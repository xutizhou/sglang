# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Layer-sharded DeepSeek V4 KV cache pool for context-parallel prefill.

Under DSA prefill context parallelism every CP rank materialises the global
token order before writing its pools, so without this class all CP ranks hold
identical copies of the whole KV cache and the prefix cache of a CP worker is
only as large as a single GPU's pool.

``LayerSplitDeepSeekV4TokenToKVPool`` shards the paged KV pools (SWA, c4, c128
and the c4 indexer) by *layer* across the CP ranks: a rank only allocates the
layers it owns, plus one full-size *scratch* buffer per pool type.  Before a
non-owned layer is read or written in a forward step, the owner gathers the
pages that step touches and broadcasts them; every other rank scatters them
into its scratch buffer at the same page positions, so attention kernels keep
indexing with the unchanged page tables.  New tokens written during the step
go to the owner's pool and to the scratch of the other ranks.

The compress-state pools stay replicated: the compressor already runs over the
all-gathered token stream on every rank, so each rank produces identical state
and no broadcast is needed.

Layer ownership is a contiguous range of the stage's layers
(``get_layer_shard_range``), so PD transfer can describe a rank's slice with
``prefill_start_layer`` / ``prefill_end_layer`` exactly like a pipeline stage.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.cp.utils import get_layer_shard_range
from sglang.kernels.ops.attention.dsa import index_buf_accessor
from sglang.kernels.ops.attention.dsv4.index_buf_accessor import NopeFp8RopeBf16Pack
from sglang.srt.mem_cache import deepseek_v4_memory_pool as _dsv4_pool
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# Pool kinds that are sharded by layer. "kv" pools (c4 / c128 / indexer) share
# the full-KV page index space; the SWA pool has its own page space.
_KIND_SWA = "swa"
_KIND_C4 = "c4"
_KIND_C128 = "c128"
_KIND_IDX = "idx"


class _LayerSlicedSingleKVPool(DeepSeekV4SingleKVPool):
    """``DeepSeekV4SingleKVPool`` whose non-owned layers are 0-page placeholders."""

    def __init__(self, *args, owned_mask: List[bool], **kwargs):
        self._owned_mask = list(owned_mask)
        super().__init__(*args, **kwargs)

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                num_pages = (self.size + self.page_size + 1) // self.page_size
                self.kv_buffer = [
                    (
                        self.create_buffer(num_pages=num_pages)
                        if owned
                        else self.create_buffer(num_pages=0)
                    )
                    for owned in self._owned_mask
                ]

    def create_scratch(self) -> torch.Tensor:
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            num_pages = (self.size + self.page_size + 1) // self.page_size
            return self.create_buffer(num_pages=num_pages)


class _LayerSlicedIndexerPool(DeepSeekV4IndexerPool):
    """``DeepSeekV4IndexerPool`` whose non-owned layers are 0-page placeholders."""

    def __init__(self, *args, owned_mask: List[bool], **kwargs):
        self._owned_mask = list(owned_mask)
        super().__init__(*args, **kwargs)

    def _page_bytes(self) -> int:
        return self.page_size * self.get_bytes_per_token()

    def _create_buffer(self):
        page_bytes = self._page_bytes()
        num_pages = (self.size + self.page_size + 1) // self.page_size
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.index_k_with_scale_buffer = [
                    torch.zeros(
                        num_pages if owned else 0,
                        page_bytes,
                        dtype=self.index_k_with_scale_buffer_dtype,
                        device=self.device,
                    )
                    for owned in self._owned_mask
                ]

    def create_scratch(self) -> torch.Tensor:
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            num_pages = (self.size + self.page_size + 1) // self.page_size
            return torch.zeros(
                num_pages,
                self._page_bytes(),
                dtype=self.index_k_with_scale_buffer_dtype,
                device=self.device,
            )

    # Buffer-explicit variants of the base accessors (the layer-split pool picks
    # the owned buffer or the scratch buffer and passes it in).
    def get_index_k_scale_buffer_from(
        self, buf, seq_len_tensor, page_indices, seq_len_sum, max_seq_len
    ):
        return index_buf_accessor.GetKAndS.execute(
            self,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer_into(self, buf, loc, index_k, index_k_scale) -> None:
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def set_index_fused_into(self, buf, loc, cache_k) -> None:
        return _dsv4_pool.fused_store_cache(
            input=cache_k, cache=buf, indices=loc, page_size=self.page_size, type="indexer"
        )

    def set_index_fp4_into(self, buf, loc, cache_k) -> None:
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import store_fp4_index_k_cache

        return store_fp4_index_k_cache(
            input=cache_k, cache=buf, loc=loc, page_size=self.page_size
        )


class LayerSplitDeepSeekV4TokenToKVPool(DeepSeekV4TokenToKVPool):
    """DeepSeek V4 KV pool sharded by layer across prefill-CP ranks."""

    def __init__(
        self,
        *args,
        layer_shard_rank: int,
        layer_shard_size: int,
        **kwargs,
    ):
        assert (
            layer_shard_rank is not None and layer_shard_size > 1
        ), "LayerSplitDeepSeekV4TokenToKVPool requires layer_shard_size > 1"
        self.layer_shard_rank = int(layer_shard_rank)
        self.layer_shard_size = int(layer_shard_size)
        self.layer_shard_enabled = True
        # Filled in by the pool-construction hooks below (they run inside
        # super().__init__, before the rest of this __init__ continues).
        self._owned_stage_range: Optional[Tuple[int, int]] = None
        super().__init__(*args, **kwargs)
        assert not self._unified_kv, "layer split is not supported with unified_kv"

        owned_start, owned_end = self._owned_stage_range
        # Global (model) layer ids of the owned contiguous range.
        self.layer_shard_start = self._stage_start + owned_start
        self.layer_shard_end = self._stage_start + owned_end

        self._scratch: Dict[str, torch.Tensor] = {}
        self._scratch_layer: Dict[str, Optional[int]] = {}
        self._alloc_scratch()

        # Pages touched by the current forward step (set by begin_step).
        self._step_kv_pages: Optional[torch.Tensor] = None
        self._step_swa_pages: Optional[torch.Tensor] = None
        self._step_active = False
        self._step_id = 0
        self._stats = {"bcast_bytes": 0, "bcast_calls": 0, "prefetch_hits": 0}
        import os

        self._debug = os.environ.get("SGLANG_DSV4_LAYER_SPLIT_DEBUG", "0") == "1"
        # Owner-broadcast prefetch of the next layer of each kind on a side stream
        # (SGLANG_DSV4_LAYER_SPLIT_PREFETCH=0 disables it).
        self._prefetch_enabled = (
            os.environ.get("SGLANG_DSV4_LAYER_SPLIT_PREFETCH", "1") == "1"
        )
        self._prefetch_stream = torch.cuda.Stream(device=self.device)
        # kind -> (layer_id, payload, event) for an in-flight prefetch.
        self._prefetched: Dict[str, Tuple[int, Optional[torch.Tensor], torch.cuda.Event]] = {}
        stage_ids = self._stage_layer_ids()
        self._layers_of_kind: Dict[str, List[int]] = {
            _KIND_SWA: list(stage_ids),
            _KIND_C4: [l for l in stage_ids if self.compression_ratios[l] == 4],
            _KIND_IDX: [l for l in stage_ids if self.compression_ratios[l] == 4],
            _KIND_C128: [l for l in stage_ids if self.compression_ratios[l] == 128],
        }
        self._log_plan()

    # ------------------------------------------------------------------ setup

    def _ensure_owned_range(self) -> Tuple[int, int]:
        if self._owned_stage_range is None:
            stage_len = self._stage_end - self._stage_start
            self._owned_stage_range = get_layer_shard_range(
                self.layer_shard_rank, self.layer_shard_size, stage_len
            )
        return self._owned_stage_range

    def _is_owned_global(self, layer_id: int) -> bool:
        start, end = self._ensure_owned_range()
        local = layer_id - self._stage_start
        return start <= local < end

    def _owner_rank_global(self, layer_id: int) -> int:
        stage_len = self._stage_end - self._stage_start
        local = layer_id - self._stage_start
        for rank in range(self.layer_shard_size):
            s, e = get_layer_shard_range(rank, self.layer_shard_size, stage_len)
            if s <= local < e:
                return rank
        raise ValueError(f"layer {layer_id} is outside the stage range")

    def _make_kv_pool(
        self,
        *,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        global_page_size: int,
        cls: type = DeepSeekV4SingleKVPool,
    ) -> DeepSeekV4SingleKVPool:
        assert cls is DeepSeekV4SingleKVPool, (
            "layer split supports the plain DSV4 KV sub-pool only " f"(got {cls})"
        )
        owned_mask = self._owned_mask_for_pool(page_size, global_page_size, layer_num)
        return _LayerSlicedSingleKVPool(
            size,
            page_size,
            dtype,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            owned_mask=owned_mask,
        )

    def _make_indexer_pool(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
    ) -> DeepSeekV4IndexerPool:
        owned_mask = self._owned_mask_by_ratio(4)
        assert len(owned_mask) == layer_num
        return _LayerSlicedIndexerPool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            owned_mask=owned_mask,
        )

    def _stage_layer_ids(self) -> List[int]:
        return list(range(self._stage_start, self._stage_end))

    def _owned_mask_by_ratio(self, ratio: int) -> List[bool]:
        return [
            self._is_owned_global(l)
            for l in self._stage_layer_ids()
            if self.compression_ratios[l] == ratio
        ]

    def _owned_mask_for_pool(
        self, page_size: int, global_page_size: int, layer_num: int
    ) -> List[bool]:
        stage_ids = self._stage_layer_ids()
        if layer_num == len(stage_ids):
            # SWA pool: one buffer per stage layer.
            return [self._is_owned_global(l) for l in stage_ids]
        ratio = global_page_size // page_size
        mask = self._owned_mask_by_ratio(ratio)
        assert len(mask) == layer_num, (
            f"cannot infer layer ownership for a pool with {layer_num} layers "
            f"(page_size={page_size}, global_page_size={global_page_size})"
        )
        return mask

    def _alloc_scratch(self) -> None:
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self._scratch[_KIND_SWA] = self.swa_kv_pool.create_scratch()
            self._scratch[_KIND_C4] = self.c4_kv_pool.create_scratch()
            self._scratch[_KIND_C128] = self.c128_kv_pool.create_scratch()
            self._scratch[_KIND_IDX] = self.c4_indexer_kv_pool.create_scratch()
        for kind in self._scratch:
            self._scratch_layer[kind] = None

    def _log_plan(self) -> None:
        stage_len = self._stage_end - self._stage_start
        parts = []
        for rank in range(self.layer_shard_size):
            s, e = get_layer_shard_range(rank, self.layer_shard_size, stage_len)
            parts.append(f"r{rank}:[{self._stage_start + s},{self._stage_start + e})")
        scratch_bytes = sum(t.nbytes for t in self._scratch.values())
        logger.info(
            "DSV4 layer split: rank=%d/%d owns layers [%d,%d) of stage [%d,%d); "
            "partitions=%s; scratch=%.1f MiB",
            self.layer_shard_rank,
            self.layer_shard_size,
            self.layer_shard_start,
            self.layer_shard_end,
            self._stage_start,
            self._stage_end,
            "; ".join(parts),
            scratch_bytes / 2**20,
        )

    # ----------------------------------------------------------- step pages

    def begin_step(self, forward_batch, req_to_token: Optional[torch.Tensor] = None) -> None:
        """Record the pages this forward step can touch and drop stale scratch.

        Called once per forward from the attention backend's metadata init,
        identically on every CP rank.  ``forward_batch`` is the full (unsharded)
        batch under CP-v1, so every rank derives the same page set from
        ``req_to_token`` (and the full->SWA mapping for the SWA pool).
        """
        if self._debug and (self._step_id < 5 or self._step_id % 200 == 0):
            logger.info(
                "DSV4 layer split step %d done: kv_pages=%s swa_pages=%s "
                "bcast_calls=%d bcast_MiB=%.1f prefetch_hits=%d",
                self._step_id,
                None if self._step_kv_pages is None else int(self._step_kv_pages.numel()),
                None if self._step_swa_pages is None else int(self._step_swa_pages.numel()),
                self._stats["bcast_calls"],
                self._stats["bcast_bytes"] / 2**20,
                self._stats["prefetch_hits"],
            )
        self._step_id += 1
        for kind in self._scratch_layer:
            self._scratch_layer[kind] = None
        self._prefetched.clear()
        self._step_kv_pages = None
        self._step_swa_pages = None
        self._step_active = True
        self._stats["bcast_calls"] = 0
        self._stats["bcast_bytes"] = 0
        self._stats["prefetch_hits"] = 0

        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        seq_lens = getattr(forward_batch, "seq_lens", None)
        if req_to_token is None:
            pool = getattr(forward_batch, "req_to_token_pool", None)
            req_to_token = pool.req_to_token if pool is not None else None
        if (
            req_pool_indices is None
            or seq_lens is None
            or req_to_token is None
            or req_pool_indices.numel() == 0
        ):
            if self._debug:
                logger.warning("DSV4 layer split: begin_step without batch info")
            return
        rows = req_to_token[req_pool_indices.long()]
        max_len = int(seq_lens.max().item()) if seq_lens.numel() else 0
        if max_len <= 0:
            return
        rows = rows[:, :max_len]
        mask = (
            torch.arange(max_len, device=rows.device)[None, :]
            < seq_lens.to(rows.device)[:, None]
        )
        tokens = rows[mask].long()
        if tokens.numel() == 0:
            return
        # c4 / c128 / indexer pools share the full-KV page index space.
        self._step_kv_pages = torch.unique(tokens // self.page_size)
        # SWA pool pages: only tokens still resident in the SWA ring map to a slot.
        mapping = getattr(self, "full_to_swa_index_mapping", None)
        if mapping is not None:
            swa_loc = mapping[tokens]
            swa_loc = swa_loc[swa_loc >= 0].long()
            if swa_loc.numel() > 0:
                self._step_swa_pages = torch.unique(swa_loc // self.swa_page_size)

    def end_step(self) -> None:
        self._step_active = False

    # ------------------------------------------------------------ broadcast

    def _pages_for(self, kind: str) -> Optional[torch.Tensor]:
        return self._step_swa_pages if kind == _KIND_SWA else self._step_kv_pages

    def _owned_buffer(self, kind: str, layer_id: int) -> torch.Tensor:
        if kind == _KIND_SWA:
            return self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)]
        if kind == _KIND_IDX:
            _, compress_layer_id, _ = self.layer_mapping[layer_id]
            return self.c4_indexer_kv_pool.index_k_with_scale_buffer[compress_layer_id]
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        return compress_kv_pool.kv_buffer[compress_layer_id]

    def _next_layer_of_kind(self, kind: str, layer_id: int) -> Optional[int]:
        layers = self._layers_of_kind[kind]
        try:
            i = layers.index(layer_id)
        except ValueError:
            return None
        return layers[i + 1] if i + 1 < len(layers) else None

    def _gather_and_broadcast(
        self, kind: str, layer_id: int, pages: torch.Tensor
    ) -> torch.Tensor:
        """Owner gathers ``pages`` of ``layer_id`` and broadcasts them; returns the
        payload (on every rank). Runs on the current stream."""
        owned = self._is_owned_global(layer_id)
        scratch = self._scratch[kind]
        if owned:
            payload = self._owned_buffer(kind, layer_id).index_select(0, pages)
        else:
            payload = torch.empty(
                (pages.numel(), scratch.shape[1]),
                dtype=scratch.dtype,
                device=scratch.device,
            )
        get_parallel().attn_cp_group.broadcast(
            payload, src=self._owner_rank_global(layer_id)
        )
        self._stats["bcast_bytes"] += payload.nbytes
        self._stats["bcast_calls"] += 1
        return payload

    def _prefetch(self, kind: str, layer_id: int) -> None:
        """Issue the owner broadcast for ``layer_id`` on the side stream so it
        overlaps the current layer's compute. Every CP rank issues the same
        sequence of prefetches, so the collectives stay matched."""
        pages = self._pages_for(kind)
        if pages is None or pages.numel() == 0 or kind in self._prefetched:
            return
        if self._scratch_layer.get(kind) == layer_id:
            return
        cur = torch.cuda.current_stream()
        self._prefetch_stream.wait_stream(cur)
        with torch.cuda.stream(self._prefetch_stream):
            payload = self._gather_and_broadcast(kind, layer_id, pages)
            event = torch.cuda.Event()
            event.record(self._prefetch_stream)
        self._prefetched[kind] = (layer_id, payload, event)

    def _materialize(self, kind: str, layer_id: int) -> torch.Tensor:
        """Return the buffer to use for ``layer_id``: the owned pool buffer on the
        owner, otherwise the scratch buffer filled from the owner's pages."""
        owned = self._is_owned_global(layer_id)
        if self._scratch_layer.get(kind) == layer_id:
            return self._owned_buffer(kind, layer_id) if owned else self._scratch[kind]
        scratch = self._scratch[kind]
        pages = self._pages_for(kind)
        if pages is not None and pages.numel() > 0:
            pf = self._prefetched.pop(kind, None)
            if pf is not None and pf[0] == layer_id:
                _, payload, event = pf
                torch.cuda.current_stream().wait_event(event)
                self._stats["prefetch_hits"] += 1
            else:
                # Prefetch for a different layer (should not happen: the touch
                # order is deterministic) -- fall back to a synchronous broadcast.
                payload = self._gather_and_broadcast(kind, layer_id, pages)
            if not owned:
                scratch.index_copy_(0, pages, payload)
            payload.record_stream(torch.cuda.current_stream())
        self._scratch_layer[kind] = layer_id
        if self._prefetch_enabled:
            nxt = self._next_layer_of_kind(kind, layer_id)
            if nxt is not None:
                self._prefetch(kind, nxt)
        return self._owned_buffer(kind, layer_id) if owned else scratch

    def _buffer_for(self, kind: str, layer_id: int) -> torch.Tensor:
        if not self._step_active and self._is_owned_global(layer_id):
            return self._owned_buffer(kind, layer_id)
        if not self._step_active and self._debug:
            logger.warning(
                "DSV4 layer split: layer %d (%s) touched outside a step", layer_id, kind
            )
        return self._materialize(kind, layer_id)

    def _write_targets(self, kind: str, layer_id: int) -> List[torch.Tensor]:
        """Buffer a write for ``layer_id`` must land in on this rank: the owner's
        pool buffer, or the (materialised) scratch on every other rank."""
        return [self._buffer_for(kind, layer_id)]

    def layer_split_stats(self) -> dict:
        return dict(self._stats)

    # ------------------------------------------------------- SWA accessors

    def get_swa_raw_buffer(self, layer_id: int) -> torch.Tensor:
        return self._buffer_for(_KIND_SWA, layer_id)

    def get_swa_key_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        return self._buffer_for(_KIND_SWA, layer_id)

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        return self._buffer_for(_KIND_SWA, layer_id)

    def set_swa_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        for buf in self._write_targets(_KIND_SWA, layer_id):
            _dsv4_pool.dsv4_index_buf_accessor.SetKAndS.execute(
                pool=self.swa_kv_pool,
                buf=buf,
                loc=loc,
                nope_fp8_rope_bf16_pack=cache_nope_fp8_rope_bf16_pack,
            )

    def set_swa_key_buffer_radix(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        self.set_swa_key_buffer(layer_id, swa_loc, cache_nope_fp8_rope_bf16_pack)

    def set_swa_key_buffer_radix_fused(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        for buf in self._write_targets(_KIND_SWA, layer_id):
            _dsv4_pool.fused_store_cache(
                input=cache_k,
                cache=buf,
                indices=swa_loc,
                page_size=self.swa_kv_pool.page_size,
                type="flashmla",
            )

    def set_swa_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        for buf in self._write_targets(_KIND_SWA, layer_id):
            _dsv4_pool.fused_k_norm_rope_flashmla(
                kv=kv,
                kv_weight=kv_weight,
                eps=eps,
                freqs_cis=freqs_cis,
                positions=positions,
                out_loc=swa_loc,
                kvcache=buf,
                page_size=self.swa_kv_pool.page_size,
            )

    # ------------------------------------------------- compressed accessors

    def _kind_for_compressed(self, layer_id: int) -> str:
        ratio, _, _ = self.layer_mapping[layer_id]
        return _KIND_C4 if ratio == 4 else _KIND_C128

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor | None:
        self.wait_layer_transfer(layer_id)
        _, _, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        buf = self._buffer_for(self._kind_for_compressed(layer_id), layer_id)
        if compress_kv_pool.store_dtype != compress_kv_pool.dtype:
            return buf.view(compress_kv_pool.dtype)
        return buf

    def set_extra_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        _, _, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        for buf in self._write_targets(self._kind_for_compressed(layer_id), layer_id):
            _dsv4_pool.dsv4_index_buf_accessor.SetKAndS.execute(
                pool=compress_kv_pool,
                buf=buf,
                loc=loc,
                nope_fp8_rope_bf16_pack=cache_nope_fp8_rope_bf16_pack,
            )

    def set_extra_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        _, _, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        for buf in self._write_targets(self._kind_for_compressed(layer_id), layer_id):
            _dsv4_pool.fused_store_cache(
                input=cache_k,
                cache=buf,
                indices=loc,
                page_size=compress_kv_pool.page_size,
                type="flashmla",
            )

    # ---------------------------------------------------- indexer accessors

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        compress_ratio, _, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self._buffer_for(_KIND_IDX, layer_id)

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.wait_layer_transfer(layer_id)
        compress_ratio, _, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        buf = self._buffer_for(_KIND_IDX, layer_id)
        return self.c4_indexer_kv_pool.get_index_k_scale_buffer_from(
            buf, seq_len_tensor, page_indices, seq_len_sum, max_seq_len
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        compress_ratio, _, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        for buf in self._write_targets(_KIND_IDX, layer_id):
            self.c4_indexer_kv_pool.set_index_k_scale_buffer_into(
                buf, loc, index_k, index_k_scale
            )

    def set_index_k_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        compress_ratio, _, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        for buf in self._write_targets(_KIND_IDX, layer_id):
            self.c4_indexer_kv_pool.set_index_fused_into(buf, loc, cache_k)

    def set_index_k_fp4(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        compress_ratio, _, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        for buf in self._write_targets(_KIND_IDX, layer_id):
            self.c4_indexer_kv_pool.set_index_fp4_into(buf, loc, cache_k)

    # ------------------------------------------------------- PD transfer

    def _owned_layers_with_ratio(self, ratio: int) -> List[int]:
        """c{ratio}-local indices of the owned layers (in bucket order)."""
        out = []
        for l in self._stage_layer_ids():
            if self.compression_ratios[l] != ratio:
                continue
            _, compress_layer_id, _ = self.layer_mapping[l]
            if self._is_owned_global(l):
                out.append(compress_layer_id)
        return out

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        c4_owned = self._owned_layers_with_ratio(4)
        c128_owned = self._owned_layers_with_ratio(128)
        groups = [
            (self.c4_kv_pool.kv_buffer, c4_owned),
            (self.c4_indexer_kv_pool.index_k_with_scale_buffer, c4_owned),
            (self.c128_kv_pool.kv_buffer, c128_owned),
        ]
        for bufs, ids in groups:
            for i in ids:
                buf = bufs[i]
                assert buf.ndim == 2 and buf.shape[0] > 0
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)
        return data_ptrs, data_lens, item_lens

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        # Layout mirrors the base class: [swa layers][c4 states][c4 indexer states],
        # restricted to the owned layer range so the decode-side slice matches.
        for l in self._stage_layer_ids():
            if not self._is_owned_global(l):
                continue
            buf = self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(l)]
            assert buf.ndim == 2 and buf.shape[0] > 0
            data_ptrs.append(buf.data_ptr())
            data_lens.append(buf.nbytes)
            item_lens.append(buf[0].nbytes)
        for pools in [self.compress_state_pools, self.indexer_compress_state_pools]:
            for l in self._stage_layer_ids():
                pool = pools[l]
                if pool is None or pool.ratio != 4 or not self._is_owned_global(l):
                    continue
                buf = pool.kv_score_buffer.kv_score
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                # One transfer item = one request's state ring (mirrors the base class).
                item_lens.append(buf[0].nbytes * pool.ring_size)
        return data_ptrs, data_lens, item_lens

    def get_c128_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        for l in self._stage_layer_ids():
            pool = self.compress_state_pools[l]
            if pool is None or pool.ratio != 128 or not self._is_owned_global(l):
                continue
            t = pool.kv_score_buffer.kv_score
            assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
            data_ptrs.append(t.data_ptr())
            data_lens.append(t.nbytes)
            item_lens.append(
                t[0].nbytes if _dsv4_pool.ONLINE_C128 else t[0].nbytes * 128
            )
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self) -> int:
        total = 0
        for pool in (self.swa_kv_pool, self.c4_kv_pool, self.c128_kv_pool):
            total += sum(b.nbytes for b in pool.kv_buffer)
        total += sum(b.nbytes for b in self.c4_indexer_kv_pool.index_k_with_scale_buffer)
        total += sum(t.nbytes for t in self._scratch.values())
        return total
