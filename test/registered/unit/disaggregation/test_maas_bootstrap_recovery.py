"""CPU regression tests for disaggregated request recovery."""

from __future__ import annotations

import ast
import asyncio
import enum
import logging
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from sglang.srt.managers import maas_ipc_delivery as delivery
from sglang.srt.managers.maas_ipc_delivery import (
    AcknowledgedDispatch,
    AdmissionRegistry,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class DeliveryTest(unittest.IsolatedAsyncioTestCase):
    async def test_loss_at_either_hop_and_ack_loss_admits_once(self):
        sender = AcknowledgedDispatch(timeout=0.005, attempts=6)
        scheduler = AdmissionRegistry()
        calls, forwarded = [], []
        attempt = 0
        key = ("attempt-a", 41)

        async def send():
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                return  # Before controller.
            forwarded.append(key)
            if attempt == 2:
                return  # Controller -> scheduler.
            scheduler.admit(key, lambda: calls.append(key))
            if attempt != 3:  # Lose first scheduler receipt.
                sender.acknowledge(key)

        await sender.send(key, send)
        self.assertEqual(attempt, 4)
        self.assertEqual(len(forwarded), 3)
        self.assertEqual(calls, [key])
        self.assertFalse(sender.pending)

    async def test_blocked_socket_send_is_bounded_and_cancelled(self):
        sender = AcknowledgedDispatch(timeout=0.005, attempts=2)
        attempts = []

        async def send():
            try:
                await asyncio.Event().wait()
            finally:
                attempts.append("cancelled")

        with self.assertRaisesRegex(TimeoutError, "admission timed out"):
            await asyncio.wait_for(sender.send(("a", 1), send), 1)
        self.assertEqual(attempts, ["cancelled", "cancelled"])
        self.assertFalse(sender.pending)

    async def test_cancellation_clears_pending(self):
        sender = AcknowledgedDispatch(timeout=0.1, attempts=2)

        async def send():
            await asyncio.Event().wait()

        task = asyncio.create_task(sender.send(("a", 1), send))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(sender.pending)

    async def test_concurrent_attempts_do_not_ack_each_other(self):
        sender = AcknowledgedDispatch(timeout=0.1, attempts=2)
        accepted = []

        async def one(i):
            key = (f"attempt-{i}", 10)

            async def send():
                await asyncio.sleep(0)
                accepted.append(key)
                sender.acknowledge(key)

            await sender.send(key, send)

        await asyncio.gather(*(one(i) for i in range(32)))
        self.assertEqual(len(set(accepted)), 32)
        self.assertFalse(sender.pending)

    def test_cancel_before_delivery_prevents_late_admission(self):
        registry = AdmissionRegistry()
        registry.abort("a")
        run = Mock()
        self.assertFalse(registry.admit(("a", 1), run))
        run.assert_not_called()

    def test_completed_admission_and_abort_are_not_reexecuted(self):
        registry = AdmissionRegistry()
        run = Mock()
        self.assertTrue(registry.admit(("a", 1), run))
        registry.abort("a")
        self.assertFalse(registry.admit(("a", 1), run))
        run.assert_called_once()

    def test_capacity_failure_does_not_evict_previous_admission(self):
        registry = AdmissionRegistry(capacity=1)
        registry.admit(("a", 1), lambda: None)
        with self.assertRaisesRegex(RuntimeError, "registry full"):
            registry.admit(("b", 2), lambda: None)
        self.assertFalse(registry.admit(("a", 1), lambda: self.fail("duplicate")))

    def test_invalid_retry_window_rejected(self):
        for timeout, attempts in [(0, 1), (1, 0), (60, 3)]:
            with self.assertRaises(ValueError):
                AcknowledgedDispatch(timeout=timeout, attempts=attempts)


def method(path, cls, name, **dependencies):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(
        n
        for n in node.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    fn.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    # Do not inject missing standard-library imports: doing so masks runtime
    # NameError failures in the generated overlays.
    imported = {
        alias.asname or alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    ns = {
        name: value
        for name, value in dict(os=os, time=time, asyncio=asyncio).items()
        if name in imported
    }
    ns["logger"] = logging.getLogger("source-test")
    ns.update(dependencies)
    exec(compile(module, str(path), "exec"), ns)
    return ns[name]


class PatchedSourceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[4]
        self.prefix = "python/sglang/srt/"

    def test_tokenizer_startup_without_distributed_process_group(self):
        fn = method(
            self.root / (self.prefix + "managers/tokenizer_manager.py"),
            "TokenizerManager",
            "init_running_status",
            get_disagg=lambda: NS(disaggregation_mode="prefill"),
            # Only config-backed properties exist in the tokenizer process.
            get_parallel=lambda: NS(enable_dp_attention=True),
            get_serving=lambda: NS(tokenizer_worker_num=1),
            AcknowledgedDispatch=AcknowledgedDispatch,
            ServerStatus=NS(Starting="starting"),
            real_time=time.time,
        )
        with patch.dict(os.environ, MAAS_PREFILL_IPC_ACK="1"):
            owner = NS(server_args=NS(tp_size=4, dp_size=4, attn_cp_size=1))
            fn(owner)
            self.assertTrue(owner._ipc_ack_enabled)
            self.assertEqual(owner.rid_to_state, {})
            for tp, dp, cp in [(4, 2, 1), (4, 4, 2)]:
                with self.assertRaises(AssertionError):
                    fn(NS(server_args=NS(tp_size=tp, dp_size=dp, attn_cp_size=cp)))

    def test_scheduler_wrapper_acks_after_admission_and_deduplicates(self):
        fn = method(
            self.root / (self.prefix + "managers/scheduler.py"),
            "Scheduler",
            "handle_generate_request",
        )
        events = []
        registry = AdmissionRegistry()
        owner = NS(
            _maas_admission_registry=lambda: registry,
            _handle_generate_request_impl=lambda req: events.append("admit"),
            ipc_channels=NS(
                send_to_tokenizer=NS(send_output=lambda ack, req: events.append("ack"))
            ),
        )
        req = NS(rid="attempt-a", bootstrap_room=1, routed_dp_rank=2)
        with patch.dict(os.environ, MAAS_PREFILL_IPC_ACK="1"), patch.dict(
            sys.modules,
            {"sglang.srt.managers.io_struct": NS(PrefillDispatchAckOutput=NS)},
        ):
            fn(owner, req)
            fn(owner, req)
        self.assertEqual(events, ["admit", "ack", "ack"])

    async def test_response_before_ack_survives_state_map_removal(self):
        path = self.root / (self.prefix + "managers/tokenizer_manager.py")
        wait = method(
            path,
            "TokenizerManager",
            "_wait_one_response",
            _REQUEST_STATE_WAIT_TIMEOUT=1,
        )
        # _handle_batch_output removes finished requests from this map. The
        # state captured before dispatch must still deliver their response.
        owner = NS(
            rid_to_state={},
            incremental_streaming_output=False,
            request_logger=NS(log_finished_request=Mock()),
            request_metrics_exporter_manager=NS(exporter_enabled=lambda: False),
        )

        async def abort_reason(*args):
            return None

        owner._handle_abort_finish_reason = abort_reason
        output = {"text": "done", "meta_info": {"finish_reason": {"type": "length"}}}
        state = NS(
            event=asyncio.Event(),
            out_list=[output],
            finished=True,
            time_stats=NS(response_sent_to_client_time=1),
        )
        state.event.set()
        response = wait(owner, NS(rid="a", stream=True), state=state)
        try:
            self.assertEqual(await response.__anext__(), output)
        finally:
            await response.aclose()

    def test_registration_transient_failure_retries_then_succeeds(self):
        post = Mock(side_effect=[TimeoutError("injected"), NS(status_code=200)])
        requests = NS(post=post, RequestException=TimeoutError)
        fn = method(
            self.root / (self.prefix + "disaggregation/common/conn.py"),
            "CommonKVSender",
            "_register_prefill_dp_rank",
            requests=requests,
            time=NS(sleep=lambda _: None),
        )
        manager = NS(attn_dp_rank=2, record_failure=Mock(), update_status=Mock())
        with patch.dict(
            sys.modules, {"sglang.srt.managers.maas_ipc_delivery": delivery}
        ):
            fn(NS(bootstrap_server_url="local", bootstrap_room=1, kv_mgr=manager))
        self.assertEqual(post.call_count, 2)
        manager.update_status.assert_not_called()

    def test_registration_permanent_failure_enters_failed_state(self):
        post = Mock(return_value=NS(status_code=503, text="unavailable"))
        requests = NS(post=post, RequestException=TimeoutError)
        fn = method(
            self.root / (self.prefix + "disaggregation/common/conn.py"),
            "CommonKVSender",
            "_register_prefill_dp_rank",
            requests=requests,
            KVPoll=NS(Failed=0),
            time=NS(sleep=lambda _: None),
        )
        manager = NS(attn_dp_rank=2, record_failure=Mock(), update_status=Mock())
        with patch.dict(
            sys.modules, {"sglang.srt.managers.maas_ipc_delivery": delivery}
        ), self.assertLogs("source-test", level="ERROR"):
            fn(NS(bootstrap_server_url="local", bootstrap_room=1, kv_mgr=manager))
        self.assertEqual(post.call_count, 3)
        manager.update_status.assert_called_once_with(1, 0)

    def test_discovery_timeout_preserves_resolved_and_terminal_requests(self):
        class Poll(enum.IntEnum):
            Failed = 0
            Bootstrapping = 1
            WaitingForInput = 2
            Success = 4

        common = self.root / (self.prefix + "disaggregation/common/conn.py")
        nixl = self.root / (self.prefix + "disaggregation/nixl/conn.py")
        check = method(
            common, "CommonKVReceiver", "_check_rank_discovery_timeout", KVPoll=Poll
        )
        poll = method(nixl, "NixlKVReceiver", "poll", KVPoll=Poll)
        for initial, expected in [
            (Poll.Bootstrapping, Poll.Failed),
            (Poll.WaitingForInput, Poll.WaitingForInput),
            (Poll.Success, Poll.Success),
        ]:
            owner = NS(
                conclude_state=None,
                started_transfer=False,
                bootstrap_started_at=time.monotonic() - 1000,
                bootstrap_room=1,
                bootstrap_addr="local",
                kv_mgr=NS(
                    waiting_timeout=180,
                    check_status=lambda _: initial,
                    record_failure=Mock(),
                    update_status=Mock(),
                ),
            )
            owner._check_rank_discovery_timeout = lambda: check(owner)
            with patch.dict(os.environ, MAAS_RANK_DISCOVERY_TIMEOUT="180"):
                self.assertEqual(poll(owner), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
