#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace sglang {

struct Mxfp8ToMxfp4Params {
  const uint8_t* __restrict__ input;
  const int32_t* __restrict__ input_scale;
  int32_t* __restrict__ output;
  int32_t* __restrict__ output_scale;
  uint32_t num_tokens;
  uint32_t hidden_size;
  uint32_t input_scale_stride;
};

__forceinline__ __device__ float fp8_e4m3_to_float(uint8_t bits) {
  const uint32_t sign = bits >> 7u;
  const uint32_t exponent = (bits >> 3u) & 0xfu;
  const uint32_t mantissa = bits & 0x7u;
  float value = exponent == 0u
                    ? static_cast<float>(mantissa) * 0.001953125f
                    : __uint_as_float(((exponent + 120u) << 23u) |
                                      (mantissa << 20u));
  return sign == 0u ? value : -value;
}

// Matches deep_gemm.utils.ceil_to_ue8m0 and clamps to finite UE8M0.
__forceinline__ __device__ uint32_t ceil_to_ue8m0(float raw_scale) {
  uint32_t bits = __float_as_uint(raw_scale);
  uint32_t exponent = (bits >> 23u) & 0xffu;
  if ((bits & 0x7fffffu) != 0u) {
    ++exponent;
  }
  return max(1u, min(254u, exponent));
}

// Matches deep_gemm.utils._quantize_to_fp4_e2m1, including midpoint ties
// rounding toward zero rather than cvt.e2m1 round-to-even behavior.
__forceinline__ __device__ uint32_t fp4_e2m1_encode(float value) {
  float magnitude = min(fabsf(value), 6.0f);
  uint32_t code = (magnitude > 0.25f) + (magnitude > 0.75f) +
                  (magnitude > 1.25f) + (magnitude > 1.75f) +
                  (magnitude > 2.5f) + (magnitude > 3.5f) +
                  (magnitude > 5.0f);
  if (value < 0.0f && code != 0u) {
    code |= 0x8u;
  }
  return code;
}

// One CTA converts one DeepEP-dispatched MXFP8 token. Each thread owns eight
// adjacent values; four adjacent threads form one 32-value MXFP4 scale group.
// The output layouts match ep_scatter's FP4 input contract.
template <bool kUsePDL>
__global__ __launch_bounds__(1024, 1) void mxfp8_to_mxfp4_kernel(
    const Mxfp8ToMxfp4Params __grid_constant__ params) {
  using namespace device;

  const uint32_t token = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  PDLWaitPrimary<kUsePDL>();

  const auto* row_in =
      params.input + static_cast<uint64_t>(token) * params.hidden_size;
  float values[8];
  float local_max = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    values[i] = fp8_e4m3_to_float(row_in[tid * 8u + i]);
    local_max = max(local_max, fabsf(values[i]));
  }

  // tid is aligned in groups of four, so XOR lanes 1 and 2 remain within the
  // 32-value quantization group.
  const uint32_t active_mask = __activemask();
  local_max = max(local_max, __shfl_xor_sync(active_mask, local_max, 1));
  local_max = max(local_max, __shfl_xor_sync(active_mask, local_max, 2));
  const uint32_t fp4_exponent = ceil_to_ue8m0(max(local_max, 1e-4f) / 6.0f);
  const float inv_scale = __uint_as_float((254u - fp4_exponent) << 23u);

  uint32_t packed = 0;
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t lo = fp4_e2m1_encode(values[2u * i] * inv_scale);
    const uint32_t hi = fp4_e2m1_encode(values[2u * i + 1u] * inv_scale);
    packed |= ((lo & 0xfu) | ((hi & 0xfu) << 4u)) << (8u * i);
  }
  params.output[static_cast<uint64_t>(token) * (params.hidden_size / 8u) +
                tid] = static_cast<int32_t>(packed);

  if ((tid & 3u) == 0u) {
    const uint32_t group_32 = tid / 4u;
    const uint32_t group_128 = group_32 / 4u;
    const auto* input_scale_bytes =
        reinterpret_cast<const uint8_t*>(params.input_scale);
    const uint32_t input_exponent =
        input_scale_bytes[(group_128 / 4u) * params.input_scale_stride * 4u +
                          token * 4u + (group_128 & 3u)];
    const int32_t combined =
        max(0, min(255, static_cast<int32_t>(input_exponent) +
                            static_cast<int32_t>(fp4_exponent) - 127));
    auto* output_scale_bytes =
        reinterpret_cast<uint8_t*>(params.output_scale);
    output_scale_bytes[static_cast<uint64_t>(token) *
                           (params.hidden_size / 32u) +
                       group_32] = static_cast<uint8_t>(combined);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct Mxfp8ToMxfp4Kernel {
  static constexpr auto kernel = mxfp8_to_mxfp4_kernel<kUsePDL>;

  static void run(const tvm::ffi::TensorView input,
                  const tvm::ffi::TensorView input_scale,
                  const tvm::ffi::TensorView output,
                  const tvm::ffi::TensorView output_scale) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    auto K = SymbolicSize{"hidden_size"};
    auto G8 = SymbolicSize{"packed_mxfp8_scale_groups"};
    auto K4 = SymbolicSize{"packed_mxfp4_values"};
    auto G4 = SymbolicSize{"packed_mxfp4_scale_groups"};
    auto input_stride = SymbolicSize{"input scale row stride"};
    device.set_options<kDLCUDA>();

    // Torch FP8 has a distinct DLPack dtype, so its byte storage is validated
    // by shape/device here and interpreted explicitly as E4M3 below.
    TensorMatcher({M, K}).with_device(device).verify(input);
    TensorMatcher({M, G8})
        .with_strides({int64_t{1}, input_stride})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(input_scale);
    TensorMatcher({M, K4})
        .with_dtype<int8_t>()
        .with_device(device)
        .verify(output);
    TensorMatcher({M, G4})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(output_scale);

    RuntimeCheck(K.unwrap() % 128 == 0,
                 "MXFP8 input K must be divisible by 128");
    RuntimeCheck(K4.unwrap() * 2 == K.unwrap(),
                 "invalid packed MXFP4 value shape");
    RuntimeCheck(G8.unwrap() * 512 == K.unwrap(),
                 "invalid packed MXFP8 scale shape");
    RuntimeCheck(G4.unwrap() * 128 == K.unwrap(),
                 "invalid packed MXFP4 scale shape");
    RuntimeCheck(K.unwrap() / 8 <= 1024,
                 "MXFP8 to MXFP4 requires K <= 8192");
    RuntimeCheck(input_stride.unwrap() >= M.unwrap(),
                 "invalid input TMA-aligned scale stride");

    // A DeepEP rank may legally receive no routed tokens. Avoid an invalid
    // zero-grid launch while retaining the same shape validation contract.
    if (M.unwrap() == 0) {
      return;
    }

    const auto params = Mxfp8ToMxfp4Params{
        .input = static_cast<const uint8_t*>(input.data_ptr()),
        .input_scale = static_cast<const int32_t*>(input_scale.data_ptr()),
        .output = static_cast<int32_t*>(output.data_ptr()),
        .output_scale = static_cast<int32_t*>(output_scale.data_ptr()),
        .num_tokens = static_cast<uint32_t>(M.unwrap()),
        .hidden_size = static_cast<uint32_t>(K.unwrap()),
        .input_scale_stride = static_cast<uint32_t>(input_stride.unwrap()),
    };
    LaunchKernel(M.unwrap(), K.unwrap() / 8, device.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
