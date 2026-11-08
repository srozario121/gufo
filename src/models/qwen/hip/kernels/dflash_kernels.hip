#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "src/models/qwen/hip/kernels/dflash_kernels.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip::kernels {

// The DFlash-2 draft head is the target LM head, which the Q8 target keeps in
// BF16 at 2.54 GB -- half of everything the draft graph reads. Drafting is
// purely weight-bandwidth bound, and the draft only steers which tokens the
// target verifies, so a draft-private Q8_0 copy cannot change an emitted token:
// the verifier still runs the target's BF16 head.
__global__ void dflash_quantize_bf16_to_q8_0_kernel(
    const hip_bfloat16* __restrict__ source,
    Q8_0Block* __restrict__ destination, uint64_t total_blocks) {
  const uint64_t block =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block >= total_blocks) {
    return;
  }
  const hip_bfloat16* values = source + (block * kQ8_0BlockSize);
  float absolute_max = 0.0F;
  for (uint32_t index = 0; index < kQ8_0BlockSize; ++index) {
    absolute_max =
        fmaxf(absolute_max, fabsf(static_cast<float>(values[index])));
  }
  const float scale = absolute_max / 127.0F;
  const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
  Q8_0Block& out = destination[block];
  out.d = __float2half(scale);
  for (uint32_t index = 0; index < kQ8_0BlockSize; ++index) {
    const float quantized = roundf(static_cast<float>(values[index]) * inverse);
    out.qs[index] =
        static_cast<int8_t>(fminf(fmaxf(quantized, -127.0F), 127.0F));
  }
}

void LaunchDFlashQuantizeBf16ToQ8_0(const void* bf16_source,
                                    void* q8_destination,
                                    std::size_t total_elements,
                                    hipStream_t stream) {
  if (bf16_source == nullptr || q8_destination == nullptr ||
      total_elements % kQ8_0BlockSize != 0) {
    return;
  }
  const std::uint64_t total_blocks = total_elements / kQ8_0BlockSize;
  constexpr std::uint32_t kThreads = 256;
  const std::uint64_t grid = (total_blocks + kThreads - 1) / kThreads;
  dflash_quantize_bf16_to_q8_0_kernel<<<static_cast<unsigned int>(grid),
                                        kThreads, 0, stream>>>(
      static_cast<const hip_bfloat16*>(bf16_source),
      static_cast<Q8_0Block*>(q8_destination), total_blocks);
}

__global__ void dflash_grouped_dynamic_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ dynamic_coefficients,
    const float* __restrict__ base_kernel, float* __restrict__ output,
    uint32_t num_tokens, uint32_t hidden_size, uint32_t kernel_size,
    uint32_t group_size, uint32_t side) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = static_cast<uint64_t>(num_tokens) * hidden_size;
  if (index >= total) {
    return;
  }

  const uint32_t token = index / hidden_size;
  const uint32_t channel = index % hidden_size;
  const uint32_t groups = hidden_size / group_size;
  const uint32_t group = channel / group_size;
  const uint32_t coefficients_per_token = 2U * kernel_size * groups;

  float sum = 0.0F;
  for (uint32_t tap = 0; tap < kernel_size; ++tap) {
    if (token < tap) {
      continue;
    }
    const uint64_t dynamic_index =
        static_cast<uint64_t>(token) * coefficients_per_token +
        (side * kernel_size + tap) * groups + group;
    const uint64_t base_index =
        static_cast<uint64_t>(side * kernel_size + tap) * hidden_size + channel;
    const uint64_t input_index =
        static_cast<uint64_t>(token - tap) * hidden_size + channel;
    sum += (base_kernel[base_index] + dynamic_coefficients[dynamic_index]) *
           input[input_index];
  }
  output[index] = sum;
}

__global__ void dflash_silu_mul_kernel(float* __restrict__ gate,
                                       const float* __restrict__ up,
                                       uint32_t total_elements) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  const float g = gate[idx];
  const float silu_g = g / (1.0F + expf(-g));
  gate[idx] = silu_g * up[idx];
}

// Non-causal draft attention in three passes over one KV head's 32
// (draft token, query head) pairs, so every K and V row is read once instead
// of once per workgroup of a token and two query heads, sixteen times per KV
// head. Each pair keeps the arithmetic of that former kernel bit for bit: the
// same per-lane QK expression and offset-16..1 xor sum, the same 256-thread
// max and sum trees, and the same ascending FMA sequence over the values.
// Scores live in a [kv head][pair chunk][key][32 pairs] scratch between the
// passes.
constexpr uint32_t kAttentionPairs = 32;
constexpr uint32_t kAttentionScoreKeys = 64;
constexpr uint32_t kAttentionSoftmaxPairs = 8;
constexpr uint32_t kAttentionValuePairs = 4;
constexpr uint32_t kAttentionValueKeys = 128;
constexpr uint32_t kAttentionThreads = 256;

__device__ __forceinline__ uint32_t AttentionPairChunks(uint32_t block_count,
                                                        uint32_t group) {
  return (block_count * group + kAttentionPairs - 1U) / kAttentionPairs;
}

// One step of the transposed xor sum: lanes with bit `Width` set keep the upper
// half of the live pairs, and each kept pair adds its partner lane's partial.
template <uint32_t Width>
__device__ __forceinline__ void HalvePairSums(float (&dot)[kAttentionPairs],
                                              uint32_t lane) {
  const bool upper = (lane & Width) != 0U;
#pragma unroll
  for (uint32_t g = 0; g < Width; ++g) {
    const float keep = upper ? dot[g + Width] : dot[g];
    const float send = upper ? dot[g] : dot[g + Width];
    dot[g] = keep + __shfl_xor(send, static_cast<int>(Width));
  }
}

// One wave per key: lane i holds the QK partial of dimensions 4i..4i+3 for all
// 32 pairs. Halving the pair set at each xor offset leaves lane i with pair
// i's sum, the value every lane of the per-pair xor butterfly converges to.
__global__ void __launch_bounds__(kAttentionThreads)
    dflash_attention_scores_kernel(
        const float* __restrict__ q, const float* __restrict__ injected_k,
        const float* __restrict__ block_k, float* __restrict__ scores,
        uint32_t current_pos, uint32_t history_start, uint32_t history_count,
        uint32_t block_count, uint32_t sliding_window, uint32_t num_q_heads,
        uint32_t num_kv_heads, uint32_t head_dim, float scale,
        uint32_t history_capacity) {
  // KV heads vary fastest across workgroups, so the resident ones read whole
  // K rows of the same keys together.
  const uint32_t kv_h = blockIdx.x, keys = blockIdx.y, chunk = blockIdx.z;
  const uint32_t group = num_q_heads / num_kv_heads,
                 pairs = block_count * group,
                 total_k = history_count + block_count,
                 q_dim = num_q_heads * head_dim,
                 kv_dim = num_kv_heads * head_dim, num_vec = head_dim / 4U;
  const uint32_t lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
  __shared__ float4 query[kAttentionPairs][32];
  for (uint32_t index = threadIdx.x; index < kAttentionPairs * 32U;
       index += kAttentionThreads) {
    const uint32_t pair = chunk * kAttentionPairs + index / 32U,
                   vec = index % 32U;
    float4 value = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    if (pair < pairs && vec < num_vec) {
      const uint32_t tok = pair / group, h = kv_h * group + pair % group;
      value = reinterpret_cast<const float4*>(
          q + uint64_t(tok) * q_dim + uint64_t(h) * head_dim)[vec];
    }
    query[index / 32U][vec] = value;
  }
  __syncthreads();
  float4 qv[kAttentionPairs];
#pragma unroll
  for (uint32_t pair = 0; pair < kAttentionPairs; ++pair)
    qv[pair] = query[pair][lane];

  const uint32_t pair = chunk * kAttentionPairs + lane;
  const uint32_t query_position = current_pos + pair / group;
  float* pair_scores =
      scores +
      (uint64_t(kv_h) * AttentionPairChunks(block_count, group) + chunk) *
          total_k * kAttentionPairs;
  constexpr uint32_t kWaves = kAttentionThreads / 32U;
  constexpr uint32_t kKeys = kAttentionScoreKeys / kWaves;
  // Issue the wave's key rows together, then score them one at a time.
  float4 rows[kKeys];
#pragma unroll
  for (uint32_t j = 0; j < kKeys; ++j) {
    const uint32_t local = keys * kAttentionScoreKeys + wave + j * kWaves;
    rows[j] = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    if (local < total_k && lane < num_vec) {
      const uint32_t position = history_start + local;
      rows[j] = reinterpret_cast<const float4*>(
          local < history_count
              ? injected_k + uint64_t(position % history_capacity) * kv_dim +
                    uint64_t(kv_h) * head_dim
              : block_k + uint64_t(local - history_count) * kv_dim +
                    uint64_t(kv_h) * head_dim)[lane];
    }
  }
#pragma unroll
  for (uint32_t j = 0; j < kKeys; ++j) {
    const uint32_t local = keys * kAttentionScoreKeys + wave + j * kWaves;
    if (local >= total_k)
      break;
    const float4 kv = rows[j];
    float dot[kAttentionPairs] = {};
    if (lane < num_vec) {
#pragma unroll
      for (unsigned g = 0; g < kAttentionPairs; ++g) {
        dot[g] += (qv[g].x * kv.x) + (qv[g].y * kv.y) + (qv[g].z * kv.z) +
                  (qv[g].w * kv.w);
      }
    }
    HalvePairSums<16>(dot, lane);
    HalvePairSums<8>(dot, lane);
    HalvePairSums<4>(dot, lane);
    HalvePairSums<2>(dot, lane);
    HalvePairSums<1>(dot, lane);
    if (pair < pairs) {
      const uint32_t position = history_start + local;
      bool visible;
      if (local < history_count) {
        visible = query_position - position < sliding_window;
      } else {
        const uint32_t key_position = current_pos + (local - history_count);
        const uint32_t distance = query_position > key_position
                                      ? query_position - key_position
                                      : key_position - query_position;
        visible = distance < sliding_window;
      }
      pair_scores[uint64_t(local) * kAttentionPairs + lane] =
          visible ? dot[0] * scale : -1e20F;
    }
  }
}

// Turns eight pairs' scores into probabilities with the former kernel's
// thread-strided partials and halving trees.
__global__ void __launch_bounds__(kAttentionThreads)
    dflash_attention_softmax_kernel(float* __restrict__ scores,
                                    uint32_t history_count,
                                    uint32_t block_count, uint32_t group) {
  const uint32_t kv_h = blockIdx.y, chunk = blockIdx.z,
                 total_k = history_count + block_count;
  float* base =
      scores +
      (uint64_t(kv_h) * AttentionPairChunks(block_count, group) + chunk) *
          total_k * kAttentionPairs +
      blockIdx.x * kAttentionSoftmaxPairs;
  constexpr uint32_t kPairs = kAttentionSoftmaxPairs;
  __shared__ float reduce[kPairs][kAttentionThreads];
  float maxima[kPairs];
#pragma unroll
  for (unsigned g = 0; g < kPairs; ++g)
    maxima[g] = -1e20F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned g = 0; g < kPairs; ++g)
      maxima[g] = fmaxf(maxima[g], base[uint64_t(p) * kAttentionPairs + g]);
  }
#pragma unroll
  for (unsigned g = 0; g < kPairs; ++g)
    reduce[g][threadIdx.x] = maxima[g];
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (unsigned g = 0; g < kPairs; ++g)
        reduce[g][threadIdx.x] =
            fmaxf(reduce[g][threadIdx.x], reduce[g][threadIdx.x + stride]);
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned g = 0; g < kPairs; ++g)
    maxima[g] = reduce[g][0];
  __syncthreads();
  float sums[kPairs] = {};
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned g = 0; g < kPairs; ++g) {
      const float e =
          expf(base[uint64_t(p) * kAttentionPairs + g] - maxima[g]);
      base[uint64_t(p) * kAttentionPairs + g] = e;
      sums[g] += e;
    }
  }
#pragma unroll
  for (unsigned g = 0; g < kPairs; ++g)
    reduce[g][threadIdx.x] = sums[g];
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (unsigned g = 0; g < kPairs; ++g)
        reduce[g][threadIdx.x] += reduce[g][threadIdx.x + stride];
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned g = 0; g < kPairs; ++g)
    sums[g] = 1.f / (reduce[g][0] + 1e-9F);
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned g = 0; g < kPairs; ++g)
      base[uint64_t(p) * kAttentionPairs + g] *= sums[g];
  }
}

// Lane d of wave w accumulates pairs 4w..4w+3 of a 32-dimension slice, each
// in its ascending FMA chain. The workgroup stages a tile of V rows and
// probabilities in LDS, loading the next tile while it accumulates this one.
__global__ void __launch_bounds__(kAttentionThreads)
    dflash_attention_values_kernel(
        const float* __restrict__ scores, const float* __restrict__ injected_v,
        const float* __restrict__ block_v, float* __restrict__ out,
        uint32_t history_start, uint32_t history_count, uint32_t block_count,
        uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim,
        uint32_t history_capacity) {
  constexpr uint32_t kTile = kAttentionValueKeys;
  constexpr uint32_t kRowVectors = 8;  // 32 floats per key
  constexpr uint32_t kLoads = kTile * kRowVectors / kAttentionThreads;
  const uint32_t kv_h = blockIdx.y, chunk = blockIdx.z;
  const uint32_t first_pair = (threadIdx.x >> 5U) * kAttentionValuePairs;
  const uint32_t group = num_q_heads / num_kv_heads,
                 pairs = block_count * group,
                 total_k = history_count + block_count,
                 q_dim = num_q_heads * head_dim,
                 kv_dim = num_kv_heads * head_dim;
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t slice = blockIdx.x * 32U, d = slice + lane;
  const float4* weights = reinterpret_cast<const float4*>(
      scores +
      (uint64_t(kv_h) * AttentionPairChunks(block_count, group) + chunk) *
          total_k * kAttentionPairs);
  __shared__ float4 tile_values[kTile][kRowVectors];
  __shared__ float4 tile_weights[kTile][kRowVectors];
  // Two register stages keep the next two tiles' loads in flight. Loads are
  // unconditional, with addresses clamped into the operands, so the compiler
  // waits only for the older stage; clamped keys are never accumulated and
  // clamped dimensions never stored.
  static_assert(kLoads == 4U, "a stage holds four rows per thread");
  struct Stage {
    float4 v0, v1, v2, v3, w0, w1, w2, w3;
  };
  const auto fetch = [&](uint32_t first, uint32_t i, float4& value,
                         float4& weight) {
    const uint32_t index = threadIdx.x + i * kAttentionThreads;
    const uint32_t wanted = first + index / kRowVectors,
                   vec = index % kRowVectors;
    const uint32_t key = wanted < total_k ? wanted : total_k - 1U;
    const uint32_t dim = slice + vec * 4U;
    const float* row =
        key < history_count
            ? injected_v +
                  uint64_t((history_start + key) % history_capacity) * kv_dim
            : block_v + uint64_t(key - history_count) * kv_dim;
    value = *reinterpret_cast<const float4*>(
        row + uint64_t(kv_h) * head_dim + (dim < head_dim ? dim : 0U));
    weight = weights[uint64_t(key) * kRowVectors + vec];
  };
  const auto load = [&](uint32_t first) {
    Stage stage;
    fetch(first, 0, stage.v0, stage.w0);
    fetch(first, 1, stage.v1, stage.w1);
    fetch(first, 2, stage.v2, stage.w2);
    fetch(first, 3, stage.v3, stage.w3);
    return stage;
  };
  const auto put = [&](uint32_t i, const float4& value, const float4& weight) {
    const uint32_t index = threadIdx.x + i * kAttentionThreads;
    tile_values[index / kRowVectors][index % kRowVectors] = value;
    tile_weights[index / kRowVectors][index % kRowVectors] = weight;
  };
  const float* lds_values = reinterpret_cast<const float*>(tile_values);
  const float* lds_weights =
      reinterpret_cast<const float*>(tile_weights) + first_pair;
  float acc[kAttentionValuePairs] = {};
  const auto stage_tile = [&](const Stage& stage) {
    __syncthreads();
    put(0, stage.v0, stage.w0);
    put(1, stage.v1, stage.w1);
    put(2, stage.v2, stage.w2);
    put(3, stage.v3, stage.w3);
    __syncthreads();
  };
  const auto consume = [&](uint32_t first) {
    const uint32_t count = total_k - first < kTile ? total_k - first : kTile;
    if (count == kTile) {
#pragma unroll 16
      for (unsigned i = 0; i < kTile; ++i) {
        const float value = lds_values[i * 32U + lane];
#pragma unroll
        for (unsigned g = 0; g < kAttentionValuePairs; ++g)
          acc[g] += lds_weights[i * kAttentionPairs + g] * value;
      }
    } else {
      for (unsigned i = 0; i < count; ++i) {
        const float value = lds_values[i * 32U + lane];
#pragma unroll
        for (unsigned g = 0; g < kAttentionValuePairs; ++g)
          acc[g] += lds_weights[i * kAttentionPairs + g] * value;
      }
    }
  };
  Stage even = load(0), odd = load(kTile);
  for (uint32_t first = 0; first < total_k; first += 2U * kTile) {
    stage_tile(even);
    even = load(first + 2U * kTile);
    consume(first);
    if (first + kTile < total_k) {
      stage_tile(odd);
      odd = load(first + 3U * kTile);
      consume(first + kTile);
    }
  }
  if (d >= head_dim)
    return;
#pragma unroll
  for (unsigned g = 0; g < kAttentionValuePairs; ++g) {
    const uint32_t pair = chunk * kAttentionPairs + first_pair + g;
    if (pair < pairs) {
      const uint32_t tok = pair / group, h = kv_h * group + pair % group;
      out[uint64_t(tok) * q_dim + uint64_t(h) * head_dim + d] = acc[g];
    }
  }
}

__global__ void dflash_noncausal_attention_kernel(
    const float* __restrict__ q, const float* __restrict__ injected_k,
    const float* __restrict__ injected_v, const float* __restrict__ block_k,
    const float* __restrict__ block_v, float* __restrict__ out,
    uint32_t current_pos, uint32_t history_length, uint32_t block_count,
    uint32_t sliding_window, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t head_dim, float scale, uint32_t history_capacity) {
  const uint32_t tok = blockIdx.x;
  const uint32_t h = blockIdx.y;
  if (tok >= block_count || h >= num_q_heads)
    return;

  const uint32_t gqa_group = num_q_heads / num_kv_heads;
  const uint32_t kv_h = h / gqa_group;
  const uint32_t q_dim = num_q_heads * head_dim;
  const uint32_t kv_dim = num_kv_heads * head_dim;
  const uint32_t query_position = current_pos + tok;
  const uint32_t history_start = current_pos + 1U > sliding_window
                                     ? current_pos + 1U - sliding_window
                                     : 0U;
  const uint32_t history_count =
      history_length > history_start ? history_length - history_start : 0U;
  const uint32_t total_k = history_count + block_count;

  const float* q_ptr = q + ((uint64_t)tok * q_dim + (uint64_t)h * head_dim);
  float* out_ptr = out + ((uint64_t)tok * q_dim + (uint64_t)h * head_dim);

  extern __shared__ float s_scores[];

  for (uint32_t local = threadIdx.x; local < history_count;
       local += blockDim.x) {
    const uint32_t position = history_start + local;
    const float* k_ptr =
        injected_k + ((uint64_t)(position % history_capacity) * kv_dim +
                      (uint64_t)kv_h * head_dim);
    float dot = 0.0F;
    for (uint32_t d = 0; d < head_dim; ++d) {
      dot += q_ptr[d] * k_ptr[d];
    }
    s_scores[local] =
        query_position - position < sliding_window ? dot * scale : -1e20F;
  }

  for (uint32_t p = threadIdx.x; p < block_count; p += blockDim.x) {
    const float* k_ptr =
        block_k + ((uint64_t)p * kv_dim + (uint64_t)kv_h * head_dim);
    float dot = 0.0F;
    for (uint32_t d = 0; d < head_dim; ++d) {
      dot += q_ptr[d] * k_ptr[d];
    }
    const uint32_t key_position = current_pos + p;
    const uint32_t distance = query_position > key_position
                                  ? query_position - key_position
                                  : key_position - query_position;
    s_scores[history_count + p] =
        distance < sliding_window ? dot * scale : -1e20F;
  }
  __syncthreads();

  // Find max score
  __shared__ float s_reduce[256];
  float local_max = -1e20F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
    local_max = fmaxf(local_max, s_scores[p]);
  }
  s_reduce[threadIdx.x] = local_max;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_reduce[threadIdx.x] =
          fmaxf(s_reduce[threadIdx.x], s_reduce[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_val = s_reduce[0];
  __syncthreads();

  // Exponentiate & sum denominator
  float local_sum = 0.0F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
    const float exp_score = expf(s_scores[p] - max_val);
    s_scores[p] = exp_score;
    local_sum += exp_score;
  }
  s_reduce[threadIdx.x] = local_sum;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0F / (s_reduce[0] + 1e-9F);
  __syncthreads();

  // Normalize scores
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
    s_scores[p] *= inv_sum;
  }
  __syncthreads();

  // Multiply by V
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0F;
    for (uint32_t local = 0; local < history_count; ++local) {
      const uint32_t position = history_start + local;
      const float* v_ptr =
          injected_v + ((uint64_t)(position % history_capacity) * kv_dim +
                        (uint64_t)kv_h * head_dim);
      acc += s_scores[local] * v_ptr[d];
    }
    for (uint32_t p = 0; p < block_count; ++p) {
      const float* v_ptr =
          block_v + ((uint64_t)p * kv_dim + (uint64_t)kv_h * head_dim);
      acc += s_scores[history_count + p] * v_ptr[d];
    }
    out_ptr[d] = acc;
  }
}

constexpr uint32_t kSelectorThreads = 128;
constexpr uint32_t kMaxSelectorTopK = 16;
constexpr uint32_t kSelectorTokensPerBlock = 2048;
/// Threads that cooperate on one candidate's low-rank transition score.
constexpr uint32_t kSelectorTransitionLanes =
    kSelectorThreads / kMaxSelectorTopK;

__device__ __forceinline__ void InsertSelectorCandidate(
    float score, uint32_t id, float* scores, uint32_t* ids,
    uint32_t selector_top_k) {
  if (score <= scores[selector_top_k - 1U]) {
    return;
  }
  uint32_t insertion = selector_top_k - 1U;
  while (insertion > 0U && score > scores[insertion - 1U]) {
    scores[insertion] = scores[insertion - 1U];
    ids[insertion] = ids[insertion - 1U];
    --insertion;
  }
  scores[insertion] = score;
  ids[insertion] = id;
}

/// Tree-merges the per-thread descending top-k lists held in shared memory into
/// slot zero. The previous implementation had thread zero fold all
/// kSelectorThreads lists serially, which dominated both selector kernels; a
/// pairwise merge does the same work in log2(threads) rounds. Ties keep the
/// lower slot, which is the same preference the serial fold had.
__device__ void MergeSelectorLists(float* scores, uint32_t* ids,
                                   uint32_t selector_top_k) {
  for (uint32_t stride = kSelectorThreads / 2U; stride > 0U; stride >>= 1U) {
    __syncthreads();
    if (threadIdx.x < stride) {
      float* mine_scores = scores + (threadIdx.x * kMaxSelectorTopK);
      uint32_t* mine_ids = ids + (threadIdx.x * kMaxSelectorTopK);
      const float* other_scores =
          scores + ((threadIdx.x + stride) * kMaxSelectorTopK);
      const uint32_t* other_ids =
          ids + ((threadIdx.x + stride) * kMaxSelectorTopK);
      float merged_scores[kMaxSelectorTopK];
      uint32_t merged_ids[kMaxSelectorTopK];
      uint32_t mine = 0;
      uint32_t other = 0;
      for (uint32_t index = 0; index < selector_top_k; ++index) {
        const bool take_mine =
            other >= selector_top_k ||
            (mine < selector_top_k && mine_scores[mine] >= other_scores[other]);
        if (take_mine) {
          merged_scores[index] = mine_scores[mine];
          merged_ids[index] = mine_ids[mine];
          ++mine;
        } else {
          merged_scores[index] = other_scores[other];
          merged_ids[index] = other_ids[other];
          ++other;
        }
      }
      for (uint32_t index = 0; index < selector_top_k; ++index) {
        mine_scores[index] = merged_scores[index];
        mine_ids[index] = merged_ids[index];
      }
    }
  }
  __syncthreads();
}

__global__ void dflash_selector_partial_topk_kernel(
    const float* __restrict__ logits, float* __restrict__ partial_scores,
    uint32_t* __restrict__ partial_ids, uint32_t vocab_size,
    uint32_t selector_top_k) {
  logits += uint64_t(blockIdx.y) * vocab_size;
  partial_scores += uint64_t(blockIdx.y) * gridDim.x * selector_top_k;
  partial_ids += uint64_t(blockIdx.y) * gridDim.x * selector_top_k;
  __shared__ float shared_scores[kSelectorThreads * kMaxSelectorTopK];
  __shared__ uint32_t shared_ids[kSelectorThreads * kMaxSelectorTopK];

  float local_scores[kMaxSelectorTopK];
  uint32_t local_ids[kMaxSelectorTopK];
#pragma unroll
  for (uint32_t index = 0; index < kMaxSelectorTopK; ++index) {
    local_scores[index] = -INFINITY;
    local_ids[index] = 0;
  }

  const uint32_t begin = blockIdx.x * kSelectorTokensPerBlock;
  const uint32_t requested_end = begin + kSelectorTokensPerBlock;
  const uint32_t end = requested_end < vocab_size ? requested_end : vocab_size;
  for (uint32_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
    InsertSelectorCandidate(logits[token], token, local_scores, local_ids,
                            selector_top_k);
  }

  const uint32_t shared_offset = threadIdx.x * kMaxSelectorTopK;
#pragma unroll
  for (uint32_t index = 0; index < kMaxSelectorTopK; ++index) {
    shared_scores[shared_offset + index] = local_scores[index];
    shared_ids[shared_offset + index] = local_ids[index];
  }

  MergeSelectorLists(shared_scores, shared_ids, selector_top_k);

  const uint32_t output_offset = blockIdx.x * selector_top_k;
  for (uint32_t index = threadIdx.x; index < selector_top_k;
       index += blockDim.x) {
    partial_scores[output_offset + index] = shared_scores[index];
    partial_ids[output_offset + index] = shared_ids[index];
  }
}

__device__ __forceinline__ void FinalizeSelector(
    const float* __restrict__ partial_scores,
    const uint32_t* __restrict__ partial_ids,
    const float* __restrict__ projected_hidden,
    const hip_bfloat16* __restrict__ predecessor_codebook,
    const hip_bfloat16* __restrict__ successor_codebook,
    const uint32_t* __restrict__ predecessor_token, uint32_t* output_token,
    float* __restrict__ output_confidence, float temperature,
    const float* __restrict__ sample_uniform,
    uint32_t* __restrict__ candidate_ids,
    float* __restrict__ candidate_probabilities, uint32_t partial_count,
    uint32_t selector_rank, uint32_t selector_top_k,
    DFlashSampling sampling = {},
    // Earlier iterations of this kernel write this history. Reload it for
    // each conditional q; cached read-only loads break sampled replay.
    const volatile uint32_t* previous_tokens = nullptr,
    uint32_t previous_count = 0) {
  __shared__ float shared_scores[kSelectorThreads * kMaxSelectorTopK];
  __shared__ uint32_t shared_ids[kSelectorThreads * kMaxSelectorTopK];
  __shared__ float
      shared_transition[kMaxSelectorTopK * kSelectorTransitionLanes];

  float local_scores[kMaxSelectorTopK];
  uint32_t local_ids[kMaxSelectorTopK];
#pragma unroll
  for (uint32_t index = 0; index < kMaxSelectorTopK; ++index) {
    local_scores[index] = -INFINITY;
    local_ids[index] = 0;
  }
  const uint32_t candidate_count = partial_count * selector_top_k;
  for (uint32_t index = threadIdx.x; index < candidate_count;
       index += blockDim.x) {
    InsertSelectorCandidate(partial_scores[index], partial_ids[index],
                            local_scores, local_ids, selector_top_k);
  }

  const uint32_t shared_offset = threadIdx.x * kMaxSelectorTopK;
#pragma unroll
  for (uint32_t index = 0; index < kMaxSelectorTopK; ++index) {
    shared_scores[shared_offset + index] = local_scores[index];
    shared_ids[shared_offset + index] = local_ids[index];
  }

  MergeSelectorLists(shared_scores, shared_ids, selector_top_k);

  if (sampling.row_ids != nullptr) {
    // Compact head rows to token ids, once, before anything reads them.
    __syncthreads();
    if (threadIdx.x < selector_top_k)
      shared_ids[threadIdx.x] =
          static_cast<uint32_t>(sampling.row_ids[shared_ids[threadIdx.x]]);
    __syncthreads();
  }
  const uint32_t predecessor = *predecessor_token;
  const hip_bfloat16* predecessor_row =
      predecessor_codebook + (uint64_t)predecessor * selector_rank;

  // Score the low-rank transitions in parallel: kSelectorTransitionLanes
  // threads share each candidate and reduce through shared memory. Thread zero
  // used to walk all selector_top_k x selector_rank products by itself.
  {
    const uint32_t candidate_index = threadIdx.x / kSelectorTransitionLanes;
    const uint32_t part = threadIdx.x % kSelectorTransitionLanes;
    if (candidate_index < selector_top_k) {
      const uint32_t candidate = shared_ids[candidate_index];
      const hip_bfloat16* successor_row =
          successor_codebook + (uint64_t)candidate * selector_rank;
      float partial = 0.0F;
      for (uint32_t rank = part; rank < selector_rank;
           rank += kSelectorTransitionLanes) {
        partial += static_cast<float>(predecessor_row[rank]) *
                   projected_hidden[rank] *
                   static_cast<float>(successor_row[rank]);
      }
      shared_transition[(candidate_index * kSelectorTransitionLanes) + part] =
          partial;
    }
  }
  __syncthreads();

  if (threadIdx.x != 0U) {
    return;
  }

  float top_scores[kMaxSelectorTopK];
  uint32_t top_ids[kMaxSelectorTopK];
  for (uint32_t index = 0; index < selector_top_k; ++index) {
    top_scores[index] = shared_scores[index];
    top_ids[index] = shared_ids[index];
  }

  float best_score = -INFINITY;
  uint32_t best_index = 0;
  float path_scores[kMaxSelectorTopK];
  for (uint32_t candidate_index = 0; candidate_index < selector_top_k;
       ++candidate_index) {
    float transition = 0.0F;
    for (uint32_t part = 0; part < kSelectorTransitionLanes; ++part) {
      transition +=
          shared_transition[(candidate_index * kSelectorTransitionLanes) +
                            part];
    }
    float score = top_scores[candidate_index] + transition;
    const auto id = top_ids[candidate_index];
    if (sampling.frequency_penalty != 0.0F ||
        sampling.presence_penalty != 0.0F || sampling.repeat_penalty != 1.0F) {
      uint32_t generated = 0;
      uint32_t low = 0, high = sampling.penalty_count;
      while (low < high) {
        const auto middle = low + (high - low) / 2;
        if (sampling.penalties[middle].token < id)
          low = middle + 1;
        else
          high = middle;
      }
      if (low < sampling.penalty_count && sampling.penalties[low].token == id)
        generated = sampling.penalties[low].generated_count;
      for (uint32_t i = 0; i < previous_count; ++i)
        generated += previous_tokens[i] == id;
      bool repeated = false;
      if (sampling.repeat_penalty != 1.0F && sampling.repeat_last_n > 0) {
        const auto retained = min(sampling.history_count,
                                  sampling.repeat_last_n > previous_count
                                      ? sampling.repeat_last_n - previous_count
                                      : 0U);
        for (uint32_t i = sampling.history_count - retained;
             i < sampling.history_count; ++i)
          repeated |= sampling.history[i] == id;
        const auto first = previous_count > sampling.repeat_last_n
                               ? previous_count - sampling.repeat_last_n
                               : 0U;
        for (uint32_t i = first; i < previous_count; ++i)
          repeated |= previous_tokens[i] == id;
      }
      if (repeated)
        score = score <= 0.0F ? score * sampling.repeat_penalty
                              : score / sampling.repeat_penalty;
      score -= sampling.frequency_penalty * static_cast<float>(generated);
      if (generated != 0)
        score -= sampling.presence_penalty;
    }
    path_scores[candidate_index] = score;
    if (score > best_score) {
      best_score = score;
      best_index = candidate_index;
    }
  }

  if (candidate_ids != nullptr) {
    for (uint32_t index = 0; index < selector_top_k; ++index) {
      candidate_ids[index] = top_ids[index];
    }
  }

  if (temperature > 0.0F) {
    const float divisor = fmaxf(temperature, FLT_MIN);
    float denominator = 0.0F;
    for (uint32_t index = 0; index < selector_top_k; ++index) {
      // Subtract before division so a finite, very small temperature cannot
      // overflow both logits and their maximum to infinity.
      const float probability_weight =
          expf((path_scores[index] - best_score) / divisor);
      path_scores[index] = probability_weight;
      denominator += probability_weight;
    }
    const float inverse_denominator = 1.0F / fmaxf(denominator, 1e-30F);
    if (sampling.integer_masses != 0U) {
      constexpr uint32_t kUnits = 1U << 24;
      uint32_t masses[kMaxSelectorTopK];
      uint32_t total = 0;
      for (uint32_t index = 0; index < selector_top_k; ++index) {
        const float share = path_scores[index] * inverse_denominator;
        masses[index] = static_cast<uint32_t>(
            fminf(floorf(share * static_cast<float>(kUnits)),
                  static_cast<float>(kUnits)));
        total += masses[index];
      }
      if (total <= kUnits) {
        masses[0] += kUnits - total;
      } else {
        // Float rounding can push the floored shares past 2^24: take the
        // excess from the largest mass so the total is exactly 2^24.
        uint32_t largest = 0;
        for (uint32_t index = 1; index < selector_top_k; ++index)
          if (masses[index] > masses[largest])
            largest = index;
        masses[largest] -= total - kUnits;
      }
      const uint32_t draw = min(
          static_cast<uint32_t>(fminf(fmaxf(*sample_uniform, 0.0F),
                                      0.99999994F) *
                                static_cast<float>(kUnits)),
          kUnits - 1U);
      uint32_t cumulative = 0;
      uint32_t sampled_index = 0;
      bool selected = false;
      for (uint32_t index = 0; index < selector_top_k; ++index) {
        const float probability =
            static_cast<float>(masses[index]) * (1.0F / kUnits);
        if (candidate_probabilities != nullptr)
          candidate_probabilities[index] = probability;
        cumulative += masses[index];
        if (!selected && draw < cumulative) {
          sampled_index = index;
          selected = true;
        }
      }
      *output_token = top_ids[sampled_index];
      if (output_confidence != nullptr)
        *output_confidence =
            static_cast<float>(masses[sampled_index]) * (1.0F / kUnits);
      return;
    }
    float threshold =
        fminf(fmaxf(*sample_uniform, 0.0F), 0.99999994F) * denominator;
    uint32_t sampled_index = best_index;
    bool selected = false;
    for (uint32_t index = 0; index < selector_top_k; ++index) {
      const float probability = path_scores[index] * inverse_denominator;
      if (candidate_probabilities != nullptr) {
        candidate_probabilities[index] = probability;
      }
      if (!selected && probability > 0.0F)
        sampled_index = index;
      threshold -= path_scores[index];
      if (threshold < 0.0F && !selected) {
        sampled_index = index;
        selected = true;
      }
    }
    *output_token = top_ids[sampled_index];
    if (output_confidence != nullptr) {
      *output_confidence = path_scores[sampled_index] * inverse_denominator;
    }
  } else {
    *output_token = top_ids[best_index];
    if (candidate_probabilities != nullptr) {
      for (uint32_t index = 0; index < selector_top_k; ++index)
        candidate_probabilities[index] = index == best_index ? 1.0F : 0.0F;
    }
    if (output_confidence != nullptr) {
      // The argmax's softmax share over the top-k path scores (temperature
      // 1): the greedy chain's gate reads it; the token is unchanged.
      float denominator = 0.0F;
      for (uint32_t index = 0; index < selector_top_k; ++index)
        denominator += expf(path_scores[index] - best_score);
      *output_confidence = 1.0F / fmaxf(denominator, 1.0F);
    }
  }
}

__global__ void dflash_selector_finalize_kernel(
    const float* partial_scores, const uint32_t* partial_ids,
    const float* projected_hidden, const hip_bfloat16* predecessor_codebook,
    const hip_bfloat16* successor_codebook, const uint32_t* predecessor_token,
    uint32_t* output_token, float* output_confidence, float temperature,
    const float* sample_uniform, uint32_t* candidate_ids,
    float* candidate_probabilities, uint32_t partial_count,
    uint32_t selector_rank, uint32_t selector_top_k) {
  FinalizeSelector(partial_scores, partial_ids, projected_hidden,
                   predecessor_codebook, successor_codebook, predecessor_token,
                   output_token, output_confidence, temperature, sample_uniform,
                   candidate_ids, candidate_probabilities, partial_count,
                   selector_rank, selector_top_k);
}

struct SelectorBatch {
  DFlashSelectorSequence sequences[8];
  uint32_t row_offsets[8];
  uint32_t token_offsets[8];
};

__global__ void dflash_selector_finalize_batch_kernel(
    const float* partial_scores, const uint32_t* partial_ids,
    const float* projected_hidden, const hip_bfloat16* predecessor_codebook,
    const hip_bfloat16* successor_codebook, uint32_t* tokens,
    float* confidences, const float* uniforms, uint32_t* candidate_ids,
    float* candidate_probabilities, SelectorBatch batch, uint32_t partial_count,
    uint32_t selector_rank, uint32_t selector_top_k) {
  const auto sequence = batch.sequences[blockIdx.x];
  __shared__ uint32_t predecessor;
  if (threadIdx.x == 0U)
    predecessor = tokens[batch.token_offsets[blockIdx.x]];
  __syncthreads();
  for (uint32_t step = 0; step < sequence.count; ++step) {
    const uint32_t row = batch.row_offsets[blockIdx.x] + step;
    const uint32_t token = batch.token_offsets[blockIdx.x] + step;
    const uint64_t partial = uint64_t(row) * partial_count * selector_top_k;
    const uint64_t candidate = uint64_t(row) * selector_top_k;
    FinalizeSelector(
        partial_scores + partial, partial_ids + partial,
        projected_hidden + uint64_t(row) * selector_rank, predecessor_codebook,
        successor_codebook, &predecessor, tokens + token + 1U,
        confidences == nullptr ? nullptr : confidences + row,
        sequence.temperature,
        sequence.temperature > 0.0F ? uniforms + row : nullptr,
        candidate_ids != nullptr ? candidate_ids + candidate : nullptr,
        candidate_probabilities != nullptr ? candidate_probabilities + candidate
                                           : nullptr,
        partial_count, selector_rank, selector_top_k, sequence.sampling,
        tokens + batch.token_offsets[blockIdx.x] + 1U, step);
    // Thread zero publishes the predecessor for this sequence's next step.
    // Other sequences own separate token, probability and partial buffers.
    if (threadIdx.x == 0U)
      predecessor = tokens[token + 1U];
    __syncthreads();
  }
}

void LaunchDFlashGroupedDynamicConv(
    const float* input, const float* dynamic_coefficients,
    const float* base_kernel, float* output, std::uint32_t num_tokens,
    std::uint32_t hidden_size, std::uint32_t kernel_size,
    std::uint32_t group_size, std::uint32_t side, hipStream_t stream) {
  const uint64_t total = static_cast<uint64_t>(num_tokens) * hidden_size;
  const uint32_t blocks = static_cast<uint32_t>((total + 255U) / 256U);
  dflash_grouped_dynamic_conv_kernel<<<blocks, 256, 0, stream>>>(
      input, dynamic_coefficients, base_kernel, output, num_tokens, hidden_size,
      kernel_size, group_size, side);
}

void LaunchDFlashSiLUMul(float* gate, const float* up,
                         std::uint32_t total_elements, hipStream_t stream) {
  const uint32_t blocks = (total_elements + 255u) / 256u;
  dflash_silu_mul_kernel<<<blocks, 256, 0, stream>>>(gate, up, total_elements);
}

namespace {

/// Shared launch geometry for both non-causal attention routes: one workgroup
/// per (draft token, query head) and one shared score slot per attended key.
struct DFlashAttentionLaunch {
  dim3 grid;
  std::size_t shared_bytes;
};

[[nodiscard]] DFlashAttentionLaunch DFlashAttentionGeometry(
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads) noexcept {
  const std::uint32_t history_start = current_pos + 1U > sliding_window
                                          ? current_pos + 1U - sliding_window
                                          : 0U;
  const std::uint32_t history_count =
      history_length > history_start ? history_length - history_start : 0U;
  return {.grid = dim3(block_count, num_q_heads),
          .shared_bytes = (history_count + block_count) * sizeof(float)};
}

}  // namespace

void LaunchDFlashNonCausalAttentionScalar(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity) {
  const auto geometry = DFlashAttentionGeometry(
      current_pos, history_length, block_count, sliding_window, num_q_heads);
  dflash_noncausal_attention_kernel<<<geometry.grid, 256, geometry.shared_bytes,
                                      stream>>>(
      q, injected_k, injected_v, block_k, block_v, out, current_pos,
      history_length, block_count, sliding_window, num_q_heads, num_kv_heads,
      head_dim, scale, history_capacity);
}

std::size_t DFlashAttentionScratchElements(
    std::uint32_t block_size, std::uint32_t num_q_heads,
    std::uint32_t num_kv_heads, std::uint32_t history_capacity) noexcept {
  const std::size_t group = num_q_heads / num_kv_heads;
  const std::size_t chunks =
      (block_size * group + kAttentionPairs - 1U) / kAttentionPairs;
  return static_cast<std::size_t>(num_kv_heads) * chunks *
         (static_cast<std::size_t>(history_capacity) + block_size) *
         kAttentionPairs;
}

void LaunchDFlashNonCausalAttentionPairs(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity, float* scores,
    std::size_t scores_elements) {
  const std::uint32_t history_start = current_pos + 1U > sliding_window
                                          ? current_pos + 1U - sliding_window
                                          : 0U;
  const std::uint32_t history_count =
      history_length > history_start ? history_length - history_start : 0U;
  const std::uint32_t group = num_q_heads / num_kv_heads;
  const std::uint32_t chunks =
      (block_count * group + kAttentionPairs - 1U) / kAttentionPairs;
  const std::uint32_t total_k = history_count + block_count;
  if (scores == nullptr || static_cast<std::size_t>(num_kv_heads) * chunks *
                                   total_k * kAttentionPairs >
                               scores_elements)
    throw std::invalid_argument("DFlash attention score scratch is too small");
  dflash_attention_scores_kernel<<<
      dim3(num_kv_heads,
           (total_k + kAttentionScoreKeys - 1U) / kAttentionScoreKeys, chunks),
      kAttentionThreads, 0, stream>>>(
      q, injected_k, block_k, scores, current_pos, history_start,
      history_count, block_count, sliding_window, num_q_heads, num_kv_heads,
      head_dim, scale, history_capacity);
  dflash_attention_softmax_kernel<<<
      dim3(kAttentionPairs / kAttentionSoftmaxPairs, num_kv_heads, chunks),
      kAttentionThreads, 0, stream>>>(scores, history_count, block_count,
                                      group);
  dflash_attention_values_kernel<<<dim3((head_dim + 31U) / 32U, num_kv_heads,
                                        chunks),
                                   kAttentionThreads, 0, stream>>>(
      scores, injected_v, block_v, out, history_start, history_count,
      block_count, num_q_heads, num_kv_heads, head_dim, history_capacity);
}

void LaunchDFlashNonCausalAttention(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity, float* scores,
    std::size_t scores_elements) {
  // One float4 per lane covers all shipped DFlash2 head dimensions.
  if ((head_dim % 4U) == 0U && head_dim <= 128U &&
      (num_q_heads % num_kv_heads) == 0U) {
    LaunchDFlashNonCausalAttentionPairs(
        q, injected_k, injected_v, block_k, block_v, out, current_pos,
        history_length, block_count, sliding_window, num_q_heads, num_kv_heads,
        head_dim, scale, stream, history_capacity, scores, scores_elements);
    return;
  }
  LaunchDFlashNonCausalAttentionScalar(
      q, injected_k, injected_v, block_k, block_v, out, current_pos,
      history_length, block_count, sliding_window, num_q_heads, num_kv_heads,
      head_dim, scale, stream, history_capacity);
}

std::size_t DFlashSelectorScratchElements(std::uint32_t vocab_size) noexcept {
  const std::size_t partial_count =
      (static_cast<std::size_t>(vocab_size) + kSelectorTokensPerBlock - 1U) /
      kSelectorTokensPerBlock;
  return partial_count * kMaxSelectorTopK;
}

void LaunchDFlashSelectorStep(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    const std::uint32_t* predecessor_token, std::uint32_t* output_token,
    float* output_confidence, float* partial_scores, std::uint32_t* partial_ids,
    float temperature, const float* sample_uniform,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::uint32_t vocab_size, std::uint32_t selector_rank,
    std::uint32_t selector_top_k, hipStream_t stream) {
  if (partial_scores == nullptr || partial_ids == nullptr || vocab_size == 0 ||
      selector_top_k == 0 || selector_top_k > kMaxSelectorTopK ||
      selector_top_k > vocab_size) {
    throw std::invalid_argument("invalid DFlash selector shape or scratch");
  }
  const std::uint32_t partial_count =
      (vocab_size + kSelectorTokensPerBlock - 1U) / kSelectorTokensPerBlock;
  dflash_selector_partial_topk_kernel<<<partial_count, kSelectorThreads, 0,
                                        stream>>>(
      logits, partial_scores, partial_ids, vocab_size, selector_top_k);
  dflash_selector_finalize_kernel<<<1, kSelectorThreads, 0, stream>>>(
      partial_scores, partial_ids, projected_hidden,
      static_cast<const hip_bfloat16*>(predecessor_codebook_bf16),
      static_cast<const hip_bfloat16*>(successor_codebook_bf16),
      predecessor_token, output_token, output_confidence, temperature,
      sample_uniform, candidate_ids, candidate_probabilities, partial_count,
      selector_rank, selector_top_k);
}

void LaunchDFlashSelectorBatch(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    std::uint32_t* tokens, float* confidences, float* partial_scores,
    std::uint32_t* partial_ids, const float* uniforms,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::span<const DFlashSelectorSequence> sequences, std::uint32_t vocab_size,
    std::uint32_t selector_rank, std::uint32_t selector_top_k,
    hipStream_t stream) {
  if (sequences.empty() || sequences.size() > 8 || logits == nullptr ||
      projected_hidden == nullptr || predecessor_codebook_bf16 == nullptr ||
      successor_codebook_bf16 == nullptr || tokens == nullptr ||
      partial_scores == nullptr || partial_ids == nullptr || vocab_size == 0 ||
      selector_rank == 0 || selector_top_k == 0 ||
      selector_top_k > kMaxSelectorTopK || selector_top_k > vocab_size)
    throw std::invalid_argument("invalid DFlash selector batch or scratch");
  SelectorBatch batch{};
  std::uint32_t rows = 0;
  for (std::size_t index = 0; index < sequences.size(); ++index) {
    const auto sequence = sequences[index];
    if (sequence.count == 0 ||
        sequence.count >
            std::numeric_limits<std::uint32_t>::max() - 8U - rows ||
        !std::isfinite(sequence.temperature) || sequence.temperature < 0.0F ||
        (sequence.temperature > 0.0F && uniforms == nullptr))
      throw std::invalid_argument("invalid DFlash selector sequence");
    batch.sequences[index] = sequence;
    batch.row_offsets[index] = rows;
    batch.token_offsets[index] = rows + static_cast<std::uint32_t>(index);
    rows += sequence.count;
  }
  const auto partial_count = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(vocab_size) + kSelectorTokensPerBlock - 1U) /
      kSelectorTokensPerBlock);
  dflash_selector_partial_topk_kernel<<<dim3(partial_count, rows),
                                        kSelectorThreads, 0, stream>>>(
      logits, partial_scores, partial_ids, vocab_size, selector_top_k);
  dflash_selector_finalize_batch_kernel<<<sequences.size(), kSelectorThreads, 0,
                                          stream>>>(
      partial_scores, partial_ids, projected_hidden,
      static_cast<const hip_bfloat16*>(predecessor_codebook_bf16),
      static_cast<const hip_bfloat16*>(successor_codebook_bf16), tokens,
      confidences, uniforms, candidate_ids, candidate_probabilities, batch,
      partial_count, selector_rank, selector_top_k);
}

}  // namespace gufo::hip::kernels
#endif  // defined(ENGINE_ENABLE_HIP)
