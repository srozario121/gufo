// dflash_attn_bench: the DFlash-2 drafter's non-causal block attention. The
// production route against the former one-workgroup-per-(token, head pair)
// kernel, kept here verbatim: byte comparison over ragged shapes, then timing
// with warm caches and with K/V rotated through more layers than the MALL
// holds.
//
// Build it with the kernel library's flags, so both routes compile as in the
// engine:
//
//   hipcc -O2 -g -DNDEBUG -std=c++20 --offload-arch=gfx1151 -I. \
//     tools/bench/dflash_attn_bench.hip -o /tmp/dflash_attn_bench
#ifndef ENGINE_ENABLE_HIP
#define ENGINE_ENABLE_HIP 1
#endif
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "src/models/qwen/hip/kernels/dflash_kernels.hip"

namespace former {
template <uint32_t QueryHeads>
__global__ void dflash_noncausal_attention_wave_kernel(
    const float* __restrict__ q, const float* __restrict__ injected_k,
    const float* __restrict__ injected_v, const float* __restrict__ block_k,
    const float* __restrict__ block_v, float* __restrict__ out,
    uint32_t current_pos, uint32_t history_length, uint32_t block_count,
    uint32_t sliding_window, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t head_dim, float scale, uint32_t history_capacity) {
  constexpr uint32_t kPrefetch = 32;
  const uint32_t tok = blockIdx.x, h = blockIdx.y * QueryHeads;
  if (tok >= block_count || h + QueryHeads > num_q_heads)
    return;
  const uint32_t kv_h = h / (num_q_heads / num_kv_heads),
                 q_dim = num_q_heads * head_dim,
                 kv_dim = num_kv_heads * head_dim;
  const uint32_t query_position = current_pos + tok;
  const uint32_t history_start = current_pos + 1U > sliding_window
                                     ? current_pos + 1U - sliding_window
                                     : 0U;
  const uint32_t history_count = history_length > history_start
                                     ? history_length - history_start
                                     : 0U,
                 total_k = history_count + block_count;
  const float4* q_vec = reinterpret_cast<const float4*>(
      q + uint64_t(tok) * q_dim + uint64_t(h) * head_dim);
  const uint32_t num_vec = head_dim / 4U, lane = threadIdx.x & 31U,
                 wave = threadIdx.x >> 5U, waves = blockDim.x >> 5U;
  extern __shared__ float scores[];
  __shared__ float reduce[QueryHeads][256];
  // A wave scores kKeys history keys at a time: their row loads are issued
  // together so the latency overlaps, then each key keeps its own lane
  // partials, FMA expression and xor reduction exactly as when scored alone.
  constexpr uint32_t kKeys = 4;
  const bool one_vec_per_lane = num_vec == 32U;
  uint32_t local = wave;
  for (; one_vec_per_lane && local + (kKeys - 1U) * waves < history_count;
       local += kKeys * waves) {
    float4 kv[kKeys];
#pragma unroll
    for (unsigned j = 0; j < kKeys; ++j) {
      const uint32_t position = history_start + local + j * waves;
      kv[j] = reinterpret_cast<const float4*>(
          injected_k + uint64_t(position % history_capacity) * kv_dim +
          uint64_t(kv_h) * head_dim)[lane];
    }
#pragma unroll
    for (unsigned j = 0; j < kKeys; ++j) {
      const uint32_t key_local = local + j * waves;
      const uint32_t position = history_start + key_local;
      float dot[QueryHeads] = {};
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g) {
        const float4 qv = q_vec[g * num_vec + lane];
        dot[g] += (qv.x * kv[j].x) + (qv.y * kv[j].y) + (qv.z * kv[j].z) +
                  (qv.w * kv[j].w);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g)
          dot[g] += __shfl_xor(dot[g], offset);
      }
      if (lane == 0) {
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g)
          scores[g * total_k + key_local] =
              query_position - position < sliding_window ? dot[g] * scale
                                                         : -1e20F;
      }
    }
  }
  for (; local < history_count; local += waves) {
    const uint32_t position = history_start + local;
    const float4* key = reinterpret_cast<const float4*>(
        injected_k + uint64_t(position % history_capacity) * kv_dim +
        uint64_t(kv_h) * head_dim);
    float dot[QueryHeads] = {};
    for (uint32_t d = lane; d < num_vec; d += 32U) {
      const float4 kv = key[d];
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g) {
        const float4 qv = q_vec[g * num_vec + d];
        dot[g] += (qv.x * kv.x) + (qv.y * kv.y) + (qv.z * kv.z) + (qv.w * kv.w);
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        dot[g] += __shfl_xor(dot[g], offset);
    }
    if (lane == 0) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        scores[g * total_k + local] = query_position - position < sliding_window
                                          ? dot[g] * scale
                                          : -1e20F;
    }
  }
  for (uint32_t p = wave; p < block_count; p += waves) {
    const float4* key = reinterpret_cast<const float4*>(
        block_k + uint64_t(p) * kv_dim + uint64_t(kv_h) * head_dim);
    float dot[QueryHeads] = {};
    for (uint32_t d = lane; d < num_vec; d += 32U) {
      const float4 kv = key[d];
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g) {
        const float4 qv = q_vec[g * num_vec + d];
        dot[g] += (qv.x * kv.x) + (qv.y * kv.y) + (qv.z * kv.z) + (qv.w * kv.w);
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        dot[g] += __shfl_xor(dot[g], offset);
    }
    if (lane == 0) {
      const uint32_t key_position = current_pos + p;
      const uint32_t distance = query_position > key_position
                                    ? query_position - key_position
                                    : key_position - query_position;
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        scores[g * total_k + history_count + p] =
            distance < sliding_window ? dot[g] * scale : -1e20F;
    }
  }
  __syncthreads();
  float maxima[QueryHeads];
#pragma unroll
  for (unsigned g = 0; g < QueryHeads; ++g)
    maxima[g] = -1e20F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g)
      maxima[g] = fmaxf(maxima[g], scores[g * total_k + p]);
  }
#pragma unroll
  for (unsigned g = 0; g < QueryHeads; ++g)
    reduce[g][threadIdx.x] = maxima[g];
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        reduce[g][threadIdx.x] =
            fmaxf(reduce[g][threadIdx.x], reduce[g][threadIdx.x + stride]);
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned g = 0; g < QueryHeads; ++g)
    maxima[g] = reduce[g][0];
  __syncthreads();
  float sums[QueryHeads] = {};
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g) {
      const float e = expf(scores[g * total_k + p] - maxima[g]);
      scores[g * total_k + p] = e;
      sums[g] += e;
    }
  }
#pragma unroll
  for (unsigned g = 0; g < QueryHeads; ++g)
    reduce[g][threadIdx.x] = sums[g];
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        reduce[g][threadIdx.x] += reduce[g][threadIdx.x + stride];
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned g = 0; g < QueryHeads; ++g)
    sums[g] = 1.f / (reduce[g][0] + 1e-9F);
  __syncthreads();
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g)
      scores[g * total_k + p] *= sums[g];
  }
  __syncthreads();
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc[QueryHeads] = {};
    uint32_t slot = history_count > 0 ? history_start % history_capacity : 0;
    // Issue independent value loads together, then retain the original
    // ascending-key FMA sequence. Advancing the ring slot avoids a division
    // for every value without changing the persistent KV layout.
    uint32_t local = 0;
    for (; history_count - local >= kPrefetch; local += kPrefetch) {
      float values[kPrefetch];
#pragma unroll
      for (unsigned i = 0; i < kPrefetch; ++i) {
        values[i] =
            injected_v[uint64_t(slot) * kv_dim + uint64_t(kv_h) * head_dim + d];
        slot = slot + 1 == history_capacity ? 0 : slot + 1;
      }
#pragma unroll
      for (unsigned i = 0; i < kPrefetch; ++i) {
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g)
          acc[g] += scores[g * total_k + local + i] * values[i];
      }
    }
    for (; local < history_count; ++local) {
      const float value =
          injected_v[uint64_t(slot) * kv_dim + uint64_t(kv_h) * head_dim + d];
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        acc[g] += scores[g * total_k + local] * value;
      slot = slot + 1 == history_capacity ? 0 : slot + 1;
    }
    for (uint32_t p = 0; p < block_count; ++p) {
      const float value =
          block_v[uint64_t(p) * kv_dim + uint64_t(kv_h) * head_dim + d];
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        acc[g] += scores[g * total_k + history_count + p] * value;
    }
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g)
      out[uint64_t(tok) * q_dim + uint64_t(h + g) * head_dim + d] = acc[g];
  }
}

void Launch(const float* q, const float* injected_k, const float* injected_v,
            const float* block_k, const float* block_v, float* out,
            uint32_t current_pos, uint32_t history_length,
            uint32_t block_count, uint32_t sliding_window,
            uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim,
            float scale, hipStream_t stream, uint32_t history_capacity) {
  const uint32_t history_start = current_pos + 1U > sliding_window
                                     ? current_pos + 1U - sliding_window
                                     : 0U;
  const uint32_t history_count =
      history_length > history_start ? history_length - history_start : 0U;
  dim3 grid(block_count, num_q_heads);
  const size_t shared_bytes = (history_count + block_count) * sizeof(float);
  const auto heads_per_kv = num_q_heads / num_kv_heads;
  if (block_count >= 4 && (heads_per_kv % 2U) == 0U &&
      shared_bytes <= (64U * 1024U - 2U * 256U * sizeof(float)) / 2U) {
    grid.y /= 2U;
    dflash_noncausal_attention_wave_kernel<2>
        <<<grid, 256, 2U * shared_bytes, stream>>>(
            q, injected_k, injected_v, block_k, block_v, out, current_pos,
            history_length, block_count, sliding_window, num_q_heads,
            num_kv_heads, head_dim, scale, history_capacity);
    return;
  }
  dflash_noncausal_attention_wave_kernel<1>
      <<<grid, 256, shared_bytes, stream>>>(
          q, injected_k, injected_v, block_k, block_v, out, current_pos,
          history_length, block_count, sliding_window, num_q_heads,
          num_kv_heads, head_dim, scale, history_capacity);
}

}  // namespace former

namespace {

constexpr uint32_t kHeads = 32, kKvHeads = 8, kDim = 128, kWindow = 2048,
                   kCapacity = 2048, kBlock = 8, kLayers = 5;
constexpr uint32_t kQDim = kHeads * kDim, kKvDim = kKvHeads * kDim;

void Check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "HIP error: %s\n", hipGetErrorString(error));
    std::exit(2);
  }
}

float* Upload(const std::vector<float>& host) {
  float* device = nullptr;
  Check(hipMalloc(&device, host.size() * sizeof(float)));
  Check(hipMemcpy(device, host.data(), host.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  return device;
}

std::vector<float> Fill(std::mt19937& rng, size_t count, float sigma) {
  std::normal_distribution<float> normal(0.0F, sigma);
  std::vector<float> values(count);
  for (auto& value : values)
    value = normal(rng);
  return values;
}

struct Operands {
  float *q, *k[kLayers], *v[kLayers], *block_k, *block_v, *out, *scores;
  size_t score_elements;
};

Operands Make(uint32_t seed, float q_sigma) {
  std::mt19937 rng(seed);
  Operands o{};
  o.q = Upload(Fill(rng, size_t(kBlock) * kQDim, q_sigma));
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    o.k[layer] = Upload(Fill(rng, size_t(kCapacity) * kKvDim, 1.0F));
    o.v[layer] = Upload(Fill(rng, size_t(kCapacity) * kKvDim, 1.0F));
  }
  o.block_k = Upload(Fill(rng, size_t(kBlock) * kKvDim, 1.0F));
  o.block_v = Upload(Fill(rng, size_t(kBlock) * kKvDim, 1.0F));
  Check(hipMalloc(&o.out, size_t(kBlock) * kQDim * sizeof(float)));
  o.score_elements = gufo::hip::kernels::DFlashAttentionScratchElements(
      kBlock, kHeads, kKvHeads, kCapacity);
  Check(hipMalloc(&o.scores, o.score_elements * sizeof(float)));
  return o;
}

const float kScale = 1.0F / sqrtf(float(kDim));

void RunFormer(const Operands& o, uint32_t layer, uint32_t pos,
               uint32_t history, uint32_t rows) {
  former::Launch(o.q, o.k[layer], o.v[layer], o.block_k, o.block_v, o.out,
                 pos, history, rows, kWindow, kHeads, kKvHeads, kDim, kScale,
                 nullptr, kCapacity);
}

void RunCurrent(const Operands& o, uint32_t layer, uint32_t pos,
                uint32_t history, uint32_t rows) {
  gufo::hip::kernels::LaunchDFlashNonCausalAttention(
      o.q, o.k[layer], o.v[layer], o.block_k, o.block_v, o.out, pos, history,
      rows, kWindow, kHeads, kKvHeads, kDim, kScale, nullptr, kCapacity,
      o.scores, o.score_elements);
}

std::vector<float> Output(const Operands& o, uint32_t rows) {
  std::vector<float> host(size_t(rows) * kQDim);
  Check(hipDeviceSynchronize());
  Check(hipMemcpy(host.data(), o.out, host.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
  return host;
}

// Returns the number of output floats whose bytes differ.
size_t Compare(const Operands& o, uint32_t pos, uint32_t history,
               uint32_t rows) {
  Check(hipMemset(o.out, 0xff, size_t(kBlock) * kQDim * sizeof(float)));
  RunFormer(o, 0, pos, history, rows);
  const auto expected = Output(o, rows);
  Check(hipMemset(o.out, 0xff, size_t(kBlock) * kQDim * sizeof(float)));
  // Poison the scratch too: every score the route reads must be written.
  Check(hipMemset(o.scores, 0xff, o.score_elements * sizeof(float)));
  RunCurrent(o, 0, pos, history, rows);
  const auto actual = Output(o, rows);
  size_t differ = 0;
  for (size_t i = 0; i < expected.size(); ++i)
    differ += std::memcmp(&expected[i], &actual[i], sizeof(float)) != 0;
  return differ;
}

// Mean microseconds per call. Rotating cycles the five layers' K/V (84 MB,
// beyond the 32 MiB MALL) as the drafter does; warm repeats layer 0.
template <typename Run>
double Time(const Operands& o, Run run, bool rotating, uint32_t pos,
            uint32_t rows) {
  hipEvent_t start, stop;
  Check(hipEventCreate(&start));
  Check(hipEventCreate(&stop));
  constexpr int kCalls = 200;
  for (int i = 0; i < 20; ++i)
    run(o, rotating ? i % kLayers : 0, pos, pos, rows);
  Check(hipEventRecord(start));
  for (int i = 0; i < kCalls; ++i)
    run(o, rotating ? i % kLayers : 0, pos, pos, rows);
  Check(hipEventRecord(stop));
  Check(hipEventSynchronize(stop));
  float ms = 0;
  Check(hipEventElapsedTime(&ms, start, stop));
  return 1000.0 * ms / kCalls;
}

}  // namespace

int main() {
  size_t failures = 0, checked = 0;
  for (const float q_sigma : {1.0F, 6.0F}) {
    const Operands o = Make(q_sigma > 1.0F ? 11U : 7U, q_sigma);
    // Positions cover no history, single keys, prefetch tails, the window
    // edge, wrapped rings and a history shorter than the position.
    const uint32_t shapes[][2] = {
        {0, 0},       {1, 1},       {2, 1},       {5, 5},       {31, 31},
        {32, 32},     {33, 33},     {63, 63},     {64, 64},     {65, 65},
        {100, 97},    {1000, 1000}, {2046, 2046}, {2047, 2047}, {2048, 2048},
        {2049, 2049}, {2051, 2051}, {2055, 2055}, {3000, 3000}, {4133, 4133},
        {4133, 4130}, {26000, 26000}};
    for (const auto& shape : shapes) {
      for (uint32_t rows = 1; rows <= kBlock; ++rows) {
        const size_t differ = Compare(o, shape[0], shape[1], rows);
        ++checked;
        if (differ != 0) {
          ++failures;
          std::printf("DIFFER sigma %.0f pos %u history %u rows %u: %zu\n",
                      q_sigma, shape[0], shape[1], rows, differ);
        }
      }
    }
  }
  std::printf("byte comparison: %zu of %zu shapes differ\n", failures,
              checked);

  const Operands o = Make(7U, 1.0F);
  for (const uint32_t pos : {3000U, 1500U, 300U}) {
    for (const uint32_t rows : {8U, 4U, 1U}) {
      for (const bool rotating : {true, false}) {
        const double before = Time(o, RunFormer, rotating, pos, rows);
        const double after = Time(o, RunCurrent, rotating, pos, rows);
        const double before2 = Time(o, RunFormer, rotating, pos, rows);
        const double after2 = Time(o, RunCurrent, rotating, pos, rows);
        std::printf(
            "pos %4u rows %u %-8s former %7.1f %7.1f us  current %7.1f %7.1f "
            "us\n",
            pos, rows, rotating ? "rotating" : "warm", before, before2, after,
            after2);
      }
    }
  }
  // The three passes alone at the profiled shape, warm and rotating.
  for (const bool rotating : {false, true}) {
    namespace k = gufo::hip::kernels;
    const uint32_t pos = 3000, rows = 8, start = pos + 1 - kWindow,
                   count = pos - start, total = count + rows;
    hipEvent_t a, b;
    Check(hipEventCreate(&a));
    Check(hipEventCreate(&b));
    const auto time = [&](const char* name, auto launch) {
      for (int i = 0; i < 10; ++i)
        launch(rotating ? i % kLayers : 0);
      Check(hipEventRecord(a));
      for (int i = 0; i < 200; ++i)
        launch(rotating ? i % kLayers : 0);
      Check(hipEventRecord(b));
      Check(hipEventSynchronize(b));
      float ms = 0;
      Check(hipEventElapsedTime(&ms, a, b));
      std::printf("pass %-8s %-8s %7.1f us\n", name,
                  rotating ? "rotating" : "warm", ms * 5.0F);
    };
    time("scores", [&](int layer) {
      k::dflash_attention_scores_kernel<<<
          dim3(kKvHeads,
               (total + k::kAttentionScoreKeys - 1) / k::kAttentionScoreKeys,
               1),
          k::kAttentionThreads>>>(o.q, o.k[layer], o.block_k, o.scores, pos,
                                  start, count, rows, kWindow, kHeads,
                                  kKvHeads, kDim, kScale, kCapacity);
    });
    time("softmax", [&](int) {
      k::dflash_attention_softmax_kernel<<<dim3(4, kKvHeads, 1),
                                           k::kAttentionThreads>>>(
          o.scores, count, rows, kHeads / kKvHeads);
    });
    time("values", [&](int layer) {
      k::dflash_attention_values_kernel<<<dim3(kDim / 32, kKvHeads, 1),
                                          k::kAttentionThreads>>>(
          o.scores, o.v[layer], o.block_v, o.out, start, count, rows, kHeads,
          kKvHeads, kDim, kCapacity);
    });
  }
  return failures != 0;
}
