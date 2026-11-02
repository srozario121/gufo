// Times the decode and verification routed experts of Flash-Next: the gated
// Q4_K gate/up vector projection (qfn_mmq_moe_gated_vec, SwiGLU out) and the
// Q5_1 down projection over the top-10 slot rows (qfn_mmq_moe_vec, one expert
// per slot), at 1 to 8 rows, as the executor calls them for verify widths.
// Weights are synthetic but valid (finite scales) and span several layers of
// 512 experts, so a pass streams from DRAM rather than the 32 MiB MALL.
// Routing is random top-10 per row; `shared` gives the rows `overlap` of their
// ten experts in common. Prints, per width, the median time per layer, the
// distinct experts streamed, the weight bandwidth and an FNV-1a hash of the
// down output (exactness across kernel variants).
//
// `part` read times a plain 16-byte-per-lane read of the same distinct
// experts' bytes in one launch (the attainable bandwidth of this access
// pattern); readl reads them with two launches per layer, gate/up then down,
// as the executor launches the projections.
//
// decode_moe_bench [reps] [layers] [overlap 0..10] [part:
// both|gate|down|read|readl]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace {

// Each block sums one 64 KiB span of one distinct expert's gate, up or down
// bytes with 16-byte loads; the sum is written so the loads are kept.
__global__ void ReadExperts(const uint4* const* bases, const int* spans,
                            int n_spans, unsigned* sink) {
  const int span = blockIdx.x;
  if (span >= n_spans)
    return;
  const uint4* p =
      bases[spans[2 * span]] + std::size_t(spans[2 * span + 1]) * 4096;
  unsigned acc = 0;
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
    const uint4 v = p[i];
    acc ^= v.x + v.y + v.z + v.w;
  }
  if (acc == 0x12345678u)
    sink[0] = acc;
}

constexpr int kExperts = 512;
constexpr int kUsed = 10;
constexpr int kHidden = 2560;
constexpr int kFf = 640;
constexpr std::size_t kQ4KBlock = 144;  // 256 elements
constexpr std::size_t kQ51Block = 24;   // 32 elements
constexpr int kTypeQ4K = 12;
constexpr int kTypeQ51 = 7;

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

std::uint32_t Next(std::uint32_t* s) {
  *s ^= *s << 13;
  *s ^= *s >> 17;
  *s ^= *s << 5;
  return *s;
}

// Random codes with finite FP16 scales: d and dmin (Q4_K) or d and m (Q5_1).
void* Weights(std::size_t blocks, std::size_t block_bytes, std::uint32_t seed) {
  std::vector<std::uint8_t> host(blocks * block_bytes);
  std::uint32_t s = seed;
  for (std::size_t i = 0; i + 4 <= host.size(); i += 4) {
    const std::uint32_t v = Next(&s);
    std::memcpy(&host[i], &v, 4);
  }
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* p = &host[b * block_bytes];
    const std::uint16_t d = 0x1c00 | (Next(&s) & 0x3ff);  // about 0.004-0.008
    const std::uint16_t m = 0x9c00 | (Next(&s) & 0x3ff);  // negative, small
    std::memcpy(p, &d, 2);
    std::memcpy(p + 2, &m, 2);
  }
  void* dev = nullptr;
  Check(hipMalloc(&dev, host.size()), "hipMalloc weights");
  Check(hipMemcpy(dev, host.data(), host.size(), hipMemcpyHostToDevice),
        "upload weights");
  return dev;
}

std::uint64_t Fnv(const std::vector<float>& v) {
  std::uint64_t h = 1469598103934665603ULL;
  const auto* b = reinterpret_cast<const std::uint8_t*>(v.data());
  for (std::size_t i = 0; i < v.size() * sizeof(float); ++i)
    h = (h ^ b[i]) * 1099511628211ULL;
  return h;
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 21;
  const int layers = argc > 2 ? std::atoi(argv[2]) : 6;
  const int overlap = argc > 3 ? std::atoi(argv[3]) : 0;
  const std::string part = argc > 4 ? argv[4] : "both";
  try {
    const std::size_t up_blocks =
        std::size_t(kExperts) * kFf * (kHidden / 256);  // per projection
    const std::size_t down_blocks =
        std::size_t(kExperts) * kHidden * (kFf / 32);
    std::vector<void*> gate(layers), up(layers), down(layers);
    for (int l = 0; l < layers; ++l) {
      gate[l] = Weights(up_blocks, kQ4KBlock, 11 + 3 * l);
      up[l] = Weights(up_blocks, kQ4KBlock, 12 + 3 * l);
      down[l] = Weights(down_blocks, kQ51Block, 13 + 3 * l);
    }
    const double expert_bytes = 2.0 * kFf * (kHidden / 256) * kQ4KBlock +
                                double(kHidden) * (kFf / 32) * kQ51Block;
    std::printf(
        "layers %d, %.2f GB of expert weights, %.3f MB per expert, "
        "overlap %d of %d\n",
        layers, layers * kExperts * expert_bytes / 1e9, expert_bytes / 1e6,
        overlap, kUsed);

    float *x = nullptr, *mid = nullptr, *out = nullptr;
    std::int32_t* ids = nullptr;
    Check(hipMalloc(&x, 8 * kHidden * sizeof(float)), "x");
    Check(hipMalloc(&mid, 8 * kUsed * kFf * sizeof(float)), "mid");
    Check(hipMalloc(&out, 8 * kUsed * kHidden * sizeof(float)), "out");
    Check(hipMalloc(&ids, std::size_t(layers) * 8 * kUsed * sizeof(int)),
          "ids");
    std::vector<float> hx(8 * kHidden);
    std::uint32_t s = 7;
    for (auto& v : hx)
      v = (int(Next(&s) % 2001) - 1000) / 1000.0F;
    Check(hipMemcpy(x, hx.data(), hx.size() * sizeof(float),
                    hipMemcpyHostToDevice),
          "x upload");
    hipEvent_t a, b;
    Check(hipEventCreate(&a), "event");
    Check(hipEventCreate(&b), "event");
    std::printf("rows  ms/layer  distinct  GB/s  hash\n");
    for (int rows = 1; rows <= 8; ++rows) {
      // Routing per layer: row 0 draws ten distinct experts; later rows keep
      // `overlap` of row 0's and draw the rest outside the row's own set.
      std::vector<std::int32_t> hid(std::size_t(layers) * rows * kUsed);
      double distinct = 0;
      for (int l = 0; l < layers; ++l) {
        std::set<int> all;
        std::vector<int> first;
        for (int r = 0; r < rows; ++r) {
          std::set<int> mine;
          for (int k = 0; k < overlap && r > 0; ++k)
            mine.insert(first[k]);
          while (int(mine.size()) < kUsed)
            mine.insert(int(Next(&s) % kExperts));
          int k = 0;
          for (int e : mine) {
            hid[(std::size_t(l) * rows + r) * kUsed + k++] = e;
            all.insert(e);
          }
          if (r == 0)
            first.assign(mine.begin(), mine.end());
        }
        distinct += all.size();
      }
      Check(hipMemcpy(ids, hid.data(), hid.size() * sizeof(int),
                      hipMemcpyHostToDevice),
            "ids upload");
      // The read control: every distinct expert's three matrices, as 64 KiB
      // spans (an expert's gate and up are 921,600 bytes, its down 1,228,800).
      std::vector<int> spans;
      std::vector<const uint4*> base_host;
      std::vector<int>
          launch_first;  // readl: span index where each launch starts
      for (int l = 0; l < layers; ++l) {
        std::set<int> all;
        for (int r = 0; r < rows * kUsed; ++r)
          all.insert(hid[std::size_t(l) * rows * kUsed + r]);
        const std::size_t gb =
            2 * std::size_t(kFf) * (kHidden / 256) * kQ4KBlock / 2;
        const std::size_t db = std::size_t(kHidden) * (kFf / 32) * kQ51Block;
        for (int which = 0; which < 2; ++which) {
          launch_first.push_back(int(spans.size() / 2));
          for (int e : all) {
            const std::pair<const void*, std::size_t> both_parts[3] = {
                {static_cast<const char*>(gate[l]) + e * gb, gb},
                {static_cast<const char*>(up[l]) + e * gb, gb},
                {static_cast<const char*>(down[l]) + e * db, db}};
            const int lo = which == 0 ? 0 : 2, hi = which == 0 ? 2 : 3;
            for (int q = lo; q < hi; ++q) {
              const auto& [p, bytes] = both_parts[q];
              base_host.push_back(static_cast<const uint4*>(p));
              for (std::size_t o = 0; o + 65536 <= bytes; o += 65536) {
                spans.push_back(int(base_host.size() - 1));
                spans.push_back(int(o / 65536));
              }
            }
          }
        }
      }
      launch_first.push_back(int(spans.size() / 2));
      const uint4** bases_dev = nullptr;
      int* spans_dev = nullptr;
      unsigned* sink = nullptr;
      Check(hipMalloc(&bases_dev, base_host.size() * sizeof(void*)), "bases");
      Check(hipMalloc(&spans_dev, spans.size() * sizeof(int)), "spans");
      Check(hipMalloc(&sink, sizeof(unsigned)), "sink");
      Check(hipMemcpy(bases_dev, base_host.data(),
                      base_host.size() * sizeof(void*), hipMemcpyHostToDevice),
            "bases upload");
      Check(hipMemcpy(spans_dev, spans.data(), spans.size() * sizeof(int),
                      hipMemcpyHostToDevice),
            "spans upload");
      const int n_spans = int(spans.size() / 2);
      const double read_bytes = double(n_spans) * 65536;
      const auto pass = [&] {
        if (part == "read") {
          ReadExperts<<<n_spans, 256>>>(bases_dev, spans_dev, n_spans, sink);
          return;
        }
        if (part == "readl") {
          for (std::size_t i = 0; i + 1 < launch_first.size(); ++i) {
            const int first = launch_first[i],
                      count = launch_first[i + 1] - first;
            ReadExperts<<<count, 256>>>(bases_dev, spans_dev + 2 * first, count,
                                        sink);
          }
          return;
        }
        for (int l = 0; l < layers; ++l) {
          const std::int32_t* lid = ids + std::size_t(l) * rows * kUsed;
          if ((part != "down" &&
               qfn_mmq_moe_gated_vec(kTypeQ4K, gate[l], up[l], x, lid, mid, kFf,
                                     kHidden, rows, kExperts, kUsed,
                                     nullptr) != 0) ||
              (part != "gate" &&
               qfn_mmq_moe_vec(kTypeQ51, down[l], mid, lid, out, kHidden, kFf,
                               rows * kUsed, kExperts, 1, nullptr, nullptr,
                               nullptr) != 0))
            throw std::runtime_error("launch failed");
        }
      };
      pass();  // warm the pool and the launch shapes
      Check(hipDeviceSynchronize(), "warm");
      std::vector<float> times;
      for (int r = 0; r < reps; ++r) {
        Check(hipEventRecord(a, nullptr), "record");
        pass();
        Check(hipEventRecord(b, nullptr), "record");
        Check(hipEventSynchronize(b), "sync");
        float ms = 0;
        Check(hipEventElapsedTime(&ms, a, b), "elapsed");
        times.push_back(ms / layers);
      }
      Check(hipFree(bases_dev), "free");
      Check(hipFree(spans_dev), "free");
      Check(hipFree(sink), "free");
      std::sort(times.begin(), times.end());
      const double med = times[times.size() / 2];
      std::vector<float> host(std::size_t(rows) * kUsed * kHidden);
      Check(hipMemcpy(host.data(), out, host.size() * sizeof(float),
                      hipMemcpyDeviceToHost),
            "out download");
      const double per_layer = distinct / layers;
      if (part == "read" || part == "readl") {
        std::printf("%4d  %8.4f  %8.1f  %5.1f  (read %.1f MB per layer)\n",
                    rows, med, per_layer,
                    read_bytes / layers / (med * 1e-3) / 1e9,
                    read_bytes / layers / 1e6);
        continue;
      }
      const double part_bytes =
          part == "gate"   ? 2.0 * kFf * (kHidden / 256) * kQ4KBlock
          : part == "down" ? double(kHidden) * (kFf / 32) * kQ51Block
                           : expert_bytes;
      std::printf("%4d  %8.4f  %8.1f  %5.1f  %016llx\n", rows, med, per_layer,
                  per_layer * part_bytes / (med * 1e-3) / 1e9,
                  static_cast<unsigned long long>(Fnv(host)));
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "decode_moe_bench: %s\n", e.what());
    return 1;
  }
  return 0;
}
