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
// decode_moe_bench [reps] [layers] [overlap 0..10]
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
    const double expert_bytes =
        2.0 * kFf * (kHidden / 256) * kQ4KBlock +
        double(kHidden) * (kFf / 32) * kQ51Block;
    std::printf("layers %d, %.2f GB of expert weights, %.3f MB per expert, "
                "overlap %d of %d\n",
                layers, layers * kExperts * expert_bytes / 1e9,
                expert_bytes / 1e6, overlap, kUsed);

    float *x = nullptr, *mid = nullptr, *out = nullptr;
    std::int32_t* ids = nullptr;
    Check(hipMalloc(&x, 8 * kHidden * sizeof(float)), "x");
    Check(hipMalloc(&mid, 8 * kUsed * kFf * sizeof(float)), "mid");
    Check(hipMalloc(&out, 8 * kUsed * kHidden * sizeof(float)), "out");
    Check(hipMalloc(&ids, std::size_t(layers) * 8 * kUsed * sizeof(int)), "ids");
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
      const auto pass = [&] {
        for (int l = 0; l < layers; ++l) {
          const std::int32_t* lid = ids + std::size_t(l) * rows * kUsed;
          if (qfn_mmq_moe_gated_vec(kTypeQ4K, gate[l], up[l], x, lid, mid, kFf,
                                    kHidden, rows, kExperts, kUsed,
                                    nullptr) != 0 ||
              qfn_mmq_moe_vec(kTypeQ51, down[l], mid, lid, out, kHidden, kFf,
                              rows * kUsed, kExperts, 1, nullptr, nullptr,
                              nullptr) != 0)
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
      std::sort(times.begin(), times.end());
      const double med = times[times.size() / 2];
      std::vector<float> host(std::size_t(rows) * kUsed * kHidden);
      Check(hipMemcpy(host.data(), out, host.size() * sizeof(float),
                      hipMemcpyDeviceToHost),
            "out download");
      const double per_layer = distinct / layers;
      std::printf("%4d  %8.4f  %8.1f  %5.1f  %016llx\n", rows, med, per_layer,
                  per_layer * expert_bytes / (med * 1e-3) / 1e9,
                  static_cast<unsigned long long>(Fnv(host)));
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "decode_moe_bench: %s\n", e.what());
    return 1;
  }
  return 0;
}
