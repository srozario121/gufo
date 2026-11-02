// The decode FFN combine with the MoE epilogue folded in (HcCombineQ8_1Moe)
// against the two launches it replaces (MoeEpilogue, then HcCombineQ8_1) on
// identical inputs: the residual streams, the normalized rows, their Q8_1
// rows and the block output must match byte for byte, at 1 to 9 rows, with
// unsaturated as well as wide-ranging values.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

constexpr std::uint32_t kSlots = 10;
constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kStreams = 4;
constexpr std::uint32_t kGateStride = 513;
constexpr std::uint32_t kParts = 4;
constexpr float kEps = 1e-6F;

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

std::vector<float> Random(std::size_t n, std::uint32_t seed, float scale) {
  std::vector<float> v(n);
  std::uint32_t s = seed;
  for (auto& x : v)
    x = scale * ((int(Next(&s) % 20001) - 10000) / 10000.0F);
  return v;
}

template<typename T>
T* Device(const std::vector<T>& host) {
  void* p = nullptr;
  Check(hipMalloc(&p, host.size() * sizeof(T)), "hipMalloc");
  Check(hipMemcpy(p, host.data(), host.size() * sizeof(T),
                  hipMemcpyHostToDevice),
        "upload");
  return static_cast<T*>(p);
}

template<typename T>
std::vector<T> Host(const T* p, std::size_t n) {
  std::vector<T> v(n);
  Check(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost),
        "download");
  return v;
}

int failures = 0;

template<typename T>
void Same(const std::vector<T>& a, const std::vector<T>& b, const char* what,
          std::uint32_t rows, float scale) {
  if (a.size() != b.size() ||
      std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) != 0) {
    std::size_t i = 0;
    while (i < a.size() && std::memcmp(&a[i], &b[i], sizeof(T)) == 0)
      ++i;
    std::cerr << "FAIL: " << what << " differs at " << i << " (rows " << rows
              << ", scale " << scale << ")\n";
    ++failures;
  }
}

void Case(std::uint32_t rows, float scale, std::uint32_t seed) {
  const std::size_t hc = std::size_t(rows) * kStreams * kHidden;
  const auto res0 = Random(hc, seed, 2.0F * scale);
  const auto expert = Random(std::size_t(rows) * kSlots * kHidden, seed + 1, scale);
  const auto weights = Random(std::size_t(rows) * kSlots, seed + 2, 0.5F);
  const auto shared = Random(std::size_t(rows) * kHidden, seed + 3, scale);
  const auto router = Random(std::size_t(rows) * kGateStride, seed + 4, 3.0F);
  const auto inject = Random(std::size_t(rows) * kStreams * kParts, seed + 5, 2.0F);
  auto gamma = Random(std::size_t(kStreams) * kHidden, seed + 6, 1.0F);
  for (auto& g : gamma)
    g = 1.0F + 0.5F * g;
  const std::size_t q8_bytes = std::size_t(rows) * kStreams * kHidden / 32 * 36;

  float* d_expert = Device(expert);
  float* d_weights = Device(weights);
  float* d_shared = Device(shared);
  float* d_router = Device(router);
  float* d_inject = Device(inject);
  float* d_gamma = Device(gamma);
  float* res_a = Device(res0);
  float* res_b = Device(res0);
  float* out_a = Device(std::vector<float>(std::size_t(rows) * kHidden, 7.0F));
  float* out_b = Device(std::vector<float>(std::size_t(rows) * kHidden, 7.0F));
  float* xn_a = Device(std::vector<float>(hc, 0.0F));
  float* xn_b = Device(std::vector<float>(hc, 0.0F));
  auto* q8_a = Device(std::vector<std::uint8_t>(q8_bytes, 0));
  auto* q8_b = Device(std::vector<std::uint8_t>(q8_bytes, 0));
  const float* gate = d_router + (kGateStride - 1);

  q::MoeEpilogue(d_expert, d_weights, d_shared, gate, kGateStride, out_a, rows,
                 kSlots, kHidden, nullptr);
  const bool ran_a =
      q::HcCombineQ8_1(res_a, out_a, d_inject, kParts, d_gamma, xn_a, q8_a,
                       rows, kHidden, kStreams, kEps, nullptr);
  q::MoeEpilogueOperands moe{d_expert, d_weights, d_shared, gate,
                             kGateStride, kSlots,  out_b};
  const bool ran_b =
      q::HcCombineQ8_1Moe(res_b, moe, d_inject, kParts, d_gamma, xn_b, q8_b,
                          rows, kHidden, kStreams, kEps, nullptr);
  Check(hipDeviceSynchronize(), "sync");
  if (!ran_a || !ran_b) {
    std::cerr << "FAIL: a combine declined the model shape (rows " << rows
              << ")\n";
    ++failures;
    return;
  }
  Same(Host(out_a, std::size_t(rows) * kHidden),
       Host(out_b, std::size_t(rows) * kHidden), "block output", rows, scale);
  Same(Host(res_a, hc), Host(res_b, hc), "residual", rows, scale);
  Same(Host(xn_a, hc), Host(xn_b, hc), "normalized rows", rows, scale);
  Same(Host(q8_a, q8_bytes), Host(q8_b, q8_bytes), "Q8_1 rows", rows, scale);
  for (void* p : {static_cast<void*>(d_expert), static_cast<void*>(d_weights),
                  static_cast<void*>(d_shared), static_cast<void*>(d_router),
                  static_cast<void*>(d_inject), static_cast<void*>(d_gamma),
                  static_cast<void*>(res_a), static_cast<void*>(res_b),
                  static_cast<void*>(out_a), static_cast<void*>(out_b),
                  static_cast<void*>(xn_a), static_cast<void*>(xn_b),
                  static_cast<void*>(q8_a), static_cast<void*>(q8_b)})
    (void)hipFree(p);
}

}  // namespace

int main() {
  try {
    std::uint32_t seed = 101;
    for (const std::uint32_t rows : {1U, 2U, 3U, 5U, 8U, 9U})
      for (const float scale : {1e-3F, 0.05F, 1.0F, 30.0F})
        Case(rows, scale, seed += 17);
    // Declines outside its shapes: no gamma.
    q::MoeEpilogueOperands moe{nullptr, nullptr, nullptr, nullptr, 0, kSlots,
                               nullptr};
    if (q::HcCombineQ8_1Moe(nullptr, moe, nullptr, kParts, nullptr, nullptr,
                            nullptr, 1, kHidden, kStreams, kEps, nullptr)) {
      std::cerr << "FAIL: the fused combine ran without gamma\n";
      ++failures;
    }
  } catch (const std::exception& e) {
    std::cerr << "hc_combine_moe_ops_test: " << e.what() << "\n";
    return 1;
  }
  if (failures != 0)
    return 1;
  std::cout << "hc_combine_moe_ops_test: all checks passed\n";
  return 0;
}
