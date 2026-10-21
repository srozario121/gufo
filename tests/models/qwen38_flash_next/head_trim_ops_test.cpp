// GPU checks for the trimmed output head's kernels (head_trim.hpp):
//  - the full-head fallback gives, for every flagged row, the same bits as the
//    scalar kernel it replaced (the reference below is that kernel, verbatim);
//  - the fallback's measurement counters match a host reference (also replayed
//    from a graph);
//  - the draft head's compact candidates and argmax, with the IDs mapped back,
//    equal the full-vocabulary kernels run over the expanded row.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

void Expect(bool ok, const std::string& what) {
  if (!ok)
    throw std::runtime_error("FAILED: " + what);
}

struct DeviceDelete {
  void operator()(void* pointer) const { (void)hipFree(pointer); }
};

template<typename T>
auto Allocate(std::size_t count) {
  T* pointer = nullptr;
  CheckHip(hipMalloc(&pointer, count * sizeof(T) + 256), "allocate");
  return std::unique_ptr<T, DeviceDelete>(pointer);
}

template<typename T>
auto Upload(const std::vector<T>& host) {
  auto device = Allocate<T>(host.size());
  CheckHip(hipMemcpy(device.get(), host.data(), host.size() * sizeof(T),
                     hipMemcpyHostToDevice),
           "upload");
  return device;
}

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  CheckHip(
      hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost),
      "download");
  return host;
}

struct Rng {
  std::uint32_t state;
  std::uint32_t Next() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  }
};

// ---- The kernel the fallback replaced (head_trim fallback, first version)
// ----
constexpr unsigned kRefRowsPerBlock = 16;

__global__ void RefQuantizeKernel(const float* x, const std::int32_t* flags,
                                  std::uint32_t cols, std::uint8_t* xq) {
  const std::uint32_t r = blockIdx.y;
  if (flags[r] == 0)
    return;
  const std::uint32_t j = blockIdx.x;
  const unsigned i = threadIdx.x;
  const float xi = x[static_cast<std::size_t>(r) * cols + j * 32 + i];
  float amax = fabsf(xi);
  for (int offset = 16; offset > 0; offset >>= 1)
    amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
  const float d = amax / 127.0F;
  const std::int8_t qv = amax == 0.0F
                             ? static_cast<std::int8_t>(0)
                             : static_cast<std::int8_t>(roundf(xi / d));
  std::uint8_t* block =
      xq + (static_cast<std::size_t>(r) * (cols / 32) + j) * 34;
  reinterpret_cast<std::int8_t*>(block + 2)[i] = qv;
  if (i == 0)
    *reinterpret_cast<__half*>(block) = __float2half(d);
}

__global__ void RefFallbackKernel(const std::uint8_t* w, std::size_t row_bytes,
                                  std::uint32_t cols, std::uint32_t vocab,
                                  const std::uint8_t* xq,
                                  const std::int32_t* flags, std::uint32_t rows,
                                  float* out) {
  __shared__ std::int32_t active[64];
  __shared__ std::int32_t active_count;
  if (threadIdx.x == 0) {
    std::int32_t count = 0;
    for (std::uint32_t r = 0; r < rows && r < 64; ++r)
      if (flags[r] != 0)
        active[count++] = static_cast<std::int32_t>(r);
    active_count = count;
  }
  __syncthreads();
  if (active_count == 0)
    return;
  const unsigned lane = threadIdx.x & 15U;
  const std::uint32_t v = blockIdx.x * kRefRowsPerBlock + (threadIdx.x >> 4);
  const bool valid = v < vocab;
  const std::uint8_t* row =
      w + static_cast<std::size_t>(valid ? v : 0) * row_bytes;
  const std::uint32_t blocks = cols / 32;
  for (std::int32_t a = 0; a < active_count; ++a) {
    const std::uint8_t* xrow =
        xq + static_cast<std::size_t>(active[a]) * blocks * 34;
    float acc = 0.0F;
    for (std::uint32_t j = lane; j < blocks; j += 16) {
      const std::uint8_t* block = row + static_cast<std::size_t>(j) * 34;
      const std::uint8_t* xblock = xrow + static_cast<std::size_t>(j) * 34;
      const auto* qw = reinterpret_cast<const std::uint16_t*>(block + 2);
      const auto* u = reinterpret_cast<const std::uint16_t*>(xblock + 2);
      int sumi = 0;
      for (int i = 0; i < 16; ++i) {
        const std::uint16_t wp = qw[i];
        const std::uint16_t xp = u[i];
        sumi += static_cast<int>(static_cast<std::int8_t>(wp & 0xFFU)) *
                static_cast<int>(static_cast<std::int8_t>(xp & 0xFFU));
        sumi += static_cast<int>(static_cast<std::int8_t>(wp >> 8)) *
                static_cast<int>(static_cast<std::int8_t>(xp >> 8));
      }
      const float dw = __half2float(*reinterpret_cast<const __half*>(block));
      const float dx = __half2float(*reinterpret_cast<const __half*>(xblock));
      acc += dw * dx * static_cast<float>(sumi);
    }
    for (int offset = 8; offset > 0; offset >>= 1)
      acc += __shfl_down(acc, offset, 16);
    if (lane == 0 && valid)
      out[static_cast<std::size_t>(active[a]) * vocab + v] = acc;
  }
}

void CheckFallback(std::uint32_t cols, std::uint32_t vocab, std::uint32_t rows,
                   const std::vector<int>& flagged, std::uint32_t seed) {
  Rng rng{seed};
  const std::uint32_t blocks = cols / 32;
  const std::size_t row_bytes = std::size_t{blocks} * 34;
  std::vector<std::uint8_t> weights(row_bytes * vocab);
  for (std::uint32_t v = 0; v < vocab; ++v) {
    for (std::uint32_t j = 0; j < blocks; ++j) {
      std::uint8_t* block = weights.data() + v * row_bytes + j * 34;
      const __half d = __float2half(
          0.001F + static_cast<float>(rng.Next() % 1000) / 200000.0F);
      std::memcpy(block, &d, 2);
      for (int i = 0; i < 32; ++i) {
        // Cover the whole int8 range, including -128.
        block[2 + i] = static_cast<std::uint8_t>(rng.Next() & 0xFFU);
      }
    }
  }
  std::vector<float> x(std::size_t{rows} * cols);
  for (auto& value : x) {
    const float u =
        static_cast<float>(static_cast<int>(rng.Next() & 0xFFFFU) - 32768) /
        32768.0F;
    value = u * (1.0F + static_cast<float>(rng.Next() % 8));
  }
  // One all-zero block, to cover the zero-scale case.
  if (rows > 1)
    std::fill_n(x.begin() + cols, 32, 0.0F);
  std::vector<std::int32_t> flags(rows, 0);
  for (const int r : flagged)
    flags[static_cast<std::size_t>(r)] = 1;

  auto d_w = Upload(weights);
  auto d_x = Upload(x);
  auto d_flags = Upload(flags);
  auto d_counters = Allocate<unsigned long long>(3);
  CheckHip(hipMemset(d_counters.get(), 0, 3 * sizeof(unsigned long long)),
           "counters");
  auto d_scratch = Allocate<std::uint8_t>(std::size_t{rows} * blocks * 36);
  auto d_ref_scratch = Allocate<std::uint8_t>(std::size_t{rows} * blocks * 34);
  const float sentinel = 12345.0F;
  const std::vector<float> filled(std::size_t{rows} * vocab, sentinel);
  auto d_out = Upload(filled);
  auto d_ref = Upload(filled);

  q::FullHeadFallback(d_w.get(), row_bytes, cols, vocab, d_x.get(),
                      d_flags.get(), rows, d_out.get(), d_counters.get(),
                      d_scratch.get(), nullptr);
  CheckHip(hipGetLastError(), "fallback launch");
  hipLaunchKernelGGL(RefQuantizeKernel, dim3(blocks, rows), dim3(32), 0,
                     nullptr, d_x.get(), d_flags.get(), cols,
                     d_ref_scratch.get());
  hipLaunchKernelGGL(RefFallbackKernel,
                     dim3((vocab + kRefRowsPerBlock - 1) / kRefRowsPerBlock),
                     dim3(kRefRowsPerBlock * 16), 0, nullptr,
                     static_cast<const std::uint8_t*>(d_w.get()), row_bytes,
                     cols, vocab, d_ref_scratch.get(), d_flags.get(), rows,
                     d_ref.get());
  CheckHip(hipDeviceSynchronize(), "fallback sync");

  const auto actual = Download(d_out.get(), filled.size());
  const auto expected = Download(d_ref.get(), filled.size());
  for (std::size_t i = 0; i < filled.size(); ++i) {
    Expect(std::bit_cast<std::uint32_t>(actual[i]) ==
               std::bit_cast<std::uint32_t>(expected[i]),
           "fallback bits differ at cols=" + std::to_string(cols) + " vocab=" +
               std::to_string(vocab) + " rows=" + std::to_string(rows) +
               " flagged=" + std::to_string(flagged.size()) +
               " element=" + std::to_string(i));
  }
  const auto counters = Download(d_counters.get(), 3);
  Expect(counters[2] == (flagged.empty() ? 0U : 1U), "one pass per call");
}

// ---- The draft head on the compact rows ----
void CheckCompactDraft(std::uint32_t vocab, std::uint32_t scored,
                       std::uint32_t seed, bool ties) {
  Rng rng{seed};
  // Ascending token IDs: a random subset of the vocabulary.
  std::vector<std::int32_t> ids;
  std::vector<std::int32_t> inverse(vocab, -1);
  {
    std::vector<bool> in(vocab, false);
    std::uint32_t chosen = 0;
    while (chosen < scored) {
      const auto id = rng.Next() % vocab;
      if (!in[id]) {
        in[id] = true;
        ++chosen;
      }
    }
    for (std::uint32_t id = 0; id < vocab; ++id)
      if (in[id]) {
        inverse[id] = static_cast<std::int32_t>(ids.size());
        ids.push_back(static_cast<std::int32_t>(id));
      }
  }
  const std::uint32_t stride = (scored + 255) / 256 * 256;
  std::vector<float> compact(stride, 0.0F);
  for (std::uint32_t i = 0; i < scored; ++i) {
    // A coarse grid makes many ties; the signed zero is exercised too.
    const int step = static_cast<int>(rng.Next() % (ties ? 40 : 40000)) - 20;
    compact[i] = ties ? static_cast<float>(step) * 0.25F
                      : static_cast<float>(step) * 0.001F;
  }
  compact[0] = -0.0F;
  if (scored > 7)
    compact[7] = 0.0F;
  std::vector<std::uint8_t> suppressed(vocab, 0);
  for (std::uint32_t id = 0; id < vocab; ++id)
    suppressed[id] = rng.Next() % 5 == 0 ? 1 : 0;

  auto d_compact = Upload(compact);
  auto d_ids = Upload(ids);
  auto d_inverse = Upload(inverse);
  auto d_suppressed = Upload(suppressed);
  auto d_full = Allocate<float>(vocab);

  // The full-vocabulary route: expand, mask, then the kernels over `vocab`.
  q::ExpandHeadLogits(d_compact.get(), d_inverse.get(), d_full.get(), 1, vocab,
                      stride, nullptr);
  q::MaskSuppressedLogits(d_full.get(), d_suppressed.get(), vocab, 1, 0,
                          nullptr);
  // The compact route: mask in place, then the kernels over `scored`.
  q::MaskSuppressedCompact(d_compact.get(), d_suppressed.get(), d_ids.get(),
                           scored, 1, 0, nullptr);

  auto run_argmax = [&](const float* logits, std::uint32_t width, bool remap) {
    auto scratch = Allocate<q::ArgmaxCandidate>(q::kArgmaxParts);
    auto out = Allocate<std::int32_t>(1);
    q::Argmax(logits, scratch.get(), out.get(), 1, width, nullptr);
    if (remap)
      q::RemapDraftIds(reinterpret_cast<std::uint32_t*>(out.get()), 1, 0,
                       d_ids.get(), nullptr);
    CheckHip(hipDeviceSynchronize(), "argmax");
    return Download(out.get(), 1)[0];
  };
  const auto full_best = run_argmax(d_full.get(), vocab, false);
  const auto compact_best = run_argmax(d_compact.get(), scored, true);
  Expect(full_best == compact_best,
         "argmax: full " + std::to_string(full_best) + " compact " +
             std::to_string(compact_best) + " vocab=" + std::to_string(vocab) +
             " scored=" + std::to_string(scored));

  struct Candidates {
    std::vector<std::uint32_t> ids;
    std::vector<float> scores;
  };
  auto run_top = [&](const float* logits, std::uint32_t width, bool remap) {
    const auto work = q::MtpCandidateWorkspaceSize(width);
    auto out_ids = Allocate<std::uint32_t>(work);
    auto scratch_ids = Allocate<std::uint32_t>(work);
    auto scores = Allocate<float>(64);
    q::MtpTopCandidates(logits, out_ids.get(), scratch_ids.get(), scores.get(),
                        width, nullptr);
    const std::uint32_t count = std::min<std::uint32_t>(width, 64);
    if (remap)
      q::RemapDraftIds(out_ids.get(), count, 0, d_ids.get(), nullptr);
    CheckHip(hipDeviceSynchronize(), "candidates");
    return Candidates{Download(out_ids.get(), count),
                      Download(scores.get(), count)};
  };
  const auto full_top = run_top(d_full.get(), vocab, false);
  const auto compact_top = run_top(d_compact.get(), scored, true);
  Expect(full_top.ids == compact_top.ids,
         "candidate IDs differ at vocab=" + std::to_string(vocab) +
             " scored=" + std::to_string(scored));
  for (std::size_t i = 0; i < full_top.scores.size(); ++i)
    Expect(std::bit_cast<std::uint32_t>(full_top.scores[i]) ==
               std::bit_cast<std::uint32_t>(compact_top.scores[i]),
           "candidate score bits differ");
}

// The timing stamps: captured in a graph and replayed, they still add one
// call per replay, a nonzero span for a phase that does work, and the call
// into the bucket of its flagged-row count.
void CheckTiming() {
  int device = 0;
  int khz = 0;
  CheckHip(hipGetDevice(&device), "device");
  CheckHip(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device),
           "wall clock rate");
  Expect(khz > 0, "the device reports a wall clock rate");
  constexpr std::size_t kBytes = 256U << 20;
  auto scratch = Allocate<std::uint8_t>(kBytes);
  auto timing = Allocate<unsigned long long>(q::kHeadTimingCounters + 5);
  CheckHip(hipMemset(timing.get(), 0,
                     (q::kHeadTimingCounters + 5) * sizeof(unsigned long long)),
           "clear");
  auto flags = Upload(std::vector<std::int32_t>{0, 1, 0, 1, 0});
  hipStream_t stream = nullptr;
  CheckHip(hipStreamCreate(&stream), "stream");
  const auto work = [&] {
    CheckHip(hipMemsetAsync(scratch.get(), 1, kBytes, stream), "work");
  };
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  CheckHip(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), "begin");
  // A call without the fallback stage, then one with two flagged rows.
  q::HeadTimingStamp(timing.get(), 0, stream);
  q::HeadTimingStamp(timing.get(), 4, stream);
  work();
  q::HeadTimingStamp(timing.get(), 1, stream);
  work();
  q::HeadTimingFinish(timing.get(), flags.get(), 5, false, stream);
  q::HeadTimingStamp(timing.get(), 0, stream);
  q::HeadTimingStamp(timing.get(), 4, stream);
  work();
  q::HeadTimingStamp(timing.get(), 1, stream);
  work();
  q::HeadTimingStamp(timing.get(), 2, stream);
  work();
  q::HeadTimingStamp(timing.get(), 3, stream);
  work();
  q::HeadTimingFinish(timing.get(), flags.get(), 5, true, stream);
  CheckHip(hipStreamEndCapture(stream, &graph), "end");
  CheckHip(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
           "instantiate");
  constexpr int kReplays = 4;
  for (int i = 0; i < kReplays; ++i)
    CheckHip(hipGraphLaunch(exec, stream), "launch");
  CheckHip(hipStreamSynchronize(stream), "sync");
  const auto c = Download(timing.get(), q::kHeadTimingCounters);
  Expect(c[0] == 2 * kReplays, "one call counted per HeadProject call");
  Expect(c[1] == kReplays, "one fallback call per replay");
  Expect(c[7] == 0 && c[8] == 0 && c[9] == kReplays && c[10] == 0,
         "two flagged rows land in the two-row bucket");
  Expect(c[2] > 0 && c[3] > 0 && c[4] > 0 && c[11 + 2] > 0,
         "phases that do work take time");
  Expect(c[5] >= c[2] + c[3] + c[4] + c[13],
         "the total spans the phases it contains");
  // 256 MiB of memset is milliseconds, far under a second of clock ticks.
  Expect(c[3] < static_cast<unsigned long long>(khz) * 1000ULL * kReplays * 4,
         "a phase is not absurdly long");
  CheckHip(hipGraphExecDestroy(exec), "destroy exec");
  CheckHip(hipGraphDestroy(graph), "destroy graph");
  CheckHip(hipStreamDestroy(stream), "destroy stream");
}

// The fallback's measurement counters against a host reference: rows whose
// argmax changes, the unscored softmax mass and its thresholds, and the
// verification round's flagged rows past the kept prefix. Run twice on the
// stream and once through a graph, so the sums are additive and replay-safe.
void CheckCounters() {
  constexpr std::uint32_t kVocab = 1003;
  constexpr std::uint32_t kRows = 6;
  constexpr std::uint32_t kStride = 512;
  std::vector<std::int32_t> inverse(kVocab, -1);
  std::vector<std::int32_t> ids;
  for (std::uint32_t id = 0; id < kVocab; id += 3) {
    inverse[id] = static_cast<std::int32_t>(ids.size());
    ids.push_back(static_cast<std::int32_t>(id));
  }
  const auto scored = static_cast<std::uint32_t>(ids.size());
  Expect(scored <= kStride, "test geometry");
  Rng rng{77};
  std::vector<float> compact(std::size_t{kRows} * kStride, -1000.0F);
  std::vector<float> full(std::size_t{kRows} * kVocab);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    for (std::uint32_t i = 0; i < scored; ++i)
      compact[r * kStride + i] = static_cast<float>(rng.Next() % 1000) / 250.0F;
    for (std::uint32_t id = 0; id < kVocab; ++id)
      full[r * kVocab + id] = static_cast<float>(rng.Next() % 1000) / 250.0F;
  }
  // Row 1: the full head agrees (the compact winner is the full winner).
  // Row 2: an unscored token wins. Row 3: flagged, an unscored token wins by a
  // wide margin (mass above 0.2). Row 4: the compact argmax tie goes to the
  // lowest token, and the full head keeps it. Row 5 is flagged, agrees.
  const auto compact_argmax = [&](std::uint32_t r) {
    std::uint32_t best = 0;
    for (std::uint32_t i = 1; i < scored; ++i)
      if (compact[r * kStride + i] > compact[r * kStride + best])
        best = i;
    return best;
  };
  for (const std::uint32_t r : {1U, 5U}) {
    const auto token = static_cast<std::uint32_t>(ids[compact_argmax(r)]);
    full[r * kVocab + token] = 50.0F;
  }
  full[2 * kVocab + 1] = 50.0F;  // token 1 is unscored
  full[3 * kVocab + 2] = 60.0F;  // token 2 is unscored
  compact[4 * kStride + 7] = compact[4 * kStride + 2] = 40.0F;  // tie
  full[4 * kVocab + static_cast<std::uint32_t>(ids[2])] = 55.0F;
  std::vector<std::int32_t> flags = {0, 1, 1, 1, 1, 1};

  auto d_compact = Upload(compact);
  auto d_full = Upload(full);
  auto d_ids = Upload(ids);
  auto d_inverse = Upload(inverse);
  auto d_flags = Upload(flags);
  auto d_rowinfo = Upload(std::vector<std::int32_t>(kRows, 9));
  auto d_counters = Allocate<unsigned long long>(q::kHeadTimingCounters);
  CheckHip(hipMemset(d_counters.get(), 0,
                     q::kHeadTimingCounters * sizeof(unsigned long long)),
           "clear");
  hipStream_t stream = nullptr;
  CheckHip(hipStreamCreate(&stream), "stream");

  // Host reference.
  std::uint64_t changed = 0, over_005 = 0, over_02 = 0, flagged = 0;
  double mass_sum = 0.0;
  std::vector<int> row_changed(kRows, 0);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    if (flags[r] == 0)
      continue;
    ++flagged;
    const auto scored_token =
        static_cast<std::uint32_t>(ids[compact_argmax(r)]);
    std::uint32_t full_token = 0;
    for (std::uint32_t id = 1; id < kVocab; ++id)
      if (full[r * kVocab + id] > full[r * kVocab + full_token])
        full_token = id;
    if (scored_token != full_token) {
      ++changed;
      row_changed[r] = 1;
    }
    double all = 0.0, unscored = 0.0;
    const double maximum = full[r * kVocab + full_token];
    for (std::uint32_t id = 0; id < kVocab; ++id) {
      const double e = std::exp(full[r * kVocab + id] - maximum);
      all += e;
      if (inverse[id] < 0)
        unscored += e;
    }
    mass_sum += unscored / all;
    over_005 += unscored / all > 0.05;
    over_02 += unscored / all > 0.2;
  }
  Expect(changed >= 2 && changed < flagged,
         "the test rows exercise both cases");

  const auto launch = [&] {
    q::HeadCountFlagged(d_compact.get(), kStride, scored, d_ids.get(),
                        d_full.get(), kVocab, d_inverse.get(), d_flags.get(),
                        d_rowinfo.get(), kRows, d_counters.get(), stream);
    // Rounds over rows 0..5 keeping 3, and over rows 1..4 (first = 1)
    // keeping 1.
    q::HeadRecordChain(d_flags.get(), d_rowinfo.get(), 0, kRows, 3,
                       d_counters.get(), stream);
    q::HeadRecordChain(d_flags.get(), d_rowinfo.get(), 1, 4, 1,
                       d_counters.get(), stream);
  };
  launch();
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  CheckHip(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), "begin");
  launch();
  CheckHip(hipStreamEndCapture(stream, &graph), "end");
  CheckHip(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
           "instantiate");
  CheckHip(hipGraphLaunch(exec, stream), "launch");
  CheckHip(hipStreamSynchronize(stream), "sync");
  const auto c = Download(d_counters.get(), 11);
  const auto rowinfo = Download(d_rowinfo.get(), kRows);
  for (std::uint32_t r = 0; r < kRows; ++r)
    Expect(rowinfo[r] == row_changed[r], "row info is 1 for a changed argmax");
  Expect(c[0] == 2 * flagged && c[1] == 2 * changed, "flagged and changed");
  Expect(c[3] == 2 * over_005 && c[4] == 2 * over_02, "mass thresholds");
  const double mass = static_cast<double>(c[2]) / 4294967296.0;
  Expect(std::fabs(mass - 2 * mass_sum) < 1e-4,
         "mass sum " + std::to_string(mass) + " against " +
             std::to_string(2 * mass_sum));
  // Round 1: all rows, keep 3: rows 3..5 are discarded. Round 2: rows 1..4,
  // keep 1: relative rows 1..3 (rows 2..4) are discarded.
  std::uint64_t chain_flagged = 0, discarded = 0, used_changed = 0,
                discarded_changed = 0;
  const auto round = [&](std::uint32_t first, std::uint32_t rows,
                         std::uint32_t keep) {
    for (std::uint32_t r = 0; r < rows; ++r) {
      if (flags[first + r] == 0)
        continue;
      ++chain_flagged;
      if (r >= keep) {
        ++discarded;
        discarded_changed += row_changed[first + r];
      } else {
        used_changed += row_changed[first + r];
      }
    }
  };
  round(0, kRows, 3);
  round(1, 4, 1);
  Expect(c[5] == 4 && c[6] == 2 * (kRows + 4), "rounds and rows");
  Expect(c[7] == 2 * chain_flagged && c[8] == 2 * discarded &&
             c[9] == 2 * used_changed && c[10] == 2 * discarded_changed,
         "chain counters");
  CheckHip(hipGraphExecDestroy(exec), "destroy exec");
  CheckHip(hipGraphDestroy(graph), "destroy graph");
  CheckHip(hipStreamDestroy(stream), "destroy stream");
}

}  // namespace

int main() {
  try {
    CheckTiming();
    CheckCounters();
    // cols: 4096 is the model's; 96 has fewer blocks than the 16 lanes; 1056
    // leaves lanes with uneven block counts. vocab values are not multiples of
    // the 16 rows a block covers.
    for (const std::uint32_t cols : {96U, 1056U, 4096U}) {
      CheckFallback(cols, 1003, 1, {}, 1);
      CheckFallback(cols, 1003, 1, {0}, 2);
      CheckFallback(cols, 1003, 6, {1, 4}, 3);
      CheckFallback(cols, 2049, 8, {0, 1, 2, 3, 4, 5, 6, 7}, 4);
      CheckFallback(cols, 517, 9, {8, 0, 3, 5, 6}, 5);
    }
    CheckFallback(4096, 20000, 64, {0, 63, 17, 18, 19, 20, 21, 30, 31}, 6);
    for (const bool ties : {false, true}) {
      CheckCompactDraft(8000, 3001, 11, ties);
      CheckCompactDraft(8000, 400, 12, ties);
      CheckCompactDraft(248320, 48696, 13, ties);
      CheckCompactDraft(248320, 38892, 14, ties);
    }
    std::cout << "head_trim_ops OK\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
