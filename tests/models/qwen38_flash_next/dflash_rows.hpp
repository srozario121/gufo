// DFlash-2 target features for drafter training (gufo-mtp-continuous-learning,
// todo 2.4 T1): the CPU-testable half of dump_hidden.cpp's DFlash-2 mode.
//
// One row for position t of a sequence x is
//   features  for each tapped layer, in the drafter's target_layer_ids order,
//             the mean over the 4 hyper-connection streams of that layer's
//             settled output at t, as bf16 (5 x 2,560 = 12,800 values),
//   input id  x[t] (int32),
//   loss mask 1 when x[t] lies in a loss span, else 0 (int32),
// so row t is SpecForge's offline DFlash sample at t: hidden_states[t],
// input_ids[t], loss_mask[t], all at the same position (no shift; MTP rows
// pair h[t-1] with x[t], these do not). 25,608 bytes at the real shape.
//
// The features themselves come from the engine's own capture: the executor
// is created with Options::dflash_tap_layers and writes each session's
// capture ring with HcTapMeanToRing (kernels/rocm/kernels.hip.cpp), the
// same code the served drafter reads. TapMeanToRingReference below is that
// kernel's CPU reference (same indexing and arithmetic order); the CPU test
// drives the walk with it, and the tap-parity run compares against it.
//
// The walk is written against a `Trunk` so the CPU test can stand in for the
// GPU executor:
//   Forward(span<const int32_t> tokens)  advance; the taps of these positions
//                                        land in the ring at position % rows
//   ReadRing(size_t first, size_t n, float* out)
//                                        n x slots x hidden floats, positions
//                                        [first, first + n), from the ring
//   Verify(span<const int32_t> tokens, size_t keep)  (verify path only)
//                                        a kVerify forward of the tokens, then
//                                        a rollback to the first `keep`
//   Decode(int32_t token)                (verify path only) a kDecode step
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "tests/models/qwen38_flash_next/dump_rows.hpp"

namespace gufo::tools::dflash2 {

inline constexpr std::uint32_t kStreams = 4;
// incoai/Qwen3.8-Flash-Next-DFlash2 config.json target_layer_ids (zero-based
// layer outputs); its GGUF stores them as layer inputs [6, 16, 26, 36, 46],
// and QwenDFlashWeights::LoadFromGguf applies the N - 1 rule.
inline constexpr std::array<std::uint32_t, 5> kDrafterTaps{5, 15, 25, 35, 45};

struct Span {
  std::uint32_t begin;  // first supervised position
  std::uint32_t end;    // one past the last
};

inline std::size_t RowBytes(std::size_t slots, std::size_t hidden) {
  return slots * hidden * 2 + 4 + 4;
}

// The loaded taps must be exactly the expected ones, strictly increasing and
// short of the last layer (whose prefill rows the executor may skip).
inline void CheckTaps(std::span<const std::uint32_t> loaded,
                      std::span<const std::uint32_t> expected,
                      std::uint32_t num_layers) {
  if (!std::equal(loaded.begin(), loaded.end(), expected.begin(),
                  expected.end())) {
    std::string got;
    for (auto l : loaded)
      got += (got.empty() ? "" : ",") + std::to_string(l);
    throw std::invalid_argument("tap layers " + got +
                                " are not the drafter's target_layer_ids");
  }
  for (std::size_t i = 0; i < loaded.size(); ++i) {
    if (i > 0 && loaded[i] <= loaded[i - 1])
      throw std::invalid_argument("tap layers must be strictly increasing");
    if (loaded[i] + 1 >= num_layers)
      throw std::invalid_argument("a tap at or past the last layer");
  }
}

// Loss spans: ascending, not overlapping, non-empty, inside the sequence.
inline void ValidateLoss(std::span<const Span> s, std::size_t len) {
  for (std::size_t i = 0; i < s.size(); ++i) {
    const auto id = "loss span " + std::to_string(i);
    if (s[i].begin >= s[i].end)
      throw std::invalid_argument(id + ": is empty");
    if (s[i].end > len)
      throw std::invalid_argument(id + ": ends past the sequence");
    if (i > 0 && s[i].begin < s[i - 1].end)
      throw std::invalid_argument(id + ": overlaps or precedes the previous");
  }
}

inline bool InLoss(std::span<const Span> s, std::size_t t) {
  for (const auto& sp : s)
    if (t >= sp.begin && t < sp.end)
      return true;
  return false;
}

// CPU reference of HcTapMeanToRingKernel: for the batch's last `rows` rows,
// ring[(position + t) % rows][slot] = mean over the 4 streams of res[t],
// each stream first settled with HcApplyDeferred's fma when block_out is set.
inline void TapMeanToRingReference(const float* res, const float* block_out,
                                   const float* w, float* ring,
                                   std::uint32_t position, std::uint32_t rows,
                                   std::uint32_t slot, std::uint32_t slots,
                                   std::uint32_t n_tokens,
                                   std::uint32_t hidden) {
  if (n_tokens == 0 || rows == 0)
    return;
  const std::uint32_t first = n_tokens > rows ? n_tokens - rows : 0U;
  const std::size_t hc_dim = std::size_t{kStreams} * hidden;
  for (std::uint32_t t = first; t < n_tokens; ++t) {
    const float* row = res + std::size_t{t} * hc_dim;
    const std::uint32_t pos = (position + t) % rows;
    float* dst = ring + (std::size_t{pos} * slots + slot) * hidden;
    for (std::uint32_t i = 0; i < hidden; ++i) {
      float sum = 0.0F;
      for (std::uint32_t s = 0; s < kStreams; ++s) {
        float v = row[std::size_t{s} * hidden + i];
        if (block_out != nullptr)
          v = std::fma(block_out[std::size_t{t} * hidden + i],
                       w[std::size_t{t} * kStreams + s], v);
        sum += v;
      }
      dst[i] = sum * (1.0F / kStreams);
    }
  }
}

// Contiguous ring row runs covering positions [first, first + n): a run of
// `count` rows starting at ring row `ring_row` lands at output row `offset`.
struct RingRun {
  std::size_t ring_row, count, offset;
};
inline std::vector<RingRun> RingRuns(std::size_t first, std::size_t n,
                                     std::size_t rows) {
  if (n > rows)
    throw std::invalid_argument("more positions than the ring holds");
  std::vector<RingRun> out;
  std::size_t done = 0;
  while (done < n) {
    const std::size_t r = (first + done) % rows;
    const std::size_t count = std::min(n - done, rows - r);
    out.push_back({r, count, done});
    done += count;
  }
  return out;
}

inline void AppendRow(std::string& out, std::span<const float> features,
                      std::int32_t token, std::int32_t mask) {
  for (float v : features) {
    const std::uint16_t b = ToBf16(v);
    out.append(reinterpret_cast<const char*>(&b), sizeof(b));
  }
  out.append(reinterpret_cast<const char*>(&token), sizeof(token));
  out.append(reinterpret_cast<const char*>(&mask), sizeof(mask));
}

// Walks one sequence and returns the packed rows of every window, in order.
// Every forward call is 2..call_width positions (never one: a one-position
// prefill takes another arithmetic route) and at most ring_rows, so each
// call's taps are all still in the ring when they are read.
// One decode cycle of the verify path (item 2.10): a kVerify forward of k
// rows (the anchor and k - 1 drafts), of which the first `keep` are committed
// and the rest rolled back, as Session::FinishDecode does. k = 1 is an
// ordinary kDecode step.
struct Cycle {
  std::uint32_t k;
  std::uint32_t keep;
};
inline constexpr std::uint32_t kMaxChain =
    8;  // the drafter's block (verify width)

inline void ValidateSchedule(std::span<const Cycle> schedule) {
  for (std::size_t i = 0; i < schedule.size(); ++i)
    if (schedule[i].k < 1 || schedule[i].k > kMaxChain ||
        schedule[i].keep < 1 || schedule[i].keep > schedule[i].k)
      throw std::invalid_argument("cycle " + std::to_string(i) +
                                  " needs 1 <= keep <= k <= 8");
}

// Walks one sequence and returns the packed rows of every window, in order.
// Prompt positions are forwarded in kPrefill calls of 2..call_width positions
// (never one: a one-position prefill takes another arithmetic route), each at
// most ring_rows, so each call's taps are all still in the ring when read.
//
// Positions inside a `verify` span are fed the way serving decodes them
// (Trunk::Verify(tokens, keep) / Trunk::Decode(token)), cycle by cycle from
// `schedule`, taken cyclically and restarted for every sequence. The true
// next tokens stand in for the drafts: a committed row only sees the tokens
// before it, so its arithmetic is the served cycle's. A cycle's k is clipped
// to the tokens left in the sequence, its keep to the verify span. A prompt
// segment of one position is widened by one (prefill rows do not depend on
// the call width), and refused if the sequence ends there.
template<class Trunk>
std::string DumpDflashSequence(Trunk& trunk, std::span<const std::int32_t> x,
                               std::span<const Window> windows,
                               std::span<const Span> loss, std::size_t slots,
                               std::size_t hidden, std::size_t call_width,
                               std::size_t ring_rows,
                               std::span<const Span> verify = {},
                               std::span<const Cycle> schedule = {},
                               std::size_t max_rows = 2000) {
  ValidateWindows(windows, x.size(), max_rows);
  ValidateLoss(loss, x.size());
  ValidateLoss(verify, x.size());
  ValidateSchedule(schedule);
  if (!verify.empty() && schedule.empty())
    throw std::invalid_argument("verify spans need a cycle schedule");
  if (call_width > ring_rows || (!verify.empty() && kMaxChain > ring_rows))
    throw std::invalid_argument("a forward call wider than the capture ring");
  std::string out;
  if (windows.empty())
    return out;
  const std::size_t width = slots * hidden;
  std::vector<float> feats(std::max<std::size_t>(call_width, kMaxChain) *
                           width);
  const std::size_t last = windows.back().end;
  const auto emit = [&](std::size_t lo, std::size_t hi) {
    for (const auto& w : windows) {  // the windows' rows inside [lo, hi)
      const std::size_t a = std::max<std::size_t>(w.begin, lo);
      const std::size_t b = std::min<std::size_t>(w.end, hi);
      if (a >= b)
        continue;
      trunk.ReadRing(a, b - a, feats.data());
      for (std::size_t t = a; t < b; ++t)
        AppendRow(out,
                  std::span<const float>(feats.data() + (t - a) * width, width),
                  x[t], InLoss(loss, t) ? 1 : 0);
    }
  };
  std::size_t pos = 0, cycle = 0, vi = 0;
  while (pos < last) {
    while (vi < verify.size() && verify[vi].end <= pos)
      ++vi;
    if (vi < verify.size() && verify[vi].begin <= pos) {
      const Cycle c = schedule[cycle++ % schedule.size()];
      const std::size_t k = std::min<std::size_t>(c.k, x.size() - pos);
      const std::size_t keep =
          std::min<std::size_t>({c.keep, k, verify[vi].end - pos});
      if (k == 1)
        trunk.Decode(x[pos]);
      else
        trunk.Verify(x.subspan(pos, k), keep);
      emit(pos, pos + keep);
      pos += keep;
      continue;
    }
    std::size_t end = vi < verify.size()
                          ? std::min<std::size_t>(verify[vi].begin, last)
                          : last;
    if (end - pos == 1) {
      if (pos + 2 > x.size() ||
          (vi < verify.size() && verify[vi].begin == pos + 1))
        throw std::invalid_argument("a one-position prompt segment");
      end = pos + 2;
    }
    for (const auto n : Split(end - pos, call_width)) {
      trunk.Forward(x.subspan(pos, n));
      emit(pos, pos + n);
      pos += n;
    }
  }
  return out;
}

}  // namespace gufo::tools::dflash2
