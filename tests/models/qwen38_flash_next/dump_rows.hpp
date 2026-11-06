// Row extraction for the MTP-head training dump (the CPU-testable half of
// dump_hidden.cpp).
//
// One dump row for the token at position i of a sequence x is
//   residual  h[i-1] as bf16 (the trunk's wide residual after position i-1),
//   token     x[i]   (int32, the token the draft step embeds),
//   top-k     the trunk's top-k ids and log-probabilities at position i (the
//   distribution of x[i+1]),
// so (h[i-1], x[i]) is the draft step whose target is that distribution. 20,644
// bytes at k = 20. A window [a, b) emits the rows i = a .. b-1; it needs the
// residuals of positions a-1 .. b-2 and the logits of positions a .. b-1, which
// is why a >= 1 and b <= len.
//
// The walk is written against a `Trunk` so the CPU test can stand in for the
// GPU executor:
//   Prefill(span<const int32_t> tokens)                          advance with
//   no outputs Batch(span<const int32_t> tokens, float* logits, float* hidden)
//   advance, logits (n x vocab) and residuals (n x hc)
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gufo::tools {

struct Window {
  std::uint32_t begin;  // first row's token position a (>= 1)
  std::uint32_t end;    // one past the last row, b (<= len)
};

inline std::uint16_t ToBf16(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  if ((u & 0x7FFFFFFFu) > 0x7F800000u)
    return static_cast<std::uint16_t>((u >> 16) | 0x40u);  // NaN stays a NaN
  std::uint16_t b = static_cast<std::uint16_t>(u >> 16);
  const std::uint32_t frac = u & 0xFFFFu;
  if (frac > 0x8000u || (frac == 0x8000u && (b & 1u) != 0u))
    ++b;  // round to nearest even
  return b;
}

struct TopK {
  std::vector<std::int32_t> ids;
  std::vector<float> logprobs;
};

// Top-k by logit (ties to the lower id), log-probability = logit - logsumexp
// over the full vocabulary (temperature 1.0).
inline TopK TopKRow(std::span<const float> logits, std::size_t k) {
  const std::size_t vocab = logits.size();
  k = std::min(k, vocab);
  std::vector<std::uint32_t> idx(vocab);
  std::iota(idx.begin(), idx.end(), 0u);
  const auto better = [&](std::uint32_t a, std::uint32_t b) {
    return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
  };
  std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k),
                    idx.end(), better);
  float max_v = logits[idx[0]];
  double sum = 0.0;
  for (float v : logits)
    sum += std::exp(static_cast<double>(v) - max_v);
  const double lse = max_v + std::log(sum);
  TopK out;
  for (std::size_t i = 0; i < k; ++i) {
    out.ids.push_back(static_cast<std::int32_t>(idx[i]));
    out.logprobs.push_back(static_cast<float>(logits[idx[i]] - lse));
  }
  return out;
}

inline std::size_t RowBytes(std::size_t hc, std::size_t k) {
  return hc * 2 + 4 + k * 4 + k * 4;
}

inline void AppendRow(std::string& out, std::span<const float> residual,
                      std::int32_t token, const TopK& top) {
  for (float v : residual) {
    const std::uint16_t b = ToBf16(v);
    out.append(reinterpret_cast<const char*>(&b), sizeof(b));
  }
  out.append(reinterpret_cast<const char*>(&token), sizeof(token));
  out.append(reinterpret_cast<const char*>(top.ids.data()),
             top.ids.size() * sizeof(std::int32_t));
  out.append(reinterpret_cast<const char*>(top.logprobs.data()),
             top.logprobs.size() * sizeof(float));
}

// Windows must be ascending and not overlapping, a >= 1, a < b <= len. Throws
// std::invalid_argument naming the first fault.
inline void ValidateWindows(std::span<const Window> w, std::size_t len,
                            std::size_t max_rows) {
  std::uint32_t prev_end = 1;
  for (std::size_t i = 0; i < w.size(); ++i) {
    const auto id = "window " + std::to_string(i);
    if (w[i].begin < 1)
      throw std::invalid_argument(
          id +
          ": begins before position 1 (a row needs the residual before it)");
    if (w[i].begin >= w[i].end)
      throw std::invalid_argument(id + ": is empty");
    if (w[i].end > len)
      throw std::invalid_argument(id + ": ends past the sequence");
    if (w[i].end - w[i].begin > max_rows)
      throw std::invalid_argument(id + ": has more rows than the maximum");
    if (w[i].begin < prev_end && i > 0)
      throw std::invalid_argument(id + ": overlaps the previous window");
    prev_end = w[i].end;
  }
}

// Chunk sizes for `length` positions in calls of at most `widest`: never a call
// of one position. The 0.19.0 trunk consumes a prefill call's known successors
// in the same call, so a one-position call leaves the draft block behind by
// more than the frontier.
inline std::vector<std::size_t> Split(std::size_t length, std::size_t widest) {
  if (widest < 3)
    throw std::invalid_argument("the widest call must be at least 3 positions");
  if (length < 2)
    throw std::invalid_argument("a call must cover at least 2 positions");
  std::vector<std::size_t> sizes(length / widest, widest);
  if (const auto rest = length % widest; rest != 0)
    sizes.push_back(rest);
  if (sizes.size() > 1 &&
      sizes.back() == 1) {  // borrow one position from the call before
    --sizes[sizes.size() - 2];
    sizes.back() = 2;
  }
  return sizes;
}

// The positions to forward are the union of [a-1, b) over the windows, merged
// when they touch or lie one position apart (a gap of one position cannot be
// prefilled alone), and starting from 0 when the first would begin at 1.
struct Range {
  std::size_t begin, end;
};
inline std::vector<Range> Ranges(std::span<const Window> windows) {
  std::vector<Range> out;
  for (const auto& w : windows) {
    const Range r{w.begin - 1, w.end};
    if (!out.empty() && r.begin <= out.back().end + 1)
      out.back().end = std::max(out.back().end, r.end);
    else
      out.push_back(r);
  }
  if (!out.empty() && out.front().begin <= 1)
    out.front().begin = 0;
  return out;
}

// Walks one sequence and returns the packed rows of every window, in order.
//   Trunk::Prefill(span tokens)                           advance, no outputs
//   (at least 2 tokens) Trunk::Batch(span tokens, float* logits, float* hidden)
//   advance with logits (n x vocab) and residuals (n x hc), at least 2 tokens
//   Trunk::Advance(int32 next)                            the draft block
//   consumes the frontier: the successor of the last
//                                                         position, called
//                                                         between every two
//                                                         forward calls and
//                                                         never after the last
// `batch` is the widest Batch call (at least 3, at most the trunk's kept rows).
template<class Trunk>
std::string DumpSequence(Trunk& trunk, std::span<const std::int32_t> x,
                         std::span<const Window> windows, std::size_t hc,
                         std::size_t vocab, std::size_t k, std::size_t batch,
                         std::size_t prefill_chunk = 512,
                         std::size_t max_rows = 2000) {
  ValidateWindows(windows, x.size(), max_rows);
  const auto ranges = Ranges(windows);
  std::string out;
  std::vector<float> logits(batch * vocab), hidden(batch * hc), carry(hc);
  std::size_t pos = 0;  // positions [0, pos) are in the trunk
  std::size_t wi = 0;   // the window the next row belongs to
  const std::size_t last = ranges.empty() ? 0 : ranges.back().end;
  const auto frontier = [&] {
    if (pos < last)
      trunk.Advance(x[pos]);
  };
  for (const auto& r : ranges) {
    if (r.begin > pos) {
      for (const auto n : Split(r.begin - pos, prefill_chunk)) {
        trunk.Prefill(x.subspan(pos, n));
        pos += n;
        frontier();
      }
    }
    for (const auto n : Split(r.end - pos, batch)) {
      trunk.Batch(x.subspan(pos, n), logits.data(), hidden.data());
      for (std::size_t j = 0; j < n; ++j) {
        const std::size_t i = pos + j;  // logits row j is position i
        while (wi < windows.size() && i >= windows[wi].end)
          ++wi;
        if (wi == windows.size() || i < windows[wi].begin)
          continue;  // not a row of any window
        const float* h = j > 0 ? hidden.data() + (j - 1) * hc : carry.data();
        AppendRow(
            out, std::span<const float>(h, hc), x[i],
            TopKRow(std::span<const float>(logits.data() + j * vocab, vocab),
                    k));
      }
      std::copy_n(hidden.data() + (n - 1) * hc, hc, carry.begin());
      pos += n;
      frontier();
    }
  }
  return out;
}

}  // namespace gufo::tools
