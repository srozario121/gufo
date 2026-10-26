// Row extraction for the MTP-head training dump (the CPU-testable half of dump_hidden.cpp).
//
// One dump row for the token at position i of a sequence x is
//   residual  h[i-1] as bf16 (the trunk's wide residual after position i-1),
//   token     x[i]   (int32, the token the draft step embeds),
//   top-k     the trunk's top-k ids and log-probabilities at position i (the distribution of x[i+1]),
// so (h[i-1], x[i]) is the draft step whose target is that distribution. 20,644 bytes at k = 20.
// A window [a, b) emits the rows i = a .. b-1; it needs the residuals of positions a-1 .. b-2 and the logits of positions
// a .. b-1, which is why a >= 1 and b <= len.
//
// The walk is written against a `Trunk` so the CPU test can stand in for the GPU executor:
//   Prefill(span<const int32_t> tokens)                          advance with no outputs
//   Batch(span<const int32_t> tokens, float* logits, float* hidden)  advance, logits (n x vocab) and residuals (n x hc)
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
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return static_cast<std::uint16_t>((u >> 16) | 0x40u);  // NaN stays a NaN
  std::uint16_t b = static_cast<std::uint16_t>(u >> 16);
  const std::uint32_t frac = u & 0xFFFFu;
  if (frac > 0x8000u || (frac == 0x8000u && (b & 1u) != 0u)) ++b;  // round to nearest even
  return b;
}

struct TopK {
  std::vector<std::int32_t> ids;
  std::vector<float> logprobs;
};

// Top-k by logit (ties to the lower id), log-probability = logit - logsumexp over the full vocabulary (temperature 1.0).
inline TopK TopKRow(std::span<const float> logits, std::size_t k) {
  const std::size_t vocab = logits.size();
  k = std::min(k, vocab);
  std::vector<std::uint32_t> idx(vocab);
  std::iota(idx.begin(), idx.end(), 0u);
  const auto better = [&](std::uint32_t a, std::uint32_t b) {
    return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
  };
  std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(), better);
  float max_v = logits[idx[0]];
  double sum = 0.0;
  for (float v : logits) sum += std::exp(static_cast<double>(v) - max_v);
  const double lse = max_v + std::log(sum);
  TopK out;
  for (std::size_t i = 0; i < k; ++i) {
    out.ids.push_back(static_cast<std::int32_t>(idx[i]));
    out.logprobs.push_back(static_cast<float>(logits[idx[i]] - lse));
  }
  return out;
}

inline std::size_t RowBytes(std::size_t hc, std::size_t k) { return hc * 2 + 4 + k * 4 + k * 4; }

inline void AppendRow(std::string& out, std::span<const float> residual, std::int32_t token, const TopK& top) {
  for (float v : residual) {
    const std::uint16_t b = ToBf16(v);
    out.append(reinterpret_cast<const char*>(&b), sizeof(b));
  }
  out.append(reinterpret_cast<const char*>(&token), sizeof(token));
  out.append(reinterpret_cast<const char*>(top.ids.data()), top.ids.size() * sizeof(std::int32_t));
  out.append(reinterpret_cast<const char*>(top.logprobs.data()), top.logprobs.size() * sizeof(float));
}

// Windows must be ascending and not overlapping, a >= 1, a < b <= len. Throws std::invalid_argument naming the first fault.
inline void ValidateWindows(std::span<const Window> w, std::size_t len, std::size_t max_rows) {
  std::uint32_t prev_end = 1;
  for (std::size_t i = 0; i < w.size(); ++i) {
    const auto id = "window " + std::to_string(i);
    if (w[i].begin < 1) throw std::invalid_argument(id + ": begins before position 1 (a row needs the residual before it)");
    if (w[i].begin >= w[i].end) throw std::invalid_argument(id + ": is empty");
    if (w[i].end > len) throw std::invalid_argument(id + ": ends past the sequence");
    if (w[i].end - w[i].begin > max_rows) throw std::invalid_argument(id + ": has more rows than the maximum");
    if (w[i].begin < prev_end && i > 0) throw std::invalid_argument(id + ": overlaps the previous window");
    prev_end = w[i].end;
  }
}

// Walks one sequence and returns the packed rows of every window, in order. `batch` is the widest Batch call.
template <class Trunk>
std::string DumpSequence(Trunk& trunk, std::span<const std::int32_t> x, std::span<const Window> windows, std::size_t hc,
                         std::size_t vocab, std::size_t k, std::size_t batch, std::size_t prefill_chunk = 512,
                         std::size_t max_rows = 2000) {
  ValidateWindows(windows, x.size(), max_rows);
  std::string out;
  std::vector<float> logits(batch * vocab), hidden(batch * hc), carry(hc);
  std::size_t pos = 0;           // positions [0, pos) are in the trunk
  std::int64_t carry_pos = -1;   // the position whose residual `carry` holds
  for (const auto& w : windows) {
    std::size_t s;
    if (carry_pos >= 0 && static_cast<std::size_t>(carry_pos) + 1 == w.begin && pos == w.begin) {
      s = pos;  // the previous window ended right before this one: its last residual is the one this window needs first
    } else {
      if (w.begin - 1 < pos) throw std::invalid_argument("window needs a residual that was not kept");
      while (pos < w.begin - 1) {
        const std::size_t n = std::min(prefill_chunk, w.begin - 1 - pos);
        trunk.Prefill(x.subspan(pos, n));
        pos += n;
      }
      carry_pos = -1;
      s = pos;
    }
    while (s < w.end) {
      const std::size_t n = std::min(batch, w.end - s);
      trunk.Batch(x.subspan(s, n), logits.data(), hidden.data());
      for (std::size_t j = 0; j < n; ++j) {
        const std::size_t i = s + j;   // logits row j is position i
        if (i < w.begin) continue;     // the residual-only position a-1
        const float* h = j > 0 ? hidden.data() + (j - 1) * hc : carry.data();
        AppendRow(out, std::span<const float>(h, hc), x[i], TopKRow(std::span<const float>(logits.data() + j * vocab, vocab), k));
      }
      std::copy_n(hidden.data() + (n - 1) * hc, hc, carry.begin());
      carry_pos = static_cast<std::int64_t>(s + n - 1);
      s += n;
      pos = s;
    }
  }
  return out;
}

}  // namespace gufo::tools
