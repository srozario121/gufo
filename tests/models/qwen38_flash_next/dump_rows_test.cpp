// CPU test of the dump-row walk (dump_rows.hpp) against a fake trunk: which
// residual, token and distribution make each row.
#include "tests/models/qwen38_flash_next/dump_rows.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace t = gufo::tools;

namespace {
constexpr std::size_t kHc = 12, kVocab = 50, kK = 5;

float Hidden(std::size_t pos, std::size_t c) {
  return static_cast<float>(pos) * 7.25F + static_cast<float>(c) * 0.5F + 1.0F;
}
float Logit(std::size_t pos, std::size_t v) {
  return std::sin(static_cast<float>(pos * 131 + v * 17)) * 4.0F;
}

// The fake enforces the rules of the 0.19.0 trunk that the first live window
// tripped over: a forward call needs the draft block at the trunk's position
// ("MTP must consume the preceding frontier before prefill"), a call consumes
// its known successors, and a one-position call does not. Advance() consumes
// the frontier and must be given the successor of the last position.
struct FakeTrunk {
  const std::vector<std::int32_t>* x = nullptr;
  std::size_t position = 0, mtp = 0;
  std::size_t prefill_calls = 0, batch_calls = 0, widest = 0, advances = 0;
  bool bad_token = false;
  void Enter(std::size_t n) {
    if (n < 2)
      throw std::runtime_error("a one-position forward call");
    if (mtp != position)
      throw std::runtime_error(
          "MTP must consume the preceding frontier before prefill");
    mtp = position + n - 1;
    position += n;
  }
  void Prefill(std::span<const std::int32_t> tokens) {
    ++prefill_calls;
    Enter(tokens.size());
  }
  void Batch(std::span<const std::int32_t> tokens, float* logits,
             float* hidden) {
    ++batch_calls;
    widest = std::max(widest, tokens.size());
    const std::size_t start = position;
    Enter(tokens.size());
    for (std::size_t j = 0; j < tokens.size(); ++j) {
      for (std::size_t v = 0; v < kVocab; ++v)
        logits[j * kVocab + v] = Logit(start + j, v);
      for (std::size_t c = 0; c < kHc; ++c)
        hidden[j * kHc + c] = Hidden(start + j, c);
    }
  }
  void Advance(std::int32_t next) {
    ++advances;
    if (position >= x->size() || next != (*x)[position])
      bad_token = true;
    if (mtp + 1 != position)
      throw std::runtime_error(
          "the draft block is not exactly one position behind");
    ++mtp;
  }
};

std::string Expected(const std::vector<std::int32_t>& x,
                     const std::vector<t::Window>& ws) {
  std::string out;
  for (const auto& w : ws) {
    for (std::size_t i = w.begin; i < w.end; ++i) {
      std::vector<float> h(kHc), lg(kVocab);
      for (std::size_t c = 0; c < kHc; ++c)
        h[c] = Hidden(i - 1, c);
      for (std::size_t v = 0; v < kVocab; ++v)
        lg[v] = Logit(i, v);
      t::AppendRow(out, h, x[i], t::TopKRow(lg, kK));
    }
  }
  return out;
}

std::string Run(const std::vector<std::int32_t>& x,
                const std::vector<t::Window>& ws, std::size_t batch,
                FakeTrunk* tr = nullptr, std::size_t prefill_chunk = 5) {
  FakeTrunk local;
  FakeTrunk& trunk = tr ? *tr : local;
  trunk.x = &x;
  return t::DumpSequence(trunk, x, ws, kHc, kVocab, kK, batch, prefill_chunk,
                         2000);
}

bool Throws(const std::vector<std::int32_t>& x,
            const std::vector<t::Window>& ws) {
  try {
    Run(x, ws, 8);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}
}  // namespace

int main() {
  int checks = 0;
  const auto ok = [&](bool c, const char* what) {
    ++checks;
    if (!c) {
      std::cerr << "FAIL: " << what << '\n';
      std::exit(1);
    }
  };
  std::vector<std::int32_t> x(200);
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = static_cast<std::int32_t>((i * 37 + 5) % kVocab);

  ok(t::RowBytes(10240, 20) == 20644,
     "row bytes for the real shape are 20,644");
  ok(t::RowBytes(kHc, kK) == kHc * 2 + 4 + kK * 8,
     "row bytes for the test shape");

  // the row contract, for every batch width: row i = (h[i-1], x[i], top-k of
  // the logits at i)
  for (std::size_t batch : {3u, 4u, 8u, 64u}) {
    const std::vector<t::Window> ws{{10, 40}};
    ok(Run(x, ws, batch) == Expected(x, ws),
       "one window: rows are (h[i-1], x[i], top-k at i) at every batch width");
  }
  const std::vector<t::Window> many{
      {1, 9}, {20, 33}, {33, 60}, {61, 62}, {100, 200}};
  for (std::size_t batch : {3u, 4u, 7u, 64u}) {
    ok(Run(x, many, batch) == Expected(x, many),
       "windows at the start, adjacent, one position apart, single-row and at "
       "the end");
  }
  const std::vector<t::Window> gap_one{
      {10, 20},
      {22, 30},
      {32, 40}};  // one skipped position between the forward ranges
  for (std::size_t batch : {3u, 5u, 64u})
    ok(Run(x, gap_one, batch) == Expected(x, gap_one),
       "windows a single position apart are merged, never prefilled alone");
  ok(Run(x, many, 8).size() == (8 + 13 + 27 + 1 + 100) * t::RowBytes(kHc, kK),
     "the output holds exactly one row per window row");

  // work done: prefill only what lies before the windows, never past the last
  // window
  FakeTrunk tr;
  Run(x, {{50, 60}}, 4, &tr, 16);
  ok(tr.position == 60,
     "the trunk advances to the end of the last window and no further");
  ok(tr.widest <= 4, "no batch is wider than asked");
  FakeTrunk tr2;
  Run(x, {{50, 60}}, 4, &tr2, 16);
  ok(tr2.prefill_calls == 4,
     "49 positions before the window are prefilled in 4 chunks, none of one "
     "position (16, 16, 15, 2)");
  ok(!tr2.bad_token && tr2.advances == tr2.prefill_calls + tr2.batch_calls - 1,
     "the frontier is consumed between every two calls with the right token, "
     "and not after the last");

  // the first row's residual is the one before the window, whatever the batch
  // width
  const auto first = Run(x, {{5, 6}}, 8);
  std::vector<float> h(kHc);
  for (std::size_t c = 0; c < kHc; ++c)
    h[c] = Hidden(4, c);
  std::string head;
  for (float v : h) {
    const std::uint16_t b = t::ToBf16(v);
    head.append(reinterpret_cast<const char*>(&b), 2);
  }
  ok(first.compare(0, head.size(), head) == 0,
     "a one-row window carries the residual of the position before it");

  // Split never yields a one-position call and always covers the length
  bool split_ok = true;
  for (std::size_t widest : {3u, 4u, 5u, 64u, 512u})
    for (std::size_t length = 2; length < 1300; ++length) {
      const auto sizes = t::Split(length, widest);
      std::size_t sum = 0;
      for (auto n : sizes) {
        split_ok &= n >= 2 && n <= widest;
        sum += n;
      }
      split_ok &= sum == length;
    }
  ok(split_ok,
     "Split covers every length with calls of 2..widest positions, never one");
  ok(t::Split(65, 64) == std::vector<std::size_t>({63, 2}),
     "65 positions in calls of 64 are 63 and 2, not 64 and 1");
  bool threw = false;
  try {
    t::Split(10, 2);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  ok(threw, "a widest call under 3 positions is refused");

  // refusals
  ok(Throws(x, {{0, 5}}),
     "a window starting at 0 is refused (no residual before it)");
  ok(Throws(x, {{5, 5}}), "an empty window is refused");
  ok(Throws(x, {{150, 201}}), "a window past the sequence is refused");
  ok(Throws(x, {{10, 30}, {20, 40}}), "overlapping windows are refused");
  ok(Throws(x, {{50, 60}, {10, 20}}), "descending windows are refused");
  ok(!Throws(x, {{10, 30}, {30, 40}}),
     "control: adjacent windows are accepted");

  // bf16 and top-k
  ok(t::ToBf16(1.0F) == 0x3F80, "bf16 of 1.0");
  ok(t::ToBf16(1.00390625F) == 0x3F80, "bf16 rounds a tie to even (down)");
  ok(t::ToBf16(1.01171875F) == 0x3F82, "bf16 rounds a tie to even (up)");
  ok(t::ToBf16(std::numeric_limits<float>::quiet_NaN()) != 0x7F80,
     "bf16 keeps a NaN a NaN");
  std::vector<float> lg{1.0F, 3.0F, 3.0F, 0.5F, 2.0F};
  const auto top = t::TopKRow(lg, 3);
  ok(top.ids[0] == 1 && top.ids[1] == 2 && top.ids[2] == 4,
     "top-k orders by logit and breaks a tie to the lower id");
  double z = 0;
  for (float v : lg)
    z += std::exp(static_cast<double>(v));
  ok(std::abs(top.logprobs[0] - (3.0 - std::log(z))) < 1e-6,
     "log-probabilities are logit minus the log-sum-exp over the whole "
     "vocabulary");

  std::printf("dump_rows_test: %d checks passed\n", checks);
  return 0;
}
