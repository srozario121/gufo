// CPU test of the DFlash-2 feature walk (dflash_rows.hpp) against a fake
// trunk that writes its capture ring with the CPU reference of the engine's
// HcTapMeanToRing kernel. Each guarantee has a negative control: a tap at the
// wrong layer, no stream mean, and an off-by-one position are all caught.
#include "tests/models/qwen38_flash_next/dflash_rows.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace d = gufo::tools::dflash2;
namespace t = gufo::tools;

namespace {
constexpr std::uint32_t kLayers = 48, kHidden = 8;
constexpr std::size_t kSlots = d::kDrafterTaps.size();

// A layer's settled output, stream s, position p, channel c: on a 2^-12 grid
// so the deferred form (res - 0.5 * 0.25) and its fma are exact.
float Out(std::uint32_t layer, std::uint32_t s, std::size_t p, std::size_t c) {
  const double v =
      std::sin(layer * 1.7 + s * 0.9 + static_cast<double>(p) * 0.31 +
               static_cast<double>(c) * 0.13);
  return static_cast<float>(std::round(v * 4096.0) / 4096.0);
}

// The fake's mode-dependent arithmetic: exact multiples of 2^-6.
float Bias(char mode, std::uint32_t n, std::uint32_t j) {
  if (mode == 'p')
    return 0.0F;
  if (mode == 'd')
    return 1.0F;
  return 0.5F + 0.015625F * static_cast<float>(n * 8 + j);
}

struct FakeTrunk {
  // the capture under test; the defaults are the engine's contract
  std::vector<std::uint32_t> taps{d::kDrafterTaps.begin(),
                                  d::kDrafterTaps.end()};
  std::uint32_t layer_offset = 0;  // breakage: tap the GGUF layer-input index
  bool stream0_only = false;       // breakage: no mean over the streams
  long write_skew = 0;             // breakage: ring row of position p + skew
  long read_skew = 0;              // breakage: read position p + skew
  bool deferred = false;           // the residual still owes its update
  bool ignore_deferred = false;    // breakage: the tap does not fold it in
  std::uint32_t rows = 16;
  std::vector<float> ring;
  std::size_t position = 0, calls = 0, narrowest = 1 << 30, widest = 0;

  // verify path: the arithmetic of a row may depend on the mode, the call
  // width and its offset in the call (Bias); prefill rows are unbiased
  bool ignore_keep = false;  // breakage: commit every row of a verify call
  std::size_t verifies = 0, decodes = 0;
  void Verify(std::span<const std::int32_t> tokens, std::size_t keep) {
    ++verifies;
    Write(static_cast<std::uint32_t>(tokens.size()), 'v');
    position -= tokens.size() - (ignore_keep ? tokens.size() : keep);
  }
  void Decode(std::int32_t) {
    ++decodes;
    Write(1, 'd');
  }
  void Forward(std::span<const std::int32_t> tokens) {
    const auto n = static_cast<std::uint32_t>(tokens.size());
    ++calls;
    narrowest = std::min<std::size_t>(narrowest, n);
    widest = std::max<std::size_t>(widest, n);
    Write(n, 'p');
  }
  void Write(std::uint32_t n, char mode) {
    if (ring.empty())
      ring.assign(std::size_t{rows} * kSlots * kHidden, NAN);
    for (std::uint32_t slot = 0; slot < taps.size(); ++slot) {
      const std::uint32_t layer = taps[slot] + layer_offset;
      std::vector<float> res(std::size_t{n} * d::kStreams * kHidden);
      std::vector<float> bo(std::size_t{n} * kHidden, 0.5F),
          w(std::size_t{n} * d::kStreams, 0.25F);
      for (std::uint32_t j = 0; j < n; ++j)
        for (std::uint32_t s = 0; s < d::kStreams; ++s)
          for (std::uint32_t c = 0; c < kHidden; ++c) {
            const std::uint32_t src = stream0_only ? 0 : s;
            float v = Out(layer, src, position + j, c) + Bias(mode, n, j);
            if (deferred)
              v -= 0.5F * 0.25F;
            res[(std::size_t{j} * d::kStreams + s) * kHidden + c] = v;
          }
      const bool fold = deferred && !ignore_deferred;
      d::TapMeanToRingReference(
          res.data(), fold ? bo.data() : nullptr, fold ? w.data() : nullptr,
          ring.data(), static_cast<std::uint32_t>(position + rows + write_skew),
          rows, slot, kSlots, n, kHidden);
    }
    position += n;
  }
  void ReadRing(std::size_t first, std::size_t n, float* out) const {
    for (const auto& r : d::RingRuns(
             static_cast<std::size_t>(static_cast<long>(first) + read_skew), n,
             rows))
      std::copy_n(ring.data() + r.ring_row * kSlots * kHidden,
                  r.count * kSlots * kHidden,
                  out + r.offset * kSlots * kHidden);
  }
};

// The contract, computed independently of the walk and the ring.
std::string Expected(const std::vector<std::int32_t>& x,
                     const std::vector<t::Window>& ws,
                     const std::vector<d::Span>& loss) {
  std::string out;
  for (const auto& w : ws)
    for (std::size_t p = w.begin; p < w.end; ++p) {
      std::vector<float> f;
      for (auto layer : d::kDrafterTaps)
        for (std::size_t c = 0; c < kHidden; ++c) {
          float sum = 0.0F;
          for (std::uint32_t s = 0; s < d::kStreams; ++s)
            sum += Out(layer, s, p, c);
          f.push_back(sum * 0.25F);
        }
      bool m = false;
      for (const auto& sp : loss)
        m |= p >= sp.begin && p < sp.end;
      d::AppendRow(out, f, x[p], m ? 1 : 0);
    }
  return out;
}

std::string Run(FakeTrunk& tr, const std::vector<std::int32_t>& x,
                const std::vector<t::Window>& ws,
                const std::vector<d::Span>& loss, std::size_t width) {
  return d::DumpDflashSequence(tr, x, ws, loss, kSlots, kHidden, width,
                               tr.rows);
}

// The verify-path contract, computed independently of the walk: every
// committed position's mode, call width and offset under the schedule.
std::string ExpectedVerify(const std::vector<std::int32_t>& x,
                           const std::vector<t::Window>& ws,
                           const std::vector<d::Span>& loss,
                           const std::vector<d::Span>& verify,
                           const std::vector<d::Cycle>& sched) {
  struct Info {
    char mode = 'p';
    std::uint32_t n = 0, j = 0;
  };
  std::vector<Info> info(x.size());
  std::size_t cycle = 0;
  for (const auto& v : verify)
    for (std::size_t p = v.begin; p < v.end;) {
      const auto c = sched[cycle++ % sched.size()];
      const std::size_t k = std::min<std::size_t>(c.k, x.size() - p);
      const std::size_t keep = std::min<std::size_t>({c.keep, k, v.end - p});
      for (std::size_t j = 0; j < keep; ++j)
        info[p + j] = {k == 1 ? 'd' : 'v', static_cast<std::uint32_t>(k),
                       static_cast<std::uint32_t>(j)};
      p += keep;
    }
  std::string out;
  for (const auto& w : ws)
    for (std::size_t p = w.begin; p < w.end; ++p) {
      std::vector<float> f;
      for (auto layer : d::kDrafterTaps)
        for (std::size_t c = 0; c < kHidden; ++c) {
          float sum = 0.0F;
          for (std::uint32_t st = 0; st < d::kStreams; ++st)
            sum +=
                Out(layer, st, p, c) + Bias(info[p].mode, info[p].n, info[p].j);
          f.push_back(sum * 0.25F);
        }
      bool m = false;
      for (const auto& sp : loss)
        m |= p >= sp.begin && p < sp.end;
      d::AppendRow(out, f, x[p], m ? 1 : 0);
    }
  return out;
}

std::string RunVerify(FakeTrunk& tr, const std::vector<std::int32_t>& x,
                      const std::vector<t::Window>& ws,
                      const std::vector<d::Span>& loss,
                      const std::vector<d::Span>& verify,
                      const std::vector<d::Cycle>& sched, std::size_t width) {
  return d::DumpDflashSequence(tr, x, ws, loss, kSlots, kHidden, width, tr.rows,
                               verify, sched);
}

template<class F>
bool Throws(F f) {
  try {
    f();
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
  std::vector<std::int32_t> x(300);
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = static_cast<std::int32_t>((i * 37 + 5) % 1000);
  const std::vector<t::Window> ws{{1, 9}, {20, 60}, {60, 61}, {100, 230}};
  const std::vector<d::Span> loss{{5, 9}, {40, 61}, {150, 230}};
  const auto want = Expected(x, ws, loss);

  ok(d::RowBytes(5, 2560) == 25608, "row bytes for the real shape are 25,608");
  ok(want.size() == (8 + 40 + 1 + 130) * d::RowBytes(kSlots, kHidden),
     "one row per window position");

  // the contract at every call width and ring size, including ring wrap-around
  for (std::uint32_t rows : {7u, 16u, 64u, 2056u})
    for (std::size_t width : {3u, 5u, 7u, 64u, 512u}) {
      if (width > rows)
        continue;
      FakeTrunk tr;
      tr.rows = rows;
      ok(Run(tr, x, ws, loss, width) == want,
         "rows are (mean of 4 streams at layers 5..45 at t, x[t], mask[t])");
      ok(tr.narrowest >= 2 && tr.widest <= width,
         "every forward call is 2..width positions");
      ok(tr.position == 230,
         "the trunk advances to the last window's end and no further");
    }
  {
    FakeTrunk tr;
    tr.deferred = true;
    ok(Run(tr, x, ws, loss, 7) == want,
       "a deferred residual update is folded in before the mean");
  }

  // negative controls: each breakage of the capture is caught by the contract
  {
    FakeTrunk tr;
    tr.layer_offset = 1;
    ok(Run(tr, x, ws, loss, 7) != want,
       "caught: taps at the GGUF layer-input indices 6..46 (no N-1 rule)");
  }
  {
    FakeTrunk tr;
    tr.stream0_only = true;
    ok(Run(tr, x, ws, loss, 7) != want,
       "caught: stream 0 instead of the mean over the 4 streams");
  }
  {
    FakeTrunk tr;
    tr.deferred = true;
    tr.ignore_deferred = true;
    ok(Run(tr, x, ws, loss, 7) != want,
       "caught: a tap that reads the residual before its deferred update");
  }
  for (long skew : {-1L, 1L}) {
    FakeTrunk tr;
    tr.write_skew = skew;
    ok(Run(tr, x, ws, loss, 7) != want,
       "caught: the ring written one position off");
    FakeTrunk rd;
    rd.read_skew = skew;
    ok(Run(rd, x, ws, loss, 7) != want,
       "caught: the ring read one position off (h[t-1] paired with x[t])");
  }
  {
    std::vector<d::Span> shifted;
    for (auto s : loss)
      shifted.push_back({s.begin + 1, s.end + 1});
    FakeTrunk tr;
    ok(Run(tr, x, ws, loss, 7) != Expected(x, ws, shifted),
       "caught: a loss mask one position off");
    std::vector<std::int32_t> y(x.begin() + 1, x.end());
    y.push_back(0);
    FakeTrunk tr2;
    ok(Run(tr2, y, ws, loss, 7) != want, "caught: input ids one position off");
  }

  // the drafter's taps
  const std::vector<std::uint32_t> real{5, 15, 25, 35, 45},
      gguf{6, 16, 26, 36, 46};
  ok(!Throws([&] { d::CheckTaps(real, d::kDrafterTaps, kLayers); }),
     "control: taps 5,15,25,35,45 are accepted");
  ok(Throws([&] { d::CheckTaps(gguf, d::kDrafterTaps, kLayers); }),
     "taps 6..46 (GGUF numbering) are refused");
  ok(Throws([&] {
       d::CheckTaps(std::vector<std::uint32_t>{5, 15, 25, 35}, d::kDrafterTaps,
                    kLayers);
     }),
     "a missing tap is refused");
  ok(Throws([&] { d::CheckTaps(real, real, 46); }),
     "a tap at the last layer is refused");

  // ring runs
  const auto runs = d::RingRuns(14, 5, 16);
  ok(runs.size() == 2 && runs[0].ring_row == 14 && runs[0].count == 2 &&
         runs[1].ring_row == 0 && runs[1].count == 3 && runs[1].offset == 2,
     "ring runs wrap at the ring's end");
  ok(Throws([] { d::RingRuns(0, 17, 16); }),
     "more positions than the ring holds are refused");

  // refusals
  {
    FakeTrunk tr;
    ok(Throws([&] { Run(tr, x, ws, loss, 17); }),
       "a call wider than the ring is refused");
    ok(Throws([&] { Run(tr, x, ws, {{10, 20}, {15, 25}}, 7); }),
       "overlapping loss spans are refused");
    ok(Throws([&] { Run(tr, x, ws, {{10, 10}}, 7); }),
       "an empty loss span is refused");
    ok(Throws([&] { Run(tr, x, ws, {{290, 301}}, 7); }),
       "a loss span past the sequence is refused");
    ok(Throws([&] { Run(tr, x, {{0, 5}}, loss, 7); }),
       "a window at position 0 is refused (as in MTP mode)");
  }

  // ---- the verify path (item 2.10): prompt rows prefilled, output rows in
  // decode cycles of the schedule, committed keep rows at a time
  {
    const std::vector<d::Span> vspans{{20, 59}, {150, 229}};
    const std::vector<d::Cycle> served{{6, 4}, {8, 8}, {1, 1}, {5, 2}, {7, 7}};
    const auto vwant = ExpectedVerify(x, ws, loss, vspans, served);
    for (std::uint32_t rows : {16u, 64u, 2056u})
      for (std::size_t width : {3u, 7u, 16u}) {
        FakeTrunk tr;
        tr.rows = rows;
        ok(RunVerify(tr, x, ws, loss, vspans, served, width) == vwant,
           "verify rows carry their cycle's mode, width and offset; prompt "
           "rows stay prefill rows");
        // the last window's final prompt segment [229, 230) is one position,
        // widened to two: the trunk ends at 231
        ok(tr.position == 231 && tr.verifies > 0 && tr.decodes > 0 &&
               tr.narrowest >= 2,
           "a one-position prompt segment is widened to two, and the walk used "
           "verify and decode steps");
      }
    FakeTrunk plain;
    ok(RunVerify(plain, x, ws, loss, {}, {}, 7) == want,
       "control: without verify spans the walk is the prompt path, unchanged");
    const std::vector<d::Cycle> fixed{{8, 8}};
    FakeTrunk f8;
    ok(RunVerify(f8, x, ws, loss, vspans, fixed, 7) ==
           ExpectedVerify(x, ws, loss, vspans, fixed),
       "a fixed 8:8 schedule commits every row of each width-8 call");
    // negative controls
    FakeTrunk nk;
    nk.ignore_keep = true;
    ok(RunVerify(nk, x, ws, loss, vspans, served, 7) != vwant,
       "caught: a verify call whose rejected rows are not rolled back");
    std::vector<d::Cycle> rotated(served.begin() + 1, served.end());
    rotated.push_back(served.front());
    FakeTrunk r1;
    ok(RunVerify(r1, x, ws, loss, vspans, served, 7) !=
           ExpectedVerify(x, ws, loss, vspans, rotated),
       "caught: the schedule consumed one cycle off");
    FakeTrunk r2;
    ok(RunVerify(r2, x, ws, loss, vspans, served, 7) != want,
       "caught: output rows fed through the prompt path");
    std::vector<d::Span> shifted{{21, 59}, {151, 229}};
    FakeTrunk r3;
    ok(RunVerify(r3, x, ws, loss, vspans, served, 7) !=
           ExpectedVerify(x, ws, loss, shifted, served),
       "caught: verify spans one position off");
    FakeTrunk r4;
    ok(Throws([&] {
         RunVerify(r4, x, ws, loss, vspans, std::vector<d::Cycle>{{9, 9}}, 7);
       }),
       "a cycle wider than the drafter's block (8) is refused");
    ok(Throws([&] {
         RunVerify(r4, x, ws, loss, vspans, std::vector<d::Cycle>{{4, 5}}, 7);
       }),
       "a cycle keeping more rows than it forwarded is refused");
    ok(Throws([&] { RunVerify(r4, x, ws, loss, vspans, {}, 7); }),
       "verify spans without a schedule are refused");
    ok(Throws([&] {
         RunVerify(r4, x, ws, loss, {{20, 59}, {60, 229}}, served, 7);
       }),
       "a one-position prompt segment before a verify span is refused");
  }

  std::printf("dflash_rows_test: %d checks passed\n", checks);
  return 0;
}
