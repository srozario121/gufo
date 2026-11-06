// DFlash-2 tap parity probe (gufo-mtp-continuous-learning, todo 2.4 T3): the
// port's own in-engine capture, written in dump_hidden's DFlash-2 row layout
// so that the two can be compared value by value.
//
// It loads the target and the drafter through Model::Load, as `gufo serve
// --speculative dflash2 --dflash-model` does (optionally with the trimmed
// head), and feeds each sequence through the engine's own Session, then reads
// the session's capture ring (Session::CopyDflashTaps): the rows the drafter
// reads at serve time.
//
//   dflash_tap_parity --model FIRST_SHARD.gguf --dflash-model DRAFTER.gguf
//       [--head-trim-set IDS --head-trim-threshold 0.25] [--max-context N]
//   request: out=/path/rows.bin seq=/path/a.i32 first=A end=B feed=sync
//            out=/path/rows.bin seq=/path/a.i32 first=A end=B feed=evaluate:K
//
// feed=sync forwards x[0, B) with Session::Sync (the prompt path).
// feed=evaluate:K forwards x[0, K) with Sync, then x[K, B) one token at a time
// with Session::Evaluate (the decode path). Either way rows A..B-1 are
// written: bf16 features[taps x hidden] | int32 x[t] | int32 0. B - A must fit
// the ring (the drafter's window + 8 rows). Reply `ok <out> <rows>` or
// `err <message>`.
//
// feed=decode:N and feed=decode4:N (item 2.10, the verify path) decode for
// real, with the drafter, and record each step's cycle; see ServeDecode.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/engine.hpp"
#include "tests/models/qwen38_flash_next/dflash_rows.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace d = gufo::tools::dflash2;

namespace {

struct Fail : std::runtime_error {
  using std::runtime_error::runtime_error;
};

std::vector<std::int32_t> ReadTokens(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in)
    throw Fail("cannot open " + path);
  const auto bytes = static_cast<std::size_t>(in.tellg());
  if (bytes % 4 != 0 || bytes == 0)
    throw Fail(path + " is not int32 tokens");
  std::vector<std::int32_t> x(bytes / 4);
  in.seekg(0);
  in.read(reinterpret_cast<char*>(x.data()),
          static_cast<std::streamsize>(bytes));
  return x;
}

void WriteTokens(const std::string& path, std::span<const std::int32_t> x) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(x.data()),
            static_cast<std::streamsize>(x.size() * sizeof(std::int32_t)));
  if (!out)
    throw Fail("write failed on " + path);
}

std::string SplitAt(std::string& list) {
  const auto comma = list.find(',');
  std::string head = list.substr(0, comma);
  list = comma == std::string::npos ? std::string() : list.substr(comma + 1);
  return head;
}

// feed=decode:N (one session, Session::DecodeStep) or feed=decode4:N (four
// sessions, Session::DecodeBatch, as serving decodes four requests): x[0, K)
// is synced (K = first), then the engine decodes greedily with its drafter
// until a session holds N generated positions (stop tokens do not stop it).
// Written per session: the committed tokens (tokens=), each step's cycle as
// "k keep" lines (sched=; k = 1 + drafted, or 1 for a plain decode step) and
// the ring rows of positions [K, P) (out=), P the session's final position.
// For decode4, seq=, out=, tokens= and sched= are four comma-separated paths.
// temp= and seed= set the sampler (default greedy); session i uses seed + i.
std::size_t ServeDecode(q::Model& model, std::size_t width,
                        std::string out_list, std::string seq_list, long first,
                        const std::string& feed, std::string tokens_list,
                        std::string sched_list, float temp, long long seed) {
  const bool batched = feed.rfind("decode4:", 0) == 0;
  if (!batched && feed.rfind("decode:", 0) != 0)
    throw Fail("feed must be decode:N or decode4:N");
  const std::size_t n_gen = std::stoul(feed.substr(batched ? 8 : 7));
  const std::size_t sessions = batched ? 4 : 1;
  if (tokens_list.empty() || sched_list.empty() || first < 1 || n_gen < 1)
    throw Fail("decode needs first=K, tokens= and sched=");
  struct One {
    std::string out, tokens, sched;
    std::vector<std::int32_t> x;
    std::unique_ptr<q::Session> session;
    gufo::sampling::SamplerState sampler;
    q::Session::DecodeResult result;
    q::Session::BatchOutcome outcome;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> steps;
    std::uint64_t cycles = 0, drafted = 0;
  };
  std::vector<One> all(sessions);
  std::string error;
  for (std::size_t i = 0; i < all.size(); ++i) {
    auto& o = all[i];
    o.sampler = gufo::sampling::SamplerState(
        {.temperature = temp,
         .seed = seed < 0 ? -1 : seed + static_cast<long long>(i)});
    o.out = SplitAt(out_list);
    o.tokens = SplitAt(tokens_list);
    o.sched = SplitAt(sched_list);
    o.x = ReadTokens(SplitAt(seq_list));
    if (o.out.empty() || static_cast<std::size_t>(first) > o.x.size())
      throw Fail("each session needs out=, tokens=, sched= and seq= of >= K");
    o.session = model.CreateSession(
        gufo::core::SessionMode::kSpeculative,
        static_cast<std::uint32_t>(first + n_gen + 64), &error);
    if (!o.session)
      throw Fail("session failed: " + error);
    if (!o.session->Sync(std::span<const std::int32_t>(
                             o.x.data(), static_cast<std::size_t>(first)),
                         &error))
      throw Fail("sync failed: " + error);
  }
  const auto generated = [&](const One& o) {
    return o.session->Position() - static_cast<std::uint32_t>(first);
  };
  const auto done = [&] {
    for (const auto& o : all)
      if (generated(o) >= n_gen)
        return true;
    return false;
  };
  while (!done()) {
    for (auto& o : all) {
      o.cycles = o.session->Statistics().cycles;
      o.drafted = o.session->Statistics().drafted;
    }
    if (batched) {
      std::vector<q::Session::DecodeRequest> reqs;
      for (auto& o : all)
        reqs.push_back(
            {o.session.get(), 64, &o.sampler, &o.result, false, &o.outcome});
      if (!q::Session::DecodeBatch(reqs, &error))
        throw Fail("decode batch failed: " + error);
    } else if (!all[0].session->DecodeStep(64, all[0].sampler, &all[0].result,
                                           &error, false)) {
      throw Fail("decode step failed: " + error);
    }
    for (auto& o : all) {
      const auto& st = o.session->Statistics();
      const auto keep = static_cast<std::uint32_t>(o.result.tokens.size());
      const auto k =
          st.cycles != o.cycles
              ? static_cast<std::uint32_t>(1 + st.drafted - o.drafted)
              : 1U;
      if (keep == 0)
        throw Fail("a decode step committed nothing");
      o.steps.emplace_back(k, keep);
    }
  }
  std::size_t rows = 0;
  for (auto& o : all) {
    const std::uint32_t p = o.session->Position();
    const auto n = p - static_cast<std::uint32_t>(first);
    std::vector<float> feats(std::size_t{n} * width);
    if (!o.session->CopyDflashTaps(static_cast<std::uint32_t>(first), n, feats,
                                   &error))
      throw Fail(error);
    const auto toks = o.session->Tokens();
    WriteTokens(o.tokens, toks);
    {
      std::ofstream sf(o.sched);
      for (const auto& [k, keep] : o.steps)
        sf << k << ' ' << keep << '\n';
    }
    std::string bytes;
    for (std::uint32_t r = 0; r < n; ++r)
      d::AppendRow(bytes,
                   std::span<const float>(feats.data() + r * width, width),
                   toks[static_cast<std::size_t>(first) + r], 0);
    std::ofstream out(o.out, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out)
      throw Fail("write failed on " + o.out);
    rows += n;
  }
  return rows;
}

std::size_t Serve(q::Model& model, std::size_t width, const std::string& line,
                  std::string& out_path) {
  std::stringstream ss(line);
  std::string tok, seq, feed, tokens_out, sched_out;
  long first = -1, end = -1;
  float temp = 0.0F;
  long long seed = -1;
  while (ss >> tok) {
    const auto eq = tok.find('=');
    if (eq == std::string::npos)
      throw Fail("field '" + tok + "' is not k=v");
    const auto k = tok.substr(0, eq), v = tok.substr(eq + 1);
    if (k == "out")
      out_path = v;
    else if (k == "seq")
      seq = v;
    else if (k == "first")
      first = std::stol(v);
    else if (k == "end")
      end = std::stol(v);
    else if (k == "feed")
      feed = v;
    else if (k == "tokens")
      tokens_out = v;
    else if (k == "sched")
      sched_out = v;
    else if (k == "temp")
      temp = std::stof(v);
    else if (k == "seed")
      seed = std::stoll(v);
    else
      throw Fail("unknown field '" + k + "'");
  }
  if (out_path.empty() || seq.empty() || feed.empty())
    throw Fail("a request needs out=, seq=, first=, end= and feed=");
  if (feed.rfind("decode", 0) == 0)
    return ServeDecode(model, width, out_path, seq, first, feed, tokens_out,
                       sched_out, temp, seed);
  const auto x = ReadTokens(seq);
  if (first < 0 || end <= first || static_cast<std::size_t>(end) > x.size())
    throw Fail("first/end are outside the sequence");
  std::size_t split = static_cast<std::size_t>(end);
  if (feed.rfind("evaluate:", 0) == 0) {
    split = std::stoul(feed.substr(9));
    if (split < 1 || split > static_cast<std::size_t>(end))
      throw Fail("evaluate:K needs 1 <= K <= end");
  } else if (feed != "sync") {
    throw Fail("feed must be sync or evaluate:K");
  }
  std::string error;
  auto session =
      model.CreateSession(gufo::core::SessionMode::kSpeculative,
                          static_cast<std::uint32_t>(x.size() + 1), &error);
  if (!session)
    throw Fail("session failed: " + error);
  if (!session->Sync(std::span<const std::int32_t>(x.data(), split), &error))
    throw Fail("sync failed: " + error);
  for (std::size_t t = split; t < static_cast<std::size_t>(end); ++t)
    if (!session->Evaluate(x[t], &error))
      throw Fail("evaluate failed: " + error);
  if (session->Position() != static_cast<std::uint32_t>(end))
    throw Fail("the session holds " + std::to_string(session->Position()) +
               " positions, not " + std::to_string(end));
  const auto n = static_cast<std::uint32_t>(end - first);
  std::vector<float> feats(std::size_t{n} * width);
  if (!session->CopyDflashTaps(static_cast<std::uint32_t>(first), n, feats,
                               &error))
    throw Fail(error);
  std::string bytes;
  for (std::uint32_t r = 0; r < n; ++r)
    d::AppendRow(bytes, std::span<const float>(feats.data() + r * width, width),
                 x[static_cast<std::size_t>(first) + r], 0);
  const auto part = out_path + ".part";
  {
    std::ofstream out(part, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out)
      throw Fail("write failed on " + part);
  }
  std::filesystem::rename(part, out_path);
  return n;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  q::ModelOptions options;
  options.max_context = 8192;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (a == "--model")
      model_path = next();
    else if (a == "--dflash-model")
      options.dflash_model_path = next();
    else if (a == "--head-trim-set")
      options.head_trim_set_path = next();
    else if (a == "--head-trim-threshold")
      options.head_trim_threshold = std::stof(next());
    else if (a == "--max-context")
      options.max_context = static_cast<std::uint32_t>(std::stoul(next()));
    else {
      std::fprintf(stderr, "unknown option %s (see the file header)\n",
                   a.c_str());
      return 2;
    }
  }
  if (model_path.empty() || options.dflash_model_path.empty()) {
    std::fprintf(stderr, "--model and --dflash-model are required\n");
    return 2;
  }
  std::string error;
  auto model = q::Model::Load(model_path, options, &error);
  if (!model || !model->HasDflash()) {
    std::fprintf(stderr, "dflash_tap_parity: load failed: %s\n", error.c_str());
    return 1;
  }
  const std::size_t width =
      d::kDrafterTaps.size() * model->config().hidden_size;
  std::printf(
      "ready mode=engine-capture features=%zu row_bytes=%zu trimmed=%d\n",
      width, d::RowBytes(d::kDrafterTaps.size(), model->config().hidden_size),
      model->HeadTrimmed() ? 1 : 0);
  std::fflush(stdout);
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty())
      continue;
    std::string out_path;
    try {
      const auto n = Serve(*model, width, line, out_path);
      std::printf("ok %s %zu\n", out_path.c_str(), n);
    } catch (const std::exception& e) {
      std::error_code ec;
      if (!out_path.empty())
        std::filesystem::remove(out_path + ".part", ec);
      std::printf("err %s\n", e.what());
    }
    std::fflush(stdout);
  }
  return 0;
}
