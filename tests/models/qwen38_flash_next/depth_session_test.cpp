// Depth-14 gate-probe build (v0.34.0 port): past seven drafts a session
// decoding alone drafts to its limit and verifies the chain through
// ForwardWide (segments of at most eight rows, the decode arithmetic). Each
// limit 7..14 is checked on its own against ordinary decoding, greedy with
// and without penalties, on a prompt-lookup prompt and a prose prompt, and
// seeded sampled decoding must replay exactly. A divergence does not stop the
// run: per limit and config the test prints how many tokens matched ordinary
// decoding and where they first diverged. Only limit 7 fails the run (it
// must be exact); limits 8..14 are a recorded diagnostic.
//
// Usage: depth_session_test --model FIRST.gguf --mtp-model MTP.gguf
//          [--vocab-trim-set IDS] [--record OUT.jsonl]
// --vocab-trim-set runs the served head (configuration G: threshold 0.25,
// discard-aware, skip-scored). --record writes one JSON line per limit,
// prompt and config with the tokens, so the limit-7 tokens of this build can
// be compared with the logging-only build's: this file uses only API that
// build has too, and a build that refuses limits past seven records that.
// With GUFO_MTP_GATE_PROBE set to a fresh file the probe records of each
// limit are checked: their layout, and that MTP chains past eight rows (no
// copies) occur past limit 7.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen38_flash_next/engine.hpp"
#include "tests/models/qwen38_flash_next/mtp_replay.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

constexpr std::size_t kTokens = 384;
constexpr std::uint32_t kFirstLimit = 7;
constexpr std::uint32_t kLastLimit = 14;

struct Args {
  std::string model, mtp, trim, record;
};

qfn::ModelOptions Options(const Args& args, std::uint32_t limit) {
  qfn::ModelOptions options{.max_context = 4096,
                            .mtp_model_path = args.mtp,
                            .max_draft_tokens = limit,
                            .decode_concurrency = 1};
  if (!args.trim.empty()) {
    options.head_trim_set_path = args.trim;
    options.head_trim_threshold = 0.25F;
    options.head_trim_discard_aware = true;
    options.head_trim_skip_scored = true;
  }
  return options;
}

std::vector<std::int32_t> Ordinary(qfn::Model& model,
                                   std::span<const std::int32_t> prompt,
                                   const sampling::SamplingConfig& config) {
  std::string error;
  auto session = model.CreateSession(gufo::core::SessionMode::kAutoregressive,
                                     4096, &error);
  Require(session && session->Sync(prompt, &error), error);
  sampling::SamplerState sampler(
      config, std::vector<sampling::TokenId>(prompt.begin(), prompt.end()));
  std::vector<std::int32_t> tokens;
  while (tokens.size() < kTokens) {
    qfn::Session::DecodeResult step;
    Require(session->DecodeStep(1, sampler, &step, &error, false), error);
    Require(step.tokens.size() == 1, "ordinary decoding made no progress");
    tokens.push_back(step.tokens.front());
  }
  return tokens;
}

gufo::test::MtpReplay Speculative(qfn::Model& model,
                                  std::span<const std::int32_t> prompt,
                                  const sampling::SamplingConfig& config) {
  std::string error;
  auto session =
      model.CreateSession(gufo::core::SessionMode::kSpeculative, 4096, &error);
  Require(session && session->Sync(prompt, &error), error);
  sampling::SamplerState sampler(
      config, std::vector<sampling::TokenId>(prompt.begin(), prompt.end()));
  return gufo::test::DecodeMtpReplay(*session, sampler, kTokens);
}

struct Prompt {
  std::string name;
  std::vector<std::int32_t> tokens;
};

std::vector<Prompt> MakePrompts(const qfn::Model& model) {
  std::vector<Prompt> prompts(2);
  prompts[0].name = "repeated";  // prompt lookup copies long chains
  const auto pattern = model.Tokenize(
      "def area(width, height):\n    return width * height\n\n"
      "The quick brown fox jumps over the lazy dog. ");
  Require(!pattern.empty(), "empty prompt pattern");
  for (std::size_t i = 0; i < 1024; ++i)
    prompts[0].tokens.push_back(pattern[i % pattern.size()]);
  prompts[1].name = "prose";  // neural drafts only, mostly
  prompts[1].tokens = model.Tokenize(
      "Write a short, careful explanation of how a binary search tree keeps "
      "its keys ordered, then give a Python class with insert, search and "
      "in-order traversal methods, with comments.");
  Require(!prompts[1].tokens.empty(), "empty prose prompt");
  return prompts;
}

const std::array<sampling::SamplingConfig, 2> kGreedy{{
    {.seed = 1},
    // GreedyMtpPredictions verifies wide chains in pieces of seven rows.
    {.seed = 1,
     .repeat_penalty = 1.1F,
     .repeat_last_n = 64,
     .frequency_penalty = 0.2F,
     .presence_penalty = 0.1F},
}};
const sampling::SamplingConfig kSampled{
    .temperature = 0.7F, .top_k = 40, .top_p = 0.95F, .seed = 73};

std::string List(std::span<const std::int32_t> tokens) {
  std::string s = "[";
  for (std::size_t i = 0; i < tokens.size(); ++i)
    s += (i ? "," : "") + std::to_string(tokens[i]);
  return s + "]";
}

/// One limit's outcome for one prompt and config.
struct Outcome {
  std::uint32_t limit{0};
  std::string prompt;
  std::string config;  ///< "greedy0", "greedy1" or "sampled"
  std::string status;  ///< "ok", "diverged", "error", "refused"
  std::size_t matched{0};
  long first_divergence{-1};
  std::int32_t expected{-1}, actual{-1};
  std::size_t widest{0};
  std::vector<std::int32_t> tokens;
  std::string error;
};

std::string Json(const Outcome& o) {
  std::string s =
      "{\"limit\":" + std::to_string(o.limit) + ",\"prompt\":\"" + o.prompt +
      "\",\"config\":\"" + o.config + "\",\"status\":\"" + o.status +
      "\",\"matched\":" + std::to_string(o.matched) +
      ",\"of\":" + std::to_string(kTokens) +
      ",\"first_divergence\":" + std::to_string(o.first_divergence) +
      ",\"expected\":" + std::to_string(o.expected) +
      ",\"actual\":" + std::to_string(o.actual) +
      ",\"widest_cycle\":" + std::to_string(o.widest) +
      ",\"tokens\":" + List(o.tokens);
  if (!o.error.empty()) {
    std::string escaped;
    for (const char c : o.error)
      if (c != '"' && c != '\\' && c >= 0x20)
        escaped += c;
    s += ",\"error\":\"" + escaped + "\"";
  }
  return s + "}";
}

/// Probe records written since `offset`: their layout, and the longest MTP
/// chain without copies (the chains that verify wide past eight rows).
std::uint32_t CheckProbeRecords(const char* path, std::streamoff* offset) {
  std::ifstream in(path);
  Require(in.good(), "cannot read the gate probe records");
  in.seekg(*offset);
  std::string line;
  std::uint32_t longest = 0;
  const auto count = [](std::string_view s, std::string_view key) {
    const auto begin = s.find("\"" + std::string(key) + "\":[");
    if (begin == std::string_view::npos)
      return std::size_t{0};
    const auto end = s.find(']', begin);
    const auto list = s.substr(begin, end - begin);
    if (list.back() == '[')
      return std::size_t{0};
    return static_cast<std::size_t>(std::ranges::count(list, ',')) + 1;
  };
  const auto number = [](std::string_view s, std::string_view key) {
    const auto at = s.find("\"" + std::string(key) + "\":");
    Require(at != std::string_view::npos,
            "probe record lacks " + std::string(key));
    return static_cast<std::uint32_t>(
        std::stoul(std::string(s.substr(at + key.size() + 3))));
  };
  while (std::getline(in, line)) {
    if (line.rfind("{\"pos\":", 0) != 0)
      continue;  // anchor lines
    const auto chain = number(line, "chain");
    const auto lookup = number(line, "lookup");
    const auto drafts = count(line, "p");
    const auto keep = number(line, "keep");
    const bool lists =
        count(line, "agree") == drafts && count(line, "order") == drafts;
    Require(lists && chain == 1 + drafts + lookup && keep >= 1 && keep <= chain,
            "inconsistent record: " + line);
    Require(drafts <= kLastLimit, "a chain drafted past 14: " + line);
    if (lookup == 0)
      longest = std::max(longest, chain);
  }
  in.clear();
  in.seekg(0, std::ios::end);
  *offset = in.tellg();
  return longest;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string_view flag = argv[i];
    if (flag == "--model")
      args.model = argv[i + 1];
    else if (flag == "--mtp-model")
      args.mtp = argv[i + 1];
    else if (flag == "--vocab-trim-set")
      args.trim = argv[i + 1];
    else if (flag == "--record")
      args.record = argv[i + 1];
    else
      args.model.clear();
  }
  if (args.model.empty() || args.mtp.empty() || argc % 2 == 0) {
    std::cerr << "Usage: depth_session_test --model FIRST.gguf --mtp-model "
                 "MTP.gguf [--vocab-trim-set IDS] [--record OUT.jsonl]\n";
    return 77;
  }
  std::ofstream record;
  if (!args.record.empty()) {
    record.open(args.record, std::ios::trunc);
    if (!record) {
      std::cerr << "FAIL: cannot write " << args.record << '\n';
      return 1;
    }
  }
  const char* probe = std::getenv("GUFO_MTP_GATE_PROBE");
  std::streamoff probe_offset = 0;
  bool failed = false;
  // Ordinary decoding does not depend on the draft limit: computed once.
  std::vector<Prompt> prompts;
  std::map<std::pair<std::size_t, std::size_t>, std::vector<std::int32_t>>
      expected;
  std::vector<Outcome> outcomes;
  for (std::uint32_t limit = kFirstLimit; limit <= kLastLimit; ++limit) {
    std::string error;
    std::shared_ptr<qfn::Model> model;
    try {
      model = qfn::Model::Load(args.model, Options(args, limit), &error);
    } catch (const std::exception& e) {
      error = e.what();
    }
    if (!model) {
      // The logging-only build caps MTP at seven drafts.
      const bool refused = limit > kFirstLimit &&
                           error.find("at most seven") != std::string::npos;
      Outcome o{.limit = limit,
                .prompt = "*",
                .config = "*",
                .status = refused ? "refused" : "error",
                .error = error};
      outcomes.push_back(o);
      failed = failed || !refused;
      std::cerr << "limit " << limit << ": " << (refused ? "refused" : "FAIL")
                << ": " << error << '\n';
      continue;
    }
    try {
      if (prompts.empty()) {
        prompts = MakePrompts(*model);
        for (std::size_t p = 0; p < prompts.size(); ++p)
          for (std::size_t c = 0; c < kGreedy.size(); ++c)
            expected[{p, c}] = Ordinary(*model, prompts[p].tokens, kGreedy[c]);
      }
    } catch (const std::exception& e) {
      std::cerr << "FAIL: ordinary decoding: " << e.what() << '\n';
      return 1;
    }
    for (std::size_t p = 0; p < prompts.size(); ++p) {
      for (std::size_t c = 0; c <= kGreedy.size(); ++c) {
        const bool sampled = c == kGreedy.size();
        Outcome o{.limit = limit,
                  .prompt = prompts[p].name,
                  .config = sampled ? std::string("sampled")
                                    : "greedy" + std::to_string(c),
                  .status = "ok"};
        try {
          if (sampled) {
            // Seeded sampled decoding replays exactly; it has no ordinary
            // reference (the draws differ), so `matched` counts the tokens
            // two runs agree on.
            const auto a = Speculative(*model, prompts[p].tokens, kSampled);
            const auto b = Speculative(*model, prompts[p].tokens, kSampled);
            o.tokens = a.tokens;
            while (o.matched < kTokens &&
                   a.tokens[o.matched] == b.tokens[o.matched])
              ++o.matched;
            o.widest = *std::ranges::max_element(a.widths);
            if (!(a == b)) {
              o.status = "diverged";
              if (o.matched < kTokens) {
                o.first_divergence = static_cast<long>(o.matched);
                o.expected = a.tokens[o.matched];
                o.actual = b.tokens[o.matched];
              }
            }
          } else {
            const auto& want = expected[{p, c}];
            const auto actual =
                Speculative(*model, prompts[p].tokens, kGreedy[c]);
            o.tokens = actual.tokens;
            while (o.matched < kTokens &&
                   actual.tokens[o.matched] == want[o.matched])
              ++o.matched;
            o.widest = *std::ranges::max_element(actual.widths);
            if (o.matched < kTokens) {
              o.status = "diverged";
              o.first_divergence = static_cast<long>(o.matched);
              o.expected = want[o.matched];
              o.actual = actual.tokens[o.matched];
            }
          }
          o.tokens.resize(std::min(o.tokens.size(), kTokens));
          // A cycle keeps at most the rows it verified: a wide copy's 64.
          if (o.widest > 64) {
            o.status = "error";
            o.error = "a cycle kept more tokens than any verification has rows";
          }
        } catch (const std::exception& e) {
          o.status = "error";
          o.error = e.what();
        }
        // Limit 7 must be exact; past it every outcome is a diagnostic,
        // except an engine error, which is a defect of the patch.
        if (o.status == "error" || (limit == kFirstLimit && o.status != "ok"))
          failed = true;
        std::cout << "limit " << limit << " " << o.prompt << " " << o.config
                  << ": " << o.status << ", matched " << o.matched << "/"
                  << kTokens;
        if (o.first_divergence >= 0)
          std::cout << ", first divergence at token " << o.first_divergence
                    << " (" << o.expected << " vs " << o.actual << ")";
        std::cout << ", widest cycle " << o.widest;
        if (!o.error.empty())
          std::cout << ", error: " << o.error;
        std::cout << '\n';
        outcomes.push_back(std::move(o));
      }
    }
    if (probe != nullptr) {
      try {
        const auto longest = CheckProbeRecords(probe, &probe_offset);
        std::cout << "limit " << limit << ": probe records keep their layout; "
                  << "longest MTP chain " << longest << " rows\n";
        // Past seven drafts the routing must take: an MTP chain past eight
        // rows verified wide (the gate is off under the probe).
        if (limit > kFirstLimit && longest <= 8) {
          std::cerr << "FAIL: limit " << limit
                    << ": no MTP chain past eight rows was verified\n";
          failed = true;
        }
      } catch (const std::exception& e) {
        std::cerr << "FAIL: probe records: " << e.what() << '\n';
        failed = true;
      }
    }
  }
  // Per limit: tokens matched over the greedy configs, summed.
  std::cout << "summary (greedy tokens matching ordinary decoding):";
  for (std::uint32_t limit = kFirstLimit; limit <= kLastLimit; ++limit) {
    std::size_t matched = 0, of = 0;
    for (const auto& o : outcomes)
      if (o.limit == limit && o.config.rfind("greedy", 0) == 0) {
        matched += o.matched;
        of += kTokens;
      }
    std::cout << ' ' << limit << ':' << matched << '/' << of;
  }
  std::cout << '\n';
  if (record) {
    for (const auto& o : outcomes)
      record << Json(o) << '\n';
  }
  if (failed)
    std::cerr << "FAIL: limit 7 not exact, or an engine error (see above)\n";
  return failed ? 1 : 0;
}
