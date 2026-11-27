// Depth-14 gate-probe build: an MTP chain of up to fourteen drafts verifies
// up to fifteen target rows in one pass. Greedy decoding must still emit the
// tokens of ordinary decoding, at the default limit of seven and at
// fourteen, with and without penalties, and the long chains must actually
// occur. Rows past eight take the tiled (non-vector) projections on this
// base, so a mismatch here is the open arithmetic question of item 94.
//
// Usage: depth_session_test --model FIRST.gguf --mtp-model MTP.gguf
// With GUFO_MTP_GATE_PROBE set to a fresh file the confidence gate is off,
// so chains run to the limit, and the probe records are checked too.

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

struct Prompts {
  std::vector<std::int32_t> repeated;  ///< prompt lookup copies long chains
  std::vector<std::int32_t> prose;     ///< neural drafts only, mostly
};

Prompts MakePrompts(const qfn::Model& model) {
  Prompts prompts;
  const auto pattern = model.Tokenize(
      "def area(width, height):\n    return width * height\n\n"
      "The quick brown fox jumps over the lazy dog. ");
  Require(!pattern.empty(), "empty prompt pattern");
  for (std::size_t i = 0; i < 1024; ++i)
    prompts.repeated.push_back(pattern[i % pattern.size()]);
  prompts.prose = model.Tokenize(
      "Write a short, careful explanation of how a binary search tree keeps "
      "its keys ordered, then give a Python class with insert, search and "
      "in-order traversal methods, with comments.");
  Require(!prompts.prose.empty(), "empty prose prompt");
  return prompts;
}

// Greedy (with and without penalties) equals ordinary decoding; returns the
// histogram of kept tokens per cycle across both prompts.
std::map<std::size_t, std::size_t> CheckGreedy(qfn::Model& model,
                                               std::uint32_t limit) {
  const auto prompts = MakePrompts(model);
  const std::array<sampling::SamplingConfig, 2> configs{{
      {.seed = 1},
      // PenalizedArgmax verifies in launches of seven rows.
      {.seed = 1,
       .repeat_penalty = 1.1F,
       .repeat_last_n = 64,
       .frequency_penalty = 0.2F,
       .presence_penalty = 0.1F},
  }};
  std::map<std::size_t, std::size_t> kept;
  for (const auto* prompt : {&prompts.repeated, &prompts.prose}) {
    for (std::size_t c = 0; c < configs.size(); ++c) {
      const auto expected = Ordinary(model, *prompt, configs[c]);
      const auto actual = Speculative(model, *prompt, configs[c]);
      std::size_t first = 0;
      while (first < kTokens && actual.tokens[first] == expected[first])
        ++first;
      const auto where = "limit " + std::to_string(limit) + ", config " +
                         std::to_string(c) + ", token " + std::to_string(first);
      Require(first == kTokens, "MTP greedy left ordinary decoding: " + where);
      for (const auto width : actual.widths)
        ++kept[width];
      Require(*std::ranges::max_element(actual.widths) <= limit + 1,
              "a cycle kept more tokens than its verification rows");
    }
  }
  return kept;
}

// Sampled verification (compact candidates, launches of seven rows) replays
// exactly from the same seed at either limit.
void CheckSampledReplay(qfn::Model& model, std::uint32_t limit) {
  const auto prompts = MakePrompts(model);
  const sampling::SamplingConfig config{
      .temperature = 0.7F, .top_k = 40, .top_p = 0.95F, .seed = 73};
  const auto a = Speculative(model, prompts.repeated, config);
  const auto b = Speculative(model, prompts.repeated, config);
  const auto where = "limit " + std::to_string(limit);
  Require(a == b, "seeded sampled MTP did not replay exactly: " + where);
  Require(*std::ranges::max_element(a.widths) <= limit + 1,
          "a sampled cycle kept more tokens than its verification rows");
}

// Probe records (GUFO_MTP_GATE_PROBE): every chain is within fifteen rows and
// keeps the probe-v2 layout; returns the longest chain.
std::uint32_t CheckProbeRecords(const char* path) {
  std::ifstream in(path);
  Require(in.good(), "cannot read the gate probe records");
  std::string line;
  std::uint32_t longest = 0;
  std::size_t chains = 0;
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
    const auto missing = "probe record lacks " + std::string(key);
    Require(at != std::string_view::npos, missing);
    return static_cast<std::uint32_t>(
        std::stoul(std::string(s.substr(at + key.size() + 3))));
  };
  while (std::getline(in, line)) {
    if (line.rfind("{\"pos\":", 0) != 0)
      continue;  // anchor lines
    ++chains;
    const auto chain = number(line, "chain");
    const auto lookup = number(line, "lookup");
    const auto drafts = count(line, "p");
    const auto keep = number(line, "keep");
    const bool lists =
        count(line, "agree") == drafts && count(line, "order") == drafts;
    const bool shape = chain <= qfn::kMaxMtpDraftTokens + 1 &&
                       chain == 1 + drafts + lookup && keep >= 1;
    Require(lists && shape && keep <= chain, "inconsistent record: " + line);
    longest = std::max(longest, chain);
  }
  Require(chains != 0, "the gate probe recorded no chains");
  return longest;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5 || std::string_view(argv[1]) != "--model" ||
      std::string_view(argv[3]) != "--mtp-model") {
    std::cerr << "Usage: depth_session_test --model FIRST.gguf --mtp-model "
                 "MTP.gguf\n";
    return 77;
  }
  try {
    constexpr std::array<std::uint32_t, 2> kLimits{qfn::kDefaultMtpDraftTokens,
                                                   qfn::kMaxMtpDraftTokens};
    for (const auto limit : kLimits) {
      std::string error;
      auto model = qfn::Model::Load(argv[2],
                                    {.max_context = 4096,
                                     .mtp_model_path = argv[4],
                                     .max_draft_tokens = limit,
                                     .decode_concurrency = 1},
                                    &error);
      Require(model != nullptr, error);
      const auto kept = CheckGreedy(*model, limit);
      CheckSampledReplay(*model, limit);
      std::cout << "limit " << limit << ": greedy equals ordinary decoding; "
                << "kept tokens per cycle:";
      for (const auto& [width, cycles] : kept)
        std::cout << ' ' << width << 'x' << cycles;
      std::cout << '\n';
      if (limit > qfn::kDefaultMtpDraftTokens) {
        std::size_t wide = 0;
        for (const auto& [width, cycles] : kept)
          wide += width > 8 ? cycles : 0;
        Require(wide != 0, "no cycle kept more than eight tokens");
      }
    }
    if (const char* path = std::getenv("GUFO_MTP_GATE_PROBE")) {
      const auto longest = CheckProbeRecords(path);
      Require(longest > 8, "the probe recorded no chain past eight rows");
      std::cout << "gate probe records keep their layout; longest chain "
                << longest << " rows\n";
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
