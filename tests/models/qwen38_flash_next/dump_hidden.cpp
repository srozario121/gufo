// Training-data dump for MTP draft-head fine-tuning (gufo-mtp-continuous-learning, todo 1.3).
//
// A long-lived process: it loads the trunk and the MTP head once, then reads requests from stdin, one per line, and answers
// on stdout. A request names an output file and one or more sequences with their windows:
//
//   out=/path/chunk.bin seq=/path/a.i32 win=10:40,60:90 seq=/path/b.i32 win=5:20
//
// seq is raw little-endian int32 token ids of the whole sequence, never re-tokenised here; win is a:b row windows with
// 1 <= a < b <= length, ascending and not overlapping, at most 2,000 rows each. The reply is `ok <out> <rows>` or
// `err <message>`. Each output row is 20,644 bytes at the real shape (see dump_rows.hpp): the wide residual of position
// i-1 as bf16, the token at i, and the trunk's top-k ids and log-probabilities at i (temperature 1.0, no top-p cut).
//
// The trunk runs in kPrefill mode, whose arithmetic is the same at every chunk width, so the rows do not depend on how the
// work was batched. That is NOT the arithmetic of serving decode: the residual carries a small mode-and-build drift that the
// head-training G2 gate measures. Nothing here reads a GPU result back except logits and the kept trunk rows.
//
//   dump_hidden --model FIRST_SHARD.gguf --mtp-model MTP.gguf [--top-k 20] [--rows 64]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"
#include "tests/models/qwen38_flash_next/dump_rows.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace t = gufo::tools;

namespace {

struct Fail : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class GpuTrunk {
 public:
  GpuTrunk(const q::rocm::Executor& exec, q::rocm::Session& session, std::size_t hc)
      : exec_(exec), session_(session), hc_(hc) {}
  void Prefill(std::span<const std::int32_t> tokens) {
    std::string error;
    if (!exec_.Forward(session_, tokens, 0, nullptr, q::rocm::Executor::ForwardMode::kPrefill, &error))
      throw Fail("prefill failed: " + error);
  }
  void Batch(std::span<const std::int32_t> tokens, float* logits, float* hidden) {
    std::string error;
    const auto n = static_cast<std::uint32_t>(tokens.size());
    if (!exec_.Forward(session_, tokens, n, logits, q::rocm::Executor::ForwardMode::kPrefill, &error))
      throw Fail("forward failed: " + error);
    if (!exec_.CopyTrunkHidden(session_, std::span<float>(hidden, std::size_t{n} * hc_), &error))
      throw Fail("trunk hidden read failed: " + error);
  }

 private:
  const q::rocm::Executor& exec_;
  q::rocm::Session& session_;
  std::size_t hc_;
};

std::vector<std::int32_t> ReadTokens(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw Fail("cannot open " + path);
  const auto bytes = static_cast<std::size_t>(in.tellg());
  if (bytes % 4 != 0 || bytes == 0) throw Fail(path + " is not int32 tokens");
  std::vector<std::int32_t> x(bytes / 4);
  in.seekg(0);
  in.read(reinterpret_cast<char*>(x.data()), static_cast<std::streamsize>(bytes));
  return x;
}

std::vector<t::Window> ParseWindows(const std::string& spec) {
  std::vector<t::Window> out;
  std::stringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    const auto colon = item.find(':');
    if (colon == std::string::npos) throw Fail("window '" + item + "' is not a:b");
    out.push_back({static_cast<std::uint32_t>(std::stoul(item.substr(0, colon))),
                   static_cast<std::uint32_t>(std::stoul(item.substr(colon + 1)))});
  }
  return out;
}

struct Rig {
  std::unique_ptr<gufo::core::GgufReader> reader, mtp_reader;
  std::optional<q::ModelWeights> weights;
  std::optional<q::MtpWeights> mtp;
  std::unique_ptr<q::NgramTable> ngram;
  std::unique_ptr<q::rocm::DeviceModel> device;
  std::unique_ptr<q::rocm::Executor> exec;
  std::size_t hc = 0, vocab = 0, top_k = 20, rows = 64;
};

Rig Open(const std::string& model, const std::string& mtp_path, std::size_t top_k, std::size_t rows) {
  Rig r;
  r.top_k = top_k;
  r.rows = rows;
  std::string error;
  r.reader = gufo::core::GgufReader::OpenFile(model, &error);
  if (!r.reader) throw Fail("open failed: " + error);
  r.weights = q::ModelWeights::Bind(*r.reader, &error);
  if (!r.weights) throw Fail("bind failed: " + error);
  const auto& c = r.weights->config;
  if (c.ple_layer >= 0) {
    const auto& tb = r.weights->ple_table;
    r.ngram = q::NgramTable::Open(r.reader->GetMappedRegions()[tb.shard].file_descriptor, tb.file_offset, tb.rows,
                                  c.ple_head_dim, tb.type, &error);
    if (!r.ngram) throw Fail("n-gram table failed: " + error);
  }
  r.mtp_reader = gufo::core::GgufReader::OpenFile(mtp_path, &error);
  if (r.mtp_reader) r.mtp = q::MtpWeights::Bind(*r.mtp_reader, c, &error);
  if (!r.mtp) throw Fail("MTP bind failed: " + error);
  r.device = q::rocm::DeviceModel::Upload(*r.weights, *r.reader, &*r.mtp, r.mtp_reader.get(), &error);
  if (!r.device) throw Fail("upload failed: " + error);
  q::rocm::Executor::Options options;
  options.max_batch = 512;
  options.max_logit_rows = static_cast<std::uint32_t>(rows);
  options.max_speculative = static_cast<std::uint32_t>(rows);  // the kept trunk rows are the last max_speculative of a batch
  r.exec = q::rocm::Executor::Create(*r.device, r.ngram.get(), options, &error);
  if (!r.exec) throw Fail("executor failed: " + error);
  r.hc = c.HcDim();
  r.vocab = c.vocab_size;
  return r;
}

// One request: every sequence's rows are appended to `out` in order. -> row count.
std::size_t Serve(Rig& r, const std::string& line, std::string& out_path) {
  std::stringstream ss(line);
  std::string tok, seq;
  std::vector<std::pair<std::string, std::vector<t::Window>>> jobs;
  while (ss >> tok) {
    if (tok.rfind("out=", 0) == 0) out_path = tok.substr(4);
    else if (tok.rfind("seq=", 0) == 0) seq = tok.substr(4);
    else if (tok.rfind("win=", 0) == 0) {
      if (seq.empty()) throw Fail("win= before seq=");
      jobs.emplace_back(seq, ParseWindows(tok.substr(4)));
      seq.clear();
    } else throw Fail("unknown field '" + tok + "'");
  }
  if (out_path.empty() || jobs.empty()) throw Fail("a request needs out= and at least one seq= win= pair");
  const auto part = out_path + ".part";
  std::size_t rows = 0;
  {
    std::ofstream out(part, std::ios::binary);
    if (!out) throw Fail("cannot write " + part);
    for (const auto& [path, windows] : jobs) {
      const auto x = ReadTokens(path);
      std::string error;
      auto session = r.exec->CreateSession(gufo::core::SessionMode::kSpeculative, static_cast<std::uint32_t>(x.size() + 1), &error);
      if (!session) throw Fail("session failed: " + error);
      GpuTrunk trunk(*r.exec, *session, r.hc);
      const auto bytes = t::DumpSequence(trunk, x, windows, r.hc, r.vocab, r.top_k, r.rows);
      out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      rows += bytes.size() / t::RowBytes(r.hc, r.top_k);
    }
    out.flush();
    if (!out) throw Fail("write failed on " + part);
  }
  std::filesystem::rename(part, out_path);
  return rows;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, mtp_path;
  std::size_t top_k = 20, rows = 64;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
    if (a == "--model") model = next();
    else if (a == "--mtp-model") mtp_path = next();
    else if (a == "--top-k") top_k = std::stoul(next());
    else if (a == "--rows") rows = std::stoul(next());
    else if (a == "--help" || a == "-h") {
      std::puts("dump_hidden --model FIRST_SHARD.gguf --mtp-model MTP.gguf [--top-k 20] [--rows 64]   (requests on stdin, see the file header)");
      return 0;
    } else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  if (model.empty() || mtp_path.empty() || top_k == 0 || rows == 0 || rows > 64) {
    std::fprintf(stderr, "--model and --mtp-model are required; --top-k >= 1; --rows in 1..64\n");
    return 2;
  }
  try {
    auto rig = Open(model, mtp_path, top_k, rows);
    std::printf("ready hc=%zu vocab=%zu top_k=%zu row_bytes=%zu\n", rig.hc, rig.vocab, rig.top_k, t::RowBytes(rig.hc, rig.top_k));
    std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      std::string out_path;
      try {
        const auto n = Serve(rig, line, out_path);
        std::printf("ok %s %zu\n", out_path.c_str(), n);
      } catch (const std::exception& e) {
        std::error_code ec;
        if (!out_path.empty()) std::filesystem::remove(out_path + ".part", ec);
        std::printf("err %s\n", e.what());
      }
      std::fflush(stdout);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dump_hidden: %s\n", e.what());
    return 1;
  }
  return 0;
}
