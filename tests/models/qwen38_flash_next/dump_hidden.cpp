// Training-data dump for MTP draft-head fine-tuning
// (gufo-mtp-continuous-learning, todo 1.3).
//
// A long-lived process: it loads the trunk and the MTP head once, then reads
// requests from stdin, one per line, and answers on stdout. A request names an
// output file and one or more sequences with their windows:
//
//   out=/path/chunk.bin seq=/path/a.i32 win=10:40,60:90 seq=/path/b.i32
//   win=5:20
//
// seq is raw little-endian int32 token ids of the whole sequence, never
// re-tokenised here; win is a:b row windows with 1 <= a < b <= length,
// ascending and not overlapping, at most 2,000 rows each. The reply is `ok
// <out> <rows>` or `err <message>`. Each output row is 20,644 bytes at the real
// shape (see dump_rows.hpp): the wide residual of position i-1 as bf16, the
// token at i, and the trunk's top-k ids and log-probabilities at i
// (temperature 1.0, no top-p cut).
//
// The trunk runs in kPrefill mode, whose arithmetic is the same at every chunk
// width, so the rows do not depend on how the work was batched. That is NOT the
// arithmetic of serving decode: the residual carries a small mode-and-build
// drift that the head-training G2 gate measures. Nothing here reads a GPU
// result back except logits and the kept trunk rows.
//
//   dump_hidden --model FIRST_SHARD.gguf --mtp-model MTP.gguf [--top-k 20]
//   [--rows 64]
//
// DFlash-2 mode (todo 2.4 T1), selected by --dflash-model instead of
// --mtp-model:
//
//   dump_hidden --model FIRST_SHARD.gguf --dflash-model DRAFTER.gguf [--rows
//   512] request: out=/path/chunk.bin seq=/path/a.i32 win=10:40 loss=12:40
//   [seq=... win=... loss=...]
//
// The drafter GGUF is read on the CPU only, through
// QwenDFlashWeights::LoadFromGguf (the engine's own N-1 rule for
// dflash.target_layers), and its taps must be the drafter's [5, 15, 25, 35,
// 45]. The executor is created with those taps and the drafter's window,
// exactly as engine.cpp does, so the features are the engine's own capture
// (HcTapMeanToRing into the session's ring); no MTP head is loaded. Each row is
// 25,608 bytes (see dflash_rows.hpp): the five taps' stream means at position t
// as bf16, x[t] and the loss mask at t. --rows is the forward call width
// (3..512, at most the ring).
//
// The verify path (todo 2.10): a sequence's fields may continue with
//   verify=s:e,...  the output spans, fed as serving decodes them, and
//   sched=k:keep,... their decode cycles (taken cyclically): a kVerify forward
//                    of k true tokens, rolled back to the first keep, as
//                    Session::DecodeStep and FinishDecode do (k = 1: kDecode).
// Every other position stays on the prompt path (see DumpDflashSequence).
#include <hip/hip_runtime_api.h>

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
#include "src/models/qwen/dflash_weights.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"
#include "tests/models/qwen38_flash_next/dflash_rows.hpp"
#include "tests/models/qwen38_flash_next/dump_rows.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace t = gufo::tools;
namespace d = gufo::tools::dflash2;

namespace {

struct Fail : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class GpuTrunk {
public:
  GpuTrunk(const q::rocm::Executor& exec, q::rocm::Session& session,
           std::size_t hc)
      : exec_(exec), session_(session), hc_(hc) {}
  void Prefill(std::span<const std::int32_t> tokens) {
    std::string error;
    if (!exec_.Forward(session_, tokens, 0, nullptr,
                       q::rocm::Executor::ForwardMode::kPrefill, &error))
      throw Fail("prefill failed: " + error);
    kept_ = std::min<std::size_t>(tokens.size(), exec_.max_speculative());
  }
  void Batch(std::span<const std::int32_t> tokens, float* logits,
             float* hidden) {
    std::string error;
    const auto n = static_cast<std::uint32_t>(tokens.size());
    if (!exec_.Forward(session_, tokens, n, logits,
                       q::rocm::Executor::ForwardMode::kPrefill, &error))
      throw Fail("forward failed: " + error);
    kept_ = std::min<std::size_t>(n, exec_.max_speculative());
    if (!exec_.CopyTrunkHidden(
            session_, std::span<float>(hidden, std::size_t{n} * hc_), &error))
      throw Fail("trunk hidden read failed: " + error);
  }
  // A prefill call consumes its known successors in the draft block itself but
  // not the last one, and the next prefill call needs the draft block at the
  // trunk's position ("MTP must consume the preceding frontier before
  // prefill"): consume the frontier here, as the engine's own prefill does,
  // with the first token of the next call and the last kept trunk row.
  void Advance(std::int32_t next) {
    std::string error;
    if (!exec_.MtpForward(session_, std::span<const std::int32_t>(&next, 1),
                          static_cast<std::int32_t>(kept_ - 1),
                          {.kv_only = true}, &error))
      throw Fail("draft frontier failed: " + error);
  }

private:
  const q::rocm::Executor& exec_;
  q::rocm::Session& session_;
  std::size_t hc_;
  std::size_t kept_ = 0;
};

std::vector<std::pair<std::uint32_t, std::uint32_t>> ParseWindowsRaw(
    const std::string& spec) {
  std::vector<std::pair<std::uint32_t, std::uint32_t>> out;
  std::stringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    const auto colon = item.find(':');
    if (colon == std::string::npos)
      throw Fail("range '" + item + "' is not a:b");
    out.emplace_back(
        static_cast<std::uint32_t>(std::stoul(item.substr(0, colon))),
        static_cast<std::uint32_t>(std::stoul(item.substr(colon + 1))));
  }
  return out;
}

// DFlash-2 mode: the trunk in kPrefill mode, its taps read from the session's
// capture ring after each call.
class DflashTrunk {
public:
  DflashTrunk(const q::rocm::Executor& exec, q::rocm::Session& session)
      : exec_(exec), session_(session) {}
  void Forward(std::span<const std::int32_t> tokens) {
    std::string error;
    if (!exec_.Forward(session_, tokens, 0, nullptr,
                       q::rocm::Executor::ForwardMode::kPrefill, &error))
      throw Fail("prefill failed: " + error);
  }
  // The verify path (item 2.10), as Session::DecodeStep and FinishDecode
  // drive the executor: a kVerify forward of the chain, then a rollback to the
  // committed rows; a chain of one is a kDecode step.
  void Verify(std::span<const std::int32_t> tokens, std::size_t keep) {
    std::string error;
    const auto n = static_cast<std::uint32_t>(tokens.size());
    if (!exec_.Forward(session_, tokens, n, nullptr,
                       q::rocm::Executor::ForwardMode::kVerify, &error))
      throw Fail("verify forward failed: " + error);
    if (!exec_.Rollback(session_, static_cast<std::uint32_t>(keep), &error))
      throw Fail("rollback failed: " + error);
  }
  void Decode(std::int32_t token) {
    std::string error;
    logits_.resize(exec_.config().vocab_size);
    if (!exec_.Forward(session_, std::span<const std::int32_t>(&token, 1), 1,
                       logits_.data(), q::rocm::Executor::ForwardMode::kDecode,
                       &error))
      throw Fail("decode forward failed: " + error);
  }
  void ReadRing(std::size_t first, std::size_t n, float* out) {
    const std::size_t row =
        std::size_t{exec_.dflash_taps()} * exec_.config().hidden_size;
    if (session_.dflash_ring() == nullptr)
      throw Fail("the session has no capture ring");
    // the executor's stream is private: wait for every stream, then copy on the
    // null stream
    if (hipDeviceSynchronize() != hipSuccess)
      throw Fail("device synchronize failed");
    for (const auto& r : d::RingRuns(first, n, exec_.dflash_ring_rows()))
      if (hipMemcpy(out + r.offset * row,
                    session_.dflash_ring() + r.ring_row * row,
                    r.count * row * sizeof(float),
                    hipMemcpyDeviceToHost) != hipSuccess)
        throw Fail("capture ring download failed");
  }

private:
  const q::rocm::Executor& exec_;
  q::rocm::Session& session_;
  std::vector<float> logits_;
};

std::vector<d::Span> ParseSpans(const std::string& spec) {
  std::vector<d::Span> out;
  for (const auto& w : ParseWindowsRaw(spec))
    out.push_back({w.first, w.second});
  return out;
}

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

std::vector<t::Window> ParseWindows(const std::string& spec) {
  std::vector<t::Window> out;
  for (const auto& w : ParseWindowsRaw(spec))
    out.push_back({w.first, w.second});
  return out;
}

struct Rig {
  std::unique_ptr<gufo::core::GgufReader> reader, mtp_reader, dflash_reader;
  std::optional<q::ModelWeights> weights;
  std::optional<q::MtpWeights> mtp;
  std::unique_ptr<q::NgramTable> ngram;
  std::unique_ptr<q::rocm::DeviceModel> device;
  std::unique_ptr<q::rocm::Executor> exec;
  std::size_t hc = 0, vocab = 0, top_k = 20, rows = 64;
  bool dflash = false;  // DFlash-2 mode (--dflash-model)
  std::size_t slots = 0, hidden = 0;
  std::size_t RowBytes() const {
    return dflash ? d::RowBytes(slots, hidden) : t::RowBytes(hc, top_k);
  }
};

Rig Open(const std::string& model, const std::string& mtp_path,
         const std::string& dflash_path, std::size_t top_k, std::size_t rows) {
  Rig r;
  r.dflash = !dflash_path.empty();
  r.top_k = top_k;
  r.rows = rows;
  std::string error;
  r.reader = gufo::core::GgufReader::OpenFile(model, &error);
  if (!r.reader)
    throw Fail("open failed: " + error);
  r.weights = q::ModelWeights::Bind(*r.reader, &error);
  if (!r.weights)
    throw Fail("bind failed: " + error);
  const auto& c = r.weights->config;
  if (c.ple_layer >= 0) {
    const auto& tb = r.weights->ple_table;
    r.ngram = q::NgramTable::Open(
        r.reader->GetMappedRegions()[tb.shard].file_descriptor, tb.file_offset,
        tb.rows, c.ple_head_dim, tb.type, &error);
    if (!r.ngram)
      throw Fail("n-gram table failed: " + error);
  }
  q::rocm::Executor::Options options;
  options.max_batch = 512;
  if (r.dflash) {
    // The drafter file is read on the CPU only, for its taps and window, by the
    // engine's own loader (the N-1 rule).
    r.dflash_reader = gufo::core::GgufReader::OpenFile(dflash_path, &error);
    if (!r.dflash_reader)
      throw Fail("DFlash-2 open failed: " + error);
    const auto dw = gufo::speculative::QwenDFlashWeights::LoadFromGguf(
        *r.dflash_reader, &error);
    if (!dw)
      throw Fail("DFlash-2 bind failed: " + error);
    const auto& dc = dw->dflash_config;
    try {
      d::CheckTaps(dc.target_layer_ids, d::kDrafterTaps, c.num_layers);
    } catch (const std::invalid_argument& e) {
      throw Fail(e.what());
    }
    r.device = q::rocm::DeviceModel::Upload(*r.weights, *r.reader, nullptr,
                                            nullptr, &error);
    if (!r.device)
      throw Fail("upload failed: " + error);
    // As engine.cpp sets them for a loaded drafter (logit rows are the verify
    // width; unused by prefill).
    options.dflash_tap_layers = dc.target_layer_ids;
    options.dflash_window = dc.sliding_window;
    options.max_logit_rows = dc.block_size;
    options.max_speculative = dc.block_size;
  } else {
    r.mtp_reader = gufo::core::GgufReader::OpenFile(mtp_path, &error);
    if (r.mtp_reader)
      r.mtp = q::MtpWeights::Bind(*r.mtp_reader, c, &error);
    if (!r.mtp)
      throw Fail("MTP bind failed: " + error);
    r.device = q::rocm::DeviceModel::Upload(*r.weights, *r.reader, &*r.mtp,
                                            r.mtp_reader.get(), &error);
    if (!r.device)
      throw Fail("upload failed: " + error);
    options.max_logit_rows = static_cast<std::uint32_t>(rows);
    options.max_speculative = static_cast<std::uint32_t>(
        rows);  // the kept trunk rows are the last max_speculative of a batch
  }
  r.exec = q::rocm::Executor::Create(*r.device, r.ngram.get(), options, &error);
  if (!r.exec)
    throw Fail("executor failed: " + error);
  if (r.dflash) {
    r.slots = r.exec->dflash_taps();
    r.hidden = c.hidden_size;
    if (r.exec->dflash_ring_rows() == 0 || rows > r.exec->dflash_ring_rows())
      throw Fail("--rows is wider than the capture ring");
  }
  r.hc = c.HcDim();
  r.vocab = c.vocab_size;
  return r;
}

// One request: every sequence's rows are appended to `out` in order. -> row
// count.
std::size_t Serve(Rig& r, const std::string& line, std::string& out_path) {
  std::stringstream ss(line);
  std::string tok, seq;
  struct Job {
    std::string path;
    std::vector<t::Window> windows;
    std::vector<d::Span> loss;
    bool has_loss = false;
    std::vector<d::Span> verify;  // verify path (item 2.10): output spans
    std::vector<d::Cycle> sched;  // and their decode cycles k:keep
  };
  std::vector<Job> jobs;
  while (ss >> tok) {
    if (tok.rfind("out=", 0) == 0)
      out_path = tok.substr(4);
    else if (tok.rfind("seq=", 0) == 0)
      seq = tok.substr(4);
    else if (tok.rfind("win=", 0) == 0) {
      if (seq.empty())
        throw Fail("win= before seq=");
      jobs.push_back({seq, ParseWindows(tok.substr(4)), {}, false});
      seq.clear();
    } else if (tok.rfind("loss=", 0) == 0) {
      if (!r.dflash)
        throw Fail("loss= is a DFlash-2 mode field");
      if (jobs.empty() || jobs.back().has_loss || !seq.empty())
        throw Fail("loss= must follow its seq= win= pair");
      jobs.back().loss = ParseSpans(tok.substr(5));
      jobs.back().has_loss = true;
    } else if (tok.rfind("verify=", 0) == 0 || tok.rfind("sched=", 0) == 0) {
      if (!r.dflash || jobs.empty() || !jobs.back().has_loss || !seq.empty())
        throw Fail("verify= and sched= follow their seq= win= loss= fields");
      const bool v = tok[0] == 'v';
      const auto ranges = ParseWindowsRaw(tok.substr(v ? 7 : 6));
      for (const auto& [a, b] : ranges) {
        if (v)
          jobs.back().verify.push_back({a, b});
        else
          jobs.back().sched.push_back({a, b});
      }
    } else
      throw Fail("unknown field '" + tok + "'");
  }
  if (out_path.empty() || jobs.empty())
    throw Fail("a request needs out= and at least one seq= win= pair");
  if (r.dflash)
    for (const auto& j : jobs)
      if (!j.has_loss)
        throw Fail("DFlash-2 mode needs loss= after every seq= win= pair");
  const auto part = out_path + ".part";
  std::size_t rows = 0;
  {
    std::ofstream out(part, std::ios::binary);
    if (!out)
      throw Fail("cannot write " + part);
    for (const auto& job : jobs) {
      const auto x = ReadTokens(job.path);
      std::string error;
      auto session = r.exec->CreateSession(
          gufo::core::SessionMode::kSpeculative,
          static_cast<std::uint32_t>(x.size() + 1), &error);
      if (!session)
        throw Fail("session failed: " + error);
      std::string bytes;
      try {
        if (r.dflash) {
          DflashTrunk trunk(*r.exec, *session);
          bytes = d::DumpDflashSequence(
              trunk, x, job.windows, job.loss, r.slots, r.hidden, r.rows,
              r.exec->dflash_ring_rows(), job.verify, job.sched);
        } else {
          GpuTrunk trunk(*r.exec, *session, r.hc);
          bytes = t::DumpSequence(trunk, x, job.windows, r.hc, r.vocab, r.top_k,
                                  r.rows);
        }
      } catch (const std::invalid_argument& e) {
        throw Fail(e.what());
      }
      out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      rows += bytes.size() / r.RowBytes();
    }
    out.flush();
    if (!out)
      throw Fail("write failed on " + part);
  }
  std::filesystem::rename(part, out_path);
  return rows;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, mtp_path, dflash_path;
  std::size_t top_k = 20, rows = 0;
  bool top_k_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (a == "--model")
      model = next();
    else if (a == "--mtp-model")
      mtp_path = next();
    else if (a == "--dflash-model")
      dflash_path = next();
    else if (a == "--top-k") {
      top_k = std::stoul(next());
      top_k_set = true;
    } else if (a == "--rows")
      rows = std::stoul(next());
    else if (a == "--help" || a == "-h") {
      std::puts(
          "dump_hidden --model FIRST_SHARD.gguf --mtp-model MTP.gguf [--top-k "
          "20] [--rows 64]\n"
          "dump_hidden --model FIRST_SHARD.gguf --dflash-model DRAFTER.gguf "
          "[--rows 512]   (requests on stdin, see the file header)");
      return 0;
    } else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  const bool dflash = !dflash_path.empty();
  if (rows == 0)
    rows = dflash ? 512 : 64;
  if (model.empty() || mtp_path.empty() == dflash_path.empty()) {
    std::fprintf(stderr,
                 "--model and exactly one of --mtp-model and --dflash-model "
                 "are required\n");
    return 2;
  }
  if (!dflash && (top_k == 0 || rows > 64)) {
    std::fprintf(stderr, "MTP mode: --top-k >= 1; --rows in 1..64\n");
    return 2;
  }
  if (dflash && (top_k_set || rows < 3 || rows > 512)) {
    std::fprintf(stderr, "DFlash-2 mode: no --top-k; --rows in 3..512\n");
    return 2;
  }
  try {
    auto rig = Open(model, mtp_path, dflash_path, top_k, rows);
    if (rig.dflash) {
      std::string taps;
      for (auto l : d::kDrafterTaps)
        taps += (taps.empty() ? "" : ",") + std::to_string(l);
      std::printf(
          "ready mode=dflash2 taps=%s hidden=%zu features=%zu ring_rows=%u "
          "row_bytes=%zu\n",
          taps.c_str(), rig.hidden, rig.slots * rig.hidden,
          rig.exec->dflash_ring_rows(), rig.RowBytes());
    } else {
      std::printf("ready hc=%zu vocab=%zu top_k=%zu row_bytes=%zu\n", rig.hc,
                  rig.vocab, rig.top_k, rig.RowBytes());
    }
    std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty())
        continue;
      std::string out_path;
      try {
        const auto n = Serve(rig, line, out_path);
        std::printf("ok %s %zu\n", out_path.c_str(), n);
      } catch (const std::exception& e) {
        std::error_code ec;
        if (!out_path.empty())
          std::filesystem::remove(out_path + ".part", ec);
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
