// The engine's own float32 CPU reference of the MTP draft block, run on dumped rows (gufo-mtp-continuous-learning, todo 1.3/1.4).
//
// G2's parity gate compares the training port's argmax with "gufo's reference MTP logits' argmax" on the validation rows. The dump tool
// (dump_hidden.cpp) writes rows of (wide residual of position i-1 as bf16, token x[i], trunk top-k); this tool feeds those same pairs
// to ReferenceModel::MtpStep, which is the CPU scalar predictor the GPU head is tested against, and writes what it predicts. No GPU.
//
// A long-lived stdin/stdout process like dump_hidden. One request per line:
//
//   rows=/path/chunk.bin off=0 n=300 out=/path/ref.bin
//
// reads n rows starting at row `off` of the rows file (20,644 bytes each at the real shape), runs them as one sequence from row 0 of the
// request (the draft block's attention cache starts empty, as the port's does), and writes n records of (int32 argmax, float32 logit at the
// argmax, float32 second-best logit) to out (.part, renamed). Reply `ok <out> <n>` or `err <msg>`.
//
//   ref_mtp --model FIRST_SHARD.gguf --mtp-model MTP.gguf [--top-k 20]
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/reference.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace q = gufo::models::qwen38_flash_next;

namespace {

struct Fail : std::runtime_error {
  using std::runtime_error::runtime_error;
};

float FromBf16(std::uint16_t b) {
  const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

struct Rig {
  std::unique_ptr<gufo::core::GgufReader> reader, mtp_reader;
  std::optional<q::ModelWeights> weights;
  std::optional<q::MtpWeights> mtp;
  std::size_t hc = 0, vocab = 0, top_k = 20;
};

Rig Open(const std::string& model, const std::string& mtp_path, std::size_t top_k) {
  Rig r;
  r.top_k = top_k;
  std::string error;
  r.reader = gufo::core::GgufReader::OpenFile(model, &error);
  if (!r.reader) throw Fail("open failed: " + error);
  r.weights = q::ModelWeights::Bind(*r.reader, &error);
  if (!r.weights) throw Fail("bind failed: " + error);
  r.mtp_reader = gufo::core::GgufReader::OpenFile(mtp_path, &error);
  if (r.mtp_reader) r.mtp = q::MtpWeights::Bind(*r.mtp_reader, r.weights->config, &error);
  if (!r.mtp) throw Fail("MTP bind failed: " + error);
  r.hc = r.weights->config.HcDim();
  r.vocab = r.weights->config.vocab_size;
  return r;
}

std::size_t Serve(Rig& r, const std::string& line, std::string& out_path) {
  std::stringstream ss(line);
  std::string tok, rows_path;
  std::size_t off = 0, n = 0;
  while (ss >> tok) {
    if (tok.rfind("rows=", 0) == 0) rows_path = tok.substr(5);
    else if (tok.rfind("off=", 0) == 0) off = std::stoul(tok.substr(4));
    else if (tok.rfind("n=", 0) == 0) n = std::stoul(tok.substr(2));
    else if (tok.rfind("out=", 0) == 0) out_path = tok.substr(4);
    else throw Fail("unknown field '" + tok + "'");
  }
  if (rows_path.empty() || out_path.empty() || n == 0) throw Fail("a request needs rows=, n= and out=");
  const std::size_t row_bytes = r.hc * 2 + 4 + r.top_k * 4 + r.top_k * 4;
  std::ifstream in(rows_path, std::ios::binary);
  if (!in) throw Fail("cannot open " + rows_path);
  in.seekg(static_cast<std::streamoff>(off * row_bytes));
  std::vector<char> raw(n * row_bytes);
  in.read(raw.data(), static_cast<std::streamsize>(raw.size()));
  if (static_cast<std::size_t>(in.gcount()) != raw.size()) throw Fail("short read");

  // The attention cache of the reference grows with the request, from row 0 of the request.
  q::ReferenceModel ref(*r.weights, nullptr, static_cast<std::uint32_t>(n + 1));
  std::vector<float> hidden(r.hc), logits(r.vocab);
  const auto part = out_path + ".part";
  {
    std::ofstream out(part, std::ios::binary);
    if (!out) throw Fail("cannot write " + part);
    for (std::size_t i = 0; i < n; ++i) {
      const char* row = raw.data() + i * row_bytes;
      for (std::size_t c = 0; c < r.hc; ++c) {
        std::uint16_t b;
        std::memcpy(&b, row + c * 2, sizeof(b));
        hidden[c] = FromBf16(b);
      }
      std::int32_t token;
      std::memcpy(&token, row + r.hc * 2, sizeof(token));
      std::string error;
      if (!ref.MtpStep(*r.mtp, token, hidden, logits, nullptr, nullptr, &error)) throw Fail("MtpStep failed: " + error);
      std::size_t best = 0, second = 1;
      if (logits[second] > logits[best]) std::swap(best, second);
      for (std::size_t v = 2; v < logits.size(); ++v) {
        if (logits[v] > logits[best]) {
          second = best;
          best = v;
        } else if (logits[v] > logits[second]) {
          second = v;
        }
      }
      const std::int32_t arg = static_cast<std::int32_t>(best);
      const float top = logits[best], next = logits[second];
      out.write(reinterpret_cast<const char*>(&arg), sizeof(arg));
      out.write(reinterpret_cast<const char*>(&top), sizeof(top));
      out.write(reinterpret_cast<const char*>(&next), sizeof(next));
    }
    out.flush();
    if (!out) throw Fail("write failed on " + part);
  }
  std::filesystem::rename(part, out_path);
  return n;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, mtp_path;
  std::size_t top_k = 20;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
    if (a == "--model") model = next();
    else if (a == "--mtp-model") mtp_path = next();
    else if (a == "--top-k") top_k = std::stoul(next());
    else if (a == "--help" || a == "-h") {
      std::puts("ref_mtp --model FIRST_SHARD.gguf --mtp-model MTP.gguf [--top-k 20]   (requests on stdin, see the file header)");
      return 0;
    } else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  if (model.empty() || mtp_path.empty() || top_k == 0) {
    std::fprintf(stderr, "--model and --mtp-model are required\n");
    return 2;
  }
  try {
    auto rig = Open(model, mtp_path, top_k);
    std::printf("ready hc=%zu vocab=%zu row_bytes=%zu\n", rig.hc, rig.vocab, rig.hc * 2 + 4 + rig.top_k * 8);
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
    std::fprintf(stderr, "ref_mtp: %s\n", e.what());
    return 1;
  }
  return 0;
}
