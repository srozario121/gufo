# Gufo: The ***Open Source*** Strix Halo AI engine 

<p align="center">
  <img src="assets/gufo-logo.jpg" alt="Gufo logo" width="180">
</p>

Gufo is a vertical local inference engine for Linux and Windows, specifically built and optimized for AMD Strix Halo hardware:
Ryzen AI MAX+ 395 systems with Radeon 8060S (`gfx1151`), up to 128 GiB of unified memory.

**Contributions are welcome!**

See the [changelog](CHANGELOG.md) and [GitHub Releases](https://github.com/gufo-org/gufo/releases)
for user-facing changes and release history.

> [!TIP]
> Check out **[Gufo RDMA for Dual Strix Halo](https://github.com/neuhaus/gufo)**, a Gufo variant that has not been merged yet!

## What this fork changes

> [!IMPORTANT]
> **This is [srozario121/gufo](https://github.com/srozario121/gufo), a fork of
> [gufo-org/gufo](https://github.com/gufo-org/gufo).** Everything outside this section is upstream's
> text and still describes upstream's engine. The fork adds a short series of patches for one
> workload: serving **Qwen3.8 Flash-Next** (UD-Q4_K_XL trunk plus the Q8_0 MTP head) to a coding
> agent, with English-only, sampled traffic and cached prompt prefixes.
>
> **Every patch is off by default.** If you pass none of the flags below, the fork behaves like the
> upstream release it is based on. The branch layout, build and push rules are in `FORK.md` on the
> `ai-sdlc/tooling` branch. The plan for porting the patches onto this base is in
> [docs/fork/MERGE_PLAN.md](docs/fork/MERGE_PLAN.md).

### Summary

All numbers come from one Strix Halo box (`gfx1151`) running Qwen3.8 Flash-Next with MTP, on
upstream **v0.19.0**. The benchmark replays 36 held-out coding-agent turns, measured as paired
A/B runs with a server restart per visit, an A-against-A null run, and 95% clustered bootstrap
intervals. Quality is judged on 100 held-out turns, decoded greedily. "Lossless" is used only for
output that is byte-identical. The patches have not yet been re-measured on a newer upstream base;
see the merge plan.

| Change | Flags | Measured benefit | Quality | Status |
| --- | --- | --- | --- | --- |
| **Vocabulary-trimmed output head** ("High" level, configuration G) | `--vocab-trim-set`, `--vocab-trim-fallback-threshold 0.25`, `--vocab-trim-discard-aware`, `--vocab-trim-fallback-skip-scored` | **+5.6% decode tok/s** for a single stream (interval +3.6% to +7.5%) and **+5.7%** with 4 concurrent streams (+1.6% to +11.0%). The A/A null passes on both. Decode round time drops from 61.9 ms to 59.7 ms. | Approximate. Greedy text diverges on 1 of 100 held-out turns. Seeded sampling gives identical text on 33 of 36 turns. Every non-English script still passes through. | Meets the +3% adoption bar on both legs. Not yet the production default: the agent task-slice check and an at-scale sampled-quality check are still open |
| Fallback refinements (part of G) | `--vocab-trim-discard-aware`, `--vocab-trim-fallback-skip-scored` | Full-head fallback time per round drops from 1.20 ms to 0.33 ms (−73%). The head's share of a round drops from 3.4% to 1.9%. | Discard-aware is output-neutral. Skip-scored can change a near tie; it added no divergence on the held-out set. | Part of G |
| MTP draft-vocabulary restriction ("Low") | `--mtp-draft-vocab-prefix`, `--mtp-draft-vocab-set` | Not resolved: +3.2% single-stream, −1.0% with 4 streams, and both intervals include 0 | **Lossless**: greedy output byte-identical (72/72) | Kept as an off flag |
| `logit_bias` and soft non-ASCII suppression ("Medium") | `--logit-bias`, `--suppress-nonascii-keep`, `--suppress-nonascii-value` | Not a speed gain: −5.1% with 4 streams, because biasing runs on the host. The value is functional: the fork accepts OpenAI-style `logit_bias` on token IDs, which upstream rejects with `unsupported_sampling`. | 0/100 English divergence; Japanese and Hindi pass-through failed at −8 | Dropped as a speed lever; kept as an API feature |
| Head instrumentation | `--vocab-trim-timing`, `--vocab-trim-counters`, `--vocab-trim-rowlog`, `--vocab-trim-count-once` | Device-side timing and counters for each phase. These numbers drove the G tuning. `--vocab-trim-count-once` fixes a `/metrics` over-count: batched decode was reporting head counters 2.45× too high. | None (measurement only) | Diagnostic |
| Two-tier fallback | `--vocab-trim-tier2`, `--vocab-trim-tier2-threshold` | **Negative**: every flagged row escalated anyway, and fallback time rose 7% | — | Failed; code kept off so later commits apply cleanly. Do not use |
| Smoothed n-gram **trie drafting** (C2b, "trie-first token") | branches `ai-sdlc/c2b-*` | Offline replay predicted +8.4% decode. **Live it was slower** than upstream's prompt lookup: v1 −21%, and v2 a median −13.5% per agent replay. At live trie-first rounds the target gave the trie's token a mean probability of 0.17, against the 0.82 predicted offline. The offline data was rebuilt from trace text rather than the model's own token stream. | Distribution-exact: drafts are verified as point masses | **Dropped, not ported.** Branches kept for reference |
| Q4_1 decode copy of the Q8_0 trunk | `GUFO_DECODE_COPY=balanced`, branch `ai-sdlc/port-q4-1-decode-copy` | 4-visit screen: +6.1% single-stream and +7.6% with 4 streams. Both nulls failed, so neither leg is judged. Costs +1.66 GiB of device memory. | Approximate; greedy divergence d = 0.80 on 100 turns, a quality alarm | Screen only; not adopted. Reimplemented from the design in [mikealanni/gufo](https://github.com/mikealanni/gufo) (MIT) |
| MTP training-data tools | `dump_hidden`, `ref_mtp` (branch `feat/dump-hidden`) | Dumps per-token hidden rows and trunk top-k for training the MTP draft head. `ref_mtp` is a CPU reference for parity checks. | Not in the serving path; separate Nix derivation, `GUFO_BUILD_DUMP_HIDDEN=OFF` by default | Tooling |

### How the vocabulary trim works

Qwen3.8 Flash-Next has 248,320 output tokens. On one English coding-agent corpus, 19,878 distinct
tokens appear in output, and the top 10,000 cover 98.7% of it. The output-head projection reads
every row on every decode and verify step. The trim exploits that gap:

1. **Compact head.** `--vocab-trim-set PATH` takes one token ID per line. Special and stop tokens
   are always added. At load, the Q8_0 rows for the set (48,696 rows in the measured set) are
   gathered into a compact device copy. The projection scores only those rows. A kernel then
   expands the result to full-vocabulary logits, filling every unscored token with −∞. So the
   sampler, penalties and `logit_bias` still see real token IDs.
2. **The MTP draft head reads the same compact rows.** It also takes its argmax and top-64 there
   and maps the winners back to token IDs. This replaces the draft-vocabulary prefix and table
   (the two option sets exclude each other).
3. **Confidence fallback.** If a position's best probability over the scored rows falls below the
   threshold, a device kernel recomputes the full head for that row. Its integer dot products keep
   the bit pattern of an unrestricted head. The fallback runs on the stream, so decode graphs still
   capture it.
4. **Discard-aware flagging.** In a greedy chain, the first draft the target rejects ends the
   chain, so flagged rows after it are cleared on the device before the full-head read. A sampled
   or biased chain defers each flagged row until the verify loop actually reaches it. Either way,
   rows that would be thrown away never pay for a full-head read.
5. **Skip-scored fallback.** A full-head pass reads only the 199,624 rows the compact projection did
   not already score, instead of all 248,320.

Measured on the same turns, only 2 of 2,909 flagged greedy rows (0.07%) changed their argmax after
the full-head pass. That is why the threshold was lowered from 0.5 to 0.25 and the rows past the
first rejection are no longer paid for.

**Known limits with a trim set loaded.** These hold on every base the trim has run on:

- **`logprobs`:** a token outside the set has a logprob of −∞. The exception is a row that took the
  full-head fallback, which carries the full distribution.
- **Structured output:** a grammar or JSON schema cannot produce a token outside the set unless that
  row falls back. Leave the trim off for schema-constrained traffic, or make sure the set covers the
  schema's tokens.

### Fork branches

| Branch | Holds |
| --- | --- |
| `ai-sdlc/v0.19.0` | Upstream v0.19.0 with no patches. This is the build that is currently deployed |
| `ai-sdlc/v0.19.0-vocab` | v0.19.0 plus the vocabulary-trim, `logit_bias` and draft-vocabulary series (10 commits) |
| `ai-sdlc/upstream-main-vocab` | The same series on upstream `1167a944` (after v0.21.0) |
| `ai-sdlc/port-q4-1-decode-copy` | The Q4_1 decode copy on upstream `6c8b2ff4` |
| `ai-sdlc/port-q4-1-decode-copy-v0.28.1` | The same 5 commits on v0.28.1, kept apart from the vocab trim |
| `ai-sdlc/c2b-*`, `ai-sdlc/c2b-trie-first-token` | The trie-drafting experiments (dropped) |
| `feat/dump-hidden` | The MTP training-data tools |
| `feat/dump-hidden-v0.28.1` | The same tools on v0.28.1 |
| `ai-sdlc/v0.28.1` (this branch) | Upstream v0.28.1 plus this README and the vocabulary-trim, `logit_bias` and draft-vocabulary series, ported. CPU checks pass; GPU and speed checks are pending (see the merge plan) |
| `ai-sdlc/tooling` | `FORK.md`, the reproducible image build (`build.sh`), `test.sh`, the leak check and the planning items |

## Models and benchmarks

All model documentation lives under [docs/models](docs/models/README.md):

| Model | Inference modes | Hugging Face weights | Benchmarks | Quality |
| --- | --- | --- | --- | --- |
| [Qwen3.8 27B](docs/models/qwen3.8-27b/README.md) | Q4/Q8, images, AR, DFlash2 | Unsloth [Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf) / [Q8_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q8_K_XL.gguf) · [DFlash2 Q4_K_M](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/blob/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) | Q4: **648.89 tok/s pp**; up to **71.55 tok/s tg** single user and **128.42 aggregated tok/s** on 8 concurrent requests with DFlash2 · [Benchmarks](docs/models/qwen3.8-27b/BENCHMARKS.md) | [Quality](docs/models/qwen3.8-27b/QUALITY.md) |
| [Qwen3.8 Flash-Next](docs/models/qwen3.8-flash-next/README.md) | Q4, images, AR, MTP | Unsloth [Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q4_K_XL) · [MTP Q8_0](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf) | **1,801.01 tok/s pp** (AR); up to **116.93 tok/s tg** single user and **245.81 aggregated tok/s** on 8 concurrent requests with MTP · [Benchmarks](docs/models/qwen3.8-flash-next/BENCHMARKS.md) | [Quality](docs/models/qwen3.8-flash-next/QUALITY.md) |
| [DeepSeek V4 Flash](docs/models/deepseek-v4-flash/README.md) | AR, DSpark | [antirez Flash 0731 IQ2XXS](https://huggingface.co/antirez/deepseek-v4-gguf/blob/1cd7b564460821938add0475a60b942c409295e0/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf) · [DSpark](https://huggingface.co/antirez/deepseek-v4-gguf/blob/e7f04037032990db0346398d249baf9fb9df1ccc/DeepSeek-V4-Flash-DSpark-support-0731.gguf) | **484.62 tok/s pp**; up to **26.62 tok/s tg** single user and **54.74 aggregated tok/s** on 8 concurrent requests with DSpark · [Benchmarks](docs/models/deepseek-v4-flash/BENCHMARKS.md) | [Quality](docs/models/deepseek-v4-flash/QUALITY.md) |
| [Qwen3-ASR 1.7B](docs/models/qwen3-asr/README.md) | Speech recognition | [BF16](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5) | **15.27× realtime** · [Benchmarks](docs/models/qwen3-asr/BENCHMARKS.md) | [Quality](docs/models/qwen3-asr/QUALITY.md) |
| [EmbeddingGemma 2](docs/models/embeddinggemma-2/README.md) | Text and image embeddings | [BF16](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b) | Up to **41.5k tokens/s** over HTTP; **13.0 ms** for a 404-token request · [Benchmarks](docs/models/embeddinggemma-2/BENCHMARKS.md) | [Quality](docs/models/embeddinggemma-2/QUALITY.md) |
| [Qwen3-TTS 1.7B](docs/models/qwen3-tts/README.md) | Speech synthesis and voice cloning | BF16 [CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) / [VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) / [Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Up to **2.54× realtime**; **201 ms** to first audio (CustomVoice) · [Benchmarks](docs/models/qwen3-tts/BENCHMARKS.md) | [Quality](docs/models/qwen3-tts/QUALITY.md) |
| [Qwen-Image-2.1](docs/models/qwen-image-2.1/README.md) | BF16 image generation and editing | [Complete pipeline](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/b3179ad355be050328e483a9dfdd9e60cd62adfa) | In progress · [Benchmarks](docs/models/qwen-image-2.1/BENCHMARKS.md) | [Quality](docs/models/qwen-image-2.1/QUALITY.md) |
| [MiniMax H3](docs/models/minimax-h3/README.md) | BF16 text to video/audio | [FL2VA pipeline](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA) | In progress · [Benchmarks](docs/models/minimax-h3/BENCHMARKS.md) | [Quality](docs/models/minimax-h3/QUALITY.md) |

Peak measured workloads; text pp is autoregressive (AR), while tg uses the
named speculative mode. Peaks include repetitive output; aggregate tg sums
individual request decode rates. Qwen27B's single-user peak uses the short-prompt
C1 workload. Audio excludes loading.
Each model guide lists the required files and complete benchmark settings.

## Philosophy

- Contributions are welcome! We need the help of Strix Halo community to keep improving gufo!
- We would like this to be the one-stop shop for Strix Halo Local AI enthusiasts: batteries included for text, audio, image, and video models.
- Build and optimize specifically for the Strix Halo 128 GiB hardware. Smaller memory configurations should still work and preserve the speed benefits for models that can fit on memory.
- Support only the best available models for their size that can run on this hardware: less code to maintain, more focused optimization and testing work.
- Preserve quality when optimizing. Each model's quality report records independent numerical checks, execution consistency and unresolved gaps. Don't reuse kernels across different models to limit blast radius of a code change.
- Treat concurrent requests, cancellation and conversation caching as first-class workloads.
- Keep production dependencies small and development tools separate.

## Quickstart

```sh
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  --repo-type model \
  --local-dir models/Qwen3.8-27B-GGUF
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF \
  Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --revision 2d9571f8ce46e151f61c6499c99dee6079e1d610 \
  --repo-type model \
  --local-dir models/Qwen3.8-27B-DFlash2-GGUF
podman pull ghcr.io/gufo-org/toolboxes/gufo-runtime:latest
podman run --rm \
  --userns=keep-id:uid=1000,gid=1000 \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add keep-groups \
  --ulimit memlock=-1 \
  -p 8080:8080 \
  -v ./models:/models:ro \
  ghcr.io/gufo-org/toolboxes/gufo-runtime:latest \
  gufo serve --host 0.0.0.0 --port 8080 llm \
  --model /models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --speculative dflash2 \
  --dflash-model /models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

Rootless Podman needs `crun` for `--group-add keep-groups`. Your host user must
have read/write access to `/dev/kfd` and `/dev/dri/renderD*`, usually through the
`render` and `video` groups; log out and back in after changing membership.
Container groups named `video`/`render` do not preserve host supplementary
groups. Check `id`, `ls -l /dev/kfd /dev/dri/renderD*`, and
`podman info --format '{{.Host.OCIRuntime.Name}}'` if ROCm reports no device.
See [Podman's rootless group-access guidance](https://github.com/containers/podman/blob/main/troubleshooting.md#20-passed-in-devices-or-files-cant-be-accessed-in-rootless-container).

On Fedora or another SELinux-enforcing host, GPU enumeration can succeed while
SELinux blocks mapping `/dev/kfd`, causing ROCr to report a misleading
“Memory critical” error. Check the **host** audit log:

```sh
sudo ausearch -m avc -ts recent | grep -E '/dev/kfd|hsa_device_t'
```

If it shows a denied `map` for the container, Podman documents this fix:

```sh
sudo setsebool -P container_use_devices true
```

This persistently allows containers to access device labels for devices passed
into them; it affects all containers on that host. Review that policy scope
before enabling it. See [Podman's device documentation](https://docs.podman.io/en/latest/markdown/podman-run.1.html#device-host-device-container-device-permissions)
and the [SELinux container policy](https://github.com/containers/container-selinux/blob/main/container.te).

Then, from another terminal, ask it something through the OpenAI-compatible API:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-UD-Q8_K_XL",
    "messages": [{"role": "user", "content": "Say something"}]
  }'
```

For Open WebUI, VS Code, OpenAI SDK and Responses-API clients (for example
Codex), set the API base URL to `http://localhost:8080/v1`. Chat Completions and
Responses support text, images, function tools, structured output and streaming.
See the [API contract](docs/SERVER.md).

The text server uses the model's native context by default and generates until
EOS or the context is full. `--context N` sets context capacity per session;
`--max-tokens N` sets a default response limit that clients can override.
Reasoning tokens count toward that response limit.

## Harness notes

- **Qwen with Codex:** Codex sends developer messages mid-conversation, after
  compaction or a settings change. Qwen's template accepts only one leading
  system turn, so Gufo moves them there. The request that introduces one is
  prefilled again; later requests reuse the cache. See the
  [API contract](docs/SERVER.md).
- **Qwen with Claude Code:** not supported. Claude Code adds a system message
  to `messages` on every turn, so Gufo moves a new one to the leading system
  turn each time and every request prefills the whole history again (#522).

## Build from source

Production packages target Linux x86-64 on AMD Strix Halo (`gfx1151`).
Windows x86-64 is a supported native build target for development; see
[BUILD-WINDOWS.md](BUILD-WINDOWS.md). CMake owns
one production configuration for both Nix and ordinary Linux builds. Tests,
profilers, tuning executables and Python reference runners are not installed
with the production package. No `build.sh` wrapper is needed.

### With Nix

```sh
nix build
./result/bin/gufo diagnose
./result/bin/gufo serve llm --model /path/to/model.gguf
```

[flake.lock](flake.lock) pins the dependencies. `nix develop` adds profiling,
model-download and independent evaluation tools; these are not runtime
requirements. Optional benchmark baselines are selected separately with
`nix shell .#ds4-reference`, `.#llama-cpp-reference` or
`.#llama-cpp-mtp-reference`; see [benchmarking](docs/BENCHMARKS.md).
See [testing](docs/TESTING.md) for the small hosted CI suite and
explicit local quality checks.

Prebuilt store paths are published to the `gufo` Cachix cache. The flake
declares it in `nixConfig`: users listed in `trusted-users` are asked to accept
it, or pass `--accept-flake-config`. Nix ignores it for other users; add it
system-wide with `cachix use gufo` or in `/etc/nix/nix.conf`:

```
extra-substituters = https://gufo.cachix.org
extra-trusted-public-keys = gufo.cachix.org-1:rfZe3n4lHINgyoACKzs5EthIZmBNvEyCCtufUpth0PE=
```

The Nix package and development shell use ROCm 10.1.0 (HIP Clang 24)
from the pinned [nixpkgs ROCm 10.1 update](https://github.com/NixOS/nixpkgs/pull/573885).
Python, Python packages and general build dependencies come from the pinned
`nixos-unstable` input. The ROCm package set comes from the separate fork,
retaining its compiler/runtime dependencies to reuse the gfx1151 binaries on
`gufo.cachix.org`. All Nix workflows configure this cache explicitly.
Both include a [queue-memory patch](.devops/nix/patches/rocr-fixed-cwsr-rocm10.patch)
for Strix Halo. It keeps queue save/restore buffers, launch/copy arenas and
code-loading staging resident during host memory compaction, while retaining
managed-memory support and the ordinary fine-grained host-allocation policy.

### Without Nix

Install a C++20 compiler, CMake 3.21+, Ninja, pkg-config and the
following development libraries. The pinned Nix toolchain uses GCC
16.2 and ROCm 10.1.0 (HIP Clang 24). Attention and audio convolution kernels are compiled
directly from HIP. Python, Triton/AOTriton, Composable Kernel and MIOpen are
not production build or runtime requirements.
For the same protection against queue evictions as the Nix build, apply the
ROCr patch above when building your runtime. Flash-Next's host buffers also use
Linux long-term pins, with ordinary HIP registration retained when system
limits disallow those pins. Gufo leaves `HSA_USERPTR_FOR_PAGED_MEM` unchanged.

| Dependency | Used for |
| --- | --- |
| ROCm HIP compiler/runtime, hipBLAS, hipBLASLt, rocBLAS | GPU execution and matrix multiplication |
| hipCUB, rocPRIM, rocWMMA headers | Compiled GPU kernels |
| ICU, libcurl, OpenSSL, libpng, libjpeg, libwebp | Tokenization, HTTPS, hashing and images |
| FFmpeg development libraries | Video embedding decoding and RGB conversion |
| libsndfile, libsoxr | Audio decoding and reference-compatible resampling |
| FFmpeg and ffprobe | Video/audio output; invoked as separate executables |

Install ROCm using [AMD's Linux instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).
Use the development packages for the libraries above. ROCm normally installs
under `/opt/rocm`.

For example, on Debian/Ubuntu the ordinary system libraries are:

```sh
sudo apt install build-essential cmake ninja-build pkg-config \
  libicu-dev libcurl4-openssl-dev libssl-dev libpng-dev libjpeg-dev libwebp-dev libsndfile1-dev libsoxr-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev ffmpeg

# ROCm libraries from the table, named as AMD's repository ships them.
sudo apt install hipblas-dev hipblaslt-dev rocblas-dev \
  hipcub-dev rocprim-dev rocwmma-dev

cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
./build/release/gufo serve llm --model /path/to/model.gguf
```

Configuring fails at `find_package(hipblas)` when those ROCm packages are
missing. Other distributions name them `-devel` instead of `-dev`.
For nonstandard installations, pass ordinary CMake paths, for example
`cmake --preset release -DCMAKE_PREFIX_PATH="/opt/rocm"`.
If compiler discovery picks a system Clang, also pass
`-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++`.
Use `cmake --install build/release` to install Gufo,
its runtime data and license notices. The GPU driver must allow your user to
access `/dev/kfd` and `/dev/dri`; model weights are acquired separately.

The same source, compiler flags and install rules serve both builds. Nix pins
the complete toolchain for reproducible comparisons; changing the compiler or
math libraries requires the affected model's quality checks.

## License

Gufo's original code is [MIT licensed](LICENSE). Adapted code and dependencies
retain their own notices in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and `licenses/`, installed under `share/licenses/gufo`. Model weights are not
bundled and retain their publishers' terms.

## Reference Projects

The initial design is informed by the following open source projects:

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for compact model serving, GGUF, and CPU/GPU correctness paths.
- [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp), [Nathanw1014/strix-halo-llamacpp](https://github.com/Nathanw1014/strix-halo-llamacpp), and [gaetan-puleo/llama-cpp-strix-halo](https://github.com/gaetan-puleo/llama-cpp-strix-halo) for Strix Halo optimization inspiration.
- [vLLM](https://github.com/vllm-project/vllm) for continuous batching and paged request scheduling.
- [hipEngine](https://github.com/shisa-ai/hipEngine) for torch-free HIP execution, and native speculative-cycle work.
- [ds4](https://github.com/antirez/ds4) for DeepSeek V4 Flash, MoE scheduling, and DSpark.
- [audio.cpp](https://github.com/0xShug0/audio.cpp) for audio models for tts and asr tasks.
