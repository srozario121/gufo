#ifndef GUFO_SERVER_SAMPLING_REQUEST_HPP_
#define GUFO_SERVER_SAMPLING_REQUEST_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "src/core/json.hpp"
#include "src/core/sampling.hpp"

namespace gufo::server {

struct SamplingRequestError {
  std::string message;
  std::string code;
};

namespace detail {

inline constexpr std::int64_t kMaxExactJsonInteger = INT64_C(9007199254740991);
inline constexpr std::size_t kMaxSamplingWindow =
    std::numeric_limits<std::uint32_t>::max();

inline std::string SamplingErrorCode(std::string_view field) {
  std::string code = "invalid_";
  code.append(field);
  for (char& character : code) {
    if (character == '-') {
      character = '_';
    }
  }
  return code;
}

inline std::optional<SamplingRequestError> ReadSamplingFloat(
    const json::Value& body, std::string_view field, float* output,
    bool nullable = false) {
  const json::Value* value = body.find(std::string(field));
  if (value == nullptr || (nullable && value->is_null())) {
    return std::nullopt;
  }
  if (!value->is_number() || !std::isfinite(value->as_double()) ||
      value->as_double() <
          -static_cast<double>(std::numeric_limits<float>::max()) ||
      value->as_double() >
          static_cast<double>(std::numeric_limits<float>::max())) {
    return SamplingRequestError{
        .message = "'" + std::string(field) + "' must be a finite number",
        .code = SamplingErrorCode(field),
    };
  }
  *output = static_cast<float>(value->as_double());
  return std::nullopt;
}

template<typename Integer>
inline std::optional<SamplingRequestError> ReadSamplingInteger(
    const json::Value& body, std::string_view field, Integer minimum,
    Integer maximum, Integer* output, bool nullable = false) {
  const json::Value* value = body.find(std::string(field));
  if (value == nullptr || (nullable && value->is_null())) {
    return std::nullopt;
  }
  const double number = value->is_number() ? value->as_double() : 0.0;
  if (!value->is_number() || !std::isfinite(number) ||
      std::floor(number) != number || number < static_cast<double>(minimum) ||
      number > static_cast<double>(maximum)) {
    return SamplingRequestError{
        .message = "'" + std::string(field) + "' must be an integer between " +
                   std::to_string(minimum) + " and " + std::to_string(maximum),
        .code = SamplingErrorCode(field),
    };
  }
  *output = static_cast<Integer>(number);
  return std::nullopt;
}

inline constexpr std::string_view kLlamaSamplerFields[] = {
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "typical_p",
    "tfs_z",
    "dynatemp_range",
    "dynatemp_exponent",
    "top_n_sigma",
    "xtc_probability",
    "xtc_threshold",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
    "dry_sequence_breakers",
};

inline bool IsLlamaSamplerField(std::string_view field) {
  for (const auto known : kLlamaSamplerFields) {
    if (field == known) {
      return true;
    }
  }
  return false;
}

/// llama.cpp sampler fields that gufo does not implement. A field that is
/// absent, null, or set to a value that leaves llama.cpp's sampler chain
/// unchanged is ignored, so clients that always send their sampler settings
/// (Open WebUI 0.12 sends mirostat 0) still work. A value that would activate
/// one of these samplers is rejected rather than sampling another
/// distribution. Neutral values follow llama.cpp: the server's clamps and
/// each sampler's early return.
inline std::optional<SamplingRequestError> CheckNeutralLlamaSamplers(
    const json::Value& body) {
  const auto invalid = [](std::string_view field, std::string_view what) {
    return SamplingRequestError{
        .message = "'" + std::string(field) + "' must be " + std::string(what),
        .code = "invalid_sampling",
    };
  };
  const auto unsupported = [](std::string_view field) {
    return SamplingRequestError{
        .message = "request field '" + std::string(field) +
                   "' is not supported with a value that enables it",
        .code = "unsupported_sampling",
    };
  };
  std::optional<SamplingRequestError> error;
  // Reads a present, non-null number; null counts as absent.
  const auto number = [&](std::string_view field) -> std::optional<double> {
    const json::Value* value = body.find(std::string(field));
    if (value == nullptr || value->is_null()) {
      return std::nullopt;
    }
    if (!value->is_number() || !std::isfinite(value->as_double())) {
      if (!error) {
        error = invalid(field, "a finite number");
      }
      return std::nullopt;
    }
    return value->as_double();
  };
  const auto window = [&](std::string_view field) -> std::optional<double> {
    const auto value = number(field);
    if (value && (std::floor(*value) != *value || *value < 0.0 ||
                  *value > std::numeric_limits<std::int32_t>::max())) {
      if (!error) {
        error = invalid(field, "an integer between 0 and 2147483647");
      }
    }
    return value;
  };
  const auto mirostat = number("mirostat");
  (void)number("mirostat_tau");
  (void)number("mirostat_eta");
  const auto typical_p = number("typical_p");
  (void)number("tfs_z");
  const auto dynatemp_range = number("dynatemp_range");
  (void)number("dynatemp_exponent");
  const auto top_n_sigma = number("top_n_sigma");
  const auto xtc_probability = number("xtc_probability");
  const auto xtc_threshold = number("xtc_threshold");
  const auto dry_multiplier = number("dry_multiplier");
  (void)number("dry_base");
  (void)window("dry_allowed_length");
  const auto dry_penalty_last_n = window("dry_penalty_last_n");
  if (const json::Value* breakers = body.find("dry_sequence_breakers");
      breakers != nullptr && !breakers->is_null() && !error) {
    bool strings = breakers->is_array() && !breakers->empty();
    if (strings) {
      for (const auto& breaker : breakers->items()) {
        strings = strings && breaker.is_string();
      }
    }
    if (!strings) {
      error = invalid("dry_sequence_breakers", "a nonempty array of strings");
    }
  }
  if (error) {
    return error;
  }
  // mirostat_tau/eta only matter with mirostat 1 or 2, dynatemp_exponent
  // only with a positive dynatemp_range, and llama.cpp no longer has TFS.
  if (mirostat && *mirostat > 0.0) {
    return unsupported("mirostat");
  }
  if (typical_p && *typical_p < 1.0) {
    return unsupported("typical_p");
  }
  if (dynatemp_range && *dynatemp_range > 0.0) {
    return unsupported("dynatemp_range");
  }
  if (top_n_sigma && *top_n_sigma > 0.0) {
    return unsupported("top_n_sigma");
  }
  // XTC acts only when the clamped probability is positive and the clamped
  // threshold is at most 0.5 (llama.cpp defaults: 0.0 and 0.1).
  if (std::clamp(xtc_probability.value_or(0.0), 0.0, 1.0) > 0.0 &&
      std::clamp(xtc_threshold.value_or(0.1), 0.0, 1.0) <= 0.5) {
    return unsupported(xtc_probability ? "xtc_probability" : "xtc_threshold");
  }
  // DRY acts only with a nonzero multiplier and a nonzero window
  // (common.h defaults 0.0 and 64; the server limits the window to
  // [0, INT32_MAX]). The sampler skips dry_base < 1, but the server replaces
  // such a base with its default first, so dry_base never disables DRY.
  if (dry_multiplier.value_or(0.0) != 0.0 &&
      dry_penalty_last_n.value_or(64.0) != 0.0) {
    return unsupported("dry_multiplier");
  }
  return std::nullopt;
}

/// Most `logit_bias` entries one request may carry (OpenAI's own limit).
inline constexpr std::size_t kMaxLogitBiasEntries = 300;
inline constexpr double kMaxLogitBiasMagnitude = 100.0;

/// Reads a request's `logit_bias`: an object of decimal token IDs (no sign,
/// no leading zeros, below the vocabulary) to finite numbers in [-100, 100],
/// at most kMaxLogitBiasEntries of them. The request's entries replace the
/// server default for their tokens. Called only when the server accepts it.
inline std::optional<SamplingRequestError> ReadLogitBias(
    const json::Value& body, const sampling::LogitBias& server,
    sampling::SamplingConfig* output) {
  const json::Value* value = body.find("logit_bias");
  if (value == nullptr || value->is_null() ||
      (value->is_object() && value->empty())) {
    return std::nullopt;
  }
  const auto invalid = [](std::string message) {
    return SamplingRequestError{.message = std::move(message),
                                .code = "invalid_logit_bias"};
  };
  if (!value->is_object()) {
    return invalid("'logit_bias' must be an object of token IDs to numbers");
  }
  sampling::LogitBias bias;
  bias.vocab = server.vocab;
  bias.defaults = server.defaults;
  for (const auto& [key, entry] : value->members()) {
    if (bias.entries.size() >= kMaxLogitBiasEntries) {
      return invalid("'logit_bias' has more than " +
                     std::to_string(kMaxLogitBiasEntries) + " entries");
    }
    const bool canonical =
        !key.empty() && key.size() <= 9 && (key == "0" || key.front() != '0') &&
        std::ranges::all_of(key, [](char c) { return c >= '0' && c <= '9'; });
    const std::uint64_t token = canonical
                                    ? std::stoull(key)
                                    : std::numeric_limits<std::uint64_t>::max();
    if (!canonical || token >= server.vocab) {
      return invalid("'logit_bias' keys must be token IDs below " +
                     std::to_string(server.vocab));
    }
    if (!entry.is_number() || !std::isfinite(entry.as_double()) ||
        std::abs(entry.as_double()) > kMaxLogitBiasMagnitude) {
      return invalid(
          "'logit_bias' values must be finite numbers between -100 "
          "and 100");
    }
    bias.entries.emplace_back(static_cast<sampling::TokenId>(token),
                              static_cast<float>(entry.as_double()));
  }
  std::ranges::sort(bias.entries, {},
                    &std::pair<sampling::TokenId, float>::first);
  if (std::ranges::adjacent_find(bias.entries, {},
                                 &std::pair<sampling::TokenId, float>::first) !=
      bias.entries.end()) {
    return invalid("'logit_bias' repeats a token ID");
  }
  output->logit_bias =
      std::make_shared<const sampling::LogitBias>(std::move(bias));
  return std::nullopt;
}

}  // namespace detail

/// Reads all shared sampling controls, using defaults for omitted fields.
inline std::optional<SamplingRequestError> ParseSamplingConfig(
    const json::Value& body, const sampling::SamplingConfig& defaults,
    sampling::SamplingConfig* output) {
  if (output == nullptr) {
    return SamplingRequestError{
        .message = "sampling output must not be null",
        .code = "invalid_sampling",
    };
  }
  *output = defaults;

  // These controls alter proposal/target probabilities. Reject unsupported
  // spellings instead of accepting a request with a different distribution.
  // A server that accepts `logit_bias` (a non-null default) reads it below.
  const bool accepts_logit_bias = defaults.logit_bias != nullptr;
  for (const auto& [field, value] : body.members()) {
    if (field == "logit_bias" && (accepts_logit_bias || value.is_null() ||
                                  (value.is_object() && value.empty())))
      continue;
    const bool draft_control = field.starts_with("draft_") ||
                               field.ends_with("_draft") || field == "draft" ||
                               field == "speculative";
    if (draft_control || field == "samplers" || field == "logit_bias" ||
        ((field.starts_with("xtc_") || field.starts_with("dry_")) &&
         !detail::IsLlamaSamplerField(field))) {
      return SamplingRequestError{
          .message = "request field '" + field + "' is not supported",
          .code = "unsupported_sampling",
      };
    }
  }
  if (auto error = detail::CheckNeutralLlamaSamplers(body)) {
    return error;
  }

  if (accepts_logit_bias) {
    if (auto error =
            detail::ReadLogitBias(body, *defaults.logit_bias, output)) {
      return error;
    }
  }
  if (auto error = detail::ReadSamplingFloat(body, "temperature",
                                             &output->temperature, true)) {
    return error;
  }
  if (auto error = detail::ReadSamplingInteger(
          body, "top_k", std::int32_t{0},
          std::numeric_limits<std::int32_t>::max(), &output->top_k)) {
    return error;
  }
  if (auto error =
          detail::ReadSamplingFloat(body, "top_p", &output->top_p, true)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(body, "min_p", &output->min_p)) {
    return error;
  }
  if (auto error = detail::ReadSamplingInteger(body, "min_keep", std::size_t{0},
                                               detail::kMaxSamplingWindow,
                                               &output->min_keep)) {
    return error;
  }
  if (auto error = detail::ReadSamplingInteger(body, "seed", std::int64_t{-1},
                                               detail::kMaxExactJsonInteger,
                                               &output->seed, true)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(body, "repeat_penalty",
                                             &output->repeat_penalty)) {
    return error;
  }
  if (auto error = detail::ReadSamplingInteger(
          body, "repeat_last_n", std::size_t{0}, detail::kMaxSamplingWindow,
          &output->repeat_last_n)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(
          body, "frequency_penalty", &output->frequency_penalty, true)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(body, "presence_penalty",
                                             &output->presence_penalty, true)) {
    return error;
  }

  // Check the original JSON number, not the narrowed float: 1.00000001 must
  // not round into the permitted top-p range. Native sampler controls may
  // be broader; these bounds implement the public API contract.
  for (const auto& [field, minimum, maximum] :
       {std::tuple{"temperature", 0.0, 2.0},
        {"top_p", 0.0, 1.0},
        {"min_p", 0.0, 1.0},
        {"frequency_penalty", -2.0, 2.0},
        {"presence_penalty", -2.0, 2.0}}) {
    const auto* value = body.find(field);
    if (value && value->is_number() &&
        (value->as_double() < minimum || value->as_double() > maximum)) {
      return SamplingRequestError{
          .message = "'" + std::string(field) + "' must be between " +
                     std::to_string(minimum) + " and " +
                     std::to_string(maximum),
          .code = detail::SamplingErrorCode(field),
      };
    }
  }

  try {
    output->Validate();
  } catch (const std::invalid_argument& exception) {
    return SamplingRequestError{
        .message = exception.what(),
        .code = "invalid_sampling",
    };
  }
  if (output->temperature > 2.0F) {
    return SamplingRequestError{
        .message = "'temperature' must be between 0 and 2",
        .code = "invalid_temperature",
    };
  }
  for (const auto& [field, value] :
       {std::pair{"frequency_penalty", output->frequency_penalty},
        std::pair{"presence_penalty", output->presence_penalty}}) {
    if (value < -2.0F || value > 2.0F) {
      return SamplingRequestError{
          .message = "'" + std::string(field) + "' must be between -2 and 2",
          .code = detail::SamplingErrorCode(field),
      };
    }
  }
  return std::nullopt;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_SAMPLING_REQUEST_HPP_
