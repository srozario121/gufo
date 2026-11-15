# Text runner step counters

Text (`src/ext/text.hpp`): a runner can report model-specific counters for a
decode step without a dedicated field.
- `TextRunnerCounter`: `name`, `value`, and `rate_of`, which marks a rate
  published as the request's sum of `value` over its sum of the plain counter
  it names.
- `TextDecodeStep::counters`: the step's counters. The host sums them per
  request by name, in first-reported order, and publishes the sums in the
  usage `gufo` object and the generation log line; plain counters also appear
  in `/metrics` as `gufo_<name>_total`, summed over requests.

Compatibility: additive. The field defaults to empty, a runner that reports
no counters changes no output, and existing extensions build unchanged.
