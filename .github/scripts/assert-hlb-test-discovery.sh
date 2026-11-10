#!/usr/bin/env bash
set -euo pipefail

source_sha="${SYMTHAEA_SOURCE_SHA:?SYMTHAEA_SOURCE_SHA must bind test discovery to an exact commit}"
if [[ ! "$source_sha" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "SYMTHAEA_SOURCE_SHA must be a 40-character hexadecimal commit SHA" >&2
  exit 2
fi

mkdir -p research-results
receipt="research-results/hlb-test-discovery.txt"
{
  echo "schema=hlb-focused-test-discovery-v1"
  echo "source_sha=$source_sha"
} > "$receipt"

check_filter() {
  local package="$1"
  local filter="$2"
  local minimum="$3"
  local listing
  local matching
  local count

  listing="$(cargo test --locked --release -p "$package" --lib "$filter" -- --list 2>&1)"
  matching="$(printf '%s\n' "$listing" | awk '/: test$/ { print }')"
  count="$(printf '%s\n' "$matching" | awk 'NF { n++ } END { print n + 0 }')"

  if (( count < minimum )); then
    {
      echo "FAIL package=$package filter=$filter expected_at_least=$minimum discovered=$count"
      printf '%s\n' "$listing"
    } >&2
    exit 1
  fi

  {
    echo "PASS package=$package filter=$filter discovered=$count"
    printf '%s\n' "$matching"
  } >> "$receipt"
  echo "PASS $package $filter ($count tests discovered)"
}

# Every filtered cargo test command in the HLB contract must discover at least
# one actual test. A successful zero-test filter is not qualification.
while IFS='|' read -r package filter minimum; do
  [[ -z "$package" || "$package" == \#* ]] && continue
  check_filter "$package" "$filter" "$minimum"
done <<'FILTERS'
symthaea-core|weight_space_program::tests|1
symthaea-core|sequential_weight_program::tests|1
symthaea-core|sequential_weight_verification::tests|1
symthaea-fep|open_world_weight_program_gate::tests|1
symthaea-fep|hierarchical_information_gain::tests|1
symthaea-fep|decision_relevant_information::tests|1
symthaea-fep|causal_diagnostic_semantics::tests|1
symthaea-fep|gaussian_decision_diagnostic::tests|1
symthaea-fep|open_world_robust_decision::tests|1
symthaea-fep|sibson_alpha_information::tests|1
symthaea-fep|sequential_diagnostic_planning::tests|1
symthaea-fep|sequential_diagnostic_benchmark::tests|1
symthaea-fep|static_model_diagnostic::tests|1
symthaea-fep|vssi_mechanical_fixture::tests|1
symthaea-fep|decision_diagnostic_benchmark::tests|1
symthaea-fep|gaussian_model_information_gain::tests|1
symthaea-fep|bayesian_model_reduction::tests|1
symthaea-fep|dirichlet_model_reduction::tests|1
symthaea-fep|three_ball_rule_space::tests|1
symthaea-fep|three_ball_partial_observation::tests|1
symthaea-fep|three_ball_sequential_discovery::tests|1
symthaea-core|hdc::weight_program_adapter::tests|1
symthaea|dynamics::cfc::weight_program_adapter::tests|1
symthaea-core|hdc::structured_mixer::tests|1
FILTERS

echo "Wrote source-bound test discovery receipt: $receipt"
