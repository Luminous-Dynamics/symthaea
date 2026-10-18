#!/bin/sh
# Development-only verifier for ENG-REA-002's synthetic host-built fixture.
# It never executes recovered/vendor firmware and never accesses hardware.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SOURCE="$ROOT/fixtures/debounce_fixture.c"
MANIFEST="$ROOT/cases.json"
ORACLE="$ROOT/oracle.json"
CC_BIN=${CC:-cc}

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

for command_name in "$CC_BIN" jq sha256sum; do
    command -v "$command_name" >/dev/null 2>&1 ||
        fail "required command unavailable: $command_name"
done
[ -f "$SOURCE" ] || fail "missing fixture source: $SOURCE"
[ -f "$MANIFEST" ] || fail "missing cases.json"
[ -f "$ORACLE" ] || fail "missing oracle.json"

jq -e '.schema_version == "symthaea.rea-differential-manifest.v1" and (.cases | length) == 6' "$MANIFEST" >/dev/null ||
    fail "invalid manifest schema or unexpected case count"
jq -e '.schema_version == "symthaea.rea-differential-oracle.v1" and (.expected_cases | keys | length) == 6' "$ORACLE" >/dev/null ||
    fail "invalid oracle schema or unexpected case count"

work=$(mktemp -d "${TMPDIR:-/tmp}/rea-differential-v1.XXXXXX")
trap 'rm -rf "$work"' 0 HUP INT TERM

compile_profile() {
    profile_id=$1
    output=$2
    shift 2

    expected_args=$*
    manifest_args=$(jq -er --arg id "$profile_id" '.build_profiles[] | select(.id == $id) | .args | join(" ")' "$MANIFEST") ||
        fail "missing build profile in manifest: $profile_id"
    [ "$expected_args" = "$manifest_args" ] ||
        fail "build arguments disagree with cases.json for $profile_id"

    "$CC_BIN" "$@" "$SOURCE" -o "$output"
}

compile_profile threshold12_o0_debug "$work/threshold12_o0_debug" \
    -std=c11 -O0 -g -DDEBOUNCE_THRESHOLD_TICKS=12
compile_profile threshold12_o2_stripped "$work/threshold12_o2_stripped" \
    -std=c11 -O2 -s -DDEBOUNCE_THRESHOLD_TICKS=12
compile_profile threshold20_o0_debug "$work/threshold20_o0_debug" \
    -std=c11 -O0 -g -DDEBOUNCE_THRESHOLD_TICKS=20
compile_profile threshold20_o2_stripped "$work/threshold20_o2_stripped" \
    -std=c11 -O2 -s -DDEBOUNCE_THRESHOLD_TICKS=20

sha256() {
    sha256sum "$1" | awk '{print $1}'
}

hash_12_debug=$(sha256 "$work/threshold12_o0_debug")
hash_12_stripped=$(sha256 "$work/threshold12_o2_stripped")
hash_20_debug=$(sha256 "$work/threshold20_o0_debug")
hash_20_stripped=$(sha256 "$work/threshold20_o2_stripped")

[ "$hash_12_debug" != "$hash_12_stripped" ] ||
    fail "RDB-001 expected different artifact hashes for debug vs optimized/stripped builds"
[ "$hash_12_debug" != "$hash_20_debug" ] ||
    fail "RDB-002 expected different artifact hashes for threshold 12 vs 20"

check_vector() {
    case_id=$1
    vector_name=$2
    binary=$3
    timer_tick_us=$4
    widths=$5
    expected=$6
    actual=""

    for width in $widths; do
        ticks=$((width / timer_tick_us))
        output=$("$binary" "$ticks") || fail "$case_id: fixture failed for width=$width"
        if [ -z "$actual" ]; then
            actual=$output
        else
            actual="$actual $output"
        fi
    done

    [ "$actual" = "$expected" ] ||
        fail "$case_id/$vector_name: expected [$expected], got [$actual]"
}

# RDB-001 and RDB-002: consume the declared vectors/expectations from the JSON oracle.
for profile_id in threshold12_o0_debug threshold12_o2_stripped; do
    case "$profile_id" in
        threshold12_o0_debug) binary="$work/threshold12_o0_debug" ;;
        threshold12_o2_stripped) binary="$work/threshold12_o2_stripped" ;;
    esac
    widths=$(jq -er '.expected_cases["RDB-001"].pulse_width_microseconds | join(" ")' "$ORACLE")
    expected=$(jq -er --arg id "$profile_id" '.expected_cases["RDB-001"].expected_outputs_by_profile[$id] | join(" ")' "$ORACLE")
    check_vector RDB-001 "$profile_id" "$binary" 1000 "$widths" "$expected"
done

for profile_id in threshold12_o0_debug threshold20_o0_debug; do
    case "$profile_id" in
        threshold12_o0_debug) binary="$work/threshold12_o0_debug" ;;
        threshold20_o0_debug) binary="$work/threshold20_o0_debug" ;;
    esac
    widths=$(jq -er '.expected_cases["RDB-002"].pulse_width_microseconds | join(" ")' "$ORACLE")
    expected=$(jq -er --arg id "$profile_id" '.expected_cases["RDB-002"].expected_outputs_by_profile[$id] | join(" ")' "$ORACLE")
    check_vector RDB-002 "$profile_id" "$binary" 1000 "$widths" "$expected"
done

# RDB-003: the binary stays fixed while only the input trace changes.
for trace_id in trace_a_pulse_width_microseconds trace_b_pulse_width_microseconds; do
    widths=$(jq -er --arg id "$trace_id" '.expected_cases["RDB-003"].pulse_width_microseconds_by_trace[$id] | join(" ")' "$ORACLE")
    expected=$(jq -er --arg id "$trace_id" '.expected_cases["RDB-003"].expected_outputs_by_trace[$id] | join(" ")' "$ORACLE")
    check_vector RDB-003 "$trace_id" "$work/threshold12_o0_debug" 1000 "$widths" "$expected"
done

# RDB-004: same binary, different timer configuration.
for timer_tick_us in 1000 2000; do
    expected=$(jq -er --arg tick "$timer_tick_us" '.expected_cases["RDB-004"].expected_outputs_by_timer_tick_us[$tick]' "$ORACLE")
    ticks=$((12000 / timer_tick_us))
    actual=$("$work/threshold12_o0_debug" "$ticks")
    [ "$actual" = "$expected" ] || fail "RDB-004: tick=$timer_tick_us expected $expected, got $actual"
done

# RDB-005: one observation does not identify the cause; the 22 ms test does.
h1_initial=$("$work/threshold20_o0_debug" "$((16000 / 1000))")
h2_initial=$("$work/threshold12_o0_debug" "$((16000 / 2000))")
expected_h1=$(jq -er '.expected_cases["RDB-005"].initial_observation.H1.output' "$ORACLE")
expected_h2=$(jq -er '.expected_cases["RDB-005"].initial_observation.H2.output' "$ORACLE")
[ "$h1_initial" = "$expected_h1" ] && [ "$h2_initial" = "$expected_h2" ] ||
    fail "RDB-005: initial observations differ from oracle"
[ "$h1_initial" = "$h2_initial" ] ||
    fail "RDB-005: fixture accidentally made the initial hypotheses distinguishable"

h1_followup=$("$work/threshold20_o0_debug" "$((22000 / 1000))")
h2_followup=$("$work/threshold12_o0_debug" "$((22000 / 2000))")
expected_h1=$(jq -er '.expected_cases["RDB-005"].expected_followup.H1.output' "$ORACLE")
expected_h2=$(jq -er '.expected_cases["RDB-005"].expected_followup.H2.output' "$ORACLE")
[ "$h1_followup" = "$expected_h1" ] && [ "$h2_followup" = "$expected_h2" ] ||
    fail "RDB-005: follow-up outputs differ from oracle"
[ "$h1_followup" != "$h2_followup" ] ||
    fail "RDB-005: proposed follow-up does not discriminate"

# RDB-006: the observed before/after pair changes both code profile and trace.
baseline=$("$work/threshold12_o0_debug" "$((13000 / 1000))")
changed=$("$work/threshold20_o0_debug" "$((10000 / 1000))")
[ "$baseline" = "$(jq -er '.expected_cases["RDB-006"].observed_pair.baseline.output' "$ORACLE")" ] ||
    fail "RDB-006: baseline diagonal does not match oracle"
[ "$changed" = "$(jq -er '.expected_cases["RDB-006"].observed_pair.changed.output' "$ORACLE")" ] ||
    fail "RDB-006: changed diagonal does not match oracle"

cross_a=$("$work/threshold12_o0_debug" "$((10000 / 1000))")
cross_b=$("$work/threshold20_o0_debug" "$((13000 / 1000))")
expected_a=$(jq -er '.expected_cases["RDB-006"].proposed_factorial_controls[0].expected_output' "$ORACLE")
expected_b=$(jq -er '.expected_cases["RDB-006"].proposed_factorial_controls[1].expected_output' "$ORACLE")
[ "$cross_a" = "$expected_a" ] && [ "$cross_b" = "$expected_b" ] ||
    fail "RDB-006: cross-over controls do not match oracle"

# Malformed inputs fail with exit 65 and no stdout.
for bad in "-1" "abc" ""; do
    rc=0
    "$work/threshold12_o0_debug" "$bad" >"$work/invalid.stdout" 2>"$work/invalid.stderr" || rc=$?
    [ "$rc" -eq 65 ] || fail "invalid input '$bad' returned exit $rc instead of 65"
    [ ! -s "$work/invalid.stdout" ] || fail "invalid input '$bad' unexpectedly produced stdout"
    grep -F "non-negative decimal integer" "$work/invalid.stderr" >/dev/null ||
        fail "invalid input '$bad' had no expected diagnostic"
done

printf 'PASS: six synthetic ENG-REA-002 case contracts (RDB-001..RDB-006)\n'
printf 'PASS: invalid-input controls (-1, abc, empty)\n'
printf 'Compiler executable: %s\n' "$(command -v "$CC_BIN")"
printf 'Compiler identity: %s\n' "$("$CC_BIN" --version | sed -n '1p')"
printf 'RDB-001 threshold-12 debug SHA-256: %s\n' "$hash_12_debug"
printf 'RDB-001 threshold-12 optimized/stripped SHA-256: %s\n' "$hash_12_stripped"
printf 'RDB-002 threshold-20 debug SHA-256: %s\n' "$hash_20_debug"
printf 'RDB-002 threshold-20 optimized/stripped SHA-256: %s\n' "$hash_20_stripped"
printf 'NOTE: this verifies synthetic host fixture semantics only; REA/provider and physical behavior are not tested.\n'
