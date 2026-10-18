# ENG-REA-002 v1: differential firmware-to-behavior fixture corpus

**Status:** development known-answer corpus; not a qualified benchmark run.  
**Parent work:** [ENG-REA-002 #8057](https://github.com/Luminous-Dynamics/symthaea/issues/8057)  
**Integration boundary:** [ENG-REA-001 #8022](https://github.com/Luminous-Dynamics/symthaea/issues/8022)

## Purpose

This minimal corpus tests a core paradigm shift: distinguish a byte-level difference from a behavioral change, and a behavioral change from a well-supported causal explanation. It also tests when the correct output is abstention plus a discriminating next experiment.

The only program fixture is `fixtures/debounce_fixture.c`: a deliberately tiny, source-derived, host-buildable program that accepts or rejects a pulse according to a compile-time tick threshold. It does not access peripherals, probe a device, flash firmware, or control hardware. It is not production firmware.

The corpus currently defines six scenarios:

| Case | Controlled change | What the evaluator must get right |
|---|---|---|
| RDB-001 | Same source/threshold, different build profiles | Binary differs while known outputs remain equal |
| RDB-002 | Threshold changes from 12 to 20 ticks | Localize the behavioral divergence to the firmware threshold profile |
| RDB-003 | Firmware is fixed; input trace changes | Do not blame firmware for input-driven output changes |
| RDB-004 | Firmware is fixed; timer tick profile changes | Attribute the change to configuration evidence, not a binary change |
| RDB-005 | Threshold and timer hypotheses explain one observation equally | Abstain until a 22 ms known-width input discriminates them |
| RDB-006 | Firmware and input trace change together | Abstain on attribution from the before/after pair; request cross-over controls |

`cases.json` defines the input profiles and scenarios. `oracle.json` defines the public expected outcomes and invalid-input controls. This is a public known-answer regression corpus; it is **not** a blind or held-out evaluation set. Do not tune to this corpus and then report its score as generalization.

## Deterministic build and smoke commands

Requires an existing host C compiler. The commands produce host executables only:

```sh
set -eu
ROOT=benchmarks/rea-differential/v1
CC="${CC:-cc}"

"$CC" -std=c11 -O0 -g -DDEBOUNCE_THRESHOLD_TICKS=12 \
  "$ROOT/fixtures/debounce_fixture.c" -o /tmp/rea-debounce-o0
"$CC" -std=c11 -O2 -s -DDEBOUNCE_THRESHOLD_TICKS=12 \
  "$ROOT/fixtures/debounce_fixture.c" -o /tmp/rea-debounce-o2
"$CC" -std=c11 -O0 -g -DDEBOUNCE_THRESHOLD_TICKS=20 \
  "$ROOT/fixtures/debounce_fixture.c" -o /tmp/rea-debounce-20

# Threshold 12: REJECT at 11 ticks, ACCEPT at 12 ticks.
test "$(/tmp/rea-debounce-o0 11)" = REJECT
test "$(/tmp/rea-debounce-o0 12)" = ACCEPT

# Threshold 20: REJECT at 19 ticks, ACCEPT at 20 ticks.
test "$(/tmp/rea-debounce-20 19)" = REJECT
test "$(/tmp/rea-debounce-20 20)" = ACCEPT

# Malformed and signed values must not be treated as pulse widths.
for bad in -1 abc; do
  if /tmp/rea-debounce-o0 "$bad" >/tmp/rea-bad.out 2>/tmp/rea-bad.err; then
    echo "unexpected acceptance of invalid input: $bad" >&2
    exit 1
  fi
done

sha256sum /tmp/rea-debounce-o0 /tmp/rea-debounce-o2 /tmp/rea-debounce-20
```

The hash output must be retained as execution evidence. RDB-001 requires that the two same-threshold binaries actually have different SHA-256 values **and** identical outputs over the full vector set. The build flags make a difference likely; they do not replace checking the resulting artifacts. Capture compiler identity, exact argv, binary digests, exit status, stdout/stderr, and all test vectors/results.

Run the corpus's executable contract verifier with an existing C compiler, `jq`, and `sha256sum`:

```sh
benchmarks/rea-differential/v1/verify.sh
```

The verifier compiles all four declared profiles in a fresh temporary directory, checks their binary identities, derives expected outputs from `oracle.json`, evaluates RDB-001 through RDB-006, and verifies malformed-input rejection. It prints the actual generated hashes and compiler identity. It does not invoke REA or test real hardware.

**Recorded execution:** [local verification receipt](receipts/local-debian13-gcc14-2026-10-18.json). The verifier passed with Debian GNU/Linux 13, GCC 14.2.0-19 and jq 1.7. The source, case manifest, oracle and verifier were each checked to have the exact Git blob IDs on the tested branch before running from a temporary mirror. This is a local synthetic-fixture pass, not a GitHub CI pass, REA/provider qualification, or a physical-hardware result.

For each physical pulse width in a case, the fixture model converts `pulse_width_microseconds` to `pulse_ticks = floor(width / timer_tick_us)`. The host program receives the resulting integer ticks. This conversion is a **synthetic experimental model**, not a model of any particular MCU timer, oscillator tolerance, interrupt latency or signal conditioning circuit.

## Required negative and interpretation controls

- RDB-001: different artifact hashes are not automatically a behavioral difference.
- RDB-003/RDB-004: changed outcomes do not imply firmware changed; input trace and equipment configuration are separate causal variables.
- RDB-005: a matching observation is not enough to identify one of two hypotheses; select the stated 22 ms test because predictions differ.
- RDB-006: a diagonal before/after comparison confounds two simultaneous interventions; the two cross-over tests are necessary to estimate their separate effects.
- Invalid input: malformed or signed strings fail with exit code 65, empty stdout and a diagnostic. They must never produce ACCEPT/REJECT.
- Missing values stay missing. This corpus has no real sensor or device data and must never be used as evidence about physical hardware.

## Integration with REA

Once compiled artifacts are built, a future verifier can use REA's supported static inspection on these **source-built fixtures**. It should bind any finding to the actual generated binary digest and pinned tool identity. This directory does not currently call REA or assert that its analysis correctly detects the threshold. Nor does it implement a Rust-side Evidence importer.

The next implementation layer should add an executable, dependency-light evaluator (preferably Rust-native within the existing engineering test conventions) that checks build identities, known-answer output parity, oracle consistency, and a claim-to-evidence ledger. REA-assisted diagnosis must then be compared with a no-REA baseline on held-out source families. Keep the public oracle away from the held-out scorer and retain independent adjudication.

## Status and claims ceiling

No compiler, REA provider, Binwalk, Unblob, Ghidra, emulator or physical device was run while authoring these files. The source and oracle are committed inputs, not test results. Passing this synthetic fixture would validate only this controlled model; it would not qualify MCU firmware, establish general decompilation fidelity, prove root cause on a real board, or authorize a device action.
