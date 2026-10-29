# symthaea-iit

Rust-native foundation for a formalism-pinned Integrated Information Theory (IIT)
research engine.

## Current scope: v0.1 causal-model substrate

This crate currently provides a validated dense binary factored TPM and stable
model/evaluation digests. It does not compute IIT system integrated
information (phi_s), mechanism-level phi, a cause-effect structure, structure-level
Phi, or any consciousness result.

The intentional first step is to agree on the input contract before implementing
the combinatorial algorithms. A mathematically correct engine cannot compare with
PyPhi if the two sides silently disagree about state order, TPM layout, connectivity,
evaluation state, time grain, or formalism.

### Data contract

- Binary units and a state indexed by unit.
- Unit 0 is the least-significant bit in the TPM state index.
- TPM cells are row-major P(next_unit = 1 | current_state).
- The factored transition probability assumes next units are conditionally
  independent given the current state.
- Connectivity is a separately declared directed matrix, source-major, and is not
  inferred from the TPM.
- The dense in-memory profile accepts 1–16 units. This is a storage guard for this
  representation, not a claim that exact IIT is tractable at 16 units.
- Model digest: SHA-256 over a versioned binary encoding of unit count, exact
  TPM probabilities, and directed connectivity. Negative zero is normalized.
- Evaluation digest: SHA-256 binding the same model encoding to the requested state.
- A digest establishes byte-level identity only. It does not certify that the model,
  causal interpretation, or scientific theory is correct.

## Development gates

1. Pin the TPM/state/connectivity semantics against a documented PyPhi fixture.
2. Implement small-system probability repertoires and compare each intermediate
   distribution with PyPhi on the same model digest.
3. Add IIT 4.0 (2023) system phi_s with a versioned formalism profile and exhaustive
   partition search for the first tiny systems.
4. Keep IIT 4.0 (2026) as a distinct profile because the intrinsic-information
   requirement changes the system quantity.
5. Add certified lower/upper bounds only when they are mathematically valid for the
   named quantity; do not label heuristic estimates as bounds.
6. Add mechanism distinctions, relations, and structure-level Phi only after system
   phi_s parity is independently established.
7. Add property-based tests for state/unit relabeling, invalid TPMs, boundary
   probabilities, deterministic replay, and cross-language digest parity.

## Benchmark policy

The first comparisons must measure parity, not speed. Once equivalent outputs are
established, record wall-clock time, peak memory, units, state count, partition
search count, cache hits, search completeness, and result status. Compare exact
against exact for identical formalism/input/configuration. If a search mode changes
the partition family, report it as a different formalism/profile rather than a speed
optimization.

Required fail-closed behavior: invalid model, unsupported formalism, timeout,
incomplete exact search, missing reference output, or mismatched input digest must
never produce a qualified scalar result.

## Claim ceiling

A valid BinaryCausalModel proves only that the input satisfies this crate's shape
and probability invariants. It does not establish IIT conformance, integration, or
consciousness. The current crate is an input/data-contract milestone, not a PyPhi
port and not yet an IIT engine.
