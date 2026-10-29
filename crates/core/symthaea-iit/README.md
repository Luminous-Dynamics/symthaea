# symthaea-iit

Rust-native foundation for a formalism-pinned Integrated Information Theory (IIT)
research engine.

## Current scope: v0.1 causal-model substrate

This crate currently provides a validated dense binary factored TPM, cause/effect
repertoire primitives, a validated pointwise GID measure primitive, a bounded IIT 4.x
directed-system-partition enumerator, typed formalism/quantity identities, and stable model, evaluation, and
analysis-request digests. It does not yet compute IIT system
integrated information (phi_s), mechanism-level phi, a complete cause-effect
structure, structure-level Phi, or any consciousness result.

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
- An analysis request binds the evaluation state, candidate subsystem, explicit
  formalism profile, and requested quantity so incompatible requests are rejected.

## Development gates

1. Run the v0.1 model/repertoire unit tests and pin TPM/state/connectivity semantics
   against a documented PyPhi fixture.
2. Compare cause/effect repertoire distributions, GID point measures, and directed cut matrices against
   the reference on identical model, state, subsystem, and partition-profile inputs,
   including deterministic, independent, and zero-likelihood negative controls.
3. Add IIT 4.0 (2023) system phi_s with a versioned formalism profile, documented
   MIP repertoire-difference measure, and exhaustive partition search for tiny systems.
4. Keep IIT 4.0 (2026) as a distinct profile because its intrinsic-information
   requirement changes the system quantity.
5. Add certified lower/upper bounds only when mathematically valid for the named
   quantity; do not label heuristic estimates as bounds.
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

Required fail-closed behavior: invalid model, unsupported quantity/formalism pair,
undefined cause-repertoire mass, timeout, incomplete exact search, missing reference
output, or mismatched input digest must never produce a qualified scalar result.
Current cause/effect routines are probability-repertoire primitives under explicit
uniform-prior/background assumptions; the directed partition enumerator generates
candidate severance matrices but does not evaluate their repertoire effects or select
a minimum-information partition. These pieces are not by themselves proof of PyPhi
conformance.

## Claim ceiling

A valid BinaryCausalModel proves only that the input satisfies this crate's shape
and probability invariants. It does not establish IIT conformance, integration, or
consciousness. The current crate is an input/data-contract milestone, not a PyPhi
port and not yet an IIT engine.


## Reference material

- PyPhi TPM layout, little-endian state indexing, and connectivity validation:
  https://pyphi.readthedocs.io/en/latest/conventions.html
- PyPhi repertoire construction and computational costs:
  https://pyphi.readthedocs.io/en/latest/theory/computational-complexity.html
- Explicit formalism profiles and their quantity differences:
  https://pyphi.readthedocs.io/en/latest/reference/configuration.html
- IIT 4.0 (2023): Albantakis et al., PLOS Computational Biology (2023):
  https://doi.org/10.1371/journal.pcbi.1011465

These references define the implementation targets. They do not establish parity for this crate.


### Cross-language identity contract

The canonical model digest deliberately excludes display labels, evaluation state,
formalism, and requested quantity. The evaluation digest adds the selected state.
An analysis request digest additionally binds the candidate subsystem, formalism id,
and quantity id. The Python oracle runner must use the same byte encoding, not a
separately canonicalized JSON hash, before it can be used to verify Rust request
identity. Golden digest vectors are a release gate before claiming cross-language
parity.


## Generalized intrinsic difference (GID) primitive

The crate includes a validated pointwise GID operation as specified in PyPhi's
IIT 4.0 implementation:

    selectivity[state] * log2(forward[state] / partitioned[state])

For either zero forward or zero partitioned probability, the undefined-ratio
contribution is zero, matching PyPhi's pointwise convention. The function rejects
non-finite/out-of-range probabilities, distribution-length mismatches, unnormalized
input distributions, and invalid state indices; negative pointwise GID values are
preserved rather than silently clamped.

This is one pointwise component of the system-phi pipeline. It is not a repertoire
distance over the whole distribution, a minimum-information partition score, or
system phi_s by itself.

Reference formula: https://github.com/wmayner/pyphi/blob/main/pyphi/measures/distribution.py

## Directed system-partition enumeration

`enumerate_iit4_system_partitions` now enumerates each non-trivial set partition,
assigns Cause, Effect, or Bidirectional to each block, builds the directed severance
matrix, and deduplicates candidates by that matrix. It preserves self-loops for
ordinary directed partitions; the optional `TotalCut` severs all entries, including
self-loops. A one-unit system always includes `TotalCut`. The caller supplies a
maximum count of block-direction assignments to visit; if the budget is exceeded,
the function returns an error rather than a partial candidate list.

This is a candidate generator only. The next stage is to compare its cut matrices
with a pinned PyPhi configuration and then implement the corresponding cause/effect
repertoire comparison and minimum-information-partition selection. Different
partition schemes must not be represented as a speed-only option when they alter
the calculated quantity.
