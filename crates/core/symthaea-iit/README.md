# symthaea-iit

Rust-native foundation for a formalism-pinned Integrated Information Theory (IIT)
research engine.

## Current scope: v0.1 causal-model substrate

This crate currently provides a validated dense binary factored TPM, cause/effect
repertoire primitives, forward cause-likelihood computation, a validated pointwise
GID measure primitive, an explicit partition-cut TPM transform, a bounded IIT 4.x
directed-system-partition enumerator,
and typed formalism/quantity identities, and stable model, evaluation, and
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
2. Compare forward cause likelihoods, cause/effect repertoire distributions, GID point
   measures, directed cut matrices, and post-cut TPM rows against the reference on identical model, state,
   subsystem, and partition-profile inputs,
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

The crate includes a validated full-vector and pointwise GID operation as specified
in PyPhi's IIT 4.0 implementation:

    selectivity[state] * log2(forward[state] / partitioned[state])

For either zero forward or zero partitioned probability, the undefined-ratio
contribution is zero, matching PyPhi's pointwise convention. The function rejects non-finite/out-of-range probability entries, distribution-length
mismatches, an unnormalized selectivity distribution, and invalid state indices.
Forward likelihood arrays are not required to sum to one; negative pointwise GID
values are preserved rather than silently clamped.

The vector API returns every pointwise contribution; the selected-state API extracts
one contribution from the same implementation. These are components of the system-
phi pipeline, not a minimum-information partition score or system phi_s by themselves.

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


## Partition application

The model can apply an enumerated cut matrix by marginalizing each severed source
bit under a uniform maximum-entropy intervention. For each cut source -> target
edge, the target's transition probability is averaged over the source bit with all
other bits held fixed; repeated averaging handles multiple cut inputs. The updated
connectivity metadata removes the severed links and the resulting model is run
through the normal TPM/connectivity validator again.

This operation is tested on a directed cross-edge, a total cut including self-loops,
and a cut restricted to a candidate system that leaves links to external units intact.
It is a model-transform primitive; its exact agreement with PyPhi's induced cut
system remains to be tested differentially.


## Partition-search observability

The convenience enumerator returns the partition list. The companion
`enumerate_iit4_system_partitions_with_stats` returns a `PartitionEnumerationReport`
that separates:
- block-direction assignments actually evaluated;
- unique non-total cut matrices retained;
- duplicate matrices suppressed;
- whether the special TotalCut was included.

For the current enumerator and no TotalCut, the combinatorial fixtures specify:
- 3 units: 54 assignments, 22 unique cuts, 32 duplicates;
- 4 units: 306 assignments, 150 unique cuts, 156 duplicates;
- 5 units: 1,863 assignments, 1,061 unique cuts, 802 duplicates.

These counts are implementation regression anchors, not performance measurements.
They make future optimizations measurable: reduce wall time/memory or allocation cost
while preserving the exact cut-matrix set and reporting candidate work transparently.


## Reproducible partition microbenchmark

A standard-library-only Rust example records median/minimum elapsed nanoseconds
alongside exact search counters for the two-, three-, four-, and five-unit
partition fixtures. Every warm-up and timed iteration must reproduce the frozen
assignment and unique-cut counts; otherwise it exits nonzero.

Run:

    cargo run --release --locked -p symthaea-iit --example directed_partition_sweep
    cargo run --release --locked -p symthaea-iit --example directed_partition_sweep -- --repeats 25

Capture the CSV output with the commit SHA, rustc version, machine details, and
build profile. This gives a reproducible Rust baseline; it is not a PyPhi comparison
until the same directed cut-matrix set and workload are generated by both engines.
Do not report a speedup until those semantic parity checks pass.


## Exact-preserving cut-matrix optimization

The enumerator now builds one local block-index array per set partition and
represents each block direction with two Boolean flags (cuts outgoing, cuts
incoming). A source -> target edge is severed exactly when the units belong to
different blocks and the source block cuts outgoing edges or the target block
cuts incoming edges. This avoids repeated linear membership scans inside the
source/target matrix loops.

The test suite contains a deliberately simpler brute-force reference generator
that uses the original block-membership definition, then compares the complete set
of matrices—not merely the count—against the optimized enumerator for 2–5 units.
The existing assignment/unique-cut counts remain fixed. This is an exact-semantic
optimization candidate; runtime improvement must still be measured in the Rust
release microbenchmark before claiming a speedup.


For context before timing is available, the combinatorics imply the following
matrix-loop work counts for the previous implementation (one full n² pass per
block and assignment) versus the current implementation (one full n² pass per
block-direction assignment). These are analytical operation counts, not elapsed
time or a claim of end-to-end speedup:

| Units | Assignments | Previous block-by-matrix iterations | Current matrix-cell checks | Ratio |
|---:|---:|---:|---:|---:|
| 2 | 9 | 72 | 36 | 2.00× |
| 3 | 54 | 1,215 | 486 | 2.50× |
| 4 | 306 | 14,976 | 4,896 | 3.06× |
| 5 | 1,863 | 168,750 | 46,575 | 3.62× |

The older path also performed linear membership scans for both endpoints within
each block iteration; the new path uses the precomputed block index plus incoming/
outgoing flags. Actual performance must be established by the release microbenchmark
on a pinned build and machine, including allocation and deduplication costs.
