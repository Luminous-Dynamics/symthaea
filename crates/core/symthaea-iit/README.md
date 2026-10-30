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

The first step is to agree on the input contract before comparing mathematical
outputs. A correct engine cannot compare with PyPhi if the two sides silently
disagree about state order, TPM layout, connectivity, evaluation state, time grain,
or formalism. The partition generator is now implemented with an exact quotient
over direction-label equivalence classes; the full system-phi pipeline remains
future work.

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

## Published reference fixture

`tests/published_iit4_2023_fig1a.rs` pins the three-unit logistic substrate from
PyPhi's published IIT 4.0 (2023) Fig. 1A example. It checks all 24 state-by-unit
TPM values, little-endian state indexing for the specified state `(0, 1, 1)`,
and the directed connectivity matrix. The paper's reported `phi_s` values
(approximately 0.04 for {A}, 0.17 for {A, B}, and 0.13 for {A, B, C}) are
future output-parity anchors; this fixture does **not** compute or validate
those values. The upstream reference is the PyPhi example source:
https://github.com/wmayner/pyphi/blob/134a8f31b49db512322465ed3ba5da7194d362fd/pyphi/examples.py.

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

## Competitive benchmark gates (targets, not current results)

Use two clearly separated upstream baselines:

- **Reproducible release-candidate baseline:** `pyphi==2.0.0rc4`, pinned with
  `IIT_4_0_2023` and `IIT_4_0_2026` tested separately. As of 2026-10-29,
  PyPI lists 2.0.0rc4 (2026-10-13) as the newest 2.0 prerelease and 1.2.0 as
  the latest stable package. Do not silently treat GitHub `main` or rendered
  development docs as the same artifact:
  https://pypi.org/project/pyphi/
- **Exploratory development comparator:** current PyPhi GitHub source, recorded
  by full commit SHA and used only as a separate comparison until its packaged
  release and output artifacts are pinned.

### Qualification gates

1. **Input parity:** 100% agreement on model dimensions, all TPM cells within a
   declared tolerance, connectivity, state, candidate subsystem, formalism, and
   request digest. A mismatch blocks all downstream comparisons.
2. **Primitive parity:** compare full cause/effect repertoires, pointwise GID
   vectors, each candidate cut matrix, and every post-cut TPM cell—not just final
   scalars or counts. Include deterministic, noisy, disconnected, zero-likelihood,
   unit-relabelled, and external-unit controls.
3. **Quantity parity:** reproduce published `phi_s` anchors per named formalism,
   then use a generated fixture corpus and independent randomized small systems.
   Require zero unexplained mismatches; do not merge 2023 and 2026 results.
4. **Fail-closed behavior:** 100% of malformed inputs, unsupported profile/quantity
   pairs, missing reference outputs, digest mismatches, exhausted budgets, or
   incomplete searches must be reported as invalid/unqualified—never as an exact
   scalar.
5. **Performance only after parity:** measure release-mode end-to-end wall time
   (median and p95), peak resident memory, allocations where available, candidate
   assignments, unique cuts, cache hits/misses, and completed work. Pin machine,
   OS, toolchain, PyPhi version, thread/worker count, formalism, partition scheme,
   input digest, and cache state. Compare cold and warm runs separately.

### Performance goals after correctness gates

The first competitive goal is **at least 2× lower median end-to-end runtime** than
the pinned PyPhi baseline on the exact same complete small-system fixture set,
with identical outputs and no increase in peak memory. Treat this as an aspiration,
not a result or guarantee. If it is not met, report the regression and identify the
dominant cost before optimizing. For scale, report the largest system completed
exactly within fixed wall-time and memory budgets; one extra unit completed is
meaningful only when the algorithm, partition family, and qualification rules are
unchanged. Approximate, bounded, cached, and exact modes must be reported separately.

PyPhi 2.0 itself advertises major optimizations—including approximately 18–20×
faster partition evaluation versus its pre-2.0 implementation, content-based caches,
and reduced-dimensional cause-side inversion. Those are upstream claims against
PyPhi's own previous version, **not** speedups over this Rust crate. The Rust
benchmark must use the same mathematical workload and compare against the current
pinned baseline before making any comparative claim:
https://pyphi.readthedocs.io/en/latest/whats-new-in-2.0.html

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

`enumerate_iit4_system_partitions` enumerates each non-trivial set partition and
emits one canonical direction assignment for every distinct severance matrix. It
does not materialize all (3^k) Cause/Effect/Bidirectional labelings and then hash
away duplicates.

The exact quotient follows directly from the cut rule. For blocks (i,j), the
cross-block edge (i \to j) is uncut **only** when block (i) is Cause and block
(j) is Effect. Thus every cut matrix is uniquely represented by either:

- the all-Bidirectional labeling, which cuts every cross-block edge; or
- a nonempty Cause-block subset (C), a disjoint nonempty Effect-block subset
  (E), and Bidirectional labels for all remaining blocks. The only uncut
  cross-block edges are the complete directed biclique (C \to E).

For (k) blocks this yields exactly
(1 + (3^k - 2^{k+1} + 1) = 3^k - 2^{k+1} + 2) representatives. This matches the
matrix-equivalence relation used by PyPhi's `EdgeCut` equality (candidate node
indices plus severance matrix); redundant direction metadata is not part of the
identity of a cut. Since a cut matrix also recovers its set partition—two units
belong to the same block exactly when neither direction between them is cut—there
are no cross-partition matrix collisions. The implementation can therefore omit
the runtime deduplication hash set entirely.

Ordinary directed partitions preserve within-block self-loops; the optional
`TotalCut` severs every entry, including self-loops. A one-unit system always
includes `TotalCut`. The caller's work budget applies to canonical representatives
actually materialized, while the report records the raw equivalent labelings and
how many were pruned. Budget exhaustion returns an error rather than a partial list.

This remains a candidate generator only. It does not yet apply repertoires to each
cut, score a minimum-information partition, or compute system \(\phi_s\). The
next gate is full-matrix parity and post-cut TPM parity with pinned PyPhi inputs,
then the formalism-specific minimum search. Different partition schemes must not
be represented as a speed-only option when they alter the calculated quantity.


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
that separates the raw equivalent labelings, canonical representatives actually
evaluated, redundant labelings pruned before matrix construction, and unique
non-total cut matrices emitted. It also reports whether the special TotalCut was
included. Under this exact quotient, canonical representatives and unique
non-total cut matrices must agree exactly.

For no TotalCut, the combinatorial regression fixtures specify:

| Units | Raw direction labelings | Canonical representatives evaluated | Redundant labelings pruned | Unique cuts |
|---:|---:|---:|---:|---:|
| 2 | 9 | 3 | 6 | 3 |
| 3 | 54 | 22 | 32 | 22 |
| 4 | 306 | 150 | 156 | 150 |
| 5 | 1,863 | 1,061 | 802 | 1,061 |
| 6 | 12,348 | 7,896 | 4,452 | 7,896 |

These are exact work-count identities and regression anchors, not runtime
measurements. The independent brute-force reference still generates the original
3^k labeling family, constructs its matrices, and compares the complete result
set against the canonical generator through six units.


## Reproducible partition microbenchmark

A standard-library-only Rust example records median/minimum elapsed nanoseconds
alongside raw direction-label count, canonical representatives evaluated,
redundant labelings pruned, and unique cuts for the two- through six-unit
fixtures. Every warm-up and timed iteration must reproduce all frozen work counts;
otherwise it exits nonzero.

Run:

    cargo run --release --locked -p symthaea-iit --example directed_partition_sweep
    cargo run --release --locked -p symthaea-iit --example directed_partition_sweep -- --repeats 25

Capture the CSV output with the commit SHA, rustc version, machine details, and
build profile. This gives a reproducible Rust baseline; it is not a PyPhi comparison
until the same directed cut-matrix set and workload are generated by both engines.
Do not report a speedup until those semantic parity checks pass.


## Exact-preserving cut-matrix quotient

The first optimization precomputes one local block-index array per set partition.
The stronger optimization uses the closed-form equivalence above to emit only
canonical Cause/Effect block subsets. It avoids materializing redundant direction
vectors, allocating a candidate matrix for each redundant labeling, and maintaining
a global hash set of cut matrices. Every emitted candidate still has its explicit
parts, direction vector, and severance matrix.

The test suite retains a deliberately simple brute-force reference generator that
uses the original block-membership definition and all three directions per block.
It compares the complete matrix set—not merely counts—against the canonical
enumerator for 2–6 units. This is exact-set equality, not a heuristic or approximate
search. The six-unit fixture reduces the equivalent direction space from 12,348
raw labelings to 7,896 canonical representatives, pruning 4,452 redundant
labelings before matrix construction.

The analytical matrix-loop comparison below contrasts the previous implementation
(one n² matrix pass per block per raw labeling) with the canonical implementation
(one n² pass per distinct output cut). These are operation counts, not elapsed time
or a claim of end-to-end speedup:

| Units | Raw labelings | Previous block-by-matrix iterations | Canonical matrix-cell checks | Ratio |
|---:|---:|---:|---:|---:|
| 2 | 9 | 72 | 12 | 6.00× |
| 3 | 54 | 1,215 | 198 | 6.14× |
| 4 | 306 | 14,976 | 2,400 | 6.24× |
| 5 | 1,863 | 168,750 | 26,525 | 6.36× |
| 6 | 12,348 | 1,854,252 | 284,256 | 6.52× |

These figures count matrix-cell loop work only. They exclude set-partition recursion,
direction-subset generation, allocation, and output storage. Release benchmarks must
measure actual wall time and memory on a pinned build and machine before making any
performance claim.
