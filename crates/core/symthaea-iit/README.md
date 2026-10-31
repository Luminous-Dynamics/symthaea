# symthaea-iit

Rust-native foundation for a formalism-pinned Integrated Information Theory (IIT)
research engine.

## Current scope: bounded IIT 4.0 (2023) system search

This crate now provides a validated dense binary factored TPM; cause/effect and
forward-cause repertoire primitives; pointwise generalized intrinsic difference
(GID); maximum-entropy cut transforms; an exact-quotiented directed set-partition
generator; cause/effect intrinsic specified-state selection with tie families;
fixed-state system-partition scoring; and a bounded exhaustive IIT 4.0 (2023)
Integration-level system search for phi_s. Model, evaluation, and analysis-request
identities are versioned and SHA-256 bound.

The system search accepts both the full substrate and embedded proper
candidate systems. For proper systems it implements IIT 4.0 Eq. 3 effect-side
conditioning on the observed external state and Eq. 4 cause-side background
marginalization. Crucially, partitioned cause likelihoods retain the original
system's Eq. 4 background posterior; the cut graph then removes severed
candidate-system input axes by maximum-entropy marginalization. Recomputing the
background posterior from a globally cut TPM is not equivalent and is explicitly
avoided.

The search is complete only for the declared binary factored TPM profile, the
IIT 4.0 (2023) Integration-level quantity, the default directed-set-partition
family, and the explicit work budgets passed to the API. The three independent
caps bound canonical partition generation, tied specified-state pairs, and the
Cartesian product of state pairs × partitions scored. The product is checked
before any partition scoring begins; exhaustion returns an error, never a partial
analysis marked complete. All three limits are bound into the result receipt.
It applies the default PyPhi MIP ranking: minimum normalized phi, then maximum
nonnegative phi, then lexicographic induced cut. It evaluates every tied
cause/effect specified-state pair before applying the Integration-level
maximum-phi rule.

This remains a partial implementation, not a complete PyPhi port. It does not
construct the full cause-effect structure, compute structure-level big Phi, or
resolve nonzero specified-state ties that require Composition-level big-Phi
escalation. The IIT 4.0 (2026) intrinsic-information cap is not implemented by
the system-search API, even though the formalism identifier is represented in
the type system. No result supports a claim about consciousness.

A correct engine cannot compare with PyPhi if the two sides silently disagree
about state order, TPM layout, connectivity, evaluation state, time grain,
formalism, partition family, or tie policy. The implementation therefore exposes
these assumptions and refuses to label incomplete searches exact.

### Proof-carrying result identity

Every completed system-search result exposes `receipt_sha256()`. The versioned
canonical receipt binds the request digest, algorithm identifier, numerical
precision, explicit work budgets, TotalCut policy, formalism, result status,
phi_s (or its absence), selected/tied state-pair indices, exact partition work
counters, each candidate state pair, selected MIP cut matrix, directional GID
terms, normalization, and selection margin.

The receipt is an integrity and reproducibility mechanism: changing a search
setting or a reported score changes the digest, and non-finite fields are rejected.
It is **not** a proof that the TPM is the correct model of a physical system, nor
a proof of IIT's axioms or a consciousness claim. An independent verifier can
recompute the digest from a serialized result and separately replay the analysis.

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

## Sparse local-CPT substrate and Eq. 4 scaling

The dense reference model stores 2^n probabilities per target and remains
capped at 16 units. The separate SparseBinaryCausalModel stores one local CPT
per target: a sorted parent scope plus a table of size 2^k for k parents. Its
model and evaluation digests use a distinct versioned encoding from dense TPM
digests. Conversion from dense is allowed only when every TPM column is exactly
a function of its declared parent scope.

The sparse model's forward-cause API runs the exact IIT 4.0 Eq. 4 background
posterior with the log-domain min-fill factor engine. Candidate-past variables
are duplicated, external-background variables are shared, and the original
system observation likelihood is normalized before the target-wise cause
factors are multiplied. Zero likelihoods remain exact negative-infinity log
weights. All local and intermediate factor tables, plus the returned candidate
state vector, are checked against the explicit factor-entry budget. Each result
also exposes a deterministic receipt that binds the sparse model identity,
observed state, candidate-system order, algorithm version, factor-entry budget,
likelihood vector, and elimination telemetry. This is an integrity receipt,
not a proof of the model's scientific validity.

The sparse engine now includes Eq. 3 effect repertoires with observed-state
external-input clamping, Eq. 4 cause-likelihood contraction under the original
whole-system background posterior, and maximum-entropy directed cuts applied
directly to local CPTs. A partitioned Eq. 4 call computes its posterior from
the original uncut model and uses the cut local CPTs only for target-specific
cause factors; it never recomputes the background posterior from the cut model.
The sparse partition scorer combines Eq. 3, Eq. 4, cause/effect GID, positive-part
clamping and the dense scorer's cut-edge normalization.

### Exact multi-query reuse across partitions and state ties

The sparse system analyzer separates state-independent inference from
partition-dependent inference and state-pair score lookup:

- One shared context prepares the original uncut observation factors,
  candidate/background variable maps, posterior normalizer, and min-fill cost
  telemetry for the fixed model, candidate, observed state, and factor budget.
- The uncut cause-likelihood/selectivity and Eq. 3 effect selectivity are
  computed once per system analysis.
- Each directed cut is applied once. Its partitioned Eq. 4 likelihood vector
  and Eq. 3 effect repertoire are computed once and reused across every tied
  cause/effect specified-state pair.
- The baseline and per-partition Eq. 4 likelihood-vector receipts are hashed
  once at preparation time, then their digests are reused in every scalar score
  receipt. The state-pair-specific outer receipt is still computed separately. 
- Only the indexed scalar GID terms, combined score, and state-pair-specific
  outer receipt are produced per state-pair × partition combination.

This does **not** reuse a cut posterior across different cuts: the original
uncut posterior factors and normalizer are shared, while each target-specific
cut likelihood still performs its own exact factor contraction. The report
distinguishes the shared posterior-normalizer contraction from newly performed
target contractions. Receipt versions and algorithm identifiers were bumped
for the changed report and search contracts.

The regression test compares the sparse analyzer's complete state-pair winners
against the dense analyzer on a small system, then compares cached partition
terms and their final score receipts against independently prepared
single-partition scores. The system-result receipt now carries the exact dense
or sparse algorithm identifier; the identifier is part of the versioned receipt
rather than inferred from the request hash.

The `sparse_system_search_reuse_sweep` benchmark compares the optimized full
analyzer with a deliberately uncached exhaustive loop that calls the
single-partition API once for every state pair × partition. It checks every
state-pair winner, cut matrix, tie outcome, score, and final `phi_s` qualification
status before emitting latency. Its work counters also show the expected
reduction in baseline preparations, partition-repertoire preparations, and
nested Eq. 4 receipt hashes.

Every CSV row also binds the exact sparse model SHA-256, system-search request
SHA-256, algorithm identifier, decimal precision, factor-entry ceiling, and
partition/state-pair/score budgets. This lets a reported timing row be tied to
the exact mathematical input and search policy that was validated.

Run:

    cargo run --release --locked -p symthaea-iit --example sparse_system_search_reuse_sweep
    cargo run --release --locked -p symthaea-iit --example sparse_system_search_reuse_sweep -- --repeats 7

This measures the effect of scoped reuse inside Symthaea's sparse backend. It
is not a PyPhi speed comparison; use the benchmark policy below for any
cross-implementation claim.

Dense-vs-sparse parity covers local transitions, Eq. 3 repertoires, uncut Eq. 4,
maximum-entropy-cut TPM cells, partitioned Eq. 4 and complete per-partition
cause/effect scores for an embedded five-unit substrate. Full system-search
differential coverage now adds 18 deterministic cases across three- and
four-unit strongly connected cycles, three CPT parameterizations per size,
and three observed states per parameterization. Each case compares all
specified-state pairs, MIP cut matrices, cause/effect GID terms, combined scores,
tie status, theorem-bound metadata, and result completion against the dense
oracle. The common theorem-bound guard is tested for valid-within-tolerance,
over-bound negative control, non-finite scalar, and theorem-out-of-domain cases.

The sparse backend now also exposes a bounded exhaustive
`analyze_iit4_2023_system_phi_s` path for the declared binary local-CPT profile.
It performs the sparse intrinsic cause/effect state selection, canonical
directed-partition enumeration, per-partition Eq. 3/Eq. 4 scoring, MIP ranking,
and state-pair tie cascade. It uses the same explicit state-pair, partition,
partition-score, and factor-entry budgets as the dense path. It binds the
certified `n(n-1)` upper bound only when the theorem assumptions hold; TotalCut
and mandatory one-unit TotalCut remain outside that bound domain.

The sparse system search is exact for the declared computation when it returns
`complete = true`, but it is **not yet generally qualified against PyPhi**.
Dense-vs-sparse tests now cover 18 end-to-end small-system fixtures in addition
to the repertoire and cut-transform oracle suite; they do not replace an
independently pinned PyPhi fixture corpus. Nonzero
Composition-level specified-state ties still return an explicit unresolved
status because this crate does not yet construct distinctions, relations, or
big Phi. The IIT 4.0 (2026) intrinsic-information requirement remains a
separate unimplemented path. A large sparse benchmark must be described as
sparse IIT 4.0 (2023) system search only when it actually runs the full
bounded analyzer—not as a PyPhi-equivalent result.

Validation includes dense-to-sparse transition equality, Eq. 3 repertoire
parity, maximum-entropy cut TPM and partitioned Eq. 4 parity, full per-partition
cause/effect score parity, an independent full-joint Eq. 4 posterior oracle on
small embedded systems, and sparse ring fixtures at 64, 256, and 512 units.
The full-joint oracle enumerates the background posterior directly and does not
call the production elimination or projection helpers.

The sparse Eq. 4 scale sweep measures ring substrates at 64, 256, 512, 1,024,
and 2,048 units with a four-unit candidate. It reports local CPT cells, parent
references, posterior and target induced widths, peak factor entries, factor
multiplications, median and minimum elapsed time, and an output checksum.
These measurements are for the Eq. 4 cause-likelihood primitive only; they
must not be represented as full phi_s throughput or compared with end-to-end
PyPhi system-analysis timings.

A separate fixed-partition score sweep uses the same 64–2,048-unit ring
substrates and a four-unit embedded candidate. It records cause/effect signed
scores, normalized partition score, severed-edge count, uncut and partitioned
Eq. 4 receipts, a complete per-partition score receipt, and latency. This is
one fixed-state per-partition evaluation, not the exhaustive intrinsic-state /
all-partitions search needed for phi_s.

## Published reference fixture

The integration fixture pins the three-unit logistic substrate from PyPhi's
published IIT 4.0 (2023) Fig. 1A example. It checks all 24 state-by-unit TPM
values, little-endian state indexing for the specified state (0, 1, 1), and
directed connectivity.

End-to-end regressions run the bounded IIT 4.0 (2023) system search
for all three published candidate systems at state (A,B,C)=(0,1,1):
{A} = 0.0396 ± 0.001, {A,B} = 0.1719 ± 0.001, and
{A,B,C} = 0.1339 ± 0.001. The {A} case is especially diagnostic: it
exercises a proper subsystem with external background units and a self-loop
TotalCut. These rounded tolerances qualify the named Fig. 1A examples only;
they do not establish general PyPhi parity. Separate noisy-OR tests pin Eq. 3
and Eq. 4 proper-subsystem repertoire behavior against PyPhi reference values.

The upstream example source is pinned at:
https://github.com/wmayner/pyphi/blob/134a8f31b49db512322465ed3ba5da7194d362fd/pyphi/examples.py.

## Development gates

1. **Input and primitive parity:** compare forward cause likelihoods, cause/effect
   repertoire distributions, full GID vectors, directed cut matrices, and every
   post-cut TPM cell against pinned PyPhi on identical model, state, subsystem,
   and partition-profile inputs. Include deterministic, independent, zero-likelihood,
   disconnected, and external-unit negative controls.
2. **System-result parity:** differentially compare the end-to-end IIT 4.0 (2023)
   system search: intrinsic specified-state ties, Cartesian cause/effect state
   pairs, all partition scores, MIP selection, normalization, runner-up/margin,
   and final phi_s. The three published Fig. 1A anchors {A}, {A,B}, and {A,B,C}
   are now executable regressions; broaden beyond these named examples.
3. **Composition parity:** implement distinctions, relations, and big Phi; only then
   resolve nonzero state-pair ties that currently return an explicit unresolved
   status. Do not invent a canonical winner when PyPhi escalates to Composition.
4. **Formalism separation:** implement IIT 4.0 (2026)'s intrinsic-information
   requirement as a separate path. It caps phi_s after MIP selection; it must not
   silently alter the 2023 computation.
5. **Certified scale-up:** add mathematically valid lower/upper bounds, branch-and-bound
   certificates, and exact stopping conditions. Heuristics are not bounds, and a
   budget-exhausted search must never return an exact scalar.
6. Add property-based and metamorphic tests for unit/state relabeling, edge
   intervention invariance, invalid TPMs, boundary probabilities, deterministic
   replay, and cross-language digest parity.

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

## Certified system-phi invariant

For binary units with a conditionally independent TPM and the ordinary
directed-set-partition family that preserves self-connections, IIT 4.0
(2023) has the certified upper bound phi_s <= n(n - 1) (Zaeemzadeh &
Tononi, 2024, Table 2). Both the dense and sparse system-search paths bind this
bound into completed results and fail closed if a returned scalar exceeds it.

The helper returns no certificate when TotalCut is requested, because that
special cut severs self-connections; it also returns no certificate for a
one-unit system, where TotalCut is mandatory even when the flag is false.
This invariant is a correctness alarm, not a pruning optimization or an
independent proof of PyPhi parity.

Reference: https://doi.org/10.1371/journal.pcbi.1012323

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



## Exact binary factor-graph kernel (new research track)

The crate now includes a standalone binary sum-product engine:
`BinaryFactor` and `min_fill_sum_product`. Each factor has an explicitly
sorted variable scope and a dense table local to that scope. The deterministic
min-fill heuristic selects the next variable by (fewest fill edges, lowest
degree, lowest variable id), then performs exact multiplication and summation.
The public `plan_min_fill_elimination` API computes the elimination order,
induced width, exact predicted peak bucket-table entries, and final retained
output-table size before any factor multiplication. It rejects an over-budget
plan before contraction begins. Both ordinary and log-domain execution consume
the same planner, and their reports expose predicted versus observed peak entries,
output size, multiplication count, and variables eliminated. Each actual
intermediate allocation is still independently checked against the caller's
entry budget as a second line of defence.

This is the architectural direction for scaling IIT 4.0 Eq. 4 background
marginalization: contract the causal factor graph with cost governed by induced
width, rather than enumerating every global past state. Sparse chains and rings
should remain tractable as node count grows; densely coupled graphs should
expose their high width and stop at the explicit memory budget. This can make
proper-subsystem analysis practical in large substrates when the candidate
system and its boundary have low effective treewidth.

**Current integration boundary:** the log-domain min-fill path is now used
for system-level Eq. 4 cause factors when mechanism and purview both equal the
candidate system. It contracts the uncut observation factors over candidate-past
and external-background variables, then attaches a second candidate-past copy
for each target factor. This avoids materializing the joint background posterior
table. Multiplication is log addition and marginalization is log-sum-exp; exact
zero likelihoods remain negative infinity until the final probability factors
are emitted.

The dense exhaustive implementation is retained as a differential oracle and
remains the fallback for general mechanism/purview subsets. Regression tests
compare every Eq. 4 per-node factor across embedded subsystems, noncanonical
candidate ordering, and extreme joint-likelihood underflow. A separate test
compares the complete partitioned cause-likelihood vector against an independent
dense-factor projection for every directed cut of an embedded three-unit system.
The public `plan_min_fill_elimination` API plans the same deterministic order
without multiplying factor tables. It reports induced width, predicted largest
bucket table, and predicted final output table; the predicted peak is the maximum
of input, bucket, and retained-output sizes. Both probability-domain and
log-domain inference call this planner and expose predicted versus observed peak
entries. A plan is rejected before contraction if either an intermediate or
the fully materialized retained output exceeds the memory budget. The published
Fig. 1A system anchors remain end-to-end regression gates. This does not establish
general PyPhi parity yet; broader generated-model differential coverage and
release-profile performance comparisons are still required.

Run the topology sweep:

    cargo run --release --locked -p symthaea-iit --example factor_elimination_sweep

It emits CSV rows for chains (32–2,048 variables), rings (32–512), and
grids (4×4 through 14×14), including a separate preflight-planning latency,
induced width, predicted output and peak entries, observed peak entries,
multiplication count, end-to-end median/minimum latency, and a deterministic
plan/output check. The preflight estimate also accounts for the final product
over all retained variables, including disconnected retained components; its
memory ceiling therefore covers the complete return factor, not only elimination
buckets.
The same bucketed, log-domain kernel is used by the system-level Eq. 4 cause
path; end-to-end system-search timing must still be measured separately.


### Exact cutset conditioning under a factor-table ceiling

The factor graph also exposes
`min_fill_log_sum_product_with_cutset`. It conditions an explicit set of
eliminated binary variables, enumerates every one of the (2^k) assignments,
runs one preflighted min-fill plan on the residual factor graph, then combines
branch outputs with log-sum-exp. The residual plan is computed once because all
conditioned branches retain the same scopes.

This is an exact alternative when ordinary min-fill exceeds the caller's
largest-factor-table budget. It trades an explicit branch count for a lower
residual induced width; it does not approximate, truncate, or silently change
the factorization. A cutset variable absent from all factors is still enumerated
and correctly contributes its factor-of-two summation. Both the maximum table
entry count and the maximum number of conditioned branches have independent
fail-closed budgets.

The factor-entry ceiling bounds individual input/intermediate/output tables,
not total process RSS. The caller still needs to measure process memory when
making deployment-level memory claims. The current implementation streams one
conditioned branch at a time and accumulates only the retained output factor.

A 4x4 grid regression demonstrates the intended regime: full min-fill exceeds an
eight-entry table ceiling, while conditioning one row leaves a residual graph of
width at most two. The complete retained-state vector is checked against an
independent full-joint log-domain enumeration. The separate scale sweep compares
unconditioned and conditioned results on 4x4, 5x5, 6x6, and 8x8 grids; it records
branch count, both induced widths, both peak table sizes, conditioned table
entries materialized, branch-output log-add operations, factor multiplications,
and timing. Per-fixture factor-entry budgets are 8, 8, 16, and 32 respectively;
each case also asserts that ordinary min-fill is rejected under that same budget.
The 8x8 baseline needs 1,024 factor entries, while the conditioned residual fits
within 32. Its timing ratio is a cost tradeoff, not a speedup claim.

Run:

    cargo run --release --locked -p symthaea-iit --example cutset_conditioning_sweep
    cargo run --release --locked -p symthaea-iit --example cutset_conditioning_sweep -- --repeats 15


### Proper-system Eq. 4 background sweep

A second benchmark exercises the public production path
`forward_cause_likelihood_iit4` with a fixed four-unit candidate system embedded
in sparse ring substrates of 8, 12, and 16 units. It records total substrate
state count, candidate-system state count, external-background size, median and
minimum latency, and a deterministic result checksum.

Run:

    cargo run --release --locked -p symthaea-iit --example eq4_sparse_background_sweep

This isolates the cost of Eq. 4 cause-side inversion as the background grows while
the candidate system remains fixed. It does not measure the full system partition
sweep, and it is not a PyPhi performance comparison. The dense implementation is
still used as an exact differential oracle on small models; larger performance
claims require the same input TPM, formalism, candidate, state, and result status
on a pinned PyPhi artifact.

### Factor-graph benchmark targets

- **Exactness:** zero unexplained dense-vs-factor-graph mismatches on exhaustive
  small binary models, including zero probabilities, disconnected inputs,
  external-background conditioning, and partition interventions.
- **Width behavior:** chain width 1 and ring width 2 should keep peak factor size
  bounded as node count increases; grids should track separator width rather
  than total state count.
- **Memory safety:** 100% of cases exceeding the configured factor-entry budget
  must fail during preflight before contraction, including oversized final
  retained-scope products; actual table constructors retain independent guards.
- **Competitive scale:** report the largest exact proper subsystem and substrate
  solved under fixed memory/time budgets, alongside induced width and boundary
  size. Compare with PyPhi only after matching its exact TPM, formalism,
  background convention, partition family, and result receipt.


## Claim ceiling

A valid BinaryCausalModel proves only that the input satisfies this crate's shape
and probability invariants. The current branch now contains an exhaustive
Integration-level IIT 4.0 (2023) system search for bounded binary factored models,
but its output remains under differential qualification. It is not a complete
PyPhi port: the Composition-level cause-effect structure and big-Phi tie escalation
are absent, and the IIT 4.0 (2026) intrinsic-information requirement is a separate
unimplemented path. The Fig. 1A {A}, {A,B}, and {A,B,C} anchors and noisy-OR Eq. 3/Eq. 4
repertoire fixtures are executable end-to-end/reference checks. The min-fill
Eq. 4 path is additionally differentially compared with the dense oracle and
independent partition projection on small systems. These named fixtures do not
establish general PyPhi conformance; broader generated models and an independently
pinned PyPhi output corpus remain required.

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

The enumerator itself remains a candidate generator; the crate's separate
bounded IIT 4.0 (2023) system-search API applies repertoire and GID scores,
selects the MIP, and returns phi_s when its specified-state tie cascade is
fully resolvable. The search API's current differential qualification is limited
to the named published Fig. 1A anchors and the embedded-system fixtures described
above; it is not general PyPhi parity. Different partition schemes must not be
represented as a speed-only option when they alter the calculated quantity.

Before materializing cuts, callers can use
`estimate_iit4_system_partition_work(candidate, unit_count, include_total_cut)`.
It computes exact raw labeling, canonical cut, pruned labeling, and TotalCut
counts from Stirling numbers with O(n) counter storage. The enumerator uses the
same estimate to reject an insufficient canonical-work budget before building
partition objects. Arithmetic overflow is an explicit error, not a wrapped estimate.


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


## Preflight cost admission

The public work estimator is the cheap preflight analogue of PyPhi's
`estimate_analysis()` at the directed-partition layer. It is not a runtime
estimate and does not predict Eq. 3/Eq. 4 contraction cost; it reports exact
combinatorial counts with checked integer arithmetic. Use it before choosing
a budget or attempting to materialize cuts. A future full-analysis estimator
must additionally include tied specified-state pairs, factor-graph induced-width
forecasts, partition-score evaluations, and relation/Composition work when
those stages are implemented.

Estimator regressions pin both raw and canonical counts for 1–6 candidate units
and compare predictions against materialized reports. A budget one below the
exact canonical count must fail before recursive enumeration.

## Reproducible partition microbenchmark

A standard-library-only Rust example records preflight-estimator median/minimum
latency separately from cut-enumeration median/minimum latency, alongside raw
direction-label count, canonical representatives evaluated, redundant labelings
pruned, and unique cuts for the two- through seven-unit fixtures. Every warm-up
and timed iteration must reproduce all frozen work counts;
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
enumerator for 2–7 units. This is exact-set equality, not a heuristic or approximate
search. The seven-unit fixture reduces the equivalent direction space from 88,560
raw labelings to 61,888 canonical representatives, pruning 26,672 redundant
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
| 7 | 88,560 | 20,392,722 | 3,032,512 | 6.72× |

These figures count matrix-cell loop work only. They exclude set-partition recursion,
direction-subset generation, allocation, and output storage. Release benchmarks must
measure actual wall time and memory on a pinned build and machine before making any
performance claim.
