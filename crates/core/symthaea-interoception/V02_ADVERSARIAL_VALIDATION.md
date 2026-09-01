# Affective Emergence v0.2 — Adversarial Observatory Validation Matrix

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract defines how the observational-affect layer should be attacked before any candidate is interpreted scientifically. The goal is to falsify the measurement pipeline itself before using its outputs to evaluate affect hypotheses.

## Principle

A candidate is not trustworthy merely because its unit tests match hand-picked expected values.

The observatory must survive adversarial transformations where the correct behavior is known from invariance, symmetry, information availability, or provenance constraints.

Failures here are **integrity failures**, not interesting affect results.

## Validation classes

Every qualified candidate should have evidence across five classes:

1. **information integrity** — no future/semantic leakage;
2. **causal isolation** — observation cannot alter native execution;
3. **mathematical integrity** — temporal alignment and numeric behavior are correct;
4. **provenance integrity** — substitutions/tampering are detected;
5. **scientific discriminant integrity** — candidate is not trivially reducible to nuisance variables under designed controls.

## A1 — future suffix mutation invariance

Take a valid execution prefix ending at `t`. Generate many adversarial suffixes after `t` by mutating:

- future drive magnitudes/signs;
- future intervention timing/values;
- future run duration;
- future semantic condition meaning;
- future realized states in diagnostic fixtures.

For every `OnlinePrefixCausal` candidate:

`M_t(prefix + suffix_a) == M_t(prefix + suffix_b)`

exactly, because qualified computation sees only the prefix.

Any difference is a hard future-information leak.

## A2 — exact-prefix divergent-future twins

Use locked scenario twins whose information content is byte-identical through cut point `t` but diverges afterward.

All online candidate artifacts through `t` must be identical, including:

- availability state;
- numeric values;
- provenance fields that are allowed to depend only on prefix identity;
- candidate-level logs.

A semantic explanation of the future must not appear in the primary artifact.

## A3 — semantic mapping permutation invariance

Hold blinded execution data fixed and replace the semantic arm mapping with arbitrary valid permutations.

Before unblinding:

- forecast trajectories unchanged;
- candidate time series unchanged;
- candidate ranking by blind code unchanged;
- primary logs unchanged.

Only the later semantic report may change labels according to the mapping.

## A4 — semantic-label canaries

Inject unique high-entropy semantic canary strings into arm descriptions and scenario interpretation metadata.

Scan all qualified primary outputs for exact/subsequence occurrences:

- execution artifacts;
- prefix views/artifacts;
- forecast trajectories;
- candidate artifacts;
- blinded comparisons;
- structured logs/error messages.

Any canary leakage is an integrity failure.

## A5 — observer/no-observer bisimulation

Execute the same locked native study under two conditions:

1. observatory absent;
2. observatory attached in qualified read-only mode.

Require exact equality of the complete native v0.1 execution trace and mechanical receipts.

Repeat with:

- all candidates enabled;
- one candidate at a time;
- diagnostics disabled;
- different observatory internal iteration orders where applicable.

Observation must be causally transparent.

## A6 — candidate-order permutation invariance

If multiple candidates are computed from the same immutable evidence, permuting their evaluation order must not change any candidate output.

This detects hidden mutable shared state and accidental cross-candidate contamination.

## A7 — duplicate-candidate independence

Compute the same candidate twice under different opaque registry positions/IDs but identical canonical definition.

Numerical output must agree exactly; registry position must not affect computation.

If IDs are identity-bearing by design, provenance IDs may differ only where the contract explicitly requires them to.

## A8 — future protocol redaction equivalence

Run qualified online computation from:

- a full study object passed through a safe prefix-materialization boundary; and
- an independently materialized artifact that never contained future protocol content.

Outputs must agree exactly.

This verifies that redaction is not merely hiding fields at presentation time while retaining hidden access paths.

## A9 — oracle separation test

Build/execute online candidate code with oracle diagnostics unavailable or disabled.

Qualified online outputs must remain identical.

Attempting to register an oracle candidate as primary online evidence must fail validation/type construction.

An oracle artifact must carry explicit `OracleDiagnostic` provenance and must not satisfy online-candidate qualification.

## A10 — temporal shift/self-consistency tests

For deterministic self-consistent forecasts where actual transition equals the prior one-step forecast and policy inputs remain unchanged:

- R2 one-step residual should be zero/equivalent within declared tolerance;
- R3 overlapping-future revision should be zero/equivalent;
- forecast trajectory overlap should align by absolute time.

Introduce deliberate off-by-one fixtures to ensure the test suite catches temporal-indexing errors.

## A11 — rolling-horizon boundary dominance diagnostic

Construct cases where overlapping future predictions remain unchanged but the dropped/added boundary terms differ.

Expected:

- R3 approximately zero;
- R4 may change because of horizon turnover.

This fixture proves that the pipeline can distinguish future-outlook revision from rolling-window composition.

If R3 follows the boundary-only change, temporal alignment is broken.

## A12 — sign-crossing fixtures

Include deterministic fixtures for:

- worsening but better than expected (`R1 < 0`, `R2 > 0`);
- improving but worse than expected (`R1 > 0`, `R2 < 0`);
- unchanged current burden with changed outlook (`R1 ~= 0`, `R3 != 0`);
- current change with self-consistent outlook (`R1 != 0`, `R2 ~= 0`, `R3 ~= 0`).

These are not optional examples. They are anti-collapse tests proving R1/R2/R3 represent separable computations.

## A13 — nuisance-preserving metamorphisms

Construct transformations designed to preserve one nuisance variable while altering regulatory consequence, and vice versa.

Examples:

- equal drive magnitude, different viability margin;
- equal current homeostatic burden, different observed velocity;
- equal current burden and drive magnitude, different recovery configuration within declared valid region;
- equal peak deviation, different channel breadth/duration.

Candidate claims beyond nuisance baselines require preregistered discrimination under these paired cases.

## A14 — scale/representation sensitivity

Within valid representation rules, test transformations such as:

- serialize/deserialize every input artifact;
- reorder map/object fields in non-canonical source JSON then canonicalize;
- convert v0.1 `f32` values into v0.2 `f64` exactly before derived accumulation;
- evaluate deterministic sums using the locked accumulation order.

Canonical outputs/digests must remain stable where identity semantics say they should.

Non-finite values are validation failures, not clamp-to-zero cases.

## A15 — candidate-definition tampering

For each identity-bearing candidate field, mutate exactly one item:

- sign;
- horizon;
- discount;
- forecast class;
- temporal availability;
- alignment rule;
- normalization;
- undefined-value semantics;
- reference fixture digest;
- source/model semantic identity.

The candidate digest must change and artifacts from the old definition must fail validation against the new definition.

## A16 — scenario substitution and near-duplicate audit

Attempt to substitute:

- renamed discovery scenario into holdout;
- same content with new ID;
- same outcome-relevant content with changed blind code;
- scenario generated from an unregistered generator version/seed.

Content/cohort validation must reject or explicitly mark the substitution.

A near-duplicate policy violation cannot become confirmatory evidence merely through renaming.

## A17 — missing-scenario accounting attack

Remove one locked confirmatory scenario from the realized package.

The equality

`locked_count == included + excluded + indeterminate`

must fail and promotion must become `IntegrityFailure`.

Hypothesis-inconsistent scenarios cannot be silently omitted.

## A18 — exclusion manipulation attacks

Try:

- missing exclusion decision;
- duplicate decision;
- decision for unknown criterion;
- malformed evidence digest;
- decision bound to different execution;
- converting `Indeterminate` or `Triggered` to `NotTriggered` without changing receipt digest.

All must fail validation.

Surprising candidate output is never by itself a valid exclusion reason.

## A19 — frozen blinded artifact replacement

After creating the blinded comparison digest, alter any candidate value/order/availability state and attempt unblinding.

The unblinding receipt must reject the replacement.

Semantic evaluation must be a deterministic transformation of the already-frozen blinded artifact plus the committed mapping/analysis plan.

## A20 — analysis-plan mutation

After confirmatory root lock, mutate:

- primary candidate;
- threshold;
- equivalence band;
- cut point;
- candidate ranking rule;
- scenario subset;
- baseline set;
- tie handling.

The analysis-plan/root digest must change. Results under the modified plan are exploratory/new-lineage outputs, never the same confirmatory study.

## A21 — deterministic replay under clean process state

Recompute qualified derived artifacts in a fresh process using only serialized locked inputs.

Require matching canonical digests for:

- forecast trajectories;
- candidate time series;
- blinded comparison;
- validation reports;
- semantic report after unblinding.

This detects hidden process-local state/environment dependence.

## A22 — dependency/source boundary audit

Mechanical/source checks should verify:

- native v0.1 has no observatory dependency;
- online candidate modules have no oracle/unblinding dependency;
- semantic mapping type absent from primary module dependency graph;
- candidate outputs cannot convert to native drives/actions;
- no interpretation-bearing emotion category appears in causal/metric API;
- no qualified path calls wall clock, OS randomness, network, or mutable global state.

## A23 — deliberately malicious test implementation

During test development, create intentionally bad candidate/policy fixtures that:

- read one future step;
- inspect semantic mapping;
- mutate shared state;
- use candidate ordering;
- leak a canary to logs;
- return zero for unavailable values;
- use the oracle schedule while claiming online status.

Each malicious fixture must be caught by at least one explicit gate.

A validation suite that has never demonstrated its ability to catch known bad implementations is weaker evidence.

## Gate severity

Classify failures prospectively:

### Integrity-blocking

Examples:

- future information leak;
- semantic leak;
- observer feedback;
- replay mismatch;
- artifact substitution;
- missing locked scenario;
- invalid temporal alignment.

These block scientific interpretation entirely.

### Candidate-disqualifying

Examples:

- numerical instability;
- failed neutrality;
- failed sensitivity region;
- equivalent-to-nuisance baseline under preregistered comparison.

These invalidate/promote-against the candidate but do not imply corrupted evidence if the failure itself is faithfully recorded.

### Hypothesis-negative

A valid candidate and valid evidence chain may simply fail the primary scientific hypothesis. That is a qualified negative/null result.

## Validation artifact

The eventual implementation should emit an `ObservatoryAdversarialValidationReport` binding:

- prospective evidence-root digest;
- observatory source commit;
- candidate-definition digests;
- scenario/cohort digests;
- exact validation-suite version;
- each adversarial test ID and outcome;
- malicious-fixture detection outcomes;
- integrity-blocking summary;
- canonical report SHA-256.

This report is required before confirmatory evidence can leave `ExecutedAwaitingValidation`.

## Claim boundary

Passing this matrix would support confidence that the observatory respects its declared information, causality, mathematics, and evidence boundaries. It would not establish emotion, subjective valence, feeling, sentience, or consciousness.