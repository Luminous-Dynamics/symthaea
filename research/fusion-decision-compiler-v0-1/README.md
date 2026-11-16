# Fusion decision compiler contract v0.1

An isolated, dependency-free Rust research crate for the first FECOMP-0 compiler semantics.

It is an integration contract, not another fusion system code. It does not replace PROCESS, Bluemira, FreeGSNKE, DESC, materials solvers, thermal/structural/neutronics solvers, Symtropy's engineering twin, or independent engineering authorities.

## Initial capabilities

- Candidate and objective identity checks.
- Typed service metrics with units encoded by variant: target gain, plasma gain, driver wall-plug gain, gross thermal/electric power, net electric export, useful heat, lifetime energy and availability.
- Each typed service metric requires its own exact evidence scope (for example net-electric-export-boundary versus plasma-gain-boundary); the compiler rejects reusing a different metric's boundary evidence.
- Exact solver-run binding to candidate revision and geometry digest.
- Explicit energy-service boundary admission, resolved to the exact evidence record consumed by the energy hard gate.
- Required hard-gate evaluation for energy, material, manufacturing, inspection, assembly and replacement.
- Solver capability, regime, observable, intended model use, applicability, execution and convergence checks.
- Intended-use classes distinguish engineering screening, single-step prediction, open-loop rollout, optimization inside a surrogate, and reference-solver evaluation.
- Explicit distinction between blocked and indeterminate candidates.
- Evidence-to-claim class and revision/scope binding.
- Pairwise solver dependence classification across declared dataset, code and assumption groups; empty or missing dependency identities remain unknown.
- Observable-scoped dependency inventories for solver-build lineage, equations/closures, input/calibration datasets, boundary and initial conditions, geometry, transformations, measurement systems/data, material properties and shared scenario assumptions.
- Partial overlap is retained per observable; same source node with different revisions and byte-identical payloads under different local names are detected as shared dependencies.
- Incomplete inventories remain unknown; complete disjoint declarations mean only that no shared dependency was found in the declared inventory, not proof of physical/statistical independence.
- Deterministic transitive invalidation of downstream dependency nodes.
- Fixed claim ceiling: eligible candidates are only eligible for further engineering review.
- Candidate-funnel audit across the frozen universe at analytical, low-fidelity and high-fidelity stages; every candidate/stage pair must have an explicit disposition, including NotEvaluated.
- Pipeline advancement must follow prior passes; rejected candidates can re-enter only through a documented stratified audit or human override.
- Exact screening recall is reported only when retrospective reference labels cover the full candidate universe; partial labels never masquerade as an exact recall estimate.
- Validation-domain module distinguishes calibration exposure, measured test-stand evidence and blind holdout; it checks source/dataset/episode non-overlap, holdout exposure flags, measured-evidence class, diagnostic/uncertainty provenance, model revision and declared regime/geometry/facility/case/operating-point envelope.
- Holdout transfer requires at least one declared difference from the test-stand measurement (facility, case or operating point), while remaining inside the model's declared domain.
- A successful validation-domain audit grants only eligibility for independent metric evaluation; it is not a model-performance or physical-validation result.

## Dispositions

- `Blocked`: at least one explicit hard failure, unsupported solver capability, out-of-domain model or failed execution.
- `Indeterminate`: a required boundary, model result, gate or applicability state is missing or unknown.
- `EligibleForFurtherEngineeringReview`: every required gate and solver contract is satisfied under the declared synthetic/modelled scope. This is not physical qualification or manufacturing release.

A hard failure takes precedence over an unknown. Missing evidence is never treated as zero cost, zero uncertainty or a passing gate.

## Evidence classes

A claim must cite evidence with an exact compatible class and the same subject revision. Synthetic fixtures cannot support measured, independently verified or physically qualified claims. A digest string is an identity declaration; the crate does not fetch or hash arbitrary source payloads. Solver run records must bind the exact candidate revision and geometry, but source-byte verification remains the responsibility of a separate payload verifier.

## Solver dependence

Two runs are only classified as independent under declared dependencies when dataset, code and assumption groups are all present and pairwise distinct. Any shared group is marked common-mode. Missing group identity is marked unknown. This is not a statistical proof of independence.

## Observable-scoped solver dependence

The initial run-level comparison uses declared dataset, code and assumption groups. The follow-on observable-scoped module refines this with per-output dependency inventories. It can distinguish a shared equilibrium source for one output from an independent empirical closure used for another. A shared source node remains shared across version changes; an identical payload digest exposes aliases. Missing completeness declarations remain unknown. The strongest no-overlap result means only that no shared dependency was found within two declared-complete inventories; it is not a physical or statistical independence proof. The module does not calculate covariance; missing cross-covariance remains the responsibility of the separate Symtropy covariance contract and must not be replaced with zero.


## Matched-case cross-solver comparison

The follow-on cross-solver comparison module does not attempt to rank systems codes from unmatched outputs. It accepts a frozen comparison case and two exact solver-output records. Both outputs must refer to the same candidate revision, geometry, service boundary, observable, canonical quantity, unit, frame, sign convention, time basis and normalization contract. Out-of-domain, unknown-applicability, non-converged, unreported and unexecuted runs are retained as non-success dispositions.

A comparable pair may yield the raw signed/absolute difference. If the exact joint covariance receipt is absent, unknown or stale, the result is **descriptive-only**. No zero covariance is fabricated. A known covariance receipt must bind the ordered pair of exact run IDs, observable ID and output digests; even then, this only enables downstream uncertainty review, not a significance or correctness claim.

The module consumes the existing observable-scoped dependency manifests from the sibling module. It exposes common source lineage and incomplete inventories rather than translating “no shared dependency in complete declarations” into statistical independence. It performs no unit conversion or covariance arithmetic: it requires Symtropy's exact normalization and joint covariance receipts.

The initial 15 authored tests cover matched outputs, descriptive-only and covariance-linked cases, common and unknown dependencies, mismatched geometry/service boundary/quantity/unit/frame/sign/time/normalization/candidate revision, applicability/convergence refusal, stale/swapped covariance receipts, malformed/non-finite results, run-manifest identity mismatch and subtraction overflow.


## Dependency invalidation

The graph helper returns the changed nodes and their full transitive downstream closure. Cycles are tolerated and traversal is deterministic. This is a structural impact calculation, not automatic revocation of an external certificate or authority decision.


## Quantitative blind-holdout predictive assessment

The `validation_domain` module remains the admission gate: it checks calibration/test-stand/holdout non-overlap, measured evidence class, measurement provenance, blind exposure flags, exact model revision and intended use, and declared regime/geometry/facility/case/operating-point scope. It does not measure prediction performance.

The follow-on `validation_assessment` module calls that existing gate for every campaign case before computing any metrics. It then requires a final holdout-unseal receipt, exact model-run ID/revision/input/output digests, and complete predicted/measured values for every frozen observable.

Per observable, the campaign reports:
- mean signed error, bias, MAE and RMSE;
- normalized bias and normalized RMSE under a predeclared physical scale;
- maximum absolute normalized residual;
- empirical point-in-prediction-interval coverage;
- measurement-uncertainty interval overlap coverage, separately reported;
- mean normalized prediction-interval width.

The assessment profile freezes quantity ID, unit, service boundary, averaging window, normalization scale, nominal coverage, interval-generation method and its assumption reference, error limits, minimum coverage and maximum interval width. It also requires minimum holdout count and facility/operating-point diversity. Prediction values bind their exact model run/output digest; measurement values bind the exact measured holdout artifact. Unseal/model-run receipts cannot be reused between holdout cases.

A report only means the finite campaign met the declared profile. Empirical point coverage is descriptive; the module does not attach a confidence interval, assume independent samples, fit a model or calibrate prediction bands. Measurement-interval overlap is not substituted for point coverage. Wide intervals are penalized through the mean normalized width ceiling.

The module has 27 authored tests covering known-answer metric calculations, split leakage, stale model and output identity, reused receipts, incomplete observables, unit/boundary/time-window mismatch, malformed intervals, artifact provenance, sample-diversity gates, point versus measurement-interval coverage, order-invariant output, out-of-domain refusal, pooled-masking adversaries and predeclared slice integrity. The synthetic runner uses three holdout cases across two facilities and three operating points; all values and the expected acceptance profile are synthetic.

### External validation benchmark alignment

The DOE's June 2026 roadmap calls for an end-to-end plant-model workflow against experimental data with quantified uncertainty, independent reproducibility, an enforced validity domain and holdout validation at a different operating point/case/facility while maintaining uncertainty bounds. It also calls for model comparisons against coupled experiments. This module adds the predictive-metric side of that requirement while keeping specialist experiment, metrology and physics-model ownership external.

A July 2026 *Fusion Science and Technology* article on integral nuclear-data benchmarking emphasizes measured integral responses representative of the materials/radiation/temperature conditions and rigorous benchmark models with experimental uncertainties. A 2021 multiscale fusion VVUQ study explores distribution-similarity validation; this implementation uses transparent point errors and interval metrics as a reproducible v0.1 baseline, not as a claim to supersede specialist validation methods.



## Facility and operating-point worst-slice validation

The next layer keeps the pooled campaign metrics but also reports metrics across categorical slice dimensions frozen in the assessment profile before holdout unsealing. V0.1 supports facility, case ID, operating point, regime and geometry identity. It does not invent bins after observing residuals.

For every predeclared dimension/value and observable, the report includes sample count, bias, MAE, RMSE, normalized bias/RMSE, max normalized error, point interval coverage, measurement interval overlap and normalized interval width. Underfilled groups remain visible with `InsufficientSliceEvidence`; they are never silently dropped or pooled away.

The campaign now reports a separate slice disposition. A sufficiently sampled slice that fails any declared observable threshold makes the overall assessment `OutsideDeclaredAcceptance`, even when the pooled metrics pass. An underfilled slice makes it `InsufficientHoldoutEvidence`. A deterministic worst-slice register identifies the worst facility/case/operating-point/regime/geometry for normalized RMSE, absolute normalized bias, maximum normalized residual, point coverage and interval width; exact ties are returned in sorted order.

Four additional authored tests demonstrate:
- a synthetic pooled campaign that meets aggregate thresholds but fails one facility slice;
- deterministic worst-slice identification;
- insufficient per-slice sample count remaining explicit;
- duplicate/empty slice dimensions being rejected.

The runner emits each facility and operating-point slice and its worst-slice findings. These remain synthetic software-contract examples, not independent statistical or physical validation.


## Run locally

From this directory:

```sh
cargo fmt --manifest-path Cargo.toml -- --check
cargo test --manifest-path Cargo.toml
cargo run --manifest-path Cargo.toml --bin verify_fecomp_corpus
cargo run --manifest-path Cargo.toml --bin verify_validation_domain_corpus
cargo run --manifest-path Cargo.toml --bin compare_solver_outputs
cargo run --manifest-path Cargo.toml --bin qualify_plant_service
cargo run --manifest-path Cargo.toml --bin audit_validation_assessment
cargo clippy --manifest-path Cargo.toml --all-targets -- -D warnings
```

The independent compiler corpus binary checks fixture shape and required case identities without calling the compiler implementation. The current compiler corpus has 26 rows: 5 expected review-eligible, 7 blocked, 6 indeterminate and 8 schema-rejected. The compiler core has 32 authored Rust unit tests, including one that applies all 26 compiler corpus mutations to the actual compiler. The observable-dependence module adds 11 authored tests for per-output dependency overlap, source aliases, incomplete inventories, conflicting identities and deterministic ordering. The candidate-funnel module adds 16 authored tests for full stage matrices, missing/duplicate records, skip reasons, pipeline transitions, explicit audit overrides, failure/non-convergence and exact-vs-partial recall. The validation-domain module adds 16 authored tests for calibration/test/holdout separation, source and episode overlap, leakage, measurement class, metrology provenance, model revision and domain applicability; it also executes all 16 frozen split-integrity cases. The cross-solver comparison module adds 15 authored tests for matched-case output admission, normalization identity, run-status refusal, dependency disclosure, and covariance-receipt binding. Total authored Rust test declarations across the current five source modules: 90. These are authored test cases, not executed test results.

The separate validation-domain inventory binary checks fixture structure without calling the split validator. Its 16 synthetic cases contain 3 expected acceptances and 13 expected refusals.

The `compare_solver_outputs` binary is an executable synthetic matched-case example. It compares a declared PROCESS-like output (1200 MW) against a D0FUS-like output (1185 MW) for one frozen candidate and service boundary. The expected disposition is descriptive-only because cross-covariance is deliberately unknown. The example verifies the output comparison and refuses to manufacture inferential significance or a physical-validation claim. These values are synthetic fixtures, not PROCESS or D0FUS solver runs.

## Claim ceiling

A pass establishes only that this exact software contract behaves as tested on synthetic fixtures. It does not validate a solver, equilibrium, reactor architecture, material, component, manufacturing process, plant economics, physical experiment, safety case or net-energy claim. It cannot authorize experiments, hardware, fabrication, commissioning or operation.
