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


## Preregistered measured test-stand and blind-holdout metrics

The `measured_holdout_metrics` module composes with `validation_domain.rs` rather than duplicating split-integrity checks. A pre-reveal prediction commitment binds model revision/configuration, input/output manifests, observable profile, record IDs, artifact digests and expected sample counts. Measurement verification receipts are separate for test-stand and blind-holdout payloads.

Each successful report retains the exact `ValidationDomainAudit` returned by the existing domain gate, including model/domain/use identity, calibration counts, held-out record IDs, distinct holdout dimensions, admissibility and claim ceiling. If the domain gate fails, the report carries no successful audit and no prediction metrics are computed.

The synthetic runner uses fabricated samples and declared verifier placeholders. It must remain `DescriptiveOnly`; numerical residuals from synthetic observations are not measured validation. Empirical interval coverage remains descriptive and is not a confidence guarantee.


## Run locally

From this directory:

```sh
cargo fmt --manifest-path Cargo.toml -- --check
cargo test --manifest-path Cargo.toml
cargo run --manifest-path Cargo.toml --bin verify_fecomp_corpus
cargo run --manifest-path Cargo.toml --bin verify_validation_domain_corpus
cargo run --manifest-path Cargo.toml --bin compare_solver_outputs
cargo run --manifest-path Cargo.toml --bin qualify_plant_service
cargo run --manifest-path Cargo.toml --bin audit_measured_holdout
cargo clippy --manifest-path Cargo.toml --all-targets -- -D warnings
```

The independent compiler corpus binary checks fixture shape and required case identities without calling the compiler implementation. The current compiler corpus has 26 rows: 5 expected review-eligible, 7 blocked, 6 indeterminate and 8 schema-rejected. The compiler core has 32 authored Rust unit tests, including one that applies all 26 compiler corpus mutations to the actual compiler. The observable-dependence module adds 11 authored tests for per-output dependency overlap, source aliases, incomplete inventories, conflicting identities and deterministic ordering. The candidate-funnel module adds 16 authored tests for full stage matrices, missing/duplicate records, skip reasons, pipeline transitions, explicit audit overrides, failure/non-convergence and exact-vs-partial recall. The validation-domain module adds 16 authored tests for calibration/test/holdout separation, source and episode overlap, leakage, measurement class, metrology provenance, model revision and domain applicability; it also executes all 16 frozen split-integrity cases. The cross-solver comparison module adds 15 authored tests for matched-case output admission, normalization identity, run-status refusal, dependency disclosure, and covariance-receipt binding. Total authored Rust test declarations across the current five source modules: 90. These are authored test cases, not executed test results.

The separate validation-domain inventory binary checks fixture structure without calling the split validator. Its 16 synthetic cases contain 3 expected acceptances and 13 expected refusals.

The `compare_solver_outputs` binary is an executable synthetic matched-case example. It compares a declared PROCESS-like output (1200 MW) against a D0FUS-like output (1185 MW) for one frozen candidate and service boundary. The expected disposition is descriptive-only because cross-covariance is deliberately unknown. The example verifies the output comparison and refuses to manufacture inferential significance or a physical-validation claim. These values are synthetic fixtures, not PROCESS or D0FUS solver runs.

## Claim ceiling

A pass establishes only that this exact software contract behaves as tested on synthetic fixtures. It does not validate a solver, equilibrium, reactor architecture, material, component, manufacturing process, plant economics, physical experiment, safety case or net-energy claim. It cannot authorize experiments, hardware, fabrication, commissioning or operation.
