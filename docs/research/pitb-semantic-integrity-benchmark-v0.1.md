# PITB Semantic Integrity Benchmark (SIB) v0.1

**Status:** benchmark design proposal; no SIB tasks have been executed or scored.  
**Priority:** P0 adversarial integrity extension to Aegis, not a replacement benchmark.  
**Tracking:** [PITB Aegis integration #10272](https://github.com/Luminous-Dynamics/symthaea/issues/10272)

## 1. Research question

Can a software-reasoning system distinguish a valid proof of the intended implementation contract from a valid proof of a statement that is stale, vacuous, weakened, mistranslated, or unrelated to the implementation?

Conventional proof success is insufficient for this question. SIB evaluates the full claim chain:

Rust source -> extraction/translation -> formal semantics -> specification -> theorem -> proof/refutation -> independent replay.

SIB is intended to expose failure modes that a theorem kernel cannot detect on its own, because the kernel checks the theorem it receives, not whether that theorem faithfully represents the original software requirement.

## 2. Threat model

The benchmark includes controlled, labeled perturbations. Each perturbation is generated from a frozen parent task and records the exact transformation and affected artifact digest.

### Source mutations
- Change integer width or signedness.
- Change overflow-sensitive expressions or cast behavior.
- Change loop bounds, branch predicates, or termination conditions.
- Change failure/panic behavior.
- Change mutation order or state-update semantics.
- Change preconditions or externally visible behavior.

### Translation mutations
- Substitute a stale generated Lean file.
- Change translator version/configuration without updating provenance.
- Alter a translated operation while retaining the original source identity.
- Omit or alter a dependency/import.

### Specification mutations
- Weaken a postcondition until it becomes vacuous.
- Strengthen a precondition to exclude all meaningful inputs.
- Change a boundary condition or failure case.
- Swap program/specification identifiers.
- Retain a theorem statement after source changes.
- Replace a required behavioral property with a weaker unrelated property.

### Evidence mutations
- Alter proof bytes or proof digest.
- Remove replay evidence.
- Reuse a receipt from another task.
- Change a toolchain identity or dependency digest.
- Duplicate or omit a task from a cohort.

All mutation operators must be deterministic, versioned, and reproducible. Mutated examples must not be mixed into the official Aegis score unless its maintainers explicitly define them as part of that benchmark.

## 3. Task classes and expected response

Each task has a hidden mutation label and an expected action:

| Class | Expected response |
|---|---|
| Unmodified valid task | Prove and pass all integrity gates |
| False implementation claim | Refute or return a sound counterexample |
| Source/Lean mismatch | Detect translation disagreement |
| Stale or mismatched contract | Detect specification disagreement |
| Vacuous/overconstrained contract | Flag specification integrity failure |
| Tampered or stale evidence | Reject receipt |
| Unsupported translation construct | Report unsupported/unknown; do not silently claim verification |
| Insufficient evidence | Abstain / unresolved |

A system must not be rewarded for guessing a mutation label without a supporting witness or reproducible diagnostic artifact.

## 4. Metrics

Report metrics separately; do not collapse them into a single score:

- **Valid-task verified solve rate:** valid tasks proved with all gates passed / valid tasks.
- **False acceptance rate:** mutated tasks incorrectly accepted as verified.
- **Mutation detection recall:** mutated tasks correctly detected / mutated tasks.
- **Mutation detection precision:** correctly detected mutations / all mutation detections.
- **Abstention quality:** unresolved/unknown decisions on genuinely unsupported or underdetermined tasks.
- **Localization accuracy:** correct fault layer identified (source, translation, specification, proof, evidence).
- **Witness validity:** independently reproduced counterexamples/diagnostics / submitted witnesses.
- **Replay reproducibility:** accepted proof receipts independently reproduced / accepted receipts.
- **Coverage:** results by mutation operator, language construct, task stratum, and source program.

Report per-operator confusion matrices and confidence intervals. Cluster uncertainty estimates by parent program because multiple mutations of one program are not independent samples.

## 5. Anti-gaming rules

- Keep a hidden test partition and publish mutation-generator versions only after a frozen evaluation window.
- Use multiple semantically distinct mutation operators for each category.
- Include benign transformations as controls so systems cannot label every changed digest as a semantic defect.
- Separate byte-integrity detection from semantic-mismatch detection.
- Require witnesses for claims of semantic mismatch.
- Require kernel replay for proof acceptance.
- Require exact task and artifact identities in every result.
- Keep unknown/unsupported distinct from false and from proved.
- Never let a model-generated explanation substitute for a verifier result.

## 6. Research hypotheses

H1: source-derived invariants and dependency-aware proof decomposition improve valid-task resolution over matched tactic-only search.

H2: a separate semantic-integrity gate reduces false acceptance compared with proof-kernel-only acceptance.

H3: counterexample-guided diagnosis improves fault localization and reduces wasted proof-search budget.

H4: typed proof insight transfer improves solve rate per unit cost when preconditions and dependency closures are enforced.

H5: explicit unsupported-construct tracking reduces false confidence on translation-boundary cases.

These hypotheses require paired experiments and preregistered ablations. They are not assumed to be true.

## 7. Initial acceptance targets (prospective)

- 100% independent replay of accepted proof artifacts.
- 100% rejection of seeded byte-integrity and receipt-reuse controls.
- Zero false acceptance on the initial curated semantic mutation suite before expansion to hidden adversarial tasks.
- At least 90% mutation detection recall on each sufficiently powered, published mutation family; report precision and confidence intervals alongside recall.
- No increase in false acceptance when enabling proof retrieval, parallel search, or adaptive scheduling.
- Valid-task solve rate reported separately from all mutation-detection metrics.

The zero-false-acceptance target is a safety gate, not a statistically justified claim about all possible defects. Report sample size and an upper confidence bound; never state that a finite test suite proves zero risk.

## 8. Relationship to Aegis and PITB

- Aegis remains the external real-code proving comparator.
- SIB is an adversarial integrity evaluation layered around selected Aegis-derived tasks and separately authored controls.
- SorryDB remains a proof retrieval/transfer benchmark.
- VeriSoftBench remains a repository-scale context/dependency comparator.
- Mathematical benchmarks remain regression controls.

Publish independent scorecards and a multi-axis capability profile. Do not average these tasks into a leaderboard scalar.

## 9. Toolchain coverage ledger

Aeneas is a translation toolchain for safe Rust, not a proof that arbitrary Rust constructs are fully represented. For every task, record the exact Rust constructs encountered and the translator's documented support status for the pinned version.

At minimum classify:
- supported and translated;
- translated with explicit modeling assumptions;
- unsupported;
- excluded by dataset construction;
- unknown / not audited.

The documentation currently identifies limitations including nested loops, function pointers/closures, some type-parametricity cases, nested borrows in signatures, interior mutability, and concurrency. These limitations are version-sensitive: pin the exact Aeneas release and re-audit its documentation/source before each evaluation. Aegis-generated loop obligations must be tested against the exact pipeline rather than inferred from a general limitation statement.

## 10. Execution plan

1. Build deterministic mutation operators over a tiny, manually audited Rust corpus.
2. Generate source/translation/specification/evidence variants and preserve parent-child provenance.
3. Validate each operator with an independent reference checker and human review.
4. Add positive benign-transformation controls.
5. Connect task identity and receipts to the Aegis manifest format.
6. Run proof-only versus proof-plus-integrity-gate ablations.
7. Freeze a hidden partition and publish the evaluation harness only after the first reproducible release.

No benchmark result should be published until task labels, mutation operators, expected outcomes, and independent witnesses have been validated.
