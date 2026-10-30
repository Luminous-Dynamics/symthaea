# Luminous Dynamics DSL Portfolio RFC v1

**Status:** Proposed; cross-repository design for review  
**Date:** 2026-10-29  
**Scope:** Mycelix, Symtropy, Symthaea, Sol-Atlas, and Nixward  
**Implementation principle:** Extend existing typed Rust and schema infrastructure before adding a new parser or runtime.

## 1. Decision summary

Build a small portfolio of domain-specific specifications around an interoperable evidence profile carried by established attestation formats. Do **not** build a universal Luminous language or invent a competing envelope/signature format.

The first implementation should use:

- Rust domain types and Serde for in-process validation.
- JSON Schema Draft 2020-12 for portable structural contracts and fixtures.
- TOML for hand-authored local scenarios where its semantics fit; JSON for interoperable records and exchange.
- A versioned intermediate representation (IR) shared at the data-contract level, not a shared domain semantics layer.
- Canonical serialization for digesting only after choosing and testing one precise canonicalization rule. RFC 8785/JCS is a candidate; its number model and I-JSON restrictions must be respected.
- Domain validators that produce structured, machine-readable diagnostics.
- A CLI and CI conformance suite that can run locally without remote runners.

No DSL may silently grant authority, certify a scientific claim, or turn a simulated outcome into an observed fact.

## 2. Repository evidence and existing assets

This RFC is based on a focused source inventory, not a complete audit of every repository.

### Symthaea

Existing assets include:

- `benches/BENCHMARK_GUIDE.md` and a broad benchmark inventory.
- `api/service-protocol-v1.schema.json`, already using JSON Schema 2020-12.
- `crates/domains/symthaea-quantum-comp/src/benchmark.rs`, with a typed `BenchmarkManifest`.
- `crates/bridges/symthaea-proof-audit/src/policy.rs`, which already audits a declared axiom base.
- `crates/core/symthaea-evidence-plane`, which checks agreement between declared mechanisms and measured counters, and owns a deliberately non-cryptographic `config_hash`.
- `crates/core/symthaea-communication/src/benchmark.rs`, which already defines benchmark manifests and dataset split-separation checks.
- Existing evidence-focused workflows and validation facilities.
- Open follow-up work on predicate-scoped evidence identity (#6797) and non-scalar evidence resolution (#2617); LEP must interoperate with these rather than create a competing resolver.

**Implication:** Do not create a second benchmark manifest or parallel schema framework. Inventory and harmonize existing formats first. A new experiment DSL should compose the evidence-plane integrity checks and benchmark records rather than replace them. Never promote `config_hash` to a cryptographic digest.

### Symtropy

Existing assets include:

- `crates/bridges/symtropy-mycelix-bridge/src/scenarios.rs`, with a typed `ScenarioConfig` and declarative scenario harness.
- `crates/core/symtropy-physics/RESEARCH_VALIDATION.md`, which distinguishes local mathematical contracts, analytical references, and stronger validation tiers.
- Deterministic replay, validation examples, and physics tests.
- Existing open work already covers sealed scenario manifests and evidence identity (Symtropy #1202), plus a deterministic engineering scenario/replay substrate (Symtropy #1172).

**Implication:** Do not create a competing scenario-identity project. Reconcile the DSL with #1202 and #1172, reusing sealed manifest/replay work and adding typed physical semantics where missing. Do not replace the physics engine or invent a general equation language.

### Mycelix

The connected `Mycelix-Core` repository includes federated learning, Holochain integration, governance material, and claim-to-evidence documentation, but the repository's current scope is broader and not identical to a constitutional policy system. The older `mycelix-prototype` README describes a different prototype.

**Implication:** First establish the authoritative current repository and integrity/execution boundaries for each policy. Implement policy rules only where their actual enforcement point is known. Never infer production guarantees from roadmap documents or from the existence of a policy schema.

### Sol-Atlas

Existing assets include:

- `tools/workforce-active-evidence-verifier`, which validates a versioned scenario corpus using JSON Schema, bounded search, and hand-reference frontiers.
- Energy-trading and visual-validation code.

**Implication:** Extend the existing schema-plus-verifier pattern. A new energy DSL should express dispatch inputs, objectives, and hard constraints without granting actuator authority.

## 3. Portfolio: what to build, in order

| Priority | DSL/specification | Home | What it describes | Why it matters |
| --- | --- | --- | --- | --- |
| P0 | **Luminous Evidence Profile (LEP)** — a versioned predicate profile over in-toto Statement v1, not a custom envelope | Shared contract package/repository after placement review | Claim status, inputs, code revision, configuration/data digests, environment, validation runs, artifacts, limitations | Prevents incompatible and overstated evidence across projects |
| P0 | **Mycelix Policy / Authority DSL** | Authoritative Mycelix governance and integrity code | Principal, action, resource, scope, quorum/approvals, expiry, revocation, appeal, deny rules, explanation | Makes authority explicit and enables adversarial conformance testing |
| P1 | **Symtropy Scenario / Physical Model DSL** | Symtropy | Model selection, quantities/units, initial and boundary conditions, solver, timestep, tolerances, parameter provenance, validation tier, uncertainty | Makes simulations reproducible and physically checkable |
| P1 | **Symthaea Experiment DSL** | Symthaea | Hypotheses, tasks, baselines, datasets/splits, seeds, metrics, ablations, budgets, statistical plan | Prevents apples-to-oranges benchmark comparisons |
| P1 | **Sol-Atlas Energy / Resilience DSL** | Sol-Atlas | Assets, loads, tariffs, PV, storage, grid limits, contingencies, objectives, hard constraints, forecast uncertainty | Turns energy plans into reviewable, testable decision candidates |
| P2 | **Nixward ChangePlan DSL** | Nixward | Proposed system change, allowed capabilities, preconditions, authorization, rollback, health checks, expected diff | Constrains high-impact host changes and makes rollback auditable |
| P2 | **Manufacturing / Process Recipe DSL** | Symtropy plus domain-owned validation | Materials, process stages, machine envelope, thermal/flow settings, quality checks and traceability | Connects simulation predictions to process constraints without treating predictions as measurements |

This is a portfolio of **seven named artifacts** but only the first three domain languages should be actively built initially. LEP is a shared contract, not a universal DSL.

## 4. Shared evidence contract: minimum fields

The LEP predicate should use explicit, versioned fields. Domain-specific metadata can extend it through namespaced extension objects, not arbitrary changes to shared meanings.

Required concepts:

1. **Identity:** schema URI/version, stable artifact ID, claim ID, producer and timestamp.
2. **Target:** source repository, full immutable source revision, build/toolchain identity where relevant.
3. **Inputs:** canonical configuration digest, dataset/artifact digests, input provenance, random seed or an explicit statement that no seed applies.
4. **Claim semantics:** one controlled status from `implemented`, `tested`, `measured`, `simulated`, `inferred`, `hypothesized`, `normative`, or `aspirational`; keep normative commitments distinct from empirical claims.
5. **Validation:** checks requested, checks actually executed, verifier name/version, exact target digest, raw result references, pass/fail/not-run status.
6. **Limitations:** validity domain, assumptions, uncertainty, missing evidence, known exclusions, and allowed claim ceiling.
7. **Integrity:** digest algorithm, canonicalization profile/version, artifact digests, optional signature envelope and signer identity.

Distinguish these statements explicitly:

- A receipt is authentic.
- An event occurred.
- A test passed against an exact target.
- A measured or simulated result supports a claim.
- The claim generalizes beyond that test.

They are not interchangeable. A hash or signature binds data; it does not prove the scientific or normative truth of the data.

Avoid a permissive field like `verified: true`. Verification must be a structured result naming **what** was checked, against **which exact target**, by **which verifier**, and with **what result**. `not_run`, `queued`, missing, stale target, or digest mismatch must never be converted into pass.

## 5. Language and tooling choices

### Mycelix: compare Cedar before inventing a policy grammar

Prototype the same small authorization policy set in:
- Typed Rust policy structures and an explicit evaluator.
- Cedar policies plus a schema, using the Rust `cedar-policy` crate.
- A narrow OPA/Rego comparison for policy query expressiveness and operational integration.

Use the same request corpus, policy truth table, mutations, and performance harness for all candidates. Cedar and Rego should not be compared on unrelated features.

Cedar is purpose-built for authorization and supports schema-based validation. It is a strong comparison candidate, not an automatic production choice. Decide based on deployment target/WASM compatibility, explainability, policy migration, latency, and the ability to enforce non-overridable constitutional invariants.

Keep constitutional invariants in mandatory validation/enforcement code so an editable policy cannot disable the invariant it is supposed to obey. Unknown actions and malformed requests must deny by default. Authorization must be checked at the actual operation boundary, not only in a UI or planning layer.

### Symtropy: typed model registry plus unit-safe quantities

Use a compact scenario manifest that names registered models and parameter types. Represent units explicitly at the boundary and convert into typed quantities internally. Evaluate Rust's `uom` as a candidate for dimensional analysis rather than inventing unit arithmetic. For co-simulation, investigate FMI 3.0.2/FMU interoperability before designing proprietary model-exchange formats; FMI is an interface standard, not a certificate of physical correctness.

Never accept a physically meaningful bare number where a required unit is missing or ambiguous. Distinguish measured, calibrated, assumed, and hypothesized parameters. Each scenario must declare its validation tier; a rendered scene is not validation.

### Symthaea: a benchmark manifest compiler, not a benchmark framework rewrite

Define an experiment IR and compile it to existing benchmark harnesses. Require baseline identity, task/data split, seeds or repeated-run plan, metric direction and aggregation, hardware/software context, resource budgets, ablation plan, and raw results. Experimental claims should include uncertainty and a predeclared claim ceiling.

Do not treat Phi, an internal proxy, or one selected task as a universal measure of cognition. Each metric needs an operational definition and validity limits.

### LEP: stay interoperable and portable

Begin with versioned JSON Schema plus Rust types and conformance fixtures. CUE is a candidate for cross-field constraints and configuration generation if schemas become difficult to maintain; measure the extra toolchain burden first. Do not require CUE or Dhall at runtime. Use JSON Schema for structural contracts, typed Rust checks for domain semantics, and a single canonicalization profile for content hashing.

RFC 8785/JCS may be evaluated for canonical JSON hashing, but implementations must conform to its number/I-JSON constraints. Test canonicalization with cross-implementation vectors before using it for signatures or attestations.

## 6. Conformance architecture

Each DSL should follow the same pipeline while retaining domain-specific checks:

`source spec -> parse -> schema validation -> typed IR -> semantic validation -> plan -> execution -> raw evidence -> independent verifier`

Minimum outputs:

- A machine-readable validation report with stable error codes and JSON pointers/field paths.
- A human-readable explanation of every rejected requirement.
- Generated positive and negative test fixtures.
- An execution plan separated from authorization to execute.
- A result record that binds source revision, inputs, settings, verifier, and raw artifacts.
- A reproducibility command that works locally without waiting for hosted CI.

The independent verifier must not simply call the same code path as the producer for the property it is meant to verify. Where full independence is impractical, document shared code and residual common-mode risk.

## 7. Benchmarks and release gates

These are proposed engineering targets, not claims about current performance.

### Cross-cutting contract

- 100% of published records validate against the declared schema version.
- 100% of required negative fixtures reject (missing required fields, unknown enum values, malformed digests, stale target SHA, invalid status transitions, unsupported schema versions).
- Equivalent inputs yield the same canonical digest under the chosen profile.
- A one-field mutation changes the digest and invalidates the bound verification result.
- Zero conversion of missing, queued, stale, or failed checks into a positive verification status.
- Every consequential claim has a target revision, data/config identity, evidence status, and limitation/claim ceiling.

### Mycelix policy DSL

- Exhaustive action/principal/resource matrix for each declared policy fixture.
- 100% deny for unknown actions, expired delegation, revoked authority, malformed scope, and absent mandatory approvals.
- Property tests for least privilege and non-escalation across policy edits.
- Mutation testing target: at least 95% of seeded policy mutations detected by tests, with every constitutional invariant mutation detected.
- Decision explanations agree with the actual evaluator result for 100% of conformance fixtures.
- No privilege-bearing action executes when policy evaluation or required evidence is unavailable.

### Symtropy scenario DSL

- 100% rejection of unknown units, incompatible dimensions, missing boundary conditions required by a model, and invalid solver/timestep settings.
- Tier B tests compare against analytical references with absolute and relative error reported versus timestep.
- Tier C tests compare with independent implementations/reference data where available.
- Report invariant drift (energy, momentum, constraint residuals) only for systems where the invariant is mathematically applicable.
- Record numerical stability, convergence under timestep refinement, and uncertainty for every research-grade scenario.
- Do not claim cross-architecture bit-identical replay unless measured and continuously tested.

### Symthaea experiment DSL

- 100% required-field completion for a release-grade experiment manifest.
- Detect intentional train/test contamination and mismatched baseline resource budgets in negative fixtures.
- Every metric has an operational definition, direction, aggregation rule, and raw value trace.
- Paired-seed or repeated-run reports include confidence intervals where applicable and disclose exclusions.
- Report latency percentiles, peak memory, throughput and energy per successful task where measurable; distinguish unavailable from zero.
- Any proposed improvement must beat the baseline on a preregistered metric without violating quality/safety constraints.

### Sol-Atlas energy DSL

- Hard constraints are not traded away for a better objective score.
- 100% rejection of infeasible capacity, invalid units, contradictory limits, and missing safety-critical inputs.
- Benchmark feasible-plan rate, constraint violations, unserved load, curtailment, operating cost, resilience under explicit contingencies, and sensitivity to forecast error.
- Outputs are proposals only until a separately authorized execution system approves them.

### Performance and adoption

Measure parse/validation time, memory, diagnostics quality, schema migration effort, dependency/license burden, and operator comprehension on the same fixtures. Do not claim superiority over CUE, Cedar, OPA/Rego, or other established tooling until the same workloads and negative fixtures have been run against those baselines.

## 8. Adversarial test catalogue

Every DSL needs a stable, version-controlled conformance corpus with at least:

- Valid minimal and maximal specifications.
- Missing field, unknown field/enum, wrong type, range violation, and unsupported version.
- Conflicting constraints and unsatisfiable requirements.
- Stale source revision and stale artifact digest.
- Tampered raw results and changed configuration.
- Invalid units or parameter provenance (Symtropy).
- Unauthorized action, expired/revoked delegation, and policy shadowing (Mycelix).
- Data leakage, unpaired baselines, seed mismatch, missing raw scores, and post-hoc metric changes (Symthaea).
- Infeasible constraints, forecast perturbation, and objective gaming (Sol-Atlas).
- Timeout, resource exhaustion, and verifier failure.

Use mutation/property-based tests and independent hand-authored expected outcomes. A test suite that merely round-trips its own serializer is insufficient.

## 9. Implementation sequence

### P0 — Contract inventory and threat model

1. Inventory all current schemas, evidence records, manifests, and provenance envelopes across the target repositories.
2. Identify duplicate fields and incompatible definitions before proposing a common version.
3. Draw the authorization/evidence boundary for Mycelix and mark non-overridable invariants.
4. Write valid and invalid LEP fixtures, with exact expected errors and claim-status transition rules.
5. Select the canonicalization profile and test vectors; do not hash ad hoc serialized structs.

### P0 — First working vertical slice

**Prototype source added; release readiness remains unverified.** The isolated Rust tool now has strict duplicate-key JSON parsing, bundled Draft 2020-12 predicate-schema validation, semantic target/status checks, a once-per-process compiled-schema cache, fixture-backed unit tests, and an executable machine-readable conformance manifest. The `lep-benchmark` binary measures parsing + structural validation separately from parsing + structural + semantic validation on synthetic inputs; it has no measured result yet. A standalone lockfile and targeted GitHub Actions workflow are checked in. The first hosted workflow attempt failed at `cargo fmt --check` before compilation; later attempts were canceled by subsequent head updates before tests ran. This environment has no Rust toolchain, so no compiler/test pass is claimed. The next gate is a clean format, compile, test, Clippy, and benchmark-smoke run on one fixed head. Artifact-byte resolution, DSSE/signature verification, signer authorization, and cross-implementation conformance remain out of scope. See `LEP_CONFORMANCE_BENCHMARK_PLAN.md` for the industry comparison and measurable release gates.

### P1 — Domain pilots

1. Mycelix: prove the policy subset and compare Cedar vs typed Rust against the same adversarial matrix.
2. Symtropy: serialize the existing `ScenarioConfig` pattern, then add unit/model/solver validation and reproducibility metadata.
3. Symthaea: define the experiment IR and migrate one existing benchmark suite without changing its scoring or execution semantics.
4. Sol-Atlas: apply the envelope to one existing evidence corpus and one energy-planning slice.

### P2 — Expand only after evidence of value

Add Nixward ChangePlan and manufacturing recipes only after the first pilots show that declarative plans reduce errors and improve auditability. Do not add a custom parser unless users can demonstrate real limitations in the schema-backed representation.

## 10. Acceptance criteria for this RFC's first implementation

- Inventory identifies each source schema/manifest and the owner of its semantics.
- No duplicate evidence format is introduced without a documented migration reason.
- Schema, semantic validator, fixtures, and CLI have offline tests.
- All negative fixtures are checked into version control with expected diagnostics.
- The emitted evidence record distinguishes event provenance, test outcome, measured/simulated evidence, and claim level.
- The implementation can be reviewed and tested without waiting for remote runners.
- No existing enforcement gate is weakened and no unimplemented capability is described as validated.


## Amendment 1 — interoperability-first design (2026-10-29)

This amendment sharpens the original proposal after comparing the repository's existing contracts with current standards. The portfolio should reuse standards wherever they already solve the hard interoperability problem.

### A. LEP is a predicate profile, not a new envelope

Use the **in-toto Statement v1** as the outer statement shape: `_type`, immutable `subject` digest(s), `predicateType`, and a versioned `predicate`. LEP defines only the predicate semantics that existing projects do not already share. Reuse the established in-toto envelope for signing/authentication rather than inventing a Luminous signature envelope.

Use **SLSA Provenance v1.2** for software build provenance where it fits; the current official SLSA specification index identifies v1.2 as current. Pin the actual provenance specification version in build metadata and records. Do not force it to represent a physical simulation result or a governance decision; those require domain-specific predicate types. The evidence profile must distinguish:
- an authentic statement;
- a claim about an event;
- a test or verification outcome against an exact target;
- measured or simulated support for a claim; and
- whether that support justifies the claim's stated scope.

Existing `symthaea-evidence-plane` and `symthaea-communication` records remain domain/runtime interfaces. LEP should map to them through adapters rather than replace them. The evidence-plane crate's `config_hash` is explicitly non-cryptographic and unsuitable for signing or adversarial integrity checks.

### B. First schema and fixture prototype

A first draft predicate schema lives at `docs/research/schemas/luminous-evidence-predicate-v0.1.schema.json`; examples live in `docs/research/fixtures/luminous-evidence-v0.1/`.

The schema enforces required target identity, version and enumerated states, digest shape for common algorithms, and the rule that a verification cannot be marked passed unless execution is completed and passed. A queued or not-run record cannot claim a passed execution. Its structural rules are tested separately from semantic target binding so a schema-valid stale target remains a meaningful adversarial case.

The predicate spec deliberately documents a stricter unknown-field rule than generic in-toto parsing: unknown fields in closed LEP core objects are rejected; extensions must use the explicit `extensions` object with namespaced HTTPS keys. This matches the schema's `additionalProperties: false` checks and makes each version's accepted surface explicit. Future core fields require a reviewed schema/version change and cross-implementation conformance tests, not silent acceptance.

A Rust conformance tool now exists in `tools/lep-conformance` with an offline CLI, bundled Draft 2020-12 predicate-schema validation, duplicate-key-rejecting JSON parsing, exact structured target/checked-target equality, immutable source-revision enforcement, supported digest-shape checks, execution/result consistency, and check outcome/id consistency. The JSON conformance manifest is executable from Rust tests and now contains positive, negative, parser, unknown-field, and target-drift mutation cases. The compiled schema is cached once per process. The separate `lep-benchmark` binary measures parsing + structural validation against parsing + structural + semantic validation on synthetic records. It is a measurement harness, not yet a measured performance result. The Rust test source also contains a deterministic 1,000-record adversarial mutation suite spanning ten invalidity classes; it has not yet been executed successfully.

**Scope and limitations:** example digests are placeholders, and these fixtures are not production run receipts. The Rust implementation has not yet completed a successful compile/test run: this authoring environment has no Rust toolchain; the first hosted workflow attempt failed at formatting before compilation; subsequent attempts were canceled by later commits before tests ran. No passing Rust or hosted CI result is claimed. The earlier independent Python JSON Schema check and strict-object-pairs check verified the initial schema/fixture set only; they do not verify the Rust implementation or replace rechecking the updated manifest. A standalone Cargo.lock is checked in, but its extracted transitive closure has not been validated by Cargo here, so `cargo test --locked` must confirm it. The tool still does not authenticate in-toto/DSSE signatures, resolve artifact URIs, recompute artifact bytes, or prove that a declared run occurred. Neither schema conformance nor semantic consistency proves a scientific claim true. The measurable comparison and acceptance gates are specified in `LEP_CONFORMANCE_BENCHMARK_PLAN.md`.


### C. Physical-model interoperability before a proprietary equation language

For Symtropy, the domain DSL should primarily orchestrate typed registered models, physical quantities, conditions, solver configuration, uncertainty and acceptance tests. FMI 3.0.2 offers established Model Exchange, Co-Simulation and Scheduled Execution interfaces. Evaluate FMU import/export where useful rather than wrapping every model in a new proprietary equation syntax. Keep Symtropy's native Rust APIs for capabilities and performance they already support. The validity tier and error against analytical/independent references remain mandatory regardless of transport format.

### D. Fair industry comparison

Benchmark candidates using the same fixtures and equivalent semantics:
- **Data constraints:** JSON Schema 2020-12 vs CUE for cross-field validation, diagnostics, migration and runtime/tooling cost.
- **Authorization:** Cedar vs a typed Rust evaluator, with a narrow OPA/Rego baseline where relevant. Compare correctness, mutation detection, explainability, latency and WASM/deployment constraints.
- **Provenance:** in-toto Statements and SLSA provenance for software artifacts, plus narrowly scoped domain predicates where their semantics differ.
- **Simulation integration:** native Symtropy scenario manifests vs FMI/FMU interoperability for exchange and co-simulation. Do not count standards compatibility as physical accuracy.
- **Scientific work:** extend the existing benchmark-manifest, dataset-split and evidence-plane mechanisms; don't rewrite functioning research harnesses just to fit a new syntax.

No state-of-the-art claim should be made until the baseline tools are run on pinned versions with the same positive/negative corpus and resource limits.

## References

- Existing Symthaea benchmark guide: https://github.com/Luminous-Dynamics/symthaea/blob/main/benches/BENCHMARK_GUIDE.md
- Symthaea service schema: https://github.com/Luminous-Dynamics/symthaea/blob/main/api/service-protocol-v1.schema.json
- Symthaea proof audit policy: https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/bridges/symthaea-proof-audit/src/policy.rs
- Symtropy scenario harness: https://github.com/Luminous-Dynamics/symtropy/blob/main/crates/bridges/symtropy-mycelix-bridge/src/scenarios.rs
- Symtropy physics validation protocol: https://github.com/Luminous-Dynamics/symtropy/blob/main/crates/core/symtropy-physics/RESEARCH_VALIDATION.md
- Sol-Atlas evidence verifier: https://github.com/Luminous-Dynamics/sol-atlas-leptos/tree/main/tools/workforce-active-evidence-verifier
- CUE documentation: https://cuelang.org/docs/
- Cedar policy language: https://docs.cedarpolicy.com/
- Cedar Rust crate: https://docs.rs/cedar-policy/latest/cedar_policy/
- JSON Schema Rust validator: https://docs.rs/jsonschema/latest/jsonschema/
- Rust type-safe units: https://docs.rs/uom/latest/uom/
- RFC 8785 JSON Canonicalization Scheme: https://www.rfc-editor.org/rfc/rfc8785.html
- in-toto Statement v1: https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md
- in-toto attestation framework: https://github.com/in-toto/attestation/blob/main/spec/README.md
- SLSA Provenance v1: https://slsa.dev/spec/v1.2/provenance
- FMI 3.0.2 specification: https://fmi-standard.org/docs/3.0.2/
- Existing Symtropy sealed scenario manifest issue: https://github.com/Luminous-Dynamics/symtropy/issues/1202
- Existing Symtropy engineering replay substrate issue: https://github.com/Luminous-Dynamics/symtropy/issues/1172
- Existing Symthaea predicate-scoped evidence identity issue: https://github.com/Luminous-Dynamics/symthaea/issues/6797
- Existing Symthaea evidence resolver issue: https://github.com/Luminous-Dynamics/symthaea/issues/2617
- OPA/Rego policy language: https://www.openpolicyagent.org/docs/policy-language
