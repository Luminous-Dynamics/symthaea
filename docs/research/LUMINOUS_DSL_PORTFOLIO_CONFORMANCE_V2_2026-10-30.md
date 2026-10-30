# Luminous DSL Portfolio and Conformance Protocol v2

**Date:** 2026-10-30  
**Status:** Draft cross-project coordination contract  
**Parent:** [DSL-PORTFOLIO-001 #9246](https://github.com/Luminous-Dynamics/symthaea/issues/9246)  
**Supersedes as current status:** [Luminous Dynamics DSL Portfolio RFC v1](./LUMINOUS_DSL_PORTFOLIO_RFC_V1.md)

## 1. Executive decision

Build a portfolio of small, typed domain-specific languages around one shared conformance discipline. Do not build a universal Luminous language, universal semantic IR, or new evidence/signature envelope.

The shared layer handles source identity, strict parsing, schema/compiler identity, stable diagnostics, canonicalization profiles, conformance fixtures, and verification receipts. Each project keeps its own domain semantics, IR, executor, authorization boundary and independent verifier.

The key new distinction is between:
- a specification that is structurally valid;
- a plan whose declared domain capabilities resolve;
- an operation admitted by the relevant authority;
- an operation that actually executed;
- a result independently verified;
- a scientific or governance claim reviewed as supported.

No compiler or schema pass may collapse those stages into verified=true.

## 2. Portfolio map: current work, not hypothetical projects

| Project | Current owner and work | Next meaningful increment | Explicit non-claim |
|---|---|---|---|
| Shared evidence | LEP prototype in Symthaea #8954 | Expand independent compound-adversarial and cross-language conformance | LEP validation does not authenticate signers or prove scientific claims |
| Mycelix | Policy bake-off #5522; equivalence/CEL follow-up #5530/#5531 | Restore truthful Rust policy baseline under #5189; then execute generated decision-space parity | Cedar/OPA success is not three-engine parity while Rust oracle fails |
| Symtropy | Scenario DSL design #2985/#2987; CUE study #3001 | Freeze schema/typed IR, then compare Rust + JSON Schema + CUE | A valid NACA0012 manifest is not an executed/qualified CFD run |
| Symthaea | Experiment DSL issue #9215 | Compile one existing benchmark manifest into a frozen paired case matrix | A frozen protocol does not prove a hypothesis |
| Sol-Atlas | EnergyStudyManifest #82, exact dispatch oracle #309, regret #310/#311, design adapter #313/#314 and selector #315/#316 | Define and implement a Pareto-selector over complete oracle alternatives | The selector is not a schedule generator, site optimizer, or dispatch authority |
| Nixward | Candidate ChangePlan DSL (P2) | Start only after exact host-change and authorization ownership is mapped | A plan cannot authorize arbitrary host mutation |
| Manufacturing/process | Candidate process-recipe DSL (P2) | Extend domain-owned process models and validation contracts | Recipe schema validity is not process capability or product quality |

### 2.1 LEP: shared evidence profile

[Symthaea PR #8954](https://github.com/Luminous-Dynamics/symthaea/pull/8954) provides a versioned evidence predicate profile over established attestation formats, not a competing envelope.

The dedicated LEP conformance workflow passed formatting, locked Rust tests, Clippy, CLI positive/negative fixtures and benchmark artifact collection on its exact PR head. Other workflows associated with that commit were skipped; do not report whole-repository CI success.

The published 2026-10-30 synthetic benchmark measures the incremental cost of semantic validation over strict parse + schema validation:
- 1 KiB requested / 1,247 actual bytes: p50 9.819 µs schema-only, 11.572 µs with semantic checks;
- 10 KiB: 11.872 µs vs 13.555 µs;
- 100 KiB: 31.379 µs vs 33.281 µs;
- 1 MiB: 236.190 µs vs 235.548 µs.

That harness excludes signature authentication, signer authorization, network retrieval, artifact re-hashing, scientific execution and claim adjudication. It is an initial synthetic baseline, not an industry leaderboard or SOTA result. See [LEP benchmark results](./LEP_CONFORMANCE_BENCHMARK_RESULTS_2026-10-30.md).

### 2.2 Mycelix: policy semantics and authority

[Mycelix #5522](https://github.com/Luminous-Dynamics/mycelix/pull/5522) contains a 30-case Rust-derived authorization corpus, Cedar policy/schema and OPA/Rego implementation. The latest exact workflow run has:
- Cedar: 30/30 passed;
- OPA/Rego: 38/38 passed (the same 30 plus eight additional controls);
- corpus/adapter integrity: passed;
- Rust policy oracle: failed before evaluation because scientific_credentials is absent from the current core tree; projection tests were skipped.

This is not three-engine parity. The repair/migration decision belongs to [#5189](https://github.com/Luminous-Dynamics/mycelix/issues/5189) and [protocol-resolution PR #4968](https://github.com/Luminous-Dynamics/mycelix/pull/4968). Do not add a stub or invent credential-registry semantics to make CI green.

The next research layer is [#5530 / PR #5531](https://github.com/Luminous-Dynamics/mycelix/pull/5531): bounded decision-space enumeration, policy permission diffs with counterexamples, CEL runtime conformance, CEL-Java/Z3 authoring-time analysis, and mutation testing. Keep Cedar's Lean-verified semantics as a quality bar. A CEL-Java proof does not prove the separate Rust CEL interpreter conforms.

The non-delegable integrity kernel retains signature, identity/key lifetime/revocation, organization membership, replay, receipt, and atomic commit requirements. Pure policy authorization is not independent-evidence qualification.

### 2.3 Symtropy: scenario / physical model

[ENG-SCENARIO-DSL-001 #2985 / draft PR #2987](https://github.com/Luminous-Dynamics/symtropy/pull/2987) fills the source-authoring/compiler gap between:
- ENG-SYNTH #1172: deterministic scenario execution, capability inventory, stable entity identity and replay;
- ENG-SYNTH-CRED #1173: context-of-use, model suitability and numerical-integrity semantics;
- LEP: exact result/evidence identities and claim ceilings.

The language is a bounded authoring layer. It compiles TOML to typed JSON IR and a capability-checked plan. It does not implement a physics engine, solver, replay authority, or evidence store.

The first real input is benchmarks/aero/naca0012-inc-rans/case.toml. Its current status explicitly says input references are pinned, the run is not yet qualified. Preserve that status.

[CUE evaluation #3001](https://github.com/Luminous-Dynamics/symtropy/issues/3001) compares CUE as an independent design-time constraint/schema compatibility tool. Pin stable CUE v0.17.1; do not use the v0.18 prerelease for a release gate. CUE success means compatibility of declared data-value sets, not physical-model compatibility.

### 2.4 Symthaea: experiment design

[EXP-DSL-001 #9215](https://github.com/Luminous-Dynamics/symthaea/issues/9215) defines preregistered experiment design and case-matrix compilation. It should reuse existing typed records through adapters:
- quantum-comp benchmark manifest;
- helicopter fixed-authority benchmark;
- communication evaluation plan and dataset-split auditor.

The shared core must preserve independent experimental unit, assignment unit, cluster, repeated measure, pairing, matched information/compute budgets, immutable seeds, leakage groups, primary outcome, multiplicity and stopping rules. Existing domain records are not flattened into one generic manifest.

### 2.5 Sol-Atlas: energy dispatch

Sol-Atlas already has much of the energy study / oracle substrate:
- [#82](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/82): EnergyStudyManifestV1 binds case/data/model/policy identity.
- [#309](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/309): bounded integer-quantum exact dispatch oracle and Pareto frontier.
- [#310/#311](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/311): heuristic schedule generator, regret comparison and independent verifier.
- [#313/#314](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/314): strict candidate-subgraph adapter from Symtropy into the bounded dispatch request.
- [#315/#316](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/316): Energy Dispatch Pareto-Selector DSL.

Critical type boundary: the heuristic in #311 generates a schedule; the v0.1 selector DSL selects among already enumerated Pareto alternatives. They are different policy classes. Preserve the heuristic as a separate benchmark arm. Do not claim the selector DSL compiles or reproduces the heuristic.

The exact oracle is a calibration instrument for small finite cases, not a replacement for REopt, PyPSA or GenX. Cross-tool comparisons require genuinely matched model boundaries, units, chronology, constraints, storage state/efficiency, terminal reserve and objective priorities.

## 3. Shared conformance protocol (properties only)

The same high-level stages should be visible across projects, but domain-specific stage names and semantics remain owned locally:

~~~text
exact source bytes
  -> strict bounded parser
  -> structural/schema validation
  -> typed domain IR
  -> domain semantic validation
  -> registry/capability resolution
  -> deterministic plan/policy
  -> separately authorized admission (where needed)
  -> execution and raw artifacts
  -> independent post-execution verification
  -> evidence/claim review
~~~

Common report properties:
- source-byte digest and source revision;
- schema/compiler/registry version identities;
- stable diagnostic codes and field paths;
- input corpus and conformance-suite digests;
- typed IR / semantic plan identity;
- exact test target identity;
- explicit unresolved / unsupported capabilities;
- claim ceiling and limitations.

A universal valid boolean is forbidden. Each domain must preserve its own dispositions, e.g. Invalid, Blocked, Unsupported, NotComparable, Unknown, PlanReady, NotRun, Skipped, Indeterminate, Pass, and Fail, where these terms are precisely defined.

## 4. Canonical identity is a profile, not a universal algorithm

Do not force every DSL into one canonical encoding without examining its data model.

- RFC 8785/JCS is an interoperability candidate for JSON values that obey its I-JSON and IEEE-754 number constraints. Larger integers / high-precision values need explicit representation, commonly strings.
- Exact integer protocols, fixed-point quantities, large seeds, epoch values and enum-tagged domains may require versioned domain-separated binary encodings.
- Keep authoring-source identity separate from semantic-IR identity.
- Keep case, model, policy, registry, execution and result identities separate where semantic independence matters.
- Sort only set-like collections. Preserve order for objective priority, event schedules and other ordered sequences.
- Publish independent cross-language golden vectors before using a canonical digest in a signature/authority receipt.
- Non-cryptographic fingerprints must never be promoted to artifact commitments.

## 5. Assurance levels and benchmark discipline

Every result must identify which assurance level it reaches:

1. **Example conformance:** all frozen fixtures pass.
2. **Bounded exhaustive conformance:** every input in a declared finite abstraction passes; abstraction limits are recorded.
3. **Formal proof:** a proof establishes the property under explicit assumptions, semantic model and supported theory.

Do not present an example suite as bounded exhaustive coverage, or a bounded exhaustive result as a universal proof.

Cross-cutting hard gates:
- 100% expected outcomes on immutable positive/negative fixtures;
- zero silent drops of unknown/unsupported requirements;
- zero false positive readiness/allow on hard-invalid controls;
- stale/target-mismatched receipts rejected;
- semantic mutation changes normalized identity, harmless formatting does not;
- at least 1,000 deterministic mutations before v0.2 for mature DSLs, with equivalent-mutant accounting;
- cold/warm parse/compile/evaluate and end-to-end costs measured separately;
- p50/p95/p99, throughput, allocations/RSS and concurrency disclosed;
- no SOTA claim without public pinned same-input/same-scope comparisons and independent verification.

Correctness and security are hard gates, never weighted against speed.

## 6. Toolchain decisions to test, not assume

| Task | Candidate | Appropriate role | Main caveat |
|---|---|---|---|
| Authorization policy | Cedar | Typed policy language, schema validation and symbolic analysis | Does not own Mycelix identity/signature/commit kernel |
| General policy engine | OPA/Rego | Production evaluation, profiling and partial evaluation | Query vs compilation/network costs must be separated |
| Portable expression policies | CEL | Small non-Turing-complete expression subset | Rust runtime conformance and Java/Z3 verification are different evidence |
| Constraint composition | CUE | Schema unification, TOML/JSON interop, value-set subsumption | Go toolchain cost; design-time only unless separately justified |
| Structural interchange | JSON Schema 2020-12 | Portable structural contract | Cannot replace domain semantic validation |
| Scenario authoring | TOML | Human-edited local source where it fits | Source digest differs from normalized semantic digest |
| Evidence envelope | in-toto / DSSE / SLSA-compatible formats | Provenance and attestation interoperability | Authenticity does not prove scientific truth |
| Model interchange | FMI / SSP | External component and co-simulation boundary | Interface conformance does not validate model fidelity |

Pin exact versions. An evaluator's formal proof must bind its own implementation, input model and assumptions. A proof in CEL-Java does not automatically prove a Rust CEL runtime; a CUE subsumption proof does not prove solver semantics.

## 7. Immediate order of work

1. Resolve Mycelix #5189 through the real registry migration/protocol choice; keep #5522 honest until Rust executes.
2. Review/merge the LEP prototype only after its exact dedicated lane and target identity are independently confirmed; remember the other workflows were skipped.
3. Freeze Symtropy v0.1 typed IR and NACA0012 fixture; implement strict Rust validation before considering execution integration.
4. Run CUE v0.17.1 in a design-time reference lane on the same frozen scenario fixtures.
5. Implement Symthaea experiment DSL one adapter at a time, starting with fixed-authority helicopter and a complete expected case matrix.
6. Implement Sol-Atlas Pareto-selector only after #309/#311 semantics and frontier completeness are frozen; preserve the generator/selector distinction.
7. Defer Nixward and process-recipe DSLs until authority and model-validation owners are explicit.

## 8. Non-claims

This portfolio does not claim:
- a universal DSL or shared semantic IR;
- Cedar/OPA/Rust parity until the Rust oracle runs;
- CEL formal proof for an unverified Rust interpreter;
- solver execution/physical validation from a well-formed scenario;
- hypothesis truth from preregistration or benchmark success;
- site feasibility, resilience or safety from the finite energy oracle;
- policy selector equivalence to a schedule-generating heuristic;
- SOTA superiority from synthetic internal benchmarks.

**Success criterion:** contributors can author precise domain specifications, compile them into deterministic typed plans, and independently verify what those plans and results establish—without duplicated authorities, ambiguous semantics, or inflated scientific claims.
