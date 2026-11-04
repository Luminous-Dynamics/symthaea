# CEB specification discovery and falsification V0 — 2026-11-04

**Status:** research protocol / static authoring plan. Not an implemented capability or qualified benchmark.  
**Issue:** [#10400 — CEB-SPEC-001](https://github.com/Luminous-Dynamics/symthaea/issues/10400)  
**Composes with:** [#10173 repository twin](https://github.com/Luminous-Dynamics/symthaea/issues/10173), [#8311 mutation-tested oracles](https://github.com/Luminous-Dynamics/symthaea/issues/8311), [#10068 test-generation validity](https://github.com/Luminous-Dynamics/symthaea/issues/10068), [#10366 causal debugging](https://github.com/Luminous-Dynamics/symthaea/issues/10366), [#10385 engineering quality](https://github.com/Luminous-Dynamics/symthaea/issues/10385), [#10387 causal integration](https://github.com/Luminous-Dynamics/symthaea/issues/10387).  
**Security prerequisite:** [#10064 — qualify the OS isolation boundary](https://github.com/Luminous-Dynamics/symthaea/issues/10064).  
**Claim ceiling:** this document proposes a falsifiable research programme. It does not establish that Symthaea can infer intended specifications, generate adequate oracles, or outperform existing coding agents.

---

## 1. Research question

Can an engineering agent recover a useful, scoped, evidence-backed behavioural contract from an unfamiliar repository—then discover where that candidate contract is wrong or incomplete—before it proposes an engineering intervention?

This is not the same as code search, code summarization, test generation, or patch generation.

A repository can contain several different kinds of truth:

- what an external standard or product contract requires;
- what the public types and interfaces permit;
- what the compiler establishes for a particular configuration;
- what the implementation currently does;
- what tests happen to assert;
- what historical behaviour was promised to downstream users;
- what the agent merely suspects the authors intended.

These facts can agree, conflict, or describe different versions and configuration worlds. Symthaea must not collapse them into a single inferred specification.

A test suite passing establishes conformance to that test suite. It does not by itself establish that the test suite captures the correct requirements. Likewise, observing current implementation behaviour does not prove that the behaviour is intended.

## 2. Paradigm shift

Replace the default loop:

```text
source + existing tests -> patch -> tests pass
```

with:

```text
immutable repository/configuration snapshot
                  |
                  v
       candidate behavioural claims
                  |
                  v
      provenance and authority graph
                  |
                  v
 competing interpretations / assumptions
                  |
                  v
      discriminating falsification plan
                  |
                  v
 independent evidence and counterexamples
                  |
                  v
       qualified / bounded contract
                  |
                  v
   causal engineering intervention (CEB)
                  |
                  v
       independent outcome and regrade
```

The most important design principle is **specification claims must be falsifiable and provenance-carrying**. A system should be rewarded for identifying a contradiction, an unknown, or an under-specified requirement—not for manufacturing a complete-looking contract.

The output is not necessarily one total specification. It may be a set of supported claims with explicit applicability limits, plus unresolved questions and prohibited inferences.

## 3. Claim ontology and evidence authority

### 3.1 Claim kinds

The initial typed claim vocabulary should include:

- precondition and input-domain constraint;
- postcondition and output guarantee;
- state invariant and transition relation;
- ordering, idempotency, retry and delivery semantics;
- concurrency, isolation and consistency guarantee;
- error, failure and recovery behaviour;
- public API and compatibility promise;
- configuration / target / feature conditional;
- resource, latency, throughput or complexity bound;
- security, privacy and authority constraint.

Do not assume every claim can be reduced to a simple input/output function. Temporal and stateful claims need traces, transition systems or sequence-level predicates.

### 3.2 Epistemic kind is separate from claim kind

Each atomic claim must distinguish its *content* from the kind of evidence supporting it. At minimum:

| Evidence kind | What it can establish | What it cannot establish by itself |
|---|---|---|
| External normative | A requirement in a cited contract, standard or task instruction | That the implementation satisfies it |
| Source-declared | A statement explicitly present in source, schema, comment or documentation | That the statement is true, current or enforced |
| Syntax-derived | Parsed declarations, imports, spans and syntactic relations | Type resolution or runtime behaviour |
| Compiler-resolved | A fact established by a pinned compiler / language server in a pinned configuration | Runtime behaviour or intended product semantics |
| Test-oracle | The result of a named test under a pinned runner | That the test oracle is complete or correct |
| Runtime-observed | A behaviour measured on a specific build, input and environment | General behaviour outside the measured domain |
| Intervention-observed | An outcome after a controlled, declared intervention | Universal causality outside the intervention assumptions |
| Historical | A claim or behaviour associated with a named repository revision | That the promise is still current |
| Inferred | A model-generated or derived candidate interpretation | Independent evidence or authority |
| Unresolved | Missing, conflicting or inadequate evidence | Any positive claim |

Authority must not be promoted automatically. In particular:

- a parser edge is not a compiler-resolved symbol relation;
- a passing test is not proof of the task specification;
- a stack-trace correlation is not an intervention effect;
- a dream/counterfactual heuristic is not an observed external outcome;
- a comment is not a binding requirement unless the task contract says it is;
- a historical API guarantee must be scoped to its version and compatibility policy.

### 3.3 Claim record

A future task-local representation should bind each claim to:

- stable claim ID and revision lineage;
- exact repository, source tree, task revision, target, feature set and configuration-world digests;
- subject, relation, object and typed claim kind;
- applicability domain and known exclusions;
- source spans, symbol identities and evidence IDs;
- evidence authority and source-origin lineage;
- support and contradiction references;
- assumptions and competing interpretations;
- epistemic status: candidate, supported, contradicted, conflicted, underdetermined, out-of-domain or superseded;
- a falsifying observation or a statement of why none is currently available;
- invalidation triggers (source edit, dependency/toolchain change, feature/target change, standard revision, etc.).

A natural-language summary may accompany a typed claim. It must not replace the machine-checkable identity, scope and provenance.

Do not require a probability for every claim. Where a probability is emitted, it must carry a declared event definition, probability semantics, model/policy identity, calibration population and scoring method. Otherwise use explicit qualitative statuses or a first-class not-estimated state. An uncalibrated float is not a posterior.

## 4. Evidence graph and source independence

The evidence graph should be a typed provenance DAG, not a bag of retrieved passages.

Each evidence item should identify:

- origin (file/blob, test, compiler invocation, trace, external standard, human-authored task contract);
- exact content digest and source revision where available;
- extraction/measurement tool and version;
- configuration world and runtime environment where relevant;
- authority class;
- parent evidence and derivation method;
- known common ancestors or shared source dependencies;
- capture time and invalidation conditions.

Two tests copied from the same reference implementation are not two independent confirmations. A README copied from a source comment is not independent corroboration. A generated test and the specification that generated it share a causal origin.

The validator should be able to ask whether evidence is independent enough for the specific claim being made. It should not treat distinct file paths or distinct tool names as proof of independence.

## 5. Candidate-contract discovery pipeline

### Stage 0 — freeze the world

Capture the repository tree, dependency lock, compiler/toolchain, target triple, feature selection, relevant environment profile and task contract. Configuration worlds must be explicit; a fact extracted under one Cargo feature world must not silently apply to another.

### Stage 1 — construct the repository twin

Reuse #10173. Extract syntax facts first, then enrich with Cargo metadata and compiler/Rust Analyzer evidence where available. Keep unresolved macro, generated-code, build-script and conditional-compilation facts explicit.

Compare graph construction against Tree-sitter, Cargo metadata, rust-analyzer and SCIP as relevant. Do not use a text-search-only baseline as the sole competitor.

### Stage 2 — extract atomic candidate claims

Convert API declarations, schemas, assertions, docs, historical changes and observed traces into candidate claims. Split compound statements into atomic claims, retain exact source anchors, and record the source's own authority.

Do not turn every implementation detail into a product requirement. Explicitly label descriptive behaviour separately from normative behaviour.

### Stage 3 — generate alternatives

For each consequential claim, produce at least one plausible competing interpretation where the evidence permits it. Record missing preconditions, version boundaries, feature worlds, environmental assumptions and plausible exceptions.

Where there is no meaningful alternative, record why the claim is directly established rather than fabricating uncertainty.

### Stage 4 — design falsification probes

Choose an observation or intervention that could materially change the claim status. Candidate probe families include:

- boundary and equivalence-class cases;
- state-machine sequence exploration;
- property-based and metamorphic relations with justified preconditions;
- differential behaviour against an independently sourced implementation;
- compiler queries across target / feature worlds;
- historical compatibility and API migration checks;
- mutation of candidate implementations and oracle controls;
- controlled fault injection and configuration perturbation;
- counterexamples generated from explicit assumptions.

This stage should consume #5476's prospective value-of-computation semantics and #8286's probe distinctions where appropriate. It should not introduce another general scheduler or duplicate active-probe engine.

### Stage 5 — enforce the independent-oracle gate

A candidate specification cannot validate itself. Before a generated check is counted as oracle evidence, establish its source of authority and independence.

Use the #8311 pattern:

- a known-good control must pass;
- a plausible known-bad control must fail;
- relevant mutations must be enumerated;
- surviving mutants must be classified;
- the check must actually exercise the claimed behaviour;
- the expected result must not be derived solely from the candidate agent's own claim.

A test that compiles but does not invoke the candidate, a comment-only assertion, or a no-op property check is not a behavioural oracle. Compose with #10068 rather than duplicating test synthesis.

### Stage 6 — issue a bounded contract

The result may be:

- supported within a declared domain;
- supported only under explicit assumptions;
- contradicted;
- conflicted between authorities;
- underdetermined / non-identifiable;
- out of domain;
- not assessed.

A repair may proceed only against the contract elements required by its task, with unresolved elements exposed. It must not silently convert "not assessed" to "satisfied".

### Stage 7 — compare prediction with change outcome

After an engineering change, compare predicted and observed behaviour, impacted graph edges, regression controls and independent grades. Invalidate stale facts and update lineage. A passing patch does not retroactively prove that the diagnosis or inferred specification was correct.

## 6. Initial task families

Start with static, non-executable authoring. The initial tranche should contain 12–18 blueprints, not 12–18 claimed qualified tasks.

1. **Public API and trait contract:** infer caller obligations, generic bounds, error semantics, re-exports and downstream compatibility. Include a local-looking diagnostic whose actual cause is an upstream API change.
2. **Configuration-conditional contract:** compare Cargo default/no-default/explicit features, target-specific dependencies, cfg gates, compiler versions and conditional modules.
3. **Conflicting evidence:** source, test, docs, changelog and external specification disagree. The correct output may be a conflict report or clarification request.
4. **Stateful/distributed behaviour:** retry, duplicate delivery, idempotency, ordering, partial commit, restart and recovery. Evaluate sequences, not just one input/output example.
5. **Multiple valid implementations:** preserve semantic contract while permitting different internal architectures. Do not use overlap with one reference diff as correctness.
6. **Under-specified / non-identifiable:** omit a critical precondition, provide an unsupported version claim, or construct observationally equivalent interpretations. Correct behaviour is abstention, a bounded statement or a targeted question.

For each family, author positive controls, plausible incorrect specifications, misleading symptoms and should-abstain variants. Include conditional worlds and interactions. Keep evaluator-only root causes, gold labels, admissible-edit rubrics and hidden checks outside the agent-visible repository.

## 7. Baselines and ablations

Matched tasks, model, harness, tools, context, time/token budget and grader must be frozen.

| Arm | Capability isolated |
|---|---|
| Existing tests only | What the current test suite says, without inferred specification |
| Syntax / manifest facts | Static structure without semantic enrichment |
| Compiler-resolved facts | Stronger type/configuration evidence |
| Documentation/history retrieval | Claims found in declared materials |
| Single-pass model extraction | Candidate claims without iterative falsification |
| Multi-source claim synthesis | Evidence triangulation without active probe selection |
| Repository twin + claim synthesis | Added value from typed repository context |
| Full falsification loop | Evidence selection, counterexamples, oracle qualification and claim revision |
| Lesion / sham / rescue | Whether each evidence source or decision mechanism causes a gain |

Human-authored adjudicated contracts are an upper reference and evaluator source, not an agent competitor. An LLM judge may help triage, but it must not be the sole authority for claim correctness.

## 8. Metrics — keep the dimensions separate

**Claim accuracy**
- supported-claim precision / recall / F1 by claim kind and evidence authority;
- unsupported critical-claim rate;
- missing-critical-invariant rate;
- false-constraint / over-specification rate across valid alternative implementations;
- applicability-domain accuracy and conditional-world leakage;
- conflict, under-specification, non-identifiability and out-of-domain detection.

**Epistemic quality**
- calibration (Brier score / log loss) only for claims with defined probabilistic semantics and ground truth;
- false promotion rate across evidence-authority levels;
- provenance completeness and source-independence correctness;
- rate of claims correctly retracted after counterexamples;
- stale-claim rate after invalidation.

**Falsification quality**
- independent counterexample yield;
- mutation score over the declared relevant mutant set;
- known-good acceptance / known-bad rejection;
- time and compute to falsify a consequential false claim;
- false rejection of valid alternative behaviour;
- fraction of generated tests with demonstrated behavioural effect.

**Downstream engineering**
- independently accepted repair rate;
- protected-invariant preservation and regression rate;
- unnecessary semantic change / patch scope;
- change-impact precision and recall;
- diagnosis correctness, repair correctness and specification correctness reported separately.

**Human interaction / efficiency**
- ASK-F1, targeted question coverage and blocker recall;
- paired act/abstain correctness, selective risk, false-action and false-abstention rates;
- human review time, accepted/rejected suggestions, cost per accepted contract/repair;
- wall time, CPU/GPU, memory, tokens, tool calls and failure cost.

Do not compress these into one scalar. Publish the Pareto frontier and raw per-task dispositions.

## 9. Prospective pilot targets

These are hypotheses to calibrate on a development set and freeze before holdout evaluation. They are not observed Symthaea results and are not a substitute for sample-size planning.

| Dimension | Pilot target | Holdout target |
|---|---:|---:|
| Supported atomic-claim precision | >= 0.95 | >= 0.97 |
| Supported atomic-claim recall | >= 0.90 | >= 0.93 |
| False constraints across valid alternatives | <= 5% | <= 3% |
| Unsupported critical claims | <= 1% | <= 0.5% |
| Fabricated source/compiler evidence | 0 | 0 |
| Correct conflict / under-specification detection | >= 0.90 | >= 0.95 |
| Relevant mutation score | >= 0.85 | >= 0.90 |
| Provenance coverage on claims used to authorize a repair | 100% | 100% |
| Stale accepted facts after completed invalidation | 0 | 0 |

Report confidence intervals, effect sizes, repository/template clusters and seed variance. Determine task count from pilot variability and the smallest practically important effect. Do not infer general superiority from a handful of synthetic tasks.

## 10. External benchmark alignment

This programme should connect to, not claim to replace, existing evaluations.

- **SWE-Bench Pro V2 Full and HARD:** public issue resolution remains a useful repair anchor, but its refreshed protocol reports 642 tasks / 11 repositories, two-sided reference-patch / empty-patch admission and pristine-image regrading. The benchmark page also describes residual verifier execution of patch-provided code; it is not a substitute for Symthaea's own isolation qualification.  
  https://labs.scale.com/leaderboard/swe_bench_pro_public_v2
- **SWE Atlas Codebase QnA, Test Writing and Refactoring:** covers upstream repository comprehension, meaningful tests and larger multi-file restructuring. Keep each task family separate; rubric task resolution is not interchangeable with issue-fix pass rate.  
  https://labs.scale.com/leaderboard/sweatlas-qna  
  https://labs.scale.com/leaderboard/sweatlas-refactoring
- **Precise Debugging Benchmark:** separates bug-level recall from edit-level precision; published frontier baselines show why test pass alone is insufficient.  
  https://www.microsoft.com/en-us/research/publication/precise-debugging-benchmark-is-your-model-debugging-or-regenerating/
- **SWE-fficiency:** expert-normalized speedup on real workloads, with correctness preserved; optimization quality is a different skill from reducing an application's own baseline runtime.  
  https://proceedings.mlr.press/v306/ma26l.html
- **HiL-Bench and AgentAbstain:** evaluate targeted clarification and paired act/abstain behaviour. Preserve their distinct metrics and task definitions; do not compare their scores as though they measure the same thing.  
  https://labs.scale.com/leaderboard  
  https://github.com/AntiQuality/agentabstain

A live leaderboard score is a dated model+harness+budget+protocol result, not a universal capability constant. Freeze the exact leaderboard revision and reproduce the task protocol before using a number as a formal baseline. Public repositories are development/cross-check material; unfamiliar private repository templates and held-out contract families are needed for confirmatory claims.

## 11. Security, custody and benchmark integrity

- Static parsing, manifest analysis, source-history inspection, task blueprint authoring and schema validation can proceed before #10064.
- Do not execute repository-controlled build scripts, tests, proc macros, generated programs or candidate code until the OS isolation boundary is independently qualified.
- Keep task authoring, hidden gold labels, oracle/grader, candidate workspace and regrade environment in separate trust domains.
- Preserve every eligible task and disposition: accepted, rejected, abstained, unsupported, timed out, tool failed, ungraded, not attempted or censored.
- Reuse #10115 for the CEB task profile, #10314 for canonical run accounting, #10212 / #10167 for receipt/DSSE verification, and #10369 for signer/trust qualification. Do not create another receipt schema, DSSE envelope, scheduler or generic test-generation system.
- A digest is not authentication. A signature is not proof that a signed assertion is true. Distinct key labels are not proof of operational independence.

## 12. First implementation sequence

**S0 — static protocol**
- Freeze this claim/evidence vocabulary and review it against #10173, #8311, #10068 and #10387.
- Author 12–18 blueprints across at least four contract families.
- Establish evaluator-only gold-label custody and source-independence rules.
- Define reference / empty-patch / good-control / bad-control admission checks.

**S1 — static fixture and adjudication**
- Build a small synthetic multi-crate Rust fixture with explicitly conditional API and state-machine contracts.
- Publish syntax/manifest gold labels and compiler-resolved gold labels only for facts actually established by a pinned compiler.
- Create valid alternative implementations and adversarial false specifications.
- Do not execute fixture code in this slice.

**S2 — oracle qualification**
- Compose with #8311 and #10068 to verify that checks call the candidate and distinguish the declared known-good / known-bad controls.
- Mutation-test the independent evaluator.
- Preregister metrics, baseline arms, holdout splits and analysis before comparison.

**S3 — isolated dynamic falsification**
- Block on #10064; use qualified trust and evidence layers from #10369 / #10370 / #10212.
- Add property, metamorphic, differential and state-machine execution only after isolation and grader custody are independently verified.
- Run primary and pristine regrade; report exact task-level evidence and missingness.

**S4 — causal engineering composition**
- Link supported contracts and unresolved assumptions into #10387's hypothesis / forecast / action / outcome ledger.
- Use #5476 forecast-before-outcome and #5478 matched-budget ablation semantics.
- Evaluate whether specification discovery improves downstream repair precision, not only claim extraction scores.

## 13. Stop rules

Stop or abstain when:
- critical contract sources conflict and no authority ordering resolves them;
- the available evidence cannot distinguish competing behaviours;
- a requested claim is outside the declared version/configuration domain;
- the only test oracle is generated from the claim being tested;
- an independent known-good / known-bad control cannot be established;
- required execution or grader isolation is not qualified.

A partial, correctly bounded contract is preferable to a complete-looking but unsupported one.

---

## References and related work

- CEB task and evidence foundations: #10115, #10314, #10167, #10212, #10369, #10370.
- Repository twin: #10173.
- Mutation-tested oracle quality: #8311; generated-test correctness: #10068.
- Causal diagnosis and integration: #10366, #10367, #10385, #10387.
- Existing metareasoning / probe foundations: #5476, #5478, #8286, #8721.
- Isolation prerequisite: #10064.

**Final claim ceiling:** V0 is a research protocol and static authoring plan. No executable tasks, comparative results, production integration or benchmark qualification are claimed.
