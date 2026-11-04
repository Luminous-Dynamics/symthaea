# CEB engineering-intelligence task blueprints V0 — 2026-11-04

**Status:** eight non-executable authoring blueprints. Not task records, executable fixtures, qualified oracles, or benchmark results.  
**Parent:** [CEB-INTELLIGENCE-001 #10419](https://github.com/Luminous-Dynamics/symthaea/issues/10419)  
**Related:** #10173 repository twin; #10400 specification discovery; #8311 oracle quality; #10385 repair precision / optimization / abstention; #10369 trust; #10064 execution isolation; #10370 evidence convergence.  
**Claim ceiling:** the blueprint set defines task intent, evidence requirements, adversarial controls and metrics. It does not assert that Symthaea or any other system has achieved the targets.

---

## 1. Strategic scope

This tranche extends the engineering loop beyond issue-to-patch completion.

| Capability | New task family | Central question | External reference |
|---|---|---|---|
| Context acquisition | CEB-CONTEXT | Did the agent retrieve the smallest sufficient evidence set? | ContextBench |
| Pre-merge review | CEB-REVIEW | Did it find real defects before code was merged? | CRBugBench |
| Secure correctness | CEB-SECURE | Is the change both functional and secure? | Agent Security League |
| Rust-specific reasoning | CEB-RUST | Can it reproduce, explain and repair repository-level Rust defects? | Rust-SWE-bench / RUSTFORGER |

The benchmark must preserve distinct outcomes. Correct retrieval is not a correct specification. A review comment is not a true positive without a real defect. Functional success is not secure success. A source-based Rust explanation is not compiler evidence.

Use the existing CEB task schema and the existing receipt/accounting path. Do not create a new task schema, grader protocol, scheduler, or result envelope for these lanes. The current task schema does not yet have a dedicated code-review lane; use the narrowest existing lane in any future task adapter and document the classification. Do not modify the enum merely to make the task bank look tidier.

## 2. Common controls and custody

Every eventual task instance must pin:
- source and patch snapshot, repository-template family, task/configuration identity and leakage group;
- model, harness, prompt/tool registry, resource budget and environment when evaluation is eventually admissible;
- exact agent-visible files and read-only/writable paths;
- evaluator-only defect/requirement labels, severity, source-origin lineage and admissible alternatives;
- known-good control, plausible known-bad control, empty-answer control and adversarial mutation set;
- independent grader revision and regrade path;
- disposition for pass, fail, refusal, abstention, unsupported, tool failure, timeout, ungraded, not attempted and invalidated.

Do not place a “do not read gold file” instruction inside a shared workspace and call that hidden-answer custody. The gold material must be physically outside agent-visible and candidate-writable storage. The existing CEB repository twin currently co-locates its source and design/evaluator artifacts; it is not ready to host an agent-scored task until a source-only view is materialized.

All tasks remain `design_only`. No repository-controlled task, candidate patch, build script, procedural macro, test or grader code is to be executed until #10064 qualifies isolation. A Nix development shell or host opt-in is not an isolation boundary.

---

## 3. CEB-CONTEXT — retrieval quality

### CEB-CTX-001 — minimal sufficient context for a cross-crate contract

**Question:** Can an agent retrieve the smallest set of repository evidence sufficient to explain a scoped contract and its failure modes?

**Task idea:** A synthetic multi-crate Rust change requires reasoning about a public trait, one concrete implementation, a generic service, an API facade and one relevant test. The complete repository is available, but the agent is budgeted for a limited context window. The task asks it to identify the exact evidence spans needed before proposing a change.

**Agent output:** ranked file/symbol/span evidence, atomic claims, omitted-but-relevant evidence, unresolved questions and token/cost accounting.

**Gold labels:** independently adjudicated relevant and necessary spans; sufficient evidence sets (allow multiple valid sets); distracting but plausible spans; common-origin duplicate facts; claims that cannot be established from source alone.

**Negative controls:** lexical keyword bait in irrelevant files; copied documentation that duplicates a source fact; a relevant trait implementation reachable only through a non-obvious generic bound; one hidden counterexample in a call site.

**Metrics:** Recall@k, Precision@k, Context F1, sufficiency rate, irrelevant-token fraction, evidence-claim coverage, retrieval latency/cost, downstream claim accuracy under a fixed token budget.

**Paradigm test:** compare lexical search, AST/symbol retrieval, Cargo metadata, compiler/LSP graph enrichment, graph traversal and Symthaea/HDC retrieval using the same task, query, model and token budget. More files or tokens must not receive automatic credit.

### CEB-CTX-002 — provenance-aware context under correlated evidence

**Question:** Can an agent distinguish independent corroboration from multiple copies of the same underlying claim?

**Task idea:** A repository contains a source declaration, a README paragraph copied from an earlier version, a generated API page and tests with a shared helper. Some sources are stale or share a common origin. The agent must produce a contract explanation and identify which evidence is independent.

**Gold labels:** source origin graph, revision lineage, stale/fresh status, supported proposition set and unresolved conflicts.

**Negative controls:** duplicate a source quote into more files; change one copy but not its origin; move the source declaration while leaving stale documentation; add a test that uses the same helper implementation as the code under test.

**Metrics:** source-origin clustering accuracy, false corroboration rate, stale-claim rate, provenance coverage, conflict detection and evidence-weighted claim precision.

**Adversarial requirement:** duplicating one claim into N files must not increase independent-evidence count.

### CEB-CONTEXT evaluation targets

Prospective goals, not achieved values:
- Context Recall@10 >=0.85 pilot / >=0.90 sealed holdout.
- Context F1 >=0.45 pilot / >=0.50 sealed holdout.
- Evidence-backed claim coverage 100% for claims used to authorize a repair.
- No degradation in downstream correctness at a fixed token/tool budget.
- Report the best Recall and best F1 configurations separately; never combine values from different systems or runs.

---

## 4. CEB-REVIEW — pre-merge defect detection

### CEB-REV-001 — detect a non-local defect introduced by a patch

**Question:** Can an agent detect a real defect in a proposed change when the bug's evidence spans files outside the diff?

**Task idea:** Provide a synthetic pull request and source-only repository. The diff changes an interface or state transition; the defect occurs in a caller, retry path, authorization check or feature-gated implementation outside the changed lines. The agent must submit ranked, actionable findings before merge.

**Agent output:** one finding per defect, exact changed-line anchor, relevant source spans, violated invariant, impact/severity, minimal counterexample trace and confidence. “No finding” is allowed.

**Gold labels:** human-adjudicated bug identity, root cause, triggering conditions, severity, expected patch span and accepted paraphrases; valid “no bug” examples.

**Negative controls:** a suspicious but correct refactor; a comment-only change; a test-only change that does not affect behavior; a valid alternative implementation; a defect already fixed elsewhere in the same patch.

**Metrics:** bug Recall@1/3/5, actionable precision, severity-weighted recall, false-positive findings per 1,000 changed lines, localization precision, non-local evidence coverage and reviewer minutes per confirmed defect.

**Paradigm test:** compare a reviewer-only agent, the same model with a strong code-review scaffold and the same model with the Symthaea repository twin/specification graph. Do not count verbose speculative comments as review quality.

### CEB-REV-002 — distinguish a security-relevant finding from an attractive false positive

**Question:** Can an agent identify whether a suspected issue violates a real security or behavioral invariant, rather than merely matching a known risky pattern?

**Task idea:** Construct paired review tasks with superficially similar code patterns: one genuinely bypasses an authorization boundary, the other is safe because a caller invariant or capability check constrains it.

**Gold labels:** exact trust boundary, invariant, call-chain evidence, exploit precondition, false-positive explanation and severity.

**Negative controls:** same API name with different authority flow; unreachable-looking code; validation performed in a shared helper; test fixture that does not exercise the claimed path.

**Metrics:** security bug recall, actionable precision, severity-weighted precision, false-positive review burden, unsupported-claim rate and correct abstention when exploitability is underdetermined.

### CEB-REVIEW evaluation targets

Prospective goals:
- Human-verified bug Recall@5 >=0.80 pilot / >=0.85 sealed.
- Report actionable precision and reviewer burden independently.
- High-severity false negatives have a separate ceiling and cannot be hidden by average recall.
- Every accepted finding has source evidence and an independently adjudicated violation.

---

## 5. CEB-SECURE — functional and secure correctness

### CEB-SEC-001 — repair path/shell injection without breaking the operation

**Question:** Can an agent remove command/path injection while preserving the intended operation and its legitimate path semantics?

**Task idea:** A synthetic workspace tool constructs shell commands from a repository path or interpolates values into a shell command. The task asks for a repair that preserves file operations, handles spaces/Unicode/metacharacters as data and does not expand the agent's or process's authority.

**Agent output:** threat model, trust boundary, vulnerable flow, minimal repair, preserved behavior, security counterexamples and evidence limitations.

**Gold labels:** accepted secure alternatives (for example direct executable-plus-argument invocation), exploit preconditions, protected path set, permitted effects, positive and negative input classes.

**Negative controls:** spaces and Unicode paths; leading dashes; shell metacharacters; symlink/path traversal; relative and absolute path handling; an apparently sanitized but still shell-interpreted input; a patch that disables the feature to make attacks disappear.

**Metrics:** functional pass, secure pass, CWE-stratified outcomes, high-severity false-accept rate, path containment, allowed side effects, regression rate and patch precision. Functional pass and secure pass are separate columns.

### CEB-SEC-002 — preserve authorization invariants across a refactor

**Question:** Can an agent detect that a locally plausible refactor broadens access across tenant or user boundaries?

**Task idea:** A multi-crate service passes an identity/authority object through a generic abstraction. A proposed optimization or API refactor bypasses or weakens a permission check only on one call path. The agent must review and repair the change without disabling legitimate operations.

**Gold labels:** principal/authority mapping, protected resources, authorization invariant, bypass path and permitted alternative implementations.

**Negative controls:** valid service-to-service capability delegation; anonymous access explicitly allowed for one endpoint; an unused authorizer field; an authorization check that is present but applied to a different identity.

**Metrics:** authorization invariant recall, cross-tenant false-accept rate, legitimate-operation preservation, severity-weighted security pass, and unsupported trust-boundary claims.

### CEB-SECURE evaluation targets

Prospective goals:
- Secure task pass >=0.55 pilot / >=0.65 sealed.
- Functional correctness within a preregistered non-inferiority margin (initial proposal: no more than 2 percentage points lower).
- Zero high-severity security false accepts in a finite task set is not a zero-risk proof; publish confidence bounds and exact denominators.
- A change that disables the affected feature is not an accepted security repair.

The Agent Security League is a useful external construct, but its published task set is Python-focused. CEB's Rust-heavy slice must be treated as an independent adaptation, not a direct reproduction or score comparison.

---

## 6. CEB-RUST — repository-level Rust reasoning

### CEB-RUST-002 — reproduce and localize a trait/type error before proposing a fix

**Question:** Can an agent trace a Rust issue across workspace manifests, trait bounds, generic callers and diagnostic context without guessing?

**Task idea:** A synthetic multi-crate workspace contains a public trait/API change and a downstream consumer that violates the new bound. The agent receives a source snapshot and a pinned diagnostic excerpt prepared by an independent trusted authority; no compilation is run by the candidate in this design tranche.

**Agent output:** minimal reproduction explanation, trait/type obligation graph, affected crates and call sites, exact diagnostic/source mapping, proposed repair alternatives and unresolved compiler-dependent claims.

**Gold labels:** intended type relationships, broken bound, affected paths, accepted equivalent patches, diagnostic authority and facts that remain unresolved without the compiler.

**Negative controls:** a similar method with a different trait; blanket implementation that applies only under a specific bound; a private trait with the same name; wrong target/feature diagnostic; stale diagnostic from a different commit.

**Metrics:** issue reproduction accuracy, trait/type claim precision/recall, change-impact precision/recall, compiler-authority discipline, repair bug recall/edit precision and affected-crate recall.

**Boundary:** source inspection can support a hypothesis; it cannot establish the compiler's actual trait resolution. Compiler claims require a pinned trusted compiler/rust-analyzer receipt after #10064 is qualified.

### CEB-RUST-003 — preserve feature and target scope during API evolution

**Question:** Can an agent evolve an API without accidentally activating or breaking symbols on unsupported feature/target combinations?

**Task idea:** A public API changes under Cargo features and target gates. The task requires preserving the intended Linux/Windows behavior while changing a trait or re-export.

**Gold labels:** feature selection provenance, effective feature sets, cfg predicates, target dependencies, symbols active/inactive by world, compatible downstream patterns and allowed alternatives.

**Negative controls:** enable a Windows feature on Linux; disable defaults but explicitly select one feature; add a target dependency without changing module cfg; change a public re-export without moving the declaration; stale graph facts after the edit.

**Metrics:** feature/target claim accuracy, world-matrix exact match, false activation/inactivation rates, API compatibility, change-impact recall and stale-fact rate.

### CEB-RUST evaluation targets

Prospective goals:
- Rust issue-resolution >30% pilot / >35% sealed holdout only in a directly matched replication protocol.
- Supported trait/type relation P/R >=0.90 pilot / >=0.95 sealed.
- Feature/target world exact-match >=0.95 with zero compiler-authority inflation.
- Report issue reproduction, diagnostic comprehension, compile/repair success and regression control separately.

Rust-SWE-bench / RUSTFORGER is a historical external reference, not a November 2026 frontier leaderboard. Re-fetch model, harness and task revisions before a formal numerical comparison.

---

## 7. Cross-family compositions

After the individual families have independent gold labels, create a small number of composed tasks:

- **INT-C01 — context × review:** relevant code is outside the patch; score retrieved evidence and defect Recall@k independently.
- **INT-C02 — review × security:** a proposed patch is functionally correct but introduces an authorization bypass.
- **INT-C03 — Rust × context:** a trait bound is changed in one crate and its effect is only visible in a downstream generic consumer.
- **INT-C04 — Rust × security:** an unsafe/FFI wrapper changes ownership or validation assumptions at a trust boundary.
- **INT-C05 — context × configuration:** correct code is active only under one feature/target world.
- **INT-C06 — review × abstention:** the agent must withhold a high-severity claim because the required runtime or external authority evidence is missing.

Do not count composed tasks as independent samples when they reuse the same template, root cause or gold labels.

## 8. Matched benchmark design

For each lane, compare:
1. deterministic/static tool baseline where applicable;
2. model-only minimal fixed harness;
3. same model with strong industry-style scaffold;
4. same model plus the Symthaea component under test;
5. Symthaea-native route only where genuinely implemented without hidden foundation-model fallback.

Pin model, prompt, harness, tool set, budgets, source snapshot, environment and grader. Use paired tasks and cluster by repository template and defect/root-cause family. Retain every disposition and report cost per independently accepted engineering outcome, not cost per attempt.

Required separate measures:
- retrieval: Recall@k, Precision@k, F1, context sufficiency, token/cost/latency;
- review: bug Recall@1/3/5, actionable precision, severity and reviewer burden;
- secure correctness: functional pass, secure pass, CWE strata, high-severity false accepts;
- Rust: reproduction, trait/type/configuration reasoning, patch acceptance and regression control;
- integrity: coverage, grader qualification, exact-head evidence, trust and isolation assurance.

No omnibus engineering score should hide a regression in any one dimension.

## 9. Readiness checklist

- [ ] Every task converted to the existing CEB task schema without introducing a new receipt or grader interface.
- [ ] Agent-visible source tree and evaluator-only labels physically separated.
- [ ] Gold claims, defect identities, severity and accepted alternatives independently adjudicated.
- [ ] Valid “no defect” and “underdetermined” controls included.
- [ ] Functional and secure outcomes separately scored.
- [ ] Model/harness/resource budgets matched for comparative runs.
- [ ] Public benchmark licenses, task versions, and leakage risks reviewed.
- [ ] #10064 isolation qualified before any repository-controlled execution.
- [ ] #10369 trust and #10370 evidence convergence gates satisfied before publishing execution-grade receipts.

## 10. Status

- New authoring blueprints: **8**.
- Families: **4**.
- Cross-family compositions: **6**.
- Executable fixtures: **0**.
- Qualified tasks: **0**.
- Symthaea benchmark results: **0**.

This document is a research design. It does not establish that Symthaea currently outperforms CRBugBench, Agent Security League, ContextBench, Rust-SWE-bench or any other benchmark.
