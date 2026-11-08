# CEB Review Claim-Support Closure and Selective Invalidation Protocol V0 — 2026-11-08

**Status:** Public design-only extension to CEB-CONTINUITY-001 (#10630) and its existing evidence-continuity protocol (#10631). No new task schema, receipt, repository graph, metric lane or executable evidence engine.
**Primary owner:** #10630 owns evidence continuity across repository evolution. This document narrows one unresolved engineering question: how to prove which specific claims depend on changed evidence, and when an unchanged claim can safely be retained.
**Composes with:** #10173 repository-twin change impact; #9112 decision-use validity; #10366/#10387 causal evidence; #10314 canonical receipts; #10610 rerun/regrade/reuse; #10619 statistical claim licensing; #10630 continuity.
**Execution boundary:** candidate-controlled execution remains gated by #10064; independent pristine regrade remains gated by #10369.

## 1. The gap: selective invalidation needs a proof of non-impact

The existing continuity contract correctly distinguishes retain, revalidate, invalidate, supersede and unresolved. It also requires that an agent not invalidate everything or blindly reuse stale evidence.

The difficult case is the **retain** decision. A changed file, dependency, test, build flag or trust root may affect one claim but not another. A system must not declare a claim unaffected merely because it did not notice a textual overlap or because the claim's primary source file did not change.

This extension makes the dependency closure behind each claim inspectable. Its question is:

> Given a versioned claim, the evidence and assumptions supporting it, and an intervening change event, can the reviewer show exactly which support dependencies were affected—and why every retained claim remains inside the validity scope of its evidence?

This is a refinement of #10630's evaluation adapter, not a second evidence-continuity system. It is not a new metric lane, introduces no candidate code execution, and creates no new run receipt. Its six-case public challenge is not a sealed holdout.

## 2. Paradigm shift: from citations to claim-support closure

A citation identifies where a statement may have come from. A support closure identifies everything that must remain valid for the statement to be used.

For a claim C, define a directed typed dependency graph G=(V,E). Nodes are limited to existing CEB concepts and evidence records:

- **claim:** the proposition being used for an engineering decision;
- **evidence:** source anchor, test result, trace, configuration record, contract clause, oracle result or existing receipt subject;
- **assumption:** explicit scope or causal assumption;
- **authority:** the existing CEB evidence-authority or decision-use validity status;
- **environment:** build, feature, target, toolchain, dependency or runtime identity;
- **policy:** security, trust, approval, authorization or compatibility rule;
- **method:** evaluator, grader, metric or validation procedure identity.

Typed edges record why one item is needed by another:

- **supports:** evidence supports a proposition;
- **derived_from:** a claim is inferred from another claim or observation;
- **assumes:** a claim depends on an explicit assumption;
- **scoped_by:** a claim is limited by a contract, target, actor, resource, time or environment;
- **authorized_by:** current decision use depends on a policy, signer or authority root;
- **measured_under:** a result depends on an evaluator, metric or oracle version;
- **produced_under:** evidence was produced under an exact source/build/runtime identity;
- **contradicts:** evidence or a later event refutes the claim within its scope.

The graph is an annotation/evaluation view over existing CEB evidence and repository-twin facts. It must not create another repository graph, task schema or receipt.

### 2.1 Support closure

For claim C, the support closure Cl(C) is the set of all dependency nodes reachable by following edges that are necessary for the claim's declared meaning and current decision use.

The closure must include:
1. terminal evidence items and their exact content identity;
2. source revision/tree and relevant source anchors;
3. all assumptions and applicability boundaries;
4. contract/policy versions that define the expected behavior;
5. build/configuration/environment identity where the claim depends on it;
6. oracle/evaluator/method identity where the claim depends on a measured result;
7. authority and authorization roots needed for current use;
8. upstream claims from which the finding was derived.

A source citation without these relevant dependencies is incomplete for a consequential claim. A complete closure does not mean every file in the repository; it means every dependency necessary for that specific claim and declared scope.

### 2.2 Closure completeness is itself a claim

A system cannot prove non-impact merely by failing to find an edge. A retain decision therefore requires a bounded closure-completeness argument:

- the claim's source and caller scope are stated;
- the applicable event classes and hidden dependencies are considered;
- all dependency types required by the claim kind are represented;
- the relevant repository-twin/change-impact facts are available at the required authority;
- any uninspected dependency or unknown caller is retained as an open-world remainder.

If closure completeness is not defensible, do not infer retain. Use revalidate or unresolved according to the existing #10630 rules.

## 3. Event-to-claim impact analysis

Given a change event E and claim C, evaluate impact against Cl(C), not only against file-path overlap.

### 3.1 Change classes and default interpretation

| Event class | Examples | Default impact question |
|---|---|---|
| Source content | hunk edit, symbol rename, moved module, macro/generated source update | Does the changed source anchor or an upstream derived claim occur in Cl(C)? |
| API/contract | trait signature, public API, requirement clarification, deprecation | Does the governing contract or the claim's scoped behavior change? |
| Build world | Cargo feature, target triple, compiler, toolchain, build script, profile | Does C depend on the selected implementation or artifact identity? |
| Dependency/runtime | lockfile, dependency version, runtime service, database, environment | Does a transitive or behavioral dependency in Cl(C) change? |
| Security/authority | trust-root rotation, signer revocation, permission change, policy revision | Is current decision use no longer authorized, even if historical evidence remains authentic? |
| Evaluator/oracle | test code, grader, metric formula, expected-output correction | Is the engineering claim affected, or only the evaluation result/grade? |
| Data/measurement | workload, dataset, seed, benchmark population, sensor calibration | Does the empirical result remain applicable to the new measured population? |
| Metadata-only | formatting, path alias, non-semantic label correction | Are evaluation inputs and support identities unchanged byte-for-byte or semantically? |

The table is an impact-analysis aid. Existing #10610 determines rerun/regrade/reuse treatment; #10630 determines engineering-claim validity. Do not merge those dispositions.

### 3.2 Claim disposition rule

For each event/claim pair, produce exactly one continuity disposition from #10630:

- **retain:** no required dependency in Cl(C) changed, or equivalence is independently established for the claim's full declared scope;
- **revalidate:** a dependency changed or equivalence is not yet proven, but the claim may remain true after a specified, sufficient check;
- **invalidate:** a governing premise is contradicted, revoked or no longer applicable; withdraw the claim from current decision use;
- **supersede:** a newer scoped claim replaces it while the earlier claim remains historically true for its original scope/time;
- **unresolved:** dependency identity, graph completeness or event semantics are insufficient to decide safely.

Never convert a dependency-unknown case into retain. Never treat a stale historical result as a failed result. Preserve the distinction between historical truth and current decision-use admissibility.

## 4. Minimal impact witness and unaffectedness witness

Every non-retain disposition must provide an **impact witness**: the changed dependency, event identity, path through Cl(C), and why the claim's use is affected.

Every retain disposition on a mixed-impact or decisive-change sequence must provide an **unaffectedness witness**:
- the relevant closure of the retained claim;
- the event's affected dependency set;
- the intersection or non-intersection analysis;
- the equivalence/scope reason that makes reuse admissible;
- the open-world remainder and why it cannot alter this claim within scope.

This is stronger than saying “different file” or “unrelated commit.” Path disjointness alone does not prove semantic independence. If a dependency is shared transitively, all dependent claims must be considered even when their own source anchors did not change.

For revalidate, the agent states the smallest sufficient observation and predicts the possible outcomes before evidence is acquired, reusing the existing sequential evidence policy. For invalidation, no further test can make the old evidence current again; a new claim/evidence lineage is required.

## 5. Invalidation propagation

When a node changes, propagate impact through the typed dependency graph using the edge semantics:

1. Mark the changed evidence/environment/policy/method node with the exact event identity and change class.
2. Traverse only edges whose semantics imply dependence for the event class.
3. Mark direct claims and downstream claims as impact candidates.
4. For each candidate claim, distinguish invalidation from revalidation, supersession and unresolved.
5. Preserve unaffected sibling claims only when their closure and non-impact witness are complete.
6. Retain historical claim/evidence records immutably; current decision use points to the new validity state.
7. If the typed edge set or closure is incomplete, propagate unresolved conservatively to the nearest dependent claims rather than guessing.

A graph traversal is a candidate generator, not the final semantic verdict. For example, a new test/grader may invalidate a historical grade without invalidating the source-level claim; a revoked signer may invalidate the authority of a receipt without proving that the recorded test never passed.

## 6. Six public development cases

Use the [Claim-Support Closure Challenge V0](CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_CHALLENGE_V0_2026-11-08.md) and its separate [public answer key](CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_KEY_V0_2026-11-08.md).

The six cases cover:
- direct source anchor edit with a derived claim;
- shared upstream dependency used by two claims with different scopes;
- build/feature transition;
- verifier-only revision;
- authority-root revocation;
- metadata-only event plus incomplete closure.

The cases are public synthetic development material, not admitted task sequences, hidden labels or a sealed holdout.

## 7. Development experiment

Compare three policy variants on identical public development event sequences and matched budgets:

- **C0 — path heuristic:** decide impact from changed path/name overlap only;
- **C1 — continuity baseline:** use the existing #10630 event and claim dispositions;
- **C2 — support-closure extension:** use typed dependency closure plus impact/unaffectedness witnesses.

C2 is useful only if it improves selective invalidation and the proof of safe retention without increasing stale-evidence use, unresolved overuse or revalidation burden beyond the existing CEB continuity objectives. This is exploratory design comparison, not a new registered metric lane or confirmatory hypothesis.

### 7.1 Diagnostic annotations (not new primary metrics)

Report separately:
- correct claim disposition per event;
- impact-witness completeness;
- unaffectedness-witness completeness;
- closure completeness and open-world remainder quality;
- false retain (stale support reused);
- false invalidation (valid claim discarded);
- unnecessary revalidation count/cost;
- unresolved/escalation correctness;
- dependency closure size and evidence lineage completeness;
- human minutes and all-in evidence acquisition cost.

Do not merge these into a new score or replace the existing CEB-CONTINUITY primary metrics registered under #10632.

## 8. Adversarial and metamorphic controls

Each case family should include controls for:

- **direct hit:** an event changes a support node and must alter the dependent claim's disposition;
- **shared ancestry:** two claims share one upstream evidence item, so impact must propagate to both;
- **scope separation:** an event is in scope for one claim but explicitly outside another claim's domain;
- **equivalent rename/move:** path or symbol changes without semantic change; retain only if content/semantic equivalence is established;
- **hidden shared dependency:** apparently separate claims rely on one dependency/policy/oracle; both must be revisited;
- **verifier-only change:** regrade is required but engineering source truth is not automatically invalidated;
- **authority revocation:** current use is invalidated even if the underlying historical result remains true;
- **unknown closure:** missing caller/dependency data must prevent an unjustified retain;
- **irrelevant metadata:** formatting or a display label does not invalidate claims if evaluation inputs and semantics are identical;
- **common-origin evidence:** copied tests/logs/docs share one provenance root and are not independent support.

## 9. Integration and ownership boundaries

- **#10630** remains the owner of evidence continuity and retain/revalidate/invalidate/supersede/unresolved evaluation.
- **#10173** remains the source/repository twin and change-impact fact authority.
- **#9112** remains the decision-use validity and evidence admissibility authority.
- **#10366/#10387** remain causal hypothesis/forecast/intervention ownership.
- **#10314/#10370** remain canonical run accounting, receipt and evidence convergence.
- **#10610** owns benchmark rerun/regrade/reuse classification.
- **#10619** owns comparative statistical licensing.
- **#10064/#10369** retain execution-isolation and independent trust/regrade boundaries.

This protocol creates no second repository graph, generic dependency engine, evidence ledger, task schema, receipt, DSSE envelope, scheduler, task admission system or metric lane. It adds an evaluation/annotation view over existing claim and evidence records.

## 10. Runner-independent sequence

1. Have two reviewers independently annotate the six public cases' claim closures and event impact paths.
2. Lock the dependency closures and disposition before revealing the key.
3. Compare C0/C1/C2 expected transitions on identical event sequences.
4. Adjudicate impact and unaffectedness witnesses separately.
5. Record missing dependency classes and open-world closure gaps.
6. Revise only ambiguous case wording and retain first-pass annotations.
7. Expand to repository/defect-family-disjoint sequences only after task admission, privacy/licensing, oracle review and contamination controls.
8. Run executable change-impact evaluation only after the #10064 and #10369 gates are separately qualified.

## 11. Claim boundary

This extension defines how to make selective evidence invalidation auditable. It does not claim current Symthaea maintains complete dependency closures, does not execute a repository graph engine, and reports no model performance, human reliability or SOTA result. A static protocol pass does not establish that a claim was correctly retained or invalidated in a real repository.
