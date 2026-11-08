# CEB Branch-Aware Evidence Continuity Protocol V0 — 2026-11-08

**Status:** Design-only research adapter. The eight public cases are synthetic development material; no candidate tasks, source code, builds, tests, repairs, runtime traces or model evaluations are executed by this protocol.  
**Owner:** [CEB-CONTINUITY-001 #10630](https://github.com/Luminous-Dynamics/symthaea/issues/10630)  
**Continuity base:** [draft PR #10631](https://github.com/Luminous-Dynamics/symthaea/pull/10631)  
**Claim-support composition:** [causal review PR #11018](https://github.com/Luminous-Dynamics/symthaea/pull/11018)  
**Authority boundaries:** #10173 owns repository-twin facts; #9112 owns decision-use validity; #10366/#10387 own causal investigation; #10314 owns canonical run receipts; #10064/#10369 own execution isolation and independent trust/regrade.

## 1. Why this extension exists

The current continuity contract represents repository evolution as an ordered event stream. That is useful for a linear sequence, but real engineering histories branch:

- developers fork parallel feature branches;
- security fixes are cherry-picked or backported into release branches;
- rebases and squash merges rewrite commit identity;
- two individually safe branches can produce an unsafe merge resolution;
- a revert can restore the final tree without erasing an earlier exposure window;
- release branches can remain divergent after main has advanced;
- authority and policy changes may apply globally even when source changes are branch-local.

A review claim can therefore be correct for one branch tip and false, out of scope, or unverified on another. A merge is not simply “the newest commit.” It creates a new combined state whose resulting tree and full parent history must be assessed.

**Paradigm shift:** move from linear evidence freshness to **branch-scoped evidence validity over a version-control history DAG**. CEB should ask not only whether evidence is current, but *for which lineage, branch tip, configuration, policy epoch and time interval it is admissible*.

The Git commit DAG is the version-control ancestry input; it is not the CEB repository dependency graph and does not establish semantic dependency. This is an evaluation adapter over existing repository-twin and evidence-continuity facts. It is **not a new repository graph**, a Git implementation, a dependency engine, an evidence ledger, a task schema, a receipt or a scheduler.

## 2. Research distinction and comparator boundary

- **SWE-Milestone (ICML 2026)** reconstructs milestone dependency DAGs from real release histories and evaluates sustained software evolution. Its paper reports that performance above 80% on isolated tasks fell to at most 38% in continuous settings across 12 models and four agent frameworks. Its unit is milestone completion and preservation of tests; it is not a branch-scoped evidence-validity benchmark. Source: https://proceedings.mlr.press/v306/deng26d.html
- **SWE-Milestone's public harness** already uses a milestone DAG and persistent workspaces. CEB should not claim novelty for DAG-shaped task dependencies alone. CEB's proposed distinction is branch-specific claim/evidence applicability, merge inheritance and historical validity. Source: https://github.com/DeepCommit-ai/SWE-Milestone
- **Incremental Program Analysis in the Wild (ASE 2025)** evaluates 4,084 commits across 20 Java projects and two tools; it reports efficiency gains alongside inconsistency and peak-memory concerns. This motivates clean-recomputation oracles and complete lifecycle cost, but it measures program-analysis results, not engineering-claim dispositions. Source: https://doi.org/10.1109/ASE63991.2025.00194
- **PR-level change-impact analysis** combines call-graph dependency analysis and history mining; its three-project feasibility study reports 7.4–22.43 seconds analysis time and user-perception measures. These are contextual methods/results, not claim-validity accuracy. Source: https://doi.org/10.1007/s10664-024-10600-2
- **ConflictBench** evaluates software merge tools on Java merge-conflict cases. It is an adjacent comparator for conflict-resolution quality, not an evidence-continuity or agentic-review score. Source: https://doi.org/10.1016/j.jss.2024.112084
- Git history can be misleading when viewed through path-limited or first-parent simplification. GitHub's Git internals discussion documents cases where a faulty merge makes a valid change disappear from a simplified file-history query. This supports full-parent-DAG controls, not the assumption that every merge is defective. Source: https://github.blog/open-source/git/gits-database-internals-iii-file-history-queries/

Do not compare any of these published measurements numerically with CEB branch-applicability or stale-evidence measures.

## 3. Evaluation objects (adapter vocabulary, not a new schema)

The authoring worksheet may refer to the following concepts while reusing the existing #10173 graph facts and #10630 claim/evidence records:

- **Commit identity:** repository identity plus commit object identifier and its ordered parent identifiers.
- **Tree identity:** exact repository tree state at a revision. A changed commit identifier does not necessarily mean changed source content.
- **Branch tip:** a named ref observed at a declared time, resolved to an exact commit.
- **Event:** a fork, commit, merge, cherry-pick, backport, rebase, squash, revert, ref movement, policy change or evidence/evaluator change.
- **Claim scope:** the branches/revisions, configuration worlds, contract/policy epoch, runtime environment and time interval for which the proposition is asserted.

A **branch scope** names the lineage/branch-tip domain within which the claim is proposed for use; it must not be inferred from a mutable branch label alone.
- **Evidence scope:** the corresponding identity and applicability domain of the source, test, build, measurement, authorization or attestation supporting the claim.
- **Merge result:** the exact resulting tree plus all parent identities and any conflict-resolution evidence.

A branch name is mutable and is never sufficient identity by itself. Bind a decision to the resolved tip and its repository/tree identity. A commit SHA is not a substitute for the tree, configuration, contract or authority scope.

## 4. Branch-aware validity rules

### 4.1 Fork and divergence

A fork inherits historical records, not automatic applicability to every descendant branch. A claim scoped to `main@R` does not become a claim about `feature@R'` merely because the feature branch descends from R.

When branches diverge, assess each branch against the events on its own ancestry and any external/global events whose declared scope reaches it. Preserve valid evidence on an unaffected sibling branch. Do not invalidate every branch because one branch changed.

### 4.2 Merge reconciliation

For a merge commit, inspect the full parent set and the resulting tree. Do not union parent evidence or assume that two safe parent states imply a safe merged state.

For each inherited claim:
1. verify whether its declared branch/scope includes the merge result;
2. traverse the relevant evidence-support closure and both parent lineages;
3. inspect changes introduced by the merge resolution itself;
4. check interaction effects between the two sides;
5. retain only with an unaffectedness/equivalence witness for the resulting tree and scope;
6. otherwise revalidate, invalidate, supersede or remain unresolved under #10630.

A clean full-recomputation over the merged result is the correctness oracle for any incremental branch-aware update.

### 4.3 Cherry-pick and backport

A cherry-pick or backport creates a distinct commit in a distinct context. Equal patch identity, commit message or changed-line similarity does not prove equivalent behavior. The destination branch may have different callers, defaults, dependencies, feature selection, contracts or neighboring fixes.

Transfer evidence only after an equivalence argument covers the claim's complete dependency closure and destination scope. Otherwise require a destination-branch revalidation or mark the transfer unresolved.

### 4.4 Rebase and squash

Commit identities may change while the resulting tree or relevant behavior remains equivalent. Do not invalidate all behavioral claims solely because a commit SHA changed. Conversely, do not retain an artifact-identity, build-provenance or exact-commit claim solely because a patch message or visible diff appears unchanged.

Separate source-behavior claims from commit/artifact/provenance claims. Preserve the historical identity mapping and record which proposition is retained, superseded or requires a new build/regrade under #10610.

### 4.5 Revert and reapply

A final tree can match an earlier safe state while the intervening history records a vulnerable release or operational exposure. Revert does not erase historical events, approvals, observations or incidents.

Score current-state safety and historical-exposure claims separately. A reapply is a new event and must not silently inherit an approval that was scoped to the pre-revert interval.

### 4.6 Branch-local versus global authority

Source/configuration changes may be branch-scoped. Trust-root revocation, signer compromise, organization-wide policy changes or legal restrictions may have a global or explicitly multi-branch scope.

The evaluator must declare the authority event's scope. A branch-local configuration change must not be propagated globally; a genuinely global revocation must not be contained to whichever branch happened to receive the latest source commit.

### 4.7 Missing ancestry and shallow history

If a parent, merge resolution, ref observation, or relevant event is missing, do not silently replace the missing edge with a first-parent guess. Mark the affected transfer/retention decision unresolved, request the minimum discriminating evidence, or escalate.

## 5. Required claim/evidence decision record

For every consequential claim used at a branch tip, the annotation should capture:

1. proposition and claim kind (behavior, security, build, artifact identity, performance, authorization or historical incident);
2. exact branch tip, commit/tree identity and all known parent identities;
3. declared branch/revision and configuration/policy/time scope;
4. support-closure identity, counter-evidence and relevant branch events;
5. whether the claim is branch-local, destination-transferable or globally scoped;
6. continuity disposition from #10630: retain, revalidate, invalidate, supersede or unresolved;
7. impact witness for every non-retain disposition;
8. unaffectedness/equivalence witness for every retain or cross-branch transfer on a mixed-impact case;
9. minimum discriminating evidence and predicted outcomes for revalidation;
10. cost, reversibility and human-escalation requirements.

A path overlap is not proof of semantic impact. A path non-overlap is not proof of unaffectedness. A commit ancestry relation is not proof that evidence transfers to every descendant branch.

## 6. Public development challenge

The companion challenge contains eight synthetic anchors:

- BHC-S01: forked branch, branch-local API/security change;
- BHC-S02: first-parent simplification hides a relevant second-parent change;
- BHC-S03: cherry-pick/backport with patch identity but different destination context;
- BHC-S04: individually safe parents produce an unsafe merge resolution;
- BHC-S05: rebase/squash changes commit identity while source-tree and claim identities differ;
- BHC-S06: revert/reapply restores a safe tip but does not erase a historical exposure;
- BHC-S07: global authority revocation versus branch-scoped policy/configuration change;
- BHC-S08: missing second parent and incomplete ancestry require unresolved, not retain.

The public answer key is deliberately separate for annotation workflow convenience, but it is **not secure blinding**. These cases are public synthetic development anchors, are not independent statistical tasks, and are **not a sealed holdout** or evidence of candidate performance. They must never be counted as eight independent confirmatory observations.

## 7. Development experiment and clean oracle

Compare matched policies on identical public event histories and budgets:

- **B0 — linear-tip baseline:** use only the current named branch tip and a linear recency model.
- **B1 — first-parent/path baseline:** include first-parent ancestry and path overlap, but no full-parent or claim-scope reasoning.
- **B2 — full-DAG baseline:** use all known parents and event identities, but no claim-specific branch-scope/evidence-closure semantics.
- **B3 — branch-aware continuity:** combine full-parent ancestry, claim-support closure, scope, merge reconciliation, historical intervals and #10630 dispositions.

The independent reference should be built by reviewers from the full event history and an independently produced clean full-recomputation. Incremental branch-aware updates must match the clean result or return unresolved. The full oracle is not candidate-generated gold.

Keep model, prompt, tools, visible history, context, event order, compute, inference/tool budgets and evaluator fixed across matched arms. Separate agent-visible current history from grader-only future events and expected dispositions.

### 7.1 Existing CEB-CONTINUITY primary measures

Map results onto the existing seven measures; add no primary metric:

- sequence-complete rate;
- evidence-disposition macro-F1;
- stale-evidence use rate;
- false-invalidation rate;
- revalidation cost per accepted sequence;
- repair survival rate;
- evidence-lineage completeness.

### 7.2 Branch-specific diagnostic annotations

Report separately, without creating another lane or opaque composite:
- branch-applicability precision and recall;
- cross-branch evidence-transfer error rate;
- merge-inheritance error rate;
- incorrect first-parent-only disposition rate;
- false global propagation and missed global propagation;
- historical-exposure preservation accuracy;
- missing-ancestry unresolved precision/recall;
- merge-resolution inspection completeness;
- cost by fork count, merge count, branch depth and dependency fan-out.

Always include invalidate-all, retain-all, always-unresolved, linear-tip and first-parent-only baselines. An agent must not score well by refusing every cross-branch decision or by withdrawing all prior evidence.

## 8. Prospective target ladder (not achieved results)

Use a blinded feasibility pilot to estimate label reliability, difficulty, cluster structure and variance before freezing confirmatory targets. Candidate aspirations:

- zero critical stale-evidence transfers on the declared synthetic security/authority controls;
- at least 95% branch-applicability precision on adjudicated branch/claim pairs;
- at least 90% recall for required invalidation/revalidation on the critical branch-change slice;
- no increase in false invalidation or indiscriminate unresolved dispositions beyond a pilot-derived non-inferiority margin;
- at least 20% lower full lifecycle revalidation cost than clean full-recomputation, at non-inferior sequence-complete rate and without more stale-evidence use;
- positive paired sequence-complete improvement over the strongest matched baseline on repository-family-disjoint confirmatory histories, with cluster-aware intervals and a preregistered practical margin.

These are prospective research targets, not current CEB results, industry standards or guarantees. Do not claim superiority over SWE-Milestone, ConflictBench, incremental program analysis or a code-review leaderboard based on incomparable task units.

## 9. Sampling, independence and statistical licensing

The first expansion should produce 20–24 design-only branch-history blueprints across at least five event strata. The eight public anchors are protocol-debugging material, not the full feasibility sample.

Split by repository root/family and branch-event template, not by individual commit or sibling branch. Branches from the same repository root, repeated cherry-picks of one patch and variants of one merge scenario are clustered observations, not independent n. Report sequence-level and repository-family-level counts. Use paired treatment assignments and cluster-aware uncertainty; control multiplicity for secondary diagnostics.

Two reviewers must independently annotate the public development pack before seeing the public key. Preserve original annotations, timestamps, adjudications and disagreement types. Public-key exposure means this exercise estimates rubric usability, not blinded inter-rater reliability.

## 10. Ownership, integration and no-runner work

- #10630 remains the sole owner of evidence-continuity claims and dispositions.
- #10631 remains the continuity protocol/registration integration.
- #10173 remains authoritative for repository graph and change-impact facts; this protocol consumes its full parent/branch facts and does not create a second repository graph.
- #9112 owns decision-use validity and evidence admissibility.
- #10366/#10387 own causal forecasts, interventions and investigation selection.
- #10314/#10370 own canonical receipts, accounting and evidence convergence.
- #10610 owns rerun/regrade/reuse classification.
- #10619 owns statistical licensing and multiplicity.
- #10064/#10369 retain candidate isolation and independent pristine-regrade/trust gates.

Before those gates are qualified, permissible work is limited to protocol authoring, synthetic non-executable cases, annotation guidance, static validators, matched baseline definitions and design-only analysis. No candidate-controlled code, build, test, PoC, repair, runtime trace, graph engine, grader or performance evaluation is run by this tranche.

This extension creates **not a new metric lane** and adds no primary metric, task schema, release manifest, receipt, DSSE envelope, generic dependency engine, evidence ledger or scheduler. It is an annotation/evaluation adapter over existing CEB owners.

## 11. Acceptance and claim boundary

- [ ] Eight public cases and separate public key have the required branch/merge controls.
- [ ] Static validator checks case coverage, scope, merge-parent, historical-event and claim-boundary wording with negative self-tests.
- [ ] The existing CEB contract workflow invokes the validator on exact PR head.
- [ ] No new primary metric, graph, receipt or execution authority is introduced.
- [ ] Two independent human reviewers complete the development packet before calibration claims are made.
- [ ] Confirmatory sampling is repository-family/branch-template disjoint and separately adjudicated.

A passing documentation/static workflow validates this authored contract only. It does not establish that a candidate agent correctly handles Git histories, that branch-aware closure is complete, that human reviewers agree, that CEB is competitive with SOTA, or that execution/regrade is isolated. Those claims remain blocked by human annotation, task admission, #10064 and #10369 as applicable.
