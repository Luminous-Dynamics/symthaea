# PITB typed proof-repair track (2026-10-31)

**Status:** new research specification and operation schema. No PITB proof-repair dataset, model repair outcome, or benchmark comparison has yet been executed. The schema does not authenticate receipt bytes or independently prove an edit correct.

## Why this is a separate research direction

The October 2026 BlueprintRepair / BlueprintTrace work evaluates typed local operations on failed Lean proof blueprints. It reports 142 controlled failures with accepted and rejected trajectories. In its matched setup, the tested interfaces solved roughly similar numbers of localized failures, while typed repair was 1.30x cheaper per solved state than exact patching and 2.06x cheaper than full module rewriting. The key design lesson is not that typed repair always wins; it is that constrained local edits can reduce authoring cost while preserving a target boundary. [Khrulev, MathNLP 2026](https://aclanthology.org/2026.mathnlp-main.3/).

PITB should add a **repair endpoint**, kept separate from:
- support-antichain recovery (which routes remain sufficient under controlled ablation);
- insight transfer utility (whether an insight improves held-out target solving);
- end-to-end theorem proving (whether the target can be solved at all).

The new endpoint asks: when a proof blueprint is wrong or incomplete, can a prover make the smallest valid repair, preserve the exact target theorem, and avoid damaging unrelated proof nodes?

## Proposed trajectory contract

A repair episode starts from a content-addressed proof blueprint containing an immutable target theorem node and a directed dependency graph of intermediate obligations and supporting lemmas. The system submits schema-checked operations and retains both accepted and rejected attempts.

The operation schema is data/benchmarks/proof-insight-transfer-v0/typed-proof-repair-operation.schema.json. Each operation binds:
- exact target theorem ID and target statement digest before and after;
- blueprint digests before and after;
- the typed operation and the exact node(s) it affects;
- declared support, model decision, Lean check state, and run/operation receipts;
- a rejection reason for rejected edits.

The schema handles structural validation. A separate trajectory validator must also enforce:
1. target_statement_sha256_before equals target_statement_sha256_after for every operation and every episode;
2. the target theorem ID never appears in affected_node_ids;
3. operation node references exist in the pre-state and the state transition matches the declared edit;
4. each operation's before digest equals the prior applied operation's after digest;
5. rejected or abstained operations do not mutate the graph;
6. applied operations pass Lean checking and the semantic regression suite;
7. declared blueprint lemma dependencies match the accepted proof's checked support receipt;
8. all receipt bytes are opened and authenticated independently.

A model must not get credit for silently rewriting the target theorem or replacing a failed target with an easier proposition. Failed, rejected, abstained, timed-out, and budget-exhausted trajectories remain in the denominator.

## Controlled failure and mutation cohort

Build from manually reviewed training theorem families first. Create target-preserving failures by controlled edits to blueprint nodes/edges and proof obligations, not by changing the target proposition.

Initial mutation families:
- remove a required dependency edge;
- substitute a nearby but directionally wrong lemma;
- leave a stale or unavailable dependency;
- duplicate or over-strengthen an intermediate obligation;
- underconstrain an obligation so a later proof step no longer composes;
- introduce a dependency cycle;
- hide a transitive prerequisite while keeping the direct lemma name visible;
- add unrelated nodes and plausible distractors;
- mutate one non-target node while holding sibling subgraphs fixed.

For each episode, preserve a known accepted repair, at least one rejected near-miss, and a baseline trajectory. Use two independent reviewers for intended target correspondence, mutation validity, and repair labels. Quarantine any episode where the failed blueprint no longer has the exact target or where the intended repair is ambiguous.

## Matched comparison

Use the same starting blueprint, source visibility, checker feedback, model, operation/token/time limits, and number of episodes for:
1. typed local operation interface;
2. exact textual source patch;
3. complete module rewrite;
4. retrieval plus no automatic edit;
5. deterministic graph-repair heuristics;
6. optional human-authored oracle repair as an upper-bound reference.

Use one repair episode per state/interface for the primary comparison. Additional attempts form a separate budget-scaling curve. Select all models, prompts, repair interfaces, and thresholds on validation only.

## Metrics

| Metric | Definition / purpose |
|---|---|
| Verified repair rate | Exact target passes Lean and downstream semantic regression gates |
| Target-preservation rate | Exact target digest is unchanged throughout the entire episode |
| Invalid-edit rejection recall | Fraction of registered invalid edits rejected before graph mutation |
| False rejection rate | Valid repair operations incorrectly rejected |
| Exact repair-plan rate | Recovered accepted operation sequence modulo a preregistered equivalence |
| Locality ratio | Non-target nodes changed divided by the independently adjudicated minimum affected-node count |
| Collateral regression rate | Previously passing unaffected sibling obligations that fail after repair |
| Dependency declaration precision/recall | Declared blueprint supports versus authenticated accepted-proof dependencies |
| Cost per verified repair | Tokens, calls, wall time, checker time, memory, and money |
| Recovery frontier | Verified repairs versus progressively larger matched budgets |

Report rates and cost jointly. A system that rejects everything can score well on invalid-edit safety but has zero useful repair coverage. A system that rewrites the whole module may solve a case but should pay the measured edit and collateral-regression cost.

## Proposed qualification goals (not achieved)

| Dimension | Future target |
|---|---:|
| Target theorem identity | 100% preservation; zero accepted target mutations |
| Applied operation validity | 100% Lean-checked |
| Invalid-edit rejection recall | >= 0.99 |
| False rejection on valid edits | <= 0.05 |
| Unaffected-sibling regression | 0 on the finite adversarial suite |
| Dependency declaration F1 | >= 0.95 |
| Verified repair rate | non-inferior to patch/rewrite within 2 pp |
| Cost per verified repair | >= 20% reduction against the strongest non-typed baseline |
| Temporal/project holdout | positive paired result on unseen project components |

These are proposed gates. The 142-episode BlueprintTrace result is an external context point, not PITB's denominator. Use a later or independently curated task set for a confirmatory replication. For project-scale validation, VeriSoftBench demonstrates the challenge: 500 tasks from 23 repositories, with reported success of 41% under curated context versus 35% using raw repository context. The benchmark emphasizes layered library, project, local, and theorem context. [VeriSoftBench](https://utopia-group.github.io/VeriSoftBench/).

## How this connects to support antichains

The support-antichain track identifies alternative minimal sufficient contexts for an exact target under a fixed solver policy. The repair track changes the proof blueprint when a required dependency is missing, wrong, or overconstrained.

A promising combined agent would:
1. predict the support antichain for the current proof-state/blueprint context;
2. detect which support route was invalidated by a controlled mutation;
3. choose a local typed graph edit or an alternate route;
4. prove the unchanged target and rerun affected successor obligations;
5. record accepted and rejected trajectories with complete cost and provenance.

This creates a useful division of labour for Symthaea: HDC proposes structurally similar insight candidates; dependency-graph search discovers multi-hop support; the liquid controller tracks uncertainty and selects whether to repair, replan or abstain; Lean validates every applied change. A support prediction or HDC similarity score never substitutes for formal acceptance.

## Competitive positioning

LeanGraphSearch has established that multi-hop formal dependency traversal can improve retrieval over direct semantic similarity; its reported MathlibQR Recall@100 is 94.7% versus 82.7% for LeanSearch v2 on its fair subset. PITB should use it as a retrieval baseline and investigate whether typed repairs can operationalize the retrieved dependency structure. [LeanGraphSearch paper](https://arxiv.org/abs/2610.22339) · [results](https://pluslabnlp.github.io/LeanGraphSearch/).

T² shows compilation can overstate semantic correctness: its five-repository set contains 2,206 problems and the reported best Testing Accuracy is 38.9%. Therefore repair acceptance must require both Lean proof checking and downstream successor/invariant tests, not merely a locally compiling edit. [T², ACL 2026](https://aclanthology.org/2026.acl-industry.150/).

The distinctive PITB opportunity is a combined benchmark for **support prediction + target-preserving repair + semantic regression + cost efficiency**. No claim is made that this combination is already SOTA; it is a falsifiable research proposal that can be evaluated against typed-edit, patch, rewrite, retrieval and repository-prover baselines.
