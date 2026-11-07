# CEB Review Cross-Framing Anchor Protocol V0 — 2026-11-07

**Status:** Public research design; no task is admitted, no model is evaluated and no sealed holdout is created.  
**Owner:** Existing CEB-REVIEW / CEB-RESOLUTION / CEB-REPO / CEB-HUMAN measurement owners; #10419, #10723, #10367.  
**Purpose:** Separate engineering capability from the amount and style of information the benchmark supplies.  
**No new primary metric lane or composite score:** each condition uses its existing CEB lane and its own registered measures.

## 1. Research question

How much of an engineering agent's apparent capability comes from solving the underlying engineering problem, and how much comes from the benchmark's task framing and supplied information?

Existing benchmark families expose very different amounts of information:

- An issue-resolution task gives a natural-language issue, often a reproducer or test.
- A realistic coding-assistant task may begin with an incomplete user request and require clarification.
- A code-review task gives a proposed diff but does not disclose the defect.
- An open-world repository sweep supplies a repository without bug locations or descriptions.
- A deterministic repair task may provide the specific failing test and let a validator decide whether the fix works.

Those scores answer different questions. A high issue-resolution rate does not establish autonomous defect discovery; high code-review recall does not establish patch repair; a deterministic test pass does not establish causal diagnosis; and a user-style prompt may reveal interaction failures absent from formal issue tasks.

**Central hypothesis:** on a shared source/defect anchor, changing only the task information/framing can materially change discovery, diagnosis, repair, clarification and abstention performance. CEB should measure those changes explicitly rather than treating benchmark percentages as interchangeable evidence of general engineering intelligence.

## 2. Research motivation

Garg, Steenhoek and Huang's October 2025 paper, *Saving SWE-Bench: A Benchmark Mutation Approach for Realistic Agent Evaluation*, converts formal GitHub issue descriptions into realistic user-style queries based on observed developer interactions with chat-based agents. It reports that formal public benchmark results overestimated performance for some evaluated models by more than 50% relative to the mutated-query setting, with roughly 10–16% differences on its private benchmark. The reported effects are specific to the paper's models, benchmark subsets and mutation procedure; they are not a universal correction factor. [Microsoft Research overview](https://www.microsoft.com/en-us/research/publication/saving-swe-bench-a-benchmark-mutation-approach-for-realistic-agent-evaluation/) · [arXiv paper](https://arxiv.org/abs/2510.XXXXX).

SWE-sweep then removes issue hints entirely and asks agents to find and fix multiple concurrent bugs in real repositories. Its October 2026 paper reports 4,068 bugs across 100 repositories and 22 languages; its strongest system resolves 4.7% in the paper evaluation, while the current live board shows a different model/harness snapshot at 9.5%. [Paper](https://swesweep.com/paper) · [live leaderboard](https://swesweep.com/).

LoopControlBench uses the opposite information extreme: a specified failing maintainer test and allowlisted source edit, with a deterministic test gate deciding whether the task is complete. Its reference 46/70 = 65.7% applies to its original certified Python task pool and one-attempt policy, not to the later 89-task pool or general software bugs. [Repository](https://github.com/kltownsend/loopcontrolbench).

**CEB implication:** task framing is an experimental factor. Do not compare the headline percentages from these benchmarks directly. Build paired anchor variants that preserve the underlying source/defect and separately measure what the agent discovers, explains, repairs, asks and abstains from doing.

## 3. Common anchor definition

A **cross-framing anchor** is one independently adjudicated defect (or a declared set of interacting defects) tied to immutable repository states and a frozen behavioral contract.

Required anchor identity:
- repository and exact source tree/commit identity;
- parent, introducing and fixed source states where available;
- target defect provenance and independently reviewed change disentanglement;
- frozen requirement/specification version;
- causal mechanism: trigger, path/state, operation, violated invariant and consequence;
- relevant dependency/configuration/toolchain/environment identity;
- target-specific tests/oracle provenance and scope;
- residual-path and neighboring-mechanism controls;
- known uncertainty, confounders and applicability boundaries.

Use the existing counterfactual-control qualification protocol before relying on historical P/I/F or constructed N/M/IRR variants. A fix commit is not a minimal intervention by default. No anchor becomes admitted or sealed because it appears in this authoring document.

### Two anchor strata

**Stratum A — Single-defect common anchor.** Exactly one target defect is independently selected for the cross-framing comparison. This supports the cleanest paired analysis across review, issue repair, realistic user request and test-guided repair.

**Stratum B — Multi-defect repository anchor.** Several independently validated defects coexist in one source snapshot and may interact. This supports open-world repository sweep and task prioritization. It is not directly pooled with Stratum A; defect interactions and shared source files are retained.

The unit is the originating repository/defect family, not each prompt variant. Multiple framings of one source anchor are paired conditions, not independent samples.

## 4. Information/framing conditions

Use only conditions that can be authored without revealing protected gold labels. The source tree and behavioral contract remain fixed within an anchor; vary only the declared information, task objective and permitted output.

| Condition | Candidate receives | Candidate objective | Existing CEB lane(s) |
|---|---|---|---|
| F0 — Formal issue-conditioned repair | Formal issue description; only task-permitted context | Repair the named defect | CEB-RESOLUTION |
| F1 — Realistic user request | Human-authored/mutated ordinary developer request; no benchmark terminology | Clarify as needed and solve the user's actual request | CEB-HUMAN / CEB-RESOLUTION |
| F2 — Diff-only review | Parent→introducing diff and repository context; no issue, fix or target label | Produce up to five review findings | CEB-REVIEW |
| F3 — Repository-only single-defect discovery | Bug-bearing repository snapshot; no issue, location, reproducer or target hint | Discover and explain the target defect; do not patch unless the protocol explicitly enables it | CEB-REPO / CEB-REVIEW |
| F4 — Failing-test-guided repair | Bug-bearing repository and a maintainer test that fails on the base | Repair only allowlisted source paths; validator decides completion | CEB-RESOLUTION |
| F5 — Open-world multi-bug sweep | Multi-defect repository snapshot; no bug descriptions or locations | Discover and repair as many independently gold-supported bugs as possible | CEB-REPO / CEB-RESOLUTION |

F3 and F5 must not be conflated: F3 holds one target defect in focus for paired attribution; F5 evaluates a repository-level multi-defect objective with interaction and prioritization effects.

F1 must be human-authored or independently reviewed for realism. Do not let the candidate model generate its own user prompt, gold label or evaluation input. Preserve the mutation source, transformation procedure and human review; synthetic paraphrase is not equivalent to telemetry-derived real user distribution.

## 5. Paired experimental design

For each eligible anchor:
1. Freeze the source state, oracle/requirement, environment, tool permissions, model/harness version and total cost budget.
2. Create only the approved framing variants. An independent reviewer confirms that each variant preserves the intended underlying defect and does not accidentally disclose hidden labels.
3. Randomize condition order and use isolated fresh contexts. No model sees one framing and then another variant of the same anchor in the same conversation/session.
4. Disable unauthorized web/network access and solution retrieval for blind conditions. Apply the same repository-history access policy across conditions.
5. Run paired conditions with the same agent/model/harness when estimating framing effects. Cross-model comparisons require the same condition, task snapshot and budget.
6. Keep output contracts distinct: review findings, repair patches, clarification questions and open-world bug lists are not coerced into one format.
7. Independently adjudicate findings and patches against the same frozen requirement/oracle; preserve non-target discoveries separately.
8. Cluster and bootstrap by originating repository/defect, never by the number of prompts or counterfactual variants.

A missing task condition is missing data, not a zero or a failure. An agent's failure to ask a question when needed is adjudicated under the existing CEB-HUMAN question/abstention contracts; do not add a new metric.

## 6. Confounders and leakage controls

| Threat | Required control |
|---|---|
| Different source state between framings | Exact tree/diff binding; no silent branch-head substitution |
| Different issue difficulty or target defect | Same anchor and frozen target, paired by originating defect |
| Fix/issue history leaks target location | Apply a single, predeclared history/network policy across blind conditions |
| Formal issue text includes more detail than user prompt | Record information fields present/absent; do not claim only wording changed if information content changed |
| Prompt mutation changes requirement meaning | Independent semantic-equivalence review; reject or mark altered prompts |
| Test reveals exact location/implementation | Separate test-guided condition; do not mix with source-only discovery |
| Model remembers another variant | Fresh isolated context and randomized order |
| Open-world variants contain different bug counts | Separate Stratum B; report defect-count distribution and interactions |
| Hidden tests reward patch memorization | Provenance, contamination audit, clean controls and pristine regrade under existing gates |
| Same evaluator scores different outputs inconsistently | Freeze lane-specific graders/rubrics; blinded human audit and evaluator versioning |
| Compute/human cost differs | All-in budget ledger; report per-condition resource use, retries and human effort |
| A task condition is absent or invalid | Preserve missing/invalid/infrastructure states distinctly; do not reclassify denominator after seeing results |

The task framing itself is the treatment. If information content changes (for example, a failing test is supplied only in F4), the comparison estimates the effect of the *information package*, not a pure natural-language paraphrase effect. Use paired subcontrasts that change wording only when literal information equivalence can be established.

## 7. Primary measures and analysis — no new metrics

Use the existing CEB metric matrix unchanged:
- **F2 review:** `defect_recall_at_5`, `actionable_precision`, `reviewer_burden`.
- **F0/F1/F4/F5 repair:** `resolve_rate`, `cost_per_accepted_repair`, `regression_rate`.
- **F3 repository understanding/discovery:** `source_supported_claim_precision`, `claim_recall`, `unsupported_authority_rate`; map exact candidate outputs through existing task-specific adjudication.
- **F1 interaction/clarification:** existing CEB-HUMAN question precision, blocker recall, selective escalation, paired act/abstain accuracy and reviewer burden where applicable.

Do not create a cross-framing composite score. Do not average review recall, repair resolve rate, repository claim recall and human question precision into a single “engineering intelligence” number.

### Predeclared contrasts

- **Formal-to-realistic interaction gap:** F0 versus F1 on the same anchor where information content and objective can be matched; otherwise state which information fields differ.
- **Review discovery gap:** F2 target defect detection versus F3 target discovery, acknowledging that F2 receives a diff and F3 receives a repository snapshot.
- **Issue-hint contribution:** F0 versus F3 on the same source anchor, separating issue-conditioned repair from source-only discovery.
- **Failing-test contribution:** F0 versus F4 when a test is a legitimate additional information source; do not label the effect as pure prompt framing.
- **Open-world generalization:** Stratum B F5 outcomes, analyzed separately from single-defect paired anchors.
- **Human interaction contribution:** F1 with and without clarification/ask opportunities, using existing human-assistance metrics.

Report paired task-level differences with repository/defect-cluster confidence intervals, all-in cost, per-stratum denominators and missingness. Pre-register primary contrasts, multiple-comparison correction and power calculation before the holdout is accessed.

## 8. What would count as a paradigm-level result?

A meaningful result is not merely that one model ranks first on one framing.

The strongest evidence would show:
- the same agent's performance changes materially under realistic or lower-information framing, with the cause attributable to information conditions rather than task mix;
- a method improves discovery under F2/F3/F5 without eroding precision or causing unbounded false positives;
- the agent can detect when it lacks enough information and ask a high-value question instead of fabricating certainty;
- improvements transfer across repositories, languages and defect families;
- a repair that passes the target test also preserves pre-existing behavior and survives independent regrade;
- full human + model + tool + judge cost is favorable, not merely token cost.

Prospective goals remain those registered in the relevant CEB lanes. Cross-framing effects are analyses of existing measures, not new primary metrics.

## 9. Runner-independent work sequence

1. Build a public design-only anchor worksheet with exact source/defect identity fields.
2. Select three public synthetic examples that can be expressed in F0–F4 without executing code; do not create a sealed task.
3. Have two independent human authors review F0/F1 prompt equivalence and F2/F3 target leakage.
4. Fill the confounder matrix and mark each framing as same-information, additional-information or reduced-information.
5. Specify the lane-specific output and existing measurement IDs before any candidate run.
6. Freeze a matched-cost experimental analysis plan and repository-cluster power calculation using development-only assumptions.
7. Only after task admission and #10064/#10369 gates are qualified, decide which anchors may enter executed evaluation.

## 10. Claim boundary

This protocol is an experiment design, not a benchmark result. It creates no task IDs, admitted tasks, sealed holdout, gold labels, hidden tests, runner or metric lane. It does not claim that any current agent has a framing gap of a particular size. All candidate-controlled execution and independent regrade restrictions remain unchanged.
