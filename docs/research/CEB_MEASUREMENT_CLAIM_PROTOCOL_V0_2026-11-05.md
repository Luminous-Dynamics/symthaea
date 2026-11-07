# CEB Measurement and Comparative Claim Protocol V0 — 2026-11-05

**Status:** Design-only. No CEB performance result is claimed.  
**Machine-readable matrix:** [`benchmarks/ceb/measurement-matrix.v0.json`](../../benchmarks/ceb/measurement-matrix.v0.json)  
**Schema:** [`benchmarks/ceb/measurement-matrix.schema.json`](../../benchmarks/ceb/measurement-matrix.schema.json)  
**Execution gates:** [#10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064), [#10369](https://github.com/Luminous-Dynamics/symthaea/issues/10369)

## 1. Paradigm shift: engineering quality is a vector

SWE-Bench Pro V2 Full is now near its ceiling: Scale's live board reports Opus 5 + Claude Code at 99.40 ± 0.40, Gemini 4 Argon + mini-swe-agent at 99.22 ± 0.45 and Fable 5.1 + Claude Code at 99.10 ± 0.50 across 642 tasks and 11 repositories. Its separate HARD subset has only 51 tasks and must not be merged with Full.

The benchmark refresh itself illustrates that leaderboard scores depend on evaluator quality: Scale removed 89 invalid tasks, corrected 69 contradictory task statements, disabled agent web access after fix-commit retrieval, added pristine-image regrading and requires reference-patch pass / empty-patch fail. The live page still identifies a trusted model-endpoint relay and verifier execution of candidate-controlled code as residual risks. Source: https://labs.scale.com/leaderboard/swe_bench_pro_public_v2

Other current reference points reveal distinct capability ceilings:
- SWE Atlas Codebase QnA: 124 tasks; Opus 5 + Claude Code 63.17 ± 5.01.
- SWE Atlas Test Writing: Fable 5.1 + Claude Code 67.04 ± 5.33.
- SWE Atlas Refactoring: 70 tasks; GPT-6 Astra + Codex 59.05 ± 6.43.
- ContextBench: 1,136 tasks / 66 repositories / 8 languages; best reported Recall 0.779, Pass@1 77.6%, and Context F1 0.389 are not all from one system.
- Agent Security League: 91.6% maximum FuncPass and 36.9% maximum SecPass come from different configurations.
- CRBugBench: the best reported system surfaced 71.9% of human-verified bugs within its top five comments.
- SWE-fficiency: 498 tasks across 9 repositories; top evaluated agents averaged below 0.23× expert speedup.
- Epoch AI's Terminal-Bench 4.0.0 review identified publicly noted scoring defects in 30/66 tasks (45.5%), including pass-channel reward hacking, answer leakage and false negatives.

These results are not directly rankable. Their exact sources, metrics, task scopes and caveats are registered per lane in the matrix. The snapshot must be refreshed before each formal comparison.

## 2. The CEB portfolio

The machine-readable matrix defines 14 lanes: repository resolution, repository comprehension, context retrieval, test generation, refactoring, precise debugging, pre-merge review, secure correctness, Rust engineering, optimization, human-agent complementarity, evaluator integrity and documentation executability.

Each lane records its decision question, primary metrics, external comparators, baseline arms, evidence artifacts, independent authority, hard gates and holdout design. It contains 56 primary metrics across 14 lanes. The matrix comparator/source counts and the separate normalized external-observation counts must be read from their own versioned registries rather than conflated. The registry includes 15 external benchmark/methodology sources plus one internal documentation-audit implementation reference; the normalized external-observation registry records the 15 external sources separately. The matrix is design-only and contains no CEB performance score.

### Selective autonomy comparator — AgentAbstain

The external-observation registry now includes AgentAbstain as a second comparator for CEB-HUMAN. Its benchmark contains 263 near-identical should-act/should-abstain task pairs across 42 executable environments. The live leaderboard reports Gemini 3.1 Pro + Google ADK at 90.5% act-side accuracy, 65.4% abstain-side accuracy, 59.5% paired accuracy and 65.7% conditioned abstention rate (CAR). Paired accuracy is the primary measure because both variants must be correct; the component metrics must remain separate. Source: https://agentabstain.github.io/ . This is contextual evidence, not a matched CEB comparison.

## 3. Claim levels

**Contextual reference:** different tasks, metrics, versions, harnesses, budgets or rubrics may motivate a target, but cannot support a direct superiority claim.

**Replication:** reproduce the exact public version, task split, metric, harness and scoring semantics. Publish deviations, source revisions, exclusions and uncertainty. A replication is not automatically an improvement.

**Matched comparison:** same tasks, instructions, source snapshots, oracle semantics, metric definitions and resource budgets. Hold model, harness, prompt, tools, turns, tokens, wall time, compute and cost constant unless a preregistered ablation explicitly varies one.

**Superiority:** additionally require a preregistered primary endpoint and stopping rule; at least three independent seeds (five where pilot variance warrants); paired confidence intervals or a preregistered non-inferiority test; repository/root-cause clustering; task-level outcomes and raw denominators; pristine independent regrade; exact evidence digests; and no critical gate violation.

Do not say “beats SOTA” because a percentage is numerically higher on another set, because intervals overlap, or because different configurations lead different metric columns.

## 4. Required baseline arms

For each applicable lane compare:
1. deterministic tools / conventional static analysis;
2. the same foundation model with a minimal harness;
3. the same model with a strong industry-style harness;
4. the same model plus exactly one CEB component;
5. the Symthaea-native path where no frontier foundation model is needed;
6. lesion, sham and rescue ablations for causal claims;
7. human-only, agent-only and human+agent for complementarity claims.

If a baseline cannot be matched, classify the result as contextual. Do not infer causal contribution.

## Selective-autonomy measurement within CEB-HUMAN

The CEB-HUMAN lane now covers both human-agent complementarity and selective autonomy. In addition to assisted defect recall, risk/coverage and escalation precision, it explicitly measures paired should-act/should-abstain accuracy, act-side accuracy, abstain-side accuracy and conditioned abstention (CAR). The paired endpoint requires both counterfactual variants to be correct. AgentAbstain's 59.5% paired score, 90.5% act-side score, 65.4% abstain-side score and 65.7% CAR are stored as separate contextual observations. CEB's prospective targets are not direct cross-benchmark superiority claims.

## 5. First qualification targets

| Lane | Proposed target |
|---|---|
| CEB-EVAL | Zero critical false accepts; both negative and positive controls; false-accept and false-reject uncertainty reported separately |
| CEB-DOCS | Zero unannotated missing Cargo manifests, package selectors or binary targets; dynamic references remain explicit gaps |
| CEB-REPO | Supported-claim precision ≥0.95; claim recall ≥0.90; zero fabricated compiler/runtime authority |
| CEB-CONTEXT | Recall@10 ≥0.90 and Context F1 ≥0.50 on sealed holdout; downstream success and cost reported |
| CEB-DEBUG | Bug Recall@5 ≥0.90 and edit precision ≥0.85 on qualified holdout |
| CEB-REVIEW | Recall@5 ≥0.85 and actionable precision ≥0.80 on sealed holdout |
| CEB-SECURE | Joint functional+secure pass ≥0.65 on sealed Rust-heavy tasks; zero critical security false accepts |
| CEB-OPT | ≥0.80× expert-normalized speedup with semantic non-regression |
| CEB-HUMAN | +5 pp confirmed-defect recall at equal effort, or non-inferior recall with ≥20% lower review time |

These are proposed thresholds, not measured CEB outcomes. Sample sizes and confidence bounds must be selected from pilot variance and fault-family dependence.

## 6. Hard gates

- Critical security/evaluator false accepts are disqualifying; no aggregate can cancel them.
- CEB-OPT must reject benchmark-input memorization, harness-aware bypass and in-scope workload overfitting while accepting contract-permitted caching that preserves semantics and resource limits.
- CEB-TEST must trace each specification-required feature interaction to acceptance evidence. Hidden tests may operationalize frozen requirements but must not invent new ones; valid semantic alternatives must pass. Report raw measurements and integrity adjudication separately.
- Positive controls must pass; a reject-everything verifier cannot qualify.
- The reference verifier itself must pass verifier-of-verifiers checks: frozen replay against adjudicated labels, repeatability from reset state, exact case/verifier provenance, input-label consistency, always-accept/always-reject/exact-match-only baselines, and mutation sensitivity of necessary checks.
- At least one known-valid solution and permitted semantic alternative must be accepted before a verifier is frozen. Report positive acceptance and negative rejection separately; perfect negative rejection with zero positive acceptance is a false-reject failure, not a strong verifier.
- A validly evidenced but incorrect patch is a candidate failure, not an integrity false reject.
- If the task specification requires feature interactions, the oracle must test those compositions; isolated visible checks alone do not establish end-to-end specification compliance.
- Verifier mutations, reference replays and same-input label consistency are qualification evidence, not candidate-performance results.
- Ambiguous specifications/oracles remain unresolved until independent adjudication.
- A receipt binds evidence; it does not prove the underlying assertion is true.
- Public controls are development artifacts, not sealed holdouts.
- Candidate-controlled execution remains blocked until #10064 qualifies OS isolation and #10369 qualifies trust roots, signer custody and failure-domain independence.

## 7. Benchmark lifecycle

- **Major:** task set, agent-visible instructions, environment semantics or resource budget changes; rerun affected arms.
- **Minor:** verifier semantics, hidden-test behavior, qualification or scoring changes; regrade affected artifacts and version the result.
- **Patch:** metadata/documentation-only changes; reuse only after proving relevant execution and scoring digests are unchanged.

Quarantine tasks after a confirmed answer leak, pass-channel shortcut, invalid reference patch, false-negative oracle or material environment drift. Regrade affected scores; never silently patch a grader and retain old scores as comparable.

## 8. Sources

- Scale SWE-Bench Pro V2: https://labs.scale.com/leaderboard/swe_bench_pro_public_v2
- Scale SWE Atlas QnA: https://labs.scale.com/leaderboard/sweatlas-qna
- Scale SWE Atlas Test Writing: https://labs.scale.com/leaderboard
- Scale SWE Atlas Refactoring: https://labs.scale.com/leaderboard/sweatlas-refactoring
- ContextBench: https://contextbench.github.io/
- PDB results: https://huggingface.co/datasets/Precise-Debugging-Benchmarking/PDB-Results
- Microsoft CRBugBench: https://www.microsoft.com/en-us/research/publication/when-should-a-software-factory-be-lights-off-benchmarking-code-review-agents-on-bug-detection/
- Endor Labs Agent Security League: https://www.endorlabs.com/research/ai-code-security-benchmark
- Rust-SWE-bench / RUSTFORGER: https://arxiv.org/abs/2602.22764
- SWE-fficiency: https://proceedings.mlr.press/v306/ma26l.html
- Scale consolidated leaderboard (HiL-Bench): https://labs.scale.com/leaderboard
- Epoch AI Terminal-Bench 4.0.0 review: https://epoch.ai/benchmarks/terminal-bench-4/review
- Terminal-Bench lifecycle and verifier roadmap: https://www.tbench.ai/news/terminal-bench-4-0


## HiL-Bench selective-escalation metrics

The CEB-HUMAN lane now separates Question Precision, Blocker Recall, selective-escalation F1 (ASK-F1) and help-enabled Pass@3. HiL-Bench defines ASK-F1 as the harmonic mean of Question Precision and Blocker Recall; Pass@3 is success in at least one of three independent runs after help. These are not interchangeable. Its 44% combined ASK-F1 and 24% combined Pass@3 are separate observations; domain blocker-recall maxima of 36% SWE and 61% SQL retain separate 150-task denominators. The consolidated Muse Wonder 79.67 ± 4.17 value remains unlabeled and is not assigned to either metric. Source: https://labs.scale.com/leaderboard/hil .
