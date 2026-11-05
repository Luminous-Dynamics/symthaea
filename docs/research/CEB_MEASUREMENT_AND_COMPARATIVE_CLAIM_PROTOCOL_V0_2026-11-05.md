# CEB Measurement and Comparative Claim Protocol V0 — 2026-11-05

**Status:** Design-only measurement contract; no CEB performance result is claimed.  
**Machine-readable matrix:** [\`benchmarks/ceb/measurement-matrix.v0.json\`](../../benchmarks/ceb/measurement-matrix.v0.json)  
**Schema:** [\`benchmarks/ceb/measurement-matrix.schema.json\`](../../benchmarks/ceb/measurement-matrix.schema.json)  
**Integrity controls:** [PR #10479](https://github.com/Luminous-Dynamics/symthaea/pull/10479)  
**Execution gates:** [#10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064), [#10369](https://github.com/Luminous-Dynamics/symthaea/issues/10369)

## 1. The paradigm shift: engineering quality is a vector, not a leaderboard number

The current coding-agent frontier makes a single repair percentage an inadequate target.

Scale's SWE-Bench Pro V2 public Full leaderboard currently reports Opus 5 + Claude Code at 99.40 ± 0.40, Gemini 4 Argon + mini-swe-agent at 99.22 ± 0.45, and Fable 5.1 + Claude Code at 99.10 ± 0.50. The full set has 642 tasks across 11 repositories. This is a near-ceiling result, not proof that long-horizon software engineering is solved. The separate 51-task HARD subset has a different denominator and materially wider separation.

The same benchmark refresh is instructive: its creators removed 89 invalid tasks, corrected 69 contradictory instructions, disabled agent web access after observing fix-commit retrieval, added pristine-image regrading, and required every task to pass with the reference patch and fail with the empty patch. The live page still identifies trusted endpoint relay and verifier execution of candidate-controlled patch code as residual risks. See [Scale's official methodology and results](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2).

Other capability leaderboards show why a scalar “best agent” is misleading:

- **Comprehension:** SWE Atlas Codebase QnA has 124 tasks; Scale's current reference is Opus 5 + Claude Code at 63.17 ± 5.01.
- **Test writing:** Scale reports Fable 5.1 + Claude Code at 67.04 ± 5.33. This is test-writing quality, not a direct oracle-soundness measurement.
- **Refactoring:** SWE Atlas Refactoring has 70 tasks; GPT-6 Astra + Codex is reported at 59.05 ± 6.43.
- **Context retrieval:** ContextBench has 1,136 tasks, 66 repositories and 8 languages. Its best reported Recall (0.779), Pass@1 (77.6%) and Context F1 (0.389) are not all held by one configuration. Do not combine metric maxima into a nonexistent composite system.
- **Security:** Endor Labs reports 91.6% highest FuncPass for Muse Code + Muse Wonder and 36.9% highest SecPass for Claude Code + Claude Fable 5.1. Different configurations lead the two metrics. Functional correctness is not secure correctness.
- **Code review:** Microsoft's CRBugBench reports 71.9% of human-verified bugs in the best system's top five comments; non-local defects and private-repository generalization remain harder.
- **Optimization:** SWE-fficiency has 498 tasks across 9 repositories; top evaluated agents averaged below 0.23× the expert speedup in the paper's study.
- **Evaluator integrity:** Epoch AI rated Terminal-Bench 4.0.0 “Flawed”, reporting publicly noted scoring defects in 30/66 tasks (45.5%), including a writable pass signal, answer leakage and false negatives. Terminal-Bench's maintainers state that tamper-resistant verifiers are a 4.1 focus.

Sources and the exact metric caveats are registered per lane in the machine-readable matrix. The data above is a dated snapshot; recheck and pin the exact live board, benchmark version and task split before future comparison.

## 2. CEB's measurement portfolio

The matrix defines 13 lanes: resolution, repository comprehension, context retrieval, test generation, refactoring, precise debugging, pre-merge review, joint secure correctness, Rust-specific engineering, repository optimization, human-agent complementarity, evaluator integrity and documentation executability.

Each lane records:
- one decision question and a small set of primary metrics;
- current external comparators with exact task scope, published result, source URI and comparability classification;
- minimum baseline arms, including deterministic, model-only, strong industry-style, CEB-enabled and causal ablation arms where applicable;
- the required evidence artifacts and the authority allowed to support each claim;
- hard gates that cannot be traded against a higher score;
- a repository-/fault-family-aware holdout design.

The matrix is explicitly \`design_only\`. No row contains a CEB result. The static JSON Schema and CEB workflow enforce unique lane IDs and metric IDs, registered comparator sources, required adjudication and unresolved-state support, and negative controls against marking the matrix qualified or inventing a “beats_sota” comparison class.

## 3. How to make a defensible SOTA claim

A result may be described as **contextual**, **replicated**, **matched**, or **superior**. These are different evidence levels.

### Contextual reference

A result from another benchmark, metric, task version, model/harness, budget or rubric can motivate a target. It cannot support a direct claim that CEB beats that system.

### Replication

Run the exact public benchmark version, task split, metric, harness and scoring semantics. Publish deviations, source revision, exclusions and uncertainty. A successful reproduction is a replication, not automatically an improvement.

### Matched comparison

Use the same tasks, task instructions, source snapshots, oracle semantics, metric definitions, resource budget and evaluation environment. Hold model, harness, prompt, tools, turns, tokens, time, compute and cost constant unless the experiment preregisters one of these as the intervention. Use paired task outcomes and independent adjudication.

### Superiority

Require all matched-comparison evidence plus:
1. preregistered primary outcome and stopping rule;
2. at least three independent seeds, and five where pilot variance warrants;
3. paired confidence intervals or a preregistered non-inferiority test;
4. repository and root-cause-family clustering;
5. full task-level outcomes, exclusions, unresolved items and raw denominators;
6. pristine independent regrade and exact input/evaluator digests;
7. no critical gate violation.

Do not say “beats SOTA” because a percentage is numerically higher on a different set, because confidence intervals overlap, or because different configurations lead different metric columns.

## 4. Minimum matched baseline arms

For each lane, compare as applicable:

1. deterministic tools and conventional static analysis;
2. the same foundation model with a minimal harness;
3. the same model with a strong industry-style harness;
4. the same model plus exactly one CEB component;
5. the Symthaea-native path where a frontier foundation model is not required;
6. lesion, sham and rescue ablations for a causal mechanism claim;
7. human-only, agent-only and human+agent for complementarity claims.

All arms receive equivalent task information and resource budgets. If a baseline cannot be matched, classify the result as contextual and do not infer causal contribution from it.

## 5. Core research targets

The machine-readable matrix contains lane-specific targets. The first confirmatory portfolio should prioritize:

| Priority | Lane | Initial target |
|---|---|---|
| P0 | CEB-EVAL | Zero critical false accepts; positive controls must pass; false-accept and false-reject uncertainty reported separately |
| P0 | CEB-DOCS | Zero unannotated missing documented Cargo manifests, package selectors or binary targets; dynamic references remain explicit gaps |
| P1 | CEB-REPO | Source-supported claim precision ≥0.95; claim recall ≥0.90; zero fabricated compiler/runtime authority |
| P1 | CEB-CONTEXT | Recall@10 ≥0.90 and Context F1 ≥0.50 on sealed holdout; measure downstream success and cost |
| P1 | CEB-DEBUG | Bug Recall@5 ≥0.90 and edit precision ≥0.85 on qualified holdout |
| P1 | CEB-REVIEW | Recall@5 ≥0.85 and actionable precision ≥0.80 on sealed holdout |
| P1 | CEB-SECURE | Joint functional+secure pass ≥0.65 on sealed Rust-heavy tasks; zero critical security false accepts |
| P2 | CEB-OPT | ≥0.80× expert-normalized speedup with semantic non-regression |
| P2 | CEB-HUMAN | +5 percentage-point confirmed-defect recall at equal effort, or non-inferior recall with ≥20% lower review time |

These are proposed thresholds, not observed CEB results. Final sample sizes and confidence bounds must be set from pilot variance, task-family dependence, evaluator quality and the consequences of false acceptance.

## 6. Hard-gate philosophy

- A critical security or evaluator false accept is disqualifying; no weighted score can cancel it.
- A positive control must qualify; rejecting all runs is not a successful verifier.
- A validly evidenced but incorrect candidate is a candidate failure, not an integrity false reject.
- Ambiguous specifications and inconsistent oracles remain unresolved until independently adjudicated.
- A passing receipt is a claim about evidence binding and validation; it does not prove the underlying physical or software assertion is true.
- Public control cases are development material, not hidden holdout evidence.
- No candidate-controlled execution is authorized until #10064 qualifies the operating-system isolation boundary and #10369 qualifies trust roots, signer custody and failure-domain independence.

## 7. Benchmark lifecycle

Use semantic versioning by what changes in the experiment:
- **Major:** task set, agent-visible instructions/context, environment behavior or resource budgets change; rerun affected model/task arms.
- **Minor:** verifier semantics, hidden test behavior, qualification or scoring rules change; regrade affected artifacts and clearly separate versions.
- **Patch:** metadata or documentation changes that do not alter execution/scoring; reuse previous results only after confirming the relevant execution and scoring digests are unchanged.

Quarantine a task after a confirmed answer leak, reward-channel shortcut, invalid reference patch, false-negative oracle or material environment drift. Regrade affected results; never silently patch the grader while retaining an old score as comparable.

## 8. Sources

- [Scale — SWE-Bench Pro V2](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2)
- [Scale — SWE Atlas Codebase QnA](https://labs.scale.com/leaderboard/sweatlas-qna)
- [Scale — SWE Atlas Test Writing](https://labs.scale.com/leaderboard/sweatlas-test-writing)
- [Scale — SWE Atlas Refactoring](https://labs.scale.com/leaderboard/sweatlas-refactoring)
- [ContextBench leaderboard](https://contextbench.github.io/)
- [PDB result tables](https://huggingface.co/datasets/Precise-Debugging-Benchmarking/PDB-Results)
- [Microsoft Research — CRBugBench](https://www.microsoft.com/en-us/research/publication/when-should-a-software-factory-be-lights-off-benchmarking-code-review-agents-on-bug-detection/)
- [Endor Labs — Agent Security League](https://www.endorlabs.com/research/ai-code-security-benchmark)
- [Rust-SWE-bench / RUSTFORGER](https://arxiv.org/abs/2602.22764)
- [SWE-fficiency, ICML 2026](https://proceedings.mlr.press/v306/ma26l.html)
- [Scale — HiL-Bench](https://labs.scale.com/leaderboard/hil-bench)
- [Epoch AI — Terminal-Bench 4.0.0 review](https://epoch.ai/benchmarks/terminal-bench-4/review)
- [Terminal-Bench — 4.0 lifecycle and 4.1 verifier roadmap](https://www.tbench.ai/news/terminal-bench-4-0)
