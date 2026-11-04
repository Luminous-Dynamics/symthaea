# CEB paradigm shifts and frontier targets V0 — 2026-11-04

**Status:** research and evaluation plan; design-only.  
**Related:** #10366 causal debugging; #10385 repair precision / optimization / abstention; #10387 causal integration; #10419 context/review/security/Rust evaluation; #10436 task custody; #10173 repository twin; #10064 execution isolation; verifier follow-on [PR #10456](https://github.com/Luminous-Dynamics/symthaea/pull/10456).  
**Claim ceiling:** benchmark reports below are external model+harness+dataset results. Targets are proposed. None are Symthaea results, and no current source change has passed a Rust build or task evaluation.

## 1. Executive decision

Do not try to win the next coding-agent leaderboard by optimizing one more aggregate patch-resolution number. SWE-Bench Pro V2 Full is now close to saturation: Scale's November 4 page reports 99.40 ± 0.40 for Opus 5, 99.22 ± 0.45 for Gemini 4 Argon and 99.10 ± 0.50 for Fable 5.1 across a 642-task, 11-repository public set. Even its 51-task HARD subset has a 98.0 top score. A high score on this public set is no longer sufficient evidence of general engineering competence.

The opportunity is to build an **evidence-qualified engineering system** that can do six things together:

1. acquire the minimum sufficient context, with source provenance and dependency awareness;
2. recover the actual contract, including configuration, type, security and operational invariants;
3. distinguish a real defect from a suspicious but valid change;
4. select a discriminating investigation before spending execution budget;
5. propose a minimal, admissible repair whose collateral effects are measured;
6. produce independently verifiable evidence, while abstaining or escalating when authority or evidence is insufficient.

The research claim should be about better *engineering outcomes per unit of risk, human effort and compute*, not about an agent that writes more code.

## 2. Frontier benchmark snapshot — 2026-11-04

The numbers below must remain attached to their original metric, task set and evaluation protocol. Do not pool them into a composite.

| Capability | Public reference | Current observed result | What remains hard |
|---|---|---|---|
| Long-horizon issue resolution | SWE-Bench Pro V2 Full, 642 tasks / 11 repos | Opus 5 + Claude Code: 99.40 ± 0.40; Gemini 4 Argon: 99.22 ± 0.45; Fable 5.1: 99.10 ± 0.50 | Full-set ceiling, smaller hard slice, private-repository transfer, benchmark leakage and verifier integrity |
| Hard issue resolution | SWE-Bench Pro V2 HARD, 51 tasks | Opus 5: 98.00; Fable 5.1: 92.20; Gemini 4 Argon: 92.16; GPT-6 Astra: 90.20 | Very small denominator; do not over-interpret rank or tiny score gaps |
| Context retrieval | ContextBench, 1,136 tasks / 66 repos / 8 languages | CodeNib: Recall 0.779, Pass@1 77.6%, Context F1 0.281, $0.68/task; different system leads F1 at 0.389 | Recall, F1 and solve rate are distinct; current result uses ContextBench-specific system adaptations |
| Pre-merge review | CRBugBench, 1,000 public tasks / 98 repos plus 800 proprietary tasks | Claude Opus 5 + Copilot CLI catches 71.9% of human-verified bugs in its top five comments | Complex changes, non-local dependencies and proprietary project context |
| Secure code generation | Agent Security League, 200 tasks / 108 Python projects / 77 CWE classes | Highest reported Functional: 91.6% (Muse Code + Muse Wonder); highest Secure: 36.9% (Claude Code + Claude Fable 5.1) | Functional and secure leaders differ; security is not a side effect of passing functional tests |
| Precise debugging | PDB-Single Hard, 5,751 tasks | Claude Opus 4.7: edit precision 83.3 ± 0.8, bug recall 89.3 ± 0.6; Gemini 3.1 Pro: 82.5 ± 0.8 / 89.4 ± 0.7 | Single-line synthesized bugs are not repository repair; task and evaluator version must be pinned |
| Mixed / repository debugging | PDB-Wild, 484 tasks | Opus 4.7: 77.8 ± 3.1 precision / 83.4 ± 2.8 recall; Gemini 3.1 Pro: 77.8 ± 3.0 / 85.6 ± 2.8 | Small, mixed set (multi-line and SWE-smith repository tasks); confidence intervals matter |
| Rust repository repair | Rust-SWE-bench, 500 tasks / 34 Rust repos | Historical RUSTFORGER + Claude Sonnet 3.7: 28.6%; studied ReAct baseline up to 21.2% | Repo comprehension, Rust type/trait semantics and issue reproduction |

### Source notes and metric cautions

- **SWE-Bench Pro V2:** [Scale leaderboard and methodology](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2), [V2 paper](https://arxiv.org/abs/2610.24206). The September 22 V2 release reports 642 tasks across 11 repositories after 89 invalid tasks were removed. The paper describes 480 of 731 original tasks being repaired or retired, including instruction/test contradictions; the protocol turns off agent web tools and regrades diffs on a pristine image. The live Scale page also lists two residual trust issues: its model endpoint is a trusted relay, and code in a patch is still executed by the verifier. A pristine regrade is valuable but not equivalent to a secure execution boundary.
- **ContextBench:** [Leaderboard](https://contextbench.github.io/). Its best Recall row and best Context F1 row are different systems. CodeNib's Recall 0.779 / Pass@1 77.6 / F1 0.281 is not the same run as openJiuwen's F1 0.389 / Recall 0.411 / Pass@1 60.8. Do not synthesize a fictional best-of-each-system score.
- **CRBugBench:** [Microsoft Research, October 2026](https://www.microsoft.com/en-us/research/publication/when-should-a-software-factory-be-lights-off-benchmarking-code-review-agents-on-bug-detection/). Its published top-five result is a detection rate over human-verified bug-review tasks; it is not precision or patch-resolution rate.
- **Agent Security League:** [Endor Labs leaderboard and methodology](https://www.endorlabs.com/research/ai-code-security-benchmark). Its displayed FuncPass and SecPass are separate published columns. Highest functional and secure values belong to different combinations. Its current task collection is Python-heavy, so it is a methodology reference, not a direct Rust benchmark.
- **PDB:** [PDB results dataset](https://huggingface.co/datasets/Precise-Debugging-Benchmarking/PDB-Results) and [paper](https://arxiv.org/abs/2604.17338). PDB-Single Hard and PDB-Wild use different task mixtures and epsilon-relaxed edit matching. The 5,751 single-line task result and 484-task mixed/repository result must be reported separately.
- **Rust-SWE-bench:** [paper](https://arxiv.org/abs/2602.22764) and [repository](https://github.com/GhabiX/Rust-SWE-Bench). The 28.6% is a February 2026 historical RUSTFORGER + Claude Sonnet 3.7 result, not a current November frontier claim.

## 3. Paradigm shifts worth testing

### A. From context volume to minimal sufficient evidence

A retrieval system should not be rewarded simply for placing more files in a prompt. It should find a compact evidence set that is sufficient to support the requested claims and decision.

**Design:** return ranked file/symbol/span records, atomic claims, source-origin lineage, freshness, authority and unresolved questions. Count copied documentation and duplicate logs as one underlying source where they share provenance.

**Breakthrough hypothesis:** a provenance-aware repository twin can maintain downstream correctness while reducing irrelevant context tokens and stale-fact rate after code changes.

**Ablations:** lexical search; AST/symbol search; Cargo metadata; rust-analyzer/compiler enrichment; dependency graph traversal; HDC semantic retrieval; full Symthaea context pipeline. Keep model, prompt, tokens, tools, task and budget fixed.

### B. From patch completion to minimal, proof-carrying repair

A candidate patch must be evaluated on the defects fixed and the edits that did not need to happen. PDB's edit-level precision and bug-level recall demonstrate why a passing test suite alone is insufficient.

**Design:** require a repair to provide an invariant-to-edit rationale, exact changed-path set, evidence coverage, allowed edit scope, regression evidence and known limitations. Use symbolic diff matching where applicable, but adjudicate semantically valid alternatives rather than treating a reference patch as the only correct answer.

**Breakthrough hypothesis:** a system can match or exceed frontier bug recall while substantially reducing unnecessary and collateral edits.

**Ablations:** unconstrained code generation; minimal-edit instruction only; spec-backed change plan; evidence-backed admissible-edit plan; independent skeptical review before submission.

### C. From plausible debugging to prospective causal investigation

A good investigator forecasts what an experiment should show before running it. It does not rewrite its explanation after seeing the outcome and then pretend it predicted that outcome.

**Design:** keep hypotheses, forecast, proposed action, authorization, observed outcome, update and repair as separate immutable records. Use low-risk, reversible, matched-cost experiments. Preserve null hypotheses, confounders, non-identifiability and missing counterfactuals. Reuse the prospective value-of-computation and active-probe work already specified in #5476, #5478 and #8286; do not build another generic scheduler.

**Breakthrough hypothesis:** fewer experiments per independently accepted repair, with better forecast calibration and no regression in accepted-repair correctness.

**Ablations:** retry-only; retrieval/similarity; one-shot diagnosis; random matched-cost experiments; static heuristic; existing VoC policy; causal hypothesis + forecast + intervention policy.

### D. From secure enough to joint functional and security correctness

Security must be a first-class acceptance gate, not a label inferred from functional success. The Agent Security League shows a large gap between its best functional and secure results. CEB should include Rust authorization, path containment, secret handling, unsafe/FFI, deserialization, dependency and supply-chain cases.

**Breakthrough hypothesis:** materially improve security pass rate without disabling required functionality, with no high-severity false accepts in the declared challenge set.

**Ablations:** model-only patching; standard linter/SAST; threat-model + call-graph retrieval; independent security reviewer; full evidence-qualified pipeline.

### E. From one agent's confidence to calibrated act / abstain / escalate

The system should know when it has source evidence, compiler-resolved evidence, runtime observation, intervention evidence, or only an inference. One uncalibrated confidence number must not authorize all actions.

**Design:** score decisions over paired near-identical should-act and should-abstain tasks. Measure false action, missed action, escalation precision, selective risk and coverage separately. Treat escalation as a useful outcome when it requests the missing evidence or authority, not as a generic failure.

**Breakthrough hypothesis:** reduce severe false actions while maintaining useful coverage and engineering throughput.

### F. From code review automation to human–agent complementarity

The best target may not be lights-off. Compare human-only, agent-only and human+agent review under a fixed effort budget. Measure confirmed defect recall, false-positive burden, severity, human minutes, override rate and reviewer trust calibration.

**Breakthrough hypothesis:** +5 percentage points in confirmed-defect recall at equal effort, or non-inferior recall with at least 20% less human review time, while keeping a separate high-severity false-negative gate.

### G. From static benchmark scores to longitudinal engineering reliability

Use repository-family and time-based splits, rotating fresh tasks, mutation sequences, regressions and operational follow-up. Score stale evidence after renames/moves/features/target changes, issue recurrence, rollback rate, and whether a correct fix remains correct under later changes.

**Breakthrough hypothesis:** demonstrate retention and transfer across new repository families, not just performance on a static public snapshot.

## 4. Matched evaluation protocol

Every component claim should be supported by the same comparison design:

1. **Fixed baseline arms:** deterministic/static tool baseline; model-only fixed harness; a strong public agent-style harness; Symthaea with the component disabled; Symthaea with the component enabled.
2. **Matched budgets:** freeze task, source revision, model, tool registry, prompt, context-token budget, output budget, wall time, tool calls, compute and cost.
3. **Paired units:** evaluate the same task in every arm. Cluster confidence intervals by repository, source family and root-cause family. Repeat seeds where the harness is stochastic.
4. **Preregistered primary outcomes:** choose one primary outcome per task family; keep all secondary measures visible. Report all eligible tasks, invalidations, no-output and tool failures.
5. **Independent scoring:** do not use the agent's own tests or LLM-as-judge as the sole authority. Keep reference patch, empty patch, known-bad, wrong-scope and mutated-oracle controls.
6. **Exact-head receipts:** score only a pinned task/source/model/harness/tool/policy/grader configuration with primary and independent pristine regrade receipts. Queued, skipped, or missing CI is never a pass.
7. **No leakage:** public benchmark tasks are external calibration, not sealed holdouts after their labels or patches are public. Hold out repository families and future tasks; retain a contamination register.

No aggregate number should allow a high functional solve rate to cancel a security failure, an unsupported causal claim, or an unqualified receipt.

## 5. Proposed targets

These are research gates, not standards or achieved results. For direct “beat SOTA” claims, use the identical public benchmark split, model/harness constraints, task revision and evaluator. The CEB-specific holdout targets below are separate.

| Lane | Matched external reference | Proposed matched target | CEB qualification target |
|---|---|---|---|
| Context retrieval | ContextBench Recall 0.779; F1 0.389 (different systems) | Recall >=0.82 and F1 >=0.42 in separate runs; Pass@1 >=0.80 | Recall@10 >=0.90, Context F1 >=0.50, no downstream accuracy drop at fixed budget |
| Precise debugging | PDB-Single Hard Opus 4.7: P 0.833 / R 0.893; PDB-Wild Opus 4.7: P 0.778 / R 0.834 | Exact same sets: improve both precision and recall with paired confidence intervals; do not blend Single and Wild | Review/edit precision >=0.85, bug recall >=0.90 on qualified holdout |
| Pre-merge review | CRBugBench 71.9% human-verified bugs in top five comments | >=0.76 bug detection @5 on same public tasks and setup, with actionable precision separately reported | Recall@5 >=0.85, actionable precision >=0.80; high-severity false negatives reported independently |
| Secure correctness | Agent Security League highest SecPass 36.9; FuncPass 91.6 from different agent/model | Reproduce on exact version; target SecPass >=0.45 and functional non-inferiority within 2 pp | Joint functional+secure pass >=0.65 on sealed Rust-heavy tasks; zero high-severity false accepts in the declared control set |
| Rust repair | Rust-SWE-bench historical RUSTFORGER 28.6% | >0.35 on a frozen same-version replication | >0.35 on the sealed Rust holdout; report reproduction success and type/trait/feature strata |
| Public issue resolution | SWE-Bench Pro V2 Full top 99.4; HARD top 98.0 | Use Pro V2 as an external integrity/transfer check; aim >=98 Full and >=95 HARD without regression or protocol violations | Not the primary CEB discriminator; require exact two-sided oracle qualification and pristine regrade |
| Causal debugging | No directly equivalent widely accepted benchmark | +10 pp independently accepted repair rate vs strongest matched baseline; or >=20% lower cost per accepted repair with <=1 pp correctness regression | Forecast calibration, intervention yield, repair correctness, non-identifiability and abstention all separately qualified |
| Human review complementarity | CRBugBench top-five reference | +5 pp confirmed-defect recall at equal effort, or noninferior recall with >=20% less human time | Reproduced in a preregistered paired human-only / agent-only / assisted study |
| Expert-relative optimization | SWE-fficiency reports top evaluated agents below 0.23x expert speedup in its studied suite | >0.5x expert-normalized speedup on matched tasks | >=0.8x with equivalent solution quality and semantic correctness |

**Interpretation:** The ContextBench and PDB matched targets are specific public reproduction goals, not claims that the same numbers transfer automatically to Rust tasks. CRBugBench's top-five detection rate is not precision. The ASL FuncPass and SecPass maxima are separate configurations, not a joint score. Small leaderboard gaps need paired uncertainty intervals before claiming a win.

## 6. Engineering priorities

### P0 — before any benchmark execution
- Qualify #10064 isolation with an adversarial threat model, network and filesystem controls, syscall/device policy, credential non-inheritance, process-tree boundaries, resource caps, kill/reap semantics, and independent penetration tests.
- Finish the static agent-view verifier under [PR #10456](https://github.com/Luminous-Dynamics/symthaea/pull/10456); require exact manifest/tree pins and adversarial inventory tests. This verifies package integrity only, not custody or isolation.
- Separate evaluator and agent ACL/storage; prove with canaries. The public development fixture remains non-sealed.
- Reconcile artifact references with #10370; do not create a new receipt envelope.

### P1 — first confirmatory task slice
- Author 12–24 rights-cleared synthetic tasks across context, API/trait contracts, non-local review, authorization/path security, Rust semantics and operational diagnosis.
- Every task gets reference-pass, empty-patch-fail, plausible known-bad, wrong-scope, mutation and leakage controls.
- Keep source-only static tasks separate from execution tasks; do not invoke Cargo, procedural macros, build scripts, tests or candidate programs before the isolation gate.

### P2 — controlled comparative experiments
- Run component ablations under matched model and budgets.
- Reproduce at least one external context, debugging, security and Rust baseline where rights and protocol permit.
- Preregister primary outcome, task/repository clusters, confidence intervals, exclusions, missingness and multiple-comparison corrections.
- Publish a result card with exact head SHA, task digest, harness/model identity, tool manifest, budgets, grader digest, receipts, confidence intervals and claim ceiling.

## 7. Decision

The most defensible “revolution” is not an unbounded agent. It is a **closed-loop, evidence-qualified engineering process** with five independently testable innovations:

- minimal sufficient context with provenance;
- specification and invariant recovery;
- prospective causal investigation;
- precision-preserving, security-aware repairs;
- independent verification plus calibrated abstention.

Build and benchmark those capabilities as separable modules, then show that their composition improves accepted engineering outcomes at equal or lower cost. If one module makes no statistically credible contribution under matched ablation, remove or redesign it. Do not keep it because the architecture sounds elegant.

## 8. Sources

- [Scale AI — SWE-Bench Pro V2 leaderboard and methodology](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2)
- [SWE-Bench Pro V2 paper](https://arxiv.org/abs/2610.24206)
- [ContextBench leaderboard](https://contextbench.github.io/)
- [Microsoft Research — CRBugBench (October 2026)](https://www.microsoft.com/en-us/research/publication/when-should-a-software-factory-be-lights-off-benchmarking-code-review-agents-on-bug-detection/)
- [Endor Labs — Agent Security League](https://www.endorlabs.com/research/ai-code-security-benchmark)
- [PDB results and per-task evidence](https://huggingface.co/datasets/Precise-Debugging-Benchmarking/PDB-Results)
- [PDB paper](https://arxiv.org/abs/2604.17338)
- [Rust-SWE-bench / RUSTFORGER paper](https://arxiv.org/abs/2602.22764)
