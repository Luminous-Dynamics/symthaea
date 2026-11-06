# CEB Industry Frontier and Paradigm Targets V0 — 2026-11-06

**Status:** Research synthesis and prospective target proposal. Not a benchmark result, validated standard, or superiority claim.  
**Parent:** [#10663 CEB pilot authoring pack](https://github.com/Luminous-Dynamics/symthaea/issues/10663)  
**Companion:** [CEB Pilot Authoring Pack V0](CEB_PILOT_AUTHORING_PACK_V0_2026-11-06.md)

**Name:** CEB means **Computational Engineering Bench**, the Symthaea-native engineering evaluation programme introduced in [PR #10115](https://github.com/Luminous-Dynamics/symthaea/pull/10115). This synthesis retains that official name while reflecting the broader lifecycle scope now covered by its lanes.

## 1. Executive thesis

The next step in software-engineering-agent evaluation is not another isolated patch leaderboard. CEB should qualify whether an engineering system can make **correct, authorized, evidence-backed decisions over the lifetime of a changing codebase**.

The field is moving beyond one-shot bug fixing already:
- SWE-Bench Pro V2 has pushed single-issue resolution near a public ceiling on its full split.
- SWE-Milestone and SWE-CI measure continuous repository evolution and maintenance.
- SWE Atlas broadens coverage to repository understanding, test writing and refactoring.
- Artificial Analysis's Coding Agent Index combines software engineering, terminal work and repository Q&A, while reporting cost and time.
- HiL-Bench makes selective escalation visible rather than rewarding confident guessing through missing requirements.

CEB should integrate these benchmarks as context and use common tasks where feasible, but it should not pretend their scores are interchangeable. Its differentiator is the validity and cost of engineering evidence as the source, requirements, environment, policy, evaluator and workload evolve.

## 2. Current industry landscape — verified 2026-11-06

### 2.1 SWE-Bench Pro V2: patch resolution is approaching saturation

Scale AI's refreshed public split contains 642 tasks across 11 repositories. The release removed 89 tasks judged invalid; 69 task descriptions were corrected after instruction/test contradictions were found and blind-solved by an expert. Agent network access is restricted to the model endpoint, patches are regraded on pristine images, and every task must pass with the reference patch and fail with the empty patch. The public methodology also discloses residual concerns: the model endpoint is a trusted relay, and code introduced by a patch can still execute in the verifier. [Scale AI methodology and release notes](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full)

The refreshed public leaderboard reports Claude Opus 5 at **99.4% (638/642)**, Kimi K3 at **97.7% (627/642)** and GPT-Astra at **96.9%** on the 642-task public split. The companion 51-task Hard split is a more discriminating public probe. [Scale's V2 release and public/private results](https://labs.scale.com/blog/swe-bench-pro-v2) · [Live leaderboard](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full) · [V2 paper, 16 October 2026](https://arxiv.org/abs/2610.24206)

**Version-scoped validity warning:** Epoch AI reviewed the pre-V2 SWE-Bench Pro release on September 1, 2026 and rated that release *Flawed*, citing multiple audits estimating that roughly 30% or more of tasks had scoring defects. That review describes the older 1,865-task suite (731 public); it predates the September 22 V2 refresh and must not be mechanically transferred to V2. V2 is a materially revised release: Scale reports removing 89 invalid tasks, refreshing task instructions, verifiers and images, and adding expert/agentic review. However, V2 should not be called independently verified merely because it is newer. A V2-specific task-level audit remains the correct evidentiary standard. [Epoch AI's pre-V2 review](https://epoch.ai/benchmarks/swe-bench-pro/review) · [Scale V2 release notes](https://labs.scale.com/blog/swe-bench-pro-v2)

**Saturation is not the same as generalization:** Scale reports a large public/private gap—for Claude Opus 5, 638/642 public tasks versus 222/272 private tasks—and explicitly notes that the near-ceiling public result may reflect training-time exposure to public repositories and fixes. Network locking reduces evaluation-time retrieval; it cannot erase information learned during training. Keep public and private results separate, and do not interpret the public 99% result as proof that the underlying capability is solved. [Scale V2 public/private analysis](https://labs.scale.com/blog/swe-bench-pro-v2)

**CEB implication:** use patch correctness as a necessary baseline and safety gate, not the sole north-star. Distinguish the pre-V2 review from the V2 release, and do not claim superiority by comparing CEB's sequence or evidence metrics to SWE-Bench resolve rate.

### 2.2 SWE-Milestone: continuous evolution is the closest adjacent construct

SWE-Milestone reconstructs real repository histories into milestone dependency DAGs. An agent works through consecutive milestones without a reset; a milestone resolves only if all fail-to-pass tests pass and no previously-passing test regresses. The paper evaluates 12 frontier models across four agent frameworks and reports an isolated-to-continuous decline from over 80% to at most 38%. [ICML 2026 paper](https://proceedings.mlr.press/v306/deng26d.html) · [v1.0.2 project and leaderboard](https://swe-milestone.com/)

This is directly relevant to CEB but is not identical. SWE-Milestone measures continued implementation and regression control. CEB continuity adds a distinct question: does the agent correctly retain, revalidate, invalidate, supersede or escalate claims and their evidence as the system's governing conditions change?

**CEB implication:** build paired experiments on compatible milestone streams where possible. Compare a full evidence-continuity arm against stateless reset, persistent-history and repository-twin-only ablations under matched budgets.

### 2.3 SWE-CI: long histories and maintainability

SWE-CI contains 100 repository-maintenance tasks. Each task's evolution history averages 233 days and 71 consecutive commits. Its stated objective is to move beyond static functional correctness toward dynamic, long-term maintainability. [SWE-CI paper](https://arxiv.org/abs/2603.03823)

**CEB implication:** sequence depth and real repository diversity matter. A 24-blueprint authoring bank is only a feasibility inventory; confirmatory claims require many more independent repository clusters and real chronological history.

### 2.4 SWE Atlas: breadth outside issue resolution

SWE Atlas v1.1 covers 284 tasks from 15 production repositories in six languages and three categories: codebase comprehension, test writing and refactoring. Its tasks are single-turn: a prompt states the goal and leaves the approach to the agent. V1.1 re-ran models with a new judge (GPT-6.1 Sol, majority of three votes), refined grading criteria and restricted network access; scores from v1.0 are therefore not directly comparable. [Scale AI SWE Atlas v1.1 methodology/results](https://labs.scale.com/leaderboard/swe-atlas)

**CEB implication:** use repository Q&A, tests and refactoring as component-level probes, but measure whether their outputs remain valid across later source, configuration and policy events.

### 2.5 Artificial Analysis Coding Agent Index V2: composite breadth and economics

The October 2026 Coding Agent Index V2 is an equal-weight average of FrontierCode v1.1 (100 tasks), Terminal-Bench 4.0 (66 tasks) and SWE-Atlas-QnA-AA (124 tasks), for 290 tasks. Each component is attempted three times; the methodology averages attempt scores per task and then gives each task equal weight. FrontierCode is a private sample graded with a maintainer-oriented blocking/non-blocking rubric; Terminal-Bench covers terminal workflows; SWE-Atlas-QnA-AA covers repository questions. The V2 methodology also publishes cost, token use and execution time, classifies safety refusals/errors/timeouts explicitly, and applies reward-hacking review to passing attempts. In its Terminal-Bench review path, an LLM judge (Claude Code with Claude Sonnet 5) inspects trajectories that pass deterministic verification. This is disclosed methodology, not evidence of a defect; however, shared judge/model lineage is a dependency worth stress-testing rather than assuming independent corroboration. [Artificial Analysis V2 methodology](https://artificialanalysis.ai/methodology/coding-agents-benchmarking)

**CEB implication:** report cost, tokens, wall-clock, human effort and review effort alongside correctness. Preserve per-component and per-task outcomes rather than relying on the equal-weight index alone. Record auditor model/harness lineage and test review decisions against an independent adjudication sample; a reward-hacking filter is itself an evaluator that requires qualification.

### 2.6 HiL-Bench: selective escalation is measurable

HiL-Bench contains 300 tasks across software engineering and text-to-SQL, with 1,131 blockers (3.77 per task average). Its design hides missing, ambiguous or contradictory information and measures both ASK-F1 and task outcome, because asking well and integrating answers are distinct capabilities. [HiL-Bench overview and methodology](https://labs.scale.com/leaderboard/hil)

**CEB implication:** reuse #10612/#10613 for ASK-F1 and help-enabled outcome. Continuity and collaboration should test whether an escalation is correctly tied to a concrete claim, policy source, or evidence gap—not create a competing ASK metric.


### 2.7 SWE-Marathon: project-scale, multi-hour engineering

SWE-Marathon v1.1 describes 20 realistic multi-hour tasks spanning full-library reproduction, full-stack application cloning, ML systems/post-training and algorithmic optimization. The official page reports 3,810 logged trials and a top displayed Pass@1 of 53.1% for Muse Wonder Full with Muse Code. This is a useful project-scale pressure test, but its 20 tasks are a small challenge set, not a substitute for long repository histories or evidence continuity across releases. [Official benchmark and v1.1 leaderboard](https://www.swe-marathon.org/)

**CEB implication:** include project-scale tasks after the current evidence and custody gates are met. Report project completion, durable correctness, regression and total human/agent effort separately; do not conflate multi-hour task completion with multi-release evidence continuity.

### 2.8 GSO: optimization quality after hack adjustment

GSO contains 102 optimization tasks across 10 codebases and five languages. Opt@1 estimates the fraction of tasks where one attempt reaches at least 95% of expert reference speedup and passes correctness. Its public leaderboard shows Claude Fable 5.1 xHigh at 88.2% Opt@1 and 87.3% hack-adjusted. The benchmark explicitly documents a hack detector for deceptive optimizations such as memoization and harness hijacking, and network hardening after observed attempts to retrieve upstream changes. [GSO benchmark and methodology](https://gso-bench.github.io/)

**CEB implication:** optimize only under matched workloads, hardware and correctness constraints, and publish raw and integrity-adjusted outcomes separately. A speedup is not valid when it bypasses the task's measurement or semantic contract. GSO Opt@1 is not numerically equivalent to CEB's expert-normalized speedup.

### 2.9 VerifierBench: verifier generation is a capability in its own right

VerifierBench is now supported by a named arXiv preprint dated October 19, 2026, which supersedes the earlier characterization of the work as only an anonymous under-review submission. The paper authors are Jianing Yin, Ziyi Chen, Hao Wang, Yiyou Sun, Zijian Jin, Dawn Song and Daoan Zhang. The preprint describes 222 tasks from six agent benchmarks and 4,993 hidden cases, including valid alternatives and attacks. Its reported experiment spans 19 models in a common harness, with normalized balanced accuracy from 10.6 to 62.1; generated verifiers reject invalid solutions more reliably than they accept valid ones, and many false rejections come from verifiers that accept no tested solution. The project page separately reports 36 model–harness configurations and 45 source-task exclusions. These are different counts and should not be collapsed into "36 models" or treated as the same experimental denominator. The preprint is public research, not evidence of peer review or independent CEB reproduction. [VerifierBench paper (October 19, 2026)](https://arxiv.org/abs/2610.28679) · [Project page and dataset overview](https://verifierbench.github.io/)

The paper describes a six-check Verifier4Verifier (V4V) audit for source verifiers and test suites. The important operational failure is asymmetric discrimination: rejecting invalid submissions is not sufficient if valid alternatives are rejected or no valid solution is accepted. The exact paper results and project-page configuration inventory must remain source-scoped.

**CEB implication:** qualify the verifier itself. CEB must test positive acceptance and negative rejection separately, reject always-accept/always-reject/exact-match-only strategies, verify known-valid alternatives, and establish that necessary verifier checks actually affect decisions. This evidence motivated the V4V extension in [#10680](https://github.com/Luminous-Dynamics/symthaea/issues/10680) and [draft PR #10681](https://github.com/Luminous-Dynamics/symthaea/pull/10681). VerifierBench is contextual research, not an equivalent CEB score or independently reproduced result.

### 2.10 SpecBench: visible-test success can hide specification failure

SpecBench contains 30 systems-level coding tasks from short to ultra-long horizons. Each task separates agent-visible feature-level validation tests from hidden tests that compose behaviors already required by the specification. It defines the reward-hacking gap as visible validation pass rate minus hidden composition pass rate. The detailed paper reports that the upper-tail gap grows by about 27 percentage points per tenfold increase in reference implementation size; its abstract rounds the figure to 28. [SpecBench paper](https://arxiv.org/abs/2605.21384)

**CEB implication:** task oracles must include required cross-feature interactions and valid alternative solutions. Hidden tests must instantiate the stated contract, not introduce new requirements. A public task with only feature-isolated tests cannot support an end-to-end specification-compliance claim.

### 2.11 RewardHackingAgents: protect the measurement channel

RewardHackingAgents studies evaluator tampering and train/test leakage in agentic ML engineering. Its paper reports three ML tasks and two model backbones; scripted attacks succeeded in fully mutable workspaces, while combined evaluator-locking and data-access controls blocked both vectors. In natural-agent runs, evaluator-tampering attempts occurred in about half the episodes before evaluator locking, which imposed a 25–31% median runtime overhead. [RewardHackingAgents paper](https://arxiv.org/abs/2603.11337)

**CEB implication:** measure the cost of integrity controls as well as their effect. Keep grader and hidden-data access outside candidate-controlled workspaces, log access provenance, and distinguish genuine capability improvement from modification of the metric or its reporting channel.

### 2.12 Agent Security League: functional and secure correctness are different axes

Endor Labs' Agent Security League reports 34 agent/model configurations. Its November 3, 2026 update shows Muse Code + Muse Wonder at **91.6% FuncPass** but only **25.1% SecPass** (seventh by security), while Claude Code + Claude Fable 5.1 retains the highest reported SecPass at **36.9%**. The functional and secure maxima are different configurations and must not be combined into a fictitious best-system result. More importantly for measurement integrity, the November update describes three confirmed cheating cases for Muse Wonder; one training-recall case was found through manual spot checks after the automated pipeline missed it. Endor Labs revised and re-scored its leaderboard after that audit. [Endor Labs, November 3 update](https://www.endorlabs.com/learn/metas-new-muse-wonder-record-funcpass-but-mid-table-secpass) · [Agent Security League](https://www.endorlabs.com/research/ai-code-security-benchmark)

**CEB implication:** retain joint secure-functional pass and critical security false accepts as independent hard-gated measures. Also benchmark the integrity auditor: manual spot checks, known training-recall controls, independent trajectory review and auditor false-negative accounting should qualify the anti-cheating pipeline. A high functional pass rate cannot stand in for security correctness, and an automated auditor's "clean" label is not ground truth by itself.

### 2.13 External benchmark validity is version-scoped

External benchmark names are not validity guarantees. Epoch AI's published review list currently classifies the pre-V2 SWE-Bench Pro release, SWE-bench Verified and Terminal-Bench 4.0.0 as *Flawed*. Its review rubric treats scoring defects that can affect at least 20% of a benchmark's total score as sufficient for that verdict. This is a useful independent warning signal, not a universal ground-truth authority and not a substitute for inspecting the affected task versions. [Epoch AI reviewed-benchmark index](https://epoch.ai/data/benchmark-reviews-documentation/included-benchmarks)

For Terminal-Bench specifically, Epoch's September 4 review reports public scoring defects in 30 of 66 v4.0.0 tasks (45.5%), including writable pass signals, answer leakage and false negatives. The Terminal-Bench maintainers describe 4.1 as a verifier-hardening release, and the v4.1 GitHub milestone showed 93% completion with six open issues at its October 29 update. The official benchmark catalogue retrieved for this review still lists Terminal-Bench 4.0 as the current released version; a milestone or roadmap is not a released, regraded benchmark. Therefore, preserve 4.0.0 results as historical observations with defect warnings, and do not silently treat planned 4.1 fixes as completed remediation. [Epoch AI v4.0.0 review](https://epoch.ai/benchmarks/terminal-bench-4/review) · [Terminal-Bench 4.1 milestone](https://github.com/harbor-framework/terminal-bench/milestone/2) · [Official benchmark catalogue](https://www.tbench.ai/benchmarks) · [Maintainer 4.0 release notes](https://www.tbench.ai/news/terminal-bench-4-0)

**CEB implication:** attach validity evidence to the exact suite release, task universe, verifier revision and leaderboard observation. A family-level label ("SWE-Bench Pro", "Terminal-Bench") is too coarse. When an external suite is revised, preserve the old verdict and its scope, qualify the new release independently, and regrade/rerun only according to its documented change semantics. Reuse the existing CEB external-observation and release-lifecycle ownership (#10571/#10589 and #10610/#10611); do not create a second benchmark registry or an opaque benchmark-quality score.

## 3. What the industry does well — and what remains open

| Industry benchmark | What it measures well | Remaining CEB research opportunity |
|---|---|---|
| SWE-Bench Pro V2 | Real-issue patch resolution, task repair, reference/empty controls, pristine-image regrading | Separate V2 from the older Flawed review; independently audit V2, public/private split and training-exposure risk; then measure ongoing evidence validity and long-term decision traceability |
| Terminal-Bench 4.0.0 | Terminal, systems, science, operations and tool-use tasks; continuous versioning | Treat 4.0.0 as historically useful but externally reviewed as Flawed; require task-level adjudication and a released, regraded successor before claiming remediation |
| SWE-Milestone | Multi-milestone implementation, dependency DAGs, F2P/P2P regressions, compounding errors | Typed claim/evidence disposition and selective proof refresh across milestones |
| SWE-CI | Long repository histories and maintainability under repeated evolution | Separating product evolution from evaluator/metrology drift and claim-admissibility changes |
| SWE Atlas | Comprehension, test writing, refactoring beyond issue resolution | Whether generated understanding and tests stay valid after subsequent changes |
| Coding Agent Index V2 | Broad capability coverage plus cost, tokens and time | Human effort, reviewer reconstruction burden, durable acceptance and evidence lifecycle economics |
| HiL-Bench | Progressive blocker discovery and targeted help seeking | Authority-bounded collaboration and evidence-aware escalation across a changing repository |
| SWE-Marathon | Multi-hour, project-scale engineering across 20 tasks | Durable multi-milestone correctness, selective evidence refresh and human-effort accounting |
| GSO | Expert-relative optimization with hack-adjusted scoring | Measurement-integrity-aware optimization under matched workloads and hardware |
| VerifierBench | Generated verifier quality, positive/negative discrimination and V4V checks | Verifier qualification integrated with task admission, independent oracle review and trust evidence |
| SpecBench | Gap between visible feature tests and hidden spec-required composition | Explicit composition coverage and prevention of visible-test overfitting |
| RewardHackingAgents | Evaluator tampering and train/test leakage in ML-agent workspaces | Evidence access control and auditable measurement-channel integrity |
| Agent Security League | Joint functional and security correctness reported separately | Zero critical security false accepts and per-configuration secure-functional outcomes |

The V2 task audit is itself instructive: 480 of the original 731 tasks were repaired or retired, according to the paper. A benchmark can become invalid as agents improve or exploit its harness; benchmark maintenance and evaluator integrity are therefore ongoing engineering disciplines, not one-time dataset preparation. [SWE-Bench Pro V2 paper](https://arxiv.org/abs/2610.24206)

## 4. Paradigm shifts CEB should pursue

### Shift 1 — From patch-as-output to claim-as-unit

The core object is not only a patch or task result. It is a consequential engineering claim with:
- a precise proposition;
- an applicability domain (source revision, configuration, version, target, workload, policy and time);
- evidence provenance and authority;
- dependencies that can change;
- a current disposition and a decision-use boundary.

This makes stale evidence a first-class failure rather than a documentation nuisance. #9112 owns admissibility semantics; #10173 owns repository graph and change-impact facts; #10630 scores the continuity behavior.

### Shift 2 — From chronological memory to selective proof maintenance

Persistent memory is not enough. A long transcript can faithfully preserve an obsolete conclusion. The system must decide which evidence to retain, revalidate, invalidate, supersede or mark unresolved, and justify the minimum sufficient check.

Three baselines are essential: reset-and-recheck, remember-everything, and invalidate-everything. CEB should beat all three on a Pareto surface, not merely claim that memory helps.

### Shift 3 — From static tests to contrastive causal sequences

For every consequential change, author a matched contrast:
- one version changes a governing dependency;
- the other changes an irrelevant surface;
- a third changes only one branch of a conditional applicability domain.

The agent's change in disposition and action should match the causal difference. This is stronger than merely measuring whether it can describe a diff. #10366/#10387 supply causal hypothesis and intervention discipline; no second causal scheduler is needed.

### Shift 4 — From “ask or act” to risk-sensitive value of information

Use #10612 for selective escalation and #10366/#10387 for prospective diagnosis. The novel test is whether the agent compares the risk of acting on uncertain evidence with the cost and expected value of a targeted check or human question.

A good system should not ask for everything, rerun everything, or act confidently on nothing. It should seek the smallest sufficient information that changes a real decision. Keep the information-gathering action, the engineering action, and the human authority decision distinct.

### Shift 5 — From trustworthy artifacts to current authority

Cryptographic integrity answers whether an artifact has changed; it does not by itself prove that the signer is still trusted, the policy is still current, or the evidence applies to the present release. CEB should test authenticity, custody, policy validity and decision-use scope as separate conditions through #10314/#10370/#10369/#9112.

### Shift 6 — From code correctness to qualified collaboration

A patch that passes tests can still be unsafe to merge or deploy if the agent bypassed approval, overstated a skipped check, ignored current policy, or made the review impossible. Human-agent collaboration is a distinct socio-technical outcome, not a proxy for patch pass rate.

#10637/#10640 adds this construct without duplicating #10612's question metrics or #10385's debugging/human-complementarity outcomes.

### Shift 7 — From trusting the verifier to qualifying the verifier

A verifier is an executable hypothesis about the specification, not ground truth by declaration. Test it against known-valid alternatives, known-invalid near misses, identical-input label consistency, frozen replay, repeatability and mutation sensitivity. Always-accept and always-reject baselines must both fail. A verifier that rejects every answer may have high negative-case accuracy while being unusable. Extend this one level further: the **integrity auditor** that decides whether an agent cheated is also a verifier. It needs its own known-positive/known-negative controls, lineage disclosure, manual blinded spot checks and independent adjudication. Endor Labs' November 3 audit provides a concrete example of an automated anti-cheating blind spot discovered through manual review. The V4V protocol in #10680/#10681 extends the existing CEB-EVAL hard gates without adding a new lane or score.

### Shift 8 — From feature-isolated tests to compositional specification evidence

When a contract requires interactions among features, evaluate those interactions. The gap between visible feature-only tests and hidden composition tests reveals whether an agent built a coherent system or optimized a proxy. Hidden cases must remain derivable from the visible specification; they must not introduce new requirements.

### Shift 9 — From fixed benchmark to continuously audited measurement system

Treat task provenance, contamination, oracle validity, denominators, environment digests, evaluator versions, external review status, regrading and external observations as a maintained evidence base. Validity attaches to an exact release and task universe, not just a benchmark family name. A benchmark result should have an explicit claim class and a reproducible path back to task, system, run, grader revision and adjudication. Reuse #10610, #10619, #10626 and #10589/#10593 rather than adding another registry.

## 5. Recommended CEB scorecard — independent dimensions

Do not publish one opaque weighted number. Maintain hard gates and a Pareto surface across:

1. **Engineering outcome:** accepted behavior, regression preservation, repair survival.
2. **Evidence validity:** disposition correctness, stale-evidence use, false invalidation, lineage completeness.
3. **Decision quality:** correct action/ask/abstain/escalate, prospective diagnosis, minimum sufficient revalidation.
4. **Security and authority:** critical policy compliance, unauthorized consequential actions, custody and trust validity.
5. **Collaboration:** factual handoff, correction integration, reviewer reconstruction time, human effort.
6. **Economics:** compute cost, wall-clock, tool use, human labor and revalidation cost per accepted sequence.
7. **Measurement integrity:** contamination, oracle quality, verifier and integrity-auditor trust, exact-release validity, denominator fidelity and regrade reproducibility.

A result can be strong on one dimension and unacceptable on another. Critical unauthorized actions, critical stale-evidence use, verifier escape or invalid holdout custody must not be averaged away.

## 6. Prospective benchmark targets

These are recommended **research targets**, not measured results, industry standards or a guarantee that the first pilot can meet them. #10619 must determine confirmatory margins and statistical adequacy before scoring begins.

| Capability | Feasibility target | Hard guardrail |
|---|---:|---|
| Continuity sequence completion | ≥0.90 | Correct at every milestone; no skipping difficult events |
| Evidence disposition macro-F1 | ≥0.90 after blinded pilot calibration; ≥0.95 on sealed holdout | Publish per-disposition precision/recall and prevalence |
| Stale-evidence use | Report the exact overall rate and denominator | Zero critical use of stale, revoked or out-of-scope evidence |
| False invalidation | ≤0.05 feasibility; ≤0.02 sealed holdout | No “invalidate everything” strategy; report false invalidation cost |
| Revalidation economics | At least 20% lower lifecycle cost than reset-and-recheck | No increased critical stale use, false acceptance or protected-invariant regression |
| Repair survival | ≥0.90 on sealed holdout after sequence depth is calibrated | Verify on the exact successor revision and report horizon/event strata |
| Evidence lineage completeness | ≥0.98 on sealed holdout | Missing provenance is never inferred |
| Policy/process compliance | ≥0.95 | Zero unauthorized high-impact actions |
| Handoff factual precision | ≥0.95 | Claims must bind to exact revision and actual check state |
| Correction integration accuracy | ≥0.90 | No scope drift or regression accepted as a correction |
| Reviewer reconstruction time | ≥20% lower than matched baseline | Independently accepted-work correctness non-inferior |
| Durable outcome economics | Improve accepted engineering output per total human + agent effort | Report compute, review, interruption and revalidation costs separately |

**Interpretation:** these targets are calibration bands. Do not turn them into a success criterion by choosing a convenient denominator after the run. Report per-repository, per-policy and per-event strata with uncertainty and missingness. A 24-sequence / 24-episode authoring bank is not statistically sufficient for superiority claims.

## 7. How to compare with and outperform the state of the art

### 7.1 Same agent, same task, only the CEB mechanism changes

The primary causal experiment should keep the base model and agent scaffold constant, then compare:
- no explicit evidence continuity;
- persistent history only;
- repository-twin only;
- decision-use validity only;
- causal investigation only;
- full continuity composition.

For collaboration, compare human-only, agent-only, conventional agent, technical CEB without collaboration protocol, selective escalation only, and full CEB collaboration.

Match prompts, tools, context, time, tokens, compute, execution environment, task order and human effort. Use paired tasks and repository/policy cluster-aware inference. A stronger model or bigger budget is not evidence that CEB caused the gain.

### 7.2 Benchmark on a common anchor, but report distinct constructs

Where license, task access and protocol permit, re-run the same agent configurations on selected SWE-Milestone and SWE-CI tasks to establish functional-performance context. Then measure CEB-specific dispositions and lifecycle outcomes on the same event stream.

Do not paste a published third-party leaderboard score beside a CEB metric and call the difference a win. Published model, harness, prompt, budget, version and task split frequently differ.

### 7.3 Demonstrate value with a Pareto frontier

The desired frontier is not maximum accuracy at unlimited cost. It is:
- more correct, durable engineering decisions;
- fewer consequential stale-evidence uses;
- lower unnecessary revalidation and reviewer reconstruction cost;
- zero critical unauthorized action;
- no regression in engineering correctness;
- matched or lower total human-plus-agent effort.

An agent that saves 30% of test cost but increases critical stale acceptance is not an improvement. An agent that asks fewer questions but silently guesses is not an improvement. A system that retains every artifact but cannot explain current applicability is not an improvement.

### 7.4 Pre-register before confirmation

Use the ownership already established:
- #10614/#10617 for balanced repository and fault-family sampling;
- #10619/#10624 for paired analysis, clustering, multiplicity and claim class;
- #10610/#10611 for rerun/regrade/reuse;
- #10626/#10634 for contamination;
- #10612/#10613 for escalation;
- #10630/#10631 for continuity;
- #10637/#10640 for collaboration.

The starter pilot is exploratory. The existing research plan requires a much larger confirmatory repository-cluster set; a six-repository cohort is not a defensible basis for general superiority.

## 8. Runner-independent work that can proceed now

The following are safe and useful before #10064 isolation and #10369 trust qualification:

1. Complete the 24 + 24 public development blueprint pack and reviewer rubrics (companion document).
2. Calibrate reviewers on non-evaluative synthetic vignettes and revise ambiguous definitions.
3. Build an event-to-claim authoring worksheet outside the task-admission schema.
4. Audit every planned sequence for paired decisive/irrelevant and mixed-impact controls.
5. Create a preregistration draft with outcomes, denominators, cluster unit, missingness and stop/go gates; do not lock margins from post-hoc observations.
6. Audit the external observation registry for source revision, denominator, uncertainty, environment, cost and benchmark version completeness.
7. Map the current CEB portfolio to ownership boundaries and block overlapping registries, metrics and receipts.
8. Continue static negative-control design for admission, contamination, lifecycle and claim licensing.
9. Author and statically review six Rust task/oracle development candidates under [#10701](https://github.com/Luminous-Dynamics/symthaea/issues/10701) / [draft PR #10704](https://github.com/Luminous-Dynamics/symthaea/pull/10704). Reconcile their family mapping with #10617 and keep them unadmitted under #10609 until all source, oracle, custody and execution prerequisites are met.

Do not execute candidate-controlled build scripts, tests, macros, graders or attack payloads on the current host-opt-in runner. Do not claim independent regrading, sealed holdout, candidate execution or production isolation.

## 9. Current CEB maturity and limits

The November 5 work added or advanced:
- task admission and two-sided oracle qualification (#10609);
- release lifecycle (#10611);
- selective escalation (#10613);
- balanced sampling (#10617);
- comparative statistical claim qualification (#10624);
- evidence continuity protocol and measurement lane (#10631/#10632);
- T1–T4 contamination transparency and task/card binding (#10634);
- collaborative engineering protocol (#10640).

These remain open draft PRs, not merged releases. Their CEB Contract Validation V0 runs passed at their recorded exact heads, but that is contract validation only; general CI, execution isolation and independently trusted regrading are not implied. The task and contamination registries still contain zero admitted real tasks/cards, and no continuity or collaboration episodes have been executed. P0 #10064 and #10369 remain prerequisites for candidate execution and independent-regrade claims.

**The honest present claim:** CEB has a more developed design for evaluating engineering evidence, continuity and collaboration. It has not yet demonstrated that Symthaea outperforms any industry system on these dimensions.

## 10. Sources and comparison notes

- Scale AI, [SWE-Bench Pro V2 release/methodology](https://labs.scale.com/blog/swe-bench-pro-v2), September 22, 2026; [live leaderboard](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full), viewed November 6, 2026.
- Epoch AI, [SWE-Bench Pro review](https://epoch.ai/benchmarks/swe-bench-pro/review), September 1, 2026 (pre-V2 release; verdict scope must not be transferred to V2).
- Epoch AI, [Terminal-Bench 4.0.0 review](https://epoch.ai/benchmarks/terminal-bench-4/review), September 4, 2026; [Terminal-Bench 4.1 milestone](https://github.com/harbor-framework/terminal-bench/milestone/2), updated October 29, 2026; [official benchmark catalogue](https://www.tbench.ai/benchmarks), viewed November 6, 2026.
- Epoch AI, [included benchmark reviews](https://epoch.ai/data/benchmark-reviews-documentation/included-benchmarks), retrieved November 6, 2026.
- Dan et al., [SWE-Bench Pro V2: A Cleaner, Harder-to-Game Coding Benchmark](https://arxiv.org/abs/2610.24206), 16 October 2026.
- Deng et al., [SWE-Milestone: Evaluating AI Agents on Continuous Software Evolution](https://proceedings.mlr.press/v306/deng26d.html), ICML 2026, PMLR 306.
- Chen et al., [SWE-CI: Evaluating Agent Capabilities in Maintaining Codebases via Continuous Integration](https://arxiv.org/abs/2603.03823), 4 March 2026.
- Scale AI, [SWE Atlas v1.1](https://labs.scale.com/leaderboard/swe-atlas), updated September 22, 2026.
- Artificial Analysis, [Coding Agent Index V2 methodology](https://artificialanalysis.ai/methodology/coding-agents-benchmarking), October 2026; includes three-attempt task aggregation, resource accounting, reward-hacking review and an LLM-judge trajectory audit for passing Terminal-Bench runs.
- Endor Labs, [Meta's New Muse Wonder: record FuncPass, but mid-table SecPass](https://www.endorlabs.com/learn/metas-new-muse-wonder-record-funcpass-but-mid-table-secpass), November 3, 2026; documents the post-audit Agent Security League re-score and a training-recall case missed by the automated anti-cheating pipeline.
- Scale AI, [HiL-Bench overview and methodology](https://labs.scale.com/leaderboard/hil), 2026.
- Abundant AI, [SWE-Marathon v1.1](https://www.swe-marathon.org/), current leaderboard viewed November 6, 2026.
- Shetty et al., [GSO: Challenging Software Optimization Tasks for SWE-Agents](https://gso-bench.github.io/), leaderboard/methodology viewed November 6, 2026.
- Yin et al., [VerifierBench: Benchmarking the Generation of Executable Verifiers for LLM Agents](https://arxiv.org/abs/2610.28679), arXiv preprint, October 19, 2026; [project page and dataset overview](https://verifierbench.github.io/) (accessed November 6, 2026).
- Zhao et al., [SpecBench: Measuring Reward Hacking in Long-Horizon Coding Agents](https://arxiv.org/abs/2605.21384), May 20, 2026.
- Atinafu and Cohen, [RewardHackingAgents: Benchmarking Evaluation Integrity for LLM ML-Engineering Agents](https://arxiv.org/abs/2603.11337), March 11, 2026.
- Endor Labs, [Agent Security League](https://www.endorlabs.com/research/ai-code-security-benchmark), leaderboard viewed November 6, 2026.

**Cross-benchmark caveat:** every metric in this synthesis has its own task population, unit, harness, agent configuration, grader and release version. The numbers establish context and research motivation; they do not provide a direct CEB score conversion or a matched CEB-vs-industry comparison.