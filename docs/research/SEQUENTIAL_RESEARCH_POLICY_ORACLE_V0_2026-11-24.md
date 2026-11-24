# Exact Sequential Policy Oracles for Decision-Grade Scientific Research

**Snapshot:** 2026-11-24  
**Status:** Research and implementation contract; no implementation or benchmark qualification is claimed.  
**Owner:** Symthaea #13567, composed with #12546 and #11171.

## 1. Executive thesis

A scientific agent should be evaluated on the quality of the *complete research policy* it follows, not only on the information gain or decision value of its next experiment.

The important question is:

> Given the evidence available now, which sequence of permitted investigations gives the best defensible terminal decision under the complete resource and deadline constraints?

A locally attractive experiment can be globally inferior. An observation can reveal many bits about nuisance variables while doing nothing for the decision. Conversely, a low-immediate-value measurement can reveal which follow-up experiment is informative, making the sequence valuable even when the first action does not change the current terminal decision.

We therefore need an exact oracle for tiny finite worlds. It provides a known-answer reference for policy regret and a falsifiable target for the actual Symthaea research policy.

This is an evaluator-side diagnostic. It is not a production scheduler, learned researcher, physical experiment system, or execution authority.

## 2. Verified starting point and gap

At the public repository main snapshot used for this work:

- `research/decision-predictability-benchmark/src/lib.rs` defines a finite evaluator whose `CandidateSubmission` contains one optional `requested_probe_id` and whose evaluation result contains one `requested_probe`.
- #11060 provides a blind AgentView/AgentAction process protocol, finite reference worlds and seven one-step experiment-policy baselines. Its process bridge is explicitly process separation, not an OS sandbox.
- #11171 owns connecting a real Symthaea research-policy callable to blind episodes. It explicitly states that the finite reference policy is not the learned/production researcher.
- #12546 owns the decision-relevant predictability benchmark and its Bayesian and prior-free decision-value context.
- #13545 separates unconstrained robust minimax loss from current and post-observation risk-gate status. It complements this proposal; it does not solve multi-step policy planning.

The missing capability is an exact, independently cross-checked **multi-step policy-tree reference**. Without it, a policy can beat a one-step baseline while remaining far from the best possible strategy on the small worlds where exact comparison is feasible.

## 3. Policy and information semantics

A policy is a mapping from public histories to either another permitted research action or a caller-declared terminal decision:

`\pi: h_t \mapsto a_t \text{ or } d_t`

A public history contains only information that the candidate could legitimately have observed: prior action IDs, returned outcome IDs, evidence-prefix identity, declared time/resource state and protocol-visible status. It must not contain hidden generating state, evaluator truth, future outcomes, hidden seeds or scores.

The evaluator may use hidden truth to calculate the final score, but that truth and its derived score remain outside the candidate process until submission is final.

### 3.1 Prior-free robust terminal risk

For a non-empty possible-state set \(S\), define:

`R(S) = \min_{d\in D} \max_{s\in S} L(d,s)`

where \(D\) is exactly the caller-declared terminal decision set and \(L\) is the frozen bounded task-loss table.

For experiment \(a\), let \(S_{a,o}\) be the non-empty set of states consistent with observable outcome \(o\). The oracle does not attach probabilities to these sets.

### 3.2 Robust sequential dynamic programme

Let:

- \(\mathbf B\) be the remaining resource budget vector;
- \(h\) be the remaining finite decision horizon;
- \(q\) be the finite protocol state derivable from public history (for example, already-used actions, cooldown state or branch-specific action availability);
- \(c_a\) be the declared resource vector consumed by action \(a\);
- \(\Delta_a\) be its declared result delay.

The robust value is:

`V_R(S,\mathbf B,h,q)=\min\left\{
R(S),\;
\min_{a\in A_{\mathrm{legal}}(S,\mathbf B,h,q)}
\max_{o:S_{a,o}\ne\varnothing}
V_R(S_{a,o},\mathbf B-\mathbf c_a,h-\Delta_a,q_{a,o})
\right\}
`

The stop term \(R(S)\) means that the agent may finish now with the best currently defensible declared decision. The action term considers only actions that are permitted, affordable in every required resource dimension, result before the decision deadline, and legal after the public history.

The maximum over possible outcomes is the prior-free worst case. Every branch can choose a different continuation based on its observed outcome, but it cannot choose based on the hidden state within an overlapping support set.

This recurrence is a finite-horizon reference, not a claim that all real scientific decisions are finite or fully observable.

### 3.3 Bayesian sequential dynamic programme

For a separate, defensible prior and observation-likelihood model, define terminal Bayes risk:

`R_B(p)=\min_{d\in D}\sum_s p(s)L(d,s)`

For action \(a\) with outcomes \(o\), use the predictive outcome probability \(P(o\mid p,a)\) and the correctly conditioned posterior \(p_{a,o}\):

`V_B(p,\mathbf B,h,q)=\min\left\{
R_B(p),\;
\min_{a\in A_{\mathrm{legal}}}
\sum_o P(o\mid p,a)
V_B(p_{a,o},\mathbf B-\mathbf c_a,h-\Delta_a,q_{a,o})
\right\}
`

This is an expected terminal decision-loss objective under resource constraints. Gross EVSI is derived by comparing the no-acquisition terminal risk with the best expected terminal risk after the permitted policy. EIG, mutual information and entropy remain separate diagnostics; none is silently substituted for decision value.

The robust and Bayesian recurrences must remain separate APIs, test suites and result labels. A missing prior or likelihood does not authorize inventing one.

## 4. Resource, timing and stopping rules

Use a resource vector rather than a hidden scalar penalty:

`\mathbf B=(B_{\mathrm{experiment}},B_{\mathrm{compute}},B_{\mathrm{energy}},B_{\mathrm{time}},\ldots)`

An action is feasible only when every required component remains within its declared budget and the result arrives before the decision deadline. The implementation must not subtract dollars, seconds, joules, token counts or experiment units directly from task loss.

For each tiny world, either:

1. enumerate the attainable terminal-loss/resource Pareto frontier; or
2. use a predeclared, application-owned utility conversion.

The first implementation should prefer the Pareto frontier. Any scalarized objective must record the exact conversion rule and its owner.

The finite protocol state \(q\) must account for action repetition, cooldowns, branch-specific availability and any other rule that affects future legal actions. The horizon and action legality rules prevent infinite zero-cost loops. Equal-value policies must use deterministic tie-breaking.

Stopping is not an implicit failure. A stop chooses the best action from the declared terminal decision set. The oracle must not silently add an abstain, defer, fallback or safety action that the caller did not declare.

## 5. Canonical non-myopic challenge family

Use an explicit 16-state world with:

- \\(x\\in\\{0,1\\}\\): the decision-relevant hidden mechanism;
- \\(z\\in\\{0,1\\}\\): the branch selector;
- \\(n\\in\\{0,1,2,3\\}\\): a nuisance variable.

The state is \\(s=(x,z,n)\\). The only terminal decisions are \\(d_0,d_1\\), with frozen loss \\(L(d,s)=0\\) when \\(d=x\\) and \\(L(d,s)=1\\) otherwise. The prior-free initial minimax loss is therefore 1.0.

Define the actions:

- **A — nuisance survey:** returns the exact value of \\(n\\), with four possible outcomes. Every outcome support still contains both values of \\(x\\), so the worst-case terminal loss remains 1.0. Under a uniform diagnostic distribution used only by the entropy-greedy baseline, A reveals 2 bits.
- **B — branch probe:** returns the exact value of \\(z\\), with two possible outcomes. Each support still contains both values of \\(x\\), so B alone also leaves worst-case terminal loss at 1.0. Under the same uniform diagnostic distribution, B reveals 1 bit. Its public outcome transitions the protocol to the matching branch state.
- **C_z — branch-specific diagnostic:** only after B returns \\(z\\), the corresponding permitted C action returns \\(x\\). Each resulting support contains only one decision-relevant value, so worst-case terminal loss becomes 0.0.

Use a two-action horizon and a two-unit experiment budget. A is a terminal dead-end in the protocol; B unlocks exactly one valid C_z follow-up. Thus B-then-C is feasible and achieves worst-case loss 0.0, while A cannot resolve the terminal decision. The uniform distribution is used only to show why an entropy-greedy baseline prefers A; the robust oracle itself remains prior-free.

Freeze the state space, observation supports, loss table, protocol transitions, resource costs and horizon before candidate testing. The independent exhaustive tree enumerator and dynamic programme must both return the B-then-C policy and worst-case terminal loss 0.0.

This example makes three properties testable:

1. information about nuisance variables is not decision value;
2. the first experiment can have option value even when its immediate terminal decision value is zero;
3. the best continuation depends on the observation, so a fixed sequence is not generally equivalent to an adaptive policy tree.

Do not tune the example after observing candidate-policy results.

## 6. Required controls

The frozen suite should include at least:

1. **Non-myopic unlock:** low-immediate-value B enables C and beats one-step greedy selection.
2. **High-entropy nuisance:** entropy reduction is large while terminal decision value remains zero.
3. **Duplicate observation:** repeating a deterministic duplicate cannot create additional information or improve the exact oracle.
4. **Overlapping support:** one outcome remains consistent with multiple states; the robust branch retains all of them.
5. **Unsafe-to-safe transition:** current robust risk gate fails, but every outcome after a permitted probe admits a safe declared decision.
6. **Partial resolution:** the sequence reduces worst-case loss but does not pass the risk gate; it must not be labelled safe.
7. **Model inadequacy:** an evaluator-only counterexample outside the candidate's declared model family prevents a false singleton; compose with #11197.
8. **Permission failure:** a high-value but forbidden experiment is not a legal branch.
9. **Budget boundary:** a sequence whose total resource vector exceeds any hard budget is rejected, even if its terminal value is attractive.
10. **Deadline boundary:** a useful experiment that returns too late cannot be selected for the current decision.
11. **Irrelevant metadata:** perturbing non-decision-relevant metadata does not change the policy.
12. **Deterministic replay:** identical frozen inputs produce identical oracle policy, tie resolution and receipt.

For each world, compare the dynamic programme with a **separately implemented exhaustive policy-tree enumerator**. Sharing the same recurrence helper would weaken the independence of this known-answer check.

## 7. Tournament and metrics

Compare the following policies on identical public starting states and inclusive budgets:

- fixed predeclared sequence;
- random legal policy matched on resources;
- greedy entropy/EIG (Bayesian lane only);
- greedy one-step robust minimax reduction;
- greedy one-step Bayesian EVSI;
- the strongest simple threshold or fixed-cadence baseline where appropriate;
- the frozen actual Symthaea research policy connected through #11171;
- exact sequential policy oracle, evaluator-only diagnostic upper reference.

Report a vector, not an opaque composite score:

- terminal Bayes risk or worst-case loss, with the estimand identified;
- one-step-to-oracle policy regret;
- terminal risk-gate pass rate and false-safe rate;
- correct/incorrect terminal decisions and declared abstention outcomes;
- policy depth, branch coverage and number of experiments;
- total experiment, compute, energy, time, communication and memory consumption;
- prospective action-value forecast calibration and realized decision impact;
- legal-action violations, deadline misses, invalid outcomes, timeouts and incomplete traces;
- evidence-prefix integrity and exact replay status.

Use paired worlds and seeds. Cluster uncertainty estimates by generator family and parameter family rather than treating correlated windows as independent samples. Preserve every failed, incomplete and invalid run in the denominator.

The oracle is an upper reference for its finite world, not an attainable competitor and not a claim about real-world scientific optimality.

## 8. Integration boundaries

Compose with existing owners:

- **#12546:** decision-relevant predictability and decision-oracle benchmark.
- **#11060:** blind AgentView/AgentAction wire protocol, finite reference worlds and process bridge.
- **#11171:** production Symthaea research-policy callable and blind-policy integration.
- **#11012 / #11015:** adaptive-versus-fixed workflow comparison and prospective branch/action traces.
- **#8721 / #5476 / #5478:** value-of-computation, prospective resource allocation and metareasoning.
- **#11197:** model misspecification, support omission and unknown mechanisms.
- **#9332 / #10756:** sealed evaluator custody and evaluator-quality controls.
- **#13545 / #13534:** robust minimax value and risk-gate semantics.

This work does not create a second scheduler, posterior engine, receipt service, evidence ledger, memory store, evaluator custodian, permission authority or execution path. The production researcher remains the candidate; the exact oracle remains evaluator-side.

The child process in #11060 is not an OS sandbox. Untrusted candidate code requires a separately qualified OS/container isolation boundary before confirmatory execution.

## 9. Qualification gates and targets

### Exact-reference gates

- Dynamic programming agrees with independent exhaustive enumeration on every frozen tiny-world known-answer case.
- Exact robust and Bayesian oracle fixtures pass independently.
- All action, permission, budget, deadline and non-anticipativity invariants pass negative controls.
- Identical inputs yield exact deterministic replay.
- No evaluator-only state, seed, truth, score or oracle handle enters the candidate process.

### Candidate-policy targets

Initial promotion hypotheses, not achieved results:

| Metric | Initial target |
|---|---:|
| Tiny-world exact policy-oracle agreement | 100% for the independent oracle implementations |
| False-safe results on frozen negative controls | 0 |
| Positive-control detection | ≥95% |
| Correct insufficiency/abstention on frozen controls | ≥95% |
| Decision regret vs. strongest matched baseline | ≥10% lower |
| Resource use at unchanged decision risk | ≥20% lower in at least one measured dimension, with no hidden increase elsewhere |
| Evaluator leakage | 0 |
| Invalid/missing run accounting | 100% |
| Reproducibility | Exact task, source, model, config, budget, evaluator epoch, artifacts and replay receipts |

The 10% decision-regret and 20% resource-efficiency targets require paired uncertainty intervals, held-out task-family confirmation and no material critical-risk regression. They are not SOTA claims. External benchmark scores must follow each benchmark's official scorer and remain separate from these internal gates.

## 10. Relationship to current industry benchmarks

These external suites test related but non-identical constructs. They should be used as complementary comparison tracks, not collapsed into a single leaderboard.

| Benchmark | Public reference checked 2026-11-24 | What it measures | Proper use here |
|---|---:|---|---|
| Autonomous Science Bench 0.1 | 72.2% leading listed resolution rate, six pilot campaigns | Closed-loop experiment selection and learning over laboratory workflows | Closest external comparison for multi-round scientific action policies; reproduce its discovery and held-out learning gates separately |
| ASI-Bench | 63.90 leading listed overall score, 60 tasks | Project-level scientific autonomy over B1–B4 guidance levels | Methodological-autonomy comparison; published evaluation has no external tools and the scored task-set identifier is not supplied |
| SciWorkBench | 57.38 best reported mean | End-to-end scientific workflows across biology, chemistry, materials and physics | Workflow-procedure, multimodal evidence and self-correction comparison |
| FIRE-Bench | 46.7 leading published F1 over 30 tasks and three trials | Verifiable rediscovery of claims from published papers | Claim-support and planning comparison, not a physical experiment-success metric |
| BIABench | 16 real-world tasks, separate outcome and process scores | Raw 2D/3D/time-lapse bioimage analysis and QC | Evidence-interpretation track; preserve the official outcome/process split |
| AstaBench | Around 0.53 best score across 2,400+ problems; about $3.40/problem in the AI Index snapshot | Broad scientific research workflows and cost | Secondary breadth/cost reference; not directly comparable to sequential closed-loop resolution |
| NeuroBench Mackey–Glass | 14 frozen delay parameterizations with Lyapunov-time references | Nonlinear time-series prediction | Separate dynamical-prediction baseline lane for #12546, not a research-policy leaderboard |

A benchmark-specific SOTA claim requires identical task versions, model/tool access, scoring, resource boundaries and run policy. Where any of those differ, report a compatibility gap rather than a rank.

AutoSciBench is relevant as an adaptive **benchmark-construction method**, not an agent-performance leaderboard. Its reported 22.4- and 25.5-percentage-point reductions are solver-accuracy changes on generated variants relative to human-curated computational-biology and materials-science tasks; its task-quality comparisons use LLM judges. The generated benchmark and its independent oracle must remain separate from the candidate system's authority.

## 11. Implementation sequence

1. Freeze the task schema, resource-vector semantics, public protocol state and terminal decision contract.
2. Add tiny-world fixtures and an independent exhaustive tree enumerator.
3. Implement the bounded robust dynamic programme and compare every result against exhaustive enumeration.
4. Add the Bayesian dynamic programme as a separate implementation with separately validated priors and likelihoods.
5. Integrate one-step, fixed-sequence, random, entropy and EVSI baselines.
6. Pin the actual Symthaea research-policy entry point and connect it via #11171.
7. Execute blind paired campaigns on held-out families, recording complete resource and failure denominators.
8. Only after those gates, consider external AS-Bench campaign execution and broader benchmark submissions.

Static source review, schema checks, fixture construction and independent mathematical enumeration can proceed without GitHub runners. Rust formatting, compilation, tests, strict Clippy and exact-head CI remain mandatory before implementation qualification.

## References

- Autonomous Science Bench 0.1 leaderboard and methodology: https://asb.open-ssi.org/leaderboard and https://asb.open-ssi.org/about
- ASI-Bench leaderboard and scoring methodology: https://asibench.apexin.ai/leaderboard/
- SciWorkBench (2026): https://arxiv.org/abs/2611.22490
- FIRE-Bench: https://firebench.github.io/
- BIABench (2026): https://arxiv.org/abs/2609.34274 and https://github.com/BIABench/BIABench
- AstaBench coverage in Stanford AI Index 2026: https://hai.stanford.edu/assets/files/ai_index_report_2026_chapter_5_science.pdf
- NeuroBench Mackey–Glass frozen data: https://huggingface.co/datasets/NeuroBench/mackey_glass
- AutoSciBench (2026): https://arxiv.org/abs/2610.05140

## Claim boundary

This document specifies a research and implementation contract. It reports no Rust implementation, test pass, policy-oracle result, learned Symthaea research capability, external benchmark reproduction, SOTA improvement, physical experiment or scientific discovery.
