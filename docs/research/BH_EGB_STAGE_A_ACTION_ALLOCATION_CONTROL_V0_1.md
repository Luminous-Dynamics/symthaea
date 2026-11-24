# Stage-A research-action allocation control v0.1

**Status:** synthetic exact-known-answer control for the Stage-A / Stage-B separation. It is not an EGB prior, physical prediction, real solver-cost estimate, calibrated Symthaea forecast, or an implementation of a new scheduler.

Machine-readable fixture: [BH_EGB_STAGE_A_ACTION_ALLOCATION_CONTROL_V0_1.json](BH_EGB_STAGE_A_ACTION_ALLOCATION_CONTROL_V0_1.json).

## What this is testing

Stage A chooses **whether a research action is worth purchasing**. Stage B chooses **how to configure a selected numerical experiment**. The two decisions have different objectives.

A Stage-B experiment can sharply distinguish two models and still be a poor Stage-A purchase if its cost exceeds the value of the resulting decision improvement. A symbolic derivation may be cheap and decisive. A diagnostic may reduce all epistemic uncertainty without changing any downstream decision.

This fixture demonstrates both failure modes and a separate risk-gate veto. It reuses #8721/#5476/#5478/#11171 ownership; it does not introduce an action scheduler, VoC predictor, posterior service, or execution authority.

## Exact finite-world cases

### Case 1 — a cheap derivation beats an expensive BVP

There are two equally likely synthetic states, H_A and H_B, and two downstream choices, choose_A and choose_B. The correct decision yields utility 1; the wrong decision yields 0.

Before any action:

\[
V_0=\max\{0.5(1)+0.5(0),\;0.5(0)+0.5(1)\}=0.5.
\]

A perfectly discriminating action reveals which state is true. After either possible outcome the best decision has utility 1, so:

\[
\operatorname{EVSI}=1-0.5=0.5.
\]

The candidate action comparison is:

| Action | Gross EVSI | Cost | Net decision value | Risk gate |
|---|---:|---:|---:|---|
| Exact symbolic derivation | 0.50 | 0.10 | **+0.40** | Eligible |
| Degenerate-parameter BVP | 0 | 0.15 | -0.15 | Eligible, but wasteful |
| Precision increase with shared output | 0 | 0.02 | -0.02 | Eligible, but wasteful |
| Expensive discriminating BVP | 0.50 | 0.70 | -0.20 | Eligible, but uneconomic |
| Risky discriminating probe | 0.50 | 0.05 | +0.45 | **Vetoed** |

The known answer is to purchase the symbolic derivation. The more informative expensive BVP is not worth its cost; the high-risk probe is not legal under the frozen risk limit even though its net value is positive.

### Case 2 — information gain without decision value

There are again two equally likely states, but continue_common_plan yields utility 1 in both states while stop yields 0 in both.

A perfect diagnostic identifies the hidden state, reducing uncertainty by one bit. However, the optimal decision remains continue_common_plan regardless of the outcome:

\[
V_0=1,\qquad
\mathbb{E}[V_{\mathrm{after}}]=1,\qquad
\operatorname{EVSI}=0.
\]

At cost 0.04, net decision value is -0.04. The known answer is **do not purchase the diagnostic** if its only proposed benefit is changing that decision.

This is a deliberately stark control for the proposition:

\[
\text{information gain} \ne \text{decision value}.
\]

Scientific learning can still have independent value for future work, claim qualification, safety, or generalization. Such benefits must be declared in the utility/decision model rather than silently smuggled into an information-gain score.

## Computation contract

For a finite state set s, a frozen prior P(s), action a, outcomes o, likelihoods P(o|s,a), and declared downstream utility U(d,s):

\[
P(o\mid a)=\sum_s P(s)P(o\mid s,a)
\]

\[
P(s\mid o,a)=\frac{P(s)P(o\mid s,a)}{P(o\mid a)}
\]

\[
V(P)=\max_d \sum_s P(s)U(d,s)
\]

\[
\operatorname{EVSI}(a)=
\sum_o P(o\mid a)V(P(\cdot\mid o,a))-V(P)
\]

\[
\operatorname{NetValue}(a)=
\operatorname{EVSI}(a)-C_{\mathrm{direct}}(a)-C_{\mathrm{opportunity}}(a).
\]

The synthetic fixture sets opportunity cost to zero, so the listed net value is gross EVSI minus direct cost. A real campaign must declare opportunity cost, resource vector, utility units, risk bounds, and action legality before outcomes are observed.

The purchase rule is not “choose the highest score regardless of constraints.” It is:

1. action is in the allowed action set and within budget;
2. required risk/downside gates pass;
3. expected net decision value is positive under a calibrated or exactly enumerated forecast;
4. forecast/action choice is bound to the evidence-prefix identity before execution;
5. unknown probabilities, utilities, or costs remain unknown and yield bounded/ranked qualitative output or UNRESOLVED.

## What the fixture does not establish

- It does not estimate real EGB action costs or probabilities.
- It does not provide a calibrated VoC model for Symthaea.
- It does not establish that symbolic derivation is always better than numerical work.
- It does not rank Stage-B parameter settings.
- It does not permit a researcher to change its own risk constraints or evaluator.
- It is not a result from the live EGB campaign.

## How it should be used

1. Add it as a known-answer synthetic episode to #13418's Stage-A protocol.
2. Use the exact finite enumerator as an oracle diagnostic on this tiny world only.
3. Compare no-tool, fixed, random matched-cost, simple heuristic, calibrated prospective VoC and oracle arms at identical budgets.
4. Check forecast/action identity is immutable before outcome exposure.
5. Include a control where the outcome changes the hypothesis but not the optimal decision.
6. Include a high gross/net value action that still fails a predeclared risk gate.
7. Keep Stage-B discrimination score and Stage-A net decision value in separate report fields.

Passing this fixture is policy-contract conformance, not evidence that any adaptive agent performs better than fixed or simple baselines.
