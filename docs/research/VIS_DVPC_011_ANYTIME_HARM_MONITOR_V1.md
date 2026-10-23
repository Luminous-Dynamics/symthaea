# VIS-DVPC-011: Anytime-Valid Harm Monitoring v1

**Status:** Draft, standalone Rust reference.  
**Issue:** [#8556](https://github.com/Luminous-Dynamics/symthaea/issues/8556)  
**Fixed-sample certificate:** [PR #8553](https://github.com/Luminous-Dynamics/symthaea/pull/8553)  
**Calibrated router:** [PR #8532](https://github.com/Luminous-Dynamics/symthaea/pull/8532)  
**Prior art:** [Active, Anytime-Valid Risk-Controlling Prediction Sets (NeurIPS 2024)](https://www.microsoft.com/en-us/research/publication/active-anytime-valid-risk-controlling-prediction-sets/); [Conformal Policy Control (ICML 2026)](https://proceedings.mlr.press/v306/prinster26a.html)

## Why this is separate from fixed-sample certification

VIS-DVPC-010's Clopper–Pearson bound is appropriate for an independent, fixed certification split and a frozen policy. Repeatedly inspecting fixed-time confidence intervals while new outcomes arrive is a different statistical problem: optional stopping can break the nominal guarantee.

This prototype maintains an anytime-valid likelihood-ratio e-process for each predeclared group. It consumes **confirmed post-action harm labels for selected escalations only**. It is not a model that predicts whether the next action is harmful.

## Construction

For a fixed policy and group, let (Y_t in {0,1}) indicate whether the (t)-th selected escalation was harmful, and let (r) be the predeclared maximum harm rate. For each fixed alternative (q_j = c_j r), where (c_j in {0.10,0.25,0.50,0.75,0.90}), maintain

[
L_{j,n} = left(rac{q_j}{r}ight)^{sum_{t=1}^{n}Y_t}
left(rac{1-q_j}{1-r}ight)^{n-sum_{t=1}^{n}Y_t}.
]

The fixed equal-weight mixture is

[
E_n = rac{1}{5}sum_{j=1}^{5} L_{j,n}.
]

Under the conditional null that the harmful-outcome probability at each selected escalation is at least (r), each fixed-(q_j) likelihood-ratio process is a nonnegative supermartingale; a predeclared mixture remains one. Ville's inequality gives

[
Pr_{H_0}left(sup_n E_n ge 1/delta_gight) le delta_g.
]

The code works in log space and recomputes from integer harm/event counts on every update, avoiding underflow and reducing order-dependent floating-point artifacts.

The family-wise budget is Bonferroni allocated over all predeclared policies and groups:
[
delta_g = rac{delta_{mathrm{family}}}
{N_{mathrm{policies}} N_{mathrm{groups}}}.
]

A group is currently labeled **CERTIFIED** only while its current e-value is at or above (1/delta_g) after at least one event. It can return to **MONITORING** after subsequent harmful outcomes lower the e-value. The global monitor reports **CERTIFIED** only when every predeclared group is currently certified, **INSUFFICIENT_EVIDENCE** while at least one group has no escalation outcomes, and **MONITORING** otherwise.

## Fail-closed boundaries

- Policy ID/digest and loss-profile ID are frozen. A mismatch revokes the monitor.
- An OOD/unknown state or a changed domain revokes the monitor irreversibly. Anytime validity does not solve arbitrary distribution shift.
- Duplicate trace IDs and undeclared groups are rejected. An undeclared group revokes the monitor because the declared multiplicity/group family is incomplete.
- Group IDs are fixed before monitoring, and the correction covers the predeclared policy-family size. Changing policies requires a new separately controlled monitor.
- One trace ID must represent an independent evaluation episode/cluster. Adjacent video frames from the same drive should not be counted as independent Bernoulli trials.
- The monitor sees outcome labels only after the action. The implementation is not a secure process boundary, signed attestation, or online action controller.

## Adversarial tests

The twelve authored Rust tests cover:
1. zero observations as insufficient evidence;
2. enough clean outcomes crossing the anytime threshold;
3. harmful outcomes failing to certify low risk;
4. later harms lowering the current certificate to monitoring;
5. a certified common group not covering an unseen rare group;
6. global status waiting for every predeclared group;
7. count-based e-values and receipts invariant to outcome order;
8. absorbing shift revocation;
9. stale policy digest revocation;
10. duplicate traces and undeclared groups;
11. larger policy-family multiplicity raising the certification threshold;
12. duplicate/invalid config rejection.

These tests validate the reference calculations and state machine, not false-certification probability on real streams. The anytime guarantee additionally requires the conditional-risk null and frozen policy/domain assumptions.

## Benchmark plan

Replay chronological, held-out escalation outcomes and compare with the VIS-DVPC-010 fixed-sample gate only at predeclared look times. Report false-certification probability under null simulations, time-to-certificate under lower-risk alternatives, subgroup detection delays, shift-revocation delay, coverage by group, and policy utility/latency/energy separately. Simulated false-certification tests are useful stress tests but do not replace the supermartingale argument.

A meaningful result is valid repeated monitoring without optional-stopping inflation, with all groups and policy versions predeclared. No empirical real-world harm reduction, safety guarantee under arbitrary shift, or SOTA superiority is claimed.
