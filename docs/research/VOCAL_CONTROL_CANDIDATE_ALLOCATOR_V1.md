# VCCA-001 — Uncertainty-bounded vocal control candidate allocation

**Date:** 11 November 2026  
**Status:** finite-set allocator contract and synthetic tests; no audio-control result claimed  
**Owner:** https://github.com/Luminous-Dynamics/symthaea/issues/11543

## Why it is a new layer

VCRB-001 measures control-to-acoustic response and off-diagonal coupling. VNOC-001 estimates no-op variability. VIA-001 makes interval-aware conservative decisions. VCCR-001 qualifies a correction trajectory. None chooses a next action.

VCCA-001 fills this allocation gap without conflating measurement, optimization and action authority.

## Robust finite candidate selection

The input is a calibrated local response matrix and a finite set of candidate control vectors. For each acoustic metric, the model predicts a center and worst-case uncertainty radius:

~~~text
center_m = sum_i coefficient[m,i] * delta_control_i
radius_m = residual_uncertainty_m
           + sum_i coefficient_uncertainty[m,i] * abs(delta_control_i)
interval_m = [center_m - radius_m, center_m + radius_m]
~~~

Target intervals must be wholly contained in their signed requested response bands. Protected intervals must be wholly contained in calibrated symmetric no-op guards. Candidate effort is the weighted sum of squared normalized control deltas.

Among feasible candidates, select the one with minimum effort, tie-broken lexically by candidate ID. If none are robustly feasible, return an explicit abstention with per-candidate rejection reasons. Do not weaken tolerances to force an action.

This is an optimizer over a finite, predeclared candidate set. It is not a continuous QP solver, unconstrained pseudoinverse, global optimum, or proof of validity outside the local response envelope.

## V1 Rust implementation

The crate at tools/vocal-control-candidate-allocator:
- binds response profile, response matrix, frozen context and calibration digests;
- requires a complete metric × control sensitivity matrix with explicit units;
- declares supported controls, bounds, trust radii, effort scales and weights;
- declares target bands and protected absolute-delta guards;
- propagates coefficient uncertainty and model-residual uncertainty conservatively;
- rejects unsupported, out-of-bounds, over-trust, over-budget and wrong-unit candidates;
- emits PROPOSAL_ONLY on a robust feasible selection;
- abstains if all candidates fail and preserves all candidate rejection reasons.

No renderer interface or action authorization exists in this crate.

## Integration path

~~~text
VCRB paired interventions
  -> calibrated response coefficients + coupling cells
VNOC no-op split
  -> protected tolerances / residual bounds
VIA interval assurance
  -> conservative uncertainty provenance
VCCA finite candidate allocator
  -> minimum-effort PROPOSAL_ONLY candidate
renderer capability gate
  -> separately authorized operation (or reject)
fresh acoustic analysis
  -> VCCR trajectory evaluation + VMEA task/preservation score
~~~

The calibration profile must be backend/voicebank/analyzer/source-context-specific. The confirmatory evaluation must use disjoint singer/song/session cases. A profile change invalidates the old allocation proposal.

## Matched experiment

Compare:
1. fixed-step manual correction;
2. greedy one-control-at-a-time;
3. unconstrained damped pseudoinverse;
4. regularized least-squares allocation with box bounds;
5. VCCA robust finite candidate search.

Use identical source performances, backend/voicebank artifacts, initial score/pronunciation, no-op profile, target request, action candidate set, render budget and independent analyzers.

Report:
- target success/convergence per metric and accepted target interval;
- protected guard violations, worst excursion and false-safe proposal rate;
- intervention norm, bound utilization and abstention rate;
- render count/cost and corrections-to-convergence distribution;
- calibration coverage / empirical prediction interval coverage;
- stress under coefficient shift, residual underestimation and unsupported-control requests;
- separate singer/register/technique/SNR/backend strata.

A proposal is not a successful correction. Promotion requires measured post-render target compliance and preservation.

## Qualification targets

Synthetic controls:
- 100% detection of malformed/missing/duplicate sensitivity cells and control vectors;
- no robust proposal when any target/protected interval crosses its acceptance boundary;
- 100% rejection of unsupported, out-of-bounds, over-trust and over-budget controls;
- deterministic selection for the same profile/candidate set;
- every no-feasible case explicitly abstains.

Held-out acoustic pilot targets (not achieved results):
- at least 90% target attainment on supported interventions initially, progressing toward 95%;
- zero protected-property violations in the accepted-converged cohort;
- reduce render count or intervention effort versus fixed-step and greedy baselines at equal output quality;
- publish false-safe and abstention bounds with singer/session-clustered uncertainty.

## Limitations and safeguards

- Linear sensitivity is only a local approximation. The declared trust radius and residual uncertainty are mandatory; response drift invalidates the profile.
- Correlated coefficient errors require a joint uncertainty envelope or conservative worst-case bound. V1's sum-of-absolute-errors formula is intentionally conservative under the assumed cellwise error bounds; it is not a statistical confidence interval.
- If input confidence bounds are under-calibrated, robust arithmetic cannot rescue them. Measure coverage on held-out perturbations.
- The candidate set can omit the best feasible action; report its coverage and construction. The allocator does not claim global optimality.
- Action proposals must never bypass backend capability and safety gates.

## Related work

- #11401 VCRB-001 control-response matrix
- #11420 VNOC-001 no-op calibration
- #11456 VIA-001 interval assurance / PR #11461
- #11482 VCCR-001 correction trajectory / PR #11488
- #11503 VMEA-001 task-aware edit assurance / PR #11506
- #11356 VocalEditPatch
- AES 2026 Singing Voice Synthesis via Guided Diffusion: https://aes.org/publications/elibrary/elibrary-page/?id=23501
- SegTune (ACL 2026), temporally structured fine-grained song control: https://aclanthology.org/2026.acl-long.586/
