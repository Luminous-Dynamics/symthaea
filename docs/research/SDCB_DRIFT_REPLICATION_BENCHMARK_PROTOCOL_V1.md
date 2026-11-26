# SDCB drift-aware and multi-instrument benchmark protocol v1

**Status:** preregistration-style specification; no tournament has been run.  
**Reference under test:** `scripts/research/sdcb_drift_replication.rs`  
**Analytical fixture:** `SDCB_DRIFT_REPLICATION_V1.example.json`

## 1. Claim under test

Evaluate whether a scientific decision agent can distinguish:
- repeated observations from the same instrument with a shared latent bias;
- observations from independently biased instruments;
- observations made before and after instrument-state drift;
- calibration evidence that is current versus stale;
- calibration that improves a scientific decision versus calibration that only reduces nuisance uncertainty.

The target claim is:

> At matched task information and resource budgets, an instrument-state-aware policy lowers held-out decision regret under calibration drift and shared measurement bias, while avoiding unnecessary calibration when an independent, better-qualified measurement path is available.

This is a research target, not an achieved result.

## 2. Task families

| Family | Hidden measurement process | Required capability |
|---|---|---|
| D0: static known device | Stable, correctly calibrated instrument | Avoid redundant calibration |
| D1: static unknown offset | Persistent device-specific offset | Distinguish science state from instrument state |
| D2: same-device replication | Repeated readings share one latent device offset | Do not treat same-device repeats as independent calibration evidence |
| D3: independent-device replication | Separate instruments have independent latent offsets and residual noise | Value cross-device confirmation when it changes a decision |
| D4: stale calibration | Device state evolves after calibration | Model calibration freshness instead of using a fixed accuracy |
| D5: gradual drift | Drift probability depends on elapsed time | Requires a later continuous-time/variable-step extension |
| D6: batch common-cause | Multiple devices share a latent batch/environment effect | Detect when apparent cross-device independence is false |
| D7: calibration intervention | Calibration changes/resets instrument state | Requires a state-action transition model, not only passive Markov drift |
| D8: source mismatch | Device firmware, asset identity, or calibration certificate is misbound | Reject evidence before any belief mutation |
| D9: non-identifiability | Competing science and sensor models induce the same observable distribution | Abstain from unsupported mechanism claims |

D0–D4 and D9 can be represented in the current finite-state reference. D5, D6, and D7 are explicit later-model requirements. D8 requires production EGBT/attestation integration and is not implemented by this oracle.

## 3. Matched comparator arms

Use the same scientific hypotheses, device prior, candidate experiments, scenario set, utility schedule, allowed outcomes, and hard budget:

1. No experiment / best current action / abstention.
2. Repeat-only policy on one device.
3. Fixed calibration-first policy.
4. Fixed independent-device replication.
5. Greedy EIG across all candidate measurements.
6. R-I / R-IDeA, where their target-representativeness and proxy assumptions apply.
7. One-step EVSI.
8. Exact two-stage decision oracle on small instances.
9. Finite-scenario robust maximin policy.
10. Multi-fidelity Bayesian optimization where candidate fidelity and source assumptions match.
11. Markov instrument-state-aware policy.

The exact finite oracle is a small-instance reference, not a deployable algorithm. Preserve each baseline's original objective and implementation. Do not report raw acquisition scores as though EIG, R-IDeA, EVSI, and robust utility were commensurable.

## 4. Negative controls

Freeze the following controls:

- Repeating a device with a persistent unknown offset must not be credited as independent-device replication.
- Calibration must be independent of the scientific hypothesis conditional on the named device state.
- A stale calibration should lose value as the probability of a state transition increases.
- A calibration record from device A must not update device B's instrument belief.
- Two devices exposed to a common-cause state must not be treated as conditionally independent if the task generator encodes that shared cause.
- If every plausible state selects the same terminal action, nuisance calibration must not be rewarded solely for reducing sensor-state entropy.
- Any invalid/stale/duplicate/misbound/revoked measurement receipt must cause zero scientific-belief and instrument-belief mutation once EGBT is connected.
- Every possible first-outcome branch must fit the hard resource cap.
- An agent must abstain on observationally equivalent worlds where the requested scientific mechanism is not identifiable.

Reject-all is not a successful integrity policy: include a separate positive-control set of valid, correctly bound evidence receipts.

## 5. Metrics

Report each family separately.

**Scientific decisions**
- expected terminal utility and Bayes decision regret;
- worst-scenario regret and robust value of information;
- mechanism recovery conditioned on identifiability;
- false mechanism-claim rate and calibrated abstention;
- held-out observable prediction loss.

**Instrument inference**
- device-state posterior Brier/log score;
- calibration error and interval coverage;
- drift-state detection delay and false alarms;
- stale-calibration detection;
- cross-device error correlation and common-cause detection;
- calibration transfer errors across device identities.

**Efficiency**
- number of science and calibration measurements;
- cost and time to a useful decision;
- maximum branch cost and expected cost by scenario;
- compute, wall time, energy (only with measured telemetry);
- full resource-vector Pareto frontier.

**Evidence integrity**
- stale, duplicate, misbound, wrong-device, unauthorized, and revoked receipt acceptance;
- invalid-transition false acceptance (must be zero in frozen negative controls);
- valid-receipt false rejection rate;
- accepted-transition replay agreement;
- rejected-transition atomicity;
- calibration-certificate/device-identity coverage.

Do not reward instrument-state estimation unless it improves a declared scientific or operational target.

## 6. Statistical protocol

- Start with 20 seeds per synthetic family, increasing the count when confidence intervals remain too wide.
- Split by entire scientific mechanism, instrument-drift law, device cohort, and shared-cause family. Do not leak near-duplicate trajectories across splits.
- Use paired task/seed comparisons and family-clustered bootstrap intervals.
- Freeze transition parameters, likelihood models, hyperparameters, utility schedules, and thresholds using validation only.
- Use a preregistered multiplicity correction across acquisition arms.
- Publish failed, aborted, and invalid-evidence episodes, along with resource consumption.
- Confirm synthetic gains on at least one external mechanism-discovery or laboratory workflow benchmark without changing that benchmark's native score.
- Require a second implementation or independent team to reproduce any SOTA claim.

## 7. Proposed competitive gates

Targets, not achieved results:

1. Exact agreement with the finite oracle on every known-answer fixture.
2. Zero calibration-leakage acceptances and zero branch-wise budget violations.
3. Zero invalid evidence transitions on frozen negative controls, with at least 99% acceptance on a separately generated valid-receipt control set.
4. At least 10% lower held-out decision regret than the strongest matched same-device-repeat/EIG-only baseline on preregistered D2/D4 tasks, paired 95% confidence interval excluding zero.
5. No material regression versus simpler policies on D0 static-known-channel tasks.
6. At least 10% lower resource cost at matched decision quality on one drift-sensitive multi-fidelity task.
7. No increase in false scientific-mechanism claims or loss of calibration under distribution shift.
8. Independent reproduction on held-out device cohorts and drift laws.

The 10% thresholds are internal project targets, not accepted industry standards.

## 8. External evaluation lanes

Keep external benchmark scoring intact:
- **Petri-bench:** scientific method integrity, controlled experiments, and evidence quality.
- **ActiveSciBench:** mechanism discovery under a finite experiment budget.
- **Autonomous Science Bench:** complete laboratory-origin campaigns and held-out learning.
- **ASI-Bench / AstaBench:** broad project-level scientific workflow autonomy.
- **Fresh generated hidden tasks:** device drift, correlated errors, calibration-transfer attacks, and benchmark anti-saturation.

Use a separate instrument-aware subtrack when upstream tasks do not expose device metadata or calibration control. Do not manufacture a device-drift score from a benchmark that does not measure it.

## 9. Future extension gates

Before calling this a deployable scientific measurement architecture, implement:
1. Continuous-time or variable-delay state transitions.
2. Correlated/common-cause transitions across instruments.
3. Calibration actions that may repair or reconfigure a device.
4. Independent source attestation and revocable calibration certificates.
5. EGBT typed receipts bound to instrument asset, configuration, calibration epoch, capture time, and observation payload.
6. Atomic joint updates to scientific and instrument beliefs.
7. HDC/CfC policy proposals evaluated against the exact finite oracle and held-out tasks.

The present v1 reference implements none of the production trust services or physical metrology layers above.
