# Safety-Constrained Active Falsification v1

**Status:** deterministic synthetic research prototype; no external task integration, physical safety claim, or SOTA result.

## 1. Why this is the next step

A passive shield can only preserve actions that are certifiable under the uncertainty it already has. A separate OOD/residual monitor can trigger fallback when its available signals reveal a mismatch, but uncertainty may remain wide and persistently force the same conservative decisions.

The next research question is whether a controller can *actively distinguish candidate dynamics models* without taking an action that sacrifices the safety contract. This is closely related to safe exploration and dual control, not a new field invented by this repository. Relevant prior work includes:

- Safe exploration equilibrium between feasible regions and model uncertainty (Yang, Zheng & Li, 2026): https://pubmed.ncbi.nlm.nih.gov/41774638/
- Dual-gatekeeper: safe dual-control trajectory planning under safety and mission-cost budgets (Ben Naveed et al., 2026): https://arxiv.org/abs/2604.15507
- Learning Safe Control via On-the-Fly Bandit Exploration (Capone et al., ICML 2025): https://proceedings.mlr.press/v267/capone25a.html
- Conformalized System Level Synthesis MPC (Srinivasan, Leeman & Chou, L4DC 2026): https://proceedings.mlr.press/v331/srinivasan26a.html

This prototype is a much smaller, discrete-action experiment than those methods. It does not reproduce their optimization, proofs, controllers, tasks, or results.

## 2. Safety-constrained action selection

Given a finite family of candidate dynamics hypotheses, the implementation builds a rectangular hull over each action's reserve/load effect intervals. It evaluates multi-step robust viability against that common hull. Only actions with a valid horizon certificate enter the active-probe candidate set.

This common-hull step matters. It is not sufficient to check each model separately and union their viable actions: a different model may require a different continuation after the proposed first action, while the controller has not yet identified which model is actual. The hull enforces a common conservative transition envelope; the multi-step viability search then requires one safe continuation sequence under that envelope. Rectangularization can introduce unreachable combinations and therefore false vetoes, but it should not create extra optimism under the declared interval abstraction.

For each candidate action, calculate pairwise distances between the model-prediction boxes for reserve/load changes. If two boxes overlap on both coordinates, their distance is zero. The noise-aware selector expands each predicted interval by the declared per-coordinate observation error radius; the separation then shrinks by twice that radius on each coordinate, floored at zero. An action scores zero if any model pair's noisy observation boxes overlap. For each action, use the minimum separation over every pair of model hypotheses:

\[
D(a)=\min_{i<j}\operatorname{dist}(B_i(a),B_j(a)).
\]

The planner chooses the safe action with the largest (D(a)), subject to a predeclared minimum-discrimination threshold. A task-preferred action breaks ties. If no certified action has sufficient discriminative value, it recommends **no probe**; it does not claim that no ordinary safe task action exists. The caller must continue to use its existing certified task policy/fallback logic.

This score is a conservative geometric proxy for worst-pair distinguishability, not expected information gain, mutual information, Bayesian entropy reduction, or proof that one observation will identify the true model. The original zero-noise API remains as a wrapper for compatibility; the new `recommend_safe_probe_with_noise` path discounts separation for a caller-supplied measurement bound and fails closed for invalid bounds. This only handles a bounded, per-coordinate observation-error model; stochastic transitions and non-identifiable parameters require a richer observation model before making stronger claims.

## 3. Module contract

- `src/safe_probe.rs`: validated multi-hypothesis model hull, maximin pairwise effect separation, horizon-certified safe candidate filter, deterministic action selection and unit tests.
- `src/bin/allostatic-safe-probe-bench.rs`: deterministic contract fixture with informative-safe, informative-unsafe, indistinguishable-model, one-step-dead-end, authority-restricted and invalid-horizon cases.
- The output includes one raw row per declared case and a compact summary. The panel is intentionally a contract suite, not a statistical test set.
- Any invalid request or failed bounded search yields no recommendation. An emitted probe action must be contained in the reported horizon-viable set.

## 4. Predeclared falsification cases

1. **Informative and safe:** the most discriminating action should be chosen when its full-horizon transition tube remains inside the limits.
2. **Informative but unsafe:** the more discriminating action must be rejected when any effect in the model hull violates limits.
3. **Indistinguishable hypotheses:** no action should be called an informative probe even if safe actions exist.
4. **One-step safe dead end:** an action that is safe now but lacks a safe continuation must not be selected as a probe.
5. **Authority restricted:** an emergency action cannot become a candidate when the caller lacks authority to invoke it.
6. **Invalid horizon:** malformed requests fail closed rather than returning a partial certificate.

The unit tests also check that the model hull contains every declared candidate effect interval. They do not prove that the actual system is inside that hull.

## 5. Closed-loop hypothesis update experiment

The branch now includes a small follow-on benchmark that connects a safe probe to its measured outcome and then recomputes the viability set:

1. Build the common model hull and count the initially certified horizon-3 actions.
2. Select the probe using the common-hull multi-step safety certificate.
3. Feed the observed transition effects and a predeclared measurement radius to an interval-consistency update. A hypothesis is retained only if both its predicted reserve-effect and load-effect intervals intersect the observation box.
4. Recompute the viable-action set from the retained model hull. If no hypothesis is compatible, mark the candidate family falsified and return no posterior hull; do not quietly restore the prior or issue a certificate.

The deterministic panel covers four evaluator-only cases: an actual plant exactly matching each of the two candidate hypotheses, an intermediate unmodeled plant whose noisy observation is compatible with both, and an observation outside all candidate envelopes. The intended result is to make all of these differ visibly in the output. In this fixture the safe Maintain probe distinguishes model 0, after which the Push action becomes horizon-3 certifiable; under model 1, Push remains excluded. Ambiguous evidence leaves both hypotheses, while out-of-family evidence empties the candidate set.

These are hand-constructed contract cases, not a statistical claim. The interval update is a consistency filter, not a posterior probability calculation. An observation box that overlaps a candidate does not prove the candidate is the real plant, and a finite family can omit the real system. Actual safety is scored separately from the model-membership result.

## 6. Next benchmark beyond this fixture

Use a pinned physics environment and task-specific models. For each probe, the external evaluator should record:

- model hypotheses before and after the probe;
- which actual transition was independently observed;
- whether the prediction intervals were separated under predeclared observation noise;
- whether the action remained safe under the hidden actual plant;
- change in calibrated uncertainty and subsequent viable-action availability;
- reward/cost incurred for information gathering;
- detection and certification latency.

Compare at least: task-greedy safe action, random safe probe, maximum-disagreement safe probe, a principled information-gain policy when its likelihood model is available, fixed global-margin shielding, and a strong native MPC/CBF/MPSC baseline. Compare at matched violation budget and explicit exploration-cost budget.

A meaningful success criterion is not "probe selected." It is a better safety–capability frontier after a fixed interaction budget: lower subsequent false-veto rate or better task reward, while preserving the same predeclared violation bound. Report no-probe rate and cases where the model family stays unidentifiable. Never charge the experiment for expected information as if it were realized information; measure actual uncertainty reduction on held-out data.

## 7. Limitations and claim boundaries

- The reserve/load state and three discrete actions are synthetic engineering abstractions.
- Interval hulls are conservative but discard correlations among model parameters and effects.
- Maximum pairwise interval separation is not mutual information and can prefer an action whose observed outcome is still ambiguous after measurement noise.
- The candidate hypothesis family may omit the real plant; the certificate is conditional on the hull covering it.
- Open-loop continuation is conservative compared with state-contingent feedback.
- No live Symthaea control loop is changed.
- No external benchmark has been run and no SOTA claim is made.

The research hypothesis is that a future estimator can turn safe observations into tighter, empirically validated uncertainty sets, increasing certified action availability without increasing false certificates. This PR implements only the first testable selection contract needed to pursue that hypothesis.
