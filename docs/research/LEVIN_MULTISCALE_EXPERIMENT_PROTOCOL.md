# Levin-inspired multiscale experiment protocol

## Purpose and scope

This protocol expands the existing Cell Foundry equifinality and dose-response experiments into one deterministic, multi-factor sweep. It is deliberately labeled a **synthetic simulation experiment**: it does not establish that the model reproduces living tissue, and it does not establish cognition or consciousness.

## Factors swept

- **Pattern family:** radial bipolar target versus a linear anterior/posterior voltage target.
- **Perturbation:** two levels of tissue removal per family plus total Vmem scramble as a hard recovery boundary.
- **Electrical communication:** gap-junction permeability at 0, 0.25, 0.5, 0.75, and 1.0.
- **Additional memory channel:** positional homing disabled versus enabled.
- **Replication:** four independent deterministic seeds (17, 29, 41, 53).
- **Outcomes:** final morphology discrepancy, mean discrepancy over the trajectory, first observed day below a fixed 0.05 tolerance, and wall-clock time per condition.

The fixed sweep covers 240 conditions (four seeds × two pattern families × three perturbations × two homing settings × five fixed permeability values). The adaptive extension adds 96 controller conditions (48 seed/family/perturbation/homing contexts × two policies: trend-only and counterfactual). The fixed trajectories are written to **levin_multiscale_results.json** and adaptive traces to **levin_adaptive_coupling_results.json**. The runner prints per-condition rows and grouped means/population standard deviations. Use the same source revision, build profile, machine, and configuration when comparing runtimes.

## Run

    cargo run -p symthaea-cell-foundry --example levin_multiscale_experiments

The script is intentionally a research runner, not part of the default test suite; it executes many independent simulated recovery trajectories.

## Hypotheses and falsification

1. **Electrical-coupling hypothesis:** for perturbations that leave a surviving pattern template, increasing coupling should improve some recovery outcomes relative to zero coupling. Failure across replicated seeds is evidence against the current parameterization, not something to hide by retuning a threshold after seeing results.
2. **Independent-memory hypothesis:** positional homing may improve recovery after complete pattern scramble, including with gap junctions blocked. This is an independent mechanism and must not be credited to electrical coupling.
3. **Equifinality hypothesis:** different perturbation routes should converge toward the same target, measured by trajectory and final discrepancy rather than a single hand-picked case.
4. **Generalization hypothesis:** effects should persist across both radial and axis-defined patterns and held-out seeds. These two synthetic families are an initial test, not broad proof of out-of-distribution generalization.
5. **Cost hypothesis:** report runtime alongside recovery quality. A more complex condition is not better if it costs more and fails to improve the predefined outcome.

## Adaptive coupling extension

The current runner compares two predeclared controllers. The trend-only heuristic starts at permeability 0.5, examines three-day windows, raises coupling by 0.1 if mean discrepancy improves beyond 0.002, lowers it by 0.2 if it deteriorates beyond that tolerance, and otherwise holds. This controller is intentionally a simple baseline: natural improvement over time can confound the feedback signal.

The counterfactual controller starts from the same complete simulated state, clones it, evaluates lower/current/higher permeability branches over an identical three-day horizon with matched RNG state, and chooses the candidate with lowest forecast discrepancy if it beats the current action by the tolerance. It then applies that setting until the next decision. This reduces the time-trend confound, but its conclusions are still limited by model accuracy and short-horizon prediction.

Both policies clamp permeability to [0, 1], record each day's applied permeability and discrepancy, and are paired per seed with the mean of fixed-permeability controls at 0.0, 0.5, and 1.0. A negative final-error delta means a controller outperformed that fixed-control mean in that condition. Neither controller is presumed superior; preserve results even when fixed controls win.

## Required next controls

- Compare the full implementation against zero-coupling/no-homing and coupling-only/homing-only conditions.
- Keep all seeds, failed recoveries, timeouts, and negative results.
- Follow with a held-out set of irregular masks / fragmented networks and a simpler graph-diffusion baseline.
- For biological claims, compare calibrated model outputs to suitable published datasets and experimental uncertainty. Synthetic success alone is only verification of the code's internal behavior.
