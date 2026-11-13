# MCR² v17 — Known-truth intervention and endpoint-ascertainment benchmark

**Status:** synthetic research reference, version 0.1. No physical, simulator, HIL, ROS 2, survival-calibration, or production-control claim.

## Objective

Test whether an evaluator distinguishes reference-policy boundary risk from assigned-policy outcomes; randomized intention-to-treat (ITT) value from selected received-treatment comparisons; endpoint observation loss from event-free outcomes; supported adjustment from positivity failure; and observed assigned-policy data from evaluator-only counterfactual truth.

This is a fixed-horizon endpoint-risk DGP. It is not a time-to-event survival estimator and does not implement HAPS, IPCW, AIPCW or a causal estimator.

## Composition

- [#11880 anticipatory reflexes](https://github.com/Luminous-Dynamics/symthaea/issues/11880)
- [#11938 intervention/censoring semantics](https://github.com/Luminous-Dynamics/symthaea/issues/11938)
- [#11954 known-truth DGP](https://github.com/Luminous-Dynamics/symthaea/issues/11954)
- [#9690 censor-aware forecast methodology](https://github.com/Luminous-Dynamics/symthaea/issues/9690)
- [#6226 experiment design](https://github.com/Luminous-Dynamics/symthaea/issues/6226)
- [#4814 JointLab](https://github.com/Luminous-Dynamics/symthaea/issues/4814)

## Design

Four endpoint-ascertainment regimes: complete follow-up, MCAR loss, loss conditional on observed pre-assignment risk stratum, and positivity failure with zero high-risk endpoint observation. Each matched block randomizes one episode to each of two independently qualified synthetic safe policies. Assignment and its receipt precede time zero.

Policy-facing observations contain only the assigned policy's endpoint, when observed. Potential outcomes and both policy event times are kept in an evaluator-only oracle artifact. Missing endpoints remain unknown; competing terminal events remain distinct.

The oracle-known observation-probability calculation is a diagnostic ceiling only, not a fitted or operational estimator. Statistical censor-aware forecast methods remain owned by #9690.

## Local package

The MCR² v17 reference package contains the deterministic generator, strict observed/oracle JSON schemas, sample artifacts, report and 15 unit tests.

~~~bash
python3 run_benchmark.py
python3 -m unittest discover -v
~~~

Default configuration: 10,000 matched blocks per scenario (20,000 assigned episodes), seed 20261113, fixed horizon 20 synthetic time units.

## Claim boundary

Synthetic protocol diagnostics only. No real-world censoring assumption, survival calibration, observational causal identification, anticipatory reflex effectiveness, physical safety, timing, energy or SOTA superiority is established. Do not withhold required physical protection to create an unsafe control arm.
