# MCR² v17 — Known-truth intervention and endpoint-ascertainment benchmark

**Status:** synthetic research reference, version 0.1. No physical, simulator, HIL, ROS 2, survival-calibration, or production-control claim.

## Research objective

Test whether an evaluator can distinguish:
- the reference-policy event risk from the outcome under an assigned anticipatory policy;
- randomized intention-to-treat (ITT) policy value from a selected received-treatment comparison;
- endpoint observation loss from event-free outcomes;
- conditionally supported endpoint adjustment from a positivity failure;
- observed assigned-policy data from evaluator-only counterfactual truth.

This is a fixed-horizon endpoint-risk DGP. It is not a time-to-event survival estimator and does not implement HAPS, IPCW, AIPCW or a causal estimator.

## Composition

- [#11880 anticipatory reflexes](https://github.com/Luminous-Dynamics/symthaea/issues/11880)
- [#11938 intervention/censoring semantics](https://github.com/Luminous-Dynamics/symthaea/issues/11938)
- [#11954 known-truth DGP](https://github.com/Luminous-Dynamics/symthaea/issues/11954)
- [#9690 censor-aware forecast methodology](https://github.com/Luminous-Dynamics/symthaea/issues/9690)
- [#6226 experiment design](https://github.com/Luminous-Dynamics/symthaea/issues/6226)
- [#4800 / #4814 robotics provenance and JointLab](https://github.com/Luminous-Dynamics/symthaea/issues/4814)

## Design

Four endpoint-ascertainment regimes are generated: complete follow-up, MCAR loss, loss conditional on observed pre-assignment risk stratum, and a positivity failure with zero high-risk endpoint observation. Every matched block randomizes one episode to each of two independently qualified synthetic safe policies. Assignment and a receipt precede time zero.

The policy-facing observed record contains only the assigned policy's endpoint, when observed. Potential outcomes and both policy event times are held in a separate evaluator-only oracle artifact. Missing endpoints remain unknown. Competing terminal events remain distinct.

The oracle-known observation-probability calculation is a diagnostic ceiling only. It must not be confused with a fitted or operational estimator. Statistical censor-aware forecast methods remain owned by #9690.

## Run the local reference package

The accompanying MCR² v17 reference package contains the deterministic generator, strict observed/oracle JSON schemas, sample artifacts, report and 15 unit tests.

~~~bash
python3 run_benchmark.py
python3 -m unittest discover -v
~~~

The default configuration uses 10,000 matched blocks per scenario (20,000 assigned episodes), seed 20261113 and a fixed horizon of 20 synthetic time units.

## Claim boundary

This is a known-truth synthetic protocol diagnostic. It does not validate real-world censoring assumptions, survival calibration, causal identification from observational robot data, anticipatory reflex effectiveness, physical safety, timing, energy, or SOTA superiority. Do not deliberately withhold required protection on a physical robot to create an unsafe control arm.
