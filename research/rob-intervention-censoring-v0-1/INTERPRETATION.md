# v16 synthetic interpretation

- 15 classification fixtures exercise observed untreated events, ordinary administrative censoring, intervention-induced counterfactual unobservability, post-intervention events, competing events, telemetry loss, invalid clocks/instrumentation, unknown actuation delivery, protocol abort, same-tick ambiguity, and unsupported censor-adjustment admission.
- 8 randomized synthetic safe-policy episodes are split 4/4 across reactive_safe and predictive_safe.
- reactive_safe: 2/4 adverse fixed-horizon outcomes (risk 0.50).
- predictive_safe: 1/4 adverse fixed-horizon outcomes (risk 0.25).
- Candidate-minus-control adverse-risk difference: -0.25.

This is a deterministic, authored known-answer example, not an estimated treatment effect. No uncertainty interval is reported, and 4 episodes per arm cannot support a meaningful general performance claim. The point is to prove that the evaluator reports a policy-assignment estimand while refusing to claim that an intervened episode reveals its own untreated event time.

The untreated T0 cohort and randomized policy-outcome cohort are separate concepts. If every episode is protected by intervention and no valid reference stratum or identification design exists, the untreated event-time distribution is not identified from that cohort alone.

No survival calibration, IPCW/AIPCW/HAPS estimator, simulator, HIL, sensor, actuator or physical robot was run. Statistical time-to-event methods remain with #9690; policy assignment and physical run evidence must follow #6226 and #4800.
