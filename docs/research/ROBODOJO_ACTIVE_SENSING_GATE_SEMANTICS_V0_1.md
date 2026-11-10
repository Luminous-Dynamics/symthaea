# Active-sensing gate semantics for RoboDojo-RC

**Status:** implementation clarification and synthetic contract tests. No robot-performance, simulator, HIL or safety claim.

## Decision principle

Do not conflate a mandatory observation needed to satisfy a continuation gate with an optional observation whose expected information value exceeds its cost.

Let current estimates be p_s (task-success probability), p_u (unsafe-event probability), and let p_s_plus and p_u_plus be the predicted post-observation values after acquisition and replanning. Let p_min be the minimum continuation probability and p_max the maximum accepted unsafe-event probability.

The current state is admissible under this primitive exactly when:

`p_s >= p_min AND p_u <= p_max`

An observation is mandatory if either current gate is violated:

`required = (p_s < p_min) OR (p_u > p_max)`

The proposed post-observation state must satisfy both continuation gates:

`p_s_plus >= p_min AND p_u_plus <= p_max`

## Decision table

| Current state | Candidate posterior | Budget | Decision |
|---|---|---|---|
| Both gates pass | Both gates pass, net benefit meets threshold | Fits | Acquire (optional) |
| Both gates pass | Risk or confidence gate fails | Any | Decline optional observation; keep current admissible state |
| Both gates pass | Both gates pass, net benefit below threshold | Any | Proceed without sensing |
| Either gate fails | Both posterior gates pass | Fits | Acquire (mandatory); optional ROI threshold does not veto |
| Either gate fails | Either posterior gate fails | Any | Stop and escalate |
| Either gate fails | Forecast missing, sensor unavailable or over budget | Any | Stop and escalate |

The API returns separate risk-gate and confidence-gate requirement flags so the caller can explain why acquisition occurred.

## Why the distinction matters

A forecast about an observation that is not acquired must not veto an otherwise admissible current state. Conversely, a mandatory observation cannot be treated as optional merely because its cost exceeds the optional ROI threshold: when the current state violates a declared gate, either obtain evidence that restores both gates or stop.

This does not mean a robot is safe to move after the recommendation. It means only that this decision primitive chose an evidence-acquisition branch. The caller must obtain the observation, validate its provenance/currentness, recompute its state and pass every independent controller/physical safety gate.

## Synthetic regression tests

The five additional tests exercise:
- optional posterior risk degradation -> decline optional sensing rather than stop;
- optional posterior confidence degradation -> decline optional sensing rather than stop;
- mandatory confidence restoration -> acquire even when optional ROI threshold is higher than net benefit;
- mandatory risk restoration that degrades confidence -> stop;
- mandatory confidence restoration that leaves unsafe risk above limit -> stop.

Combined with the original 11 tests, the module now has 16 tests. They verify only the declared pure decision function under synthetic probabilities and costs.
