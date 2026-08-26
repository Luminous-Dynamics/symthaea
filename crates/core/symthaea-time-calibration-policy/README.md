# Symthaea Time Calibration Policy

`symthaea-time-calibration-policy` separates **calibration evidence** from **permission to use that evidence**.

A narrow offset interval is not automatically synchronization authority. This crate applies an explicitly supplied, versioned policy to a validated `CalibrationConsensus` and records one of three outcomes:

- `Accepted`
- `Rejected`
- `Inconclusive`

## Asymmetric decisions

The policy has two separate thresholds:

- `acceptance_max_radius_us`
- optional `practical_failure_min_radius_us`

The practical-failure threshold must be strictly larger than the acceptance threshold.

Decision semantics are:

- radius <= acceptance threshold -> `Accepted`
- radius >= practical-failure threshold -> `Rejected`
- otherwise -> `Inconclusive`

If no practical-failure threshold is configured, failure to accept can never become `Rejected`; it remains `Inconclusive`.

This prevents an underpowered or noisy calibration from being mislabeled as evidence that synchronization is impossible.

## No built-in timing threshold

The crate intentionally defines **no universal acceptable clock error**.

A camera/IMU fusion loop, distributed audio capture, robot safety interlock, chemical sensor array, and long-timescale scientific logger can require very different timing accuracy.

Thresholds belong in a use-specific, versioned policy or preregistration—not in this core crate.

## Accepted offset estimate

An accepted decision can expose:

- source/target domain + epoch;
- deterministic midpoint of the accepted offset interval;
- conservative symmetric error radius covering the interval.

The midpoint is a representation of the accepted interval, not a claim that the true offset equals the midpoint.

`AcceptedOffsetEstimate` is a convenience projection, not independent evidence or synchronization authority. Auditable workflows should retain the decision receipt and the underlying calibration evidence.

## Still not a clock transform

Even `Accepted` does **not** prove:

- mapping continuity outside the sampled exchange(s);
- target-clock continuity;
- a future validity/holdover interval;
- oscillator drift bounds;
- timestamp authenticity;
- synchronization protocol correctness.

Therefore this crate does not mint `ClockTransformReceipt`s.

The intended chain is:

1. `symthaea-time-integrity` — timestamp identity/continuity/error claims;
2. `symthaea-time-calibration` — derive admissible offset intervals;
3. this crate — decide whether the interval meets a use-specific gate;
4. later continuity/holdover evidence — justify a transform validity interval;
5. `symthaea-time-normalization` — represent and apply the resulting transform.

## Evidence count is not confidence

This policy does not use calibration exchange count as epistemic weight. Ten duplicated or correlated exchanges do not automatically constitute stronger evidence than one exchange.

A future evidence-bundle layer may bind the exact exchange set and any independence/provenance claims, but this policy gates only the validated interval produced by that set.
