# VIS-DVPC-031 — Region-aware fail-closed decision gate v1

**Status:** standalone gate reference implementation. It is not a statistical inference engine and does not manufacture bounds.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8759

## Contract

Inputs are a value interval [value_lower, value_upper], a risk interval [risk_lower, risk_upper], a minimum acceptable value, and a maximum acceptable risk. All quantities are declared on [0,1]. Either interval may be missing (both endpoints as \`NA\`), but one endpoint alone is invalid.

- \`AUTHORIZED\`: the entire value interval meets or exceeds the value floor, and the entire risk interval meets or stays below the risk cap.
- \`DENIED\`: the entire value interval is below the floor, or the entire risk interval is above the cap.
- \`INSUFFICIENT_EVIDENCE\`: neither authorization nor rejection is established, including missing bounds and intervals crossing a decision threshold.

Boundaries are intentional: value-floor equality is acceptable; risk-cap equality is acceptable. If the value interval's upper endpoint equals the value floor but its lower endpoint is below it, the result remains \`INSUFFICIENT_EVIDENCE\`, because not every compatible value clears the floor.

## Separation of evidence types

An identification region describes values compatible with the data distribution and stated assumptions. A confidence interval describes sampling uncertainty under an inference procedure. A risk bound may come from a separate harm-rate analysis. A target-transport claim requires its own assumptions/evidence. These are not interchangeable.

This tool only classifies supplied regions. It does not verify that a value region, confidence interval, or risk bound was computed correctly, and it does not authenticate their provenance. The caller must bind evidence type, assumptions, inputs, and procedure to a trusted receipt. It only authorizes with both complete regions present and both constraints cleared. A definitely rejecting condition can justify denial even if the other interval is missing.

## VIS-DVPC-028 composition control

The synthetic zero-support case has a structural value region [0.1, 0.6]. At a minimum-value threshold of 0.5, with a risk interval [0.01, 0.03] and cap 0.05, the result is \`INSUFFICIENT_EVIDENCE\`. The gate must not convert the oracle's synthetic point value into an authorization.

## Run

\`\`\`sh
rustc --edition=2024 --test tools/vision-dvpc-region-gate.rs -o /tmp/vis-dvpc-031-tests
/tmp/vis-dvpc-031-tests
rustc --edition=2024 -O tools/vision-dvpc-region-gate.rs -o /tmp/vis-dvpc-031
/tmp/vis-dvpc-031 --evaluate 0.1 0.6 0.01 0.03 0.5 0.05
/tmp/vis-dvpc-031 --evaluate NA NA 0.06 0.09 0.5 0.05
\`\`\`

This v1 gate is a synthetic contract test and not a production safety certification. It should compose with DVPC-022 only after a dedicated adapter binds the incoming value/risk intervals to their evidence provenance and the required statistical assumptions.
