# Application-event energy uncertainty and evidence-bound edge Pareto v0.1

Tracking: #9755 / #9775  
Implementation branch: `research/symthaea-tinyml-benchmark-v0.1`

## Why a separate uncertainty receipt is required

A PPK2 inference-energy uncertainty receipt cannot be reused as an application-event uncertainty receipt. The measurand is different: the event protocol can include sensor wake/acquisition, preprocessing, inference, postprocessing, communication, storage, sleep transitions and other declared phases.

This module derives uncertainty from repeated, matched application events selected from a validated phase-energy receipt.

## Frozen repeatability protocol

The repeatability protocol binds:
- benchmark and repeatability experiment identity;
- target and board revision;
- exact phase-protocol digest;
- independent repeat-design digest;
- candidate artifact digest;
- physical device/session receipt digest;
- matched input digest;
- voltage source identity and calibrated voltage-readback evidence digest;
- minimum number of repeats.

The metrology manifest must bind the exact repeat-design digest. The phase protocol is independently pinned. The event outcome row itself binds artifact, device session, input and output/result digest.

All events in the phase receipt must refer to the same candidate artifact and device session. Only events with the exact matched input digest are eligible for this repeatability estimate.

## Event selection

An event contributes to Type A repeatability only when all conditions hold:

1. the phase receipt is valid and its phase protocol digest matches the supplied phase protocol;
2. the event artifact digest and device-session digest match the repeatability protocol;
3. its input digest matches the frozen repeated input;
4. its phase status is `COMPLETE_PHASE_COVERAGE`;
5. its outcome belongs to the protocol's accepted-success policy;
6. its deadline was met;
7. total event energy and boundary-alignment bound are present, finite and valid.

Rejected attempts are retained as separate counts: rejected outcome, missed deadline, incomplete/unattributed phase coverage and invalid/missing energy. They are not silently promoted into successful-event repeats.

## Uncertainty calculation

For (n) complete accepted deadline-met event repeats with energy (E_i):

`mean = Σ(E_i)/n`

`s = sample standard deviation(E_i)`

`u_A = s / sqrt(n)`

The calibration manifest supplies a declared Type B relative standard uncertainty:

`u_B = mean × u_B_relative`

With its explicit uncorrelated Type A/Type B declaration:

`u_c = sqrt(u_A² + u_B²)`

`U_stat = k × u_c`

The clock-boundary term is a deterministic worst-case energy bound, not a random standard uncertainty. The receipt takes the maximum boundary-alignment energy bound over the included repeats and adds it linearly:

`U_total = U_stat + max(boundary_alignment_bound_i)`

The reported interval is `[max(0, mean − U_total), mean + U_total]`.

This intentionally avoids pretending that clock-edge uncertainty is Gaussian or independent. A later uncertainty budget may model covariance if the clock/calibration methodology supports it.

## Fair candidate/reference comparison

A strict energy winner is reported only when its total uncertainty interval is disjoint from the other's interval.

The comparison requires matching:
- benchmark and repeatability experiment identity;
- phase protocol and independent repeat-design digest;
- exact repeated input;
- target and board revision;
- PPK2 instrument serial and current range;
- calibration certificate digest;
- voltage source identity and voltage-readback evidence digest;
- voltage value.

Candidate artifacts and device-session digests may differ because those are the objects being compared. They are individually bound into their own event receipts and hardware measurement receipts.

The comparison does not assume systematic calibration error cancels. It uses the full expanded intervals and therefore can be conservative when candidates share systematic error.

## Evidence replay and Pareto scoring

The new edge candidate evidence evaluator requires a bundle containing:
- causal prequential execution trace and independently replayed audit receipt;
- phase-energy receipt plus phase protocol;
- event-repeatability protocol and metrology manifest;
- application-event energy uncertainty receipt;
- hardware measurement receipt with compiler, firmware, linker-map, measurement-protocol and device-session digests;
- target-board latency samples, peak SRAM and flash size.

Before a candidate can enter the frontier, the evaluator:
1. replays the candidate's predict → label release → update chronology;
2. recomputes the event-energy uncertainty receipt from the phase receipt and metrology inputs;
3. verifies all trace/artifact/device/dataset/session commitments;
4. derives p95 latency using nearest-rank from the recorded latency samples;
5. checks hard quality, false-alarm, missed-event, minimum-coverage, p95 deadline, protected SRAM and flash constraints.

The frontier uses quality, false alarms, missed events, latency, energy upper bound, peak SRAM, flash, coverage and Brier score. It does not collapse these unlike objectives into one weighted scalar. Infeasible candidates remain in the evaluation with explicit rejection reasons but cannot appear on the Pareto front.

This v0.1 evaluator covers temporal-adaptation and active-sensing application-event tracks. Static inference remains a separate MLPerf Tiny-compatible sub-benchmark; CPU-only and accelerator/NPU tiers must remain separate.

## Claim ceiling

- A digest is not proof of physical acquisition authenticity; the device-session, voltage-readback, calibration, linker-map and hardware-measurement evidence must be independently resolved by the acquisition/provenance system.
- The event-energy estimate applies to the exact matched input and declared protocol, not every possible workload input.
- A correct application outcome remains an evaluator-produced assertion; evaluator semantics are separately reviewed.
- The Type B budget must represent the full energy measurement chain, including current and voltage sources, with documented assumptions.
- Boundary-alignment energy is a conservative deterministic bound added linearly.
- No candidate win, SOTA improvement, or physical MCU measurement is claimed until the evidence bundles are acquired and replayed on target hardware.

## Authored controls

Seven synthetic tests cover matched successful repeats, non-overlapping versus overlapping event-energy intervals, minimum repeat count, artifact/session mismatch, tampered uncertainty bounds and voltage-readback mismatch. Three additional hardware-evidence tests cover p95 latency ranking, invalid latency samples and measurement-receipt digest tampering. Together with the prequential and stream tests, the benchmark branch now has a broad synthetic contract suite; no tests have been compiled or executed in the current authoring environment.
