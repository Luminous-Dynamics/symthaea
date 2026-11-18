# Atelier Lightshell — Pre-registered Trial Design Contract v0.1

## Purpose and boundary

This contract closes an experimental-design gap between paired installed-system measurements (#12831) and the admission of those measurements as comparative evidence. Same-serial matching is necessary but not sufficient: if every trial always runs uncovered first and Lightshell last, a time-dependent drift can become confounded with the garment condition.

This module owns protocol freeze, randomized counterbalanced sequence allocation, within-stratum balance, per-period reset/carryover receipts and block-level sentinel drift checks. It links each period to an exact installed-system run. It does not execute a trial, calculate the performance gate, replace the #12831 validator, control a robot, qualify safety, certify an article or authorize operation.

## Experimental design

The preferred design is a within-robot blocked comparison. A stratum fixes robot serial, firmware, robot configuration, tool/payload, mission profile and environment. Do not pool across changed strata as though they were the same physical system.

When a genuinely comparable incumbent cover exists, include three conditions:
- U = baseline_uncovered
- I = incumbent_cover
- L = lightshell_installed

Use all six first-order-balanced sequences with equal allocation inside each stratum:

| Sequence | Period 0 | Period 1 | Period 2 |
|---|---|---|---|
| U-I-L | Uncovered | Incumbent | Lightshell |
| U-L-I | Uncovered | Lightshell | Incumbent |
| I-U-L | Incumbent | Uncovered | Lightshell |
| I-L-U | Incumbent | Lightshell | Uncovered |
| L-U-I | Lightshell | Uncovered | Incumbent |
| L-I-U | Lightshell | Incumbent | Uncovered |

All six permutations balance each condition's period position and each directed transition. This is a counterbalancing scheme, not a guarantee that every possible higher-order carryover effect is eliminated; explicit transition/reset checks remain required.

When there is no genuinely comparable incumbent, use the two-arm U-L / L-U design and omit the incumbent condition. The reason and comparator applicability should remain recorded in the target #12831 campaign. Do not create a fake or unsuitable commercial comparator merely to fill a third arm.

A randomization custodian generates and allocates the sequence independently of trial outcomes. Before the first run, freeze the protocol digest and commit to the allocation seed. Preserve the allocation receipt and reveal/verification record after execution. No outcome-adaptive sequence reassignment is admitted. The validator checks structure, balance, references and state transitions; it does not implement a cryptographic random-number generator.

## Period reset and carryover

Every period requires a reset/stabilization decision. At minimum, the responsible owner should decide whether the protocol needs:
- return to a defined robot thermal state;
- battery state-of-charge and supply-voltage bounds;
- payload/tool identity and configuration restoration;
- cable/hose routing and service-interface inspection;
- textile relaxation / fastening re-tension rules;
- sensor/instrument zeroing and calibration validity;
- fixture, environment and mission-profile restoration.

These are examples of possible reset variables, not universal limits. The owner must freeze exact criteria and evidence references. Every transition after period zero receives a carryover check; failed or unknown reset/carryover blocks admission of that block's comparative evidence.

## Sentinel drift

A block starts and ends with the same pre-declared sentinel measurement. The sentinel should be independent of the garment effect being studied and sensitive to the most plausible time-dependent confounder for that test campaign.

P0 uses conservative drift guard-banding:

`D_guard = |y_end - y_start| + U_expanded`

A block is eligible only when `D_guard <= D_max`, with `D_max` frozen before execution and derived from the relevant requirement, instrument capability and test method. No universal drift threshold is proposed here. This conservative rule intentionally avoids assuming that measurement errors cancel or are independent.

The start and end values must each resolve to their own evidence records. A missing sentinel is unknown, not zero drift.

## Evidence linkage

Each executed period points to one unique installed-system run from #12831. The downstream/orchestration layer must dereference that exact run and independently verify the robot/garment condition, profile, environment, calibration, method, fixture, raw-data digest and evidence origin. This module does not trust a bare string reference as proof that the referenced run is valid.

The proposed fixture contains six planned blocks and eighteen planned periods, but zero executed periods. All names and digests in the fixture are synthetic placeholders.

## P0 invariants

- Protocol freeze and SHA-256 digest exist before first execution.
- Conditions are exactly U/L, or U/I/L with an exact incumbent comparator reference.
- Every stratum has complete and equal sequence allocation; no sequence may be absent.
- Period order exactly matches the allocated sequence.
- Every block and allocation receipt is unique; allocation-order indices are contiguous and unique per stratum.
- Every observed period has one unique #12831 run reference.
- Every observed period has a passed reset check; each period after the first has a passed carryover check.
- Every observed block has start/end sentinel evidence and conservative drift within the frozen limit.
- Allocation seed commitment is present; an observed campaign also has a reveal/verification receipt.
- Adjudication requires an independent reviewer and an explicit admit disposition.
- Proposed campaigns cannot smuggle in run outcomes or measurement receipts.
- Unknown, failed, stale or missing evidence never becomes a pass.

## Benchmarks and research targets

| Dimension | P0 target | Stretch target |
|---|---|---|
| Allocation integrity | 100% sequence balance per stratum | independently reproduced allocation receipt |
| Run linkage | 100% periods resolve to unique #12831 run IDs | replayable, machine-verifiable evidence bundle |
| Reset/carryover | 100% observed periods carry passing evidence | quantify and reduce carryover variance by condition |
| Drift | 100% blocks pass uncertainty-guarded threshold | instrument- and robot-specific drift model validated on hold-outs |
| Reproducibility | no outcome-adaptive reassignment | repeat across serials, production lots, environments and independent labs |
| False admission | zero false admits in injected adversarial controls | external red-team review and public conformance suite |

Do not claim superiority to RGBench by comparing unrelated metrics. Reproduce RGBench on its own garment simulation/manipulation task first; report this installed-system protocol as a complementary axis. The differentiator to pursue is reproducible, counterbalanced, uncertainty-bounded robot/garment qualification, not another cloth simulator.

## Known limitations

- Sequence balance does not by itself estimate all carryover effects; it reduces order/period confounding and must be paired with reset and carryover checks.
- The P0 validator does not perform statistical inference, sample-size planning, mixed-effects modeling or cross-lab meta-analysis.
- One robot serial does not establish population-level generality.
- The exact meaning and limit of each sentinel, reset and washout criterion is owner- and method-specific.
- This record cannot replace the robot integrator's safety assessment or the requirements of the applicable ISO 10218 integration/safety process.

## References

- RGBench, AAAI 2026: https://rgbench.github.io/
- ICRA 2024 Cloth Competition dataset/benchmark paper, first published 2026: https://doi.org/10.1177/02783649251414885
- 2026 systematic review of deep-learning robotic cloth manipulation: https://doi.org/10.3389/frobt.2026.1752914
- ISO 10218-1:2025 and ISO 10218-2:2025: https://www.iso.org/standard/73933.html
