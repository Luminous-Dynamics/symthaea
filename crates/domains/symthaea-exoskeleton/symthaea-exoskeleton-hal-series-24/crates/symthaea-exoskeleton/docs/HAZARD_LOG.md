# Exoskeleton Software Hazard Log

This is a living engineering hazard log, not a completed regulatory risk file.
Severity and verification ownership must be reviewed by qualified mechanical,
electrical, controls, clinical, and safety engineers before human use.

| ID | Hazard | Representative cause | Software control | Required external control | Evidence gate |
|---|---|---|---|---|---|
| H-01 | Unexpected joint torque | Corrupt learned output, sign error | Independent magnitude, slew, travel, velocity, and power envelope | Current-limited drive, mechanical stops | S2, K2 |
| H-02 | Residual resistance after authority loss | Slew-limited withdrawal or stale impedance | Immediate zero torque, stiffness, and damping on zero authority | Normally-safe power stage and emergency release | S0, S1, T1 |
| H-03 | Assistance chatter | Noisy estimate near a threshold | Hysteresis, confidence, freshness, and upgrade dwell | Independent sensor-quality monitor | A0 |
| H-04 | Wrong wearer/profile | Stale or copied calibration | Identified versioned profile and validation | Hardware serial binding and operator ceremony | K0–K5 |
| H-05 | Control-loop stall | CPU overload, deadlock, scheduler delay | Deadline budget and consecutive-miss fault | Independent hardware watchdog | RT0–RT4 |
| H-06 | Stale or replayed sensor frame | Bus delay, duplicate packet | Monotonic timestamp, age, and exact sequence checks | Timestamped sensor MCU | RT0–RT2 |
| H-07 | Sensor disagreement hidden by fusion | Failed encoder or IMU | Typed disagreement fault; no silent averaging | Redundant sensing and plausibility MCU | D0 |
| H-08 | Thermal or current injury | Motor stall, short, excessive load | Critical latched fault policy | Hardware over-current and thermal cutoff | D0 |
| H-09 | Transport sends malformed command | Channel mismatch, NaN, range error | Fail-closed command sink validation | Command acknowledgement and CRC | H0, T1 |
| H-10 | Unreproducible incident | Logging controller request instead of applied output | Fixed-capacity applied-command trace and replay | Authenticated evidence export | T0–T3 |
| H-11 | Model overconfidence | Reduced-order plant omits contact/soft tissue | Claims bounded to research simulation | Bench, HIL, multibody and human-factors validation | P1, F0 |
| H-12 | Ethical or consent violation | Unauthorized assistance mode | Moral gate can only reduce authority | Physical wearer-controlled stop and governance | S1, D0 |

## Closure rule

A hazard is not closed by code review alone. Closure requires a named control,
a test or inspection method, retained evidence, and confirmation that the
control does not introduce a more severe secondary hazard.
