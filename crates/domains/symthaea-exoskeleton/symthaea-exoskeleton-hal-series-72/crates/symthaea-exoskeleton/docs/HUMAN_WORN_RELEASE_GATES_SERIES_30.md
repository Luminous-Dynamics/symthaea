# Series 30 Human-Worn Release Gates

Passing repository tests is not authorization for powered human-worn operation.

## Software and integration gates

- Full workspace formatting, Clippy, tests, feature matrix, Miri where supported, sanitizers, dependency audit, and reproducible builds.
- Deterministic evidence campaigns for authority, safety, passivity, timing, redundancy, state estimation, contact awareness, actuator residuals, power, lifecycle, provenance, and reliability.
- Trace replay proves the final applied command, not only the requested command.
- All zero-authority decisions reach HAL disablement within the allocated deadline.

## Bench and HIL gates

- Independent E-stop and contactor verified with the main processor stalled.
- Mechanical stops, passive backdrivability, and emergency release verified without software.
- Locked-rotor, reversed-polarity, runaway-command, feedback-freeze, encoder-jump, bus-loss, undervoltage, overcurrent, and thermal-soak campaigns.
- Independent load-cell or torque-transducer comparison for every powered joint.
- Worst-case execution and communication latency measured under maximum system load.

## Human factors gates

- Fit and skin-interface assessment across the intended anthropometric range.
- Don/doff and emergency-release usability with impaired mobility.
- Explicit consent session and immediate wearer-controlled authority withdrawal.
- Staged progression from unpowered fit checks, to suspended low-energy tests, to supervised powered trials under an approved protocol.

## Release rule

Any missing, stale, ambiguous, or failed gate means the powered human-worn configuration remains disabled.

## Series 36 continuity extension

Human-worn release additionally requires trusted clock continuity, fail-closed restart recovery, authenticated persistent maintenance and rollback state, governed calibration updates, and validated attachment-load supervision. A reboot, update, or calibration change begins a new consent and arming ceremony.
