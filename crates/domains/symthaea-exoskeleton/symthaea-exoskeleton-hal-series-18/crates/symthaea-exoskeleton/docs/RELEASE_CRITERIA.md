# Exoskeleton Release Criteria

## Research-simulation release

A tagged research release requires:

- Formatting and warning-free lint across every feature combination.
- Default, Symtropy, sensor, HAL-mock, and all-feature tests passing.
- The deterministic evidence campaign passing all software gates.
- Patch-series replay producing the documented source tree.
- No real-transport path represented as implemented when it is refused.
- Updated hazard log, calibration protocol, and evidence limitations.

## Bench-only hardware release

In addition to the research criteria:

- A calibrated transport adapter with command acknowledgement.
- Independent e-stop, watchdog, current, voltage, and thermal evidence.
- Motor direction testing under current-limited power.
- Mechanical stops and unloaded fixture testing.
- Hardware-in-the-loop injection for every typed fault.
- A signed, immutable hardware/software/calibration evidence bundle.

## Human-wearable release

No repository-only milestone authorizes this stage. It additionally requires a
formal risk-management process, qualified multidisciplinary review, applicable
regulatory and ethics approvals, validated mechanical fit and release, staged
human-factors protocols, and retained incident-response procedures.
