# symthaea-physical-cognition

Neutral contracts for experimentally evaluating physical computation backends without promoting device behavior into claims about consciousness, superiority, or hardware validation.

## Boundary

This crate is intentionally distinct from `symthaea-core::hdc::substrate_independence`.

- A `PhysicalBackend` says how controls become observations.
- `ExecutionBoundary` records whether the result came from software, simulation, an emulator, or a physical device.
- `EvidenceLevel` prevents simulation/emulation results from being silently promoted into hardware measurements.
- `EnergyBoundary` distinguishes device-only energy from whole-system energy.

The initial contract does not schedule backends, mutate the active HDC/LTC cognitive loop, or make any consciousness claim.
