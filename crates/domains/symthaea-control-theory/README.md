# symthaea-control-theory

Classical control theory for Symthaea — PID controllers, second-order response,
and Routh-Hurwitz stability. Connects to the robotics platforms (which have
per-platform controllers but no shared control layer).

Pure `std`, zero deps, no `symthaea-core` link. Checked vs textbook values.

- `pid::Pid` — discrete PID, drives a plant to setpoint in the tests.
- `second_order::SecondOrder` — damping regime, percent overshoot, settling time.
- `routh::{is_stable, rhp_root_count}` — stability without root-finding.

```bash
cargo test -p symthaea-control-theory
```
