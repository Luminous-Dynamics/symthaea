# battery-inverse-design-audit

Dependency-free Rust 1.96 reference semantics for a finite battery candidate portfolio.

Implements structural checks, hard-gate-first eligibility, conservative scenario aggregation, exact finite-set Pareto filtering, and a bounded read-only experiment selector.

The experiment score is explicitly a heuristic information-gain-per-normalized-resource ranking. It is not a calibrated Bayesian expected-value-of-information estimate.

Unit tests are authored but must be run by the dedicated exact-head workflow. No physical battery property, safety, manufacturability, lifetime or commercial claim is made.

See `docs/transport/BATTERY_INVERSE_DESIGN_V0_1.md` and `fixtures/ev_battery/battery_inverse_design_v0_1.json`.