# PIE-009R metrology decision-budget oracle

Purpose: freeze an implementation-independent reference for decision-qualified metrology precision.

The central distinction is that a calibration route can exist while still being too imprecise for the decision it is meant to support.

The reference uses conservative bounded-error composition rather than a probabilistic uncertainty model. For each declared error component, the bound at evaluation age `a` is:

`component_bound = sensitivity * (bound_at_calibration + drift_bound_per_step * a)`

The total route bound is the sum of all component bounds. No independence assumption or cancellation is used.

A route is `Qualified` when it is within its declared validity interval and its total bound is no greater than the decision allowance. It is `InsufficientPrecision` when still temporally valid but above the allowance, and `Expired` after the declared valid-through step.

`metrologically_autonomous` is true only when the route is both `Qualified` and structurally `LocallyRenewable`.

The final candidate self-test was executed locally with Python 3 on 2026-09-13 and returned `ok`.

Executed fixtures cover exact-bound acceptance, drift-driven precision loss, non-improving error with age, a locally renewable but too-coarse route, an imported precise route that remains import-dependent, promotion after a better local standard, expiry, monotonic improvement from tighter bounds, deterministic metadata-sensitive digests, and malformed input rejection.

This oracle does not establish real lunar or Martian calibration performance. Physical bounds, drift, environmental stability, traceability, and lifetime require separate evidence.
