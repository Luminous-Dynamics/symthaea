# PIE-009O safe active-diagnosis oracle evidence

Date: 2026-09-12

## Scope

This note records the independent synthetic reference semantics in `scripts/pie-009o-safe-active-diagnosis-oracle.py`.

The oracle addresses a gap between passive diagnosability and executable recovery: when the current belief is ambiguous, can the system safely perform a sequence of diagnostic probes that resolves the required `PROCEED` / `BLOCK` decision?

## Independent execution

The final candidate self-test was executed locally with Python 3 on 2026-09-12 and returned:

`ok`

No Symthaea module is imported.

## Reference semantics

- probes are admitted only when safe in every hidden world still present in the current belief;
- unsafe but informative tests are rejected rather than credited;
- an uninformative probe cannot advance diagnosis;
- observations partition the current belief, after which the planner replans;
- if all worlds in the current belief imply the same decision, diagnosis stops immediately;
- probes are one-use in the reference campaign;
- the solver minimizes **worst-case diagnostic depth**, not expected depth;
- no world probabilities or expected-value weights are invented;
- deterministic tie-breaking is explicit;
- if no universally safe informative sequence resolves the decision, the result is blocked (`None` in the reference API);
- adding an available redundant safe probe cannot worsen the optimum;
- malformed probe maps and unknown references fail closed.

## Executed synthetic fixture

The hidden worlds match PIE-009N: generator health × intertie state, with `PROCEED` only for `healthy + closed`.

Available probes are:

- `probe_gen`: safe generator-health diagnosis;
- `probe_tie`: safe intertie-state diagnosis;
- `probe_gen_redundant`: redundant safe generator diagnosis;
- `live_spin_test`: informative generator test that is unsafe in failed-generator worlds.

The executed fixture demonstrates:

1. the complete ambiguous belief is resolved with worst-case depth 2;
2. deterministic tie-breaking selects `probe_gen` first;
3. an observed failed generator collapses immediately to `BLOCK` after one probe;
4. a healthy generator still requires tie diagnosis before `PROCEED`;
5. the unsafe live-spin test is not selected from the full belief;
6. removing `probe_tie` leaves the healthy branch unresolved and the plan blocked;
7. a homogeneous `BLOCK` belief requires zero additional probes;
8. adding a redundant safe generator probe does not worsen worst-case depth;
9. an uninformative probe cannot fabricate progress;
10. malformed probe definitions and unknown worlds fail closed.

## Limitations

This is a symbolic active-diagnosis reference only. It does not model physical test energy, test duration, sensor noise, calibration, equipment wear caused by probing, probabilities, Bayesian experiment selection, continuous state estimation, electrical transients, cyber compromise, or hardware authority.

A production implementation should compose probe feasibility with PIE-009J startup-energy/islanding accounting and PIE-009I authority, and should use PIE-009N to determine whether a surviving sensor/probe portfolio is structurally sufficient.

## Promotion boundary

`passive diagnosability -> safe active-diagnosis plan -> partially observed recovery plan -> proof-carrying admissibility verification -> local authority/safety kernel`

Tracks #2036, #2030, #1972, #1978, #1928, #1857, #1647 and master #1604.
