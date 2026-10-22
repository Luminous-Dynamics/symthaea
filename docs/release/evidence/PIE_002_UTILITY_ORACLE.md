# PIE-002A Utility / Thermal Oracle Evidence

Status: independent reference implementation; synthetic execution evidence only.

## Purpose

`scripts/pie-utility-accounting-oracle.py` freezes first-order utility-accounting semantics independently of the production Rust implementation.

It does not import Symthaea code and does not claim that any lunar or Martian industrial process is feasible.

## Exact-byte execution evidence

The hardened oracle bytes checked in on 2026-09-13 were executed locally with Python 3.13.5 before commit.

```text
python3 -m py_compile                         PASS
scripts/pie-utility-accounting-oracle.py --self-test   PASS (`ok`)
raw source SHA-256                           b4e1a403d6a67697230c3077fadcfab01cdab88fde050e75a33d2e1238eba291
locally computed Git blob SHA-1              cd1743ae4ad6247a3a90298734fd9db427621fdb
GitHub stored Git blob                       cd1743ae4ad6247a3a90298734fd9db427621fdb
```

This is reference-oracle execution evidence only. It is not production Rust qualification, repository-wide CI evidence, or physical-process validation.

## Feasibility semantics

For an uncertain demand interval `D=[Dmin,Dmax]` and capacity interval `C=[Cmin,Cmax]`:

- `Guaranteed` iff `Dmax <= Cmin`;
- `Impossible` iff `Dmin > Cmax`;
- otherwise `Possible`.

This intentionally prevents overlapping uncertainty ranges from being presented as guaranteed utility feasibility.

## Gross supply is not silently reduced by recovery

The original PIE-002A draft reported gross energy, recovery and net energy separately, but it then used net energy to screen the same cycle's energy capacity and average continuous-power requirement. That could over-credit energy recovered later in the cycle.

The hardened contract separates two propositions:

1. **gross/current-cycle screening** uses gross electrical energy and gross average power and makes no recovery-timing credit;
2. **steady-cycle accounting** may report lower net energy/average power only for energy that survives explicit capture, charge-power, delivery-efficiency and discharge-power limits.

Therefore:

```text
recoverable energy
!= accepted recovery
!= usable steady-cycle recovery
!= proof of current-cycle supply
```

A process can be `Impossible` on its gross/current supply screen while a separately named steady-cycle accounting screen is `Guaranteed`. The latter does not overwrite the former.

## Power-screen boundary

The reported gross and steady-cycle power feasibility fields are **cycle-average power screens** derived from energy divided by declared batch duration and compared with declared sustained capacity. They are not time-resolved load-profile, ramp-rate, dispatch, transient-stability, or continuous-delivery proofs. Peak power remains a separate screen.

## Electrical recovery bounds

Accepted recovery is bounded by all of:

- physically recoverable energy;
- storage energy acceptance;
- storage charge power integrated over the declared recovery window;
- gross source energy.

Usable steady-cycle recovery is then additionally bounded by:

- explicit recovery delivery / round-trip fraction in `[0,1]`;
- storage discharge power integrated over the next batch duration;
- gross process energy.

Gross energy, accepted recovery, usable steady-cycle recovery and net steady-cycle energy remain distinct reported quantities.

The oracle fails closed if multiplication/division overflows to a non-finite value, if a batch or recovery duration is non-positive, or if an efficiency fraction is outside `[0,1]`.

## Executed synthetic fixtures

The self-test covers:

- identical batch energy with different durations produces different gross average power;
- gross energy capacity can be guaranteed while peak power is impossible;
- `100 J` gross with `30 J` captured/reusable recovery reports `70 J` net steady-cycle energy while preserving `100 J` gross;
- charge power limits accepted recovery;
- delivery/round-trip efficiency limits reusable recovery;
- discharge power limits reusable recovery;
- recovery may strengthen a separately named steady-cycle screen while the gross/current-cycle screen remains impossible;
- low-temperature waste heat cannot satisfy a hotter process even when energy quantity is sufficient;
- sufficiently hot/energetic heat is `Guaranteed` by the screening rule;
- overlapping uncertain thermal envelopes are only `Possible`;
- widening uncertainty cannot strengthen a conservative demand/capacity conclusion;
- malformed, non-finite, zero-duration, invalid-fraction and arithmetic-overflow cases fail closed.

## Thermal screening boundary

Temperature compatibility is a first-order screening rule only. PIE-002A does **not** calculate entropy, exergy, heat-exchanger approach temperatures, phase change, thermal losses, reaction enthalpy, kinetics, equipment thermal limits, or temporal heat-network dispatch.

A closed utility ledger is not proof of thermodynamic or industrial feasibility.

## Missing-data boundary

The oracle requires every field used by a screen to be supplied explicitly. Missing demand/capacity/timing/storage fields fail rather than defaulting to zero. Production code should preserve the same rule: absence of a utility fact is not evidence of zero demand.

Tracks #1610 and master program #1604.
