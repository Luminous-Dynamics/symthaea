# PIE-002A Utility / Thermal Oracle Evidence

Status: independent reference implementation; synthetic execution evidence only.

## Purpose

`scripts/pie-utility-accounting-oracle.py` freezes first-order utility-accounting semantics independently of the production Rust implementation.

It does not import Symthaea code and does not claim that any lunar or Martian industrial process is feasible.

## Executed self-test

Executed locally on 2026-09-11 with Python 3. The script returned `ok`.

Synthetic fixtures exercised:

- identical batch energy with different durations produces different average power;
- energy capacity may be guaranteed while peak power is impossible;
- gross electrical demand remains explicit when recovered energy is credited;
- recovered energy is bounded by source energy and storage acceptance;
- low-temperature waste heat cannot satisfy a hotter process even when energy quantity is sufficient;
- sufficiently hot/energetic heat is `Guaranteed` by the screening rule;
- overlapping uncertain thermal envelopes are only `Possible`;
- widening uncertainty cannot strengthen a conservative demand/capacity conclusion;
- malformed/non-finite/zero-duration inputs fail closed.

## Feasibility semantics

For an uncertain demand interval `D=[Dmin,Dmax]` and capacity interval `C=[Cmin,Cmax]`:

- `Guaranteed` iff `Dmax <= Cmin`;
- `Impossible` iff `Dmin > Cmax`;
- otherwise `Possible`.

This intentionally prevents overlapping uncertainty ranges from being presented as guaranteed utility feasibility.

## Electrical recovery

Accepted recovery is bounded by all three of:

1. physically recoverable energy;
2. storage acceptance;
3. gross source energy.

Gross energy, accepted recovery, and net energy remain separate reported quantities.

## Thermal screening boundary

Temperature compatibility is a first-order screening rule only. PIE-002A does **not** calculate entropy, exergy, heat-exchanger approach temperatures, phase change, thermal losses, reaction enthalpy, kinetics, or equipment thermal limits.

A closed utility ledger is not proof of thermodynamic or industrial feasibility.

Tracks #1610 and master program #1604.
