# PIE Phase-0 equipment lifecycle / spares oracle

## Purpose

Freeze implementation-independent semantics for scheduled maintenance, critical spares, local repairability, local reproducibility, spare production, and correlated/common-mode failures.

The oracle is `scripts/pie-lifecycle-oracle.py` and imports no Symthaea code.

## Semantics

- an imported machine being operational now does not imply local repairability or local reproducibility;
- scheduled maintenance consumes real time and declared spare inventory;
- a required spare that is unavailable can stop the machine;
- locally produced spares may extend availability without implying that the complete machine is reproducible;
- unscheduled failures require both local repair capability and required parts;
- correlated failures may affect multiple nominally redundant machines simultaneously;
- future spare production cannot be consumed before its declared production time;
- campaign uptime and downtime remain explicit.

## Executed synthetic fixtures

The final self-test was executed locally on 2026-09-11 and returned `ok` before this reference was committed.

Fixtures cover:

1. a machine serviced every 100 h consumes two seals over a 250 h campaign and accrues 20 h planned downtime;
2. with only one seal, the same machine stops at the second required service;
3. producing a replacement seal before the second service extends campaign availability without changing `locally_reproducible = false`;
4. an unscheduled bearing failure stops a machine when no bearing is available;
5. supplying the bearing enables repair and adds explicit downtime but still does not imply machine reproduction;
6. a correlated controller failure affecting two machines with only one controller spare leaves only one operational;
7. a machine marked non-repairable locally cannot be repaired merely because a spare part is present;
8. malformed/nonpositive campaign horizon fails closed.

## Important limitations

This is a deterministic reference scheduler, not a reliability prediction. It intentionally does not model stochastic hazard functions, Weibull/Poisson failure processes, overlapping maintenance windows, degraded-but-operational states, work crews, robotics capacity, repair skill queues, cannibalization, salvage yield, tool calibration, environmental stress models, or true lunar/Mars machine lifetime.

A later production layer should compose this contract with PIE-007/008 dependency closure, PIE-006 inventory/circularity, process throughput, utilities, and evidence-bearing failure/lifetime distributions.

## Non-claims

No real equipment lifetime, maintenance interval, spare rate, settlement uptime, or autonomous maintenance authority is established.

Tracks #1648, #1647, #1618, and master #1604.
