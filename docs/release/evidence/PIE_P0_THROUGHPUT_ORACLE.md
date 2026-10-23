# PIE Phase-0 scale / throughput oracle

## Purpose

Freeze implementation-independent semantics for process throughput, batch/continuous bases, duty/availability, and explicit multi-unit scale-up.

The oracle is `scripts/pie-throughput-oracle.py` and imports no Symthaea code.

## Semantics

- continuous and batch processes use explicit, different bases;
- batch throughput derives from batch mass / cycle time;
- effective output includes duty cycle and availability;
- more than one process unit may not be credited unless an explicit parallelization rule exists;
- parallelization efficiency is evidence-bearing and may be less than 1.0;
- conservative feasibility is `Guaranteed`, `Possible`, or `Impossible`;
- widening rate/duty/availability/parallelization uncertainty cannot strengthen a conservative claim;
- malformed bases and zero cycle times fail closed.

## Executed synthetic fixtures

The final self-test was executed locally on 2026-09-11 and returned `ok` before this reference was committed.

Fixtures cover:

1. one exact 10 kg/h continuous unit produces 100 kg in 10 h;
2. requesting two units without a declared multi-unit rule fails closed;
3. two independent exact units produce 200 kg only when the rule explicitly permits it;
4. a two-unit parallel-efficiency range of 0.8..1.0 yields 160..200 kg, making an 180 kg requirement only `Possible`;
5. an exact 20 kg / 2 h batch process normalizes to 10 kg/h;
6. duty-cycle and availability uncertainty weaken guaranteed output;
7. demand above the admissible upper output is `Impossible`;
8. zero/invalid cycle time fails closed.

## Important limitations

This oracle does not prove real process scaling. It deliberately omits startup/ramp transients, buffers/queues, utility constraints, heat rejection, equipment footprint, feedstock starvation, maintenance events, correlated failures, and nonlinear scale-up physics. PIE-002 must separately screen energy/continuous power/peak power, and the lifecycle oracle must separately screen uptime/spares.

No lab datum is converted into a plant-scale default by this reference.

## Non-claims

No Moon/Mars process rate, plant capacity, production forecast, or hardware authority is established.

Tracks #1648, #1647, and master #1604.
