# Symthaea Space Exosuit — Integrated EVA Mission v0.1

Status: **simulation architecture only**. This document and the associated mission harness do not establish human rating, flight readiness, medical validity, or superiority over any existing suit.

## Purpose

The Space Exosuit now has enough independent research models that subsystem-only tests can hide system-level mistakes. `eva_mission` therefore composes metabolism, protected multi-bus power, the PLSS reference twin, and radiation operations into a single deterministic mission lineage.

The key rule is that optimization never outranks survival constraints:

```text
mission plan / Symthaea recommendation
                |
                v
       protected power allocation
                |
        survival power satisfied? ---- no ---> ABORT
                |
               yes
                v
       actual assist power delivered
                |
                v
       actual human mechanical work
                |
                v
       metabolic O2 / CO2 / heat
                |
                v
          PLSS reference twin -------- failure ---> ABORT
                |
                v
      radiation exposure manager ----- alert ---> SHELTER / RETURN
                |
                v
 optional mobility/mission shortfall ---> DEGRADE
                |
                v
             CONTINUE
```

## What the harness intentionally couples

- electrical power actually delivered to mobility;
- mechanical assistance actually available from that electrical power;
- human positive/negative mechanical workload;
- metabolic power, oxygen use, carbon-dioxide production, and heat;
- PLSS oxygen inventory and ventilation/thermal viability;
- protected survival, mobility, and mission power buses;
- personal radiation dose accumulation and safe-haven timing;
- degraded mission behavior when optional buses cannot satisfy demand.

A mobility-power shortfall cannot leave the metabolic model assuming assistance that was never delivered. Undelivered mobility power also cannot appear as fictitious equipment heat.

## What remains deliberately separate

- primary pressure/oxygen/thermal hardware authority;
- certified actuator admission (`SpaceExosuitSafetyKernel`);
- rescue propulsion authority and no-fire geometry;
- detailed full-frame rigid-body dynamics;
- radiation transport / shielding attenuation;
- medical claims about exercise efficacy or long-duration health.

Future integration may observe or couple these systems, but it must not collapse their independent safety authorities.

## Reference scenario

The first reference mission is a small lunar EVA:

1. outbound traverse;
2. surface work;
3. return traverse.

The harness records elapsed time, oxygen consumed, CO2 produced, metabolic energy, human mechanical energy, survival/mobility/mission electrical energy, accumulated radiation, and degraded segments.

All reference numbers remain `Simulation` evidence.

## Required regression behaviors

1. Moderate assist may consume more mobility energy while reducing human work and oxygen use.
2. If mobility power is unavailable, the model moves task work back to the human rather than assuming nonexistent assistance.
3. Optional mission-bus loss degrades the mission and never consumes protected survival reserve.
4. Protected survival-power loss aborts the EVA.
5. PLSS loss of a viable survival path aborts the EVA.
6. An energetic-particle alert demands immediate shelter.
7. Invalid radiation instrumentation or a return-budget violation sends the astronaut toward safe haven.

## Next evidence tranche

The highest-value next work is not another isolated feature set. It is to increase fidelity around the integrated mission:

- wire explicit Earth/Moon/Mars gravity into the 20-DOF full-frame Symtropy world;
- add pressure-garment resistance to per-joint full-body workload rather than only aggregate workload;
- model dust exposure/degradation and electrodynamic-dust-shield availability;
- represent suitport ingress/egress and contamination transfer;
- add powered glove/tendon assistance and dexterity tasks;
- couple suit/rover/habitat safe-haven and external charging/refill services;
- run seeded multi-fault mission campaigns, not only isolated subsystem faults;
- compare assist policies on Pareto fronts rather than making one weighted score authoritative;
- graduate only through simulation -> hardware bench -> HIL -> human-in-loop -> relevant-environment qualification.
