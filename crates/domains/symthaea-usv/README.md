# Symthaea USV

`symthaea-usv` is a deliberately small, mission-neutral surface-vessel domain built on
`symthaea-maritime-core`.

It currently owns only:

- validated local USV motion/energy state;
- an optional geodetic navigation fix supplied by an external positioning stack;
- projection into shared `MaritimeState`;
- a fail-closed collision-risk **review boundary** based on range, DCPA, TCPA and
  source diversity.

## What it does not own

- no fleet manager;
- no cryptography or identity system;
- no Holochain governance;
- no weapons/targets/engagement model;
- no autonomous COLREG manoeuvre selector;
- no claim of regulatory compliance;
- no hydrodynamic simulator yet.

Those boundaries are intentional. HAL remains responsible for local actuator/safety
mechanisms, maritime-core for shared assurance semantics, positioning for navigation
estimation, and a future verified navigation-policy adapter for concrete COLREG conduct.

## Collision-review boundary

The 1972 COLREGs require proper lookout, safe speed, use of available means to determine
collision risk, and action to avoid collision. In particular, the IMO summary of Rule 7
warns against assumptions based on scanty information.

Accordingly, this crate does **not** infer a safe manoeuvre from a single weak track. It
requires escalation when:

- DCPA/TCPA evidence is missing;
- source diversity is below configured policy;
- immediate proximity crosses an operator-defined threshold; or
- projected DCPA/TCPA crosses operator-defined risk thresholds.

The output is only `NoTrigger`, `Monitor`, or `ColregReviewRequired`. A separate,
validated policy layer must decide what manoeuvre—if any—is appropriate under the
applicable COLREG situation, local rules, vessel characteristics and human authority.

Reference: IMO, *Convention on the International Regulations for Preventing Collisions
at Sea, 1972 (COLREGs)*, especially Rules 5–8 and Rules 13–19.

## Next evidence steps

1. compare DCPA/TCPA calculations against an independent navigation reference;
2. add noisy/lost/spoofed contact-track perturbations;
3. connect weather/sea-state inputs through existing Symthaea environment/physics domains;
4. add a simple surface hydrodynamics backend or external simulator bridge;
5. test navigation-policy adapters against published COLREG encounter cases before any
   real-vessel use.
