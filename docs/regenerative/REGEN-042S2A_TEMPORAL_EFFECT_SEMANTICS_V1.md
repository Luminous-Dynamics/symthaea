# REGEN-042S2A — Temporal Effect Semantics v1

Status: preregistration hardening only.

Parent effect contract: REGEN-042S2 / #3834.
State predecessors: REGEN-042S1/S1A/S1B / #3823, #3832, #3833.
Primitive predecessor: REGEN-042S0A / #3831.
Authoritative transition semantics: Luminous-Dynamics/mycelix#1519 / REGEN-042A.

## 1. Purpose

S2 freezes typed exogenous effects and target/timing conflict validation. Before S3 can apply those effects, the model must distinguish effects that occur once from effects that constrain state throughout an interval.

Without that distinction, an interval event can accidentally mean any of the following:

```text
apply one mutation at interval start
apply the mutation once per tick
hold an override while active
apply once at interval end
restore a snapshotted old value at interval end
```

Those semantics are materially different.

S2A freezes:

```text
Impulse
!= TemporalOverlay
```

and forbids implicit repeated application.

## 2. Core theorem

```text
validated typed effect
+ explicit temporal application class
+ exact timing compatibility
+ overlay/base-state separation
= unambiguous temporal effect plan
```

not successor state, physical truth, hazard probability, emergency authority, or actuation.

## 3. Impulse effect

An impulse changes modeled state exactly once at one campaign tick.

Conceptually:

```rust
TemporalApplication::Impulse { at: Tick }
```

An impulse is not active before or after that tick.

It MUST NOT be represented as a one-tick interval merely to reuse interval code.

## 4. Temporal overlay

An overlay modifies the effective modeled view while a non-empty half-open interval is active.

Conceptually:

```rust
TemporalApplication::Overlay { interval: TimeInterval }
```

For `[start, end)`:

```text
active at start
active before end
inactive at end
```

The overlay is evaluated as a constraint/modifier, not re-applied as a fresh mutation each tick.

## 5. Initial v1 effect/application mapping

S2A freezes the first intended application classes:

```text
NoOpFixture                         -> explicit control; no mutation
DependencyUnavailable               -> Overlay
DependencyCapacityScaled            -> Overlay
DependencyCapacityCapped            -> Overlay
ScenarioDemandScaled                -> Overlay
ProvisionLeadTimeExtended           -> Overlay
InventoryLoss                       -> Impulse
FailureDomainUnavailable            -> Overlay
```

A later revision may add other effect classes, but it must not silently change these v1 temporal meanings.

## 6. Inventory loss is impulse-only in v1

A quantity-valued physical stock loss represents one modeled loss event.

Therefore:

```text
InventoryLoss + Interval
=> invalid effect input
```

This prevents a five-tick loss event from accidentally subtracting the full quantity five times.

A future continuous loss/rate model requires a different typed effect with explicit rate/unit/time semantics.

## 7. Overlay effects are interval-only in v1

The initial availability/capacity/demand/lead-time/failure-domain modifiers are temporary scenario conditions.

They therefore require non-empty interval timing.

An instantaneous persistent outage or permanent parameter mutation is not inferred from an `Instant` timestamp. If later required, it must use a separately specified persistent-transition effect.

## 8. No hidden persistent mutation

S2A v1 contains no generic `SetForever` or “instant means persistent from now on” convention.

Persistent model-state changes require explicit semantics in a later stage/revision.

This prevents recovery/end-of-shock behavior from depending on undocumented conventions.

## 9. Base state vs effective state

S3 must distinguish the enduring/base state from temporary overlay-derived effective state.

Conceptually:

```text
BaseState(t)
+ ActiveOverlays(t)
-> EffectiveState(t)
```

An overlay modifies only the effective state while active unless a separately typed enduring transition occurs.

## 10. Overlay expiry must not restore stale snapshots

When an overlay expires, the model removes that overlay from the active set.

It MUST NOT write back a value copied from the state that existed when the overlay started.

Example:

```text
base capacity = 100
overlay caps effective capacity to 40
legitimate enduring base change during overlay -> base capacity = 80
overlay expires
```

Correct result:

```text
effective capacity = 80
```

not stale restoration to `100`.

This is a critical anti-time-travel invariant.

## 11. Overlay stacking remains fail-closed unless qualified

S2's same-target overlapping-effect conflict rule remains authoritative for v1.

Two overlapping overlays on the same direct target are invalid unless a later explicit composition rule is separately preregistered and qualified.

S2A does not introduce implicit multiplication, min/max, strongest-wins, last-wins, or event-ID ordering.

## 12. Overlay on one target and impulse on another may coexist

An active overlay and an impulse with disjoint direct targets may coexist at the same tick subject to ordinary S2 validation.

This does not prove causal independence; it only removes direct-write ambiguity.

## 13. Impulse during same-target overlay conflicts in v1

If an impulse and overlay target the same direct state dimension at the impulse tick, S2A v1 treats the plan as conflicting unless an explicit later reducer profile defines their composition.

Canonical event order does not resolve the ambiguity.

## 14. Adjacent overlays

For the same target:

```text
A = [t0,t1)
B = [t1,t2)
```

A and B do not temporally overlap.

They may form ordered history if independently valid.

S3 must transition cleanly at `t1` without evaluating both overlays simultaneously.

## 15. No-op timing

`NoOpFixture` may use an instant or interval solely to test campaign/control timing.

It contributes no mutating target and must never trigger fallback, recovery, stock consumption, or state restoration merely because it becomes active/inactive.

## 16. Demand overlay semantics

`ScenarioDemandScaled` modifies the effective scenario demand while active.

It never mutates the adopted service floor.

At overlay expiry, effective scenario demand returns to the then-current base scenario demand, not necessarily the numeric demand that existed when the overlay began.

This preserves S1B's authority boundary and the no-stale-restore invariant.

## 17. Capacity overlay semantics

A capacity scale/cap modifies effective usable capacity according to the later reducer rules while active.

It does not erase nominal design capacity and does not by itself rewrite enduring base capacity.

At expiry, current base capacity again becomes visible.

## 18. Availability overlay semantics

`DependencyUnavailable` and `FailureDomainUnavailable` make the targeted effective availability unavailable while active.

They do not destroy the underlying dependency/failure-domain record.

Their downstream consequences are derived by later propagation logic, not stored as duplicate exogenous effects.

## 19. Provision lead-time overlay semantics

`ProvisionLeadTimeExtended` modifies effective modeled lead time while active.

When it expires, the then-current base lead time is visible again.

A permanent route/process change requires a distinct enduring-state transition.

## 20. Impulse history is append-only

Once an impulse has been applied, its event/effect identity remains in transition history.

Advancing to a later tick must not apply it again merely because the original event remains present in a full campaign schedule.

The reducer must distinguish:

```text
scheduled impulse
applied impulse
```

or otherwise prove exactly-once application from immutable transition history.

## 21. Exactly-once key

The first implementation should derive an unambiguous identity for one impulse application from at least:

```text
campaign revision
event ID
effect index/identity
impulse tick
prior transition lineage
```

The exact representation may differ, but replay must not double-apply an already-consumed impulse.

## 22. Replay theorem

Replaying the same transition from the same prior committed state with the same validated temporal effect plan must produce the same successor commitment and receipt.

Replaying from an already advanced successor as though the impulse were new is a different transition request and must not silently duplicate the impulse.

## 23. Tick-step size must not multiply overlays

An overlay's meaning is tied to the declared interval, not to the number of evaluation steps.

Refining a simulation from one-tick to smaller substeps in a future model must not multiply an overlay effect simply because it is evaluated more often.

Any timestep-sensitive dynamics require separately specified rate/differential semantics.

## 24. Interval splitting metamorphism

For a constant overlay with no intervening base-state transition, splitting:

```text
[t0,t2)
```

into adjacent equivalent overlays:

```text
[t0,t1) + [t1,t2)
```

should preserve effective-state trajectory at declared observation boundaries, provided the profile explicitly treats the split as the same scenario semantics.

This is a metamorphic qualification target, not a universal law across all future stateful effects.

## 25. Base-state mutation during overlay must be retained

A separately valid enduring/base-state transition that occurs while an overlay is active is retained beneath the overlay.

Overlay expiration cannot erase it.

This property should be tested directly because snapshot-restoration implementations commonly violate it.

## 26. Effect plan canonical identity binds application class

Two otherwise identical effects with different temporal application classes are different effect plans.

The canonical plan/receipt must bind whether the effect is an `Impulse` or `Overlay` and its exact timing.

## 27. Invalid timing is not an adverse modeled outcome

Examples:

```text
InventoryLoss(interval)
DependencyUnavailable(instant)
empty overlay interval
```

are invalid effect fixtures under S2A v1.

They are not resilience failures or service failures.

## 28. Recovery remains separate

Overlay expiry is not recovery evidence.

It means only that a synthetic exogenous modifier is no longer active.

If the base state is still failed because of an enduring failure or recovery has not completed, service/dependency state remains failed/degraded accordingly.

This prevents campaign-duration semantics from manufacturing recovery success.

## 29. First regression campaign

A later executable S2A/S3 campaign should include at least:

1. inventory loss instant accepted;
2. inventory loss interval rejected;
3. dependency-unavailable interval accepted;
4. dependency-unavailable instant rejected;
5. capacity-scale interval accepted;
6. demand-scale interval accepted;
7. lead-time-extension interval accepted;
8. failure-domain-unavailable interval accepted;
9. explicit no-op instant accepted;
10. explicit no-op interval accepted;
11. overlay not repeatedly multiplied per active tick;
12. impulse applied exactly once;
13. replay from same prior state deterministic;
14. stale replay cannot double-apply an impulse to an already advanced state;
15. overlay expiry reveals current base state;
16. base-state change during overlay survives expiry;
17. overlay expiry does not imply recovery;
18. adjacent same-target overlays do not overlap;
19. overlapping same-target overlays remain conflict;
20. same-target impulse inside overlay remains conflict;
21. disjoint overlay + impulse may coexist;
22. application-class-only mutation changes plan identity;
23. interval-boundary mutation changes plan identity;
24. no-op activation/deactivation changes no substantive state.

## 30. Independent temporal oracle

At least the active-at-tick / exactly-once / expiry semantics should have a small implementation-independent expected table.

The table should cover ticks immediately before start, at start, before end, at end, and after end, plus impulse consumption state.

## 31. Handoff to S3

S3 receives only a validated temporal effect plan whose effects already have:

- exact targets;
- exact effect types;
- exact temporal application classes;
- exact timing;
- no unresolved same-target temporal conflicts;
- exact binding to the prior committed state/campaign revision.

S3 owns deterministic state reduction. It must not reinterpret an overlay as an impulse or vice versa.

## 32. Deliberate non-claims

REGEN-042S2A establishes no executable S0/S1/S2 PASS, no successor-state correctness, no physical rate model, no recovery theorem, no real hazard timing, no disaster forecast, no authority, and no physical-action permission.

Its proposition is narrow:

> compound-shock effects need explicit temporal application semantics so one-time losses are not multiplied across intervals, temporary conditions are not made permanent, and overlay expiry cannot restore stale state or masquerade as recovery.
