# GEOM-003B3 — Real-Loop GWT Handler Delivery Gate

## Status

Architecture-integration tranche under #3141, stacked on GEOM-001C exact head `09bd26d627ff0d5d68ce73bfc2cd5a6ef332f604`.

This tranche brings the GEOM-003B2 delivery-boundary idea into the actual `CognitiveLoopService` GWT path without adding experiment state to the service or changing normal construction.

## Existing production path

When GWT is enabled, the cognitive-loop constructor creates a `UnifiedGlobalWorkspace` and registers two built-in handlers:

- `memory`: set the shared memory-consolidation flag;
- `perception`: increment the shared perception-broadcast counter.

`UnifiedGlobalWorkspace::process()` constructs broadcasts, iterates recipients, looks up matching registered handlers, and invokes them.

GEOM-003B3 preserves that architecture.

## Zero-default-impact rule

No constructor code changes.

No new `CognitiveLoopService` field.

No new `GwtManager` state field.

No changed GWT default.

Unless an experiment explicitly calls `install_gwt_builtin_delivery_gate(...)`, the repository runs the original constructor-installed handlers exactly as before.

## Installation boundary

The experiment supplies three shared atomics:

1. `delivery_enabled: Arc<AtomicBool>`;
2. `handler_invocations: Arc<AtomicUsize>`;
3. `blocked_deliveries: Arc<AtomicUsize>`.

The explicit accessor replaces only the existing named `memory` and `perception` handlers using the already-public `UnifiedGlobalWorkspace::register_handler` replacement semantics.

When enabled, each replacement performs the exact original side effect.

When disabled, each replacement:

1. increments `handler_invocations`;
2. observes the disabled gate;
3. increments `blocked_deliveries`;
4. returns without performing the original side effect.

Thus the existing workspace still owns:

- competition and entry;
- broadcast creation;
- recipient lists;
- recipient iteration;
- handler lookup;
- wrapper invocation;
- GWT cross-broadcast statistics and unrelated workspace dynamics.

Only the two built-in downstream side effects are gated.

## Why the caller owns the atomics

Keeping the experiment handles outside `CognitiveLoopService` makes the intervention explicit and disposable. It avoids:

- a persistent experiment-mode field;
- another service field-count-ratchet exception;
- a hidden environment variable;
- a globally mutable switch;
- exposing the private GWT object for arbitrary mutation.

The caller can preregister and retain exact gate provenance for an experiment run.

## Controls

### Enabled delivery

With GWT enabled and the gate installed/enabled:

- a matched probe must emit at least one workspace broadcast;
- wrapper invocation count must increase;
- blocked count must remain zero;
- memory flag and perception count must show their original side effects.

### Disabled delivery

With the same probe and gate disabled:

- workspace broadcasts must still occur;
- wrapper invocation count must increase;
- blocked count must equal wrapper invocations for the two gated built-in handlers;
- memory flag must remain false;
- perception count must remain zero.

### Rescue

After an initially blocked probe, changing only the caller-owned atomic gate to enabled and running the probe again must:

- increase invocation count further;
- leave blocked count at its previous value for enabled deliveries;
- restore memory and perception side effects.

### GWT absent

Installing the gate when the service was constructed with GWT disabled must return `false` and leave experiment counters unchanged.

## Relation to B1/B2

B1 is a broad emission ablation using core `WorkspaceConfig.enable_broadcasting=false`.

B2 proves a delivery-gate mechanism at the core workspace boundary.

B3 supplies the narrow experiment hook needed to test the same delivery concept inside the real cognitive-loop integration without changing default production behavior.

B3 does not make B1 or B2 redundant: their agreement/disagreement remains part of the preregistered falsifier matrix.

## Remaining compute caveat

GEOM-003B3 preserves all upstream GWT work through handler-wrapper invocation, but it does not and cannot claim identical downstream compute: when delivery is blocked, the memory/perception side effect itself is not executed.

This must remain explicit in later interpretation.

## Claim boundary

Allowed:

> The built-in GWT handler wrappers were invoked under both conditions, while the lesion blocked their downstream side effects; measured GEOM observables changed by X.

Not allowed:

> This gate disables consciousness.

Not allowed:

> A GWT-dependent geometric effect proves Global Workspace Theory.

Not allowed:

> Any such effect establishes a gravity-consciousness physical coupling.

## Next experiment gate

Once this implementation and its predecessors qualify, a real-loop runner can construct matched intact/lesion/sham/rescue services from the same deterministic genesis configuration, install B3 gates, collect `CycleResult.thought_vector` sequences, and feed those sequences through the frozen GEOM-001B/001C projection pipeline.
