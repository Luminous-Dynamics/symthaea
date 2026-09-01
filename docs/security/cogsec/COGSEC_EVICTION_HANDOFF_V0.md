# CogSec Eviction Handoff v0

Status: **current lifecycle characterized; canonical handoff state commitment implemented on dormant branch; ownership repair tracked by #290; no runtime behavior changed**.

This document freezes the first-hook interpretation of working-memory eviction before any CogSec enforcement is introduced.

## Why this is a separate owner

`ContinuousMind::evicted_items` is not merely a debugging cache. The public `take_evicted_tagged()` API drains it, and outer `Symthaea` consumes the drained records for both graduation routing and asynchronous database persistence.

Therefore the handoff is a distinct mutable resource:

`mind/memory/eviction-handoff`

It must not be silently folded into the working-memory root, the graduation queue root, or persistent-memory authority.

## Current legacy fan-out

One full-capacity `ContinuousMind::process_inputs()` eviction currently performs, in order:

1. remove the oldest item from the five parallel working-memory arrays;
2. compute `steps_survived`;
3. enqueue a `GraduationEvent` into the **mind-local** `MemoryCoordinator`;
4. append an `EvictedMemory` record to `evicted_items`;
5. append the incoming item to working memory;
6. continue active-state and goal/affect processing.

Outer `Symthaea`, after `mind.tick()`, then:

1. drains `take_evicted_tagged()`;
2. reconstructs another `GraduationEvent` for every drained record;
3. queues that event into a **separate outer** `MemoryCoordinator`;
4. immediately processes the outer graduation queue into outer episodic memory;
5. when a database is configured, asynchronously persists the same drained eviction records.

`ContinuousMind::new()` and outer `Symthaea` instantiate their coordinators independently. Repository search currently finds no runtime `process_graduations()` call for the mind-local coordinator; that coordinator is nevertheless used for other legitimate functions such as dream-state pruning.

## Semantic divergence

The two graduation routes do not construct the same event.

The mind-local route currently derives:

- label from `metadata["topic"]` or empty;
- `final_activation = 0.5`;
- Psi from `self.state.consciousness_level`;
- coherence from `self.state.consciousness_level`.

The outer route currently derives:

- label `wm_eviction_step_{interaction_count}`;
- `final_activation = 0.5`;
- Psi from the post-tick snapshot consciousness;
- coherence from post-tick `snapshot.meta_awareness`.

So deleting either route without an explicit semantic decision could change behavior.

## ObserverOnly resource model

Until #290 resolves the lifecycle, first-hook shadow evidence must treat these as separate causal transitions/resources:

1. **WorkingMemory** — removal/admission state;
2. **EvictionHandoff** — append/drain state;
3. **mind-local GraduationQueue** — current legacy enqueue, if observed;
4. **outer GraduationQueue** — separate outer enqueue/process path, if observed;
5. **persistent database effect** — separate external persistence action.

No one transition grants authority to the next.

In particular:

> WM eviction does not authorize graduation.
>
> Eviction handoff does not authorize persistent memory.
>
> Graduation candidate validity does not authorize database persistence.

## Canonical handoff state — dormant source

`agent/cogsec-canonical-effects-v0` defines:

- `EvictionHandoffItemView`;
- `EvictionHandoffItemCommitmentV1`;
- `EvictionHandoffStateCommitmentV1`;
- canonical resource ID `mind/memory/eviction-handoff`.

The state commitment binds:

- buffer count and order;
- exact HDC content bits;
- `steps_survived`;
- `MemorySource`;
- legacy `is_verified`;
- deterministic metadata commitment.

It intentionally defines **state identity only**. No frozen-K0 mutation class is assigned to append or drain while #290 is unresolved.

## Target ownership repair

The preferred end state is a single semantic construction point:

`WorkingMemory eviction`

→ `EvictionHandoffRecord`

→ inert `GraduationCandidate` captured once

→ one authoritative graduation coordinator/queue

→ separate persistent-memory qualification/commit

Database persistence remains a separate effect.

The candidate should capture the exact semantic fields once at the eviction boundary. Outer orchestration may decide whether to submit it, but must not reconstruct label/Psi/coherence from a later snapshot.

Do not remove the mind-local `MemoryCoordinator` wholesale: it currently has legitimate non-graduation responsibilities such as dream pruning. Remove or redirect the duplicated **graduation ownership**, not every coordinator function.

## Graduation canonicalization ratchet

Before runtime integration, one `GraduationEvent` semantic object must have one canonical event commitment.

Preferred dependency direction:

1. `symthaea-memory` owns `GraduationEventCommitmentV1` and `GraduationEvent::commitment_v1()`;
2. the private pending-queue root composes ordered event commitments;
3. the CogSec `GraduationEnqueue` effect domain-separates `enqueue this event commitment`;
4. the effects bridge never independently reserializes the event fields;
5. `symthaea-memory` never depends on CogSec/effects.

This prevents queue-state and effect encoders from drifting when `GraduationEvent` evolves.

## Migration ratchets

A lifecycle repair is not complete until tests prove:

1. one WM eviction yields exactly one authoritative graduation candidate;
2. the obsolete/unconsumed graduation queue no longer grows across repeated evictions;
3. the selected graduation semantics are explicit and stable;
4. the eviction handoff remains exactly observable before and after drain;
5. `LegacyBehaviorProjection` accounts for handoff contents/drain state during migration;
6. database persistence remains separately attributable from episodic graduation;
7. no item can be graduated twice through duplicated coordinator paths;
8. no shadow/evidence sequence is reused as owner freshness or authority;
9. ObserverOnly instrumentation remains behavior-preserving for the reviewed projection;
10. owner-aware/enforcement claims remain blocked until the lifecycle is singular and qualified.

## Qualification boundary

This dormant-branch source is **not compiled or qualified yet**. #270 remains the executable pre-runtime gate. The handoff commitment and eventual lifecycle repair must be added to the focused pinned-Cargo package/test/Clippy scope before any enforcement claim.
