# CogSec Episodic Memory Protected Owner v0

Status: **pre-runtime owner census / design contract**.

This contract records the canonical production `EpisodicMemory` mutation surface discovered while tracing the real persistence sink behind graduation. It does not change runtime behavior and does not claim enforcement.

## Why this owner matters

`EpisodicMemory` is a long-lived cognitive state owner. Its contents influence later replay, retrieval, consolidation, pruning, and training. A mutation to this store therefore has materially different security consequences from merely observing or staging a graduation candidate.

The current API is intentionally feature-rich, but from a CogSec perspective it exposes several distinct privileged mutation families through ordinary `&mut self` methods.

## Persistence terminology

Keep two meanings separate:

1. **Cognitive persistence** — information survives beyond transient working state and can influence later cognition. `EpisodicMemory` is such an owner even when its heap is only in RAM.
2. **Durable storage** — information is written to a database/file/remote store and survives process loss. SQLite/database persistence is a separate owner/sink.

`PersistentMemoryCommit` in cognitive policy must never silently imply permission for durable external storage.

## Current owner state

Security-relevant `EpisodicMemory` state includes at least:

- configured capacity and Psi threshold;
- ordered/logical set of `Episode` records in the priority heap;
- each episode's input/output, Psi, timestamp, prediction error, valence, coherence, replay count, consolidation strength, retrieval count, dopamine-at-encoding, bath state and semantic embedding;
- current cycle / replay cadence state;
- priority scores and ordering semantics;
- total stored/evicted/replay counters where they affect future behavior or qualification;
- demand-replay trigger state.

A future canonical owner root must define exactly which of these are protected semantics and which are telemetry. It must not hash a `BinaryHeap` through debug/iteration order unless that order is explicitly canonicalized.

## Current mutation families

### Episode admission / insertion

`store_if_significant(episode)`:

- updates current-cycle/replay cadence state before the threshold decision;
- may reject the episode;
- if accepted, updates average/minimum Psi statistics;
- inserts a prioritized episode;
- increments total stored;
- may increment capacity-eviction accounting.

This is the actual episodic cognitive-persistence transition behind graduation.

Direct callers exist outside `MemoryCoordinator`; therefore mediating only the graduation consumer does not close the owner.

### Reconsolidation / replay-state mutation

`replay_session(...)` and `replay_session_conditioned(...)`:

- sample episodes;
- call `replay_training_step` on an external `TrainableNetwork`;
- reset replay-control state;
- rebuild the episodic heap;
- increment episode replay counts;
- reconsolidate matching episodes, changing consolidation strength/retrieval count;
- update replay statistics.

This crosses **two protected owners**:

1. episodic memory;
2. trainable model/network state.

A replay-session call must therefore never be modeled as one implicit authority grant. Model mutation remains under the learning-promotion boundary (#210).

### Consolidation-strength mutation

`boost_causal_consolidation(...)` and `boost_recent_consolidation(...)` change stored episode consolidation strength and recalculate priority scores.

These are persistence/retention influence, not read-only analytics.

### Deletion / pruning

`prune(...)` removes stored episodes according to survival value.

`clear()` deletes the complete episodic store.

Deletion authority is not implied by insertion authority.

### Replay-policy/control mutation

Examples include:

- `set_batch_size(...)`;
- `trigger_demand_replay()`;
- `adapt_replay_interval(...)`.

These do not directly insert/delete an episode but change future replay/training behavior. Treat them separately from content persistence and from ordinary observation.

## Required writer topology

Long-term target:

`EpisodicMemoryReader`

- retrieval/query/statistics only;
- no mutation handle.

`EpisodicCandidateSink`

- accepts inert episode/persistence candidates;
- cannot mutate the owner.

`EpisodicMutationControl`

- privileged owner adapter for exact typed mutations;
- small reviewed surface;
- consumes exact authorization/commit context in enforcement mode.

`EpisodicReplayReader`

- may select/read replay candidates;
- cannot obtain training authority merely by reading memory.

`LearningPromotionControl`

- separately authorizes model mutation under #210.

Ordinary cognition should not carry a broad `&mut EpisodicMemory` together with a broad `&mut TrainableNetwork` when a narrower capability can express the required operation.

## Exact mutation classes required by K0.1

Do not force all episodic operations into one coarse class merely because frozen K0 has `PersistentMemoryCommit`.

At minimum distinguish exact effects for:

- `EpisodeInsert`;
- `EpisodeReplaceOrEvict` if capacity policy performs a real replacement;
- `EpisodeReconsolidate`;
- `EpisodeConsolidationBoost`;
- `EpisodePrune`;
- `EpisodeClear`;
- replay-control/config changes where security-significant;
- model `LearningPromotion` as a separate owner/effect.

Frozen K0 may map only the exact subset for which semantics are honest. Missing classes remain explicit limitations.

## Compound-effect rule

> One API call may cross several protected owners, but authority is evaluated per owner transition.

For replay:

`read episodes -> select batch -> propose model learning -> authorize/install model update -> propose episodic reconsolidation -> authorize episodic transition -> receipts`

The causal relationship between replay and learning grants neither side authority over the other.

## Direct-writer closure

The enforcement target is not "graduation goes through the coordinator."

It is:

> **Every security-relevant mutation of the canonical episodic owner is reachable only through the reviewed episodic mutation boundary.**

Required census includes direct calls from:

- outer `Symthaea` facade;
- cognitive-loop memory phase;
- memory coordinator;
- engineering/domain adapters;
- replay/consolidation paths;
- tests/examples only as non-production exceptions where appropriate.

Compile-fail/type ratchets should eventually prove ordinary readers cannot call writer methods.

## Owner-root requirements

A canonical episodic state commitment must:

- bind every security-relevant stored episode field;
- use a deterministic logical order independent of incidental `BinaryHeap` iteration layout;
- bind owner policy/config fields only when they are part of the protected state schema;
- distinguish content state from replay-control state if they have different mutation authorities;
- use semantic atom commitments rather than effect digests, per `COGSEC_CANONICAL_COMMITMENT_LAYERS_V0.md`;
- version schema changes explicitly.

## Graduation interaction

The target path is:

`EvictionHandoff`

→ inert `GraduationCandidate`

→ assessment/policy

→ exact `EpisodeInsert` proposal

→ `PersistentMemoryCommit` authorization for the episodic owner

→ owner mutation

→ receipt.

`GraduationEnqueue` is therefore not `PersistentMemoryCommit`.

## Durable database interaction

An accepted episodic insert does not authorize SQLite/file/network persistence.

Durable export requires a separate exact sink/effect and confidentiality/declassification analysis under #214 where applicable.

## Shadow-phase obligations

Before enforcement:

1. census all production episodic writer call sites;
2. define a deterministic legacy episodic behavior projection;
3. prove ObserverOnly instrumentation does not change insert/replay/prune behavior;
4. report every direct writer not yet behind the owner boundary;
5. do not claim P0 persistence coverage while any production writer bypass remains;
6. keep model-learning effects separate from episodic reconsolidation effects.

## Exit gate

This owner boundary is closed only when:

- all production episodic mutations pass through a small typed owner adapter;
- exact mutation classes are represented honestly;
- model learning is separately authorized;
- durable storage remains separate;
- owner roots are canonical and deterministic;
- `episodic_mutations_without_owner_transition == 0` under enforcement qualification.
