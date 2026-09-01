# CogSec First-Hook Evaluation Adapter v0

Status: **design + canonical-commitment substrate on dormant branch; runtime integration still blocked on #270 qualification**.

This contract is stacked conceptually after consolidated pre-runtime PR #270. It does not authorize opening the runtime PR before #270 obtains real pinned-Cargo qualification evidence.

## Purpose

The first `ContinuousMind` hook must produce real monitor-origin, exact-effect-bound evidence without turning the evidence observer into a state owner or manufacturing security facts merely to satisfy frozen K0.

Four roles remain distinct:

1. **Legacy owner — `ContinuousMind`** performs unchanged cognition.
2. **Shadow evaluation adapter** owns `ReferenceMonitor`, `TrustedFactAuthority`, sealed shadow policy, truthful read-only state commitments, and a monotonic shadow-evaluation sequence.
3. **`PortableEffectBoundShadowRuntimeObserver`** owns evidence bookkeeping and exact effect pairing only.
4. **`symthaea-cogsec-effects`** owns canonical effect/resource-state representations and SHA-256 commitments outside the logical monitor core.

No monitor/policy/fact authority or permit belongs inside ordinary `ContinuousMind` in ObserverOnly mode.

## Identity/freshness separation

Never interchange:

- evidence `EventId` sequence;
- shadow-evaluation sequence;
- cognitive tick;
- protected-owner `ResourceVersion`;
- protected-resource state root;
- request/proposal identity.

For ObserverOnly:

- `resource_version_before = None`;
- `resource_version_after = None`;
- `CognitiveTick` is correlation only;
- evidence sequence is evidence continuity only;
- no observer may synthesize owner freshness.

Frozen K0 also carries `MutationRequest.sequence`. The shadow adapter may allocate a private monotonic **evaluation sequence** only for non-capability, non-enforcing audit receipts. It must never be described as `ResourceVersion`. Before capability-bearing/enforcing use, sequence semantics must be rebound to the real owner/transaction model.

## No fake resource roots

K0 requires concrete expected/current resource roots. Therefore `Digest32::default()`, arbitrary constants, pointer/debug hashes, ticks, or observer counters MUST NOT stand for “unknown root.”

If a truthful root is unavailable, the corresponding monitor-origin claim is unavailable.

## Canonical effect substrate — implemented on dormant branch

`agent/cogsec-canonical-effects-v0` currently defines opaque-by-construction `CognitiveEffectV1` values. External callers provide actual legacy values; they cannot inject arbitrary nested HDC/metadata/active-state digests.

Implemented effect families:

- working-memory admission;
- working-memory replacement/eviction;
- graduation enqueue;
- complete active-state replacement;
- goal activation;
- affect transition.

Canonicalization rules include:

- explicit v1 domain tags and effect discriminants;
- big-endian fixed-width integers;
- length-prefixed UTF-8 strings;
- HDC vectors committed as ordered exact `f32::to_bits()` values;
- metadata maps sorted by UTF-8 key/value bytes;
- explicit `MemorySource` discriminants;
- floats committed as exact IEEE bits after legacy computation;
- full `LiquidHolocell` commitment: state, tau, exact dimensionality variant/value, pressure;
- active state additionally binds separately stored `current_thought`.

Matching effect commitments prove identity only, not authority/authentication/truth.

### Working-memory replacement

The replacement effect binds:

- admitted content/arrival tick/source/legacy verified flag/metadata/index;
- exact evicted content/arrival tick/source/legacy verified flag/metadata/index;
- exact `steps_survived` written to the legacy eviction record.

The graduation queue write remains a separate effect lineage. Replacement must never silently grant persistence authority.

## Canonical resource roots — implemented substrate

### Working memory

`working_memory_state_digest_v1(...)` commits to:

- configured WM capacity;
- item count/order;
- every HDC vector;
- every arrival tick;
- every `MemorySource`;
- every legacy verification flag;
- every metadata map.

It **fails closed** if the five parallel arrays differ in length instead of zipping/truncating inconsistent state.

### Active cognitive state

`active_state_digest_v1(...)` commits to the full `LiquidHolocell` plus `current_thought`.

### Goal store

`goal_store_state_digest_v1(...)` commits to ordered goal records using the same fields as `GoalActivateV1`: ID, description, embedding, priority, progress, active flag.

### Affect

`affect_state_digest_v1(...)` commits to exact `emotional_valence` bits. Widening the protected affect owner later requires a new schema/domain.

### Graduation queue

`graduation_queue_state_digest_v1(&[GraduationEvent])` canonically commits to an **explicitly supplied ordered queue** using the same fields as `GraduationEnqueueV1`.

This helper does **not** bypass `MemoryCoordinator` privacy. The live owner queue is still private, so general runtime graduation qualification remains blocked on a narrow owner-side commitment seam.

Preferred owner correction: expose only a read-only commitment API from `MemoryCoordinator` (or equivalently narrow owner-owned commitment capability), not the raw private queue.

A deterministic test fixture may use the canonical empty-queue root only when it separately proves the coordinator begins empty. Do not generalize that assumption to arbitrary live minds.

## Exact-effect flow

For each mapped transition:

1. capture exact pre-state;
2. construct one canonical effect from actual legacy values;
3. compute its v1 digest;
4. compute the truthful pre-resource root;
5. allocate adapter evaluation sequence;
6. issue verified transition/request using the same digest/root;
7. obtain opaque monitor receipt;
8. append `...Evaluated`, receiving one-use pending token;
9. execute unchanged legacy mutation regardless of decision;
10. reconstruct the **actual** effect from resulting owner values with the same constructor;
11. compute actual digest and truthful post-root;
12. consume pending token through observed-mutation API;
13. any mismatch/evidence failure invalidates the evidence claim but does not deny/rollback legacy cognition.

ObserverOnly remains:

> **fail open for legacy behavior; fail closed for evidence/qualification claims.**

## Active-state prediction

Before the real `holocell.step(input, 0.1)`, the trusted adapter may clone the captured pre-Holocell and apply the exact deterministic step to prepare the expected effect. The real owner still performs the unchanged step. Post-commit reconstruction from the actual resulting Holocell/current-thought detects prediction/runtime drift through digest mismatch.

The clone is proposal preparation, not a state owner and not permission.

## Graduation owner seam — remaining implementation blocker

Current `MemoryCoordinator::graduation_queue` is private. Before general S1 monitor-origin graduation evidence, add a narrow owner-side commitment seam that returns the canonical queue root without exposing raw queued memory contents or mutable access.

The owner seam should use the same v1 queue/event representation as `symthaea-cogsec-effects`; do not create a second serialization/hashing scheme.

## First runtime PR prerequisites

Do not add a `ContinuousMind` observer field/hook until:

1. #270 has a Cargo-generated committed lock state;
2. #270's seven focused gates pass on that exact head;
3. its schema-v2 PASS receipt is independently verified;
4. `symthaea-cogsec-effects` compiles/tests/clippies under pinned Rust/Cargo 1.96.0;
5. the graduation owner commitment seam is implemented or graduation is explicitly excluded from the first monitor-origin claim;
6. no placeholder/default resource root remains;
7. evaluation sequence remains distinct from owner `ResourceVersion`;
8. no monitor authority, trusted-fact authority, policy authority, or permit is stored in ordinary `ContinuousMind`;
9. default/no-feature behavior remains unchanged.

## First allowed claim

A successful first hook may claim only:

> For frozen deterministic S0/S1/S2 scenarios with truthful scoped resource commitments, enabling ObserverOnly CogSec instrumentation preserves the reviewed legacy behavior projection exactly while producing structurally valid, monitor-origin, exact-effect-bound shadow evidence for the mapped transitions.

It still does not establish P0 enforcement, owner-issued freshness, complete mutation coverage, authenticated/witnessed evidence, trusted-runner attestation, unresolved-taxonomy closure, or production security closure.
