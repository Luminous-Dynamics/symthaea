# CogSec First-Hook Evaluation Adapter v0

Status: **design + canonical effect/state/owner-commitment substrate on dormant branch; runtime integration still blocked on #270 qualification and real Cargo qualification of this branch**.

This contract is stacked conceptually after consolidated pre-runtime PR #270. It does not authorize opening the runtime PR before #270 obtains real pinned-Cargo qualification evidence.

## Purpose

The first `ContinuousMind` hook must produce real monitor-origin, exact-effect-bound evidence without turning the evidence observer into a state owner or manufacturing security facts merely to satisfy frozen K0.

Four roles remain distinct:

1. **Legacy owner — `ContinuousMind`** performs unchanged cognition.
2. **Shadow evaluation adapter** owns `ReferenceMonitor`, `TrustedFactAuthority`, sealed shadow policy, truthful read-only state commitments, and a monotonic shadow-evaluation sequence.
3. **`PortableEffectBoundShadowRuntimeObserver`** owns evidence bookkeeping and exact effect pairing only.
4. **`symthaea-cogsec-effects`** owns canonical effect/resource/transition commitments outside the logical monitor core.

`MemoryCoordinator` remains the owner of its private graduation queue and now mints its own opaque queue commitment; CogSec never receives the raw queue merely to hash it.

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

## Public commitment boundary — implemented on dormant branch

`agent/cogsec-canonical-effects-v0` no longer exposes raw hashing helpers as the normal application API.

Public callers use:

- `EffectCommitmentV1` — exact effect digest + exact semantic effect class;
- `ResourceStateCommitmentV1` — resource-tagged canonical pre-state commitment;
- `CanonicalTransitionCommitmentV1` — effect + pre-state pair that can exist only when both name the same protected resource.

The inner `CognitiveEffectV1` representation and raw SHA-256 helper functions are crate-private.

This prevents three different substitution classes:

1. arbitrary nested digests cannot be injected into an effect;
2. a goal-store root cannot silently stand in for a WM root;
3. a valid effect digest cannot be paired with the wrong protected resource without an explicit typed error.

The generic `Digest32` is deliberately recovered only at the trusted monitor-adapter boundary because frozen K0 remains cryptography-neutral.

## Exact effect taxonomy

Canonical effect classes are more precise than frozen K0:

- `WorkingMemoryAdmit`;
- `WorkingMemoryReplace`;
- `GraduationEnqueue`;
- `ActiveStateReplace`;
- `GoalActivate`;
- `AffectSet`.

The v1 class itself names its canonical protected resource.

Exact frozen-K0 mappings exist only for:

- `WorkingMemoryAdmit -> WorkingMemoryAdmission`;
- `GraduationEnqueue -> PersistentMemoryCommit`;
- `GoalActivate -> GoalActivation`;
- `AffectSet -> Affect`.

`WorkingMemoryReplace` and `ActiveStateReplace` deliberately return **no K0 mapping**. They remain explicit #201 taxonomy gaps rather than being coerced into admission/attention or another convenient existing variant.

This means S1 replacement and working-state influence may be observed canonically before K0.1, but they cannot contribute to a strong all-stage K0 authorization/attribution claim.

## Canonicalization rules

The underlying v1 encoder uses:

- explicit domain tags and effect discriminants;
- big-endian fixed-width integers;
- length-prefixed UTF-8 strings;
- HDC vectors committed as ordered exact `f32::to_bits()` values;
- metadata maps sorted deterministically by UTF-8 bytes;
- explicit `MemorySource` discriminants;
- floats committed as exact IEEE bits after legacy computation;
- full `LiquidHolocell` commitment: state, tau, exact dimensionality variant/value, pressure;
- active state additionally binds separately stored `current_thought`.

Matching commitments prove identity only, not authority/authentication/truth.

### Working-memory replacement

The replacement effect binds:

- admitted content/arrival tick/source/legacy verified flag/metadata/index;
- exact evicted content/arrival tick/source/legacy verified flag/metadata/index;
- exact `steps_survived` written to the legacy eviction record.

The graduation queue write remains a separate effect lineage. Replacement must never silently grant persistence authority.

## Canonical resource states

### Working memory

The typed WM constructor commits to:

- configured WM capacity;
- item count/order;
- every HDC vector;
- every arrival tick;
- every `MemorySource`;
- every legacy verification flag;
- every metadata map.

It **fails closed** if the five parallel arrays differ in length instead of zipping/truncating inconsistent state.

### Active cognitive state

The active-state commitment binds the full `LiquidHolocell` plus `current_thought`.

### Goal store

The goal-store commitment binds ordered records: ID, description, embedding, priority, progress, active flag.

### Affect

The v1 affect root commits to exact `emotional_valence` bits. Widening the protected affect owner later requires a new schema/domain.

### Graduation queue — owner seam implemented

`MemoryCoordinator::graduation_queue` remains private.

The memory owner now exposes only:

- `pending_graduation_count()` — cardinality, not content;
- `pending_graduation_commitment_v1()` — an opaque `PendingGraduationCommitmentV1`.

`PendingGraduationCommitmentV1`:

- has a private constructor;
- has no serde implementation;
- can only be minted through safe public API by `MemoryCoordinator` from its actual private queue;
- commits to queue length/order and every stored event field using an owner-specific domain-separated SHA-256 schema;
- can be passed to `ResourceStateCommitmentV1::graduation_queue_owner(...)` without revealing queued HDCs, labels or metadata.

The memory-owner state schema is intentionally an **owner-state commitment**, not a duplicate of the effect serialization. Effect identity and resource-state identity are separate commitments with separate domains.

Safe external code may copy/forward an already minted token, but cannot construct or deserialize one and claim it originated from the memory owner.

This closes the previous graduation privacy/root blocker for first-hook design. It does **not** create `ResourceVersion` or authorization authority.

## Exact-effect flow

For each mapped transition:

1. capture exact pre-state;
2. construct a class-bound `EffectCommitmentV1` from actual legacy values;
3. obtain the canonical/resource-owner pre-state commitment;
4. bind effect + state into `CanonicalTransitionCommitmentV1`;
5. reject if effect class and state commitment name different resources;
6. reject a strong K0 claim when `k0_mutation_kind()` is absent;
7. allocate adapter evaluation sequence;
8. issue verified transition/request using the bound effect digest and truthful pre-root;
9. obtain opaque monitor receipt;
10. append `...Evaluated`, receiving one-use pending token;
11. execute unchanged legacy mutation regardless of decision;
12. reconstruct the **actual** effect from resulting owner values with the same typed constructor;
13. compute actual post-state commitment from the owner;
14. consume pending token through observed-mutation API;
15. any mismatch/evidence failure invalidates the evidence claim but does not deny/rollback legacy cognition.

ObserverOnly remains:

> **fail open for legacy behavior; fail closed for evidence/qualification claims.**

## Active-state prediction

Before the real `holocell.step(input, 0.1)`, the trusted adapter may clone the captured pre-Holocell and apply the exact deterministic step to prepare the expected effect. The real owner still performs the unchanged step. Post-commit reconstruction from the actual resulting Holocell/current-thought detects prediction/runtime drift through digest mismatch.

The clone is proposal preparation, not a state owner and not permission.

## First runtime PR prerequisites

Do not add a `ContinuousMind` observer field/hook until:

1. #270 has a Cargo-generated committed lock state;
2. #270's seven focused gates pass on that exact head;
3. its schema-v2 PASS receipt is independently verified;
4. `symthaea-cogsec-effects` **and the additive `symthaea-memory` owner-commitment change** compile/test/clippy under pinned Rust/Cargo 1.96.0;
5. no placeholder/default resource root remains;
6. evaluation sequence remains distinct from owner `ResourceVersion`;
7. no monitor authority, trusted-fact authority, policy authority, or permit is stored in ordinary `ContinuousMind`;
8. default/no-feature behavior remains unchanged;
9. the first-hook integration uses the typed public commitment boundary rather than internal raw digest helpers;
10. unresolved `WorkingMemoryReplace` / `ActiveStateReplace` K0 mappings remain limitations, never implicit success.

The former graduation-owner-commitment blocker is now **implemented in source but unqualified** until real Cargo runs.

## First allowed claim

A successful first hook may claim only:

> For frozen deterministic S0/S1/S2 scenarios with truthful scoped resource commitments, enabling ObserverOnly CogSec instrumentation preserves the reviewed legacy behavior projection exactly while producing structurally valid, monitor-origin, exact-effect-bound shadow evidence for the K0-mapped transitions and explicitly reporting unresolved taxonomy stages as limitations.

It still does not establish P0 enforcement, owner-issued `ResourceVersion` freshness, complete mutation coverage, authenticated/witnessed evidence, trusted-runner attestation, unresolved-taxonomy closure, or production security closure.
