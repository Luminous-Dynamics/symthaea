# CogSec Canonical Commitment Layers v0

Status: **pre-runtime design contract on dormant `agent/cogsec-canonical-effects-v0`**.

This document refines the canonical-effect work discovered during the first-hook static audit. It introduces a three-layer commitment model so one semantic object has one field encoding while protected-resource state and state-transition effects remain independently domain-separated.

No runtime behavior, authority, enforcement, or qualification claim is introduced here.

## Problem

The current dormant canonicalization substrate is directionally correct but still has two classes of coupling risk:

1. some resource-state commitments reuse an entire effect digest to represent one stored record;
2. some semantic objects are independently serialized in more than one canonicalizer.

Examples found during the audit:

- working-memory state currently commits each item by constructing a `WorkingMemoryAdmit` effect and nesting its digest;
- goal-store state currently commits each goal by constructing a `GoalActivate` effect and nesting its digest;
- `GraduationEvent` fields are serialized once by the memory owner's pending-queue commitment and again by the CogSec `GraduationEnqueue` effect encoder;
- `MemorySource` discriminants are repeated across canonicalization modules.

These do not automatically make the current digests incorrect. They do create schema-drift and semantic-coupling hazards.

## Core rule

> **Canonical field semantics are shared; state identity and effect identity are not.**

A protected owner state must not depend on an operation/effect discriminator merely to identify the data it stores.

An effect must not re-serialize a semantic object if an authoritative canonical atom for that object already exists.

## Three commitment layers

### Layer A — semantic atom commitments

A semantic atom commits to one complete domain object independent of where it is stored or what operation is being proposed.

Examples:

- `WorkingMemoryItemCommitmentV1`;
- `GoalRecordCommitmentV1`;
- `GraduationEventCommitmentV1`;
- `EvictionHandoffItemCommitmentV1` (already implemented in the dormant branch);
- future provenance/security-envelope commitments.

Properties:

- explicit schema/domain tag;
- deterministic canonical field encoding;
- no mutation/effect discriminator;
- no owner container index unless index is intrinsically part of the object;
- no authorization semantics;
- no `ResourceVersion` semantics;
- no claim that the committed data is factual or trusted.

### Layer B — protected resource-state commitments

A resource-state commitment identifies the complete state owned by one protected resource.

It composes semantic atoms plus owner/container facts such as:

- capacity;
- item count;
- ordering/index;
- queue order;
- owner-specific configuration only where that configuration is part of the protected state schema.

Examples:

`WorkingMemoryStateV1`:

`domain || capacity || count || (index || WorkingMemoryItemCommitmentV1)*`

`GoalStoreStateV1`:

`domain || count || (index || GoalRecordCommitmentV1)*`

`PendingGraduationStateV1`:

`domain || count || (index || GraduationEventCommitmentV1)*`

`EvictionHandoffStateV1`:

`domain || count || (index || EvictionHandoffItemCommitmentV1)*`

A resource-state root MUST NOT change merely because the operation used to insert the same records is renamed, gets a different effect discriminant, or moves to a newer effect schema.

### Layer C — effect / transition commitments

An effect commitment describes one exact proposed or observed transition.

It composes semantic atoms plus transition-specific context.

Examples:

`WorkingMemoryAdmitV1`:

`effect-domain || WorkingMemoryItemCommitmentV1 || insertion_index`

`WorkingMemoryReplaceV1`:

`effect-domain || admitted_item_atom || admitted_index || evicted_item_atom || evicted_index || steps_survived/other exact transition facts`

`GraduationEnqueueV1`:

`effect-domain || GraduationEventCommitmentV1`

`GoalActivateV1`:

`effect-domain || GoalRecordCommitmentV1`

Effect identity remains distinct from owner state identity even when both reference the same canonical atom.

## Why this is stronger

### 1. Schema changes stay local

Changing an effect taxonomy/discriminant does not silently redefine the resource-state root.

Changing container ordering/state semantics does not silently redefine the semantic record itself.

### 2. One object gets one encoder

A `GraduationEvent` should not have two field serializers that must remain manually synchronized forever.

The owner/domain crate defines the canonical event atom. Queue-state and effect commitments compose that atom with different outer domains.

### 3. Review becomes tractable

A reviewer can separately ask:

- does this atom include every field of the semantic object?
- does this state root include every owner/container fact?
- does this effect include every transition-specific fact?

instead of auditing one monolithic serializer for all three concerns.

### 4. Formal properties become clearer

Required identities can be expressed directly:

- identical semantic objects -> identical atom commitment;
- different committed semantic fields -> different atom commitment except cryptographic collision;
- same atoms in different owner order -> different state root;
- same atom under different effect classes -> different effect commitments;
- changing an effect discriminator alone -> state commitment unchanged;
- changing a state-container schema alone -> semantic atom unchanged.

## `MemorySource` discriminant ownership

The audit also found repeated `MemorySource -> u8` mappings.

Do not let every canonicalizer freeze its own copy indefinitely.

Target options, in preference order:

1. define a versioned canonical `MemorySource` commitment/discriminant in the memory domain and reuse it;
2. define one dependency-neutral canonical memory-source adapter shared by all CogSec commitment layers;
3. until one of the above is implemented, keep explicit frozen test vectors in every duplicate encoder and treat divergence as a qualification failure.

Do **not** rely on Rust enum layout or serde enum representation as a security commitment.

## Graduation-specific refinement

`GraduationEvent` needs both this canonical layering fix and the semantic split tracked in #207/#290.

The eventual atom must commit to fields that have explicitly defined meanings. In particular:

- a constant fallback must not be named/claimed as an owner-observed `final_activation`;
- Psi/coherence must identify their measurement point/schema;
- legacy verification is epistemic/application data, not authority or affect;
- policy-derived persistence bonuses belong to policy/decision state, not owner observation facts.

Therefore do not freeze `GraduationEventCommitmentV1` until the compatibility semantics are explicitly named. A temporary legacy atom may be introduced if required, but its name must make the legacy/default semantics unambiguous.

## Migration plan

### C0 — current dormant substrate

Preserve existing digests while qualification is unavailable. Do not silently rewrite schemas under the same `v1` domains.

### C1 — introduce atom types

Add new, independently domain-separated atom commitments with frozen test vectors.

No existing state/effect digest changes yet.

### C2 — introduce v2 state/effect schemas

Build new `v2` state/effect commitments from the atoms.

Do not change the meaning of existing `v1` domains in place.

### C3 — cross-schema equivalence fixtures

For representative legacy records, prove that v1 and v2 commit the same logical field values even though their outer digests intentionally differ.

### C4 — first-hook adoption

ObserverOnly runtime instrumentation uses one reviewed schema version consistently for proposal and observation.

Evidence records include the commitment-schema identifier; digest equality without schema identity is insufficient.

## Qualification ratchets

Before runtime integration:

1. each public canonical atom has deterministic and field-sensitivity tests;
2. state roots never call an effect constructor as their semantic-record encoder in the new schema;
3. effect constructors never re-serialize an atom's fields in the new schema;
4. duplicate canonical enum/source mappings are eliminated or cross-tested with frozen vectors;
5. schema/domain identity accompanies every digest used for qualification;
6. no current v1 digest is silently redefined under the same domain tag;
7. pinned Cargo check/test/doc-test/Clippy passes on the exact branch head before any runtime claim.

## Constitutional relation

This layering supports the Cognitive Non-Escalation Invariant but does not itself grant security authority.

A commitment proves only deterministic identity under a named schema.

It does not prove:

- truth;
- provenance authenticity;
- authorization;
- resource freshness;
- ownership;
- trusted execution;
- policy endorsement.

Those remain separate CogSec facts and boundaries.
