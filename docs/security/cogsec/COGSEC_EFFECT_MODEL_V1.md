# CogSec Exact Effect Model v1

Status: **post-K0 design contract; no runtime enforcement change**

Related: #143 frozen K0 kernel, #195 shadow qualification, #199 attribution gate, #201 K0.1 compound-mutation taxonomy.

## 1. Purpose

The v0 kernel already binds a `MutationRequest` to a caller-supplied `mutation_digest`, a protected `resource`, and an expected resource-state root. That is a strong starting point, but the live cognitive runtime shows that one nominal operation can contain several materially different state effects.

The most obvious example is a full working-memory admission:

```text
remove(oldest_item)
queue_graduation_candidate(oldest_item)
record_evicted_copy(oldest_item)
admit(new_item)
```

The same input then separately influences active Holocell/current-thought state and may activate a goal or mutate affect.

The authorization model therefore needs two ideas that must not be confused:

1. **atomic compound effects inside one protected resource owner**; and
2. **causally related but independently authorized transitions across protected resources/owners**.

Primary invariant:

> A permit authorizes the complete security-relevant effect on one protected owner boundary, and nothing outside that effect.

A second invariant prevents accidental over-bundling:

> Causal relationship does not imply shared authority.

One input may cause several proposed transitions. Permission for one does not automatically authorize the others.

## 2. Why a single mutation kind is not enough

`MutationKind` is useful for policy classification, capability attenuation, and stable reason codes. It should remain a coarse security class.

It must not be treated as the complete effect description.

For example:

```text
WorkingMemoryAdmission
```

cannot distinguish:

```text
Admit(item A)
```

from:

```text
Replace(item B with item A)
```

or from:

```text
Replace(item C with item A)
```

Those are different protected state transitions and must have different effect commitments.

Therefore K0.1 should preserve the useful split:

```text
MutationKind        = coarse policy/capability class
ExactEffect         = canonical complete state-transition description
EffectCommitment    = digest of canonical ExactEffect + relevant context
```

## 3. Protected-owner boundary

A **protected owner** is the serialization boundary that can authoritatively state the current version of a resource and atomically apply one accepted transition.

Examples:

- working-memory owner;
- active cognitive-state owner;
- goal-set owner;
- affect-state owner;
- persistent-memory owner;
- learning/model owner;
- trust-policy owner;
- security-policy owner;
- external-action executor.

A logical `ResourceId` identifies the protected resource inside or at that owner.

The owner is responsible for:

1. issuing/read-serving the authoritative current `ResourceVersion` and, where required, `ResourceStateRoot`;
2. preparing the exact effect against that pre-state;
3. participating in trusted-fact construction;
4. performing precommit freshness validation;
5. atomically applying the effect and advancing the resource version;
6. emitting the authoritative commit observation/receipt.

## 4. Canonical exact effect

The exact effect should be a canonical typed value rather than arbitrary prose or model-authored JSON.

Conceptually:

```text
ExactEffect {
    resource,
    precondition,
    operation,
}
```

where the operation is domain typed.

The security kernel does not need to understand every domain payload field. It does need a trusted canonical effect commitment and enough typed classification to apply policy/capability rules.

Recommended split:

```text
EffectDescriptor       // domain-owned, canonical, serializable data
EffectCommitment       // typed digest of descriptor + resource + pre-state domain
VerifiedEffectFacts    // trusted adapter assertion about descriptor classification
MutationRequest        // untrusted proposal referencing exact commitment
```

The requester may propose an effect. It may not manufacture the trusted classification/freshness facts used to authorize it.

## 5. Working-memory effect algebra

K0.1 should distinguish at least:

```text
WorkingMemoryEffect::Admit {
    admitted_item,
}

WorkingMemoryEffect::Replace {
    admitted_item,
    evicted_item,
    evicted_slot,
}

WorkingMemoryEffect::Merge {
    source_items,
    result_item,
}
```

A standalone `Evict` variant is appropriate only if the runtime has or later gains an independently callable eviction operation.

The item identifiers are opaque commitments/references, not raw private memory content.

### 5.1 Admit

`Admit` is valid only when the owner pre-state has capacity for admission without replacing protected content.

Changing the admitted item changes the effect commitment.

### 5.2 Replace

`Replace` is the correct semantic effect when capacity is full.

It MUST bind:

- admitted item commitment;
- exact evicted item commitment;
- exact slot/index or equivalent owner-stable position identity where ordering is semantically relevant;
- working-memory pre-version/root;
- replacement policy if selection is policy-derived and security relevant.

Changing the eviction target must change the effect commitment.

An `Admit` permit MUST NOT be accepted for `Replace`.

### 5.3 Merge

Dream consolidation currently replaces two adjacent working-memory items with one bundled item. The exact merge effect MUST bind:

- all source item references;
- their owner positions/identity where required;
- the resulting item reference;
- the conservative security/provenance-label result;
- pre-state owner version/root.

A merge is not ordinary admission and must not inherit admission authority implicitly.

## 6. Candidate-producing side effects

The current full-buffer path queues a graduation candidate and records an evicted copy immediately after removal.

The clean security architecture is to make those outputs **inert proposals/evidence**, not implicit persistence authority.

Rule:

> A protected transition may emit an unprivileged candidate or observation, but that candidate cannot itself cause another privileged transition without a new authorization boundary.

Therefore:

```text
WorkingMemory Replace
    -> EvictedMemory observation/candidate
    -> GraduationCandidate
    -> [separate PersistentMemoryCommit evaluation/permit]
```

If the graduation queue is architecturally guaranteed to be inert until later admission, appending a candidate need not consume persistent-memory authority.

If any queue consumer can persist or promote without a fresh CogSec decision, the queue itself becomes a protected resource and must be mediated.

This should be established by the mutation census, not assumed.

## 7. One input, several security transitions

The current input path is not one security transaction.

For a goal input with no eviction, the conceptual transition DAG is:

```text
Ingress observation
   ├─> WorkingMemory Admit
   ├─> ActiveState Influence
   └─> Goal Activation
```

For a full-buffer goal input:

```text
Ingress observation
   ├─> WorkingMemory Replace
   │      └─> GraduationCandidate
   │              └─> PersistentMemoryCommit proposal
   ├─> ActiveState Influence
   └─> Goal Activation
```

The arrows express causality/dependency, not shared authority.

A capability for working-memory replacement does not authorize goal activation. A capability for goal activation does not authorize persistent memory. A persistence denial does not retroactively falsify the historical fact that a working-memory replacement occurred.

## 8. Active-state influence

The live runtime steps the Holocell with the input and then replaces `state.current_thought` with the new Holocell state.

This is security-relevant active cognitive influence and deserves its own coarse mutation class if it remains in the required CogSec scope.

Provisional class:

```text
MutationKind::ActiveCognitiveStateInfluence
```

Do not reuse `Attention` merely because attention also affects cognition. Attention/salience weighting and direct replacement/evolution of active cognitive state are different state transitions.

The exact effect should bind at least:

- active-state resource;
- pre-state version/root;
- input/driver commitment or safe derivation reference;
- result commitment if deterministically available at preparation time, otherwise a typed transition-program commitment plus post-state verification rule.

The kernel must never authorize this from Phi, familiarity, confidence, or another floating cognitive score alone.

## 9. Affect mutation

Affect remains a separate mutation class and owner.

A feedback input may causally lead to both active-state influence and affect mutation, but those effects are independently classified and authorized.

For bounded influence profiles, the exact affect effect should bind:

- affected variable/domain;
- signed delta or bounded transformation;
- pre-state version;
- maximum allowed cumulative budget consumption;
- resulting value/range where deterministic.

This composes with the earlier `InfluenceBudget` design: per-event bounds are insufficient against slow cumulative manipulation.

## 10. Goal activation

Goal activation remains independent from working-memory and active-state mutation.

The exact effect should bind:

- goal-set resource;
- goal proposal identity/content commitment;
- active/inactive state;
- priority ceiling/classification;
- pre-state version;
- any replacement/deactivation side effects if goal capacity becomes bounded later.

A goal description stored in untrusted input is data. It is not authority to activate itself.

## 11. Transaction model

### 11.1 Single-owner transaction

For one protected owner:

```text
Prepare exact effect
      ↓
MutationRequest(effect_commitment, expected_version/root)
      ↓
CogSec evaluate / authorize
      ↓
MutationPermit
      ↓
owner lock / serialized boundary
      ↓
re-read authoritative version/root
      ↓
CogSec precommit
      ↓
CommitPermit
      ↓
apply complete ExactEffect atomically
      ↓
advance ResourceVersion
      ↓
CommitReceipt
```

The commit permit is consumed by the owner.

### 11.2 Multi-owner causal plan

For transitions spanning multiple owners, default to a causal plan of independent transactions rather than a distributed atomic transaction.

```text
TransitionPlan {
    plan_id,
    nodes: [single-owner proposed transitions],
    dependency_edges,
}
```

Each node receives its own authorization and receipt.

This keeps authority least-privileged and avoids making cognition depend on a heavyweight distributed two-phase commit.

### 11.3 When true cross-owner atomicity is required

Only introduce cross-owner atomic commit if a concrete safety invariant requires all-or-none semantics that cannot be expressed with compensation, staging, or dependency conditions.

If required later, the transaction coordinator itself becomes a high-assurance protected owner/TCB component and must not be smuggled into the LLM/planner layer.

## 12. Effect-set closure

A proposed effect is complete only if every protected write performed during commit is represented in the committed effect set.

Invariant:

> No protected write may be an uncommitted side effect of an authorized write.

Qualification should instrument owner commits and compare the observed changed-resource/slot set against the permitted effect set.

A mismatch invalidates enforcement qualification even when the nominal `MutationKind` was authorized.

## 13. Capability semantics

Capabilities should continue to authorize coarse mutation classes and resource scopes, but exact effect commitment is checked separately.

Conceptually:

```text
Capability says:
  principal P may perform WorkingMemoryReplacement
  on resource R
  up to consequence C
  during epochs/window W

Permit says:
  this exact replacement E
  against exact state/version S
  under exact policy/auth/revocation context
  may commit once
```

This preserves delegation attenuation without turning capabilities into enormous domain-specific payloads.

K0.1 should consider whether `WorkingMemoryAdmission` remains the capability class for both `Admit` and `Replace`, or whether replacement deserves a stronger separate class. The safe default is a distinct replacement class because replacement destroys/evicts existing protected content.

## 14. Consequence monotonicity

Compound effects use the maximum consequence of their protected sub-effects, never an average.

```text
consequence(compound) = max(consequence(effect_i))
```

A low-risk admission cannot conceal a higher-risk eviction by averaging the two.

Similarly, taint/confidentiality/control-integrity combination remains conservative; compound transformation cannot improve security labels by aggregation.

## 15. State versions versus state roots

Keep two concepts distinct:

- `ResourceVersion` — cheap monotonic owner freshness/CAS token;
- `ResourceStateRoot` — cryptographic commitment to exact content/state where required.

Hot-path precommit may use the owner version for freshness while high-consequence/recovery evidence additionally binds a state root.

A version is not a cryptographic commitment. A root is not a monotonic serialization token.

## 16. Post-state commitments

There are two legitimate transition forms:

### Deterministic prepared result

The exact post-state is known before authorization.

Bind:

```text
pre_state_root
exact_effect
expected_post_state_root
```

### Deterministic transition program

The result depends on protected internal state or computation not safely exposed before commit.

Bind:

```text
pre_state_root/version
transition_program/effect commitment
bounded parameters
postcondition verifier
```

The owner emits the actual post-state root after commit and the receipt records whether the postcondition held.

Do not allow an untrusted model to substitute an arbitrary postcondition verifier.

## 17. Failure semantics

Within one owner, a compound effect commits all-or-none.

Across separate owner transactions, failure does not rewrite history. Later dependent transitions may be skipped, quarantined, compensated, or re-planned according to explicit policy.

Examples:

- working-memory replacement succeeds, persistent graduation denied: memory remains evicted from working memory, but it is not silently persisted;
- active-state influence denied, goal activation independently allowed: policy must explicitly say whether the goal node depends on active-state success;
- stale goal-set version: reprepare/re-evaluate goal transition only; do not replay unrelated memory effects.

## 18. Shadow-mode mapping

Shadow mode keeps legacy behavior unchanged but should model the future transaction boundaries faithfully.

For each observed legacy effect:

1. construct the exact proposed effect that *would* have been authorized;
2. evaluate it without blocking;
3. record the evaluation;
4. observe the actual legacy state transition;
5. compare observed changed-resource/effect facts to the proposed exact effect;
6. record divergence as evidence rather than modifying legacy behavior.

This is stronger than merely recording `applied=true`.

A future #161 refinement should therefore add **effect correspondence** to the observation evidence, using opaque item/resource commitments rather than raw private content.

## 19. Required K0.1 proof/property obligations

At minimum:

1. `Admit` authority cannot commit `Replace`.
2. Changing an eviction target changes the exact effect commitment.
3. Changing any protected sub-effect changes the compound commitment.
4. Compound consequence cannot be lower than any sub-effect consequence.
5. Precommit rejects stale owner version/root.
6. Commit consumes the permit exactly once.
7. Owner applies every protected sub-effect atomically or none.
8. Owner performs no protected write outside the permitted effect set.
9. Graduation/persistence requires an independent authorization path.
10. Dream merge conservatively combines provenance, confidentiality, control integrity, artifact integrity and taint.
11. Active-state influence has an explicit mutation class rather than an approximate alias.
12. A causal-plan edge grants no authority by itself.
13. A child capability cannot broaden the parent into a stronger compound-effect class.
14. Replay/recovery cannot resurrect an already-consumed replacement permit.

## 20. Migration order

Do not modify frozen K0 merely to satisfy this design.

Recommended order:

1. obtain focused package evidence for frozen K0 source;
2. qualify #195 evidence/qualification substrate enough to trust mapped-stage findings;
3. freeze K0.1 typed effect semantics and canonical serialization/commitment rules;
4. add explicit active-state and working-memory replacement/merge taxonomy;
5. property-test effect closure and capability attenuation;
6. extend shadow events with exact-effect references/correspondence;
7. run deterministic audit-on/audit-off `LegacyBehaviorProjection` scenarios;
8. only then design first permit-consuming protected owner.

## 21. Constitutional summary

The exact-effect model extends Cognitive Non-Escalation to side effects:

> Authority for one named operation never implies authority for an uncommitted side effect.

And it extends the mediation theorem to compound state transitions:

> Every protected write is either part of the exact authorized effect on its owner boundary or is a separately mediated transition.
