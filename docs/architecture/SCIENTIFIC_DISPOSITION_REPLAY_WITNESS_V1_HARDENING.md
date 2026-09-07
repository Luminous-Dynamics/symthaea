# Scientific Disposition Replay Witness v1 — view-scoped currentness hardening

**Status:** semantic hardening companion to `SCIENTIFIC_DISPOSITION_REPLAY_WITNESS_V1.md`.

**Parent:** `#783@16febe6615072f9a1a3d07560c66e53845a2f04b`

## 1. Correction: current scientific state is view-relative

The parent replay contract uses the phrase `AuthoritativeScientificStateSnapshotV1` to describe the source of current generation/head information.

That phrase must not be read as:

```text
one registry is authoritative over scientific truth
```

or:

```text
one current head represents all science everywhere
```

The stronger theorem is:

```text
registry authority over one declared view
    != authority over proposition truth
    != global evidence completeness
    != universal scientific consensus
```

A registry or Atlas service may be authoritative only for questions such as:

> What is the exact current state of this declared scientific view/namespace according to this registered source set and policy?

It is not authoritative for:

> What is ultimately true?

---

## 2. Prefer view-scoped naming

The eventual positive type should avoid implying global currentness.

Prefer a name conceptually like:

```text
CurrentWithinScientificViewDispositionV1
```

or:

```text
ViewCurrentScientificDispositionWitnessV1
```

over a bare global-sounding:

```text
CurrentScientificDispositionWitnessV1
```

unless the latter always carries an unavoidable exact view namespace/profile in its type or fields.

Likewise, the current-state input is more accurately:

```text
QualifiedScientificViewStateV1
```

than an unqualified global scientific state.

---

## 3. Scientific view identity

A currentness-capable scientific view should bind at least:

```text
ScientificViewProfileV1 {
    view_namespace,
    view_semantics_commitment,
    registry/source-set identity,
    discovery/query policy identity,
    historical/current availability policy,
    proposition namespace/profile set,
    lifecycle/adjudication registry identities,
    dependency/triangulation registry identities,
    disposition-policy registry identity,
    evaluator registry identity,
}
```

The exact Rust shape is deferred.

The view identity itself must be immutable/content-bound strongly enough that changing what counts as part of the view cannot preserve the same semantic identity by friendly-name reuse.

Therefore:

```text
view label != view semantics identity
```

and:

```text
same registry URL != same scientific view semantics
```

if the source set, inclusion policy, or interpretation rules changed.

---

## 4. Currentness means current within one exact view

The positive theorem is:

> This disposition faithfully replayed and matches the current registered scientific-state snapshot for view V, use U, and exact state head H.

It does not mean:

> This is the globally current scientific conclusion.

Formally:

```text
CurrentWithin(V, H, P, U)
    != GloballyCurrent(P)
    != True(P)
```

This makes the witness composable across:

- institutional registries;
- independent replication networks;
- domain-specific observatories;
- public evidence mirrors;
- historical corpora;
- federated Mycelix scientific views;
- local research workspaces.

---

## 5. Multiple current views may legitimately disagree

Suppose two qualified views exist:

```text
V_A @ H17 -> SupportedWithinScope
V_B @ H42 -> Contested
```

This is not automatically a consistency failure.

The views may differ in:

- source coverage;
- historical cutoff;
- measurement-admission policy;
- evidence-use policy;
- domain scope;
- proposition schema/profile;
- current lifecycle/adjudication knowledge;
- disposition policy.

The Atlas should preserve the disagreement and permit an explicit **view comparison** rather than silently declaring one globally canonical.

A future `ScientificViewComparisonReceipt` may establish which differences explain the divergent dispositions.

---

## 6. Federation requires explicit mapping, not namespace erasure

If two views are federated, never infer:

```text
same proposition text
+ same friendly registry name
    -> same scientific view
```

Cross-view composition should require explicit namespace-qualified identities and mapping/equivalence evidence.

Conceptually:

```text
View A / Proposition P_A
View B / Proposition P_B
        ↓
qualified proposition-target compatibility
        +
qualified view/source compatibility
        ↓
CrossViewComparison
```

Currentness from view A cannot be transplanted into view B merely because the two views currently contain similar data.

---

## 7. Registry coherence authority is not evidentiary authority

A scientific registry may legitimately provide an exact current head and prove that a snapshot belongs to that head.

That establishes:

```text
coherence / currentness within registry view
```

not:

```text
scientific admissibility of every object
truth of every contribution
quality of every source
correctness of the disposition policy
```

Those remain separately qualified through #668/#701/#729/#769/#783 and domain-owned scientific admission policies.

Therefore the current-view witness must bind both:

```text
view-state/currentness provenance
```

and:

```text
replay-verified scientific reasoning
```

without allowing either to substitute for the other.

---

## 8. Root provenance and caller selection

The privileged view-current verifier must know which view-state root it is checking against.

Invalid shape:

```text
caller provides:
    assessment
    registry head H
    matching generation ids

verifier says current
```

when H has no independently established relationship to the requested view.

Preferred architecture:

```text
registered ScientificViewProfile
        ↓
owner-local / explicitly trusted view-state resolver
        ↓
QualifiedScientificViewState { view_id, head, generations }
        ↓
view-current disposition verifier
```

The view-state resolver may consume untrusted network/storage transport, but the final binding between view identity and state head must be verified at the owner boundary.

Do not create a generic public trait whose semantic promise alone allows downstream code to mint `QualifiedScientificViewState`.

---

## 9. Currentness snapshot must be immutable during use

A positive view-current witness should retain the exact immutable head/snapshot against which it was checked.

If the underlying registry advances:

```text
H17 -> H18
```

then a witness bound to H17 remains a valid statement about H17 but is no longer current for the live H18 view.

For safe point-of-use behavior:

```text
verify against H17
        ↓
operate/query only against H17 snapshot
```

or:

```text
verify against current H17
        ↓
re-check head still H17 at use
```

or use a transaction/snapshot lease that guarantees the visible state cannot change mid-operation.

This is epistemic TOCTOU prevention, not action authority.

---

## 10. Natural validity is state-bound, not necessarily time-bound

Scientific view currentness often changes because a head/generation advances, not because a fixed wall-clock expiry arrives.

Therefore a witness should not pretend that:

```text
valid_until = tomorrow
```

is sufficient currentness semantics.

The primary boundary is:

```text
valid while exact view-state head/generation still matches
```

A time horizon may additionally apply for source freshness or operational caching, but it cannot substitute for generation/head currentness.

---

## 11. Currentness unavailable is not opposition

If the view registry is unreachable or its head cannot be verified:

```text
CurrentnessUnavailable
```

is the correct epistemic result.

Do not map it to:

```text
OpposedWithinScope
Refuted
NoEvidence
```

Likewise, do not silently fall back to a cached `current=true` field.

A replay-verified record may still be shown with an explicit:

```text
last verified against V@H17
currentness now unavailable
```

status.

---

## 12. Evidence discovery completeness remains bounded

Even a perfectly current view is not omniscient.

The witness must retain the evidence-view/discovery-scope limitations from #783.

Thus:

```text
current within V
    != all relevant evidence in the universe was discovered
```

A view may be current and still carry:

```text
coverage limitations
unindexed sources
language exclusions
unresolved external repositories
pending replication feeds
```

These should remain visible reason/coverage metadata rather than being erased by currentness qualification.

---

## 13. No implicit consensus layer

If several qualified scientific views disagree, the shared kernel must not automatically majority-vote them into a global consensus.

Any cross-view synthesis requires another explicit scientific assessment with:

- exact participating view identities;
- proposition compatibility;
- evidence dependency across views;
- duplicated-source detection;
- policy compatibility;
- source-coverage differences;
- exact synthesis policy;
- unresolved disagreements.

Therefore:

```text
N_views supporting > N_views opposing
    != global scientific truth
```

---

## 14. Suggested type separation

A cleaner future type chain is:

```text
PersistedScientificDispositionRecordV1
        ↓ replay
ReplayVerifiedScientificDispositionV1
        ↓ bind exact registered view state
CurrentWithinScientificViewDispositionV1
```

with a separate serializable:

```text
ScientificViewStateRecordV1
```

and private/non-deserializable:

```text
QualifiedScientificViewStateV1
```

if the registry-verification layer needs an opaque positive capability.

This gives four distinct theorems:

```text
record identity
replay correctness
view-state qualification
current-within-view disposition
```

rather than one overloaded `verified/current` bit.

---

## 15. Negative qualification cases

Future implementation should prove at least:

1. same friendly view label with different semantics commitment does not preserve view identity;
2. caller-supplied matching heads cannot mint a qualified view state;
3. a replay-positive disposition cannot become view-current without a qualified exact view state;
4. currentness for view A cannot satisfy view B;
5. advancing H17 -> H18 makes an H17 witness stale for the live view without invalidating its historical statement;
6. registry unavailability returns currentness-unavailable rather than opposition or cached currentness;
7. two valid disagreeing views can coexist without forced global winner;
8. current-within-view never implies exhaustive evidence discovery;
9. current-within-view never implies proposition truth;
10. current-within-view never grants governance/action authority.

---

## 16. Refined non-equivalence theorem

The final boundary is:

```text
persisted record
    != replay-verified reasoning
    != qualified scientific view state
    != current-within-view disposition
    != cross-view consensus
    != truth
    != action authority
```

This preserves a federated scientific architecture while still giving each declared Atlas view strong replay/currentness semantics.
