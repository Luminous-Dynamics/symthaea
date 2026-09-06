# Symthaea RCA Shadow Disposition Preflight

RCA-003b.3b binds all currently issued shadow-disposition inputs into **one exact evaluation scope before any disposition algorithm exists**.

## Core theorem

```text
individually valid artifacts
        !=
a valid joint evaluation input
```

A case, evidence witness, interpretation lineage, interpretation witness, policy, and experiment contract may each be valid independently while belonging to different propositions, currentness contexts, qualification generations, or evidence selections.

Preflight exists to make that splice impossible.

## Inputs

`preflight_shadow_disposition_inputs_v1(...)` accepts exactly:

```text
BoundShadowEvidenceCaseV1
+
3 fixed evidence-witness slots:
    support / opposition / defeater
+
exact current DispositionEligibleRelationDeclarationV1[]
+
InterpretationLineageV1
+
3 fixed interpretation-witness slots:
    support / opposition / defeater
+
RegisteredEffectiveShadowDispositionPolicyV1
+
RegisteredExperimentContractV1
```

The fixed slots are deliberate. An arbitrary witness list would create a new unregistered resource/selection dimension. The same exact witness may appear in more than one compatible slot; reuse does not multiply evidence because identity is preserved.

## Exact case / eligibility / lineage join

Preflight requires:

```text
case declaration ids
        ==
current eligible declaration ids
        ==
interpretation-lineage declaration ids
```

and each interpretation-lineage entry must bind the exact eligibility id supplied for that declaration.

All eligible declarations must also share the exact eligibility-context commitment carried by the interpretation lineage.

Because relation eligibility excludes `Supersedes` as a disposition relation, a bound case containing a `Supersedes` declaration cannot satisfy this exact eligibility coverage. It fails closed rather than silently treating supersession as support/opposition evidence.

## Evidence witness binding

For every supplied evidence witness:

- its profile must equal the policy-bound evidence-witness profile;
- every selected item must be an exact bound-case candidate;
- every selected item must be current-runtime relevant in that case;
- every selected item must belong to the semantic slot;
- every selected item must bind the exact case observation root.

### Deliberately narrow V1 root model

Current instrumented runtime candidates have exactly one observation-event root. RCA-003b.3b therefore requires:

```text
witness item root set
        ==
{ case item observation_root_id }
```

Composite/multi-root evidence is **not** generalized into this profile. A future composite-evidence preflight must introduce a new explicit contract/profile rather than silently widening V1 semantics.

## Exact evidence ↔ interpretation co-grounding

This is the most important cross-artifact rule.

An interpretation witness may not merely contain roots that interpret *some* support/opposition evidence somewhere in the case.

For each semantic slot:

```text
slot evidence witness candidate ids
        ↓
exact bound-case declarations
        ↓
exact interpretation-lineage declaration → root mapping
        ↓
canonical expected interpretation-root set
```

must equal:

```text
slot interpretation witness root set
```

Therefore this splice fails:

```text
independent evidence set A
+
independent interpreters of unrelated evidence set B
```

If an interpretation witness is supplied without a corresponding evidence witness in the same slot, preflight fails closed.

An evidence witness without an interpretation witness is allowed; it simply cannot satisfy a later interpretation-root threshold and therefore remains underdetermined at the future disposition stage.

## Semantic slots

V1 freezes:

```text
support:
    Supports | Corroborates

opposition:
    Contradicts | Weakens | Defeats

defeater:
    Defeats only
```

`Corroborates` remains only a relation label; evidence independence is still proven separately by the evidence-set witness.

## Policy and experiment joins

Preflight verifies:

- exact proposition equality;
- exact bound-case profile;
- exact interpretation-lineage profile;
- exact evidence-witness profile per supplied slot;
- exact interpretation-witness profile from the effective policy;
- exact interpretation-lineage id inside every interpretation witness;
- actual `RegisteredExperimentContractV1::verify_integrity()`;
- actual experiment-contract digest equals the policy preregistration binding.

A digest-shaped experiment reference is not enough: the actual registered artifact must be present and verify.

## Runtime resource ceilings

Policy registration proves its thresholds are *theoretically* feasible. Preflight separately verifies the actual evaluation input obeys the registered ceilings:

```text
case.items.len() <= policy.max_case_items
interpretation_lineage.root_pair_assessments.len()
    <= policy.max_interpretation_pairs
```

Thus a caller cannot register a bounded policy and then evaluate an oversized case.

## Zero-witness preflight

All six witness slots may be empty. This is valid because preflight proves **input coherence**, not epistemic sufficiency.

```text
valid preflight
        !=
threshold satisfied
```

A later pure disposition engine must compare actual issued witness cardinalities to the preregistered policy. With no witnesses, that engine should remain underdetermined under V1 policy rather than treating preflight success as support.

## Issued capability

`ShadowDispositionPreflightV1` binds:

- proposition id;
- bound case id and case-scope digest;
- case evidence-lineage graph id;
- exact relation-eligibility context commitment;
- six optional witness ids by semantic slot;
- canonical declaration → eligibility bindings;
- interpretation-lineage id;
- effective policy id;
- actual registered experiment-contract digest;
- preflight profile/schema.

The complete binding receives a serializer-independent domain-separated BLAKE3 `preflight_id`.

The issued type has private fields and no `Deserialize` implementation. Archived bytes cannot recreate current evaluation eligibility.

## Non-scope

This crate intentionally does **not**:

- compare witness cardinalities with policy thresholds;
- choose an independent evidence set;
- discover an interpretation clique;
- aggregate relation strengths;
- compute a posterior/confidence;
- emit `Supported`, `Contested`, `Defeated`, or any other disposition;
- admit canonical belief;
- modify workspace/GWT state;
- authorize an external action;
- authorize recursive self-improvement promotion.

```text
ShadowDispositionPreflightV1
        !=
ShadowDispositionV1
        !=
canonical epistemic state
        !=
workspace/GWT authority
        !=
action authority
        !=
self-improvement promotion
```
