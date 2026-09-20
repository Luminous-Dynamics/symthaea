# SCIP Semantic Review Profile v1

Status: research architecture + implementation candidate. This profile stacks directly on the v15 accepted-surface receipt boundary.

## Governing theorem

```text
surface accepted
    != semantic review structurally complete
    != semantic fidelity qualified
    != grounded truth
    != epistemic authority
```

V15 establishes an exact content identity for an accepted SCIP language surface without claiming that the surface preserved the grounded meaning. V16 makes the *review contract* exact before any semantic verifier is allowed to mint a stronger capability.

## Exact subject binding

A `SemanticReviewSubjectV1` can be constructed only when all of the following agree:

- the source `GroundedConceptGraph` recomputes to the semantic hash retained by the v15 receipt;
- the exact accepted UTF-8 surface recomputes to the v15 surface digest;
- the surface byte count equals the receipt-bound byte count;
- the receipt claim scope remains `SurfaceAcceptedOnly`.

This creates one exact review subject:

```text
canonical grounded source identity
+ accepted-surface identity
+ v15 realization-receipt identity
        -> SemanticReviewSubjectV1
```

The subject is an integrity binding, not a semantic-fidelity result.

## Closed review dimensions

V1 requires exactly one finding for each of twelve dimensions:

1. polarity / negation;
2. epistemic strength;
3. modal force;
4. attribution;
5. quantifier / cardinality;
6. temporal scope;
7. entity / reference identity;
8. causal force;
9. uncertainty qualification;
10. unsupported addition;
11. required-detail omission;
12. ambiguity introduction.

A finding reports one of:

- `Preserved`;
- `Violated`;
- `Inconclusive`;
- `NotApplicable`.

These are verifier-reported values. Structural validation does not establish that a `Preserved` or `NotApplicable` report is correct.

## Surface localization

Findings may bind zero or more exact UTF-8 byte spans in the accepted surface. Every supplied span must be non-empty, in bounds, and begin/end on UTF-8 character boundaries.

Zero spans are allowed because an omission can concern text that is absent by definition.

Every finding also carries a bounded non-empty source anchor. The anchor is descriptive verifier metadata, not proof that the verifier interpreted the grounded graph correctly.

## Structurally validated report

`StructurallyValidatedSemanticReviewV1` establishes only that:

- the review is bound to the exact v15 realization receipt;
- the exact accepted surface is still present;
- all twelve dimensions occur exactly once;
- source anchors and surface spans satisfy the profile's structural constraints;
- findings are canonicalized by semantic dimension;
- a deterministic content digest binds the subject, verifier labels, and complete finding set.

Its positive claim scope is therefore exactly:

```text
ReportStructureOnly
```

The output deliberately exposes:

```text
semantic_fidelity_established() == false
verifier_authenticated() == false
verifier_independence_established() == false
```

A report with twelve `Preserved` findings is still only a structurally valid report saying that a verifier reported preservation.

## Verifier identity boundary

V1 binds a `verifier_profile` and `verifier_run_id` as ordinary metadata. They are not signatures, runtime attestations, model-weight identities, organizational-independence evidence, or proof that a named verifier actually ran.

Therefore:

```text
bound verifier label
    != authenticated verifier
    != independent verifier
    != qualified verifier
```

Broca must not grade itself and then reinterpret that self-report as independent semantic authority.

## Adversarial semantic corpus

The crate freezes an initial falsification corpus containing mutations such as:

```text
"may"                         -> "will"
"evidence suggests"           -> "evidence proves"
"Alice reported X"            -> "X"
"some"                        -> "all"
"3 of 10"                     -> "10 of 10"
"during the test window"      -> unqualified present-tense claim
"Sensor A"                    -> "Sensor B"
"associated with"             -> "caused"
"10 ± 2"                      -> "10"
checksum result                -> checksum + deployment-safety claim
pass + failure                 -> pass only
explicit entities              -> ambiguous pronouns
```

The corpus covers every required dimension and expects the mutation to be reported as `Violated`.

Passing this finite corpus will remain falsification evidence only. It cannot establish general semantic equivalence.

## Why v16 stops here

The tempting implementation would be:

```text
Broca/LLM generates text
        -> another model says "looks faithful"
        -> boolean faithful=true
```

That would collapse generation, grading, verifier identity, and semantic authority into one weak boundary.

Instead the intended sequence is:

```text
canonical grounded source
        -> accepted Broca surface
        -> v15 SurfaceAcceptedOnly receipt
        -> v16 exact semantic-review subject + complete report contract
        -> later independently qualified verifier execution
        -> separate typed semantic-fidelity capability
```

The later verifier-execution tranche should bind at least:

- exact verifier implementation/artifact identity;
- exact verification profile and corpus revision;
- exact input subject and surface identities;
- execution/result evidence;
- explicit independence assumptions or evidence;
- fail-closed handling of `Inconclusive`;
- cross-verifier disagreement rather than silent majority laundering.

A stronger capability must be a new type. V15 or V16 must never acquire semantic authority by adding a boolean.

## Non-claims

V16 does not establish semantic equivalence, truth, attribution correctness, causal correctness, verifier competence, verifier authentication, verifier independence, generalization beyond its adversarial corpus, user authorization, action authority, or production routing eligibility.

The canonical grounded graph remains the semantic source of record. Native Broca/SSM direct-neural paths remain untouched.
