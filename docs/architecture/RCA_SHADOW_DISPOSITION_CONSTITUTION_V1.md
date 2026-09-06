# RCA Shadow Disposition Constitution v1

Status: **normative pre-implementation contract**

This document defines the boundary for the first RCA shadow disposition policy. It exists to prevent a coherent evidence case from silently becoming a belief engine.

The first implementation is intentionally restricted to **instrumented Symthaea runtime propositions** represented by provenance-bound RCA shadow cases. It is not a universal epistemic policy and does not create canonical belief, workspace, action, or self-improvement authority.

## 1. Core theorem

```text
BoundShadowEvidenceCaseV1
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

A disposition is an explicitly qualified **shadow interpretation of one exact content-addressed case under one exact preregistered policy**.

## 2. Required input class

RCA-003b may consume only:

```text
BoundShadowEvidenceCaseV1
```

from `symthaea-rca-bound-shadow-case`.

It must not accept:

- raw `ShadowEvidenceCaseV1`;
- arbitrary candidate/relation arrays;
- caller-supplied independence counts;
- caller-supplied relevance reports;
- producer relation ids as provenance authority;
- persisted disposition bytes as live authority.

The bound case already commits the proposition, relevance context, candidate/claim identities, evidence-root topology, relation declaration provenance, relation strengths, relevance defects, independence topology, and declared relation topology.

## 3. Evidence independence is not interpretation independence

RCA v1 distinguishes two independent causal questions.

### 3.1 Evidence-root independence

```text
Did these observations arise from independent evidence roots?
```

This is answered only by the existing lineage graph.

Multiple fields from one frozen cycle share one observation-event root and are not independent evidence.

### 3.2 Interpretation-root independence

```text
Were these evidence → proposition relation judgments produced by independent interpretation roots?
```

This is a separate graph.

Five independent observations all labeled `Supports` by the same rule/model/version are:

```text
5 independent evidence roots
1 interpretation root
```

They must never be counted as five independent interpretive confirmations.

A future RCA-003b implementation therefore requires explicit relation-declaration interpretation lineage before any minimum-independent-support rule may be satisfied.

## 4. Relation provenance is not relation qualification

`BoundEvidenceRelationDeclarationV1` proves:

- who/what declared a relation;
- the declared method;
- immutable provenance artifact identity;
- the exact relation body;
- canonical declaration identity.

It does **not** prove the declarer is qualified for shadow disposition use.

A separate qualification artifact is required before a declaration may influence disposition.

Conceptually:

```text
relation declaration
        +
independent declarer qualification
        +
exact permitted use
        ->
DispositionEligibleRelationDeclarationV1
```

Qualification must bind at least:

- declarer id;
- declarer version when applicable;
- declaration method;
- qualification artifact/corpus identity;
- evaluator identity/version;
- proposition scope or exact proposition id;
- allowed relation kinds;
- validity/currentness boundary;
- explicit permitted use = `ShadowRuntimeDisposition`;
- qualification-policy contract identity.

A declaration producer may not be the sole authority that qualifies its own relation declarations for disposition use.

## 5. Opaque proposition hashes do not create universal semantics

The current shadow case binds an exact proposition digest. A digest proves identity, not semantic class.

RCA-003b v1 therefore operates only under a policy registered for:

```text
one exact proposition id
or
one separately qualified typed proposition scope
```

It must not infer proposition semantics from a digest and must not generalize a policy learned for one proposition to another opaque proposition id.

A future universal epistemic layer should introduce a canonical typed proposition/expression contract before cross-domain disposition policy is attempted.

## 6. Disposition policy must be preregistered

Disposition rules must be committed **before inspecting result-bearing cases**.

A `ShadowDispositionPolicyV1` or equivalent registered artifact must bind:

- exact disposition-policy contract/version;
- permitted case profile;
- exact proposition id or qualified proposition scope;
- declarer-qualification profile;
- interpretation-lineage policy;
- minimum evidence-root requirements by outcome;
- minimum interpretation-root requirements by outcome;
- treatment of relation strengths;
- defeater precedence rules;
- opposition/contestation rules;
- underdetermination/abstention rules;
- OOD/fail-closed rules;
- evaluation corpus/seed/metric identity when experimental;
- resource ceilings when applicable.

Post-result threshold changes require a new policy identity and a new experimental lineage. They do not rewrite the previous result.

## 7. Relation strength is not probability

`EvidenceRelationV1::strength_ppm` remains a declared relation strength.

It is not automatically:

- a probability;
- a likelihood ratio;
- a calibrated confidence;
- additive evidence weight;
- a voting weight.

RCA-003b v1 must not sum, average, normalize, multiply, or Bayesian-update declared strengths unless the exact declarer/strength semantics have separately qualified calibration evidence and the policy explicitly enables that interpretation.

Default v1 policy should treat relation strength as preserved diagnostic metadata rather than an arithmetic evidence score.

## 8. No candidate-count voting

The following are prohibited disposition rules:

```text
number of Supports > number of Contradicts
mean support strength > threshold
sum support strength > sum opposition strength
most modules agree
```

unless an independently qualified policy explicitly proves why those operations are valid for the exact evidence and interpretation lineage.

Module count is never a substitute for root independence.

## 9. Defeaters are qualified blockers, not truth oracles

A declaration labeled `Defeats` may block a positive disposition only when all of the following hold:

```text
current-runtime relevant
AND exact-case joined
AND declaration provenance bound
AND declarer qualified for Defeats on this proposition/use
AND interpretation-lineage requirements satisfied
```

An unqualified or stale `Defeats` label cannot veto the case.

A qualified defeater means **the positive support case is defeated under this policy**. It does not, by itself, establish the negation of the proposition as canonical truth.

## 10. First shadow disposition vocabulary

The first engine may emit only a closed diagnostic vocabulary such as:

```text
NoRelevantEvidence
UnqualifiedInterpretation
Underdetermined
TentativelySupported
Supported
TentativelyOpposed
Opposed
Contested
Defeated
```

These terms are shadow-policy outputs only.

### 10.1 `NoRelevantEvidence`

No candidate is currently relevant under the exact bound case context.

### 10.2 `UnqualifiedInterpretation`

Relevant evidence exists, but no adequate relation declaration is qualified for disposition use.

### 10.3 `Underdetermined`

Qualified relevant relations exist, but preregistered evidence-root, interpretation-root, coverage, or policy requirements are insufficient to prefer support or opposition.

### 10.4 `TentativelySupported`

Qualified support exists and no qualified opposition/defeater blocks it, but the preregistered `Supported` qualification threshold is not met.

### 10.5 `Supported`

Only when the preregistered support policy is satisfied by qualified, current, correctly joined evidence and interpretation roots, with no blocking qualified opposition/defeater.

### 10.6 `TentativelyOpposed` / `Opposed`

Symmetric shadow diagnostics for qualified opposition under preregistered policy. `Opposed` does not mean the proposition's negation is canonically true.

### 10.7 `Contested`

Qualified current support and qualified current opposition both survive the policy. Disagreement is preserved rather than collapsed into a scalar.

### 10.8 `Defeated`

At least one policy-qualified defeater blocks positive support under the exact preregistered rules. This is a defeat of the support case, not automatic proof of the negation.

## 11. Abstention/underdetermination is a first-class success state

The engine must never force every case onto a support/opposition axis.

When qualification, currentness, root independence, interpretation independence, proposition scope, or policy requirements are insufficient, the correct result is an explicit abstaining/underdetermined state.

```text
unknown != 0.5 confidence
insufficient qualification != weak support
missing independence != one vote
```

## 12. Evidence-root and interpretation-root thresholds are policy, not constants

There is no universal hardcoded rule such as "two independent sources means Supported."

Different proposition classes may require different evidence structures. A formal derivation, direct instrument reading, noisy behavioral signal, and external retrieved claim do not justify identical thresholds.

The exact thresholds must live in the preregistered policy and participate in its identity.

## 13. Interpretation-lineage requirements

Before RCA-003b can count distinct interpretation roots, the architecture must represent how declarations relate causally.

At minimum, declarations produced by the same exact:

```text
declarer id
+ declarer version
+ declaration method
+ shared rule/model/procedure lineage
```

must not silently count as independent interpretive confirmation.

If interpretation lineage is unknown, independence fails closed as unknown/not-established rather than independent.

## 14. Role separation

The following conceptual roles remain distinct:

```text
Evidence Producer
Relation Declarer
Declarer Qualifier
Disposition Evaluator
Canonical Belief Admitter        [future]
Action Authorizer                 [outside RCA epistemics]
Self-Improvement Promoter         [outside RCA epistemics]
```

One component may perform multiple non-conflicting engineering roles, but no artifact gains authority merely because the same producer generated both the claim and the qualification that says its interpretation is correct.

## 15. Disposition identity

A future issued `ShadowDispositionV1` must be content-addressed by at least:

```text
disposition_id = H(
    disposition contract identity,
    exact BoundShadowEvidenceCaseV1.case_id,
    exact preregistered policy id,
    exact declarer-qualification-set id,
    exact interpretation-lineage identity,
    emitted disposition,
    machine-readable reasons
)
```

The issued result should have private fields and no `Deserialize` implementation. Archived bytes are audit material; trusted disposition must be recomputed from revalidated inputs.

## 16. Required reason trace

Every disposition must retain machine-readable reasons sufficient to answer:

- which evidence roots were current and eligible;
- which relation declarations were qualified/unqualified and why;
- which interpretation roots were established or shared;
- which candidate relations supported/opposed/defeated;
- which preregistered thresholds were met or missed;
- which defeater, if any, blocked support;
- why the engine abstained or declared underdetermination;
- the exact case/policy/qualification identities used.

A bare enum without a reason trace is insufficient.

## 17. No canonical belief admission

Even a `Supported` shadow disposition does not create a canonical epistemic state.

The future chain remains:

```text
BoundShadowEvidenceCaseV1
        ↓
qualified ShadowDispositionV1
        ↓
[separate future canonical epistemic admission policy]
        ↓
CanonicalEpistemicClaimV1
```

No RCA-003b API may directly mutate or create canonical belief/workspace state.

## 18. No action authority

No shadow disposition grants tool, motor, network, filesystem, Xenia, execution-fence, or external-effect authority.

The action chain remains separately governed:

```text
cognitive/epistemic result
        !=
action authority
```

## 19. No recursive-improvement promotion authority

Shadow disposition success cannot qualify or promote a new architecture.

Recursive-improvement promotion remains a separate independently qualified chain with preregistered experiment contracts, evaluator/adversary separation, held-out evidence, and external promotion authorization.

## 20. Scope limitation

RCA-003b v1 is an **instrumented-runtime shadow research policy**.

It does not establish:

- a universal theory of truth;
- universal Bayesian semantics;
- phenomenal self-knowledge;
- global canonical belief admission;
- AGI or consciousness;
- live recursive self-improvement.

Its purpose is narrower and testable:

> Can Symthaea interpret its own provenance-bound runtime evidence cases more reliably while preserving uncertainty, disagreement, causal independence, and authority boundaries?

## 21. Dependency order before the first engine

The recommended implementation order is:

```text
RCA-003b.0  Shadow Disposition Constitution        [this contract]
RCA-003b.1  Relation-declarer qualification
RCA-003b.2  Interpretation lineage + independence
RCA-003b.3  Preregistered disposition-policy type
RCA-003b.4  Pure shadow disposition engine
RCA-003b.5  Qualification benchmark / adversarial cases
```

Do not skip directly from provenance-bound case identity to a live disposition engine.

## 22. Qualification targets

Before any future RCA-003b engine is considered qualified, tests should include at least:

- ten candidate fields from one observation still count as one evidence root;
- ten independent observations interpreted by one rule remain one interpretation root;
- same evidence + independently qualified distinct interpretation roots remains distinct from independent evidence;
- unqualified `Supports` does not contribute to `Supported`;
- unqualified `Defeats` cannot veto;
- stale evidence cannot contribute to a current disposition;
- mixed qualified support/opposition becomes `Contested`, not an averaged score;
- insufficient roots become `Underdetermined`, not weak support;
- changing policy thresholds after seeing a case creates a different policy/result lineage;
- changing declarer qualification changes disposition identity;
- changing case identity changes disposition identity;
- persistence cannot recreate a live trusted disposition;
- no disposition has a path into GWT, canonical belief, action, or promotion authority.

## 23. Constitutional summary

```text
case coherence
        !=
relation provenance
        !=
relation qualification
        !=
interpretation independence
        !=
disposition
        !=
belief
        !=
action
        !=
self-improvement promotion
```

The design goal is not to make Symthaea more certain.

The design goal is to make it **harder for Symthaea to manufacture certainty from correlated evidence, correlated interpretation, stale state, unqualified declarations, or post-hoc policy changes**.
