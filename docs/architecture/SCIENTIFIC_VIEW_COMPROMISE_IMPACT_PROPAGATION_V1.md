# Scientific View Compromise Impact Propagation v1

Status: hardening companion to the SCI-014 scientific-view currentness architecture in #908.

## 1. Purpose

The bootstrap-root and selection-policy lineage contracts define how administrative authority becomes current and how compromise/recovery can be represented without rewriting history.

The next missing boundary is impact propagation after a root, selection policy, administrator set, attestation profile, or head-selection execution lineage is later determined to be compromised or otherwise unqualified.

A compromise event must not produce either unsafe extreme:

```text
compromised administrative authority
    -> everything admitted during the interval remains current
```

or:

```text
compromised administrative authority
    -> every scientific proposition in the affected view is false
```

Freeze instead:

```text
administrative compromise
    -> qualified authority-dependency impact analysis
    -> affected currentness lineages quarantined/requalified
    -> scientific evidence retained for independent reconstruction
```

This is an authority/currentness impact theorem, not a scientific falsification theorem.

## 2. Core theorem

Preserve:

```text
administrative authority dependency
    != scientific evidence dependency
    != scientific argument relation
    != proposition truth
```

and:

```text
root/policy compromise
    != evidence artifact corruption by default

root/policy compromise
    != scientific claim falsification

recovery of administrative authority
    != automatic requalification of affected descendants

same scientific payload under clean lineage
    != same administrative currentness lineage
```

The system should invalidate or quarantine the authority-bearing paths that depended on compromised authority while retaining independently verifiable scientific material for replay and reconstruction.

## 3. Separate authority dependency graph

SCI-006's Evidence Dependency Graph answers questions such as:

- which studies share data;
- which estimates share assumptions;
- which results share models, estimators, transformations, or learned priors.

Compromise propagation needs a different graph, conceptually:

```text
ScientificAdministrativeAuthorityDependencyGraphV1
```

Its nodes may include:

```text
bootstrap trust roots
root transitions
selection-policy transitions/heads
view-state captures
view-state transitions
head-selection receipts
selected view heads
currentness witnesses
continuity receipts
administrative execution/attestation artifacts
```

Its edges answer:

> Which positive administrative/currentness capability depended on which earlier authority capability or execution lineage?

Do not overload evidence dependency with administrative trust semantics.

## 4. Compromise finding is itself qualified evidence

Do not allow a caller to submit:

```text
compromised = true
```

and invalidate a view.

A compromise finding should itself have exact identity and provenance, conceptually:

```text
QualifiedAdministrativeCompromiseFindingV1
```

binding at least:

```text
finding profile identity
exact affected authority subject
finding class
known or bounded compromise interval
supporting evidence/artifacts
adjudication policy/profile
finding implementation/execution lineage
information-availability cutoff
```

A finding may be revised, contested, superseded, or withdrawn through append-only adjudication semantics rather than mutable flags.

## 5. Compromise time is not one timestamp

Preserve at least these distinct temporal concepts when relevant:

```text
possible compromise start
possible compromise end
compromise detection time
finding qualification time
revocation/recovery effective time
information availability time
```

A key may have been compromised on day 1, discovered on day 20, and formally revoked on day 21.

Historical reconstruction at day 10 must not contain the future day-20 finding, while a present-day impact analysis may legitimately re-examine descendants created during the suspected day-1..day-20 interval.

Freeze:

```text
discovery time
    != compromise start
    != historical information availability
```

## 6. Unknown compromise interval fails conservatively

Sometimes the exact compromise start cannot be established.

Do not choose a convenient boundary silently.

Represent uncertainty explicitly, for example conceptually:

```text
ExactInterval
EarliestKnownSafeBoundary
OpenBeginning
OpenEnd
UnknownInterval
```

The exact taxonomy is deferred.

If the impact frontier cannot prove a descendant lies outside the compromised interval, currentness should remain quarantined/unknown rather than automatically unaffected.

## 7. Impact propagation follows qualified authority reachability

Given a qualified compromise finding for root R20, the impact engine should traverse only scientifically/currentness-material authority edges that depend on R20 or descendants whose authority was derived from R20 during the affected interval.

Conceptually:

```text
CompromiseFinding(R20)
        ↓
AuthorityDependencyProjection
        ↓
AffectedSelectionPolicyHeads
        ↓
AffectedHeadSelections
        ↓
AffectedScientificViewHeads
        ↓
AffectedCurrentnessWitnesses
```

The transitive closure is over administrative authority, not over proposition semantics.

## 8. Descendant quarantine is not automatic scientific invalidation

Suppose H18 was selected by policy P18, and P18 was bootstrapped by compromised root R17.

Then H18's currentness lineage may lose qualification.

That does not imply:

```text
all evidence in H18 is false
```

or:

```text
all propositions represented by H18 are refuted
```

Instead preserve the underlying artifacts and mark the affected view/currentness lineage for requalification.

A clean reconstruction may recover the same scientific disposition through a new administrative lineage.

## 9. Requalification under clean authority

A recovered administrative root does not simply restore old descendant capabilities.

Prefer:

```text
clean current root
    + clean current selection-policy head
    + retained scientific artifacts
        ↓
reconstruct coherent view state
        ↓
new view-state transition/head lineage
        ↓
replay/reconstruct disposition
        ↓
new currentness witness
```

If the scientific state is byte-for-byte identical to an affected prior snapshot, the new head/currentness identities still differ because the administrative lineage differs.

Freeze:

```text
same scientific payload
    != same authority lineage
```

## 10. Revalidation can preserve scientific conclusions without preserving old capabilities

A post-compromise reconstruction may reach the same proposition disposition as the affected view.

For example:

```text
old compromised-lineage disposition = SupportedWithinScope
new clean-lineage disposition        = SupportedWithinScope
```

This may establish semantic/evaluative agreement under the new qualified reconstruction.

It does not revive the old `CurrentWithinScientificViewDispositionV1` capability.

The old witness remains historical and stale/quarantined.

## 11. Administrative quarantine states

Avoid a universal `valid: bool`.

A future impact assessment may need bounded states conceptually like:

```text
UnaffectedByQualifiedImpactAnalysis
AffectedPendingRequalification
AffectedAndSuperseded
RequalifiedUnderNewAuthorityLineage
ImpactUnknown
ImpactContested
HistoricallyValidButNotCurrent
```

These are administrative/currentness dispositions, not proposition truth states.

The exact production enum is deferred.

## 12. Compromise propagation must be reason-preserving

A result such as:

```text
AffectedPendingRequalification
```

must retain why.

Conceptually a `ScientificAdministrativeImpactAssessmentV1` should bind:

```text
compromise finding identity
source authority subject
exact authority dependency graph/root
impact traversal profile
affected-node set
unaffected-node set where positively established
unresolved-node set
reason edges / propagation trace
information cutoff
impact implementation/execution lineage
```

Do not reduce blast radius to an unexplained list of invalid IDs.

## 13. Candidate-set closure matters for impact too

An impact engine cannot prove a complete blast radius from a caller-selected subset of descendants.

For an owner-local authority store, use a qualified closed descendant/index snapshot.

For federated views, use an explicit bounded discovery/frontier theorem.

Freeze:

```text
no affected descendant found in supplied subset
    != no affected descendant exists
```

Incomplete discovery yields `ImpactUnknown` or equivalent rather than a positive unaffected result.

## 14. Proving unaffected is stronger than failing to find a path

There is a difference between:

```text
path found from compromised root to object X
```

and:

```text
no material path exists from compromised root to X
```

The latter is a closed-world negative claim over the exact authority graph/frontier.

Reuse #848's negative-by-absence principle:

```text
not found
    != does not exist
```

A positive `UnaffectedByQualifiedImpactAnalysis` result therefore requires a sufficiently closed authority-dependency universe for the claimed scope.

## 15. Policy compromise vs root compromise

Different compromise classes have different frontiers.

Examples:

```text
bootstrap root compromise
    -> potentially policy bootstrap/rotation descendants

selection policy compromise
    -> head selections made under that policy interval

head-selection implementation compromise
    -> receipts produced by that implementation/execution lineage

capture verifier compromise
    -> state captures qualified through that verifier lineage
```

Do not inflate every finding to deployment-wide impact if the exact authority graph proves a narrower scope.

Likewise, do not artificially narrow the scope when transitive dependencies exist.

## 16. Execution implementation compromise is first-class

Administrative authority may be semantically correct while its verifier implementation is later found defective.

Example:

```text
policy P is correct
verifier implementation V incorrectly accepts invalid signatures
```

Affected positive capabilities depend on V's execution lineage even though P itself is not compromised.

The authority graph therefore needs implementation/execution dependencies where they are material.

Freeze:

```text
policy semantics uncompromised
    != verifier-produced capability remains qualified
```

## 17. Cryptographic algorithm/profile compromise

If a cryptographic profile is broken or found misimplemented, impact analysis should bind the exact scheme/profile/wire identity and the exact artifacts/capabilities that depended on it.

Do not invalidate unrelated algorithms by friendly-name similarity.

Do not preserve affected capabilities merely because their signatures still parse.

Migration to a replacement scheme creates new authority lineage; it does not rewrite historical signatures.

## 18. Partial compromise and scoped delegation

If a delegated administrator had authority only for one transition class, compromise should ordinarily propagate only through objects that depended on that delegated capability.

For example:

```text
delegate D may approve policy metadata updates
```

must not automatically taint root rotations that D had no authority to authorize.

This is another reason the authority graph should model exact capability scope, not just actor identity.

## 19. Cross-view / federation impact

A compromised root in view A does not automatically invalidate view B.

Impact crosses views only through explicit qualified bridge/import/delegation dependencies.

If B imported A's selected-head/currentness capability as an administrative dependency, the bridge becomes part of the impact frontier.

If B independently reconstructed the same scientific evidence without depending on A's administrative lineage, B's scientific result may remain unaffected.

Freeze:

```text
same scientific evidence
    != shared administrative compromise lineage
```

## 20. Compromise finding itself can be contested

Scientific/administrative findings can be wrong.

A compromise allegation should therefore not erase lineage immediately through mutable deletion.

Use append-only adjudication with bounded outcomes such as:

```text
FindingQualified
FindingContested
FindingSuperseded
FindingWithdrawn
FindingScopeNarrowed
FindingScopeExpanded
```

The exact semantics belong to the adjudication layer.

Currentness policy may fail closed while a high-severity compromise finding is unresolved, but that is explicit policy behavior rather than hidden truth inference.

## 21. Emergency quarantine vs final adjudication

A deployment may need to stop issuing new currentness witnesses before a complete compromise investigation finishes.

Keep emergency administrative quarantine separate from final compromise adjudication.

Conceptually:

```text
EmergencyAuthorityQuarantineV1
```

may temporarily block affected transition/currentness issuance under an exact pre-existing emergency policy.

It must not:

- declare propositions false;
- erase evidence;
- retroactively rewrite history;
- become permanent compromise proof by mere duration.

Emergency policy must itself be pre-established and non-self-authorizing.

## 22. Recovery completion needs explicit exit criteria

Do not treat:

```text
new root installed
```

as sufficient to declare the view fully recovered.

Recovery may require:

```text
clean root current
clean selection policy current
compromised interval bounded/adjudicated
impact graph closed for required scope
affected heads quarantined
scientific state reconstructed
current dispositions replayed/reissued
external bridge consumers notified/requalified where required
```

The exact deployment profile may vary.

## 23. Historical state remains reconstructible

Even when an old head is later known to have depended on compromised authority, historical queries should be able to distinguish:

```text
what the view treated as current at time t
```

from:

```text
what a later impact analysis says about that historical authority lineage
```

Do not rewrite the historical selected-head record.

This allows honest retrospective questions such as:

> What did the Atlas present on date t, and what later compromise findings now qualify our trust in that presentation?

## 24. Interaction with replay/reconstruction

#825 and #839/#848 answer whether a disposition evaluation/reconstruction is scientifically replay-consistent with its exact scientific inputs.

This document answers whether the administrative/currentness lineage that presented that disposition was affected by a compromise finding.

Therefore:

```text
scientific replay still succeeds
+ administrative lineage compromised
```

is possible.

In that case the old currentness witness is not current/qualified, but the science may be reissued under clean administrative authority after reconstruction.

## 25. Interaction with #886 continuity

Material continuity must fail or suspend when either endpoint head lies inside an unresolved compromised administrative interval unless an explicit impact/requalification theorem clears the relevant dependency slice.

A continuity optimization cannot bypass compromise analysis.

Freeze:

```text
material scientific dependencies unchanged
    != administrative currentness preserved across compromised authority
```

## 26. Qualification vectors

A first executable compromise-impact qualification should prove at minimum:

1. caller-supplied `compromised=true` cannot mint a compromise finding;
2. compromise of root R affects descendant policy/head/currentness capabilities that depend on R;
3. unrelated authority branch remains unaffected only when closed graph analysis proves no dependency path;
4. scientific evidence artifacts are retained after administrative quarantine;
5. proposition truth/disposition is not inverted by administrative compromise;
6. recovery root installation does not automatically requalify old descendant witnesses;
7. identical scientific snapshot reconstructed under clean authority yields new administrative/currentness identity;
8. unknown compromise start yields conservative unresolved impact rather than guessed cutoff;
9. detection time is not substituted for compromise-start time;
10. implementation/verifier compromise propagates through capabilities produced by that lineage;
11. scoped delegate compromise does not taint capabilities outside its proven authority reach;
12. incomplete descendant discovery cannot prove an object unaffected;
13. cross-view impact requires explicit bridge/dependency lineage;
14. emergency quarantine cannot mint final compromise truth;
15. withdrawn/superseded compromise findings remain historical and trigger explicit re-evaluation rather than deletion;
16. continuity cannot bypass unresolved compromise impact;
17. historical Atlas state remains reconstructible with later compromise annotations kept separate;
18. recovered currentness never implies proposition truth, recommendation, governance, medical/resource, or physical-effect authority.

## 27. First implementation order

After lower SCI replay/reconstruction and #908 root/policy/head primitives qualify, prefer:

```text
1. pure administrative authority-dependency edge types
2. synthetic root -> policy -> head -> currentness fixture
3. qualified compromise finding fixture
4. deterministic affected-descendant traversal
5. private impact-assessment witness
6. quarantine-state projection
7. clean-lineage reconstruction/requalification fixture
8. unknown-interval + closed-world unaffected negatives
9. cross-view bridge impact later
10. emergency quarantine/recovery orchestration later
```

Do not begin with global revocation broadcasting or automatically deleting affected scientific state.

## 28. Exit gate

This contract is complete when the repository can truthfully state:

> A qualified compromise of scientific-view administrative authority propagates only through exact authority/currentness dependencies that actually relied on the affected root, policy, verifier, or transition interval. Affected live capabilities are quarantined or requalified under a clean lineage, while underlying scientific evidence and historical state remain available for independent reconstruction. Administrative compromise never directly falsifies a scientific proposition.

This is a compromise-impact/currentness theorem, not a truth theorem.
