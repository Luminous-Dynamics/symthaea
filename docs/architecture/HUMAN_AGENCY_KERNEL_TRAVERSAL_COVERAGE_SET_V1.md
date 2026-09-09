# HAK-018b — Traversal Coverage-Set Conservation v1

Status: candidate audit/evidence tooling. No runtime authority changes.

## Purpose

HAK-016 makes selector coverage precise inside one retained HAK-014 normalization receipt. HAK-017 makes collection-completeness declarations precise. HAK-018a verifies deterministic structure and deduplication over one retained traversal transcript.

None of those facts alone establishes that a selector was assessed across every page contributing to the retained traversal.

The HAK-018b boundary is:

```text
CoverageSourceIncludedInTraversal
!=
CoverageExhaustsTraversalPopulation
```

and, more specifically:

```text
CoverageSetConservedAcrossRetainedTraversal
!=
ProviderPopulationQualifiedCoverage
```

HAK-018b therefore accounts for every page in the exact HAK-018a retained traversal and derives selector coverage across those retained pages without upgrading the result into a provider-population claim.

## Artifact

The coverage-set record is:

```text
hak.traversal-coverage-set.v1
```

implemented by:

```text
scripts/hak_traversal_coverage_set.py
```

with schema:

```text
docs/architecture/hak/traversal-coverage-set-v1.schema.json
```

The record is canonicalized with the exact inherited HAK-015 profile and digest domain:

```text
hak.traversal-coverage-set.v1
```

## Page-disposition conservation

Let `P` be the exact ordered page set in the validated HAK-018a retained traversal.

HAK-018b requires:

```text
forall p in P:
    ExactlyOneDisposition(p)
```

where v1 dispositions are:

```text
Assessed
Unassessed(reason)
```

The witness page-ID set must exactly equal the traversal page-ID set.

Therefore:

```text
NoMissingPage
NoDuplicatePageWitness
NoExtraPageWitness
```

An `Unassessed` page is not absence of evidence metadata. It is an explicit first-class record with a non-empty reason.

```text
MissingWitness
!=
Unassessed(reason)
```

An `Unassessed` witness may not carry receipt, coverage, or profile-binding artifacts.

```text
Unassessed
!=
AssessedButHidden
```

## Exact assessed-page join

An `Assessed` page must provide exactly:

```text
page_id
HAK-014 receipt
HAK-016 coverage record
HAK-016 canonical-profile binding
```

HAK-018b validates the HAK-014 receipt, validates that the HAK-016 record is the deterministic derivation of that receipt, and validates the HAK-016 profile-binding artifact.

The receipt is then joined to the exact HAK-018a page:

```text
HAK014.raw_source.source_ref
==
TraversalPage.response_ref
```

and:

```text
HAK014.raw_source.raw_response_digest
==
TraversalPage.raw_response_digest
```

The typed resource domain must also match:

```text
HAK014.resource_kind
==
HAK016.resource_kind
==
Traversal.resource_kind
```

Thus:

```text
SameBytes
!=
SameTypedResourceDomain
```

## Exact selector result

For the HAK-018b selector path, every assessed HAK-016 page must contain exactly one optional-selector result for that path.

```text
ZeroSelectorResults -> reject Assessed
MultipleSelectorResults -> reject Assessed
```

The page result retains HAK-016's existing state/count semantics:

```text
Present
PartiallyPresent
Absent
NotApplicable
Failed
```

HAK-018b does not redefine HAK-016 state meaning.

## HAK-015 profile binding

A coverage record naming:

```text
hak.canonical-json.v1
```

is not sufficient on its own.

Each assessed page must include the exact HAK-016 adjacent profile-binding artifact that binds the HAK-016 coverage digest to:

```text
exact HAK-015 Git profile artifact ref
+
raw SHA-256 of the profile bytes
```

Therefore:

```text
ProfileName
!=
ProfileArtifactIdentity
```

and profile-binding substitution is rejected.

## Cross-page normalization-policy identity

All assessed pages must agree on the recorded normalization-policy identity:

```text
policy_id
policy_digest
artifact_ref
```

The `artifact_ref` must be an exact Git artifact reference.

This prevents two assessed pages from silently using different recorded normalization-policy lineages while being aggregated as one semantic coverage set.

However:

```text
NormalizationPolicyRef/DigestAgreement
!=
NormalizationPolicyContentVerified
```

HAK-018b does not independently resolve the policy artifact or prove that the policy bytes match the recorded digest/reference.

The machine-visible boundary is:

```text
normalization_policy_content_verification = NotEstablished
```

## HAK-014 input-replay boundary

HAK-018b validates the HAK-014 receipt and validates HAK-016's deterministic derivation from that receipt.

It does not possess the exact raw provider response bytes as an input to HAK-014 replay.

Therefore:

```text
CoverageDerivedFromReceipt
!=
NormalizationReceiptReplayedAgainstRawBytes
```

The record fixes:

```text
normalization_input_replay = NotEstablished
```

A stronger future consumer must supply raw bytes, exact HAK-013 policy content, and exact HAK-014 interpreter identity/bytes to invoke HAK-014's replay join.

## Aggregation semantics

For every assessed non-failed page, HAK-018b sums:

```text
applicable
matches
missing
```

and enforces:

```text
aggregate.applicable
==
aggregate.matches + aggregate.missing
```

The retained-traversal aggregate is derived as follows:

```text
if any page is Unassessed:
    Indeterminate
else if any assessed page is Failed:
    Failed
else if aggregate.applicable == 0:
    NotApplicable
else if aggregate.matches == aggregate.applicable:
    Present
else if aggregate.matches == 0:
    Absent
else:
    PartiallyPresent
```

This deliberately makes denominator uncertainty dominate reassuring partial evidence.

```text
OneUnassessedPage
+
AllObservedPagesPresent
->
Indeterminate
```

not:

```text
Present
```

## Retained traversal scope

HAK-018b fixes:

```text
coverage_scope = VerifiedRetainedTraversalPagesOnly
```

This says exactly what is conserved: coverage dispositions across the pages in the exact HAK-018a retained transcript.

It does not establish that HAK-018a's transcript is provider-authenticated or that it represents the complete real provider population.

## Population anti-oracle boundary

The record fixes:

```text
population_qualification = NotEstablished
```

Even when:

```text
HAK017 = DeclaredComplete
HAK018a = VerifiedAgainstRetainedTranscript
HAK018b aggregate = Present
```

HAK-018b still does not emit:

```text
ProviderPopulationPresent
```

because HAK-018a intentionally leaves provider/raw-content/continuation provenance unverified, and HAK-017 intentionally leaves actual exhaustion verification outside its boundary.

```text
DeclaredCollectionComplete
+
RetainedTraversalVerified
+
RetainedCoverageSetPresent
!=
VerifiedProviderPopulationPresent
```

## Source-assurance preservation

HAK-018b copies the exact HAK-018a source-assurance boundaries:

```text
provider_authentication = NotEstablished
raw_response_content_verification = NotEstablished
entity_projection_verification = NotEstablished
continuation_source_verification = NotEstablished
temporal_snapshot_verification = NotEstablished
```

None may be promoted by HAK-018b.

Therefore:

```text
RetainedTranscriptComposition
!=
ProviderSourceAuthentication
```

and:

```text
AllPagesObservedSequentially
!=
OneStableSnapshot
```

## Canonical identity

The coverage-set digest uses the exact inherited HAK-015 canonicalization profile:

```text
profile_id:
    hak.canonical-json.v1

artifact_ref:
    git:Luminous-Dynamics/symthaea@cf440ce4f813bb30a6b1948e5caac96feda10610:docs/architecture/hak/canonical-json-v1.profile.json

raw_sha256:
    sha256:f76152db3cce567bb4d24ca46b5ed94d95db698daa8882a6fcdb38828d40e830
```

The digest domain is:

```text
hak.traversal-coverage-set.v1
```

This does not upgrade HAK-015's evidence tier.

## Qualification contract

HAK-018b preregisters an E5-target tooling claim before the focused workflow runs.

Plan:

```text
hak018b-traversal-coverage-set-e5-v1
```

Plan digest:

```text
sha256:ddecff671c320dfb0e3d20b018a088b73473d642f01a1f79557244fc0f36eba7
```

HAK-010 obligation-binding policy:

```text
hak018b-traversal-coverage-set-binding-policy-v1
```

Policy digest:

```text
sha256:23ff73519b100dc2c34d79cdc614e6b9de55dbb55be41d87a8a832f19f665912
```

The exact hosted step bindings are:

```text
3  Assert exact-head checkout
6  Compile HAK-018b tooling
7  Validate HAK-018b schema syntax
8  Lint HAK-018b qualification plan
9  Lint HAK-018b check-binding policy
10 Run HAK-018b regressions
```

A green run can qualify only the bounded tooling contract at the exact subject.

```text
HAK018bE5
!=
HAK018aE5
!=
HAK017E5
!=
ProviderPopulationTruth
```

## Adversarial requirements

The preregistered negative surface includes:

- traversal page omitted from the coverage set;
- duplicate page witness;
- unassessed page smuggling assessed artifacts;
- valid HAK-014 receipt from a different source ref;
- valid HAK-014 receipt with a different raw-response digest;
- HAK-014/016 resource domain differing from traversal domain;
- HAK-016 profile-binding substitution;
- selector missing from an assessed page;
- cross-page normalization-policy artifact-ref mismatch;
- coherently redigested `Indeterminate -> Present` promotion;
- provider-population qualification promotion;
- HAK-014 input-replay promotion;
- normalization-policy content-verification promotion;
- source-assurance promotion;
- schema root-field expansion;
- unassessed schema smuggling;
- missing assessed profile-binding digest;
- unauthorized aggregate-state expansion.

## What HAK-018b proves

Given exact HAK-018a / HAK-017 witnesses and exact per-page HAK-014 / HAK-016 / profile-binding witnesses, HAK-018b can establish that:

1. every retained traversal page is explicitly accounted for exactly once;
2. every assessed page is joined to the exact retained traversal response identity/digest and typed resource domain;
3. every assessed page has one exact HAK-016 selector result;
4. assessed pages use one common recorded normalization-policy identity;
5. retained-page counts are aggregated deterministically;
6. unassessed or failed evidence cannot disappear from the aggregate;
7. tested assurance labels cannot be promoted by coherent redigest.

## What HAK-018b does not prove

HAK-018b does not establish:

- provider authentication;
- raw provider-response content replay;
- HAK-014 replay from exact raw response bytes;
- normalization-policy artifact-content verification;
- entity projection provenance;
- continuation-source provenance;
- actual provider exhaustion;
- temporal snapshot consistency;
- retry lineage;
- provider-population-qualified selector coverage;
- scientific or semantic truth;
- legal/governance legitimacy;
- human worth;
- runtime authority.

## Next composition boundary

The remaining strong claim needs additional evidence:

```text
Provider/source-content verification
+
Verified exhaustion under exact provider contract
+
Temporal snapshot semantics where required
+
HAK-018b conserved retained coverage set
->
Candidate population-qualified coverage
```

Even then:

```text
PopulationQualifiedCoverage
!=
SemanticTruth
!=
Authority
```
