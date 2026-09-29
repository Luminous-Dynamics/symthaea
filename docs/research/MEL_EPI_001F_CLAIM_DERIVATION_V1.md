# MEL-EPI-001F — Fail-Closed Claim Derivation Projection V1

Tracker: #5355
Parent architecture: #5275
Source parent: MEL-EPI-001D draft #5369 / `a4f5351882b08f044319d5e61a01c792341f3726`

Status: source-staged only. This tranche defines generic derivation declarations and structural consistency. It establishes no executable qualification and no source-native claim admission.

## Purpose

MEL-EPI must never derive child authority by unioning ancestor claims or by following provenance edges.

```text
parent A declares X
parent B declares Y
child C derived from A + B

!=

child C establishes X + Y
```

V1 makes claim transfer explicit, profile-relative, provenance-bound, preservation-aware, and fail-closed.

The governing rule is:

```text
no registered derivation profile
-> no derived positive claim
```

Even when a generic derivation receipt validates:

```text
declared prerequisite satisfied
!= source-admitted prerequisite
!= scientific theorem verified
```

Source/domain-specific verification remains mandatory before any resulting positive claim becomes trustworthy authority.

## Profile projection

V1 defines:

```text
ClaimDerivationProfileProjectionV1 {
  profile_version,
  profile_id,
  mechanism_activity_id,
  input_requirements,
  output_scope_profile_id,
  output_claims,
  explicit_nonclaims,
}
```

The name `Projection` is deliberate. This is ordinary serializable data, not a credential or theorem proof.

## Input requirements

Each exact role binds:

```text
ClaimDerivationInputRequirementV1 {
  role_id,
  required_scope_profile_id,
  required_claims,
  preservation,
}
```

Input roles are strictly sorted and unique. Required claims are strictly sorted and unique.

This prevents a derivation profile from saying merely “some input with claim X.” It must name the exact logical role and exact claim-scope interpretation profile expected for that role.

## Preservation policy

V1 freezes two input preservation policies:

```text
RequireLosslessUnderProfile

AllowProjectedWithLoss {
  allowed_loss_codes
}
```

`RequireLosslessUnderProfile` accepts only an input source ref whose 001B preservation ceiling is `LosslessUnderProfile`.

`AllowProjectedWithLoss` accepts:

- `LosslessUnderProfile`; or
- `ProjectedWithLoss` only when every actual loss code is explicitly present in the profile's allowed-loss set.

An empty allowed-loss set is invalid; use `RequireLosslessUnderProfile` instead.

Core rule:

```text
loss not explicitly permitted
-> prerequisite rejected
```

Allowing a projected input does not expand the output claim ceiling.

## Mechanism binding

Every profile binds an exact `mechanism_activity_id: EvidenceSemanticIdV1`.

A concrete receipt must contain that exact semantic-only provenance node typed as `Activity`, and the output source reference must have an exact:

```text
output --GeneratedBy--> mechanism activity
```

association edge.

This establishes only structural provenance linkage under V1. It does not prove that the activity really executed, that code was correct, or that the scientific transformation is valid.

## Receipt projection

V1 defines:

```text
ClaimDerivationReceiptProjectionV1 {
  receipt_version,
  profile,
  inputs,
  output,
  provenance,
}
```

Each input binding contains:

```text
role_id
exact ClaimScopeProjectionV1
```

The receipt input role set must exactly equal the profile input role set.

## Prerequisite declaration checks

For every role, generic validation requires:

1. exact required claim-scope profile match;
2. preservation policy satisfied;
3. every required claim returns `DeclaredEstablished` from 001C;
4. `ExplicitNonclaim` fails with a distinct error;
5. `NotDeclared` fails closed.

This check is intentionally named/understood as **declaration consistency**.

```text
DeclaredEstablished in generic projection
!= source-admitted established claim
```

The generic receipt does not possess enough authority to erase that distinction.

## Exact output ceiling

The output `ClaimScopeProjectionV1` must match the profile exactly:

```text
output.scope_profile_id == profile.output_scope_profile_id
output.establishes       == profile.output_claims
output.explicit_nonclaims == profile.explicit_nonclaims
```

No extra positive output claim is allowed, even if some parent input declares it.

Therefore:

```text
parent authority union
!= output authority
```

and:

```text
valid signature on input bytes
!= additional scientific output claim
```

unless a separate derivation profile explicitly requires/authenticates that theorem.

## Provenance requirements

The exact output source reference must appear in the 001D graph.

Every exact input source reference must also appear and must be reachable from the output through one or more **derivation** edges:

```text
output -> ... -> input
```

Association/context reachability does not satisfy this requirement.

The exact input source ref may not equal the exact output source ref.

Because 001D endpoints retain full source-reference identity, two native artifacts sharing a semantic ID are not interchangeable here.

## Provenance is necessary but insufficient

A derivation path is a structural prerequisite only.

```text
provenance path exists
!= prerequisite claim admitted
!= transformation scientifically valid
!= output claim trustworthy
```

001F combines provenance structure with claim declarations and preservation constraints, but generic validation still stops before source/domain admission.

## Deterministic identity

`ClaimDerivationProfileProjectionV1::semantic_payload()` binds:

- profile version and profile ID;
- exact mechanism semantic identity;
- exact ordered input-role requirements;
- required claim-scope profiles;
- prerequisite claim sets;
- preservation policies and allowed losses;
- exact output scope profile;
- exact output claims and nonclaims.

`ClaimDerivationReceiptProjectionV1::semantic_payload()` then binds:

- receipt version;
- complete profile payload;
- exact ordered input claim-scope payloads;
- exact output claim-scope payload;
- complete canonical 001D provenance payload.

No arbitrary Serde JSON is hashed as epistemic identity.

## Typestate

V1 adds:

```text
ValidatedClaimDerivationReceiptProjectionV1
```

It caches canonical receipt bytes after full generic validation and exposes no mutable state.

The helper `from_validated_parts(...)` may consume 001G validated claim-scope wrappers and an 001D validated provenance graph at the construction seam, but returns ordinary raw serializable data followed by whole-receipt validation.

Even the validated receipt wrapper remains declaration-only typestate:

```text
ValidatedClaimDerivationReceiptProjectionV1
!= DomainAdmittedClaimDerivation
```

A future source/domain-specific admitted wrapper must retain exact source verification evidence for every prerequisite and for the derivation mechanism/profile itself.

## Source controls

The source tests cover:

1. canonical receipt structural validation;
2. absent prerequisite -> fail closed;
3. explicit nonclaim cannot satisfy positive prerequisite;
4. output cannot exceed exact profile claim ceiling;
5. projected input rejected when role requires lossless;
6. explicitly allowed projected loss accepted without expanding output;
7. unregistered projection loss rejected;
8. exact derivation path required;
9. exact `GeneratedBy` mechanism link required;
10. validated receipt remains declaration-only typestate.

## First downstream use

MEL-EPI-MUSE-001 / #5351 can eventually define a domain-specific derivation profile for collection-close or confirmatory analysis relationships, but only after exact source adapters can produce source-admitted prerequisite scopes.

A future confirmatory-analysis profile might require independently admitted claims for exact lifecycle closure, unblinding, input binding, analysis plan binding, and cross-check execution before admitting a bounded output claim such as:

```text
muse.analysis.confirmatory-under-profile
```

It must not derive broader claims such as general causal validity or artistic quality unless a separate theorem explicitly establishes those.

## Explicit nonclaims

001F does not establish:

- source-native artifact validity;
- source admission of prerequisite claims;
- correctness or truth of a derivation theorem;
- execution of the bound mechanism activity;
- cryptographic authentication or signer authorization;
- trusted chronology;
- preregistration authenticity;
- consent/export authority;
- statistical significance;
- causal validity;
- generalization;
- listener preference;
- artistic or musical quality;
- product authority.

It establishes only a fail-closed, deterministic structural language for declaring and checking bounded claim-transfer prerequisites once exact executable qualification eventually confirms the implementation subject.
