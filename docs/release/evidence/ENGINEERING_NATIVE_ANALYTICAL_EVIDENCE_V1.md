# ETK-3C — Native Analytical Evidence Reference Boundary

Status: **independent exact-byte reference theorem; production Rust source exists but bounded-currentness V2 is not yet production-qualified**

This tranche defines how Symthaea should treat in-process, closed-form, algebraic, or otherwise native engineering calculations without pretending they are external-solver simulation evidence.

## Core theorem

```text
native calculation result
!= admitted analytical evidence
!= historical analytical discharge receipt
!= current analytical discharge fact
!= complete requirement satisfaction
!= qualified design / certification / manufacturing / deployment / actuation authority
```

The legacy engineering facade violates the first boundary in explicit `discharge_*_check` helpers, where native assessments can directly mutate proof-obligation authority state. ETK-3C replaces that pattern; it does not validate it.

## Distinct evidence class

Native analytical evidence is **not** external-solver `Simulation`. The reference uses the distinct semantic evidence class `Analysis`, introduced separately by #2001.

Changing the evidence class therefore changes both accepted-requirement and proof-obligation identities. Existing `Simulation` revisions must never be silently reinterpreted as native analysis.

## Canary semantics

The reference uses the existing Euler-Bernoulli rectangular cantilever / end-point-load calculation as an authority canary. It binds exact method and algorithm-artifact identities, explicit assumptions, Canonical Binary64 inputs, an accepted Civil/Blocking `Analysis` requirement, exact Analysis obligation semantics, analytical policy, model-qualification premise, semantic subject/twin/validity context, bounded currentness, and execution-artifact identity.

The fixed implementation, qualification, currentness-attestation, and execution digests are fixtures. They do not establish that a present Rust binary, material model, physical design, or attestation source has been qualified or authenticated.

## Requirement-policy compatibility

A syntactically valid analytical policy is not automatically adequate for the accepted requirement. The canary verifies:

```text
(yield_strength / FoS_threshold) * (1 + max_model_error)
    <= accepted_requirement_max_stress
```

For the frozen fixture, `FoS >= 2.0`, maximum model error `0.05`, and yield strength `250 MPa` imply a worst permitted stress of `131.25 MPa`, which is conservative relative to the accepted `250 MPa` bound. A deliberately weak `FoS >= 0.5` policy is rejected at plan construction.

## Exact result-to-input binding

A result candidate carries the exact method and input revisions it claims to describe. Admission independently recomputes the canary equations from those exact bound inputs:

```text
M = P L
section_modulus = b h^2 / 6
stress = M / section_modulus
I = b h^3 / 12
deflection = P L^3 / (3 E I)
FoS = yield_strength / stress
```

Reported moment, bending stress, deflection, and FoS must agree with independently recomputed values to relative tolerance `1e-12`.

Thus:

```text
internally consistent numbers
!= numbers belonging to this exact analytical input
```

## Conservative admission

After exact plan and equation binding, admission additionally requires a canonical execution-artifact identity, finite values, actual model-relative error within policy, error-adjusted bending stress inside the accepted requirement, and conservative `FoS / (1 + actual_error)` satisfying the policy threshold.

A native faculty's convenience `passes` boolean is not an authority input.

## Bounded currentness V2

The previous reference proved that refreshing the currentness assertion invalidates an old receipt, but it left a subtler authority hole: an unchanged assertion could remain "current" forever.

V2 therefore makes freshness explicit:

```text
same currentness assertion
!= currently applicable forever
```

The currentness assertion now commits:

```text
attestation_digest
observed_at_unix_ms
valid_until_unix_ms
twin_revision_id
validity_domain_revision_id
```

under `symthaea.etk-currentness-assertion.v2`. Construction fails if `valid_until_unix_ms <= observed_at_unix_ms`.

Deriving a present-tense analytical discharge fact additionally requires an explicit `evaluated_at_unix_ms` satisfying the inclusive interval:

```text
observed_at_unix_ms
    <= evaluated_at_unix_ms
    <= valid_until_unix_ms
```

Evaluation before observation fails closed as `freshness_not_yet_valid`; evaluation after expiry fails closed as `freshness_expired`. Both exact interval boundaries are valid.

The present-tense fact uses `symthaea.etk-current-native-analytical-discharge-fact.v2` and commits the observation time, expiry, and evaluation time. Historical receipts remain immutable historical evidence after freshness expires; they simply cease to justify a present-tense fact.

## Vector migration

The V2 freshness theorem deliberately leaves semantics that do not depend on present applicability unchanged:

```text
Analysis requirement revision
sha256:10891514f6551c85b10674a3df04ab2c6d19f74f014adeacc671fa31deecd419

Analysis obligation revision
sha256:ea6e2f0f5e3ec37524325eda24b95ead75fa502f5d2a0b9410a407aed3a12a8c

method revision
sha256:73bb060e12b166cecfea9a73c1f274f4443f1a540a777e7ac8260c65808bedc3

input revision
sha256:3b9a62ea2ce031b73c188e8c0138fb1ebcda20e77d7f2401502ec493323d3c86

acceptance policy
sha256:7edc8fcbc03184413cc9c275537bdefb2d1c18f281406dc1adacd9a53a0a3bb5
```

Only the freshness-dependent chain re-keys:

```text
currentness assertion V2
sha256:deab1cf24d0cb23c7e72697f9bc41277b82997696a13ea1c25fb3c7a7d0994c6

analytical plan
sha256:bc308f5399ca9032131dcbed3f53991ab104465b15471521b0388be7f611b5e1

admitted analytical evidence
sha256:228f0ceb1566fa133d8647ccc5859ffd45957ab5138d67fa4b943a79fcdf0ec7

historical analytical discharge receipt
sha256:ce0b9d96d83e829d3878e7303e35ac607dbed8512a2202e7c224ca4c32ee0b81

current analytical discharge fact V2
sha256:2d6e90e20460efc8f2f3d24c72e2c7a07be77e0143119d38f23385a868dad6eb
```

The frozen V2 fixture uses:

```text
observed_at_unix_ms = 1789123456000
valid_until_unix_ms = 1789209856000
evaluated_at_unix_ms = 1789123457000
```

## Adversarial reference cases

The checked-in self-test fails closed on changed analytical input under an old plan, result method/input drift, requirement-inadequate policy, nominal-but-not-conservative acceptance, excessive model error, inconsistent FoS, incorrect beam deflection, incorrect moment, malformed execution-artifact identity, invalid currentness windows, evaluation before observation, evaluation after expiry, and historical receipt reuse after a refreshed attestation.

It also proves the exact observation and expiry boundaries remain valid and freezes Canonical Binary64 signed-zero normalization.

## Exact-byte reference execution evidence

The V2 checked-in oracle is exactly Git blob:

```text
5fee41abb9fccd3bde0158f19babe5bc77716f20
```

Those exact bytes were executed locally before check-in:

```text
--self-test              PASS
python3 -m py_compile    PASS
raw SHA-256              18d053f4b683f2ffc6e81197c2218c256c363c583f5b9b022b2c62c7d91fedb2
Git blob SHA-1           5fee41abb9fccd3bde0158f19babe5bc77716f20
checked-in Git blob      5fee41abb9fccd3bde0158f19babe5bc77716f20
```

This is exact-byte **reference execution evidence** only. It is not Rust qualification, physical validation, model qualification, or authentication of fixture premises.

Exact oracle source commit before this documentation-only reconciliation: `17e52321f3bf7a4e98508a7ec214015ab72a520e`.

Repository CI run `34729742948` and Showroom Integrity run `34729742969` are queued. Benchmark run `34729742961` is skipped and is not qualification evidence.

## Production sequence

```text
#2001 distinct Analysis evidence kind + canonical names
    -> #1999 independent analytical theorem with bounded currentness V2
    -> #2058 production analytical boundary
    -> #2201 shared EvidenceKind canonical-name consumption
    -> production bounded-currentness V2 tranche
    -> exact Rust qualification
    -> compose with explicit requirement->obligation relationship/completeness
    -> migrate structural discharge helper as first canary
    -> migrate remaining native disciplines
    -> delete free-form native discharge mutation path
```

Production must not leave the current unbounded V1 present-tense derivation callable as an alternate authority route. Historical V1 records may remain auditable, but only bounded-currentness semantics should mint new present-tense analytical facts.

## Deliberate nonclaims

Neither #1999 nor the production ETK-3C stack establishes Euler-Bernoulli applicability to a particular physical structure, material-property truth, scientific validity of the fixture 5% error bound, authenticity of acceptance/currentness/qualification records, evidence independence, requirement-derivation completeness, design qualification, certification, manufacturing approval, deployment approval, or physical actuation authority.
