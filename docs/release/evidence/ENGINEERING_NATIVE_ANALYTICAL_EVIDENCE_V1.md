# ETK-3C — Native Analytical Evidence V1 Reference Boundary

Status: **independent exact-byte reference theorem; production Rust source exists in #2058 but is not yet execution-qualified**

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

Native analytical evidence is **not** external-solver `Simulation`. The reference vectors use the distinct semantic evidence class `Analysis`, introduced separately by #2001.

Changing the evidence class therefore changes both accepted-requirement and proof-obligation identities. Existing `Simulation` revisions must never be silently reinterpreted as native analysis.

## Canary semantics

The reference uses the existing Euler-Bernoulli rectangular cantilever / end-point-load calculation as an authority canary. It binds:

- exact method and algorithm-artifact identities;
- explicit method assumptions and supported load cases;
- Canonical Binary64 analytical inputs and SI units;
- an accepted Civil/Blocking `Analysis` requirement with `stress <= 250 MPa`;
- exact Analysis proof-obligation semantics;
- evidence-policy identity and model-qualification-record premise;
- subject, twin, validity-domain, and currentness identities;
- exact execution-artifact identity.

The fixed implementation, qualification, and execution digests are fixtures. They do not establish that a present Rust binary, material model, or physical design has been qualified.

## Requirement-policy compatibility

A syntactically valid analytical policy is not automatically adequate for the accepted requirement.

For the canary, plan construction now verifies that the policy's worst-case permitted stress, including its declared maximum model-relative-error allowance, cannot exceed the accepted 250 MPa requirement:

```text
(yield_strength / FoS_threshold) * (1 + max_model_error)
    <= accepted_requirement_max_stress
```

The frozen positive policy `FoS >= 2.0`, maximum model error `0.05`, and fixture yield strength `250 MPa` implies at most `131.25 MPa`, so it is conservative relative to the 250 MPa requirement.

A deliberately weak `FoS >= 0.5` policy is rejected at plan construction even though it is otherwise syntactically valid.

## Exact result-to-input binding

A result candidate now carries the exact method and input revisions it claims to describe. Admission rejects a candidate whose method/input lineage differs from the bound plan.

Admission also independently recomputes the canary equations from the exact bound inputs before trusting the reported outputs:

```text
M = P L
section_modulus = b h^2 / 6
stress = M / section_modulus
I = b h^3 / 12
deflection = P L^3 / (3 E I)
FoS = yield_strength / stress
```

The reported moment, bending stress, deflection, and FoS must agree with those independently recomputed values to relative tolerance `1e-12`.

Thus:

```text
internally consistent numbers
!= numbers belonging to this exact analytical input
```

## Conservative admission

After exact plan and equation binding, admission additionally requires:

1. canonical execution-artifact SHA-256 identity;
2. finite result values;
3. model-relative-error within the bound policy;
4. error-adjusted bending stress still below the accepted 250 MPa requirement;
5. conservative `FoS / (1 + actual_error_bound)` still satisfying the FoS policy.

A native faculty's convenience `passes` boolean is not an authority input.

## Present-tense applicability

The historical receipt binds one exact admitted analytical evidence identity, exact plan, and exact obligation revision. A current analytical-discharge fact can be derived only for that same exact plan.

The stale-receipt test compares two independently schema-derived currentness assertions over the same twin/validity context, changing the attestation record. The old receipt becomes historical under the refreshed plan.

## Frozen reference vectors

The strengthened gates do **not** change the positive protocol identities:

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

analytical plan
sha256:b86b76c504bd7b7601981d40988ceafd6c0a6580c941e4c91a65054c57749432

admitted analytical evidence
sha256:7ea504bb399216df14d5792b44f574e18c62163049c4406e23e4dc3783e86bde

historical analytical discharge receipt
sha256:5d740e2cfe67fa0bc0145cc7d79a9410e9c36b276ac78f12243ad5e013f7a1ec

current analytical discharge fact
sha256:47e1d641da233d0d70e6084a97bbe55fbf9d0487a3b6f7978ca03790fbeb494c
```

## Adversarial reference cases

The checked-in self-test now fails closed on:

- changed analytical input under an old plan;
- result-level method/input binding drift;
- a policy too weak for the accepted stress requirement;
- a nominal FoS pass whose conservative FoS fails;
- excessive model-relative error;
- inconsistent reported FoS;
- incorrect reported beam deflection;
- incorrect reported beam moment;
- malformed execution-artifact digest;
- reuse of a historical receipt after a schema-derived currentness refresh.

It also freezes signed-zero Canonical Binary64 normalization and the baseline currentness identity.

## Exact-byte reference execution evidence

The strengthened checked-in oracle is exactly Git blob:

```text
deb6d11f12cd026e63eef6620643585cbacaa899
```

Those exact bytes were executed locally before check-in:

```text
--self-test              PASS
python3 -m py_compile    PASS
raw SHA-256              9fc125197e2b33a311ab272d18d1a7f907ce1371001d19a04a79b551ef01c5e8
Git blob SHA-1           deb6d11f12cd026e63eef6620643585cbacaa899
checked-in Git blob      deb6d11f12cd026e63eef6620643585cbacaa899
```

This is exact-byte **reference execution evidence** only. It is not Rust qualification, physical validation, model qualification, or authentication of the fixture premises.

## Production sequence

```text
#2001 distinct Analysis evidence kind
    -> #1999 independent analytical authority theorem
    -> #2058 typed production analytical boundary
    -> qualify exact Rust vectors
    -> compose with explicit requirement->obligation relationship/completeness
    -> migrate structural discharge helper as first canary
    -> migrate remaining native disciplines
    -> delete free-form native discharge mutation path
```

#2058 already implements the production canary source boundary, including exact method/input result binding, independent equation recomputation, requirement-policy adequacy, immutable receipts, currentness invalidation, and one-way authority IDs. Its exact Rust execution remains pending.

## Deliberate nonclaims

Neither #1999 nor #2058 establishes Euler-Bernoulli applicability to a particular physical structure, material-property truth, scientific validity of the fixture 5% error bound, authenticity of acceptance/currentness/qualification records, evidence independence, requirement-derivation completeness, design qualification, certification, manufacturing approval, deployment approval, or physical actuation authority.
