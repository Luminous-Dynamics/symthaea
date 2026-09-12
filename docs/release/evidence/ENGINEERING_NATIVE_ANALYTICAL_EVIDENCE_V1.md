# ETK-3C — Native Analytical Evidence V1 Reference Boundary

Status: **independent reference theorem; production implementation not yet established**

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

The current engineering facade violates the first boundary in its explicit `discharge_*_check` helpers: a native assessment's `passes` boolean can directly mutate a matching proof obligation to `Discharged` and attach a free-form evidence string. ETK-3C is the replacement path; it does not validate those legacy mutations.

## Distinct evidence class

Native analytical evidence is **not** `EvidenceKind::Simulation`. The current formal-safety enum describes `Simulation` as a result from an external solver.

The reference vectors therefore use a distinct semantic evidence class:

```text
Analysis
```

That changes both accepted-requirement and proof-obligation content identities. A legacy requirement that explicitly requests `Simulation` must not be silently satisfied by a native calculation.

Production prerequisite: add a first-class formal-safety analytical evidence variant in a separately reviewable compatibility change, then update the accepted-requirement/evidence-plan canonicalizers to recognize it. Do not reinterpret old `Simulation` revisions in place.

## Canary method

The independent oracle uses the existing `symthaea-structural` Euler-Bernoulli beam calculation as a canary because the method has explicit inputs, SI units, a documented validity envelope, and several outputs.

The method identity commits to:

- a method key;
- implementation-artifact and algorithm-revision digests;
- explicit assumptions;
- supported load cases;
- output names/units;
- unit system.

The fixed digests in the reference vector are **fixtures**. They do not claim that the present Rust source/binary has already been reproducibly bound to those values. Production must replace fixture artifact identities with exact reproducibility/supply-chain evidence.

## Exact analytical inputs

The canary input identity commits to the exact method revision and Canonical Binary64 encodings of:

- beam length;
- rectangular section dimensions;
- Young's modulus;
- yield strength;
- load-case kind, value, and unit.

Input ordering is non-semantic; numeric input changes are semantic. A result produced for changed inputs cannot be admitted under an old plan.

## Acceptance policy

The reference policy binds:

- required metric `factor_of_safety`;
- operator `>=`;
- threshold `2.0`;
- maximum allowed model-relative-error bound `0.05`;
- an explicit model-qualification-record digest.

The positive vector uses fixture qualification/error premises solely to test authority semantics. The oracle does **not** establish that a 5% bound is scientifically justified for Euler-Bernoulli analysis, nor that the qualification record is authentic.

A production analytical evidence path must obtain such error/qualification premises from actual validation/calibration evidence rather than inventing them.

## Conservative admission

For the structural canary, admission checks:

1. exact method, input, and policy binding;
2. canonical execution-artifact content identity;
3. finite result values;
4. factor-of-safety self-consistency with yield strength and reported bending stress;
5. model-error bound within policy;
6. conservative factor of safety `FoS / (1 + error_bound)` still satisfies the threshold.

Therefore:

```text
assessment.passes == true
```

is neither an input to authority nor sufficient for admission.

## Present-tense applicability

The historical receipt binds one exact admitted analytical evidence identity, plan, and obligation revision. A current analytical-discharge fact can be minted only for the exact current plan. A currentness refresh changes the plan and makes the old receipt historical.

## Frozen reference vectors

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

The checked-in self-test fails closed on:

- changed analytical input under an old plan;
- a nominal pass whose conservative margin falls below threshold;
- a model-error bound exceeding policy;
- inconsistent reported factor of safety vs yield/stress;
- malformed execution-artifact digest;
- currentness refresh attempting to reuse a historical receipt.

It also freezes signed-zero Canonical Binary64 normalization.

## Exact-byte reference execution evidence

The checked-in `scripts/etk-native-analytical-evidence-oracle.py` bytes were executed locally after accounting for the repository file's no-final-newline representation:

```text
--self-test              PASS
python3 -m py_compile    PASS
raw SHA-256              68351dc9133a942fbe3210a51af39562656bd1b7e94d1b1b64f48b84f1602034
Git blob SHA-1           4214dad1bd63694220aaf535c569bf57dc9bf6af
checked-in Git blob      4214dad1bd63694220aaf535c569bf57dc9bf6af
```

This establishes execution of the exact reference bytes only. It does not establish production Rust parity, physical truth, model qualification, or evidence authentication.

## Production sequence

The recommended migration order is:

```text
first-class Analysis evidence class
    -> typed analytical method/input/policy/plan
    -> native result candidate
    -> analytical admission capability
    -> historical receipt
    -> current analytical fact
    -> replace one legacy discharge helper (structural canary)
    -> extend discipline-by-discipline
    -> delete shared free-form discharge mutation path
```

Keep the existing `evaluate_*` methods as computation seams. Replace authority-mutating `discharge_*` wrappers rather than conflating calculation with authority.

## Deliberate nonclaims

This reference does not establish the correctness of Euler-Bernoulli theory, applicability to a particular physical structure, material-property truth, calibration validity, the authenticity of the fixture qualification record, evidence independence, design qualification, certification, manufacturing approval, deployment approval, or physical actuation authority.