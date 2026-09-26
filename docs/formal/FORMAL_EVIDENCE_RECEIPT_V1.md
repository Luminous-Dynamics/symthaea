# SYM-FV-006 — Formal Evidence Receipt v1

Tracking issue: #5719  
Authority: **EvidenceOnly**  
Parent architecture: `SYM-FV-000`

## Purpose

This contract gives Symthaea one machine-readable receipt shape for formal and
qualification evidence without flattening different assurance classes into one
"verified" label.

A receipt records **what exact subject was checked, what claim was checked, which
tools and assumptions were involved, what hostile controls ran, and what the
result may and may not establish**.

It does not create runtime authority.

```text
proof/evidence receipt
!= truth by digest alone
!= currentness
!= authenticity
!= authorization
!= permission for external effects
```

## Primary-evidence rule

Every receipt has exactly one `primary_evidence_class`, drawn from the
`SYM-FV-000` evidence-class registry:

- `AbstractFormalTheorem`
- `ExtractedSourceRefinement`
- `DeductiveImplementationProof`
- `BoundedModelSafety`
- `TemporalModelEvidence`
- `BoundedTraceConformance`
- `RuntimeQualification`

These are **not a strength ranking**. They are different evidence planes.

For example:

```text
AbstractFormalTheorem
+
ExtractedSourceRefinement
+
RuntimeQualification
```

may jointly support a larger reviewed claim, but none of the three receipts is
silently relabelled as another class.

## Typed composition

`composition.dependencies` records explicit typed edges between receipts:

- `depends_on`
- `refines`
- `conforms_to`
- `qualified_by`
- `imports_assurance_from`

Every edge retains the dependency receipt's own evidence class and content
digest. The mandatory composition rule is:

```text
DependenciesDoNotPromotePrimaryEvidenceClass
```

This gives us a traversable proof/evidence DAG while preventing authority
amplification by graph composition.

## Exact subject binding

A receipt binds at minimum:

```text
repository
commit
path
Git blob
symbol
optional raw source SHA-256
```

A Git commit or digest is an identity anchor, not evidence of correctness by
itself.

For a positive `Pass`, the qualifier must also record matching pre/post subject
digests and a clean checkout.

## Claim binding

A receipt binds:

- stable `claim_id`;
- human-readable theorem/property statement;
- SHA-256 of those exact statement bytes;
- optional digest of an elaborated/formal statement;
- claim ceiling;
- explicit nonclaims.

The validator recomputes the human-readable statement SHA-256. Tool-specific
qualifiers should additionally bind the strongest stable elaborated
representation available under their pinned toolchain.

```text
same theorem name
!= same theorem statement

same proof bytes
!= proof of a changed statement
```

## Toolchain and intermediate artifacts

The toolchain is an ordered census of named tools and exact identities. Examples
include:

- Rust toolchain;
- Charon revision and options;
- retained LLBC digest;
- Aeneas revision and options;
- generated Lean digests;
- Lean version;
- Mathlib revision;
- Verus revision;
- TLC/Alloy/Quint checker identity and bounds.

Generated and intermediate artifacts are content-bound separately so a changed
translation cannot reuse an old proof receipt invisibly.

## Trust boundary

The receipt records three different things rather than collapsing them:

1. declared assumptions;
2. external/trusted/unmodeled definitions;
3. proof-kernel axiom or verifier-assumption census.

External models are classified as:

- `Proved`
- `Trusted`
- `Unmodeled`
- `UnsupportedBoundary`

A successful `ExtractedSourceRefinement` or `DeductiveImplementationProof`
receipt cannot retain an `UnsupportedBoundary`.

This is deliberately stricter than merely recording that a template or external
function exists.

## Mutation and negative controls

Each declared hostile control records:

```text
id
expected_detection
observed_result
```

where `observed_result` is one of:

- `ExpectedFail`
- `UnexpectedPass`
- `NotRun`
- `InfrastructureFailure`

A positive receipt may be `Pass` only when every declared mutation control
actually reaches `ExpectedFail`.

This prevents a tool-installation error, parser crash, or skipped mutant from
being misreported as semantic mutation sensitivity.

## Qualification result

The receipt result is one of:

- `Pass`
- `Fail`
- `Blocked`
- `EnvironmentFailure`

`Blocked` is a useful outcome. For example, an Aeneas extraction that reaches a
real unsupported dependency boundary should retain that exact result rather than
substituting an axiom or weakening the theorem.

## Immutability rule

For `Pass`:

```text
pre_subject_sha256 == post_subject_sha256
checkout_clean == true
```

This does not prove the runner itself is trustworthy; it proves the qualifier
did not knowingly award positive evidence to bytes that changed during the run.

## Claim ceiling examples

### Lean theorem

```text
AbstractFormalTheorem
!= production Rust refinement
!= native binary correctness
!= runtime authority
```

### Aeneas refinement

```text
ExtractedSourceRefinement
!= rustc correctness
!= LLVM correctness
!= SIMD/native-machine proof
!= runtime authority
```

### Verus proof

```text
DeductiveImplementationProof
!= compiler correctness
!= whole-program correctness
!= runtime authority
```

### Model checker

```text
BoundedModelSafety / TemporalModelEvidence
!= theorem outside the retained model, fairness assumptions, and bounds
!= implementation refinement
```

### Runtime qualification

```text
RuntimeQualification
!= universal theorem
!= behavior outside the retained profile
!= external-effect authority
```

## TCB placement

`symthaea-proof-audit` remains the narrow theorem/axiom-policy authority for Lean
proof admission. This receipt contract does **not** replace it.

The receipt layer records and composes evidence. Tool-specific qualifiers remain
responsible for producing authoritative raw checker output and invoking their
existing proof/spec gates.

In particular, the generic receipt parser should not become a second proof
kernel.

## Aeneas-specific instantiation

`SYM-FV-003` should instantiate this schema with at least:

```text
Rust commit/blob/path/symbol
Rust toolchain
Charon revision/options
LLBC digest
Aeneas revision/options
generated Lean digest census
external-model/template census
abstract HDC theorem/spec digest
Lean identity
theorem statement + digest
#print axioms output digest
symthaea-proof-audit policy/result
mutation-control results
qualification head/run identity
pre/post subject digests
```

Aeneas' current Lean workflow explicitly uses a Rust -> Charon -> LLBC -> Aeneas
-> Lean pipeline, and generated external-function templates can require
hand-maintained models. Those models therefore belong in the receipt's explicit
trust boundary, not in an invisible build step.

## Verus-specific instantiation

Verus receipts should record all assumptions introduced by `assume`,
`external_body`, external specifications, and ignored external items that affect
the verified subject. A verifier PASS with a changed assumption census is a
different evidence subject.

## Model-based-testing instantiation

Quint/TLA+/Alloy/trace-conformance receipts should retain:

- exact model digest;
- checker identity;
- bounds/configuration/fairness;
- projection/adapter identity;
- generated or retained trace digests;
- counterexample digests when present;
- mutation-control results.

Bounded conformance must remain `BoundedTraceConformance`; it cannot be promoted
to full refinement merely because many traces passed.

## Validator

Run:

```bash
python3 scripts/validate_formal_evidence_receipt_v1.py
```

To validate one or more concrete receipts:

```bash
python3 scripts/validate_formal_evidence_receipt_v1.py path/to/receipt.json [...]
```

The validator is intentionally zero-dependency. It verifies both the repository
schema contract and semantic invariants JSON Schema alone cannot express.

## Deliberate nonclaims

This tranche defines evidence representation and fail-closed validation only.

It does **not** establish:

- any HDC theorem;
- any Rust-to-Lean refinement theorem;
- compiler correctness;
- verifier completeness or soundness beyond imported trust;
- distributed protocol correctness;
- runtime authorization;
- currentness/authenticity merely from content digests;
- that all future formal evidence can be represented without a reviewed schema
  revision.
