# Independent Lean Admission Profile V1

Tracking: SYM-FV-024 / #5733, SYM-FV-024A / #6011, profile freeze #6014.

## Purpose

High-assurance Lean theorem admission should not depend on one checker implementation or a moving toolchain.

```text
ordinary Lean elaboration/typecheck
!= independent proof validation

one kernel accepts
!= independent kernels agree

independent kernels agree
!= theorem statement matches intended meaning
```

V1 freezes an immutable **candidate** profile. It does not execute or qualify that profile.

## Why this exists

Recent Lean releases have fixed kernel/runtime soundness-relevant defects that did not affect independently implemented kernels in the same way. Lean's validation guidance therefore recommends comparator-based validation with independent external checkers for stronger protection against checker implementation bugs.

The security benefit is implementation diversity, not a stronger logical theorem.

## Candidate toolchain

V1 proposes:

```text
Lean        v4.34.1
landrun     5ed4a3db3a4ad930d577215c6b9abaa19df7f99f
lean4export 076e8e57707e813375e8f9da8bf989799ace9680
comparator  d03acab154d269c06e60e4de7e4cc85deebff94b
nanoda      68d5ca9db226849b41a6fff59d796ff19d0a8840
```

The external-tool pins are adopted from the reviewed `leanprover/lean-eval` security pin profile observed on 2026-09-26.

The upstream reference profile currently names Lean v4.34.0. Symthaea deliberately proposes v4.34.1 because it is the current stable patch release, but **does not assume** that the pinned export/comparator toolchain is compatible with it.

Therefore every compatibility coordinate remains:

```text
NeedsQualification
```

until the exact pipeline executes.

## Required checker agreement

Initial profile:

```text
OfficialLeanKernel
+ Nanoda
+ constitutional axiom policy
+ exact statement binding
```

All must accept before canonical `result = Pass` is possible.

Checker disagreement maps to `Blocked`, not to majority vote.

Tool/sandbox unavailability maps to `EnvironmentFailure`, not semantic failure and never PASS.

Malformed/rejected proof subjects map to `Fail`.

## Statement binding

Independent kernels do not protect against proving the wrong theorem.

The challenge statement must therefore be independently reviewable and content-addressed. The solution bridge must be fixed/reviewed, and the retained proof/export receipt must be invalidated whenever the challenge statement identity changes.

## Axiom policy

The profile retains the existing Symthaea constitutional boundary:

- `sorryAx` forbidden;
- undeclared axioms forbidden;
- exact axiom-policy identity retained in the receipt.

Independent checker agreement does not override an axiom-policy rejection.

## Later paranoid profile

Comparator can support additional external checker implementations. A future `IndependentLeanParanoid` profile may use them for R3/R4 trust-boundary claims, but those checkers require their own immutable pins, compatibility tests, resource profile, and review.

They are intentionally **not admitted by this V1 manifest**.

## First executable pilot

#6018 should apply this profile to a small stable theorem family such as repaired BinaryHV XOR algebra before authority-critical theorem families use it.

The pilot must retain:

- exact challenge/solution source identities;
- exact exported proof artifact identity;
- exact checker pins;
- official-kernel result;
- nanoda result;
- axiom census/policy result;
- sandbox/toolchain identity;
- canonical qualification result plus diagnostic class;
- immutable postflight.

## Boundary

```text
independent checker agreement
!= stronger theorem statement
!= Rust source refinement
!= compiler correctness
!= machine-code verification
!= empirical truth
!= runtime authority
```

This profile is a trust-diversity contract around proof checking. It does not change the evidence class of the theorem merely because two kernels agree.