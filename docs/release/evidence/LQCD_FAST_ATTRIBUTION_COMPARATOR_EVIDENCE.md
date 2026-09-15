# LQCD Fast-Lane Attribution Comparator Evidence

Authority: **independent engineering evidence only**.

This tranche converts the causal-classification policy from issue #3404 into a deterministic standard-library comparator over normalized lint-identity evidence.

## Classification states

- `InheritedBaselineFailure`
- `CandidateRegression`
- `BaselineDifferentFailure`
- `VerifierOrToolchainShift`
- `BaseComparisonUnavailable`

## Equivalence context

Before comparing lint identities, the comparator requires equality of:

- qualification profile ID;
- recipe-semantics SHA-256;
- rustc identity;
- Clippy identity;
- exact Clippy command.

A mismatch yields `VerifierOrToolchainShift` and does not authorize source repair.

## Synthetic controls

Six controls pass:

1. identical base/candidate lint sets -> `InheritedBaselineFailure`;
2. equivalent passing base with empty lint set -> `CandidateRegression`;
3. failing base with different identity set -> `BaselineDifferentFailure`;
4. toolchain mismatch -> `VerifierOrToolchainShift`;
5. absent base evidence -> `BaseComparisonUnavailable`;
6. malformed base identity evidence -> `BaseComparisonUnavailable`.

## Current real result

Using #3442 candidate identity evidence without a completed #3434 base artifact:

- classification: `BaseComparisonUnavailable`
- Rust repair authorized: `false`
- baseline-maintenance scope required: `false`

This is intentional. The comparator is ready, but exact-base hosted evidence is still required before causal classification.

## Claim ceiling

Engineering causal attribution only. The comparator does not qualify Rust code, establish numerical correctness, or support any lattice-QCD physics claim.
