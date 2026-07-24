# Patch 0011: test run periodic independent verifier conformance

**Series:** 29

## Objective

Detect semantic drift between the Rust implementation and independent verifiers.

## Intended changes

- Run positive, mutation, ambiguity, policy, lineage, retirement, and resource fixtures.
- Pin verifier implementation and policy identities.
- Promote disagreements into blocking regression records.

## Required evidence

- Disagreement never resolves by majority vote.
- No release proceeds with unresolved required-vector disagreement.
- Unsupported vectors are explicit.

## Non-claims

- Does not prove verifier implementations are independently developed.
- Does not require every deployment to run every verifier.
