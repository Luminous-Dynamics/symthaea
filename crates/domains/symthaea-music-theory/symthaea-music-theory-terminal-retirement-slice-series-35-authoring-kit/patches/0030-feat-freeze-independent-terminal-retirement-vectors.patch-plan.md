# Patch 0030: feat freeze independent terminal retirement vectors

**Series:** 35

## Objective

Publish language-neutral vectors for retirement, revocation, terminal checkpoint, archive, and handoff contracts.

## Intended changes

- Include canonical bytes, positive cases, policy mutations, role replay, capability omission, receipt mutations, and archive audits.
- Version every role and expected policy.
- Provide independent verifier protocol examples.

## Acceptance evidence

- Rust and at least one independent implementation agree.
- Required disagreement blocks qualification.
- Vectors are architecture independent.

## Non-claims

- Does not prove organizational independence.
- Does not expose private keys.
