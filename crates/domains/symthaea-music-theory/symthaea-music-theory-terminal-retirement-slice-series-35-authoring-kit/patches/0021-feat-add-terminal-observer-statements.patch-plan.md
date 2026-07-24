# Patch 0021: feat add terminal observer statements

**Series:** 35

## Objective

Allow external observers to attest that they verified the terminal package without becoming retirement authorities.

## Intended changes

- Bind observer identity, verifier policy, terminal checkpoint, package identity, observation epoch, and limitations.
- Support conflict reporting.
- Keep observations outside the retirement authorization.

## Acceptance evidence

- Wrong package, pre-checkpoint observation, duplicate observer, and policy mismatch fail.
- Observer absence does not invalidate retirement.
- Observer statements cannot restart mutation.

## Non-claims

- Does not prove observer independence.
- Does not make observations trusted timestamps automatically.
