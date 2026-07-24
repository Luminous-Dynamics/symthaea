# Patch 0008: feat implement cycle specific resumption statements

**Series:** 31

## Objective

Implement signed recovery-authority and witness statements for the exact plan.

## Intended changes

- Use distinct signer roles and domain separators.
- Bind expected policy epochs and signer identities.
- Return canonical signable payloads.

## Acceptance evidence

- Wrong role, plan, policy epoch, and signer substitution fail.
- Closure signatures cannot replay as resumption statements.
- Payload vectors match the independent specification.

## Non-claims

- Does not manage private keys.
- Does not prove organizational independence.
