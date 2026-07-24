# Patch 0011: feat implement cycle two authorization statements

**Series:** 33

## Objective

Add cycle-bound recovery-authority and witness statements over the exact plan.

## Intended changes

- Use cycle-two and role-specific domain separators.
- Bind active policy epochs, signer identities, plan digest, and intended anchor.
- Expose external-verifier request payloads.

## Acceptance evidence

- Cycle-one, closure, resumption, and reopening signatures cannot replay.
- Wrong cycle, plan, role, policy epoch, or signer fails.
- Independent canonical vectors agree.

## Non-claims

- Does not manage private keys.
- Does not make witnesses branch selectors by themselves.
