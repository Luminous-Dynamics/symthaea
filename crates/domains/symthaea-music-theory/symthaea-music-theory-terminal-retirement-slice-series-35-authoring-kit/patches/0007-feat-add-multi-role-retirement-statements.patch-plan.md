# Patch 0007: feat add multi role retirement statements

**Series:** 35

## Objective

Add distinct retirement statements for configured governance, recovery, witness, and preservation roles.

## Intended changes

- Bind signer role, active policy epoch, plan digest, catalog head, and retirement domain.
- Permit caller-owned role combinations and thresholds.
- Expose external-verifier payloads.

## Acceptance evidence

- Closure, recovery, reopening, and resumption signatures cannot replay.
- Wrong role, plan, head, or policy fails.
- Independent canonical vectors agree.

## Non-claims

- Does not prove signer independence.
- Does not require every role in every deployment.
