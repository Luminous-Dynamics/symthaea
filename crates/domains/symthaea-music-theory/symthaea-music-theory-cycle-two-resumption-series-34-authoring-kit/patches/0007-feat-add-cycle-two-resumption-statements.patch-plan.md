# Patch 0007: feat add cycle two resumption statements

**Series:** 34

## Objective

Add recovery-authority and witness statements over the exact successor-segment plan.

## Intended changes

- Use cycle-two and resumption-specific domain separators.
- Bind current policy epochs, signer identity, plan digest, successor segment, and target head.
- Expose shell-free external-verifier payloads.

## Acceptance evidence

- Cycle-one resumption, cycle-two recovery, and cycle-two closure signatures cannot replay.
- Wrong role, plan, cycle, policy, segment, or head fails.
- Independent vectors agree.

## Non-claims

- Does not manage private keys.
- Does not make witnesses publishers.
