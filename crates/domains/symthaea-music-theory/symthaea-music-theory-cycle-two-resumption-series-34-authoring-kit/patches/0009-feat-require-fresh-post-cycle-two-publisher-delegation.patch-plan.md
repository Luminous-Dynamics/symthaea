# Patch 0009: feat require fresh post cycle two publisher delegation

**Series:** 34

## Objective

Prevent the earlier resumed segment's publisher authority from carrying into the successor segment.

## Intended changes

- Bind new delegation to cycle-two closure epoch, successor segment, channel, subject, issuer, expiry, and expected verifier policy.
- Reject all delegations issued before cycle-two closure.
- Preserve historical delegation evidence.

## Acceptance evidence

- Series 31, pre-freeze, pre-closure, wrong-segment, wrong-channel, expired, and externally rejected delegations fail.
- No extra quorum compensates for missing delegation.
- Dependent plan bytes change under delegation substitution.

## Non-claims

- Does not destroy old keys.
- Does not contact issuers.
