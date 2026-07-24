# Patch 0010: feat implement fresh publisher delegation binding

**Series:** 31

## Objective

Require publisher authority created after closure and scoped to the new segment.

## Intended changes

- Bind delegation identity, issuer, subject, channel, segment, issue/expiry epoch, and expected external-verifier policy.
- Reject pre-incident and pre-closure delegations.
- Expose external-verifier request bytes.

## Acceptance evidence

- Old, wrong-segment, wrong-channel, expired, and externally rejected delegations fail.
- Identity substitution changes all dependent plan bytes.
- No extra quorum compensates for missing publisher authority.

## Non-claims

- Does not define key storage.
- Does not contact the issuer.
