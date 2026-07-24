# Patch 0006: feat implement cycle two resumption plan

**Series:** 34

## Objective

Build one exact plan for the first publication of the successor segment.

## Intended changes

- Bind cycle-two closure and certification, successor segment, exact catalog head, active policy epochs, quarantine state, channel, publication identity, delegation constraints, allowance constraints, and limitations.
- Require global predecessor ordinals.
- Expose canonical signable payloads.

## Acceptance evidence

- Wrong cycle, stale head, wrong segment, wrong channel, active forbidden quarantine, and expired plan fail.
- Every semantic mutation changes plan bytes.
- The plan is non-mutating.

## Non-claims

- Does not authorize publication.
- Does not consume an allowance.
