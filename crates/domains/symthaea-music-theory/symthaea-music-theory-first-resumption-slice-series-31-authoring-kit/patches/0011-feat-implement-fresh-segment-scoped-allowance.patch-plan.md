# Patch 0011: feat implement fresh segment scoped allowance

**Series:** 31

## Objective

Provide one consumable publication allowance that cannot cross the closure boundary.

## Intended changes

- Bind allowance issuer, subject, segment, channel, maximum count, consumed count, issue/expiry epoch, and predecessor reference.
- Support reservation and transactional consumption.
- Reject implicit carryover.

## Acceptance evidence

- Old, copied, wrong-segment, exhausted, and expired allowances fail.
- Failed commits do not consume allowance.
- Successful first mutation consumes exactly one unit.

## Non-claims

- Does not authorize publication alone.
- Does not erase prior allowance history.
