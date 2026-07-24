# Patch 0010: feat require fresh successor segment allowance

**Series:** 34

## Objective

Ensure no earlier segment allowance survives into the new segment.

## Intended changes

- Issue a fresh allowance scoped to successor segment, subject, channel, expiry, and maximum count.
- Retain predecessor allowance reference only as history.
- Support transactional reservation and consumption.

## Acceptance evidence

- Series 31 allowance, copied remaining count, wrong-segment, exhausted, and expired allowances fail.
- Failed commits consume nothing.
- Successful commit consumes exactly one.

## Non-claims

- Does not authorize publication alone.
- Does not erase prior allowance history.
