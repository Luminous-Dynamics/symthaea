# Patch 0007: feat implement cycle scoped recovered witness policy

**Series:** 33

## Objective

Require fresh cycle-two witness policy and checkpoint statements.

## Intended changes

- Bind policy epoch, activation point, accepted witnesses, threshold, cycle identity, and quarantine state.
- Require observations strictly after the cycle-two recovery anchor.
- Keep earlier witness evidence historical-only for freshness.

## Acceptance evidence

- Cycle-one, pre-anchor, wrong-policy, duplicate, and quarantined witness statements do not count.
- Threshold and freshness are separate report dimensions.
- Embedded weaker policy cannot override expected policy.

## Non-claims

- Does not prove witness independence.
- Does not create trusted wall-clock time.
