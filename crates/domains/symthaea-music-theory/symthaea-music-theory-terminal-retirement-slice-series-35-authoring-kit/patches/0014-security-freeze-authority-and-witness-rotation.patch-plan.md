# Patch 0014: security freeze authority and witness rotation

**Series:** 35

## Objective

Prevent retirement reversal through new policy epochs or enrollments.

## Intended changes

- Append terminal authority and witness ledger events.
- Disallow new rotations, enrollments, or activation under the retired lineage.
- Retain old epochs for historical verification only.

## Acceptance evidence

- Post-retirement rotation and enrollment fail.
- Historical signatures still verify under original policy.
- Terminal events cannot be superseded.

## Non-claims

- Does not prevent a genuinely new catalog identity.
- Does not guarantee old algorithms remain secure indefinitely.
