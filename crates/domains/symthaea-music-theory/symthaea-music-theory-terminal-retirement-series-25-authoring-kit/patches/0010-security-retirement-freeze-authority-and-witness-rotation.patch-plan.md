# Patch 0010: security retirement freeze authority and witness rotation

**Series:** 25

## Objective

Prevent terminal retirement from being reversed by rotating to new authority policies.

## Intended changes

- Append terminal authority and witness ledger events.
- Disallow new policy epochs, enrollments, or rotation sets under the retired lineage.
- Keep old policies available for historical verification only.

## Required tests

- Post-retirement rotation and enrollment attempts fail.
- Historical signatures continue to verify under their original policy epochs.
- Terminal ledger events cannot be removed or superseded.

## Non-claims

- Does not prevent creating a genuinely new catalog identity.
- Does not claim old algorithms remain secure forever.
