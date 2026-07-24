# Patch 0006: feat retirement model terminal retirement receipt

**Series:** 25

## Objective

Bind the final authority transition to exact pre-state, actions, and archive-only result.

## Intended changes

- Add planned and committed receipt forms.
- Bind trigger report, authorization, catalog head, segment and cycle states, delegation and allowance revocations, endpoint decommission plan, archive policy, and post-state.
- Include mandatory non-reversibility and non-deletion limitations.

## Required tests

- Any pre-state, action, or post-state mutation breaks the receipt.
- A planned receipt cannot masquerade as committed retirement.
- The committed receipt has one exact catalog lineage.

## Non-claims

- Does not prove keys were physically destroyed.
- Does not make local shutdown a trusted global timestamp.
