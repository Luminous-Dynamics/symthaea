# Patch 0007: feat implement resumption plan

**Series:** 31

## Objective

Build one exact plan binding current state and intended first mutation.

## Intended changes

- Bind segment, closure, certification, exact pre-head, authority epochs, witness policy, delegation constraints, allowance constraints, channel, and publication content identity.
- Embed limitations separating plan from authorization and commit.
- Expose structural validation.

## Acceptance evidence

- Stale head, wrong segment, wrong channel, expired plan, and missing limitation variants fail.
- Every semantic mutation changes canonical bytes.
- A plan cannot mutate state.

## Non-claims

- Does not authorize the plan.
- Does not execute the publication.
