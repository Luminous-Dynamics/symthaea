# Patch 0021: test incident cover reopen authorization freeze and races

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Prove governed reopening and atomic freeze under replay and concurrency.

## Intended changes

- Cover reused closure signatures, stale recovery authorities, wrong witnesses, duplicate signers, quarantine changes, publication/freeze race, and repeated commit.
- Inject failure at each transaction stage.
- Verify post-freeze mutation rejection across every publication path.

## Required tests

- Zero or one of publication and freeze commits from the same head.
- Failed attempts leave byte-identical pre-state.
- A successful freeze preserves all prior closure and resumption evidence.

## Non-claims

- Does not benchmark distributed writer contention.
- Does not certify external storage durability.
