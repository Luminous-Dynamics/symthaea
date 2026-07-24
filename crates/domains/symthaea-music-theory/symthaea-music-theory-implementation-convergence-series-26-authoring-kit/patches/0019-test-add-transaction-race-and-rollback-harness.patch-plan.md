# Patch 0019: test add transaction race and rollback harness

**Series:** 26

## Objective

Exercise every state-changing transition under deterministic races and injected failure.

## Intended changes

- Race publication versus freeze, freeze versus recovery, recovery versus retirement, two first mutations, two branch selections, and two terminal transitions.
- Inject failures before and after each staged write.
- Record exact scenario seeds and expected state hashes.

## Required tests

- Zero or one conflicting transition commits.
- Failed attempts leave byte-identical pre-state.
- Successful output passes cumulative audit.

## Non-claims

- Does not benchmark distributed consensus.
- Does not rely on nondeterministic sleeps.
