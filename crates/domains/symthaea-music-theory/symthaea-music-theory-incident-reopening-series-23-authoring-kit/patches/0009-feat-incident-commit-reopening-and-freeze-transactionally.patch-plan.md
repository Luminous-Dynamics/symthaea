# Patch 0009: feat incident commit reopening and freeze transactionally

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Apply segment freeze, reopening ledger append, and quarantine changes as one atomic transition.

## Intended changes

- Reverify the adverse evidence, authorization, active policies, current head, and closure status at commit time.
- Stage all changes before mutation and return exact post-state.
- Reject any catalog mutation that races the expected freeze head.

## Required tests

- Injected failures leave byte-identical pre-state.
- A simultaneous publication and freeze cannot both commit from one head.
- Successful output passes closure-history, segment, challenge, quarantine, and reopening audits.

## Non-claims

- Does not implement network consensus.
- Does not guarantee external notification delivery.
