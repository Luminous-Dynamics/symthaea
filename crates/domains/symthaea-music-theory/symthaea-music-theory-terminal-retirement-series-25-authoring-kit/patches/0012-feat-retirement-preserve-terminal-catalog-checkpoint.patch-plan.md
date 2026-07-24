# Patch 0012: feat retirement preserve terminal catalog checkpoint

**Series:** 25

## Objective

Create a final checkpoint over the exact retired catalog and complete lifecycle history.

## Intended changes

- Bind final catalog head, global ordinals, active and terminal segment states, cycle ledger, incident ledger, authority ledgers, revocations, retirement receipt, manifests, and claim matrix.
- Permit optional external attestations without treating them as retirement authority.
- Support independent offline reconstruction.

## Required tests

- Any omitted or substituted lifecycle object breaks the checkpoint.
- Checkpoint verification distinguishes structural completeness from external authentication.
- Local mtime is never used as retirement time proof.

## Non-claims

- Does not prove no later unauthorized copy exists.
- Does not establish universal canonicality.
