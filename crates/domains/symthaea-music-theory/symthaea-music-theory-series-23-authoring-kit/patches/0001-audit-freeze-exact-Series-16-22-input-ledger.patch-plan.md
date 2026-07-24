# Patch 0001: Audit and freeze the exact Series 16–22 input ledger

**Series:** 23

## Objective

Create a machine-readable inventory of every required base tree, patch archive, source archive, outer checksum, and expected final tree before cumulative replay begins.

## Intended changes

- Add a strict ledger schema with fixed-width fields and lowercase hexadecimal digests.
- Require external archive digests rather than trusting only manifests stored inside archives.
- Reject duplicate series numbers, missing predecessors, ambiguous filenames, and unpinned baselines.

## Required tests

- Accept one complete exact chain.
- Reject a missing Series 17 manifest or any substituted archive digest.
- Reject a chain whose advertised predecessor tree does not match the replayed tree.

## Non-claims

- Does not create new publication authority.
- Does not claim support for lanes that were not executed.
