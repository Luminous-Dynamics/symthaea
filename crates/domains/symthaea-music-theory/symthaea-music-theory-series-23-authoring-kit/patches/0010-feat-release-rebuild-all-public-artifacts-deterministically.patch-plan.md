# Patch 0010: Rebuild all public artifacts deterministically

**Series:** 23

## Objective

Produce public archives and manifests from tracked inputs with canonical metadata.

## Intended changes

- Normalize file order, path names, ownership, permissions, timestamps, gzip headers, and locale.
- Generate internal manifests and an external outer digest list.
- Forbid undeclared files and private records in public kits.

## Required tests

- Two independent builds are byte-identical.
- A timestamp or file-order perturbation is normalized.
- A private-field canary blocks publication.

## Non-claims

- Does not create new publication authority.
- Does not claim support for lanes that were not executed.
