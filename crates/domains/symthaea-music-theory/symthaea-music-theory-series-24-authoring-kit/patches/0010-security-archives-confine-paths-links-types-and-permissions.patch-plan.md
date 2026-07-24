# Patch 0010: Confine archive paths, links, file types, and permissions

**Series:** 24

## Objective

Prevent filesystem escape and special-file attacks during optional extraction.

## Intended changes

- Reject absolute paths, `..`, empty components, platform-prefix paths, symlinks, hardlinks, devices, sockets, and FIFOs by default.
- Extract through a capability-scoped destination with create-new semantics.
- Normalize or reject unsafe permission bits.

## Required tests

- Traversal corpus cannot write outside destination.
- Link chains cannot escape.
- Pre-existing destination files are not overwritten silently.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.
