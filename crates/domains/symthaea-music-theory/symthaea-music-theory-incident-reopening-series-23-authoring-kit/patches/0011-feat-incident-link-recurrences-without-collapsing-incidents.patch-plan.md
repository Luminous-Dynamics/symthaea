# Patch 0011: feat incident link recurrences without collapsing incidents

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Relate repeated or connected incidents while preserving each incident's independent evidence and authority history.

## Intended changes

- Add recurrence links with predecessor incident, relationship class, supporting evidence, and verifier policy.
- Support same-root-cause-suspected, same-authority-compromise, same-branch-family, repeated-equivocation, and unknown relationship classes.
- Keep recurrence assertions separate from proven facts.

## Required tests

- Self-links, cycles forbidden by policy, target substitution, and unsupported classes fail.
- Two incidents remain independently auditable.
- Relationship changes append new events rather than rewriting earlier assertions.

## Non-claims

- Does not prove common human responsibility.
- Does not merge incident IDs or erase distinct closures.
