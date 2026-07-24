# Patch 0005: Close target, module, and feature integration gaps

**Series:** 23

## Objective

Repair only those integration defects that prevent the documented Series 16–22 surface from compiling as declared.

## Intended changes

- Add missing module declarations, dependencies, feature edges, and required-features discovered by the inventory.
- Remove dead or accidental targets rather than hiding them behind broad default features.
- Keep each correction traceable to an existing public contract or tool.

## Required tests

- Default feature build passes.
- No-default-features build passes where supported.
- Every declared feature and target appears in at least one successful lane.

## Non-claims

- Does not create new publication authority.
- Does not claim support for lanes that were not executed.
