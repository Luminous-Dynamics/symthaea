# Patch 0015: Split release stages and bounded evidence capture

**Series:** 23

## Objective

Keep the release orchestrator auditable and avoid one opaque script.

## Intended changes

- Separate acquisition, replay, build, conformance, packaging, and claim-generation modules.
- Use typed stage results and stable failure categories.
- Bound captured output and redact environment secrets.

## Required tests

- Each stage can be replayed independently.
- Failure evidence is deterministic where inputs are deterministic.
- No stage marks a skipped dependency as success.

## Non-claims

- Does not create new publication authority.
- Does not claim support for lanes that were not executed.
