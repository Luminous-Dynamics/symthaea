# Patch 0020: security apply series 24 resource and privacy bounds to new code

**Series:** 26

## Objective

Ensure new lifecycle artifacts cannot bypass hostile-input and privacy constraints.

## Intended changes

- Apply byte, depth, collection, reference, signature, verifier-call, archive, subprocess, and output limits to every new decoder and command.
- Classify prohibited private fields and deterministic redaction behavior.
- Keep exact artifact identifiers out of unbounded metric labels.

## Required tests

- Archive bombs, path traversal, deep nesting, fan-out, oversized strings, and secret-field exports fail safely.
- Limit failures produce no partial authoritative state.
- Valid worst-case artifacts remain processable within budget.

## Non-claims

- Does not select universal deployment limits.
- Does not guarantee anonymity against all auxiliary information.
