# ADR-032: Harden RSK witness-suite CI self-check against self-satisfaction

- **Status**: Accepted
- **Change Class**: A
- **Scope**: Replicator Safety Kernel qualification only

## Decision

The focused RSK workflow MUST verify that the Xenia witness trust suites remain present as actual executable `run:` steps. A self-check that searches for the raw command string anywhere in the workflow is insufficient, because the command can appear inside the self-check's own expected-command list and therefore satisfy the check after the real execution step has been deleted.

The self-check MUST instead require the exact executable YAML line for each suite:

```text
run: python3 docs/architecture/replicator-safety/reference/test_rsk_xenia_witness_trust.py -v
run: python3 docs/architecture/replicator-safety/reference/test_rsk_xenia_witness_admission.py -v
```

The named execution steps MUST also remain present.

## Rationale

Qualification machinery is itself Class A. A self-referential assertion that can pass after the protected action is removed is not a valid guard. This hardening makes the workflow prove the protected commands are actually wired for execution rather than merely mentioned as text.

## Non-claims

This ADR does not establish that either suite has executed successfully. Exact-head workflow completion remains required before treating the suites as executed evidence. Production admission remains **DENIED / NOT YET ELIGIBLE**.
