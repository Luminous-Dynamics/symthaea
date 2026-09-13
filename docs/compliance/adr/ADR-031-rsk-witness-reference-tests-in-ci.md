# ADR-031: Execute Xenia witness trust reference suites in focused RSK CI

**Status**: Accepted
**Change Class**: A
**Scope**: Replicator Safety Kernel qualification only

## Decision

The focused `RSK Class A ADR gate` job MUST execute the standalone reference suites for:

1. Xenia verified-key -> RSK key/signer/failure-domain trust-context reconstruction; and
2. exact signed-commitment `trust_context_digest` admission.

The workflow MUST continue to treat these suites as reference/semantic evidence only. Passing them does not establish Xenia cryptographic correctness, trusted time, durable external retention, crash-safe persistence, runtime admission, or replication authority.

## Required commands

```text
python3 docs/architecture/replicator-safety/reference/test_rsk_xenia_witness_trust.py
python3 docs/architecture/replicator-safety/reference/test_rsk_xenia_witness_admission.py
```

The workflow self-protection check MUST retain both commands.

## Rationale

The two suites are Class A because they encode the boundary between verified cryptographic keys, governed signer/failure-domain independence, and the trust context cryptographically bound into a Xenia state commitment. Leaving them authored but unexecuted would allow this boundary to drift without blocking qualification.

## Non-claims

This ADR does not mark either suite as executed until an exact-head workflow run actually completes successfully. Production admission remains **DENIED / NOT YET ELIGIBLE**.
