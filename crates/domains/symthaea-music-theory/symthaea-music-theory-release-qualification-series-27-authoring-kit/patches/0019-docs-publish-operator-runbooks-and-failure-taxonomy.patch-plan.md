# Patch 0019: docs publish operator runbooks and failure taxonomy

**Series:** 27

## Objective

Make failures actionable without blurring semantic meaning.

## Intended changes

- Document stable stages and issue codes for structural, authentication, policy, lineage, state, resource, transaction, privacy, compatibility, and retirement failures.
- Map each to safe diagnostic and next-step procedures.
- Distinguish retryable, stale-input, authority-required, corrupted, unsupported, and terminal conditions.

## Required tests

- Every emitted stable code appears in generated runbooks.
- No runbook suggests bypassing evidence checks.
- Examples use synthetic identities only.

## Non-claims

- Does not automate governance decisions.
- Does not make operational advice authority.
