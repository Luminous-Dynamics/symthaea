# VART-WORLD-CREATIVE-001 — Confirmatory Source Closure v1

Status: transition/qualification contract. It cannot authorize confirmatory execution or scientific claims.

## Purpose

A boolean such as `confirmatory_source_fetchable = true` is not sufficient evidence. Before a confirmatory freeze is prepared, the exact source lineage intended for confirmatory execution must be durably reachable and independently reproducible from a fresh checkout.

The source-closure receipt binds four distinct facts:

1. **identity** — the exact confirmatory `HEAD` and `TREE`;
2. **remote reachability** — a named repository/ref resolves to that exact commit;
3. **fresh-checkout equivalence** — an independently created checkout resolves to the same `HEAD` and `TREE`;
4. **reproduction context** — the source is bound to an environment digest, lock/reproduction manifest, and qualification receipt.

## Receipt schema

`schema = "symthaea.vart-world-creative-001.source-closure.v1"`

Required top-level fields:

- `experiment_id = "VART-WORLD-CREATIVE-001"`
- `status = "qualified"`
- `confirmatory_execution_authorized = false`
- `claim_authorized = false`

### `confirmatory_source`

- `head` — 40-hex commit ID;
- `tree` — 40-hex tree ID;
- `parent_v05a_head`;
- `parent_v05a_tree`.

### `pilot_predecessor`

- `head`;
- `tree`;
- `is_ancestor_of_confirmatory_source = true`.

Instrumentation-only pilot fixes may create a new descendant source HEAD. This is permitted only when the post-pilot disposition classifies the change as plumbing and does not alter the frozen scientific mechanism/contract. Scientific-mechanism or scientific-contract changes require a new preregistration lineage.

### `remote`

- `repository_full_name`;
- `ref` — durable branch/tag/ref used for retrieval;
- `fetch_verified = true`;
- `fetched_head`;
- `fetched_tree`;
- `fresh_checkout_verified = true`;
- `fresh_checkout_head`;
- `fresh_checkout_tree`.

All fetched/fresh identities must equal `confirmatory_source.head/tree`.

### `reproduction`

- `environment_digest` — 64-hex digest of the execution environment identity;
- `lock_manifest_sha256` — 64-hex digest of the source/toolchain/lock manifest;
- `qualification_receipt_sha256` — 64-hex digest of the qualification receipt for the intended source;
- `independent_checkout_gate = true`.

The reproduction block proves identity/context closure, not semantic scientific efficacy.

## External binding

The raw SHA-256 of the completed source-closure receipt is recorded by the freeze-eligibility receipt and later by the prospective confirmatory freeze. Editing the source-closure receipt therefore creates a different transition lineage.

## Claim boundary

A source-closure PASS establishes only that the exact intended source is fetchable and context-bound. It does not establish that the VART hypothesis is true, that the pilot was scientifically valid, or that confirmatory execution is authorized.
