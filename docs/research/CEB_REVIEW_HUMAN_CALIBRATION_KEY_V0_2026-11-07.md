# CEB Review Human Calibration Key V0 — 2026-11-07

**Status:** Public development answer key for [Human Calibration Pack V0](CEB_REVIEW_HUMAN_CALIBRATION_PACK_V0_2026-11-07.md). Not hidden ground truth, a task oracle, or sealed benchmark material.  
**Release procedure:** The maintainer may withhold this file from reviewers until both independent worksheets are received, but because it is public, it offers no cryptographic blinding or holdout protection.

## 1. Variant mapping

| Case | Variant | Causal state | Target mechanism live? | Rationale |
|---|---|---|---|---|
| CAL-01 | X1 | N — neighboring mechanism | No | Uses read permission rather than write permission. This is a separate authorization flaw, not the target write-before-check mechanism. |
| CAL-01 | X2 | F — fixed | No | Write permission is checked before persistent mutation. |
| CAL-01 | X3 | I — introducing | Yes | Persistent write occurs before the write-permission check; later denial does not undo the mutation. |
| CAL-01 | X4 | M — minimal causal intervention | No | The write occurs only on the authorized branch. The necessary ordering/bypass condition is removed. |
| CAL-01 | X5 | IRR — irrelevant intervention | Yes | Audit/error wording changes, but the persistent write still precedes authorization. |
| CAL-01 | X6 | P — parent | No | Permission check precedes the write, as required. |
| CAL-02 | X1 | I — introducing | Yes | Allocation proportional to the untrusted declared length happens before the maximum-size check. |
| CAL-02 | X2 | F — fixed | No | Maximum bound is checked before proportional allocation. |
| CAL-02 | X3 | N — neighboring mechanism | No | Rejects an empty payload contrary to the normative contract, but the target unbounded-allocation ordering is absent. This is a valid non-target correctness defect. |
| CAL-02 | X4 | IRR — irrelevant intervention | Yes | Checksum validation is added, but the oversized allocation still precedes the size bound. |
| CAL-02 | X5 | M — minimal causal intervention | No | Checked conversion and maximum bound precede capacity allocation. |
| CAL-02 | X6 | P — parent | No | Size conversion and maximum bound precede allocation; target mechanism absent. |
| CAL-03 | X1 | P — parent | No | Cache is nested by tenant and lookup passes tenant identity. |
| CAL-03 | X2 | I — introducing | Yes | Global cache key omits tenant identity and caller cannot partition the lookup by tenant. |
| CAL-03 | X3 | F — fixed | No | Tenant is included in the cache key and caller. |
| CAL-03 | X4 | N — neighboring mechanism | No | Tenant identity is compared against the cached value before reusing the decision; no cross-tenant hit. |
| CAL-03 | X5 | IRR — irrelevant intervention | Yes | No tenant is included in the key; adding policy evaluation on a miss does not fix cross-tenant reuse on a hit. |
| CAL-03 | X6 | M — minimal causal intervention | No | Minimal key expansion adds tenant identity; expiry metadata is orthogonal. |

## 2. Key reasoning

### CAL-01

The distinguishing event is the persistent `db.put`, not the returned `Result`. Returning `Err(Denied)` after writing does not roll back the side effect. X1 demonstrates a nearby but different failure: read permission does not imply write permission. It must not be credited as the target write-before-check finding, though it may be a valid independently actionable authorization finding.

### CAL-02

The normative contract specifically says the maximum must be enforced before any allocation proportional to attacker-controlled length. A check after `vec!`, `with_capacity` or `resize` is too late. X3's empty-payload rejection is a separate defect because empty payloads are allowed; it does not activate the target large-allocation mechanism. X4 adds checksum validation but leaves the allocation-order defect intact. Distinguish a useful non-target finding from a target hit.

### CAL-03

The relevant unit is the cross-file request/cache path. X2 has a shared global map keyed only by user/resource; because the task contract permits the same identifiers in separate tenants, a cached decision can cross the tenant boundary. X5 preserves the same unsafe key; evaluating policy only on a cache miss does not protect a cache hit from another tenant. X1's nested per-tenant map and X4's tenant check both prevent the target cross-tenant reuse.

## 3. Scoring guidance

This key is not a new metric. Record:
- target-mechanism classification over 18 variants;
- whether each reviewer identified the correct distinct non-target issue in CAL-01 X1 and CAL-02 X3;
- whether causal reasoning mentions the decisive path/operation, not only a location;
- whether the reviewer overclaims runtime/compiler evidence;
- unresolved judgments and adjudication time.

Do not report a benchmark score or generalize from 18 intentionally small synthetic variants. The useful result is the disagreement log and rubric ambiguities.
