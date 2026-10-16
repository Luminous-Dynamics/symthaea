# INTX-PROGRAM-BENCH-001: API Ownership and Readiness Matrix

**Audit date:** 2026-10-16  
**Baseline branch:** `main`  
**Baseline commit:** `5e83cea4986ba3845aa50e0d5dd03493c5adeddf`  
**Corpus:** `crates/intx-program-bench/fixtures/cases.tsv` (50 frozen seed scenarios)  
**Purpose:** prevent a test fixture from being mistaken for an implemented or qualified production API.

## Finding

The repository contains detailed INTX design issues for interaction identity, authority composition, credential brokerage, event normalization, reconciliation, protocol adapters, preflight, and connector qualification. During this audit, all listed owner issues below were still open. An open issue alone does not prove that no related code exists, but a proposed issue contract is not source-level evidence that the API exists, is consumed by the target path, or has passed qualification.

Code-search queries for the specific proposed core names `ResourceRef`, `InteractionIntent`, `ExternalObservation`, and `DispatchPermit` did not return a matching implementation on the audited default branch. Treat core/API integration as **not yet demonstrated** until source and tests are verified at an exact commit.

Related owners checked:
- [INTX-001 / #4399](https://github.com/Luminous-Dynamics/symthaea/issues/4399) — pure identity/intent/observation core
- [INTX-002 / #4400](https://github.com/Luminous-Dynamics/symthaea/issues/4400) — authority and action-runtime binding
- [INTX-003 / #4401](https://github.com/Luminous-Dynamics/symthaea/issues/4401) — credential and signer broker contracts
- [INTX-004 / #4402](https://github.com/Luminous-Dynamics/symthaea/issues/4402) — replay-resistant external observations
- [INTX-005 / #4403](https://github.com/Luminous-Dynamics/symthaea/issues/4403) — reconciliation/idempotency/unknown outcomes
- [INTX-006 / #4404](https://github.com/Luminous-Dynamics/symthaea/issues/4404) — non-authoritative operational traces
- [INTX-WEB-001 / #4405](https://github.com/Luminous-Dynamics/symthaea/issues/4405) — bounded connector reference implementation
- [INTX-IND-001 / #4415](https://github.com/Luminous-Dynamics/symthaea/issues/4415) — OPC UA profile
- [INTX-MODBUS-001 / #4432](https://github.com/Luminous-Dynamics/symthaea/issues/4432) — Modbus profile
- [INTX-008 / #4445](https://github.com/Luminous-Dynamics/symthaea/issues/4445) — preflight and simulation semantics
- [INTX-016 / #4494](https://github.com/Luminous-Dynamics/symthaea/issues/4494) — connector conformance SDK
- [OT/cyber-physical security / #2187](https://github.com/Luminous-Dynamics/symthaea/issues/2187) — OT safety/security boundaries
- [MFG-PROC-000 / #5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686) — engineering process/recipe semantics

## Corpus ownership map

“Ready” in the last column means ready to validate the *fixture data shape only*, not ready to claim production API behavior.

| Fixture IDs | Primary ownership/dependency | What the scenario is intended to detect | Current status |
|---|---|---|---|
| C0-01..C0-08, C0-10 | INTX-001/#4399; OPC UA #4415 or Modbus #4432; freshness/currentness semantics | Target/profile identity, namespace collision, stale data, endpoint changes, and untrusted discovery | API mapping and exact-source qualification required |
| C0-09 | Read-only observation profile plus canonical identity | Valid, fresh, bounded inventory observation | Fixture-shape validation only; protocol behavior deferred |
| C1-01..C1-08 | Equipment artifact/provenance semantics; MFG-PROC-000/#5686 where applicable | Source/dependency/digest/target/constraint errors block export or commissioning | Offline fixture only; compiler/artifact backend not implemented by this corpus |
| C1-09..C1-10 | Offline artifact pipeline and pinned toolchain | Valid candidate export/build/validation control cases | Expected-state fixture only; no actual vendor compiler/backend invoked |
| C2-01..C2-09 | INTX-002/#4400; credential broker #4401; preflight #4445; OT safety #2187 | Missing/revoked approval, stale window/precondition, protected function, data-egress and metadata authority confusion | Dispatch gate/API readiness not proven by this PR; must be blocked/deferred until exact owner paths qualify |
| C2-10 | Preflight semantics #4445 composed with authority #4400 | Offline preflight allowed without dispatch | Fixture only; no dispatch permit may be inferred |
| C3-01..C3-10 | INTX-004/#4402; INTX-005/#4403; protocol owner #4415/#4432; connector conformance #4494 | Replay, timeout, duplicate response, endpoint swap, readback mismatch, uncertain outcome and reconciliation | Runtime outcome behavior not proven by the fixture harness |
| C4-01..C4-10 | INTX-005/#4403; trace/evidence #4404; conformance #4494; OT security #2187 | Supersession, approval expiry, sensitive logging, rollback incompatibility, receipts and requalification | Lifecycle/receipt owner integration required |

## Status vocabulary for follow-on work

Every seed must eventually be tagged with exactly one implementation status in the runtime-mapping report:

- `EXECUTABLE_QUALIFIED`: exact API/path exists and required focused tests passed at the exact source head.
- `EXECUTABLE_UNQUALIFIED`: code exists, but required qualification evidence is absent or failing.
- `DESIGN_ONLY`: the owner contract is still a proposal or no matching API was found in the audited source.
- `DEFERRED_DEPENDENCY`: test depends on a named predecessor that is not yet qualified.
- `FIXTURE_ONLY`: corpus structure can be checked, but no runtime implementation is under test.

Never convert `DESIGN_ONLY`, `DEFERRED_DEPENDENCY`, skipped, queued, unavailable, or unrun tests into passes. Missing authorization gate code is itself a blocker, not a bypass permission.

## Next implementation decision

1. At a fresh exact branch head, inspect the relevant code and tests for every owner before asserting source readiness. The default-branch search above is a point-in-time audit, not a permanent absence proof.
2. Publish the exact source path/symbol, tests, and qualification artifact for each `EXECUTABLE_QUALIFIED` classification.
3. Keep C0/C1 fixture-data checks running now. Defer runtime assertions whose INTX or domain APIs are still design-only.
4. When INTX-001/#4399 and INTX-002/#4400 have qualified successors, bind only the relevant cases to those real APIs; keep the oracle independent and mutation-test it.
5. Do not add a protocol adapter or dispatch implementation as a shortcut around these prerequisites.

## Claim ceiling

This matrix is a repository/source-readiness audit plus a proposed ownership mapping. It does not show runtime conformance, OPC UA interoperability, any actual PLC capability, functional-safety assurance, medical-device compliance, or field qualification.
