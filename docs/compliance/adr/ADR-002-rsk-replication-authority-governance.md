# ADR-002: Treat Replicator Safety Kernel Authority as Class A

**Date**: 2026-09-10  
**Status**: Proposed  
**Change Class**: A (Safety-Critical)

## Context

The Replicator Safety Kernel (RSK) introduces a deny-by-default authority boundary for bounded autonomous fabrication and future replicating systems. Its core rule is that creation does not confer authority: descendants require fresh external authorization and remain constrained by lineage, capability, population, depth, resource, containment, monitoring, evidence, and revocation invariants.

The repository's existing Class A detector predates RSK and therefore does not classify either of the new RSK crates or their dedicated verification workflow as safety-critical. A future change to these paths could consequently appear to the generic governance check as an ordinary change even though it can alter the conditions under which replication authority is granted or denied.

The generic governance script is currently advisory in CI for missing ADRs. Changing that behavior globally in the same tranche would have repository-wide effects unrelated to RSK. RSK therefore needs an explicit local governance gate while the broader governance policy remains unchanged.

This ADR governs the **reference safety-semantics stack**. It does not assert that RSK is production-admitted. Production admission remains explicitly denied until the separate admission gates have executed evidence.

## Decision

1. Treat these paths as Class A safety-critical surfaces:
   - `crates/domains/symthaea-replicator-safety/`
   - `crates/domains/symthaea-replicator-ledger/`
   - `.github/workflows/rsk-safety.yml`
2. Treat the generic Class A detector itself and the Governance Charter as Class A governance surfaces so weakening the classifier is visible as a safety-critical change.
3. Add an RSK-specific CI governance job that requires a changed ADR whenever RSK authority/ledger code or the RSK verification workflow changes in a pull request.
4. Keep documentation-only edits under `docs/architecture/replicator-safety/` visible to the RSK workflow but do not require a new ADR for every documentation correction. Normative implementation changes still require an ADR through the code/workflow path gate.
5. Keep the repository-wide Class A check advisory for now. Any future decision to make all Class A ADR requirements blocking should be a separate governance ADR.
6. Keep both RSK crates `publish = false` while production admission is denied.
7. Keep format, governance, and semantic verification as independent RSK workflow jobs so a style failure cannot suppress compile/test/Clippy evidence.
8. Treat exact intent → authorization → commit binding as a Class A production blocker. The normative contract is `docs/architecture/replicator-safety/RSK_EXACT_ACTION_BINDING_V0_1.md`; implementation is tracked in #1335.

### Parameter Changes

No physical, fabrication, biological, molecular, or replication-process parameter is changed.

| Parameter | Old Value | New Value | File |
|-----------|-----------|-----------|------|
| RSK authority/ledger Class A classification | Not classified | Class A | `scripts/check-class-a-changes.sh` |
| RSK code/workflow ADR requirement | Advisory/implicit | Blocking in focused RSK CI | `.github/workflows/rsk-safety.yml` |
| RSK package publication | Not explicitly blocked | `publish = false` while admission denied | RSK crate manifests |
| RSK validation job topology | Format before all semantic checks | Independent governance, format, and semantic jobs | `.github/workflows/rsk-safety.yml` |

### Scientific / Engineering Basis

This ADR is an engineering-governance decision, not a claim of certification.

Relevant architectural guidance includes:

- NASA runtime-assurance/Simplex work: keep the trusted safety path smaller and independently reasoned about than the advanced autonomy it constrains. RSK applies that separation to replication authority.
- IETF RFC 9943 (SCITT): separate signed claims, transparent registration, and independently verifiable receipts. RSK similarly separates authoritative local safety decisions from external transparency evidence.
- Existing Symthaea Governance Charter: Class A changes require stronger evidence and review because they affect safety-critical behavior.
- Existing Fabrication Kernel governance: successful authorization retains verified context and records domain-separated authorization-context evidence rather than reconstructing positive authority from loose execution-time caller values.

## Impact Analysis

### Downstream Systems Affected

- [x] Safety monitoring / runtime assurance
- [ ] Ethics evaluation (EthicsEngine)
- [ ] Consciousness scoring (ConsciousnessEquationV2)
- [x] Governance permissions / change control
- [ ] Learning dynamics (FEP, CfC)
- [x] Other: RSK authority, lineage/budget accounting, future durable evidence and recovery

### Risk Register Impact

This change reduces governance-bypass and validation-blindness risk for RSK. It does not create a physical replication capability.

Risks explicitly reduced:

- safety-critical authority code changed without an ADR;
- dedicated RSK verification weakened without governance visibility;
- future reviewers assuming the generic Class A detector already covered RSK;
- style/check ordering hiding whether authority code actually compiles or tests;
- reference crates becoming publishable before production admission;
- exact-action substitution being mistaken for ordinary budget exhaustion;
- autonomy/evidence features expanding faster than the safety governance surrounding them.

Residual risks / blockers:

- exact evaluated resource/action binding is not yet implemented (#1335);
- commit time is still a reference-model caller value rather than verified monotonic-time evidence;
- grants, quorum, and runtime witness are reference structs rather than verified production evidence types;
- durable canonical replay/CAS/checkpoint/fork handling is specified but not implemented;
- GitHub branch protection / required-check configuration could not be verified through the connected integration;
- runner availability can delay executed evidence.

## Evidence State

RSK evidence is intentionally split into three categories.

### A. Design / authored evidence

- [x] Constitutional safety model authored (#1232).
- [x] Reference authority semantics authored (#1233).
- [x] Reference lineage/budget ledger authored (#1239).
- [x] Durable evidence/replay contract authored (#1241).
- [x] Runtime assurance profile authored (#1326).
- [x] Production-admission matrix authored.
- [x] Exact action-binding contract authored.
- [x] Black-box integration tests authored for stale-cursor double commit, runtime-monitor veto, ancestral capability attenuation, ancestral population budgets, lineage quarantine, branch-policy non-widening, and duplicate delivery.

Authored evidence is not treated as executed evidence.

### B. Executed CI evidence

On 2026-09-11, focused RSK workflow run `34455150887` executed far enough to install the pinned Rust 1.96 toolchain and run the RSK formatting gate.

Observed result:

- [x] pinned Rust toolchain installation executed;
- [x] `cargo fmt --check` executed;
- [x] rustfmt reported formatting differences in the RSK Rust sources/tests;
- [ ] compile checks executed — **no; skipped because the sequential format step failed**;
- [ ] unit/integration tests executed — **no; skipped**;
- [ ] Clippy executed — **no; skipped**.

Therefore the run is evidence of a formatting defect and of the previous workflow's validation-order blind spot. It is **not** evidence of a Rust compile/test failure and is not represented as such.

The workflow has since been split into independent governance, format, and compile+test+Clippy jobs so future format failures cannot mask semantic verification.

### C. Required promotion evidence still missing

- [ ] focused RSK format gate green on exact candidate commit;
- [ ] focused RSK compile/test/Clippy gate green on exact candidate commit;
- [ ] repository-wide relevant checks green or explicitly scoped/explained;
- [ ] exact action-binding implementation and negative substitution tests green;
- [ ] trusted-time continuity boundary implemented and tested;
- [ ] verified grant/quorum/runtime-witness production boundary implemented and tested;
- [ ] canonical durable evidence/replay/CAS/checkpoint/fork tests green;
- [ ] generated state-machine/property testing green;
- [ ] protected-branch required-check enforcement independently verified.

No missing item is inferred from authored code or a queued/cancelled run.

## Production Admission

**DENIED / NOT YET ELIGIBLE.**

This ADR only establishes governance and evidence discipline around the reference model. It does not authorize RSK to control a hazardous or physically self-replicating system.

Production-admission requirements are tracked in:

- `docs/architecture/replicator-safety/RSK_PRODUCTION_ADMISSION_GATES_V0_1.md`
- #1335

## Rollback Plan

If this governance change causes an unintended CI failure unrelated to RSK safety semantics:

1. preserve this ADR and the RSK safety contract;
2. correct the path matcher or CI implementation narrowly;
3. do not remove RSK from Class A merely to regain CI availability;
4. if the blocking ADR gate itself must be replaced, land an equivalent or stronger RSK governance mechanism in the same change;
5. do not make either RSK crate publishable as a workaround.

Removing the authority/ledger Class A classification without a replacement should require a superseding Class A ADR.

## Consequences

### Positive

- RSK's safety-critical status becomes machine-visible rather than conventional knowledge.
- Authority and ledger code changes receive an enforceable ADR requirement in their dedicated CI lane.
- The governance detector becomes self-visible as a safety-critical surface.
- Authored, executed, and missing evidence are explicitly separated.
- A formatting failure can no longer hide semantic verification in the focused workflow.
- Reference crates cannot quietly become release artifacts while production admission is denied.
- The exact action-binding defect is promoted from an informal observation to a normative Class A blocker.
- The safety boundary is less dependent on institutional memory.

### Negative / Cost

- RSK code changes carry additional documentation/review overhead.
- Contributors must update or add an ADR for authority/ledger implementation changes.
- The focused CI lane consumes separate runner jobs for format and semantic verification.
- Production promotion remains intentionally blocked while evidence is incomplete.

The added friction is intentional for code that can change replication-authority semantics.
