# ADR-002: Treat Replicator Safety Kernel Authority as Class A

**Date**: 2026-09-10  
**Status**: Proposed  
**Change Class**: A (Safety-Critical)

## Context

The Replicator Safety Kernel (RSK) introduces a deny-by-default authority boundary for bounded autonomous fabrication and future replicating systems. Its core rule is that creation does not confer authority: descendants require fresh external authorization and remain constrained by lineage, capability, population, depth, resource, containment, monitoring, evidence, and revocation invariants.

The repository's existing Class A detector predates RSK and therefore does not classify either of the new RSK crates or their dedicated verification workflow as safety-critical. A future change to these paths could consequently appear to the generic governance check as an ordinary change even though it can alter the conditions under which replication authority is granted or denied.

The generic governance script is currently advisory in CI for missing ADRs. Changing that behavior globally in the same tranche would have repository-wide effects unrelated to RSK. RSK therefore needs an explicit local governance gate while the broader governance policy remains unchanged.

## Decision

1. Treat these paths as Class A safety-critical surfaces:
   - `crates/domains/symthaea-replicator-safety/`
   - `crates/domains/symthaea-replicator-ledger/`
   - `.github/workflows/rsk-safety.yml`
2. Treat the generic Class A detector itself and the Governance Charter as Class A governance surfaces so weakening the classifier is visible as a safety-critical change.
3. Add an RSK-specific CI governance job that requires a changed ADR whenever RSK authority/ledger code or the RSK verification workflow changes in a pull request.
4. Keep documentation-only edits under `docs/architecture/replicator-safety/` visible to the RSK workflow but do not require a new ADR for every documentation correction. Normative implementation changes still require an ADR through the code/workflow path gate.
5. Keep the repository-wide Class A check advisory for now. Any future decision to make all Class A ADR requirements blocking should be a separate governance ADR.

### Parameter Changes

No physical, fabrication, biological, molecular, or replication-process parameter is changed.

| Parameter | Old Value | New Value | File |
|-----------|-----------|-----------|------|
| RSK authority/ledger Class A classification | Not classified | Class A | `scripts/check-class-a-changes.sh` |
| RSK code/workflow ADR requirement | Advisory/implicit | Blocking in focused RSK CI | `.github/workflows/rsk-safety.yml` |

### Scientific / Engineering Basis

This ADR is an engineering-governance decision, not a claim of certification.

Relevant architectural guidance includes:

- NASA runtime-assurance/Simplex work: keep the trusted safety path smaller and independently reasoned about than the advanced autonomy it constrains. RSK applies that separation to replication authority.
- IETF RFC 9943 (SCITT): separate signed claims, transparent registration, and independently verifiable receipts. RSK similarly separates authoritative local safety decisions from external transparency evidence.
- Existing Symthaea Governance Charter: Class A changes require stronger evidence and review because they affect safety-critical behavior.

## Impact Analysis

### Downstream Systems Affected

- [x] Safety monitoring / runtime assurance
- [ ] Ethics evaluation (EthicsEngine)
- [ ] Consciousness scoring (ConsciousnessEquationV2)
- [x] Governance permissions / change control
- [ ] Learning dynamics (FEP, CfC)
- [x] Other: RSK authority, lineage/budget accounting, future durable evidence and recovery

### Risk Register Impact

This change reduces governance-bypass risk for RSK. It does not create a physical replication capability.

Risks explicitly reduced:

- safety-critical authority code changed without an ADR;
- dedicated RSK verification weakened without governance visibility;
- future reviewers assuming the generic Class A detector already covered RSK;
- autonomy/evidence features expanding faster than the safety governance surrounding them.

Residual risk:

- GitHub runner unavailability can still prevent the gate from executing. Branch protection / required-check policy is an external repository setting and is not asserted by this ADR.

## Test Evidence

- [x] Black-box integration tests authored for stale-cursor double commit, runtime-monitor veto, ancestral capability attenuation, ancestral population budgets, lineage quarantine, branch-policy non-widening, and duplicate delivery.
- [x] Focused RSK CI lane authored with pinned Rust 1.96 format/check/test/Clippy across all RSK targets.
- [ ] RSK focused CI execution confirmed green — pending GitHub runner availability at time of this ADR.
- [ ] Repository-wide CI execution confirmed green — pending GitHub runner availability at time of this ADR.
- [ ] Generated state-machine/property testing — follow-up promotion gate before durable RSK graduation.

No test is represented as executed when it has not executed.

## Rollback Plan

If this governance change causes an unintended CI failure unrelated to RSK safety semantics:

1. preserve this ADR and the RSK safety contract;
2. correct the path matcher or CI implementation narrowly;
3. do not remove RSK from Class A merely to regain CI availability;
4. if the blocking ADR gate itself must be replaced, land an equivalent or stronger RSK governance mechanism in the same change.

Removing the authority/ledger Class A classification without a replacement should require a superseding Class A ADR.

## Consequences

### Positive

- RSK's safety-critical status becomes machine-visible rather than conventional knowledge.
- Authority and ledger code changes receive an enforceable ADR requirement in their dedicated CI lane.
- The governance detector becomes self-visible as a safety-critical surface.
- The safety boundary is less dependent on institutional memory.
- Future durable-evidence, replay, and recovery work has an explicit governance foundation.

### Negative / Cost

- RSK code changes carry additional documentation/review overhead.
- Contributors must update or add an ADR for even small authority/ledger implementation changes.
- The focused CI gate remains unavailable while GitHub runners are unavailable; this ADR does not solve infrastructure capacity.

The added friction is intentional for code that can change replication-authority semantics.
