# Aesthetic Operational Resilience Patch Series

This series continues from the federated-stewardship baseline and adds the
contracts needed to rebuild releases independently, constrain extractor
execution, investigate failures accountably, restore critical evidence, and
block deployment when operational resilience has not been demonstrated.

The aesthetic scoring model is intentionally unchanged. This wave strengthens
the machinery around the model rather than adding another heuristic.

## Security truth

The crate still uses stable FNV-derived identifiers for deterministic replay and
cross-linking. They are not cryptographic hashes, signatures, or MACs. Build,
runtime, incident, recovery, and release records can be bound to external
attestations through the existing trust interfaces, but this crate does not
implement signing algorithms, key custody, process isolation, containerization,
or backup storage.

A declared capability manifest is also not an operating-system sandbox. It
becomes security evidence only when an external executor actually enforces the
manifest and records observed access, denials, resource use, termination, and
subject identity.

## Bundle A — Supply-chain integrity

1. Canonical build recipes, materials, software bills of materials, and release
   provenance.
2. Independent build replicas and reproducibility criteria.
3. A fail-closed supply-chain release assessment.
4. Recomputed canonical digests and derived metrics.
5. Complete, unique finding-set validation.

Important properties:

- Build provenance binds the release, recipe, SBOM, and produced output.
- Production policy requires pinned dependency-lock and toolchain materials.
- Reproducibility distinguishes replica count from independent builder count.
- Recipe, SBOM, and output agreement are evaluated separately.
- A favorable serialized status cannot replace the complete source evidence.
- License omissions, prohibited components, missing materials, or output
  substitution block release under production criteria.

## Bundle B — Capability sandboxing

6. Declarative capability grants and resource budgets.
7. Runtime access observations and sandbox-compliance reports.
8. A runtime-security bundle tying the extractor descriptor, build provenance,
   supply-chain assessment, manifest, execution, and report together.
9. Complete, unique sandbox finding validation.

The sandbox evidence model records default-deny behavior, declared and
undeclared access, denied attempts, required-grant use, process outcome,
policy-triggered termination, and resource consumption. Exact digests prevent a
compliant report from being replayed with a different extractor build,
capability manifest, or execution record.

## Bundle C — Incident response

10. Append-only incident cases with constrained state transitions.
11. Response plans, action ownership, deadlines, and resolution evidence.
12. Post-incident reviews, causal factors, corrective actions, and recurrence
    tests.
13. Closure gates that require report, timeline, action, and root-cause
    agreement.
14. Complete, unique response and closure findings.

A closed incident is not represented by a mutable boolean. Closure is a derived
claim requiring the final incident state, completed response actions,
resolution evidence, a review, reviewer diversity, corrective-action status,
and recurrence-test results to agree.

## Bundle D — Disaster recovery and release closure

15. Backup manifests covering objects, tiers, encryption, immutability,
    retention, and failure domains.
16. Restore drills with measured recovery-point and recovery-time evidence.
17. Service-continuity plans and readiness assessment.
18. Complete, unique recovery and continuity findings.
19. A unified operational-resilience release bundle.
20. New trust roles and attestation scopes for builds, runtime execution,
    incidents, recovery, and release resilience.
21. Public API generation `1.3.0` and 21 new persisted-schema families.
22. Prelude, architecture, and compatibility publication.
23. Final derived-report forgery hardening.
24. This application and security guide.

The authored Git history contains additional narrowly scoped hardening commits
inside these four bundles. Apply every numbered mail patch in bundle order.

## Unified release decision

The final gate combines:

```text
authenticated build inputs and output lineage
        +
independent reproducibility evidence
        +
extractor capability and runtime observations
        +
incident status and closure evidence
        +
backup, restore, and continuity readiness
        +
trusted transparency checkpoint
        =
operational resilience outcome
```

The outcome is `Ready`, `Conditional`, or `Blocked`. Production criteria fail
closed when runtime evidence is absent, continuity is not ready, high-severity
incidents remain active, resolved incidents lack closure, or the transparency
checkpoint is not trusted.

## Application order

Apply the bundles in this order:

1. Supply-chain integrity.
2. Capability sandboxing.
3. Incident response.
4. Disaster recovery and resilience closure.

The series is additive except for contract publication:

- Public API advances from `1.2.0` to `1.3.0`.
- Trust-store and attestation-statement schemas advance from version 1 to 2.
- The persisted schema catalog expands from 36 to 57 families.

Existing version-1 trust records remain readable because the minimum readable
version is not narrowed. Existing aesthetic, governance, deployment,
operational-evolution, and federated-stewardship records are unchanged.

New downstream systems that require this wave should pin
`ApiRequirement::operational_resilience()`.

## Recommended integration sequence

1. Generate the canonical build recipe and SBOM from the real release process.
2. Produce at least two independent build replicas on distinct builders.
3. Authenticate provenance and reproducibility records at the external trust
   boundary.
4. Declare the extractor's minimum capabilities and strict resource budgets.
5. Execute it under a real default-deny sandbox and capture observations.
6. Verify the complete `RuntimeSecurityBundle` before accepting extractor
   output in production.
7. Connect incident creation to deployment, SLO, drift, replay, and security
   alarms.
8. Rehearse response actions and require evidence-backed postmortems.
9. Inventory critical records across independent storage failure domains.
10. Run restore drills that measure actual RPO, RTO, object coverage, and digest
    integrity.
11. Evaluate continuity and operational resilience before promotion.
12. Append the final report to the transparency log and authenticate the
    checkpoint externally.

## Mandatory parent-workspace gates

- `cargo fmt --all -- --check`
- `cargo test -p symthaea-aesthetic --all-features`
- `cargo clippy -p symthaea-aesthetic --all-targets --all-features -- -D warnings`
- Feature-matrix and minimum-supported-Rust-version checks.
- Independent rebuilds on at least two genuinely separate builders.
- SBOM generation and license-policy tests against the production dependency
  graph.
- Linux sandbox tests proving denied filesystem, network, process, device, and
  environment access where applicable.
- Resource-exhaustion and policy-termination tests.
- Incident state-machine, deadline, escalation, and recurrence-drill tests.
- Destructive restore rehearsals against isolated recovery infrastructure.
- Downstream Muse, Canvas, voice, poetry, and game-director integration tests.

## Non-goals

This series does not provide a package manager, hermetic build engine, SBOM
scanner, vulnerability database, cryptographic signer, HSM, OS sandbox,
container runtime, incident paging service, ticketing system, object store,
backup agent, or disaster-recovery orchestrator. It provides strict,
serializable contracts through which those external systems can produce and
verify evidence without silently weakening the release gate.
