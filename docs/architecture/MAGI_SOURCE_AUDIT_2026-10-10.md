# MAGI and Operational Self-Model: Source Audit

**Audit date:** 2026-10-10  
**Repository:** Luminous-Dynamics/symthaea  
**Scope:** source inspection of recursive improvement, world prediction, calibration, resolution, constraint gating, and capability self-modeling.

This audit distinguishes source presence from verified behavior. A type, module, test, or API does not prove that a production path invokes it correctly. This is a source audit, not a claim that the full MAGI loop is qualified.

## Findings

### Components that exist

| Area | Source location | What source inspection supports |
|---|---|---|
| World-prediction types and resolution contracts | \`src/consciousness/recursive_improvement/world_prediction.rs\` | Predictions, outcome categories, resolution contracts, and declared authority types exist. |
| Calibration | \`src/consciousness/recursive_improvement/calibration.rs\` | Brier/ECE-style domain calibration, sample counters, and confidence-adjustment logic exist. The code explicitly tracks whether ECE was actually computed. |
| Execution-mode constraint gate | \`src/consciousness/recursive_improvement/constraint_gate.rs\` | Gate decisions can return autonomous, dry-run, or supervised modes based on current calibration/risk inputs. This does not establish that every production action path consults the gate. |
| Outcome-to-domain attribution | \`src/consciousness/recursive_improvement/magi_integration.rs\` | Failed predictions can trigger attribution records, responsible-domain heuristics, and missing-information suggestions. These are hypotheses, not identified causes. |
| Calibration-derived EFE values | \`src/consciousness/recursive_improvement/magi_integration.rs\` | \`compute_efe_contribution\` and \`compute_calibrated_efe\` exist. Source presence does not prove that the live decision route uses them or that the adjustment improves outcomes. |
| External resolver implementations | \`src/consciousness/recursive_improvement/resolution.rs\` | Command-exit and resource-state resolvers exist. See the material limitations below. |

### Gap 1 — raw outcomes are no longer a public method, but admission is still unverified

On this branch, the previous public WorldGroundedSelfModel::resolve_prediction(id, outcome, confidence) entry point has been renamed to the crate-private resolve_prediction_from_unverified_observation. A downstream crate can no longer call that method directly to inject an outcome into calibration.

This is **API-surface containment plus calibration fail-closed behavior, not receipt verification**. The current follow-up branch resolves unverified runtime outcomes for operational accounting but does not add them to Brier/ECE calibration, generate calibration-based causal attributions from them, or count them as completed calibration loops. Focused regression tests now cover both the internal admission method and the live async runtime resolution path, but they have not yet been run. Operational monitoring/PAC observation remains separate and is not promoted to trusted evidence. The deterministic magi_simulation example uses a public resolve_prediction_for_simulation method compiled only with the explicit, non-default magi-simulation feature; enabling that feature permits synthetic outcomes to affect that model instance's calibration, so it must not be enabled for a live model or qualification-grade evaluation. The resolver-contract follow-up now checks the runtime observer against the canonical model-side pending forecast before execution: the action-context action_type must equal the resolution-contract action_type (including before task-deadline fallback); resource checks require an exact path and supported Exists/NotExists state; command/service checks require an ExitCode authority with zero-success and explicit metadata pinning the exact program/argv; unsupported observer/authority combinations remain unclear. The runtime queue's caller-provided prediction clone is not trusted to supply this binding; a live-runtime regression test mutates that clone after forecast creation and asserts that the observer is rejected and the prediction retained as unclear. These are consistency checks, not cryptographic evidence: prediction/action metadata can still be caller-authored before the forecast is recorded, the result is not backed by a signed or independently verified receipt, and the runtime still does not provide exact subject/config/evaluation identity or replay protection. Do not describe the MAGI resolution path as authority-enforced or ungameable.

Tracking item: [#7336 — enforce resolver-backed outcome admission and real timeouts](https://github.com/Luminous-Dynamics/symthaea/issues/7336).

### Gap 2 — command timeouts are now enforced in both resolver paths; evidence provenance remains open

The timeout portion has been implemented on the stacked branches: PR [#7350](https://github.com/Luminous-Dynamics/symthaea/pull/7350) bounds the synchronous ExitCodeResolver, caps retained output while draining, and returns unresolved outcomes on timeout/ambiguous pipe state; PR [#7352](https://github.com/Luminous-Dynamics/symthaea/pull/7352) separately bounds async runtime command and port observations. Process-group termination is best-effort and does not itself authenticate an observation.

These source changes are not yet test-qualified in this audit. They also do not close the separate receipt/authority gap: a bounded command is not, by itself, proof that the command was the resolver committed in the prediction contract.

### Additional call-path findings — current fixes and remaining provenance limits

- In src/consciousness/recursive_improvement/runtime.rs, PR [#7352](https://github.com/Luminous-Dynamics/symthaea/pull/7352) adds bounded async command/port observation and keeps timeout/spawn/wait ambiguity out of calibration. PR [#7350](https://github.com/Luminous-Dynamics/symthaea/pull/7350) bounds the separate synchronous resolver. This is a source-level implementation change, not evidence that the associated CI or tests have passed.
- In src/coding_agent/magi_code_bridge.rs, PR [#7351](https://github.com/Luminous-Dynamics/symthaea/pull/7351) requires explicit test evidence for compile-and-test predictions and retains missing-test evidence as pending. Callers that pass Some(bool) still need the compile/test result bound to independently attributable receipts.
- In src/coding_agent/generation.rs, the orchestrator passes response.accepted as both compilation and test result. The orchestrator's acceptance result is intended to reflect its internal verification, so it would be inaccurate to call it definitely untested. However, the boolean loses per-stage verifier provenance, and the calibration bridge does not independently verify receipts for those stages.

These observations distinguish timeout/evidence-presence handling from provenance. None of PRs #7350, #7351, or #7352 establishes a cryptographic or independently verified receipt chain for the full MAGI path.

Tracking remains [#7336 — enforce resolver-backed outcome admission and real timeouts](https://github.com/Luminous-Dynamics/symthaea/issues/7336).

### Gap 3 — the MAGI capability self-model is a generic estimate, not yet qualified expertise

The \`SelfModel\` stub in \`magi_integration.rs\` initializes capability values from a shared prior, updates an estimate through an observed scalar and learning rate, and uses that estimate as the confidence reported by \`predict_behavior\`. It does not, by itself, bind a particular skill claim to a forecast committed before execution and a verified later outcome for the exact source/configuration/evaluation profile.

The new capability-ledger primitive in [PR #7349](https://github.com/Luminous-Dynamics/symthaea/pull/7349) is the first proposed foundation for that missing link. It is intentionally not described as a complete learning loop yet: runtime forecast emission, trusted receipt verification, and end-to-end skill evaluation remain to be integrated.

### Gap 4 — the full loop is not qualified by source presence

The following remain required before calling the loop operationally established:

- Each forecast is committed before its outcome, bound to exact subject/configuration/evaluation identity, and resolved against independently verified evidence.
- Resolver timeout, spawn failure, stale results, missing evidence, and ambiguous outcomes remain distinguishable from observed task failure.
- The verified outcome path is the only path that can update qualification-grade calibration; arbitrary caller-supplied outcomes cannot qualify a capability.
- Self-model claims are tested prospectively on held-out and transfer contexts. Training-only results, skipped or queued jobs, stale-head evidence, and unauthenticated receipts are excluded.
- Any EFE influence is observed at the real action-selection boundary and evaluated against a matched baseline.
- Cross-domain claims are qualified on an exact integrated subject in at least two unrelated domains before any MAGI-crossing claim is made.

## Proposed evidence-bound capability lifecycle

The ledger proposed in PR #7349 uses this lifecycle:

\`Proposed → Discovered → Candidate → Qualified\`

with \`Restricted\`, \`Suspended\`, and \`Retired\` as fail-closed or terminal states. A candidate can only become qualified by passing policy thresholds on resolved held-out/transfer evidence from an explicitly pinned evaluator identity and revision. Training data may inform development, but it cannot promote the claim.

The ledger computes Brier score and ECE for prospective success probabilities, retains expected/actual compute values for later analysis, requires monotonic forecast/outcome sequence numbers, prevents cross-subject resolution and rejects duplicate receipt roots within one ledger. Empty calibration metrics are represented as unavailable, not perfect.

**Trust boundary:** the ledger checks metadata and bindings; it is not a cryptographic artifact verifier. Its input receipt must already have been authenticated by the external evidence pipeline. A caller-supplied digest string alone is not proof that a run occurred.

## Next implementation order

1. Replace the crate-private unverified-outcome adapter with opaque receipt admission bound to prediction ID, exact subject/configuration/evaluation profile, committed forecast sequence, resolver identity/revision, and content-addressed evidence root. Until then, ensure raw runtime observations remain excluded from qualification-grade calibration (implemented in the follow-up branch; tests pending).
2. Extend the contract-binding prototype to an immutable forecast commitment and opaque, independently verified receipt; current runtime checks prevent queue-copy resolver rerouting and reject mismatched action/contract type labels, but caller-authored metadata is not yet a trust anchor. Unsupported/unbound observer authorities must remain unresolved, and caller-constructible digest strings are not proof.
3. Connect MAGI per-capability forecasts/outcomes to the ledger in PR #7349, with receipt verification preceding qualification-grade admission.
4. Preserve per-stage compile/test verifier provenance through the coding-agent bridge; distinguish an aggregate acceptance boolean from separately attributable evidence.
5. Add exact-head integration tests for wrong ID/subject, replayed receipts, stale chronology, resolver mismatch, unsupported authority, timeout, spawn failure, and ambiguous output.
6. Run focused tests on the exact integrated commit, then measure held-out transfer, calibration, and the actual downstream effect on decision selection.

No formatting, compilation, or test pass is asserted by this document.
