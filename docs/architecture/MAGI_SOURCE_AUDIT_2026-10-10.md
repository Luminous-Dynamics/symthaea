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

### Gap 1 — outcome resolution is not enforced by the public MAGI method

In \`WorldGroundedSelfModel::resolve_prediction\` in \`magi_integration.rs\`, the caller supplies \`observed_outcome\` and \`resolution_confidence\` directly. The method then resolves the pending prediction and updates calibration from those arguments. It does not itself execute or verify the prediction's declared \`ResolutionAuthority\`, and it does not require an independently verified receipt.

This leaves a gap between the intent documented by \`ResolutionContract\` and the authority enforced by the public resolution path. This finding does not assert that all current callers self-grade; it says this API does not prevent them from doing so.

Tracking item: [#7336 — enforce resolver-backed outcome admission and real timeouts](https://github.com/Luminous-Dynamics/symthaea/issues/7336).

### Gap 2 — the advertised command timeout is not enforced

In \`resolution.rs\`, \`ExitCodeResolver::execute(&self, _timeout: Duration)\` ignores its timeout argument and calls \`Command::output()\`. A hung child can therefore block resolution indefinitely. The timeout needs to be a real execution bound, with child cleanup/reaping and bounded output handling; timeout or ambiguous execution must not be relabelled success.

### Additional call-path findings — async resolution and untested code

The synchronous `ExitCodeResolver` in PR #7337 is not the only command-resolution path:

- In `src/consciousness/recursive_improvement/runtime.rs`, `check_auto_resolve` invokes `sh -c ... .output().await` for `CommandSucceeds` and `systemctl ... .output().await` for `ServiceRunning`, without an explicit Tokio timeout. The polling runtime later maps a boolean to `Success`/`SafeFailure` and calls `resolve_prediction(..., 1.0)`. PR #7337 fixes the synchronous standalone resolver. [PR #7346](https://github.com/Luminous-Dynamics/symthaea/pull/7346) now proposes the async runtime timeout, process-group cleanup, and no-calibration unresolved path; these changes are still awaiting exact-head qualification.
- In `src/coding_agent/magi_code_bridge.rs`, `compiled && tests_passed.unwrap_or(true)` means `tests_passed: None` is treated as `Success` if compilation succeeded. The API comment says `None` means no tests were run, so absent test evidence is currently promoted to success for this bridge's calibration statistics. This should remain unqualified evidence, or be represented as a distinct partial/unresolved outcome.
- In `src/coding_agent/generation.rs`, the orchestrator passes `response.accepted` as both the compilation and test result. The code-orchestrator source says acceptance follows compiler/test verification, so this is not evidence that accepted output was untested. The concern is narrower: the bridge receives one aggregate boolean for two distinct fields and does not retain the response's verification-layer provenance. Qualification should bind the actual verification results, not only the aggregate.

These are call-path findings from source inspection. They do not assert that the orchestrator always accepts untested code; they show that these APIs do not preserve the distinction between “not tested” and “verified successful” in all paths.

Tracking: [#7336 — enforce resolver-backed outcome admission and real timeouts](https://github.com/Luminous-Dynamics/symthaea/issues/7336). PR #7337 addresses the synchronous resolver, PR #7346 addresses async runtime command/port deadlines, and PR #7339 addresses the coding bridge's explicit missing-test and contradictory-evidence cases. These remain unqualified until exact-head checks pass.

### Gap 3 — the MAGI capability self-model is a generic estimate, not yet qualified expertise

The \`SelfModel\` stub in \`magi_integration.rs\` initializes capability values from a shared prior, updates an estimate through an observed scalar and learning rate, and uses that estimate as the confidence reported by \`predict_behavior\`. It does not, by itself, bind a particular skill claim to a forecast committed before execution and a verified later outcome for the exact source/configuration/evaluation profile.

The new capability-ledger primitive in [PR #7335](https://github.com/Luminous-Dynamics/symthaea/pull/7335) is the first proposed foundation for that missing link. It is intentionally not described as a complete learning loop yet: runtime forecast emission, trusted receipt verification, and end-to-end skill evaluation remain to be integrated.

### Gap 4 — the full loop is not qualified by source presence

The following remain required before calling the loop operationally established:

- Each forecast is committed before its outcome, bound to exact subject/configuration/evaluation identity, and resolved against independently verified evidence.
- Resolver timeout, spawn failure, stale results, missing evidence, and ambiguous outcomes remain distinguishable from observed task failure.
- The verified outcome path is the only path that can update qualification-grade calibration; arbitrary caller-supplied outcomes cannot qualify a capability.
- Self-model claims are tested prospectively on held-out and transfer contexts. Training-only results, skipped or queued jobs, stale-head evidence, and unauthenticated receipts are excluded.
- Any EFE influence is observed at the real action-selection boundary and evaluated against a matched baseline.
- Cross-domain claims are qualified on an exact integrated subject in at least two unrelated domains before any MAGI-crossing claim is made.

## Proposed evidence-bound capability lifecycle

The ledger in PR #7335 uses this lifecycle:

\`Proposed → Discovered → Candidate → Qualified\`

with \`Restricted\`, \`Suspended\`, and \`Retired\` as fail-closed or terminal states. A candidate can only become qualified by passing policy thresholds on resolved held-out/transfer evidence from an explicitly pinned evaluator identity and revision. Training data may inform development, but it cannot promote the claim.

The ledger computes Brier score and ECE for prospective success probabilities, retains expected/actual compute values for later analysis, enforces a single monotonically increasing event sequence across all accepted forecasts and outcomes, prevents cross-subject resolution and rejects duplicate receipt roots within one ledger. Empty calibration metrics are represented as unavailable, not perfect.

**Trust boundary:** the ledger checks metadata and bindings; it is not a cryptographic artifact verifier. Its input receipt must already have been authenticated by the external evidence pipeline. A caller-supplied digest string alone is not proof that a run occurred.

## Next implementation order

1. Enforce timeouts and unresolved results in both the synchronous resolver and the async runtime subprocess path.
2. Preserve `tests_passed: None` as unresolved/not-qualified rather than success; bind compile/test claims to the relevant independent receipts.
3. Add a resolver result/receipt type that binds outcome to prediction ID, exact subject, evaluator revision, chronology, and evidence root.
4. Make verified receipt admission—not an arbitrary outcome argument—the only production route into qualification-grade calibration.
5. Connect MAGI's per-capability forecast emission and resolved outcomes to the ledger in PR #7335.
6. Run focused tests on the exact integrated commit, then measure held-out transfer, calibration, and the actual downstream effect on decision selection.

No formatting, compilation, or test pass is asserted by this document.
