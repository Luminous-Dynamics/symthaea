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

The synchronous `ExitCodeResolver` in PR #7350 is not the only command-resolution path:

- In `src/consciousness/recursive_improvement/runtime.rs`, `check_auto_resolve` invokes `sh -c ... .output().await` for `CommandSucceeds` and `systemctl ... .output().await` for `ServiceRunning`, without an explicit Tokio timeout. The polling runtime later maps a boolean to `Success`/`SafeFailure` and calls `resolve_prediction(..., 1.0)`. PR #7350 proposes the synchronous standalone resolver fix. [PR #7352](https://github.com/Luminous-Dynamics/symthaea/pull/7352) proposes the async runtime timeout, process-group cleanup, and no-calibration unresolved path; these changes are still awaiting exact-head qualification.
- In `src/coding_agent/magi_code_bridge.rs`, `compiled && tests_passed.unwrap_or(true)` means `tests_passed: None` is treated as `Success` if compilation succeeded. The API comment says `None` means no tests were run, so absent test evidence is currently promoted to success for this bridge's calibration statistics. This should remain unqualified evidence, or be represented as a distinct partial/unresolved outcome.
- In `src/coding_agent/generation.rs`, the orchestrator passes `response.accepted` as both the compilation and test result. The code-orchestrator source says acceptance follows compiler/test verification, so this is not evidence that accepted output was untested. The concern is narrower: the bridge receives one aggregate boolean for two distinct fields and does not retain the response's verification-layer provenance. Qualification should bind the actual verification results, not only the aggregate.

These are call-path findings from source inspection. They do not assert that the orchestrator always accepts untested code; they show that these APIs do not preserve the distinction between “not tested” and “verified successful” in all paths.

Tracking: [#7336 — enforce resolver-backed outcome admission and real timeouts](https://github.com/Luminous-Dynamics/symthaea/issues/7336). PR #7350 proposes the synchronous resolver and lockfile fix, PR #7352 the async runtime command/port deadlines, and PR #7351 the coding bridge's explicit missing-test and contradictory-evidence handling. These remain unqualified until exact-head checks pass.

### Gap 3 — the MAGI capability self-model is a generic estimate, not yet qualified expertise

The \`SelfModel\` stub in \`magi_integration.rs\` initializes capability values from a shared prior, updates an estimate through an observed scalar and learning rate, and uses that estimate as the confidence reported by \`predict_behavior\`. It does not, by itself, bind a particular skill claim to a forecast committed before execution and a verified later outcome for the exact source/configuration/evaluation profile.

The capability-ledger primitive in [PR #7349](https://github.com/Luminous-Dynamics/symthaea/pull/7349) requires an opaque `VerifiedCapabilityOutcomeReceipt` at outcome admission. The token has private fields and no serde implementation; qualification policy pins evaluator identity/revision, the evaluator's Ed25519 public key, and verifier identity/revision. The policy is validated and frozen when the ledger is constructed rather than being supplied at each qualification call, so thresholds and allowlists cannot be relaxed after seeing outcomes. With the existing opt-in `identity` feature, `verify_and_resolve_signed_receipt` verifies an Ed25519 signature over a versioned receipt payload and hashes the exact supplied evidence bytes. It now requires a canonical, field-ordered `CapabilityEvidencePayloadV1` envelope (strict schema version, unknown fields rejected, exact re-serialization, and a 1 MiB input bound). The envelope must agree with the signed receipt on forecast ID and full forecast commitment, subject, outcome sequence, observed success, split, evaluator identity/revision, and realized compute. It also carries runner, immutable run/artifact identities, and an artifact SHA-256 for a subsequent trusted resolver to fetch and verify. The independent verifier revision is now `capability-evidence-envelope-v2`; the stale `capability-receipt-v1` revision is rejected by the policy-pinned allowlist. Regression coverage includes stale verifier-revision rejection, semantic receipt/evidence mismatch, noncanonical JSON rejection, evidence-byte mismatch, post-signature tampering, forecast commitment variations, and signed forecast transplant rejection. This authenticates the receipt as signed by the policy-pinned key and checks internal consistency between signed outcome and evidence envelope; it does **not** independently prove that the evaluator's run actually occurred, that the runner-artifact identifiers are truthful, or that the evaluator is truthful. Evaluator key custody and the trusted outer layer's initial policy selection remain trust assumptions. Default builds without `identity` have no production signed-receipt ingress and cannot mint the token. Runtime forecast emission, trusted runner-artifact retrieval, and end-to-end skill evaluation remain open.

### Gap 4 — the full loop is not qualified by source presence

The following remain required before calling the loop operationally established:

- Each forecast is committed before its outcome, bound to exact subject/configuration/evaluation identity, and resolved against independently verified evidence.
- Resolver timeout, spawn failure, stale results, missing evidence, and ambiguous outcomes remain distinguishable from observed task failure.
- The verified outcome path is the only path that can update qualification-grade calibration; arbitrary caller-supplied outcomes cannot qualify a capability.
- Self-model claims are tested prospectively on held-out and transfer contexts. Training-only results, skipped or queued jobs, stale-head evidence, and unauthenticated receipts are excluded.
- Any EFE influence is observed at the real action-selection boundary and evaluated against a matched baseline.
- Cross-domain claims are qualified on an exact integrated subject in at least two unrelated domains before any MAGI-crossing claim is made.

## Proposed evidence-bound capability lifecycle

The ledger in PR #7349 uses this lifecycle:

\`Proposed → Discovered → Candidate → Qualified\`

with \`Restricted\`, \`Suspended\`, and \`Retired\` as fail-closed or terminal states. A candidate can only become qualified by passing policy thresholds on resolved held-out/transfer evidence from an explicitly pinned evaluator identity and revision. Training data may inform development, but it cannot promote the claim.

The ledger computes Brier score and ECE for prospective success probabilities, retains expected/actual compute values for later analysis, enforces a single monotonically increasing event sequence across all accepted forecasts and outcomes, prevents cross-subject resolution and rejects duplicate receipt roots within one ledger. Empty calibration metrics are represented as unavailable, not perfect.

**Trust boundary:** raw `CapabilityOutcomeReceipt` data is not accepted by the qualification admission method. Qualification stores only the opaque `VerifiedCapabilityOutcomeReceipt` token and filters on policy-pinned evaluator identity/revision, Ed25519 public key, and verifier identity/revision. That validated policy is immutable for the lifetime of the ledger and cannot be swapped at metric/qualification call time; binding it to the correct evaluation profile remains the responsibility of the trusted outer layer. The token constructor is private and the token cannot be serialized/deserialized. Under the opt-in `identity` feature, the nested verifier checks an Ed25519 signature over a versioned receipt payload, verifies SHA-256 against the exact supplied bytes, requires a canonical `CapabilityEvidencePayloadV1`, and checks that its outcome fields exactly match the signed receipt. The envelope carries immutable runner/run/artifact identities and an artifact SHA-256, while the ledger compares the signed forecast commitment with the stored commitment for the referenced forecast. These checks prevent a signed payload from silently disagreeing with its structured evidence envelope, and prevent same-ID forecast substitution. They do not retrieve the external runner artifact, verify the artifact digest against fetched bytes, or establish that the evaluator reported truthfully. The key's provenance/custody, the policy that pins it, and the evaluator's execution environment remain explicit trust assumptions. Builds without `identity` fail closed because the production signed-receipt ingress is absent.

The existing `symthaea-evidence-plane` crate is a measured-counter/integrity contract (`EvidenceCounters`, `Expectation`, `check_integrity`), not an outcome-signature verifier. Its `config_hash` documentation explicitly states that it uses `DefaultHasher` over a debug representation and is not cryptographic or security-sensitive. It may be useful for recording instrumentation counters, but it must not be repurposed as the trust anchor for resolver receipts. Outcome qualification needs authenticated provenance and exact prediction/subject binding.

## Runner-artifact resolver design constraints (researched 2026-10-11)

The new evidence envelope is only a typed binding contract. Its runner/artifact identities and artifact SHA-256 remain claims until a trusted resolver independently retrieves the referenced bytes and verifies the lineage. A digest copied out of the same untrusted response is not independent verification.

For a GitHub Actions adapter, prefer a two-part check rather than treating the Actions artifact ID as provenance:

- Retrieve from the expected repository and exact workflow run/attempt using an immutable artifact identity. GitHub documents uploaded artifacts as immutable from `upload-artifact@v4` onward, unless they are deleted; immutability prevents mutation-in-place but does not by itself establish that the artifact came from an authorized workflow. See the [upload-artifact documentation](https://github.com/actions/upload-artifact/blob/main/README.md).
- Verify a signed artifact attestation against a reviewed policy: expected source repository, signer workflow, predicate type, source ref, source digest, and subject digest. GitHub's [artifact-attestation guidance](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations) documents the generate/verify flow, and the [GitHub CLI verification contract](https://cli.github.com/manual/gh_attestation_verify) exposes signer-workflow, signer-repository, source-ref, source-digest, predicate-type, and digest-algorithm constraints. These must be explicit policy values; do not accept whichever signer or source is returned by the artifact itself.

The adapter should produce a private verified-lineage result only after checking fetched bytes against the envelope's expected artifact digest and verifying the attestation subject and workflow/source identity against the frozen subject and policy. That verified result—not a public boolean or caller-provided digest—must feed the verification root used to mint the opaque capability token. Keep the adapter outside the pure ledger state machine so the ledger does not acquire ambient credentials or silently trust provider metadata. A generic interface without a policy-pinned implementation is not a security boundary.

This is a concrete implementation direction, not a claim that the resolver has already been implemented or that current receipts possess artifact provenance.

## Next implementation order

1. Enforce timeouts and unresolved results in both the synchronous resolver and the async runtime subprocess path.
2. Preserve `tests_passed: None` as unresolved/not-qualified rather than success; bind compile/test claims to the relevant independent receipts.
3. Add a trusted runner-artifact retrieval/verification adapter that fetches by immutable artifact identity, verifies the artifact bytes and lineage independently of the caller, and binds that authenticated lineage to the signed evidence root.
4. Connect the private verified-token issuance boundary only to that reviewed verifier path; caller-created digests, metadata-only checks, serialized tokens, and synthetic test fixtures must never become production qualification evidence.
5. Connect MAGI's per-capability forecast emission and resolved outcomes to the ledger in PR #7349.
6. Run focused tests on the exact integrated commit, then measure held-out transfer, calibration, and the actual downstream effect on decision selection.

No formatting, compilation, or test pass is asserted by this document.
