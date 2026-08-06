<!-- Tables and appendix generated from git, not typed. Regenerate with section 5. -->
# Deletion ledger — the 2026-07-29 music-crate cleanup

**629 files across 4 commits, all on 2026-07-29.** Written 2026-07-31 in response to an
external review whose criticism was fair: a deletion this size left its justification only in
commit messages, which nobody greps. This is the grep-able record — what went, why, how to get
it back, and the evidence that nothing has needed it since.

Nothing here is a recommendation to restore anything. It exists so that *deciding* to restore
something is a two-minute lookup instead of an archaeology session.

## 1. The four commits

| Commit | Date | Files deleted | What |
|---|---|---:|---|
| `53ea9e9a48` | 2026-07-29 | 148 | delete 148 orphaned `// placeholder` files, de-quarantine 9 crates |
| `0d2a2d2090` | 2026-07-29 | 19 | archive and delete evidence_calibration, close the orphan-cleanup increment |
| `154363798a` | 2026-07-29 | 1 | delete muse_motif_foundry_pilot, restore the wider studio check |
| `333445d5e4` | 2026-07-29 | 461 | delete the stranded evidence apparatus, scope PartId to tests |
| | | **629** | |

## 2. What was deleted, by category

| Category | Files | Recovery source |
|---|---:|---|
| authoring-kit process scaffolding (.md only) | 404 | `333445d5e4` |
| orphaned `// placeholder` src files | 139 | `53ea9e9a48` |
| examples/evidence_*.rs (stranded consumers) | 35 | `333445d5e4` |
| src/evidence_calibration/ (orphan subtree) | 28 | `0d2a2d2090`, `53ea9e9a48` |
| root EVIDENCE_*/PATCH_SERIES_* docs | 20 | `333445d5e4` |
| examples/support/ (undeclared module) | 2 | `333445d5e4` |
| src/bin/ (non-compiling binary) | 1 | `154363798a` |

## 3. Why each category went

**`// placeholder` src files (139 + 9 inside `evidence_calibration/`).** One-line files whose
entire content is a comment. They were *orphans* — no `mod` declared them, so `rustc` never
compiled them and no gate could see them (`feedback_orphan_modules_are_invisible_to_rustc`).
They spanned 9 crates beyond the music work; deleting them de-quarantined those crates.

**`src/evidence_calibration/` (28).** A subtree beneath a one-line `// placeholder`
`evidence_calibration.rs`. Because the parent was a placeholder rather than a `mod` declaration,
the whole subtree was unreachable from the crate root. Its design intent was archived *before*
deletion in `docs/design/EVIDENCE_CALIBRATION_STRANDED_DESIGN_2026-07-29.md`, which carries
recovery commands and a resurrection gate — that file is deliberately still present.

**`examples/evidence_*.rs` (35).** The consumer half of the above. 28 imported ~39
`Calibration*` types deleted that morning; 7 were one-line `// placeholder` with no `fn main`.
**They had been failing to build, and no gate noticed**: `cargo check -p <crate>` builds the lib
only, and `check-orphan-modules.sh` scanned `src/**`. Examples are auto-discovered by Cargo and
declared by no `mod`, so they were invisible to both by construction. 35 of 45 examples in the
crate were broken. Both gates were repaired in the same arc — CI now runs
`cargo build -p symthaea-music-theory --examples`, and the orphan script grew `broken_examples`
detection.

**`*-authoring-kit/` (404 files, all `.md`, zero `.patch`).** Process scaffolding that produced
no code. Series 25 declares "terminal retirement"; ten further series follow it, ending at
Series 35, "terminal retirement slice." Self-perpetuating and unreplayable — the kits describe
patches that do not exist in them.

**Root `EVIDENCE_*` / `PATCH_SERIES_*` docs (20).** Documentation-only consumers of the deleted
examples; 3 were themselves `// placeholder`.

**`src/bin/muse_motif_foundry_pilot.rs` (1).** Did not compile, and its presence had forced CI
to narrow to `--bin muse_studio`. Deleting it let the wider studio check come back.

## 4. Kept deliberately — do not "finish the job"

| Path | Why it survived |
|---|---|
| `docs/design/EVIDENCE_CALIBRATION_STRANDED_DESIGN_2026-07-29.md` | The intent archive for the deleted subtree, with recovery commands and a resurrection gate |
| `examples/diversity_census.rs` | Sat in the same not-compiling bucket, but is the only instrument measuring style differentiation. **Repaired first** in `a074498803`, then explicitly excluded from the sweep — the obvious predicate would have destroyed it |
| `DESIGN.md`, `HARMONIC_SYNTAX_REWORK_SCOPE_2026-07-26.md` | Live design docs, not evidence-apparatus output |
| the ~21.5K-line study apparatus (§4.3 of the review) | Unexercised, but real infrastructure with a plausible consumer — a different judgement from "produced nothing and cannot be replayed" |

## 5. Recovery

Every deleted file is intact in git. Nothing here needed a special archive.

```sh
# What did commit X delete?
git show --diff-filter=D --name-only --format= 333445d5e4

# Read one deleted file without touching the tree
git show 333445d5e4^:symthaea/crates/domains/symthaea-music-theory/examples/evidence_survey.rs

# Restore one file
git checkout 333445d5e4^ -- <path>

# Restore an entire category
git checkout 333445d5e4^ -- 'symthaea/crates/domains/symthaea-music-theory/examples/evidence_*.rs'

# Regenerate this ledger's tables
git show --diff-filter=D --name-only --format= 53ea9e9a48 0d2a2d2090 154363798a 333445d5e4 | sort -u
```

`git gc` does not endanger these: they are reachable from `main`'s history, not from a pruned
branch. (Contrast `.claude/rules/CONCURRENT_SESSIONS.md`, where the recovery concern is
*unreachable* objects.)

## 6. Has anything missed them? — verified 2026-07-31

Two days after the deletions, the tree was swept for live references to any deleted path:

```
$ grep -rlE "evidence_calibration|authoring-kit|muse_motif_foundry_pilot" \
    --include='*.rs' --include='*.toml' --include='*.yml' --include='*.sh' symthaea/
symthaea/scripts/check-orphan-modules.sh     # comments describing the history
symthaea/.github/workflows/ci.yml            # comments describing the history
```

**Both hits are comments about the cleanup itself. Zero live references.** No build, script, CI
job or module refers to anything deleted. That is the evidence the deletion was correct, and it
is the check to re-run before anyone argues for restoration.

## 7. Appendix — complete file list

### `53ea9e9a48` — 148 files

```
symthaea/crates/domains/symthaea-acoustics/src/acoustic_two_port.rs
symthaea/crates/domains/symthaea-acoustics/src/atmospheric_absorption.rs
symthaea/crates/domains/symthaea-acoustics/src/error.rs
symthaea/crates/domains/symthaea-acoustics/src/layered_interface.rs
symthaea/crates/domains/symthaea-acoustics/src/medium.rs
symthaea/crates/domains/symthaea-acoustics/src/meteorology.rs
symthaea/crates/domains/symthaea-acoustics/src/propagation.rs
symthaea/crates/domains/symthaea-aesthetic/src/api_contract.rs
symthaea/crates/domains/symthaea-aesthetic/src/assessment.rs
symthaea/crates/domains/symthaea-aesthetic/src/deployment.rs
symthaea/crates/domains/symthaea-aesthetic/src/governance.rs
symthaea/crates/domains/symthaea-aesthetic/src/pipeline.rs
symthaea/crates/domains/symthaea-aesthetic/src/prelude.rs
symthaea/crates/domains/symthaea-aesthetic/src/reference_extractors.rs
symthaea/crates/domains/symthaea-aesthetic/src/registry.rs
symthaea/crates/domains/symthaea-aesthetic/src/schema.rs
symthaea/crates/domains/symthaea-aesthetic/src/study.rs
symthaea/crates/domains/symthaea-aesthetic/src/test_support.rs
symthaea/crates/domains/symthaea-aesthetic/src/trust.rs
symthaea/crates/domains/symthaea-aesthetic/src/validation.rs
symthaea/crates/domains/symthaea-canvas/src/accessibility.rs
symthaea/crates/domains/symthaea-canvas/src/aesthetic_report.rs
symthaea/crates/domains/symthaea-canvas/src/attestation.rs
symthaea/crates/domains/symthaea-canvas/src/comparison.rs
symthaea/crates/domains/symthaea-canvas/src/document.rs
symthaea/crates/domains/symthaea-canvas/src/dom_patch.rs
symthaea/crates/domains/symthaea-canvas/src/fingerprint.rs
symthaea/crates/domains/symthaea-canvas/src/manifest.rs
symthaea/crates/domains/symthaea-canvas/src/mapping_trace.rs
symthaea/crates/domains/symthaea-canvas/src/migration.rs
symthaea/crates/domains/symthaea-canvas/src/readiness.rs
symthaea/crates/domains/symthaea-canvas/src/recording.rs
symthaea/crates/domains/symthaea-canvas/src/render_receipt.rs
symthaea/crates/domains/symthaea-canvas/src/scene_audit.rs
symthaea/crates/domains/symthaea-canvas/src/scene_index.rs
symthaea/crates/domains/symthaea-canvas/src/session.rs
symthaea/crates/domains/symthaea-canvas/src/telemetry.rs
symthaea/crates/domains/symthaea-coding-theory/src/bit_packing.rs
symthaea/crates/domains/symthaea-coding-theory/src/convolutional.rs
symthaea/crates/domains/symthaea-coding-theory/src/crc.rs
symthaea/crates/domains/symthaea-coding-theory/src/soft.rs
symthaea/crates/domains/symthaea-hal/src/arming.rs
symthaea/crates/domains/symthaea-hal/src/audit_export.rs
symthaea/crates/domains/symthaea-hal/src/boot.rs
symthaea/crates/domains/symthaea-hal/src/capability.rs
symthaea/crates/domains/symthaea-hal/src/evidence.rs
symthaea/crates/domains/symthaea-hal/src/fault_campaign.rs
symthaea/crates/domains/symthaea-hal/src/fault_ledger.rs
symthaea/crates/domains/symthaea-hal/src/feedback.rs
symthaea/crates/domains/symthaea-hal/src/hil.rs
symthaea/crates/domains/symthaea-hal/src/operator_authority.rs
symthaea/crates/domains/symthaea-hal/src/output_gate.rs
symthaea/crates/domains/symthaea-hal/src/security_admission.rs
symthaea/crates/domains/symthaea-hal/src/startup.rs
symthaea/crates/domains/symthaea-humanoid/src/calibration.rs
symthaea/crates/domains/symthaea-humanoid/src/contract.rs
symthaea/crates/domains/symthaea-humanoid/src/evidence_signature.rs
symthaea/crates/domains/symthaea-humanoid/src/interlock.rs
symthaea/crates/domains/symthaea-humanoid/src/observation.rs
symthaea/crates/domains/symthaea-humanoid/src/proprioceptive_calibration.rs
symthaea/crates/domains/symthaea-humanoid/src/runtime.rs
symthaea/crates/domains/symthaea-humanoid/src/safety_export.rs
symthaea/crates/domains/symthaea-humanoid/src/safety_journal.rs
symthaea/crates/domains/symthaea-humanoid/src/servo.rs
symthaea/crates/domains/symthaea-humanoid/src/supervisor.rs
symthaea/crates/domains/symthaea-legal-reasoning/src/inference.rs
symthaea/crates/domains/symthaea-legal-reasoning/src/proof.rs
symthaea/crates/domains/symthaea-legal-reasoning/src/session.rs
symthaea/crates/domains/symthaea-manipulator/src/command_authority.rs
symthaea/crates/domains/symthaea-manipulator/src/hil.rs
symthaea/crates/domains/symthaea-manipulator/src/operator_policy.rs
symthaea/crates/domains/symthaea-manipulator/src/release_assurance.rs
symthaea/crates/domains/symthaea-manipulator/src/robustness.rs
symthaea/crates/domains/symthaea-manipulator/src/safety_protocol.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/bundle.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/migration.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/incident_response_tests.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/integration_tests.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/schema.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/signature.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/study/integration_tests.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/study/link.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/study/retention.rs
symthaea/crates/domains/symthaea-statistics/src/autoregression.rs
symthaea/crates/domains/symthaea-statistics/src/conjugate.rs
symthaea/crates/domains/symthaea-statistics/src/cox.rs
symthaea/crates/domains/symthaea-statistics/src/cross_validation.rs
symthaea/crates/domains/symthaea-statistics/src/error.rs
symthaea/crates/domains/symthaea-statistics/src/fitting.rs
symthaea/crates/domains/symthaea-statistics/src/glm.rs
symthaea/crates/domains/symthaea-statistics/src/linalg.rs
symthaea/crates/domains/symthaea-statistics/src/logrank.rs
symthaea/crates/domains/symthaea-statistics/src/matching.rs
symthaea/crates/domains/symthaea-statistics/src/model_matrix.rs
symthaea/crates/domains/symthaea-statistics/src/multiple_regression.rs
symthaea/crates/domains/symthaea-statistics/src/multivariate.rs
symthaea/crates/domains/symthaea-statistics/src/process_control.rs
symthaea/crates/domains/symthaea-statistics/src/resampling.rs
symthaea/crates/domains/symthaea-statistics/src/robust_regression.rs
symthaea/crates/domains/symthaea-statistics/src/sampling.rs
symthaea/crates/domains/symthaea-statistics/src/sequential.rs
symthaea/crates/domains/symthaea-subterranean/src/accountability_supervisor.rs
symthaea/crates/domains/symthaea-subterranean/src/accountability_validation.rs
symthaea/crates/domains/symthaea-subterranean/src/decision_trace.rs
symthaea/crates/domains/symthaea-subterranean/src/near_miss.rs
symthaea/crates/domains/symthaea-subterranean/src/team_resource.rs
symthaea/crates/domains/symthaea-therapeutic/src/change_control.rs
symthaea/crates/domains/symthaea-therapeutic/src/consent.rs
symthaea/crates/domains/symthaea-therapeutic/src/evidence.rs
symthaea/crates/domains/symthaea-therapeutic/src/orchestrator.rs
symthaea/crates/domains/symthaea-therapeutic/src/release_evidence.rs
symthaea/crates/domains/symthaea-vocal-tract/src/branched_waveguide.rs
symthaea/crates/domains/symthaea-vocal-tract/src/gesture_projection.rs
symthaea/crates/domains/symthaea-vocal-tract/src/gesture_timing.rs
symthaea/crates/domains/symthaea-vocal-tract/src/glottal_source.rs
symthaea/crates/domains/symthaea-vocal-tract/src/observed_waveguide.rs
symthaea/crates/domains/symthaea-vocal-tract/src/physical_speech.rs
symthaea/crates/domains/symthaea-vocal-tract/src/physiology.rs
symthaea/crates/domains/symthaea-vocal-tract/src/series23_contract.rs
symthaea/crates/domains/symthaea-vocal-tract/src/series23_evidence.rs
symthaea/crates/domains/symthaea-vocal-tract/src/series23_perceptual_promotion.rs
symthaea/crates/domains/symthaea-vocal-tract/src/series23_statistical_promotion.rs
symthaea/crates/domains/symthaea-vocal-tract/src/speech_evidence.rs
symthaea/crates/domains/symthaea-vocal-tract/src/transmission_line_reference.rs
symthaea/crates/domains/symthaea-wisdom/src/archive.rs
symthaea/crates/domains/symthaea-wisdom/src/archive_replay.rs
symthaea/crates/domains/symthaea-wisdom/src/archive_store.rs
symthaea/crates/domains/symthaea-wisdom/src/audit.rs
symthaea/crates/domains/symthaea-wisdom/src/authority_checkpoint.rs
symthaea/crates/domains/symthaea-wisdom/src/authority_recovery.rs
symthaea/crates/domains/symthaea-wisdom/src/checkpoint.rs
symthaea/crates/domains/symthaea-wisdom/src/coordination.rs
symthaea/crates/domains/symthaea-wisdom/src/deployment.rs
symthaea/crates/domains/symthaea-wisdom/src/ethics.rs
symthaea/crates/domains/symthaea-wisdom/src/evidence.rs
symthaea/crates/domains/symthaea-wisdom/src/execution.rs
symthaea/crates/domains/symthaea-wisdom/src/orchestration.rs
symthaea/crates/domains/symthaea-wisdom/src/postgres_sync.rs
symthaea/crates/domains/symthaea-wisdom/src/production_network.rs
symthaea/crates/domains/symthaea-wisdom/src/release.rs
symthaea/crates/domains/symthaea-wisdom/src/replay.rs
symthaea/crates/domains/symthaea-wisdom/src/replication.rs
symthaea/crates/domains/symthaea-wisdom/src/rotation_store.rs
symthaea/crates/domains/symthaea-wisdom/src/runtime.rs
symthaea/crates/domains/symthaea-wisdom/src/service.rs
symthaea/crates/domains/symthaea-wisdom/src/source_auth.rs
symthaea/crates/domains/symthaea-wisdom/src/startup_attempt.rs
symthaea/crates/domains/symthaea-wisdom/src/storage.rs
```

### `0d2a2d2090` — 19 files

```
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/closure/integrity.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/closure/tests.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/continuity/integrity.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/continuity/mod.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/continuity/model.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/gossip/integrity.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/gossip/mod.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/gossip/model.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/lineage.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/mod.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/recovery_authority/integrity.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/recovery_authority/tests.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/reentry/integrity.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/reentry/tests.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/witness_policy.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/witness_policy/integrity.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/study/mod.rs
symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/study/model.rs
```

### `154363798a` — 1 files

```
symthaea/crates/domains/symthaea-muse/src/bin/muse_motif_foundry_pilot.rs
```

### `333445d5e4` — 461 files

```
symthaea/crates/domains/symthaea-music-theory/EVIDENCE_GOVERNANCE_DISCLOSURE_RELEASE.md
symthaea/crates/domains/symthaea-music-theory/EVIDENCE_GOVERNANCE_RELEASE.md
symthaea/crates/domains/symthaea-music-theory/EVIDENCE_HUMAN_STUDY_RELEASE.md
symthaea/crates/domains/symthaea-music-theory/EVIDENCE_INCIDENT_RECOVERY_RELEASE.md
symthaea/crates/domains/symthaea-music-theory/EVIDENCE_PRIVACY_PORTFOLIO_RELEASE.md
symthaea/crates/domains/symthaea-music-theory/EVIDENCE_PUBLICATION_RELEASE.md
symthaea/crates/domains/symthaea-music-theory/EVIDENCE_RECOVERY_REENTRY_CLOSURE_RELEASE.md
symthaea/crates/domains/symthaea-music-theory/PATCH_SERIES_14_PLAN_2026-07-20.md
symthaea/crates/domains/symthaea-music-theory/PATCH_SERIES_16_APPLY_README.md
symthaea/crates/domains/symthaea-music-theory/PATCH_SERIES_16_PLAN_2026-07-20.md
symthaea/crates/domains/symthaea-music-theory/PATCH_SERIES_17_APPLY_README.md
symthaea/crates/domains/symthaea-music-theory/PATCH_SERIES_17_PLAN_2026-07-20.md
symthaea/crates/domains/symthaea-music-theory/PATCH_SERIES_18_APPLY_README.md
symthaea/crates/domains/symthaea-music-theory/PATCH_SERIES_18_PLAN_2026-07-21.md
symthaea/crates/domains/symthaea-music-theory/PATCH_SERIES_19_APPLY_README.md
symthaea/crates/domains/symthaea-music-theory/PATCH_SERIES_19_PLAN_2026-07-21.md
symthaea/crates/domains/symthaea-music-theory/PATCH_SERIES_20_APPLY_README.md
symthaea/crates/domains/symthaea-music-theory/PATCH_SERIES_20_PLAN_2026-07-21.md
symthaea/crates/domains/symthaea-music-theory/PATCH_SERIES_21_APPLY_README.md
symthaea/crates/domains/symthaea-music-theory/PATCH_SERIES_21_PLAN_2026-07-21.md
symthaea/crates/domains/symthaea-music-theory/examples/evidence_attach_listener_response.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_enforce_study_retention.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_governance_attestation_payload.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_governance_export.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_governance_receipt.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_governance_receipt_chain.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_listener_response_payload.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_privacy_release.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_catalog.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_checkpoint.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_continuity.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_delegation.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_gossip.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_head_bundle.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_incident.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_incident_closure.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_incident_response.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_lineage.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_mirror.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_policy.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_post_recovery.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_quarantine.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_recovery.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_recovery_authority.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_witness.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_publication_witness_policy.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_retention_snapshot.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_selective_disclosure.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_study_assignment.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_study_books.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_study_portfolio.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_study_public_report.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_third_party_audit_package.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_verify_listener_response.rs
symthaea/crates/domains/symthaea-music-theory/examples/evidence_withdraw_study_response.rs
symthaea/crates/domains/symthaea-music-theory/examples/support/checkpoint_verifier.rs
symthaea/crates/domains/symthaea-music-theory/examples/support/mod.rs
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/APPLY_README_TEMPLATE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/AUTHORING_CHECKLIST.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/EVIDENCE_CYCLE_TWO_RESUMPTION_CONTRACT.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/PATCH_ORDER.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/README.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0001-audit-freeze-cycle-two-resumption-scenario.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0002-test-add-qualified-cycle-two-closure-baseline.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0003-refactor-generalize-trust-segment-successor-identity.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0004-feat-extend-segment-ledger-for-successor-genesis.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0005-feat-add-cycle-two-resumption-policy-context.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0006-feat-implement-cycle-two-resumption-plan.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0007-feat-add-cycle-two-resumption-statements.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0008-feat-implement-cycle-two-dual-quorum-resumption.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0009-feat-require-fresh-post-cycle-two-publisher-delegation.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0010-feat-require-fresh-successor-segment-allowance.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0011-refactor-extend-transition-gate-for-successor-resumption.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0012-feat-extend-reference-store-for-successor-first-mutation.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0013-feat-implement-successor-first-mutation-receipt.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0014-feat-commit-successor-first-publication-atomically.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0015-security-prevent-cross-cycle-double-first-mutation.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0016-feat-enforce-global-ordinal-continuity-through-cycle-two.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0017-feat-add-cross-cycle-operating-state-audit.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0018-feat-register-cycle-two-resumption-schema-prefix.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0019-feat-add-external-verifier-cycle-two-resumption-roles.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0020-feat-add-cycle-two-resumption-plan-cli.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0021-feat-add-successor-first-publication-cli.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0022-test-run-positive-cycle-two-closure-to-publication.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0023-test-run-cross-cycle-resumption-replay-and-staleness-corpus.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0024-test-run-successor-race-crash-and-rollback-matrix.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0025-feat-freeze-independent-cycle-two-resumption-vectors.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0026-ci-add-cycle-two-resumption-qualification-lanes.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-cycle-two-resumption-series-34-authoring-kit/patches/0027-docs-publish-series-34-cycle-two-resumption-report.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/APPLY_README_TEMPLATE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/AUTHORING_CHECKLIST.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/EVIDENCE_FIRST_RESUMPTION_SLICE_CONTRACT.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/PATCH_ORDER.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/README.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0001-audit-freeze-first-resumption-slice-scenario.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0002-test-add-series-21-closure-baseline-fixture.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0003-refactor-add-minimal-shared-lifecycle-identities.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0004-feat-add-verifier-owned-resumption-policy-context.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0005-feat-implement-trust-segment-genesis.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0006-feat-implement-minimal-segment-ledger.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0007-feat-implement-resumption-plan.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0008-feat-implement-cycle-specific-resumption-statements.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0009-feat-implement-dual-quorum-authorization.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0010-feat-implement-fresh-publisher-delegation-binding.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0011-feat-implement-fresh-segment-scoped-allowance.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0012-refactor-add-minimal-typed-transition-gate.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0013-feat-add-reference-compare-and-commit-store.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0014-feat-implement-first-mutation-receipt.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0015-feat-commit-first-resumed-publication-atomically.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0016-security-prevent-concurrent-double-first-mutation.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0017-feat-enforce-global-publication-and-event-ordinals.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0018-feat-add-shell-free-external-verifier-resumption-role.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0019-feat-add-resumption-plan-and-verify-cli.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0020-feat-add-first-resumed-publication-cli.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0021-test-run-positive-end-to-end-resumption-scenario.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0022-test-run-resumption-replay-staleness-and-policy-corpus.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0023-test-run-crash-rollback-and-race-matrix.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0024-feat-freeze-independent-resumption-conformance-vectors.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0025-ci-add-first-slice-build-test-and-reproduction-lanes.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-first-resumption-slice-series-31-authoring-kit/patches/0026-docs-publish-series-31-first-slice-qualification-report.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/APPLY_README_TEMPLATE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/AUTHORING_CHECKLIST.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/EVIDENCE_IMPLEMENTATION_CONVERGENCE_RELEASE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/PATCH_ORDER.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/README.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0001-audit-build-plan-to-code-convergence-matrix.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0002-chore-freeze-exact-series-21-baseline-and-source-inventory.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0003-refactor-create-shared-lifecycle-domain-primitives.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0004-feat-implement-trust-segment-model-and-ledger.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0005-feat-implement-resumption-policy-authorization-and-receipts.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0006-feat-implement-challenge-ledger-and-reopening-model.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0007-feat-implement-cycle-aware-recovery-model.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0008-feat-implement-terminal-retirement-and-archive-mode.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0009-refactor-unify-verifier-owned-expected-policy-context.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0010-refactor-add-single-typed-transition-gate.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0011-feat-add-compare-and-commit-state-store-contract.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0012-security-inventory-and-block-all-mutation-bypasses.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0013-feat-register-series-22-25-schema-prefixes.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0014-feat-add-persisted-state-migration-and-backward-verification.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0015-feat-implement-shell-free-external-verifier-roles.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0016-feat-implement-curated-cli-workflows.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0017-test-add-frozen-positive-cumulative-lifecycle-vectors.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0018-test-add-frozen-replay-staleness-and-policy-substitution-corpus.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0019-test-add-transaction-race-and-rollback-harness.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0020-security-apply-series-24-resource-and-privacy-bounds-to-new-code.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0021-test-add-property-fuzz-and-model-checking-seeds.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0022-docs-generate-implementation-and-claim-matrices-from-code.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0023-chore-produce-real-mail-series-22-through-25.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0024-ci-add-cumulative-all-target-feature-and-nix-lanes.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-implementation-convergence-series-26-authoring-kit/patches/0025-docs-publish-series-26-implementation-convergence-release.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/APPLY_README_TEMPLATE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/AUTHORING_CHECKLIST.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/EVIDENCE_INCIDENT_REOPENING_RELEASE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/PATCH_ORDER.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/README.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0001-audit-freeze-future-evidence-and-reopening-threat-model.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0002-feat-incident-model-bounded-evidence-challenges.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0003-feat-incident-add-append-only-challenge-ledger.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0004-feat-incident-model-reopening-trigger-policy.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0005-feat-incident-evaluate-objective-reopening-triggers.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0006-feat-incident-model-reopening-policy-plan-and-limitations.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0007-feat-incident-add-dual-quorum-reopening-authorization.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0008-feat-incident-model-segment-freeze-and-reopen-receipts.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0009-feat-incident-commit-reopening-and-freeze-transactionally.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0010-security-publication-block-mutations-after-authorized-reopening.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0011-feat-incident-link-recurrences-without-collapsing-incidents.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0012-feat-incident-model-reopened-lifecycle-state.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0013-security-incident-bound-challenge-storage-and-verification.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0014-feat-incident-add-reopening-review-package.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0015-feat-schema-register-challenge-reopening-and-recurrence-contracts.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0016-feat-api-export-curated-reopening-surface.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0017-feat-tooling-extend-external-verifier-for-reopening.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0018-feat-tooling-add-incident-challenge-command.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0019-feat-tooling-add-incident-reopening-and-freeze-command.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0020-test-incident-cover-hidden-fork-equivocation-and-invalid-resumption.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0021-test-incident-cover-reopen-authorization-freeze-and-races.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0022-test-incident-cover-challenge-spam-privacy-and-resource-limits.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0023-docs-incident-define-reopening-recurrence-and-freeze-release.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-incident-reopening-series-23-authoring-kit/patches/0024-docs-add-series-23-landing-and-application-guides.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/APPLY_README_TEMPLATE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/AUTHORING_CHECKLIST.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/EVIDENCE_RECURSIVE_RECOVERY_RELEASE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/PATCH_ORDER.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/README.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0001-audit-freeze-repeated-recovery-threat-model.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0002-feat-recovery-model-content-derived-cycle-identity.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0003-feat-recovery-add-append-only-cycle-ledger.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0004-security-recovery-reject-cross-cycle-authority-replay.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0005-feat-recovery-model-cycle-scoped-authority-epochs.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0006-feat-recovery-model-cycle-scoped-witness-policy.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0007-feat-recovery-bind-branch-selection-to-frozen-segment.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0008-feat-recovery-carry-forward-quarantines-explicitly.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0009-feat-recovery-model-cycle-plan-and-limitations.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0010-feat-recovery-add-dual-quorum-cycle-authorization.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0011-feat-recovery-commit-cycle-selection-transactionally.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0012-feat-recovery-model-post-cycle-fresh-checkpoint.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0013-feat-recovery-verify-cycle-reentry-certification.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0014-feat-recovery-model-cycle-closure.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0015-feat-recovery-generalize-segment-genesis-from-cycle.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0016-feat-recovery-audit-multi-cycle-lifecycle.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0017-security-recovery-bound-number-of-active-attempts.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0018-feat-schema-register-cycle-aware-recovery-contracts.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0019-feat-api-export-curated-cycle-recovery-surface.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0020-feat-tooling-add-recovery-cycle-command.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0021-test-recovery-cover-cross-cycle-replay-and-lineage.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0022-test-recovery-cover-transaction-races-and-abandonment.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0023-docs-recovery-define-recursive-cycle-release.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-recursive-recovery-series-24-authoring-kit/patches/0024-docs-add-series-24-landing-and-application-guides.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/APPLY_README_TEMPLATE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/AUTHORING_CHECKLIST.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/EVIDENCE_RELEASE_QUALIFICATION_CONTRACT.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/PATCH_ORDER.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/README.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0001-audit-freeze-release-qualification-and-compatibility-scope.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0002-feat-publish-versioned-public-contract-and-stability-tiers.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0003-refactor-minimize-and-curate-crate-root-api.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0004-feat-add-compatibility-adapters-for-series-21-clients.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0005-security-remove-or-hard-fail-legacy-direct-mutation-paths.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0006-feat-freeze-canonical-byte-and-schema-conformance-corpus.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0007-feat-qualify-independent-verifier-implementations.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0008-test-run-end-to-end-lifecycle-scenarios.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0009-test-run-clean-room-mail-series-replay.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0010-test-run-cross-platform-fixed-width-serialization-lanes.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0011-test-run-worst-case-valid-resource-benchmarks.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0012-test-run-long-history-and-restart-soak.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0013-test-run-fault-injection-and-crash-recovery-qualification.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0014-security-run-complete-privacy-and-disclosure-audit.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0015-security-run-supply-chain-license-and-advisory-gates.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0016-security-run-mutation-surface-and-endpoint-inventory-gate.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0017-feat-add-release-candidate-claim-matrix-and-waiver-ledger.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0018-feat-add-reproducible-release-evidence-bundle.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0019-docs-publish-operator-runbooks-and-failure-taxonomy.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0020-docs-publish-api-schema-and-migration-reference.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0021-ci-add-release-candidate-freeze-gate.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0022-docs-publish-series-27-release-qualification-report.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-release-qualification-series-27-authoring-kit/patches/0023-chore-tag-grounded-lifecycle-release-candidate.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/APPLY_README_TEMPLATE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/AUTHORING_CHECKLIST.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/EVIDENCE_SECOND_RECOVERY_CYCLE_CONTRACT.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/PATCH_ORDER.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/README.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0001-audit-freeze-second-recovery-cycle-scenario.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0002-test-add-qualified-series-32-frozen-baseline-fixture.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0003-refactor-extend-shared-identities-for-cycle-two.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0004-feat-implement-content-derived-cycle-two-identity.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0005-feat-implement-minimal-recovery-cycle-ledger.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0006-feat-implement-cycle-scoped-recovery-authority-policy.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0007-feat-implement-cycle-scoped-recovered-witness-policy.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0008-feat-implement-explicit-quarantine-carry-forward.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0009-feat-implement-cycle-two-branch-candidate-set.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0010-feat-implement-cycle-two-recovery-plan.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0011-feat-implement-cycle-two-authorization-statements.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0012-feat-implement-dual-quorum-cycle-two-authorization.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0013-refactor-extend-transition-gate-for-cycle-selection.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0014-feat-extend-reference-store-for-cycle-selection.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0015-feat-commit-cycle-two-branch-selection-atomically.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0016-security-bound-active-cycle-two-attempts.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0017-feat-implement-post-cycle-two-fresh-checkpoint-input.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0018-feat-implement-cycle-two-reentry-certification.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0019-feat-implement-cycle-two-closure-plan.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0020-feat-implement-cycle-two-closure-authorization.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0021-feat-commit-cycle-two-closure-transactionally.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0022-feat-implement-multi-cycle-lifecycle-audit-through-cycle-two.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0023-feat-register-cycle-two-slice-schema-prefix.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0024-feat-add-shell-free-external-verifier-cycle-two-roles.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0025-feat-add-cycle-two-recovery-cli.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0026-test-run-positive-freeze-to-cycle-two-closure-scenario.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0027-test-run-cross-cycle-replay-lineage-and-quarantine-corpus.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0028-test-run-cycle-two-race-crash-abandonment-and-limit-matrix.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0029-feat-freeze-independent-cycle-two-conformance-vectors.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0030-ci-add-cycle-two-build-test-and-reproduction-lanes.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-second-recovery-cycle-series-33-authoring-kit/patches/0031-docs-publish-series-33-cycle-two-qualification-report.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/APPLY_README_TEMPLATE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/AUTHORING_CHECKLIST.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/EVIDENCE_POST_RECOVERY_RESUMPTION_RELEASE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/PATCH_ORDER.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/README.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/SYMTHAEA_MUSIC_THEORY_PATCH_SERIES_21_PLAN_2026-07-21.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0001-test-publication-freeze-Series-20-prerequisite-invariants.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0002-feat-publication-add-trust-segment-models.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0003-feat-recovery-bind-recovered-anchor-to-segment-genesis.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0004-feat-publication-assess-fresh-post-anchor-checkpoints.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0005-feat-publication-gate-resumption-on-mirrors-conflicts-and-quarantine.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0006-feat-publication-add-resumption-authorization-contracts.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0007-feat-publication-verify-resumption-with-external-adapters.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0008-security-publication-enforce-resumption-at-mutation-boundary.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0009-security-publication-refuse-cross-segment-delegation-carryover.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0010-feat-publication-bind-records-events-and-proofs-to-trust-segments.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0011-feat-publication-authorize-explicit-cross-segment-status-bridges.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0012-feat-publication-build-resumed-head-bundles.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0013-feat-tools-operate-post-recovery-resumption-workflow.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0014-feat-schema-register-post-recovery-resumption-contracts.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0015-test-publication-reject-resumption-boundary-attacks.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0016-test-publication-cover-recovery-to-resumption-end-to-end.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0017-refactor-publication-split-resumption-model-integrity-and-tests.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-21-authoring-kit/patches/0018-docs-publication-define-post-recovery-resumption-release.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/APPLY_README_TEMPLATE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/AUTHORING_CHECKLIST.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/EVIDENCE_CROSS_IMPLEMENTATION_CONFORMANCE_RELEASE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/PATCH_ORDER.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/README.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/SYMTHAEA_MUSIC_THEORY_PATCH_SERIES_22_PLAN_2026-07-21.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/patches/0001-audit-conformance-inventory-public-persistence-and-canonical-APIs.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/patches/0002-feat-conformance-add-versioned-envelope-and-failure-taxonomy.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/patches/0003-feat-conformance-export-positive-canonical-vectors.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/patches/0004-feat-conformance-export-single-field-mutation-vectors.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/patches/0005-test-conformance-cover-order-duplicates-unknown-fields-and-integer-bounds.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/patches/0006-feat-conformance-add-publication-recovery-resumption-scenarios.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/patches/0007-feat-tools-run-no-shell-differential-verifiers.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/patches/0008-docs-conformance-specify-independent-verifier-interface.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/patches/0009-fix-persistence-remove-platform-dependent-public-encodings.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/patches/0010-feat-release-build-deterministic-evidence-archives.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/patches/0011-feat-tools-export-offline-independent-verification-kit.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/patches/0012-feat-schema-report-consumer-compatibility.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/patches/0013-test-conformance-freeze-fuzz-seeds-and-property-replay.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/patches/0014-test-release-reproduce-source-vectors-and-verification-kit.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/patches/0015-feat-schema-register-conformance-contracts.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-22-authoring-kit/patches/0016-docs-evidence-define-cross-implementation-conformance-release.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/APPLY_README_TEMPLATE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/AUTHORING_CHECKLIST.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/EVIDENCE_CUMULATIVE_INTEGRATION_RELEASE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/PATCH_ORDER.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/README.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/SYMTHAEA_MUSIC_THEORY_PATCH_SERIES_23_PLAN_2026-07-21.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/patches/0001-audit-freeze-exact-Series-16-22-input-ledger.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/patches/0002-feat-tools-add-clean-room-patch-replay.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/patches/0003-test-release-require-authored-and-replayed-tree-identity.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/patches/0004-audit-cargo-inventory-all-targets-features-and-optional-dependencies.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/patches/0005-fix-build-close-target-module-and-feature-integration-gaps.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/patches/0006-ci-add-minimal-default-and-all-feature-cargo-matrix.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/patches/0007-ci-add-supported-architecture-and-endianness-serialization-lanes.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/patches/0008-ci-add-clean-nix-build-and-offline-reproduction-lanes.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/patches/0009-test-conformance-run-rust-and-independent-verifiers.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/patches/0010-feat-release-rebuild-all-public-artifacts-deterministically.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/patches/0011-test-release-add-negative-control-campaign.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/patches/0012-feat-evidence-generate-machine-readable-claim-matrix.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/patches/0013-docs-generate-implementation-status-from-claim-matrix.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/patches/0014-test-release-run-clean-room-end-to-end-rehearsal.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/patches/0015-refactor-release-split-replay-build-conformance-and-package-stages.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-23-authoring-kit/patches/0016-docs-evidence-publish-cumulative-integration-release.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/APPLY_README_TEMPLATE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/AUTHORING_CHECKLIST.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/EVIDENCE_BOUNDED_VERIFICATION_RELEASE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/PATCH_ORDER.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/README.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/SYMTHAEA_MUSIC_THEORY_PATCH_SERIES_24_PLAN_2026-07-21.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0001-audit-map-all-untrusted-input-and-complexity-surfaces.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0002-feat-verification-add-caller-owned-limit-policy.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0003-feat-verification-add-preflight-measurement-and-early-rejection.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0004-security-decoding-bound-depth-strings-and-collections.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0005-security-publication-bound-catalog-events-status-and-lineage.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0006-security-witness-bound-signers-mirrors-conflicts-and-external-calls.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0007-security-canonicalization-bound-total-and-per-object-bytes.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0008-security-tools-bound-subprocess-runtime-output-and-protocol.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0009-security-archives-stream-and-verify-before-extraction.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0010-security-archives-confine-paths-links-types-and-permissions.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0011-feat-verification-add-transactional-cancellation-safe-workflow.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0012-security-verification-cache-bind-complete-context.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0013-feat-conformance-add-stable-resource-failure-dimensions.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0014-test-security-freeze-malicious-artifact-and-archive-corpus.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0015-test-security-add-fuzz-property-and-complexity-replay.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0016-bench-security-measure-worst-case-valid-reference-bundles.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0017-feat-tools-report-verification-resource-usage.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-series-24-authoring-kit/patches/0018-docs-evidence-publish-bounded-verification-release.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/APPLY_README_TEMPLATE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/AUTHORING_CHECKLIST.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/EVIDENCE_STEWARDSHIP_AND_MAINTENANCE_CONTRACT.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/PATCH_ORDER.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/README.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0001-audit-freeze-post-release-stewardship-scope.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0002-feat-publish-support-and-versioning-policy.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0003-feat-add-structured-regression-intake.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0004-feat-add-regression-triage-ledger.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0005-test-promote-every-confirmed-defect-to-frozen-fixture.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0006-feat-add-corrective-patch-template.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0007-feat-add-backport-and-supported-branch-policy.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0008-security-add-coordinated-vulnerability-disclosure-workflow.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0009-security-run-recurring-dependency-and-advisory-gates.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0010-test-run-periodic-clean-room-reproduction.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0011-test-run-periodic-independent-verifier-conformance.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0012-test-run-long-history-and-restart-surveillance.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0013-test-run-periodic-crash-and-fault-game-days.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0014-feat-add-schema-and-api-deprecation-ledger.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0015-feat-add-release-delta-and-risk-report.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0016-feat-add-maintenance-release-evidence-bundle.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0017-ci-add-maintenance-branch-release-gates.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0018-feat-add-maintainer-rotation-and-stewardship-handoff.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0019-feat-add-quarterly-maintenance-scorecard.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0020-security-add-emergency-software-release-procedure.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0021-test-audit-maintenance-process-with-seeded-regressions.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0022-docs-publish-series-29-stewardship-and-maintenance-contract.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-stewardship-series-29-authoring-kit/patches/0023-chore-freeze-maintenance-ready-release-line.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/APPLY_README_TEMPLATE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/AUTHORING_CHECKLIST.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/EVIDENCE_TERMINAL_RETIREMENT_RELEASE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/PATCH_ORDER.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/README.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0001-audit-freeze-trust-exhaustion-and-retirement-threat-model.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0002-feat-retirement-model-caller-owned-trust-exhaustion-policy.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0003-feat-retirement-evaluate-trust-exhaustion-triggers.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0004-feat-retirement-model-terminal-retirement-plan.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0005-feat-retirement-add-multi-role-retirement-authorization.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0006-feat-retirement-model-terminal-retirement-receipt.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0007-feat-retirement-commit-terminal-transition-transactionally.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0008-security-retirement-block-all-authoritative-mutations.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0009-security-retirement-terminally-revoke-delegations-and-allowances.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0010-security-retirement-freeze-authority-and-witness-rotation.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0011-feat-retirement-add-archive-only-operating-mode.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0012-feat-retirement-preserve-terminal-catalog-checkpoint.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0013-feat-retirement-model-successor-handoff-with-explicit-discontinuity.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0014-security-retirement-require-new-identity-for-any-successor-publication.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0015-feat-retirement-add-public-terminal-disclosure-package.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0016-feat-retirement-model-custody-and-preservation-obligations.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0017-feat-retirement-add-independent-retirement-observer-statements.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0018-feat-retirement-decommission-mutation-tools-and-endpoints.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0019-feat-schema-register-retirement-and-archive-contracts.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0020-feat-api-export-curated-retirement-surface.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0021-feat-tooling-add-trust-exhaustion-report-command.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0022-feat-tooling-add-terminal-retirement-command.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0023-test-retirement-cover-trigger-policy-and-authorization-replay.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0024-test-retirement-cover-atomicity-and-post-retirement-blocking.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0025-test-retirement-cover-archive-successor-and-privacy-boundaries.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0026-docs-retirement-define-terminal-authority-release.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-series-25-authoring-kit/patches/0027-docs-add-series-25-landing-and-application-guides.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/APPLY_README_TEMPLATE.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/AUTHORING_CHECKLIST.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/EVIDENCE_TERMINAL_RETIREMENT_SLICE_CONTRACT.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/PATCH_ORDER.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/README.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0001-audit-freeze-terminal-retirement-scenario.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0002-test-add-qualified-active-lineage-baseline.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0003-refactor-add-terminal-retirement-identities.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0004-feat-add-verifier-owned-trust-exhaustion-policy.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0005-feat-implement-trust-exhaustion-report.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0006-feat-implement-terminal-retirement-plan.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0007-feat-add-multi-role-retirement-statements.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0008-feat-implement-multi-role-retirement-authorization.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0009-feat-implement-planned-and-committed-retirement-receipts.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0010-refactor-extend-transition-gate-for-retirement.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0011-feat-extend-reference-store-for-terminal-transition.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0012-feat-commit-terminal-retirement-atomically.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0013-security-terminally-revoke-all-delegations-and-allowances.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0014-security-freeze-authority-and-witness-rotation.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0015-security-block-every-authoritative-mutation-path.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0016-feat-implement-archive-only-operating-mode.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0017-feat-implement-terminal-catalog-checkpoint.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0018-feat-implement-archive-custody-ledger.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0019-feat-implement-successor-handoff-with-discontinuity.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0020-feat-implement-public-retirement-disclosure-package.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0021-feat-add-terminal-observer-statements.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0022-feat-register-terminal-retirement-schema-prefix.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0023-feat-add-shell-free-external-verifier-retirement-roles.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0024-feat-add-trust-exhaustion-and-retirement-plan-cli.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0025-feat-add-terminal-retirement-and-archive-cli.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0026-test-run-positive-active-lineage-to-archive-only.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0027-test-run-retirement-policy-replay-and-staleness-corpus.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0028-test-run-retirement-race-crash-and-post-blocking-matrix.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0029-test-run-archive-successor-privacy-and-observer-corpus.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0030-feat-freeze-independent-terminal-retirement-vectors.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0031-ci-add-terminal-retirement-qualification-lanes.patch-plan.md
symthaea/crates/domains/symthaea-music-theory/symthaea-music-theory-terminal-retirement-slice-series-35-authoring-kit/patches/0032-docs-publish-series-35-terminal-retirement-report.patch-plan.md
```
