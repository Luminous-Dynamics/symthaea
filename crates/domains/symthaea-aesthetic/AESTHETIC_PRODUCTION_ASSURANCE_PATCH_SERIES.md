# Symthaea Aesthetic Production-Assurance Patch Series

This series extends the empirical-governance baseline without changing the meaning of intrinsic evidence or policy utility. It adds the machinery required to operate the aesthetic critic as an accountable production subsystem.

## Design boundary

The series does **not** claim to solve aesthetic judgement with one model. It ensures that deployed judgements are versioned, replayable, drift-aware, consent-aware, robust to bounded perturbations, and subject to explicit human accountability.

## Bundle A — Runtime contracts

1. `feat(registry): add versioned extractor contracts`
2. `feat(registry): verify extraction reports against contracts`

Adds a deterministic extractor registry with strict semantic versions, declared modalities and evidence channels, artifact/evidence schemas, deterministic-build flags, build fingerprints, compatibility resolution, stable snapshot digests, and exact report verification.

## Bundle B — Drift and robustness

3. `feat(drift): add batch assessment drift baselines`
4. `feat(drift): add sustained online drift alarms`
5. `feat(robustness): add policy perturbation and extractor audits`

Adds privacy-light assessment telemetry, Welford moments, batch drift comparison, optional-signal missingness drift, EWMA sustained alarms, policy sensitivity probes, monotonic confidence-collapse checks, grounding-collapse checks, and suspicious extractor-report audits.

## Bundle C — Consent and study privacy

6. `feat(privacy): add consent-aware k-anonymous study exports`
7. `feat(privacy): audit scoped consent for model training`

Adds scoped grant/revoke consent ledgers, contribution clipping, small-cohort suppression, non-identifying aggregate exports, explicit claim boundaries, and sequence-level audits for research, training, or publication consent. The export intentionally states that it is not differential privacy.

## Bundle D — Deployment closure

8. `feat(deployment): bind evidence lineage and operational release gates`
9. `feat(replay): add deterministic production replay corpora`
10. `feat(operations): add runtime service-level evidence`
11. `feat(overrides): add assessment-bound human override ledger`
12. `fix(operations): keep missing health metrics JSON-safe`
13. `docs: document production-assurance architecture`

Adds evaluation envelopes binding registry snapshot, extractor build, extraction report, and assessment evidence; operational gates combining governance, drift, robustness, and consent; conservative downstream decisions; deterministic replay corpora; runtime SLO evidence; and human overrides bound to the exact assessment and release.

## Recommended integration flow

```text
artifact reference
  -> resolve registered extractor contract
  -> extract and verify report against descriptor
  -> evaluate typed aesthetic assessment
  -> create evaluation envelope
  -> record assessment telemetry
  -> run batch / sequential drift checks
  -> run robustness suite
  -> audit study consent and evidence governance
  -> evaluate operational release gate
  -> produce downstream decision
  -> apply only assessment-bound, reasoned human overrides
  -> capture envelope in replay corpus
  -> record runtime service-health evidence
```

## Migration notes

- Existing assessment, extraction, study, and governance APIs remain unchanged.
- All new modules are additive and re-exported from `lib.rs`.
- Callers supply audit timestamps; this crate does not trust or query wall-clock time.
- Artifact payloads and raw participant identity are excluded from new persisted contracts.
- Registry and envelope digests use deterministic FNV-1a for reproducibility and cache identity, **not** as a cryptographic authenticity primitive. Downstream evidence bundles should additionally sign or cryptographically hash serialized artifacts.
- Human overrides never mutate the underlying assessment. They produce a separate effective decision retaining the original automated decision.
- Missing optional operational metrics serialize as `null`, never NaN.

## Verification expected in the parent workspace

```bash
cargo fmt --all -- --check
cargo test -p symthaea-aesthetic
cargo clippy -p symthaea-aesthetic --all-targets --all-features -- -D warnings
```

Also exercise at least one real integration for Muse, Canvas, voice, and the game director, then capture a production replay corpus before enabling automatic downstream decisions.
