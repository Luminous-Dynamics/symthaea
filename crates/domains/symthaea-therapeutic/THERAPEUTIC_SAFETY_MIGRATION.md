# Symthaea Therapeutic Safety Migration

This snapshot changes the crate boundary from a clinically suggestive research API
to a supportive, fail-closed default surface. It does **not** establish clinical
validity, regulatory compliance, or fitness for autonomous care.

## Default-build guarantees

A default build now:

- discards response drafts containing scope violations instead of echoing them
  beneath a disclaimer;
- requires explicit intervention authorization context and exposes hard blockers;
- excludes computational psychiatry, consciousness protocols, named diagnostic
  hypotheses, and clinical-scale-like analogue outputs;
- preserves observed-versus-inferred signal provenance across the experimental
  protocol bridge;
- reports normalized model-inferred outcome metrics as `Option<f32>` and marks
  named instruments as `NotAdministered`;
- classifies crisis-indicator context and requires clarification for negated,
  historical, third-party, quoted, hypothetical, affect-only, or unclear-subject
  matches;
- uses jurisdiction-neutral crisis-resource placeholders unless a deployment
  explicitly selects or supplies a reviewed resource profile;
- uses compositional lexical HDC encodings for formulation, narrative, and shadow
  memory instead of whole-sentence random hashes;
- restores shadow encodings, migrates legacy dream-queue indices, and uses stable
  fragment IDs across compaction and eviction.

## Opt-in research and compatibility features

The following features are disabled by default:

| Feature | Surface | Prohibited default use |
|---|---|---|
| `experimental-computational-psychiatry` | Simulated psychopharmacology scenarios | Diagnosis, prescribing, treatment selection |
| `experimental-consciousness-protocols` | Experimental protocol and bridge APIs | Autonomous intervention or crisis decisions |
| `experimental-diagnostic-hypotheses` | Named diagnostic profile storage | Presenting hypotheses as diagnoses |
| `legacy-clinical-scale-analogues` | PHQ-9/GAD-7/ORS-range compatibility values | Displaying values as administered instruments |

Enabling a feature does not validate the feature. Callers remain responsible for
isolating research outputs from production decision paths.

## Required orchestration order

Production callers should maintain one authoritative pipeline:

1. Validate input and establish locale, consent, and data-use policy.
2. Detect crisis indicators and inspect `CrisisDisposition`.
3. Classify the requested support within scope.
4. Propose a bounded supportive strategy.
5. Call `EthicalEvaluator::evaluate_with_context`.
6. Refuse execution when any `EthicalBlocker` is present.
7. Render the response draft.
8. Pass the draft through `ScopeGuard::guard_response`.
9. Render only `GuardedResponse::rendered()`.
10. Record a redacted decision receipt without raw therapeutic content.

No experimental module should bypass this sequence.

## Migration notes

- Replace direct string use from `ScopeGuard::apply_disclaimers` with
  `ScopeGuard::guard_response` where violation metadata is needed.
- Replace `EthicalEvaluator::evaluate` with `evaluate_with_context`; the legacy
  method intentionally uses an unverified, fail-closed context.
- Replace transient shadow indices with `ShadowFragment::fragment_id` and
  `record_dream_result_by_id`.
- Replace `phq9_analogue`, `gad7_analogue`, `ors_analogue`, and `outcome_summary`
  with `inferred_outcome_metrics` and its normalized fields.
- Treat `None` inferred values as insufficient data, never as a zero symptom score.
- Use `SafetyPlan::template_with_resources` with deployment-reviewed resources.
- Honor `CrisisDisposition::ClarifyBeforeEscalation` before executing the
  provisional action.

## Validation status

The patch series has been checked for clean diffs, balanced delimiters, bounded
queue/state migrations, and internal API consistency by static inspection.
The construction environment did not contain `cargo`, `rustc`, or `rustfmt`, so
compilation, formatting, clippy, and test execution remain required before merge.

Recommended commands in the full Symthaea workspace:

```bash
cargo fmt --all -- --check
cargo check -p symthaea-therapeutic
cargo test -p symthaea-therapeutic
cargo test -p symthaea-therapeutic --all-features
cargo clippy -p symthaea-therapeutic --all-targets --all-features -- -D warnings
```

## Remaining blockers before clinical-facing deployment

- External crisis-detection evaluation with adversarial, multilingual, and
  subgroup-aware corpora.
- Human-factors testing of clarification and escalation flows.
- A consent, encryption, retention, deletion, export, and access-audit data vault.
- Independently reviewed jurisdiction and mandatory-reporting policy registries.
- Calibration and evidence ledgers for every model-derived threshold and proxy.
- Explicit prohibition tests proving experimental modules cannot influence
  default intervention selection.
