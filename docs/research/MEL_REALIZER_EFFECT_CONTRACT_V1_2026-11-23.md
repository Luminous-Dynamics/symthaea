# Melothaea realizer capability and observed-effect contract v1

**Date:** 23 November 2026  
**Owner:** [#10196 — capability-qualified musical realization backends](https://github.com/Luminous-Dynamics/symthaea/issues/10196)  
**Related:** [#12841 — factorial causal control surfaces](https://github.com/Luminous-Dynamics/symthaea/issues/12841), [#13349 — independent output observation](https://github.com/Luminous-Dynamics/symthaea/issues/13349), [#13348 — output-observation closure](https://github.com/Luminous-Dynamics/symthaea/pull/13348), [#13358 — immutable RMI policy identity](https://github.com/Luminous-Dynamics/symthaea/pull/13358)

## The shift

A renderer's control list is not a capability guarantee. An HTTP 200 is not evidence a control took effect. An echoed "effective setting" is not proof the output changed. A symbolic score change is not proof the waveform changed, and a waveform change is not proof a musician perceived the requested effect.

The contract separates five stages:

1. **Declared capability** — exact renderer/model, adapter and environment identity; task and modality; control semantics; evidence level.
2. **Admission** — compare a backend-neutral request to the capability manifest. Unsupported tasks/modalities abstain before execution. A hard/exact requirement cannot be silently routed to a best-effort or advisory control; an explicit downgrade decision is required.
3. **Execution receipt** — exact input/output digest, backend/version/adapter, request/manifest digest, accepted/effective/ignored controls and timing.
4. **Independent observation** — complete output capture, independently identified observer and manifest, baseline/treatment measurements, sham/no-op measurement, optional direction reversal and collateral metrics.
5. **Assessment** — observed_as_expected, no_observed_effect, sham_confounded, direction_reversal_failed, collateral_contract_violated, provider_ignored, provider_rejected, or evidence_insufficient.

The Rust implementation is crates/domains/symthaea-muse/src/realization_contract.rs.

## Implemented in this increment

- BackendCapabilityManifestV1: version-pinned backend, renderer, adapter and environment identity; task/input/output modalities; control guarantee classes; evidence levels; conformance identifiers; seed/determinism/reset declarations; optional measured clock contract.
- MusicalRealizationRequestV1: source artifact identity, backend-neutral task and modality, exact target/protected regions, requested values, explicit guarantee requirements, consent/purpose reference and independent-observation policy.
- admit_realization_v1: pre-execution route admission; abstains on unsupported task/modality/control, requires explicit downgrade for weaker-than-requested control semantics.
- MusicalRealizationReceiptV1: binds exact request and capability manifest, backend/renderer/adapter, input/output bytes, control results, latency and verification policy.
- validate_against: checks receipt integrity and exact request/manifest identity, refuses receipts for unadmitted requests, verifies one-to-one control outcomes, binds output capture to output digest and checks required independent capture completeness.
- assess_control_effect_v1: only confirms expected effect when the provider says the operation was applied, a complete independent observation exists, the measured effect is in the preregistered interval, the sham stays within tolerance, required reversal is present and points the other way, and protected collateral dimensions remain within bounds.

The module contains deterministic synthetic fixtures only. It has no external API credentials, network calls or commercial backend dependency.

## Evidence semantics

| Evidence | Establishes | Does not establish |
|---|---|---|
| Provider documentation | Interface/control is described | A deployment accepts or applies it |
| API acknowledgement | Request was accepted syntactically | The effective setting or sound changed |
| Provider effective-setting receipt | Provider reports a configuration | Output followed the setting |
| Independent complete output capture | The exact output bytes and measured features | The effect is musically useful or perceived |
| Sham and reversal controls | Basic evidence against a generic drift/session explanation | All possible confounding is absent |
| Musician/listener study | Perceived target success, usefulness and agency for its sample/protocol | Exact event preservation unless separately measured |

This increment's digest receipts are not signatures, trusted timestamps or trusted execution attestations. Do not treat a caller-supplied observer_independent=true flag as cryptographic proof of independence. The trusted runner and attestation layer remain separate work, including [#13254](https://github.com/Luminous-Dynamics/symthaea/issues/13254). A production implementation must bind the observer identity to an independently managed execution role, not let a backend assert its own independence.

## Initial qualification targets

These are prospective engineering gates, not measured results:

| Measure | Initial target |
|---|---:|
| Unsupported task, modality or control executed | 0 |
| Exact request silently routed to best-effort/advisory control | 0 |
| Scored receipts missing request, manifest or output digest | 0 |
| Scored control outcomes missing independent complete capture | 0 |
| Sham/no-op effects above preregistered tolerance credited as causal | 0 |
| Required direction reversal absent or inconsistent but credited | 0 |
| Protected collateral metric outside tolerance but marked successful | 0 |
| Valid synthetic known-answer admission/effect fixtures accepted | 100% |

For actual musical operation, report target effect size and confidence interval, sham distribution, control reversal, cross-session persistence, collateral drift, output-capture eligibility, p50/p95/p99 end-to-end latency and human-rated relevance/agency separately. Never aggregate these into a single score that allows a preference win to conceal a preservation or control failure.

## Connect to existing benchmark owners

- [#12841](https://github.com/Luminous-Dynamics/symthaea/issues/12841) owns the 3^4 factorial symbolic response surface, contrast kernel, confirmatory seeds and exact symbolic invariance. Do not fork a second factorial harness.
- [#13349](https://github.com/Luminous-Dynamics/symthaea/issues/13349) owns live MIDI/audio loopback capture and event manifests.
- [#13348](https://github.com/Luminous-Dynamics/symthaea/pull/13348) owns complete-window observation closure.
- [#13358](https://github.com/Luminous-Dynamics/symthaea/pull/13358) binds immutable agent/partner identity across counterfactual arms.
- [#12886](https://github.com/Luminous-Dynamics/symthaea/issues/12886) owns the musician-perceived control, agency and authorship study.

## Competitive comparison policy

Keep separate scoreboards:

1. finished-track preference (e.g. Artificial Analysis vocal and instrumental boards);
2. timed control success (e.g. Taan TimingBench; it reports a ±500 ms event tolerance, which is not comparable to beat-phase error);
3. localized editing and preservation (ICASSP 2027 Audio Editing Challenge's IFR, Consistency Rate and Exact Match Rate);
4. symbolic response and invariance (#12841);
5. rendered effect and latency (this contract plus #13349);
6. reciprocal interaction and agency (RMI-Bench and #12886).

A backend is compared only on tasks its interface exposes. Use identical source assets, intent, protected regions, seed/candidate budget, model version and evaluation budget. Report modality limitations instead of imputing unavailable note-level metrics for audio-only backends.

## Remaining work

- Bind observer_identity_sha256 to an independently issued role certificate/attestation; a boolean alone is not an authority proof.
- Add actual symbolic/MIDI and audio-stream mock backends that exercise the same request while proving distinct output/verification capabilities.
- Add renderer-specific conformance probes with randomized sham and reversal trials; do not use provider self-report as the output observer.
- Add session carryover, reset, provider fallback, stale-version and cancellation fault injection.
- Only then add live backend adapters, each with version-pinned capabilities and documented rights/consent terms.

**Claim ceiling:** this is a first Rust contract and synthetic conformance kernel. It is not a live renderer integration, a musical-quality result, a real-world SOTA comparison or evidence of human collaboration.
