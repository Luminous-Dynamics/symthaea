# MEL-MR-EVAL-035 — Replay-verifiable, topology-specific tail-profile registry

**Status:** first source implementation authored on 12 November 2026. Ten Rust unit tests are authored. Compilation, formatting, test execution, production signature-adapter integration, real renderer calibration and production tail-guard activation are not yet qualified.

**Tracking issue:** [#9337 — Build replay-verifiable, topology-specific tail-profile registry](https://github.com/Luminous-Dynamics/symthaea/issues/9337)

**Stack:** EVAL-017 canonical [PR #9128](https://github.com/Luminous-Dynamics/symthaea/pull/9128) → EVAL-025 [PR #9218](https://github.com/Luminous-Dynamics/symthaea/pull/9218) → EVAL-034 [PR #9330](https://github.com/Luminous-Dynamics/symthaea/pull/9330) → this EVAL-035 tranche.

## 1. Research question

Can Muse promote a non-zero post-event tail guard only when it is tied to the exact renderer and complete output topology; estimated from replay-verified event measurements; calibrated and qualified on disjoint source-composition lineages; supported by finite-sample confidence bounds; signed by an authorized publisher; and rejected when a runtime render differs from the exact qualified topology?

This is not another acoustic detector. EVAL-017 measures paired-render influence. EVAL-034 keeps raw measured support, hysteresis support and separately declared guard padding distinct. EVAL-035 determines whether a particular non-zero guard policy is qualified for a particular renderer topology.

## 2. Paradigm shift: a tail guard becomes a calibrated capability

A profile digest alone proves only that a report refers to some bytes. It does not prove that the publisher is trusted, that the renderer/route graph matches, that the guard covers unseen source compositions, or that repeated events/windows from one composition were not counted as independent samples.

EVAL-035 introduces an opaque VerifiedTailProfile. It can be created only by resolving a signed, qualified profile through a registry snapshot whose digest is pinned by a trusted caller. The token can apply its guard to an otherwise unguarded EVAL-034 configuration only when renderer identity, renderer configuration, sample rate and channel count match exactly.

The profile is topology-specific. There is no global music-tail profile and no default guard. Unknown routing remains unknown.

## 3. Evidence and trust model

### 3.1 Complete topology evidence

TailTopologyEvidence binds:
- renderer identity and exact render configuration;
- SoundFont digest;
- effects-group assignment and effects settings;
- complete downstream route graph, including sends/returns, buses, plugins, master processing and output stage;
- output chain and PCM format;
- EVAL-034 segmentation-policy digest;
- dataset revision, source-domain profile and lineage-sampling protocol;
- profile-builder artifact;
- operation family, sample rate, channel count and publisher key ID.

The evidence is accepted only when all digests are valid, its domain-separated digest replays, a signature verifier accepts the publisher key, and routing_complete is true.

The EVAL-025 adapter currently marks routing incomplete because it has only a typed effects-group subgraph, not the complete downstream/master/plugin/output graph. Therefore current EVAL-025 reports cannot mint a verified topology proof or qualify a production tail profile. This is the intended fail-closed result.

TailEvidenceSignatureVerifier is an integration boundary. The production implementation must use the project's trusted key registry, key validity and revocation policy. The unit-test verifier is deterministic scaffolding, not a production signature algorithm. A caller-supplied public key is not itself a trust root.

### 3.2 Source-composition lineage

Events, channels, windows, repeated takes and repeated renders from one composition are not independent samples.

TailSourceLineageEvidence binds:
- a root composition-lineage digest;
- the exact observed source Score digest;
- a provenance-chain digest;
- an authorized publisher identity and signature.

The EVAL-017 report and replay-verified renderer binding must agree with the observed Score digest. Tail observations are grouped by the signed root composition lineage, so different edits or versions from the same composition remain one statistical unit.

This does not independently discover composition ancestry. The upstream lineage publisher and trusted-key registry remain explicit dependencies; a free-form caller string is not accepted as the independent-sample identity.

The signed source-domain profile defines the target composition population (for example, instrument families, register, polyphony, rhythmic/meter families, duration and style strata); the sampling-protocol digest freezes how lineages enter calibration and qualification. A profile is not transferable to a different source-domain or sampling policy without new evidence.

### 3.3 Replay-verified event tail

verify_tail_event_observation:
1. replays the complete EVAL-017 v0.2 report from its renderer-bound inputs and verified repeatability calibration;
2. requires the single-event Remove intervention currently supported by canonical EVAL-017;
3. requires EffectDetected above the calibrated no-op floor and non-empty hysteresis support;
4. rejects pre-applied tail guards and clipped-guard evidence;
5. checks baseline and counterfactual renderer identity/configuration, Score identity, sample rate and channels against the verified topology;
6. extracts the exact scheduled note-off frame from the baseline renderer token's performed-event lineage;
7. measures tail frames as max(0, hysteresis_support_end - scheduled_note_off);
8. binds the EVAL-017 report, both renderer receipts, repeatability calibration, source lineage and topology.

No-effect-above-noise-floor observations cannot silently contribute zero-length tails to a tail profile. The result is a conditional renderer-specific measurement, not exclusive sample ownership or perceptual irrelevance.

## 4. Statistical contract

### 4.1 One independent value per composition

For each source-composition lineage, the profile builder takes the maximum measured event tail over all admitted event observations. Every observed Score version must have complete, unique canonical event-index coverage for each exact baseline-render receipt. This permits multiple independent baseline takes/scenarios of the same Score while preventing an incomplete render scenario from silently omitting a long-tail event.

All events, channels, windows, and multiple baseline render scenarios from one root composition are still collapsed to one maximum-tail value for that root lineage. This is deliberately conservative and prevents pseudoreplication from compositions with many notes, windows, channels or repeats.

Calibration and qualification lineage sets must be digest-disjoint. The profile contains sorted lineage IDs, event-observation receipt digests and one maximum-tail value per lineage.

Before clustering, every observed source Score version must have exact coverage of its canonical event-index range: each Score event index appears exactly once, the declared event count is consistent, and duplicates or missing events reject. This prevents a profile from being built from a conveniently selected subset of low-tail events. Different Score versions may belong to the same signed root composition lineage; each version must be complete, then the maximum across all of its events/versions contributes one value for that root lineage.

### 4.2 Distribution-free upper confidence bound for the requested quantile

The builder uses an exact binomial order-statistic rule.

For n independent calibration lineages and requested population quantile q, choose the smallest one-based rank k satisfying:

P(Binomial(n, q) <= k - 1) >= c

where c is the requested one-sided confidence level. The sample order statistic X(k) is then a distribution-free upper confidence bound for the population q-quantile, under independent, exchangeable source-lineage sampling from the declared topology domain.

For the default 95th percentile at 95% confidence:
- n=58 cannot satisfy the condition at any observed order statistic;
- n=59 can, using the sample maximum.

The implementation therefore defaults to at least 59 independent calibration lineages. The profile records the rank, observed order statistic and upper-confidence frame value separately.

The guard is:

G = X(k) + M

where M is an explicitly declared safety margin in output frames. A guard above the predeclared maximum is rejected.

This is a distribution-free quantile bound, not a parametric tail model. It assumes the calibration lineages are independent/exchangeable for the exact topology and operation family. A renderer/topology change requires a different profile or a separately qualified invariance claim.

### 4.3 Independent qualification split

The separate qualification split measures the rate at which a held-out composition's maximum event tail exceeds the proposed guard.

The profile reports a one-sided exact Clopper-Pearson upper bound for that miss probability, computed at the declared confidence level and rounded upward to integer parts per million. Promotion requires this upper bound to meet the frozen maximum miss-rate target.

With the defaults:
- at least 59 independent qualification lineages;
- 95% one-sided confidence;
- maximum miss-rate upper bound of 5%.

For 0 misses in 59 independent qualification lineages, the one-sided upper bound is just under 5%. One miss in 59 exceeds the 5% target. This is a stringent initial screen; larger samples are required to tolerate observed misses while maintaining the same upper-bound requirement.

Both splits are grouped by root composition lineage. Repeated event observations cannot increase n. No threshold is tuned on the qualification split.

## 5. Registry and runtime resolution

TailProfile commits to exact topology, calibration/qualification lineage sets, event-observation digests, per-lineage maximum tails, quantile rank/confidence bound, safety margin/final guard, qualification miss count/upper bound, and disposition/reasons.

verify_tail_profile recomputes the profile from opaque verified observations. A report digest alone is not enough.

SignedTailProfile binds the profile digest, publisher key ID, issue time and expiry time into the signature message. Changing the expiry or publisher metadata invalidates the signature.

TailProfileRegistrySnapshot is content-addressed and includes exact profile/topology digest pairs, authorized publisher ID, entry validity interval, profile revocation timestamp, revoked publisher key IDs and snapshot validity interval.

The resolver requires the caller's expected registry digest to match the snapshot. That expected digest must come from a trusted/pinned distribution channel; an untrusted local registry cannot establish its own authority. The resolver rejects missing or duplicate entries, unknown or revoked publisher keys, expired snapshots/profiles/entries, revoked profiles, invalid signatures, unqualified profiles and topology mismatches.

Revocation and supersession create a new registry snapshot. Historical reports are never rewritten.

## 6. Implementation

Added and exported src/tail_profile_registry.rs with:
- signed topology-evidence contract and opaque verified topology token;
- signed source-composition lineage contract and opaque verified lineage token;
- EVAL-017 v0.2 replay adapter to event-specific tail observations;
- per-composition maximum-tail clustering across complete per-render Score-event maps;
- exact binomial order-statistic quantile UCB;
- one-sided Clopper-Pearson miss-rate upper bound;
- replay-sealed profile and independent replay verifier;
- signed profile and digest-pinned registry snapshot;
- validity, expiry, publisher/profile revocation and exact topology checks;
- opaque VerifiedTailProfile and exact-context EVAL-034 configuration handoff.

The module deliberately does not add a renderer, route parser, identity service, cryptographic key store, artifact registry, acoustic detector or editor.

The focused workflow requires root Cargo metadata to resolve. The current repository tree contains three source-only paths matched by workspace globs but lacking Cargo manifests (two core prototypes and one README-only REA verifier plan). This branch explicitly excludes those non-package paths rather than adding placeholder packages or weakening the metadata check. The exclusions should be removed when each directory becomes a real Cargo package.

## 7. Initial tests authored

Ten Rust tests cover:
1. incomplete route evidence and topology-digest tampering reject;
2. signed source-lineage evidence binds root composition and observed Score;
3. altered source lineage and invalid signature reject;
4. 58/59 sample-size boundary for q95/95% quantile UCB;
5. zero/one held-out misses against the 5% upper-bound target;
6. multiple events within one composition collapse to one maximum;
7. calibration/qualification lineage overlap rejects;
8. excessive held-out misses prevent profile qualification;
9. profile digest replay rejects a modified guard;
10. registry resolution checks publisher signature, revocation, pinned snapshot and exact topology; resolved profile cannot be applied to a different renderer.

These are source-authored tests, not executed results.

## 8. Benchmark plan and targets

Use a renderer/topology matrix, with source-composition lineage held out as the independent unit.

Required topology strata:
- FluidSynth binary/build identity;
- SoundFont digest and instrument family;
- effects-group assignment/count;
- reverb, chorus and other effects settings;
- shared sends/returns and master buses;
- plugin/master processing and output chain;
- sample rate, channels and PCM format;
- EVAL-034 segmentation policy and operation family.

The complete route graph must be captured and independently verified. If it cannot be, no topology-specific profile is promoted.

Baselines and ablations:
1. zero tail guard;
2. one fixed global guard (diagnostic baseline only; not a production candidate);
3. per-topology empirical maximum without a confidence bound;
4. per-topology distribution-free quantile UCB;
5. UCB plus safety margin;
6. incorrect event/window-level counting (negative control that should reveal pseudoreplication);
7. correct source-lineage clustering;
8. incomplete route/topology evidence (must be unqualified).

Report independently:
- held-out tail-miss count and one-sided upper bound;
- requested quantile and upper-confidence frames;
- guard duration in frames and milliseconds;
- over-guard cost (excess protected duration and render/search cost);
- missed influence energy and miss severity;
- output-horizon clipping rate;
- worst topology/SoundFont/instrument/operation-family bounds;
- number of independent calibration and qualification lineages;
- profile resolution/revocation/expiry failures;
- runtime and memory cost.

Initial targets — not measured results:

| Dimension | Initial target |
|---|---:|
| Calibration composition lineages | >=59 per exact topology |
| Qualification composition lineages | >=59 per exact topology |
| q95 tail upper-bound confidence | >=95% |
| Held-out miss-rate upper bound | <=5% |
| Known-tail synthetic fixture coverage | 100% above declared measurement floor |
| Profile/registry/topology tamper controls | 100% rejection |
| Incomplete-routing handoff | 100% rejection |
| Cross-topology profile reuse | 0 without independent invariance evidence |

The 59+59 sample minimum is an initial distribution-free statistical floor, not a claim that 118 samples are sufficient for every topology or all possible SoundFont distributions. Worst-group bounds and uncertainty remain visible.

## 9. Competitive positioning

Raw music-generation leaderboards do not measure tail calibration, source preservation, provenance or renderer qualification. The relevant comparison is production-audio editing reliability and reproducible signal-processing evidence, not a direct Elo contest with prompt-to-song generators.

The current ICASSP 2027 MMAE challenge has two tracks: a Single Model system and an Agent system that can orchestrate multiple open-source models and DSP tools. The public development set is 2,000 examples with 17,741 rubrics; hidden test sets are 500 per track. The leaderboard ranks by EMR, then IFR, then CR. The official timeline lists November 25, 2026 as the final submission deadline; registration closed October 8, so entry eligibility/credentials must be confirmed rather than assumed.
Official sources: https://audio-editing-challenge.github.io/ and https://audio-editing-challenge.github.io/leaderboard/

SMEBench (October 27, 2026) provides 3,600 paired synthetic test examples across twelve edit operations plus a real-audio multitrack complement. It separates source preservation, target agreement and text compatibility; those must remain separate Muse metrics.
Paper: https://arxiv.org/abs/2610.36715

EVAL-035 is not intended to win those benchmark scores directly. It supplies a renderer-specific calibration capability that can reduce unsafe support assumptions and improve EVAL-019/022 downstream decisions. It must be evaluated by matched renderer topology, paired source-lineage holds and independently verified guard outcomes.

## 10. Claim boundary and next steps

This tranche implements a profile/registry contract and deterministic reference logic. It does not establish:
- that current FluidSynth topology is complete;
- that any real SoundFont/effects topology has 59+59 independent qualified compositions;
- that a production tail profile is currently promotable;
- that local evidence is attested against a malicious host;
- that measured influence captures all audible or perceptual effects;
- that a guard proves zero influence beyond its boundary.

Next:
1. obtain a signed, complete downstream route-graph capture; current EVAL-025 evidence is deliberately insufficient;
2. connect the trusted signature verifier to an existing project key/registry service;
3. build an independently verified source-lineage receipt adapter;
4. collect calibration and qualification compositions per topology;
5. validate exact binomial bounds independently and compare to conservative alternatives;
6. integrate only the opaque VerifiedTailProfile into EVAL-034 configuration handoff;
7. run exact-head Rust formatting, tests and Clippy; then perform held-out real FluidSynth/SoundFont studies.

No CI pass, acoustic calibration result, SOTA superiority or production tail guard is claimed.
