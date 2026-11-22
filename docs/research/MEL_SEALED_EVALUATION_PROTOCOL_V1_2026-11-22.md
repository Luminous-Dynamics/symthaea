# MEL Sealed Evaluation Protocol v1

**Date:** 22 November 2026  
**Programme:** Melothaea / MEL-BENCH  
**Implementation:** tools/mel-sealed-evaluation  
**Status:** standalone Rust reference evaluator and synthetic known-answer fixture; no human-study result.

## Why this increment exists

MEL-PREF-ASSAY-001 (#13212) checks whether human preference presentations are assigned and recorded correctly. MEL-JUDGE-GAP-001 (#13218) checks whether a machine judge preserves decisions under reversed presentation order and agrees with an independently supplied human-derived ranking. Those are important measurement contracts, but they do not, by themselves, create an operational barrier between a judge's inference context and the held-out human reference.

This increment makes that boundary explicit. The frozen manifest contains candidate assets and their provenance, the judge identity, the comparison protocol, a held-out split declaration and a committed pair-order seed. It deliberately has no human labels, human item strengths or reference-estimator outputs. The judge predictions bind the exact manifest bytes. A separately released human-reference package then binds both the manifest and frozen prediction bytes.

## Protocol: three immutable stages

### 1. Candidate and protocol freeze

The manifest binds each candidate's audio bytes, prompt, generator/version/configuration, rights-receipt reference, modality and genre. It also binds the judge adapter, judge prompt and settings, audio delivery, normalization, presentation UI and comparison protocol. Pair identities and rank-gap bins are fixed before evaluation. The assignment seed is committed in the manifest and revealed only with the completed prediction bundle.

### 2. Judge prediction freeze

The judge receives the permitted candidate inputs, not the human-reference package. Each pair is presented twice, with the left/right order reversed. The output records the selected side or explicit tie/abstention, response digest and optional probability for candidate A. The completed prediction JSON is frozen and its SHA-256 is recorded in the later reference package.

### 3. Human-reference release and independent scoring

A distinct reference custodian releases human-derived Bradley–Terry item strengths and pair-specific standard errors after the prediction freeze. The evaluator checks that the release binds the exact manifest and predictions, verifies the declared release chronology, reconstructs the order assignment and scores stable judge decisions against uncertainty-qualified human preferences.

The evaluator reports:
- total pairs and human-directional pairs;
- stable directional, stable tie, order-inconsistent and incomplete judge outcomes;
- gold-reference coverage, conditional accuracy and effective accuracy;
- Brier score when judge probabilities are supplied;
- near/medium/far results using pre-registered human-reference logit-gap bins;
- the source digests and claim ceilings carried by every input.

A pair is human-directional only when the absolute human-reference logit gap exceeds the pre-registered minimum gap plus the configured confidence multiplier times that pair's standard error. Close or uncertain pairs remain in the report but do not become forced gold votes.

## Paradigm shift

The important change is to move from a single benchmark file that co-locates judge inputs and gold labels to a **two-custodian, three-stage measurement protocol**. The generator and judge can be developed without access to the held-out human reference. The scorer only sees the reference after predictions are frozen. This turns benchmark leakage from a statement in a paper into a concrete data-flow boundary with byte identities.

The protocol also treats the evaluation instrument as part of the experimental subject: candidate assets, generator settings, normalization, audio delivery, UI, judge model, judge prompt, order assignment and estimator all receive stable identities. A score without those identities is not a reproducible result.

## Threat model and limitations

The byte-level guarantees in v1 are deliberately scoped. The evaluator recomputes SHA-256 over the three raw JSON documents and verifies the cross-document links. Audio, prompt, generator configuration, rights receipt, protocol, judge response and external cohort/estimator digest fields are currently validated as 64-character digest references; this crate does not open those underlying assets or receipts and recompute their hashes. Likewise, timestamp strings are syntax/order checked, not validated against a trusted timestamp authority. The workflow evidence bundle preserves the exact JSON inputs, source files, resolved Cargo.lock and known-answer report, with a SHA-256 manifest for those run artifacts.

The validator can check file hashes, identity equality, candidate/pair coverage, seed commitments, swapped-order conformance, response digests, and declared timestamp ordering. It cannot prove that a judge process never accessed a separate copy of the human labels, that audio was actually played, that a human heard it, or that a timestamp receipt is authentic. The current receipt fields are digest references only. Timestamp strings are declared metadata.

Production qualification therefore requires:
- a separate reference custodian and separate inference/scoring identities;
- external timestamp or transparency-log receipts whose signatures and trust chains are independently verified;
- access logs and policy enforcement showing that the judge cannot read the sealed package;
- held-out split lineage and overlap audits for prompts, audio, near-duplicates, generator training data and judge training data;
- rights and consent receipts validated by the relevant policy authority;
- replayable UI/playback receipts and clock qualification for human studies.

A future adapter may validate a signed timestamp authority or append-only transparency receipt, but v1 deliberately does not claim that capability.

## Benchmark plan and prospective gates

These are targets, not achieved Melothaea results.

| Dimension | First real-study gate | Stretch target |
|---|---:|---:|
| Manifest/prediction/reference byte-binding failures | 0 tolerated | 0 |
| Candidate/pair coverage | 100% | 100% |
| Swapped-order integrity | 100% assigned and replayed | 100% with independent receipt validation |
| Judge order-inconsistency on clear human-reference pairs | Report all; no hidden exclusions | ≤2% with 95% interval |
| Judge effective accuracy on far-gap held-out pairs | ≥90% | ≥95% |
| Conditional accuracy on directional held-out pairs | ≥90% | ≥95% |
| Coverage of human-directional pairs | ≥90% | ≥95% |
| Spearman correlation to human Bradley–Terry ranking | ≥0.85 | ≥0.90 |
| Brier score and calibration | Report with uncertainty | Improve over CMI-RM and strong multimodal baselines on the same split |
| Cross-generator and cross-genre transfer | Report every stratum | No material subgroup collapse |
| Gold leakage / unauthorized reference access | 0 | 0 |

Do not optimize only the point estimate. Require cluster-aware bootstrap intervals over raters and prompts, report the item/prompt split unit, and keep a final untouched private set. For a fair industry comparison, reproduce MIREX CMI-RewardBench on its official 249-example core and 618-example extension, preserve the official input/output conventions, report vocal and instrumental subsets separately, and disclose data, parameters, latency and compute. Only then run a separate Melothaea holdout study.

## Industry context

Artificial Analysis' Music Arena currently uses blind human pairwise comparisons, Bradley–Terry maximum-likelihood ratings and 95% intervals, with distinct vocal and instrumental boards and standardized audio delivery. MIREX 2026 CMI-RewardBench evaluates reward models against human preferences over musicality, text–music alignment and compositional instruction alignment. Its published core-set results list 73.9% for the leading entry, 65.9% for CMI-RM and 59.0% for the always-choose-B baseline. These are external baselines on their own datasets, not Melothaea results.

Suno Studio 2.0 now combines generative workflows with automation, effects, MIDI, a synth and conversational control; Suno MCP exposes creation, editing, stem/MIDI operations and library management to connected assistants. ACE Studio 2.0 spans AI voices, instruments, generative kits and an arrangement-oriented workstation. Google DeepMind describes Lyria 3.5 as its latest music model and Lyria RealTime as designed for interactive generation. These capabilities make a generator wrapper an increasingly weak differentiator. Melothaea's defensible product thesis is the rights-aware, renderer-independent collaboration and agency layer above generators, with independently qualified preference and interaction evidence.

## Synthetic conformance fixture

The five-pair fixture intentionally contains two correct stable directional judgments, one stable but incorrect judgment, one order-inconsistent judgment, and one stable tie on a human-uncertain pair. The known-answer report therefore has 4 human-directional pairs, 3 stable directional judge outputs, 1 order-inconsistent output, 1 stable tie, 2 correct directional outputs, 75% reference coverage and 50% effective accuracy. These values test the evaluator; they are not empirical quality results.

## Claim ceiling

A successful validator run means that the supplied files satisfy this v1 reference contract. It does not establish real listener preference, musical quality, judge superiority, rights clearance, a trusted timestamp, sealed runtime isolation, proof of human playback or production integration. No external benchmark result is inferred from the synthetic fixture.
