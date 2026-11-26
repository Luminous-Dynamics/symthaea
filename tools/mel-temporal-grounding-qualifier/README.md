# MEL temporal-grounding qualification harness

This nested Rust package imports the exact production scorer and digest module by path. It is intentionally isolated from the main workspace so scorer qualification does not inherit unrelated workspace build or lint debt.

Run:

    cargo +1.96.0 test --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml
    cargo +1.96.0 clippy --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --all-targets -- -D warnings

The base temporal-grounding tests exercise artifact/clock-bound interval scoring and count-gated one-to-many aggregation.

The temporal-revision V2 API accepts one frozen ground truth, one scoring plan, and raw draft/final predictions. It computes both scores using the exact production scorer. The caller cannot inject derived score fields. The resulting audit reports revision gain, harmful edits, perfect-draft preservation, count-gated one-to-many improvement, and explicit absence-classification changes.

Evidence conditions are distinct: no evidence, local evidence bound to the exact recording and in-range frame window, or yoked evidence from a different audio digest. Structural digest references do not authenticate evidence bytes or prove that a model consumed them.

The revision audit is an evaluation layer, not a model-training loop, waveform observer, capture verifier or musician study.


## Matched evidence-ablation triplets

The evaluator can compare one NoEvidence, one LocalRetrieved, and one YokedRetrieved V2 case under the same experiment_unit_id. All three arms must share the identical raw draft prediction, truth annotations, scoring plan, model, prompt, adapter and source recording. Only the final prediction and evidence condition may differ.

The contrast reports local-minus-no-evidence and local-minus-yoked final F1, count-gated F1, draft-to-final gains, and whether local evidence beats yoked evidence on both revision gain and final F1. Missing/duplicate arms and changed drafts are rejected. This does not itself prove causal effects in a population; it enforces the paired experimental design so subsequent statistics can estimate them correctly.


## CLI replay

Run a single V2 case or a matched evidence-triplet JSON file:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-temporal-revision-audit -- path/to/case-or-triplet.json

Use a hyphen as the path to read JSON from standard input. For an array input, the CLI audits each V2 case and emits matched evidence-ablation contrasts. The committed synthetic triplet fixture is at fixtures/mel-temporal-revision-triplet.synthetic.json.


## Matched-trial runtime and budget identity

V2 case inputs also bind a source-lineage identifier, immutable runtime identity digest, decoding-configuration digest, maximum output-token budget, actual output tokens, maximum wall-time budget and observed elapsed wall time. The evaluator rejects actual usage beyond the declared budget. A matched NoEvidence/LocalRetrieved/YokedRetrieved triplet must share source lineage, runtime, decoding configuration and maximum budgets; actual token/time consumption is reported separately for each arm.

These fields are self-reported run metadata until backed by an independently authenticated execution receipt. They make the paired protocol's intended controls explicit; they do not authenticate the runtime or prove that the model consumed retrieved audio.


## Source-work-clustered uncertainty

The matched-evidence API returns per-experiment-unit paired contrasts. For inferential summaries, call bootstrap_matched_evidence_contrasts on those contrasts. It:
- averages repeated query units within each source_lineage_id;
- resamples source-work clusters with replacement using deterministic SplitMix64;
- uses the supplied nonzero seed and at least 1,000 replicates;
- reports percentile 95% intervals separately for F1 and count-gated one-to-many F1;
- reports separate ordinary and one-to-many cluster counts.

A computable interval is not automatically confirmatory. The report exposes a proposed 30-source-work floor and separate booleans for ordinary and one-to-many cohorts. The floor is a minimum reporting warning, not a substitute for preregistered power analysis; a credible study may need more clusters. Do not treat two synthetic source works as evidence of model improvement.

V2 runtime identity, decoding-configuration, output-token budget, actual token use and wall-time fields are included in reports and matched-arm checks. They remain self-reported unless bound to an independently authenticated execution receipt.


## Declared-action vs observed-edit consistency

V2 reports an observed edit class and a scorer-derived edit delta (predicted-count, TP/FP/FN and start/end boundary MAE changes). For declared actions other than Other, it checks the action against that observed delta:

- AddOccurrence / SplitMergedOccurrence must increase predicted count and true positives.
- RemoveOccurrence / MergeDuplicateOccurrence must decrease predicted count and false positives without reducing true positives.
- AdjustStartBoundary / AdjustEndBoundary must preserve predicted count and improve the corresponding matched-boundary MAE.
- NoChange requires identical raw prediction objects.
- Other remains unclassified rather than being treated as verified.

These checks reuse the production scorer's existing matches and TP/FP/FN; they do not introduce a second interval matcher. They assess whether the declared action is consistent with the observed change, not whether the model's internal reasoning was authentic or whether the named sub-action (for example, split vs add) is causally correct.


## Exact audio evidence-window verifier

The audio_evidence_window module recomputes SHA-256 over the exact source-audio bytes, decoded PCM bytes, each selected half-open PCM frame window, the ordered context manifest, and the full evidence manifest. Frame-to-byte offsets use checked integer arithmetic over channels × bytes-per-sample. The context digest binds the query-text digest, target/source audio, rights-scope receipt reference, PCM sample clock and format, decoder/profile identity, window coordinates, model/runtime/decoding identity, prompt/adapter/retrieval policy and exact context order.

Run the synthetic CLI replay with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-audio-evidence-window-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-audio-evidence-window.synthetic.json tools/mel-temporal-grounding-qualifier/fixtures/mel-audio-evidence-source.synthetic.bin tools/mel-temporal-grounding-qualifier/fixtures/mel-audio-evidence-pcm.synthetic.hex

The decoded PCM fixture is stored as hexadecimal text and decoded to raw bytes by the CLI before hashing. The source-audio fixture is read byte-for-byte. The report marks rights, decoder, model/runtime and runner receipt authentication false; request-body recomputation and model-internal input attestation also remain false. A digest reference is not legal authority, a signed execution receipt or proof of internal model consumption. Those stages must compose with #13254 and the selected backend's attestation capability.


## Exact audio evidence-window verifier (MEL-GROUND-009)

The `audio_evidence_window` module verifies a versioned manifest against actual source-audio bytes and decoded PCM bytes. It recomputes the source-file SHA-256, full decoded-PCM SHA-256, each selected half-open frame-window SHA-256, the ordered context-manifest digest, and the canonical full-manifest digest.

The context commitment includes the exact query-text digest, target/source audio identities, rights-scope receipt reference, PCM sample rate/channel/frame/sample-format tuple, decoder/profile identifiers, window coordinates and payload digests, model/runtime/decoding identities, prompt/adapter/retrieval policy, and exact ordered window list. Frame-to-byte offsets use checked integer arithmetic over channels × bytes-per-sample.

Run:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-audio-evidence-window-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-audio-evidence-window.synthetic.json tools/mel-temporal-grounding-qualifier/fixtures/mel-audio-evidence-source.synthetic.bin tools/mel-temporal-grounding-qualifier/fixtures/mel-audio-evidence-pcm.synthetic.hex

The decoded PCM fixture is stored as hexadecimal text and decoded to raw bytes by the CLI before hashing. The source-audio fixture is read byte-for-byte.

This is a byte/coordinate verification layer only. The report explicitly keeps rights-scope authentication, decoder identity authentication, model/runtime identity authentication, runner-receipt verification, request-body recomputation and model-input attestation false. The digests are content commitments, not signatures. The receipt/authentication bridge remains owned by #13254; capture closure remains #13348/#13349.


## Binding verified audio to revision trials

The `audio_revision_binding` module provides `audit_revision_with_audio_evidence`. It invokes the raw-byte audio verifier and the production V2 temporal-revision scorer in one path, then joins experiment unit, source-work lineage, query text, target artifact, evidence condition, model/runtime/decoding configuration, prompt, adapter, retrieval policy and the exact selected window coordinates/payload digest.

Local evidence must use the scored target's exact audio clock. Yoked evidence must remain a distinct source recording while retaining the same target query. A no-evidence revision case cannot be promoted by attaching an audio-window manifest. The resulting report says that materialized audio bytes are bound to the revision case but always keeps `model_input_attested=false`.

This integration prevents a byte-valid manifest from being reused with a different query or revision arm. It remains a client-side evidence binding; provider/runtime input acceptance and internal model consumption require authenticated execution evidence from #13254.


The evidence-window contract now distinguishes two work identities: `target_source_lineage_id` is the work being queried, while `source_lineage_id` is the work from which the audio window was materialized. LocalRetrieved requires equal lineages and the exact target audio digest. YokedRetrieved requires a different lineage and a different audio digest. A different encoding or crop from the same work is not accepted as a yoked source. The verification report returns both lineage IDs separately.

The revision binding also requires the selected frame-range/payload tuple to match **exactly one** manifest window. A manifest with duplicate matching windows is rejected as ambiguous rather than picking an arbitrary occurrence.


## Unsigned in-toto Statement consumer for #13254

The `audio_evidence_in_toto` module emits a standard in-toto Statement v1 JSON payload for the verified audio↔revision join. Its subjects bind the source-audio bytes, decoded PCM, every selected window, ordered context manifest, full evidence manifest, byte-verification report and temporal-revision report. The predicate records the experiment/source lineages, query, target/evidence audio identities, model/runtime/decoder configuration, window byte coordinates and each achieved assurance stage.

The payload type is `application/vnd.in-toto+json`; the function returns the exact statement JSON bytes and their SHA-256 so the existing #13254 DSSE signer/verifier can consume them. The target audio appears as a `target-audio-reference` subject because a yoked evidence run may not have supplied the target's raw bytes to this verifier.

This module deliberately sets `dsse_envelope_created=false` and `signature_verified=false`. It does not implement DSSE, Sigstore, key distribution, trust roots, signer-role policy or transparency verification. A signed runner/provider receipt must be produced and verified by #13254 before any stronger execution or model-input claim is allowed.


## Unsigned in-toto Statement payload

Run the integrated byte-verification, temporal-revision join and in-toto Statement builder with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-audio-evidence-in-toto -- tools/mel-temporal-grounding-qualifier/fixtures/mel-audio-bound-revision.synthetic.json tools/mel-temporal-grounding-qualifier/fixtures/mel-audio-evidence-window.synthetic.json tools/mel-temporal-grounding-qualifier/fixtures/mel-audio-evidence-source.synthetic.bin tools/mel-temporal-grounding-qualifier/fixtures/mel-audio-evidence-pcm.synthetic.hex

The CLI emits a wrapper containing the exact compact in-toto Statement v1 JSON bytes, payload type and payload SHA-256. Its subjects bind target-audio reference, source-audio bytes, decoded PCM, each window, context/full manifests, verification report and revision report. The predicate records `DigestLinked` evidence and keeps all trust/authentication flags false.

This is unsigned payload material for #13254. It is not a DSSE envelope, it has no signature, and it must not be treated as authenticated provenance until the existing signer and independent verifier validate it under the configured trust-root/role policy.
