# MEL-MR-EVAL-008 — Symbolic-to-Audio Render-Binding Envelope

**Status:** implementation draft; Rust tests not executed  
**Tracking:** issue #8937  
**Depends on:** MEL-MR-EVAL-006 audio receipts and MEL-MR-EVAL-007 symbolic edit-locality receipts

## Research question

Can an edit system join two independently replayable evidence channels—the symbolic requested-change/protected-property result and the rendered-audio preservation result—without accidentally mixing source scores, target scores, audio files, renderer configurations, or receipts?

## Implemented API

The theory-feature-gated module `symthaea_muse::symbolic_audio_binding` provides:

- `RendererManifest`: renderer ID/version, render-config digest, optional instrument-bank and effects-config digests, sample rate, channel count, and latency policy.
- `renderer_manifest_sha256`: versioned, domain-separated manifest digest.
- `create_symbolic_audio_render_binding`: consumes source/target event sequences, symbolic evidence, source/target audio views, acoustic and onset reports, and renderer metadata.
- `verify_symbolic_audio_render_binding`: replays symbolic verification, recomposes the audio result from the two child reports, recomputes buffer and manifest digests, then requires exact receipt equality.

The binder does not accept a standalone composite audio report as authoritative. It recomposes the audio-only result from the two child reports so both child receipt digests are checked before a cross-modal envelope is created.

The resulting binding commits to source/target symbolic hashes, the symbolic receipt hash, source/target audio hashes, the recomposed audio receipt hash, renderer-manifest hash, audio format, overall verdict, and reasons. The combined rule is non-compensatory: a symbolic failure or audio failure dominates; if symbolic passes but audio abstains, the overall result abstains; pass requires both to pass.

## Trust boundary — important

These are unkeyed content commitments. They detect substitution and report modification, but they do not prove that an untrusted renderer actually consumed the target score or that its output was causally generated from that score. Generate the envelope at the render boundary inside a trusted adapter. Stronger provenance requires trusted execution and/or a signed record with key lifecycle and verification rules.

A caller can still lie when building an envelope. The envelope describes a reproducible binding claim; it is not cryptographic attestation of a renderer.

## Regression cases

Six Rust unit tests cover:

1. matching symbolic/audio/manifest inputs verify and produce a pass;
2. substituting the target audio buffer is rejected;
3. substituting the target score is rejected;
4. modifying the embedded binding manifest is detected;
5. changing the manifest audio format is rejected;
6. tampering with an audio child receipt is rejected, and a failed symbolic result dominates passing audio evidence.

The render-binding module has six Rust test functions covering the seven checks listed above (the changed-score and tampered-binding checks share one test). These tests are committed but not executed here. Rust compilation, rustfmt, clippy, and actual rendering validation remain pending.

## Next evidence gates

- Execute the feature-gated test suite and the existing MEL-MR-EVAL-006 conformance runner on the exact commit.
- Run a controlled render adapter that derives score hashes and audio hashes from the same invocation, rather than accepting caller-authored bindings.
- Test renderer version drift, instrument-bank swaps, effect-chain changes, latency-policy changes, and altered source/target buffers.
- Preserve actual generated JSON receipts as immutable artifacts with exact code, renderer, asset, and configuration identities.
- Evaluate a matched editor ablation on a public task set; report IFR, CR, EMR, critical false accepts, abstentions, runtime, and cost separately.

This provides a typed join between symbolic and audio evidence, while keeping claims narrower than “the renderer has been proven trustworthy.”
