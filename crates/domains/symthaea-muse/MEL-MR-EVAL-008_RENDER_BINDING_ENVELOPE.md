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
- `create_symbolic_audio_render_binding` / `verify_symbolic_audio_render_binding`: lower-level API for explicit `LocalityEvent` sequences; it replays symbolic verification, re-runs the acoustic/onset verifiers against exact buffers and stored masks/configs, recomposes child reports, checks buffer/manifest digests, and requires exact receipt equality.
- `create_full_score_render_binding` / `verify_full_score_render_binding`: adapter for Muse's actual `symthaea_music_theory::Score` type. It records full serialized source/target Score hashes in addition to the narrower event projection receipt, and refuses unsupported scores or out-of-contract metadata changes.
- `AudioPreservationProfile` / `audio_preservation_profile_sha256`: explicit, versioned profile pin for the exact acoustic/onset configurations. The full-Score gate requires both child configs to equal the selected profile exactly and binds the profile digest into its receipt. A profile must be resolved from a trusted registry; hashing a caller-invented profile is not evidence that it is approved.

The binder does not accept a standalone composite audio report as authoritative. Acoustic and onset reports store their own masks and are independently replayed against the exact audio buffers before composition. The full-Score adapter derives the acoustic mask from the requested symbolic operation, tempo, sample rate, and renderer frame-latency offset; a caller cannot broaden the acoustic exemption. The onset mask is intentionally different: for pitch edits/transposition it must be empty, so the edited note's attack remains a required event. A naive broad time mask would otherwise hide deletion or delay of the very note that was supposedly edited. Re-sealing a fabricated metric is insufficient because it must reproduce from the supplied buffers. The acoustic report also records per-channel RMS-envelope deltas *inside* the authorized spectral region; pitch edits and transposition must preserve this envelope within the configured RMS tolerance, preventing an edit from deleting/attenuating the target note while passing because its spectrum was masked. The full-Score gate also requires the exact detector config embedded in the selected preservation profile—an arbitrary calibration string or permissive config is not enough. Profile approval itself remains an external trusted-registry responsibility.

The resulting binding commits to source/target symbolic hashes, the symbolic receipt hash, source/target audio hashes, the recomposed audio receipt hash, renderer-manifest hash, audio format, overall verdict, and reasons. The full-Score wrapper separately hashes every serialized Score field so changes to the original Score cannot be hidden by projecting only pitch/onset/duration into `LocalityEvent`.

The full-Score adapter derives the authorized audio-time mask from the symbolic operation and the source/target event intervals, then requires the acoustic and onset reports to carry exactly that mask. Beat positions are converted using the score tempo and audio sample rate; the renderer manifest includes a signed-integer frame-latency offset (not a digital signature), which is applied during mask derivation and must keep the expected mask inside the buffer. This prevents widening the ignored region or accidentally checking an unshifted region when a renderer adds known latency.

The full-Score adapter deliberately refuses scores outside the current verifier's scope: the score must be a single non-overlapping part/role with consistent total length; event insertions/deletions, key/tempo/meter changes, and changes to part, role, velocity, emphasis, or section intensity are rejected. The current audio gate accepts pitch-only edits and transposition; duration/onset edits fail closed until the detector can compare operation-conditioned timing/length rather than simple source/output onset equality.

The combined rule is non-compensatory: a symbolic failure or audio failure dominates; if symbolic passes but audio abstains, the overall result abstains; pass requires both to pass.

## Trust boundary — important

These are unkeyed content commitments. They detect substitution and report modification, but they do not prove that an untrusted renderer actually consumed the target score or that its output was causally generated from that score. Generate the envelope at the render boundary inside a trusted adapter. Stronger provenance requires trusted execution and/or a signed record with key lifecycle and verification rules.

A caller can still lie when building an envelope. The envelope describes a reproducible binding claim; it is not cryptographic attestation of a renderer. Likewise, `AudioPreservationProfile` is only a hash-bound policy object: the caller must obtain it from a trusted, versioned calibration registry or signature-verified manifest. This module does not authenticate that registry or profile approval.

## Regression cases

Six Rust unit tests cover:

1. matching symbolic/audio/manifest inputs verify and produce a pass;
2. substituting the target audio buffer is rejected;
3. substituting the target score is rejected;
4. modifying the embedded binding manifest is detected;
5. changing the manifest audio format is rejected;
6. tampering with an audio child receipt is rejected, and a failed symbolic result dominates passing audio evidence.

The render-binding module now has thirteen Rust test functions: six lower-level envelope tests, six full-Score tests for full-score hash/replay, rejected edited-note deletion, out-of-contract velocity mutation, simultaneous polyphonic notes, broadened authorization masks, and latency-shifted mask mismatch, plus a preservation-profile/config-drift test. An independent Python/NumPy sanity run over the synthetic source/target fixture found 110 protected acoustic windows, zero violations, and four matching protected onset events after the declared edit mask and onset-analysis guard. This only checks that the synthetic fixture is well-formed; it does not execute Rust or prove a causal render. All Rust tests are committed but not executed here. Compilation, rustfmt, clippy, and actual rendering validation remain pending.

## Next evidence gates

- Execute the feature-gated test suite and the existing MEL-MR-EVAL-006 conformance runner on the exact commit.
- Connect the full-Score adapter to a trusted render-boundary adapter that derives score hashes and audio hashes from the same invocation, rather than accepting caller-authored bindings.
- Test renderer version drift, instrument-bank swaps, effect-chain changes, numeric latency shifts, latency-policy changes, and altered source/target buffers.
- Preserve actual generated JSON receipts as immutable artifacts with exact code, renderer, asset, and configuration identities.
- Evaluate a matched editor ablation on a public task set; report IFR, CR, EMR, critical false accepts, abstentions, runtime, and cost separately.

This provides a typed join between symbolic and audio evidence, while keeping claims narrower than “the renderer has been proven trustworthy.”
