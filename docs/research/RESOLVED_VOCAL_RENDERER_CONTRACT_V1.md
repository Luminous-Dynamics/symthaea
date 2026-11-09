# Resolved Vocal Renderer Contract V1

**Date:** 9 November 2026  
**Status:** research implementation on a branch stacked on VOCAL-STRICT-001; exact-head CI pending. No acoustic-quality or SOTA claim.

## Problem

The existing v2 SingingVoiceEngine consumes VocalPerformance. That object couples lyrics to the small Muse English G2P frontend and the legacy syllable-to-note shape. The newer pronunciation-plan and score-alignment work can represent selected pronunciation candidates, arbitrary syllable intervals, explicit rests, multi-note melisma, multiple syllables within a note, and beat-domain tempo maps. Before this change, the resolved-plan route was an inherent FormantSingingEngine method rather than a renderer-neutral interface.

## Implementation

The new feature-gated module is src/voice/resolved_singing_engine.rs.

It defines:
- ResolvedScoreInput: either a seconds-domain VocalScoreAlignment or a BeatDomainScore with explicit syllable bindings and phrase spans.
- ResolvedSingingRequest: a selected PronunciationPlan, score input, and executable SingingRenderPolicy.
- ResolvedSingingVoiceEngine: a renderer-neutral trait separate from the v2 SingingVoiceEngine. Its default render_request validates the selected pronunciation, score/tempo materialization, route-specific capabilities and backend identity before dispatch.
- ResolvedSingingOutput: exact audio stem plus a ResolvedRenderReceipt.
- ResolvedRenderReceipt v1: BLAKE3 digests of the canonical selected pronunciation plan, exact score input (preserving the beat-domain tempo map rather than only its materialized seconds), capability manifest, render policy, sample rate, sample count and exact f32 output buffer.

The output digest is domain-separated and includes sample-rate metadata. It is an audio-buffer identity, not a hash of the encoded WAV container. The surrounding Muse run bundle should continue to hash the exact WAV separately.

The integrated source-filter baseline, FormantSingingEngine, now implements the trait for its resolved-plan route. It handles both SecondsAlignment and BeatDomain sources by dispatching to the existing validated resolved-plan methods. Kokoro and DiffSinger remain v2-only; they have not been silently upgraded or treated as equivalent v3 renderers.

## Safety and claim boundaries

- The request is rejected before synthesis if the selected pronunciation is unresolved/OOV/unsupported, score intervals are invalid, or a required control is unsupported or disallowed as approximated/conditional.
- The renderer capability ID must match the route's returned stem backend ID.
- A receipt can be independently recomputed against the original inputs and exact f32 buffer; sample tampering, sample-rate tampering, changed score, changed selected pronunciation, changed capabilities or changed policy invalidates it.
- Receipt verification proves integrity and linkage only. It does not prove that the pronunciation is linguistically correct, that audio matches its score, or that the singer sounds natural.
- The current resolved formant path remains English-table limited, fixed-heuristic and source-filter based; it is not the unintegrated physical waveguide tract renderer, not a neural singer, and not a production-quality winner.
- The v2 interface remains for compatibility. The new trait is additive and does not change Kokoro/DiffSinger behavior.

## Tests

Authored tests cover:
- exact output-buffer digest binding to rate and float samples;
- object-safe trait dispatch through a resolved-plan request;
- successful independently rechecked receipt;
- output sample tampering rejection;
- capability/backend identity mismatch rejection;
- unsupported phrase-breath requirement rejected before dispatch;
- beat-domain materialization and receipt binding that retains original tempo-map inputs.

The exact-head workflow includes pinned Rust 1.96 formatting, the library feature compile, the existing formant resolved-plan regression suite, the new resolved-contract primitive tests, and the frozen-lockfile check. Do not treat the PR as qualified until those exact-head checks complete.

## Competitive significance

This moves Muse closer to a compiler architecture with a defined source language (selected pronunciation and beat-domain score), a route-specific capability contract, and a reproducible renderer result. A future neural score-conditioned backend can implement the same trait without inheriting the legacy G2P path or pretending to support controls it ignores.

The next comparison should feed one selected pronunciation/score graph into every backend that can honestly consume it. A renderer that only accepts text or a simplified v2 plan belongs in a separate compatibility track, not the same exact-control denominator.
