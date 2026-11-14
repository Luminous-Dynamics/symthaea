# MEL-MR-RIGHTS-001 — Component-level music benchmark rights policy v1

**Date:** 14 November 2026  
**Status:** First executable contract slice; exact-head qualification pending  
**Tracking:** [Issue #12029](https://github.com/Luminous-Dynamics/symthaea/issues/12029)

## Purpose

The EVAL-035 materializer hashes actual rights/access-evidence bytes, but a digest match does not authenticate the issuer or decide whether a use is allowed. This increment converts a human-reviewed per-component rights record into a deterministic, purpose-specific admission decision. It deliberately does not parse license prose or make legal judgments.

## Paradigm

Use a component-license graph rather than one dataset-level cleared bit. Audio, MIDI/score, SoundFonts, stems, annotations, captions and derived target renders can inherit different terms. The effective collection decision is bounded by the most restrictive applicable component.

Unknown and machine-extracted-only terms are blocked. Explicit deny is deny. Allow is possible only when every relevant component has a human-review receipt and explicitly grants every purpose requested.

## Implemented contract

- Versioned component rights bundle and use-request schemas.
- Distinct purposes: internal research, evaluation, commercial evaluation, training, fine-tuning, redistribution and public release.
- Explicit permission states: allowed, denied, unknown.
- Separate review states: human reviewed, machine extracted, unreviewed. Human-reviewed records require an opaque reviewer reference, role, jurisdiction, UTC review timestamp, rationale and decision-record digest. These are traceability fields, not signature/identity authentication.
- Structured commercial-use, output redistribution and public-release gates.
- Attribution, notice and share-alike obligations are retained per component and carried into the decision receipt.
- Deterministic decision receipt binds the normalized component bundle, request, policy version, decision, reasons and output obligations.
- Replay recomputes the policy decision from the same input bundle and request.
- Sixteen authored tests cover explicit evaluation allow, non-commercial restriction, commercial-context gating, unknown permission, machine-extracted-only terms, missing/malformed review records, UTC timestamps, inherited restrictive components, attribution/share-alike propagation, receipt tampering, canonical ordering, unknown share-alike and distribution/public-release guards.

## Known SMEBench license facts — not a legal decision

The SMEBench paper says SME-Bench is CC BY-NC-SA 4.0 and component resources retain their original licenses, including MidiCaps CC BY-SA 4.0, MoisesDB CC BY-NC-SA 4.0, Arachno SoundFont CC BY-NC-SA 4.0, GeneralUser GS and FluidR3-GM CC BY 4.0, TimGM6mb public domain, and General Montage CC BY 3.0.

A commercial entity's internal evaluation is not automatically classified as non-commercial by this policy. The caller must make the real-use context explicit and a qualified reviewer must approve the purpose-specific record. No SMEBench audio/MIDI/SoundFont data is embedded in the test fixtures; only synthetic metadata records are included.

Primary source: https://arxiv.org/abs/2610.36715, Licensing and Ethics appendix.

## Explicit limits

- No license parser, automated legal interpretation, legal advice, issuer authentication or compliance certification.
- Reviewer references and decision-record digests are traceability/integrity fields, not digital signatures or identity proof. The reviewer is not authenticated by this policy.
- The evaluator checks the permissions declared in the record; it cannot prove that the human review was correct.
- The caller supplies the commercial-context declaration; the policy does not infer commerciality from an organization or transaction.
- Attribution/share-alike output flags are caller attestations. Downstream publishing workflows must independently verify the actual notices and license choices.
- Rights for performers, recording owners, publishers, composers, labels and collection contributors may differ; component records must enumerate each relevant rightsholder/claim as known.

## Validation state

Nine Rust test functions are authored. They have not yet been compiled or run. The exact-head Rust 1.96 workflow must pass formatting, tests and Clippy before any implementation qualification claim.
