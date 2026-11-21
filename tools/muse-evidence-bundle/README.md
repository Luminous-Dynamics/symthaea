# MUSE Evidence Bundle Verifier

Standalone Rust CLI/library for validating the cross-layer MUSE evidence manifest.

## Command

```sh
cargo run --manifest-path tools/muse-evidence-bundle/Cargo.toml -- verify \
  path/to/muse-evidence-bundle.json path/to/evidence-directory
```

The evidence directory may contain:

| Path | Purpose |
|---|---|
| `case.json` | Frozen case JSON; the verifier replays its case digest and source-audio binding |
| `source-audio.wav` | Exact source WAV bytes, independently hashed and used for acoustic replay |
| `gain-envelope-contract.json` | Frozen gain-envelope contract used to reproduce the acoustic observer |
| `symbolic-result.json` | Sealed v1 symbolic evaluation result |
| `render-receipt.json` | Sealed v1 render execution receipt |
| `render-bundle/` | Full render artifact directory accepted by the render receipt verifier |
| `acoustic-result.json` | Sealed v1 acoustic observation result |
| `score-audio-result.json` | Reserved for a future registered Score-to-audio verifier |
| `human-acceptance.json` | Reserved for a future registered human-receipt verifier |

The CLI writes a JSON verification report to stdout. Exit codes are:

- `0`: all five gates independently pass;
- `2`: a verified failure or hard binding mismatch;
- `3`: one or more required gates abstain or have no registered verifier;
- `4`: malformed manifest, invalid bundle seal, unreadable inputs or verifier error.

## Verification design

- Parses a closed manifest wire shape and verifies its recursively key-normalized, domain-separated bundle seal.
- Calls `symthaea_muse::edit_evidence::verify_result_seal` and re-runs `evaluate_symbolic_case` against the frozen case, source/candidate Scores and actual rendered candidate WAV; compute-time-only fields are excluded from semantic comparison.
- Calls `muse_render_receipt::verify_receipt_bytes` and `muse_render_receipt::verify_bundle` for render receipts and full artifact directories.
- Calls `symthaea_muse::audio_edit_observer::verify_gain_envelope_observation_seal` and re-runs `observe_gain_envelope_v1` from the exact source WAV, rendered candidate WAV and gain-envelope contract; the supplied observer result must exactly match the recomputed result.
- Rechecks case/source/candidate Score/MIDI/WAV/contract/receipt digests across components. Candidate Score/MIDI/WAV bytes come from the render bundle; the frozen case and source WAV are supplied separately.
- Recomputes the five-gate outcome: any failure dominates abstention; any missing or unsupported required verifier abstains; no aggregate score compensates across gates.

## Deliberate abstention boundaries

The v0.1 verifier cannot yet produce an overall `pass`:

1. The current render receipt verifier checks byte and artifact integrity but does not establish trusted executor identity or prove that an untrusted renderer consumed the declared MIDI. Render execution therefore remains abstain even after its artifact directory verifies.
2. The Score-to-audio conformance receipt schema/public verifier is not yet registered.
3. The human-acceptance receipt schema/public verifier is not yet registered.
4. Isolated-stem acoustic observations abstain until a separate stem/material-lineage receipt authenticates the claimed source. Full-mix gain observations may pass their own acoustic gate.

These are intentional safety boundaries, not successful work deferred to a background process. The missing verifier interfaces must be implemented and tested before those gates can pass.

## Qualification

Seven Rust unit tests are authored for failure precedence, missing verifier abstention, stale bundle seal rejection, case/candidate binding mismatch, missing receipt rejection, and key-order-independent seals. They have **not** been compiled or executed. The local environment has no Cargo/rustc/rustfmt; no exact-head compile, format, test or Clippy result is claimed.

The component receipt formats are versioned and allowlisted in code. Unknown producer IDs or versions are abstentions, not passes.
