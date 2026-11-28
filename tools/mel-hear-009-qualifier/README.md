# MEL-HEAR-009 matched perception/reasoning qualifier

Standalone Rust 1.96 contract checker for the matched condition panel.

Run from the repository root:

```bash
cargo +1.96.0 fmt --manifest-path tools/mel-hear-009-qualifier/Cargo.toml --check
cargo +1.96.0 test --locked --manifest-path tools/mel-hear-009-qualifier/Cargo.toml
cargo +1.96.0 clippy --locked --manifest-path tools/mel-hear-009-qualifier/Cargo.toml --all-targets -- -D warnings
cargo +1.96.0 run --locked --manifest-path tools/mel-hear-009-qualifier/Cargo.toml -- benchmarks/mel/registry/mel-hear-009-perception-reasoning-synthetic-v0.json
```

This is a strict **episode-level matched-condition validator and diagnostic counter**. It consumes existing owner references for audio materialization, identity triads, musician annotations, sampling design and matched execution budgets. It does not reimplement any of those verifiers.

V0 condition arms:
- `oracle_symbolic`: diagnostic reasoning upper bound with an independently produced symbolic reference and a fixed downstream reasoning head.
- `audio_derived_symbolic`: audio-derived transcription/alignment and the same downstream reasoning head as the oracle arm.
- `raw_audio`: direct audio-only model.
- `audio_plus_predicted_structure`: direct audio plus model-predicted structure, with the same model/runtime/prompt/adapter/decoding identity as raw audio.
- `no_evidence_control`: prompt-only control, using the same model identity as raw audio.

The report gives per-arm item outcomes and paired diagnostic flags. All five arms must carry the same canonical question digest and shared budget-profile digest. V0 accepts only the declared `local_context_verified` evidence stage; client/provider attestation requires the upstream trusted stage verifier. One item is not a performance estimate, and no confidence interval or population inference is produced here. The oracle arm is an upper-bound diagnostic, not a deployable competitor. Digest references are commitments, not proof that remote providers consumed the input.
