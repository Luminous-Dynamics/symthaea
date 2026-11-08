# Vocal evaluation bundle verifier

This standalone Rust utility independently verifies the integrity and consistency of the Muse vocal release-gate bundle.

## Usage

Run after the release gate has completed:

```bash
cargo run --manifest-path tools/vocal-run-verifier/Cargo.toml -- \
  verify --blind-dir audio_output/muse_vocal_gate/blind \
  --private-dir audio_output/muse_vocal_gate/private
```

Expected directory split:

```text
muse_vocal_gate/
  blind/
    blind_manifest.json
    ratings.template.json
    vocal-<id>.wav
  private/
    run_manifest.json
    private_backend_key.json
    objective_report.json
    human_reference_receipts.json
    release_status.json
```

## What is verified

- Exact source revision and run-manifest schema.
- Contiguous, unique case IDs and finite generation controls.
- Recomputed score BLAKE3 digest from the serialized note sequence.
- One-to-one blind clip/private-key correspondence.
- Blind identifiers recomputed from private salt, backend ID and case ID.
- Exact WAV bytes against the blind-manifest BLAKE3 digest.
- Safe single-component WAV filenames; no traversal or symlinked audio.
- Performance digest agreement between case receipt, blind clip and private key.
- Capability-manifest digest agreement between private key and objective report.
- One objective entry for every generated model clip and none for human references.
- Human-reference score/audio linkage and presence of pseudonym, consent and license references.
- Separation of blind and private directories.

## What is not verified

This verifier does not prove that the renderer actually loaded the model/voicebank tree whose hash was recorded, that a license is legally valid, that the audio matches its intended score, or that the singer is natural. It verifies receipt integrity and cross-file consistency only. The model/voicebank tree digest is a separate artifact receipt and must be bound to the renderer process before the bundle can claim full model provenance. Human-reference consent and license references are identifiers for external records; reviewers still need access to those records under the study's governance rules.

## Security posture

The verifier fails closed on malformed/missing receipts, duplicate IDs, unknown references, path traversal, symlinked audio, and changed WAV bytes. It reads only a caller-selected bundle and performs no network access or model execution.
