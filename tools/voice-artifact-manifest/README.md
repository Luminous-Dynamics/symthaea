# Voice/model artifact tree manifest

A dependency-isolated Rust utility for producing and verifying content-addressed manifests for provisioned model, voicebank, tokenizer and configuration trees.

## Commands

```bash
cargo run --manifest-path tools/voice-artifact-manifest/Cargo.toml -- \
  create --root /models/kokoro --kind kokoro-model-and-voice --output kokoro-artifacts.json

cargo run --manifest-path tools/voice-artifact-manifest/Cargo.toml -- \
  verify --root /models/kokoro --manifest kokoro-artifacts.json
```

## Integrity model

Every regular file is represented by:
- normalized relative path using forward slashes;
- byte size;
- BLAKE3 digest of its content.

Entries are sorted by normalized relative path. The tree root digest is BLAKE3 over canonical JSON containing the schema, caller-supplied artifact kind, and complete sorted file-entry list. Absolute provisioning paths are excluded.

The utility rejects symlinks, special files, non-UTF8 path components, duplicate normalized paths, empty trees, and invalid root directories. Extra or missing files alter the root digest. Verification recomputes the entire tree and requires exact manifest equality.

This manifest is an integrity receipt, not a cryptographic signature or proof of licensing. Store a separate source/model revision and rights receipt; sign or independently anchor the manifest digest when the trust model requires authenticity.

## Intended voice assets

- Kokoro: model ONNX files, config/tokenizer, selected voice binary and any required phoneme tables.
- DiffSinger: linguistic, variance, acoustic and vocoder ONNX files; config; phone map; language IDs; speaker embeddings and voicebank metadata.
- Future physical renderer: all parameter tables, calibration data and source-model residual assets that affect output.

Do not include caches that are guaranteed not to influence rendering. Conversely, do not exclude a file because it is large or inconvenient if the runtime can read it.

## Important boundaries

- This utility lives outside the production Cargo workspace.
- A successful tree hash proves only that the listed directory contents match the receipt at verification time.
- It does not prove the model is the claimed upstream release, that the license permits the intended use, or that a renderer actually loaded these bytes.
- A production run must bind the verified root digest to the renderer instance that produced the audio; a CLI path or user-supplied model name is not sufficient.
