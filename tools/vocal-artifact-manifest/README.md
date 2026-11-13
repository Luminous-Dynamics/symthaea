# Vocal artifact tree manifest

A standalone Rust utility for content-addressing a provisioned voicebank or model directory. It is independent of the main Cargo workspace and does not download, load, or execute model files.

## Create a manifest

```bash
cargo run --manifest-path tools/vocal-artifact-manifest/Cargo.toml -- \
  --create /models/diffsinger-voicebank /evidence/diffsinger-voicebank.manifest.json
```

## Verify a manifest

```bash
cargo run --manifest-path tools/vocal-artifact-manifest/Cargo.toml -- \
  --verify /models/diffsinger-voicebank /evidence/diffsinger-voicebank.manifest.json
```

The JSON contains sorted normalized relative paths, per-file BLAKE3-256 hashes and byte sizes, aggregate file/byte counts, and a domain-separated root digest. It does not contain absolute host paths.

## Fail-closed rules

- The root must be a real directory.
- Symlinks and special files are rejected rather than followed.
- Non-UTF-8 or non-canonical relative paths are rejected.
- Files that appear to change during hashing are rejected.
- A manifest must have a supported schema, strictly sorted unique paths, valid file digests, consistent aggregate counts, and a correct root digest.
- A verification mismatch reports a bounded list of missing, added, and changed paths.
- A manifest output file must be outside the artifact root, preventing self-inclusion and unstable repeated hashing.

The tool hashes file contents and relative names, not timestamps, permissions, absolute paths, or owner metadata. Two byte-identical artifact trees with the same relative paths therefore produce the same root digest. An external signature or trusted receipt is still required to establish who produced a manifest; a hash alone proves integrity relative to the manifest, not authenticity.

## Intended Muse integration

Provision the renderer outside the production process, run this tool, then attach the root digest to the corresponding backend identity and case receipts. Never treat a voice ID, directory path, model version label, or URL as a content digest.

## Validation

Six deterministic tests cover stable ordering, content and path changes, tampered manifest entries, missing/added/changed files, and symlink rejection. Exact-head CI evidence is required before claiming those tests pass.
