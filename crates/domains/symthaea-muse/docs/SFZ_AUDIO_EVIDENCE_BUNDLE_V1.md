# Muse SFZ Audio Evidence Bundle v1

This note specifies the input manifest accepted by `sfz_audio_evidence_verify`. It is an **integrity/provenance-link verifier**, not an audio-quality score, cryptographic signature verifier, or SFZ compatibility certification.

## Run

From the repository root:

```sh
cargo run --manifest-path crates/domains/symthaea-muse/Cargo.toml \
  --bin sfz_audio_evidence_verify -- --manifest /path/to/bundle/bundle.json

cargo run --manifest-path crates/domains/symthaea-muse/Cargo.toml \
  --bin sfz_audio_evidence_verify -- --manifest /path/to/bundle/bundle.json \
  --output /path/to/new-verification.json
```

The optional output path must not already exist. The verifier emits a JSON receipt to stdout. Exit 0 means the listed bytes and cross-links verified; exit 1 means evidence mismatches; exit 2 means an invalid manifest or setup/IO error. **Exit 0 is not an audio-quality pass.**

## Bundle layout

Keep all referenced artifacts under the directory containing `bundle.json`, with no symlinks. A minimal directory layout is:

```text
bundle/
  bundle.json
  fixture-pack/
    pack.sha256
    fixtures.json
    instrument.sfz
    performance.mid
    samples/
      note-01.wav
  renderer/
    muse-renderer
    reference-renderer
    muse-build-config.json
    reference-build-config.json
  captures/
    muse.wav
    reference.wav
  comparison.json
```

Paths in the manifest are relative to the manifest directory. Path traversal, absolute paths, symlinks in any component, non-regular files, and artifacts above 2 GiB are rejected. The tool never executes renderer commands. Keep the captured command as provenance text, not as a shell string for this program to run.

## Manifest example

The following is a shape example. Replace every `REPLACE_...` digest with an actual 64-character SHA-256 hex digest before using it.

```json
{
  "schema": "muse-sfz-audio-evidence-bundle-v1",
  "bundle_id": "sfz-fixture-0001-render-pair-01",
  "fixture_id": "round_robin_two_position",
  "fixture_pack_digest": "REPLACE_WITH_FIXTURE_TREE_DIGEST",
  "fixture_pack_receipt": {
    "path": "fixture-pack/pack.sha256",
    "sha256": "REPLACE_WITH_SHA256_OF_PACK_SHA256_FILE"
  },
  "fixture_manifest": {
    "path": "fixture-pack/fixtures.json",
    "sha256": "REPLACE_WITH_SHA256_OF_FIXTURES_JSON"
  },
  "sfz": {
    "path": "fixture-pack/instrument.sfz",
    "sha256": "REPLACE_WITH_SFZ_SHA256"
  },
  "midi": {
    "path": "fixture-pack/performance.mid",
    "sha256": "REPLACE_WITH_MIDI_SHA256"
  },
  "sample_assets": [
    {
      "path": "fixture-pack/samples/note-01.wav",
      "sha256": "REPLACE_WITH_SAMPLE_SHA256"
    }
  ],
  "candidate": {
    "engine_name": "Muse",
    "engine_version": "symthaea-muse <release-or-version>",
    "source_revision": "<full-commit-sha>",
    "executable": {
      "path": "renderer/muse-renderer",
      "sha256": "REPLACE_WITH_EXECUTABLE_SHA256"
    },
    "build_configuration": {
      "path": "renderer/muse-build-config.json",
      "sha256": "REPLACE_WITH_BUILD_CONFIG_SHA256"
    },
    "render_command": "muse-renderer --sfz fixture-pack/instrument.sfz --midi fixture-pack/performance.mid --wav captures/muse.wav",
    "output_wav": {
      "path": "captures/muse.wav",
      "sha256": "REPLACE_WITH_MUSE_WAV_SHA256"
    }
  },
  "reference": {
    "engine_name": "sfizz",
    "engine_version": "<exact-version>",
    "source_revision": "<full-commit-sha>",
    "executable": {
      "path": "renderer/reference-renderer",
      "sha256": "REPLACE_WITH_REFERENCE_EXECUTABLE_SHA256"
    },
    "build_configuration": {
      "path": "renderer/reference-build-config.json",
      "sha256": "REPLACE_WITH_REFERENCE_BUILD_CONFIG_SHA256"
    },
    "render_command": "reference-renderer --sfz fixture-pack/instrument.sfz --midi fixture-pack/performance.mid --wav captures/reference.wav",
    "output_wav": {
      "path": "captures/reference.wav",
      "sha256": "REPLACE_WITH_REFERENCE_WAV_SHA256"
    }
  },
  "comparison_receipt": {
    "path": "comparison.json",
    "sha256": "REPLACE_WITH_COMPARISON_JSON_SHA256"
  }
}
```

The fixture pack generator writes the tree digest as the first whitespace-delimited token in `pack.sha256`; the verifier compares that token with `fixture_pack_digest`, while separately checking the SHA-256 of the receipt file itself.

## What is verified

- Every declared SFZ, MIDI, sample, engine executable, build-configuration file, captured WAV and comparison JSON file exists as a regular, non-symlink file and matches the declared SHA-256.
- The fixture-pack receipt content matches the declared tree digest.
- The selected `fixture_id` exists in `fixtures.json`; its SFZ hash and byte length match the declared SFZ, and its sample-asset hash multiset matches the bundle declaration.
- The differential receipt uses `muse-sfz-audio-differential-receipt-v1`, has a `measured_not_quality_qualified` status and metrics, and names the same candidate/reference WAV hashes as the bundle.
- Failures are included in the JSON receipt, rather than converting absent, corrupt or mismatched artifacts into a pass.

## What is not verified

- Engine identity, software authorship or the truth of the recorded render command are **declared, not authenticated**. SHA-256 proves byte identity relative to a declaration, not who produced those bytes. Signed attestations or independently trusted build provenance are future work.
- The verifier does not recalculate the fixture pack's tree hash from all fixture files; it checks the supplied pack receipt file hash and that the receipt's digest token matches the declared digest. To independently validate the entire pack, run the existing fixture-pack validator too.
- The verifier does not render the SFZ, compare expected MIDI events with real-engine behavior, judge the spectral numbers, or qualify perceptual equivalence. A correct provenance bundle may faithfully document a bad render.
- Render command text is not executed, and build-configuration files are not interpreted; hash identity is the only automated claim made about these files.

## Qualification policy

Do not adopt a universal spectral-distance or waveform-RMSE threshold from the first corpus. Build a calibration panel with at least: byte-identical repeat, controlled gain shift, controlled pitch shift, one-frame onset shift, wrong-region sample, deliberate silence, truncation, and a changed sample rate. Verify that each negative control is detected by an appropriate metric, then freeze thresholds by fixture family and engine pair.

Keep four statuses separate in dashboards: `integrity_passed`, `semantic_behavior_passed`, `signal_thresholds_passed`, and `perceptual_study_passed`. This verifier emits only the first category and always writes `quality_qualified: false`.
