# MUSE Render Receipt Verifier

A standalone Rust workspace for checking the content links in one score-edit-to-render attempt. It is deliberately outside the root Cargo workspace so its protocol kernel can be compiled and tested independently of monorepo member enumeration.

## Commands

Receipt-only check (shape, digest seal, and internal cross-field constraints):

```sh
cargo run --manifest-path tools/muse-render-receipt/Cargo.toml -- verify path/to/receipt.json
```

Full bundle check:

```sh
cargo run --manifest-path tools/muse-render-receipt/Cargo.toml -- verify path/to/receipt.json path/to/bundle
```

The bundle directory must contain:

| File | Purpose |
|---|---|
| `source-score.json` | Exact serialized source Score bytes |
| `candidate-score.json` | Exact serialized candidate Score bytes |
| `edit-plan.json` | Exact serialized ScoreEditPlanV1 |
| `edit-receipt.json` | Exact serialized ScoreEditReceiptV1 |
| `edit-verification.json` | Independent candidate-verification report |
| `midi-manifest.json` | Part-aware MIDI performance manifest |
| `performance.mid` | Exact Standard MIDI File bytes |
| `renderer-binary` | Exact renderer executable bytes |
| `renderer-environment.json` | Exact pinned environment/closure manifest |
| `renderer-config.json` | Exact renderer configuration bytes |
| `renderer-asset.bin` | Exact SoundFont/asset bytes when the receipt names an asset digest |
| `render-invocation.json` | Canonical invocation record |
| `render.stdout`, `render.stderr` | Captured process streams when their hashes are present |
| `output.wav` | Exact rendered WAV bytes |

The verifier checks:
- the receipt's RFC 8785-style canonical JSON seal (receipt hash omitted before hashing);
- exact byte hashes for source/candidate scores, edit plan/receipt/report, MIDI, renderer artifacts, invocation, process streams, and WAV;
- consistency of the score-edit input digest, operation IDs, protected-window IDs and independent verifier result;
- consistency of the receipt with the PartId-aware MIDI manifest, including the repository's 960 ticks-per-quarter resolution;
- MIDI track/part/program/note event correspondence by parsing the actual SMF bytes, not by trusting manifest summaries;
- WAV parseability, declared sample format, frame count, finite samples, non-silence, and exact output-byte digest.

## Important boundaries

- A passing integrity check is not an edit-acceptance decision.
- This tool does not independently recompute the score-edit candidate transformation. It verifies the exact edit verifier report bytes and checks their source/operation bindings. The score-edit verifier remains a separate component.
- MIDI parsing verifies the MIDI file against its own manifest. It does not prove the renderer consumed that MIDI. The invocation/executor evidence and trust boundary remain separate.
- WAV non-silence and metadata checks do not prove score-to-audio conformance, perceptual quality, musical coherence or human preference.
- The tool does not verify digital signatures or trusted hardware/workflow attestations. It reports the attestation class without upgrading it to trusted identity.
- `outcome_unknown` is returned as unresolved (exit code 3), never as a successful render and never as permission to retry under the same attempt ID.
- Hashes prove byte identity only. They do not establish rights, authorship or publication permission.

No Rust compile/test or real render result is claimed until the standalone workflow runs on the exact head.
