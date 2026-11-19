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
| `orchestration-timeline.json` | Exact source-Score-bound timeline whose segments must explain each MIDI track |
| `instrument-bindings.json` | Canonical, instrument-ID-sorted renderer/program/patch bindings |
| `instrument-capabilities.json` | Canonical, instrument-ID-sorted capability profiles used by the MIDI realization |
| `performance.mid` | Exact Standard MIDI File bytes |
| `renderer-binary` | Exact renderer executable bytes |
| `renderer-environment.json` | Exact pinned environment/closure manifest |
| `renderer-config.json` | Exact renderer configuration bytes |
| `renderer-asset.bin` | Exact SoundFont/asset bytes when the receipt names an asset digest |
| `render-invocation.json` | Structured invocation record validated against renderer identity, binary/environment/configuration hashes, MIDI input, WAV output, timing and exit code |
| `render.stdout`, `render.stderr` | Captured process streams when their hashes are present |
| `output.wav` | Exact rendered WAV bytes |

The verifier checks:
- the structured [render-invocation-v1 schema](../../docs/research/schemas/muse-render-invocation-v1.schema.json), including exact links to the renderer binary/environment/configuration, MIDI input digest, WAV output path, timestamps and exit code;
- the receipt's project-specific v1 canonical serialization seal (typed integer/string/bool/null/array fields, lexicographically sorted object keys, compact JSON, receipt hash omitted, domain-separated SHA-256);
- exact byte hashes for source/candidate scores, edit plan/receipt/report, MIDI, renderer artifacts, invocation, process streams, and WAV;
- independent recomputation of the v1 velocity-only candidate from the exact source Score and authorized edit plan; exact comparison of the edit receipt deltas and verifier report counters/issues;
- event-by-event candidate Score ↔ MIDI-note receipt comparison for source ordinal, PartId, pitch, exact onset/offset at 960 PPQ and the exporter's velocity quantization;
- consistency of the score-edit input digest, operation IDs, protected-window IDs and independent verifier result;
- consistency of the receipt with the PartId-aware MIDI manifest, including the repository's 960 ticks-per-quarter resolution;
- exact timeline, instrument-binding and capability-manifest digests; one-to-one timeline-segment/track correspondence; renderer/program/patch agreement; typed instrument-kind validation; per-note sounding-range checks; per-instrument polyphony sweeps against the declared capability budget; and resource ceilings matching the producer (256 parts, 100,000 notes, 4,096 segments, 128 profiles, 64-note maximum polyphony, 255-byte MIDI text);
- MIDI track/part/program/note event correspondence by parsing the actual SMF bytes, not by trusting manifest summaries; unexpected controller messages, unrecognized metadata, and system-exclusive/escape events are rejected because the exporter does not declare them; each instrument instance must own one stable MIDI channel, channels cannot be shared across instrument IDs, and sidecar track names must be unique;
- WAV parseability, declared sample format, frame count, finite samples, non-silence, and exact output-byte digest.

## Important boundaries

- A passing integrity check is not an edit-acceptance decision.
- `validate_receipt` and receipt-only reporting recompute the receipt seal in the public library API; callers cannot bypass seal verification by constructing a typed `ReceiptV1` and calling library report functions directly.
- V1 binds one shared renderer asset digest per attempt. Per-instrument heterogeneous asset sets require a future receipt schema; the verifier rejects missing or mismatched binding/receipt asset digests.
- The standalone verifier independently re-derives the current velocity-only ScoreEditPlanV1 transformation from the source Score, checks every candidate note field and velocity, validates exact edit receipt deltas, and checks verifier report counts/issues. It does not yet independently support future ScoreEdit operation classes; unknown schema/operation classes must not be silently accepted.
- MIDI parsing verifies the MIDI file against its own manifest. It does not prove the renderer consumed that MIDI. The invocation/executor evidence and trust boundary remain separate.
- WAV non-silence and metadata checks do not prove score-to-audio conformance, perceptual quality, musical coherence or human preference.
- The tool does not verify digital signatures or trusted hardware/workflow attestations. It reports the attestation class without upgrading it to trusted identity.
- `outcome_unknown` is returned as unresolved (exit code 3), never as a successful render and never as permission to retry under the same attempt ID.
- Hashes prove byte identity only. They do not establish rights, authorship or publication permission.

The direct dependency versions are exact-pinned to the current root lockfile. The standalone Cargo.lock contains the 49-package dependency closure extracted from that root lockfile; it has not yet been validated by Cargo. The focused workflow runs tests and Clippy with `--locked`, so any incomplete or inconsistent graph fails closed. No Rust compile/test or real render result is claimed until that exact-head workflow passes.
