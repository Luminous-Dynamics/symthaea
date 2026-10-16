# Communication capability inventory

Status date: 2026-07-13. "Supported" is reserved for benchmarked capability that
has passed a pinned release gate. This inventory describes code paths, not release
claims. No entry in this table constitutes a supported capability unless its
`CapabilityLevel` is marked **Released** and a corresponding `SupportRegistry` file
exists in the repository.

## Architecture separation

The `symthaea-communication` crate enforces four independently measurable stages
that must not be conflated:

1. **Signal detection and segmentation** — `CapabilityLevel::Signal` and `Unit`
2. **Structural pattern discovery** — `CapabilityLevel::Structure`
3. **Grounded meaning inference** — `CapabilityLevel::Reference` (requires preregistered intervention)
4. **Language-specific comprehension** — `CapabilityLevel::Intent` / `Dialogue`

NSM semantic primes are an optional human-language adapter only. They are never
assumed as a universal substrate for animal or unknown communication.

## Capability table

| Path | Status | Demonstrated ceiling | Notes |
|---|---|---:|---|
| `symthaea-communication` core contracts | **released architecture** | — | 40+ unit tests; benchmark gates, evidence chain, and expression policy. No claims are released without passing evidence. |
| `symthaea-communication` human pilot (Whisper large-v3) | pilot infrastructure ready | Structure | Workers, FLEURS preparation, plan/provider templates in `communication/`. Not yet released: no gate has been run on production data. |
| `symthaea-communication` human pilot (SeamlessM4T-v2) | pilot infrastructure ready | Structure | SeamlessM4T worker with MMS-LID identity. No gate run. |
| `src/language` LLM providers and language manager | active / feature-gated | Intent | Human text orchestration; providers have separate quality characteristics. No 100-language gate exists yet. |
| `crates/core/symthaea-stt` speech pipeline | active / experimental training paths | Unit | English phoneme/audio components; several CLIs contain placeholder projections and are not release evidence. |
| `symthaea-stt::discovery` | active experimental | Structure | Segmentation, clustering, recurrence, and transitions only. Does not establish reference or intent. |
| `symthaea-stt` cetacean modules | active experimental | Structure | Fixed acoustic and HDC/CfC comparison models. Synthetic examples and heuristic labels cannot pass a release gate. |
| `symthaea-stt::communication_adapter` | active | Structure | Bridges acoustic discovery outputs to `symthaea-communication` contracts; compile-checked in CI. |
| `crates/domains/symthaea-broca` native generator | feature-gated / experimental | Intent | Thought-to-text provider. Mock backends and synthetic curricula are test fixtures, never benchmark evidence. |
| `symthaea-broca` cognitive REPL | disconnected demo | Intent | Uses a mock generator and is not a measured comprehension pipeline. |

## Cetacean research contract

Dolphin and whale work is part of the same evidence ladder, but the acoustic task and input-bandwidth contract must be species/signal-specific. See `docs/research/CETACEAN_COMMUNICATION_RESEARCH_CONTRACT_V1.md` for existing module inventory, data/license admission, profile-specific metrics and next gates.

The safer `CetaceanClassifier::try_new` / `try_classify_audio` paths validate analysis-band Nyquist limits and refuse implicit downsampling via the current linear interpolator. The safer `WhaleSentinel::try_with_config` / `try_process_buffer` paths validate temporal configuration and require the source rate to match the configured sample clock. Older infallible APIs remain for compatibility and do not provide the same input checks.

The shared `symthaea-communication::bioacoustic_corpus` module provides a versioned `BioacousticCorpusManifest` (schema v3), rights-admission evidence, a canonical BLAKE3 manifest identity, annotation bounds/Nyquist checks, source-crop offsets, explicit unscored/ambiguous regions, and cross-split source/session/site/individual leakage checks. `bioacoustic-corpus-preflight` checks declared metadata; `bioacoustic-asset-audit` verifies local file bytes against SHA-256. Neither authenticates legal rights or annotation truth.

The `bioacoustic-event-eval` CLI evaluates a precomputed prediction bundle against Test annotations. It enforces non-overlapping scored source coverage, performs deterministic one-to-one event matching, reports event AP/F1/precision/recall and false alarms per scored hour, and explicitly counts predictions discarded because they fell in unscored regions. It does not run the model or authenticate predictions/threshold provenance. Use per-task metrics and keep unknown intervals out of the negative class, as in established DCASE bioacoustic event-evaluation practice.

High-rate dolphin recordings must not be silently converted to the 44.1 kHz default before click analysis. A successful config check is only input-contract evidence; it does not prove click/whistle detection accuracy, coda classification, communicative meaning or dialogue.

## Expression policy

Animal playback requires reviewed `ExpressionAuthorization` and `CapabilityLevel::Intent`.
Autonomous transmission to unknown or extraterrestrial targets is unconditionally blocked.
Experimental evidence alone never authorises expression.

## Release gate requirements

A `SupportRegistry` is only written by the CLI after:

1. A `ProviderManifest` pins the artifact URI, hash (BLAKE3), and license.
2. An `EvaluationPlan` pins the dataset manifests with per-sample content hashes.
3. The plan's release gate passes the benchmark report.
4. Calibration, sample counts, and hardware are recorded.
5. All evidence records have content-addressed IDs.

Run bundles are written with `create_new` semantics and marked read-only. They are
verified before comparison or publication with `verify-run`.
