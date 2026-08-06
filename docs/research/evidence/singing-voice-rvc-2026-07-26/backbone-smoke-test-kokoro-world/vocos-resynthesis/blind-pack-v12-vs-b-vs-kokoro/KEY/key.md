# Unblinding key — open only after `01_BLIND_PASS/RESPONSE_SHEET.md` is filled in

| clip | phrase | condition | true target text | duration |
|---|---|---|---|---|
| `clip_01` | consonant_clusters | **K** — spoken Kokoro TTS (reference anchor -- not a singing candidate) | "strong streams splashed strangely" | 2.50s |
| `clip_02` | positive_control | **B** — Arm B (event-informed masking, WORLD vocoder) | "won't you sing along with me" | 2.25s |
| `clip_03` | short_unstressed | **K** — spoken Kokoro TTS (reference anchor -- not a singing candidate) | "it is what it is to me" | 1.82s |
| `clip_04` | positive_control | **K** — spoken Kokoro TTS (reference anchor -- not a singing candidate) | "won't you sing along with me" | 1.98s |
| `clip_05` | fricative_heavy | **K** — spoken Kokoro TTS (reference anchor -- not a singing candidate) | "she sells seashells by the seashore" | 2.38s |
| `clip_06` | long_sustained_vowels | **B** — Arm B (event-informed masking, WORLD vocoder) | "moon over the blue lagoon" | 2.20s |
| `clip_07` | positive_control | **V** — v12 (Arm B resynthesized through Vocos charactr/vocos-mel-24khz) | "won't you sing along with me" | 2.25s |
| `clip_08` | short_unstressed | **B** — Arm B (event-informed masking, WORLD vocoder) | "it is what it is to me" | 2.32s |
| `clip_09` | fricative_heavy | **V** — v12 (Arm B resynthesized through Vocos charactr/vocos-mel-24khz) | "she sells seashells by the seashore" | 2.55s |
| `clip_10` | fricative_heavy | **B** — Arm B (event-informed masking, WORLD vocoder) | "she sells seashells by the seashore" | 2.55s |
| `clip_11` | long_sustained_vowels | **K** — spoken Kokoro TTS (reference anchor -- not a singing candidate) | "moon over the blue lagoon" | 2.08s |
| `clip_12` | consonant_clusters | **V** — v12 (Arm B resynthesized through Vocos charactr/vocos-mel-24khz) | "strong streams splashed strangely" | 1.97s |
| `clip_13` | short_unstressed | **V** — v12 (Arm B resynthesized through Vocos charactr/vocos-mel-24khz) | "it is what it is to me" | 2.31s |
| `clip_14` | consonant_clusters | **B** — Arm B (event-informed masking, WORLD vocoder) | "strong streams splashed strangely" | 1.98s |
| `clip_15` | long_sustained_vowels | **V** — v12 (Arm B resynthesized through Vocos charactr/vocos-mel-24khz) | "moon over the blue lagoon" | 2.20s |

## Reveal codes

- **B** = Arm B baseline (existing WER-winning render, WORLD vocoder, no naturalization)
- **V** = v12 (Arm B resynthesized through Vocos, `charactr/vocos-mel-24khz`)
- **K** = spoken Kokoro TTS reference (quality anchor -- naturally will sound best; not a singing candidate, don't score it against B/V on singing quality)
