# Sample audio

The actual `.wav` files referenced by this bundle live at
`symthaea/audio_output/diffsinger_csd_poc_2026-07-25/` (sibling directory
in this repo, not duplicated here to avoid committing large binaries
alongside documentation — see `manifests/outputs.sha256` for integrity
hashes of every file).

| File | What it is |
|---|---|
| `en001a-step500-early.wav` | DiffSinger acoustic model, mid-training (step 500/2000), before voice conversion |
| `en001a-step2000-final.wav` | DiffSinger acoustic model, fully trained (step 2000/2000), before voice conversion — the "source" for the final RVC comparison |
| `en010a-step2000-final.wav`, `en040a-step2000-final.wav` | Same fully-trained DiffSinger checkpoint, different held-out CSD test lines |
| `en001a_clip12s_ORIGINAL_diffsinger.wav` | First 12s of `en001a-step2000-final.wav`, trimmed for the RVC checkpoint-comparison tests |
| `en001a_af_heart_ep50_12s.wav` | RVC epoch 50 checkpoint, converting the 12s clip above |
| `en001a_af_heart_ep75_12s.wav` | RVC epoch 75 checkpoint, same 12s clip |
| `en001a_af_heart_FINAL_ep200.wav` | RVC epoch 200 (final) checkpoint, converting the FULL 64s `en001a-step2000-final.wav` |

All CSD text (`en001a`, `en010a`, `en040a`) is genuinely held-out — listed
in `test_prefixes` in `pipeline-configs/diffsinger/csd_en_acoustic.yaml` and never
seen during DiffSinger training.

See `LICENSE_STATUS.md` before using any of these files for anything
beyond internal research listening.
