# Reproducing this run

## What's preserved here vs. what's gone
This bundle preserves configs, scripts, logs, hashes, and manifests — the
*record* of the run. It does **not** preserve the multi-GB corpus audio,
venvs, or checkpoint weights themselves (see `manifests/*.sha256` for
integrity-checkable references to where those lived). Those artifacts were
built under `/var/lib/symthaea/training-runs/` — a persistent-but-not-git
location chosen specifically because the previous attempt at this same
pipeline lost an entire training run to `/tmp` scratch-cleanup (see
project memory: `feedback_scratchpad_is_ephemeral_use_var_lib_symthaea`).
As of this bundle's creation, those files still exist at that path; there
is no guarantee they will remain there indefinitely.

## Full reproduction path
1. **Environment**: NixOS, RTX 2070 (8GB VRAM, compute capability 7.5),
   driver 595.84 — see `environment/`. Three separate Python 3.11 venvs
   were used (DiffSinger, Kokoro-corpus-generation, RVC) rather than one
   shared environment, to avoid dependency conflicts between DiffSinger's
   and RVC's differing `torch`/`transformers` pins. Build each from a
   nix-provided interpreter (`nix-build '<nixpkgs>' -A python311`) plus
   `pip install` from the relevant `environment/python-freeze-*.txt`.
2. **Pin the exact source revisions** in `environment/source-revisions.txt`
   (`git clone` + `git checkout <sha>` for both DiffSinger and RVC).
3. **Data**:
   - CSD: Zenodo record 4916302, English subset. Convert via
     `pipeline-configs/diffsinger/convert_csd.py`. Manifest of exactly which 100
     files and their train/test split:
     `manifests/corpus-manifest-csd.csv`.
   - `af_heart` corpus: regenerate via
     `pipeline-configs/rvc-training/generate_af_heart_corpus.py` (needs the `kokoro`
     pip package + Alice in Wonderland text, Project Gutenberg #11, public
     domain — the script fetches its own source sentences). Manifest:
     `manifests/corpus-manifest-af-heart.csv`.
4. **Preprocess**: `commands/preprocess.sh` (edit paths for your
   environment first — this bundle records exact historical paths, not a
   portable script).
5. **Train**: `commands/train.sh`. Expect ~1.5-2 hours for the DiffSinger
   2000-step run and ~18 hours for the RVC 200-epoch run on equivalent
   hardware (RTX 2070-class, 8GB).
6. **Infer**: `commands/infer.sh`.
7. **Verify**: `manifests/checkpoints.sha256` and `manifests/outputs.sha256`
   let you confirm a re-run produced bit-identical artifacts — it likely
   won't (GPU nondeterminism, unpinned RNG seeds in a few places), but the
   loss curves in `metrics/*.csv` should be closely reproducible in shape.
8. **Analyze**: `metrics/analyze_audio.py` — see `metrics/methodology.md`
   for exactly what it measures and its known limitations.

## Known non-reproducibility gaps (disclosed, not fixed)
- No RNG seed was explicitly pinned for the Kokoro corpus generation, the
  DiffSinger training run's data shuffling, or RVC's training run — a
  re-run will not produce bit-identical checkpoints.
- The `af_heart` corpus was generated from a *specific* sentence extraction
  of Alice in Wonderland (see `generate_af_heart_corpus.py`'s sentence
  splitter); minor differences in whitespace normalization or the source
  text encoding could change which/how many sentences are selected.
- Two of the three RVC-inference test clips used a 12-second trimmed
  excerpt (`ffmpeg -t 12`) rather than the full source file, originally as
  a workaround for a real GPU-memory contention issue when running
  inference concurrently with training (documented in this project's
  session history, not reproduced in a standalone file here). The final
  (epoch 200) sample used the full 64-second clip once training had
  finished and GPU contention was no longer a factor.
