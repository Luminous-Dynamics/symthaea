# Audio-comparison methodology

Produced by `analyze_audio.py` in this directory. Run with the RVC venv's
Python (has `parselmouth` + `soundfile`):
```
LD_LIBRARY_PATH="/nix/store/8lahnh9pn3lrrnhax5nk7ibvjcbjmnkm-gcc-15.2.0-lib/lib:/nix/store/b2swxfi8srrbsafvh9iyyhd26mz9giwf-zlib-1.3.2/lib:/run/opengl-driver/lib" \
  /var/lib/symthaea/training-runs/voice-conversion/rvc-venv/bin/python3 analyze_audio.py
```
(paths are session-local scratch; regenerate the venv per `REPRODUCE.md` if it
no longer exists).

## F0 extraction
- Praat autocorrelation via `parselmouth.Sound.to_pitch_ac`.
- Time step: 20ms. Pitch floor 65 Hz, ceiling 1100 Hz (same range used by
  DiffSinger's own `pe: parselmouth` pitch extractor at training/binarize time).
- No manual octave-jump correction, no smoothing.

## F0 correlation
- **Direct Pearson correlation, no DTW / time-alignment.** Frames are
  compared index-for-index after truncating both curves to the shorter
  length; only frames voiced in *both* signals are included.
- This is a **stricter, lower-bound** measure of pitch fidelity: any timing
  drift introduced by RVC's synthesis path (which does not guarantee
  frame-exact alignment to the source) directly lowers the correlation, even
  if the melody is subjectively identical.
- **Observed effect of clip length on this metric**: the two 12-second test
  clips (epoch 50, epoch 75) both scored ~0.994; the one 64-second full clip
  (epoch 200 final) scored 0.868. This is consistent with drift accumulating
  over a longer clip and being penalized more by a non-aligned correlation —
  not necessarily evidence that the epoch-200 checkpoint tracks pitch worse
  than epoch 50/75. **A DTW-aligned F0 correlation across matched clip
  lengths is the correct follow-up if a single canonical number is needed.**
- An earlier verbal review of this run (not this bundle's author) reported
  0.956 for the full-clip case using an unspecified methodology. This bundle
  could not reproduce that exact figure (`0.868` here) and does not know
  what alignment/extractor/filtering choices produced 0.956. Both numbers
  are recorded; neither should be treated as canonical without matching
  methodology.

## Silence fraction
- RMS computed per 2048-sample frame, 512-sample hop (44.1kHz source /
  40kHz RVC output — hop is **not** re-scaled for the differing sample
  rates between source and compare files, so time resolution differs
  slightly, ~11.6ms vs ~12.8ms per hop; this was not corrected for in this
  pass).
- A frame is "silent" if `20*log10(rms) < -50 dB`.
- Fraction reported = silent frames / total frames.
- **This measure needs no alignment and is length-independent**, so it is
  the more trustworthy of the two metrics in this bundle.

## Verified results (this bundle's own run, not carried over from any
external review)

| pair | F0 corr (no DTW) | co-voiced frames | silence: source | silence: compare |
|---|---|---|---|---|
| ep50 vs 12s source | 0.9936 | 275 | 26.8% | 41.2% |
| ep75 vs 12s source | 0.9939 | 278 | 26.8% | 41.4% |
| final (ep200) vs 64s source | 0.8682 | 1622 | 24.8% | 34.3% |

Raw JSON: `audio-comparison.json` (regenerated each run of `analyze_audio.py`).

## What this does and does not establish
- **Does establish**: RVC's output is consistently more aggressively
  gated/silent than the DiffSinger source across every checkpoint tested
  (+14-15 percentage points of below-threshold frames on the short clips,
  +9.6 on the long clip). This is a real, repeatable, alignment-independent
  finding.
- **Does establish**: pitch tracking between source and converted output is
  strongly correlated on short clips (0.99+).
- **Does NOT establish**: a single canonical "fidelity score" for the full
  pipeline — the 64s-clip F0 number is sensitive to the lack of time
  alignment in this particular script, and should not be quoted as
  equivalent to the 12s-clip numbers.
- **Does NOT establish**: speaker-identity similarity to `af_heart` (no
  reference `af_heart` speech was compared against in this pass — see
  `CLAIMS.md`).
