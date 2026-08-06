# Step 4: Vocos neural-vocoder resynthesis — a real, mixed result

Step 4 of `/home/tstoltz/.claude/plans/synthetic-tumbling-raccoon.md`, reached
after Step 2/3 (pitch micro-naturalization) produced only a marginal
naturalness gain at a real WER cost — evidence favoring the hypothesis that
WORLD's own vocoder timbre, not the F0 trajectory, is the naturalness
ceiling. This step tests that hypothesis directly.

## Method: analysis-resynthesis, not a WORLD-pipeline rewrite

Per the user's explicit instruction ("use the flake not pip please"), Vocos
(`charactr/vocos-mel-24khz`) is packaged via Nix (`buildPythonPackage` +
`fetchPypi`-style `fetchurl` of the official PyPI wheel) in
`symthaea/flake.nix`'s new `voice-vocoder` devShell, not `pip install vocos`
into an ad-hoc venv. All of Vocos's declared runtime deps (torch,
torchaudio, numpy, scipy, einops, pyyaml, huggingface-hub, `encodec==0.1.1`
exact) are already in nixpkgs. A real, reproducible build issue was found
and fixed along the way: `sdl3`'s own test suite (pulled in transitively via
`torchaudio -> torchcodec -> ffmpeg -> sdl2-compat -> sdl3`) has a
real-time-scheduling test (`testrwlock`) that reproducibly times out under
this host's routine concurrent-session CPU oversubscription (load 30-60 on
12 cores) — confirmed by reverting the check-skip override and watching the
identical test fail again on the vanilla derivation. Fixed via an overlay
disabling checks for `sdl3` specifically, not a broader hack.

Rather than rewriting WORLD's `harvest`/`cheaptrick`/`d4c`/`pw.synthesize()`
pipeline to route through Vocos's mel representation (which WORLD's
spectral envelope isn't natively compatible with), this step takes the
existing Arm-B baseline (WER-winning, non-naturalized) v10 renders and feeds
each waveform through Vocos's own `forward()` — internal mel extraction +
neural decode, i.e. genuine analysis-resynthesis. This tests whether
Vocos's decoder produces a more natural timbre for the same spectral content
WORLD already computed, without touching Arm B's tuned F0/duration control.

`17_vocos_resynthesis.py` — loads Arm B's 10 Gate-2 renders, resamples to
24kHz (Vocos's trained rate) if needed, runs `vocos(wav)`, writes output to
`v12_vocos_resynth/`.

## Sanity check: clean, no artifacts

10/10 renders pass NaN/inf/blowup checks — peaks 0.20-0.46 (well below
clipping), zero NaN, zero inf, zero clipped samples. See
`v12_vocos_resynth_sanity.json`.

## Real incident found and fixed mid-experiment: a stale nix-store path

Both scoring venvs (`voice-conversion/venv`, `ace-step/venv`) suddenly
failed with `ModuleNotFoundError` for packages known to be installed —
root-caused to their shared base interpreter,
`/nix/store/28wlfb25i3q4wq06ap0n9gb04qkjdjyn-python3-3.11.15`, having been
garbage-collected from the local store (likely incidental GC pressure from
this session's own heavy Vocos-devShell build, or another concurrent
session's GC run — this machine routinely runs 14+ concurrent Claude
sessions). The venvs' own site-packages were untouched; only the symlinked
base interpreter was gone. Fixed by re-fetching the exact same store path
(`nix shell nixpkgs#python311`, which resolved to the identical hash via
this repo's pinned `flake.lock`) — confirms the venvs are recoverable from
this failure mode without reinstalling anything, as long as the pin is
intact.

## Results: WER regressed, naturalness split between the two metrics

| | mean WER | mean DNSMOS (OVR) | mean UTMOS |
|---|---|---|---|
| Spoken reference | — | 3.319 | 4.373 |
| Arm B baseline (v10full, no naturalization) | 0.284 | 1.784 | 1.881 |
| Naturalized B (Step 2/3, pitch scoop/drift/jitter) | 0.365 | 1.852 | 1.913 |
| **Vocos resynthesis of Arm B (v12)** | **0.385** | **2.118** | **1.700** |

**WER**: regressed again, and by more than the naturalization attempt
(0.385 vs 0.365) — Vocos's mel round-trip is lossy for this out-of-domain
(Kokoro+WORLD sung) content in a way that costs intelligibility, similar in
kind to (if slightly worse than) the naturalization experiment's cost.

**DNSMOS**: the largest naturalness gain of any experiment in this arc —
+0.334 over baseline (vs. naturalization's +0.068), closing **21.8%** of the
gap to the spoken reference (vs. naturalization's 4.5%).

**UTMOS**: moved in the OPPOSITE direction — **-0.181 below baseline**, a
**-7.3%** move away from the spoken reference, the only experiment so far
where the two naturalness metrics disagree on direction.

### Per-phrase detail (DNSMOS / UTMOS)

`rapid_letter_names` shows the largest DNSMOS jump (2.799, vs ~2.0 for other
arms) despite being the one phrase every arm/renderer in this entire arc has
failed WER on outright (structurally non-speech-like letter-name content) —
a reminder that DNSMOS/UTMOS proxy general audio naturalness, not
intelligibility, and can diverge sharply from WER on the same file. Full
per-phrase numbers in `v12_vocos_naturalness_results.json` /
`v12_vocos_wer_results.json`.

## Honest verdict: a real, mixed result — not a clean win

This is neither a clean win nor a clean null, and the divergence between
DNSMOS and UTMOS is itself the notable finding, not just the direction of
either individual number. A plausible explanation given the plan's own
disclosed risk up front ("Vocos wasn't trained on Kokoro's voice, so a
domain/timbre mismatch is possible"): the vocos-mel-24khz model's decoder
may be resolving artifacts one metric weighs heavily (DNSMOS emphasizes
background/noise-suppression-style quality) while introducing a different
kind of distortion the other weighs heavily (UTMOS is SSL-based and more
sensitive to unnatural articulation/timbre) — consistent with Vocos
operating well outside its training distribution (spoken, non-Kokoro,
non-WORLD-processed audio) for this specific mel content.

Per the plan's own framing, this result does **not** cleanly clear the bar
to declare Vocos a fix, but it moved DNSMOS more than any other experiment
in the arc — a stronger signal on one axis than anything tried so far, at
real costs on the other two (WER, UTMOS). Whether this is worth adopting is
a genuine judgment call this document does not make: the previous human
listening check (v10 4-arm matrix) was decisive and unambiguous ("none
sound good, A/B indistinguishable"); this result is quantitatively mixed
enough that another human listen would be the right next step before
concluding anything further, rather than trusting either automated metric
alone.

## What this does and doesn't establish

**Does establish**: a working, nix-managed (no pip) neural-vocoder
integration exists and is cheap to re-run; Vocos's analysis-resynthesis
measurably changes naturalness-proxy scores in a real, non-trivial way
(unlike the near-flat pitch-naturalization result); the two naturalness
proxies can disagree, which is itself useful to know before trusting either
in isolation on a future experiment.

**Does NOT establish**: that Vocos actually sounds better (no listening
check performed this pass); that this specific analysis-resynthesis
integration pattern is the right one for production (a true WORLD-features
-> Vocos-mel adapter, or NSF-HiFiGAN's more feature-native F0-conditioned
approach, could behave very differently); whether the WER regression is
recoverable (e.g. by wiring Vocos earlier in the pipeline rather than as a
post-hoc waveform round-trip).

## Not yet done

- A human listening check on the v12 files (recommended before any further
  decision).
- NSF-HiFiGAN (the plan's stated fallback if Vocos "doesn't help") — not
  yet attempted; this result is ambiguous enough that trying it may still
  be worthwhile regardless.
- Retuning (e.g. wiring Vocos differently, earlier in the pipeline, rather
  than as a post-hoc waveform-in/waveform-out round trip) — not attempted,
  would be a distinct experiment from what's reported here.

## Files

- `17_vocos_resynthesis.py` — the resynthesis script (nix `voice-vocoder`
  devShell).
- `18_vocos_wer_evaluate.py` / `19_vocos_naturalness_screen.py` — scoring
  scripts (existing pip venvs, no new dependencies).
- `v12_vocos_resynth_sanity.json`, `v12_vocos_wer_results.json`,
  `v12_vocos_naturalness_results.json` — raw results.
- Audio: `symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v12_vocos_resynth/*_sung_v12_vocos.wav`
  (gitignored, not duplicated here).
- Flake: `symthaea/flake.nix`'s new `voice-vocoder` devShell (`nix develop .#voice-vocoder`).
