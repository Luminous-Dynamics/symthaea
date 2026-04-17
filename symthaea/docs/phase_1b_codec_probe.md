# Phase I.B.0 — Codec Ladder Probe Results

**Date**: 2026-04-14
**Device**: Google Pixel 8 Pro, serial `41201FDJG000UM`
**Android version**: 16
**scrcpy-server version**: v2.4 (SHA256 `93c272b7438605c055e127f7444064ed78fa9ca49f81156777fd201e79ce7ba3`)
**Probe command**: `app_process / com.genymobile.scrcpy.Server 2.4 list_encoders=true audio=false`

## Raw output

```
[server] INFO: Device: [Google] google Pixel 8 Pro (Android 16)
[server] INFO: List of video encoders:
    --video-codec=h264 --video-encoder='c2.exynos.h264.encoder'
    --video-codec=h264 --video-encoder='c2.android.avc.encoder'
    --video-codec=h264 --video-encoder='OMX.google.h264.encoder'
    --video-codec=h265 --video-encoder='c2.exynos.hevc.encoder'
    --video-codec=h265 --video-encoder='c2.android.hevc.encoder'
    --video-codec=av1  --video-encoder='c2.google.av1.encoder'
    --video-codec=av1  --video-encoder='c2.android.av1.encoder'
```

## Classification

| Codec | Encoder | Type | Verdict |
|---|---|---|---|
| H.264 | `c2.exynos.h264.encoder` | **Hardware** (Tensor G3 silicon) | Usable |
| H.264 | `c2.android.avc.encoder` | Software | Ignored — HW available |
| H.264 | `OMX.google.h264.encoder` | Legacy software | Ignored |
| H.265 | `c2.exynos.hevc.encoder` | **Hardware** (Tensor G3 silicon) | **PRIMARY** |
| H.265 | `c2.android.hevc.encoder` | Software | Ignored — HW available |
| AV1 | `c2.google.av1.encoder` | Software (libgav1) | Research only |
| AV1 | `c2.android.av1.encoder` | Software | Research only |

Naming convention: `c2.exynos.*` = hardware (Tensor/Exynos silicon);
`c2.google.*` and `c2.android.*` = software fallbacks shipped in the
Android platform image.

## Decision

**PRIMARY: HEVC (H.265) via `c2.exynos.hevc.encoder` (hardware).**

Rationale:
- The v1.3 roadmap's AV1 pivot was grounded in the premise that
  Pixel 8 Pro exposes hardware AV1 encoding to MediaCodec. **This
  premise is false on Android 16.** Only software AV1 encoders
  are enumerated.
- Software AV1 encoding at 30 fps on real UI content is untenable
  on a phone: battery drain, thermal throttling, unlikely to
  sustain the target rate, contaminates the Phase IV PE signal
  with thermally-modulated latency.
- HEVC compresses ~50% smaller than H.264 at equal perceptual
  quality — close enough to AV1's 30-40% that the USB 2.0 budget
  relief motivating the v1.2 AV1 pivot survives this change.
- HEVC HW encode on Tensor G3 is stable, low-power, and proven in
  production video-record workloads.

## Fallback ladder (revised)

1. **HEVC** via `c2.exynos.hevc.encoder` — hardware, primary
2. **H.264** via `c2.exynos.h264.encoder` — hardware, fallback if
   the desktop HEVC decoder is unavailable for any reason
3. **AV1 SW** via `c2.google.av1.encoder` — research tier only,
   for battery-insensitive bench runs (e.g. tethered Phase III
   experiments where we care about bandwidth not power)

## Decoder implications

HEVC primary means `ffmpeg-next` is back on the critical path
(v1.2 had demoted it to `hevc-fallback`). The v1.2 concern about
C-library integration friction on NixOS is real but tractable:
- NixOS has `ffmpeg` + `ffmpeg-dev` in nixpkgs
- `ffmpeg-next` uses pkg-config to find them — no custom build
  script pathology like `openh264`'s Cisco binary download
- First `cargo build --features scrcpy` in the worktree must run
  inside `nix develop` with ffmpeg in the shell (documented in
  Phase I.B.4)

Pure-Rust alternatives considered and rejected:
- `rav1d` — AV1 only, doesn't help for HEVC
- `ffmpeg-the-third` — thin wrapper over ffmpeg-next, same deps
- No pure-Rust HEVC decoder exists at production quality as of
  2026-04

## Zero-latency tuning (inherited from v1.3)

The v1.3 encoder-tuning flags still apply, adjusted for HEVC:
- `--display-buffer=0`
- `--max-fps=30`
- `--video-codec-options=profile=1` — HEVC Main Profile (integer 1
  in MediaCodec's `CodecProfileLevel` for HEVC). Must be validated
  in I.B.3 against actual encoder init.

## Action items flowing from this probe

1. Roadmap main: bump v1.3 → v1.4, codec pivot AV1 → HEVC
2. Cargo.toml: `ffmpeg-next` moves from optional `hevc-fallback`
   feature to required under the `scrcpy` feature; `rav1d` moves
   to optional `av1-research` feature for Phase II.5
3. `src/scrcpy.rs`: use `--video-codec=h265` in the default
   app_process command line
4. Phase I.B.4 (Cargo.toml + features): document the `nix develop`
   requirement for the first build; add a pkg-config probe to the
   build so the error message is actionable if ffmpeg is missing
5. Phase II.5 (compressed-domain): still on the table, but now
   reframed as "AV1 software encode is already expensive; if we're
   paying that cost we should at least extract its motion vectors"
   — the cost/benefit ratio of II.5 actually *improves* when AV1
   is SW-only, because the redundant work becomes more expensive
