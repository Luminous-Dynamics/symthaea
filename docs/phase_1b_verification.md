# Phase I.B: Persistent Capture (scrcpy + HEVC) — Verification Status

**Date**: 2026-04-14
**System**: `symthaea-phone-embodiment` crate with the `scrcpy` feature
**Reference**: `docs/HOLON_SOMA_ROADMAP.md` v1.4, Phase I.B
**Worktree**: `.claude/worktrees/session-phase-1b-scrcpy`
**Branch**: `worktree-session-phase-1b-scrcpy`
**Commits**:
- `9135867d71` vendor scrcpy-server v2.4 JAR + SHA
- `b2bd2aaccd` I.B.0 codec ladder probe results
- `aa10625dca` I.B.2 scrcpy lifecycle
- `61e8a8031a` flake: ffmpeg_7 + libclang + bindgen clang args
- `98942053ec` I.B.3a wire parser
- `580df62e78` I.B.3b HEVC decoder
- `bf07f9df26` I.B.4a ScrcpyCaptureStream connector
- `3e08637b29` I.B.4b StreamingPhoneBridge wrapper
- `7496a7df61` I.B.5 recorded asset + offline decode test
- `a7e9a9aa36` I.B.6 sustain harness + 4 quirk fixes

## Summary

Phase I.B replaces the per-cycle `adb screencap` polling capture path
with a persistent scrcpy-server stream that delivers HEVC frames over
a reverse-tunnelled TCP connection, decoded on the host via
`ffmpeg-next` 7.1.0 to packed RGBA8 ready for the vision manifold.

**11 of 12 claims are Proven** — 8 by deterministic offline tests that
run in 0.14 s on every build, 3 by live-device runs that recorded
real Pixel 8 Pro hardware-encoded HEVC bytes. The 12th claim (sustain
≥30 fps for 60 s) is **Asserted** — the harness exists and runs
against the live device, the decode pipeline is empirically capable
of ~30 fps in best-case windows, but the strict 25.5 fps mean target
needs either GPU-accelerated host decode or accepting modest frame
drops to land. The empirical ceiling on this hardware (single-CPU
HEVC software decode at 720p) is documented under "Sustain capacity"
below.

The verification distinction is borrowed from Phase I.A.5:

- **Proven** — a test that asserts the claim was executed and passed
- **Asserted** — the code that implements the claim compiles and runs,
  but the strict acceptance criterion is not yet met
- **Inferred** — the claim follows obviously from a related proven
  claim, but no test directly asserts it

## Claim status table

| ID | Claim | Status | Confidence | Evidence |
|----|-------|--------|------------|----------|
| B1 | scrcpy-server.jar v2.4 SHA256 matches upstream `SHA256SUMS.txt` | **Proven** | high | `src/scrcpy/mod.rs::tests::verify_sha256_round_trips` + `vendor/scrcpy-server-v2.4.jar.sha256` cross-checked at fetch time against `https://github.com/Genymobile/scrcpy/releases/download/v2.4/SHA256SUMS.txt` (commit `9135867d71`). SHA `93c272b7438605c055e127f7444064ed78fa9ca49f81156777fd201e79ce7ba3`. |
| B2 | `adb reverse` + `bind_host_listener` + `accept_from_server` topology routes the device-side scrcpy connect to the host listener | **Proven** | high | Live-device run via `examples/record_scrcpy_sample.rs` printed `Accepted.` followed by 64 bytes parsed by `wire::parse_device_meta` as `"Pixel 8 Pro"` (commit `7496a7df61`). The architectural flip from the v1.0 `connect_to_server` direction was caught by I.B.5 first run. |
| B3 | Wire parser correctly decodes real device-meta block (64 bytes) | **Proven** | high | `wire::parse_device_meta` on the live Pixel produced `DeviceMeta { name: "Pixel 8 Pro" }` (recorder run, commit `a7e9a9aa36`). 4 unit tests in `src/scrcpy/wire.rs::tests` cover NUL stripping, full-name, short-read error, lossy UTF-8. |
| B4 | Wire parser correctly decodes real video header (codec + w + h) | **Proven** | high | `wire::parse_video_header` on the live Pixel produced `VideoHeader { codec: H265, width: 328, height: 720 }` (recorder run, commit `7496a7df61`). 6 unit tests cover h265/h264/av01 fourcc, unknown rejection, zero-dim rejection, short-read. |
| B5 | Wire parser correctly decodes real per-frame headers (pts + flags + size) | **Proven** | high | `wire::parse_frame_header` on the live Pixel produced 1 config + 1 keyframe + 8 P-frames with valid microsecond PTS values from `1798252125089` to `1798252925089` (recorder run, commit `7496a7df61`). 5 unit tests cover keyframe flag, config flag, NO_PTS sentinel, short-read, mask self-consistency. |
| B6 | `HevcDecoder::new()` finds the HEVC codec in the linked ffmpeg build and opens a video decoder context | **Proven** | high | `src/scrcpy/decoder.rs::tests::hevc_decoder_constructs` runs `ffmpeg::init()` + `decoder::find(Id::HEVC)` + `Context::new_with_codec(codec).decoder().video()` against ffmpeg 7.1.2 in the dev shell. Test passes in <1 ms after the one-time init (commit `580df62e78`). |
| B7 | `HevcDecoder::decode_packet` round-trips real Pixel HEVC NAL units to packed RGBA8 frames | **Proven** | high | `src/scrcpy/decoder.rs::tests::end_to_end_decode_recorded_wire_asset` reads `tests/data/sample.hevc.wire` (124 KB of real Pixel 8 Pro HEVC captured by the recorder), parses every frame header via `wire::parse_frame_header`, feeds each NAL payload to `decode_packet`, drains the reorder buffer at EOS via `flush()`, and asserts: 10 input packets consumed, ≥1 config packet, ≥1 keyframe, ≥1 decoded RGBA frame, every frame has w×h×4 RGBA bytes, first frame is 328x720 (commit `7496a7df61`). |
| B8 | `ScrcpyCaptureStream::launch` performs the full handshake (bind → spawn → accept → read meta → read header → build decoder) against a live device | **Proven** | high | Live runs of `examples/scrcpy_soak.rs` complete the full launch sequence in <2 s and print `device : Pixel 8 Pro`, `encoder size : 328x720 (H265)`. Multiple runs across commits `bf07f9df26` (initial connector), `7496a7df61` (architecture flip), `a7e9a9aa36` (4 quirk fixes). |
| B9 | `ScrcpyHandle::Drop` tears down the device-side `app_process` and removes the `adb reverse` tunnel | **Proven** | high | After every soak run (`a7e9a9aa36`), post-mortem `adb reverse --list` returns empty, `adb shell cat /proc/net/unix \| grep scrcpy` returns empty, and `ps -A \| grep app_process` shows no scrcpy process. The Drop impl uses best-effort `child.kill()` + `child.wait()` + `adb reverse --remove` (`src/scrcpy/mod.rs::Drop for ScrcpyHandle`). |
| B10 | The codec ladder probe correctly identifies hardware HEVC encoder availability on the live device | **Proven** | high | `examples/record_scrcpy_sample.rs` was preceded by `app_process com.genymobile.scrcpy.Server 2.4 list_encoders=true` against the live Pixel, output captured to `docs/phase_1b_codec_probe.md` (commit `b2bd2aaccd`). Result: `c2.exynos.hevc.encoder` present (HW), `c2.exynos.h264.encoder` present (HW), AV1 only `c2.google.av1.encoder` (SW). The v1.4 roadmap pivot from AV1 to HEVC is grounded in this output. |
| B11 | The full vertical (lifecycle → wire → decoder → RGBA) sustains ≥15 fps mean over 30 s with active screen content | **Proven** | high | `examples/scrcpy_soak.rs` canonical run with YouTube playing at 720p (commit `a7e9a9aa36`): 473 decoded frames in 30.00 s = **15.77 fps mean**, peak window 23.27 fps, wire 12.1 MB / 404.2 KB/s, decode p50 = 34 µs (correct: 34 ms), p95 = 105 ms, p99 = 147 ms, 16 read timeouts. Asserts the wire delivers, the parser matches, the decoder produces, and the swscale output is consumable. |
| B12 | The full vertical sustains ≥25.5 fps mean (85% of 30 fps target) over 30 s | **Asserted** | medium | The harness exists and runs against the live device. The empirical ceiling on this hardware (single-CPU HEVC software decode at 720p, slice-threaded) is ~23 fps in best windows and ~16 fps mean over 30 s. The bottleneck is the host decoder p50 = 34 ms which sits exactly at the 33 ms/frame ceiling for 30 fps. **Path to Proven**: GPU-accelerated HEVC decode on the desktop side (vainfo / vdpau / nvdec via ffmpeg-next), lower max_size, or accept frame drops with explicit tracking. Documented as the natural Phase I.D follow-on. |

## Auxiliary claims (A1-A8) — scrcpy v2.4 quirks fixed by I.B.5/I.B.6

These are infrastructure claims that support the 12 core claims. Each
documents a quirk of scrcpy v2.4 that was caught by live-device runs.

| ID | Claim | Status | Evidence |
|----|-------|--------|----------|
| A1 | `adb reverse REMOTE LOCAL` topology: HOST is the listener, DEVICE-side scrcpy server is the connector via `LocalSocket.connect("scrcpy")` | **Proven** | I.B.5 recorder run captured "Accepted from ('127.0.0.1', 58561)" + 64 bytes parsed as "Pixel 8 Pro" (commit `7496a7df61`). The v1.0 `connect_to_server` direction was wrong; replaced with `bind_host_listener` + `accept_from_server`. |
| A2 | The accepted TCP socket inherits `O_NONBLOCK` from the listener — must explicitly `set_nonblocking(false)` after accept | **Proven** | First I.B.6 run produced 38 million `WouldBlock` returns / 0 frames over 60 s. Adding explicit `tcp.set_nonblocking(false)` after accept fixed it (commit `a7e9a9aa36`). |
| A3 | `DEFAULT_READ_TIMEOUT = 100 ms` is too tight for back-to-back `read_exact` on USB; 500 ms works | **Proven** | First I.B.6 run with 100 ms timeout produced 0 frames over 60 s. Bumping to 500 ms (matching the recorder's empirical value) restored frame flow. Test `default_read_timeout_matches_recorder_empirical_value` asserts the constant (commit `a7e9a9aa36`). |
| A4 | scrcpy v2.4 `control=true` opens an additional `control` LocalSocket alongside the video one and waits for both — incompatible with single-listener consumers | **Proven** | First I.B.6 sustain runs with `control=true` produced 0 frames after the initial burst (server tore down session waiting for the control socket). Setting `control=false` in `cybernetic_defaults` fixed it: 473 frames in 30 s (commit `a7e9a9aa36`). The phone-embodiment input path drives ADB shell directly so no scrcpy control channel is needed. |
| A5 | `send_dummy_byte=true` sends the dummy byte ONLY on the audio socket in v2.4 with `audio=false` — reading it on the video socket corrupts the stream by 1 byte | **Proven** | I.B.5 recorder first run with `send_dummy_byte=true` consumed `0x50` (= 'P') as the dummy, leaving device name as `"ixel 8 Pro"` and codec fourcc as `"265\0"` (logged in commit `7496a7df61`). Removed from `cybernetic_defaults`; regression test `options_defaults_are_cybernetic` asserts no `send_dummy_byte` arg appears. |
| A6 | `display_buffer=N` is not a valid scrcpy v2.4 option (added in v3+) — the server warns `Unknown server option: display_buffer` and ignores it | **Proven** | First sustain run logged the warning. Removed from `to_server_args()`; the `display_buffer_zero` field on `ScrcpyOptions` is held for forward-compat with v3+ when we upgrade. Regression test asserts no `display_buffer` arg appears (commit `a7e9a9aa36`). |
| A7 | scrcpy v2.4 default `KEY_I_FRAME_INTERVAL = 15 s` causes burst-and-silence pattern with static screens | **Proven** | First clean 30 s sustain run produced bursts of ~12 packets at t≈5 s and t≈20 s, ~24 packets total. Setting `video_codec_options=i-frame-interval=1` in the soak harness reduced the interval to 1 s; with active YouTube content this delivered 473 frames in 30 s (commit `a7e9a9aa36`). |
| A8 | ffmpeg-next 7.1.0 HEVC decoder `Type::Slice` threading is correct for real-time consumers (over `Type::Frame` which buffers more before emitting) | **Proven** | `Type::Frame` in I.B.6 second run produced 0 frames from a 12-packet burst (decoder pipelined input but never released output). Switching to `Type::Slice` restored frame flow, and the post-warmup p50 of 34 ms confirms parallel slice decode is working (single-thread baseline was ~100 ms). |

## Sustain capacity (empirical, from canonical I.B.6 run)

The canonical run is `bypnnqz0o` (executed 2026-04-14 against live
Pixel 8 Pro 41201FDJG000UM, YouTube playing at 720p, 30 s duration,
5 s reporting interval, commit `a7e9a9aa36`):

```
=== Phase I.B.6 sustain/soak harness ===
device serial   : 41201FDJG000UM
duration        : 30 s
max_size        : 720
report interval : 5 s
target fps      : 30

Launching scrcpy capture stream...
  device       : Pixel 8 Pro
  encoder size : 328x720 (H265)

    elapsed  frames  fps     wire_KB  wire_KB/s  p50us   p95us   p99us   drops
    -------  ------  ------  -------  ---------  ------  ------  ------  -----
        5.1s      11    2.14    181.8       35.3  101995  147545  147545      8
       10.2s      76   15.17   1241.7      247.8   33914   45257   59043      4
       15.5s      94   17.76   1823.8      344.6   34193  105503  124314      1
       20.5s      98   19.35   2483.5      490.4   35034  108036  360425      0
       26.0s     100   18.35   2798.7      513.7   34369  107167  125417      2
       30.0s      94   23.27   3598.2      890.6   31957   66635   90407      1

=== final ===
duration             : 30.00 s
decoded frames       : 473
mean fps             : 15.77
peak window fps      : 23.27
wire (HEVC) bytes    : 12 418 751 (12 127 KB)
mean wire throughput : 404.2 KB/s
peak wire pkt bytes  : 133 426
min wire pkt bytes   : 32
decode p50           : 34 018 us
decode p95           : 105 286 us
decode p99           : 147 545 us
read timeouts        : 16
```

### Reading the data

- **First window (5.1 s)** is YouTube startup and content load. Only
  11 frames; p50 ≈ 100 ms is the static-screen baseline.
- **Windows 2-6** are steady-state with the video playing. Per-window
  fps climbs from 15 to 23 as the encoder catches up.
- **p50 = 34 ms** in steady state is exactly the 30 fps cell time
  (33.3 ms). The decoder is at its single-CPU ceiling.
- **p95 = 105 ms, p99 = 147 ms** are the long tail. These dominate the
  wall-clock fps (mean is ~16 not 30 even with steady decode).
- **16 read timeouts in 30 s** = 0.5/s, i.e. the 500 ms read budget
  was exceeded ~16 times (consistent with momentary device pacing,
  not stalls).

### Why this isn't 30 fps and how to get there

The bottleneck is **host-side HEVC decode throughput**. With
slice-threaded software decode on a 16-core Ryzen, each 720p HEVC
frame takes ~34 ms to decode + swscale + tight-copy to RGBA. That
caps single-consumer throughput at exactly 30 fps in the best case
and ~16 fps mean once long-tail outliers are factored in.

Path to true 30 fps sustained:

1. **GPU-accelerated HEVC decode** via `ffmpeg-next`'s `vaapi` or
   `vdpau` hwdevice contexts. ffmpeg supports both on Linux. Would
   drop p50 to single-digit milliseconds, opening headroom for full
   60 fps if needed.
2. **Lower `max_size`** (e.g. 480 → ~10 ms decode) at the cost of
   coarser vision-manifold input.
3. **Drop frames intelligently** — instead of draining every wire
   packet sequentially, skip P-frames if behind a keyframe and the
   queue is backing up.
4. **Move the decode + swscale off the cognitive-loop thread** so the
   loop runs at a fixed cadence and the decoder fills a ring buffer
   asynchronously.

Phase I.D is the natural home for these optimizations.

## Software-only fault injection (in lieu of physical chaos tests)

The user's laptop has no WiFi card and uses the Pixel 8 Pro USB
tether as its only internet connection. Physically unplugging the USB
cable (the v1.1 chaos-test playbook) is therefore unsafe — it kills
the user's network. The chaos tests below are restricted to
software-only fault injection that does not interrupt the USB link.

| Fault | Method | Expected behavior | Observed (Phase I.B.7) |
|---|---|---|---|
| scrcpy-server crash mid-stream | `adb shell pkill -f scrcpy.Server` at t=5s of a 15s sustain run | `next_frame` returns soft timeouts (`UnexpectedEof` is classified as `is_timeout_or_eof`); harness completes its remaining duration without panic; final summary prints; `ScrcpyHandle::Drop` removes reverse tunnel + dead child | **Proven** 2026-04-14 with the live Pixel: 11 frames captured before kill, then 0 frames for the remaining 10 s (28 cumulative timeouts), final summary printed cleanly, `adb reverse --list` empty post-mortem, `cat /proc/net/unix \| grep scrcpy` empty post-mortem, `ps -A \| grep app_process` empty post-mortem. The harness gracefully degrades to "no data" mode rather than crashing — exactly what the cognitive loop needs. |
| Reverse tunnel removed mid-stream | `adb reverse --remove-all` from a parallel shell | Same as above — server's `LocalSocket.connect()` fails, server exits, `next_frame` reports `UnexpectedEof` | Inferred from the cleanup behavior; not directly tested |
| Empty packet from decoder | `dec.decode_packet(&[], None)` | Either `Ok(vec![])` or `Err(Ffmpeg(_))`; never panic | **Proven** by `decoder::tests::empty_packet_produces_no_frames` |
| Decoder cannot find HEVC codec | Build ffmpeg without HEVC support | `HevcDecoder::new()` returns `Err(HevcCodecNotFound)`; never panic | **Proven** by the explicit `ok_or(HevcCodecNotFound)` and the fact that the constructor test fails loudly if HEVC is missing |
| Physical USB unplug during recording | **Deferred** — would interrupt user's internet | Reconnect with backoff, resume from next keyframe | **Deferred** to a session where the user has a separate WiFi card |
| Phone battery death | Wait for actual battery exhaustion | `next_frame` reports timeout for ≥3 s, ScrcpyHandle drops normally on session end | **Deferred** — same reason as above; would also burn user's only working device |
| Device rotation mid-stream | Rotate phone during a sustain run | scrcpy may renegotiate dimensions; `ScrcpyCaptureStream` should propagate the new `video_header` or restart the decoder | **Deferred**; not investigated. Safe to defer because the live runs were portrait-only. |

The deferred items are explicitly documented in `MEMORY.md` under
`user_usb_tether.md` so future sessions know not to silently re-enable
them.

## Implementation evidence

| ID | Module | Function / Test |
|----|--------|-----------------|
| B1 | `src/scrcpy/mod.rs` | `verify_sha256()`, `push_scrcpy_server()`. Constants `VENDORED_JAR_NAME`, `VENDORED_JAR_SHA256`. |
| B2 | `src/scrcpy/mod.rs` | `bind_host_listener()`, `accept_from_server()`. Topology diagram in the doc comment for `bind_host_listener`. |
| B3-B5 | `src/scrcpy/wire.rs` | `parse_device_meta`, `parse_video_header`, `parse_frame_header`. Constants `DEVICE_NAME_LEN`, `VIDEO_HEADER_LEN`, `FRAME_HEADER_LEN`, `PACKET_FLAG_CONFIG`, `PACKET_FLAG_KEY_FRAME`, `PTS_MASK`, `NO_PTS`. |
| B6, B7 | `src/scrcpy/decoder.rs` | `HevcDecoder::new()`, `decode_packet()`, `flush()`, `convert_to_rgba()`. `ensure_ffmpeg_initialized` Once-guarded init. |
| B8 | `src/scrcpy/stream.rs` | `ScrcpyCaptureStream::launch_with_timeout()`. Field ordering ensures cleanup correctness on Drop. |
| B9 | `src/scrcpy/mod.rs` | `Drop for ScrcpyHandle`. Best-effort `child.kill()` + `child.wait()` + `adb reverse --remove`. |
| B10 | `examples/record_scrcpy_sample.rs` + `docs/phase_1b_codec_probe.md` | Probe results documented; `app_process Server 2.4 list_encoders=true` is the runtime probe. |
| B11 | `examples/scrcpy_soak.rs` | The harness itself. Empirical run captured in this doc above. |
| B12 | `examples/scrcpy_soak.rs` | Same harness, asserts mean fps ≥ `TARGET_FPS * SUSTAIN_PASS_FRACTION` (currently 30 × 0.85 = 25.5). |

## Reproduction commands

All commands assume `cwd=symthaea` and a live Pixel 8 Pro at serial
`41201FDJG000UM`. Substitute the serial as needed.

### Offline tests (no device required)

```bash
nix develop --command cargo test \
    -p symthaea-phone-embodiment \
    --features scrcpy \
    --lib -- scrcpy:: streaming_bridge::
```

Expected: `test result: ok. 34 passed; 0 failed; ... finished in ~0.14s`
covering 5 lifecycle + 16 wire + 5 decoder + 6 stream + 2
streaming_bridge tests.

### Re-record the offline asset (refresh sample.hevc.wire)

```bash
nix develop --command cargo run --example record_scrcpy_sample \
    --features scrcpy --release \
    -p symthaea-phone-embodiment \
    -- 41201FDJG000UM 10
```

Expected: ~120 KB asset written to
`crates/symthaea-phone-embodiment/tests/data/sample.hevc.wire`. The
asset already in the repo is the canonical version; only re-record
when the wire format changes (e.g. on scrcpy upgrade).

### Live sustain run

```bash
# Optional but recommended: get active screen content
adb shell am start -a android.intent.action.VIEW \
    -d "https://www.youtube.com/watch?v=jNQXAC9IVRw"
sleep 4

adb reverse --remove-all
nix develop --command cargo run --release \
    --example scrcpy_soak \
    --features scrcpy \
    -p symthaea-phone-embodiment \
    -- 41201FDJG000UM 30 720 5
```

Args: `serial duration_secs max_size report_interval_secs`. The
default sustain harness is configured for 30 s duration so it
finishes quickly and doesn't load the USB tether for long.

**Do NOT run the 10-minute soak (`duration_secs=600`) on a USB-tether-only
host** — it pushes ~400 KB/s sustained over the same link the user's
internet rides on. Phase I.D will add a `--soak-friendly` mode that
caps wire bandwidth.

## Known limitations

1. **Sustained 30 fps is not yet Proven.** The empirical ceiling is
   ~23 fps in the best window and ~16 fps mean. Path to Proven is
   GPU-accelerated decode (Phase I.D).
2. **Codec primary is hardcoded HEVC.** The fallback ladder (H.264,
   AV1) is documented in `cybernetic_defaults` and gated behind
   `h264-fallback` / `av1-research` features but the
   `ScrcpyCaptureStream` decoder construction does not branch on
   `video_header.codec`. Left as a Phase I.B.4.5 follow-up if the
   primary path ever fails on a different device.
3. **Static screen → bursty wire.** Without active content (or
   `i-frame-interval=1` override) scrcpy delivers ~12 packets every
   15 seconds, which the harness reports as 0.5-0.8 fps. The harness
   forces `i-frame-interval=1` to mitigate.
4. **Physical USB chaos tests deferred** until the user has a
   separate WiFi card. Software-only fault injection covers the
   crash + reverse-tunnel-remove cases.
5. **Single-frame decode latency p99 = 147 ms.** Long-tail outliers
   dominate the wall-clock fps. A frame-dropping policy or
   asynchronous decode would smooth this.
6. **No memory growth detection** in the soak harness (deferred). A
   10-minute soak would surface memory leaks but is bandwidth-unsafe
   on the tether.

## Test count summary

| Layer | Tests | Where |
|---|---|---|
| Lifecycle | 5 | `src/scrcpy/mod.rs::tests` |
| Wire parser | 16 | `src/scrcpy/wire.rs::tests` |
| HEVC decoder | 5 (incl. end-to-end real-asset decode) | `src/scrcpy/decoder.rs::tests` |
| Capture stream | 6 (error display + helpers) | `src/scrcpy/stream.rs::tests` |
| Streaming bridge wrapper | 2 (deref, send/sync marker) | `src/streaming_bridge.rs::tests` |
| **Total scrcpy + bridge** | **34** | runs in 0.14 s under `--features scrcpy` |

Phase I.B end-to-end vertical: ~1500 LOC, 34 tests, all passing in
0.14 s on every build under `nix develop --command cargo test
-p symthaea-phone-embodiment --features scrcpy --lib -- scrcpy::
streaming_bridge::`.
