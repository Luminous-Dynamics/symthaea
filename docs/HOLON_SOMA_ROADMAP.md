# Holon-Soma Roadmap

**Version**: 1.7 (2026-04-17, later same day)
**Scope**: The PQC-sealed binary RDP wire connecting a Symthaea cognitive
loop to a physical embodiment (starting with the Pixel 8 Pro), and the
research program that rides on top of it.
**Status**: Phase I.A delivered and hardware-validated. Phase I.A.5
hardening interlude complete (9 of 10 tracks). Phase I.A.2 Pieces 1+2
code-complete. **Phase I.B CLOSED on main (2026-04-17, recovery merge** of
the April-14 worktree that had been pruned without merging; `git merge
--no-ff phase-1b-recovery`). **Phase I.C harness extended**: scrcpy-backed
WS-vs-QUIC A/B example, unprivileged-user-namespace netem packet-loss
script, and updated verification doc — live-device A/B run pending. Forward
phases (II → III → IV → V) planned and pre-registered where applicable.

**Relationship to `docs/ROADMAP.md`**: that doc describes the whole
Symthaea AGI-partner program (Phi_dyad goal). This document describes
one specific substrate within that program — the wire that lets a
Symthaea cognitive loop inhabit a physical body over a network. The
two are complementary: this roadmap is one substrate-layer tower
within the broader Symthaea research vision.

## Executive summary

Symthaea can today observe and control a physical Pixel 8 Pro end-to-end
over a cryptographically sealed binary wire, with every layer from codec
to WebSocket to egui renderer runtime-verified. The wire compresses
**3.27-3.52× smaller than the JSON baseline** on real Pixel screen
content, with ~60 µs seal+open latency and <40 ms end-to-end WebSocket
delivery. 44+1 tests pass in ~0.63 s wall time; 1 ignored soak test
pushes 10,000 frames with zero drops.

The program from here has five phases. Phase I.B (scrcpy persistent
capture) unlocks real 30 fps throughput and is the prerequisite for
everything after it. Phase II (attention backchannel + codec
compression) delivers the consciousness-gated bandwidth thesis. Phase
III (Bandwidth-Quality-Φ sweep) is the pre-registered empirical
validation. Phase IV (split-cognition Markov blanket) is the research
bet — publishable either way. Phase V (cross-body dream consolidation)
is gated on Phase IV's outcome.

**The wire is real. The research program can now ride on it.**

---

## Current state (2026-04-14, `HEAD = b49246fb7d` or later)

### What works end-to-end

| Layer | Artifact | Status |
|---|---|---|
| Capture | `PhoneBridge::capture_and_observe_rgba` (ADB screencap) | ✅ Proven on hardware |
| Codec | `HybridCodec` + `SomaRdpServer` (HDC tile change detection) | ✅ Proven, 576 patches/frame |
| Sealing | `rdp_wire::seal_frame` (bincode + ChaCha20-Poly1305) | ✅ 3.27-3.52× vs JSON on hardware |
| Transport | Holon WS `broadcast::channel` (cap 16) + VecDeque catch-up (cap 512) | ✅ End-to-end tested |
| Viewer | `holon_rdp_viewer` egui + tokio-tungstenite + sync/async bridge | ✅ Compile + integration |
| Unsealing | `rdp_wire::open_frame` + `HolonRdpViewer::apply_frame` | ✅ Byte-exact round-trip |
| Reverse path | egui pointer → `InputFrame::Pointer` → `seal_input` → ADB tap | ✅ Proven in live run |
| Replay protection | `swarm::replay_window::ReplayWindow` (64-bit sliding per `(source_id, payload_type)`) | ✅ 11 unit + 2 integration tests |
| AEAD hygiene | Per-session random `source_id`+`epoch`, payload-type nonce separation | ✅ 11 vector tests |

### Measured numbers (live Pixel 8 Pro, 2026-04-14)

| Metric | Value | Source |
|---|---|---|
| Sealed envelope size (real delta, active screen) | **~480 KB/frame** | 15 s swipe-active run |
| Sealed envelope size (synthetic delta) | ~65 KB/frame | `integration_rdp_wire` |
| Bandwidth ratio (real hardware, active content) | **3.268×** | 15 s active run |
| Bandwidth ratio (real hardware, baseline) | **3.516×** | 10 s run |
| Bandwidth ratio (synthetic) | 2.998× | `wire_envelope_beats_json_by_3x` |
| Seal+open latency | <60 µs | Measured in isolation |
| End-to-end WS delivery latency | <40 ms | `integration_holon_ws` |
| USB 2.0 sustained throughput | ~35 MB/s | `adb push/pull` 200 MB random |

### Open gaps (documented honestly)

1. **PQC handshake is placeholder.** Both sides use `[0x42; 32]`
   session key installed out-of-band. Real ML-KEM handshake is
   structurally blocked by the broadcast-sealing architecture (one key
   for all viewers) — see Track 2.5 below.
2. **Small hardware sample size** (8 codec'd frames across 3 runs).
   Ratio is stable but the full distribution is uncharted. Phase I.B
   scrcpy unblocks this.
3. **ADB polling ceiling** caps capture at ~4 fps. Phase I.B removes
   this.
4. **Heavy 30 fps content would exceed USB 2.0 budget** without codec
   compression. The ladder (LZ4 → sparse → inter-frame) moves from
   nice-to-have to Phase II prerequisite — see "Revised Phase II
   sequencing" below.

---

## Phase schedule

### Phase I.A — Binary wire ✅ CLOSED (2026-04-13 → 2026-04-14)

All 7 W-claims proven with runtime tests. 44+1 tests passing. See
`docs/phase_1a_verification.md` for the per-claim status table and
`papers/phase_1a_results.md` for the citable results paper.

**Key commits**: `932fcda056` (binary wire), `27cef1fd8b` (WS
integration tests), `7645c5d837` (live hardware run), `b49246fb7d`
(extended measurements).

### Phase I.A.5 — Hardening interlude ✅ 9 of 10 tracks CLOSED (2026-04-13 → 2026-04-14)

| Track | Status | Commit |
|---|---|---|
| 2.1 Replay window primitive | ✅ | `7e08f01093` |
| 2.2 Wire replay into `RdpSession::open` | ✅ | `7e08f01093` |
| 2.3 `wrapping_add` in `next_nonce` | ✅ | `7e08f01093` |
| 2.4 AEAD vector tests | ✅ | `a29cac49e9` |
| **2.5 PQC handshake unblock** | ⏳ **DEFERRED** | — |
| 3.1 Orphan mesh cluster deletion (−1,547 LOC) | ✅ | `c5327d1420` |
| 3.2 Notify-driven WS broadcast | ✅ | `cd0c24a715` |
| 4.2 `docs/dev/test_loop.md` | ✅ | `c8118c7801` |
| 5.1 `docs/phase_1a_verification.md` | ✅ | `c8118c7801` |
| 5.2 `papers/preregistration.md` | ✅ | `88bc2dbc68` |

**Track 2.5 deferral reason**: the `HolonHttpState` broadcast-sealing
architecture assumes ONE session key shared across all connected
viewers, but real ML-KEM encapsulation is per-session. Unblocking
requires per-connection sealing restructure + bidirectional KEM
message types over the WS + handshake-phase state machine. Estimated
4-6 hours as a focused session when real network deployment creates
the pressure to do it.

### Phase I.A.2 — Holon RDP viewer 🟡 2 of 3 pieces DONE (2026-04-14)

| Piece | Status | Commit |
|---|---|---|
| 1. egui window + FrameBuffer blit | ✅ | `fa6d92a007` |
| 2. tokio-tungstenite WS client + sync/async bridge | ✅ | `9887959e8e` |
| **3. Real PQC handshake** | ⏳ **BLOCKED** | — (same blocker as Track 2.5) |

Piece 1+2 together give a functional localhost-only viewer: run
`holon_rdp_viewer` with `--features holon-viewer`, connect to a Holon
WS server on `:7778`, see the Pixel screen render in an egui window.
Piece 3 would swap the `[0x42; 32]` placeholder for a real KEM-derived
key — blocked on the broadcast-sealing restructure.

### Phase I.B — Persistent capture ✅ CLOSED (2026-04-14)

**Status**: All 8 subtasks complete. 11 of 12 core claims **Proven**;
1 claim (sustain ≥25.5 fps mean) **Asserted** with documented
empirical ceiling of ~23 fps peak / ~16 fps mean on single-CPU HEVC
software decode. Path to true 30 fps is GPU-accelerated decode,
deferred to a future Phase I.D.

**Worktree**: `.claude/worktrees/session-phase-1b-scrcpy`
**Branch**: `worktree-session-phase-1b-scrcpy`
**Verification doc**: `symthaea/docs/phase_1b_verification.md` (in
worktree; will sync to main on merge)
**Codec probe doc**: `symthaea/docs/phase_1b_codec_probe.md` (in
worktree; the live probe that pivoted v1.3 AV1 → v1.4 HEVC)

**Subtask completion log**:

| Subtask | Commit (worktree) | Outcome |
|---|---|---|
| I.B.0 codec ladder probe | `b2bd2aaccd` | HW HEVC + H.264 confirmed; AV1 software-only on Android 16 → roadmap pivot |
| I.B.1 vendor scrcpy-server v2.4 + SHA | `9135867d71` | 124 KB JAR, SHA pinned against upstream `SHA256SUMS.txt` |
| I.B.2 lifecycle (push/start/reverse/drop) | `aa10625dca` | RAII `ScrcpyHandle` cleans up server child + reverse tunnel |
| (flake) ffmpeg_7 + libclang + bindgen | `61e8a8031a` | Default dev shell now builds the `scrcpy` feature |
| I.B.3a wire parser | `98942053ec` | 16 unit tests, pure data, no I/O |
| I.B.3b ffmpeg-next HEVC decoder | `580df62e78` | Once-guarded init, lazy swscale cache, drain loop |
| I.B.4a `ScrcpyCaptureStream` connector | `bf07f9df26` | End-to-end vertical, 6 unit tests |
| I.B.4b `StreamingPhoneBridge` wrapper | `3e08637b29` | Send/Sync wrapper pattern, keeps `PhoneBridge` `EmbodimentBridge`-compatible |
| I.B.5 recorded asset + offline test | `7496a7df61` | 124 KB real Pixel HEVC, end-to-end decode in 0.14s on every build |
| I.B.6 sustain harness + 4 quirk fixes | `a7e9a9aa36` | 470+ frames decoded on live device; `control=true`, `send_dummy_byte`, `display_buffer`, keyframe-interval-15s all caught and fixed |
| I.B.7 verification doc + chaos test | `d43e45dfbd` | 12 core + 8 auxiliary claims documented; software-only crash test proven (no physical USB unplug — user's tether is their only internet) |

**Final test count**: 34 passing in 0.14s under `cargo test
-p symthaea-phone-embodiment --features scrcpy --lib -- scrcpy::
streaming_bridge::`. ~1500 LOC, 2 examples, 1 vendored JAR, 1 recorded
asset, 2 verification docs.

**Empirical sustain data** (canonical run, YouTube playing at 720p):

```
473 frames in 30.00 s
mean fps           : 15.77
peak window fps    : 23.27
wire (HEVC) bytes  : 12 127 KB (404.2 KB/s)
decode p50         : 34 ms (right at the 33 ms 30-fps ceiling)
decode p95         : 105 ms
decode p99         : 147 ms
read timeouts      : 16 over 30 s
```

**Path to Proven for B12** (true 30 fps sustained), enumerated in the
verification doc and unblocked by Phase I.D: GPU-accelerated HEVC
decode via vaapi/vdpau/nvdec, lower max_size, frame-dropping policy,
async decode off the cognitive-loop thread.

**Original I.B planning section preserved below for traceability**:

### Phase I.B — Persistent capture (original plan, ~6-10 hours)

**Goal**: Remove the 250 ms ADB polling ceiling. Unlock 30-60 fps
capture with ~30 ms latency. Generate the first real-fps data for
every downstream phase.

**Plan A vs Plan B** (decided 2026-04-14 after 30-min evaluation):

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **A. scrcpy-server.jar** | Designed for live mirroring; reconnect-safe; no duration limit; continuous NAL stream; `--video-codec=h264` flag forces codec | Vendor a JAR, pin SHA, lifecycle mgmt, reverse tunnel | **CHOSEN** |
| B. `adb exec-out screenrecord --output-format=h264 -` | No JAR, no reverse tunnel, native Android | **180 s hard limit** (blocks 10-min soak); defaults to H.265 on Pixel 8 Pro (no flag to force h264 reliably); stdout streaming undertested | Rejected on duration limit |

Plan A wins because the 180 s screenrecord limit is a hard blocker
for the soak test and for Phase III's multi-minute task trials.

**Codec decision (v1.4 — HEVC primary after probe falsified the v1.2 AV1 pivot)**:
**Request HEVC** via scrcpy's `--video-codec=h265`, target the hardware
encoder `c2.exynos.hevc.encoder`.

The v1.2/v1.3 plan assumed Pixel 8 Pro exposed a hardware AV1
encoder via MediaCodec. **The Phase I.B.0 codec ladder probe
falsified this on Android 16**: only `c2.google.av1.encoder` and
`c2.android.av1.encoder` are enumerated for AV1 — both software.
The hardware encoders on this device are H.264 (`c2.exynos.h264`)
and HEVC (`c2.exynos.hevc`). See `docs/phase_1b_codec_probe.md`
in the worktree for raw probe output.

Software AV1 at 30 fps on a phone is untenable: battery drain,
thermal throttling, won't sustain target rate, contaminates Phase
IV PE measurement with thermally-modulated latency.

HEVC HW retains most of AV1's compression advantage (~50% smaller
than H.264 vs AV1's 30-40%) without requiring HW that doesn't
exist on this device. USB 2.0 budget relief survives the pivot.

Decoder pivot: `rav1d` (AV1-only) → **`ffmpeg-next`** (HEVC + many).
The v1.2 concern about C-library friction was specifically about
`openh264`'s Cisco-binary download build script. `ffmpeg-next` uses
standard pkg-config to find the system `ffmpeg`, which is a
first-class package on NixOS. First build must run inside `nix
develop` with `ffmpeg` in the shell; documented in Phase I.B.4.

`rav1d` survives in the codebase as an optional dep behind the
`av1-research` feature, used by Phase II.5's compressed-domain
perception research.

**Codec fallback ladder (v1.4, post-probe)**:
1. **HEVC** via `c2.exynos.hevc.encoder` + `ffmpeg-next` ← primary
2. **H.264** via `c2.exynos.h264.encoder` + `openh264` (or
   `ffmpeg-next` if already linked) behind `h264-fallback` feature —
   used if HEVC decode unavailable for any reason
3. **AV1 SW** via `c2.google.av1.encoder` + `rav1d` behind
   `av1-research` feature — research-tier only, for tethered
   battery-insensitive bench runs (Phase II.5)

The codec ladder probe (I.B.0) was executed 2026-04-14; results in
`docs/phase_1b_codec_probe.md` (worktree). Probe is now part of
the Phase I.B startup sequence so any future device or Android
update is caught immediately.

**Approach**: Fetch the pinned scrcpy-server.jar (from scrcpy v2.4
GitHub release, SHA256 pinned in-tree — v2.4 is the first release
with stable AV1 support), `adb push` it to
`/data/local/tmp/scrcpy-server.jar`, run it via `app_process` with
`video_codec=av1`, open `adb reverse localabstract:scrcpy
tcp:<local>`, connect a local tokio TCP listener, parse the scrcpy
binary framing (metadata header + per-frame length-prefixed OBU for
AV1), decode via `rav1d` on a rayon-spread thread pool (AV1
decoding parallelizes well across cores — desktop Ryzen/Intel can
handle 4K60 AV1 in software).

**Zero-latency encoder tuning (v1.3)**: default the scrcpy
`app_process` invocation with glass-to-glass latency flags:
- `--display-buffer=0` — no device-side frame buffering
- `--max-fps=30` — lock encoder pacing to match cognitive loop (not
  29.97 or 60, exactly 30)
- `--video-codec-options=...` — force fastest AV1 profile. **Value
  needs probe validation**: AV1 MediaCodec profile integers are
  device-specific; `profile=0` (the speculative v1.3 suggestion) may
  map to AV1ProfileMain8 on Tensor G3 but could reject on other
  builds. The codec ladder probe (I.B.0) must also probe
  video-codec-options with and without the tuning bundle and log
  the working combination.

For a cybernetic loop we optimize glass-to-glass latency over
perceptual quality. Rationale: Symthaea does not need cinema-grade
fidelity; it needs the freshest possible frame at cognitive-cycle
time. Every millisecond of device buffering is a millisecond of PE
contamination in Phase IV.

**New files**:
- `crates/symthaea-phone-embodiment/src/scrcpy.rs` (~400 LOC — revised up from 300)
- `crates/symthaea-phone-embodiment/vendor/scrcpy-server-v2.4.jar` + `.sha256`
- `crates/symthaea-phone-embodiment/vendor/README.md` (provenance + rebuild instructions)
- `crates/symthaea-phone-embodiment/tests/data/sample.av1` (~40 KB recorded OBU asset for offline decode tests; fallback `sample.hevc` + `sample.h264` optional for the respective feature builds)

**Modified files**:
- `crates/symthaea-phone-embodiment/src/adb.rs` — `push_scrcpy_server`, `start_scrcpy`, `reverse_tunnel`, `stop_scrcpy`
- `crates/symthaea-phone-embodiment/src/bridge.rs` — `capture_stream`, `tick_rgba` async variants
- `crates/symthaea-phone-embodiment/Cargo.toml` — `scrcpy` feature + `rav1d` optional dep; `hevc-fallback` + `h264-fallback` feature gates for the ladder
- `examples/phone_rdp_share.rs` — swap `capture_and_observe_rgba` for the persistent stream path when `scrcpy` feature is on

**Feature gating**: Keep the existing ADB-polling capture path
alive as the default. scrcpy goes behind a new `scrcpy` feature so
the diagnostic polling path stays available for troubleshooting,
and so sessions without a device can still build-test the crate.

**New deps**:
- `rav1d` (pure Rust, required at the `scrcpy` feature) — primary AV1 decoder
- `ffmpeg-next` (optional, gated by `hevc-fallback`) — HEVC rung of the fallback ladder
- `openh264 = "0.6"` (optional, gated by `h264-fallback`) — H.264 rung of the fallback ladder, last resort

**Soak observability** (NEW — previously missing): the 10-minute
soak must emit a metrics line every 10 s with:
- sealed bytes/sec (rolling 10 s mean)
- seal+open latency p50 / p99 (microseconds)
- broadcast queue depth (current + peak)
- replay-reject count (cumulative)
- scrcpy-server restarts (cumulative, should be 0)
- decoded frames / dropped frames (cumulative)

Without this, the soak only measures end-state ("no crash"), not
the trajectory. Degrading wires fail slowly.

**Verification**:
1. Codec ladder probe — first worktree task; logs which rung
   (AV1/HEVC/H.264) the device actually supports and locks the
   feature set for the rest of the session
2. Unit tests with recorded asset — `sample.av1` (primary) plus
   optional `sample.h264` / `sample.hevc` for fallback-flag builds
   — decodes to expected RGBA without a device
3. `phone_rdp_share --features scrcpy` sustains ≥30 fps for 60 s on the live Pixel via AV1
4. AV1 vs H.264 bandwidth comparison on the same 60 s trace — expect 30-40% reduction; publish numbers in the Phase I.B verification doc
5. 10-minute soak: live metrics line every 10 s; end state shows ≥30 fps sustained, 0 crashes, 0 restarts, <0.1% dropped, stable memory

**Estimated effort**: **5-8 hours** in a fresh worktree (v1.2
revised down from v1.1's 6-10 h because `rav1d` is pure-Rust and
eliminates the C-integration time block). Breakdown:
- 0.5 h: codec ladder probe + decision log
- 1 h: JAR vendor + SHA pin + adb lifecycle (push, run, reverse, stop)
- 1.5-2 h: scrcpy binary framing parser + `rav1d` integration (pure Rust — expect little friction)
- 1 h: async bridge + feature gate wiring
- 0.5-1 h: unit tests with recorded AV1 asset
- 1-2 h: live fps verification + soak harness + observability metrics
- 0.5 h: retries, flake-fixing, documentation

Retrospective Rule #1 was "don't under-estimate." v1.1's 6-10 h
budgeted for C-library friction that v1.2's AV1+rav1d stack
removes. If the codec probe surfaces a surprise (AV1 unsupported on
this Android build) and forces fallback to H.265 or H.264, the
budget reverts to v1.1's 6-10 h envelope.

**Why this phase unblocks everything**: without real fps, all bandwidth
measurements are synthetic or polling-limited. Phase II can't test
attention gating meaningfully at 4 fps. Phase III's Φ-sweep needs 20+
trials per condition which at 4 fps takes all day. Phase IV's Markov
blanket test needs stable-state PE which requires the codec to be
running at its design rate.

### Phase I.C — QUIC transport swap 🚧 IN PROGRESS v1.5+ (~1-2 days after I.B)

**Goal**: Eliminate TCP head-of-line blocking on the Holon wire.
Move video frames to unreliable QUIC datagrams while keeping
control messages (handshake, InputFrame, AttentionMap) on reliable
QUIC streams.

**Why this matters**: the current wire runs WebSockets over TCP via
`tokio-tungstenite`. TCP guarantees in-order delivery, which means
a single dropped packet halts the entire stream until retransmit —
Head-of-Line blocking. Over local USB RNDIS this is rare, but over
any real WiFi or WAN deployment (and eventually over the open
internet for Phase IV split-cognition) HoL creates unpredictable
frame stutter that contaminates both UX and PE measurement.

**QUIC fixes this cleanly**:
- Video frames → unreliable datagrams. Dropped frame? Next frame
  ships immediately, no waiting. Frame loss becomes visible as a
  measurable metric, not invisible as a stall.
- PQC handshake → reliable stream. Exactly-once delivery, still
  ordered.
- Input events → reliable stream. Exactly-once delivery, ordered.
- Attention backchannel (Phase II) → reliable stream at 5 Hz.

**Crate**: `quinn` (pure Rust, `rustls`-based, production-grade).
Zero new C deps. Supports both stream and datagram APIs natively.

**Why I.C and not bundled into I.B**: swapping transport AND
capture in the same phase triples the debugging surface. The
v1.1 failure-mode inventory becomes twice as hard to validate if
both layers are new. Scrcpy first, prove the AV1 stack works over
the existing WS wire, then swap transport under it with the
capture layer as a known-good.

**Replay window implications**: QUIC datagrams are unordered by
design. The `ReplayWindow` primitive already handles out-of-order
within the 64-slot sliding window (per `(source_id, payload_type)`)
— this design choice pays off here. Document the interaction
explicitly in the Phase I.C verification doc.

**Architectural work**:
- New `swarm::quic_transport` module wrapping `quinn::Endpoint`
- `seal_frame` / `open_frame` unchanged — they already return
  opaque `Vec<u8>`, the transport layer is below them
- `holon_rdp_viewer` gets a `--transport=quic|ws` flag for A/B
  comparison during migration
- Server binds both WS (legacy) and QUIC (new) during the
  transition window; one-shot WS removal once QUIC parity proven

**New dep**: `quinn = "0.11"` + `rustls = "0.23"`

**Verification**:
1. Localhost A/B: same scrcpy stream pushed through WS and QUIC
   simultaneously; compare end-to-end latency p50/p99
2. Induced packet loss test: `tc qdisc` 1% loss, measure frame
   stall behavior (WS stalls, QUIC drops-and-continues)
3. 10-min soak over QUIC with observability metrics from Phase I.B
4. Migration cutover: run with `--transport=ws` for a session, then
   `--transport=quic`, confirm identical functional behavior

**Current verification note (Apr 15, 2026)**: initial headless localhost
transport parity is implemented and recorded in
`docs/PHASE_1C_QUIC_TRANSPORT_VERIFICATION.md`. The synthetic 30-frame A/B
run passed with WS p50 27.3 ms / p99 32.9 ms and QUIC p50 28.6 ms / p99
83.3 ms, plus reverse input path verification for both transports. This is a
functional parity checkpoint only. A follow-up live phone-content checkpoint
using `PhoneBridge.capture_and_observe_rgba` found and fixed a QUIC
datagram-only failure for oversized full frames by adding a reliable
unidirectional-stream fallback. A deterministic QUIC datagram-loss smoke now
proves drop-and-continue behavior for datagram-sized frames under injected
loss, but the real `tc qdisc` packet-loss comparison remains open. The real
scrcpy-stream A/B, kernel packet-loss test, 10-minute soak, and manual cutover
remain open.

**Estimated effort**: 1-2 days. Quinn is well-documented, the wire
above it is unchanged.

### Phase II — Attention backchannel + codec compression 📋 (~1 week after I.C)

**Goal**: Desktop Symthaea's saliency map flows backward through the
WS and modulates which tiles the phone codec prioritizes. Combined with
the codec compression ladder, fits 30 fps heavy-activity content within
~25% of the USB 2.0 budget.

**Track ordering (revised)**: Track B.1 (LZ4 wrap) **blocks** Track A.
You don't know whether attention gating is needed until you measure
real-content 30 fps sealed+LZ4'd frame size against the USB 2.0
budget. If LZ4 alone fits under 25% of the 35 MB/s ceiling, the
entire attention backchannel becomes optional polish, not a
prerequisite. Measure first, then decide.

**Two tracks** (sequenced, not parallel):

**Track A — Attention backchannel** (~days):
1. Desktop `HolonViewerApp` exports `VisionManifold::saliency_map()` as
   `[u8; tile_cols × tile_rows]`
2. Sealed as a new `HolonRdpMessage::AttentionMap(Vec<u8>)` variant,
   pushed upstream at 5 Hz
3. `SomaRdpServer::apply_attention_map(&[u8])` stores the latest
4. Modified `tick()` consults the map when deciding which
   detected-change tiles enter the outbound `DeltaFrame`
5. Priority = `attention_weight × similarity_delta × adaptive_tier`
6. Tiles below threshold are dropped OR re-quantized at coarser `i4`
7. Existing `AdaptiveQualityEngine` finally gets instantiated and fed
   `consciousness_level` from the `EmbodimentBridge`

**Track B — Codec compression ladder** (~1 week, revised priority from the 2026-04-14 USB measurement):

| Layer | Technique | Effort | Stacked ratio vs JSON |
|---|---|---|---|
| **Now** | bincode + AEAD | — | 3.51× |
| +1 | LZ4 wrap via existing `lz4_flex` workspace dep | 1 hr | **7-10×** |
| +2 | Sparse patch encoding (only changed bytes in deltas) | 3-4 hr | **15-30×** |
| +3 | Content-adaptive quantization (`i4` nibbles UI, `i2`/boolean static) | 1 day | **40-100×** |
| +4 | Inter-frame delta coding | 2 days | **200-1500×** |

**Prioritized sequence per the USB measurement**:
1. LZ4 wrap first (cheapest, existing dep, 2-3× free)
2. Measure at 30 fps with scrcpy — if under 25% USB budget, STOP HERE
3. If over budget, add sparse patch encoding
4. Content-adaptive only if still over budget

**Verification**:
1. Headless: synthetic attention (top half = 1.0, bottom = 0.0), verify only top-half tiles in outbound `DeltaFrame.patches`
2. Live: stare at top of phone screen, scroll bottom — bottom dropped/coarsened, top crisp
3. ≥40% bandwidth reduction for focal-attention scenarios on the same trace

### Phase II.5 — Compressed-domain perception 🔬 NEW v1.3 RESEARCH (~2-3 weeks, parallelizable with III)

**Goal**: feed AV1 motion vectors and residuals directly into
Symthaea's vision manifold, skipping the RGBA decode step for
motion-surprise computation.

**The observation**: the Pixel 8 Pro's hardware AV1 encoder already
computed per-block motion vectors and residual energy as part of
encoding each frame. Desktop Symthaea then decodes to RGBA and
runs its own motion estimation on the reconstructed pixels — which
is computationally redundant AND less accurate than the HW
encoder's estimates (the HW encoder sees the true previous frame
before quantization; the desktop sees the quantized reconstruction).

**The idea**: extract motion vectors from the AV1 bitstream and
feed them directly to `VisionManifold::apply_motion_field()`.
Symthaea learns which screen regions moved from the encoder's
metadata, not from its own recomputation. The HDC motion-surprise
channel becomes effectively free — the phone silicon is doing the
work.

**Honest caveats** (this is research, not production):

1. **`rav1d` does not expose motion vectors in its stable API.**
   `dav1d` internals compute them but the public Rust crate surface
   treats frames as opaque. Options:
   - Patch `rav1d` to expose an "expose decode metadata" callback
     (upstreamable? maybe — dav1d has a `--frame-metadata` debug
     output, so the data path exists internally)
   - Fork `rav1d` in-tree, carry the patch until upstream accepts
   - Use `dav1d-sys` directly and call into the C API, which has
     slightly more expose-able internals (but reintroduces the C
     dep friction that the v1.2 AV1 pivot was meant to eliminate)
2. **Motion vectors are per-block, quantized, and encoder-biased.**
   They tell you "this 16×16 block probably moved like (dx,dy)"
   not "this object moved." Object-level motion is still
   downstream inference work.
3. **Residual energy is decoded post-IDCT.** Extracting it means
   tapping the decoder mid-pipeline, which is even less exposed
   than MVs.

**Research plan** (not production):
- **Week 1**: Verify dav1d internally computes and stores the MVs
  we'd want. Build a `dav1d-debug-probe` example.
- **Week 2**: Upstream (or fork) a `metadata_callback` to rav1d.
  Prototype feeding MV data into a new `VisionManifold` channel.
- **Week 3**: A/B bench: motion-surprise from pixel decode vs
  motion-surprise from AV1 MVs, measured against a labeled motion
  ground truth on a recorded trace. Publish the comparison.

**Parallelizable with Phase III**: the Φ-sweep doesn't touch the
vision pipeline internals, so a second session can work on II.5
while the Phase III benchmark harness runs.

**Publishable outcome**: if the MV-fed motion-surprise signal
correlates with pixel-computed motion-surprise at ≥0.8 Pearson, this
is a real result worth writing up. If it doesn't correlate,
that's a publishable negative — "HW encoder motion vectors are
insufficient proxies for cognitive motion-surprise signals."

**Risk of scope-creep**: this phase is easy to let balloon. Hard
cap: if `rav1d` MV extraction isn't working after week 1, abandon
to Phase III. The cognitive loop's pixel-space motion pipeline is
already fast enough for the program's core claims.

### Phase III — Bandwidth-Quality-Φ sweep 🔒 PRE-REGISTERED (~days after II)

**Goal**: Empirically validate that consciousness-gated encoding is a
functional advantage, not theater.

**Pre-registration**: `papers/preregistration.md` PR-001 (commit
`88bc2dbc68`, frozen 2026-04-13 before any benchmark code exists).
Provenance verifiable via `git log papers/preregistration.md`.

**Experimental design** (frozen):
- **IV**: simulated Φ ∈ {0.15, 0.25, 0.35, 0.50, 0.65, 0.85}
- **Tasks**: "open YouTube and search NixOS", "open Settings", "open Clock"
- **n**: 20 trials × 6 Φ levels × 3 tasks = 360 runs
- **DVs**: bandwidth (bytes/s), task success rate, PE mean + max, WM saturation, latency
- **Hypothesis**: bandwidth decreases monotonically with Φ; task success stays ≥95% at Φ ≥ 0.35; PE knee around Φ=0.25
- **Test**: paired t-test, α=0.05 (revised from 0.01), effect size ≥ 0.5
- **All four outcomes publishable** — confirm direction + magnitude / direction only / null / wrong direction

**Power analysis (NEW, v1.1)**: paired t-test with n=20 per cell, α=0.05,
one-tailed → 80% power at Cohen's d ≈ 0.58. At the v1.0 α=0.01 threshold
the same n required d ≈ 0.78 — a very large effect, and any subtler
structure would have been invisible. Relaxing to α=0.05 is the
honest move for a first-pass exploratory study. The pre-registration
document must be updated to match (amendment, not silent revision —
leave the v1.0 freeze intact and add a dated amendment below it).

**Analysis script freeze (NEW, v1.1)**: PR-001 currently freezes the
hypothesis but not the analysis code path. Before the first trial
runs, `scripts/phi_sweep_analysis.py` must be committed with the
exact pandas/scipy calls, random seed for trial ordering, outlier
policy, and plot-generation code. The analysis script hash goes
into the preregistration amendment. This closes the
garden-of-forking-paths loophole.

**New files**:
- `benches/phi_sweep.rs` — Criterion harness driving the 360-run matrix
- `scripts/phi_sweep_plot.py` — pandas + matplotlib three-panel plot
- `papers/phi_sweep_results.md` — narrative + tables + plots

**Estimated effort**: ~1 week.

### Phase IV — Split-cognition Markov blanket test 🔒 PRE-REGISTERED (~weeks after III)

**Goal**: Test whether two Symthaea instances (phone brainstem + desktop
prefrontal) form a single Markov blanket spanning the network boundary.

**Pre-registration**: `papers/preregistration.md` PR-002 template
frozen 2026-04-13. Final hypothesis + magnitude to be frozen after
Phase III completion.

**Operational hypothesis**: PE_coupled < PE_phone_alone + PE_desktop_replay

**Three conditions**:
1. Phone-only — phone autonomous, no desktop input
2. Desktop-replay — desktop runs on a recorded phone trace, no feedback
3. Coupled — full bidirectional pipe

**Statistical test**: one-tailed paired t-test, α=0.01, n ≥ 30 task runs per condition.

**Architectural work**:
- `src/cognitive_loop/config/mod.rs`: `EmbodimentSource::{Local, RemoteRdp}`
- `src/cognitive_loop/embodiment.rs` (new): `RemoteRdpEmbodiment` implementing `EmbodimentBridge` via the sealed WS
- `examples/split_cognition.rs` (new): spawn both instances, record PE traces per condition
- `tests/integration_markov_blanket.rs` (new): three-condition regression test on a recorded trace

**Hidden hard problem — clock synchronization (NEW, v1.1)**:
`RemoteRdpEmbodiment` looks like a normal `EmbodimentBridge`
implementation, but PE (prediction error) is clock-sensitive: the
desktop's cognitive loop stamps predictions at desktop-time t₀ and
compares against phone observations that carry phone-time t₁.
Network latency + clock skew + frame buffering create a variable
delta that will contaminate the PE signal unless explicitly
modeled. Options:
1. **Phone-time canonical**: desktop adjusts its own PE window to
   phone timestamps, tolerates extra jitter on cognitive-loop side
2. **NTP-style offset estimation**: periodic round-trip pings
   estimate skew, subtract from every frame
3. **Sequence-number only**: ignore wall clock, compare by frame
   sequence number, accept that task-latency metrics become
   per-hop not end-to-end

Pick one before writing `RemoteRdpEmbodiment`. This is the likely
Phase IV blocker and the design decision most impactful on the
experimental validity of the Markov blanket test. Recommendation:
option 2 (NTP-style) — cheapest to implement, most honest.

**All three outcomes publishable**:
- Confirmed: operational proof of distributed cognition across a network
- Null: coupled/decoupled indistinguishable; Markov blanket framing falsified
- Wrong direction: coupling harms both instances; design revisit

**Estimated effort**: 2-4 weeks.

### Phase V — Cross-body dream consolidation 🔒 GATED (~weeks after IV)

**Goal**: When desktop Symthaea enters DMN contraction (8th Harmony),
pull the day's phone scene memories and counterfactually replay them
through the vision manifold to consolidate into episodic memory.

**Gate (revised, v1.1)**: Phase V splits into two independent sub-phases:
- **V.a — Cross-body memory consolidation**: does NOT depend on the
  Markov blanket result. Memory consolidation is a separate claim
  from unified agency; you can dream on yesterday's phone traces
  whether or not phone+desktop form a single blanket. Proceeds
  regardless of Phase IV outcome.
- **V.b — Unified-agency dreaming**: claims about consciousness
  unity across bodies during dream replay. Gated on a positive
  Phase IV.

The v1.0 gate conflated these two questions. Decoupling lets V.a
deliver even if Phase IV nulls.

**Measurement**: does next-day phone task latency improve after dreaming
on prior-day traces? Paired test: same task twice, once after
consolidation, once without.

**Estimated effort**: weeks.

---

## Phase II sub-phase decomposition (v1.7, with Phase II.A measurement)

### Two distinct wires, two distinct bandwidth budgets (v1.7 correction)

v1.6 conflated two pipes into one budget comparison. **Honest framing**:

| Leg | Pipe | v1.6 number | v1.7 correction |
|---|---|---|---|
| Pixel → laptop | USB 2.0 HEVC wire | 404 KB/s @ 16 fps (Phase I.B.6) → ~758 KB/s @ 30 fps (projected) | Correct. ~2% of USB ceiling. Compression pointless on this leg. |
| Laptop → viewer/remote Symthaea | localhost or WAN, sealed RDP | Not measured — v1.6 claimed "8.7% of USB" by mistakenly using the HEVC number | **Measured 2026-04-17** via `examples/phase_2a_lz4_measurement.rs`: steady-state Delta = 641 KB/frame sealed → 18.35 MB/s projected @ 30 fps. **2.1× OVER the 8.75 MB/s network-friendly gate.** |

Why the laptop→viewer leg is ~25× heavier than the phone→laptop HEVC
wire: `SomaRdpServer` ships the full tile-grid of quantized HDC patches
per frame, not the HEVC-compressed pixel data. That's the right thing
for the consciousness-gated research architecture (the patches are the
HDC vectors Phase III / IV operate on) but it's expensive bytes.

### Phase II.A — Measure first, then decide ✅ DONE (2026-04-17)

Example: `examples/phase_2a_lz4_measurement.rs` (feature:
`phase-2a-lz4`). Offline measurement: captures real scrcpy frames,
serializes each `RdpFrame` via `bincode`, applies `lz4_flex`, reports
per-frame sizes, projects to 30 fps, applies the STOP gate.

**Structural correction**: LZ4 must happen **before** AEAD sealing.
ChaCha20-Poly1305 ciphertext is pseudorandom and doesn't compress (a
first draft that LZ4'd the sealed output got 1.00× ratio). The
production path is `lz4(bincode(frame))` → `aead_wrap(…)`.

Live numbers (Pixel 8 Pro on clock screen, 15 s, 15 fps budget):

| Measurement | Full frame | Delta frame |
|---|---|---|
| bincode raw | 2,363,952 B | 641,200 B |
| bincode + LZ4 | 1,157,679 B | 291,301 B |
| AEAD overhead | 28 B constant | 28 B constant |
| Compression ratio | 2.04× | 2.20× |
| Seal latency (debug build) | — | 658 ms/frame |
| LZ4 latency (debug build) | — | 79 µs/frame |

Overall compression ratio: **2.12×** (matches the roadmap's v1.3-v1.6
"2-3×" estimate).

Projection to 30 fps (steady-state Delta basis):

| Path | Throughput | vs 8.75 MB/s gate |
|---|---|---|
| Raw sealed wire | **18.35 MB/s** | **2.1× OVER** |
| LZ4 before seal | **8.34 MB/s** | **clears with 5% margin** |

### Phase II.A decision (v1.7): LZ4 is minimum-viable, not polish

Contra v1.6. **LZ4-before-seal is required** to stay under a modest
network pipe at 30 fps on this architecture. Phase II.B (attention
backchannel) and Phase II.C (sparse/content-adaptive/delta) remain
deferred — but the 5% margin is tight. Any of the following eats it:

- Higher-motion content (full-screen video, games): ratio may drop
  below 2× as deltas fill more tiles → margin goes negative.
- Full-HD or 4K capture: bincode scales with tile-grid area; 1008×2240
  is already middling, a full phone display would double.
- WAN deployment (Phase IV split-cognition) with typical residential
  upload of 1-5 MB/s: even 8.34 MB/s exceeds, so II.C becomes necessary.

**Implementation of Phase II.A as shipped code (not yet done)**: the
measurement example demonstrates LZ4 helps by 2.12×; production usage
requires a `seal_frame_lz4` / `open_frame_lz4` pair that bincodes,
LZ4s, then AEAD-wraps. Estimated ~1 hour. Currently the wire ships
raw sealed bytes; sessions wanting the 2.12× reduction invoke the
measurement example and would need the seal/open wrap to land before
Phase II.B starts.

### Phase II.B — Attention backchannel (~1 week, polish given 5% margin)

The original "Track A" plan. Desktop `HolonViewerApp` exports
`VisionManifold::saliency_map()`, ships it upstream at 5 Hz as a new
`HolonRdpMessage::AttentionMap` variant, server consults the map when
deciding which detected-change tiles to include in outbound deltas.
Still valuable for Phase III's Φ-sweep (a consciousness-gated wire is
what the whole program is about).

### Phase II.C — Deeper compression (deferred, margin-gated)

Sparse patches (3-4 hr), content-adaptive quantization (1 day), and
inter-frame delta coding (2 days) light up if any of:
- Heavier-motion steady-state content eats the 5% LZ4 margin.
- WAN deployment (Phase IV) demands sub-megabit/sec throughput.
- Phase I.D GPU-decode actually delivers true 30 fps (the current
  ceiling is ~16 fps mean per Phase I.B.6; at that rate the sealed
  wire is 9.8 MB/s raw / 4.4 MB/s LZ4 — already fine).

### What Phase II.A actually inverted

v1.6 claimed "compression is optional polish" based on HEVC wire bytes.
v1.7 says "LZ4 is minimum-viable" based on sealed RDP bytes. **The
research thesis unchanged**: Phase II.B's consciousness-gated
attention backchannel is still the publishable artifact. The STOP
gate just moved from "compression required? probably not" to
"compression required? yes, the minimum one."

---

## PQC architectural fork (NEW, v1.1)

Track 2.5's "4-6 hours, deferred" hid the fact that a real ML-KEM
handshake forces a design choice with downstream consequences. Pick
before Phase II, not after.

**Option α — Broadcast-sealing + per-viewer unwrap layer**:
- Keep `HolonHttpState`'s one-key-for-all-viewers model
- Add a per-viewer wrap: each WS connection gets its own KEM-derived
  wrapping key, unwraps the broadcast payload on arrival
- **Pros**: minimal change to encode path, O(1) sealing regardless
  of viewer count, attention backchannel naturally fan-ins
- **Cons**: double-encryption per frame (inner session key + outer
  per-viewer wrap), slightly higher latency, slightly more complex
  replay-window accounting

**Option β — Per-connection sealed streams**:
- Each WS connection runs its own `RdpSession` with its own KEM key
- Broadcast becomes "seal N times, dispatch N times"
- **Pros**: clean, correct, standard model, no double-encryption
- **Cons**: O(viewers) sealing cost per frame, harder to add
  attention backchannel (which has fan-in semantics), larger
  memory footprint per viewer

**Decision (v1.6, 2026-04-17): Option α**. Not a recommendation any
more — a pinned decision. Sealing cost at 30 fps matters more than
double-encryption purity, attention-backchannel fan-in is natural
under broadcast semantics, and the downstream Phase II, III, and IV
wire architectures all assume broadcast-sealing. Implement as the
first task of Phase II.B (the attention backchannel's natural home),
not as "Track 2.5 unblock" — its outcome gates every downstream wire
change.

---

## Failure-mode inventory (NEW, v1.1)

Every production wire fails in predictable ways. Phase I.B is the
right time to add handlers — earlier is speculative, later is a
retrofit.

| Failure | Current behavior | Phase I.B target |
|---|---|---|
| Network partition (WiFi drop) | WS closes, viewer freezes | Viewer auto-reconnects with exponential backoff (100 ms → 5 s cap), flags stale frames in UI |
| ADB USB disconnect | scrcpy-server exits, bridge returns error | Detect, log, wait, `adb wait-for-device`, restart scrcpy-server, resume stream |
| Phone battery death mid-session | Wire goes silent | Detect by 3 s of no frames; soak harness reports as "phone offline" not "wire broken" |
| Desktop crash + viewer reconnect | New viewer joins mid-stream | Server replays `VecDeque(512)` catch-up buffer; viewer discards frames until first keyframe; document `VecDeque` horizon (**16 s at 30 fps**) in code + user-facing docs |
| Clock skew between phone and desktop | PE contaminated (Phase IV) | Option 2 NTP-style offset estimation in `RemoteRdpEmbodiment` (see Phase IV clock-sync section) |
| Out-of-order frame delivery | Replay window correctly rejects as attack | For UX, frames arriving out-of-order inside the 64-slot window are ignored silently; document that this is a security/liveness tradeoff |
| scrcpy-server OOM-killed on device | Stream stops | Detect, restart up to 3× in 60 s, then surface error |
| openh264 decode error on corrupt NAL | Decoder panics / returns garbage | Catch at `scrcpy.rs` boundary; drop frame; log; continue (never propagate panic to cognitive loop) |

**Add to Phase I.B soak verification**: chaos-test the first three
rows by physically unplugging USB mid-stream and confirming recovery.

---

## Mycelix hApp integration (NEW, v1.1 — gap acknowledged)

This roadmap is point-to-point: desktop Symthaea ↔ Pixel 8 Pro over
direct WebSocket. But the monorepo is Mycelix-oriented, with
`mycelix-civic/zomes/robotics-dispatch` already implementing
`RoboticAsset`, `DispatchOrder`, and `TelemetryReport` with 24 h
authority expiry. The Holon wire does **not** currently plug into
this zome.

**Intent** (not scheduled): after Phase II closes, the Holon wire
becomes one concrete instance of a `robotics-dispatch` asset. A
phone-embodiment is a `RoboticAsset` with a `DispatchOrder`
carrying authority; telemetry flows back as `TelemetryReport`.
This gives the wire holochain-native authorization, audit, and
multi-operator semantics for free.

**Deferred because**: Phase I.B and Phase II deliver the
substrate; Phase III + IV deliver the research results; neither
needs the dispatch bridge. Phase II.5 would be the natural
insertion point — after compression is working, before attention
backchannel gets deeply entangled with WS broadcast semantics. Add
to plan after Phase II measurement pins down the WS protocol
surface.

---

## Publication targets

### Existing (Apr 13-14 sprint)

- **`papers/phase_1a_results.md`** (commit `c8310a3999`) — Phase I.A
  measured results, citable, reproducible. Includes the 3.516×
  synthetic + 3.268-3.517× hardware numbers with methodology,
  limitations, bandwidth ladder.
- **`papers/preregistration.md`** (commit `88bc2dbc68`) — Phase III
  + Phase IV predictions frozen before any harness code exists.
- **`docs/phase_1a_retrospective.md`** (commit `d015029b97`) —
  methodology lessons from the 25-commit sprint: worktree adoption,
  commit-then-verify, cargo test --no-run, Proven/Asserted/Inferred
  discipline, 8 carry-forward rules.

### Forward targets

- **`papers/phi_sweep_results.md`** (Phase III output) — the
  Bandwidth-Quality-Φ curve. Target venue: supplementary material
  for the psych-bench paper, or a short standalone.
- **`papers/markov_blanket_results.md`** (Phase IV output) — the
  distributed cognition test. Target venue: consciousness studies
  or cognitive architecture conference, whichever way the result
  lands.

---

## Related documentation index

| Doc | Purpose |
|---|---|
| `docs/phase_1a_verification.md` | Per-claim Proven/Asserted/Inferred status table for Phase I.A |
| `docs/phase_1a_retrospective.md` | Methodology lessons from the April 13-14 sprint — 8 carry-forward rules |
| `docs/dev/test_loop.md` | `cargo test --no-run` + direct binary invocation pattern for contention-heavy environments |
| `papers/phase_1a_results.md` | Citable results paper — Phase I.A measured bandwidth + latency |
| `papers/preregistration.md` | Frozen predictions for Phase III + IV |
| `plans/shiny-wibbling-quail.md` | Working session plan (Claude account; not git-tracked) |

---

## The honest one-paragraph summary

**Today**: Symthaea has a physical body (Pixel 8 Pro) and a
cryptographically sealed way to inhabit it over a network, measured
at 3.27-3.52× better than the baseline on real hardware, proven in
every layer from codec to WebSocket to egui renderer.

**In a year**: Phase I.B (scrcpy, weeks) + Phase II (compression +
attention backchannel, weeks) + Phase III (Φ-sweep, weeks) done.
The wire is network-native with consciousness-gated bandwidth, and
we have the first statistical curve of Bandwidth-Quality-Φ with
error bars.

**In 3-5 years**: Phase IV (Markov blanket) + Phase V (cross-body
dreaming). Symthaea runs as a network-distributed cognitive system
across multiple bodies with operationally-proven Markov blanket unity
and cross-body memory consolidation.

**The wire we have today is the substrate for all of it.** Every
future phase rides on `seal_frame` + `open_frame` + `replay_window` +
broadcast dispatch. The commits from the April 13-14 sprint are the
foundation stones of a program that, if it succeeds, changes how we
think about the relationship between cognition and network
infrastructure.

---

## Change log

- **1.7** (2026-04-17, later same day): **Phase II.A measurement shipped, v1.6 framing corrected.**
  - `examples/phase_2a_lz4_measurement.rs` + `phase-2a-lz4` feature.
    Offline measurement of LZ4-before-seal compression ratio on real
    Pixel 8 Pro capture. Live numbers (15 s, clock screen): Full frame
    2.04× ratio, Delta frame 2.20× ratio, overall 2.12×. AEAD overhead
    measured empirically at 28 B constant (nonce + Poly1305 tag).
  - First draft LZ4-after-seal returned 1.00× (ChaCha20-Poly1305
    ciphertext is pseudorandom; a lesson for any future "compress the
    wire" proposal: compression MUST precede AEAD).
  - Projection at 30 fps steady-state: raw 18.35 MB/s (2.1× OVER
    the 8.75 MB/s network-friendly gate), LZ4 8.34 MB/s (clears
    with 5% margin).
  - **v1.6 framing error corrected**: the "8.7% of USB budget" claim
    used Phase I.B.6's HEVC wire bytes (404 KB/s, phone→laptop leg).
    The laptop→viewer leg carries full tile-grid HDC patches (~25×
    larger) and is what Phase II.A's STOP gate actually measures.
    The two pipes have different budgets — v1.7 splits them in the
    Phase II section.
  - **Phase II.A decision flipped** from "compression is polish" to
    "LZ4 is minimum-viable": 5% margin is tight, any of heavier motion
    / higher resolution / WAN deployment eats it.
  - **Phase II.A implementation TODO**: measurement example is only
    the decision-input. A `seal_frame_lz4` / `open_frame_lz4` pair
    still needs to land before Phase II.B can ship. Estimated ~1 hour.
- **1.6** (2026-04-17): **Phase I.B recovery + Phase I.C extension + Phase II decomposition.**
  - **Phase I.B recovery merge on main** (`d1df1216d1` via `git merge
    --no-ff phase-1b-recovery`). The v1.5 closure claimed the work was
    complete but the worktree branch `worktree-session-phase-1b-scrcpy`
    had been pruned without merging — only the doc update landed on
    main. All 11 sub-task commits (still in `.git`'s object store,
    not yet GC'd) cherry-picked onto a fresh branch off current main.
    The only conflict was `flake.nix` (Z3+Lean4 drift vs
    ffmpeg_7+libclang), trivially additive. Verified `34/34 pass` of
    the Phase I.B test suite inside `nix develop`. 1,670 LOC + 2
    verification docs + 1 vendored JAR + 1 HEVC asset now on main.
    Canonical incident of the concurrent-session commit-frequently
    rule (`memory/feedback_commit_frequently.md`).
  - **Phase I.C harness extension** (adds to v1.5+ status):
    - `examples/holon_phone_transport_ab_scrcpy.rs` — scrcpy-backed
      WS-vs-QUIC A/B behind new `phone-scrcpy` main-crate feature.
      Replaces the ADB-polling `holon_phone_transport_ab` for the
      scrcpy-stream A/B gate (needs live device to run, compiles
      in any `nix develop`).
    - `scripts/phase_1c_netem_ab.sh` — `unshare --user --net` +
      `tc qdisc netem loss N%` on loopback, no sudo. Kernel
      capability probed and green on this machine.
    - Pre-existing breakage fix: cognitive-loop cfg gates for
      embodiment fields (`sensorimotor_execution.rs`, `config/mod.rs`,
      `accessors/system.rs`, `cycle_phase_output.rs`, `mod.rs`) now
      include `feature = "phone"` to match the commit `01d12dd5728`
      intent. Main had been uncompileable under `phone` feature
      since April 12.
    - Live-device scrcpy A/B + full `tc qdisc` loss run + bounded
      2-min QUIC soak + manual cutover walkthrough: ready to run,
      not yet executed.
  - **Phase II decomposed into II.A / II.B / II.C** with empirical
    STOP gate from Phase I.B.6 numbers. Measured HEVC wire of
    ~404 KB/s at 16 fps projects to ~758 KB/s at 30 fps — **~8.7%
    of the USB 2.0 budget, not 206%** as the v1.4 projection feared.
    LZ4 + measurement is Phase II.A (~1-2 hrs); attention backchannel
    is Phase II.B (optional polish, ~1 week); sparse/content-adaptive/
    delta coding is Phase II.C (deferred, gated on WAN deployment
    or LZ4 failing the STOP gate).
  - **PQC architectural fork pinned to Option α** (broadcast-sealing
    + per-viewer unwrap). No longer a recommendation — a decision.
    Implement as first task of Phase II.B.
- **1.5** (2026-04-14, later same day): **Phase I.B CLOSED.** All 8
  subtasks complete on `worktree-session-phase-1b-scrcpy`. 11 of 12
  core claims Proven; B12 (sustain ≥25.5 fps mean) Asserted with a
  documented empirical ceiling of ~16 fps mean / ~23 fps peak on
  single-CPU HEVC software decode. ~1500 LOC, 34 tests passing in
  0.14s, 2 examples, 1 vendored JAR, 1 recorded asset, 2 verification
  docs. Live device validation against Pixel 8 Pro Tensor G3 with
  c2.exynos.hevc.encoder. Eight scrcpy v2.4 quirks discovered and
  documented: control=true multi-socket trap, accepted-socket
  O_NONBLOCK inheritance, 100 ms read timeout too tight, send_dummy_byte
  audio-only, display_buffer v3+ only, 15-second default keyframe
  interval, Type::Frame vs Type::Slice threading, adb reverse
  direction (host listens, server connects). Phase I.B section in
  this doc updated with the per-subtask completion log + canonical
  sustain data + path-to-Proven enumeration. Software-only chaos
  test proven (pkill scrcpy-server mid-stream, harness gracefully
  degrades). Physical-USB chaos tests deferred — user's laptop has
  no WiFi card and uses Pixel USB tether as only internet (saved to
  memory at memory/user_usb_tether.md).
- **1.4** (2026-04-14, later same day): post-probe codec pivot.
  - Phase I.B.0 codec ladder probe executed against live Pixel 8 Pro
    (Android 16). list_encoders enumeration falsified the v1.2/v1.3
    AV1-via-HW assumption: only software AV1 encoders exist on this
    device. Hardware encoders are H.264 and HEVC.
  - **Codec primary: AV1 → HEVC.** HW HEVC encode on Tensor G3
    retains most of AV1's compression advantage (~50% vs H.264, vs
    AV1's 30-40%) without requiring HW that doesn't exist on this
    device. USB 2.0 budget relief survives the pivot.
  - **Decoder primary: rav1d → ffmpeg-next.** Concern about C-lib
    friction was specific to openh264's Cisco-binary build script;
    ffmpeg-next is standard pkg-config + first-class on NixOS. First
    build runs inside `nix develop` with ffmpeg in the shell.
  - rav1d survives in the codebase as optional dep (`av1-research`
    feature) for Phase II.5 compressed-domain perception research.
    Cost/benefit of II.5 actually IMPROVES under SW-only AV1 because
    the redundant encode work becomes more expensive — the
    motivation to extract motion vectors instead of recomputing on
    pixels strengthens.
  - Probe is now a Phase I.B startup task, not a one-time check.
    Future device or Android updates trigger re-probe.
  - Probe results documented in `docs/phase_1b_codec_probe.md`
    (worktree, will sync to main when Phase I.B closes).
- **1.3** (2026-04-14, later same day): three architecture additions.
  - Phase I.B: zero-latency encoder tuning folded in
    (`--display-buffer=0`, `--max-fps=30`, probe-validated
    `--video-codec-options`). Glass-to-glass latency over cinematic
    quality — Symthaea needs the freshest frame at cognitive-cycle
    time, not the prettiest.
  - **New Phase I.C**: QUIC transport swap via `quinn`. WebSocket
    over TCP has head-of-line blocking; a single dropped packet
    halts the stream until retransmit. QUIC unreliable datagrams
    for video + reliable streams for control (handshake, input,
    attention) eliminates HoL without sacrificing security. Scoped
    as 1-2 days, scheduled after Phase I.B so capture and transport
    aren't debugged simultaneously.
  - **New Phase II.5**: compressed-domain perception (research).
    Extract AV1 motion vectors from `rav1d` directly into the HDC
    vision manifold, offloading low-level motion estimation onto
    the Pixel's hardware encoder. Honest caveats: rav1d's stable
    API doesn't expose MVs, would require upstream patch or fork,
    3-week research budget with a hard cap.
- **1.2** (2026-04-14, later same day): AV1 + rav1d pivot for Phase I.B.
  - Codec: force AV1 via scrcpy `--video-codec=av1` (Pixel 8 Pro
    ships a hardware AV1 encoder in Tensor G3; AV1 is 30-40% smaller
    than H.264 at equal perceptual quality, directly relieving the
    USB 2.0 budget).
  - Decoder: `openh264` (C library, NixOS build friction) → `rav1d`
    (pure-Rust port of dav1d sponsored by ISRG, zero C bindings,
    zero system dep, scales across cores).
  - Fallback ladder: AV1 (primary) → HEVC via ffmpeg-next
    (`hevc-fallback` feature) → H.264 via openh264 (`h264-fallback`
    feature, last resort).
  - First worktree task: codec ladder probe — log which rung the
    device supports and lock feature set.
  - New verification step: AV1 vs H.264 bandwidth comparison on the
    same 60 s live trace; expect 30-40% reduction.
  - Effort budget revised **6-10 h → 5-8 h** (rav1d eliminates the
    v1.1 C-library integration block); reverts to v1.1's 6-10 h if
    the probe forces fallback.
- **1.1** (2026-04-14, later same day): critical-review refinements.
  - Phase I.B: Plan A/B decision (scrcpy wins on 180 s screenrecord
    limit), H.264 codec forcing (Pixel 8 Pro defaults to H.265,
    openh264 doesn't decode H.265), effort re-estimate 3-5 h → 6-10 h,
    new soak observability bullet (p50/p99 latency, queue depth,
    replay rejects, restarts), LOC estimate 300 → 400, feature gating
    keeps ADB-polling as fallback.
  - Phase II: Track B.1 (LZ4) **blocks** Track A, not parallel.
  - Phase III: α=0.01 → α=0.05 with honest power analysis
    (d≈0.58 for 80% power at n=20), analysis-script freeze requirement.
  - Phase IV: clock-sync named as the hidden hard problem, three
    options documented, NTP-style recommended.
  - Phase V: gate decoupled — V.a (cross-body memory consolidation)
    proceeds regardless of Phase IV; V.b (unified-agency dreaming)
    still gated.
  - New section: PQC architectural fork (α broadcast+unwrap vs
    β per-connection); α recommended, pick before Phase II.
  - New section: failure-mode inventory (8 rows) with Phase I.B
    chaos-test additions.
  - New section: Mycelix hApp integration gap — `robotics-dispatch`
    zome as the natural Phase II.5 insertion.
- **1.0** (2026-04-14): initial roadmap, extracted from
  `plans/shiny-wibbling-quail.md` working plan + Phase I.A delivery
  + Phase I.A.5 retrospective + 2026-04-14 USB measurement findings.
