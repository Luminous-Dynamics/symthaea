# Phase I.C QUIC Transport Verification

Status: in progress
Date: 2026-04-15

This document records the first reproducible Phase I.C verification slice. It
does not close Phase I.C; the roadmap still requires a real scrcpy-stream A/B,
packet-loss testing, soak testing, and migration cutover validation.

## Implemented Surface

- `swarm::quic_transport` provides a Quinn/rustls QUIC transport for sealed
  Holon RDP frames.
- Outbound display frames move over QUIC unreliable datagrams with
  fragmentation/reassembly.
- Oversized sealed frames fall back to a reliable QUIC unidirectional stream.
  This is required for full-frame/bootstrap delivery: live phone full frames
  were large enough that datagram-only fragmentation timed out on localhost.
- Viewer-to-Holon input events move over a reliable QUIC bidirectional stream.
- `seal_frame` / `open_frame` remain unchanged; transport sees opaque sealed
  bytes only.
- `symthaea-holon` can bind both legacy HTTP/WebSocket and QUIC during the
  migration window.
- `holon_rdp_viewer` supports `--transport=ws|quic` for manual A/B.

## Headless Verification

Build command:

```bash
TMPDIR=/srv/luminous-dynamics/.phase1c-tmp \
CARGO_TARGET_DIR=/srv/luminous-dynamics/.phase1c-target \
RUSTFLAGS='-Awarnings' \
RUSTC_WRAPPER= \
SCCACHE_DISABLE=1 \
cargo build --no-default-features \
  --example holon_ws_smoke \
  --example holon_quic_smoke \
  --example holon_quic_loss_smoke \
  --example holon_transport_ab \
  --example holon_phone_transport_ab \
  --features holon-viewer,phone \
  --message-format short
```

Result:

```text
Finished `dev` profile [unoptimized + debuginfo] target(s) in 5.55s
```

WebSocket parity smoke:

```bash
/srv/luminous-dynamics/.phase1c-target/debug/examples/holon_ws_smoke
```

Result:

```text
WS smoke passed via ws://127.0.0.1:43019/holon/ws
```

Synthetic 30-frame localhost A/B:

```bash
/srv/luminous-dynamics/.phase1c-target/debug/examples/holon_transport_ab
```

Result:

```text
WS   samples=30 p50=27266us p99=32941us max=39954us
QUIC samples=30 p50=28613us p99=83328us max=88021us
Reverse input path OK for WS and QUIC
```

Interpretation: this headless synthetic localhost run proves functional parity
for sealed display-frame delivery and reverse sealed input delivery. Latency is
not stable enough in this harness to claim performance superiority: this sample
has comparable p50 but a worse QUIC p99/max. The next performance claim must
come from the real scrcpy-stream A/B plus packet-loss testing.

QUIC smoke regression after adding oversized-frame reliable fallback:

```bash
/srv/luminous-dynamics/.phase1c-target/debug/examples/holon_quic_smoke
```

Result:

```text
QUIC smoke passed via 127.0.0.1:34518 (connected (quic://127.0.0.1:34518))
```

Synthetic 10-frame regression after adding oversized-frame reliable fallback:

```bash
/srv/luminous-dynamics/.phase1c-target/debug/examples/holon_transport_ab --frames=10
```

Result:

```text
WS   samples=10 p50=31942us p99=33763us max=34768us
QUIC samples=10 p50=36452us p99=47082us max=55146us
Reverse input path OK for WS and QUIC
```

Deterministic QUIC datagram-loss smoke:

```bash
SYMTHAEA_QUIC_DROP_EVERY_N_DATAGRAM=3 \
  /srv/luminous-dynamics/.phase1c-target/debug/examples/holon_quic_loss_smoke
```

Result:

```text
QUIC loss smoke passed via 127.0.0.1:57867: received [1, 2, 4, 5, 7, 8] (SYMTHAEA_QUIC_DROP_EVERY_N_DATAGRAM=3)
```

Default no-loss regression:

```bash
/srv/luminous-dynamics/.phase1c-target/debug/examples/holon_quic_loss_smoke
```

Result:

```text
QUIC loss smoke passed via 127.0.0.1:48785: received [1, 2, 3, 4, 5, 6, 7, 8, 9] (SYMTHAEA_QUIC_DROP_EVERY_N_DATAGRAM=0)
```

Interpretation: the deterministic drop hook proves the QUIC datagram path has
the intended drop-and-continue behavior for datagram-sized display frames. It
does not replace the roadmap's `tc qdisc` packet-loss test because it does not
exercise kernel/network behavior or TCP/WebSocket head-of-line stalls.

## Live Phone-Content Checkpoint (ADB screenshot path)

The initial live-content checkpoint used the ADB screenshot path because
Phase I.B's scrcpy implementation had not yet been merged:

```text
PhoneBridge.capture_and_observe_rgba -> SomaRdpServer -> sealed RdpFrame replay -> WS/QUIC
```

Command:

```bash
/srv/luminous-dynamics/.phase1c-target/debug/examples/holon_phone_transport_ab \
  --duration 2 --fps 1 --serial 41201FDJG000UM
```

Result:

```text
Capturing live phone frames: serial=41201FDJG000UM duration=2s fps=1 nominal=1008x2244
captured frame=1 source=1008x2244 prediction_error=0.000
captured frame=2 source=1008x2244 prediction_error=0.008
Captured 2 RDP frame(s); replaying transports...
WS   samples=2 p50=1441246us p99=1441246us max=1480697us
QUIC samples=2 p50=1378447us p99=1378447us max=1716470us
Reverse input path OK for WS and QUIC
Source: live PhoneBridge screenshot capture, not scrcpy
```

Interpretation: this proves that real phone-derived `SomaRdpServer` frames,
including full-frame bootstrap payloads, now traverse both transports. The first
attempt failed on QUIC with a frame timeout; the reliable fallback for oversized
sealed frames fixed it. The latency numbers are dominated by huge screenshot
full-frame payloads and are not representative of the target scrcpy stream.

## scrcpy-Stream A/B (harness ready, live run pending)

Phase I.B merged to main on 2026-04-17, so the scrcpy-stream A/B is now
buildable. The harness lives at
`symthaea/examples/holon_phone_transport_ab_scrcpy.rs` and pulls real HEVC
frames through `StreamingPhoneBridge` from Phase I.B, converts them to
`RdpFrame`s via `SomaRdpServer`, then replays the same frame vector through
both the baseline WebSocket transport and the QUIC transport.

Gate: `--features holon-viewer,phone-scrcpy`. The `phone-scrcpy` main-crate
feature transitively enables `symthaea-phone-embodiment/scrcpy`, which pulls
in ffmpeg-next 7 + sha2, so the build must run inside `nix develop`
(LIBCLANG_PATH + BINDGEN_EXTRA_CLANG_ARGS come from the dev shell).

Build command:

```bash
nix develop ./symthaea --command cargo build \
  --no-default-features --features holon-viewer,phone-scrcpy \
  --example holon_phone_transport_ab_scrcpy
```

Run command (10-second bounded run; see Soak section for longer):

```bash
target/debug/examples/holon_phone_transport_ab_scrcpy \
  --duration 10 --fps 15 --serial 41201FDJG000UM --tcp-port 8401
```

Expected output shape: per-frame `captured frame=N source=WxH
prediction_error=...` lines during capture, then `WS samples=N p50=... p99=...`
and `QUIC samples=N p50=... p99=...` latency stats, then `Reverse input path
OK for WS and QUIC`.

### Live run (2026-04-17, Pixel 8 Pro, idle home screen)

Result (10 s, 15 fps budget, 1008×2240 native):

```text
Capturing scrcpy frames: serial=41201FDJG000UM duration=10s fps-budget=15 nominal=1008x2244 port=8401
captured frame=1 source=1008x2240 prediction_error=0.000
captured frame=2 source=1008x2240 prediction_error=0.086
captured frame=3 source=1008x2240 prediction_error=0.054
scrcpy stream: 3 device frames observed, 3 RDP frames produced
Captured 3 RDP frame(s); replaying transports...
WS   samples=3 p50=2136057us p99=2136057us max=2337146us
QUIC samples=3 p50=2407376us p99=2407376us max=2593081us
Reverse input path OK for WS and QUIC
Source: scrcpy persistent HEVC capture via StreamingPhoneBridge
```

**End-to-end works**: scrcpy-server launched, 3 device frames decoded,
3 `SomaRdpServer` RDP frames emitted, same frame vector replayed through
both WS and QUIC, reverse input path validated for both.

**Latency caveat (same as the ADB-path checkpoint above)**: these
numbers are **dominated by `Full`-frame bootstrap payloads**, not by
steady-state delta traffic. Idle home-screen capture produced 0.3 fps
(3 frames in 10 s — the HDC tile-change detector only emits RDP
frames on change, and a static screen barely changes). Each of those
3 frames was a fresh `Full` frame (~500 KB at 1008×2240 RGBA) rather
than a small delta. The 2.1-2.6 s transport time is the
fragmentation-over-localhost budget for sub-MB sealed envelopes, not
the 30-fps-delta-stream latency Phase III will actually measure.

A fair steady-state number requires either (a) running longer with
motion (scrolling, video playback) so the encoder emits deltas after
the bootstrap, or (b) reducing `SomaRdpServer`'s `Full`-emission
cadence. Both are Phase II.A measurement work, not Phase I.C closeout.

Startup note: the default `--tcp-port 8408` collided with a local
Python process on this machine; `--tcp-port 8401` worked. The
`holon_phone_transport_ab_scrcpy` help text lists the flag. Dev-test
range 8400-8409 per `.claude/rules/PORTS.md`; pick any free port.

## tc qdisc packet-loss A/B (scripted, user-namespace based)

Script: `symthaea/scripts/phase_1c_netem_ab.sh`. Uses Linux unprivileged
user+net namespaces (no sudo) to apply `tc qdisc ... netem loss N%` to
loopback inside an isolated namespace, then runs the existing synthetic
`holon_transport_ab` example under that loss.

Kernel requirement probed and OK on this machine:

```bash
$ unshare --user --net --map-root-user /usr/bin/env true && echo OK
OK
$ unshare --user --net --map-root-user -- sh -c \
    'ip link set lo up && tc qdisc add dev lo root netem loss 1% && tc qdisc show dev lo'
qdisc netem 8001: root refcnt 2 limit 1000 loss 1% seed ...
```

Run:

```bash
cargo build --no-default-features --features holon-viewer \
  --example holon_transport_ab
LOSS=1 FRAMES=60 symthaea/scripts/phase_1c_netem_ab.sh
```

Results (2026-04-17, three loss levels, same machine):

| LOSS | FRAMES | WS p50 | WS p99 | WS max | QUIC p50 | QUIC p99 | QUIC max |
|---|---|---|---|---|---|---|---|
| 0% (control) | 30 | 43 ms | 52 ms | 60 ms | 59 ms | 71 ms | 74 ms |
| 1% | 60 | 48 ms | 93 ms | **280 ms** | 63 ms | 92 ms | **97 ms** |
| 5% | 60 | 59 ms | **277 ms** | **314 ms** | 59 ms | 97 ms | **115 ms** |

**Head-of-line blocking signal at 1% loss**: WS max inflates from 60 ms
→ 280 ms (4.7× worse), while QUIC max moves 74 ms → 97 ms (1.3× worse).
At 5% loss the WS tail explodes further (p99=277 ms, max=314 ms — 5.3×
and 5.2× vs control) while QUIC stays within 2× (p99=97 ms,
max=115 ms). Exact match to the Phase I.C design thesis: QUIC
unreliable datagrams drop-and-continue; TCP retransmit stalls the
entire stream until recovery.

The WS p50 also degrades more gracefully than the tail — at 1% loss
WS p50 is 48 ms vs QUIC's 63 ms, so the *median* frame looks fine on
WS. The distinction is entirely in the distribution tail. A
cognitive-loop consumer sampling at the 30 fps frame rate would see
the occasional 280 ms WS stall as a visible hitch; QUIC's 97 ms max
stays under a 3-frame window even at p99+.

## Remaining Phase I.C Verification

- [x] **Execute the scrcpy-stream A/B live against the Pixel 8 Pro and
      record numbers for this document.** Done 2026-04-17; numbers in
      the scrcpy-stream A/B section above. End-to-end path works; the
      captured latency is bootstrap-dominated, not steady-state.
      Steady-state measurement deferred to Phase II.A.
- [x] **Execute `phase_1c_netem_ab.sh` for LOSS ∈ {0, 1, 5} and record the
      WS-vs-QUIC stall behavior.** Done 2026-04-17; numbers in the tc qdisc
      section above. WS tail inflates 4.7-5.3× at 1%/5% loss; QUIC stays
      within 1.3-2×. Head-of-line blocking thesis confirmed on this
      machine.
- [ ] Run a **bounded 2-minute soak** over QUIC with Phase I.B observability
      metrics (sealed bytes/s, seal+open p50/p99, queue depth, replay-reject
      count, restarts, decoded/dropped). The 10-minute full soak is deferred
      out of respect for the user's USB tether budget (see
      `memory/user_usb_tether.md`).
- [ ] Manual migration cutover walkthrough: run `holon_rdp_viewer
      --transport=ws`, then the same with `--transport=quic`, against the
      same phone session. Confirm the viewer renders in both cases. This is
      largely proven functionally by the A/B harness but not yet by a
      human-observable cutover.
