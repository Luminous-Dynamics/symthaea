# Holon-Soma Roadmap

**Version**: 1.0 (2026-04-14)
**Scope**: The PQC-sealed binary RDP wire connecting a Symthaea cognitive
loop to a physical embodiment (starting with the Pixel 8 Pro), and the
research program that rides on top of it.
**Status**: Phase I.A delivered and hardware-validated. Phase I.A.5
hardening interlude complete (9 of 10 tracks). Phase I.A.2 Pieces 1+2
code-complete. Forward phases (I.B → II → III → IV → V) planned and
pre-registered where applicable.

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

### Phase I.B — scrcpy persistent capture 📋 NEXT (~3-5 hours)

**Goal**: Remove the 250 ms ADB polling ceiling. Unlock 30-60 fps
capture with ~30 ms latency. Generate the first real-fps data for
every downstream phase.

**Approach**: Push `scrcpy-server.jar` via `adb push`, run it as a
shell process via `app_process`, open an ADB reverse tunnel for the
H.264 NAL output stream, decode via `openh264` (Cisco BSD-2, small,
no system dep).

**New files**:
- `crates/symthaea-phone-embodiment/src/scrcpy.rs` (~300 LOC)
- `crates/symthaea-phone-embodiment/vendor/scrcpy-server.jar` + `.sha256`

**Modified files**:
- `crates/symthaea-phone-embodiment/src/adb.rs` — `push_scrcpy_server`, `start_scrcpy`
- `crates/symthaea-phone-embodiment/src/bridge.rs` — `capture_stream`, `tick_rgba`
- `examples/phone_rdp_share.rs` — swap `capture_and_observe_rgba` for the persistent stream path

**New dep**: `openh264 = "0.6"` (prefer over `ffmpeg-next` — smaller, no system dep)

**Verification**:
1. Unit tests with recorded NAL asset
2. `phone_rdp_share --features scrcpy` sustains 30 fps for 60 s
3. 10-minute soak: no memory growth, no scrcpy-server crash, <0.1% dropped

**Estimated effort**: 3-5 hours in a fresh worktree.

**Why this phase unblocks everything**: without real fps, all bandwidth
measurements are synthetic or polling-limited. Phase II can't test
attention gating meaningfully at 4 fps. Phase III's Φ-sweep needs 20+
trials per condition which at 4 fps takes all day. Phase IV's Markov
blanket test needs stable-state PE which requires the codec to be
running at its design rate.

### Phase II — Attention backchannel + codec compression 📋 (~1 week after I.B)

**Goal**: Desktop Symthaea's saliency map flows backward through the
WS and modulates which tiles the phone codec prioritizes. Combined with
the codec compression ladder, fits 30 fps heavy-activity content within
~25% of the USB 2.0 budget.

**Two parallel tracks**:

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
- **Test**: paired t-test, α=0.01, effect size ≥ 0.5
- **All four outcomes publishable** — confirm direction + magnitude / direction only / null / wrong direction

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

**All three outcomes publishable**:
- Confirmed: operational proof of distributed cognition across a network
- Null: coupled/decoupled indistinguishable; Markov blanket framing falsified
- Wrong direction: coupling harms both instances; design revisit

**Estimated effort**: 2-4 weeks.

### Phase V — Cross-body dream consolidation 🔒 GATED (~weeks after IV)

**Goal**: When desktop Symthaea enters DMN contraction (8th Harmony),
pull the day's phone scene memories and counterfactually replay them
through the vision manifold to consolidate into episodic memory.

**Gate**: Only proceeds if Phase IV produces a positive Markov blanket
result. A null Phase IV cancels Phase V.

**Measurement**: does next-day phone task latency improve after dreaming
on prior-day traces? Paired test: same task twice, once after
consolidation, once without.

**Estimated effort**: weeks.

---

## Revised Phase II sequencing (from 2026-04-14 USB measurement)

Before the USB ceiling measurement, Phase II was planned as "attention
backchannel" with codec compression as optional polish. The measurement
revealed that heavy-activity 30 fps content would exceed USB 2.0 budget
without compression (72 MB/s projected vs 35 MB/s ceiling). This
changes the sequencing:

1. **Phase I.B first** (scrcpy at real fps — prerequisite for meaningful measurement)
2. Measure real-content sealed per-frame size with scrcpy
3. **LZ4 wrap before sealing** (1 hour, zero new deps, 2-3× reduction)
4. Measure again — if under 25% USB budget, SKIP step 5
5. Sparse patch encoding (3-4 hours, 5-10× additional)
6. Attention backchannel (the original Phase II task)
7. Content-adaptive quantization (only if still over budget)
8. Inter-frame delta coding (Phase II-plus, if we want serious compression)

**The key insight**: compression was optional polish before the USB
measurement. It's now required to hit Phase II's 30 fps target on USB
2.0 without saturating the bus (and thereby degrading the user's
laptop internet via RNDIS tethering contention).

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

- **1.0** (2026-04-14): initial roadmap, extracted from
  `plans/shiny-wibbling-quail.md` working plan + Phase I.A delivery
  + Phase I.A.5 retrospective + 2026-04-14 USB measurement findings.
