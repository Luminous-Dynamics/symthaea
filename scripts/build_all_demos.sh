#!/usr/bin/env bash
# Build all Symthaea demo artifacts from source.
#
# This script chains all demo generation steps in order:
#   1. Moral drone video (MuJoCo split-screen simulation)
#   2. Narrated version (LTC vocal tract synthesis + ffmpeg mux)
#   3. Consciousness dashboard video (cognitive loop telemetry)
#   4. Hero video (composited showcase with subtitles)
#   5. GIF preview (beam-release moment)
#
# Prerequisites:
#   - MUJOCO_DYNAMIC_LINK_DIR set (MuJoCo shared libs)
#   - EGL available (NVIDIA GPU for offscreen rendering)
#   - alsa-lib dev headers (for live-voice feature)
#   - ffmpeg
#
# Output (all in video_output/):
#   moral_drone.mp4              — Split-screen FEP vs PD simulation
#   moral_drone_narrated.mp4     — Narrated version with ambient audio
#   consciousness_dashboard.mp4  — Cognitive loop telemetry visualization
#   symthaea_hero.mp4            — Combined showcase (45s)
#   moral_drone_preview.gif      — GIF for README/social
#   narration_segments/          — Individual WAV segments
#
# Usage:
#   bash scripts/build_all_demos.sh
#
# Timing (approximate, release mode):
#   Step 1: ~2 min (first build) / ~30s (incremental)
#   Step 2: ~3 min (vocal tract training + synthesis)
#   Step 3: ~1 min (500 cognitive loop cycles)
#   Step 4: ~30s (ffmpeg compositing)
#   Step 5: ~5s (GIF extraction)

set -euo pipefail

cd "$(dirname "$0")/.."

echo "========================================"
echo "  Symthaea Demo Build Pipeline"
echo "========================================"
echo ""

# ── Verify environment ───────────────────────────────────────────
if [[ -z "${MUJOCO_DYNAMIC_LINK_DIR:-}" ]]; then
  echo "WARNING: MUJOCO_DYNAMIC_LINK_DIR not set."
  echo "  Step 1 (drone video) may fail without MuJoCo libs."
  echo ""
fi

mkdir -p video_output

# ── Step 1: Moral Drone Video ────────────────────────────────────
echo "Step 1/5: Moral Drone Video"
echo "─────────────────────────────"
if [[ -f "video_output/moral_drone.mp4" ]]; then
  echo "  SKIP: video_output/moral_drone.mp4 already exists"
  echo "  (delete it to regenerate)"
else
  cargo run --example moral_drone_video --features multirotor-mujoco-renderer --release
fi
echo ""

# ── Step 2: Narrated Version ─────────────────────────────────────
echo "Step 2/5: Narrated Version (LTC Vocal Tract)"
echo "─────────────────────────────────────────────"
if [[ -f "video_output/moral_drone_narrated.mp4" ]]; then
  echo "  SKIP: video_output/moral_drone_narrated.mp4 already exists"
  echo "  (delete it to regenerate)"
else
  cargo run --example narrate_moral_drone --features live-voice --release
fi
echo ""

# ── Step 3: Consciousness Dashboard ──────────────────────────────
echo "Step 3/5: Consciousness Dashboard Video"
echo "───────────────────────────────────────"
if [[ -f "video_output/consciousness_dashboard.mp4" ]]; then
  echo "  SKIP: video_output/consciousness_dashboard.mp4 already exists"
  echo "  (delete it to regenerate)"
else
  cargo run --example consciousness_dashboard_video --release
fi
echo ""

# ── Step 4: Hero Video ───────────────────────────────────────────
echo "Step 4/5: Hero Video (Composited Showcase)"
echo "──────────────────────────────────────────"
bash scripts/build_hero_video.sh
echo ""

# ── Step 5: GIF Preview ─────────────────────────────────────────
echo "Step 5/5: GIF Preview"
echo "─────────────────────"
bash scripts/extract_demo_gif.sh
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "========================================"
echo "  Build Complete"
echo "========================================"
echo ""
echo "Generated artifacts:"
for f in video_output/moral_drone.mp4 \
         video_output/moral_drone_narrated.mp4 \
         video_output/consciousness_dashboard.mp4 \
         video_output/symthaea_hero.mp4 \
         video_output/moral_drone_preview.gif; do
  if [[ -f "$f" ]]; then
    SIZE=$(stat -c%s "$f" 2>/dev/null || echo 0)
    SIZE_HUMAN=$(awk "BEGIN {
      if ($SIZE > 1048576) printf \"%.1f MB\", $SIZE/1048576;
      else printf \"%.0f KB\", $SIZE/1024
    }")
    printf "  %-45s %s\n" "$f" "$SIZE_HUMAN"
  else
    printf "  %-45s MISSING\n" "$f"
  fi
done
echo ""
echo "Demo page: docs/demos/moral_drone.html"
echo "  Open in browser or deploy to a web server."
