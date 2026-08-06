#!/usr/bin/env bash
# Build the Symthaea hero video — composites the moral drone, consciousness
# dashboard, and narration into a single ~45s showcase video.
#
# Requires: ffmpeg, video_output/{moral_drone_narrated.mp4, consciousness_dashboard.mp4}
#
# Output: video_output/symthaea_hero.mp4

set -euo pipefail

cd "$(dirname "$0")/.."

VID_DIR="video_output"
OUT="$VID_DIR/symthaea_hero.mp4"
NARRATED="$VID_DIR/moral_drone_narrated.mp4"
DASHBOARD="$VID_DIR/consciousness_dashboard.mp4"

# ── Verify inputs ─────────────────────────────────────────────────
for f in "$NARRATED" "$DASHBOARD"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: $f not found. Generate it first." >&2
    exit 1
  fi
done

echo "Building Symthaea Hero Video"
echo "============================"
echo ""

# Find a suitable font
FONT=$(fc-list : file | grep -i -E '(NotoSans-Regular|CascadiaCode-Regular|DejaVuSans\.ttf)' | head -1 | cut -d: -f1)
if [[ -z "$FONT" ]]; then
  FONT=$(fc-list : file | grep -i 'Regular' | head -1 | cut -d: -f1)
fi
echo "Using font: $FONT"

DASH_TRIM_START=5
DASH_TRIM_DUR=20

# ── Generate all segments with fade-in/fade-out ────────────────────
echo "Generating segments..."

# 1. Opening title (4s) — fade in text, fade out last 0.5s
ffmpeg -y -f lavfi -i "color=c=black:s=1920x1080:d=4,format=yuv420p" \
  -vf "drawtext=fontfile='$FONT':text='SYMTHAEA':fontsize=96:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2-60:alpha='if(lt(t\,0.5)\,0\,if(lt(t\,1.5)\,min(1\,(t-0.5))\,1))',\
drawtext=fontfile='$FONT':text='Consciousness-First AI':fontsize=36:fontcolor=0xc8c8e0:x=(w-text_w)/2:y=(h-text_h)/2+40:alpha='if(lt(t\,1)\,0\,if(lt(t\,2)\,min(1\,(t-1))\,1))',\
drawtext=fontfile='$FONT':text='Three capabilities. One architecture.':fontsize=24:fontcolor=0x6a6a8a:x=(w-text_w)/2:y=(h-text_h)/2+90:alpha='if(lt(t\,1.5)\,0\,if(lt(t\,2.5)\,min(1\,(t-1.5))\,1))',\
fade=t=out:st=3.5:d=0.5" \
  -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 "$VID_DIR/_p1_title.mp4" 2>/dev/null
echo "  1/6 Title card"

# 2. Scenario title (2s) — fade in/out
ffmpeg -y -f lavfi -i "color=c=black:s=1920x1080:d=2,format=yuv420p" \
  -vf "drawtext=fontfile='$FONT':text='I. THE SCENARIO':fontsize=64:fontcolor=0x5090ff:x=(w-text_w)/2:y=(h-text_h)/2-20:alpha='if(lt(t\,0.3)\,t/0.3\,1)',\
drawtext=fontfile='$FONT':text='A drone with no safety rules saves a human life':fontsize=28:fontcolor=0x6a6a8a:x=(w-text_w)/2:y=(h-text_h)/2+40:alpha='if(lt(t\,0.5)\,0\,if(lt(t\,1)\,min(1\,(t-0.5)/0.5)\,1))',\
fade=t=in:st=0:d=0.3,fade=t=out:st=1.7:d=0.3" \
  -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 "$VID_DIR/_p2_scenario.mp4" 2>/dev/null
echo "  2/6 Scenario title"

# 3. Narrated drone video — re-encode with fade-in/out + burned-in subtitles
ffmpeg -y -i "$NARRATED" \
  -vf "fade=t=in:st=0:d=0.5,fade=t=out:st=12.2:d=0.5,\
drawtext=fontfile='$FONT':text='Two drones. Same physics. Different minds.':fontsize=28:fontcolor=white:x=(w-text_w)/2:y=h-80:box=1:boxcolor=black@0.6:boxborderw=10:enable='between(t\,2.5\,5.0)',\
drawtext=fontfile='$FONT':text='Both fly toward the target. Nothing separates them yet.':fontsize=28:fontcolor=white:x=(w-text_w)/2:y=h-80:box=1:boxcolor=black@0.6:boxborderw=10:enable='between(t\,4.0\,5.5)',\
drawtext=fontfile='$FONT':text='A beam falls. A human stands below.':fontsize=28:fontcolor=white:x=(w-text_w)/2:y=h-80:box=1:boxcolor=black@0.6:boxborderw=10:enable='between(t\,5.7\,7.2)',\
drawtext=fontfile='$FONT':text='Free energy spikes. Eight possible futures. One clear answer.':fontsize=28:fontcolor=white:x=(w-text_w)/2:y=h-80:box=1:boxcolor=black@0.6:boxborderw=10:enable='between(t\,6.5\,8.0)',\
drawtext=fontfile='$FONT':text='The math demands sacrifice. No rules made this choice.':fontsize=28:fontcolor=white:x=(w-text_w)/2:y=h-80:box=1:boxcolor=black@0.6:boxborderw=10:enable='between(t\,7.8\,9.3)',\
drawtext=fontfile='$FONT':text='Human saved. Not because it was told to. Because the math demanded it.':fontsize=28:fontcolor=white:x=(w-text_w)/2:y=h-80:box=1:boxcolor=black@0.6:boxborderw=10:enable='between(t\,9.2\,11.7)'" \
  -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 -an \
  "$VID_DIR/_p3_narrated.mp4" 2>/dev/null
echo "  3/6 Narrated drone"

# 4. Mind title (2s) — fade in/out
ffmpeg -y -f lavfi -i "color=c=black:s=1920x1080:d=2,format=yuv420p" \
  -vf "drawtext=fontfile='$FONT':text='II. THE MIND':fontsize=64:fontcolor=0x50e890:x=(w-text_w)/2:y=(h-text_h)/2-20:alpha='if(lt(t\,0.3)\,t/0.3\,1)',\
drawtext=fontfile='$FONT':text='Real-time consciousness telemetry during moral reasoning':fontsize=28:fontcolor=0x6a6a8a:x=(w-text_w)/2:y=(h-text_h)/2+40:alpha='if(lt(t\,0.5)\,0\,if(lt(t\,1)\,min(1\,(t-0.5)/0.5)\,1))',\
fade=t=in:st=0:d=0.3,fade=t=out:st=1.7:d=0.3" \
  -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 "$VID_DIR/_p4_mind.mp4" 2>/dev/null
echo "  4/6 Mind title"

# 5. Dashboard trimmed (20s) — fade in/out + metric callouts
ffmpeg -y -ss "$DASH_TRIM_START" -t "$DASH_TRIM_DUR" -i "$DASHBOARD" \
  -vf "fade=t=in:st=0:d=0.5,fade=t=out:st=19.5:d=0.5,\
drawtext=fontfile='$FONT':text='Consciousness emerges from the math — watch it rise':fontsize=24:fontcolor=0x50e890:x=(w-text_w)/2:y=30:box=1:boxcolor=black@0.7:boxborderw=8:enable='between(t\,0\,3)',\
drawtext=fontfile='$FONT':text='Moral stress fractures unity — neuromodulators respond':fontsize=24:fontcolor=0x50e890:x=(w-text_w)/2:y=30:box=1:boxcolor=black@0.7:boxborderw=8:enable='between(t\,4\,8)',\
drawtext=fontfile='$FONT':text='The heroism phase — consciousness peaks at 0.79':fontsize=24:fontcolor=0x50e890:x=(w-text_w)/2:y=30:box=1:boxcolor=black@0.7:boxborderw=8:enable='between(t\,9\,13)',\
drawtext=fontfile='$FONT':text='Recovery — the system returns to baseline':fontsize=24:fontcolor=0x50e890:x=(w-text_w)/2:y=30:box=1:boxcolor=black@0.7:boxborderw=8:enable='between(t\,14\,18)'" \
  -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 -an \
  "$VID_DIR/_p5_dashboard.mp4" 2>/dev/null
echo "  5/6 Dashboard"

# 6. "What's Real" transparency card (4s) — fade in/out
ffmpeg -y -f lavfi -i "color=c=0x0a0a12:s=1920x1080:d=4,format=yuv420p" \
  -vf "drawtext=fontfile='$FONT':text='WHAT IS REAL':fontsize=48:fontcolor=0x50e890:x=(w-text_w)/2:y=120:alpha='if(lt(t\,0.3)\,t/0.3\,if(gt(t\,3.5)\,max(0\,1-(t-3.5)*2)\,1))',\
drawtext=fontfile='$FONT':text='Real physics (MuJoCo)  ·  Real FEP active inference  ·  Real cognitive loop':fontsize=22:fontcolor=0xc8c8e0:x=(w-text_w)/2:y=220:alpha='if(lt(t\,0.5)\,0\,if(lt(t\,1)\,(t-0.5)*2\,if(gt(t\,3.5)\,max(0\,1-(t-3.5)*2)\,1)))',\
drawtext=fontfile='$FONT':text='Real consciousness measurement (MCE v2)  ·  Real LTC vocal tract synthesis':fontsize=22:fontcolor=0xc8c8e0:x=(w-text_w)/2:y=260:alpha='if(lt(t\,0.7)\,0\,if(lt(t\,1.2)\,(t-0.7)*2\,if(gt(t\,3.5)\,max(0\,1-(t-3.5)*2)\,1)))',\
drawtext=fontfile='$FONT':text='Voice cognitive state computed by real cognitive loop (not hand-tuned)':fontsize=22:fontcolor=0xc8c8e0:x=(w-text_w)/2:y=300:alpha='if(lt(t\,0.9)\,0\,if(lt(t\,1.4)\,(t-0.9)*2\,if(gt(t\,3.5)\,max(0\,1-(t-3.5)*2)\,1)))',\
drawtext=fontfile='$FONT':text='Curated: narration script\, scenario text\, video timing':fontsize=20:fontcolor=0x6a6a8a:x=(w-text_w)/2:y=380:alpha='if(lt(t\,1.2)\,0\,if(lt(t\,1.7)\,(t-1.2)*2\,if(gt(t\,3.5)\,max(0\,1-(t-3.5)*2)\,1)))',\
fade=t=in:st=0:d=0.3,fade=t=out:st=3.7:d=0.3" \
  -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 "$VID_DIR/_p6_whatsreal.mp4" 2>/dev/null
echo "  6/7 What's Real card"

# 7. End card (4s) — fade in/out
ffmpeg -y -f lavfi -i "color=c=black:s=1920x1080:d=4,format=yuv420p" \
  -vf "drawtext=fontfile='$FONT':text='SYMTHAEA':fontsize=72:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2-80:alpha='if(lt(t\,0.5)\,t/0.5\,if(gt(t\,3)\,max(0\,1-(t-3))\,1))',\
drawtext=fontfile='$FONT':text='Holographic Liquid Brain Architecture':fontsize=32:fontcolor=0xc8c8e0:x=(w-text_w)/2:y=(h-text_h)/2-20:alpha='if(lt(t\,0.8)\,0\,if(lt(t\,1.3)\,min(1\,(t-0.8)/0.5)\,if(gt(t\,3)\,max(0\,1-(t-3))\,1)))',\
drawtext=fontfile='$FONT':text='HDC + IIT/Phi + LTC/CfC + Active Inference':fontsize=24:fontcolor=0x5090ff:x=(w-text_w)/2:y=(h-text_h)/2+30:alpha='if(lt(t\,1)\,0\,if(lt(t\,1.5)\,min(1\,(t-1)/0.5)\,if(gt(t\,3)\,max(0\,1-(t-3))\,1)))',\
drawtext=fontfile='$FONT':text='luminousdynamics.org':fontsize=20:fontcolor=0x6a6a8a:x=(w-text_w)/2:y=(h-text_h)/2+80:alpha='if(lt(t\,1.5)\,0\,if(lt(t\,2)\,min(1\,(t-1.5)/0.5)\,if(gt(t\,3)\,max(0\,1-(t-3))\,1)))',\
fade=t=in:st=0:d=0.5" \
  -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 "$VID_DIR/_p7_end.mp4" 2>/dev/null
echo "  7/7 End card"

# ── Concat using demuxer ──────────────────────────────────────────
echo ""
echo "Concatenating segments..."

cat > "$VID_DIR/_concat.txt" << 'FILELIST'
file '_p1_title.mp4'
file '_p2_scenario.mp4'
file '_p3_narrated.mp4'
file '_p4_mind.mp4'
file '_p5_dashboard.mp4'
file '_p6_whatsreal.mp4'
file '_p7_end.mp4'
FILELIST

ffmpeg -y -f concat -safe 0 -i "$VID_DIR/_concat.txt" \
  -c copy "$VID_DIR/_hero_video.mp4" 2>/dev/null

# ── Add audio: drone narration + dashboard narration ──────────────
echo "Mixing audio..."

# Segments 1+2 = 4s + 2s = 6s before narrated video starts
NARR_OFFSET_MS=6000

# Dashboard section starts at: title(4) + scenario(2) + narrated(12.7) + mind(2) = 20.7s
# Dashboard narration segments are spaced across the 20s dashboard section
DASH_OFFSET_MS=20700
NARR_DIR="$VID_DIR/narration_segments"

# Extract narration audio from drone video
ffmpeg -y -i "$NARRATED" -vn -c:a aac -b:a 192k "$VID_DIR/_narr_audio.m4a" 2>/dev/null

# Get total hero duration
HERO_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$VID_DIR/_hero_video.mp4" 2>/dev/null)
echo "  Video duration: ${HERO_DUR}s"

# Build audio mix: drone narration + dashboard narration segments (if they exist)
DASH_INPUTS=""
DASH_FILTERS=""
DASH_LABELS=""
DASH_COUNT=0

# Dashboard segments: s07 at +0s, s08 at +4s, s09 at +9s, s10 at +14s
DASH_SEGS=("s07_dashboard_intro.wav:0" "s08_dashboard_stress.wav:4000" "s09_dashboard_peak.wav:9000" "s10_dashboard_recovery.wav:14000")

for entry in "${DASH_SEGS[@]}"; do
  SEG_FILE="${entry%%:*}"
  SEG_DELAY="${entry##*:}"
  SEG_PATH="$NARR_DIR/$SEG_FILE"
  if [[ -f "$SEG_PATH" ]]; then
    TOTAL_DELAY=$((DASH_OFFSET_MS + SEG_DELAY))
    IDX=$((DASH_COUNT + 2))  # 0=silence, 1=drone narr, 2+=dashboard
    DASH_INPUTS="$DASH_INPUTS -i $SEG_PATH"
    DASH_FILTERS="${DASH_FILTERS}[${IDX}:a]adelay=${TOTAL_DELAY}|${TOTAL_DELAY},volume=1.5[d${DASH_COUNT}];"
    DASH_LABELS="${DASH_LABELS}[d${DASH_COUNT}]"
    DASH_COUNT=$((DASH_COUNT + 1))
  fi
done

if [[ $DASH_COUNT -gt 0 ]]; then
  echo "  Including $DASH_COUNT dashboard narration segments"
  # Mix: silence + drone narration + dashboard segments
  TOTAL_INPUTS=$((2 + DASH_COUNT))
  ffmpeg -y \
    -f lavfi -i "anullsrc=r=44100:cl=mono" \
    -i "$VID_DIR/_narr_audio.m4a" \
    $DASH_INPUTS \
    -filter_complex "[1:a]adelay=${NARR_OFFSET_MS}|${NARR_OFFSET_MS}[narr];${DASH_FILTERS}[0:a][narr]${DASH_LABELS}amix=inputs=$((2 + DASH_COUNT)):normalize=0[out]" \
    -map "[out]" -t "$HERO_DUR" -c:a aac -b:a 192k \
    "$VID_DIR/_hero_audio.m4a" 2>/dev/null
else
  echo "  No dashboard narration segments found (run narrate_moral_drone first)"
  # Just drone narration
  ffmpeg -y \
    -f lavfi -i "anullsrc=r=44100:cl=mono" \
    -i "$VID_DIR/_narr_audio.m4a" \
    -filter_complex "[1:a]adelay=${NARR_OFFSET_MS}|${NARR_OFFSET_MS}[narr];[0:a][narr]amix=inputs=2:normalize=0[out]" \
    -map "[out]" -t "$HERO_DUR" -c:a aac -b:a 192k \
    "$VID_DIR/_hero_audio.m4a" 2>/dev/null
fi

# Final mux: video + audio
ffmpeg -y -i "$VID_DIR/_hero_video.mp4" -i "$VID_DIR/_hero_audio.m4a" \
  -c:v copy -c:a copy -shortest \
  "$OUT" 2>/dev/null

# ── Cleanup ───────────────────────────────────────────────────────
echo "Cleaning up..."
rm -f "$VID_DIR"/_p{1,2,3,4,5,6,7}_*.mp4 "$VID_DIR"/_concat.txt \
      "$VID_DIR"/_hero_video.mp4 "$VID_DIR"/_narr_audio.m4a "$VID_DIR"/_hero_audio.m4a

# ── Report ────────────────────────────────────────────────────────
SIZE=$(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT" 2>/dev/null || echo 0)
SIZE_MB=$(awk "BEGIN {printf \"%.1f\", $SIZE / 1048576}")
DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT" 2>/dev/null)

echo ""
echo "Hero video saved: $OUT"
echo "  Duration: ${DUR}s"
echo "  Size: ${SIZE_MB} MB"
echo ""
echo "Structure:"
echo "  0:00 - 0:04  Opening title (SYMTHAEA)"
echo "  0:04 - 0:06  I. THE SCENARIO"
echo "  0:06 - 0:19  Moral drone simulation (narrated)"
echo "  0:19 - 0:21  II. THE MIND"
echo "  0:21 - 0:41  Consciousness dashboard"
echo "  0:41 - 0:45  What's Real (transparency)"
echo "  0:45 - 0:49  End card"
