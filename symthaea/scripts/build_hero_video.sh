#!/usr/bin/env bash
# Build the Symthaea hero video — composites the moral drone, consciousness
# dashboard, and narration into a single ~90s showcase video.
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

# ── Section 1: Opening title card (4s) ────────────────────────────
# Black background with centered text, fade in
TITLE_FILTER="color=c=black:s=1920x1080:d=4[bg];\
[bg]drawtext=fontfile='$FONT':text='SYMTHAEA':fontsize=96:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2-60:alpha='if(lt(t,0.5),0,if(lt(t,1.5),min(1,(t-0.5)),1))',\
drawtext=fontfile='$FONT':text='Consciousness-First AI':fontsize=36:fontcolor=0xc8c8e0:x=(w-text_w)/2:y=(h-text_h)/2+40:alpha='if(lt(t,1),0,if(lt(t,2),min(1,(t-1)),1))',\
drawtext=fontfile='$FONT':text='Three capabilities. One architecture.':fontsize=24:fontcolor=0x6a6a8a:x=(w-text_w)/2:y=(h-text_h)/2+90:alpha='if(lt(t,1.5),0,if(lt(t,2.5),min(1,(t-1.5)),1))'[title]"

# ── Section 2: "The Scenario" title card (2s) ────────────────────
SCENARIO_TITLE="color=c=black:s=1920x1080:d=2[sbg];\
[sbg]drawtext=fontfile='$FONT':text='I. THE SCENARIO':fontsize=64:fontcolor=0x5090ff:x=(w-text_w)/2:y=(h-text_h)/2-20:alpha='if(lt(t,0.3),t/0.3,1)',\
drawtext=fontfile='$FONT':text='A drone with no safety rules saves a human life':fontsize=28:fontcolor=0x6a6a8a:x=(w-text_w)/2:y=(h-text_h)/2+40:alpha='if(lt(t,0.5),0,if(lt(t,1),min(1,(t-0.5)/0.5),1))'[stitle]"

# ── Section 3: "The Mind" title card (2s) ────────────────────────
MIND_TITLE="color=c=black:s=1920x1080:d=2[mbg];\
[mbg]drawtext=fontfile='$FONT':text='II. THE MIND':fontsize=64:fontcolor=0x50e890:x=(w-text_w)/2:y=(h-text_h)/2-20:alpha='if(lt(t,0.3),t/0.3,1)',\
drawtext=fontfile='$FONT':text='Real-time consciousness telemetry during moral reasoning':fontsize=28:fontcolor=0x6a6a8a:x=(w-text_w)/2:y=(h-text_h)/2+40:alpha='if(lt(t,0.5),0,if(lt(t,1),min(1,(t-0.5)/0.5),1))'[mtitle]"

# ── Section 4: End card (4s) ─────────────────────────────────────
END_FILTER="color=c=black:s=1920x1080:d=4[ebg];\
[ebg]drawtext=fontfile='$FONT':text='SYMTHAEA':fontsize=72:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2-80:alpha='if(lt(t,0.5),t/0.5,if(gt(t,3),max(0,1-(t-3)),1))',\
drawtext=fontfile='$FONT':text='Holographic Liquid Brain Architecture':fontsize=32:fontcolor=0xc8c8e0:x=(w-text_w)/2:y=(h-text_h)/2-20:alpha='if(lt(t,0.8),0,if(lt(t,1.3),min(1,(t-0.8)/0.5),if(gt(t,3),max(0,1-(t-3)),1)))',\
drawtext=fontfile='$FONT':text='HDC + IIT/Phi + LTC/CfC + Active Inference':fontsize=24:fontcolor=0x5090ff:x=(w-text_w)/2:y=(h-text_h)/2+30:alpha='if(lt(t,1),0,if(lt(t,1.5),min(1,(t-1)/0.5),if(gt(t,3),max(0,1-(t-3)),1)))',\
drawtext=fontfile='$FONT':text='luminousdynamics.org':fontsize=20:fontcolor=0x6a6a8a:x=(w-text_w)/2:y=(h-text_h)/2+80:alpha='if(lt(t,1.5),0,if(lt(t,2),min(1,(t-1.5)/0.5),if(gt(t,3),max(0,1-(t-3)),1)))'[end]"

# ── Trim dashboard to ~20s (most interesting part: moral stress onset ~cycle 100-300) ──
# The dashboard is 38s; take 5s to 25s (moral stress → heroism phases)
DASH_TRIM_START=5
DASH_TRIM_DUR=20

echo "Generating title cards..."

# Generate title cards as intermediate files
ffmpeg -y -f lavfi -i "color=c=black:s=1920x1080:d=4,format=yuv420p" \
  -vf "drawtext=fontfile='$FONT':text='SYMTHAEA':fontsize=96:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2-60:alpha='if(lt(t\,0.5)\,0\,if(lt(t\,1.5)\,min(1\,(t-0.5))\,1))',\
drawtext=fontfile='$FONT':text='Consciousness-First AI':fontsize=36:fontcolor=0xc8c8e0:x=(w-text_w)/2:y=(h-text_h)/2+40:alpha='if(lt(t\,1)\,0\,if(lt(t\,2)\,min(1\,(t-1))\,1))',\
drawtext=fontfile='$FONT':text='Three capabilities. One architecture.':fontsize=24:fontcolor=0x6a6a8a:x=(w-text_w)/2:y=(h-text_h)/2+90:alpha='if(lt(t\,1.5)\,0\,if(lt(t\,2.5)\,min(1\,(t-1.5))\,1))'" \
  -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 "$VID_DIR/_title.mp4" 2>/dev/null
echo "  Title card done."

ffmpeg -y -f lavfi -i "color=c=black:s=1920x1080:d=2,format=yuv420p" \
  -vf "drawtext=fontfile='$FONT':text='I. THE SCENARIO':fontsize=64:fontcolor=0x5090ff:x=(w-text_w)/2:y=(h-text_h)/2-20:alpha='if(lt(t\,0.3)\,t/0.3\,1)',\
drawtext=fontfile='$FONT':text='A drone with no safety rules saves a human life':fontsize=28:fontcolor=0x6a6a8a:x=(w-text_w)/2:y=(h-text_h)/2+40:alpha='if(lt(t\,0.5)\,0\,if(lt(t\,1)\,min(1\,(t-0.5)/0.5)\,1))'" \
  -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 "$VID_DIR/_scenario_title.mp4" 2>/dev/null
echo "  Scenario title done."

ffmpeg -y -f lavfi -i "color=c=black:s=1920x1080:d=2,format=yuv420p" \
  -vf "drawtext=fontfile='$FONT':text='II. THE MIND':fontsize=64:fontcolor=0x50e890:x=(w-text_w)/2:y=(h-text_h)/2-20:alpha='if(lt(t\,0.3)\,t/0.3\,1)',\
drawtext=fontfile='$FONT':text='Real-time consciousness telemetry during moral reasoning':fontsize=28:fontcolor=0x6a6a8a:x=(w-text_w)/2:y=(h-text_h)/2+40:alpha='if(lt(t\,0.5)\,0\,if(lt(t\,1)\,min(1\,(t-0.5)/0.5)\,1))'" \
  -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 "$VID_DIR/_mind_title.mp4" 2>/dev/null
echo "  Mind title done."

ffmpeg -y -f lavfi -i "color=c=black:s=1920x1080:d=4,format=yuv420p" \
  -vf "drawtext=fontfile='$FONT':text='SYMTHAEA':fontsize=72:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2-80:alpha='if(lt(t\,0.5)\,t/0.5\,if(gt(t\,3)\,max(0\,1-(t-3))\,1))',\
drawtext=fontfile='$FONT':text='Holographic Liquid Brain Architecture':fontsize=32:fontcolor=0xc8c8e0:x=(w-text_w)/2:y=(h-text_h)/2-20:alpha='if(lt(t\,0.8)\,0\,if(lt(t\,1.3)\,min(1\,(t-0.8)/0.5)\,if(gt(t\,3)\,max(0\,1-(t-3))\,1)))',\
drawtext=fontfile='$FONT':text='HDC + IIT/Phi + LTC/CfC + Active Inference':fontsize=24:fontcolor=0x5090ff:x=(w-text_w)/2:y=(h-text_h)/2+30:alpha='if(lt(t\,1)\,0\,if(lt(t\,1.5)\,min(1\,(t-1)/0.5)\,if(gt(t\,3)\,max(0\,1-(t-3))\,1)))',\
drawtext=fontfile='$FONT':text='luminousdynamics.org':fontsize=20:fontcolor=0x6a6a8a:x=(w-text_w)/2:y=(h-text_h)/2+80:alpha='if(lt(t\,1.5)\,0\,if(lt(t\,2)\,min(1\,(t-1.5)/0.5)\,if(gt(t\,3)\,max(0\,1-(t-3))\,1)))'" \
  -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 "$VID_DIR/_end.mp4" 2>/dev/null
echo "  End card done."

# ── Trim dashboard ──────────────────────────────────────────────
echo "Trimming consciousness dashboard ($DASH_TRIM_START to $((DASH_TRIM_START + DASH_TRIM_DUR))s)..."
ffmpeg -y -ss "$DASH_TRIM_START" -t "$DASH_TRIM_DUR" -i "$DASHBOARD" \
  -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 \
  "$VID_DIR/_dashboard_trimmed.mp4" 2>/dev/null
echo "  Dashboard trimmed."

# ── Add crossfades to transitions ──────────────────────────────
echo "Adding crossfade transitions..."

# Re-encode narrated video to ensure compatible codec for concat
ffmpeg -y -i "$NARRATED" \
  -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 -c:a aac -b:a 192k \
  "$VID_DIR/_narrated_reenc.mp4" 2>/dev/null
echo "  Narrated re-encoded."

# ── Concat with crossfades using xfade filter ──────────────────
# Structure: title(4s) → xfade → scenario_title(2s) → xfade → narrated(12.7s) → xfade → mind_title(2s) → xfade → dashboard(20s) → xfade → end(4s)
# xfade duration: 0.5s

echo "Compositing hero video..."

# Step 1: title + scenario_title
ffmpeg -y -i "$VID_DIR/_title.mp4" -i "$VID_DIR/_scenario_title.mp4" \
  -filter_complex "[0:v][1:v]xfade=transition=fade:duration=0.5:offset=3.5[v]" \
  -map "[v]" -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 \
  "$VID_DIR/_s1.mp4" 2>/dev/null

# Step 2: s1 + narrated
# s1 duration = 4 + 2 - 0.5 = 5.5s
ffmpeg -y -i "$VID_DIR/_s1.mp4" -i "$VID_DIR/_narrated_reenc.mp4" \
  -filter_complex "[0:v][1:v]xfade=transition=fade:duration=0.5:offset=5.0[v]" \
  -map "[v]" -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 \
  "$VID_DIR/_s2.mp4" 2>/dev/null

# Step 3: s2 + mind_title
# s2 duration = 5.5 + 12.7 - 0.5 = 17.7s
ffmpeg -y -i "$VID_DIR/_s2.mp4" -i "$VID_DIR/_mind_title.mp4" \
  -filter_complex "[0:v][1:v]xfade=transition=fade:duration=0.5:offset=17.2[v]" \
  -map "[v]" -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 \
  "$VID_DIR/_s3.mp4" 2>/dev/null

# Step 4: s3 + dashboard
# s3 duration = 17.7 + 2 - 0.5 = 19.2s
ffmpeg -y -i "$VID_DIR/_s3.mp4" -i "$VID_DIR/_dashboard_trimmed.mp4" \
  -filter_complex "[0:v][1:v]xfade=transition=fade:duration=0.5:offset=18.7[v]" \
  -map "[v]" -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 \
  "$VID_DIR/_s4.mp4" 2>/dev/null

# Step 5: s4 + end card
# s4 duration = 19.2 + 20 - 0.5 = 38.7s
ffmpeg -y -i "$VID_DIR/_s4.mp4" -i "$VID_DIR/_end.mp4" \
  -filter_complex "[0:v][1:v]xfade=transition=fade:duration=0.5:offset=38.2[v]" \
  -map "[v]" -c:v libx264 -preset fast -pix_fmt yuv420p -r 30 \
  "$VID_DIR/_s5.mp4" 2>/dev/null

# ── Add audio from narrated video at the right offset ──────────
# The narrated audio starts at the scenario section (after 5.5s offset)
echo "Mixing audio..."

# Generate silent base audio for full duration, then overlay narrated audio at offset
HERO_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$VID_DIR/_s5.mp4" 2>/dev/null)
echo "  Hero duration: ${HERO_DUR}s"

# Extract audio from narrated video
ffmpeg -y -i "$NARRATED" -vn -c:a aac -b:a 192k "$VID_DIR/_narr_audio.m4a" 2>/dev/null

# Create silence + delayed narration audio
ffmpeg -y \
  -f lavfi -i "anullsrc=r=44100:cl=mono" \
  -i "$VID_DIR/_narr_audio.m4a" \
  -filter_complex "[1:a]adelay=5500|5500[narr];[0:a][narr]amix=inputs=2:duration=shortest:normalize=0[out]" \
  -map "[out]" -t "$HERO_DUR" -c:a aac -b:a 192k \
  "$VID_DIR/_hero_audio.m4a" 2>/dev/null

# Mux video + audio
ffmpeg -y -i "$VID_DIR/_s5.mp4" -i "$VID_DIR/_hero_audio.m4a" \
  -c:v copy -c:a copy -shortest \
  "$OUT" 2>/dev/null

# ── Cleanup intermediate files ──────────────────────────────────
echo "Cleaning up..."
rm -f "$VID_DIR"/_title.mp4 "$VID_DIR"/_scenario_title.mp4 "$VID_DIR"/_mind_title.mp4 \
      "$VID_DIR"/_end.mp4 "$VID_DIR"/_dashboard_trimmed.mp4 "$VID_DIR"/_narrated_reenc.mp4 \
      "$VID_DIR"/_s1.mp4 "$VID_DIR"/_s2.mp4 "$VID_DIR"/_s3.mp4 "$VID_DIR"/_s4.mp4 "$VID_DIR"/_s5.mp4 \
      "$VID_DIR"/_narr_audio.m4a "$VID_DIR"/_hero_audio.m4a

# ── Report ─────────────────────────────────────────────────────
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
echo "  0:06 - 0:18  Moral drone simulation (narrated)"
echo "  0:18 - 0:20  II. THE MIND"
echo "  0:20 - 0:40  Consciousness dashboard"
echo "  0:40 - 0:44  End card"
