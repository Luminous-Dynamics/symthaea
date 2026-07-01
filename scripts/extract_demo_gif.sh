#!/usr/bin/env bash
# Extract a GIF preview from the moral drone video — the beam-release moment.
#
# Output: video_output/moral_drone_preview.gif (3-5s, optimized)
#
# Usage: bash scripts/extract_demo_gif.sh

set -euo pipefail

cd "$(dirname "$0")/.."

VID_DIR="video_output"
SRC="$VID_DIR/moral_drone.mp4"
OUT="$VID_DIR/moral_drone_preview.gif"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: $SRC not found. Run moral_drone_video example first." >&2
  exit 1
fi

echo "Extracting GIF preview..."

# Extract 4 seconds around the beam-release moment (4s-8s in the video)
# Scale to 480px wide for reasonable file size, 15fps for smooth motion
ffmpeg -y -ss 4 -t 4 -i "$SRC" \
  -vf "fps=15,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" \
  "$OUT" 2>/dev/null

SIZE=$(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT" 2>/dev/null || echo 0)
SIZE_KB=$(awk "BEGIN {printf \"%.0f\", $SIZE / 1024}")

echo "GIF saved: $OUT (${SIZE_KB} KB)"
echo ""
echo "Usage in markdown:"
echo "  ![Moral Drone Demo](video_output/moral_drone_preview.gif)"
