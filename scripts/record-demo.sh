#!/usr/bin/env bash
# Record the Sovereign Inoculation demo video
# Usage: ./scripts/record-demo.sh
#
# Prerequisites:
#   1. QEMU VM running (./scripts/test-vm-dual-nvme.sh)
#   2. eval-api running on :8090
#   3. ssh-relay running on :8091
#   4. Portal open in browser at portal.html
#
# This script:
#   - Starts wf-recorder (Wayland screen capture)
#   - Records to ~/Videos/sovereign-inoculation-demo.mp4
#   - Press Ctrl+C to stop recording

set -euo pipefail

OUTPUT="${1:-$HOME/Videos/sovereign-inoculation-demo-$(date +%Y%m%d-%H%M%S).mp4}"
mkdir -p "$(dirname "$OUTPUT")"

echo "=== Sovereign Inoculation Demo Recording ==="
echo ""
echo "Before recording, ensure:"
echo "  1. QEMU VM is running:  ./scripts/test-vm-dual-nvme.sh"
echo "  2. eval-api is running: eval-api &"
echo "  3. ssh-relay is running: ssh-relay &"
echo "  4. Portal is open in browser"
echo ""
echo "Recording will start in 3 seconds..."
echo "Press Ctrl+C to stop recording."
echo "Output: $OUTPUT"
echo ""

sleep 3

# Use wf-recorder for Wayland, fall back to ffmpeg x11grab
if command -v wf-recorder &>/dev/null; then
    wf-recorder -f "$OUTPUT" -c libx264 -p crf=23 -p preset=medium
elif [ -n "${WAYLAND_DISPLAY:-}" ]; then
    nix-shell -p wf-recorder --run "wf-recorder -f '$OUTPUT' -c libx264 -p crf=23 -p preset=medium"
else
    # X11 fallback
    ffmpeg -f x11grab -i "$DISPLAY" -c:v libx264 -crf 23 -preset medium "$OUTPUT"
fi

echo ""
echo "Recording saved: $OUTPUT"
echo "Size: $(du -h "$OUTPUT" | cut -f1)"
echo ""
echo "To trim: ffmpeg -i '$OUTPUT' -ss 00:00:05 -to 00:03:00 -c copy trimmed.mp4"
