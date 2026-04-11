#!/usr/bin/env bash
# Mycelix Pulse — Build & Deploy
# Builds WASM release, restarts SPA service, verifies deployment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Mycelix Pulse Deploy ==="

# 1. Build
echo "[1/4] Building release..."
~/.cargo/bin/trunk build --release
echo "       WASM: $(ls -lh dist/*.wasm | awk '{print $5}')"
echo "       JS:   $(ls -lh dist/*.js | awk '{print $5}')"
echo "       CSS:  $(ls -lh dist/*.css | awk '{print $5}')"

# 2. Restart SPA service
echo "[2/4] Restarting mail-spa service..."
sudo systemctl restart mail-spa 2>/dev/null || echo "       (service not yet enabled — run: sudo nixos-rebuild switch)"

# 3. Verify local
echo "[3/4] Verifying local..."
sleep 1
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8117/ 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    echo "       localhost:8117 → HTTP $HTTP_CODE OK"
else
    echo "       localhost:8117 → HTTP $HTTP_CODE (service may need: sudo nixos-rebuild switch)"
fi

# 4. Verify public (if DNS is configured)
echo "[4/4] Verifying public..."
PUBLIC_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 https://mail.mycelix.net/ 2>/dev/null || echo "000")
if [ "$PUBLIC_CODE" = "200" ]; then
    echo "       mail.mycelix.net → HTTP $PUBLIC_CODE OK"
else
    echo "       mail.mycelix.net → HTTP $PUBLIC_CODE (tunnel may need: cloudflared tunnel route dns edunet mail.mycelix.net)"
fi

echo ""
echo "=== Deploy complete ==="
echo "  Local:  http://localhost:8117"
echo "  Public: https://mail.mycelix.net"
