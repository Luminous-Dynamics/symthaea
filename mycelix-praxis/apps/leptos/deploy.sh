#!/usr/bin/env bash
# Deploy EduNet to IPFS + Cloudflare gateway
# Usage: ./deploy.sh
#
# Builds the Leptos app, pins to local IPFS, updates DNS record.
# Requires: BWS_ACCESS_TOKEN, IPFS daemon running, trunk

set -euo pipefail

DIST_DIR="$(cd "$(dirname "$0")" && pwd)/dist"
ZONE_ID="3f3a2baaa758ed78459447577d2fa3bb"  # luminousdynamics.io
RECORD_NAME="_dnslink.edunet"

echo "=== EduNet Deploy to IPFS ==="

# Pre-flight checks
if ! curl -s --max-time 3 "http://127.0.0.1:5001/api/v0/id" > /dev/null 2>&1; then
  echo "ERROR: IPFS daemon not running on 127.0.0.1:5001"
  echo "Start it with: ipfs daemon &"
  exit 1
fi

if [ -z "${BWS_ACCESS_TOKEN:-}" ] && ! grep -q BWS_ACCESS_TOKEN ~/.zshrc 2>/dev/null; then
  echo "ERROR: BWS_ACCESS_TOKEN not set and not found in ~/.zshrc"
  exit 1
fi

# 1. Build
echo "[1/4] Building..."
~/.cargo/bin/trunk build --release 2>&1 | tail -1

# 2. Pin to IPFS
echo "[2/4] Pinning to IPFS..."
CID=$(cd "$DIST_DIR" && curl -s -X POST -F "file=@." -F "recursive=true" \
  "http://127.0.0.1:5001/api/v0/add?recursive=true&wrap-with-directory=true" \
  | tail -1 | python3 -c "import sys,json; print(json.load(sys.stdin)['Hash'])")

if [ -z "$CID" ] || ! echo "$CID" | grep -qE '^Qm[a-zA-Z0-9]{44}$|^bafy[a-z0-9]{50,}$'; then
  echo "ERROR: Invalid CID returned from IPFS: '$CID'"
  exit 1
fi

curl -s -X POST "http://127.0.0.1:5001/api/v0/pin/add?arg=$CID" > /dev/null
echo "  CID: $CID"

# 3. Get Cloudflare token
echo "[3/4] Updating DNS..."
export BWS_ACCESS_TOKEN="${BWS_ACCESS_TOKEN:-$(grep BWS_ACCESS_TOKEN ~/.zshrc | cut -d'"' -f2)}"
CF_TOKEN=$(~/.cargo/bin/bws secret get aa195733-a93a-4284-97b5-b4130141fce9 2>/dev/null \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['value'])")

# Find existing _dnslink record
RECORD_ID=$(curl -s "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records?name=$RECORD_NAME.luminousdynamics.io&type=TXT" \
  -H "Authorization: Bearer $CF_TOKEN" \
  | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print(r[0]['id'] if r else '')")

if [ -n "$RECORD_ID" ]; then
  # Update existing record
  curl -s -X PUT "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records/$RECORD_ID" \
    -H "Authorization: Bearer $CF_TOKEN" \
    -H "Content-Type: application/json" \
    --data "{\"type\":\"TXT\",\"name\":\"$RECORD_NAME\",\"content\":\"dnslink=/ipfs/$CID\",\"ttl\":1}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('  DNS updated' if d['success'] else d['errors'])"
else
  # Create new record
  curl -s -X POST "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records" \
    -H "Authorization: Bearer $CF_TOKEN" \
    -H "Content-Type: application/json" \
    --data "{\"type\":\"TXT\",\"name\":\"$RECORD_NAME\",\"content\":\"dnslink=/ipfs/$CID\",\"ttl\":1}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('  DNS created' if d['success'] else d['errors'])"
fi

# 4. Done
echo "[4/4] Done!"
echo ""
echo "  IPFS: ipfs://$CID"
echo "  URL:  https://edunet.luminousdynamics.io"
echo "  Gateway: https://cloudflare-ipfs.com/ipfs/$CID/"
echo ""
echo "  DNS propagation may take 1-5 minutes."
