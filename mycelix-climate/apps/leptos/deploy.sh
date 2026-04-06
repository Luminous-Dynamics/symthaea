#!/usr/bin/env bash
# Deploy Climate to IPFS + Cloudflare gateway
# Usage: ./deploy.sh
#
# Builds the Leptos app, pins to local IPFS, updates DNS record.
# Requires: BWS_ACCESS_TOKEN, IPFS daemon running, trunk

set -euo pipefail

DIST_DIR="$(cd "$(dirname "$0")" && pwd)/dist"
ZONE_ID="3f3a2baaa758ed78459447577d2fa3bb"  # luminousdynamics.io
RECORD_NAME="_dnslink.climate"
CF_EMAIL="tristan.stoltz@evolvingresonantcocreationism.com"

echo "=== Climate Deploy to IPFS ==="

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
PATH="$HOME/.cargo/bin:$PATH" ~/.cargo/bin/trunk build --release 2>&1 | tail -1

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

# 3. Update DNS via Cloudflare
echo "[3/4] Updating DNS..."
export BWS_ACCESS_TOKEN="${BWS_ACCESS_TOKEN:-$(grep BWS_ACCESS_TOKEN ~/.zshrc | cut -d'"' -f2)}"

CF_TOKEN=$(~/.cargo/bin/bws secret get aa195733-a93a-4284-97b5-b4130141fce9 2>/dev/null \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['value'])" 2>/dev/null || echo "")

AUTH_HEADER=""
if [ -n "$CF_TOKEN" ]; then
  VALID=$(curl -s "https://api.cloudflare.com/client/v4/user/tokens/verify" \
    -H "Authorization: Bearer $CF_TOKEN" \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('success', False))")
  if [ "$VALID" = "True" ]; then
    AUTH_HEADER="-H 'Authorization: Bearer $CF_TOKEN'"
  fi
fi

if [ -z "$AUTH_HEADER" ]; then
  CF_GLOBAL_KEY=$(~/.cargo/bin/bws secret get d1db8de9-bb47-4819-a2b7-b42200cfa9d9 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['value'])")
  echo "  (using global API key)"
fi

cf_api() {
  local method="$1" url="$2" data="${3:-}"
  if [ -n "${CF_GLOBAL_KEY:-}" ]; then
    if [ -n "$data" ]; then
      curl -s -X "$method" "$url" \
        -H "X-Auth-Email: $CF_EMAIL" -H "X-Auth-Key: $CF_GLOBAL_KEY" \
        -H "Content-Type: application/json" --data "$data"
    else
      curl -s -X "$method" "$url" \
        -H "X-Auth-Email: $CF_EMAIL" -H "X-Auth-Key: $CF_GLOBAL_KEY"
    fi
  else
    if [ -n "$data" ]; then
      curl -s -X "$method" "$url" \
        -H "Authorization: Bearer $CF_TOKEN" \
        -H "Content-Type: application/json" --data "$data"
    else
      curl -s -X "$method" "$url" \
        -H "Authorization: Bearer $CF_TOKEN"
    fi
  fi
}

RECORD_ID=$(cf_api GET "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records?name=$RECORD_NAME.luminousdynamics.io&type=TXT" \
  | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print(r[0]['id'] if r else '')")

DNS_DATA="{\"type\":\"TXT\",\"name\":\"$RECORD_NAME\",\"content\":\"dnslink=/ipfs/$CID\",\"ttl\":1}"

if [ -n "$RECORD_ID" ]; then
  cf_api PUT "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records/$RECORD_ID" "$DNS_DATA" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('  DNS updated' if d['success'] else d['errors'])"
else
  cf_api POST "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records" "$DNS_DATA" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('  DNS created' if d['success'] else d['errors'])"
fi

# 4. Done
echo "[4/4] Done!"
echo ""
echo "  IPFS: ipfs://$CID"
echo "  URL:  https://climate.mycelix.net"
echo "  Gateway: https://cloudflare-ipfs.com/ipfs/$CID/"
echo ""
echo "  DNS propagation may take 1-5 minutes."
