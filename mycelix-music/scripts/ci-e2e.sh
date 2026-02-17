#!/usr/bin/env bash
set -euo pipefail

# CI harness: anvil → deploy → migrate → indexer (background) → api (background) → web (background) → playwright smoke

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pushd "$ROOT_DIR" >/dev/null

API_PORT=${API_PORT:-3100}
WEB_PORT=${WEB_PORT:-3000}
RPC_URL=${RPC_URL:-http://127.0.0.1:8545}
API_CHAIN_RPC_URL=${API_CHAIN_RPC_URL:-$RPC_URL}
ROUTER_ADDRESS=${ROUTER_ADDRESS:-}
API_ADMIN_KEY=${API_ADMIN_KEY:-ci-admin-key}
SMOKE_BASE_URL=${SMOKE_BASE_URL:-http://localhost:${WEB_PORT}}

die() { echo "ERROR: $*" >&2; exit 1; }

command -v anvil >/dev/null || die "anvil is required"

if [ -z "$ROUTER_ADDRESS" ]; then
  echo "ROUTER_ADDRESS not set; deploy step must set it"
fi

echo "Starting anvil..."
anvil --block-time 1 --silent --port "${RPC_URL##*:}" >/tmp/anvil.log 2>&1 &
ANVIL_PID=$!
sleep 3

LOG_DIR=${LOG_DIR:-/tmp/mycelix-ci}
ARTIFACT_DIR=${ARTIFACT_DIR:-$LOG_DIR}
mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

echo "Deploying contracts..."
npm run contracts:deploy:local >/tmp/contracts-deploy.log 2>&1 || (cat /tmp/contracts-deploy.log && exit 1)

if [ -z "$ROUTER_ADDRESS" ]; then
  ROUTER_ADDRESS=$(node -e "const fs=require('fs');const path=require('path');const dir=path.join('contracts','broadcast','DeployLocal.s.sol','31337');const files=fs.readdirSync(dir).filter(f=>f.startsWith('run-') && f.endsWith('.json')).sort().reverse();let addr='';for(const f of files){const j=JSON.parse(fs.readFileSync(path.join(dir,f)));const tx=j.transactions?.find(t=>t.contractName==='EconomicStrategyRouter');if(tx?.contractAddress){addr=tx.contractAddress;break;}}console.log(addr);")
  [ -n "$ROUTER_ADDRESS" ] || die "Could not extract ROUTER_ADDRESS from deploy artifacts"
  export ROUTER_ADDRESS
fi

export DATABASE_URL=${DATABASE_URL:-postgresql://mycelix:mycelix_dev_pass@localhost:5432/mycelix_music}
export REDIS_URL=${REDIS_URL:-redis://localhost:6379}
export API_CHAIN_RPC_URL
export API_ADMIN_KEY

echo "Running migrations..."
npm run migrate --workspace=apps/api

echo "Starting indexer..."
npm run indexer --workspace=apps/api >"$LOG_DIR/indexer.log" 2>&1 &
INDEXER_PID=$!
sleep 2

echo "Starting API..."
API_PORT=$API_PORT npm run dev --workspace=apps/api >"$LOG_DIR/api.log" 2>&1 &
API_PID=$!
sleep 4

echo "Starting web..."
NEXT_PUBLIC_API_URL="http://localhost:${API_PORT}" PORT=$WEB_PORT npm run dev --workspace=apps/web >"$LOG_DIR/web.log" 2>&1 &
WEB_PID=$!
sleep 6

echo "Checking metrics endpoints..."
curl -sf "http://localhost:${API_PORT}/metrics" >/dev/null || die "API metrics not reachable"
curl -sf "http://localhost:9400/metrics" >/dev/null || die "Indexer metrics not reachable"

echo "Scraping metrics snapshots..."
curl -s "http://localhost:${API_PORT}/metrics" >"$LOG_DIR/api.metrics" || true
curl -s "http://localhost:9400/metrics" >"$LOG_DIR/indexer.metrics" || true

echo "Running Playwright smoke..."
SMOKE_E2E=1 SMOKE_BASE_URL=$SMOKE_BASE_URL npm run test:e2e -- --project=chromium
SMOKE_STATUS=$?

cleanup() {
  kill $WEB_PID $API_PID $INDEXER_PID $ANVIL_PID >/dev/null 2>&1 || true
}
trap cleanup EXIT

on_exit() {
  status=$?
  if [ $status -ne 0 ] || [ $SMOKE_STATUS -ne 0 ]; then
    echo "CI harness failed (exit $status). Logs:"
    for f in "$LOG_DIR"/{indexer.log,api.log,web.log}; do
      [ -f "$f" ] && { echo "== $f =="; tail -n 200 "$f"; }
    done
    cp "$LOG_DIR"/*.log "$LOG_DIR"/*.metrics "$ARTIFACT_DIR"/ 2>/dev/null || true
  fi
}
trap on_exit EXIT

popd >/dev/null
