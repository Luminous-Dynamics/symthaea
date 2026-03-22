#!/usr/bin/env bash
# Deploy Trust-Weighted Allocation Network to production Supabase
#
# Usage:
#   cd terra-atlas-mvp && ./scripts/deploy-trust-scores.sh
#
# Requires:
#   - BWS_ACCESS_TOKEN in environment (from ~/.zshrc)
#   - bws CLI at ~/.cargo/bin/bws
#   - npx, node available
#   - Network access to Supabase

set -euo pipefail

echo "=== Trust-Weighted Allocation Network: Deploy ==="

# 1. Fetch credentials from BWS
echo "Fetching Supabase credentials from BWS..."
BWS=${HOME}/.cargo/bin/bws
SECRETS=$($BWS secret list 2>/dev/null)
SUPABASE_URL=$(echo "$SECRETS" | python3 -c "import sys,json; secrets=json.load(sys.stdin); print(next(s['value'] for s in secrets if s['key']=='supabase-url'))")
SUPABASE_KEY=$(echo "$SECRETS" | python3 -c "import sys,json; secrets=json.load(sys.stdin); print(next(s['value'] for s in secrets if s['key']=='supabase-anon-key'))")

if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_KEY" ]; then
    echo "ERROR: Could not fetch Supabase credentials from BWS"
    exit 1
fi
echo "  URL: ${SUPABASE_URL:0:30}..."
echo "  Key: ${SUPABASE_KEY:0:10}..."

# 2. Test connectivity
echo "Testing Supabase connectivity..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 \
    "${SUPABASE_URL}/rest/v1/" \
    -H "apikey: ${SUPABASE_KEY}")

if [ "$HTTP_CODE" = "000" ]; then
    echo "ERROR: Cannot reach Supabase. Check DNS/network."
    exit 1
fi
echo "  Connected (HTTP $HTTP_CODE)"

# 3. Run migrations via Supabase SQL endpoint
echo "Running migration: 005_consciousness_scores.sql..."
MIGRATION_005=$(cat supabase/migrations/005_consciousness_scores.sql)
curl -s -X POST "${SUPABASE_URL}/rest/v1/rpc/exec_sql" \
    -H "apikey: ${SUPABASE_KEY}" \
    -H "Authorization: Bearer ${SUPABASE_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"query\": $(echo "$MIGRATION_005" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')}" \
    > /dev/null 2>&1 || {
    echo "  Note: RPC exec_sql not available. Run migration manually in Supabase SQL Editor."
    echo "  File: supabase/migrations/005_consciousness_scores.sql"
}

echo "Running migration: 006_impact_metrics.sql..."
MIGRATION_006=$(cat supabase/migrations/006_impact_metrics.sql)
curl -s -X POST "${SUPABASE_URL}/rest/v1/rpc/exec_sql" \
    -H "apikey: ${SUPABASE_KEY}" \
    -H "Authorization: Bearer ${SUPABASE_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"query\": $(echo "$MIGRATION_006" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')}" \
    > /dev/null 2>&1 || {
    echo "  Note: Run migration manually in Supabase SQL Editor."
    echo "  File: supabase/migrations/006_impact_metrics.sql"
}

# 4. Seed consciousness scores
echo "Seeding trust scores..."
export NEXT_PUBLIC_SUPABASE_URL="$SUPABASE_URL"
export NEXT_PUBLIC_SUPABASE_ANON_KEY="$SUPABASE_KEY"
export SUPABASE_SERVICE_ROLE_KEY="$SUPABASE_KEY"

npx tsx scripts/seed-consciousness-scores.ts

echo ""
echo "=== Deploy Complete ==="
echo "  Trust scores seeded to Supabase"
echo "  View at: ${SUPABASE_URL}"
echo ""
echo "Next steps:"
echo "  1. Verify at: https://atlas.luminousdynamics.io"
echo "  2. Check /api/projects/:id/trust-score endpoint"
echo "  3. Verify HarmonyLeaderboard on dashboard"
