---
description: "Check Mycelix ecosystem health - builds, tests, versions, bridge coverage"
---

# Mycelix Ecosystem Status

Quick health check across the entire Mycelix ecosystem.

Arguments: $ARGUMENTS

If args are provided, focus on that area (e.g., "builds", "tests", "versions", "bridge").

## 1. Build Status

Check which hApps compile successfully:

!echo "=== WASM Build Status ===" && for dir in mycelix-commons mycelix-civic mycelix-core mycelix-mail mycelix-identity mycelix-finance mycelix-governance mycelix-knowledge mycelix-energy mycelix-health mycelix-space mycelix-marketplace mycelix-supplychain mycelix-edunet mycelix-music; do if [ -d "/srv/luminous-dynamics/$dir" ] && [ -f "/srv/luminous-dynamics/$dir/Cargo.toml" ]; then echo -n "  $dir: "; if [ -d "/srv/luminous-dynamics/$dir/target/wasm32-unknown-unknown" ]; then echo "has WASM artifacts"; else echo "no WASM artifacts (needs build)"; fi; fi; done

## 2. Test Counts

Quick survey of test files:

!echo "=== Test Counts ===" && echo "  Bridge Common:  $(grep -c '#\[test\]' /srv/luminous-dynamics/crates/mycelix-bridge-common/src/lib.rs 2>/dev/null || echo 0) tests" && echo "  Commons Cluster: $(grep -rc '#\[test\]' /srv/luminous-dynamics/mycelix-commons/zomes/ 2>/dev/null | tail -1 | cut -d: -f2 || echo 0) tests" && echo "  Civic Cluster:  $(grep -rc '#\[test\]' /srv/luminous-dynamics/mycelix-civic/zomes/ 2>/dev/null | tail -1 | cut -d: -f2 || echo 0) tests" && echo "  Sweettest:      $(ls /srv/luminous-dynamics/mycelix-workspace/tests/sweettest/tests/*.rs 2>/dev/null | grep -v harness | wc -l) test files" && echo "  SDK TS:         $(grep -rc 'it(' /srv/luminous-dynamics/mycelix-workspace/sdk-ts/tests/ 2>/dev/null | awk -F: '{s+=$2}END{print s}') test cases" && echo "  SDK Python:     $(grep -rc 'def test_' /srv/luminous-dynamics/mycelix-workspace/sdk-python/tests/ 2>/dev/null | awk -F: '{s+=$2}END{print s}') test cases"

## 3. Version Consistency

Check that Holochain dependency versions are consistent across workspaces:

!echo "=== Holochain Versions ===" && for f in /srv/luminous-dynamics/mycelix-commons/Cargo.toml /srv/luminous-dynamics/mycelix-civic/Cargo.toml /srv/luminous-dynamics/crates/mycelix-bridge-common/Cargo.toml /srv/luminous-dynamics/mycelix-workspace/sdk/Cargo.toml; do if [ -f "$f" ]; then dir=$(echo "$f" | sed 's|/srv/luminous-dynamics/||' | sed 's|/Cargo.toml||'); hdk=$(grep 'hdk.*=' "$f" | head -1 | tr -d ' '); echo "  $dir: $hdk"; fi; done

## 4. Bridge Coverage

Check allowlist completeness:

!echo "=== Bridge Allowlists ===" && echo "Commons ALLOWED_ZOMES:" && grep -c '"[a-z_]*"' /srv/luminous-dynamics/mycelix-commons/zomes/commons-bridge/coordinator/src/lib.rs 2>/dev/null | head -1 && echo "Civic ALLOWED_ZOMES:" && grep -c '"[a-z_]*"' /srv/luminous-dynamics/mycelix-civic/zomes/civic-bridge/coordinator/src/lib.rs 2>/dev/null | head -1

## 5. Workspace Members

Count zomes per cluster:

!echo "=== Zome Counts ===" && echo "  Commons members: $(grep -c '"zomes/' /srv/luminous-dynamics/mycelix-commons/Cargo.toml 2>/dev/null)" && echo "  Civic members:   $(grep -c '"zomes/' /srv/luminous-dynamics/mycelix-civic/Cargo.toml 2>/dev/null)" && echo "  Total standalone hApps: $(ls -d /srv/luminous-dynamics/mycelix-*/Cargo.toml 2>/dev/null | wc -l)"

## 6. Git Status

!echo "=== Git Status ===" && cd /srv/luminous-dynamics && echo "Branch: $(git branch --show-current)" && echo "Uncommitted changes:" && git status --porcelain mycelix-* crates/mycelix-* 2>/dev/null | head -20

## 7. SDK Versions

!echo "=== SDK Versions ===" && echo "  Rust SDK: $(grep '^version' /srv/luminous-dynamics/mycelix-workspace/sdk/Cargo.toml 2>/dev/null | head -1)" && echo "  TS SDK:   $(grep '"version"' /srv/luminous-dynamics/mycelix-workspace/sdk-ts/package.json 2>/dev/null | head -1 | tr -d ' ,')" && echo "  Python:   $(grep 'version' /srv/luminous-dynamics/mycelix-workspace/sdk-python/pyproject.toml 2>/dev/null | head -1)"

## Summary

After gathering all the above data, provide a concise health report:

1. **Build Health**: X/Y hApps have WASM artifacts
2. **Test Health**: Total test count, any known failures
3. **Version Consistency**: Any mismatches?
4. **Bridge Coverage**: All zomes reachable?
5. **Action Items**: What needs attention?

Rate overall health: Healthy / Needs Attention / Degraded
