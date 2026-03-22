#!/usr/bin/env bash
# Terra Atlas MVP - Comprehensive Live Testing Script
# Tests all pages, features, and user flows end-to-end

set -e

BASE_URL="http://localhost:3000"
FAILED_TESTS=()
PASSED_TESTS=()

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "======================================"
echo "Terra Atlas MVP - Live Testing Suite"
echo "======================================"
echo ""

# Helper Functions
test_page() {
  local name="$1"
  local url="$2"
  local expected_status="${3:-200}"

  echo -n "Testing $name... "
  status=$(curl -s -o /dev/null -w '%{http_code}' "$url")

  if [ "$status" = "$expected_status" ]; then
    echo -e "${GREEN}✓ PASS${NC} (HTTP $status)"
    PASSED_TESTS+=("$name")
  else
    echo -e "${RED}✗ FAIL${NC} (HTTP $status, expected $expected_status)"
    FAILED_TESTS+=("$name")
  fi
}

test_api() {
  local name="$1"
  local url="$2"
  local expected_pattern="$3"

  echo -n "Testing API: $name... "
  response=$(curl -s "$url")

  if echo "$response" | grep -q "$expected_pattern"; then
    echo -e "${GREEN}✓ PASS${NC}"
    PASSED_TESTS+=("API: $name")
  else
    echo -e "${RED}✗ FAIL${NC}"
    echo "  Expected pattern: $expected_pattern"
    echo "  Got: ${response:0:100}..."
    FAILED_TESTS+=("API: $name")
  fi
}

test_json_response() {
  local name="$1"
  local url="$2"

  echo -n "Testing JSON: $name... "
  response=$(curl -s "$url")

  if echo "$response" | jq . >/dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC} (Valid JSON)"
    PASSED_TESTS+=("JSON: $name")

    # Show first 200 chars of response
    echo "  Preview: $(echo "$response" | jq -c . | head -c 100)..."
  else
    echo -e "${RED}✗ FAIL${NC} (Invalid JSON)"
    FAILED_TESTS+=("JSON: $name")
  fi
}

echo "${BLUE}=== Section 1: Core Pages ===${NC}"
test_page "Homepage" "$BASE_URL"
test_page "Explore Page" "$BASE_URL/explore"
test_page "Dashboard" "$BASE_URL/dashboard"
test_page "Portfolio" "$BASE_URL/portfolio"
test_page "SMR Projects" "$BASE_URL/smr"
test_page "Landing Page" "$BASE_URL/landing"
test_page "Horizon Page" "$BASE_URL/horizon"
echo ""

echo "${BLUE}=== Section 2: Auth Pages ===${NC}"
test_page "Login" "$BASE_URL/auth/login"
test_page "Signup" "$BASE_URL/auth/signup"
test_page "Reset Password" "$BASE_URL/auth/reset-password"
echo ""

echo "${BLUE}=== Section 3: API Endpoints - Stats ===${NC}"
test_json_response "Stats API" "$BASE_URL/api/stats"
echo ""

echo "${BLUE}=== Section 4: API Endpoints - Sites ===${NC}"
test_json_response "Sites API (world level)" "$BASE_URL/api/sites?level=world"
test_json_response "Sites API (country level)" "$BASE_URL/api/sites?level=country&country=United%20States"
test_json_response "Sites API (state level)" "$BASE_URL/api/sites?level=state&state=California"
echo ""

echo "${BLUE}=== Section 5: API Endpoints - Projects ===${NC}"
test_json_response "Projects API" "$BASE_URL/api/projects"
test_json_response "Projects API (solar filter)" "$BASE_URL/api/projects?type=solar"
test_json_response "Projects API (state filter)" "$BASE_URL/api/projects?state=California"
echo ""

echo "${BLUE}=== Section 6: Data Integrity ===${NC}"

# Test sites data structure
echo -n "Testing Sites Data Structure... "
sites_data=$(curl -s "$BASE_URL/api/sites?level=world")
if echo "$sites_data" | jq -e '.sites | length > 0' >/dev/null 2>&1; then
  site_count=$(echo "$sites_data" | jq -r '.sites | length')
  echo -e "${GREEN}✓ PASS${NC} ($site_count sites returned)"
  PASSED_TESTS+=("Sites Data: Has sites")

  # Check first site has required fields
  echo -n "Testing Site Fields... "
  if echo "$sites_data" | jq -e '.sites[0] | has("id") and has("latitude") and has("longitude")' >/dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC} (id, lat, lon present)"
    PASSED_TESTS+=("Sites Data: Required fields")
  else
    echo -e "${RED}✗ FAIL${NC} (Missing required fields)"
    FAILED_TESTS+=("Sites Data: Required fields")
  fi
else
  echo -e "${RED}✗ FAIL${NC} (No sites returned)"
  FAILED_TESTS+=("Sites Data: Has sites")
fi
echo ""

echo "${BLUE}=== Section 7: Feature Testing ===${NC}"

# Test Investment Scorecard Data
echo -n "Testing Investment Scorecard Data... "
if echo "$sites_data" | jq -e '.sites[0] | has("estimated_irr")' >/dev/null 2>&1; then
  echo -e "${GREEN}✓ PASS${NC} (IRR data present)"
  PASSED_TESTS+=("Feature: Investment Scorecard Data")
else
  echo -e "${YELLOW}⚠ WARN${NC} (No IRR data)"
  FAILED_TESTS+=("Feature: Investment Scorecard Data")
fi

# Test Tier 1 Features (Search & Filter)
echo -n "Testing Search & Filter... "
filter_test=$(curl -s "$BASE_URL/api/sites?level=world&type=solar")
if echo "$filter_test" | jq . >/dev/null 2>&1; then
  echo -e "${GREEN}✓ PASS${NC} (Filter API responds)"
  PASSED_TESTS+=("Feature: Search & Filter")
else
  echo -e "${RED}✗ FAIL${NC}"
  FAILED_TESTS+=("Feature: Search & Filter")
fi
echo ""

echo "${BLUE}=== Section 8: Performance ===${NC}"

# Test API response time
echo -n "Testing API Response Time... "
start=$(date +%s%3N)
curl -s "$BASE_URL/api/sites?level=world" >/dev/null
end=$(date +%s%3N)
duration=$((end - start))

if [ "$duration" -lt 5000 ]; then
  echo -e "${GREEN}✓ PASS${NC} (${duration}ms < 5000ms)"
  PASSED_TESTS+=("Performance: API < 5s")
else
  echo -e "${YELLOW}⚠ SLOW${NC} (${duration}ms)"
  FAILED_TESTS+=("Performance: API < 5s")
fi

# Test page load time
echo -n "Testing Homepage Load Time... "
start=$(date +%s%3N)
curl -s "$BASE_URL" >/dev/null
end=$(date +%s%3N)
duration=$((end - start))

if [ "$duration" -lt 3000 ]; then
  echo -e "${GREEN}✓ PASS${NC} (${duration}ms < 3000ms)"
  PASSED_TESTS+=("Performance: Homepage < 3s")
else
  echo -e "${YELLOW}⚠ SLOW${NC} (${duration}ms)"
  FAILED_TESTS+=("Performance: Homepage < 3s")
fi
echo ""

echo "${BLUE}=== Section 9: Database & Real Data ===${NC}"

# Check if we have real FERC data
echo -n "Testing FERC Data Import... "
projects=$(curl -s "$BASE_URL/api/projects")
if echo "$projects" | jq -e 'length > 100' >/dev/null 2>&1; then
  count=$(echo "$projects" | jq 'length')
  echo -e "${GREEN}✓ PASS${NC} ($count projects)"
  PASSED_TESTS+=("Data: FERC Import")
else
  echo -e "${RED}✗ FAIL${NC} (< 100 projects, likely demo data)"
  FAILED_TESTS+=("Data: FERC Import")
fi

# Check if we have USACE dam data
echo -n "Testing USACE Dam Data... "
dams=$(curl -s "$BASE_URL/api/sites?level=world&type=hydro")
if echo "$dams" | jq -e '.sites | length > 1000' >/dev/null 2>&1; then
  count=$(echo "$dams" | jq -r '.sites | length')
  echo -e "${GREEN}✓ PASS${NC} ($count dams)"
  PASSED_TESTS+=("Data: USACE Dams")
else
  count=$(echo "$dams" | jq -r '.sites | length // 0')
  echo -e "${RED}✗ FAIL${NC} ($count dams, expected > 1000)"
  FAILED_TESTS+=("Data: USACE Dams")
fi

# Check if we have SMR pipeline data
echo -n "Testing SMR Pipeline Data... "
smr=$(curl -s "$BASE_URL/api/sites?level=world&type=nuclear")
if echo "$smr" | jq -e '.sites | length > 40' >/dev/null 2>&1; then
  count=$(echo "$smr" | jq -r '.sites | length')
  echo -e "${GREEN}✓ PASS${NC} ($count SMR sites)"
  PASSED_TESTS+=("Data: SMR Pipeline")
else
  count=$(echo "$smr" | jq -r '.sites | length // 0')
  echo -e "${RED}✗ FAIL${NC} ($count SMR sites, expected > 40)"
  FAILED_TESTS+=("Data: SMR Pipeline")
fi
echo ""

echo "${BLUE}=== Section 10: Vision Features ===${NC}"

# Test features from vision document
echo -n "Testing Real-Time Data Stream... "
echo -e "${RED}✗ NOT IMPLEMENTED${NC}"
FAILED_TESTS+=("Vision: Real-Time Data Stream")

echo -n "Testing GIS-MCDA Toolkit... "
echo -e "${RED}✗ NOT IMPLEMENTED${NC}"
FAILED_TESTS+=("Vision: GIS-MCDA")

echo -n "Testing Economic Impact Calculator... "
echo -e "${RED}✗ NOT IMPLEMENTED${NC}"
FAILED_TESTS+=("Vision: Economic Impact")

echo -n "Testing Corridor Discovery... "
# Check if API exists
if curl -s "$BASE_URL/api/discovery/corridors" | jq . >/dev/null 2>&1; then
  echo -e "${YELLOW}⚠ PARTIAL${NC} (API exists but no real data)"
  PASSED_TESTS+=("Vision: Corridor Discovery (Partial)")
else
  echo -e "${RED}✗ NOT IMPLEMENTED${NC}"
  FAILED_TESTS+=("Vision: Corridor Discovery")
fi

echo -n "Testing Investment Flow (Pledging)... "
if curl -s -o /dev/null -w '%{http_code}' "$BASE_URL/invest" | grep -q "200"; then
  echo -e "${YELLOW}⚠ PARTIAL${NC} (Page exists, needs testing)"
  PASSED_TESTS+=("Vision: Investment Flow (Partial)")
else
  echo -e "${RED}✗ NOT FOUND${NC}"
  FAILED_TESTS+=("Vision: Investment Flow")
fi

echo -n "Testing User Portfolio... "
if curl -s -o /dev/null -w '%{http_code}' "$BASE_URL/portfolio" | grep -q "200"; then
  echo -e "${YELLOW}⚠ PARTIAL${NC} (Page exists, needs auth)"
  PASSED_TESTS+=("Vision: Portfolio (Partial)")
else
  echo -e "${RED}✗ NOT FOUND${NC}"
  FAILED_TESTS+=("Vision: Portfolio")
fi
echo ""

# ========================================
# FINAL SUMMARY
# ========================================

echo "======================================"
echo "           TEST SUMMARY"
echo "======================================"
echo ""

TOTAL_TESTS=$((${#PASSED_TESTS[@]} + ${#FAILED_TESTS[@]}))
PASS_RATE=$(echo "scale=1; ${#PASSED_TESTS[@]} * 100 / $TOTAL_TESTS" | bc)

echo -e "${GREEN}PASSED: ${#PASSED_TESTS[@]}${NC}"
echo -e "${RED}FAILED: ${#FAILED_TESTS[@]}${NC}"
echo -e "TOTAL: $TOTAL_TESTS"
echo -e "PASS RATE: ${PASS_RATE}%"
echo ""

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
  echo -e "${RED}Failed Tests:${NC}"
  for test in "${FAILED_TESTS[@]}"; do
    echo "  ✗ $test"
  done
  echo ""
fi

echo "======================================"

if [ "$PASS_RATE" = "100.0" ]; then
  echo -e "${GREEN}🎉 ALL TESTS PASSED!${NC}"
  exit 0
elif [ "${PASS_RATE%.*}" -ge 80 ]; then
  echo -e "${YELLOW}⚠️  MOSTLY PASSING (>80%)${NC}"
  exit 0
elif [ "${PASS_RATE%.*}" -ge 50 ]; then
  echo -e "${YELLOW}⚠️  NEEDS IMPROVEMENT (50-80%)${NC}"
  exit 1
else
  echo -e "${RED}❌ CRITICAL ISSUES (<50%)${NC}"
  exit 1
fi
