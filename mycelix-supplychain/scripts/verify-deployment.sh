#!/bin/bash

# Deployment Verification Script
# Purpose: Verify production deployment health and functionality
# Usage: ./verify-deployment.sh [API_URL]
# Example: ./verify-deployment.sh https://api.example.com

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
API_URL="${1:-http://localhost:8080}"
VERBOSE="${VERBOSE:-false}"
TIMEOUT=10

# Test data
TEST_BATCH_ID="VERIFY-$(date +%s)"
TEST_PRODUCT_ID="TEST-PRODUCT"

# Counters
TESTS_PASSED=0
TESTS_FAILED=0
TOTAL_TESTS=0

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}${BOLD}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}${BOLD}║  $1${NC}"
    echo -e "${BLUE}${BOLD}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_test() {
    echo -e "${YELLOW}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
    ((TESTS_PASSED++))
    ((TOTAL_TESTS++))
}

print_failure() {
    echo -e "${RED}✗${NC} $1"
    ((TESTS_FAILED++))
    ((TOTAL_TESTS++))
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

verbose() {
    if [ "$VERBOSE" = "true" ]; then
        echo -e "${NC}  $1${NC}"
    fi
}

# Check dependencies
check_dependencies() {
    print_header "Checking Dependencies"

    local deps=("curl" "jq")
    local all_present=true

    for dep in "${deps[@]}"; do
        if command -v "$dep" &> /dev/null; then
            print_success "$dep is installed"
        else
            print_failure "$dep is not installed"
            all_present=false
        fi
    done

    if [ "$all_present" = "false" ]; then
        echo ""
        echo -e "${RED}Missing dependencies. Please install:${NC}"
        echo "  brew install curl jq  # macOS"
        echo "  apt-get install curl jq  # Ubuntu"
        exit 1
    fi
}

# Test 1: Health Check
test_health_check() {
    print_header "Test 1: Health Check"
    print_test "Checking ${API_URL}/health"

    local response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT "${API_URL}/health" 2>/dev/null || echo "000")
    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | sed '$d')

    verbose "HTTP Code: $http_code"
    verbose "Response: $body"

    if [ "$http_code" -eq 200 ]; then
        print_success "Health endpoint returned 200 OK"

        # Parse health response
        local status=$(echo "$body" | jq -r '.status' 2>/dev/null || echo "unknown")
        local version=$(echo "$body" | jq -r '.version' 2>/dev/null || echo "unknown")

        print_info "Status: $status"
        print_info "Version: $version"

        if [ "$status" = "healthy" ]; then
            print_success "Service status is 'healthy'"
        elif [ "$status" = "degraded" ]; then
            print_warning "Service status is 'degraded' - check component health"
        else
            print_failure "Service status is '$status' (expected 'healthy' or 'degraded')"
        fi

        # Check components
        local db_status=$(echo "$body" | jq -r '.components.database.status' 2>/dev/null || echo "unknown")
        if [ "$db_status" = "healthy" ]; then
            print_success "Database component is healthy"
        else
            print_warning "Database component status: $db_status"
        fi

    elif [ "$http_code" -eq 000 ]; then
        print_failure "Cannot connect to API (connection timeout or refused)"
        echo ""
        echo -e "${RED}Deployment verification failed: API is not reachable${NC}"
        echo "Please check:"
        echo "  1. Is the service running?"
        echo "  2. Is the URL correct? (provided: ${API_URL})"
        echo "  3. Are there firewall rules blocking access?"
        exit 1
    else
        print_failure "Health endpoint returned HTTP $http_code (expected 200)"
    fi
}

# Test 2: Metrics Endpoint
test_metrics_endpoint() {
    print_header "Test 2: Metrics Endpoint"
    print_test "Checking ${API_URL}/metrics"

    local response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT "${API_URL}/metrics" 2>/dev/null || echo "000")
    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | sed '$d')

    verbose "HTTP Code: $http_code"

    if [ "$http_code" -eq 200 ]; then
        print_success "Metrics endpoint returned 200 OK"

        # Check for expected metrics
        if echo "$body" | grep -q "supplychain_events_ingested_total"; then
            print_success "Found 'supplychain_events_ingested_total' metric"
        else
            print_failure "Missing 'supplychain_events_ingested_total' metric"
        fi

        if echo "$body" | grep -q "supplychain_api_request_duration_seconds"; then
            print_success "Found 'supplychain_api_request_duration_seconds' metric"
        else
            print_failure "Missing 'supplychain_api_request_duration_seconds' metric"
        fi

        if echo "$body" | grep -q "supplychain_claims_stored_total"; then
            print_success "Found 'supplychain_claims_stored_total' metric"
        else
            print_failure "Missing 'supplychain_claims_stored_total' metric"
        fi

    else
        print_failure "Metrics endpoint returned HTTP $http_code (expected 200)"
    fi
}

# Test 3: Create Test Event
test_create_event() {
    print_header "Test 3: Create Test Event"
    print_test "Creating test supply chain event"

    local event_json=$(cat <<EOF
{
  "@context": ["https://www.w3.org/2018/credentials/v1"],
  "type": ["VerifiableCredential"],
  "issuer": "did:mycelix:org:deployment-verification",
  "issuanceDate": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "credentialSubject": {
    "eventType": "PRODUCED",
    "productId": "${TEST_PRODUCT_ID}",
    "batchId": "${TEST_BATCH_ID}",
    "quantity": 100.0,
    "unit": "kg",
    "facility": {
      "id": "VERIFY-FACILITY",
      "name": "Deployment Verification Facility",
      "location": {
        "country": "USA",
        "region": "California",
        "city": "San Francisco"
      }
    },
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "metadata": {
      "purpose": "deployment-verification",
      "automated": true
    }
  }
}
EOF
)

    verbose "Event JSON:"
    verbose "$event_json"

    local start_time=$(date +%s%3N)
    local response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT \
        -X POST "${API_URL}/v1/events" \
        -H "Content-Type: application/json" \
        -d "$event_json" 2>/dev/null || echo "000")
    local end_time=$(date +%s%3N)
    local duration=$((end_time - start_time))

    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | sed '$d')

    verbose "HTTP Code: $http_code"
    verbose "Response: $body"
    verbose "Duration: ${duration}ms"

    if [ "$http_code" -eq 201 ]; then
        print_success "Event created (HTTP 201)"
        print_info "Response time: ${duration}ms"

        # Extract claim_id for later tests
        CLAIM_ID=$(echo "$body" | jq -r '.claim_id' 2>/dev/null)

        if [ -n "$CLAIM_ID" ] && [ "$CLAIM_ID" != "null" ]; then
            print_success "Received claim_id: $CLAIM_ID"

            # Check for lineage_hash
            local lineage_hash=$(echo "$body" | jq -r '.lineage_hash' 2>/dev/null)
            if [ -n "$lineage_hash" ] && [ "$lineage_hash" != "null" ]; then
                print_success "Lineage hash present: ${lineage_hash:0:16}..."
            else
                print_warning "No lineage_hash in response"
            fi
        else
            print_failure "No claim_id in response"
        fi

        # Performance check
        if [ "$duration" -lt 100 ]; then
            print_success "Performance: <100ms (excellent)"
        elif [ "$duration" -lt 200 ]; then
            print_success "Performance: <200ms (good)"
        else
            print_warning "Performance: ${duration}ms (slower than expected <200ms)"
        fi

    else
        print_failure "Event creation returned HTTP $http_code (expected 201)"
        verbose "Error: $body"
    fi
}

# Test 4: Retrieve Test Claim
test_retrieve_claim() {
    print_header "Test 4: Retrieve Test Claim"

    if [ -z "$CLAIM_ID" ]; then
        print_warning "Skipping claim retrieval (no claim_id from previous test)"
        return
    fi

    print_test "Retrieving claim: $CLAIM_ID"

    local start_time=$(date +%s%3N)
    local response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT \
        "${API_URL}/v1/claims/${CLAIM_ID}" 2>/dev/null || echo "000")
    local end_time=$(date +%s%3N)
    local duration=$((end_time - start_time))

    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | sed '$d')

    verbose "HTTP Code: $http_code"
    verbose "Duration: ${duration}ms"

    if [ "$http_code" -eq 200 ]; then
        print_success "Claim retrieved (HTTP 200)"
        print_info "Response time: ${duration}ms"

        # Validate claim structure
        local batch_id=$(echo "$body" | jq -r '.credentialSubject.batchId' 2>/dev/null)
        if [ "$batch_id" = "$TEST_BATCH_ID" ]; then
            print_success "Claim contains correct batch_id"
        else
            print_failure "Claim batch_id mismatch (expected: $TEST_BATCH_ID, got: $batch_id)"
        fi

        local product_id=$(echo "$body" | jq -r '.credentialSubject.productId' 2>/dev/null)
        if [ "$product_id" = "$TEST_PRODUCT_ID" ]; then
            print_success "Claim contains correct product_id"
        else
            print_failure "Claim product_id mismatch"
        fi

        # Check for proof
        local proof=$(echo "$body" | jq -r '.proof' 2>/dev/null)
        if [ -n "$proof" ] && [ "$proof" != "null" ]; then
            print_success "Claim includes cryptographic proof"
        else
            print_failure "Claim missing cryptographic proof"
        fi

    else
        print_failure "Claim retrieval returned HTTP $http_code (expected 200)"
    fi
}

# Test 5: Verify Lineage
test_lineage() {
    print_header "Test 5: Verify Lineage"

    if [ -z "$CLAIM_ID" ]; then
        print_warning "Skipping lineage verification (no claim_id)"
        return
    fi

    print_test "Verifying lineage for batch: $TEST_BATCH_ID"

    # Note: This assumes a /v1/lineage/:batch_id endpoint exists
    # If not implemented, we skip this test
    local response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT \
        "${API_URL}/v1/lineage/${TEST_BATCH_ID}" 2>/dev/null || echo "000")
    local http_code=$(echo "$response" | tail -n1)

    if [ "$http_code" -eq 200 ]; then
        local body=$(echo "$response" | sed '$d')
        print_success "Lineage query successful"

        local claim_count=$(echo "$body" | jq 'length' 2>/dev/null || echo "0")
        print_info "Found $claim_count claim(s) in lineage"

    elif [ "$http_code" -eq 404 ]; then
        print_info "Lineage endpoint not implemented (404) - skipping"
    else
        print_warning "Lineage query returned HTTP $http_code"
    fi
}

# Test 6: Security Headers
test_security_headers() {
    print_header "Test 6: Security Headers"
    print_test "Checking security headers"

    local headers=$(curl -s -I --max-time $TIMEOUT "${API_URL}/health" 2>/dev/null)

    verbose "Headers:"
    verbose "$headers"

    # Check for important security headers
    if echo "$headers" | grep -qi "X-Frame-Options"; then
        print_success "X-Frame-Options header present"
    else
        print_warning "X-Frame-Options header missing"
    fi

    if echo "$headers" | grep -qi "X-Content-Type-Options"; then
        print_success "X-Content-Type-Options header present"
    else
        print_warning "X-Content-Type-Options header missing"
    fi

    if echo "$headers" | grep -qi "X-XSS-Protection"; then
        print_success "X-XSS-Protection header present"
    else
        print_warning "X-XSS-Protection header missing"
    fi

    if echo "$headers" | grep -qi "Content-Security-Policy"; then
        print_success "Content-Security-Policy header present"
    else
        print_warning "Content-Security-Policy header missing"
    fi

    # Check for HSTS (only on HTTPS)
    if [[ "$API_URL" == https://* ]]; then
        if echo "$headers" | grep -qi "Strict-Transport-Security"; then
            print_success "Strict-Transport-Security header present (HTTPS)"
        else
            print_warning "Strict-Transport-Security header missing (recommended for HTTPS)"
        fi
    else
        print_info "Skipping HSTS check (not HTTPS)"
    fi
}

# Test 7: Error Handling
test_error_handling() {
    print_header "Test 7: Error Handling"
    print_test "Testing invalid request handling"

    # Test 1: Invalid JSON
    local response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT \
        -X POST "${API_URL}/v1/events" \
        -H "Content-Type: application/json" \
        -d "invalid json" 2>/dev/null || echo "000")
    local http_code=$(echo "$response" | tail -n1)

    if [ "$http_code" -eq 400 ] || [ "$http_code" -eq 422 ]; then
        print_success "Invalid JSON returns 4xx error (HTTP $http_code)"
    else
        print_failure "Invalid JSON returns HTTP $http_code (expected 400 or 422)"
    fi

    # Test 2: Non-existent claim
    local response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT \
        "${API_URL}/v1/claims/non-existent-claim-id-12345" 2>/dev/null || echo "000")
    local http_code=$(echo "$response" | tail -n1)

    if [ "$http_code" -eq 404 ]; then
        print_success "Non-existent claim returns 404"
    else
        print_failure "Non-existent claim returns HTTP $http_code (expected 404)"
    fi

    # Test 3: Missing required fields
    local invalid_event='{"@context": ["https://www.w3.org/2018/credentials/v1"], "type": ["VerifiableCredential"]}'
    local response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT \
        -X POST "${API_URL}/v1/events" \
        -H "Content-Type: application/json" \
        -d "$invalid_event" 2>/dev/null || echo "000")
    local http_code=$(echo "$response" | tail -n1)

    if [ "$http_code" -eq 400 ] || [ "$http_code" -eq 422 ]; then
        print_success "Missing required fields returns 4xx error (HTTP $http_code)"
    else
        print_failure "Missing required fields returns HTTP $http_code (expected 400 or 422)"
    fi
}

# Test 8: CORS Configuration
test_cors() {
    print_header "Test 8: CORS Configuration"
    print_test "Checking CORS headers"

    local headers=$(curl -s -I --max-time $TIMEOUT \
        -H "Origin: https://example.com" \
        -H "Access-Control-Request-Method: POST" \
        "${API_URL}/health" 2>/dev/null)

    verbose "CORS Headers:"
    verbose "$headers"

    if echo "$headers" | grep -qi "Access-Control-Allow-Origin"; then
        local cors_origin=$(echo "$headers" | grep -i "Access-Control-Allow-Origin" | cut -d: -f2- | tr -d '[:space:]')
        print_success "CORS enabled: Access-Control-Allow-Origin: $cors_origin"

        if [ "$cors_origin" = "*" ]; then
            print_warning "CORS allows all origins (*) - consider restricting in production"
        fi
    else
        print_info "CORS headers not detected (may not be configured)"
    fi
}

# Summary
print_summary() {
    print_header "Verification Summary"

    echo -e "${BOLD}Total Tests:${NC} $TOTAL_TESTS"
    echo -e "${GREEN}${BOLD}Passed:${NC}      $TESTS_PASSED"
    echo -e "${RED}${BOLD}Failed:${NC}      $TESTS_FAILED"
    echo ""

    local pass_rate=$((TESTS_PASSED * 100 / TOTAL_TESTS))

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}${BOLD}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}${BOLD}║         ✓ ALL CHECKS PASSED - DEPLOYMENT VERIFIED         ║${NC}"
        echo -e "${GREEN}${BOLD}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${GREEN}Deployment is healthy and ready for production traffic.${NC}"
        exit 0
    elif [ $pass_rate -ge 80 ]; then
        echo -e "${YELLOW}${BOLD}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${YELLOW}${BOLD}║      ⚠ DEPLOYMENT VERIFIED WITH WARNINGS ($pass_rate% passed)       ║${NC}"
        echo -e "${YELLOW}${BOLD}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${YELLOW}Deployment is functional but has some issues to address.${NC}"
        echo "Please review the failures and warnings above."
        exit 0
    else
        echo -e "${RED}${BOLD}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}${BOLD}║     ✗ DEPLOYMENT VERIFICATION FAILED ($pass_rate% passed)          ║${NC}"
        echo -e "${RED}${BOLD}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${RED}Critical issues detected. Please review and fix before proceeding.${NC}"
        exit 1
    fi
}

# Main execution
main() {
    echo ""
    echo -e "${BLUE}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}${BOLD}║       Mycelix Supply Chain Deployment Verification          ║${NC}"
    echo -e "${BLUE}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}API URL:${NC} $API_URL"
    echo -e "${BOLD}Timestamp:${NC} $(date)"
    echo ""

    check_dependencies
    test_health_check
    test_metrics_endpoint
    test_create_event
    test_retrieve_claim
    test_lineage
    test_security_headers
    test_error_handling
    test_cors
    print_summary
}

# Run main function
main
