#!/usr/bin/env bash
#
# Mycelix Deployment Pre-flight Check
#
# This script validates the environment before deployment to ensure:
# - No exposed credentials in the codebase
# - Required environment variables are set
# - Dependencies are available
# - Configuration files are valid
#
# Usage: ./scripts/deployment-preflight.sh [--env FILE] [--fix] [--verbose]
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
ERRORS=0
WARNINGS=0
PASSED=0

# Options
ENV_FILE=""
FIX_MODE=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV_FILE="$2"
            shift 2
            ;;
        --fix)
            FIX_MODE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--env FILE] [--fix] [--verbose]"
            echo ""
            echo "Options:"
            echo "  --env FILE    Specify environment file to validate"
            echo "  --fix         Attempt to fix issues automatically"
            echo "  --verbose     Show detailed output"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging functions
log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((++PASSED))
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((++WARNINGS))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((++ERRORS))
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_verbose() {
    if $VERBOSE; then
        echo -e "       $1"
    fi
}

# Header
echo ""
echo "=============================================="
echo "  Mycelix Deployment Pre-flight Check"
echo "=============================================="
echo ""

# Determine repository root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
log_info "Repository root: $REPO_ROOT"
echo ""

# =============================================================================
# Section 1: Credential Exposure Check
# =============================================================================
echo "--- Section 1: Credential Exposure Check ---"
echo ""

# Check for exposed JWT tokens (only in config files, not entire codebase)
log_info "Checking for exposed JWT tokens..."
if find "$REPO_ROOT" -maxdepth 3 -type f \( -name "*.env*" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" \) \
    ! -path "*/target/*" ! -path "*/.git/*" ! -path "*/node_modules/*" ! -path "*/*.example" \
    -exec grep -l "eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\." {} \; 2>/dev/null | head -1 | grep -q .; then
    log_fail "Exposed JWT token found in repository"
    log_verbose "Run: grep -r 'eyJ' --include='*.env*' to find locations"
else
    log_pass "No exposed JWT tokens found"
fi

# Check for Supabase keys (only in env files)
log_info "Checking for exposed Supabase keys..."
if find "$REPO_ROOT" -maxdepth 3 -type f -name "*.env*" ! -name "*.example" \
    -exec grep -l "sb-[a-z]*-[A-Za-z0-9]*" {} \; 2>/dev/null | head -1 | grep -q .; then
    log_fail "Exposed Supabase key found in repository"
else
    log_pass "No exposed Supabase keys found"
fi

# Check for age identity keys (only in specific locations)
log_info "Checking for exposed age identity keys..."
if find "$REPO_ROOT" -maxdepth 4 -type f \( -name "*.env*" -o -name ".age-identity" -o -name "*.age" \) \
    -exec grep -l "AGE-SECRET-KEY-" {} \; 2>/dev/null | head -1 | grep -q .; then
    log_fail "Exposed age identity key found in repository"
else
    log_pass "No exposed age identity keys found"
fi

# Check for private keys (only in key/pem files)
log_info "Checking for exposed private keys..."
if find "$REPO_ROOT" -maxdepth 3 -type f \( -name "*.pem" -o -name "*.key" -o -name "*.env*" \) ! -name "*.example" \
    -exec grep -l "PRIVATE.*KEY\|-----BEGIN.*PRIVATE" {} \; 2>/dev/null | head -1 | grep -q .; then
    log_warn "Potential private key found - verify it's not committed"
else
    log_pass "No obvious private keys in tracked files"
fi

# Check for common weak passwords in env files (quick check)
log_info "Checking for weak passwords..."
FOUND_WEAK=false
if find "$REPO_ROOT" -maxdepth 2 -type f -name "*.env*" ! -name "*.example" \
    -exec grep -Eiq "PASSWORD.*(password|admin|changeme|secret|test123)" {} \; 2>/dev/null; then
    FOUND_WEAK=true
fi
if $FOUND_WEAK; then
    log_fail "Weak password detected in environment files"
else
    log_pass "No weak passwords detected"
fi

echo ""

# =============================================================================
# Section 2: Environment Variables
# =============================================================================
echo "--- Section 2: Environment Variables ---"
echo ""

# Required variables for deployment
REQUIRED_VARS=(
    "POSTGRES_PASSWORD"
    "GRAFANA_PASSWORD"
)

RECOMMENDED_VARS=(
    "JWT_SECRET"
    "REDIS_PASSWORD"
    "ETH_RPC_URL"
)

if [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]]; then
    log_info "Validating environment file: $ENV_FILE"

    # Source the env file to check variables
    set +u  # Allow unset variables temporarily
    source "$ENV_FILE" 2>/dev/null || true
    set -u

    for var in "${REQUIRED_VARS[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_fail "Required variable not set: $var"
        else
            # Check if it's a placeholder
            if [[ "${!var}" == *"CHANGEME"* || "${!var}" == *"your_"* || "${!var}" == *"<"* ]]; then
                log_fail "$var contains placeholder value"
            else
                log_pass "$var is set"
            fi
        fi
    done

    for var in "${RECOMMENDED_VARS[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_warn "Recommended variable not set: $var"
        else
            log_pass "$var is set"
        fi
    done
else
    log_info "No environment file specified (use --env FILE)"
    log_warn "Skipping environment variable validation"
fi

echo ""

# =============================================================================
# Section 3: Git Status
# =============================================================================
echo "--- Section 3: Git Status ---"
echo ""

# Check for .env files in git
log_info "Checking if .env files are tracked by git..."
if git -C "$REPO_ROOT" ls-files --error-unmatch "*.env" "*.env.local" "*.env.development" 2>/dev/null; then
    log_fail ".env files are tracked by git - remove them!"
else
    log_pass "No .env files tracked by git"
fi

# Check .gitignore
log_info "Checking .gitignore for secrets patterns..."
if [[ -f "$REPO_ROOT/.gitignore" ]]; then
    PATTERNS_TO_CHECK=(".env" ".env.local" "*.pem" "*.key" ".age-identity")
    ALL_IGNORED=true
    for pattern in "${PATTERNS_TO_CHECK[@]}"; do
        if ! grep -q "$pattern" "$REPO_ROOT/.gitignore" 2>/dev/null; then
            log_warn "Pattern not in .gitignore: $pattern"
            ALL_IGNORED=false
        fi
    done
    if $ALL_IGNORED; then
        log_pass "All sensitive patterns in .gitignore"
    fi
else
    log_fail "No .gitignore file found"
fi

# Check for uncommitted changes to sensitive files
log_info "Checking for uncommitted sensitive files..."
if git -C "$REPO_ROOT" status --porcelain 2>/dev/null | grep -E "\.env|\.pem|\.key|credentials" > /dev/null; then
    log_warn "Uncommitted changes to potentially sensitive files"
else
    log_pass "No uncommitted sensitive files"
fi

echo ""

# =============================================================================
# Section 4: Dependency Check
# =============================================================================
echo "--- Section 4: Dependencies ---"
echo ""

# Check for required tools
REQUIRED_TOOLS=("docker" "git" "curl")
OPTIONAL_TOOLS=("kubectl" "helm" "age" "sops")

for tool in "${REQUIRED_TOOLS[@]}"; do
    if command -v "$tool" &> /dev/null; then
        log_pass "$tool is installed"
    else
        log_fail "$tool is required but not installed"
    fi
done

for tool in "${OPTIONAL_TOOLS[@]}"; do
    if command -v "$tool" &> /dev/null; then
        log_pass "$tool is installed"
    else
        log_warn "$tool is recommended but not installed"
    fi
done

# Check Docker is running
if docker info &> /dev/null; then
    log_pass "Docker daemon is running"
else
    log_fail "Docker daemon is not running"
fi

echo ""

# =============================================================================
# Section 5: Configuration Files
# =============================================================================
echo "--- Section 5: Configuration Files ---"
echo ""

# Check for required deployment files
DEPLOYMENT_FILES=(
    "deployment/docker-compose.yml"
    "deployment/prometheus.yml"
    "deployment/alertmanager.yml"
)

for file in "${DEPLOYMENT_FILES[@]}"; do
    if [[ -f "$REPO_ROOT/$file" ]]; then
        log_pass "Found: $file"
    else
        log_warn "Missing: $file"
    fi
done

# Check for hardcoded secrets in docker-compose
log_info "Checking docker-compose for hardcoded secrets..."
if find "$REPO_ROOT/deployment" -maxdepth 2 -type f \( -name "*.yml" -o -name "*.yaml" \) ! -name "*.example*" \
    -exec grep -l "password:.*['\"][^$]" {} \; 2>/dev/null | head -1 | grep -q .; then
    log_warn "Potential hardcoded password in docker-compose files"
else
    log_pass "No hardcoded passwords in docker-compose"
fi

echo ""

# =============================================================================
# Section 6: SSL/TLS Certificates
# =============================================================================
echo "--- Section 6: SSL/TLS ---"
echo ""

# Check for self-signed certificates in production config
if [[ -d "$REPO_ROOT/deployment/certs" ]]; then
    log_info "Checking certificate directory..."
    if find "$REPO_ROOT/deployment/certs" -name "*.pem" -o -name "*.crt" 2>/dev/null | head -1 | xargs -I {} openssl x509 -in {} -noout -issuer 2>/dev/null | grep -qi "self"; then
        log_warn "Self-signed certificates detected - OK for testnet, not for mainnet"
    else
        log_pass "No self-signed certificates in deployment/certs"
    fi
else
    log_info "No certs directory found (may use cert-manager)"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "  Pre-flight Check Summary"
echo "=============================================="
echo ""
echo -e "  ${GREEN}Passed:${NC}   $PASSED"
echo -e "  ${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "  ${RED}Errors:${NC}   $ERRORS"
echo ""

if [[ $ERRORS -gt 0 ]]; then
    echo -e "${RED}❌ Pre-flight check FAILED${NC}"
    echo ""
    echo "Please fix the errors above before deploying."
    echo "Run with --verbose for more details."
    exit 1
elif [[ $WARNINGS -gt 0 ]]; then
    echo -e "${YELLOW}⚠️  Pre-flight check PASSED with warnings${NC}"
    echo ""
    echo "Review warnings above. Deployment can proceed with caution."
    exit 0
else
    echo -e "${GREEN}✅ Pre-flight check PASSED${NC}"
    echo ""
    echo "All checks passed. Ready for deployment!"
    exit 0
fi
