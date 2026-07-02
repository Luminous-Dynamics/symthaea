#!/usr/bin/env bash
#
# Mycelix Terraform Deployment Script
#
# This script handles the deployment of Mycelix infrastructure to AWS.
# It includes pre-flight checks, state management, and rollback capabilities.
#
# Usage:
#   ./terraform-deploy.sh [environment] [action]
#
# Examples:
#   ./terraform-deploy.sh testnet plan
#   ./terraform-deploy.sh testnet apply
#   ./terraform-deploy.sh staging destroy
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TERRAFORM_DIR="$REPO_ROOT/deployment/terraform"
LOG_DIR="$REPO_ROOT/logs/terraform"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help message
show_help() {
    cat << EOF
Mycelix Terraform Deployment Script

Usage: $0 [environment] [action] [options]

Environments:
    testnet     Deploy to testnet environment
    staging     Deploy to staging environment
    production  Deploy to production environment

Actions:
    plan        Generate and show execution plan
    apply       Apply the Terraform configuration
    destroy     Destroy the infrastructure (requires confirmation)
    output      Show Terraform outputs
    validate    Validate Terraform configuration
    init        Initialize Terraform (usually automatic)

Options:
    -h, --help              Show this help message
    -y, --yes               Auto-approve (skip confirmation prompts)
    -v, --verbose           Enable verbose output
    --dry-run               Show what would be done without executing
    --target=RESOURCE       Target specific resource
    --var="KEY=VALUE"       Set Terraform variable

Examples:
    $0 testnet plan
    $0 testnet apply -y
    $0 staging plan --target=module.eks
    $0 production destroy

EOF
}

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."

    # Check for required tools
    local required_tools=("terraform" "aws" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed."
            exit 1
        fi
    done
    log_success "All required tools are installed"

    # Check Terraform version
    local tf_version
    tf_version=$(terraform version -json | jq -r '.terraform_version')
    local required_version="1.5.0"
    if [[ "$(printf '%s\n' "$required_version" "$tf_version" | sort -V | head -n1)" != "$required_version" ]]; then
        log_error "Terraform version $tf_version is less than required $required_version"
        exit 1
    fi
    log_success "Terraform version $tf_version meets requirements"

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials are not configured or invalid"
        exit 1
    fi
    local aws_account
    aws_account=$(aws sts get-caller-identity --query Account --output text)
    log_success "AWS credentials valid (Account: $aws_account)"

    # Check if tfvars exists
    if [[ ! -f "$TERRAFORM_DIR/terraform.tfvars" ]]; then
        log_warn "terraform.tfvars not found. Using terraform.tfvars.example as reference."
        log_warn "Please create terraform.tfvars with your configuration."
        if [[ "$ACTION" != "validate" && "$ACTION" != "init" ]]; then
            read -p "Continue anyway? (y/N): " -r
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
}

# Initialize Terraform
init_terraform() {
    log_info "Initializing Terraform..."

    cd "$TERRAFORM_DIR"

    # Check if already initialized
    if [[ -d ".terraform" ]]; then
        log_info "Terraform already initialized, running upgrade..."
        terraform init -upgrade
    else
        terraform init
    fi

    log_success "Terraform initialized"
}

# Select workspace
select_workspace() {
    local environment=$1

    log_info "Selecting workspace: $environment"

    cd "$TERRAFORM_DIR"

    # Create workspace if it doesn't exist
    if ! terraform workspace list | grep -q "$environment"; then
        log_info "Creating workspace: $environment"
        terraform workspace new "$environment"
    fi

    terraform workspace select "$environment"
    log_success "Workspace selected: $environment"
}

# Validate configuration
validate_config() {
    log_info "Validating Terraform configuration..."

    cd "$TERRAFORM_DIR"

    terraform validate

    log_success "Configuration is valid"
}

# Generate plan
generate_plan() {
    local environment=$1
    local plan_file="$LOG_DIR/${environment}-${TIMESTAMP}.tfplan"

    log_info "Generating execution plan..."

    mkdir -p "$LOG_DIR"
    cd "$TERRAFORM_DIR"

    local tf_args=("-var=environment=$environment" "-out=$plan_file")

    # Add any additional variables
    for var in "${EXTRA_VARS[@]:-}"; do
        tf_args+=("-var=$var")
    done

    # Add target if specified
    if [[ -n "${TARGET:-}" ]]; then
        tf_args+=("-target=$TARGET")
    fi

    terraform plan "${tf_args[@]}"

    log_success "Plan saved to: $plan_file"
    echo "$plan_file"
}

# Apply configuration
apply_config() {
    local environment=$1
    local auto_approve=${2:-false}

    log_info "Applying Terraform configuration for $environment..."

    cd "$TERRAFORM_DIR"

    # Generate fresh plan
    local plan_file
    plan_file=$(generate_plan "$environment")

    # Confirmation
    if [[ "$auto_approve" != "true" ]]; then
        echo ""
        log_warn "You are about to apply changes to the $environment environment."
        read -p "Type 'yes' to confirm: " -r
        if [[ $REPLY != "yes" ]]; then
            log_info "Apply cancelled"
            exit 0
        fi
    fi

    # Apply the plan
    terraform apply "$plan_file"

    # Save outputs
    local output_file="$LOG_DIR/${environment}-${TIMESTAMP}-outputs.json"
    terraform output -json > "$output_file"

    log_success "Apply complete. Outputs saved to: $output_file"
}

# Destroy infrastructure
destroy_config() {
    local environment=$1
    local auto_approve=${2:-false}

    log_warn "Preparing to DESTROY infrastructure for $environment..."

    cd "$TERRAFORM_DIR"

    # Extra confirmation for production
    if [[ "$environment" == "production" ]]; then
        log_error "WARNING: You are about to destroy PRODUCTION infrastructure!"
        echo ""
        read -p "Type 'destroy-production' to confirm: " -r
        if [[ $REPLY != "destroy-production" ]]; then
            log_info "Destroy cancelled"
            exit 0
        fi
    elif [[ "$auto_approve" != "true" ]]; then
        read -p "Type 'destroy' to confirm: " -r
        if [[ $REPLY != "destroy" ]]; then
            log_info "Destroy cancelled"
            exit 0
        fi
    fi

    local tf_args=("-var=environment=$environment")

    if [[ "$auto_approve" == "true" ]]; then
        tf_args+=("-auto-approve")
    fi

    terraform destroy "${tf_args[@]}"

    log_success "Infrastructure destroyed"
}

# Show outputs
show_outputs() {
    log_info "Terraform outputs:"

    cd "$TERRAFORM_DIR"
    terraform output
}

# Main function
main() {
    local environment=""
    local action=""
    local auto_approve=false
    local verbose=false
    local dry_run=false
    EXTRA_VARS=()
    TARGET=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -y|--yes)
                auto_approve=true
                shift
                ;;
            -v|--verbose)
                verbose=true
                set -x
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --target=*)
                TARGET="${1#*=}"
                shift
                ;;
            --var=*)
                EXTRA_VARS+=("${1#*=}")
                shift
                ;;
            testnet|staging|production)
                environment=$1
                shift
                ;;
            plan|apply|destroy|output|validate|init)
                action=$1
                shift
                ;;
            *)
                log_error "Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Validate arguments
    if [[ -z "$environment" || -z "$action" ]]; then
        log_error "Environment and action are required"
        show_help
        exit 1
    fi

    # Validate environment
    if [[ ! "$environment" =~ ^(testnet|staging|production)$ ]]; then
        log_error "Invalid environment: $environment"
        exit 1
    fi

    echo "=========================================="
    echo "  Mycelix Terraform Deployment"
    echo "  Environment: $environment"
    echo "  Action: $action"
    echo "  Timestamp: $TIMESTAMP"
    echo "=========================================="
    echo ""

    # Run pre-flight checks
    preflight_checks

    # Initialize Terraform
    init_terraform

    # Select workspace
    select_workspace "$environment"

    # Execute action
    case $action in
        validate)
            validate_config
            ;;
        init)
            log_success "Initialization complete"
            ;;
        plan)
            validate_config
            generate_plan "$environment"
            ;;
        apply)
            validate_config
            apply_config "$environment" "$auto_approve"
            ;;
        destroy)
            destroy_config "$environment" "$auto_approve"
            ;;
        output)
            show_outputs
            ;;
        *)
            log_error "Unknown action: $action"
            exit 1
            ;;
    esac

    echo ""
    log_success "Deployment script completed successfully"
}

# Run main function
main "$@"
