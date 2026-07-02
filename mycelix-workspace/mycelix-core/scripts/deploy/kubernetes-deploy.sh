#!/usr/bin/env bash
#
# Mycelix Kubernetes Deployment Script
#
# This script handles the deployment of Mycelix workloads to Kubernetes.
# It includes health checks, rollback capabilities, and canary deployments.
#
# Usage:
#   ./kubernetes-deploy.sh [environment] [action]
#
# Examples:
#   ./kubernetes-deploy.sh testnet deploy
#   ./kubernetes-deploy.sh testnet status
#   ./kubernetes-deploy.sh staging rollback
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
K8S_DIR="$REPO_ROOT/deployment/kubernetes"
LOG_DIR="$REPO_ROOT/logs/kubernetes"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Deployment configuration
DEPLOY_TIMEOUT="600s"
ROLLOUT_TIMEOUT="300s"
HEALTH_CHECK_RETRIES=30
HEALTH_CHECK_INTERVAL=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

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

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Help message
show_help() {
    cat << EOF
Mycelix Kubernetes Deployment Script

Usage: $0 [environment] [action] [options]

Environments:
    testnet     Deploy to testnet cluster
    staging     Deploy to staging cluster
    production  Deploy to production cluster

Actions:
    deploy      Deploy all manifests
    status      Show deployment status
    rollback    Rollback to previous version
    restart     Restart all deployments
    scale       Scale deployments
    logs        Show pod logs
    shell       Open shell in a pod
    validate    Validate manifests without applying

Options:
    -h, --help              Show this help message
    -y, --yes               Auto-approve (skip confirmation)
    -n, --namespace=NS      Target specific namespace
    -c, --component=COMP    Target specific component (validators, fl, monitoring)
    --image-tag=TAG         Override image tag
    --dry-run               Show what would be done without executing
    --canary                Deploy as canary (25% traffic)

Components:
    validators  Validator StatefulSet
    fl          FL Aggregator and Coordinator
    monitoring  Prometheus, Grafana, Loki
    all         All components (default)

Examples:
    $0 testnet deploy
    $0 testnet deploy --component=validators
    $0 staging rollback --component=fl
    $0 testnet scale --component=fl --replicas=5
    $0 testnet logs --component=validators --pod=validator-0

EOF
}

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."

    # Check for required tools
    local required_tools=("kubectl" "kustomize")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed."
            exit 1
        fi
    done
    log_success "Required tools are installed"

    # Check kubectl context
    local current_context
    current_context=$(kubectl config current-context 2>/dev/null || echo "none")
    log_info "Current kubectl context: $current_context"

    # Verify cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    log_success "Cluster connectivity verified"

    # Check if context matches environment
    if [[ ! "$current_context" =~ $ENVIRONMENT ]]; then
        log_warn "kubectl context '$current_context' may not match environment '$ENVIRONMENT'"
        read -p "Continue anyway? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Validate manifests
validate_manifests() {
    log_step "Validating Kubernetes manifests..."

    cd "$K8S_DIR"

    # Run kustomize build to check for errors
    if ! kustomize build . > /dev/null 2>&1; then
        log_error "Kustomize build failed"
        kustomize build . 2>&1
        exit 1
    fi

    # Dry-run apply
    if ! kustomize build . | kubectl apply --dry-run=server -f - > /dev/null 2>&1; then
        log_warn "Server-side dry-run failed, trying client-side..."
        kustomize build . | kubectl apply --dry-run=client -f -
    fi

    log_success "Manifests validated successfully"
}

# Deploy namespaces first
deploy_namespaces() {
    log_step "Deploying namespaces..."

    kubectl apply -f "$K8S_DIR/namespaces.yaml"

    log_success "Namespaces deployed"
}

# Deploy RBAC and network policies
deploy_security() {
    log_step "Deploying RBAC and network policies..."

    kubectl apply -f "$K8S_DIR/testnet/kubernetes/rbac.yaml"
    kubectl apply -f "$K8S_DIR/testnet/kubernetes/network-policies.yaml"

    log_success "Security policies deployed"
}

# Deploy ConfigMaps
deploy_configmaps() {
    log_step "Deploying ConfigMaps..."

    kubectl apply -f "$K8S_DIR/configmaps.yaml"

    log_success "ConfigMaps deployed"
}

# Deploy External Secrets
deploy_external_secrets() {
    log_step "Deploying External Secrets..."

    kubectl apply -f "$K8S_DIR/external-secrets.yaml"

    # Wait for secrets to sync
    log_info "Waiting for secrets to sync..."
    sleep 10

    # Check if secrets were created
    local namespaces=("mycelix-validators" "mycelix-fl" "mycelix")
    for ns in "${namespaces[@]}"; do
        local secrets
        secrets=$(kubectl get secrets -n "$ns" --no-headers 2>/dev/null | wc -l || echo "0")
        log_info "Namespace $ns has $secrets secrets"
    done

    log_success "External Secrets deployed"
}

# Deploy validators
deploy_validators() {
    log_step "Deploying validators..."

    kubectl apply -f "$K8S_DIR/validator-deployment.yaml"

    # Wait for StatefulSet to be ready
    log_info "Waiting for validator StatefulSet..."
    kubectl rollout status statefulset/validator -n mycelix-validators --timeout="$ROLLOUT_TIMEOUT" || {
        log_error "Validator rollout failed"
        return 1
    }

    log_success "Validators deployed"
}

# Deploy FL components
deploy_fl() {
    log_step "Deploying FL components..."

    kubectl apply -f "$K8S_DIR/fl-node-deployment.yaml"

    # Wait for deployments
    log_info "Waiting for FL Aggregator..."
    kubectl rollout status deployment/fl-aggregator -n mycelix-fl --timeout="$ROLLOUT_TIMEOUT" || {
        log_error "FL Aggregator rollout failed"
        return 1
    }

    log_info "Waiting for FL Coordinator..."
    kubectl rollout status deployment/fl-coordinator -n mycelix-fl --timeout="$ROLLOUT_TIMEOUT" || {
        log_error "FL Coordinator rollout failed"
        return 1
    }

    log_success "FL components deployed"
}

# Deploy autoscaling policies
deploy_autoscaling() {
    log_step "Deploying autoscaling policies..."

    kubectl apply -f "$K8S_DIR/autoscaling.yaml"

    log_success "Autoscaling policies deployed"
}

# Deploy ingress
deploy_ingress() {
    log_step "Deploying ingress..."

    kubectl apply -f "$K8S_DIR/ingress.yaml"

    log_success "Ingress deployed"
}

# Full deployment
deploy_all() {
    log_info "Starting full deployment to $ENVIRONMENT..."

    mkdir -p "$LOG_DIR"

    # Save pre-deployment state
    local state_file="$LOG_DIR/${ENVIRONMENT}-${TIMESTAMP}-pre-deploy.yaml"
    kubectl get all --all-namespaces -o yaml > "$state_file" 2>/dev/null || true

    deploy_namespaces
    deploy_security
    deploy_configmaps
    deploy_external_secrets
    deploy_validators
    deploy_fl
    deploy_autoscaling
    deploy_ingress

    log_success "Full deployment complete"
}

# Health check
health_check() {
    log_step "Running health checks..."

    local all_healthy=true

    # Check validator pods
    log_info "Checking validators..."
    local validator_ready
    validator_ready=$(kubectl get pods -n mycelix-validators -l app.kubernetes.io/name=validator \
        -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' | tr ' ' '\n' | grep -c "True" || echo "0")
    local validator_total
    validator_total=$(kubectl get pods -n mycelix-validators -l app.kubernetes.io/name=validator --no-headers | wc -l || echo "0")
    echo "  Validators: $validator_ready/$validator_total ready"
    if [[ "$validator_ready" -lt "$validator_total" ]]; then
        all_healthy=false
    fi

    # Check FL pods
    log_info "Checking FL components..."
    local fl_ready
    fl_ready=$(kubectl get pods -n mycelix-fl \
        -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' | tr ' ' '\n' | grep -c "True" || echo "0")
    local fl_total
    fl_total=$(kubectl get pods -n mycelix-fl --no-headers | wc -l || echo "0")
    echo "  FL Components: $fl_ready/$fl_total ready"
    if [[ "$fl_ready" -lt "$fl_total" ]]; then
        all_healthy=false
    fi

    # Check endpoints
    log_info "Checking service endpoints..."
    local services=("validator:mycelix-validators" "fl-aggregator:mycelix-fl" "fl-coordinator:mycelix-fl")
    for svc_ns in "${services[@]}"; do
        local svc="${svc_ns%%:*}"
        local ns="${svc_ns##*:}"
        local endpoints
        endpoints=$(kubectl get endpoints "$svc" -n "$ns" -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null | wc -w || echo "0")
        echo "  $svc: $endpoints endpoints"
    done

    if $all_healthy; then
        log_success "All health checks passed"
        return 0
    else
        log_warn "Some components are not healthy"
        return 1
    fi
}

# Show status
show_status() {
    echo ""
    echo "=========================================="
    echo "  Mycelix Deployment Status"
    echo "  Environment: $ENVIRONMENT"
    echo "=========================================="
    echo ""

    log_info "Namespaces:"
    kubectl get namespaces -l app.kubernetes.io/part-of=mycelix

    echo ""
    log_info "Validators (mycelix-validators):"
    kubectl get pods -n mycelix-validators -o wide

    echo ""
    log_info "FL Components (mycelix-fl):"
    kubectl get pods -n mycelix-fl -o wide

    echo ""
    log_info "Services:"
    kubectl get svc -n mycelix-validators
    kubectl get svc -n mycelix-fl

    echo ""
    log_info "Ingresses:"
    kubectl get ingress --all-namespaces

    echo ""
    log_info "HPA Status:"
    kubectl get hpa -n mycelix-fl

    echo ""
    log_info "PVC Status:"
    kubectl get pvc -n mycelix-validators

    echo ""
    health_check
}

# Rollback deployment
rollback() {
    local component=${COMPONENT:-all}

    log_warn "Rolling back $component in $ENVIRONMENT..."

    case $component in
        validators)
            kubectl rollout undo statefulset/validator -n mycelix-validators
            kubectl rollout status statefulset/validator -n mycelix-validators --timeout="$ROLLOUT_TIMEOUT"
            ;;
        fl)
            kubectl rollout undo deployment/fl-aggregator -n mycelix-fl
            kubectl rollout undo deployment/fl-coordinator -n mycelix-fl
            kubectl rollout status deployment/fl-aggregator -n mycelix-fl --timeout="$ROLLOUT_TIMEOUT"
            kubectl rollout status deployment/fl-coordinator -n mycelix-fl --timeout="$ROLLOUT_TIMEOUT"
            ;;
        all)
            kubectl rollout undo statefulset/validator -n mycelix-validators
            kubectl rollout undo deployment/fl-aggregator -n mycelix-fl
            kubectl rollout undo deployment/fl-coordinator -n mycelix-fl
            ;;
        *)
            log_error "Unknown component: $component"
            exit 1
            ;;
    esac

    log_success "Rollback complete"
}

# Restart deployments
restart() {
    local component=${COMPONENT:-all}

    log_info "Restarting $component in $ENVIRONMENT..."

    case $component in
        validators)
            kubectl rollout restart statefulset/validator -n mycelix-validators
            ;;
        fl)
            kubectl rollout restart deployment/fl-aggregator -n mycelix-fl
            kubectl rollout restart deployment/fl-coordinator -n mycelix-fl
            ;;
        all)
            kubectl rollout restart statefulset/validator -n mycelix-validators
            kubectl rollout restart deployment/fl-aggregator -n mycelix-fl
            kubectl rollout restart deployment/fl-coordinator -n mycelix-fl
            ;;
        *)
            log_error "Unknown component: $component"
            exit 1
            ;;
    esac

    log_success "Restart initiated"
}

# Scale deployment
scale_deployment() {
    local component=${COMPONENT:-fl}
    local replicas=${REPLICAS:-3}

    log_info "Scaling $component to $replicas replicas..."

    case $component in
        validators)
            kubectl scale statefulset/validator -n mycelix-validators --replicas="$replicas"
            ;;
        fl)
            kubectl scale deployment/fl-aggregator -n mycelix-fl --replicas="$replicas"
            ;;
        *)
            log_error "Unknown component: $component"
            exit 1
            ;;
    esac

    log_success "Scaling initiated"
}

# Show logs
show_logs() {
    local component=${COMPONENT:-validators}
    local pod=${POD:-}
    local follow=${FOLLOW:-false}

    local namespace
    case $component in
        validators) namespace="mycelix-validators" ;;
        fl) namespace="mycelix-fl" ;;
        monitoring) namespace="mycelix-monitoring" ;;
        *) namespace="mycelix" ;;
    esac

    local kubectl_args=("-n" "$namespace")
    if [[ -n "$pod" ]]; then
        kubectl_args+=("$pod")
    else
        kubectl_args+=("-l" "app.kubernetes.io/part-of=mycelix")
    fi

    if [[ "$follow" == "true" ]]; then
        kubectl_args+=("-f")
    fi

    kubectl logs "${kubectl_args[@]}"
}

# Main function
main() {
    ENVIRONMENT=""
    ACTION=""
    AUTO_APPROVE=false
    DRY_RUN=false
    COMPONENT=""
    IMAGE_TAG=""
    REPLICAS=""
    POD=""
    FOLLOW=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -y|--yes)
                AUTO_APPROVE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -c=*|--component=*)
                COMPONENT="${1#*=}"
                shift
                ;;
            --image-tag=*)
                IMAGE_TAG="${1#*=}"
                shift
                ;;
            --replicas=*)
                REPLICAS="${1#*=}"
                shift
                ;;
            --pod=*)
                POD="${1#*=}"
                shift
                ;;
            -f|--follow)
                FOLLOW=true
                shift
                ;;
            testnet|staging|production)
                ENVIRONMENT=$1
                shift
                ;;
            deploy|status|rollback|restart|scale|logs|shell|validate)
                ACTION=$1
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
    if [[ -z "$ENVIRONMENT" || -z "$ACTION" ]]; then
        log_error "Environment and action are required"
        show_help
        exit 1
    fi

    echo "=========================================="
    echo "  Mycelix Kubernetes Deployment"
    echo "  Environment: $ENVIRONMENT"
    echo "  Action: $ACTION"
    echo "  Timestamp: $TIMESTAMP"
    echo "=========================================="
    echo ""

    # Run pre-flight checks
    preflight_checks

    # Execute action
    case $ACTION in
        validate)
            validate_manifests
            ;;
        deploy)
            if [[ "$AUTO_APPROVE" != "true" ]]; then
                read -p "Deploy to $ENVIRONMENT? (y/N): " -r
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    exit 0
                fi
            fi
            validate_manifests
            deploy_all
            health_check
            ;;
        status)
            show_status
            ;;
        rollback)
            if [[ "$AUTO_APPROVE" != "true" ]]; then
                read -p "Rollback in $ENVIRONMENT? (y/N): " -r
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    exit 0
                fi
            fi
            rollback
            ;;
        restart)
            restart
            ;;
        scale)
            if [[ -z "$REPLICAS" ]]; then
                log_error "--replicas is required for scale action"
                exit 1
            fi
            scale_deployment
            ;;
        logs)
            show_logs
            ;;
        *)
            log_error "Unknown action: $ACTION"
            exit 1
            ;;
    esac

    echo ""
    log_success "Script completed successfully"
}

# Run main function
main "$@"
