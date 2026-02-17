#!/usr/bin/env bash
# Mycelix Testnet - One-Command Deployment Script
# Usage: ./deploy-testnet.sh [docker|kubernetes|terraform] [options]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
DEPLOYMENT_TYPE="${1:-docker}"
CLOUD_PROVIDER="${CLOUD_PROVIDER:-gcp}"
ENVIRONMENT="${ENVIRONMENT:-testnet}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing_deps=()

    case "$DEPLOYMENT_TYPE" in
        docker)
            command -v docker >/dev/null 2>&1 || missing_deps+=("docker")
            command -v docker-compose >/dev/null 2>&1 || command -v docker >/dev/null 2>&1 || missing_deps+=("docker-compose")
            ;;
        kubernetes)
            command -v kubectl >/dev/null 2>&1 || missing_deps+=("kubectl")
            command -v kustomize >/dev/null 2>&1 || missing_deps+=("kustomize")
            ;;
        terraform)
            command -v terraform >/dev/null 2>&1 || missing_deps+=("terraform")
            case "$CLOUD_PROVIDER" in
                gcp) command -v gcloud >/dev/null 2>&1 || missing_deps+=("gcloud") ;;
                aws) command -v aws >/dev/null 2>&1 || missing_deps+=("aws-cli") ;;
                azure) command -v az >/dev/null 2>&1 || missing_deps+=("azure-cli") ;;
            esac
            ;;
    esac

    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install the missing dependencies and try again."
        exit 1
    fi

    log_success "All prerequisites met!"
}

# Generate secrets if not present
generate_secrets() {
    log_info "Generating secrets..."

    local secrets_dir="$PROJECT_DIR/secrets"
    mkdir -p "$secrets_dir"

    # Generate bootstrap node key
    if [ ! -f "$secrets_dir/bootstrap.key" ]; then
        openssl rand -hex 32 > "$secrets_dir/bootstrap.key"
        chmod 600 "$secrets_dir/bootstrap.key"
        log_info "Generated bootstrap.key"
    fi

    # Generate .env file if not present
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        cat > "$PROJECT_DIR/.env" << EOF
# Mycelix Testnet Environment Variables
# Generated on $(date)

# Database
POSTGRES_PASSWORD=$(openssl rand -hex 16)

# Grafana
GRAFANA_PASSWORD=$(openssl rand -hex 12)

# JWT Secret
JWT_SECRET=$(openssl rand -hex 32)

# Network
MYCELIX_NETWORK=testnet
MYCELIX_LOG_LEVEL=info
EOF
        chmod 600 "$PROJECT_DIR/.env"
        log_info "Generated .env file"
    fi

    log_success "Secrets generated!"
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."

    cd "$PROJECT_DIR"

    # Source environment variables
    if [ -f .env ]; then
        set -a
        source .env
        set +a
    fi

    # Create required directories
    mkdir -p config/grafana/provisioning/{datasources,dashboards}
    mkdir -p config/grafana/dashboards
    mkdir -p secrets

    # Pull latest images
    log_info "Pulling latest images..."
    docker compose -f docker-compose.testnet.yml pull

    # Start services
    log_info "Starting services..."
    docker compose -f docker-compose.testnet.yml up -d

    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30

    # Check service status
    docker compose -f docker-compose.testnet.yml ps

    log_success "Docker deployment complete!"
    echo ""
    log_info "Access points:"
    echo "  - Bootstrap API:  http://localhost:9001"
    echo "  - Faucet:         http://localhost:8080"
    echo "  - Grafana:        http://localhost:3000"
    echo "  - Prometheus:     http://localhost:9091"
}

# Deploy with Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."

    cd "$PROJECT_DIR/kubernetes"

    # Create namespace
    kubectl apply -f namespace.yaml

    # Create secrets (should be customized)
    log_warning "Using default secrets. Update secrets.yaml for production!"
    kubectl apply -f secrets.yaml

    # Apply all resources with kustomize
    log_info "Applying Kubernetes manifests..."
    kubectl apply -k .

    # Wait for deployments
    log_info "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment --all -n mycelix-testnet

    # Get status
    kubectl get pods -n mycelix-testnet
    kubectl get services -n mycelix-testnet

    log_success "Kubernetes deployment complete!"
    echo ""
    log_info "Access points:"
    echo "  Run: kubectl port-forward svc/bootstrap 9001:9001 -n mycelix-testnet"
    echo "  Run: kubectl port-forward svc/grafana 3000:3000 -n mycelix-testnet"
    echo ""
    log_info "Get external IP:"
    echo "  kubectl get svc bootstrap-external -n mycelix-testnet"
}

# Deploy with Terraform
deploy_terraform() {
    log_info "Deploying infrastructure with Terraform..."

    cd "$PROJECT_DIR/terraform"

    # Check for tfvars file
    if [ ! -f terraform.tfvars ]; then
        log_warning "No terraform.tfvars found. Copying example..."
        cp terraform.tfvars.example terraform.tfvars
        log_warning "Please edit terraform.tfvars with your configuration and run again."
        exit 1
    fi

    # Initialize Terraform
    log_info "Initializing Terraform..."
    terraform init -upgrade

    # Validate configuration
    log_info "Validating configuration..."
    terraform validate

    # Plan deployment
    log_info "Planning deployment..."
    terraform plan -out=tfplan

    # Confirm deployment
    echo ""
    read -p "Do you want to apply this plan? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log_warning "Deployment cancelled."
        exit 0
    fi

    # Apply deployment
    log_info "Applying Terraform plan..."
    terraform apply tfplan

    # Get outputs
    log_info "Deployment outputs:"
    terraform output

    log_success "Terraform deployment complete!"
    echo ""
    log_info "Next steps:"
    echo "  1. Configure kubectl with your cluster credentials"
    echo "  2. Deploy Kubernetes manifests: kubectl apply -k ../kubernetes/"
    echo "  3. Update DNS records with the external IP"
}

# Main function
main() {
    echo ""
    echo "========================================"
    echo "   Mycelix Testnet Deployment Script   "
    echo "========================================"
    echo ""

    log_info "Deployment type: $DEPLOYMENT_TYPE"

    # Check prerequisites
    check_prerequisites

    # Generate secrets
    generate_secrets

    # Deploy based on type
    case "$DEPLOYMENT_TYPE" in
        docker)
            deploy_docker
            ;;
        kubernetes|k8s)
            deploy_kubernetes
            ;;
        terraform|tf)
            deploy_terraform
            ;;
        *)
            log_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            echo "Usage: $0 [docker|kubernetes|terraform]"
            exit 1
            ;;
    esac

    echo ""
    log_success "Deployment complete!"
    echo ""
    echo "For monitoring, run: ./monitor-testnet.sh"
    echo "To join as a node, run: ./join-testnet.sh"
    echo ""
}

# Run main function
main "$@"
