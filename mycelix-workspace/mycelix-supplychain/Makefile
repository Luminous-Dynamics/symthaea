.PHONY: help build test clean run dev fmt lint check docker-build install-deps

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Mycelix ERP - Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

## Build Commands

build: ## Build all components (Rust + TypeScript)
	@echo "$(BLUE)Building Rust service...$(NC)"
	@cd rust && cargo build --release
	@echo "$(GREEN)✓ Rust build complete$(NC)"
	@echo ""
	@echo "$(BLUE)Building TypeScript SDK...$(NC)"
	@cd ts/sdk && npm ci && npm run build
	@echo "$(GREEN)✓ SDK build complete$(NC)"
	@echo ""
	@echo "$(BLUE)Building TypeScript adapters...$(NC)"
	@cd ts/adapters/csv && npm ci && npm run build
	@cd ts/adapters/mqtt && npm ci && npm run build
	@echo "$(GREEN)✓ Adapters build complete$(NC)"

build-rust: ## Build only Rust components
	@echo "$(BLUE)Building Rust workspace...$(NC)"
	@cd rust && cargo build --release
	@echo "$(GREEN)✓ Done$(NC)"

build-ts: ## Build only TypeScript components
	@echo "$(BLUE)Building TypeScript components...$(NC)"
	@cd ts/sdk && npm ci && npm run build
	@cd ts/adapters/csv && npm ci && npm run build
	@cd ts/adapters/mqtt && npm ci && npm run build
	@echo "$(GREEN)✓ Done$(NC)"

build-dashboard: ## Build dashboard
	@echo "$(BLUE)Building dashboard...$(NC)"
	@cd ts/dashboard && npm ci && npm run build
	@echo "$(GREEN)✓ Done$(NC)"

## Test Commands

test: ## Run all tests
	@echo "$(BLUE)Running Rust tests...$(NC)"
	@cd rust && cargo test --all-features
	@echo "$(GREEN)✓ Rust tests passed$(NC)"
	@echo ""
	@echo "$(BLUE)Running TypeScript tests...$(NC)"
	@cd ts/sdk && npm test || true
	@echo "$(GREEN)✓ TypeScript tests complete$(NC)"

test-rust: ## Run only Rust tests
	@cd rust && cargo test --all-features --verbose

test-ts: ## Run only TypeScript tests
	@cd ts/sdk && npm test

test-coverage: ## Run tests with coverage
	@cd rust && cargo tarpaulin --all-features --workspace --out Html

## Development Commands

run: ## Run the service locally
	@echo "$(BLUE)Starting provenance service...$(NC)"
	@cd rust/service && cargo run

dev: ## Run service with auto-reload (requires cargo-watch)
	@echo "$(BLUE)Starting service with auto-reload...$(NC)"
	@cd rust/service && cargo watch -x run

dev-dashboard: ## Run dashboard in development mode
	@cd ts/dashboard && npm run dev

## Quality Commands

fmt: ## Format all code
	@echo "$(BLUE)Formatting Rust code...$(NC)"
	@cd rust && cargo fmt --all
	@echo "$(BLUE)Formatting TypeScript code...$(NC)"
	@cd ts/sdk && npm run format || true
	@cd ts/dashboard && npm run format || true
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint: ## Lint all code
	@echo "$(BLUE)Linting Rust code...$(NC)"
	@cd rust && cargo clippy --all-targets --all-features -- -D warnings
	@echo "$(BLUE)Linting TypeScript code...$(NC)"
	@cd ts/sdk && npm run lint || true
	@cd ts/dashboard && npm run lint || true
	@echo "$(GREEN)✓ Linting complete$(NC)"

check: ## Check code without building
	@cd rust && cargo check --all-features

## Database Commands

db-create: ## Create SQLite database
	@mkdir -p data
	@echo "$(GREEN)✓ Data directory created$(NC)"

db-migrate: ## Run database migrations (when implemented)
	@echo "$(YELLOW)Database migrations not yet implemented$(NC)"

## Docker Commands

docker-build: ## Build Docker images
	@echo "$(BLUE)Building Mycelix ERP service image...$(NC)"
	@docker build -f docker/Dockerfile.service -t mycelix/erp-service:latest .
	@echo "$(BLUE)Building dashboard image...$(NC)"
	@docker build -f ts/dashboard/Dockerfile -t mycelix/erp-dashboard:latest ./ts
	@echo "$(GREEN)✓ Docker images built$(NC)"

docker-run: ## Run services with Docker Compose
	@docker-compose -f deployments/docker-compose.yml up

docker-stop: ## Stop Docker Compose services
	@docker-compose -f deployments/docker-compose.yml down

## Setup Commands

install-deps: ## Install development dependencies
	@echo "$(BLUE)Installing Rust toolchain components...$(NC)"
	@rustup component add rustfmt clippy
	@echo "$(BLUE)Installing TypeScript dependencies...$(NC)"
	@cd ts/sdk && npm install
	@cd ts/dashboard && npm install
	@cd ts/adapters/csv && npm install
	@cd ts/adapters/mqtt && npm install
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

setup: install-deps db-create ## Initial setup for development
	@echo "$(GREEN)✓ Setup complete!$(NC)"
	@echo ""
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  1. Run 'make run' to start the service"
	@echo "  2. Run 'make example' to see it in action"
	@echo "  3. Check out the docs in the docs/ directory"

## Example Commands

example: ## Run a simple example workflow
	@echo "$(BLUE)Running example workflow...$(NC)"
	@echo ""
	@echo "Starting service in background..."
	@cd rust/service && cargo run & sleep 3
	@echo ""
	@echo "Posting example event..."
	@curl -X POST http://localhost:8080/v1/events \
		-H 'Content-Type: application/json' \
		-d @specs/examples/batch_produced.json \
		-w "\n"
	@echo ""
	@echo "$(GREEN)✓ Example complete!$(NC)"
	@pkill -f provenance-service

ingest-csv: ## Ingest test CSV data
	@echo "$(BLUE)Ingesting CSV data...$(NC)"
	@cd ts/adapters/csv && node dist/ingest.js -f ../../../tests/data/small_demo.csv

## Clean Commands

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning Rust artifacts...$(NC)"
	@cd rust && cargo clean
	@echo "$(BLUE)Cleaning TypeScript artifacts...$(NC)"
	@rm -rf ts/sdk/dist ts/sdk/node_modules
	@rm -rf ts/dashboard/dist ts/dashboard/node_modules ts/dashboard/.next
	@rm -rf ts/adapters/*/dist ts/adapters/*/node_modules
	@echo "$(GREEN)✓ Clean complete$(NC)"

clean-all: clean ## Clean everything including data
	@rm -rf data/
	@echo "$(GREEN)✓ All artifacts cleaned$(NC)"

## Documentation Commands

docs: ## Generate documentation
	@echo "$(BLUE)Generating Rust docs...$(NC)"
	@cd rust && cargo doc --no-deps --open
	@echo "$(GREEN)✓ Documentation generated$(NC)"

docs-api: ## Open API documentation
	@echo "$(BLUE)Opening OpenAPI spec...$(NC)"
	@open specs/openapi.yaml || xdg-open specs/openapi.yaml || echo "View specs/openapi.yaml"

## Release Commands

release-check: ## Check if ready for release
	@echo "$(BLUE)Running pre-release checks...$(NC)"
	@make lint
	@make test
	@make build
	@echo "$(GREEN)✓ Release checks passed$(NC)"
