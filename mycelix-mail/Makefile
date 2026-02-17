# Mycelix Mail - Production Build & Optimization Pipeline
# ========================================================
#
# Comprehensive build system for Holochain zomes, frontend, and SDK
#
# Usage:
#   make build          - Build all components
#   make build-prod     - Full production build with optimizations
#   make optimize-wasm  - Optimize WASM binaries
#   make analyze-bundle - Analyze frontend bundle size
#
# ========================================================

.PHONY: all build build-zomes build-frontend build-sdk build-prod build-happ build-dna \
        optimize-wasm analyze-bundle clean test lint docs \
        test-zomes test-frontend test-sdk test-e2e test-integration \
        lint-rust lint-ts format format-rust format-ts check \
        dev dev-holochain dev-frontend docker docker-up docker-down \
        deploy deploy-staging install help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
CYAN := \033[0;36m
BOLD := \033[1m
NC := \033[0m

# Project directories
ROOT_DIR := $(shell pwd)
HOLOCHAIN_DIR := $(ROOT_DIR)/holochain
FRONTEND_DIR := $(ROOT_DIR)/ui/frontend
SDK_TS_DIR := $(ROOT_DIR)/sdk/typescript
SDK_PY_DIR := $(ROOT_DIR)/sdk/python
CLIENT_DIR := $(HOLOCHAIN_DIR)/client
SCRIPTS_DIR := $(ROOT_DIR)/scripts
DOCKER_DIR := $(ROOT_DIR)/docker

# Build output directories
RELEASE_DIR := $(ROOT_DIR)/release
OPTIMIZED_WASM_DIR := $(RELEASE_DIR)/optimized
ARTIFACTS_DIR := $(RELEASE_DIR)/artifacts

# WASM target
WASM_TARGET := wasm32-unknown-unknown

# Version from git or fallback
VERSION := $(shell git describe --tags --always 2>/dev/null || echo "0.1.0-dev")
BUILD_DATE := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")

# Default target
all: build

# ============================================
# PRIMARY BUILD TARGETS
# ============================================

## Build all components (development)
build: build-zomes build-frontend build-sdk
	@echo "$(GREEN)$(BOLD)[SUCCESS]$(NC) All components built successfully"

## Build Holochain WASM zomes with release optimizations
build-zomes:
	@echo "$(BLUE)$(BOLD)[BUILD]$(NC) Building Holochain zomes..."
	@echo "$(CYAN)  Target: $(WASM_TARGET)$(NC)"
	@echo "$(CYAN)  Profile: release$(NC)"
	cd $(HOLOCHAIN_DIR) && RUSTFLAGS="-C link-arg=-zstack-size=65536" \
		cargo build --release --target $(WASM_TARGET)
	@echo "$(GREEN)[OK]$(NC) Zomes built successfully"

## Build zomes in debug mode (faster compilation)
build-zomes-debug:
	@echo "$(BLUE)[BUILD]$(NC) Building zomes (debug mode)..."
	cd $(HOLOCHAIN_DIR) && cargo build --target $(WASM_TARGET)

## Build frontend with production optimizations
build-frontend:
	@echo "$(BLUE)$(BOLD)[BUILD]$(NC) Building frontend..."
	cd $(FRONTEND_DIR) && npm run build
	@echo "$(GREEN)[OK]$(NC) Frontend built successfully"

## Build TypeScript SDK
build-sdk: build-sdk-ts
	@echo "$(GREEN)[OK]$(NC) SDK built successfully"

## Build TypeScript SDK
build-sdk-ts:
	@echo "$(BLUE)[BUILD]$(NC) Building TypeScript SDK..."
	cd $(SDK_TS_DIR) && npm run build
	@echo "$(GREEN)[OK]$(NC) TypeScript SDK built"

## Build Holochain client library
build-client:
	@echo "$(BLUE)[BUILD]$(NC) Building Holochain client..."
	cd $(CLIENT_DIR) && npm run build
	@echo "$(GREEN)[OK]$(NC) Holochain client built"

## Package DNA from zomes
build-dna: build-zomes
	@echo "$(BLUE)[BUILD]$(NC) Packaging DNA..."
	cd $(HOLOCHAIN_DIR) && hc dna pack dna
	@echo "$(GREEN)[OK]$(NC) DNA packaged"

## Package hApp bundle
build-happ: build-dna
	@echo "$(BLUE)[BUILD]$(NC) Packaging hApp..."
	cd $(HOLOCHAIN_DIR) && hc app pack .
	@echo "$(GREEN)[OK]$(NC) hApp packaged"

# ============================================
# PRODUCTION BUILD TARGETS
# ============================================

## Full production build with all optimizations
build-prod: clean-release
	@echo "$(BOLD)$(BLUE)======================================$(NC)"
	@echo "$(BOLD)$(BLUE)  PRODUCTION BUILD PIPELINE$(NC)"
	@echo "$(BOLD)$(BLUE)  Version: $(VERSION)$(NC)"
	@echo "$(BOLD)$(BLUE)======================================$(NC)"
	@echo ""
	@mkdir -p $(RELEASE_DIR) $(OPTIMIZED_WASM_DIR) $(ARTIFACTS_DIR)
	@$(MAKE) build-zomes
	@$(MAKE) optimize-wasm
	@$(MAKE) build-happ
	@$(MAKE) build-frontend
	@$(MAKE) build-sdk
	@$(MAKE) build-client
	@$(MAKE) collect-artifacts
	@$(MAKE) generate-checksums
	@echo ""
	@echo "$(BOLD)$(GREEN)======================================$(NC)"
	@echo "$(BOLD)$(GREEN)  PRODUCTION BUILD COMPLETE$(NC)"
	@echo "$(BOLD)$(GREEN)  Artifacts: $(ARTIFACTS_DIR)$(NC)"
	@echo "$(BOLD)$(GREEN)======================================$(NC)"

## Optimize all WASM files with wasm-opt
optimize-wasm:
	@echo "$(BLUE)$(BOLD)[OPTIMIZE]$(NC) Optimizing WASM binaries..."
	@mkdir -p $(OPTIMIZED_WASM_DIR)
	@echo "$(CYAN)  Using wasm-opt with -Oz (size optimization)$(NC)"
	@echo "$(CYAN)  Stripping debug info$(NC)"
	@for wasm_file in $(HOLOCHAIN_DIR)/target/$(WASM_TARGET)/release/*.wasm; do \
		if [ -f "$$wasm_file" ]; then \
			filename=$$(basename "$$wasm_file"); \
			echo "$(YELLOW)  Processing: $$filename$(NC)"; \
			original_size=$$(stat -c%s "$$wasm_file" 2>/dev/null || stat -f%z "$$wasm_file"); \
			wasm-opt -Oz \
				--strip-debug \
				--strip-dwarf \
				--strip-producers \
				--vacuum \
				--dce \
				--optimize-casts \
				--remove-unused-names \
				--remove-unused-module-elements \
				"$$wasm_file" -o "$(OPTIMIZED_WASM_DIR)/$$filename" 2>/dev/null || \
				cp "$$wasm_file" "$(OPTIMIZED_WASM_DIR)/$$filename"; \
			optimized_size=$$(stat -c%s "$(OPTIMIZED_WASM_DIR)/$$filename" 2>/dev/null || stat -f%z "$(OPTIMIZED_WASM_DIR)/$$filename"); \
			reduction=$$(echo "scale=1; (1 - $$optimized_size / $$original_size) * 100" | bc 2>/dev/null || echo "N/A"); \
			echo "$(GREEN)    $$filename: $$(echo "scale=1; $$original_size/1024" | bc)KB -> $$(echo "scale=1; $$optimized_size/1024" | bc)KB ($$reduction% reduction)$(NC)"; \
		fi; \
	done
	@echo "$(GREEN)[OK]$(NC) WASM optimization complete"

## Analyze frontend bundle size
analyze-bundle:
	@echo "$(BLUE)$(BOLD)[ANALYZE]$(NC) Analyzing frontend bundle..."
	@echo "$(CYAN)  Output: $(FRONTEND_DIR)/bundle-analysis.html$(NC)"
	cd $(FRONTEND_DIR) && ANALYZE=true npm run build
	@if [ -f "$(FRONTEND_DIR)/dist/stats.html" ]; then \
		mv $(FRONTEND_DIR)/dist/stats.html $(FRONTEND_DIR)/bundle-analysis.html; \
		echo "$(GREEN)[OK]$(NC) Bundle analysis saved to $(FRONTEND_DIR)/bundle-analysis.html"; \
	fi
	@echo "$(YELLOW)  Tip: Open bundle-analysis.html in a browser to visualize$(NC)"

## Collect artifacts into release directory
collect-artifacts:
	@echo "$(BLUE)[COLLECT]$(NC) Collecting build artifacts..."
	@mkdir -p $(ARTIFACTS_DIR)/wasm
	@mkdir -p $(ARTIFACTS_DIR)/frontend
	@mkdir -p $(ARTIFACTS_DIR)/sdk
	@mkdir -p $(ARTIFACTS_DIR)/happ
	@# Copy optimized WASM files
	@cp -r $(OPTIMIZED_WASM_DIR)/*.wasm $(ARTIFACTS_DIR)/wasm/ 2>/dev/null || true
	@# Copy hApp bundle
	@cp $(HOLOCHAIN_DIR)/*.happ $(ARTIFACTS_DIR)/happ/ 2>/dev/null || true
	@cp $(HOLOCHAIN_DIR)/dna/*.dna $(ARTIFACTS_DIR)/happ/ 2>/dev/null || true
	@# Copy frontend build
	@cp -r $(FRONTEND_DIR)/dist/* $(ARTIFACTS_DIR)/frontend/ 2>/dev/null || true
	@# Copy SDK builds
	@cp -r $(SDK_TS_DIR)/dist $(ARTIFACTS_DIR)/sdk/typescript 2>/dev/null || true
	@# Create manifest
	@echo "Mycelix Mail Release $(VERSION)" > $(ARTIFACTS_DIR)/MANIFEST.txt
	@echo "Build Date: $(BUILD_DATE)" >> $(ARTIFACTS_DIR)/MANIFEST.txt
	@echo "Git SHA: $(shell git rev-parse HEAD 2>/dev/null || echo 'N/A')" >> $(ARTIFACTS_DIR)/MANIFEST.txt
	@echo "$(GREEN)[OK]$(NC) Artifacts collected"

## Generate checksums for all artifacts
generate-checksums:
	@echo "$(BLUE)[CHECKSUM]$(NC) Generating checksums..."
	@cd $(ARTIFACTS_DIR) && find . -type f ! -name "*.sha256" ! -name "CHECKSUMS.txt" -exec sha256sum {} \; > CHECKSUMS.txt 2>/dev/null || \
		cd $(ARTIFACTS_DIR) && find . -type f ! -name "*.sha256" ! -name "CHECKSUMS.txt" -exec shasum -a 256 {} \; > CHECKSUMS.txt
	@echo "$(GREEN)[OK]$(NC) Checksums saved to $(ARTIFACTS_DIR)/CHECKSUMS.txt"

# ============================================
# TEST TARGETS
# ============================================

## Run all tests
test: test-zomes test-frontend test-sdk
	@echo "$(GREEN)$(BOLD)[SUCCESS]$(NC) All tests passed"

## Run Rust zome tests
test-zomes:
	@echo "$(BLUE)[TEST]$(NC) Running zome tests..."
	cd $(HOLOCHAIN_DIR) && cargo test
	@echo "$(GREEN)[OK]$(NC) Zome tests passed"

## Run zome tests with verbose output
test-zomes-verbose:
	cd $(HOLOCHAIN_DIR) && cargo test -- --nocapture

## Run frontend tests
test-frontend:
	@echo "$(BLUE)[TEST]$(NC) Running frontend tests..."
	cd $(FRONTEND_DIR) && npm test -- --run
	@echo "$(GREEN)[OK]$(NC) Frontend tests passed"

## Run SDK tests
test-sdk:
	@echo "$(BLUE)[TEST]$(NC) Running SDK tests..."
	cd $(SDK_TS_DIR) && npm test -- --run
	@echo "$(GREEN)[OK]$(NC) SDK tests passed"

## Run client tests
test-client:
	@echo "$(BLUE)[TEST]$(NC) Running client tests..."
	cd $(CLIENT_DIR) && npm test -- --run
	@echo "$(GREEN)[OK]$(NC) Client tests passed"

## Run integration tests
test-integration:
	@echo "$(BLUE)[TEST]$(NC) Running integration tests..."
	cd $(CLIENT_DIR) && npm run test:integration
	@echo "$(GREEN)[OK]$(NC) Integration tests passed"

## Run E2E tests with Playwright
test-e2e:
	@echo "$(BLUE)[TEST]$(NC) Running E2E tests..."
	cd $(FRONTEND_DIR) && npx playwright test
	@echo "$(GREEN)[OK]$(NC) E2E tests passed"

## Run E2E tests with UI
test-e2e-ui:
	cd $(FRONTEND_DIR) && npx playwright test --ui

## Run E2E tests headed
test-e2e-headed:
	cd $(FRONTEND_DIR) && npx playwright test --headed

## Generate E2E test report
test-e2e-report:
	cd $(FRONTEND_DIR) && npx playwright show-report

## Run test coverage
test-coverage: test-coverage-zomes test-coverage-frontend
	@echo "$(GREEN)[OK]$(NC) Coverage reports generated"

## Run zome test coverage
test-coverage-zomes:
	@echo "$(BLUE)[COVERAGE]$(NC) Generating zome coverage..."
	cd $(HOLOCHAIN_DIR) && cargo tarpaulin --out Html

## Run frontend test coverage
test-coverage-frontend:
	@echo "$(BLUE)[COVERAGE]$(NC) Generating frontend coverage..."
	cd $(FRONTEND_DIR) && npm run test:coverage

# ============================================
# CODE QUALITY TARGETS
# ============================================

## Lint all code
lint: lint-rust lint-ts
	@echo "$(GREEN)$(BOLD)[SUCCESS]$(NC) Linting complete"

## Lint Rust code with clippy
lint-rust:
	@echo "$(BLUE)[LINT]$(NC) Linting Rust code..."
	cd $(HOLOCHAIN_DIR) && cargo clippy --all-targets --target $(WASM_TARGET) -- -D warnings
	@echo "$(GREEN)[OK]$(NC) Rust linting passed"

## Lint TypeScript code
lint-ts:
	@echo "$(BLUE)[LINT]$(NC) Linting TypeScript code..."
	cd $(FRONTEND_DIR) && npm run lint
	cd $(SDK_TS_DIR) && npm run lint
	cd $(CLIENT_DIR) && npm run lint
	@echo "$(GREEN)[OK]$(NC) TypeScript linting passed"

## Format all code
format: format-rust format-ts
	@echo "$(GREEN)[OK]$(NC) Formatting complete"

## Format Rust code
format-rust:
	@echo "$(BLUE)[FORMAT]$(NC) Formatting Rust code..."
	cd $(HOLOCHAIN_DIR) && cargo fmt

## Format TypeScript code
format-ts:
	@echo "$(BLUE)[FORMAT]$(NC) Formatting TypeScript code..."
	cd $(SDK_TS_DIR) && npm run format 2>/dev/null || true
	cd $(CLIENT_DIR) && npm run format 2>/dev/null || true

## Check all code (lint + types)
check: lint typecheck
	@echo "$(GREEN)$(BOLD)[SUCCESS]$(NC) All checks passed"

## Type check TypeScript
typecheck:
	@echo "$(BLUE)[TYPECHECK]$(NC) Type checking TypeScript..."
	cd $(FRONTEND_DIR) && npm run type-check 2>/dev/null || npm run typecheck 2>/dev/null || true
	cd $(SDK_TS_DIR) && npm run typecheck 2>/dev/null || true
	cd $(CLIENT_DIR) && npm run typecheck 2>/dev/null || true
	@echo "$(GREEN)[OK]$(NC) Type checking passed"

# ============================================
# DOCUMENTATION TARGETS
# ============================================

## Generate all API documentation (Rust + TypeScript)
docs:
	@echo "$(BLUE)$(BOLD)[DOCS]$(NC) Generating all documentation..."
	$(SCRIPTS_DIR)/generate-docs.sh
	@echo "$(GREEN)$(BOLD)[SUCCESS]$(NC) Documentation generated"
	@echo "$(CYAN)  View at: docs/api/index.html$(NC)"

## Generate Rust/Holochain documentation only
docs-rust:
	@echo "$(BLUE)[DOCS]$(NC) Generating Rust documentation..."
	$(SCRIPTS_DIR)/generate-docs.sh --rust-only
	@echo "$(GREEN)[OK]$(NC) Rust docs: docs/api/rust/"

## Generate TypeScript SDK documentation only
docs-typescript:
	@echo "$(BLUE)[DOCS]$(NC) Generating TypeScript documentation..."
	$(SCRIPTS_DIR)/generate-docs.sh --ts-only
	@echo "$(GREEN)[OK]$(NC) TypeScript docs: docs/api/typescript/"

## Clean and regenerate all documentation
docs-clean:
	@echo "$(BLUE)[DOCS]$(NC) Cleaning and regenerating documentation..."
	$(SCRIPTS_DIR)/generate-docs.sh --clean
	@echo "$(GREEN)[OK]$(NC) Documentation regenerated"

## Open documentation in browser
docs-open:
	@echo "$(BLUE)[DOCS]$(NC) Opening documentation..."
	$(SCRIPTS_DIR)/generate-docs.sh --open

# ============================================
# DEVELOPMENT TARGETS
# ============================================

## Install all dependencies
install:
	@echo "$(BLUE)[INSTALL]$(NC) Installing dependencies..."
	cd $(HOLOCHAIN_DIR) && cargo fetch
	cd $(FRONTEND_DIR) && npm install
	cd $(SDK_TS_DIR) && npm install
	cd $(CLIENT_DIR) && npm install
	@echo "$(GREEN)[OK]$(NC) Dependencies installed"

## Start development environment
dev:
	@echo "$(BLUE)[DEV]$(NC) Starting development environment..."
	@echo "$(YELLOW)  Starting Holochain sandbox...$(NC)"
	cd $(HOLOCHAIN_DIR) && hc sandbox generate --run 8888 &
	@sleep 5
	@echo "$(YELLOW)  Starting frontend dev server...$(NC)"
	cd $(FRONTEND_DIR) && npm run dev

## Start Holochain sandbox only
dev-holochain:
	cd $(HOLOCHAIN_DIR) && hc sandbox generate --num-sandboxes 2 --run 8888

## Start frontend dev server only
dev-frontend:
	cd $(FRONTEND_DIR) && npm run dev

## Run Storybook
storybook:
	cd $(FRONTEND_DIR) && npm run storybook

## Build Storybook static
storybook-build:
	cd $(FRONTEND_DIR) && npm run build-storybook

# ============================================
# DOCKER TARGETS
# ============================================

## Build Docker images
docker:
	@echo "$(BLUE)[DOCKER]$(NC) Building Docker images..."
	docker-compose -f $(DOCKER_DIR)/docker-compose.yml build
	@echo "$(GREEN)[OK]$(NC) Docker images built"

## Start Docker containers
docker-up:
	docker-compose -f $(DOCKER_DIR)/docker-compose.yml up -d

## Stop Docker containers
docker-down:
	docker-compose -f $(DOCKER_DIR)/docker-compose.yml down

## View Docker logs
docker-logs:
	docker-compose -f $(DOCKER_DIR)/docker-compose.yml logs -f

## Clean Docker resources
docker-clean:
	docker-compose -f $(DOCKER_DIR)/docker-compose.yml down -v --rmi local

# ============================================
# DEPLOYMENT TARGETS
# ============================================

## Deploy to production
deploy: build-prod test
	@echo "$(BLUE)[DEPLOY]$(NC) Deploying to production..."
	$(SCRIPTS_DIR)/deploy.sh
	@echo "$(GREEN)[OK]$(NC) Deployment complete"

## Deploy to staging
deploy-staging: build
	@echo "$(BLUE)[DEPLOY]$(NC) Deploying to staging..."
	$(SCRIPTS_DIR)/deploy.sh --staging
	@echo "$(GREEN)[OK]$(NC) Staging deployment complete"

# ============================================
# CLEANUP TARGETS
# ============================================

## Clean all build artifacts
clean: clean-rust clean-frontend clean-sdk
	@echo "$(GREEN)[OK]$(NC) Clean complete"

## Clean Rust build artifacts
clean-rust:
	@echo "$(BLUE)[CLEAN]$(NC) Cleaning Rust artifacts..."
	cd $(HOLOCHAIN_DIR) && cargo clean

## Clean frontend build artifacts
clean-frontend:
	@echo "$(BLUE)[CLEAN]$(NC) Cleaning frontend artifacts..."
	rm -rf $(FRONTEND_DIR)/dist
	rm -rf $(FRONTEND_DIR)/node_modules/.cache
	rm -rf $(FRONTEND_DIR)/bundle-analysis.html

## Clean SDK build artifacts
clean-sdk:
	@echo "$(BLUE)[CLEAN]$(NC) Cleaning SDK artifacts..."
	rm -rf $(SDK_TS_DIR)/dist
	rm -rf $(CLIENT_DIR)/dist

## Clean release directory
clean-release:
	@echo "$(BLUE)[CLEAN]$(NC) Cleaning release directory..."
	rm -rf $(RELEASE_DIR)

## Clean Holochain sandbox data
clean-sandbox:
	hc sandbox clean

## Full clean (including node_modules)
clean-all: clean clean-release
	@echo "$(BLUE)[CLEAN]$(NC) Performing full clean..."
	rm -rf $(FRONTEND_DIR)/node_modules
	rm -rf $(SDK_TS_DIR)/node_modules
	rm -rf $(CLIENT_DIR)/node_modules
	@echo "$(GREEN)[OK]$(NC) Full clean complete"

# ============================================
# SECURITY & AUDIT TARGETS
# ============================================

## Security audit
audit:
	@echo "$(BLUE)[AUDIT]$(NC) Running security audit..."
	cd $(HOLOCHAIN_DIR) && cargo audit 2>/dev/null || echo "cargo-audit not installed"
	cd $(FRONTEND_DIR) && npm audit
	cd $(SDK_TS_DIR) && npm audit
	@echo "$(GREEN)[OK]$(NC) Audit complete"

## Update dependencies
update:
	cd $(HOLOCHAIN_DIR) && cargo update
	cd $(FRONTEND_DIR) && npm update
	cd $(SDK_TS_DIR) && npm update
	cd $(CLIENT_DIR) && npm update

# ============================================
# UTILITY TARGETS
# ============================================

## Print version information
version:
	@echo "Mycelix Mail $(VERSION)"
	@echo "Build Date: $(BUILD_DATE)"

## Print build info
info:
	@echo "$(BOLD)Mycelix Mail Build System$(NC)"
	@echo ""
	@echo "$(CYAN)Version:$(NC)      $(VERSION)"
	@echo "$(CYAN)Build Date:$(NC)   $(BUILD_DATE)"
	@echo "$(CYAN)Root Dir:$(NC)     $(ROOT_DIR)"
	@echo "$(CYAN)WASM Target:$(NC)  $(WASM_TARGET)"
	@echo ""
	@echo "$(CYAN)Directories:$(NC)"
	@echo "  Holochain:  $(HOLOCHAIN_DIR)"
	@echo "  Frontend:   $(FRONTEND_DIR)"
	@echo "  SDK:        $(SDK_TS_DIR)"
	@echo "  Release:    $(RELEASE_DIR)"

## Show help
help:
	@echo "$(BOLD)$(BLUE)Mycelix Mail - Production Build System$(NC)"
	@echo ""
	@echo "$(YELLOW)Primary Build Targets:$(NC)"
	@echo "  $(CYAN)make build$(NC)          - Build all components (development)"
	@echo "  $(CYAN)make build-prod$(NC)     - Full production build with optimizations"
	@echo "  $(CYAN)make build-zomes$(NC)    - Build Holochain WASM zomes"
	@echo "  $(CYAN)make build-frontend$(NC) - Build frontend with Vite"
	@echo "  $(CYAN)make build-sdk$(NC)      - Build TypeScript SDK"
	@echo "  $(CYAN)make build-happ$(NC)     - Package hApp bundle"
	@echo ""
	@echo "$(YELLOW)Optimization Targets:$(NC)"
	@echo "  $(CYAN)make optimize-wasm$(NC)  - Optimize WASM with wasm-opt (-Oz)"
	@echo "  $(CYAN)make analyze-bundle$(NC) - Analyze frontend bundle size"
	@echo ""
	@echo "$(YELLOW)Test Targets:$(NC)"
	@echo "  $(CYAN)make test$(NC)           - Run all tests"
	@echo "  $(CYAN)make test-zomes$(NC)     - Run Rust zome tests"
	@echo "  $(CYAN)make test-frontend$(NC)  - Run frontend tests"
	@echo "  $(CYAN)make test-sdk$(NC)       - Run SDK tests"
	@echo "  $(CYAN)make test-e2e$(NC)       - Run E2E tests (Playwright)"
	@echo "  $(CYAN)make test-coverage$(NC)  - Generate coverage reports"
	@echo ""
	@echo "$(YELLOW)Code Quality:$(NC)"
	@echo "  $(CYAN)make lint$(NC)           - Lint all code"
	@echo "  $(CYAN)make format$(NC)         - Format all code"
	@echo "  $(CYAN)make check$(NC)          - Full code check (lint + types)"
	@echo ""
	@echo "$(YELLOW)Documentation:$(NC)"
	@echo "  $(CYAN)make docs$(NC)           - Generate all API documentation"
	@echo "  $(CYAN)make docs-rust$(NC)      - Generate Rust/Holochain docs"
	@echo "  $(CYAN)make docs-typescript$(NC) - Generate TypeScript SDK docs"
	@echo "  $(CYAN)make docs-clean$(NC)     - Clean and regenerate docs"
	@echo "  $(CYAN)make docs-open$(NC)      - Open docs in browser"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  $(CYAN)make install$(NC)        - Install dependencies"
	@echo "  $(CYAN)make dev$(NC)            - Start dev environment"
	@echo "  $(CYAN)make storybook$(NC)      - Run Storybook"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@echo "  $(CYAN)make docker$(NC)         - Build Docker images"
	@echo "  $(CYAN)make docker-up$(NC)      - Start containers"
	@echo "  $(CYAN)make docker-down$(NC)    - Stop containers"
	@echo ""
	@echo "$(YELLOW)Deployment:$(NC)"
	@echo "  $(CYAN)make deploy$(NC)         - Deploy to production"
	@echo "  $(CYAN)make deploy-staging$(NC) - Deploy to staging"
	@echo ""
	@echo "$(YELLOW)Cleanup:$(NC)"
	@echo "  $(CYAN)make clean$(NC)          - Clean build artifacts"
	@echo "  $(CYAN)make clean-release$(NC)  - Clean release directory"
	@echo "  $(CYAN)make clean-all$(NC)      - Full clean (inc. node_modules)"
	@echo ""
	@echo "$(YELLOW)Utility:$(NC)"
	@echo "  $(CYAN)make version$(NC)        - Print version info"
	@echo "  $(CYAN)make info$(NC)           - Print build system info"
	@echo "  $(CYAN)make audit$(NC)          - Security audit"
	@echo "  $(CYAN)make help$(NC)           - Show this help"
