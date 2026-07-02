SHELL := /bin/bash

.PHONY: help dev test build clean web rust zomes fmt lint bench check integration-test conductor-up conductor-down dna happ

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

dev: build start ## Build and start development environment

build: rust web ## Build all components

rust: ## Build Rust workspace
	@echo "▶ Building Rust workspace..."
	cargo fmt --all
	cargo clippy --all -- -D warnings
	cargo build --workspace

web: ## Build web application
	@echo "▶ Building web application..."
	cd apps/web && npm install && npm run build

zomes: ## Build Holochain zomes to WASM
	@echo "▶ Building Holochain zomes..."
	./scripts/build-wasm.sh

fmt: ## Format all code
	cargo fmt --all
	cd apps/web && npm run format || true

lint: ## Lint all code
	cargo clippy --all -- -D warnings
	cd apps/web && npm run lint || true

test: ## Run all tests
	@echo "▶ Running Rust tests..."
	cargo test --all
	@echo "▶ Running web tests..."
	cd apps/web && npm test -- --watch=false || true

bench: ## Run benchmarks
	@echo "▶ Running benchmarks..."
	cargo bench --workspace

check: ## Quick sanity check (fmt + clippy + test)
	@echo "▶ Running quick sanity check..."
	cargo fmt --all -- --check
	cargo clippy --all -- -D warnings
	cargo test --all --quiet

start: ## Start development servers
	./scripts/dev.sh

clean: ## Clean all build artifacts
	@echo "⚠ Cleaning all build artifacts..."
	cargo clean
	rm -rf apps/web/node_modules apps/web/dist
	find . -name "*.dna" -o -name "*.happ" -o -name "*.webhapp" | xargs rm -f

reset: ## Reset Holochain dev environment
	./scripts/hc-reset.sh

dna: zomes ## Package DNA bundle
	@echo "▶ Packaging DNA bundle..."
	cd dna && hc dna pack . -o edunet.dna
	@echo "  DNA: dna/edunet.dna"

happ: dna ## Package hApp bundle
	@echo "▶ Packaging hApp bundle..."
	cp dna/edunet.dna happ/edunet.dna
	hc app pack happ/ -o happ/mycelix-edunet.happ
	@echo "  hApp: happ/mycelix-edunet.happ"

integration-test: ## Run ignored integration tests (builds WASM + DNA + hApp first)
	./scripts/run-integration-tests.sh

conductor-up: happ ## Start conductor via docker-compose
	@echo "▶ Starting Holochain conductor..."
	docker compose up -d conductor
	@echo "  Admin: ws://localhost:8888"
	@echo "  App:   ws://localhost:8889"

conductor-down: ## Stop conductor
	docker compose down
