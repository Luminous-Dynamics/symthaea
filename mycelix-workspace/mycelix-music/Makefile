# Mycelix Music Platform - Makefile
# Common operations for development, testing, and deployment

.PHONY: help install dev build test lint clean \
        docker-build docker-push docker-up docker-down \
        k8s-deploy-staging k8s-deploy-prod k8s-status \
        db-migrate db-backup db-restore \
        contracts-build contracts-test contracts-deploy

# Default target
help:
	@echo "Mycelix Music Platform - Available Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install          Install all dependencies"
	@echo "  make dev              Start development servers"
	@echo "  make build            Build all packages"
	@echo "  make test             Run all tests"
	@echo "  make lint             Run linters"
	@echo "  make clean            Clean build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build Docker images"
	@echo "  make docker-push      Push images to registry"
	@echo "  make docker-up        Start services with docker-compose"
	@echo "  make docker-down      Stop docker-compose services"
	@echo "  make docker-logs      Follow docker-compose logs"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make k8s-deploy-staging  Deploy to staging"
	@echo "  make k8s-deploy-prod     Deploy to production"
	@echo "  make k8s-status          Show deployment status"
	@echo "  make k8s-logs            Follow pod logs"
	@echo ""
	@echo "Database:"
	@echo "  make db-migrate       Run database migrations"
	@echo "  make db-backup        Create database backup"
	@echo "  make db-restore       Restore from backup (BACKUP=path)"
	@echo ""
	@echo "Contracts:"
	@echo "  make contracts-build  Build smart contracts"
	@echo "  make contracts-test   Test smart contracts"
	@echo "  make contracts-deploy Deploy to local network"

# ============================================================
# Development
# ============================================================

install:
	npm ci

dev:
	npm run dev

build:
	npm run build

test:
	npm run test

test-coverage:
	npm run test -- --coverage

lint:
	npm run lint

type-check:
	npm run type-check

clean:
	npm run clean
	rm -rf coverage .turbo

# ============================================================
# Docker
# ============================================================

REGISTRY ?= ghcr.io
IMAGE_PREFIX ?= $(REGISTRY)/mycelix
TAG ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "latest")

docker-build:
	docker build -t $(IMAGE_PREFIX)/api:$(TAG) -f docker/api.Dockerfile .
	docker build -t $(IMAGE_PREFIX)/web:$(TAG) -f docker/web.Dockerfile .
	docker build -t $(IMAGE_PREFIX)/streaming:$(TAG) -f docker/streaming.Dockerfile .
	docker tag $(IMAGE_PREFIX)/api:$(TAG) $(IMAGE_PREFIX)/api:latest
	docker tag $(IMAGE_PREFIX)/web:$(TAG) $(IMAGE_PREFIX)/web:latest
	docker tag $(IMAGE_PREFIX)/streaming:$(TAG) $(IMAGE_PREFIX)/streaming:latest

docker-push: docker-build
	docker push $(IMAGE_PREFIX)/api:$(TAG)
	docker push $(IMAGE_PREFIX)/web:$(TAG)
	docker push $(IMAGE_PREFIX)/streaming:$(TAG)
	docker push $(IMAGE_PREFIX)/api:latest
	docker push $(IMAGE_PREFIX)/web:latest
	docker push $(IMAGE_PREFIX)/streaming:latest

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-ps:
	docker-compose ps

docker-prod-up:
	docker-compose -f docker-compose.prod.yml up -d

docker-prod-down:
	docker-compose -f docker-compose.prod.yml down

# ============================================================
# Kubernetes
# ============================================================

KUBE_CONTEXT ?= $(shell kubectl config current-context)

k8s-deploy-staging:
	@echo "Deploying to staging (context: $(KUBE_CONTEXT))"
	kubectl apply -k k8s/overlays/staging

k8s-deploy-prod:
	@echo "Deploying to production (context: $(KUBE_CONTEXT))"
	@read -p "Are you sure you want to deploy to production? [y/N] " confirm && [ "$$confirm" = "y" ]
	kubectl apply -k k8s/overlays/production

k8s-status:
	@echo "=== Deployments ==="
	kubectl get deployments -n mycelix-music
	@echo ""
	@echo "=== Pods ==="
	kubectl get pods -n mycelix-music
	@echo ""
	@echo "=== Services ==="
	kubectl get services -n mycelix-music
	@echo ""
	@echo "=== Ingress ==="
	kubectl get ingress -n mycelix-music

k8s-logs:
	kubectl logs -n mycelix-music -l app.kubernetes.io/part-of=mycelix-music -f --max-log-requests=10

k8s-logs-api:
	kubectl logs -n mycelix-music -l app.kubernetes.io/name=mycelix-api -f

k8s-logs-web:
	kubectl logs -n mycelix-music -l app.kubernetes.io/name=mycelix-web -f

k8s-restart-api:
	kubectl rollout restart deployment/api -n mycelix-music

k8s-restart-web:
	kubectl rollout restart deployment/web -n mycelix-music

k8s-port-forward-api:
	kubectl port-forward -n mycelix-music svc/api 3100:3100

k8s-port-forward-web:
	kubectl port-forward -n mycelix-music svc/web 3000:3000

# ============================================================
# Database
# ============================================================

db-migrate:
	npm run migrate --workspace=@mycelix/api

db-migrate-status:
	npm run migrate:status --workspace=@mycelix/api

db-backup:
	@mkdir -p backups
	./scripts/backup/backup-db.sh ./backups

db-restore:
ifndef BACKUP
	$(error BACKUP is required. Usage: make db-restore BACKUP=./backups/mycelix_backup_xxx.sql.gz)
endif
	./scripts/backup/restore-db.sh $(BACKUP)

# ============================================================
# Smart Contracts
# ============================================================

contracts-build:
	cd contracts && forge build

contracts-test:
	cd contracts && forge test

contracts-test-verbose:
	cd contracts && forge test -vvv

contracts-deploy:
	npm run contracts:deploy:local

contracts-deploy-testnet:
	npm run contracts:deploy:testnet

# ============================================================
# Local Development Setup
# ============================================================

setup-local: docker-up
	@echo "Waiting for services to be ready..."
	sleep 10
	$(MAKE) db-migrate
	$(MAKE) contracts-deploy
	npm run seed:local
	@echo ""
	@echo "Local development environment ready!"
	@echo "  API: http://localhost:3100"
	@echo "  Web: http://localhost:3000"
	@echo "  Swagger: http://localhost:3100/swagger"

# ============================================================
# CI/CD Helpers
# ============================================================

ci-test:
	npm ci
	npm run lint
	npm run type-check
	npm run test -- --ci --coverage

ci-build:
	npm ci
	npm run build

# ============================================================
# Utilities
# ============================================================

format:
	npx prettier --write "**/*.{ts,tsx,json,md,yml,yaml}"

check-deps:
	npx npm-check-updates

update-deps:
	npx npm-check-updates -u
	npm install

security-audit:
	npm audit
	cd contracts && forge audit 2>/dev/null || echo "No forge audit available"
