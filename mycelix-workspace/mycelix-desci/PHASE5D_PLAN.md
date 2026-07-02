# Phase 5D: NixOS Integration & Production Hardening

**Goal:** Add ultimate reproducibility with NixOS and production-grade features that make Mycelix-DeSci the gold standard for decentralized science infrastructure.

**Duration:** 2-3 focused sessions
**Status:** Planning → Implementation
**Priority:** CRITICAL (reproducibility is essential for science!)

---

## 🎯 Executive Summary

We've built an amazing platform with:
- ✅ Core library (400K+ claims/sec)
- ✅ REST API (15 endpoints)
- ✅ CLI tool (15+ commands)
- ✅ Docker deployment
- ✅ Comprehensive examples & docs

**Now we need:** Maximum reproducibility with NixOS and production hardening!

---

## 📋 Detailed Implementation Plan

### **Track 1: NixOS Configuration** (Priority: CRITICAL)

**Goal:** Provide bit-for-bit reproducible builds for scientific research

#### **Why NixOS Matters for Science**
- 🔬 **Bit-for-bit reproducibility** - Essential for scientific validity
- 📦 **Declarative dependencies** - No "works on my machine" issues
- ⏪ **Atomic rollbacks** - Safe deployments
- 🔒 **Isolation** - No dependency conflicts
- 📚 **Self-documenting** - Build process is the documentation

#### **File 1: flake.nix** (Main Nix Flake)

```nix
{
  description = "Mycelix-DeSci - Reproducible Decentralized Science Platform";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Rust toolchain (pinned version for reproducibility)
        rustToolchain = pkgs.rust-bin.stable."1.75.0".default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        # Common build inputs
        buildInputs = with pkgs; [
          openssl
          pkg-config
        ];

        nativeBuildInputs = with pkgs; [
          rustToolchain
          pkg-config
        ];

        # Core library package
        mycelix-core = pkgs.rustPlatform.buildRustPackage {
          pname = "mycelix-desci-core";
          version = "0.1.0";
          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          inherit buildInputs nativeBuildInputs;

          # Build only core
          cargoBuildFlags = [ "--package" "mycelix-desci-core" ];

          meta = with pkgs.lib; {
            description = "Core library for Mycelix-DeSci decentralized science platform";
            homepage = "https://github.com/Luminous-Dynamics/mycelix-desci";
            license = licenses.mit;
            maintainers = [ ];
          };
        };

        # API server package
        mycelix-api = pkgs.rustPlatform.buildRustPackage {
          pname = "mycelix-api";
          version = "0.1.0";
          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          inherit buildInputs nativeBuildInputs;

          cargoBuildFlags = [ "--package" "mycelix-desci-api" ];

          # Install binary
          postInstall = ''
            mkdir -p $out/share/doc/mycelix-api
            cp README.md $out/share/doc/mycelix-api/
            cp -r docs $out/share/doc/mycelix-api/
          '';

          meta = with pkgs.lib; {
            description = "REST API server for Mycelix-DeSci";
            homepage = "https://github.com/Luminous-Dynamics/mycelix-desci";
            license = licenses.mit;
          };
        };

        # CLI tool package
        mycelix-cli = pkgs.rustPlatform.buildRustPackage {
          pname = "mycelix";
          version = "0.1.0";
          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          inherit buildInputs nativeBuildInputs;

          cargoBuildFlags = [ "--package" "mycelix-cli" ];

          postInstall = ''
            # Install shell completions (future)
            # installShellCompletion ...
          '';

          meta = with pkgs.lib; {
            description = "Command-line tool for Mycelix-DeSci";
            homepage = "https://github.com/Luminous-Dynamics/mycelix-desci";
            license = licenses.mit;
          };
        };

        # Docker image (built with Nix for reproducibility!)
        dockerImage = pkgs.dockerTools.buildLayeredImage {
          name = "mycelix-api";
          tag = "latest";

          contents = with pkgs; [
            mycelix-api
            cacert  # SSL certificates
            coreutils
          ];

          config = {
            Cmd = [ "${mycelix-api}/bin/mycelix-api" ];
            ExposedPorts = {
              "8080/tcp" = {};
            };
            Env = [
              "PORT=8080"
              "RUST_LOG=mycelix_api=info"
            ];
          };
        };

      in {
        # Packages available via `nix build`
        packages = {
          inherit mycelix-core mycelix-api mycelix-cli dockerImage;
          default = mycelix-cli;
        };

        # Apps available via `nix run`
        apps = {
          api = {
            type = "app";
            program = "${mycelix-api}/bin/mycelix-api";
          };
          cli = {
            type = "app";
            program = "${mycelix-cli}/bin/mycelix";
          };
          default = self.apps.${system}.cli;
        };

        # Development shell
        devShells.default = pkgs.mkShell {
          buildInputs = buildInputs ++ nativeBuildInputs ++ [
            # Additional dev tools
            pkgs.rust-analyzer
            pkgs.cargo-watch
            pkgs.cargo-edit
            pkgs.cargo-audit
            pkgs.cargo-tarpaulin
          ];

          shellHook = ''
            echo "🔬 Mycelix-DeSci Development Environment"
            echo "────────────────────────────────────────"
            echo "Rust version: $(rustc --version)"
            echo ""
            echo "Available commands:"
            echo "  cargo build --release     # Build all packages"
            echo "  cargo test --all          # Run tests"
            echo "  cargo run --bin mycelix-api  # Run API server"
            echo "  cargo run --bin mycelix   # Run CLI"
            echo ""
          '';

          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
        };
      }
    ) // {
      # NixOS module
      nixosModules.default = import ./nixos-module.nix;
    };
}
```

#### **File 2: nixos-module.nix** (NixOS Service Module)

```nix
{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.mycelix-desci;

  # Import packages from flake
  mycelix-api = pkgs.callPackage ./flake.nix {}.packages.${pkgs.system}.mycelix-api;
in {
  options.services.mycelix-desci = {
    enable = mkEnableOption "Mycelix-DeSci API server";

    package = mkOption {
      type = types.package;
      default = mycelix-api;
      description = "Package to use for mycelix-api";
    };

    port = mkOption {
      type = types.port;
      default = 8080;
      description = "Port for the API server to listen on";
    };

    host = mkOption {
      type = types.str;
      default = "0.0.0.0";
      description = "Host address to bind to";
    };

    logLevel = mkOption {
      type = types.enum [ "trace" "debug" "info" "warn" "error" ];
      default = "info";
      description = "Logging level";
    };

    corsOrigins = mkOption {
      type = types.str;
      default = "*";
      description = "CORS allowed origins";
    };

    user = mkOption {
      type = types.str;
      default = "mycelix";
      description = "User account under which mycelix-api runs";
    };

    group = mkOption {
      type = types.str;
      default = "mycelix";
      description = "Group under which mycelix-api runs";
    };

    dataDir = mkOption {
      type = types.path;
      default = "/var/lib/mycelix";
      description = "Directory for mycelix data";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = "Open firewall port for API server";
    };
  };

  config = mkIf cfg.enable {
    # Create user and group
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      home = cfg.dataDir;
      createHome = true;
      description = "Mycelix-DeSci service user";
    };

    users.groups.${cfg.group} = {};

    # Systemd service
    systemd.services.mycelix-api = {
      description = "Mycelix-DeSci API Server";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      environment = {
        PORT = toString cfg.port;
        RUST_LOG = "mycelix_api=${cfg.logLevel}";
        CORS_ORIGINS = cfg.corsOrigins;
      };

      serviceConfig = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/mycelix-api";
        Restart = "always";
        RestartSec = "10s";

        # Security hardening
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = cfg.dataDir;

        # Sandboxing
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ cfg.dataDir ];
        NoNewPrivileges = true;

        # Additional security
        PrivateDevices = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" ];
        RestrictNamespaces = true;
        LockPersonality = true;
        RestrictRealtime = true;
        RestrictSUIDSGID = true;
        RemoveIPC = true;
      };
    };

    # Open firewall if requested
    networking.firewall = mkIf cfg.openFirewall {
      allowedTCPPorts = [ cfg.port ];
    };
  };
}
```

#### **File 3: .envrc** (direnv integration)

```bash
#!/usr/bin/env bash
# Automatic Nix environment loading with direnv
# Install: nix-env -iA nixpkgs.direnv

use flake

# Optional: Layout using rust
# layout rust
```

#### **File 4: docs/NIX.md** (NixOS Documentation)

Complete guide covering:
- Why NixOS for science
- Installation instructions
- Building with Nix
- Running as NixOS service
- Development workflow
- Troubleshooting

---

### **Track 2: Integration Tests** (Priority: HIGH)

Create comprehensive test suite:

#### **File: tests/integration_tests.rs**

```rust
// Integration tests for the complete platform
mod helpers;

use helpers::TestServer;

#[tokio::test]
async fn test_complete_claim_lifecycle() {
    let server = TestServer::start().await;
    let client = reqwest::Client::new();

    // 1. Create E0 claim
    // 2. Add 5 verifications
    // 3. Verify tier upgrades (E0→E1→E2→E3→E4)
    // 4. Add provenance
    // 5. Query and verify
}

#[tokio::test]
async fn test_concurrent_operations() {
    // Test handling multiple simultaneous requests
}

#[tokio::test]
async fn test_error_handling() {
    // Test all error scenarios
}
```

---

### **Track 3: Production Features** (Priority: MEDIUM)

#### **1. Prometheus Metrics** (src/api/src/metrics.rs)

```rust
use prometheus::{Registry, Counter, Histogram};

pub struct Metrics {
    pub requests_total: Counter,
    pub request_duration: Histogram,
    pub claims_created: Counter,
    pub verifications_added: Counter,
}

// Export at /metrics endpoint
```

#### **2. Rate Limiting** (src/api/src/middleware/rate_limit.rs)

```rust
use tower_governor::{GovernorLayer, GovernorConfig};

// Per-IP rate limiting
// - Claims: 100/hour
// - Queries: 1000/hour
// - Trust updates: 50/hour
```

#### **3. Enhanced Health Checks**

```rust
// Check:
// - Database connectivity (future)
// - Disk space
// - Memory usage
// - Response time thresholds
```

---

### **Track 4: Deployment Documentation** (Priority: HIGH)

#### **File: docs/DEPLOYMENT.md**

Complete production deployment guide covering:

**1. Deployment Options**
- Docker Compose (simplest)
- NixOS (reproducible)
- Kubernetes (scalable)
- Cloud platforms (AWS/GCP/Azure)

**2. NixOS Deployment**
```nix
# In your configuration.nix:
services.mycelix-desci = {
  enable = true;
  port = 8080;
  logLevel = "info";
  openFirewall = true;
};
```

**3. Docker Deployment**
```bash
docker-compose up -d
```

**4. Kubernetes Deployment**
```yaml
# Complete k8s manifests
```

**5. Security**
- TLS/SSL setup
- Firewall rules
- Rate limiting
- Input validation

**6. Monitoring**
- Prometheus metrics
- Grafana dashboards
- Alert rules
- Log aggregation

**7. Backup & Recovery**
- Data backup strategies
- Disaster recovery
- RTO/RPO targets

---

### **Track 5: API Reference** (Priority: MEDIUM)

#### **File: docs/API_REFERENCE.md**

Complete API documentation with:

**For each endpoint:**
- URL and method
- Request parameters
- Request body schema
- Response schema
- Example curl command
- Example response
- Error codes
- Rate limits

**Example:**
```markdown
## POST /api/v1/claims

Create a new scientific claim.

### Request

```bash
curl -X POST http://localhost:8080/api/v1/claims \
  -H "Content-Type: application/json" \
  -d '{
    "tier": "E0",
    "content": {
      "dataset_hash": "blake3:abc123",
      "description": "Novel research finding",
      "category": "longevity",
      "keywords": ["aging", "NAD+"]
    },
    "creator": "researcher@uni.edu"
  }'
```

### Response (201 Created)

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "tier": "E0",
  ...
}
```

### Error Responses

- `400 Bad Request` - Invalid claim data
- `500 Internal Server Error` - Server error
```

---

## 🗓️ Implementation Timeline

### **Session 1: NixOS Foundation** (Current)
- ✅ Create flake.nix with all packages
- ✅ Create nixos-module.nix with systemd service
- ✅ Add .envrc for direnv
- ✅ Write NIX.md documentation
- ✅ Test builds and deployment

### **Session 2: Testing & Production Features**
- ✅ Integration test suite
- ✅ Prometheus metrics
- ✅ Rate limiting middleware
- ✅ Enhanced health checks

### **Session 3: Documentation Complete**
- ✅ DEPLOYMENT.md guide
- ✅ API_REFERENCE.md complete
- ✅ Final polish and review

---

## 📊 Success Metrics

**NixOS Quality:**
- [ ] Reproducible builds work across machines
- [ ] NixOS service starts correctly
- [ ] Declarative configuration complete
- [ ] Development shell functional

**Testing Quality:**
- [ ] >80% code coverage
- [ ] All critical paths tested
- [ ] Concurrent operations tested
- [ ] Error scenarios covered

**Production Readiness:**
- [ ] Metrics exportable to Prometheus
- [ ] Rate limiting functional
- [ ] Health checks comprehensive
- [ ] Security hardening complete

**Documentation Quality:**
- [ ] Every deployment option documented
- [ ] Every API endpoint documented
- [ ] Troubleshooting guides complete
- [ ] Examples for every feature

---

## 🎯 Phase 5D Deliverables Checklist

- [ ] **NixOS Configuration**
  - [ ] flake.nix (packages, apps, devShell)
  - [ ] nixos-module.nix (systemd service)
  - [ ] .envrc (direnv integration)
  - [ ] NIX.md (documentation)

- [ ] **Integration Tests**
  - [ ] Test infrastructure
  - [ ] Lifecycle tests
  - [ ] Concurrent operation tests
  - [ ] Error handling tests

- [ ] **Production Features**
  - [ ] Prometheus metrics
  - [ ] Rate limiting
  - [ ] Enhanced health checks
  - [ ] Security hardening

- [ ] **Documentation**
  - [ ] DEPLOYMENT.md (complete guide)
  - [ ] API_REFERENCE.md (all endpoints)
  - [ ] NIX.md (NixOS guide)

---

## 🚀 Impact

After Phase 5D, Mycelix-DeSci will be:

✅ **Maximally Reproducible** - NixOS ensures bit-for-bit builds
✅ **Production-Ready** - Metrics, rate limiting, health checks
✅ **Well-Tested** - Comprehensive integration tests
✅ **Fully Documented** - Every feature, every endpoint
✅ **Multiple Deployment Options** - Docker, Nix, K8s, Cloud

**This makes Mycelix-DeSci the gold standard for decentralized science infrastructure!** 🔬✨

---

**Let's build the most reproducible scientific platform ever! 🎉**
