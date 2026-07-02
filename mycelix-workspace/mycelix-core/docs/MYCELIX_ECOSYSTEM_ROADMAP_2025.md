# 🍄 Mycelix Ecosystem Comprehensive Roadmap 2025-2026

**Version**: 1.0.0
**Created**: December 30, 2025
**Last Updated**: December 30, 2025
**Status**: Active Development

---

## Executive Summary

This roadmap transforms Mycelix from **5 isolated projects** into a **unified, production-ready ecosystem** built on Holochain 0.6. The plan spans 12 months across 4 major phases, designed for execution by a small team using the Sacred Trinity development model (Human + Cloud AI + Local AI).

### Key Objectives
1. **Unify** all hApps on Holochain 0.6 stable
2. **Extract** shared infrastructure into `mycelix-sdk`
3. **Connect** hApps via Bridge Protocol
4. **Launch** production pilots in 3 verticals
5. **Validate** academically with peer-reviewed publications

### Success Metrics (End of 2026)
| Metric | Target |
|--------|--------|
| Active Users | 1,000+ |
| hApps on Production | 5 |
| Academic Citations | 10+ |
| Byzantine Tolerance | 45% (validated) |
| Cross-hApp Transactions | 10,000+ |

---

## Technology Stack (Standardized)

### Holochain 0.6 Ecosystem (REQUIRED FOR ALL)

| Component | Version | Purpose |
|-----------|---------|---------|
| holochain | 0.6.0 | Core conductor |
| hc CLI | 0.6.0 | Development tooling |
| hc-scaffold | 0.600.0 | Project scaffolding |
| hc-spin | 0.600.0 | Local development |
| hcterm | 0.6.0 | Terminal interface |
| Lair keystore | 0.6.3 | Key management |
| hdk | 0.6.0 | Holochain Dev Kit |
| hdi | 0.7.0 | Holochain Data Integrity |
| @holochain/client | 0.20.0 | JavaScript client |
| holochain_client (Rust) | 0.8.0 | Rust client |
| tryorama | 0.19.0 | Testing framework |

### Supporting Stack

| Layer | Technology | Version |
|-------|------------|---------|
| Frontend | SvelteKit | 5.x |
| Desktop | Tauri | 2.x |
| Backend ML | Python | 3.11+ |
| Cryptography | liboqs (PQC) | Latest |
| Database | SQLite (local) | 3.x |
| Testing | Vitest + Tryorama | Latest |
| CI/CD | GitHub Actions | - |
| Documentation | Astro Starlight | Latest |

---

## Phase 0: Foundation Reset (Weeks 1-4)
**January 2025**

### Objective
Establish unified development environment and migrate all projects to Holochain 0.6.

### 0.1 Development Environment Standardization

#### Task 0.1.1: Create Unified Nix Flake
**Duration**: 3 days
**Owner**: Core Team

```nix
# /srv/luminous-dynamics/mycelix-workspace/flake.nix
{
  description = "Mycelix Ecosystem Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # Holochain 0.6 from holonix
    holonix = {
      url = "github:holochain/holonix/main";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Rust toolchain
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, holonix, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        # Holochain 0.6 packages
        holochain = holonix.packages.${system}.holochain;
        lair-keystore = holonix.packages.${system}.lair-keystore;
        hc = holonix.packages.${system}.hc;

        # Rust toolchain for zome development
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          targets = [ "wasm32-unknown-unknown" ];
        };
      in
      {
        devShells = {
          # Full development environment
          default = pkgs.mkShell {
            buildInputs = with pkgs; [
              # Holochain 0.6
              holochain
              lair-keystore
              hc

              # Rust for zomes
              rustToolchain

              # Node.js for frontends
              nodejs_20
              nodePackages.pnpm

              # Python for ML/FL
              python311
              python311Packages.pip
              python311Packages.virtualenv

              # Build tools
              pkg-config
              openssl

              # Development utilities
              just  # Task runner
              watchexec  # File watcher
              jq
            ];

            shellHook = ''
              echo "🍄 Mycelix Development Environment (Holochain 0.6)"
              echo "   holochain: $(holochain --version)"
              echo "   hc: $(hc --version)"
              echo "   node: $(node --version)"
              echo ""
              echo "Commands:"
              echo "  just dev      - Start all services"
              echo "  just test     - Run all tests"
              echo "  just build    - Build all hApps"
            '';
          };

          # CI environment (minimal)
          ci = pkgs.mkShell {
            buildInputs = with pkgs; [
              holochain
              hc
              rustToolchain
              nodejs_20
            ];
          };
        };
      });
}
```

**Deliverables**:
- [ ] Unified flake.nix in workspace root
- [ ] All developers can run `nix develop`
- [ ] CI uses same environment
- [ ] Documentation for setup

#### Task 0.1.2: Create Justfile Task Runner
**Duration**: 1 day

```just
# /srv/luminous-dynamics/mycelix-workspace/justfile

# Default recipe
default:
    @just --list

# === Development ===

# Start all services for local development
dev:
    @echo "Starting Mycelix development environment..."
    just _start-lair &
    just _start-conductor &
    just _start-frontend &
    @echo "All services started!"

# Stop all services
stop:
    pkill -f "lair-keystore" || true
    pkill -f "holochain" || true
    pkill -f "vite" || true

# === Building ===

# Build all hApps
build:
    just build-sdk
    just build-mail
    just build-marketplace
    just build-praxis
    just build-supplychain

build-sdk:
    cd mycelix-sdk && cargo build --release

build-mail:
    cd Mycelix-Mail/happ && hc dna pack . && hc app pack .

build-marketplace:
    cd mycelix-marketplace/holochain && hc dna pack . && hc app pack .

build-praxis:
    cd mycelix-praxis/holochain && hc dna pack . && hc app pack .

build-supplychain:
    cd mycelix-supplychain/holochain && hc dna pack . && hc app pack .

# === Testing ===

# Run all tests
test:
    just test-sdk
    just test-happs
    just test-integration

test-sdk:
    cd mycelix-sdk && cargo test

test-happs:
    cd Mycelix-Mail/happ && npm test
    cd mycelix-marketplace && npm test
    cd mycelix-praxis && npm test

test-integration:
    cd tests/integration && npm test

# Byzantine fault tolerance tests
test-byzantine:
    cd Mycelix-Core/0TML && python -m pytest tests/byzantine/ -v

# === Holochain 0.6 Migration ===

# Check migration status for all hApps
migration-status:
    @echo "=== Holochain 0.6 Migration Status ==="
    @echo ""
    @for dir in Mycelix-Mail mycelix-marketplace mycelix-praxis mycelix-supplychain; do \
        echo "[$dir]"; \
        grep -r "hdk = " $dir/*/Cargo.toml 2>/dev/null || echo "  No Cargo.toml found"; \
        echo ""; \
    done

# === Internal recipes ===

_start-lair:
    lair-keystore server run --piped

_start-conductor:
    holochain -c conductor-config.yaml

_start-frontend:
    cd mycelix-observatory && pnpm dev
```

**Deliverables**:
- [ ] Justfile with all common tasks
- [ ] Documented commands
- [ ] CI integration

### 0.2 Holochain 0.6 Migration

#### Task 0.2.1: Mycelix-Mail Migration
**Duration**: 5 days
**Priority**: HIGH

**Current State**: Holochain 0.5.x
**Target State**: Holochain 0.6.0

**Migration Steps**:

```toml
# Mycelix-Mail/happ/dna/zomes/integrity/Cargo.toml
# BEFORE (0.5.x)
[dependencies]
hdi = "0.5.0"
hdk = "0.5.0"

# AFTER (0.6.0)
[dependencies]
hdi = "0.7.0"
hdk = "0.6.0"
```

**Breaking Changes to Address**:
1. **Entry Definition Syntax**
   ```rust
   // OLD (0.5.x)
   #[hdk_entry_helper]
   pub struct MailMessage { ... }

   // NEW (0.6.0)
   #[hdk_entry_helper]
   #[derive(Clone, PartialEq)]
   pub struct MailMessage { ... }
   ```

2. **Link Type Registration**
   ```rust
   // OLD
   #[hdk_link_types]
   pub enum LinkTypes { ... }

   // NEW - same syntax, but check for new validation requirements
   ```

3. **Client Library Update**
   ```json
   // package.json
   {
     "dependencies": {
       "@holochain/client": "^0.20.0"
     }
   }
   ```

**Validation Checklist**:
- [ ] All zomes compile with hdk 0.6.0
- [ ] All integrity zomes compile with hdi 0.7.0
- [ ] DNA packs successfully
- [ ] hApp packs successfully
- [ ] Unit tests pass with tryorama 0.19.0
- [ ] Frontend connects with @holochain/client 0.20.0

#### Task 0.2.2: Mycelix-Marketplace Migration
**Duration**: 5 days
**Priority**: HIGH

Same migration pattern as Mail. Additional considerations:
- SvelteKit frontend already modern
- Ensure MRC (Mutual Reputation Consensus) zome migrates cleanly

#### Task 0.2.3: Mycelix-Praxis Migration
**Duration**: 7 days
**Priority**: HIGH

**Special Considerations**:
- Currently stuck between 0.4 and 0.6
- More complex zome structure
- W3C VC integration needs testing

**Migration Path**:
1. First migrate to 0.5.x (if needed for incremental)
2. Then migrate to 0.6.0
3. Verify all credential flows work

#### Task 0.2.4: Mycelix-SupplyChain Migration
**Duration**: 5 days
**Priority**: MEDIUM

- Alpha status - simpler migration
- Focus on core provenance functionality

### 0.3 Workspace Structure

#### Task 0.3.1: Reorganize Repository Structure
**Duration**: 2 days

```
/srv/luminous-dynamics/mycelix-workspace/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Unified CI
│       ├── release.yml         # Release automation
│       └── docs.yml            # Documentation deployment
├── flake.nix                   # Unified Nix flake
├── justfile                    # Task runner
├── README.md                   # Ecosystem overview
├── ROADMAP.md                  # This document
│
├── sdk/                        # mycelix-sdk (shared library)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs
│   │   ├── matl/               # MATL trust scoring
│   │   ├── epistemic/          # Epistemic Charter
│   │   ├── credentials/        # Verifiable Credentials
│   │   └── bridge/             # Inter-hApp communication
│   └── bindings/
│       ├── typescript/
│       └── python/
│
├── happs/                      # All Holochain applications
│   ├── mail/                   # Mycelix-Mail
│   │   ├── dna/
│   │   │   ├── zomes/
│   │   │   │   ├── integrity/
│   │   │   │   └── coordinator/
│   │   │   └── dna.yaml
│   │   ├── ui/
│   │   ├── happ.yaml
│   │   └── README.md
│   ├── marketplace/            # Mycelix-Marketplace
│   ├── praxis/                 # Mycelix-Praxis
│   └── supplychain/            # Mycelix-SupplyChain
│
├── core/                       # Mycelix-Core (0TML + Governance)
│   ├── 0tml/                   # Byzantine-resistant FL
│   ├── governance/             # Constitutional framework
│   └── epistemic/              # Epistemic Charter v2.0
│
├── observatory/                # Unified dashboard
│   ├── src/
│   └── package.json
│
├── docs/                       # Unified documentation
│   ├── astro.config.mjs
│   └── src/content/
│
└── tests/
    ├── unit/
    ├── integration/
    └── byzantine/
```

**Deliverables**:
- [ ] New directory structure created
- [ ] All projects moved/symlinked
- [ ] Git history preserved
- [ ] CI updated for new structure

### Phase 0 Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| All hApps on HC 0.6 | `hc --version` returns 0.6.0 in all projects |
| Unified dev environment | `nix develop` works from workspace root |
| All tests passing | `just test` returns 0 |
| Documentation updated | All READMEs reflect new versions |

---

## Phase 1: SDK Extraction (Weeks 5-10)
**February - Mid-March 2025**

### Objective
Extract shared code from all hApps into `mycelix-sdk` for consistency and reuse.

### 1.1 MATL Core Library

#### Task 1.1.1: Extract MATL from Mycelix-Core
**Duration**: 10 days
**Priority**: CRITICAL

```rust
// sdk/src/matl/mod.rs
pub mod pogq;           // Proof of Gradient Quality
pub mod reputation;     // Reputation system
pub mod composite;      // Composite trust scoring
pub mod cartel;         // Cartel detection
pub mod adaptive;       // Adaptive thresholds (new!)
pub mod hierarchical;   // Hierarchical detection (new!)

// Re-exports
pub use pogq::ProofOfGradientQuality;
pub use reputation::{ReputationScore, ReputationHistory};
pub use composite::CompositeTrustScore;
pub use cartel::CartelDetector;
pub use adaptive::AdaptiveThreshold;
pub use hierarchical::HierarchicalDetector;
```

**Implementation**:

```rust
// sdk/src/matl/pogq.rs
use serde::{Deserialize, Serialize};

/// Proof of Gradient Quality - Core trust mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofOfGradientQuality {
    /// Quality score [0.0, 1.0]
    pub quality: f64,
    /// Consistency over time
    pub consistency: f64,
    /// Entropy measure
    pub entropy: f64,
    /// Timestamp of measurement
    pub timestamp: u64,
}

impl ProofOfGradientQuality {
    /// Create new PoGQ measurement
    pub fn new(quality: f64, consistency: f64, entropy: f64) -> Self {
        Self {
            quality: quality.clamp(0.0, 1.0),
            consistency: consistency.clamp(0.0, 1.0),
            entropy,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Compute composite score using MATL formula
    pub fn composite_score(&self, reputation: f64) -> f64 {
        // MATL composite: weighted combination
        const W_QUALITY: f64 = 0.4;
        const W_CONSISTENCY: f64 = 0.3;
        const W_REPUTATION: f64 = 0.3;

        W_QUALITY * self.quality
            + W_CONSISTENCY * self.consistency
            + W_REPUTATION * reputation
    }

    /// Check if score indicates Byzantine behavior
    pub fn is_byzantine(&self, threshold: f64) -> bool {
        self.quality < threshold
    }
}

// sdk/src/matl/adaptive.rs
/// Adaptive threshold based on node history
pub struct AdaptiveThreshold {
    history: Vec<f64>,
    window_size: usize,
}

impl AdaptiveThreshold {
    pub fn new(window_size: usize) -> Self {
        Self {
            history: Vec::with_capacity(window_size),
            window_size,
        }
    }

    /// Update with new observation
    pub fn observe(&mut self, quality: f64) {
        if self.history.len() >= self.window_size {
            self.history.remove(0);
        }
        self.history.push(quality);
    }

    /// Compute adaptive threshold (mean - 2*std)
    pub fn threshold(&self) -> f64 {
        if self.history.is_empty() {
            return 0.5; // Default threshold
        }

        let mean = self.history.iter().sum::<f64>() / self.history.len() as f64;
        let variance = self.history.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / self.history.len() as f64;
        let std = variance.sqrt();

        (mean - 2.0 * std).max(0.1) // Minimum threshold 0.1
    }
}
```

**Deliverables**:
- [ ] MATL core extracted to SDK
- [ ] Unit tests (>90% coverage)
- [ ] Benchmarks for performance
- [ ] Documentation with examples

#### Task 1.1.2: Create Holochain Zome Wrapper
**Duration**: 5 days

```rust
// sdk/src/matl/holochain.rs
use hdk::prelude::*;
use crate::matl::{ProofOfGradientQuality, CompositeTrustScore};

/// Entry type for storing MATL scores on DHT
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MatlScoreEntry {
    pub agent: AgentPubKey,
    pub pogq: ProofOfGradientQuality,
    pub reputation: f64,
    pub composite: f64,
}

/// Get MATL score for an agent
pub fn get_agent_matl_score(agent: &AgentPubKey) -> ExternResult<Option<MatlScoreEntry>> {
    let path = Path::from(format!("matl_scores/{}", agent));
    let links = get_links(
        GetLinksInputBuilder::try_new(path.path_entry_hash()?, LinkTypes::MatlScore)?
            .build()
    )?;

    if let Some(link) = links.first() {
        let record = get(link.target.clone().into_action_hash().unwrap(), GetOptions::default())?;
        if let Some(r) = record {
            let entry: MatlScoreEntry = r.entry().to_app_option()?.unwrap();
            return Ok(Some(entry));
        }
    }
    Ok(None)
}

/// Update MATL score for calling agent
pub fn update_my_matl_score(pogq: ProofOfGradientQuality, reputation: f64) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_latest_pubkey;
    let composite = pogq.composite_score(reputation);

    let entry = MatlScoreEntry {
        agent: agent.clone(),
        pogq,
        reputation,
        composite,
    };

    create_entry(&EntryTypes::MatlScore(entry))
}
```

### 1.2 Epistemic Charter Library

#### Task 1.2.1: Implement 3D Epistemic Cube
**Duration**: 7 days

```rust
// sdk/src/epistemic/cube.rs
use serde::{Deserialize, Serialize};

/// Empirical axis - How to verify
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EmpiricalLevel {
    E0Null,           // Unverifiable belief
    E1Testimonial,    // Personal attestation
    E2PrivateVerify,  // Audit guild verification
    E3Cryptographic,  // ZKP verified
    E4PublicRepro,    // Publicly reproducible
}

/// Normative axis - Who agrees
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum NormativeLevel {
    N0Personal,       // Self only
    N1Communal,       // Local DAO
    N2Network,        // Global consensus
    N3Axiomatic,      // Constitutional/mathematical
}

/// Materiality axis - How long matters
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MaterialityLevel {
    M0Ephemeral,      // Discard immediately
    M1Temporal,       // Prune after state change
    M2Persistent,     // Archive after time
    M3Foundational,   // Preserve forever
}

/// A claim positioned in 3D epistemic space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicClaim {
    /// The claim content
    pub content: String,
    /// Empirical classification
    pub empirical: EmpiricalLevel,
    /// Normative classification
    pub normative: NormativeLevel,
    /// Materiality classification
    pub materiality: MaterialityLevel,
    /// Supporting evidence (hashes)
    pub evidence: Vec<String>,
    /// Issuer
    pub issuer: String,
    /// Timestamp
    pub timestamp: u64,
}

impl EpistemicClaim {
    /// Create a new claim with full classification
    pub fn new(
        content: impl Into<String>,
        empirical: EmpiricalLevel,
        normative: NormativeLevel,
        materiality: MaterialityLevel,
    ) -> Self {
        Self {
            content: content.into(),
            empirical,
            normative,
            materiality,
            evidence: vec![],
            issuer: String::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Check if claim meets minimum verification standards
    pub fn meets_standard(&self, min_e: EmpiricalLevel, min_n: NormativeLevel) -> bool {
        self.empirical as u8 >= min_e as u8 && self.normative as u8 >= min_n as u8
    }

    /// Should this claim be retained based on materiality?
    pub fn should_retain(&self, current_time: u64, retention_days: u64) -> bool {
        match self.materiality {
            MaterialityLevel::M0Ephemeral => false,
            MaterialityLevel::M1Temporal => true, // Caller decides based on state
            MaterialityLevel::M2Persistent => {
                current_time - self.timestamp < retention_days * 86400
            }
            MaterialityLevel::M3Foundational => true,
        }
    }
}
```

### 1.3 Bridge Protocol

#### Task 1.3.1: Design Inter-hApp Communication
**Duration**: 10 days

```rust
// sdk/src/bridge/mod.rs
pub mod protocol;
pub mod registry;
pub mod queries;

// sdk/src/bridge/protocol.rs
use serde::{Deserialize, Serialize};

/// Message types for inter-hApp communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BridgeMessage {
    /// Query reputation across hApps
    ReputationQuery {
        agent: String,
        source_happ: String,
    },

    /// Response with aggregated reputation
    ReputationResponse {
        agent: String,
        scores: Vec<HappReputationScore>,
        aggregate: f64,
    },

    /// Verify credential from another hApp
    CredentialVerify {
        credential_hash: String,
        issuer_happ: String,
    },

    /// Credential verification result
    CredentialResult {
        valid: bool,
        issuer: String,
        claims: Vec<String>,
    },

    /// Broadcast event to interested hApps
    EventBroadcast {
        event_type: String,
        payload: Vec<u8>,
        source_happ: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HappReputationScore {
    pub happ_id: String,
    pub happ_name: String,
    pub score: f64,
    pub interactions: u64,
    pub last_updated: u64,
}

// sdk/src/bridge/registry.rs
use hdk::prelude::*;

/// Path for hApp registry
const HAPP_REGISTRY_PATH: &str = "mycelix_bridge/happs";

/// Register a hApp with the bridge
pub fn register_happ(happ_id: &str, happ_name: &str, capabilities: Vec<String>) -> ExternResult<()> {
    let path = Path::from(HAPP_REGISTRY_PATH);
    path.ensure()?;

    let registration = HappRegistration {
        happ_id: happ_id.to_string(),
        happ_name: happ_name.to_string(),
        capabilities,
        registered_at: sys_time()?.as_secs(),
    };

    create_entry(&EntryTypes::HappRegistration(registration))?;
    Ok(())
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct HappRegistration {
    pub happ_id: String,
    pub happ_name: String,
    pub capabilities: Vec<String>,
    pub registered_at: u64,
}
```

### 1.4 TypeScript Bindings

#### Task 1.4.1: Create TypeScript SDK
**Duration**: 7 days

```typescript
// sdk/bindings/typescript/src/index.ts
export * from './matl';
export * from './epistemic';
export * from './bridge';
export * from './client';

// sdk/bindings/typescript/src/matl.ts
import { AgentPubKey } from '@holochain/client';

export interface ProofOfGradientQuality {
  quality: number;
  consistency: number;
  entropy: number;
  timestamp: number;
}

export interface MatlScore {
  agent: AgentPubKey;
  pogq: ProofOfGradientQuality;
  reputation: number;
  composite: number;
}

export class MatlClient {
  constructor(private cellClient: CellClient) {}

  async getAgentScore(agent: AgentPubKey): Promise<MatlScore | null> {
    return this.cellClient.callZome({
      zome_name: 'matl',
      fn_name: 'get_agent_matl_score',
      payload: agent,
    });
  }

  async updateMyScore(pogq: ProofOfGradientQuality, reputation: number): Promise<ActionHash> {
    return this.cellClient.callZome({
      zome_name: 'matl',
      fn_name: 'update_my_matl_score',
      payload: { pogq, reputation },
    });
  }

  computeComposite(pogq: ProofOfGradientQuality, reputation: number): number {
    const W_QUALITY = 0.4;
    const W_CONSISTENCY = 0.3;
    const W_REPUTATION = 0.3;

    return W_QUALITY * pogq.quality
         + W_CONSISTENCY * pogq.consistency
         + W_REPUTATION * reputation;
  }
}

// sdk/bindings/typescript/src/epistemic.ts
export enum EmpiricalLevel {
  E0Null = 0,
  E1Testimonial = 1,
  E2PrivateVerify = 2,
  E3Cryptographic = 3,
  E4PublicRepro = 4,
}

export enum NormativeLevel {
  N0Personal = 0,
  N1Communal = 1,
  N2Network = 2,
  N3Axiomatic = 3,
}

export enum MaterialityLevel {
  M0Ephemeral = 0,
  M1Temporal = 1,
  M2Persistent = 2,
  M3Foundational = 3,
}

export interface EpistemicClaim {
  content: string;
  empirical: EmpiricalLevel;
  normative: NormativeLevel;
  materiality: MaterialityLevel;
  evidence: string[];
  issuer: string;
  timestamp: number;
}

export function classifyClaim(
  content: string,
  empirical: EmpiricalLevel,
  normative: NormativeLevel,
  materiality: MaterialityLevel
): EpistemicClaim {
  return {
    content,
    empirical,
    normative,
    materiality,
    evidence: [],
    issuer: '',
    timestamp: Date.now(),
  };
}

export function meetsStandard(
  claim: EpistemicClaim,
  minE: EmpiricalLevel,
  minN: NormativeLevel
): boolean {
  return claim.empirical >= minE && claim.normative >= minN;
}
```

### Phase 1 Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| SDK published | `cargo publish` successful |
| TS bindings published | `npm publish` successful |
| All hApps use SDK | No duplicate MATL code |
| Test coverage | >90% for SDK |
| Documentation | API docs generated |

---

## Phase 2: hApp Integration (Weeks 11-18)
**Mid-March - May 2025**

### Objective
Integrate SDK into all hApps and implement Bridge Protocol for cross-hApp communication.

### 2.1 Mycelix-Mail Integration

#### Task 2.1.1: Replace Internal MATL with SDK
**Duration**: 5 days

```rust
// happs/mail/dna/zomes/coordinator/src/trust_filter.rs
use mycelix_sdk::matl::{MatlClient, ProofOfGradientQuality};
use mycelix_sdk::bridge::BridgeClient;

/// Check sender trust before accepting message
pub fn check_sender_trust(sender: AgentPubKey, message: &MailMessage) -> ExternResult<TrustDecision> {
    // Get sender's MATL score from local hApp
    let local_score = MatlClient::get_agent_score(&sender)?;

    // Query cross-hApp reputation via Bridge
    let bridge = BridgeClient::new();
    let cross_happ_scores = bridge.query_reputation(&sender)?;

    // Compute aggregate trust
    let aggregate = compute_aggregate_trust(local_score, cross_happ_scores);

    // Threshold-based decision
    let threshold = get_trust_threshold()?;

    if aggregate >= threshold {
        Ok(TrustDecision::Accept)
    } else if aggregate >= threshold * 0.7 {
        Ok(TrustDecision::Review) // Needs manual review
    } else {
        Ok(TrustDecision::Reject)
    }
}

#[derive(Debug)]
pub enum TrustDecision {
    Accept,
    Review,
    Reject,
}
```

#### Task 2.1.2: Add Epistemic Classification to Messages
**Duration**: 3 days

```rust
// Messages can carry epistemic metadata
use mycelix_sdk::epistemic::{EpistemicClaim, EmpiricalLevel, NormativeLevel, MaterialityLevel};

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EnhancedMailMessage {
    pub subject: String,
    pub body: String,
    pub sender: AgentPubKey,
    pub recipient: AgentPubKey,
    pub timestamp: Timestamp,
    /// Optional epistemic classification
    pub epistemic: Option<EpistemicClaim>,
}
```

### 2.2 Mycelix-Marketplace Integration

#### Task 2.2.1: Unified Reputation Display
**Duration**: 5 days

```svelte
<!-- happs/marketplace/ui/src/lib/components/SellerProfile.svelte -->
<script lang="ts">
  import { MatlClient, BridgeClient } from '@mycelix/sdk';
  import { TrustBadge, ReputationHistory } from '@mycelix/ui';

  export let seller: AgentPubKey;

  let matlScore: MatlScore | null = null;
  let crossHappScores: HappReputationScore[] = [];

  onMount(async () => {
    const matl = new MatlClient(cellClient);
    const bridge = new BridgeClient(cellClient);

    // Get local marketplace score
    matlScore = await matl.getAgentScore(seller);

    // Get cross-hApp reputation
    crossHappScores = await bridge.queryReputation(seller);
  });
</script>

<div class="seller-profile">
  <h2>Seller Reputation</h2>

  {#if matlScore}
    <TrustBadge
      score={matlScore.composite}
      label="Marketplace Trust"
    />

    <div class="cross-happ-scores">
      <h3>Mycelix Ecosystem Reputation</h3>
      {#each crossHappScores as score}
        <div class="happ-score">
          <span class="happ-name">{score.happ_name}</span>
          <TrustBadge score={score.score} size="small" />
          <span class="interactions">({score.interactions} interactions)</span>
        </div>
      {/each}
    </div>

    <ReputationHistory agent={seller} />
  {:else}
    <p>No reputation data available (new user)</p>
  {/if}
</div>
```

#### Task 2.2.2: Credential-Based Verification
**Duration**: 5 days

```typescript
// happs/marketplace/ui/src/lib/services/verification.ts
import { BridgeClient } from '@mycelix/sdk';

export async function verifySellerCredentials(
  seller: AgentPubKey,
  requiredCredentials: string[]
): Promise<VerificationResult> {
  const bridge = new BridgeClient(cellClient);

  const results: CredentialCheck[] = [];

  for (const credType of requiredCredentials) {
    // Check if seller has this credential from any Mycelix hApp
    const credential = await bridge.findCredential(seller, credType);

    if (credential) {
      // Verify credential is valid and not revoked
      const isValid = await bridge.verifyCredential(credential.hash);
      results.push({
        type: credType,
        found: true,
        valid: isValid,
        issuer: credential.issuer,
        issuedAt: credential.timestamp,
      });
    } else {
      results.push({
        type: credType,
        found: false,
        valid: false,
      });
    }
  }

  return {
    allMet: results.every(r => r.found && r.valid),
    checks: results,
  };
}
```

### 2.3 Mycelix-Praxis Integration

#### Task 2.3.1: W3C VC with Epistemic Classification
**Duration**: 7 days

```rust
// happs/praxis/dna/zomes/coordinator/src/credentials.rs
use mycelix_sdk::credentials::{VerifiableCredential, CredentialBuilder};
use mycelix_sdk::epistemic::{EpistemicClaim, EmpiricalLevel, NormativeLevel, MaterialityLevel};

/// Issue an educational credential with epistemic classification
pub fn issue_credential(
    recipient: AgentPubKey,
    credential_type: String,
    claims: Vec<String>,
    evidence: Vec<String>,
) -> ExternResult<VerifiableCredential> {
    let issuer = agent_info()?.agent_latest_pubkey;

    // Build W3C-compliant VC
    let vc = CredentialBuilder::new()
        .issuer(issuer.to_string())
        .subject(recipient.to_string())
        .credential_type(&credential_type)
        .claims(claims.clone())
        .evidence(evidence.clone())
        // Add epistemic classification
        .epistemic(EpistemicClaim::new(
            format!("Educational credential: {}", credential_type),
            // Cryptographically verifiable (signed by institution)
            EmpiricalLevel::E3Cryptographic,
            // Network-wide recognition
            NormativeLevel::N2Network,
            // Persistent (important record)
            MaterialityLevel::M2Persistent,
        ))
        .build()?;

    // Store on DHT
    create_entry(&EntryTypes::VerifiableCredential(vc.clone()))?;

    // Register with Bridge for cross-hApp discovery
    BridgeClient::new().register_credential(&vc)?;

    Ok(vc)
}
```

### 2.4 Mycelix-SupplyChain Integration

#### Task 2.4.1: Provenance with MATL Trust
**Duration**: 5 days

```rust
// happs/supplychain/dna/zomes/coordinator/src/provenance.rs
use mycelix_sdk::matl::MatlClient;
use mycelix_sdk::epistemic::{EpistemicClaim, EmpiricalLevel};

/// Record provenance event with trust context
pub fn record_provenance_event(
    product_id: String,
    event_type: ProvenanceEventType,
    location: Option<GeoLocation>,
    evidence: Vec<String>,
) -> ExternResult<ProvenanceRecord> {
    let reporter = agent_info()?.agent_latest_pubkey;

    // Get reporter's MATL trust score
    let matl = MatlClient::new();
    let trust_score = matl.get_agent_score(&reporter)?
        .map(|s| s.composite)
        .unwrap_or(0.5); // Default for new reporters

    // Classify evidence quality
    let evidence_level = classify_evidence(&evidence);

    let record = ProvenanceRecord {
        product_id,
        event_type,
        reporter: reporter.clone(),
        reporter_trust: trust_score,
        location,
        evidence,
        evidence_classification: evidence_level,
        timestamp: sys_time()?.as_secs(),
        // Epistemic claim about this provenance event
        epistemic: EpistemicClaim::new(
            format!("{:?} event for product {}", event_type, product_id),
            evidence_level,
            NormativeLevel::N1Communal, // Supply chain participants agree
            MaterialityLevel::M2Persistent, // Keep for audit trail
        ),
    };

    create_entry(&EntryTypes::ProvenanceRecord(record.clone()))?;

    Ok(record)
}

fn classify_evidence(evidence: &[String]) -> EmpiricalLevel {
    // Classify based on evidence types present
    let has_cryptographic = evidence.iter().any(|e| e.starts_with("sig:") || e.starts_with("zkp:"));
    let has_iot = evidence.iter().any(|e| e.starts_with("iot:"));
    let has_image = evidence.iter().any(|e| e.starts_with("img:"));

    if has_cryptographic {
        EmpiricalLevel::E3Cryptographic
    } else if has_iot {
        EmpiricalLevel::E2PrivateVerify
    } else if has_image {
        EmpiricalLevel::E1Testimonial
    } else {
        EmpiricalLevel::E0Null
    }
}
```

### 2.5 Bridge Protocol Implementation

#### Task 2.5.1: Implement Core Bridge Zome
**Duration**: 10 days

```rust
// sdk/src/bridge/zome/src/lib.rs
use hdk::prelude::*;

mod types;
mod registry;
mod reputation;
mod credentials;

pub use types::*;

/// Initialize bridge for this hApp
#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    // Create paths for bridge data
    let paths = vec![
        "mycelix_bridge/happs",
        "mycelix_bridge/reputation",
        "mycelix_bridge/credentials",
        "mycelix_bridge/events",
    ];

    for path_str in paths {
        Path::from(path_str).ensure()?;
    }

    Ok(InitCallbackResult::Pass)
}

/// Query reputation for an agent across all registered hApps
#[hdk_extern]
pub fn query_cross_happ_reputation(agent: AgentPubKey) -> ExternResult<CrossHappReputation> {
    let registered_happs = registry::get_registered_happs()?;
    let mut scores = Vec::new();

    for happ in registered_happs {
        if let Some(score) = reputation::get_happ_reputation(&agent, &happ.happ_id)? {
            scores.push(score);
        }
    }

    // Compute aggregate using MATL weighting
    let aggregate = if scores.is_empty() {
        0.5 // Default for unknown agents
    } else {
        let total_interactions: u64 = scores.iter().map(|s| s.interactions).sum();
        scores.iter()
            .map(|s| s.score * (s.interactions as f64 / total_interactions as f64))
            .sum()
    };

    Ok(CrossHappReputation {
        agent,
        scores,
        aggregate,
        queried_at: sys_time()?.as_secs(),
    })
}

/// Verify a credential issued by any Mycelix hApp
#[hdk_extern]
pub fn verify_cross_happ_credential(hash: ActionHash) -> ExternResult<CredentialVerification> {
    credentials::verify_credential(hash)
}

/// Broadcast event to all interested hApps
#[hdk_extern]
pub fn broadcast_event(event: BridgeEvent) -> ExternResult<()> {
    let path = Path::from(format!("mycelix_bridge/events/{}", event.event_type));
    path.ensure()?;

    create_link(
        path.path_entry_hash()?,
        hash_entry(&event)?,
        LinkTypes::BridgeEvent,
        event.event_type.as_bytes().to_vec(),
    )?;

    Ok(())
}

/// Subscribe to events of a specific type
#[hdk_extern]
pub fn get_events(event_type: String, since: u64) -> ExternResult<Vec<BridgeEvent>> {
    let path = Path::from(format!("mycelix_bridge/events/{}", event_type));

    let links = get_links(
        GetLinksInputBuilder::try_new(path.path_entry_hash()?, LinkTypes::BridgeEvent)?
            .build()
    )?;

    let mut events = Vec::new();
    for link in links {
        if let Some(record) = get(link.target.into_action_hash().unwrap(), GetOptions::default())? {
            let event: BridgeEvent = record.entry().to_app_option()?.unwrap();
            if event.timestamp >= since {
                events.push(event);
            }
        }
    }

    events.sort_by_key(|e| e.timestamp);
    Ok(events)
}
```

### Phase 2 Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| All hApps use SDK | No duplicate implementations |
| Bridge Protocol working | Cross-hApp reputation queries succeed |
| Credential portability | Praxis credentials verifiable in Marketplace |
| Integration tests | 100% pass rate |
| Performance | <500ms for cross-hApp queries |

---

## Phase 3: Observatory & Production (Weeks 19-30)
**June - August 2025**

### Objective
Build unified monitoring dashboard and prepare for production pilots.

### 3.1 Mycelix Observatory

#### Task 3.1.1: Dashboard Architecture
**Duration**: 10 days

```
observatory/
├── src/
│   ├── routes/
│   │   ├── +page.svelte           # Dashboard home
│   │   ├── network/
│   │   │   └── +page.svelte       # Network health
│   │   ├── trust/
│   │   │   └── +page.svelte       # Trust distribution
│   │   ├── happs/
│   │   │   ├── +page.svelte       # hApp overview
│   │   │   ├── mail/
│   │   │   ├── marketplace/
│   │   │   ├── praxis/
│   │   │   └── supplychain/
│   │   ├── governance/
│   │   │   └── +page.svelte       # DAO proposals
│   │   └── byzantine/
│   │       └── +page.svelte       # Attack detection
│   ├── lib/
│   │   ├── components/
│   │   │   ├── NetworkGraph.svelte
│   │   │   ├── TrustHeatmap.svelte
│   │   │   ├── ByzantineAlerts.svelte
│   │   │   └── GovernanceWidget.svelte
│   │   ├── stores/
│   │   │   ├── network.ts
│   │   │   ├── trust.ts
│   │   │   └── alerts.ts
│   │   └── services/
│   │       ├── holochain.ts
│   │       └── metrics.ts
├── package.json
└── svelte.config.js
```

#### Task 3.1.2: Network Health Visualization
**Duration**: 7 days

```svelte
<!-- observatory/src/lib/components/NetworkGraph.svelte -->
<script lang="ts">
  import * as d3 from 'd3';
  import { onMount } from 'svelte';
  import { networkStore } from '$lib/stores/network';

  let svg: SVGElement;
  let nodes: NetworkNode[] = [];
  let links: NetworkLink[] = [];

  networkStore.subscribe(data => {
    nodes = data.nodes;
    links = data.links;
    if (svg) updateVisualization();
  });

  function updateVisualization() {
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id))
      .force('charge', d3.forceManyBody().strength(-100))
      .force('center', d3.forceCenter(400, 300));

    // Draw nodes colored by trust score
    d3.select(svg)
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', 8)
      .attr('fill', d => trustToColor(d.trustScore))
      .attr('stroke', d => d.isByzantine ? '#ff0000' : '#333')
      .attr('stroke-width', d => d.isByzantine ? 3 : 1);

    // Draw links weighted by connection strength
    d3.select(svg)
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', d => d.strength);
  }

  function trustToColor(score: number): string {
    // Green for high trust, red for low
    const hue = score * 120; // 0 = red, 120 = green
    return `hsl(${hue}, 70%, 50%)`;
  }
</script>

<div class="network-graph">
  <h2>Mycelix Network Topology</h2>
  <svg bind:this={svg} width="800" height="600">
    <!-- D3 renders here -->
  </svg>
  <div class="legend">
    <div class="legend-item">
      <span class="dot" style="background: hsl(120, 70%, 50%)"></span>
      High Trust (>0.8)
    </div>
    <div class="legend-item">
      <span class="dot" style="background: hsl(60, 70%, 50%)"></span>
      Medium Trust (0.5-0.8)
    </div>
    <div class="legend-item">
      <span class="dot" style="background: hsl(0, 70%, 50%)"></span>
      Low Trust (<0.5)
    </div>
    <div class="legend-item">
      <span class="dot" style="background: #333; border: 3px solid red"></span>
      Detected Byzantine
    </div>
  </div>
</div>
```

#### Task 3.1.3: Byzantine Detection Dashboard
**Duration**: 7 days

```svelte
<!-- observatory/src/routes/byzantine/+page.svelte -->
<script lang="ts">
  import { byzantineStore } from '$lib/stores/byzantine';
  import { ByzantineAlerts, AttackTimeline, DefenseMetrics } from '$lib/components';

  let alerts: ByzantineAlert[] = [];
  let attackHistory: AttackEvent[] = [];
  let defenseStats: DefenseStats;

  byzantineStore.subscribe(data => {
    alerts = data.activeAlerts;
    attackHistory = data.attackHistory;
    defenseStats = data.defenseStats;
  });
</script>

<div class="byzantine-dashboard">
  <header>
    <h1>Byzantine Fault Detection</h1>
    <div class="status-indicator" class:healthy={alerts.length === 0}>
      {alerts.length === 0 ? '✓ Network Healthy' : `⚠ ${alerts.length} Active Alerts`}
    </div>
  </header>

  <div class="metrics-grid">
    <div class="metric-card">
      <h3>Byzantine Tolerance</h3>
      <div class="big-number">{defenseStats.tolerance}%</div>
      <p>Current adversarial capacity</p>
    </div>

    <div class="metric-card">
      <h3>Detection Rate</h3>
      <div class="big-number">{defenseStats.detectionRate}%</div>
      <p>Attacks successfully identified</p>
    </div>

    <div class="metric-card">
      <h3>False Positive Rate</h3>
      <div class="big-number">{defenseStats.falsePositiveRate}%</div>
      <p>Honest nodes incorrectly flagged</p>
    </div>

    <div class="metric-card">
      <h3>Active Nodes</h3>
      <div class="big-number">{defenseStats.activeNodes}</div>
      <p>Participating in consensus</p>
    </div>
  </div>

  <section class="alerts-section">
    <h2>Active Alerts</h2>
    <ByzantineAlerts {alerts} />
  </section>

  <section class="timeline-section">
    <h2>Attack History (Last 30 Days)</h2>
    <AttackTimeline events={attackHistory} />
  </section>

  <section class="defense-section">
    <h2>Defense Mechanisms</h2>
    <DefenseMetrics stats={defenseStats} />
  </section>
</div>
```

### 3.2 Production Infrastructure

#### Task 3.2.1: Bootstrap Server Deployment
**Duration**: 5 days

```yaml
# infrastructure/k8s/bootstrap-server.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mycelix-bootstrap
  namespace: mycelix
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mycelix-bootstrap
  template:
    metadata:
      labels:
        app: mycelix-bootstrap
    spec:
      containers:
      - name: kitsune2-bootstrap
        image: holochain/kitsune2-bootstrap:0.3.2
        ports:
        - containerPort: 8787
        env:
        - name: RUST_LOG
          value: "info"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8787
          initialDelaySeconds: 10
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: mycelix-bootstrap
  namespace: mycelix
spec:
  selector:
    app: mycelix-bootstrap
  ports:
  - port: 8787
    targetPort: 8787
  type: LoadBalancer
```

#### Task 3.2.2: Monitoring Stack
**Duration**: 5 days

```yaml
# infrastructure/monitoring/prometheus-config.yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093

rule_files:
  - /etc/prometheus/rules/*.yml

scrape_configs:
  - job_name: 'mycelix-conductors'
    static_configs:
      - targets: ['conductor-1:8888', 'conductor-2:8888', 'conductor-3:8888']
    metrics_path: /metrics

  - job_name: 'mycelix-bootstrap'
    static_configs:
      - targets: ['bootstrap:8787']

  - job_name: 'mycelix-observatory'
    static_configs:
      - targets: ['observatory:3000']

# infrastructure/monitoring/alerts.yml
groups:
- name: mycelix
  rules:
  - alert: HighByzantineActivity
    expr: mycelix_byzantine_detected_nodes > 10
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High Byzantine activity detected"
      description: "{{ $value }} nodes flagged as Byzantine in the last 5 minutes"

  - alert: LowNetworkParticipation
    expr: mycelix_active_nodes < 50
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: "Network participation critically low"
      description: "Only {{ $value }} nodes active, below minimum threshold"

  - alert: HighLatency
    expr: histogram_quantile(0.95, mycelix_request_duration_seconds_bucket) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High request latency detected"
      description: "95th percentile latency is {{ $value }}s"
```

### 3.3 CLI Tool

#### Task 3.3.1: Mycelix CLI
**Duration**: 10 days

```rust
// cli/src/main.rs
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "mycelix")]
#[command(about = "Mycelix ecosystem CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new Mycelix project
    Init {
        /// Project name
        name: String,
        /// Template to use
        #[arg(long, default_value = "minimal")]
        template: String,
    },

    /// Start local development network
    Dev {
        /// hApps to include
        #[arg(long)]
        happs: Vec<String>,
    },

    /// Build hApps
    Build {
        /// Specific hApp to build (or all)
        #[arg(long)]
        happ: Option<String>,
    },

    /// Run tests
    Test {
        /// Test type: unit, integration, byzantine
        #[arg(long, default_value = "unit")]
        r#type: String,
    },

    /// Query network status
    Status,

    /// Manage trust scores
    Trust {
        #[command(subcommand)]
        action: TrustCommands,
    },
}

#[derive(Subcommand)]
enum TrustCommands {
    /// Get trust score for an agent
    Get { agent: String },
    /// List all agents with trust scores
    List,
    /// Export trust data
    Export { output: String },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init { name, template } => {
            println!("🍄 Creating new Mycelix project: {}", name);
            init_project(&name, &template).await?;
        }
        Commands::Dev { happs } => {
            println!("🚀 Starting development network...");
            start_dev_network(&happs).await?;
        }
        Commands::Build { happ } => {
            build_happs(happ.as_deref()).await?;
        }
        Commands::Test { r#type } => {
            run_tests(&r#type).await?;
        }
        Commands::Status => {
            show_network_status().await?;
        }
        Commands::Trust { action } => {
            handle_trust_command(action).await?;
        }
    }

    Ok(())
}

async fn init_project(name: &str, template: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Scaffold new project from template
    let templates = vec!["minimal", "marketplace", "education", "supply-chain", "social"];

    if !templates.contains(&template) {
        return Err(format!("Unknown template: {}. Available: {:?}", template, templates).into());
    }

    // Create directory structure
    std::fs::create_dir_all(format!("{}/dna/zomes/integrity", name))?;
    std::fs::create_dir_all(format!("{}/dna/zomes/coordinator", name))?;
    std::fs::create_dir_all(format!("{}/ui/src", name))?;
    std::fs::create_dir_all(format!("{}/tests", name))?;

    // Copy template files
    copy_template_files(name, template)?;

    println!("✅ Project created at ./{}", name);
    println!("");
    println!("Next steps:");
    println!("  cd {}", name);
    println!("  nix develop");
    println!("  just dev");

    Ok(())
}
```

### Phase 3 Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| Observatory live | Public URL accessible |
| Bootstrap servers | 3+ replicas running |
| Monitoring | Prometheus + Grafana dashboards |
| CLI published | `cargo install mycelix-cli` works |
| Documentation | docs.mycelix.net live |

---

## Phase 4: Pilots & Validation (Weeks 31-48)
**September - December 2025**

### Objective
Deploy production pilots and validate academic claims.

### 4.1 Healthcare FL Pilot

#### Task 4.1.1: Partner Onboarding
**Duration**: Ongoing (Q3-Q4)

**Target Institutions**: 3-5 medical research centers

**Requirements**:
- HIPAA compliance verified
- IRB approval obtained
- Data sharing agreements signed
- Technical integration planned

**Pilot Scope**:
- Federated learning on medical imaging
- 45% Byzantine tolerance demonstration
- Privacy-preserving model updates
- Differential privacy verification

#### Task 4.1.2: Healthcare-Specific Configuration
**Duration**: 10 days

```python
# core/0tml/configs/healthcare.py
from mycelix_0tml import FederatedConfig, PrivacyConfig, ByzantineConfig

HEALTHCARE_CONFIG = FederatedConfig(
    # HIPAA compliance
    privacy=PrivacyConfig(
        differential_privacy=True,
        epsilon=1.0,  # Strong privacy guarantee
        delta=1e-5,
        gradient_clipping=1.0,
        noise_multiplier=1.1,
    ),

    # Byzantine resistance
    byzantine=ByzantineConfig(
        tolerance=0.45,  # 45% adversarial capacity
        detection_algorithm="adaptive_pogq",
        hierarchical=True,  # O(n log n) scaling
        cartel_detection=True,
    ),

    # Model settings
    model=ModelConfig(
        architecture="resnet18",  # Medical imaging
        local_epochs=5,
        batch_size=32,
        learning_rate=0.01,
    ),

    # Communication
    communication=CommConfig(
        rounds=100,
        min_participants=10,
        async_updates=True,
        compression="topk_sparsification",
    ),

    # Audit trail
    audit=AuditConfig(
        enabled=True,
        log_gradients=False,  # Privacy: don't log raw gradients
        log_aggregations=True,
        immutable_storage="holochain",
    ),
)
```

### 4.2 Supply Chain Pilot

#### Task 4.2.1: Manufacturing Partner Integration
**Duration**: Ongoing (Q4)

**Target**: 1 large manufacturer with supply chain

**Integration Points**:
- ERP system (SAP/Oracle)
- IoT sensors
- Quality management system
- Logistics tracking

**Pilot Scope**:
- Track raw material → finished goods
- Certification verification
- Recall simulation
- Product passport generation

### 4.3 Education Pilot

#### Task 4.3.1: University Partnership
**Duration**: Ongoing (Q4)

**Target**: 1 university, 1 department

**Pilot Scope**:
- Course completion credentials
- Cross-institution verification
- Privacy-preserving learning analytics
- Federated model for adaptive learning

### 4.4 Academic Validation

#### Task 4.4.1: Paper Submissions
**Duration**: Q4 2025 - Q1 2026

**Target Venues**:

| Paper | Venue | Deadline | Topic |
|-------|-------|----------|-------|
| 0TML Main | MLSys 2026 | Oct 2025 | Byzantine-resistant FL |
| PoGQ Analysis | ICML 2026 | Jan 2026 | Trust mechanisms |
| Healthcare FL | CHIL 2026 | Feb 2026 | Medical imaging FL |
| MATL Theory | NeurIPS 2026 | May 2026 | Theoretical analysis |

#### Task 4.4.2: Grant Applications
**Duration**: Q1-Q2 2026

| Grant | Agency | Deadline | Amount |
|-------|--------|----------|--------|
| CISE Research | NSF | Jun 2026 | $500K |
| R01 | NIH | Jun 2026 | $1.5M |
| ERC Starting | EU | Oct 2026 | €1.5M |

### Phase 4 Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| Healthcare pilot | 3+ institutions active |
| Supply chain pilot | 1 manufacturer live |
| Education pilot | 1 university live |
| Academic papers | 2+ submitted |
| Grant applications | 2+ submitted |
| Total users | 500+ |

---

## Phase 5: Ecosystem Growth (2026)
**January - December 2026**

### Objective
Scale ecosystem, enable third-party development, achieve sustainability.

### 5.1 Developer Ecosystem

#### Task 5.1.1: Developer Portal
**Duration**: Q1 2026

- Comprehensive tutorials
- Interactive playground
- Template gallery
- Community showcase

#### Task 5.1.2: Hackathons & Grants
**Duration**: Ongoing

- Quarterly hackathons
- Developer grants program
- Bug bounty program
- Community contributions

### 5.2 Governance Activation

#### Task 5.2.1: DAO Launch
**Duration**: Q2 2026

- Governance token (if decided)
- Proposal system
- Voting mechanisms
- Treasury management

#### Task 5.2.2: Charter Ratification
**Duration**: Q2 2026

- Community review of all charters
- Voting on constitutional amendments
- Multi-stakeholder governance

### 5.3 Multi-Chain Expansion

#### Task 5.3.1: Ethereum Bridge
**Duration**: Q3 2026

- EVM compatibility layer
- Asset bridging
- Cross-chain credentials

#### Task 5.3.2: Cosmos Integration
**Duration**: Q4 2026

- IBC protocol support
- Interchain accounts
- Cross-ecosystem trust

### Phase 5 Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| Active developers | 100+ |
| Third-party hApps | 10+ |
| DAO participation | 50%+ voter turnout |
| Multi-chain live | 2+ chains |
| Total users | 5,000+ |
| Academic citations | 50+ |

---

## Resource Requirements

### Team (Minimal Viable)

| Role | FTE | Phase 0-2 | Phase 3-4 | Phase 5 |
|------|-----|-----------|-----------|---------|
| Lead Developer | 1 | ✅ | ✅ | ✅ |
| Protocol Engineer | 0.5 | ✅ | ✅ | ✅ |
| Frontend Developer | 0.5 | - | ✅ | ✅ |
| DevOps Engineer | 0.25 | - | ✅ | ✅ |
| Technical Writer | 0.25 | - | ✅ | ✅ |
| Community Manager | 0.25 | - | - | ✅ |

### Infrastructure Costs (Monthly)

| Component | Phase 0-2 | Phase 3-4 | Phase 5 |
|-----------|-----------|-----------|---------|
| Bootstrap Servers | $0 (local) | $300 | $500 |
| Monitoring | $0 | $100 | $200 |
| Documentation | $0 | $50 | $100 |
| CI/CD | $0 | $100 | $200 |
| **Total** | **$0** | **$550** | **$1,000** |

### Development Tools (One-Time)

| Tool | Cost |
|------|------|
| GitHub Pro | Included |
| Vercel Pro | $20/mo |
| Domain renewals | ~$100/yr |
| Security audit | $10-50K (Phase 4) |

---

## Risk Mitigation

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Holochain 0.6 instability | Medium | High | Pin versions, test extensively |
| Byzantine attack discovery | Low | Critical | Continuous security research |
| Performance issues at scale | Medium | Medium | Load testing, optimization |
| Integration complexity | High | Medium | Modular architecture, SDK |

### Business Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Pilot partner dropout | Medium | High | Multiple partners per vertical |
| Funding gap | Medium | High | Grant applications, consulting |
| Regulatory changes | Low | High | Legal review, compliance focus |
| Competition | Medium | Medium | Unique value proposition, speed |

### Contingency Plans

1. **If Holochain 0.6 proves unstable**: Fall back to 0.5.x, contribute fixes upstream
2. **If pilot partners drop out**: Pivot to open community testing
3. **If funding is insufficient**: Focus on core 0TML, defer hApps
4. **If academic papers rejected**: Iterate, target alternative venues

---

## Appendices

### A. Holochain 0.6 Migration Checklist

```markdown
# Holochain 0.6 Migration Checklist

## Per-hApp Checklist

### 1. Dependencies
- [ ] Update hdk to 0.6.0
- [ ] Update hdi to 0.7.0
- [ ] Update @holochain/client to 0.20.0
- [ ] Update tryorama to 0.19.0

### 2. Integrity Zomes
- [ ] Check entry definitions
- [ ] Verify link type registrations
- [ ] Update validation functions
- [ ] Test all constraints

### 3. Coordinator Zomes
- [ ] Update extern function signatures
- [ ] Check HDK API changes
- [ ] Verify signal handling
- [ ] Test all functions

### 4. Packaging
- [ ] DNA packs successfully
- [ ] hApp packs successfully
- [ ] Bundle includes all assets

### 5. Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing complete
- [ ] Performance benchmarked

### 6. Frontend
- [ ] Client library updated
- [ ] Connection handling updated
- [ ] All API calls verified
- [ ] UI tested end-to-end
```

### B. SDK API Reference (Summary)

```rust
// mycelix-sdk crate structure

// MATL (Trust)
pub mod matl {
    pub struct ProofOfGradientQuality { ... }
    pub struct ReputationScore { ... }
    pub struct CompositeTrustScore { ... }
    pub struct AdaptiveThreshold { ... }
    pub struct HierarchicalDetector { ... }
    pub struct CartelDetector { ... }
}

// Epistemic (Truth)
pub mod epistemic {
    pub enum EmpiricalLevel { E0, E1, E2, E3, E4 }
    pub enum NormativeLevel { N0, N1, N2, N3 }
    pub enum MaterialityLevel { M0, M1, M2, M3 }
    pub struct EpistemicClaim { ... }
}

// Bridge (Inter-hApp)
pub mod bridge {
    pub struct BridgeClient { ... }
    pub enum BridgeMessage { ... }
    pub struct CrossHappReputation { ... }
}

// Credentials (W3C VC)
pub mod credentials {
    pub struct VerifiableCredential { ... }
    pub struct CredentialBuilder { ... }
    pub fn verify(vc: &VerifiableCredential) -> bool;
}
```

### C. Key Contacts & Resources

| Resource | URL/Contact |
|----------|-------------|
| Holochain Discord | discord.gg/holochain |
| Holochain Docs | developer.holochain.org |
| Mycelix Repo | github.com/Luminous-Dynamics/mycelix |
| Lead Contact | tristan.stoltz@evolvingresonantcocreationism.com |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-30 | Initial comprehensive roadmap |

---

*This roadmap is a living document. Updates will be made as the project evolves.*

**Next Review**: End of Phase 0 (January 31, 2025)
