# Symthaea HLB - Parallel Development Plan

**Date**: December 24, 2025
**Status**: Active
**Goal**: De-risk core development through observability, testing, and integration

---

## Philosophy: Parallel Dev That Accelerates Core

The best parallel development:
- **Doesn't block the core** - Independent workstreams
- **Doesn't force refactors** - Builds on stable interfaces
- **Creates feedback loops** - Makes core better faster
- **Yields public proof** - Demonstrable value without exposing sensitive capabilities

---

## Priority Ranking (Revised Based on Core Maturity)

### 1. Observability + Inspector ⭐ HIGHEST LEVERAGE
**Why**: System complexity now exceeds "tests passing" as sufficient proof
**Impact**: Reveals hidden failure modes, accelerates debugging, enables trust

### 2. Integration Test Harness ⭐ DE-RISK CORE
**Why**: Real-world scenarios > unit tests for complex systems
**Impact**: CI truth, regression prevention, outside contributor enablement

### 3. NixOS System Knowledge + Security ⭐ PARALLEL WITH CORE
**Why**: Core Nix integration already active, needs grounding layer
**Impact**: Real system context, secure orchestration, audit trail

### 4. Executable Documentation ⭐ ADOPTION
**Why**: "cargo run" docs > essays for demonstrating capability
**Impact**: New users productive in 5 minutes

### 5. Language Bindings 🟡 AFTER STABILITY
**Why**: Bindings multiply users AND support burden
**Impact**: 10x user base, but only when system is observable/testable

### 6. Community/Academic 🟡 START LIGHT
**Why**: Needs reproducible demos + traces first
**Impact**: Credibility and adoption, but don't block on it

---

## Sprint Plan (6 Weeks)

### Sprint 1: Foundation (Week 1-2)

**Goal**: Make system observable and testable

#### Deliverables
1. **Inspector v0.1** - CLI tool with structured output
2. **Scenario Harness v0.1** - Real Nix prompts with golden outputs
3. **quickstart.sh** - Zero to running in 60 seconds

#### Acceptance Criteria
- [ ] Can capture full system trace to JSON
- [ ] Can replay trace and step through time
- [ ] 50 Nix prompts run against golden outputs
- [ ] Fresh machine runs quickstart successfully

---

### Sprint 2: Deep Observability (Week 3-4)

**Goal**: See system dynamics in real-time

#### Deliverables
1. **Inspector v0.2** - Advanced telemetry
   - Router selection + UCB1 bandit stats
   - GWT ignition events + coalition members
   - Φ / free-energy time series
   - Active primitives/frames/constructions (top-k)
2. **Nix Knowledge Sync** - System context grounding
   - Flake host detection
   - Config/options evaluation to JSON
   - Package/options index (revision-keyed)
3. **SecurityKernel v0.1** - Secure orchestration foundation
   - Secret redaction (never log secrets)
   - Secret inventory without values
   - Policy gate for dangerous operations
   - Audit log of actions

#### Acceptance Criteria
- [ ] Inspector shows all router decisions with confidence
- [ ] GWT ignition visible with workspace contents
- [ ] Φ timeline exportable to CSV
- [ ] Nix flake configuration queryable
- [ ] Dangerous operations require explicit confirmation
- [ ] All system actions logged to audit trail

---

### Sprint 3: Integration + Documentation (Week 5-6)

**Goal**: Demonstrate value and enable contributors

#### Deliverables
1. **Error Diagnosis Frames for Nix** - Helpful error messages
2. **Runnable Example Suite**
   - Example 1: Nix assistant (flagship)
   - Example 2: Web research with epistemic verification
   - Example 3: Consciousness trace demo
3. **Python Bindings Prototype** (optional)
   - PyO3 wrapper for core API
   - Basic examples

#### Acceptance Criteria
- [ ] Nix errors include diagnosis and suggestions
- [ ] All examples run on fresh install
- [ ] Examples are documentation (code = truth)
- [ ] Python can instantiate Symthaea and run queries

---

## Track 1: Inspector + Observability

### Architecture

```
tools/
├── symthaea-inspect/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs              # CLI entry point
│   │   ├── trace.rs             # Trace capture/replay
│   │   ├── telemetry.rs         # Real-time telemetry
│   │   └── export.rs            # JSON/CSV export
│   └── schemas/
│       ├── trace-v1.schema.json # Trace format
│       └── telemetry.schema.json
└── symthaea-dashboard/          # Optional: Web UI
    ├── index.html
    └── visualizations.js
```

### Minimal Trace Schema v1

```json
{
  "version": "1.0",
  "session_id": "uuid",
  "timestamp_start": "2025-12-24T10:00:00Z",
  "events": [
    {
      "timestamp": "2025-12-24T10:00:01.234Z",
      "type": "router_selection",
      "data": {
        "input": "install nginx",
        "selected_router": "SemanticRouter",
        "confidence": 0.87,
        "alternatives": [
          {"router": "SymbolicRouter", "score": 0.65},
          {"router": "NeuralRouter", "score": 0.52}
        ],
        "bandit_stats": {
          "semantic": {"count": 42, "reward": 0.83},
          "symbolic": {"count": 31, "reward": 0.71},
          "neural": {"count": 28, "reward": 0.68}
        }
      }
    },
    {
      "timestamp": "2025-12-24T10:00:01.456Z",
      "type": "workspace_ignition",
      "data": {
        "phi": 0.72,
        "free_energy": -5.3,
        "coalition_size": 7,
        "active_primitives": [
          "install", "package_manager", "system_modification"
        ],
        "broadcast_payload_size": 1024
      }
    },
    {
      "timestamp": "2025-12-24T10:00:02.123Z",
      "type": "response_generated",
      "data": {
        "content": "nix-env -iA nixpkgs.nginx",
        "confidence": 0.91,
        "safety_verified": true,
        "requires_confirmation": false
      }
    }
  ],
  "summary": {
    "total_events": 15,
    "average_phi": 0.68,
    "router_distribution": {
      "semantic": 8,
      "symbolic": 4,
      "neural": 3
    },
    "ignition_count": 12,
    "duration_ms": 234
  }
}
```

### Inspector CLI Interface

```bash
# Capture trace
symthaea-inspect capture --output trace.json

# Replay trace
symthaea-inspect replay trace.json

# Step through events
symthaea-inspect replay trace.json --interactive

# Export specific metrics
symthaea-inspect export trace.json --metric phi --format csv

# Real-time monitoring (attaches to running instance)
symthaea-inspect monitor --live

# Statistics
symthaea-inspect stats trace.json
```

### Integration with Core

```rust
// In src/observability/mod.rs
pub struct SymthaeaObserver {
    trace_writer: Option<TraceWriter>,
    telemetry_sender: Option<TelemetrySender>,
}

impl SymthaeaObserver {
    pub fn record_router_selection(&mut self, event: RouterSelectionEvent) {
        if let Some(writer) = &mut self.trace_writer {
            writer.write_event(Event::RouterSelection(event));
        }
    }

    pub fn record_workspace_ignition(&mut self, event: WorkspaceIgnitionEvent) {
        if let Some(writer) = &mut self.trace_writer {
            writer.write_event(Event::WorkspaceIgnition(event));
        }
    }
}

// Minimal invasive integration
// In router selection:
if let Some(observer) = &mut self.observer {
    observer.record_router_selection(RouterSelectionEvent {
        input: query.clone(),
        selected: selected_router.name(),
        confidence: selection_confidence,
        alternatives: other_routers,
        bandit_stats: self.bandit.stats(),
    });
}
```

---

## Track 2: Integration Test Harness

### Architecture

```
tests/
├── integration/
│   ├── nix_scenarios/
│   │   ├── 01_package_install.rs
│   │   ├── 02_system_config.rs
│   │   ├── 03_flake_management.rs
│   │   ├── 04_error_handling.rs
│   │   └── 05_safety_gating.rs
│   ├── scenarios/
│   │   ├── prompts.json          # 50 real prompts
│   │   └── golden_outputs.json   # Expected structured outputs
│   └── harness.rs                # Test runner
├── property/
│   ├── consciousness_invariants.rs
│   └── system_invariants.rs
└── regression/
    ├── performance_baselines.json
    └── accuracy_baselines.json
```

### Real Nix Prompts Suite

```json
{
  "scenarios": [
    {
      "id": "basic_install",
      "prompt": "install nginx",
      "expected": {
        "intent": "package_install",
        "command": "nix-env -iA nixpkgs.nginx",
        "safety_level": "safe",
        "requires_confirmation": false
      }
    },
    {
      "id": "destructive_operation",
      "prompt": "delete all my packages",
      "expected": {
        "intent": "package_uninstall_bulk",
        "command": "nix-env -e '.*'",
        "safety_level": "destructive",
        "requires_confirmation": true,
        "warning": "This will remove ALL installed packages"
      }
    },
    {
      "id": "config_generation",
      "prompt": "create a web server configuration",
      "expected": {
        "intent": "config_generation",
        "config_type": "nginx",
        "requires_review": true
      }
    },
    {
      "id": "error_diagnosis",
      "prompt": "why did my build fail",
      "expected": {
        "intent": "error_diagnosis",
        "requires_context": ["build_log", "flake.nix"],
        "diagnosis_mode": true
      }
    }
  ]
}
```

### Harness Implementation

```rust
// tests/integration/harness.rs
use symthaea::SymthaeaHLB;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct ScenarioSuite {
    scenarios: Vec<Scenario>,
}

#[derive(Deserialize)]
struct Scenario {
    id: String,
    prompt: String,
    expected: ExpectedOutput,
}

#[derive(Deserialize, Serialize, PartialEq)]
struct ExpectedOutput {
    intent: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    command: Option<String>,
    safety_level: String,
    requires_confirmation: bool,
}

#[tokio::test]
async fn test_nix_scenarios() -> Result<()> {
    let suite: ScenarioSuite = serde_json::from_str(
        include_str!("scenarios/prompts.json")
    )?;

    let mut symthaea = SymthaeaHLB::new(16384, 1024)?;
    let mut failures = Vec::new();

    for scenario in suite.scenarios {
        let response = symthaea.process(&scenario.prompt).await?;

        // Compare structured output
        let actual = ExpectedOutput {
            intent: response.intent.clone(),
            command: response.command.clone(),
            safety_level: response.safety_level.clone(),
            requires_confirmation: response.requires_confirmation,
        };

        if actual != scenario.expected {
            failures.push((scenario.id, scenario.expected, actual));
        }
    }

    if !failures.is_empty() {
        eprintln!("Failures:");
        for (id, expected, actual) in failures {
            eprintln!("  {}: expected {:?}, got {:?}", id, expected, actual);
        }
        panic!("Scenario tests failed");
    }

    Ok(())
}
```

### Property-Based Tests

```rust
// tests/property/consciousness_invariants.rs
use proptest::prelude::*;

proptest! {
    #[test]
    fn phi_is_bounded(
        semantic in any::<Vec<f64>>(),
        dynamic in any::<Vec<f64>>()
    ) {
        let phi = calculate_phi(&semantic, &dynamic);
        prop_assert!(phi >= 0.0 && phi <= 1.0);
    }

    #[test]
    fn workspace_capacity_stable(
        inputs in prop::collection::vec(any::<String>(), 1..100)
    ) {
        let mut symthaea = SymthaeaHLB::new(16384, 1024)?;

        for input in inputs {
            symthaea.process(&input).await?;
            let workspace_size = symthaea.workspace_size();
            prop_assert!(workspace_size <= MAX_WORKSPACE_SIZE);
        }
    }
}
```

### Regression Baselines

```json
{
  "performance": {
    "phi_computation_ms": 0.011,
    "router_selection_ms": 0.005,
    "full_query_ms": 21.0,
    "memory_mb": 10.5
  },
  "accuracy": {
    "intent_recognition": 0.985,
    "command_correctness": 0.94,
    "safety_classification": 0.99
  }
}
```

---

## Track 3: NixOS System Knowledge + Security

### Knowledge Provider Architecture

```
src/
├── nix_knowledge/
│   ├── mod.rs
│   ├── flake_provider.rs      # Read flake configuration
│   ├── config_eval.rs         # Evaluate NixOS config to JSON
│   ├── package_index.rs       # Searchable package index
│   └── options_index.rs       # NixOS options reference
└── security/
    ├── mod.rs
    ├── redaction.rs           # Secret redaction
    ├── secret_inventory.rs    # Track secrets without values
    ├── policy_gate.rs         # Authorization for operations
    └── audit_log.rs           # Immutable action log
```

### Nix Knowledge Provider

```rust
// src/nix_knowledge/flake_provider.rs
pub struct FlakeKnowledgeProvider {
    cache: LruCache<FlakeLockHash, FlakeConfig>,
}

impl FlakeKnowledgeProvider {
    /// Detect current flake and load configuration
    pub async fn current_flake(&mut self) -> Result<FlakeConfig> {
        let flake_dir = std::env::current_dir()?;
        let lock_file = flake_dir.join("flake.lock");

        if !lock_file.exists() {
            return Err(Error::NoFlake);
        }

        let lock_hash = self.hash_lock_file(&lock_file)?;

        // Check cache
        if let Some(config) = self.cache.get(&lock_hash) {
            return Ok(config.clone());
        }

        // Evaluate flake to JSON
        let config = self.evaluate_flake(&flake_dir).await?;
        self.cache.put(lock_hash, config.clone());

        Ok(config)
    }

    /// Evaluate flake configuration to structured JSON
    async fn evaluate_flake(&self, dir: &Path) -> Result<FlakeConfig> {
        let output = Command::new("nix")
            .args(&["eval", ".#nixosConfigurations", "--json"])
            .current_dir(dir)
            .output()
            .await?;

        Ok(serde_json::from_slice(&output.stdout)?)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct FlakeConfig {
    pub hosts: Vec<String>,
    pub options: HashMap<String, OptionDefinition>,
    pub packages: Vec<PackageInfo>,
}
```

### Security Kernel

```rust
// src/security/mod.rs
pub struct SecurityKernel {
    redactor: SecretRedactor,
    inventory: SecretInventory,
    policy: PolicyGate,
    audit: AuditLog,
}

impl SecurityKernel {
    /// Redact secrets from text (never log actual values)
    pub fn redact(&self, text: &str) -> String {
        self.redactor.redact(text)
    }

    /// Check if operation is allowed
    pub fn authorize(&mut self, operation: &Operation) -> Result<Authorization> {
        let decision = self.policy.evaluate(operation)?;

        // Log decision (but not secret content)
        self.audit.log_authorization(operation.name(), decision);

        if decision.requires_confirmation() {
            return Ok(Authorization::RequiresConfirmation(decision.reason()));
        }

        if decision.denied() {
            return Ok(Authorization::Denied(decision.reason()));
        }

        Ok(Authorization::Allowed)
    }

    /// Record secret exists (without its value)
    pub fn register_secret(&mut self, name: &str, secret_type: SecretType) {
        self.inventory.register(name, secret_type);
    }
}

// src/security/redaction.rs
pub struct SecretRedactor {
    patterns: Vec<RedactionPattern>,
}

impl SecretRedactor {
    pub fn redact(&self, text: &str) -> String {
        let mut result = text.to_string();

        for pattern in &self.patterns {
            result = pattern.apply(&result);
        }

        result
    }
}

enum RedactionPattern {
    /// AWS access keys
    AwsAccessKey,
    /// Private keys
    PrivateKey,
    /// Passwords
    Password,
    /// API tokens
    ApiToken,
}

impl RedactionPattern {
    fn apply(&self, text: &str) -> String {
        match self {
            Self::AwsAccessKey => {
                // AKIA[0-9A-Z]{16}
                let re = Regex::new(r"AKIA[0-9A-Z]{16}").unwrap();
                re.replace_all(text, "[REDACTED_AWS_KEY]").to_string()
            }
            Self::PrivateKey => {
                let re = Regex::new(r"-----BEGIN.*PRIVATE KEY-----.*-----END.*PRIVATE KEY-----").unwrap();
                re.replace_all(text, "[REDACTED_PRIVATE_KEY]").to_string()
            }
            // ... other patterns
        }
    }
}

// src/security/audit_log.rs
pub struct AuditLog {
    entries: Vec<AuditEntry>,
    log_file: File,
}

#[derive(Serialize)]
pub struct AuditEntry {
    timestamp: DateTime<Utc>,
    operation: String,
    decision: String,
    reason: Option<String>,
    user_confirmed: bool,
}

impl AuditLog {
    pub fn log_authorization(&mut self, operation: &str, decision: AuthDecision) {
        let entry = AuditEntry {
            timestamp: Utc::now(),
            operation: operation.to_string(),
            decision: decision.name(),
            reason: decision.reason().map(String::from),
            user_confirmed: false,
        };

        // Write to file (append-only)
        writeln!(self.log_file, "{}", serde_json::to_string(&entry).unwrap());
        self.entries.push(entry);
    }
}
```

---

## Track 4: Executable Documentation

### Structure

```
docs/
└── examples/
    ├── 01-quickstart.md         # 5-minute start
    ├── 02-nix-assistant.md      # Flagship example
    ├── 03-web-research.md       # Epistemic verification
    ├── 04-consciousness-trace.md # Inspector usage
    └── assets/
        ├── quickstart.sh
        ├── nix-assistant-demo.rs
        ├── web-research-demo.rs
        └── trace-demo.rs
```

### Quickstart (Executable)

```markdown
# Symthaea HLB - 5-Minute Quickstart

## Prerequisites
- Rust 1.70+
- NixOS or Linux with Nix

## Install

```bash
# Clone repository
git clone https://github.com/luminous-dynamics/symthaea-hlb
cd symthaea-hlb

# Run quickstart
./quickstart.sh
```

That's it! The script will:
1. Build Symthaea in release mode
2. Run basic tests
3. Execute a demo query
4. Show you the trace

## Your First Query

```rust
use symthaea::SymthaeaHLB;

#[tokio::main]
async fn main() -> Result<()> {
    let mut symthaea = SymthaeaHLB::new(16384, 1024)?;

    let response = symthaea.process("install nginx").await?;

    println!("Response: {}", response.content);
    println!("Confidence: {:.1}%", response.confidence * 100.0);
    println!("Φ: {:.3}", response.phi);

    Ok(())
}
```

Run it:
```bash
cargo run --example quickstart
```

## Next Steps
- [Nix Assistant Tutorial](02-nix-assistant.md)
- [Inspector Guide](04-consciousness-trace.md)
- [Architecture Overview](../architecture/README.md)
```

### quickstart.sh (Actual Implementation)

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "🧠 Symthaea HLB - Quickstart"
echo ""

# Check prerequisites
echo "Checking prerequisites..."
command -v cargo >/dev/null 2>&1 || { echo "Error: Rust not found. Install from https://rustup.rs"; exit 1; }
echo "✓ Rust found"

# Build release
echo ""
echo "Building Symthaea (release mode)..."
cargo build --release

echo "✓ Build complete"

# Run tests
echo ""
echo "Running tests..."
cargo test --quiet -- --test-threads=1

echo "✓ Tests passed"

# Run demo
echo ""
echo "Running demo query: 'install nginx'"
cargo run --release --example quickstart

echo ""
echo "🎉 Success! Symthaea is ready."
echo ""
echo "Next steps:"
echo "  - Try the Nix assistant: cargo run --example nix_assistant"
echo "  - Inspect a trace: symthaea-inspect replay trace.json"
echo "  - Read the docs: docs/examples/02-nix-assistant.md"
```

---

## Role Assignment (If Multiple Developers)

### Dev A: Core (Main Thread)
- Routers/GWT/meta-router stability
- Refactor planning
- API surface design

### Dev B: Observability
- Inspector CLI
- Trace format
- Telemetry infrastructure

### Dev C: Nix Integration
- CLI wiring
- System knowledge sync
- Error diagnosis frames

### Dev D: Security
- SecurityKernel
- Redaction patterns
- Audit logging

### Dev E: Docs/Examples
- Quickstart script
- Runnable examples
- Contributor guide

---

## Success Metrics

### Sprint 1
- [ ] Inspector captures complete trace
- [ ] 50 Nix scenarios run against golden outputs
- [ ] Fresh install completes quickstart in <5 minutes
- [ ] Zero test failures in CI

### Sprint 2
- [ ] Inspector shows router selection with confidence
- [ ] GWT ignition events visible
- [ ] Φ timeline exportable
- [ ] Nix flake configuration queryable
- [ ] Dangerous operations require confirmation
- [ ] All actions logged to audit trail

### Sprint 3
- [ ] Nix errors include helpful diagnosis
- [ ] All examples run on fresh install
- [ ] Python can instantiate and query Symthaea
- [ ] Documentation is executable code

---

## What NOT to Prioritize Yet

- ❌ Template marketplace
- ❌ Massive doc site
- ❌ Big CLI exposing everything
- ❌ Major refactors
- ❌ Dozens of visualizers (do one well first)

---

## Bottom Line

**Single best parallel investment**: **Inspector + Scenario Harness**

This transforms Symthaea from "complex code" into an **engineered system you can trust, debug, and demonstrate**.

Every other track accelerates once you can:
1. **See what's happening** (Inspector)
2. **Prove it works** (Scenarios)
3. **Ship with confidence** (Audit trail)

---

*Let's build systems we can trust.* 🧠✨
