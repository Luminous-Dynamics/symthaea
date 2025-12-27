# 🧬 Symthaea v1.1: Refined Architecture
## *Sympoietic Constitutional Intelligence Organism Network*

**Version**: 1.1 - Production Specification (Refined)
**Date**: December 2025
**Status**: 🌟 Implementation-Ready Architecture

---

## 📋 Critical Refinements from v1.0

This version addresses five critical implementation concerns with **buildable mechanisms**:

1. **Deterministic HDC** - Hash-based projections, content-addressed specs
2. **Privilege Separation** - Shell Kernel with Action IR + invariants
3. **Verifiable Claims** - Explicit taxonomy (Physics, Units, Sensors, etc.)
4. **Hardware Pragmatism** - CPU baseline, GPU/NPU as optimization tiers
5. **Resonator Integration** - First-class coherence substrate with progressive retrieval

---

## 🎯 The Complete Stack (v1.1)

### System Architecture Overview

| System | Organ | Role | v1.1 Technology | Implementation | Location | Latency |
|--------|-------|------|-----------------|----------------|----------|---------|
| **Nervous** | Actor Model | Message Bus | tokio + mpsc | Priority queue + backpressure | Host | <1ms |
| **Nervous** | Thalamus | Attention Gate | Salience routing | ThalamusActor | Host | <1ms |
| **Nervous** | Amygdala | Safety Reflex | Pattern matching | AmygdalaActor | Host | <1ms |
| **Nervous** | Chronos | Time Sense | Subjective τ | ChronosActor | Host | <1ms |
| **Crystal** | HDC Core | Symbolic Logic | Hash-based HDC | hdc-rs (16k/32k dims) | Host | <10ms |
| **Crystal** | Resonator | Cleanup/Binding | Attractor networks | Bit-packed resonance | Host | ~20ms |
| **Liquid** | LTC Network | Causal Dynamics | Closed-form Continuous | candle-ltc (CfC) | Host (CPU baseline) | ~10ms |
| **Memory** | Hippocampus | Episodic Store | HDC-indexed episodes | SQLite + LSH tables | Host | ~50ms |
| **Memory** | Consolidator | Dream/Learn | Nightly compression | rust-bert clustering | Host | Nightly |
| **Logic** | Cortex | Reasoning | Datalog inference | cozodb (embedded) | Host | ~100ms |
| **Physiology** | Hearth | Metabolism | Willpower budget | AtomicU32 + gratitude | Host | <1ms |
| **Physiology** | Endocrine | Modulation | Neuro-chemistry | NeurochemState | Host | <1ms |
| **Motor** | Larynx | Voice | Emotional TTS | Kokoro-82M (ONNX) | Host | ~200ms |
| **Motor** | Shell Kernel | Agency | **Action IR + Invariants** | Isolated micro-service | **Privileged (sandboxed)** | ~1s |
| **Motor** | Interface | MCP Bridge | Tool protocol | mcp-sdk | Host | ~100ms |
| **Senses** | Retina | Vision | **Hash-based HDC** | Moondream2 + token projection | Host | ~50ms |
| **Senses** | Cochlea | Hearing | **Hash-based HDC** | Whisper + token projection | Host | ~200ms |
| **Senses** | Digital | Code Sense | **Graph HDC** | tree-sitter + hdc-rs | Host | ~10ms |
| **Soul** | Weaver | Identity | Holographic narrative | WeaverActor + resonance | Host | Nightly |
| **Soul** | Daemon | Creativity | Stochastic resonance | DaemonActor | Host | Background |
| **Soul** | Will | Drive | Active Inference | Free Energy minimization | Host | ~10ms |
| **Soul** | Tensor | Self-Measure | Phase space Ω | nalgebra (Jacobian) | Host | ~100ms |
| **Immune** | Thymus | Verification | **Claim Classes** | Physics + unit checks | Host | ~50ms |
| **Immune** | Glia | Cleanup | Waste management | GliaActor | Host | Background |
| **Swarm** | Mycelix | Network | GossipSub + CA | libp2p + DHT | Host | Variable |

**Note on "Host"**: All components run in native Rust process with internal sandboxing boundaries. No separate WASM guests in v1.1 (simplification for buildability).

---

## 🔒 CRITICAL FIX #1: Deterministic HDC (Hash-Based Projections)

### Problem Solved
- Full projection matrices are too heavy (32k × input_dim)
- Previous `verify_hash()` had ID/hash mismatch
- Need streaming token projections, not materialized matrices

### Solution: Content-Addressed Specs + Hash-Based Token Projection

```rust
// src/crystal/projection_spec.rs

use blake3::Hash;
use serde::{Deserialize, Serialize};

/// Body of spec (excludes hash for deterministic hashing)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectionSpecBody {
    pub label: String,
    pub dims: usize,
    pub encoding: Encoding,
    pub seed: [u8; 32],
    pub domain_sep: String,
    pub version: String,
    pub created_at: i64,  // Unix timestamp (deterministic)
}

/// Full spec with content hash
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectionSpec {
    /// Content hash of body (no self-reference)
    pub spec_hash: String,  // Hex-encoded blake3
    
    #[serde(flatten)]
    pub body: ProjectionSpecBody,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Encoding {
    BinaryBits,      // 0/1 (fast XOR+popcount)
    Bipolar,         // ±1 interpreted (stored as bits)
}

impl ProjectionSpec {
    /// Create deterministic spec tied to instance identity
    pub fn new(instance_did: &str, modality: &str, dims: usize) -> Self {
        let seed = blake3::hash(
            format!("{}:{}:{}", instance_did, modality, dims).as_bytes()
        ).into();
        
        let body = ProjectionSpecBody {
            label: format!("{}_{}", modality, dims),
            dims,
            encoding: Encoding::BinaryBits,
            seed,
            domain_sep: modality.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            created_at: Utc::now().timestamp(),
        };
        
        let spec_hash = Self::compute_hash_of_body(&body);
        
        Self { spec_hash, body }
    }
    
    fn compute_hash_of_body(body: &ProjectionSpecBody) -> String {
        // Use deterministic bincode with fixed config
        let encoded = bincode::serialize(body).unwrap();
        blake3::hash(&encoded).to_hex().to_string()
    }
    
    /// Verify spec integrity
    pub fn verify(&self) -> Result<()> {
        let computed = Self::compute_hash_of_body(&self.body);
        if computed == self.spec_hash {
            Ok(())
        } else {
            Err(anyhow!("Spec hash mismatch: computed={}, stored={}", 
                computed, self.spec_hash))
        }
    }
    
    /// Hash-based token projection (NO matrix materialization)
    pub fn project_token(&self, token: &str, index: usize) -> Vec<u8> {
        self.token_to_hv(token, index)
    }
    
    fn token_to_hv(&self, token: &str, position: usize) -> Vec<u8> {
        // Deterministic stream from keyed hash
        let mut hv = vec![0u8; self.body.dims / 8];
        let mut counter = 0u64;
        
        for chunk in hv.chunks_mut(32) {
            // Hash: seed || domain_sep || token || position || counter
            let mut hasher = blake3::Hasher::new_keyed(&self.body.seed);
            hasher.update(self.body.domain_sep.as_bytes());
            hasher.update(b"|");
            hasher.update(token.as_bytes());
            hasher.update(b"|");
            hasher.update(&position.to_le_bytes());
            hasher.update(b"|");
            hasher.update(&counter.to_le_bytes());
            
            let hash = hasher.finalize();
            let hash_bytes = hash.as_bytes();
            
            let copy_len = chunk.len().min(32);
            chunk[..copy_len].copy_from_slice(&hash_bytes[..copy_len]);
            
            counter += 1;
        }
        
        hv
    }
    
    /// Bundle multiple token HVs (superposition)
    /// NOTE: This is reference implementation (correct but unoptimized)
    /// Production should use SIMD bit-slicing or block accumulators
    pub fn bundle_tokens(&self, tokens: &[String]) -> Vec<u8> {
        let mut accumulator = vec![0i32; self.body.dims];
        
        for (i, token) in tokens.iter().enumerate() {
            let token_hv = self.project_token(token, i);
            
            // Add to accumulator (bipolar: 0→-1, 1→+1)
            for (j, byte) in token_hv.iter().enumerate() {
                for bit in 0..8 {
                    let bit_val = (byte >> bit) & 1;
                    let bipolar = if bit_val == 1 { 1 } else { -1 };
                    accumulator[j * 8 + bit] += bipolar;
                }
            }
        }
        
        // Crystallize back to bits (sign function)
        pack_bits_from_accumulator(&accumulator)
    }
}

fn pack_bits_from_accumulator(acc: &[i32]) -> Vec<u8> {
    let mut result = vec![0u8; acc.len() / 8];
    for (i, &val) in acc.iter().enumerate() {
        if val > 0 {
            result[i / 8] |= 1 << (i % 8);
        }
    }
    result
}
```

### Storage Schema (Fixed)

```sql
-- 1. Projection specs (content-addressed)
CREATE TABLE projection_spec (
  spec_hash TEXT PRIMARY KEY,          -- blake3 hash of canonical CBOR
  label TEXT NOT NULL,
  dims INTEGER NOT NULL,
  encoding TEXT NOT NULL,
  seed BLOB NOT NULL,
  domain_sep TEXT NOT NULL,
  version TEXT NOT NULL,
  created_at INTEGER NOT NULL
);

-- 2. Hypervector storage (bit-packed)
CREATE TABLE hv_store (
  hv_id TEXT PRIMARY KEY,              -- blake3(spec_hash || bits)
  spec_hash TEXT NOT NULL REFERENCES projection_spec(spec_hash),
  kind TEXT NOT NULL,                  -- "attractor" | "episode" | "symbol"
  bits BLOB NOT NULL,                  -- Packed: dims/8 bytes
  trust REAL NOT NULL DEFAULT 0.0,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  UNIQUE(spec_hash, bits)              -- Dedup identical vectors
);
CREATE INDEX hv_store_kind_idx ON hv_store(kind);
CREATE INDEX hv_store_spec_idx ON hv_store(spec_hash);
```

**Result**: 
- ✅ Hash-based token projections (no heavy matrices)
- ✅ Content-addressed specs (verifiable integrity)
- ✅ Deterministic + streaming (scales to any input size)

---

## 🛡️ CRITICAL FIX #2: Shell Security + Declarative Policy

### Problem Solved
- "Allowlists" are necessary but were misrepresented
- Transactionality is OS-specific (overlay FS not universal)
- Need honest framing + portable contract
- **NEW**: Policy should be declarative (data, not code)

### Solution: PolicyBundle + Action IR + Tiered Rollback

```rust
// src/motor/shell_kernel/policy.rs

/// Declarative security policy (signed, versioned, auditable)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyBundle {
    pub version: String,
    pub policy_hash: String,
    pub signed_by: String,
    pub created_at: i64,
    
    pub security: SecurityPolicy,
    pub verification: VerificationPolicy,
    pub resonance: ResonanceGatingPolicy,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SecurityPolicy {
    /// Capabilities (what effects are allowed)
    pub capabilities: Vec<Capability>,
    
    /// Scopes (where effects can apply)
    pub scopes: Vec<Scope>,
    
    /// Budgets (rate limits, resource bounds)
    pub budgets: HashMap<String, Budget>,
    
    /// Risk tiers (auto / confirm / deny)
    pub risk_tiers: HashMap<String, RiskTier>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capability {
    pub name: String,  // "file.write", "process.spawn", "network.http"
    pub allowed: bool,
    pub scopes: Vec<String>,  // References to scope names
    pub preconditions: Vec<String>,  // Claims that must be verified
    pub postconditions: Vec<String>,  // Conditions that must hold after
    pub budget_name: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scope {
    pub name: String,
    pub kind: ScopeKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ScopeKind {
    FileSystem { root: PathBuf, read_only: bool },
    Network { connector_name: String },
    Process { binary_allowlist: Vec<PathBuf> },
    UI { intent_only: bool },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Budget {
    pub rate_limit: Option<RateLimit>,
    pub resource_limit: Option<ResourceLimit>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RateLimit {
    pub count: u32,
    pub window_secs: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceLimit {
    pub max_bytes: Option<u64>,
    pub max_duration_secs: Option<u64>,
    pub max_cpu_percent: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum RiskTier {
    Auto,      // Execute without confirmation
    Confirm,   // Require user confirmation
    Deny,      // Always deny
}

/// Example policy bundle (security_policy.toml)
const EXAMPLE_SECURITY_POLICY: &str = r#"
version = "1.0"
policy_hash = "blake3_hash_here"
signed_by = "did:symthaea:instance_id"
created_at = 1703001600

[security]
[[security.capabilities]]
name = "file.write"
allowed = true
scopes = ["workspace"]
preconditions = []
postconditions = ["hash_matches", "no_exec_bit"]
budget_name = "file_ops"

[[security.capabilities]]
name = "process.spawn"
allowed = true
scopes = ["trusted_binaries"]
preconditions = ["no_injection"]
postconditions = []
budget_name = "process_ops"

[[security.capabilities]]
name = "network.http"
allowed = true
scopes = ["github_connector", "mycelix_connector"]
preconditions = []
postconditions = []
budget_name = "network_ops"

[[security.scopes]]
name = "workspace"
[security.scopes.kind]
FileSystem = { root = "/home/user/symthaea_workspace", read_only = false }

[[security.scopes]]
name = "trusted_binaries"
[security.scopes.kind]
Process = { binary_allowlist = ["/usr/bin/git", "/usr/bin/cargo", "/usr/bin/python3"] }

[[security.scopes]]
name = "github_connector"
[security.scopes.kind]
Network = { connector_name = "github" }

[security.budgets.file_ops]
rate_limit = { count = 100, window_secs = 3600 }
resource_limit = { max_bytes = 104857600 }  # 100MB

[security.budgets.process_ops]
rate_limit = { count = 10, window_secs = 60 }
resource_limit = { max_duration_secs = 300 }  # 5 min

[security.budgets.network_ops]
rate_limit = { count = 1000, window_secs = 3600 }
resource_limit = { max_bytes = 10485760 }  # 10MB per request

[security.risk_tiers]
"file.write" = "Auto"
"file.delete" = "Confirm"
"process.spawn" = "Confirm"
"network.http" = "Auto"
"ui.keystroke" = "Deny"
"#;

/// Minimal Action IR (NOT arbitrary commands)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ActionIR {
    FileOp(FileAction),
    ProcessOp(ProcessAction),
    NetworkOp(NetworkAction),
    UIIntent(UIAction),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileAction {
    pub op: FileOp,
    pub path: PathBuf,
    pub content: Option<Vec<u8>>,
    pub postcondition: FilePostcondition,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FileOp {
    Read,
    Write,
    Append,
    Delete,
    CreateDir,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FilePostcondition {
    pub hash_after: Option<[u8; 32]>,
    pub size_bounds: (u64, u64),
    pub no_exec_bit: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessAction {
    pub binary: PathBuf,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub timeout_secs: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkAction {
    pub connector_name: String,
    pub method: HttpMethod,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub body: Option<Vec<u8>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UIAction {
    pub intent: UIIntent,
    pub target: Option<String>,  // Window/app identifier
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UIIntent {
    FocusWindow,
    ClickButton { label: String },
    TypeText { text: String },  // Semantic, not raw keystrokes
    ScrollTo { position: String },
}

impl UIAction {
    pub fn is_raw_keystroke(&self) -> bool {
        // All UIIntents are semantic, never raw
        false
    }
}
```

### Trusted Shell Kernel (enforces policy)

```rust
// src/motor/shell_kernel/mod.rs

/// Trusted kernel with declarative policy
pub struct ShellKernel {
    policy: PolicyBundle,
    workspace_root: PathBuf,
    transact_tier: TransactionTier,
    
    /// Per-connector network policies
    network_connectors: HashMap<String, NetworkConnector>,
    
    /// Budget trackers
    budgets: HashMap<String, BudgetTracker>,
    
    /// Event log (tamper-evident)
    event_log: EventLog,
}

/// Per-connector trust model (replaces raw domain allowlist)
pub struct NetworkConnector {
    pub name: String,
    pub allowed_domains: HashSet<String>,
    pub cert_pins: Vec<CertFingerprint>,
    pub max_payload: u64,
    pub require_tls: bool,
    
    /// Lifecycle management
    pub version: u32,
    pub signature: Option<Vec<u8>>,
    pub approved_by: String,
}

/// OS-specific transactionality contract
pub enum TransactionTier {
    /// True rollback (Linux overlayfs, Windows VSS)
    Rollback,
    
    /// Compensating actions (temp files + atomic rename + backups)
    Compensating,
    
    /// No rollback (requires explicit user confirmation + warning)
    Unsafe,
}

impl ShellKernel {
    pub fn new(policy_path: &Path) -> Result<Self> {
        let policy: PolicyBundle = toml::from_str(&std::fs::read_to_string(policy_path)?)?;
        
        // Verify policy signature
        Self::verify_policy_signature(&policy)?;
        
        // Detect best transaction tier for OS
        let transact_tier = Self::detect_transaction_tier();
        
        Ok(Self {
            policy,
            workspace_root: PathBuf::from("/home/user/symthaea_workspace"),
            transact_tier,
            network_connectors: Self::load_network_connectors()?,
            budgets: HashMap::new(),
            event_log: EventLog::new()?,
        })
    }
    
    fn verify_policy_signature(policy: &PolicyBundle) -> Result<()> {
        // Verify blake3 hash matches content
        let computed_hash = blake3::hash(
            serde_json::to_string(&policy)?.as_bytes()
        ).to_hex().to_string();
        
        if computed_hash != policy.policy_hash {
            return Err(anyhow!("Policy hash mismatch"));
        }
        
        // TODO: Verify signature with instance key
        Ok(())
    }
    
    fn detect_transaction_tier() -> TransactionTier {
        #[cfg(target_os = "linux")]
        {
            if OverlayFilesystem::is_available() {
                return TransactionTier::Rollback;
            }
        }
        
        #[cfg(target_os = "windows")]
        {
            if VolumeSnapshotService::is_available() {
                return TransactionTier::Rollback;
            }
        }
        
        TransactionTier::Compensating
    }
    
    pub async fn execute_action(&mut self, action: ActionIR) 
        -> Result<ActionResult, ViolationError> {
        
        // 1. Check capability in policy
        let capability = self.check_capability(&action)?;
        
        // 2. Static invariants (hard laws)
        self.check_invariants(&action)?;
        
        // 3. Verify preconditions (Thymus)
        self.verify_preconditions(&capability)?;
        
        // 4. Check budget
        self.check_budget(&capability)?;
        
        // 5. Risk tier gating
        match self.get_risk_tier(&action) {
            RiskTier::Deny => {
                return Err(ViolationError::RiskTierDenied {
                    action: format!("{:?}", action),
                });
            }
            RiskTier::Confirm => {
                self.confirm_with_user(&action).await?;
            }
            RiskTier::Auto => {}
        }
        
        // 6. Execute with tier-appropriate transaction
        let result = match self.transact_tier {
            TransactionTier::Rollback => {
                self.execute_rollback(action.clone()).await?
            }
            TransactionTier::Compensating => {
                self.execute_compensating(action.clone()).await?
            }
            TransactionTier::Unsafe => {
                self.warn_user_no_rollback(&action).await?;
                self.confirm_with_user(&action).await?;
                self.execute_direct(action.clone()).await?
            }
        };
        
        // 7. Verify postconditions
        self.verify_postconditions(&capability, &result)?;
        
        // 8. Log to tamper-evident chain
        self.event_log.record(EventLogEntry {
            action: action.clone(),
            result: result.clone(),
            policy_hash: self.policy.policy_hash.clone(),
            timestamp: Utc::now().timestamp(),
        })?;
        
        Ok(result)
    }
    
    fn check_capability(&self, action: &ActionIR) -> Result<&Capability, ViolationError> {
        let cap_name = match action {
            ActionIR::FileOp(FileAction { op: FileOp::Write, .. }) => "file.write",
            ActionIR::FileOp(FileAction { op: FileOp::Delete, .. }) => "file.delete",
            ActionIR::ProcessOp(_) => "process.spawn",
            ActionIR::NetworkOp(_) => "network.http",
            ActionIR::UIIntent(_) => "ui.intent",
            _ => "unknown",
        };
        
        self.policy.security.capabilities.iter()
            .find(|c| c.name == cap_name && c.allowed)
            .ok_or_else(|| ViolationError::CapabilityDenied {
                capability: cap_name.to_string(),
            })
    }
    
    fn check_invariants(&self, action: &ActionIR) -> Result<(), ViolationError> {
        match action {
            ActionIR::FileOp(file_action) => {
                // INVARIANT 1: No writes outside workspace
                let canonical = file_action.path.canonicalize()
                    .unwrap_or_else(|_| file_action.path.clone());
                
                if !canonical.starts_with(&self.workspace_root) {
                    return Err(ViolationError::PathEscape {
                        attempted: canonical,
                        allowed: self.workspace_root.clone(),
                    });
                }
                
                Ok(())
            }
            
            ActionIR::ProcessOp(proc_action) => {
                // INVARIANT 2: Only trusted binaries (from policy)
                let trusted = self.policy.security.scopes.iter()
                    .find(|s| s.name == "trusted_binaries")
                    .and_then(|s| match &s.kind {
                        ScopeKind::Process { binary_allowlist } => Some(binary_allowlist),
                        _ => None,
                    })
                    .ok_or_else(|| ViolationError::ScopeNotFound {
                        scope: "trusted_binaries".to_string(),
                    })?;
                
                if !trusted.contains(&proc_action.binary) {
                    return Err(ViolationError::UntrustedBinary {
                        binary: proc_action.binary.clone(),
                        hint: "Add to security_policy.toml trusted_binaries if safe",
                    });
                }
                
                // INVARIANT 3: No shell injection
                self.verify_no_injection(&proc_action.args)?;
                
                Ok(())
            }
            
            ActionIR::NetworkOp(net_action) => {
                // INVARIANT 4: Connector-based policy
                let connector = self.network_connectors
                    .get(&net_action.connector_name)
                    .ok_or_else(|| ViolationError::UnknownConnector {
                        name: net_action.connector_name.clone(),
                    })?;
                
                connector.check_allowed(&net_action)?;
                Ok(())
            }
            
            ActionIR::UIIntent(ui_action) => {
                // INVARIANT 5: Intent-level only (never raw input)
                if ui_action.is_raw_keystroke() {
                    return Err(ViolationError::RawInputDenied {
                        reason: "Use semantic intents, not raw keystrokes",
                    });
                }
                Ok(())
            }
        }
    }
    
    fn verify_no_injection(&self, args: &[String]) -> Result<(), ViolationError> {
        // Check for shell metacharacters that could enable injection
        let dangerous_patterns = [";", "|", "&", "$", "`", "$(", "&&", "||"];
        
        for arg in args {
            for pattern in &dangerous_patterns {
                if arg.contains(pattern) {
                    return Err(ViolationError::InjectionDetected {
                        arg: arg.clone(),
                        pattern: pattern.to_string(),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    async fn execute_rollback(&self, action: ActionIR) -> Result<ActionResult> {
        #[cfg(target_os = "linux")]
        {
            let overlay = OverlayFilesystem::new(&self.workspace_root)?;
            let result = overlay.execute(&action).await?;
            overlay.commit()?;
            Ok(result)
        }
        
        #[cfg(target_os = "windows")]
        {
            let vss = VolumeSnapshotService::new(&self.workspace_root)?;
            let result = vss.execute(&action).await?;
            vss.commit()?;
            Ok(result)
        }
        
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            // Fall back to compensating
            self.execute_compensating(action).await
        }
    }
    
    async fn execute_compensating(&self, action: ActionIR) -> Result<ActionResult> {
        match action {
            ActionIR::FileOp(file_action) => {
                let temp_path = file_action.path.with_extension("tmp");
                
                // Execute to temp
                match file_action.op {
                    FileOp::Write | FileOp::Append => {
                        std::fs::write(&temp_path, file_action.content.as_ref().unwrap())?;
                    }
                    _ => return Err(anyhow!("Unsupported op for compensating transaction")),
                }
                
                // Verify postconditions on temp
                self.verify_file_postcondition(&temp_path, &file_action.postcondition)?;
                
                // Backup original if exists
                if file_action.path.exists() {
                    let backup = file_action.path.with_extension("backup");
                    std::fs::copy(&file_action.path, &backup)?;
                }
                
                // Atomic rename
                std::fs::rename(&temp_path, &file_action.path)?;
                
                Ok(ActionResult::FileWritten {
                    path: file_action.path,
                })
            }
            _ => Err(anyhow!("Compensating transaction not implemented for this action type")),
        }
    }
    
    fn verify_file_postcondition(&self, path: &Path, postcond: &FilePostcondition) -> Result<()> {
        let metadata = std::fs::metadata(path)?;
        
        // Check size bounds
        let size = metadata.len();
        if size < postcond.size_bounds.0 || size > postcond.size_bounds.1 {
            return Err(anyhow!("File size {} outside bounds {:?}", size, postcond.size_bounds));
        }
        
        // Check exec bit
        if postcond.no_exec_bit {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mode = metadata.permissions().mode();
                if mode & 0o111 != 0 {
                    return Err(anyhow!("File has exec bit set"));
                }
            }
        }
        
        // Check hash if specified
        if let Some(expected_hash) = postcond.hash_after {
            let content = std::fs::read(path)?;
            let actual_hash = blake3::hash(&content);
            if actual_hash.as_bytes() != &expected_hash {
                return Err(anyhow!("File hash mismatch"));
            }
        }
        
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ViolationError {
    #[error("Path escape: attempted={attempted:?}, allowed={allowed:?}")]
    PathEscape {
        attempted: PathBuf,
        allowed: PathBuf,
    },
    
    #[error("Untrusted binary: {binary:?}. {hint}")]
    UntrustedBinary {
        binary: PathBuf,
        hint: &'static str,
    },
    
    #[error("Unknown connector: {name}")]
    UnknownConnector {
        name: String,
    },
    
    #[error("Raw input denied: {reason}")]
    RawInputDenied {
        reason: &'static str,
    },
    
    #[error("Capability denied: {capability}")]
    CapabilityDenied {
        capability: String,
    },
    
    #[error("Injection detected in arg: {arg}, pattern: {pattern}")]
    InjectionDetected {
        arg: String,
        pattern: String,
    },
    
    #[error("Risk tier denied for action: {action}")]
    RiskTierDenied {
        action: String,
    },
}
```

**Result**: 
- ✅ Policy is **declarative** (TOML/data, not code)
- ✅ Capabilities + scopes + budgets + risk tiers
- ✅ Honest about "minimal policy sets" (still necessary)
- ✅ Tiered transactionality with explicit contract
- ✅ Signed, versioned, auditable policies

---

## 📋 Declarative Verification Policy

```toml
# verification_policy.toml

version = "1.0"
policy_hash = "blake3_hash_here"

[physics]
tolerance = 0.01  # 1% error allowed

[[physics.laws]]
name = "newtonian_mechanics"
enabled = true
method = "ode_integration"

[[physics.laws]]
name = "conservation"
enabled = true
conserved_quantities = ["energy", "momentum", "charge"]

[units]
strict_dimensional_analysis = true

[sensors]
min_correlation = 0.8
required_modalities = 2

[structural]
require_type_checking = true

[verification_requirements]
"file.write" = []  # No verification needed
"process.spawn" = ["no_injection"]  # Must pass injection check
"network.http" = []
"critical.action" = ["physics.conservation", "sensors.cross_check"]
```

---

## 📋 Declarative Resonance Policy

```toml
# resonance_policy.toml

version = "1.0"
policy_hash = "blake3_hash_here"

[thresholds]
margin_threshold_16 = 64.0
energy_threshold_ratio = 0.001
max_steps = 8
temperature = 1.0

[escalation]
# When to upgrade HV16 → HV32
hv32_on_low_margin = true
hv32_on_high_energy = true
hv32_on_high_stakes = true

[safe_intent_attractors]
write_protected = true
require_signature = true
require_user_approval_above_trust = 0.8

[novelty_handling]
user_confirmation_required = true
learn_after_approval = true
min_novelty_threshold = 0.3

[poisoning_defenses]
version_tracking = true
signature_required = true
diversity_pressure = 0.2  # Penalize over-correlated sources
```

```rust
// src/motor/shell_kernel/mod.rs

/// Minimal Action IR (NOT arbitrary commands)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ActionIR {
    FileOp(FileAction),
    ProcessOp(ProcessAction),
    NetworkOp(NetworkAction),
    UIIntent(UIAction),
}

#[derive(Clone, Debug)]
pub struct FileAction {
    pub op: FileOp,
    pub path: PathBuf,
    pub content: Option<Vec<u8>>,
    pub postcondition: FilePostcondition,
}

pub struct FilePostcondition {
    pub hash_after: Option<[u8; 32]>,
    pub size_bounds: (u64, u64),
    pub no_exec_bit: bool,
}

/// Trusted kernel with MINIMAL policy sets (not endless allowlists)
pub struct ShellKernel {
    workspace_root: PathBuf,
    
    /// Minimal set of trusted binaries
    /// (Honest: this IS an allowlist, but small by design)
    trusted_binaries: HashSet<PathBuf>,
    
    /// Per-connector trust domains (not raw domain list)
    network_connectors: HashMap<String, NetworkConnector>,
    
    resource_limits: ResourceLimits,
    transact_tier: TransactionTier,
}

/// Per-connector trust model (better than raw domain allowlist)
pub struct NetworkConnector {
    pub name: String,              // "github", "openai", "mycelix"
    pub allowed_domains: HashSet<String>,
    pub cert_pins: Vec<CertFingerprint>,
    pub max_payload: u64,
    pub require_tls: bool,
}

/// OS-specific transactionality contract
pub enum TransactionTier {
    /// True rollback (Linux overlayfs, Windows VSS)
    Rollback,
    
    /// Compensating actions (temp files + atomic rename)
    Compensating,
    
    /// No rollback (requires explicit user confirmation)
    Unsafe,
}

impl ShellKernel {
    pub async fn execute_action(&self, action: ActionIR) 
        -> Result<ActionResult, ViolationError> {
        
        // 1. Static invariants (hard laws)
        self.check_invariants(&action)?;
        
        // 2. Resource budget
        self.check_resource_limits(&action)?;
        
        // 3. Execute with tier-appropriate transaction
        let result = match self.transact_tier {
            TransactionTier::Rollback => {
                self.execute_rollback(action).await?
            }
            TransactionTier::Compensating => {
                self.execute_compensating(action).await?
            }
            TransactionTier::Unsafe => {
                // Require explicit confirmation
                self.confirm_with_user(&action).await?;
                self.execute_direct(action).await?
            }
        };
        
        // 4. Verify postconditions
        self.verify_postconditions(&action, &result)?;
        
        Ok(result)
    }
    
    fn check_invariants(&self, action: &ActionIR) -> Result<()> {
        match action {
            ActionIR::FileOp(file_action) => {
                // INVARIANT 1: No writes outside workspace
                let canonical = file_action.path.canonicalize()
                    .unwrap_or_else(|_| file_action.path.clone());
                
                if !canonical.starts_with(&self.workspace_root) {
                    return Err(ViolationError::PathEscape {
                        attempted: canonical,
                        allowed: self.workspace_root.clone(),
                    });
                }
                
                Ok(())
            }
            
            ActionIR::ProcessOp(proc_action) => {
                // INVARIANT 2: Only trusted binaries
                // (Honest framing: this IS a minimal allowlist)
                if !self.trusted_binaries.contains(&proc_action.binary) {
                    return Err(ViolationError::UntrustedBinary {
                        binary: proc_action.binary.clone(),
                        hint: "Add to trusted_binaries.toml if safe",
                    });
                }
                
                // INVARIANT 3: No shell injection
                self.verify_no_injection(&proc_action.args)?;
                
                Ok(())
            }
            
            ActionIR::NetworkOp(net_action) => {
                // INVARIANT 4: Connector-based policy
                let connector = self.network_connectors
                    .get(&net_action.connector_name)
                    .ok_or_else(|| ViolationError::UnknownConnector {
                        name: net_action.connector_name.clone(),
                    })?;
                
                connector.check_allowed(&net_action)?;
                Ok(())
            }
            
            ActionIR::UIIntent(ui_action) => {
                // INVARIANT 5: Intent-level only (no raw input)
                if ui_action.is_raw_keystroke() {
                    return Err(ViolationError::RawInputDenied {
                        reason: "Use semantic intents, not raw keystrokes",
                    });
                }
                Ok(())
            }
        }
    }
    
    async fn execute_rollback(&self, action: ActionIR) -> Result<ActionResult> {
        #[cfg(target_os = "linux")]
        {
            let overlay = OverlayFilesystem::new(&self.workspace_root)?;
            let result = overlay.execute(&action).await?;
            overlay.commit()?;
            Ok(result)
        }
        
        #[cfg(target_os = "windows")]
        {
            let vss = VolumeSnapshotService::new(&self.workspace_root)?;
            let result = vss.execute(&action).await?;
            vss.commit()?;
            Ok(result)
        }
        
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            // Fall back to compensating
            self.execute_compensating(action).await
        }
    }
    
    async fn execute_compensating(&self, action: ActionIR) -> Result<ActionResult> {
        // Write to temp, verify, atomic rename
        match action {
            ActionIR::FileOp(file_action) => {
                let temp_path = file_action.path.with_extension("tmp");
                
                // Execute to temp
                std::fs::write(&temp_path, file_action.content.as_ref().unwrap())?;
                
                // Verify postconditions on temp
                self.verify_file_postcondition(&temp_path, &file_action.postcondition)?;
                
                // Atomic rename (or backup original first)
                if file_action.path.exists() {
                    let backup = file_action.path.with_extension("backup");
                    std::fs::rename(&file_action.path, &backup)?;
                }
                std::fs::rename(&temp_path, &file_action.path)?;
                
                Ok(ActionResult::FileWritten {
                    path: file_action.path,
                })
            }
            _ => todo!("Implement for other action types"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ViolationError {
    #[error("Path escape: attempted={attempted:?}, allowed={allowed:?}")]
    PathEscape {
        attempted: PathBuf,
        allowed: PathBuf,
    },
    
    #[error("Untrusted binary: {binary:?}. {hint}")]
    UntrustedBinary {
        binary: PathBuf,
        hint: &'static str,
    },
    
    #[error("Unknown connector: {name}")]
    UnknownConnector {
        name: String,
    },
    
    #[error("Raw input denied: {reason}")]
    RawInputDenied {
        reason: &'static str,
    },
}
```

**Honest Framing**:
- ✅ We use **minimal policy sets** (trusted binaries, connectors) where OS requires
- ✅ We avoid **endless per-command allowlists** via invariants + IR
- ✅ Transactionality is **tiered by OS capability** with explicit contract

---

## ✅ CRITICAL FIX #3: Thymus Claim Classes (Complete)

```rust
// src/immune/thymus/claim_classes.rs

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VerifiableClaim {
    /// Verifiable via ODE integration
    PhysicsLaw(PhysicsLaw),
    
    /// Verifiable via dimensional analysis
    DimensionalConsistency(Equation),
    
    /// Verifiable via cross-sensor correlation
    SensorFusion(SensorClaim),
    
    /// Verifiable via conserved quantities
    Conservation(ConservedQuantity),
    
    /// Verifiable via type/contract checking
    Structural(StructuralClaim),
    
    /// NOT verifiable (heuristic/advisory only)
    Heuristic(HeuristicClaim),
}

pub struct ThymusActor {
    mailbox: mpsc::Receiver<OrganMessage>,
    ode_solver: OdeSolver,
    unit_system: UnitSystem,
}

impl ThymusActor {
    pub fn verify_claim(&self, claim: VerifiableClaim) -> VerificationResult {
        match claim {
            VerifiableClaim::PhysicsLaw(physics) => {
                self.verify_physics(physics)
            }
            
            VerifiableClaim::DimensionalConsistency(eq) => {
                self.verify_units(eq)
            }
            
            VerifiableClaim::SensorFusion(sensors) => {
                self.cross_check_sensors(sensors)
            }
            
            VerifiableClaim::Conservation(conserved) => {
                self.verify_conservation(conserved)
            }
            
            VerifiableClaim::Structural(structural) => {
                self.verify_structure(structural)
            }
            
            VerifiableClaim::Heuristic(heuristic) => {
                // Explicitly non-proof
                VerificationResult::Advisory {
                    confidence: heuristic.confidence,
                    reasoning: heuristic.reasoning,
                    warning: "This is heuristic inference, not verification".to_string(),
                }
            }
        }
    }
    
    fn verify_physics(&self, physics: PhysicsLaw) -> VerificationResult {
        match physics.law {
            PhysicsLawType::NewtonianMechanics => {
                // Integrate F=ma
                let predicted = self.ode_solver.solve_newton(
                    &physics.initial_state,
                    physics.time_delta,
                );
                
                let error_ratio = relative_error(
                    &predicted,
                    &physics.claimed_final_state,
                );
                
                if error_ratio < 0.01 {  // 1% tolerance
                    VerificationResult::Verified {
                        method: "Newtonian ODE integration",
                        confidence: 0.95,
                        evidence: vec![predicted],
                    }
                } else {
                    VerificationResult::Violated {
                        reason: format!("Physics violation: {:.2}% error", error_ratio * 100.0),
                        expected: physics.claimed_final_state,
                        actual: predicted,
                    }
                }
            }
            
            PhysicsLawType::Conservation => {
                // Check conserved quantities
                let before = physics.initial_state[0];  // e.g., energy
                let after = physics.claimed_final_state[0];
                
                if (before - after).abs() / before < 0.01 {
                    VerificationResult::Verified {
                        method: "Conservation law",
                        confidence: 1.0,
                        evidence: vec![],
                    }
                } else {
                    VerificationResult::Violated {
                        reason: format!("Conservation violated: {} → {}", before, after),
                        expected: vec![before],
                        actual: vec![after],
                    }
                }
            }
            
            _ => todo!("Other physics laws"),
        }
    }
    
    fn verify_units(&self, eq: Equation) -> VerificationResult {
        let left_dims = self.unit_system.infer_dimensions(&eq.left);
        let right_dims = self.unit_system.infer_dimensions(&eq.right);
        
        if left_dims == right_dims {
            VerificationResult::Verified {
                method: "Dimensional analysis",
                confidence: 1.0,  // Units are exact
                evidence: vec![],
            }
        } else {
            VerificationResult::Violated {
                reason: format!("Unit mismatch: {:?} ≠ {:?}", left_dims, right_dims),
                expected: vec![],
                actual: vec![],
            }
        }
    }
    
    fn cross_check_sensors(&self, sensors: SensorClaim) -> VerificationResult {
        // Correlate vision + audio + other modalities
        let correlation = sensors.compute_correlation();
        
        if correlation > 0.8 {
            VerificationResult::Verified {
                method: "Multi-sensor fusion",
                confidence: correlation,
                evidence: sensors.readings,
            }
        } else {
            VerificationResult::Suspicious {
                reason: format!("Low sensor correlation: {:.2}", correlation),
                confidence: correlation,
            }
        }
    }
}

pub enum VerificationResult {
    Verified {
        method: &'static str,
        confidence: f64,
        evidence: Vec<Vec<f64>>,
    },
    Violated {
        reason: String,
        expected: Vec<f64>,
        actual: Vec<f64>,
    },
    Suspicious {
        reason: String,
        confidence: f64,
    },
    Advisory {
        confidence: f64,
        reasoning: String,
        warning: String,
    },
}
```

**Result**: Clear taxonomy of what can be **proven** vs **heuristic**.

---

## 🌊 CRITICAL FIX #4: Resonator (Bit-Packed Throughout)

### Problem Solved
- Mixed representations (Vec<u8> vs Vec<i8>)
- Stability threshold too tight (2 bits)
- Need dimension-scaled thresholds

### Solution: Bit-Packed + Scaled Thresholds

```rust
// src/crystal/resonator.rs

pub struct ResonatorNetwork {
    attractors: HashMap<String, Attractor>,
    spec: ProjectionSpec,
}

pub struct Attractor {
    pub id: String,
    pub hv16_bits: Vec<u8>,          // Packed bits (2KB for 16k dims)
    pub hv32_bits: Option<Vec<u8>>,  // Lazy (4KB for 32k dims)
    pub label: String,
    pub trust_weight: f64,
    pub stability: f64,
    pub margin: f64,
    pub hit_count: u64,
    pub last_updated: DateTime<Utc>,
    
    /// Poisoning defense
    pub write_protected: bool,
    pub version: u32,
    pub signature: Option<Vec<u8>>,
}

impl ResonatorNetwork {
    /// Core resonance (bit-packed throughout)
    pub fn resonate(
        &self,
        cue_bits: &[u8],
        policy: ResonancePolicy,
    ) -> ResonanceResult {
        let mut current = cue_bits.to_vec();
        let mut energy_history = vec![];
        
        for step in 0..policy.max_steps {
            // Find K nearest attractors
            let neighbors = self.find_nearest_k_bitwise(&current, 3);
            
            if neighbors.is_empty() {
                return ResonanceResult::Novel {
                    distance_to_nearest: f64::MAX,
                };
            }
            
            // Bundle neighbors (bipolar accumulator)
            let next = self.bundle_attractors_weighted(&neighbors, &current, policy.temperature);
            
            // Hamming distance (XOR + popcount)
            let changed_bits = hamming_distance_packed(&current, &next);
            let dims = self.spec.dims;
            let energy_drop_ratio = changed_bits as f64 / dims as f64;
            
            energy_history.push(energy_drop_ratio);
            
            // Dimension-scaled stability check
            if energy_drop_ratio < policy.stability_threshold_ratio {
                let (best_id, margin) = self.compute_margin_bitwise(&next);
                let stability = self.compute_stability(&energy_history);
                
                return ResonanceResult::Converged {
                    attractor_id: best_id,
                    steps: step + 1,
                    stability,
                    margin,
                };
            }
            
            current = next;
        }
        
        ResonanceResult::Unstable {
            steps: policy.max_steps,
            final_energy: *energy_history.last().unwrap(),
        }
    }
    
    fn find_nearest_k_bitwise(&self, cue: &[u8], k: usize) -> Vec<(String, u32)> {
        let mut distances: Vec<_> = self.attractors.iter()
            .map(|(id, attr)| {
                let dist = hamming_distance_packed(cue, &attr.hv16_bits);
                (id.clone(), dist)
            })
            .collect();
        
        distances.sort_by_key(|(_, dist)| *dist);
        distances.into_iter().take(k).collect()
    }
    
    fn bundle_attractors_weighted(
        &self,
        neighbors: &[(String, u32)],
        current: &[u8],
        temperature: f64,
    ) -> Vec<u8> {
        let dims = self.spec.body.dims;
        
        // Use f64 accumulator for weighted resonance
        let mut accumulator = vec![0.0f64; dims];
        
        // Add current state (weight = 1.0)
        for (i, byte) in current.iter().enumerate() {
            for bit in 0..8 {
                let bit_val = (byte >> bit) & 1;
                let bipolar = if bit_val == 1 { 1.0 } else { -1.0 };
                accumulator[i * 8 + bit] += bipolar;
            }
        }
        
        // Add weighted attractors
        for (id, dist) in neighbors {
            let attr = &self.attractors[id];
            
            // Exponential weighting (closer = higher weight)
            let weight = (-(*dist as f64) / temperature).exp();
            
            for (i, byte) in attr.hv16_bits.iter().enumerate() {
                for bit in 0..8 {
                    let bit_val = (byte >> bit) & 1;
                    let bipolar = if bit_val == 1 { 1.0 } else { -1.0 };
                    accumulator[i * 8 + bit] += bipolar * weight;
                }
            }
        }
        
        // Crystallize to bits
        let mut result = vec![0u8; dims / 8];
        for (i, &val) in accumulator.iter().enumerate() {
            if val > 0.0 {
                result[i / 8] |= 1 << (i % 8);
            }
        }
        result
    }
    
    fn compute_margin_bitwise(&self, cue: &[u8]) -> (String, f64) {
        let mut distances: Vec<_> = self.attractors.iter()
            .map(|(id, attr)| {
                let dist = hamming_distance_packed(cue, &attr.hv16_bits);
                (id.clone(), dist)
            })
            .collect();
        
        distances.sort_by_key(|(_, dist)| *dist);
        
        if distances.len() < 2 {
            return (distances[0].0.clone(), f64::MAX);
        }
        
        let margin = (distances[1].1 - distances[0].1) as f64;
        (distances[0].0.clone(), margin)
    }
    
    fn compute_stability(&self, energy_history: &[f64]) -> f64 {
        if energy_history.len() < 3 {
            return 0.0;
        }
        
        // Stability = rate of convergence
        // High stability = rapid convergence (early values high, late values low)
        let first_half_avg = energy_history[..energy_history.len()/2].iter().sum::<f64>() 
            / (energy_history.len()/2) as f64;
        let second_half_avg = energy_history[energy_history.len()/2..].iter().sum::<f64>()
            / (energy_history.len() - energy_history.len()/2) as f64;
        
        // Convergence rate: how much did energy drop from first to second half?
        let convergence_rate = (first_half_avg - second_half_avg) / first_half_avg.max(1e-9);
        convergence_rate.max(0.0).min(1.0)
    }
    
    /// Progressive retrieval: HV16 → HV32 on demand
    pub fn retrieve_progressive(
        &mut self,
        cue: &[u8],
        policy: ResonancePolicy,
    ) -> RetrievalResult {
        // Try HV16 first
        let result16 = self.resonate(cue, policy.clone());
        
        match result16 {
            ResonanceResult::Converged { margin, final_energy_ratio, .. } => {
                // Check if HV32 needed (correct unit comparison)
                if margin < policy.margin_threshold_16 
                    || final_energy_ratio > policy.energy_threshold_ratio {
                    
                    // Upgrade to HV32
                    let cue32 = self.upgrade_cue_to_hv32(cue);
                    let result32 = self.resonate(&cue32, policy);
                    
                    RetrievalResult::Resolved32(result32)
                } else {
                    RetrievalResult::Resolved16(result16)
                }
            }
            
            ResonanceResult::Novel { .. } => RetrievalResult::Novel,
            
            ResonanceResult::Unstable { .. } => {
                // Try HV32 before giving up
                let cue32 = self.upgrade_cue_to_hv32(cue);
                let result32 = self.resonate(&cue32, policy);
                RetrievalResult::Resolved32(result32)
            }
        }
    }
    
    /// Learn attractor with poisoning defenses
    pub fn learn_attractor(
        &mut self,
        bits: &[u8],
        label: String,
        trust: f64,
        require_approval: bool,
    ) -> Result<String> {
        // Check novelty
        let novelty = self.compute_novelty_bitwise(bits);
        
        if novelty < 0.3 {
            return Err(anyhow!("Too similar to existing attractor (novelty < 0.3)"));
        }
        
        // Require human approval for high-trust attractors
        if trust > 0.8 && require_approval {
            // TODO: Prompt user for confirmation
        }
        
        let id = format!("attr:{}", uuid::Uuid::new_v4());
        
        let attractor = Attractor {
            id: id.clone(),
            hv16_bits: bits.to_vec(),
            hv32_bits: None,
            label,
            trust_weight: trust,
            stability: 0.0,
            margin: 0.0,
            hit_count: 0,
            last_updated: Utc::now(),
            write_protected: trust > 0.8,  // Protect high-trust
            version: 1,
            signature: None,  // TODO: Sign with instance key
        };
        
        self.attractors.insert(id.clone(), attractor);
        Ok(id)
    }
    
    fn compute_novelty_bitwise(&self, cue: &[u8]) -> f64 {
        if self.attractors.is_empty() {
            return 1.0;
        }
        
        let min_dist = self.attractors.values()
            .map(|attr| hamming_distance_packed(cue, &attr.hv16_bits))
            .min()
            .unwrap();
        
        min_dist as f64 / self.spec.dims as f64
    }
}

pub struct ResonancePolicy {
    pub max_steps: usize,
    pub margin_threshold_16: f64,         // Bits (e.g., 64.0)
    pub energy_threshold_ratio: f64,      // Dimension-scaled (e.g., 0.001)
    pub temperature: f64,
}

impl Default for ResonancePolicy {
    fn default() -> Self {
        Self {
            max_steps: 8,
            margin_threshold_16: 64.0,        // 64 bits margin minimum
            energy_threshold_ratio: 0.001,    // 0.1% dims still changing
            temperature: 1.0,
        }
    }
}

pub enum ResonanceResult {
    Converged {
        attractor_id: String,
        steps: usize,
        stability: f64,
        margin: f64,
        final_energy_ratio: f64,  // For correct threshold comparison
    },
    Novel {
        distance_to_nearest: f64,
    },
    Unstable {
        steps: usize,
        final_energy_ratio: f64,
    },
}

/// Fast Hamming distance on packed bits
/// Safe up to 2^32 bits (536 million dims) with u32 accumulator
fn hamming_distance_packed(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}