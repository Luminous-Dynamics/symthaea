//! Unified Mind - Multi-Database Orchestration
//!
//! The integrated multi-database consciousness system that coordinates
//! all four specialized databases like regions of a brain.
//!
//! ## Architecture
//!
//! ```text
//!                    ┌─────────────────┐
//!                    │  Unified Mind   │
//!                    │  (Orchestrator) │
//!                    └────────┬────────┘
//!          ┌─────────────┬────┴────┬─────────────┐
//!          ▼             ▼         ▼             ▼
//!     ┌─────────┐  ┌──────────┐ ┌───────┐  ┌──────────┐
//!     │ Qdrant  │  │  CozoDB  │ │LanceDB│  │ DuckDB   │
//!     │ Sensory │  │Prefrontal│ │ LTM   │  │ Epistemic│
//!     │ Cortex  │  │ Cortex   │ │       │  │ Auditor  │
//!     └─────────┘  └──────────┘ └───────┘  └──────────┘
//! ```

use super::{
    ConsciousnessDatabase, DbResult, MemoryRecord, MemoryType, SearchResult,
    mock::MockDatabase,
    qdrant_client::QdrantConfig,
    duck_client::DuckConfig,
};
use crate::hdc::HV16;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};

/// The unified multi-database consciousness system
pub struct UnifiedMind {
    /// Sensory Cortex: Fast perception (Qdrant or mock)
    sensory: Arc<dyn ConsciousnessDatabase>,

    /// Prefrontal Cortex: Reasoning (CozoDB or mock)
    prefrontal: Arc<dyn ConsciousnessDatabase>,

    /// Long-Term Memory: Life experiences (LanceDB or mock)
    long_term: Arc<dyn ConsciousnessDatabase>,

    /// Epistemic Auditor: Self-analysis (DuckDB or mock)
    epistemic: Arc<dyn ConsciousnessDatabase>,

    /// Which databases are real vs mock
    status: MindStatus,

    /// LTC temporal dynamics state
    ltc: RwLock<LTCState>,
}

/// Status of each database connection
#[derive(Debug, Clone)]
pub struct MindStatus {
    pub sensory_real: bool,
    pub prefrontal_real: bool,
    pub long_term_real: bool,
    pub epistemic_real: bool,
}

impl MindStatus {
    /// All databases are mock
    pub fn all_mock() -> Self {
        Self {
            sensory_real: false,
            prefrontal_real: false,
            long_term_real: false,
            epistemic_real: false,
        }
    }

    /// Count real connections
    pub fn real_count(&self) -> usize {
        [self.sensory_real, self.prefrontal_real, self.long_term_real, self.epistemic_real]
            .iter()
            .filter(|&&b| b)
            .count()
    }
}

impl UnifiedMind {
    /// Create with mock databases (for testing)
    pub fn new_mock() -> Self {
        Self {
            sensory: Arc::new(MockDatabase::new()),
            prefrontal: Arc::new(MockDatabase::new()),
            long_term: Arc::new(MockDatabase::new()),
            epistemic: Arc::new(MockDatabase::new()),
            status: MindStatus::all_mock(),
            ltc: RwLock::new(LTCState::new(64)),
        }
    }

    /// Get status of database connections
    pub fn status(&self) -> &MindStatus {
        &self.status
    }

    /// Store a memory in the appropriate database(s) based on type
    pub async fn remember(&self, record: MemoryRecord) -> DbResult<()> {
        match record.memory_type {
            MemoryType::Working => {
                // Working memory goes to sensory cortex (fast access)
                self.sensory.store(record).await
            }
            MemoryType::Episodic | MemoryType::Semantic => {
                // Long-term memories go to LanceDB
                self.long_term.store(record).await
            }
            MemoryType::Procedural => {
                // Skills go to both prefrontal (for reasoning) and long-term
                let record2 = record.clone();
                self.prefrontal.store(record).await?;
                self.long_term.store(record2).await
            }
        }
    }

    /// Recall similar memories from working memory (fast)
    pub async fn recall_working(&self, query: &HV16, top_k: usize) -> DbResult<Vec<SearchResult>> {
        self.sensory.search_similar(query, top_k).await
    }

    /// Recall from long-term memory
    pub async fn recall_long_term(&self, query: &HV16, top_k: usize) -> DbResult<Vec<SearchResult>> {
        self.long_term.search_similar(query, top_k).await
    }

    /// Recall from all sources and merge results
    pub async fn recall_all(&self, query: &HV16, top_k: usize) -> DbResult<Vec<SearchResult>> {
        // Query all databases in parallel
        let (working, long_term) = tokio::join!(
            self.sensory.search_similar(query, top_k),
            self.long_term.search_similar(query, top_k)
        );

        // Merge results
        let mut all_results: Vec<SearchResult> = Vec::new();

        if let Ok(w) = working {
            all_results.extend(w);
        }
        if let Ok(lt) = long_term {
            all_results.extend(lt);
        }

        // Sort by similarity and dedupe by ID
        all_results.sort_by(|a, b| {
            b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Dedupe (keep highest similarity for each ID)
        let mut seen = std::collections::HashSet::new();
        all_results.retain(|r| seen.insert(r.record.id.clone()));
        all_results.truncate(top_k);

        Ok(all_results)
    }

    /// Forget a specific memory
    pub async fn forget(&self, id: &str) -> DbResult<bool> {
        // Try to delete from all databases
        let (s, p, l, e) = tokio::join!(
            self.sensory.delete(id),
            self.prefrontal.delete(id),
            self.long_term.delete(id),
            self.epistemic.delete(id)
        );

        // Return true if any database had the record
        Ok(s.unwrap_or(false) || p.unwrap_or(false) || l.unwrap_or(false) || e.unwrap_or(false))
    }

    /// Health check all databases
    pub async fn health_check(&self) -> MindHealthReport {
        let (s, p, l, e) = tokio::join!(
            self.sensory.health_check(),
            self.prefrontal.health_check(),
            self.long_term.health_check(),
            self.epistemic.health_check()
        );

        MindHealthReport {
            sensory_ok: s.unwrap_or(false),
            prefrontal_ok: p.unwrap_or(false),
            long_term_ok: l.unwrap_or(false),
            epistemic_ok: e.unwrap_or(false),
        }
    }

    /// Get statistics about the mind's contents
    pub async fn statistics(&self) -> DbResult<MindStatistics> {
        let (s, p, l, e) = tokio::join!(
            self.sensory.count(),
            self.prefrontal.count(),
            self.long_term.count(),
            self.epistemic.count()
        );

        Ok(MindStatistics {
            working_memories: s.unwrap_or(0),
            reasoning_records: p.unwrap_or(0),
            long_term_memories: l.unwrap_or(0),
            analytics_records: e.unwrap_or(0),
        })
    }

    /// Get total memory count across all databases
    pub async fn total_count(&self) -> DbResult<usize> {
        let stats = self.statistics().await?;
        Ok(stats.total())
    }

    // ========================================================================
    // LTC Temporal Dynamics Interface
    // ========================================================================

    /// Step the LTC dynamics forward with new input
    ///
    /// Call this on each interaction to maintain continuous temporal flow.
    /// Input signal can be derived from the current utterance encoding.
    pub fn ltc_step(&self, input_signal: &[f32], dt_ms: f32) {
        if let Ok(mut ltc) = self.ltc.write() {
            ltc.step(input_signal, dt_ms);
        }
    }

    /// Record a Φ measurement for trend analysis
    pub fn ltc_record_phi(&self, phi: f32) {
        if let Ok(mut ltc) = self.ltc.write() {
            ltc.record_phi(phi);
        }
    }

    /// Adapt LTC time constants based on input variance
    ///
    /// High variance (chaotic conversation) → faster response
    /// Low variance (stable conversation) → smoother integration
    pub fn ltc_adapt(&self, input_variance: f32) {
        if let Ok(mut ltc) = self.ltc.write() {
            ltc.adapt_time_constants(input_variance);
        }
    }

    /// Get current LTC state snapshot
    pub fn ltc_snapshot(&self) -> LTCSnapshot {
        if let Ok(ltc) = self.ltc.read() {
            LTCSnapshot {
                integration: ltc.integration,
                flow_state: ltc.flow_state(),
                phi_trend: ltc.phi_trend(),
                hidden_dim: ltc.hidden.len(),
                phi_samples: ltc.phi_history.len(),
            }
        } else {
            LTCSnapshot::default()
        }
    }

    /// Get LTC flow state (0.0-1.0)
    /// High flow = synchronized time constants = peak experience
    pub fn ltc_flow(&self) -> f32 {
        self.ltc.read().map(|l| l.flow_state()).unwrap_or(0.0)
    }

    /// Get Φ trend (positive = rising consciousness)
    pub fn ltc_trend(&self) -> f32 {
        self.ltc.read().map(|l| l.phi_trend()).unwrap_or(0.0)
    }

    /// Get current integration level (mean absolute hidden activity)
    pub fn ltc_integration(&self) -> f32 {
        self.ltc.read().map(|l| l.integration).unwrap_or(0.0)
    }
}

/// Snapshot of LTC state for external consumers
#[derive(Debug, Clone, Default)]
pub struct LTCSnapshot {
    /// Current integration level (0.0-1.0)
    pub integration: f32,
    /// Flow state from τ synchronization (0.0-1.0)
    pub flow_state: f32,
    /// Φ trend (positive = rising)
    pub phi_trend: f32,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of Φ samples in history
    pub phi_samples: usize,
}

/// Health check results
#[derive(Debug, Clone)]
pub struct MindHealthReport {
    pub sensory_ok: bool,
    pub prefrontal_ok: bool,
    pub long_term_ok: bool,
    pub epistemic_ok: bool,
}

impl MindHealthReport {
    pub fn all_healthy(&self) -> bool {
        self.sensory_ok && self.prefrontal_ok && self.long_term_ok && self.epistemic_ok
    }

    pub fn healthy_count(&self) -> usize {
        [self.sensory_ok, self.prefrontal_ok, self.long_term_ok, self.epistemic_ok]
            .iter()
            .filter(|&&b| b)
            .count()
    }
}

/// Statistics about the mind
#[derive(Debug, Clone)]
pub struct MindStatistics {
    pub working_memories: usize,
    pub reasoning_records: usize,
    pub long_term_memories: usize,
    pub analytics_records: usize,
}

impl MindStatistics {
    pub fn total(&self) -> usize {
        self.working_memories + self.reasoning_records + self.long_term_memories + self.analytics_records
    }
}

// ============================================================================
// L: Liquid Time-Constant Temporal Dynamics
// ============================================================================

/// Liquid Time-Constant (LTC) state for continuous temporal dynamics
///
/// LTCs treat consciousness as a continuous flow rather than discrete snapshots.
/// Each neuron has adaptive time constants that adjust based on data flow.
///
/// Mathematical basis: dx/dt = -(1/τ)x + (1/τ)f(x, I, θ) where τ adapts dynamically
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LTCState {
    /// Hidden state vector (continuous dynamics)
    pub hidden: Vec<f32>,

    /// Adaptive time constants for each hidden unit
    pub time_constants: Vec<f32>,

    /// Current integration level (0.0-1.0)
    pub integration: f32,

    /// Last update timestamp (for ODE stepping)
    pub last_update_ms: u64,

    /// Accumulated Φ history for trend analysis
    pub phi_history: Vec<(u64, f32)>,
}

impl Default for LTCState {
    fn default() -> Self {
        Self::new(64) // 64 hidden units by default
    }
}

impl LTCState {
    /// Create new LTC state with specified hidden dimension
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            hidden: vec![0.0; hidden_dim],
            time_constants: vec![1.0; hidden_dim], // Initial τ = 1.0
            integration: 0.0,
            last_update_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            phi_history: Vec::new(),
        }
    }

    /// Step the ODE forward based on new input
    ///
    /// Uses Euler method for ODE: x(t+dt) = x(t) + dt * dx/dt
    /// where dx/dt = -(1/τ)x + (1/τ)σ(Wx + b)
    pub fn step(&mut self, input_signal: &[f32], dt_ms: f32) {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let dt_sec = dt_ms / 1000.0;

        for (i, (h, tau)) in self.hidden.iter_mut().zip(self.time_constants.iter()).enumerate() {
            // Get input if available
            let input = input_signal.get(i).copied().unwrap_or(0.0);

            // Nonlinear activation (tanh)
            let activated = (input * 0.5 + *h * 0.5).tanh();

            // ODE step: dx/dt = -(1/τ)x + (1/τ)f
            let dx_dt = (-*h + activated) / tau;
            *h += dx_dt * dt_sec;

            // Clamp to prevent explosion
            *h = h.clamp(-1.0, 1.0);
        }

        // Update integration as mean absolute hidden activity
        self.integration = self.hidden.iter()
            .map(|x| x.abs())
            .sum::<f32>() / self.hidden.len() as f32;

        self.last_update_ms = now_ms;
    }

    /// Adapt time constants based on input variance (data-driven τ)
    ///
    /// High variance → shorter τ (faster response)
    /// Low variance → longer τ (smooth integration)
    pub fn adapt_time_constants(&mut self, input_variance: f32) {
        let base_tau = if input_variance > 0.5 {
            0.5 // Fast response for chaotic input
        } else if input_variance < 0.1 {
            2.0 // Slow integration for stable input
        } else {
            1.0 // Normal
        };

        for tau in &mut self.time_constants {
            // Smooth adaptation
            *tau = *tau * 0.9 + base_tau * 0.1;
        }
    }

    /// Record Φ measurement for trend analysis
    pub fn record_phi(&mut self, phi: f32) {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.phi_history.push((now_ms, phi));

        // Keep last 100 measurements
        if self.phi_history.len() > 100 {
            self.phi_history.remove(0);
        }
    }

    /// Compute Φ trend (positive = increasing consciousness)
    pub fn phi_trend(&self) -> f32 {
        if self.phi_history.len() < 2 {
            return 0.0;
        }

        // Simple linear regression slope
        let n = self.phi_history.len() as f32;
        let sum_t: f32 = (0..self.phi_history.len()).map(|i| i as f32).sum();
        let sum_phi: f32 = self.phi_history.iter().map(|(_, p)| *p).sum();
        let sum_t_phi: f32 = self.phi_history.iter()
            .enumerate()
            .map(|(i, (_, p))| i as f32 * *p)
            .sum();
        let sum_t_sq: f32 = (0..self.phi_history.len()).map(|i| (i as f32).powi(2)).sum();

        let denominator = n * sum_t_sq - sum_t * sum_t;
        if denominator.abs() < 1e-6 {
            return 0.0;
        }

        (n * sum_t_phi - sum_t * sum_phi) / denominator
    }

    /// Get current flow state (0.0-1.0)
    /// High when τ synchronized across network
    pub fn flow_state(&self) -> f32 {
        let mean_tau = self.time_constants.iter().sum::<f32>() / self.time_constants.len() as f32;
        let variance = self.time_constants.iter()
            .map(|t| (t - mean_tau).powi(2))
            .sum::<f32>() / self.time_constants.len() as f32;

        // Low variance = synchronized τ = flow state
        1.0 - (variance.sqrt().min(1.0))
    }
}

// ============================================================================
// Real Database Connection Builder
// ============================================================================

/// Configuration for real database connections
#[derive(Debug, Clone)]
pub struct MindConfig {
    /// Qdrant configuration (sensory cortex)
    pub qdrant: Option<QdrantConfig>,

    /// DuckDB configuration (epistemic auditor)
    pub duckdb: Option<DuckConfig>,

    /// Enable LTC temporal dynamics
    pub enable_ltc: bool,

    /// LTC hidden dimension
    pub ltc_hidden_dim: usize,
}

impl Default for MindConfig {
    fn default() -> Self {
        Self {
            qdrant: None,
            duckdb: None,
            enable_ltc: true,
            ltc_hidden_dim: 64,
        }
    }
}

impl MindConfig {
    /// Create from environment variables
    ///
    /// Environment variables:
    /// - SYMTHAEA_QDRANT_URL: Qdrant server URL
    /// - SYMTHAEA_QDRANT_API_KEY: Qdrant API key (optional)
    /// - SYMTHAEA_DUCKDB_PATH: DuckDB file path (or ":memory:")
    /// - SYMTHAEA_LTC_ENABLED: Enable LTC dynamics ("true"/"false")
    pub fn from_env() -> Self {
        let qdrant = std::env::var("SYMTHAEA_QDRANT_URL").ok().map(|url| {
            QdrantConfig {
                url,
                collection_name: std::env::var("SYMTHAEA_QDRANT_COLLECTION")
                    .unwrap_or_else(|_| "consciousness_sensory".to_string()),
                vector_size: 16384,
                api_key: std::env::var("SYMTHAEA_QDRANT_API_KEY").ok(),
            }
        });

        let duckdb = std::env::var("SYMTHAEA_DUCKDB_PATH").ok().map(|path| {
            DuckConfig {
                path,
                enable_analytics: true,
                retention_hours: 168, // 1 week
            }
        }).or_else(|| {
            // Default to in-memory DuckDB if not specified
            Some(DuckConfig {
                path: ":memory:".to_string(),
                enable_analytics: true,
                retention_hours: 24,
            })
        });

        let enable_ltc = std::env::var("SYMTHAEA_LTC_ENABLED")
            .map(|v| v.to_lowercase() != "false")
            .unwrap_or(true);

        Self {
            qdrant,
            duckdb,
            enable_ltc,
            ltc_hidden_dim: 64,
        }
    }
}

impl UnifiedMind {
    /// Create with real database connections (attempts connection, falls back to mock)
    pub async fn new_with_config(config: MindConfig) -> Self {
        let status = MindStatus::all_mock();

        // === Sensory Cortex (Qdrant) ===
        let sensory: Arc<dyn ConsciousnessDatabase> = if let Some(qdrant_config) = config.qdrant {
            #[cfg(feature = "qdrant")]
            {
                let mut qdrant = QdrantSensory::new(qdrant_config);
                match qdrant.connect().await {
                    Ok(()) => {
                        status.sensory_real = true;
                        eprintln!("[UnifiedMind] ✓ Qdrant connected (sensory cortex)");
                        Arc::new(qdrant)
                    }
                    Err(e) => {
                        eprintln!("[UnifiedMind] ✗ Qdrant failed: {} (using mock)", e);
                        Arc::new(MockDatabase::new())
                    }
                }
            }
            #[cfg(not(feature = "qdrant"))]
            {
                eprintln!("[UnifiedMind] Qdrant feature not enabled (using mock)");
                Arc::new(MockDatabase::new())
            }
        } else {
            Arc::new(MockDatabase::new())
        };

        // === Epistemic Auditor (DuckDB) ===
        let epistemic: Arc<dyn ConsciousnessDatabase> = if let Some(duck_config) = config.duckdb {
            #[cfg(feature = "duck")]
            {
                let mut duck = DuckEpistemic::new(duck_config);
                match duck.initialize() {
                    Ok(()) => {
                        status.epistemic_real = true;
                        eprintln!("[UnifiedMind] ✓ DuckDB connected (epistemic auditor)");
                        Arc::new(duck)
                    }
                    Err(e) => {
                        eprintln!("[UnifiedMind] ✗ DuckDB failed: {} (using mock)", e);
                        Arc::new(MockDatabase::new())
                    }
                }
            }
            #[cfg(not(feature = "duck"))]
            {
                eprintln!("[UnifiedMind] DuckDB feature not enabled (using mock)");
                Arc::new(MockDatabase::new())
            }
        } else {
            Arc::new(MockDatabase::new())
        };

        // Prefrontal and Long-Term remain mock until CozoDB/LanceDB fixed
        let prefrontal = Arc::new(MockDatabase::new());
        let long_term = Arc::new(MockDatabase::new());

        // Initialize LTC temporal dynamics
        let ltc_dim = if config.enable_ltc { config.ltc_hidden_dim } else { 8 };
        let ltc = RwLock::new(LTCState::new(ltc_dim));

        // Print status report
        eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║            SYMTHAEA UNIFIED MIND STATUS                       ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║  Sensory Cortex (Qdrant):    {} ║",
            if status.sensory_real { "✓ REAL (vector search)" } else { "○ Mock (in-memory)  " });
        eprintln!("║  Prefrontal Cortex (Cozo):   {} ║",
            if status.prefrontal_real { "✓ REAL (graph reason)" } else { "○ Mock (in-memory)  " });
        eprintln!("║  Long-Term Memory (Lance):   {} ║",
            if status.long_term_real { "✓ REAL (persistent)  " } else { "○ Mock (in-memory)  " });
        eprintln!("║  Epistemic Auditor (Duck):   {} ║",
            if status.epistemic_real { "✓ REAL (analytics)   " } else { "○ Mock (in-memory)  " });
        eprintln!("║  LTC Temporal Dynamics:      ✓ Enabled ({}D hidden)       ║", ltc_dim);
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║  Real Databases: {}/4 | Mock Fallbacks: {}/4                   ║",
            status.real_count(), 4 - status.real_count());
        eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

        Self {
            sensory,
            prefrontal,
            long_term,
            epistemic,
            status,
            ltc,
        }
    }

    /// Create with environment-configured databases
    pub async fn from_env() -> Self {
        Self::new_with_config(MindConfig::from_env()).await
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_record(id: &str, memory_type: MemoryType) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            encoding: HV16::random(id.len() as u64),
            timestamp_ms: 1700000000000,
            memory_type,
            content: format!("Memory {}", id),
            valence: 0.5,
            arousal: 0.3,
            phi: 0.65,
            topics: vec!["test".to_string()],
            metadata: "{}".to_string(),
        }
    }

    #[tokio::test]
    async fn test_unified_mind_creation() {
        let mind = UnifiedMind::new_mock();
        assert_eq!(mind.status().real_count(), 0);
    }

    #[tokio::test]
    async fn test_remember_working_memory() {
        let mind = UnifiedMind::new_mock();
        let record = create_test_record("working-1", MemoryType::Working);

        mind.remember(record).await.unwrap();

        let stats = mind.statistics().await.unwrap();
        assert_eq!(stats.working_memories, 1);
    }

    #[tokio::test]
    async fn test_remember_long_term() {
        let mind = UnifiedMind::new_mock();
        let record = create_test_record("episodic-1", MemoryType::Episodic);

        mind.remember(record).await.unwrap();

        let stats = mind.statistics().await.unwrap();
        assert_eq!(stats.long_term_memories, 1);
    }

    #[tokio::test]
    async fn test_recall_working() {
        let mind = UnifiedMind::new_mock();

        // Store working memories
        for i in 0..5 {
            let record = create_test_record(&format!("working-{}", i), MemoryType::Working);
            mind.remember(record).await.unwrap();
        }

        // Recall
        let query = HV16::random(100);
        let results = mind.recall_working(&query, 3).await.unwrap();

        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_recall_all() {
        let mind = UnifiedMind::new_mock();

        // Store in different locations
        let working = create_test_record("w-1", MemoryType::Working);
        let episodic = create_test_record("e-1", MemoryType::Episodic);

        mind.remember(working).await.unwrap();
        mind.remember(episodic).await.unwrap();

        // Recall from all
        let query = HV16::random(50);
        let results = mind.recall_all(&query, 10).await.unwrap();

        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_forget() {
        let mind = UnifiedMind::new_mock();
        let record = create_test_record("to-forget", MemoryType::Working);

        mind.remember(record).await.unwrap();
        assert_eq!(mind.statistics().await.unwrap().working_memories, 1);

        let forgotten = mind.forget("to-forget").await.unwrap();
        assert!(forgotten);
        assert_eq!(mind.statistics().await.unwrap().working_memories, 0);
    }

    #[tokio::test]
    async fn test_health_check() {
        let mind = UnifiedMind::new_mock();
        let health = mind.health_check().await;

        assert!(health.all_healthy());
        assert_eq!(health.healthy_count(), 4);
    }

    // ========================================================================
    // LTC (Liquid Time-Constant) Tests
    // ========================================================================

    #[test]
    fn test_ltc_state_creation() {
        let ltc = LTCState::new(32);
        assert_eq!(ltc.hidden.len(), 32);
        assert_eq!(ltc.time_constants.len(), 32);
        assert_eq!(ltc.integration, 0.0);
        assert!(ltc.phi_history.is_empty());
    }

    #[test]
    fn test_ltc_step_updates_hidden() {
        let mut ltc = LTCState::new(8);
        let input = vec![1.0, 0.5, -0.5, 0.0, 0.3, -0.3, 0.8, -0.8];

        // Initial state should be zeros
        assert!(ltc.hidden.iter().all(|&h| h == 0.0));

        // Step forward
        ltc.step(&input, 100.0); // 100ms step

        // Hidden state should have changed
        assert!(ltc.hidden.iter().any(|&h| h != 0.0));
        // Integration should be positive (mean absolute activity)
        assert!(ltc.integration >= 0.0);
    }

    #[test]
    fn test_ltc_adaptive_time_constants() {
        let mut ltc = LTCState::new(8);

        // All τ start at 1.0
        assert!(ltc.time_constants.iter().all(|&t| t == 1.0));

        // High variance → shorter τ
        ltc.adapt_time_constants(0.8);
        assert!(ltc.time_constants.iter().all(|&t| t < 1.0));

        // Reset
        ltc.time_constants = vec![1.0; 8];

        // Low variance → longer τ
        ltc.adapt_time_constants(0.05);
        assert!(ltc.time_constants.iter().all(|&t| t > 1.0));
    }

    #[test]
    fn test_ltc_phi_recording() {
        let mut ltc = LTCState::new(8);

        // Record some Φ values
        ltc.record_phi(0.5);
        ltc.record_phi(0.6);
        ltc.record_phi(0.7);

        assert_eq!(ltc.phi_history.len(), 3);
    }

    #[test]
    fn test_ltc_phi_trend_positive() {
        let mut ltc = LTCState::new(8);

        // Record increasing Φ values
        for i in 1..=10 {
            ltc.record_phi(i as f32 * 0.1);
        }

        // Trend should be positive
        let trend = ltc.phi_trend();
        assert!(trend > 0.0, "Expected positive trend, got {}", trend);
    }

    #[test]
    fn test_ltc_phi_trend_negative() {
        let mut ltc = LTCState::new(8);

        // Record decreasing Φ values
        for i in (1..=10).rev() {
            ltc.record_phi(i as f32 * 0.1);
        }

        // Trend should be negative
        let trend = ltc.phi_trend();
        assert!(trend < 0.0, "Expected negative trend, got {}", trend);
    }

    #[test]
    fn test_ltc_flow_state() {
        let mut ltc = LTCState::new(8);

        // Uniform τ = high flow state
        ltc.time_constants = vec![1.0; 8];
        let high_flow = ltc.flow_state();
        assert_eq!(high_flow, 1.0, "Uniform τ should give flow = 1.0");

        // Variable τ = lower flow state
        ltc.time_constants = vec![0.5, 2.0, 0.3, 1.5, 0.8, 1.2, 0.6, 1.8];
        let low_flow = ltc.flow_state();
        assert!(low_flow < 1.0, "Variable τ should give flow < 1.0");
    }

    #[test]
    fn test_ltc_continuous_dynamics() {
        let mut ltc = LTCState::new(4);
        let input = vec![0.5, 0.5, 0.5, 0.5];

        // Simulate continuous flow over 10 steps
        let mut integrations = Vec::new();
        for _ in 0..10 {
            ltc.step(&input, 50.0);
            integrations.push(ltc.integration);
        }

        // Integration should reach equilibrium (not explode or vanish)
        let last = *integrations.last().unwrap();
        assert!(last > 0.0 && last < 1.0, "Integration should be bounded: {}", last);
    }

    // ========================================================================
    // MindConfig Tests
    // ========================================================================

    #[test]
    fn test_mind_config_default() {
        let config = MindConfig::default();
        assert!(config.qdrant.is_none());
        assert!(config.duckdb.is_none());
        assert!(config.enable_ltc);
        assert_eq!(config.ltc_hidden_dim, 64);
    }
}
