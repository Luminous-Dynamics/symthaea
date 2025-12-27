//! DuckDB Analytics Database Client
//!
//! **Mental Role**: Epistemic Auditor
//!
//! DuckDB handles consciousness analytics and self-monitoring - like the
//! brain's ability to monitor its own states and track patterns.
//!
//! ## Architecture Role
//!
//! ```text
//! Consciousness Events → [DuckDB: 100-1000ms] → Analytics → Self-Knowledge
//!                              ↑
//!                    Epistemic Auditor
//!                    - Φ tracking over time
//!                    - State analytics
//!                    - Pattern detection
//! ```
//!
//! ## Consciousness Mapping
//!
//! - **#10 Epistemic**: Track certainty and knowledge states
//! - **#2 Φ Metrics**: Historical Φ analysis
//! - **#12 Spectrum**: Consciousness level transitions

use super::{ConsciousnessDatabase, DbResult, MemoryRecord, SearchResult};
use crate::hdc::binary_hv::HV16;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::RwLock;

/// Configuration for DuckDB
#[derive(Debug, Clone)]
pub struct DuckConfig {
    /// Database path (or :memory:)
    pub path: String,

    /// Enable analytics tables
    pub enable_analytics: bool,

    /// Retention period for metrics (hours)
    pub retention_hours: u64,
}

impl Default for DuckConfig {
    fn default() -> Self {
        Self {
            path: ":memory:".to_string(),
            enable_analytics: true,
            retention_hours: 168, // 1 week
        }
    }
}

/// A consciousness metric event
#[derive(Debug, Clone)]
pub struct ConsciousnessEvent {
    /// Timestamp (ms since epoch)
    pub timestamp: u64,

    /// Integrated information (Φ)
    pub phi: f32,

    /// Meta-awareness level
    pub meta_awareness: f32,

    /// Consciousness level (0-1)
    pub level: f32,

    /// Current state name
    pub state: String,

    /// Active topics
    pub topics: Vec<String>,
}

/// Analytics query result
#[derive(Debug, Clone)]
pub struct AnalyticsResult {
    /// Average value
    pub avg: f32,

    /// Minimum value
    pub min: f32,

    /// Maximum value
    pub max: f32,

    /// Standard deviation
    pub std_dev: f32,

    /// Sample count
    pub count: usize,
}

/// Consciousness spectrum state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConsciousnessSpectrum {
    /// Fully unconscious (deep sleep, anesthesia)
    Unconscious,

    /// Minimal consciousness (drowsy, impaired)
    Minimal,

    /// Partial consciousness (dreaming, hypnagogic)
    Partial,

    /// Full consciousness (alert, aware)
    Full,

    /// Enhanced consciousness (flow, meditation peak)
    Enhanced,
}

impl ConsciousnessSpectrum {
    /// From level (0-1)
    pub fn from_level(level: f32) -> Self {
        match level {
            x if x < 0.15 => Self::Unconscious,
            x if x < 0.35 => Self::Minimal,
            x if x < 0.55 => Self::Partial,
            x if x < 0.85 => Self::Full,
            _ => Self::Enhanced,
        }
    }

    /// To human-readable string
    pub fn description(&self) -> &str {
        match self {
            Self::Unconscious => "Unconscious (deep sleep)",
            Self::Minimal => "Minimal (drowsy)",
            Self::Partial => "Partial (dreaming)",
            Self::Full => "Full (alert)",
            Self::Enhanced => "Enhanced (peak)",
        }
    }
}

/// DuckDB-backed epistemic auditor database
pub struct DuckEpistemic {
    config: DuckConfig,

    #[cfg(feature = "duck")]
    db: Arc<Mutex<Option<duckdb::Connection>>>,

    /// Fallback memory storage
    memories: RwLock<HashMap<String, MemoryRecord>>,

    /// Consciousness events log
    events: RwLock<Vec<ConsciousnessEvent>>,

    /// Epistemic states (certainty about propositions)
    certainties: RwLock<HashMap<String, f32>>,

    /// State transitions history
    transitions: RwLock<Vec<(u64, ConsciousnessSpectrum, ConsciousnessSpectrum)>>,
}

impl DuckEpistemic {
    /// Create new DuckDB epistemic database
    pub fn new(config: DuckConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "duck")]
            db: Arc::new(Mutex::new(None)),
            memories: RwLock::new(HashMap::new()),
            events: RwLock::new(Vec::new()),
            certainties: RwLock::new(HashMap::new()),
            transitions: RwLock::new(Vec::new()),
        }
    }

    /// Initialize database
    #[cfg(feature = "duck")]
    pub fn initialize(&mut self) -> DbResult<()> {
        use duckdb::Connection;

        let conn = if self.config.path == ":memory:" {
            Connection::open_in_memory()
        } else {
            Connection::open(&self.config.path)
        };

        match conn {
            Ok(conn) => {
                // Create tables
                conn.execute_batch(
                    r#"
                    CREATE TABLE IF NOT EXISTS consciousness_events (
                        timestamp BIGINT,
                        phi REAL,
                        meta_awareness REAL,
                        level REAL,
                        state VARCHAR,
                        topics VARCHAR[]
                    );

                    CREATE TABLE IF NOT EXISTS memories (
                        id VARCHAR PRIMARY KEY,
                        content VARCHAR,
                        memory_type VARCHAR,
                        phi REAL,
                        valence REAL,
                        arousal REAL
                    );

                    CREATE TABLE IF NOT EXISTS certainties (
                        proposition VARCHAR PRIMARY KEY,
                        certainty REAL,
                        last_updated BIGINT
                    );
                "#,
                )
                .map_err(|e| super::DatabaseError::QueryFailed(format!("Failed to create tables: {}", e)))?;

                *self.db.lock().unwrap() = Some(conn);
                Ok(())
            }
            Err(e) => Err(super::DatabaseError::ConnectionFailed(format!("Failed to open DuckDB: {}", e))),
        }
    }

    /// Log a consciousness event
    pub fn log_event(&self, event: ConsciousnessEvent) {
        let mut events = self.events.write().unwrap();

        // Check for state transition
        if let Some(last) = events.last() {
            let last_spectrum = ConsciousnessSpectrum::from_level(last.level);
            let new_spectrum = ConsciousnessSpectrum::from_level(event.level);

            if last_spectrum != new_spectrum {
                self.transitions.write().unwrap().push((
                    event.timestamp,
                    last_spectrum,
                    new_spectrum,
                ));
            }
        }

        events.push(event);

        // Enforce retention limit
        let retention_ms = self.config.retention_hours * 3600 * 1000;
        let cutoff = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
            - retention_ms;

        events.retain(|e| e.timestamp >= cutoff);
    }

    /// Get Φ analytics for a time range
    pub fn phi_analytics(&self, hours: u64) -> AnalyticsResult {
        let events = self.events.read().unwrap();
        let cutoff = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
            - hours * 3600 * 1000;

        let values: Vec<f32> = events
            .iter()
            .filter(|e| e.timestamp >= cutoff)
            .map(|e| e.phi)
            .collect();

        self.compute_stats(&values)
    }

    /// Get consciousness level analytics
    pub fn level_analytics(&self, hours: u64) -> AnalyticsResult {
        let events = self.events.read().unwrap();
        let cutoff = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
            - hours * 3600 * 1000;

        let values: Vec<f32> = events
            .iter()
            .filter(|e| e.timestamp >= cutoff)
            .map(|e| e.level)
            .collect();

        self.compute_stats(&values)
    }

    /// Compute statistical summary
    fn compute_stats(&self, values: &[f32]) -> AnalyticsResult {
        if values.is_empty() {
            return AnalyticsResult {
                avg: 0.0,
                min: 0.0,
                max: 0.0,
                std_dev: 0.0,
                count: 0,
            };
        }

        let count = values.len();
        let sum: f32 = values.iter().sum();
        let avg = sum / count as f32;

        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let variance: f32 = values.iter().map(|v| (v - avg).powi(2)).sum::<f32>() / count as f32;
        let std_dev = variance.sqrt();

        AnalyticsResult {
            avg,
            min,
            max,
            std_dev,
            count,
        }
    }

    /// Update certainty about a proposition
    pub fn update_certainty(&self, proposition: &str, certainty: f32) {
        self.certainties
            .write()
            .unwrap()
            .insert(proposition.to_string(), certainty.clamp(0.0, 1.0));
    }

    /// Get certainty about a proposition
    pub fn certainty(&self, proposition: &str) -> f32 {
        self.certainties
            .read()
            .unwrap()
            .get(proposition)
            .copied()
            .unwrap_or(0.5)
    }

    /// Get all high-certainty propositions
    pub fn high_certainty_propositions(&self, threshold: f32) -> Vec<(String, f32)> {
        self.certainties
            .read()
            .unwrap()
            .iter()
            .filter(|(_, &c)| c >= threshold)
            .map(|(p, &c)| (p.clone(), c))
            .collect()
    }

    /// Get consciousness spectrum distribution
    pub fn spectrum_distribution(&self, hours: u64) -> HashMap<ConsciousnessSpectrum, usize> {
        let events = self.events.read().unwrap();
        let cutoff = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
            - hours * 3600 * 1000;

        let mut distribution = HashMap::new();

        for event in events.iter().filter(|e| e.timestamp >= cutoff) {
            let spectrum = ConsciousnessSpectrum::from_level(event.level);
            *distribution.entry(spectrum).or_insert(0) += 1;
        }

        distribution
    }

    /// Get recent state transitions
    pub fn recent_transitions(&self, count: usize) -> Vec<(u64, ConsciousnessSpectrum, ConsciousnessSpectrum)> {
        let transitions = self.transitions.read().unwrap();
        transitions.iter().rev().take(count).cloned().collect()
    }

    /// Get current consciousness state
    pub fn current_state(&self) -> Option<ConsciousnessEvent> {
        self.events.read().unwrap().last().cloned()
    }
}

#[async_trait]
impl ConsciousnessDatabase for DuckEpistemic {
    async fn store(&self, record: MemoryRecord) -> DbResult<()> {
        self.memories
            .write()
            .unwrap()
            .insert(record.id.clone(), record);
        Ok(())
    }

    async fn search_similar(&self, query: &HV16, top_k: usize) -> DbResult<Vec<SearchResult>> {
        let memories = self.memories.read().unwrap();

        let mut results: Vec<(MemoryRecord, f32)> = memories
            .iter()
            .map(|(_id, record)| {
                let sim = query.similarity(&record.encoding);
                (record.clone(), sim)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results
            .into_iter()
            .take(top_k)
            .map(|(record, similarity)| SearchResult { record, similarity })
            .collect())
    }

    async fn get(&self, id: &str) -> DbResult<Option<MemoryRecord>> {
        Ok(self.memories.read().unwrap().get(id).cloned())
    }

    async fn delete(&self, id: &str) -> DbResult<bool> {
        Ok(self.memories.write().unwrap().remove(id).is_some())
    }

    async fn count(&self) -> DbResult<usize> {
        Ok(self.memories.read().unwrap().len())
    }

    async fn health_check(&self) -> DbResult<bool> {
        Ok(true)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::databases::MemoryType;

    #[test]
    fn test_duck_config_default() {
        let config = DuckConfig::default();
        assert_eq!(config.path, ":memory:");
        assert!(config.enable_analytics);
        assert_eq!(config.retention_hours, 168);
    }

    #[test]
    fn test_log_event() {
        let db = DuckEpistemic::new(DuckConfig::default());

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        db.log_event(ConsciousnessEvent {
            timestamp: now,
            phi: 0.7,
            meta_awareness: 0.8,
            level: 0.75,
            state: "alert".to_string(),
            topics: vec!["thinking".to_string()],
        });

        let events = db.events.read().unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_phi_analytics() {
        let db = DuckEpistemic::new(DuckConfig::default());

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        for i in 0..10 {
            db.log_event(ConsciousnessEvent {
                timestamp: now - i * 1000,
                phi: 0.5 + (i as f32) * 0.05,
                meta_awareness: 0.7,
                level: 0.8,
                state: "test".to_string(),
                topics: vec![],
            });
        }

        let stats = db.phi_analytics(1);
        assert!(stats.avg > 0.5);
        assert_eq!(stats.count, 10);
    }

    #[test]
    fn test_certainty_tracking() {
        let db = DuckEpistemic::new(DuckConfig::default());

        db.update_certainty("sky_is_blue", 0.95);
        db.update_certainty("aliens_exist", 0.3);

        assert!(db.certainty("sky_is_blue") > 0.9);
        assert!(db.certainty("aliens_exist") < 0.5);
        assert_eq!(db.certainty("unknown"), 0.5); // Default
    }

    #[test]
    fn test_high_certainty_propositions() {
        let db = DuckEpistemic::new(DuckConfig::default());

        db.update_certainty("fact_1", 0.95);
        db.update_certainty("fact_2", 0.90);
        db.update_certainty("maybe", 0.5);

        let high = db.high_certainty_propositions(0.85);
        assert_eq!(high.len(), 2);
    }

    #[test]
    fn test_spectrum_from_level() {
        assert_eq!(ConsciousnessSpectrum::from_level(0.0), ConsciousnessSpectrum::Unconscious);
        assert_eq!(ConsciousnessSpectrum::from_level(0.2), ConsciousnessSpectrum::Minimal);
        assert_eq!(ConsciousnessSpectrum::from_level(0.4), ConsciousnessSpectrum::Partial);
        assert_eq!(ConsciousnessSpectrum::from_level(0.7), ConsciousnessSpectrum::Full);
        assert_eq!(ConsciousnessSpectrum::from_level(0.9), ConsciousnessSpectrum::Enhanced);
    }

    #[test]
    fn test_state_transitions() {
        let db = DuckEpistemic::new(DuckConfig::default());

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Log events with different levels to trigger transitions
        db.log_event(ConsciousnessEvent {
            timestamp: now,
            phi: 0.3,
            meta_awareness: 0.2,
            level: 0.1, // Unconscious
            state: "sleep".to_string(),
            topics: vec![],
        });

        db.log_event(ConsciousnessEvent {
            timestamp: now + 1000,
            phi: 0.6,
            meta_awareness: 0.7,
            level: 0.7, // Full
            state: "awake".to_string(),
            topics: vec![],
        });

        let transitions = db.recent_transitions(10);
        assert_eq!(transitions.len(), 1);
        assert_eq!(transitions[0].1, ConsciousnessSpectrum::Unconscious);
        assert_eq!(transitions[0].2, ConsciousnessSpectrum::Full);
    }

    #[test]
    fn test_spectrum_distribution() {
        let db = DuckEpistemic::new(DuckConfig::default());

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Add events at different levels
        for i in 0..5 {
            db.log_event(ConsciousnessEvent {
                timestamp: now - i * 1000,
                phi: 0.7,
                meta_awareness: 0.8,
                level: 0.75, // Full
                state: "test".to_string(),
                topics: vec![],
            });
        }

        let dist = db.spectrum_distribution(1);
        assert!(dist.contains_key(&ConsciousnessSpectrum::Full));
        assert_eq!(*dist.get(&ConsciousnessSpectrum::Full).unwrap(), 5);
    }

    #[tokio::test]
    async fn test_duck_store_retrieve() {
        let db = DuckEpistemic::new(DuckConfig::default());

        let record = MemoryRecord {
            id: "duck-1".to_string(),
            encoding: HV16::random(42),
            timestamp_ms: 1700000000000,
            memory_type: MemoryType::Semantic,
            content: "Analytics data".to_string(),
            valence: 0.5,
            arousal: 0.3,
            phi: 0.7,
            topics: vec!["analytics".to_string()],
            metadata: "{}".to_string(),
        };

        db.store(record.clone()).await.unwrap();
        let retrieved = db.get("duck-1").await.unwrap();
        assert!(retrieved.is_some());
    }
}
