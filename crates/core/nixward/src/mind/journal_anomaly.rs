// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Journal Anomaly Detection via HDC
//!
//! Encodes systemd journal entries into hyperdimensional vectors and detects
//! anomalous patterns by comparing against a learned baseline. Uses an
//! exponential moving average of journal entry vectors to build a "normal"
//! baseline, then flags entries whose HDC similarity to the baseline falls
//! below a configurable threshold.
//!
//! Key techniques:
//! - Token-level encoding of log messages via the NixCodebook
//! - Unit-aware encoding (different units occupy different HDC subspaces)
//! - Priority weighting (errors contribute more to the anomaly signal)
//! - Sliding window baseline for adapting to system changes

use crate::encoding::NixCodebook;
#[cfg(feature = "native")]
use crate::observe::journal::JournalEntry;
use symthaea_core::hdc::ContinuousHV;

/// A detected anomaly in the journal.
#[cfg(feature = "native")]
#[derive(Debug, Clone)]
pub struct JournalAnomaly {
    /// The journal entry that triggered the anomaly.
    pub entry: JournalEntry,
    /// How different this entry is from the baseline: 0.0 = identical,
    /// 0.5 = orthogonal/no correlation, 1.0 = maximally anti-correlated.
    /// See `anomaly_score_from_similarity` for the mapping.
    pub anomaly_score: f32,
    /// The similarity to the baseline (higher = more normal).
    pub baseline_similarity: f32,
    /// Why this was flagged.
    pub reason: String,
}

/// HDC-based journal anomaly detector.
pub struct JournalAnomalyDetector {
    /// HDC codebook for encoding log tokens.
    codebook: NixCodebook,
    /// Baseline "normal" state vector (EMA of recent entries).
    baseline: Option<ContinuousHV>,
    /// EMA smoothing factor (0.0-1.0, higher = more recent bias).
    alpha: f32,
    /// Anomaly threshold: entries with similarity below this are flagged.
    threshold: f32,
    /// Number of entries processed.
    entries_processed: usize,
    /// Dimensionality of the HDC space.
    dim: usize,
}

impl JournalAnomalyDetector {
    /// Create a new detector with default settings.
    pub fn new() -> Self {
        Self {
            codebook: NixCodebook::with_dim(256),
            baseline: None,
            alpha: 0.05,
            threshold: 0.3,
            entries_processed: 0,
            dim: 256,
        }
    }

    /// Set the dimensionality of the detector (re-initializes the codebook with the correct dimension).
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self.codebook = NixCodebook::with_dim(dim);
        self
    }

    /// Set the anomaly threshold (0.0-1.0).
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the EMA alpha (0.0-1.0).
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.01, 1.0);
        self
    }

    /// Encode a single journal entry into an HDC vector.
    ///
    /// The encoding combines:
    /// - Unit identity (permuted by unit name hash)
    /// - Priority level (scaled by severity)
    /// - Message content tokens (bundled)
    #[cfg(feature = "native")]
    pub fn encode_entry(&mut self, entry: &JournalEntry) -> ContinuousHV {
        // Encode the unit name
        let unit_hv = self.codebook.get_or_create(&entry.unit).clone();

        // Encode message tokens
        let tokens: Vec<&str> = entry
            .message
            .split_whitespace()
            .filter(|t| t.len() > 2)
            .take(20) // Cap token count for performance
            .collect();

        let token_hvs: Vec<ContinuousHV> = tokens
            .iter()
            .map(|t| self.codebook.get_or_create(t).clone())
            .collect();

        // Bundle message tokens
        let message_hv = if token_hvs.is_empty() {
            ContinuousHV::random(self.dim, entry.message.len() as u64 + 1)
        } else {
            let refs: Vec<&ContinuousHV> = token_hvs.iter().collect();
            ContinuousHV::bundle(&refs)
        };

        // Combine unit + message (bind operation)
        let combined = unit_hv.bind(&message_hv);

        // Weight by priority (errors get amplified)
        let priority_weight = priority_to_weight(entry.priority);
        combined.scale(priority_weight)
    }

    /// Encode a single journal entry by combining its symbolic unit name
    /// with the offline-safe semantic log embedding from the Neural Bridge.
    #[cfg(feature = "native")]
    pub fn encode_entry_hybrid_offline(
        &mut self,
        entry: &JournalEntry,
        bridge: &super::neural_bridge::NeuralBridge,
    ) -> Result<ContinuousHV, super::neural_bridge::BridgeError> {
        // 1. Encode symbolic unit identity
        let unit_hv = self.codebook.get_or_create(&entry.unit).clone();

        // 2. Encode semantic log message via offline deterministic bridge
        let embed = bridge.embed_deterministic(&entry.message)?;
        let msg_hv = embed.continuous;

        // 3. Combine unit + message (bind)
        let combined = unit_hv.bind(&msg_hv);

        // 4. Scale by priority weight
        let priority_weight = priority_to_weight(entry.priority);
        Ok(combined.scale(priority_weight))
    }

    /// Encode a single journal entry asynchronously by combining its symbolic
    /// unit name with the live semantic log embedding from the Neural Bridge.
    #[cfg(feature = "native")]
    pub async fn encode_entry_hybrid_async(
        &mut self,
        entry: &JournalEntry,
        bridge: &super::neural_bridge::NeuralBridge,
    ) -> Result<ContinuousHV, super::neural_bridge::BridgeError> {
        // 1. Encode symbolic unit identity
        let unit_hv = self.codebook.get_or_create(&entry.unit).clone();

        // 2. Encode semantic log message via live bridge (network)
        let embed = bridge.embed_text(&entry.message).await?;
        let msg_hv = embed.continuous;

        // 3. Combine unit + message (bind)
        let combined = unit_hv.bind(&msg_hv);

        // 4. Scale by priority weight
        let priority_weight = priority_to_weight(entry.priority);
        Ok(combined.scale(priority_weight))
    }

    /// Process a batch of journal entries using offline-safe hybrid encoding,
    /// updating the baseline and returning anomalies.
    #[cfg(feature = "native")]
    pub fn process_entries_hybrid_offline(
        &mut self,
        entries: &[JournalEntry],
        bridge: &super::neural_bridge::NeuralBridge,
    ) -> Result<Vec<JournalAnomaly>, super::neural_bridge::BridgeError> {
        let mut anomalies = Vec::new();

        for entry in entries {
            let hv = self.encode_entry_hybrid_offline(entry, bridge)?;

            // Compare against baseline
            if let Some(ref baseline) = self.baseline {
                let similarity = hv.similarity(baseline);

                if similarity < self.threshold {
                    let anomaly_score = anomaly_score_from_similarity(similarity);
                    let reason = classify_anomaly(entry, similarity);

                    anomalies.push(JournalAnomaly {
                        entry: entry.clone(),
                        anomaly_score,
                        baseline_similarity: similarity,
                        reason,
                    });
                }
            }

            // Update baseline via EMA
            self.update_baseline(&hv);
            self.entries_processed += 1;
        }

        Ok(anomalies)
    }

    /// Process a batch of journal entries asynchronously using live hybrid encoding,
    /// updating the baseline and returning anomalies.
    #[cfg(feature = "native")]
    pub async fn process_entries_hybrid_async(
        &mut self,
        entries: &[JournalEntry],
        bridge: &super::neural_bridge::NeuralBridge,
    ) -> Result<Vec<JournalAnomaly>, super::neural_bridge::BridgeError> {
        let mut anomalies = Vec::new();

        for entry in entries {
            let hv = self.encode_entry_hybrid_async(entry, bridge).await?;

            // Compare against baseline
            if let Some(ref baseline) = self.baseline {
                let similarity = hv.similarity(baseline);

                if similarity < self.threshold {
                    let anomaly_score = anomaly_score_from_similarity(similarity);
                    let reason = classify_anomaly(entry, similarity);

                    anomalies.push(JournalAnomaly {
                        entry: entry.clone(),
                        anomaly_score,
                        baseline_similarity: similarity,
                        reason,
                    });
                }
            }

            // Update baseline via EMA
            self.update_baseline(&hv);
            self.entries_processed += 1;
        }

        Ok(anomalies)
    }

    /// Get the current baseline similarity for a given entry using offline hybrid encoding.
    #[cfg(feature = "native")]
    pub fn score_entry_hybrid_offline(
        &mut self,
        entry: &JournalEntry,
        bridge: &super::neural_bridge::NeuralBridge,
    ) -> Result<f32, super::neural_bridge::BridgeError> {
        let hv = self.encode_entry_hybrid_offline(entry, bridge)?;
        Ok(match &self.baseline {
            Some(baseline) => hv.similarity(baseline),
            None => 0.5, // No baseline yet
        })
    }

    /// Process a batch of journal entries, updating the baseline and returning anomalies.
    #[cfg(feature = "native")]
    pub fn process_entries(&mut self, entries: &[JournalEntry]) -> Vec<JournalAnomaly> {
        let mut anomalies = Vec::new();

        for entry in entries {
            let hv = self.encode_entry(entry);

            // Compare against baseline
            if let Some(ref baseline) = self.baseline {
                let similarity = hv.similarity(baseline);

                if similarity < self.threshold {
                    let anomaly_score = anomaly_score_from_similarity(similarity);
                    let reason = classify_anomaly(entry, similarity);

                    anomalies.push(JournalAnomaly {
                        entry: entry.clone(),
                        anomaly_score,
                        baseline_similarity: similarity,
                        reason,
                    });
                }
            }

            // Update baseline via EMA
            self.update_baseline(&hv);
            self.entries_processed += 1;
        }

        anomalies
    }

    /// Update the baseline using exponential moving average.
    fn update_baseline(&mut self, observation: &ContinuousHV) {
        self.baseline = Some(match &self.baseline {
            None => observation.clone(),
            Some(current) => {
                // EMA: new_baseline = alpha * observation + (1 - alpha) * current
                let scaled_obs = observation.scale(self.alpha);
                let scaled_current = current.scale(1.0 - self.alpha);
                // Add the two scaled vectors
                let refs = [&scaled_obs, &scaled_current];
                ContinuousHV::bundle(&refs)
            }
        });
    }

    /// Get the current baseline similarity for a given entry (without updating).
    #[cfg(feature = "native")]
    pub fn score_entry(&mut self, entry: &JournalEntry) -> f32 {
        let hv = self.encode_entry(entry);
        match &self.baseline {
            Some(baseline) => hv.similarity(baseline),
            None => 0.5, // No baseline yet
        }
    }

    /// Number of entries processed so far.
    pub fn entries_processed(&self) -> usize {
        self.entries_processed
    }

    /// Whether the detector has enough data for meaningful anomaly detection.
    pub fn is_warmed_up(&self) -> bool {
        self.entries_processed >= 10
    }

    /// Get the current baseline vector, if any.
    pub fn baseline(&self) -> Option<&ContinuousHV> {
        self.baseline.as_ref()
    }
}

impl Default for JournalAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Map an HDC cosine similarity (range `[-1, 1]`) to an anomaly-confidence
/// score (range `[0, 1]`), preserving the full similarity range instead of
/// clamping.
///
/// The previous formula (`1.0 - similarity.max(0.0)`) clamped every
/// non-positive similarity to the same `anomaly_score = 1.0` — so a barely
/// negative, essentially-uncorrelated similarity (`sim=-0.02`, i.e. noise)
/// and a genuinely maximally anti-correlated one (`sim=-0.99`) were
/// reported with identical, maximum confidence. Verified live: journal
/// anomalies with `sim=-0.02` were reported at `score: 1.00`. This
/// destroyed exactly the information that distinguishes "weak/uncertain
/// signal" from "strong signal", which is the whole point of a confidence
/// score. See SYMTHAEA_NIXOS_MANAGEMENT_IMPROVEMENT_PLAN_2026-07-26.md
/// "Doctor severity/confidence calibration".
///
/// `similarity=1.0` (identical) -> `0.0` (no anomaly).
/// `similarity=0.0` (orthogonal/uncorrelated) -> `0.5` (genuinely uncertain).
/// `similarity=-1.0` (maximally anti-correlated) -> `1.0` (strong anomaly).
fn anomaly_score_from_similarity(similarity: f32) -> f32 {
    (1.0 - similarity) / 2.0
}

/// Convert syslog priority (0-7) to a weight factor.
/// Lower priority = higher severity = higher weight.
fn priority_to_weight(priority: u8) -> f32 {
    match priority {
        0 => 4.0, // emerg
        1 => 3.5, // alert
        2 => 3.0, // crit
        3 => 2.5, // err
        4 => 1.5, // warning
        5 => 1.0, // notice
        6 => 0.8, // info
        7 => 0.5, // debug
        _ => 1.0,
    }
}

/// Classify why an entry is anomalous based on content and similarity.
#[cfg(feature = "native")]
fn classify_anomaly(entry: &JournalEntry, similarity: f32) -> String {
    let msg_lower = entry.message.to_lowercase();

    if entry.priority <= 3 {
        format!(
            "Error-level log from {} (priority {})",
            entry.unit, entry.priority
        )
    } else if msg_lower.contains("fail") || msg_lower.contains("error") {
        format!("Error keyword in {} log", entry.unit)
    } else if msg_lower.contains("oom") || msg_lower.contains("out of memory") {
        format!("Memory pressure detected from {}", entry.unit)
    } else if msg_lower.contains("segfault")
        || msg_lower.contains("core dumped")
        || msg_lower.contains("dumped core")
    {
        format!("Crash detected in {}", entry.unit)
    } else if msg_lower.contains("timeout") || msg_lower.contains("timed out") {
        format!("Timeout in {}", entry.unit)
    } else if similarity < 0.0 {
        format!(
            "Anti-correlated pattern from {} (sim={:.2})",
            entry.unit, similarity
        )
    } else {
        format!(
            "Unusual pattern from {} (sim={:.2})",
            entry.unit, similarity
        )
    }
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;

    fn make_entry(unit: &str, priority: u8, message: &str) -> JournalEntry {
        JournalEntry {
            timestamp: "Jan 15 10:30:00".to_string(),
            unit: unit.to_string(),
            priority,
            message: message.to_string(),
        }
    }

    #[test]
    fn test_encode_entry() {
        let mut detector = JournalAnomalyDetector::new();
        let entry = make_entry("nginx", 6, "GET /index.html 200 OK");
        let hv = detector.encode_entry(&entry);
        assert_eq!(hv.as_slice().len(), 256);
    }

    #[test]
    fn test_same_entry_same_encoding() {
        let mut det1 = JournalAnomalyDetector::new();
        let mut det2 = JournalAnomalyDetector::new();
        let entry = make_entry("sshd", 6, "Accepted publickey for user");
        let hv1 = det1.encode_entry(&entry);
        let hv2 = det2.encode_entry(&entry);
        let sim = hv1.similarity(&hv2);
        assert!(
            (sim - 1.0).abs() < 1e-4,
            "Same entry should produce identical encodings, got {}",
            sim
        );
    }

    #[test]
    fn test_different_units_different_encoding() {
        let mut detector = JournalAnomalyDetector::new();
        let entry_a = make_entry("nginx", 6, "request received");
        let entry_b = make_entry("postgresql", 6, "request received");
        let hv_a = detector.encode_entry(&entry_a);
        let hv_b = detector.encode_entry(&entry_b);
        let sim = hv_a.similarity(&hv_b);
        // Same message, different unit → should have moderate similarity (not identical)
        assert!(
            sim < 0.9,
            "Different units should differ, got similarity {}",
            sim
        );
    }

    #[test]
    fn test_warmup() {
        let mut detector = JournalAnomalyDetector::new();
        assert!(!detector.is_warmed_up());

        let normal_entries: Vec<JournalEntry> = (0..15)
            .map(|i| make_entry("nginx", 6, &format!("GET /page/{} 200 OK", i)))
            .collect();

        detector.process_entries(&normal_entries);
        assert!(detector.is_warmed_up());
        assert_eq!(detector.entries_processed(), 15);
    }

    #[test]
    fn test_normal_entries_no_anomalies() {
        let mut detector = JournalAnomalyDetector::new().with_threshold(0.1); // Low threshold = only flag very unusual

        // Train on normal entries
        let normal: Vec<JournalEntry> = (0..20)
            .map(|i| {
                make_entry(
                    "nginx",
                    6,
                    &format!("GET /page/{} 200 OK processed request", i),
                )
            })
            .collect();
        let _ = detector.process_entries(&normal);

        // Process more of the same — should see few/no anomalies
        let more_normal: Vec<JournalEntry> = (20..25)
            .map(|i| {
                make_entry(
                    "nginx",
                    6,
                    &format!("GET /page/{} 200 OK processed request", i),
                )
            })
            .collect();
        let anomalies = detector.process_entries(&more_normal);
        // With similar entries, anomaly count should be low
        assert!(
            anomalies.len() <= 2,
            "Expected few anomalies for normal entries, got {}",
            anomalies.len()
        );
    }

    #[test]
    fn test_error_entry_flagged() {
        let mut detector = JournalAnomalyDetector::new().with_threshold(0.4);

        // Train baseline on normal info logs
        let normal: Vec<JournalEntry> = (0..20)
            .map(|i| {
                make_entry(
                    "nginx",
                    6,
                    &format!("GET /page/{} 200 OK request processed", i),
                )
            })
            .collect();
        let _ = detector.process_entries(&normal);

        // Now send a very different error entry
        let error_entry = make_entry(
            "postgresql",
            3,
            "FATAL: could not open relation mapping file global pg_filenode.map: No such file",
        );
        let anomalies = detector.process_entries(&[error_entry]);
        // This should be flagged because it's error-priority and from a different unit
        assert!(
            !anomalies.is_empty(),
            "Error entry should be flagged as anomaly"
        );
        assert!(anomalies[0].reason.contains("Error"));
    }

    #[test]
    fn test_priority_weighting() {
        assert_eq!(priority_to_weight(0), 4.0);
        assert_eq!(priority_to_weight(3), 2.5);
        assert_eq!(priority_to_weight(6), 0.8);
        assert_eq!(priority_to_weight(7), 0.5);
    }

    #[test]
    fn test_classify_anomaly() {
        let entry = make_entry("nginx", 3, "connection refused");
        assert!(classify_anomaly(&entry, 0.1).contains("Error-level"));

        let entry = make_entry("app", 6, "request failed with timeout");
        assert!(classify_anomaly(&entry, 0.1).contains("Error keyword"));

        let entry = make_entry("kernel", 4, "Out of memory: Killed process");
        assert!(classify_anomaly(&entry, 0.1).contains("Memory pressure"));

        let entry = make_entry("app", 6, "something totally normal happened");
        assert!(classify_anomaly(&entry, 0.2).contains("Unusual pattern"));
    }

    #[test]
    fn test_anomaly_score_from_similarity_preserves_full_range() {
        // Regression test: the old formula (1.0 - similarity.max(0.0))
        // clamped every non-positive similarity to anomaly_score=1.0,
        // reporting a barely-negative (essentially uncorrelated) similarity
        // and a genuinely maximally anti-correlated one as identically
        // "maximum confidence". Verified live: sim=-0.02 was reported at
        // score=1.00. The fixed formula must distinguish them.
        assert!((anomaly_score_from_similarity(1.0) - 0.0).abs() < 1e-6);
        assert!((anomaly_score_from_similarity(0.0) - 0.5).abs() < 1e-6);
        assert!((anomaly_score_from_similarity(-1.0) - 1.0).abs() < 1e-6);

        let near_zero_score = anomaly_score_from_similarity(-0.02);
        let maximally_anti_correlated_score = anomaly_score_from_similarity(-0.99);
        assert!(
            near_zero_score < maximally_anti_correlated_score,
            "a barely-negative (noise-level) similarity must score strictly \
             lower confidence than a genuinely maximal anti-correlation, \
             got {near_zero_score} vs {maximally_anti_correlated_score}"
        );
        assert!(
            near_zero_score < 0.6,
            "sim=-0.02 (essentially uncorrelated) should not read as \
             high-confidence anomaly, got {near_zero_score}"
        );
    }

    #[test]
    fn test_score_entry() {
        let mut detector = JournalAnomalyDetector::new();

        // No baseline yet — should return 0.5
        let entry = make_entry("nginx", 6, "hello world");
        assert!((detector.score_entry(&entry) - 0.5).abs() < 0.01);

        // Train baseline
        let normal: Vec<JournalEntry> = (0..10)
            .map(|i| make_entry("nginx", 6, &format!("GET /page/{} 200 OK", i)))
            .collect();
        let _ = detector.process_entries(&normal);

        // Score a similar entry — should be positive
        let similar = make_entry("nginx", 6, "GET /page/99 200 OK");
        let score = detector.score_entry(&similar);
        assert!(
            score > 0.0,
            "Similar entry should have positive similarity, got {}",
            score
        );
    }

    #[test]
    fn test_hybrid_encode_entry_determinism() {
        let bridge = crate::mind::neural_bridge::NeuralBridge::new();
        // Run at 16,384-D
        let mut detector =
            JournalAnomalyDetector::new().with_dim(symthaea_core::hdc::HDC_DIMENSION);
        let entry = make_entry("nginx", 6, "system initialization completed");

        let hv1 = detector
            .encode_entry_hybrid_offline(&entry, &bridge)
            .unwrap();
        let hv2 = detector
            .encode_entry_hybrid_offline(&entry, &bridge)
            .unwrap();

        let sim = hv1.similarity(&hv2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Hybrid log encoding must be deterministic: sim={sim}"
        );
        assert_eq!(hv1.as_slice().len(), symthaea_core::hdc::HDC_DIMENSION);
    }

    #[test]
    fn test_hybrid_anomaly_detection_workflow() {
        let bridge = crate::mind::neural_bridge::NeuralBridge::new();
        let dim = symthaea_core::hdc::HDC_DIMENSION;
        let mut detector = JournalAnomalyDetector::new()
            .with_dim(dim)
            .with_threshold(0.3);

        // Train baseline on normal logs
        let normal_logs = [
            make_entry("nginx", 6, "GET /status 200 OK"),
            make_entry("nginx", 6, "GET /index.html 200 OK"),
            make_entry("nginx", 6, "GET /api/v1/health 200 OK"),
        ];
        let _ = detector
            .process_entries_hybrid_offline(&normal_logs, &bridge)
            .unwrap();

        // Process a very anomalous/different error log
        let err_log = make_entry(
            "sshd",
            3,
            "Failed password for invalid user admin from 192.168.1.100 port 50430",
        );
        let anomalies = detector
            .process_entries_hybrid_offline(&[err_log], &bridge)
            .unwrap();

        assert!(
            !anomalies.is_empty(),
            "Anomalous/error log must be flagged as anomaly"
        );
        assert!(anomalies[0].reason.contains("Error"));
    }
}
