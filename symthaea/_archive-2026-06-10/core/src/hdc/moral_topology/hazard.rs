// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Hazard signature registry and escalation policy.

use super::*;

// ═══════════════════════════════════════════════════════════════════════════════
// Escalation Policy
// ═══════════════════════════════════════════════════════════════════════════════

/// Concrete response action for a given severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum EscalationLevel {
    /// Normal operation: log convergence metrics for telemetry.
    #[default]
    Log,
    /// Elevated concern: emit a warning visible in dashboards.
    Warn,
    /// Active mitigation: reduce learning rate, increase safety margin.
    Throttle,
    /// Critical: refuse to continue processing until human review.
    Block,
}

/// Maps convergence severity ranges to concrete response actions with cooldowns.
///
/// Prevents response oscillation: once an escalation fires, it holds for
/// `cooldown_cycles` before it can de-escalate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationPolicy {
    /// Severity thresholds for each level (exclusive lower bounds).
    /// Default: Log < 0.3, Warn < 0.5, Throttle < 0.7, Block >= 0.7.
    pub warn_threshold: f64,
    pub throttle_threshold: f64,
    pub block_threshold: f64,
    /// Minimum cycles before de-escalation is permitted.
    pub cooldown_cycles: u64,
    /// Current escalation level.
    pub(crate) current_level: EscalationLevel,
    /// Cycles remaining in current cooldown (0 = can change).
    pub(crate) cooldown_remaining: u64,
}

impl Default for EscalationPolicy {
    fn default() -> Self {
        Self {
            warn_threshold: 0.3,
            throttle_threshold: 0.5,
            block_threshold: 0.7,
            cooldown_cycles: 10,
            current_level: EscalationLevel::Log,
            cooldown_remaining: 0,
        }
    }
}

impl EscalationPolicy {
    /// Compute the target escalation level for a given severity.
    fn target_level(&self, severity: f64) -> EscalationLevel {
        if severity >= self.block_threshold {
            EscalationLevel::Block
        } else if severity >= self.throttle_threshold {
            EscalationLevel::Throttle
        } else if severity >= self.warn_threshold {
            EscalationLevel::Warn
        } else {
            EscalationLevel::Log
        }
    }

    /// Update the escalation state based on current severity.
    ///
    /// - **Escalation** (severity rises): immediate, resets cooldown.
    /// - **De-escalation** (severity falls): only after cooldown expires.
    ///
    /// Returns the new effective escalation level.
    pub fn update(&mut self, severity: f64) -> EscalationLevel {
        let target = self.target_level(severity);

        // Tick cooldown
        self.cooldown_remaining = self.cooldown_remaining.saturating_sub(1);

        let current_rank = self.current_level as u8;
        let target_rank = target as u8;

        if target_rank > current_rank {
            // Escalate immediately
            self.current_level = target;
            self.cooldown_remaining = self.cooldown_cycles;
        } else if target_rank < current_rank && self.cooldown_remaining == 0 {
            // De-escalate only after cooldown
            self.current_level = target;
            self.cooldown_remaining = self.cooldown_cycles;
        }

        self.current_level
    }

    /// Current escalation level.
    pub fn current_level(&self) -> EscalationLevel {
        self.current_level
    }

    /// Cycles remaining before de-escalation is allowed.
    pub fn cooldown_remaining(&self) -> u64 {
        self.cooldown_remaining
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Escalation Audit Log
// ═══════════════════════════════════════════════════════════════════════════════

/// Immutable record of an escalation state transition.
///
/// Appended to the audit log every time the escalation level changes or a
/// convergence detection fires. Provides forensic evidence for governance
/// reviews and post-incident debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationAuditEntry {
    /// Monotonic sequence number (unique across the session).
    pub sequence: u64,
    /// Cycle count at which this entry was recorded.
    pub cycle: u64,
    /// Previous escalation level (before this transition).
    pub from_level: EscalationLevel,
    /// New escalation level (after this transition).
    pub to_level: EscalationLevel,
    /// Raw severity that triggered this transition.
    pub severity: f64,
    /// Calibrated severity.
    pub calibrated_severity: f64,
    /// Which of the 4 signals were triggered.
    pub signals_triggered: [bool; 4],
    /// Signal values: [similarity_anomaly, entropy_decline, flourishing_deficit, spectral_gap_decline].
    pub signal_values: [f64; 4],
    /// Matched hazard template, if any.
    pub matched_hazard: Option<String>,
    /// Fingerprint velocity at time of event.
    pub fingerprint_velocity: f64,
    /// Persistence diagram Wasserstein distance.
    pub persistence_distance: f64,
    /// Scenario IDs (monotonic counters) of the requests in the recent window
    /// at the time of this event. Used to trace which inputs contributed.
    pub window_scenario_ids: Vec<u64>,
    /// BLAKE3 hash of the serialized entry (excluding this field) for tamper evidence.
    #[serde(default)]
    pub integrity_hash: String,
}

impl EscalationAuditEntry {
    /// Compute the BLAKE3 integrity hash for this entry.
    ///
    /// Hashes all fields except `integrity_hash` itself.
    fn compute_hash(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.sequence.to_le_bytes());
        hasher.update(&self.cycle.to_le_bytes());
        hasher.update(&[self.from_level as u8, self.to_level as u8]);
        hasher.update(&self.severity.to_le_bytes());
        hasher.update(&self.calibrated_severity.to_le_bytes());
        for &s in &self.signals_triggered {
            hasher.update(&[s as u8]);
        }
        for &v in &self.signal_values {
            hasher.update(&v.to_le_bytes());
        }
        if let Some(ref h) = self.matched_hazard {
            hasher.update(h.as_bytes());
        }
        hasher.update(&self.fingerprint_velocity.to_le_bytes());
        hasher.update(&self.persistence_distance.to_le_bytes());
        for &id in &self.window_scenario_ids {
            hasher.update(&id.to_le_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }

    /// Seal this entry with its integrity hash.
    pub fn seal(&mut self) {
        self.integrity_hash = self.compute_hash();
    }

    /// Verify the integrity hash.
    pub fn verify(&self) -> bool {
        self.integrity_hash == self.compute_hash()
    }
}

/// Append-only audit log for escalation events.
///
/// Bounded to `max_entries` to prevent unbounded memory growth.
/// Oldest entries are evicted when the log is full, but they should
/// have been persisted to disk via snapshot before eviction.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EscalationAuditLog {
    pub(super) entries: VecDeque<EscalationAuditEntry>,
    max_entries: usize,
    next_sequence: u64,
}

impl EscalationAuditLog {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries,
            next_sequence: 0,
        }
    }

    /// Append a new audit entry. Seals it with BLAKE3 before insertion.
    pub fn append(&mut self, mut entry: EscalationAuditEntry) {
        entry.sequence = self.next_sequence;
        self.next_sequence += 1;
        entry.seal();
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// All entries in chronological order.
    pub fn entries(&self) -> &VecDeque<EscalationAuditEntry> {
        &self.entries
    }

    /// Number of entries in the log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Most recent entry, if any.
    pub fn last(&self) -> Option<&EscalationAuditEntry> {
        self.entries.back()
    }

    /// Verify integrity of all entries in the log.
    ///
    /// Returns the index of the first tampered entry, or None if all are valid.
    pub fn verify_integrity(&self) -> Option<usize> {
        for (i, entry) in self.entries.iter().enumerate() {
            if !entry.verify() {
                return Some(i);
            }
        }
        None
    }

    /// Entries since a given sequence number (for incremental export).
    pub fn entries_since(&self, since_sequence: u64) -> Vec<&EscalationAuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.sequence >= since_sequence)
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Hazard Signature Templates
// ═══════════════════════════════════════════════════════════════════════════════

/// A known hazardous convergence pattern in 8D harmony space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HazardSignature {
    pub name: String,
    pub centroid: [f64; N_HARMONIES],
    pub radius: f64,
    pub severity_boost: f64,
}

/// Registry of known hazard signature templates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HazardSignatureRegistry {
    pub(crate) signatures: Vec<HazardSignature>,
}

impl HazardSignatureRegistry {
    /// Create a registry with built-in hazard templates.
    pub fn with_defaults() -> Self {
        Self {
            signatures: vec![
                HazardSignature {
                    name: "weaponization".into(),
                    centroid: [0.2, -0.6, -0.5, -0.2, -0.5, -0.1, 0.1, 0.0],
                    radius: 0.6,
                    severity_boost: 0.3,
                },
                HazardSignature {
                    name: "coercive_control".into(),
                    centroid: [0.0, -0.2, 0.0, -0.5, -0.7, 0.3, -0.6, 0.0],
                    radius: 0.5,
                    severity_boost: 0.25,
                },
                HazardSignature {
                    name: "ecological_destruction".into(),
                    centroid: [-0.5, -0.3, -0.7, 0.2, 0.0, -0.3, -0.1, 0.0],
                    radius: 0.55,
                    severity_boost: 0.2,
                },
                HazardSignature {
                    name: "surveillance_state".into(),
                    centroid: [0.1, -0.1, 0.1, -0.3, -0.6, 0.5, -0.7, 0.0],
                    radius: 0.5,
                    severity_boost: 0.25,
                },
            ],
        }
    }

    /// Add a custom hazard signature.
    pub fn add(&mut self, sig: HazardSignature) {
        self.signatures.push(sig);
    }

    /// Check if a trajectory centroid matches any hazard template.
    pub fn match_trajectory(&self, centroid: &[f64; N_HARMONIES]) -> (Option<&str>, f64) {
        let mut best_name: Option<&str> = None;
        let mut best_boost = 0.0_f64;
        for sig in &self.signatures {
            let dist_sq: f64 = centroid
                .iter()
                .zip(sig.centroid.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let dist = dist_sq.sqrt();
            if dist <= sig.radius && sig.severity_boost > best_boost {
                best_name = Some(&sig.name);
                best_boost = sig.severity_boost;
            }
        }
        (best_name, best_boost)
    }
}
