// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Decay Garden - Wealth Circulation Mechanism
//!
//! Implementation of MIP-E-002 Article VI: Wealth Half-Life
//!
//! The Decay Garden is a cultural reframing of circulation tax as "composting":
//! dormant value naturally returns to commons like autumn leaves becoming soil.

use serde::{Deserialize, Serialize};

/// Decay Garden configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayGarden {
    /// Annual decay rate (default 2%)
    pub annual_decay_rate: f64,
    /// Dormancy threshold in seconds (default 180 days)
    pub dormancy_threshold: u64,
    /// Minimum balance exempt from decay
    pub minimum_exempt: u64,
    /// Compost distribution ratios
    pub distribution: CompostDistribution,
}

impl Default for DecayGarden {
    fn default() -> Self {
        Self {
            annual_decay_rate: 0.02,                // 2%
            dormancy_threshold: 180 * 24 * 60 * 60, // 180 days
            minimum_exempt: 10_000,                 // Protect small holders
            distribution: CompostDistribution::default(),
        }
    }
}

/// How composted value is distributed
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CompostDistribution {
    /// Ratio to member's Local DAO HEARTH
    pub local_hearth_ratio: f64,
    /// Ratio to Regional DAO HEARTH
    pub regional_hearth_ratio: f64,
    /// Ratio to Global Commons Fund
    pub global_commons_ratio: f64,
}

impl Default for CompostDistribution {
    fn default() -> Self {
        Self {
            local_hearth_ratio: 0.50,    // 50%
            regional_hearth_ratio: 0.30, // 30%
            global_commons_ratio: 0.20,  // 20%
        }
    }
}

impl CompostDistribution {
    /// Validate distribution ratios sum to 1.0
    pub fn is_valid(&self) -> bool {
        let sum = self.local_hearth_ratio + self.regional_hearth_ratio + self.global_commons_ratio;
        (sum - 1.0).abs() < 0.001
    }

    /// Calculate distribution amounts
    pub fn distribute(&self, total: u64) -> CompostAllocation {
        CompostAllocation {
            local_hearth: (total as f64 * self.local_hearth_ratio) as u64,
            regional_hearth: (total as f64 * self.regional_hearth_ratio) as u64,
            global_commons: (total as f64 * self.global_commons_ratio) as u64,
        }
    }
}

/// Allocation of composted value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompostAllocation {
    /// Amount to local HEARTH
    pub local_hearth: u64,
    /// Amount to regional HEARTH
    pub regional_hearth: u64,
    /// Amount to global commons
    pub global_commons: u64,
}

impl CompostAllocation {
    /// Total composted
    pub fn total(&self) -> u64 {
        self.local_hearth + self.regional_hearth + self.global_commons
    }
}

/// Check for member dormancy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DormancyCheck {
    /// Member DID
    pub member_did: String,
    /// SAP balance subject to decay
    pub eligible_balance: u64,
    /// Days dormant
    pub days_dormant: u64,
    /// Is dormant (past threshold)
    pub is_dormant: bool,
    /// Amount subject to decay
    pub decay_amount: u64,
    /// Visual stage (for UI)
    pub visual_stage: DecayStage,
}

/// Visual decay stage for UI
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecayStage {
    /// Active - green/healthy
    Active,
    /// Cooling - 90+ days, yellow tint
    Cooling,
    /// Dormant - 180+ days, orange/amber
    Dormant,
    /// Composting - actively decaying, brown
    Composting,
}

impl DecayStage {
    /// Get stage from dormancy days
    pub fn from_days(days: u64) -> Self {
        match days {
            d if d < 90 => DecayStage::Active,
            d if d < 180 => DecayStage::Cooling,
            d if d < 270 => DecayStage::Dormant,
            _ => DecayStage::Composting,
        }
    }

    /// Get color for UI rendering
    pub fn color_hex(&self) -> &'static str {
        match self {
            DecayStage::Active => "#4CAF50",     // Green
            DecayStage::Cooling => "#FFC107",    // Amber
            DecayStage::Dormant => "#FF9800",    // Orange
            DecayStage::Composting => "#795548", // Brown
        }
    }

    /// Get description for UI
    pub fn description(&self) -> &'static str {
        match self {
            DecayStage::Active => "Active - Healthy circulation",
            DecayStage::Cooling => "Cooling - Consider activity soon",
            DecayStage::Dormant => "Dormant - Decay begins next epoch",
            DecayStage::Composting => "Composting - Nourishing the commons",
        }
    }
}

/// Compost event for audit/gratitude ledger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompostEvent {
    /// Member whose value composted
    pub member_did: String,
    /// Total amount composted
    pub amount: u64,
    /// Distribution breakdown
    pub allocation: CompostAllocation,
    /// Epoch in which composting occurred
    pub epoch: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Gratitude entry (for Gratitude Ledger)
    pub gratitude_entry: String,
}

impl DecayGarden {
    /// Check member's dormancy status
    pub fn check_dormancy(
        &self,
        member_did: &str,
        sap_balance: u64,
        last_activity: u64,
        current_time: u64,
    ) -> DormancyCheck {
        let seconds_dormant = current_time.saturating_sub(last_activity);
        let days_dormant = seconds_dormant / (24 * 60 * 60);
        let is_dormant = seconds_dormant > self.dormancy_threshold;

        // Calculate eligible balance (above minimum)
        let eligible_balance = sap_balance.saturating_sub(self.minimum_exempt);

        // Calculate decay if dormant
        let decay_amount = if is_dormant && eligible_balance > 0 {
            // Pro-rate annual decay for elapsed period
            let years_dormant =
                (seconds_dormant - self.dormancy_threshold) as f64 / (365.25 * 24.0 * 60.0 * 60.0);
            let decay_factor = 1.0 - (1.0 - self.annual_decay_rate).powf(years_dormant);
            (eligible_balance as f64 * decay_factor) as u64
        } else {
            0
        };

        DormancyCheck {
            member_did: member_did.to_string(),
            eligible_balance,
            days_dormant,
            is_dormant,
            decay_amount,
            visual_stage: DecayStage::from_days(days_dormant),
        }
    }

    /// Execute composting for a dormant member
    pub fn compost(
        &self,
        check: &DormancyCheck,
        _local_dao_id: &str,
        _regional_dao_id: &str,
        epoch: u64,
        timestamp: u64,
    ) -> Option<CompostEvent> {
        if check.decay_amount == 0 {
            return None;
        }

        let allocation = self.distribution.distribute(check.decay_amount);

        let gratitude_entry = format!(
            "🍂 COMPOST: {} SAP from {} nourishes the commons. \
             Local HEARTH: {}, Regional: {}, Global: {}",
            check.decay_amount,
            check.member_did,
            allocation.local_hearth,
            allocation.regional_hearth,
            allocation.global_commons,
        );

        Some(CompostEvent {
            member_did: check.member_did.clone(),
            amount: check.decay_amount,
            allocation,
            epoch,
            timestamp,
            gratitude_entry,
        })
    }

    /// Calculate decay preview for UI (what would decay at current trajectory)
    pub fn decay_preview(&self, sap_balance: u64, days_inactive: u64) -> DecayPreview {
        let eligible = sap_balance.saturating_sub(self.minimum_exempt);

        // Project decay at 1 year dormancy
        let annual_decay = (eligible as f64 * self.annual_decay_rate) as u64;

        // Current decay if dormant now
        let current_decay = if days_inactive > 180 {
            let extra_days = days_inactive - 180;
            let years = extra_days as f64 / 365.25;
            (eligible as f64 * self.annual_decay_rate * years) as u64
        } else {
            0
        };

        DecayPreview {
            eligible_balance: eligible,
            days_until_decay: 180u64.saturating_sub(days_inactive),
            annual_decay_if_dormant: annual_decay,
            current_decay_amount: current_decay,
            exempt_amount: self.minimum_exempt,
        }
    }

    /// Activities that reset dormancy clock
    pub fn dormancy_reset_activities() -> Vec<&'static str> {
        vec![
            "Any SAP/FLOW transaction",
            "Governance vote (MIP, DAO proposal)",
            "HEARTH warming or tending",
            "CGC/SPORE gifting (sending)",
            "Validator participation",
            "Profile update with attestation",
        ]
    }
}

/// Decay preview for member dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayPreview {
    /// Balance subject to decay
    pub eligible_balance: u64,
    /// Days until decay starts
    pub days_until_decay: u64,
    /// Annual decay if fully dormant
    pub annual_decay_if_dormant: u64,
    /// Current decay amount (if already dormant)
    pub current_decay_amount: u64,
    /// Amount protected from decay
    pub exempt_amount: u64,
}

/// Rotation detection (Sybil attack on decay)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationDetection {
    /// Member DID
    pub member_did: String,
    /// Suspected rotation pattern detected
    pub rotation_detected: bool,
    /// Related addresses in rotation cluster
    pub cluster_addresses: Vec<String>,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Recommendation
    pub recommendation: RotationRecommendation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Recommendation for credential rotation action.
pub enum RotationRecommendation {
    /// No action
    Clear,
    /// Monitor for patterns
    Monitor,
    /// Merge dormancy clocks
    MergeClocks,
    /// CIV penalty for evasion
    Penalize,
}

/// Detect rotation patterns (transfers between addresses to reset dormancy)
pub fn detect_rotation(
    member_did: &str,
    recent_transfers: &[(String, String, u64)], // (from, to, amount)
    _time_window_days: u64,
) -> RotationDetection {
    // Simplified detection - full implementation uses MATL graph analysis
    let mut cluster = Vec::new();
    let mut rotation_amount = 0u64;

    for (from, to, amount) in recent_transfers {
        if from == member_did || to == member_did {
            // Check for round-trip patterns
            let reverse_exists = recent_transfers
                .iter()
                .any(|(f, t, _)| f == to && t == from);
            if reverse_exists {
                rotation_amount += amount;
                if !cluster.contains(to) {
                    cluster.push(to.clone());
                }
            }
        }
    }

    let rotation_detected = !cluster.is_empty();
    let confidence = if rotation_detected {
        (rotation_amount as f64 / 10_000.0).min(0.9)
    } else {
        0.0
    };

    let recommendation = if confidence > 0.7 {
        RotationRecommendation::Penalize
    } else if confidence > 0.4 {
        RotationRecommendation::MergeClocks
    } else if rotation_detected {
        RotationRecommendation::Monitor
    } else {
        RotationRecommendation::Clear
    };

    RotationDetection {
        member_did: member_did.to_string(),
        rotation_detected,
        cluster_addresses: cluster,
        confidence,
        recommendation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dormancy_check_active() {
        let garden = DecayGarden::default();
        let now = 100_000_000u64; // Large enough timestamp
        let last_activity = now - (100 * 24 * 60 * 60); // 100 days ago

        let check = garden.check_dormancy("did:test:1", 50_000, last_activity, now);

        assert!(!check.is_dormant);
        assert_eq!(check.decay_amount, 0);
        assert_eq!(check.visual_stage, DecayStage::Cooling);
    }

    #[test]
    fn test_dormancy_check_dormant() {
        let garden = DecayGarden::default();
        let now = 100_000_000u64; // Large enough timestamp
        let last_activity = now - (365 * 24 * 60 * 60); // 1 year ago

        let check = garden.check_dormancy("did:test:1", 50_000, last_activity, now);

        assert!(check.is_dormant);
        assert!(check.decay_amount > 0);
        assert_eq!(check.visual_stage, DecayStage::Composting);

        // Eligible: 50,000 - 10,000 = 40,000
        // ~185 days past threshold, ~0.5 years
        // Decay: 40,000 * 0.02 * 0.5 ≈ 400
        assert!(check.decay_amount > 200 && check.decay_amount < 1000);
    }

    #[test]
    fn test_minimum_exempt() {
        let garden = DecayGarden::default();
        let now = 100_000_000u64; // Large enough timestamp
        let last_activity = now - (365 * 24 * 60 * 60); // 1 year ago

        // Balance below minimum - no decay
        let check = garden.check_dormancy("did:test:1", 5_000, last_activity, now);
        assert_eq!(check.eligible_balance, 0);
        assert_eq!(check.decay_amount, 0);
    }

    #[test]
    fn test_compost_distribution() {
        let dist = CompostDistribution::default();
        assert!(dist.is_valid());

        let allocation = dist.distribute(10_000);
        assert_eq!(allocation.local_hearth, 5_000);
        assert_eq!(allocation.regional_hearth, 3_000);
        assert_eq!(allocation.global_commons, 2_000);
    }

    #[test]
    fn test_decay_stages() {
        assert_eq!(DecayStage::from_days(30), DecayStage::Active);
        assert_eq!(DecayStage::from_days(120), DecayStage::Cooling);
        assert_eq!(DecayStage::from_days(200), DecayStage::Dormant);
        assert_eq!(DecayStage::from_days(300), DecayStage::Composting);
    }

    #[test]
    fn test_rotation_detection() {
        let transfers = vec![
            ("did:a".to_string(), "did:b".to_string(), 5000),
            ("did:b".to_string(), "did:a".to_string(), 4500), // Round trip!
        ];

        let detection = detect_rotation("did:a", &transfers, 30);
        assert!(detection.rotation_detected);
        assert!(detection.confidence > 0.0);
    }
}
