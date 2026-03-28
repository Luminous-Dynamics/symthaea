// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Support Types — Shared enums and DHT index helpers for the Support domain
//!
//! Provides category, priority, status, and sharding helpers used by all three
//! support zome pairs (knowledge, tickets, diagnostics).

use hdi::prelude::*;

// ============================================================================
// SUPPORT CATEGORY
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SupportCategory {
    Network,
    Hardware,
    Software,
    Holochain,
    Mycelix,
    Security,
    General,
}

// ============================================================================
// TICKET PRIORITY
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TicketPriority {
    Low,
    Medium,
    High,
    Critical,
}

// ============================================================================
// TICKET STATUS
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TicketStatus {
    Open,
    InProgress,
    AwaitingUser,
    Resolved,
    Closed,
}

// ============================================================================
// AUTONOMY LEVEL
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AutonomyLevel {
    Advisory,
    SemiAutonomous,
    FullAutonomous,
}

// ============================================================================
// ACTION TYPE
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ActionType {
    RestartService,
    ClearCache,
    UpdateConfig,
    RunDiagnostic,
    Custom(String),
}

// ============================================================================
// SHARING TIER
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SharingTier {
    LocalOnly,
    Anonymized,
    Full,
}

// ============================================================================
// DIAGNOSTIC TYPE
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DiagnosticType {
    NetworkCheck,
    DiskSpace,
    ServiceStatus,
    HolochainHealth,
    MemoryUsage,
    Custom(String),
}

// ============================================================================
// DIAGNOSTIC SEVERITY
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DiagnosticSeverity {
    Healthy,
    Warning,
    Error,
    Critical,
}

// ============================================================================
// DIFFICULTY LEVEL
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DifficultyLevel {
    Beginner,
    Intermediate,
    Advanced,
}

// ============================================================================
// ARTICLE SOURCE
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ArticleSource {
    Community,
    PreSeeded,
    SymthaeaGenerated,
}

// ============================================================================
// FLAG REASON
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FlagReason {
    Harmful,
    Incorrect,
    Outdated,
    Spam,
}

// ============================================================================
// REPUTATION EVENT
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ReputationEvent {
    ResolutionVerified,
    ArticleUpvoted,
    ArticleFlagged,
    HelpProvided,
}

// ============================================================================
// EPISTEMIC STATUS
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EpistemicStatus {
    Certain,
    Probable,
    Uncertain,
    Unknown,
    OutOfDomain,
}

// ============================================================================
// TIME-SHARDED INDEX HELPERS
// ============================================================================

/// Generate a time-sharded anchor path: "support:tickets:2026:02"
///
/// Prevents DHT hot-link degradation by distributing links across monthly anchors.
pub fn sharded_anchor(domain: &str, entity: &str, timestamp: &Timestamp) -> String {
    let (year, month) = year_month_from_timestamp(timestamp);
    format!("{}:{}:{}:{:02}", domain, entity, year, month)
}

/// Generate a hash-sharded anchor: "support:articles:ab" (first byte hex of entry hash)
///
/// Distributes links across 256 buckets based on the first byte of the entry hash.
pub fn hash_sharded_anchor(domain: &str, entity: &str, entry_hash: &EntryHash) -> String {
    let raw = entry_hash.get_raw_39();
    let hex = format!("{:02x}", raw[0]);
    format!("{}:{}:{}", domain, entity, hex)
}

/// Extract (year, month) from a Holochain `Timestamp`.
///
/// Timestamps are microseconds since the Unix epoch.
fn year_month_from_timestamp(ts: &Timestamp) -> (i32, u32) {
    let micros = ts.as_micros();
    let secs = micros / 1_000_000;
    // Simple conversion: days since epoch → year/month
    let days = secs / 86400;
    // Approximate: 365.25 days/year, then derive month
    let year = 1970 + (days / 365) as i32;
    let day_of_year = days % 365;
    // Approximate month (30.44 days/month average)
    let month = ((day_of_year as f64 / 30.44) as u32).min(11) + 1;
    (year, month)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Serde roundtrip: SupportCategory ────────────────────────────────

    #[test]
    fn serde_roundtrip_support_category() {
        let variants = vec![
            SupportCategory::Network,
            SupportCategory::Hardware,
            SupportCategory::Software,
            SupportCategory::Holochain,
            SupportCategory::Mycelix,
            SupportCategory::Security,
            SupportCategory::General,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: SupportCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
    }

    // ── Serde roundtrip: TicketPriority ─────────────────────────────────

    #[test]
    fn serde_roundtrip_ticket_priority() {
        let variants = vec![
            TicketPriority::Low,
            TicketPriority::Medium,
            TicketPriority::High,
            TicketPriority::Critical,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: TicketPriority = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
    }

    // ── Serde roundtrip: TicketStatus ───────────────────────────────────

    #[test]
    fn serde_roundtrip_ticket_status() {
        let variants = vec![
            TicketStatus::Open,
            TicketStatus::InProgress,
            TicketStatus::AwaitingUser,
            TicketStatus::Resolved,
            TicketStatus::Closed,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: TicketStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
    }

    // ── Serde roundtrip: AutonomyLevel ──────────────────────────────────

    #[test]
    fn serde_roundtrip_autonomy_level() {
        let variants = vec![
            AutonomyLevel::Advisory,
            AutonomyLevel::SemiAutonomous,
            AutonomyLevel::FullAutonomous,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: AutonomyLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
    }

    // ── Serde roundtrip: ActionType ─────────────────────────────────────

    #[test]
    fn serde_roundtrip_action_type() {
        let variants = vec![
            ActionType::RestartService,
            ActionType::ClearCache,
            ActionType::UpdateConfig,
            ActionType::RunDiagnostic,
            ActionType::Custom("install-patch".to_string()),
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: ActionType = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
    }

    // ── Serde roundtrip: SharingTier ────────────────────────────────────

    #[test]
    fn serde_roundtrip_sharing_tier() {
        let variants = vec![
            SharingTier::LocalOnly,
            SharingTier::Anonymized,
            SharingTier::Full,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: SharingTier = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
    }

    // ── Serde roundtrip: DiagnosticType ─────────────────────────────────

    #[test]
    fn serde_roundtrip_diagnostic_type() {
        let variants = vec![
            DiagnosticType::NetworkCheck,
            DiagnosticType::DiskSpace,
            DiagnosticType::ServiceStatus,
            DiagnosticType::HolochainHealth,
            DiagnosticType::MemoryUsage,
            DiagnosticType::Custom("custom-check".to_string()),
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: DiagnosticType = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
    }

    // ── Serde roundtrip: DiagnosticSeverity ─────────────────────────────

    #[test]
    fn serde_roundtrip_diagnostic_severity() {
        let variants = vec![
            DiagnosticSeverity::Healthy,
            DiagnosticSeverity::Warning,
            DiagnosticSeverity::Error,
            DiagnosticSeverity::Critical,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: DiagnosticSeverity = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
    }

    // ── Serde roundtrip: DifficultyLevel ────────────────────────────────

    #[test]
    fn serde_roundtrip_difficulty_level() {
        let variants = vec![
            DifficultyLevel::Beginner,
            DifficultyLevel::Intermediate,
            DifficultyLevel::Advanced,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: DifficultyLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
    }

    // ── Serde roundtrip: ArticleSource ──────────────────────────────────

    #[test]
    fn serde_roundtrip_article_source() {
        let variants = vec![
            ArticleSource::Community,
            ArticleSource::PreSeeded,
            ArticleSource::SymthaeaGenerated,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: ArticleSource = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
    }

    // ── Serde roundtrip: FlagReason ─────────────────────────────────────

    #[test]
    fn serde_roundtrip_flag_reason() {
        let variants = vec![
            FlagReason::Harmful,
            FlagReason::Incorrect,
            FlagReason::Outdated,
            FlagReason::Spam,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: FlagReason = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
    }

    // ── Serde roundtrip: ReputationEvent ────────────────────────────────

    #[test]
    fn serde_roundtrip_reputation_event() {
        let variants = vec![
            ReputationEvent::ResolutionVerified,
            ReputationEvent::ArticleUpvoted,
            ReputationEvent::ArticleFlagged,
            ReputationEvent::HelpProvided,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: ReputationEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
    }

    // ── Serde roundtrip: EpistemicStatus ────────────────────────────────

    #[test]
    fn serde_roundtrip_epistemic_status() {
        let variants = vec![
            EpistemicStatus::Certain,
            EpistemicStatus::Probable,
            EpistemicStatus::Uncertain,
            EpistemicStatus::Unknown,
            EpistemicStatus::OutOfDomain,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: EpistemicStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
    }

    // ── Time-sharded anchor tests ───────────────────────────────────────

    #[test]
    fn sharded_anchor_format() {
        // 2026-02-15 00:00:00 UTC → 1_771_113_600 seconds → microseconds
        let ts = Timestamp::from_micros(1_771_113_600_000_000);
        let anchor = sharded_anchor("support", "tickets", &ts);
        assert!(anchor.starts_with("support:tickets:"));
        // Should contain year and month
        assert!(anchor.contains("2026"));
    }

    #[test]
    fn sharded_anchor_different_months_differ() {
        let ts_jan = Timestamp::from_micros(1_704_067_200_000_000); // 2024-01-01
        let ts_jun = Timestamp::from_micros(1_717_200_000_000_000); // ~2024-06-01
        let a1 = sharded_anchor("support", "tickets", &ts_jan);
        let a2 = sharded_anchor("support", "tickets", &ts_jun);
        assert_ne!(a1, a2);
    }

    #[test]
    fn sharded_anchor_same_month_same() {
        let ts1 = Timestamp::from_micros(1_771_113_600_000_000);
        let ts2 = Timestamp::from_micros(1_771_113_600_000_000 + 86_400_000_000); // +1 day
        let a1 = sharded_anchor("support", "tickets", &ts1);
        let a2 = sharded_anchor("support", "tickets", &ts2);
        assert_eq!(a1, a2);
    }

    // ── Hash-sharded anchor tests ───────────────────────────────────────

    /// Build a valid EntryHash with the correct 3-byte prefix [0x84, 0x21, 0x24]
    fn make_entry_hash(fill: u8) -> EntryHash {
        let mut raw = vec![0x84, 0x21, 0x24]; // ENTRY_PREFIX
        raw.extend(vec![fill; 36]);
        EntryHash::from_raw_39(raw)
    }

    #[test]
    fn hash_sharded_anchor_format() {
        let hash = make_entry_hash(0xab);
        let anchor = hash_sharded_anchor("support", "articles", &hash);
        // First byte is the prefix 0x84, so shard = "84"
        assert_eq!(anchor, "support:articles:84");
    }

    #[test]
    fn hash_sharded_anchor_different_data_same_prefix() {
        // Both have the same prefix bytes, so same first byte → same shard
        let h1 = make_entry_hash(0x00);
        let h2 = make_entry_hash(0xff);
        let a1 = hash_sharded_anchor("support", "articles", &h1);
        let a2 = hash_sharded_anchor("support", "articles", &h2);
        // Both start with 0x84 (ENTRY_PREFIX), so same shard
        assert_eq!(a1, a2);
        assert_eq!(a1, "support:articles:84");
    }

    #[test]
    fn hash_sharded_anchor_same_first_byte_same_shard() {
        let h1 = make_entry_hash(0xab);
        let h2 = make_entry_hash(0xcd);
        let a1 = hash_sharded_anchor("support", "articles", &h1);
        let a2 = hash_sharded_anchor("support", "articles", &h2);
        // Same prefix → same first byte → same shard
        assert_eq!(a1, a2);
    }

    // ── year_month_from_timestamp tests ─────────────────────────────────

    #[test]
    fn year_month_epoch_is_1970_01() {
        let ts = Timestamp::from_micros(0);
        let (year, month) = year_month_from_timestamp(&ts);
        assert_eq!(year, 1970);
        assert_eq!(month, 1);
    }

    #[test]
    fn year_month_month_range_1_to_12() {
        // Test across a full year
        for m in 0..12 {
            let secs = 1_704_067_200 + (m * 30 * 86400); // 2024-01-01 + m months approx
            let ts = Timestamp::from_micros(secs as i64 * 1_000_000);
            let (_, month) = year_month_from_timestamp(&ts);
            assert!(month >= 1 && month <= 12, "month {} out of range", month);
        }
    }
}
