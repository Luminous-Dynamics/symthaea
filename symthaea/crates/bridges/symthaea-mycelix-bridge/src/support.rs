// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea-Mycelix Support Bridge
//!
//! Adapters between Symthaea's support intelligence (triage, diagnostics,
//! knowledge, predictive) and Mycelix's DHT-based support domain.

use serde::{Deserialize, Serialize};

// ============================================================================
// TRIAGE → TICKET FIELD SUGGESTIONS
// ============================================================================

/// Suggested field values for a new support ticket, derived from triage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TicketFieldSuggestions {
    pub suggested_priority: String,
    pub suggested_category: String,
    pub confidence: f32,
    pub suggested_articles: Vec<String>,
    pub epistemic_status: String,
}

/// Convert a triage result into ticket field suggestions.
///
/// Maps Symthaea's internal types to serializable strings suitable for
/// passing to the Holochain zome coordinator.
pub fn triage_to_ticket_fields(
    priority: &str,
    category: &str,
    confidence: f32,
    articles: Vec<String>,
    epistemic_status: &str,
) -> TicketFieldSuggestions {
    TicketFieldSuggestions {
        suggested_priority: priority.to_string(),
        suggested_category: category.to_string(),
        confidence,
        suggested_articles: articles,
        epistemic_status: epistemic_status.to_string(),
    }
}

// ============================================================================
// DIAGNOSTIC PLAN → DHT ENTRIES
// ============================================================================

/// A diagnostic entry ready to be stored on the DHT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticEntryData {
    pub diagnostic_type: String,
    pub findings: String,
    pub severity: String,
    pub recommendations: Vec<String>,
    pub scrubbed: bool,
}

/// Convert a diagnostic step result into a DHT-storable entry.
pub fn step_to_diagnostic_entry(
    diagnostic_type: &str,
    findings: &str,
    severity: &str,
    recommendations: Vec<String>,
    scrubbed: bool,
) -> DiagnosticEntryData {
    DiagnosticEntryData {
        diagnostic_type: diagnostic_type.to_string(),
        findings: findings.to_string(),
        severity: severity.to_string(),
        recommendations,
        scrubbed,
    }
}

// ============================================================================
// SYMTHAEA RESPONSE → TICKET COMMENT
// ============================================================================

/// Data for a ticket comment derived from a Symthaea response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TicketCommentData {
    pub content: String,
    pub is_symthaea_response: bool,
    pub confidence: Option<f32>,
    pub epistemic_status: Option<String>,
}

/// Package a Symthaea cognitive response as a ticket comment.
pub fn response_to_comment(
    content: &str,
    confidence: Option<f32>,
    epistemic_status: Option<&str>,
) -> TicketCommentData {
    TicketCommentData {
        content: content.to_string(),
        is_symthaea_response: true,
        confidence,
        epistemic_status: epistemic_status.map(|s| s.to_string()),
    }
}

// ============================================================================
// MEMORY → COGNITIVE UPDATE
// ============================================================================

/// A cognitive update package ready for DHT gossip.
///
/// HDC vectors are natively PII-free (high-dimensional binary encodings
/// contain no recoverable personal information).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveUpdatePackage {
    pub category: String,
    pub encoding: Vec<u8>,
    pub phi: f64,
    pub resolution_pattern: String,
}

/// Package a BinaryHV encoding into a cognitive update for DHT sharing.
pub fn memory_to_cognitive_update(
    encoding: Vec<u8>,
    phi: f64,
    category: &str,
    pattern: &str,
) -> CognitiveUpdatePackage {
    CognitiveUpdatePackage {
        category: category.to_string(),
        encoding,
        phi,
        resolution_pattern: pattern.to_string(),
    }
}

/// A memory record derived from absorbing a DHT cognitive update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecordData {
    pub encoding: Vec<u8>,
    pub phi: f64,
    pub category: String,
    pub pattern: String,
}

/// Convert a DHT cognitive update into a local memory record.
pub fn cognitive_update_to_memory_record(update: &CognitiveUpdatePackage) -> MemoryRecordData {
    MemoryRecordData {
        encoding: update.encoding.clone(),
        phi: update.phi,
        category: update.category.clone(),
        pattern: update.resolution_pattern.clone(),
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triage_to_ticket_fields_roundtrip() {
        let suggestions = triage_to_ticket_fields(
            "High",
            "Network",
            0.85,
            vec!["article-1".to_string()],
            "Probable",
        );
        let json = serde_json::to_string(&suggestions).unwrap();
        let back: TicketFieldSuggestions = serde_json::from_str(&json).unwrap();
        assert_eq!(back.suggested_priority, "High");
        assert_eq!(back.suggested_category, "Network");
        assert!((back.confidence - 0.85).abs() < 1e-6);
        assert_eq!(back.suggested_articles.len(), 1);
    }

    #[test]
    fn diagnostic_entry_roundtrip() {
        let entry = step_to_diagnostic_entry(
            "NetworkCheck",
            r#"{"dns":"ok","ping":"timeout"}"#,
            "Warning",
            vec!["Check firewall rules".to_string()],
            true,
        );
        let json = serde_json::to_string(&entry).unwrap();
        let back: DiagnosticEntryData = serde_json::from_str(&json).unwrap();
        assert_eq!(back.diagnostic_type, "NetworkCheck");
        assert!(back.scrubbed);
    }

    #[test]
    fn response_to_comment_with_confidence() {
        let comment =
            response_to_comment("Try restarting the conductor", Some(0.9), Some("Probable"));
        assert!(comment.is_symthaea_response);
        assert_eq!(comment.confidence, Some(0.9));
        assert_eq!(comment.epistemic_status, Some("Probable".to_string()));
    }

    #[test]
    fn response_to_comment_without_confidence() {
        let comment = response_to_comment("I'm not sure about this", None, None);
        assert!(comment.is_symthaea_response);
        assert_eq!(comment.confidence, None);
        assert_eq!(comment.epistemic_status, None);
    }

    #[test]
    fn cognitive_update_round_trip() {
        let encoding = vec![0u8; 2048];
        let update =
            memory_to_cognitive_update(encoding.clone(), 0.75, "Network", "restart_fixes_dns");
        assert_eq!(update.encoding.len(), 2048);
        assert_eq!(update.phi, 0.75);

        let record = cognitive_update_to_memory_record(&update);
        assert_eq!(record.encoding.len(), 2048);
        assert_eq!(record.phi, 0.75);
        assert_eq!(record.category, "Network");
        assert_eq!(record.pattern, "restart_fixes_dns");
    }

    #[test]
    fn cognitive_update_serializable() {
        let update =
            memory_to_cognitive_update(vec![42u8; 2048], 0.5, "Holochain", "clear_cache_pattern");
        let json = serde_json::to_string(&update).unwrap();
        let back: CognitiveUpdatePackage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.category, "Holochain");
        assert_eq!(back.encoding.len(), 2048);
    }
}
