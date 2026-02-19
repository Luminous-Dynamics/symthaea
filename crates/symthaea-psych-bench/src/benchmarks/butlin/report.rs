//! Butlin indicator report structures.

use serde::{Deserialize, Serialize};

/// Status of a consciousness indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndicatorStatus {
    /// The architectural property is clearly present.
    Present,
    /// The property is partially implemented or ambiguous.
    Partial,
    /// The property is absent.
    Absent,
}

impl std::fmt::Display for IndicatorStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndicatorStatus::Present => write!(f, "PRESENT"),
            IndicatorStatus::Partial => write!(f, "PARTIAL"),
            IndicatorStatus::Absent => write!(f, "ABSENT"),
        }
    }
}

/// Evidence for a single consciousness indicator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorEvidence {
    /// Indicator ID (e.g., "RPT-1", "GWT-3").
    pub id: String,
    /// Theory of origin (e.g., "Recurrent Processing Theory").
    pub theory: String,
    /// Description of the indicator.
    pub description: String,
    /// Assessment status.
    pub status: IndicatorStatus,
    /// Detailed evidence string.
    pub evidence: String,
    /// Quantitative measure (if applicable, 0.0-1.0).
    pub score: Option<f64>,
}

/// Complete report of all consciousness indicators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ButlinIndicatorReport {
    /// All indicator evaluations.
    pub indicators: Vec<IndicatorEvidence>,
    /// Count of Present indicators.
    pub present_count: usize,
    /// Count of Partial indicators.
    pub partial_count: usize,
    /// Count of Absent indicators.
    pub absent_count: usize,
}

impl ButlinIndicatorReport {
    /// Build from a list of indicator evaluations.
    pub fn from_indicators(indicators: Vec<IndicatorEvidence>) -> Self {
        let present_count = indicators.iter().filter(|i| i.status == IndicatorStatus::Present).count();
        let partial_count = indicators.iter().filter(|i| i.status == IndicatorStatus::Partial).count();
        let absent_count = indicators.iter().filter(|i| i.status == IndicatorStatus::Absent).count();
        Self {
            indicators,
            present_count,
            partial_count,
            absent_count,
        }
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "=== Butlin et al. Consciousness Indicators ===".to_string(),
            format!(
                "  Present: {}, Partial: {}, Absent: {}",
                self.present_count, self.partial_count, self.absent_count
            ),
        ];
        for ind in &self.indicators {
            let score_str = ind
                .score
                .map(|s| format!(" ({:.2})", s))
                .unwrap_or_default();
            lines.push(format!(
                "  [{}] {} - {}: {}{}",
                ind.id, ind.status, ind.description, ind.evidence, score_str
            ));
        }
        lines.join("\n")
    }
}
