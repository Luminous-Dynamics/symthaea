// K-Index Client - Minimal API for Resonant Speech Integration (v0.3)
//
// Philosophy: Speech layer asks simple questions, doesn't know how K-Index computes answers.
// - "What changed in Knowledge over the past 7 days?"
// - "Any significant movement in Governance over the past 30 days?"
//
// Clean separation:
// - K-Index engine (data, computation, smoothing) lives elsewhere
// - ResonantSpeech just consumes pre-computed deltas

/// Delta in a K-Index dimension over a timeframe
#[derive(Debug, Clone)]
pub struct KDelta {
    /// Dimension name ("Knowledge", "Governance", "Wellbeing", etc.)
    pub dimension: String,

    /// Normalized change (-1.0 to +1.0)
    pub delta: f32,

    /// Timeframe of measurement
    pub timeframe: String,  // "Past7Days", "Past30Days", "ThisWeek", etc.

    /// Human-readable sources/drivers of change
    pub drivers: Vec<String>,  // ["O/R manuscript", "DKG expansion", ...]

    /// Optional: confidence in this measurement
    pub confidence: Option<f32>,
}

/// Minimal K-Index client interface for speech layer
///
/// Speech layer doesn't need full K-Index machinery - just ability to query
/// pre-computed deltas for specific dimensions and timeframes.
pub trait KIndexClient {
    /// Get delta for a dimension over a timeframe
    ///
    /// Returns None if:
    /// - Dimension doesn't exist
    /// - Timeframe data not available
    /// - Delta below significance threshold
    fn get_delta(&self, dimension: &str, timeframe: &str) -> Option<KDelta>;

    /// Get deltas for multiple dimensions at once
    fn get_deltas(&self, dimensions: &[&str], timeframe: &str) -> Vec<KDelta> {
        dimensions
            .iter()
            .filter_map(|dim| self.get_delta(dim, timeframe))
            .collect()
    }

    /// Get the most significant delta across all dimensions
    fn get_strongest_delta(&self, timeframe: &str) -> Option<KDelta> {
        // Default implementation: get all deltas and pick largest by abs(delta)
        // Concrete implementations can optimize this
        let all_dimensions = vec![
            "Knowledge",
            "Governance",
            "Wellbeing",
            "Resources",
            "Impact",
        ];

        self.get_deltas(&all_dimensions, timeframe)
            .into_iter()
            .max_by(|a, b| {
                a.delta
                    .abs()
                    .partial_cmp(&b.delta.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

/// Mock K-Index client for v0.3 development
///
/// Returns fake but plausible deltas for testing integration without
/// real K-Index backend.
pub struct MockKIndexClient {
    /// Fake deltas to return
    fake_deltas: Vec<KDelta>,
}

impl MockKIndexClient {
    pub fn new() -> Self {
        Self {
            fake_deltas: vec![
                KDelta {
                    dimension: "Knowledge".to_string(),
                    delta: 0.12,
                    timeframe: "Past7Days".to_string(),
                    drivers: vec![
                        "O/R manuscript editing".to_string(),
                        "Phase 11 documentation".to_string(),
                    ],
                    confidence: Some(0.85),
                },
                KDelta {
                    dimension: "Governance".to_string(),
                    delta: -0.03,
                    timeframe: "Past7Days".to_string(),
                    drivers: vec!["Lower participation in DAO votes".to_string()],
                    confidence: Some(0.72),
                },
            ],
        }
    }

    /// Add a fake delta for testing
    pub fn add_delta(&mut self, delta: KDelta) {
        self.fake_deltas.push(delta);
    }
}

impl Default for MockKIndexClient {
    fn default() -> Self {
        Self::new()
    }
}

impl KIndexClient for MockKIndexClient {
    fn get_delta(&self, dimension: &str, timeframe: &str) -> Option<KDelta> {
        self.fake_deltas
            .iter()
            .find(|d| d.dimension == dimension && d.timeframe == timeframe)
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_client_returns_delta() {
        let client = MockKIndexClient::new();
        let delta = client.get_delta("Knowledge", "Past7Days");

        assert!(delta.is_some());
        let delta = delta.unwrap();
        assert_eq!(delta.dimension, "Knowledge");
        assert!(delta.delta > 0.0);
    }

    #[test]
    fn test_get_deltas_multiple() {
        let client = MockKIndexClient::new();
        let deltas = client.get_deltas(&["Knowledge", "Governance"], "Past7Days");

        assert_eq!(deltas.len(), 2);
    }

    #[test]
    fn test_strongest_delta() {
        let client = MockKIndexClient::new();
        let strongest = client.get_strongest_delta("Past7Days");

        assert!(strongest.is_some());
        let strongest = strongest.unwrap();
        assert_eq!(strongest.dimension, "Knowledge"); // 0.12 > |−0.03|
    }

    #[test]
    fn test_custom_delta() {
        let mut client = MockKIndexClient::new();
        client.add_delta(KDelta {
            dimension: "Wellbeing".to_string(),
            delta: -0.25,
            timeframe: "Past30Days".to_string(),
            drivers: vec!["Increased stress".to_string()],
            confidence: Some(0.9),
        });

        let delta = client.get_delta("Wellbeing", "Past30Days");
        assert!(delta.is_some());
    }
}
