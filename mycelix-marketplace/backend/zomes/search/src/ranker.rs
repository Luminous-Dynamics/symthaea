/// Ranking engine for search results
///
/// Combines multiple signals to produce final relevance scores:
/// - TF-IDF relevance score (60%)
/// - MATL trust score (30%)
/// - Recency score (10%)
pub struct Ranker {
    /// Weight for TF-IDF relevance
    relevance_weight: f64,

    /// Weight for MATL trust score
    trust_weight: f64,

    /// Weight for recency
    recency_weight: f64,

    /// Recency decay half-life in days
    recency_half_life: f64,
}

impl Ranker {
    /// Create a new ranker with default weights
    pub fn new() -> Self {
        Self {
            relevance_weight: 0.6,
            trust_weight: 0.3,
            recency_weight: 0.1,
            recency_half_life: 30.0,
        }
    }

    /// Create a ranker with custom weights
    pub fn with_weights(
        relevance_weight: f64,
        trust_weight: f64,
        recency_weight: f64,
    ) -> Self {
        Self {
            relevance_weight,
            trust_weight,
            recency_weight,
            recency_half_life: 30.0,
        }
    }

    /// Calculate final score combining TF-IDF, MATL, and recency
    ///
    /// # Arguments
    /// * `tfidf_score` - TF-IDF relevance score (0.0 to 1.0)
    /// * `matl_score` - MATL trust score (0.0 to 1.0)
    /// * `created_at` - Creation timestamp (microseconds since Unix epoch)
    ///
    /// # Returns
    /// Final score (0.0 to 1.0)
    pub fn calculate_final_score(
        &self,
        tfidf_score: f64,
        matl_score: f64,
        created_at: u64,
    ) -> f64 {
        let recency_score = self.calculate_recency_score(created_at);

        let final_score = (self.relevance_weight * tfidf_score)
            + (self.trust_weight * matl_score)
            + (self.recency_weight * recency_score);

        // Ensure score is in [0, 1] range
        final_score.max(0.0).min(1.0)
    }

    /// Calculate recency score using exponential decay
    ///
    /// Formula: e^(-age_days / half_life)
    ///
    /// Recent listings (0-7 days): ~0.8-1.0
    /// Month-old listings: ~0.37
    /// 3-month-old listings: ~0.05
    fn calculate_recency_score(&self, created_at: u64) -> f64 {
        // Get current time
        let now = get_current_time_micros();

        // Calculate age in days
        let age_micros = now.saturating_sub(created_at);
        let age_days = (age_micros as f64) / (86400.0 * 1_000_000.0);

        // Exponential decay
        (-age_days / self.recency_half_life).exp()
    }
}

/// Get current time in microseconds
///
/// In production, this would use sys_time() from HDK.
/// For testing, uses a fixed timestamp.
fn get_current_time_micros() -> u64 {
    // In real code: sys_time().unwrap_or(0)
    // For now, return a fixed value
    1703980800000000 // Dec 30, 2024 in microseconds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ranking() {
        let ranker = Ranker::new();

        // High relevance, high trust, recent
        let score1 = ranker.calculate_final_score(0.9, 0.9, get_current_time_micros() - 86400 * 1_000_000);

        // Low relevance, low trust, old
        let score2 = ranker.calculate_final_score(0.3, 0.3, get_current_time_micros() - (30 * 86400 * 1_000_000));

        assert!(score1 > score2);
        assert!(score1 > 0.7);
        assert!(score2 < 0.5);
    }

    #[test]
    fn test_trust_weighting() {
        let ranker = Ranker::new();
        let now = get_current_time_micros();

        // Same relevance and recency, different trust
        let high_trust = ranker.calculate_final_score(0.8, 0.9, now);
        let low_trust = ranker.calculate_final_score(0.8, 0.3, now);

        assert!(high_trust > low_trust);

        // Difference should be ~0.3 * (0.9 - 0.3) = ~0.18
        let diff = high_trust - low_trust;
        assert!(diff > 0.15 && diff < 0.20);
    }

    #[test]
    fn test_recency_scoring() {
        let ranker = Ranker::new();
        let now = get_current_time_micros();

        // Same relevance and trust, different recency
        let recent = ranker.calculate_final_score(0.5, 0.5, now - (1 * 86400 * 1_000_000)); // 1 day old
        let old = ranker.calculate_final_score(0.5, 0.5, now - (90 * 86400 * 1_000_000)); // 90 days old

        assert!(recent > old);
    }

    #[test]
    fn test_score_bounds() {
        let ranker = Ranker::new();
        let now = get_current_time_micros();

        // All scores should be in [0, 1]
        let score = ranker.calculate_final_score(1.0, 1.0, now);
        assert!(score >= 0.0 && score <= 1.0);

        let score2 = ranker.calculate_final_score(0.0, 0.0, now - (365 * 86400 * 1_000_000));
        assert!(score2 >= 0.0 && score2 <= 1.0);
    }

    #[test]
    fn test_custom_weights() {
        // Trust-focused ranker
        let ranker = Ranker::with_weights(0.2, 0.7, 0.1);
        let now = get_current_time_micros();

        let high_trust = ranker.calculate_final_score(0.5, 0.9, now);
        let high_relevance = ranker.calculate_final_score(0.9, 0.5, now);

        // High trust should score better with trust-focused weights
        assert!(high_trust > high_relevance);
    }
}
