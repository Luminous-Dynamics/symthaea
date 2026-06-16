use std::collections::HashMap;

pub struct RankingFeedback {
    // Map of tuple hash to 'formal-success' weight adjustment
    pub feedback_map: HashMap<Vec<u64>, f64>,
}

impl RankingFeedback {
    pub fn new() -> Self {
        Self {
            feedback_map: HashMap::new(),
        }
    }

    pub fn update_score(&mut self, tuple: Vec<u64>, success: bool) {
        let adjustment = if success { 1.5 } else { 0.5 };
        self.feedback_map.insert(tuple, adjustment);
    }

    pub fn apply_to_score(&self, tuple: &[u64], current_score: f64) -> f64 {
        let multiplier = self.feedback_map.get(tuple).unwrap_or(&1.0);
        current_score * multiplier
    }
}
