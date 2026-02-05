//! # Pattern Memory and Regime Detection
//!
//! Uses HDC for pattern storage, retrieval, and market regime classification.

use super::market_state::{MarketHV, MarketStateEncoder, OHLCV, TechnicalIndicators};
use serde::{Deserialize, Serialize};

/// Market regime categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Strong upward trend
    BullTrend,
    /// Strong downward trend
    BearTrend,
    /// Sideways consolidation
    Ranging,
    /// High volatility, no clear direction
    Volatile,
    /// Low volatility consolidation
    Quiet,
    /// Breakout from consolidation
    Breakout,
    /// Breakdown from support
    Breakdown,
    /// Reversal pattern forming
    Reversal,
}

impl MarketRegime {
    /// Get all regime types
    pub fn all() -> &'static [MarketRegime] {
        &[
            MarketRegime::BullTrend,
            MarketRegime::BearTrend,
            MarketRegime::Ranging,
            MarketRegime::Volatile,
            MarketRegime::Quiet,
            MarketRegime::Breakout,
            MarketRegime::Breakdown,
            MarketRegime::Reversal,
        ]
    }

    /// Suggested position bias for this regime
    pub fn position_bias(&self) -> f32 {
        match self {
            MarketRegime::BullTrend => 0.8,
            MarketRegime::BearTrend => -0.8,
            MarketRegime::Ranging => 0.0,
            MarketRegime::Volatile => 0.0,
            MarketRegime::Quiet => 0.0,
            MarketRegime::Breakout => 0.6,
            MarketRegime::Breakdown => -0.6,
            MarketRegime::Reversal => 0.0,
        }
    }

    /// Suggested risk level (0-1)
    pub fn risk_level(&self) -> f32 {
        match self {
            MarketRegime::BullTrend => 0.3,
            MarketRegime::BearTrend => 0.5,
            MarketRegime::Ranging => 0.2,
            MarketRegime::Volatile => 0.8,
            MarketRegime::Quiet => 0.1,
            MarketRegime::Breakout => 0.5,
            MarketRegime::Breakdown => 0.7,
            MarketRegime::Reversal => 0.6,
        }
    }
}

/// A stored pattern with metadata
#[derive(Clone)]
pub struct StoredPattern {
    /// Pattern hypervector
    pub hv: MarketHV,
    /// Associated market regime
    pub regime: MarketRegime,
    /// Outcome after this pattern (price change %)
    pub outcome: f32,
    /// Pattern label/description
    pub label: String,
    /// How many times this pattern has been observed
    pub count: u32,
}

/// Pattern match result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// Pattern label
    pub label: String,
    /// Similarity score
    pub similarity: f32,
    /// Associated regime
    pub regime: MarketRegime,
    /// Expected outcome
    pub expected_outcome: f32,
    /// Confidence based on count
    pub confidence: f32,
}

/// Pattern memory using HDC associative memory
pub struct PatternMemory {
    /// Stored patterns
    patterns: Vec<StoredPattern>,
    /// Regime prototype vectors (bundled from examples)
    regime_prototypes: std::collections::HashMap<MarketRegime, MarketHV>,
    /// State encoder
    encoder: MarketStateEncoder,
    /// Similarity threshold for pattern matching
    similarity_threshold: f32,
}

impl PatternMemory {
    /// Create a new pattern memory
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            regime_prototypes: std::collections::HashMap::new(),
            encoder: MarketStateEncoder::new(),
            similarity_threshold: 0.6,
        }
    }

    /// Set similarity threshold
    pub fn set_threshold(&mut self, threshold: f32) {
        self.similarity_threshold = threshold;
    }

    /// Store a pattern
    pub fn store_pattern(
        &mut self,
        candles: &[OHLCV],
        indicators: &[TechnicalIndicators],
        regime: MarketRegime,
        outcome: f32,
        label: &str,
    ) {
        let hv = self.encoder.encode_sequence(candles, indicators);

        // Check if similar pattern exists
        let mut found = false;
        for pattern in &mut self.patterns {
            if pattern.hv.similarity(&hv) > 0.9 && pattern.regime == regime {
                // Update existing pattern with running average
                pattern.count += 1;
                pattern.outcome = (pattern.outcome * (pattern.count - 1) as f32 + outcome)
                    / pattern.count as f32;
                found = true;
                break;
            }
        }

        if !found {
            self.patterns.push(StoredPattern {
                hv: hv.clone(),
                regime,
                outcome,
                label: label.to_string(),
                count: 1,
            });
        }

        // Update regime prototype
        self.regime_prototypes
            .entry(regime)
            .and_modify(|proto| *proto = proto.bundle(&hv).normalized())
            .or_insert_with(|| hv);
    }

    /// Find matching patterns
    pub fn find_matches(
        &self,
        candles: &[OHLCV],
        indicators: &[TechnicalIndicators],
        top_k: usize,
    ) -> Vec<PatternMatch> {
        let query = self.encoder.encode_sequence(candles, indicators);

        let mut matches: Vec<PatternMatch> = self.patterns.iter()
            .map(|p| {
                let sim = query.similarity(&p.hv);
                let confidence = (p.count as f32).ln().min(3.0) / 3.0;  // Log scale, max 3
                PatternMatch {
                    label: p.label.clone(),
                    similarity: sim,
                    regime: p.regime,
                    expected_outcome: p.outcome,
                    confidence,
                }
            })
            .filter(|m| m.similarity >= self.similarity_threshold)
            .collect();

        // Sort by similarity
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(top_k);
        matches
    }

    /// Classify current market regime
    pub fn classify_regime(
        &self,
        candles: &[OHLCV],
        indicators: &[TechnicalIndicators],
    ) -> (MarketRegime, f32) {
        if self.regime_prototypes.is_empty() {
            // Fallback to heuristic classification
            return self.heuristic_regime(candles, indicators);
        }

        let query = self.encoder.encode_sequence(candles, indicators);

        let mut best_regime = MarketRegime::Ranging;
        let mut best_sim = -1.0f32;

        for (regime, proto) in &self.regime_prototypes {
            let sim = query.similarity(proto);
            if sim > best_sim {
                best_sim = sim;
                best_regime = *regime;
            }
        }

        (best_regime, best_sim)
    }

    /// Heuristic regime classification (when no learned patterns)
    fn heuristic_regime(
        &self,
        candles: &[OHLCV],
        indicators: &[TechnicalIndicators],
    ) -> (MarketRegime, f32) {
        if candles.is_empty() || indicators.is_empty() {
            return (MarketRegime::Ranging, 0.0);
        }

        let last_ind = indicators.last().unwrap();

        // Calculate recent trend
        let recent_change: f64 = candles.iter()
            .rev()
            .take(5)
            .map(|c| c.change_pct())
            .sum::<f64>();

        // Volatility from ATR
        let atr_ratio = if last_ind.sma_short > 0.0 {
            last_ind.atr / last_ind.sma_short * 100.0
        } else {
            1.0
        };

        // Classify
        let (regime, confidence) = if recent_change > 5.0 && last_ind.is_uptrend() {
            (MarketRegime::BullTrend, 0.7)
        } else if recent_change < -5.0 && !last_ind.is_uptrend() {
            (MarketRegime::BearTrend, 0.7)
        } else if atr_ratio > 3.0 {
            (MarketRegime::Volatile, 0.6)
        } else if atr_ratio < 0.5 {
            (MarketRegime::Quiet, 0.6)
        } else if recent_change > 3.0 && atr_ratio > 2.0 {
            (MarketRegime::Breakout, 0.5)
        } else if recent_change < -3.0 && atr_ratio > 2.0 {
            (MarketRegime::Breakdown, 0.5)
        } else {
            (MarketRegime::Ranging, 0.4)
        };

        (regime, confidence as f32)
    }

    /// Predict outcome based on pattern matching
    pub fn predict_outcome(
        &self,
        candles: &[OHLCV],
        indicators: &[TechnicalIndicators],
    ) -> (f32, f32) {
        let matches = self.find_matches(candles, indicators, 5);

        if matches.is_empty() {
            return (0.0, 0.0);  // No prediction, no confidence
        }

        // Weighted average of outcomes
        let total_weight: f32 = matches.iter()
            .map(|m| m.similarity * m.confidence)
            .sum();

        if total_weight == 0.0 {
            return (0.0, 0.0);
        }

        let predicted: f32 = matches.iter()
            .map(|m| m.expected_outcome * m.similarity * m.confidence)
            .sum::<f32>() / total_weight;

        let avg_confidence = total_weight / matches.len() as f32;

        (predicted, avg_confidence)
    }

    /// Get pattern count
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Get regime prototype count
    pub fn regime_count(&self) -> usize {
        self.regime_prototypes.len()
    }

    /// Clear all patterns
    pub fn clear(&mut self) {
        self.patterns.clear();
        self.regime_prototypes.clear();
    }

    /// Add common canonical patterns
    pub fn add_canonical_patterns(&mut self) {
        // These are placeholder patterns - in a real system you'd learn these from data
        // For now, we just document what patterns would look like

        // The patterns would be learned from actual market data sequences
        // This method serves as documentation of the pattern types we recognize
    }
}

impl Default for PatternMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle(open: f64, high: f64, low: f64, close: f64, ts: i64) -> OHLCV {
        OHLCV::new(open, high, low, close, 1000.0, ts)
    }

    fn make_indicators(rsi: f64, sma_short: f64, sma_long: f64, atr: f64) -> TechnicalIndicators {
        TechnicalIndicators {
            rsi,
            sma_short,
            sma_long,
            atr,
            bb_upper: sma_short * 1.02,
            bb_lower: sma_short * 0.98,
            volume_ma: 1000.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_pattern_memory_creation() {
        let memory = PatternMemory::new();
        assert_eq!(memory.pattern_count(), 0);
    }

    #[test]
    fn test_store_and_find_pattern() {
        let mut memory = PatternMemory::new();

        // Create a bullish pattern
        let candles: Vec<OHLCV> = (0..5)
            .map(|i| make_candle(100.0 + i as f64, 102.0 + i as f64, 99.0 + i as f64, 101.0 + i as f64, i))
            .collect();

        let indicators: Vec<TechnicalIndicators> = (0..5)
            .map(|_| make_indicators(60.0, 100.0, 98.0, 2.0))
            .collect();

        memory.store_pattern(&candles, &indicators, MarketRegime::BullTrend, 2.5, "bullish_momentum");

        assert_eq!(memory.pattern_count(), 1);

        // Find similar pattern
        let matches = memory.find_matches(&candles, &indicators, 5);
        assert!(!matches.is_empty());
        assert!(matches[0].similarity > 0.9);
    }

    #[test]
    fn test_regime_classification() {
        let memory = PatternMemory::new();

        // Create bullish data
        let candles: Vec<OHLCV> = (0..5)
            .map(|i| make_candle(100.0 + i as f64 * 2.0, 103.0 + i as f64 * 2.0, 99.0 + i as f64 * 2.0, 102.0 + i as f64 * 2.0, i))
            .collect();

        let indicators: Vec<TechnicalIndicators> = (0..5)
            .map(|_| make_indicators(65.0, 105.0, 100.0, 2.0))
            .collect();

        let (regime, _confidence) = memory.classify_regime(&candles, &indicators);

        // Should classify as bullish
        assert!(matches!(regime, MarketRegime::BullTrend | MarketRegime::Breakout));
    }

    #[test]
    fn test_market_regime_properties() {
        assert!(MarketRegime::BullTrend.position_bias() > 0.0);
        assert!(MarketRegime::BearTrend.position_bias() < 0.0);
        assert_eq!(MarketRegime::Ranging.position_bias(), 0.0);

        assert!(MarketRegime::Volatile.risk_level() > MarketRegime::Quiet.risk_level());
    }
}
