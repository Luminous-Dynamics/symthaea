//! # Market State HDC Encoding
//!
//! Encodes financial market data into hyperdimensional vectors for
//! pattern matching and similarity-based analysis.
//!
//! Uses unified ContinuousHV (16,384 dimensions) from symthaea-core
//! for compatibility with other HDC modules (voice, cognitive loop).

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

/// HDC dimension for market encoding (unified with symthaea-core)
const MARKET_HDC_DIM: usize = HDC_DIMENSION;  // 16,384

/// OHLCV (Open, High, Low, Close, Volume) candle data
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OHLCV {
    /// Opening price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
    /// Timestamp (Unix seconds)
    pub timestamp: i64,
}

impl OHLCV {
    /// Create a new OHLCV candle
    pub fn new(open: f64, high: f64, low: f64, close: f64, volume: f64, timestamp: i64) -> Self {
        Self { open, high, low, close, volume, timestamp }
    }

    /// Price change (close - open)
    pub fn change(&self) -> f64 {
        self.close - self.open
    }

    /// Price change as percentage
    pub fn change_pct(&self) -> f64 {
        if self.open != 0.0 {
            (self.close - self.open) / self.open * 100.0
        } else {
            0.0
        }
    }

    /// True range (for ATR calculation)
    pub fn true_range(&self, prev_close: f64) -> f64 {
        let hl = self.high - self.low;
        let hc = (self.high - prev_close).abs();
        let lc = (self.low - prev_close).abs();
        hl.max(hc).max(lc)
    }

    /// Candle body size
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Upper wick size
    pub fn upper_wick(&self) -> f64 {
        self.high - self.open.max(self.close)
    }

    /// Lower wick size
    pub fn lower_wick(&self) -> f64 {
        self.open.min(self.close) - self.low
    }

    /// Is this a bullish (green) candle?
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Technical indicators computed from OHLCV data
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct TechnicalIndicators {
    /// Relative Strength Index (0-100)
    pub rsi: f64,
    /// MACD line
    pub macd: f64,
    /// MACD signal line
    pub macd_signal: f64,
    /// Average True Range
    pub atr: f64,
    /// Simple Moving Average (short)
    pub sma_short: f64,
    /// Simple Moving Average (long)
    pub sma_long: f64,
    /// Bollinger Band upper
    pub bb_upper: f64,
    /// Bollinger Band lower
    pub bb_lower: f64,
    /// Volume moving average
    pub volume_ma: f64,
    /// Momentum (rate of change)
    pub momentum: f64,
}

impl TechnicalIndicators {
    /// Check if price is above short-term MA
    pub fn above_short_ma(&self, price: f64) -> bool {
        price > self.sma_short
    }

    /// Check if in uptrend (short MA > long MA)
    pub fn is_uptrend(&self) -> bool {
        self.sma_short > self.sma_long
    }

    /// Check if RSI is oversold
    pub fn is_oversold(&self, threshold: f64) -> bool {
        self.rsi < threshold
    }

    /// Check if RSI is overbought
    pub fn is_overbought(&self, threshold: f64) -> bool {
        self.rsi > threshold
    }

    /// Check if price is near lower Bollinger Band
    pub fn near_lower_bb(&self, price: f64, tolerance: f64) -> bool {
        price <= self.bb_lower * (1.0 + tolerance)
    }

    /// Check if price is near upper Bollinger Band
    pub fn near_upper_bb(&self, price: f64, tolerance: f64) -> bool {
        price >= self.bb_upper * (1.0 - tolerance)
    }
}

/// Market hypervector - newtype wrapper around ContinuousHV for type safety
///
/// Uses unified 16,384-dimensional representation compatible with
/// voice, STT, and cognitive loop HDC modules.
#[derive(Clone)]
pub struct MarketHV(ContinuousHV);

impl MarketHV {
    /// Create a zero vector
    pub fn zeros() -> Self {
        Self(ContinuousHV::zero(MARKET_HDC_DIM))
    }

    /// Create a random bipolar vector from seed
    pub fn random(seed: u64) -> Self {
        Self(ContinuousHV::random(MARKET_HDC_DIM, seed))
    }

    /// Create from ContinuousHV
    pub fn from_continuous(hv: ContinuousHV) -> Self {
        Self(hv)
    }

    /// Get the underlying ContinuousHV
    pub fn as_continuous(&self) -> &ContinuousHV {
        &self.0
    }

    /// Bind two vectors (element-wise multiplication)
    pub fn bind(&self, other: &MarketHV) -> MarketHV {
        Self(self.0.bind(&other.0))
    }

    /// Bundle vectors (element-wise addition)
    pub fn bundle(&self, other: &MarketHV) -> MarketHV {
        Self(ContinuousHV::bundle(&[&self.0, &other.0]))
    }

    /// Bundle with weighting
    pub fn weighted_bundle(&self, other: &MarketHV, self_weight: f32, other_weight: f32) -> MarketHV {
        let a = self.0.scale(self_weight);
        let b = other.0.scale(other_weight);
        Self(ContinuousHV::bundle(&[&a, &b]))
    }

    /// Permute (rotate) the vector
    pub fn permute(&self, amount: i32) -> MarketHV {
        // Convert signed to unsigned (wrap around)
        let dim = MARKET_HDC_DIM as i32;
        let shift = ((amount % dim) + dim) % dim;
        Self(self.0.permute(shift as usize))
    }

    /// Normalize to unit length (in-place)
    pub fn normalize(&mut self) {
        self.0 = self.0.normalize();
    }

    /// Get normalized copy
    pub fn normalized(&self) -> MarketHV {
        Self(self.0.normalize())
    }

    /// Cosine similarity
    pub fn similarity(&self, other: &MarketHV) -> f32 {
        self.0.similarity(&other.0)
    }

    /// Get raw values
    pub fn values(&self) -> &[f32] {
        &self.0.values
    }

    /// Get mutable access to values (for level encoder)
    pub fn values_mut(&mut self) -> &mut [f32] {
        &mut self.0.values
    }

    /// Scale by a factor
    pub fn scale(&self, factor: f32) -> MarketHV {
        Self(self.0.scale(factor))
    }
}

/// Level encoding for continuous values
struct LevelEncoder {
    levels: Vec<MarketHV>,
    min_val: f64,
    max_val: f64,
}

impl LevelEncoder {
    fn new(num_levels: usize, seed: u64, min_val: f64, max_val: f64) -> Self {
        // Generate orthogonal level vectors using flipping approach
        let mut levels = Vec::with_capacity(num_levels);
        let base = MarketHV::random(seed);
        levels.push(base.clone());

        let flip_per_level = MARKET_HDC_DIM / num_levels;

        for i in 1..num_levels {
            let mut prev = levels[i - 1].clone();
            // Flip some bits from previous level
            for j in 0..flip_per_level {
                let idx = ((i - 1) * flip_per_level + j) % MARKET_HDC_DIM;
                prev.values_mut()[idx] *= -1.0;
            }
            levels.push(prev);
        }

        Self { levels, min_val, max_val }
    }

    fn encode(&self, value: f64) -> MarketHV {
        let clamped = value.clamp(self.min_val, self.max_val);
        let normalized = (clamped - self.min_val) / (self.max_val - self.min_val);
        let level_idx = ((normalized * (self.levels.len() - 1) as f64).round() as usize)
            .min(self.levels.len() - 1);
        self.levels[level_idx].clone()
    }
}

/// Market state encoder using HDC
pub struct MarketStateEncoder {
    /// Feature base vectors (one per feature)
    feature_bases: Vec<MarketHV>,
    /// Level encoders for different value ranges
    price_levels: LevelEncoder,
    volume_levels: LevelEncoder,
    rsi_levels: LevelEncoder,
    momentum_levels: LevelEncoder,
    /// Position encoding for time series
    position_bases: Vec<MarketHV>,
}

impl MarketStateEncoder {
    /// Create a new market state encoder
    pub fn new() -> Self {
        // Base vectors for different features
        let feature_names = [
            "price_change", "body_size", "upper_wick", "lower_wick",
            "volume_ratio", "rsi", "macd", "atr", "trend", "bb_position",
            "momentum"
        ];
        let feature_bases: Vec<MarketHV> = feature_names.iter()
            .enumerate()
            .map(|(i, _)| MarketHV::random(100 + i as u64))
            .collect();

        // Level encoders
        let price_levels = LevelEncoder::new(32, 200, -10.0, 10.0);  // % change
        let volume_levels = LevelEncoder::new(16, 300, 0.0, 5.0);    // Volume ratio
        let rsi_levels = LevelEncoder::new(21, 400, 0.0, 100.0);     // RSI 0-100
        let momentum_levels = LevelEncoder::new(32, 500, -20.0, 20.0);

        // Position encoding for lookback
        let position_bases: Vec<MarketHV> = (0..50)
            .map(|i| MarketHV::random(600 + i as u64))
            .collect();

        Self {
            feature_bases,
            price_levels,
            volume_levels,
            rsi_levels,
            momentum_levels,
            position_bases,
        }
    }

    /// Encode a single candle with indicators
    pub fn encode_candle(&self, candle: &OHLCV, indicators: &TechnicalIndicators) -> MarketHV {
        let mut components = Vec::new();

        // Price change encoding
        let change_hv = self.price_levels.encode(candle.change_pct());
        components.push(self.feature_bases[0].bind(&change_hv));

        // Candle structure (normalized by ATR)
        let atr = indicators.atr.max(0.001);
        let body_norm = (candle.body_size() / atr).min(5.0);
        let body_hv = self.price_levels.encode(body_norm);
        components.push(self.feature_bases[1].bind(&body_hv));

        // Wicks
        let upper_norm = (candle.upper_wick() / atr).min(3.0);
        let upper_hv = self.price_levels.encode(upper_norm);
        components.push(self.feature_bases[2].bind(&upper_hv));

        let lower_norm = (candle.lower_wick() / atr).min(3.0);
        let lower_hv = self.price_levels.encode(lower_norm);
        components.push(self.feature_bases[3].bind(&lower_hv));

        // Volume ratio to moving average
        let vol_ratio = if indicators.volume_ma > 0.0 {
            candle.volume / indicators.volume_ma
        } else {
            1.0
        };
        let vol_hv = self.volume_levels.encode(vol_ratio);
        components.push(self.feature_bases[4].bind(&vol_hv));

        // RSI
        let rsi_hv = self.rsi_levels.encode(indicators.rsi);
        components.push(self.feature_bases[5].bind(&rsi_hv));

        // MACD (normalized)
        let macd_norm = (indicators.macd / atr).clamp(-5.0, 5.0);
        let macd_hv = self.price_levels.encode(macd_norm);
        components.push(self.feature_bases[6].bind(&macd_hv));

        // Trend (SMA relationship)
        let trend = if indicators.sma_long > 0.0 {
            (indicators.sma_short - indicators.sma_long) / indicators.sma_long * 100.0
        } else {
            0.0
        };
        let trend_hv = self.price_levels.encode(trend.clamp(-10.0, 10.0));
        components.push(self.feature_bases[8].bind(&trend_hv));

        // BB position (where price is within bands)
        let bb_range = indicators.bb_upper - indicators.bb_lower;
        let bb_pos = if bb_range > 0.0 {
            (candle.close - indicators.bb_lower) / bb_range * 2.0 - 1.0
        } else {
            0.0
        };
        let bb_hv = self.price_levels.encode(bb_pos * 5.0);  // Scale to [-5, 5]
        components.push(self.feature_bases[9].bind(&bb_hv));

        // Momentum
        let momentum_hv = self.momentum_levels.encode(indicators.momentum);
        components.push(self.feature_bases[10].bind(&momentum_hv));

        // Bundle all components
        let mut result = MarketHV::zeros();
        for comp in components {
            result = result.bundle(&comp);
        }
        result.normalize();
        result
    }

    /// Encode a sequence of candles with temporal binding
    pub fn encode_sequence(
        &self,
        candles: &[OHLCV],
        indicators: &[TechnicalIndicators],
    ) -> MarketHV {
        if candles.is_empty() {
            return MarketHV::zeros();
        }

        let mut result = MarketHV::zeros();

        for (i, (candle, ind)) in candles.iter().zip(indicators.iter()).enumerate() {
            let candle_hv = self.encode_candle(candle, ind);

            // Bind with position to encode temporal order
            // More recent candles get less permutation
            let pos_idx = candles.len() - 1 - i;
            if pos_idx < self.position_bases.len() {
                let positioned = candle_hv.bind(&self.position_bases[pos_idx]);
                // Weight recent candles more heavily
                let weight = 1.0 / (1.0 + pos_idx as f32 * 0.2);
                result = result.weighted_bundle(&positioned, 1.0, weight);
            }
        }

        result.normalize();
        result
    }
}

impl Default for MarketStateEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle(open: f64, high: f64, low: f64, close: f64) -> OHLCV {
        OHLCV::new(open, high, low, close, 1000.0, 0)
    }

    #[test]
    fn test_ohlcv_change() {
        let candle = make_candle(100.0, 105.0, 98.0, 103.0);
        assert!((candle.change() - 3.0).abs() < 0.001);
        assert!((candle.change_pct() - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_ohlcv_structure() {
        let candle = make_candle(100.0, 105.0, 95.0, 102.0);
        assert!((candle.body_size() - 2.0).abs() < 0.001);
        assert!((candle.upper_wick() - 3.0).abs() < 0.001);
        assert!((candle.lower_wick() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_market_hv_similarity() {
        let a = MarketHV::random(42);
        let b = MarketHV::random(42);
        let c = MarketHV::random(43);

        // Same seed should give identical vectors
        assert!((a.similarity(&b) - 1.0).abs() < 0.001);

        // Different seeds should give near-orthogonal
        assert!(a.similarity(&c).abs() < 0.2);
    }

    #[test]
    fn test_encoder_similar_candles() {
        let encoder = MarketStateEncoder::new();

        let indicators = TechnicalIndicators {
            rsi: 50.0,
            atr: 2.0,
            sma_short: 100.0,
            sma_long: 100.0,
            bb_upper: 105.0,
            bb_lower: 95.0,
            volume_ma: 1000.0,
            ..Default::default()
        };

        // Two similar bullish candles
        let candle1 = make_candle(100.0, 103.0, 99.0, 102.0);
        let candle2 = make_candle(100.0, 103.5, 99.5, 102.5);

        let hv1 = encoder.encode_candle(&candle1, &indicators);
        let hv2 = encoder.encode_candle(&candle2, &indicators);

        let sim = hv1.similarity(&hv2);
        assert!(sim > 0.7, "Similar candles should have high similarity: {}", sim);
    }

    #[test]
    fn test_encoder_opposite_candles() {
        let encoder = MarketStateEncoder::new();

        // Use different indicators to represent different market states
        let bullish_indicators = TechnicalIndicators {
            rsi: 70.0,
            atr: 2.0,
            sma_short: 105.0,
            sma_long: 100.0,  // Short above long = uptrend
            bb_upper: 108.0,
            bb_lower: 98.0,
            volume_ma: 1000.0,
            momentum: 5.0,
            ..Default::default()
        };

        let bearish_indicators = TechnicalIndicators {
            rsi: 30.0,
            atr: 3.0,
            sma_short: 95.0,
            sma_long: 100.0,  // Short below long = downtrend
            bb_upper: 102.0,
            bb_lower: 92.0,
            volume_ma: 1500.0,
            momentum: -5.0,
            ..Default::default()
        };

        // Bullish vs bearish candle with matching indicators
        let bullish = make_candle(100.0, 105.0, 99.0, 104.0);
        let bearish = make_candle(100.0, 101.0, 95.0, 96.0);

        let hv1 = encoder.encode_candle(&bullish, &bullish_indicators);
        let hv2 = encoder.encode_candle(&bearish, &bearish_indicators);

        let sim = hv1.similarity(&hv2);
        // With different indicators, similarity should be lower
        assert!(sim < 0.7, "Opposite market states should have moderate to low similarity: {}", sim);
    }
}
