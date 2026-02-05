//! # Market Simulator
//!
//! Generates synthetic market data for testing pattern recognition systems.
//! Uses HDC pattern memory to generate realistic market sequences.

use super::market_state::{OHLCV, TechnicalIndicators};
use super::pattern_hdc::{MarketRegime, PatternMemory};
use serde::{Deserialize, Serialize};

/// Configuration for the market simulator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatorConfig {
    /// Starting price
    pub initial_price: f64,
    /// Base volatility (daily %)
    pub base_volatility: f64,
    /// Mean return (daily %)
    pub mean_return: f64,
    /// Volume base
    pub base_volume: f64,
    /// Regime persistence (0-1, higher = longer regimes)
    pub regime_persistence: f64,
    /// Trend strength multiplier
    pub trend_strength: f64,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            initial_price: 100.0,
            base_volatility: 1.5,
            mean_return: 0.02,
            base_volume: 1_000_000.0,
            regime_persistence: 0.95,
            trend_strength: 1.5,
        }
    }
}

/// Simple random number generator (xorshift)
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        self.next_u64() as f64 / u64::MAX as f64
    }

    /// Standard normal using Box-Muller
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-10);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Market simulator with regime-based generation
pub struct MarketSimulator {
    /// Configuration
    config: SimulatorConfig,
    /// Random state
    rng: Rng,
    /// Current regime
    current_regime: MarketRegime,
    /// Current price
    current_price: f64,
    /// Price history for indicator calculation
    history: Vec<OHLCV>,
    /// Current timestamp
    timestamp: i64,
    /// Pattern memory for regime-aware generation
    patterns: Option<PatternMemory>,
}

impl MarketSimulator {
    /// Create a new simulator
    pub fn new(config: SimulatorConfig, seed: u64) -> Self {
        Self {
            current_price: config.initial_price,
            config,
            rng: Rng::new(seed),
            current_regime: MarketRegime::Ranging,
            history: Vec::new(),
            timestamp: 0,
            patterns: None,
        }
    }

    /// Create with pattern memory for more realistic generation
    pub fn with_patterns(config: SimulatorConfig, seed: u64, patterns: PatternMemory) -> Self {
        let mut sim = Self::new(config, seed);
        sim.patterns = Some(patterns);
        sim
    }

    /// Get current price
    pub fn price(&self) -> f64 {
        self.current_price
    }

    /// Get current regime
    pub fn regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Get price history
    pub fn history(&self) -> &[OHLCV] {
        &self.history
    }

    /// Maybe switch regime based on persistence
    fn maybe_switch_regime(&mut self) {
        if self.rng.next_f64() > self.config.regime_persistence {
            // Switch to a new regime
            let regimes = MarketRegime::all();
            let idx = (self.rng.next_u64() as usize) % regimes.len();
            self.current_regime = regimes[idx];
        }
    }

    /// Get regime-specific parameters
    fn regime_params(&self) -> (f64, f64) {
        // (drift, volatility_mult)
        match self.current_regime {
            MarketRegime::BullTrend => (self.config.mean_return * self.config.trend_strength, 0.8),
            MarketRegime::BearTrend => (-self.config.mean_return * self.config.trend_strength, 0.9),
            MarketRegime::Ranging => (0.0, 0.5),
            MarketRegime::Volatile => (0.0, 2.5),
            MarketRegime::Quiet => (0.0, 0.3),
            MarketRegime::Breakout => (self.config.mean_return * 2.0, 1.5),
            MarketRegime::Breakdown => (-self.config.mean_return * 2.0, 1.8),
            MarketRegime::Reversal => (-self.regime_momentum() * 0.5, 1.2),
        }
    }

    /// Calculate recent momentum
    fn regime_momentum(&self) -> f64 {
        if self.history.len() < 5 {
            return 0.0;
        }
        let recent: f64 = self.history.iter()
            .rev()
            .take(5)
            .map(|c| c.change_pct())
            .sum();
        recent / 5.0
    }

    /// Generate the next candle
    pub fn next_candle(&mut self) -> OHLCV {
        self.maybe_switch_regime();

        let (drift, vol_mult) = self.regime_params();
        let volatility = self.config.base_volatility * vol_mult / 100.0;

        // Generate random walk with drift
        let daily_return = drift / 100.0 + volatility * self.rng.next_normal();

        // Generate OHLC from daily return
        let open = self.current_price;

        // Intraday high/low based on volatility
        let intraday_vol = volatility * 0.5;
        let high_dev = intraday_vol * self.rng.next_f64().abs();
        let low_dev = intraday_vol * self.rng.next_f64().abs();

        let close = open * (1.0 + daily_return);

        let high = open.max(close) * (1.0 + high_dev);
        let low = open.min(close) * (1.0 - low_dev);

        // Volume with regime influence
        let volume_mult = match self.current_regime {
            MarketRegime::Volatile | MarketRegime::Breakout | MarketRegime::Breakdown => 1.5 + self.rng.next_f64(),
            MarketRegime::Quiet => 0.5 + self.rng.next_f64() * 0.3,
            _ => 0.8 + self.rng.next_f64() * 0.4,
        };
        let volume = self.config.base_volume * volume_mult;

        let candle = OHLCV::new(open, high, low, close, volume, self.timestamp);

        // Update state
        self.current_price = close;
        self.timestamp += 86400; // Daily
        self.history.push(candle);

        candle
    }

    /// Generate n candles
    pub fn generate(&mut self, n: usize) -> Vec<OHLCV> {
        (0..n).map(|_| self.next_candle()).collect()
    }

    /// Calculate technical indicators for current history
    pub fn calculate_indicators(&self) -> Vec<TechnicalIndicators> {
        let mut indicators = Vec::with_capacity(self.history.len());

        for i in 0..self.history.len() {
            let ind = self.calc_indicators_at(i);
            indicators.push(ind);
        }

        indicators
    }

    /// Calculate indicators at a specific point
    fn calc_indicators_at(&self, idx: usize) -> TechnicalIndicators {
        let closes: Vec<f64> = self.history[..=idx].iter().map(|c| c.close).collect();
        let highs: Vec<f64> = self.history[..=idx].iter().map(|c| c.high).collect();
        let lows: Vec<f64> = self.history[..=idx].iter().map(|c| c.low).collect();
        let volumes: Vec<f64> = self.history[..=idx].iter().map(|c| c.volume).collect();

        let sma_short = Self::sma(&closes, 10);
        let sma_long = Self::sma(&closes, 20);

        let rsi = Self::rsi(&closes, 14);

        let (macd, signal) = Self::macd(&closes, 12, 26, 9);

        let atr = Self::atr(&highs, &lows, &closes, 14);

        let (bb_upper, bb_lower) = Self::bollinger(&closes, 20, 2.0);

        let volume_ma = Self::sma(&volumes, 20);

        let momentum = if closes.len() >= 10 {
            (closes[closes.len() - 1] - closes[closes.len() - 10]) / closes[closes.len() - 10] * 100.0
        } else {
            0.0
        };

        TechnicalIndicators {
            rsi,
            macd,
            macd_signal: signal,
            atr,
            sma_short,
            sma_long,
            bb_upper,
            bb_lower,
            volume_ma,
            momentum,
        }
    }

    /// Simple Moving Average
    fn sma(data: &[f64], period: usize) -> f64 {
        if data.len() < period {
            return data.iter().sum::<f64>() / data.len().max(1) as f64;
        }
        let sum: f64 = data.iter().rev().take(period).sum();
        sum / period as f64
    }

    /// RSI calculation
    fn rsi(closes: &[f64], period: usize) -> f64 {
        if closes.len() < 2 {
            return 50.0;
        }

        let mut gains = 0.0;
        let mut losses = 0.0;
        let look_back = period.min(closes.len() - 1);

        for i in (closes.len() - look_back)..closes.len() {
            let change = closes[i] - closes[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }

        let avg_gain = gains / look_back as f64;
        let avg_loss = losses / look_back as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// MACD calculation
    fn macd(closes: &[f64], fast: usize, slow: usize, signal: usize) -> (f64, f64) {
        let ema_fast = Self::ema(closes, fast);
        let ema_slow = Self::ema(closes, slow);
        let macd_line = ema_fast - ema_slow;

        // Simplified signal calculation
        let signal_line = macd_line * (2.0 / (signal + 1) as f64);

        (macd_line, signal_line)
    }

    /// EMA calculation
    fn ema(data: &[f64], period: usize) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let multiplier = 2.0 / (period + 1) as f64;
        let mut ema = data[0];

        for &price in data.iter().skip(1) {
            ema = (price - ema) * multiplier + ema;
        }

        ema
    }

    /// ATR calculation
    fn atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
        if closes.len() < 2 {
            return if !highs.is_empty() { highs[0] - lows[0] } else { 0.0 };
        }

        let mut tr_sum = 0.0;
        let look_back = period.min(closes.len() - 1);

        for i in (closes.len() - look_back)..closes.len() {
            let hl = highs[i] - lows[i];
            let hc = (highs[i] - closes[i - 1]).abs();
            let lc = (lows[i] - closes[i - 1]).abs();
            tr_sum += hl.max(hc).max(lc);
        }

        tr_sum / look_back as f64
    }

    /// Bollinger Bands
    fn bollinger(closes: &[f64], period: usize, std_mult: f64) -> (f64, f64) {
        let sma = Self::sma(closes, period);

        let look_back = period.min(closes.len());
        let variance: f64 = closes.iter()
            .rev()
            .take(look_back)
            .map(|c| (c - sma).powi(2))
            .sum::<f64>() / look_back as f64;

        let std_dev = variance.sqrt();

        (sma + std_mult * std_dev, sma - std_mult * std_dev)
    }

    /// Reset the simulator
    pub fn reset(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
        self.current_price = self.config.initial_price;
        self.current_regime = MarketRegime::Ranging;
        self.history.clear();
        self.timestamp = 0;
    }
}

impl Default for MarketSimulator {
    fn default() -> Self {
        Self::new(SimulatorConfig::default(), 42)
    }
}

/// Backtest results
#[derive(Debug, Clone, Default)]
pub struct BacktestResult {
    /// Total return percentage
    pub total_return: f64,
    /// Number of trades
    pub trade_count: u32,
    /// Win rate
    pub win_rate: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Sharpe ratio (simplified)
    pub sharpe_ratio: f64,
    /// Returns by regime
    pub regime_returns: std::collections::HashMap<String, f64>,
}

/// Simple backtester
pub struct Backtester {
    /// Pattern memory for predictions
    patterns: PatternMemory,
    /// Lookback window for pattern matching
    lookback: usize,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(patterns: PatternMemory, lookback: usize) -> Self {
        Self { patterns, lookback }
    }

    /// Run a simple backtest
    pub fn run(&self, candles: &[OHLCV], indicators: &[TechnicalIndicators]) -> BacktestResult {
        if candles.len() < self.lookback + 1 {
            return BacktestResult::default();
        }

        let mut equity: f64 = 1.0;
        let mut peak_equity: f64 = 1.0;
        let mut max_drawdown: f64 = 0.0;
        let mut returns: Vec<f64> = Vec::new();
        let mut wins = 0u32;
        let mut trades = 0u32;
        let mut regime_returns: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();

        for i in self.lookback..candles.len() - 1 {
            let window_candles = &candles[i - self.lookback..i];
            let window_indicators = &indicators[i - self.lookback..i];

            // Get prediction
            let (predicted_return, confidence) = self.patterns.predict_outcome(window_candles, window_indicators);

            // Get regime
            let (regime, _) = self.patterns.classify_regime(window_candles, window_indicators);

            // Only trade if confidence above threshold
            if confidence < 0.3 {
                continue;
            }

            // Calculate actual return
            let actual_return = candles[i + 1].change_pct() / 100.0;

            // Position size based on confidence and regime risk
            let base_position: f64 = if predicted_return > 0.0 { 1.0 } else { -1.0 };
            let position_size = base_position * (confidence as f64) * (1.0 - regime.risk_level() as f64);

            // Trade return
            let trade_return = position_size * actual_return;
            returns.push(trade_return);

            // Track
            equity *= 1.0 + trade_return;
            peak_equity = peak_equity.max(equity);
            let drawdown = (peak_equity - equity) / peak_equity;
            max_drawdown = max_drawdown.max(drawdown);

            trades += 1;
            if trade_return > 0.0 {
                wins += 1;
            }

            // Track by regime
            let regime_name = format!("{:?}", regime);
            regime_returns.entry(regime_name).or_default().push(trade_return);
        }

        // Calculate metrics
        let total_return = (equity - 1.0) * 100.0;
        let win_rate = if trades > 0 { wins as f64 / trades as f64 } else { 0.0 };

        let avg_return = if !returns.is_empty() {
            returns.iter().sum::<f64>() / returns.len() as f64
        } else {
            0.0
        };

        let variance: f64 = returns.iter()
            .map(|r| (r - avg_return).powi(2))
            .sum::<f64>() / returns.len().max(1) as f64;
        let std_dev = variance.sqrt();

        let sharpe_ratio = if std_dev > 0.0 {
            avg_return / std_dev * (252.0_f64).sqrt()  // Annualized
        } else {
            0.0
        };

        // Aggregate regime returns
        let regime_avg: std::collections::HashMap<String, f64> = regime_returns.into_iter()
            .map(|(k, v)| {
                let avg = v.iter().sum::<f64>() / v.len().max(1) as f64 * 100.0;
                (k, avg)
            })
            .collect();

        BacktestResult {
            total_return,
            trade_count: trades,
            win_rate,
            max_drawdown: max_drawdown * 100.0,
            sharpe_ratio,
            regime_returns: regime_avg,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulator_creation() {
        let sim = MarketSimulator::default();
        assert_eq!(sim.price(), 100.0);
    }

    #[test]
    fn test_generate_candles() {
        let mut sim = MarketSimulator::default();
        let candles = sim.generate(10);

        assert_eq!(candles.len(), 10);
        assert_eq!(sim.history().len(), 10);

        // Check candle validity
        for candle in &candles {
            assert!(candle.high >= candle.low);
            assert!(candle.high >= candle.open);
            assert!(candle.high >= candle.close);
            assert!(candle.low <= candle.open);
            assert!(candle.low <= candle.close);
            assert!(candle.volume > 0.0);
        }
    }

    #[test]
    fn test_indicators_calculation() {
        let mut sim = MarketSimulator::default();
        sim.generate(30);

        let indicators = sim.calculate_indicators();
        assert_eq!(indicators.len(), 30);

        let last = indicators.last().unwrap();
        assert!(last.rsi >= 0.0 && last.rsi <= 100.0);
        assert!(last.atr > 0.0);
    }

    #[test]
    fn test_regime_switching() {
        let config = SimulatorConfig {
            regime_persistence: 0.5, // Low persistence = frequent switching
            ..Default::default()
        };
        let mut sim = MarketSimulator::new(config, 42);

        let mut regimes = std::collections::HashSet::new();
        for _ in 0..100 {
            sim.next_candle();
            regimes.insert(format!("{:?}", sim.regime()));
        }

        // Should have seen multiple regimes
        assert!(regimes.len() > 1);
    }

    #[test]
    fn test_deterministic() {
        let mut sim1 = MarketSimulator::new(SimulatorConfig::default(), 12345);
        let mut sim2 = MarketSimulator::new(SimulatorConfig::default(), 12345);

        let candles1 = sim1.generate(10);
        let candles2 = sim2.generate(10);

        for (c1, c2) in candles1.iter().zip(candles2.iter()) {
            assert!((c1.close - c2.close).abs() < 1e-10);
        }
    }
}
