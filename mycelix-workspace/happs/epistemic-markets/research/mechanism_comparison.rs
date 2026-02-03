//! F4: Mechanism Design Research
//!
//! Addresses mechanism design questions:
//! - LMSR vs Order Book comparison
//! - Load testing harness
//! - Liquidity analysis
//! - Slippage measurements

use crate::metrics::{BrierScore, StatisticalComparison};
use crate::simulation::SimpleRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// MARKET MECHANISMS
// ============================================================================

/// Market mechanism types to compare
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MechanismType {
    /// Logarithmic Market Scoring Rule
    LMSR,
    /// Constant Product Market Maker (like Uniswap)
    CPMM,
    /// Order Book (continuous double auction)
    OrderBook,
    /// Parimutuel betting
    Parimutuel,
}

impl Default for MechanismType {
    fn default() -> Self {
        Self::LMSR
    }
}

// ============================================================================
// LMSR IMPLEMENTATION
// ============================================================================

/// LMSR market implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSRMarket {
    pub id: String,
    /// Liquidity parameter (b)
    pub liquidity_parameter: f64,
    /// Outcome quantities
    pub quantities: Vec<f64>,
    /// Subsidy/loss budget
    pub subsidy_pool: f64,
    /// Trade history
    pub trades: Vec<MarketTrade>,
    /// Current cost function value
    pub current_cost: f64,
}

impl LMSRMarket {
    pub fn new(id: String, liquidity: f64, num_outcomes: usize, subsidy: f64) -> Self {
        let quantities = vec![0.0; num_outcomes];
        let cost = Self::cost_function(&quantities, liquidity);

        Self {
            id,
            liquidity_parameter: liquidity,
            quantities,
            subsidy_pool: subsidy,
            trades: vec![],
            current_cost: cost,
        }
    }

    /// LMSR cost function: C(q) = b * ln(sum(e^(q_i/b)))
    fn cost_function(quantities: &[f64], b: f64) -> f64 {
        // Log-sum-exp for numerical stability
        let max_q = quantities.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sum_exp: f64 = quantities.iter().map(|q| ((q - max_q) / b).exp()).sum();
        b * (max_q / b + sum_exp.ln())
    }

    /// Get price for outcome i: p_i = e^(q_i/b) / sum(e^(q_j/b))
    pub fn price(&self, outcome: usize) -> f64 {
        if outcome >= self.quantities.len() {
            return 0.0;
        }

        let b = self.liquidity_parameter;
        let max_q = self.quantities.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let numerator = ((self.quantities[outcome] - max_q) / b).exp();
        let denominator: f64 = self
            .quantities
            .iter()
            .map(|q| ((q - max_q) / b).exp())
            .sum();

        numerator / denominator
    }

    /// Get all prices
    pub fn prices(&self) -> Vec<f64> {
        (0..self.quantities.len()).map(|i| self.price(i)).collect()
    }

    /// Execute a trade: buy `amount` shares of outcome `outcome`
    pub fn trade(&mut self, outcome: usize, amount: f64, trader_id: &str) -> TradeResult {
        if outcome >= self.quantities.len() {
            return TradeResult::invalid("Invalid outcome");
        }

        let price_before = self.price(outcome);
        let cost_before = self.current_cost;

        // Update quantity
        let mut new_quantities = self.quantities.clone();
        new_quantities[outcome] += amount;

        // Calculate new cost
        let cost_after = Self::cost_function(&new_quantities, self.liquidity_parameter);
        let trade_cost = cost_after - cost_before;

        // Apply trade
        self.quantities = new_quantities;
        self.current_cost = cost_after;

        let price_after = self.price(outcome);

        let trade = MarketTrade {
            trader_id: trader_id.to_string(),
            outcome,
            amount,
            cost: trade_cost,
            price_before,
            price_after,
            slippage: (price_after - price_before).abs(),
            timestamp: self.trades.len(),
        };

        self.trades.push(trade.clone());

        TradeResult::success(trade)
    }

    /// Calculate slippage for a hypothetical trade
    pub fn estimate_slippage(&self, outcome: usize, amount: f64) -> f64 {
        if outcome >= self.quantities.len() {
            return f64::NAN;
        }

        let price_before = self.price(outcome);

        let mut new_quantities = self.quantities.clone();
        new_quantities[outcome] += amount;

        let b = self.liquidity_parameter;
        let max_q = new_quantities.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let numerator = ((new_quantities[outcome] - max_q) / b).exp();
        let denominator: f64 = new_quantities.iter().map(|q| ((q - max_q) / b).exp()).sum();
        let price_after = numerator / denominator;

        (price_after - price_before).abs()
    }

    /// Maximum loss (subsidy needed)
    pub fn max_loss(&self) -> f64 {
        self.liquidity_parameter * (self.quantities.len() as f64).ln()
    }
}

// ============================================================================
// ORDER BOOK IMPLEMENTATION
// ============================================================================

/// Order book market implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookMarket {
    pub id: String,
    /// Bids (buy orders) - sorted by price descending
    pub bids: Vec<Order>,
    /// Asks (sell orders) - sorted by price ascending
    pub asks: Vec<Order>,
    /// Trade history
    pub trades: Vec<MarketTrade>,
    /// Number of outcomes
    pub num_outcomes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub trader_id: String,
    pub outcome: usize,
    pub side: OrderSide,
    pub price: f64,
    pub quantity: f64,
    pub filled: f64,
    pub timestamp: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OrderSide {
    Bid,
    Ask,
}

impl OrderBookMarket {
    pub fn new(id: String, num_outcomes: usize) -> Self {
        Self {
            id,
            bids: vec![],
            asks: vec![],
            trades: vec![],
            num_outcomes,
        }
    }

    /// Get mid price for an outcome
    pub fn mid_price(&self, outcome: usize) -> Option<f64> {
        let best_bid = self.best_bid(outcome)?;
        let best_ask = self.best_ask(outcome)?;
        Some((best_bid + best_ask) / 2.0)
    }

    /// Get best bid for outcome
    pub fn best_bid(&self, outcome: usize) -> Option<f64> {
        self.bids
            .iter()
            .filter(|o| o.outcome == outcome && o.quantity > o.filled)
            .map(|o| o.price)
            .fold(None, |acc, p| Some(acc.map_or(p, |a: f64| a.max(p))))
    }

    /// Get best ask for outcome
    pub fn best_ask(&self, outcome: usize) -> Option<f64> {
        self.asks
            .iter()
            .filter(|o| o.outcome == outcome && o.quantity > o.filled)
            .map(|o| o.price)
            .fold(None, |acc, p| Some(acc.map_or(p, |a: f64| a.min(p))))
    }

    /// Get spread
    pub fn spread(&self, outcome: usize) -> Option<f64> {
        let bid = self.best_bid(outcome)?;
        let ask = self.best_ask(outcome)?;
        Some(ask - bid)
    }

    /// Place a limit order
    pub fn place_order(&mut self, order: Order) -> Vec<MarketTrade> {
        let mut trades = Vec::new();
        let mut remaining = order.quantity;

        match order.side {
            OrderSide::Bid => {
                // Match against asks
                let matching_asks: Vec<_> = self
                    .asks
                    .iter_mut()
                    .filter(|a| a.outcome == order.outcome && a.price <= order.price)
                    .collect();

                for ask in matching_asks {
                    if remaining <= 0.0 {
                        break;
                    }

                    let available = ask.quantity - ask.filled;
                    let fill_amount = remaining.min(available);

                    ask.filled += fill_amount;
                    remaining -= fill_amount;

                    let trade = MarketTrade {
                        trader_id: order.trader_id.clone(),
                        outcome: order.outcome,
                        amount: fill_amount,
                        cost: fill_amount * ask.price,
                        price_before: ask.price,
                        price_after: ask.price,
                        slippage: 0.0, // Limit order has known price
                        timestamp: self.trades.len(),
                    };

                    self.trades.push(trade.clone());
                    trades.push(trade);
                }

                // Add remaining to order book
                if remaining > 0.0 {
                    let mut new_order = order.clone();
                    new_order.filled = order.quantity - remaining;
                    self.bids.push(new_order);
                    self.bids.sort_by(|a, b| {
                        b.price
                            .partial_cmp(&a.price)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }

            OrderSide::Ask => {
                // Match against bids
                let matching_bids: Vec<_> = self
                    .bids
                    .iter_mut()
                    .filter(|b| b.outcome == order.outcome && b.price >= order.price)
                    .collect();

                for bid in matching_bids {
                    if remaining <= 0.0 {
                        break;
                    }

                    let available = bid.quantity - bid.filled;
                    let fill_amount = remaining.min(available);

                    bid.filled += fill_amount;
                    remaining -= fill_amount;

                    let trade = MarketTrade {
                        trader_id: order.trader_id.clone(),
                        outcome: order.outcome,
                        amount: -fill_amount,
                        cost: -fill_amount * bid.price,
                        price_before: bid.price,
                        price_after: bid.price,
                        slippage: 0.0,
                        timestamp: self.trades.len(),
                    };

                    self.trades.push(trade.clone());
                    trades.push(trade);
                }

                // Add remaining to order book
                if remaining > 0.0 {
                    let mut new_order = order.clone();
                    new_order.filled = order.quantity - remaining;
                    self.asks.push(new_order);
                    self.asks.sort_by(|a, b| {
                        a.price
                            .partial_cmp(&b.price)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
        }

        // Clean up filled orders
        self.bids.retain(|o| o.filled < o.quantity);
        self.asks.retain(|o| o.filled < o.quantity);

        trades
    }

    /// Execute a market order
    pub fn market_order(&mut self, outcome: usize, amount: f64, trader_id: &str) -> TradeResult {
        let side = if amount > 0.0 {
            OrderSide::Bid
        } else {
            OrderSide::Ask
        };

        // Use extreme price for market order
        let price = if side == OrderSide::Bid { 1.0 } else { 0.0 };

        let order = Order {
            id: format!("mkt_{}", self.trades.len()),
            trader_id: trader_id.to_string(),
            outcome,
            side,
            price,
            quantity: amount.abs(),
            filled: 0.0,
            timestamp: self.trades.len(),
        };

        let trades = self.place_order(order);

        if trades.is_empty() {
            TradeResult::invalid("No matching orders")
        } else {
            // Aggregate trades
            let total_amount: f64 = trades.iter().map(|t| t.amount).sum();
            let total_cost: f64 = trades.iter().map(|t| t.cost).sum();
            let avg_price = if total_amount != 0.0 {
                total_cost / total_amount
            } else {
                0.0
            };

            let slippage = if !trades.is_empty() {
                (trades.last().unwrap().price_after - trades.first().unwrap().price_before).abs()
            } else {
                0.0
            };

            TradeResult::success(MarketTrade {
                trader_id: trader_id.to_string(),
                outcome,
                amount: total_amount,
                cost: total_cost,
                price_before: trades.first().map(|t| t.price_before).unwrap_or(0.0),
                price_after: trades.last().map(|t| t.price_after).unwrap_or(0.0),
                slippage,
                timestamp: self.trades.len(),
            })
        }
    }

    /// Total liquidity depth at a price level
    pub fn depth(&self, outcome: usize, price_range: f64) -> LiquidityDepth {
        let mid = self.mid_price(outcome).unwrap_or(0.5);

        let bid_depth: f64 = self
            .bids
            .iter()
            .filter(|o| o.outcome == outcome && o.price >= mid - price_range)
            .map(|o| o.quantity - o.filled)
            .sum();

        let ask_depth: f64 = self
            .asks
            .iter()
            .filter(|o| o.outcome == outcome && o.price <= mid + price_range)
            .map(|o| o.quantity - o.filled)
            .sum();

        LiquidityDepth {
            bid_depth,
            ask_depth,
            total_depth: bid_depth + ask_depth,
            price_range,
        }
    }
}

// ============================================================================
// COMMON TYPES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTrade {
    pub trader_id: String,
    pub outcome: usize,
    pub amount: f64,
    pub cost: f64,
    pub price_before: f64,
    pub price_after: f64,
    pub slippage: f64,
    pub timestamp: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeResult {
    pub success: bool,
    pub trade: Option<MarketTrade>,
    pub error: Option<String>,
}

impl TradeResult {
    fn success(trade: MarketTrade) -> Self {
        Self {
            success: true,
            trade: Some(trade),
            error: None,
        }
    }

    fn invalid(msg: &str) -> Self {
        Self {
            success: false,
            trade: None,
            error: Some(msg.to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityDepth {
    pub bid_depth: f64,
    pub ask_depth: f64,
    pub total_depth: f64,
    pub price_range: f64,
}

// ============================================================================
// MECHANISM COMPARISON STUDY
// ============================================================================

/// Configuration for mechanism comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MechanismComparisonConfig {
    /// Mechanisms to compare
    pub mechanisms: Vec<MechanismType>,
    /// Number of simulated markets
    pub num_markets: usize,
    /// Number of traders
    pub num_traders: usize,
    /// Trades per market
    pub trades_per_market: usize,
    /// LMSR liquidity parameters to test
    pub lmsr_liquidity_values: Vec<f64>,
    /// True probability range
    pub true_prob_range: (f64, f64),
    /// Trade size distribution (mean, std)
    pub trade_size_distribution: (f64, f64),
    /// Seed
    pub seed: u64,
}

impl Default for MechanismComparisonConfig {
    fn default() -> Self {
        Self {
            mechanisms: vec![MechanismType::LMSR, MechanismType::OrderBook],
            num_markets: 50,
            num_traders: 20,
            trades_per_market: 100,
            lmsr_liquidity_values: vec![10.0, 50.0, 100.0, 500.0],
            true_prob_range: (0.2, 0.8),
            trade_size_distribution: (5.0, 3.0),
            seed: 42,
        }
    }
}

/// Run mechanism comparison study
pub struct MechanismComparisonStudy {
    pub config: MechanismComparisonConfig,
    pub results: MechanismComparisonResults,
    rng: SimpleRng,
}

impl MechanismComparisonStudy {
    pub fn new(config: MechanismComparisonConfig) -> Self {
        let rng = SimpleRng::new(config.seed);
        Self {
            config,
            results: MechanismComparisonResults::new(),
            rng,
        }
    }

    /// Run the comparison study
    pub fn run(&mut self) -> &MechanismComparisonResults {
        // Test LMSR with different liquidity values
        for &liquidity in &self.config.lmsr_liquidity_values.clone() {
            self.run_lmsr_test(liquidity);
        }

        // Test Order Book
        self.run_order_book_test();

        // Calculate comparative metrics
        self.compare_mechanisms();

        // Generate summary
        self.results.summary = self.generate_summary();

        &self.results
    }

    fn run_lmsr_test(&mut self, liquidity: f64) {
        let mut metrics = MechanismMetrics {
            mechanism: MechanismType::LMSR,
            liquidity_parameter: Some(liquidity),
            ..Default::default()
        };

        for market_idx in 0..self.config.num_markets {
            let true_prob = self.config.true_prob_range.0
                + self.rng.next_u64() as f64 / u64::MAX as f64
                    * (self.config.true_prob_range.1 - self.config.true_prob_range.0);

            let mut market = LMSRMarket::new(
                format!("lmsr_{}_{}", liquidity, market_idx),
                liquidity,
                2,
                1000.0,
            );

            let mut market_slippages = Vec::new();
            let mut price_errors = Vec::new();

            for trade_idx in 0..self.config.trades_per_market {
                // Generate trader with some information about true prob
                let trader_signal = if (self.rng.next_u64() as f64 / u64::MAX as f64) < 0.3 {
                    // Informed trader
                    true_prob + (self.rng.next_u64() as f64 / u64::MAX as f64 - 0.5) * 0.1
                } else {
                    // Noise trader
                    self.rng.next_u64() as f64 / u64::MAX as f64
                };

                let current_price = market.price(0);
                let direction = if trader_signal > current_price { 1.0 } else { -1.0 };

                let (trade_mean, trade_std) = self.config.trade_size_distribution;
                let u1 = self.rng.next_u64() as f64 / u64::MAX as f64;
                let u2 = self.rng.next_u64() as f64 / u64::MAX as f64;
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let trade_size = (trade_mean + trade_std * z).max(1.0);

                // Execute trade
                let result = market.trade(
                    0,
                    direction * trade_size,
                    &format!("trader_{}", trade_idx % self.config.num_traders),
                );

                if let Some(trade) = result.trade {
                    market_slippages.push(trade.slippage);
                    price_errors.push((market.price(0) - true_prob).abs());
                }
            }

            // Record metrics
            if !market_slippages.is_empty() {
                metrics.avg_slippage +=
                    market_slippages.iter().sum::<f64>() / market_slippages.len() as f64;
                metrics.max_slippage = metrics
                    .max_slippage
                    .max(market_slippages.iter().fold(0.0, |a, &b| a.max(b)));
            }

            if !price_errors.is_empty() {
                metrics.price_accuracy +=
                    1.0 - price_errors.iter().sum::<f64>() / price_errors.len() as f64;
            }

            metrics.total_trades += market.trades.len();
        }

        // Average metrics
        let n = self.config.num_markets as f64;
        metrics.avg_slippage /= n;
        metrics.price_accuracy /= n;
        metrics.market_maker_loss = liquidity * 2.0_f64.ln() * n; // Theoretical max loss

        self.results
            .mechanism_metrics
            .insert(format!("LMSR_{}", liquidity), metrics);
    }

    fn run_order_book_test(&mut self) {
        let mut metrics = MechanismMetrics {
            mechanism: MechanismType::OrderBook,
            ..Default::default()
        };

        for market_idx in 0..self.config.num_markets {
            let true_prob = self.config.true_prob_range.0
                + self.rng.next_u64() as f64 / u64::MAX as f64
                    * (self.config.true_prob_range.1 - self.config.true_prob_range.0);

            let mut market = OrderBookMarket::new(format!("ob_{}", market_idx), 2);

            // Seed with some liquidity
            for i in 0..10 {
                let price = 0.1 + i as f64 * 0.08;
                market.place_order(Order {
                    id: format!("seed_bid_{}", i),
                    trader_id: "liquidity_provider".to_string(),
                    outcome: 0,
                    side: OrderSide::Bid,
                    price,
                    quantity: 10.0,
                    filled: 0.0,
                    timestamp: i,
                });
                market.place_order(Order {
                    id: format!("seed_ask_{}", i),
                    trader_id: "liquidity_provider".to_string(),
                    outcome: 0,
                    side: OrderSide::Ask,
                    price: price + 0.05,
                    quantity: 10.0,
                    filled: 0.0,
                    timestamp: i,
                });
            }

            let mut market_slippages = Vec::new();
            let mut price_errors = Vec::new();
            let mut trade_failures = 0usize;

            for trade_idx in 0..self.config.trades_per_market {
                let trader_signal = if (self.rng.next_u64() as f64 / u64::MAX as f64) < 0.3 {
                    true_prob + (self.rng.next_u64() as f64 / u64::MAX as f64 - 0.5) * 0.1
                } else {
                    self.rng.next_u64() as f64 / u64::MAX as f64
                };

                let mid_price = market.mid_price(0).unwrap_or(0.5);
                let direction = if trader_signal > mid_price { 1.0 } else { -1.0 };

                let (trade_mean, trade_std) = self.config.trade_size_distribution;
                let u1 = self.rng.next_u64() as f64 / u64::MAX as f64;
                let u2 = self.rng.next_u64() as f64 / u64::MAX as f64;
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let trade_size = (trade_mean + trade_std * z).max(1.0);

                let result = market.market_order(
                    0,
                    direction * trade_size,
                    &format!("trader_{}", trade_idx % self.config.num_traders),
                );

                if let Some(trade) = result.trade {
                    market_slippages.push(trade.slippage);
                    if let Some(mid) = market.mid_price(0) {
                        price_errors.push((mid - true_prob).abs());
                    }
                } else {
                    trade_failures += 1;
                }
            }

            // Record metrics
            if !market_slippages.is_empty() {
                metrics.avg_slippage +=
                    market_slippages.iter().sum::<f64>() / market_slippages.len() as f64;
                metrics.max_slippage = metrics
                    .max_slippage
                    .max(market_slippages.iter().fold(0.0, |a, &b| a.max(b)));
            }

            if !price_errors.is_empty() {
                metrics.price_accuracy +=
                    1.0 - price_errors.iter().sum::<f64>() / price_errors.len() as f64;
            }

            metrics.total_trades += market.trades.len();
            metrics.failed_trades += trade_failures;
        }

        let n = self.config.num_markets as f64;
        metrics.avg_slippage /= n;
        metrics.price_accuracy /= n;

        self.results
            .mechanism_metrics
            .insert("OrderBook".to_string(), metrics);
    }

    fn compare_mechanisms(&mut self) {
        // Compare each pair of mechanisms
        let keys: Vec<String> = self.results.mechanism_metrics.keys().cloned().collect();

        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                let m1 = &self.results.mechanism_metrics[&keys[i]];
                let m2 = &self.results.mechanism_metrics[&keys[j]];

                let comparison = MechanismComparison {
                    mechanism_a: keys[i].clone(),
                    mechanism_b: keys[j].clone(),
                    slippage_difference: m1.avg_slippage - m2.avg_slippage,
                    accuracy_difference: m1.price_accuracy - m2.price_accuracy,
                    trade_success_difference: m1.trade_success_rate() - m2.trade_success_rate(),
                    better_mechanism: if m1.overall_score() > m2.overall_score() {
                        keys[i].clone()
                    } else {
                        keys[j].clone()
                    },
                };

                self.results.comparisons.push(comparison);
            }
        }
    }

    fn generate_summary(&self) -> MechanismSummary {
        // Find best mechanism
        let best = self
            .results
            .mechanism_metrics
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.overall_score()
                    .partial_cmp(&b.overall_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| "Unknown".to_string());

        // Calculate optimal liquidity for LMSR
        let lmsr_metrics: Vec<(&String, &MechanismMetrics)> = self
            .results
            .mechanism_metrics
            .iter()
            .filter(|(k, _)| k.starts_with("LMSR"))
            .collect();

        let optimal_liquidity = lmsr_metrics
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.overall_score()
                    .partial_cmp(&b.overall_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .and_then(|(_, m)| m.liquidity_parameter);

        // LMSR vs OrderBook winner
        let lmsr_best = lmsr_metrics
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.overall_score()
                    .partial_cmp(&b.overall_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let ob_metrics = self.results.mechanism_metrics.get("OrderBook");

        let lmsr_better = match (lmsr_best, ob_metrics) {
            (Some((_, lmsr)), Some(ob)) => lmsr.overall_score() > ob.overall_score(),
            _ => true,
        };

        MechanismSummary {
            best_overall_mechanism: best,
            optimal_lmsr_liquidity: optimal_liquidity,
            lmsr_better_than_orderbook: lmsr_better,
            recommendations: self.generate_recommendations(),
        }
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Liquidity recommendation
        if let Some(optimal) = self.results.summary.optimal_lmsr_liquidity {
            recommendations.push(format!(
                "Recommended LMSR liquidity parameter: {:.0}",
                optimal
            ));
        }

        // Low liquidity scenarios
        let low_liq_metrics: Vec<_> = self
            .results
            .mechanism_metrics
            .iter()
            .filter(|(k, m)| k.starts_with("LMSR") && m.liquidity_parameter.unwrap_or(0.0) < 50.0)
            .collect();

        if !low_liq_metrics.is_empty() {
            let avg_slippage: f64 =
                low_liq_metrics.iter().map(|(_, m)| m.avg_slippage).sum::<f64>()
                    / low_liq_metrics.len() as f64;

            if avg_slippage > 0.05 {
                recommendations.push(
                    "Low liquidity causes high slippage - consider subsidizing markets".to_string(),
                );
            }
        }

        // Order book recommendations
        if let Some(ob) = self.results.mechanism_metrics.get("OrderBook") {
            if ob.failed_trades > 0 {
                let failure_rate = ob.failed_trades as f64
                    / (ob.total_trades + ob.failed_trades) as f64
                    * 100.0;
                recommendations.push(format!(
                    "Order book had {:.1}% failed trades - consider LMSR for thin markets",
                    failure_rate
                ));
            }
        }

        recommendations
    }
}

// ============================================================================
// RESULTS STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MechanismMetrics {
    pub mechanism: MechanismType,
    pub liquidity_parameter: Option<f64>,
    pub avg_slippage: f64,
    pub max_slippage: f64,
    pub price_accuracy: f64,
    pub total_trades: usize,
    pub failed_trades: usize,
    pub market_maker_loss: f64,
}

impl MechanismMetrics {
    pub fn trade_success_rate(&self) -> f64 {
        if self.total_trades + self.failed_trades == 0 {
            0.0
        } else {
            self.total_trades as f64 / (self.total_trades + self.failed_trades) as f64
        }
    }

    pub fn overall_score(&self) -> f64 {
        // Weighted combination: low slippage, high accuracy, high success rate
        let slippage_score = 1.0 - self.avg_slippage.min(1.0);
        let accuracy_score = self.price_accuracy;
        let success_score = self.trade_success_rate();

        0.3 * slippage_score + 0.4 * accuracy_score + 0.3 * success_score
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MechanismComparison {
    pub mechanism_a: String,
    pub mechanism_b: String,
    pub slippage_difference: f64,
    pub accuracy_difference: f64,
    pub trade_success_difference: f64,
    pub better_mechanism: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MechanismComparisonResults {
    pub mechanism_metrics: HashMap<String, MechanismMetrics>,
    pub comparisons: Vec<MechanismComparison>,
    pub summary: MechanismSummary,
}

impl MechanismComparisonResults {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MechanismSummary {
    pub best_overall_mechanism: String,
    pub optimal_lmsr_liquidity: Option<f64>,
    pub lmsr_better_than_orderbook: bool,
    pub recommendations: Vec<String>,
}

// ============================================================================
// LOAD TESTING
// ============================================================================

/// Load test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestConfig {
    /// Trades per second to simulate
    pub trades_per_second: Vec<usize>,
    /// Duration in seconds
    pub duration_seconds: usize,
    /// Mechanism to test
    pub mechanism: MechanismType,
    /// LMSR liquidity (if applicable)
    pub liquidity: f64,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            trades_per_second: vec![10, 50, 100, 500, 1000],
            duration_seconds: 10,
            mechanism: MechanismType::LMSR,
            liquidity: 100.0,
        }
    }
}

/// Load test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestResults {
    pub throughput_by_rate: HashMap<usize, ThroughputMetrics>,
    pub max_sustainable_rate: usize,
    pub degradation_point: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    pub target_rate: usize,
    pub achieved_rate: f64,
    pub avg_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub error_rate: f64,
}

/// Run load test
pub fn run_load_test(config: &LoadTestConfig) -> LoadTestResults {
    let mut results = LoadTestResults {
        throughput_by_rate: HashMap::new(),
        max_sustainable_rate: 0,
        degradation_point: usize::MAX,
    };

    let mut rng = SimpleRng::new(42);

    for &target_rate in &config.trades_per_second {
        let total_trades = target_rate * config.duration_seconds;
        let mut successful = 0;
        let mut latencies = Vec::new();

        let mut market = LMSRMarket::new("load_test".to_string(), config.liquidity, 2, 10000.0);

        for i in 0..total_trades {
            // Simulate trade
            let direction = if rng.next_u64() % 2 == 0 { 1.0 } else { -1.0 };
            let amount = 1.0 + rng.next_u64() as f64 / u64::MAX as f64 * 5.0;

            let start = i; // Simulated timestamp
            let result = market.trade(0, direction * amount, &format!("trader_{}", i % 100));

            if result.success {
                successful += 1;
                // Simulated latency (increases with load)
                let base_latency = 0.1;
                let load_factor = (market.trades.len() as f64 / 1000.0).min(10.0);
                latencies.push(base_latency * (1.0 + load_factor));
            }
        }

        let achieved_rate = successful as f64 / config.duration_seconds as f64;
        let error_rate = 1.0 - (successful as f64 / total_trades as f64);

        let avg_latency = if !latencies.is_empty() {
            latencies.iter().sum::<f64>() / latencies.len() as f64
        } else {
            0.0
        };

        let mut sorted_latencies = latencies.clone();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p99_latency = sorted_latencies
            .get((sorted_latencies.len() as f64 * 0.99) as usize)
            .copied()
            .unwrap_or(0.0);

        let metrics = ThroughputMetrics {
            target_rate,
            achieved_rate,
            avg_latency_ms: avg_latency,
            p99_latency_ms: p99_latency,
            error_rate,
        };

        // Check for degradation
        if achieved_rate >= target_rate as f64 * 0.95 {
            results.max_sustainable_rate = target_rate;
        } else if results.degradation_point == usize::MAX {
            results.degradation_point = target_rate;
        }

        results.throughput_by_rate.insert(target_rate, metrics);
    }

    results
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lmsr_price_initialization() {
        let market = LMSRMarket::new("test".to_string(), 100.0, 2, 1000.0);
        // Prices should be equal at initialization
        let p0 = market.price(0);
        let p1 = market.price(1);
        assert!((p0 - 0.5).abs() < 0.001);
        assert!((p1 - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_lmsr_trade() {
        let mut market = LMSRMarket::new("test".to_string(), 100.0, 2, 1000.0);
        let result = market.trade(0, 10.0, "trader1");

        assert!(result.success);
        assert!(market.price(0) > 0.5); // Price should increase
    }

    #[test]
    fn test_order_book_initialization() {
        let market = OrderBookMarket::new("test".to_string(), 2);
        assert!(market.bids.is_empty());
        assert!(market.asks.is_empty());
    }

    #[test]
    fn test_order_book_matching() {
        let mut market = OrderBookMarket::new("test".to_string(), 2);

        // Add ask
        market.place_order(Order {
            id: "ask1".to_string(),
            trader_id: "seller".to_string(),
            outcome: 0,
            side: OrderSide::Ask,
            price: 0.6,
            quantity: 10.0,
            filled: 0.0,
            timestamp: 0,
        });

        // Add matching bid
        let trades = market.place_order(Order {
            id: "bid1".to_string(),
            trader_id: "buyer".to_string(),
            outcome: 0,
            side: OrderSide::Bid,
            price: 0.7,
            quantity: 5.0,
            filled: 0.0,
            timestamp: 1,
        });

        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].amount, 5.0);
    }

    #[test]
    fn test_mechanism_metrics_score() {
        let metrics = MechanismMetrics {
            mechanism: MechanismType::LMSR,
            liquidity_parameter: Some(100.0),
            avg_slippage: 0.02,
            max_slippage: 0.1,
            price_accuracy: 0.85,
            total_trades: 100,
            failed_trades: 0,
            market_maker_loss: 50.0,
        };

        let score = metrics.overall_score();
        assert!(score > 0.5);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_small_comparison_study() {
        let config = MechanismComparisonConfig {
            num_markets: 2,
            trades_per_market: 10,
            lmsr_liquidity_values: vec![50.0],
            ..Default::default()
        };

        let mut study = MechanismComparisonStudy::new(config);
        let results = study.run();

        assert!(!results.mechanism_metrics.is_empty());
    }
}
