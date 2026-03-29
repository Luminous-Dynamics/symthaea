// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Economic simulation tests for Mycelix finance.
//!
//! These simulate multi-agent economic behavior using the pure types from lib.rs.
//! No HDK dependency — runs as standard Rust tests.

#[cfg(test)]
mod economic_simulations {
    use crate::*;

    // =========================================================================
    // Simulation infrastructure
    // =========================================================================

    /// A simulated agent with SAP balance and TEND balance.
    #[allow(dead_code)]
    struct SimAgent {
        did: String,
        sap_balance: u64,
        tend_balance: i32,
        mycel_score: f64,
    }

    impl SimAgent {
        fn new(id: usize, initial_sap: u64) -> Self {
            Self {
                did: format!("did:mycelix:agent_{}", id),
                sap_balance: initial_sap,
                tend_balance: 0,
                mycel_score: 0.5, // default Member tier
            }
        }
    }

    /// Simple PRNG (xorshift64) for deterministic simulations without external deps.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn next_f64(&mut self) -> f64 {
            (self.next_u64() % 1_000_000) as f64 / 1_000_000.0
        }
        fn next_usize(&mut self, max: usize) -> usize {
            (self.next_u64() as usize) % max
        }
    }

    // =========================================================================
    // Simulation 1: Demurrage Impact Over 1 Year
    // =========================================================================

    /// Simulate 100 agents over 365 days with 2% annual SAP demurrage.
    /// Verify:
    /// - Total SAP decreases due to demurrage (compost collected)
    /// - No individual balance goes negative
    /// - Exempt floor protects small balances
    /// - Agents above exempt floor lose ~2% annually
    #[test]
    fn sim_demurrage_impact_100_agents_1_year() {
        let n_agents = 100;
        let days = 365;
        let initial_sap = 10_000_000_000u64; // 10,000 SAP per agent
        let mut agents: Vec<SimAgent> = (0..n_agents)
            .map(|i| SimAgent::new(i, initial_sap))
            .collect();

        let mut total_compost = 0u64;
        let seconds_per_day = 86_400u64;

        for _day in 0..days {
            for agent in agents.iter_mut() {
                let deduction = compute_demurrage_deduction(
                    agent.sap_balance,
                    DEMURRAGE_EXEMPT_FLOOR,
                    DEMURRAGE_RATE,
                    seconds_per_day,
                );
                agent.sap_balance = agent.sap_balance.saturating_sub(deduction);
                total_compost += deduction;
            }
        }

        // Verify no negative balances
        for agent in &agents {
            assert!(
                agent.sap_balance > 0,
                "Agent {} balance went to 0",
                agent.did
            );
        }

        // Verify total compost collected is approximately 2% of eligible balance
        let total_initial: u64 = n_agents as u64 * initial_sap;
        let total_remaining: u64 = agents.iter().map(|a| a.sap_balance).sum();
        let total_loss = total_initial - total_remaining;

        assert_eq!(
            total_loss, total_compost,
            "Conservation: loss must equal compost"
        );

        // Each agent started with 10,000 SAP, exempt floor is 1,000 SAP.
        // Eligible = 9,000 SAP. After 1 year at 2%, should lose ~2% of eligible.
        // But demurrage is applied daily on declining balance, so actual loss < 2%.
        // Expected: ~1.98% of eligible (exponential decay).
        let eligible_initial = initial_sap - DEMURRAGE_EXEMPT_FLOOR;
        let expected_remaining = eligible_initial as f64 * (-DEMURRAGE_RATE * 1.0f64).exp();
        let actual_remaining = agents[0].sap_balance - DEMURRAGE_EXEMPT_FLOOR;
        let error_pct =
            ((actual_remaining as f64 - expected_remaining) / expected_remaining * 100.0).abs();

        // Allow 1% error due to daily discretization
        assert!(
            error_pct < 1.0,
            "Demurrage error too large: {:.2}% (expected remaining ~{:.0}, got {})",
            error_pct,
            expected_remaining,
            actual_remaining
        );
    }

    /// Verify that agents below the exempt floor (1,000 SAP) pay no demurrage.
    #[test]
    fn sim_demurrage_exempt_floor_protects_small_balances() {
        let small_balance = DEMURRAGE_EXEMPT_FLOOR; // exactly at floor
        let mut agent = SimAgent::new(0, small_balance);

        for _ in 0..365 {
            let deduction = compute_demurrage_deduction(
                agent.sap_balance,
                DEMURRAGE_EXEMPT_FLOOR,
                DEMURRAGE_RATE,
                86_400,
            );
            agent.sap_balance = agent.sap_balance.saturating_sub(deduction);
        }

        assert_eq!(
            agent.sap_balance, small_balance,
            "Balance at exempt floor should not decay"
        );
    }

    /// Verify demurrage doesn't reduce balance below exempt floor.
    #[test]
    fn sim_demurrage_never_below_exempt_floor() {
        let just_above = DEMURRAGE_EXEMPT_FLOOR + 1_000_000; // floor + 1 SAP
        let mut agent = SimAgent::new(0, just_above);

        // Run for 100 years — should converge to floor, never below
        for _ in 0..(365 * 100) {
            let deduction = compute_demurrage_deduction(
                agent.sap_balance,
                DEMURRAGE_EXEMPT_FLOOR,
                DEMURRAGE_RATE,
                86_400,
            );
            agent.sap_balance = agent.sap_balance.saturating_sub(deduction);
            assert!(
                agent.sap_balance >= DEMURRAGE_EXEMPT_FLOOR,
                "Balance {} dropped below exempt floor {}",
                agent.sap_balance,
                DEMURRAGE_EXEMPT_FLOOR
            );
        }
    }

    // =========================================================================
    // Simulation 2: TEND Zero-Sum Invariant Under Random Exchanges
    // =========================================================================

    /// Simulate 50 agents making 10,000 random TEND exchanges.
    /// Verify the zero-sum invariant holds after every exchange.
    #[test]
    fn sim_tend_zero_sum_10000_exchanges() {
        let n_agents = 50;
        let n_exchanges = 10_000;
        let mut agents: Vec<SimAgent> = (0..n_agents).map(|i| SimAgent::new(i, 0)).collect();
        let mut rng = Rng::new(42);
        let limit = TendLimitTier::Normal.limit(); // ±40

        let mut successful_exchanges = 0u32;

        for _ in 0..n_exchanges {
            let provider = rng.next_usize(n_agents);
            let mut receiver = rng.next_usize(n_agents);
            while receiver == provider {
                receiver = rng.next_usize(n_agents);
            }
            let hours = (rng.next_u64() % 8 + 1) as i32; // 1-8 hours

            // Check limits
            let provider_new = agents[provider].tend_balance - hours;
            let receiver_new = agents[receiver].tend_balance + hours;

            if provider_new >= -limit
                && provider_new <= limit
                && receiver_new >= -limit
                && receiver_new <= limit
            {
                agents[provider].tend_balance -= hours;
                agents[receiver].tend_balance += hours;
                successful_exchanges += 1;
            }

            // Verify zero-sum after every exchange
            let total: i32 = agents.iter().map(|a| a.tend_balance).sum();
            assert_eq!(
                total, 0,
                "Zero-sum violated after exchange {}",
                successful_exchanges
            );
        }

        assert!(
            successful_exchanges > 1000,
            "Too few successful exchanges: {}",
            successful_exchanges
        );

        // Final zero-sum check
        let total: i32 = agents.iter().map(|a| a.tend_balance).sum();
        assert_eq!(total, 0, "Final zero-sum invariant violated");
    }

    /// Test TEND under stress: Emergency tier (±120 limits) with aggressive exchanges.
    #[test]
    fn sim_tend_zero_sum_emergency_tier() {
        let n_agents = 20;
        let n_exchanges = 5_000;
        let mut agents: Vec<SimAgent> = (0..n_agents).map(|i| SimAgent::new(i, 0)).collect();
        let mut rng = Rng::new(137);
        let limit = TendLimitTier::Emergency.limit(); // ±120

        for _ in 0..n_exchanges {
            let provider = rng.next_usize(n_agents);
            let mut receiver = rng.next_usize(n_agents);
            while receiver == provider {
                receiver = rng.next_usize(n_agents);
            }
            let hours = (rng.next_u64() % 16 + 1) as i32; // 1-16 hours (larger exchanges)

            let provider_new = agents[provider].tend_balance - hours;
            let receiver_new = agents[receiver].tend_balance + hours;

            if provider_new >= -limit && receiver_new <= limit {
                agents[provider].tend_balance -= hours;
                agents[receiver].tend_balance += hours;
            }
        }

        let total: i32 = agents.iter().map(|a| a.tend_balance).sum();
        assert_eq!(total, 0, "Emergency-tier zero-sum invariant violated");
    }

    // =========================================================================
    // Simulation 3: Collateral Stress Test (50% Price Crash)
    // =========================================================================

    /// Simulate 50 collateral positions with a 50% price crash.
    /// Verify LTV monitoring correctly identifies endangered positions.
    #[test]
    fn sim_collateral_stress_50pct_crash() {
        let n_positions = 50;
        let mut rng = Rng::new(99);

        #[allow(dead_code)]
        struct Position {
            initial_value: u64,
            obligation: u64,
            initial_ltv: f64,
        }

        // Create positions with varying LTV ratios (30% to 75%)
        let mut positions: Vec<Position> = Vec::new();
        for _ in 0..n_positions {
            let value = 1_000_000 + (rng.next_u64() % 9_000_000); // 1M-10M
            let ltv_target = 0.30 + rng.next_f64() * 0.45; // 30%-75%
            let obligation = (value as f64 * ltv_target) as u64;
            positions.push(Position {
                initial_value: value,
                obligation,
                initial_ltv: ltv_target,
            });
        }

        // Apply 50% crash
        let crash_factor = 0.50;
        let mut healthy = 0u32;
        let mut warning = 0u32;
        let mut margin_call = 0u32;
        let mut liquidation = 0u32;

        for pos in &positions {
            let crashed_value = (pos.initial_value as f64 * crash_factor) as u64;
            let new_ltv = if crashed_value > 0 {
                pos.obligation as f64 / crashed_value as f64
            } else {
                f64::INFINITY
            };
            let status = CollateralHealthStatus::from_ltv(new_ltv);
            match status {
                CollateralHealthStatus::Healthy => healthy += 1,
                CollateralHealthStatus::Warning => warning += 1,
                CollateralHealthStatus::MarginCall => margin_call += 1,
                CollateralHealthStatus::Liquidation => liquidation += 1,
            }
        }

        // After a 50% crash, positions that were at 50%+ LTV should now be at 100%+ LTV
        // Positions at 40-47.5% → Warning (80-95%)
        // Positions at 47.5-50% → MarginCall/Liquidation
        // Positions below 40% → Healthy (still under 80%)

        let total = healthy + warning + margin_call + liquidation;
        assert_eq!(total, n_positions as u32);

        // With LTVs uniformly distributed in [30%, 75%] and a 50% crash:
        // New LTVs = old * 2, so range [60%, 150%]
        // Healthy (<80%): old LTV < 40% → about 22% of positions
        // Liquidation (>95%): old LTV > 47.5% → about 61% of positions
        assert!(liquidation > 0, "50% crash should cause some liquidations");
        assert!(
            healthy > 0,
            "Some conservative positions should survive 50% crash"
        );

        // Print distribution for manual inspection
        let _ = (healthy, warning, margin_call, liquidation); // used in asserts above
    }

    /// Verify LTV thresholds are correctly ordered.
    #[test]
    fn sim_ltv_thresholds_ordered() {
        assert!(LTV_WARNING_THRESHOLD < LTV_MARGIN_CALL_THRESHOLD);
        assert!(LTV_MARGIN_CALL_THRESHOLD < LTV_LIQUIDATION_THRESHOLD);
        assert!(LTV_LIQUIDATION_THRESHOLD < 1.0); // liquidation before insolvency
    }

    // =========================================================================
    // Simulation 4: Diversification Bonus Incentive Analysis
    // =========================================================================

    /// Verify that diversification bonus creates meaningful incentive.
    /// Compare: single-asset position vs. diversified position with same total value.
    #[test]
    fn sim_diversification_incentive() {
        // Single asset: 1M in real estate
        let single = vec![CollateralComponent {
            asset_id: "re-1".into(),
            asset_class: CollateralAssetClass::RealEstate,
            weight: 1.0,
            current_value: 1_000_000,
        }];

        // Diversified: 250K each in 4 asset classes
        let diversified = vec![
            CollateralComponent {
                asset_id: "re-1".into(),
                asset_class: CollateralAssetClass::RealEstate,
                weight: 0.25,
                current_value: 250_000,
            },
            CollateralComponent {
                asset_id: "ec-1".into(),
                asset_class: CollateralAssetClass::EnergyCertificate,
                weight: 0.25,
                current_value: 250_000,
            },
            CollateralComponent {
                asset_id: "ag-1".into(),
                asset_class: CollateralAssetClass::AgriculturalAsset,
                weight: 0.25,
                current_value: 250_000,
            },
            CollateralComponent {
                asset_id: "cc-1".into(),
                asset_class: CollateralAssetClass::CarbonCredit,
                weight: 0.25,
                current_value: 250_000,
            },
        ];

        let single_bonus = compute_diversification_bonus(&single);
        let diverse_bonus = compute_diversification_bonus(&diversified);

        assert_eq!(single_bonus, 0.0);
        assert!((diverse_bonus - 0.15).abs() < f64::EPSILON); // 3 bonus classes * 5%

        // With 80% base LTV threshold, diversification bonus effectively lowers it:
        // Single: liquidation at 95% LTV
        // Diverse: effective LTV = actual_ltv * (1.0 - bonus) = actual_ltv * 0.85
        // So liquidation at 95% / 0.85 = ~111.8% actual LTV — significantly more headroom
        let base_liquidation = LTV_LIQUIDATION_THRESHOLD;
        let effective_liquidation_single = base_liquidation / (1.0 - single_bonus);
        let effective_liquidation_diverse = base_liquidation / (1.0 - diverse_bonus);

        assert!(
            effective_liquidation_diverse > effective_liquidation_single,
            "Diversification should increase effective liquidation threshold"
        );

        // The bonus gives ~17.6% more headroom
        let headroom_increase = (effective_liquidation_diverse - effective_liquidation_single)
            / effective_liquidation_single
            * 100.0;
        assert!(
            headroom_increase > 15.0 && headroom_increase < 20.0,
            "Headroom increase should be ~17.6%, got {:.1}%",
            headroom_increase
        );
    }

    /// Verify diminishing returns: going from 4 to 5 asset classes gives
    /// the same 5% bonus, but overall position may not justify the complexity.
    #[test]
    fn sim_diversification_marginal_returns() {
        let mut components = vec![CollateralComponent {
            asset_id: "a".into(),
            asset_class: CollateralAssetClass::RealEstate,
            weight: 1.0,
            current_value: 1000,
        }];

        let bonuses: Vec<f64> = (0..6)
            .map(|i| {
                if i > 0 {
                    let class = match i {
                        1 => CollateralAssetClass::EnergyCertificate,
                        2 => CollateralAssetClass::AgriculturalAsset,
                        3 => CollateralAssetClass::CarbonCredit,
                        4 => CollateralAssetClass::Vehicle,
                        _ => CollateralAssetClass::Equipment,
                    };
                    components.push(CollateralComponent {
                        asset_id: format!("asset_{}", i),
                        asset_class: class,
                        weight: 1.0,
                        current_value: 1000,
                    });
                }
                compute_diversification_bonus(&components)
            })
            .collect();

        // Verify: 0%, 5%, 10%, 15%, 20%, 20% (capped)
        assert_eq!(bonuses[0], 0.0);
        assert!((bonuses[1] - 0.05).abs() < f64::EPSILON);
        assert!((bonuses[2] - 0.10).abs() < f64::EPSILON);
        assert!((bonuses[3] - 0.15).abs() < f64::EPSILON);
        assert!((bonuses[4] - 0.20).abs() < f64::EPSILON);
        assert!((bonuses[5] - 0.20).abs() < f64::EPSILON); // capped
    }

    // =========================================================================
    // Simulation 5: Minted Currency Demurrage + Zero-Sum
    // =========================================================================

    /// Simulate a community currency with 3% demurrage over 6 months.
    /// Verify zero-sum is maintained even after demurrage deductions.
    #[test]
    fn sim_minted_currency_demurrage_zero_sum() {
        let n_agents = 30;
        let days = 180;
        let demurrage_rate = 0.03; // 3% annual
        let credit_limit = 100i32;
        let mut balances: Vec<i32> = vec![0; n_agents];
        let mut total_compost = 0i32;
        let mut rng = Rng::new(2026);

        // Phase 1: Random exchanges (build up balances)
        for _ in 0..500 {
            let provider = rng.next_usize(n_agents);
            let mut receiver = rng.next_usize(n_agents);
            while receiver == provider {
                receiver = rng.next_usize(n_agents);
            }
            let hours = (rng.next_u64() % 4 + 1) as i32;

            if balances[provider] - hours >= -credit_limit
                && balances[receiver] + hours <= credit_limit
            {
                balances[provider] -= hours;
                balances[receiver] += hours;
            }
        }

        // Phase 2: Apply daily demurrage for 6 months
        for _ in 0..days {
            for balance in balances.iter_mut() {
                let deduction = compute_minted_demurrage(*balance, demurrage_rate, 86_400);
                if deduction > 0 {
                    *balance -= deduction;
                    total_compost += deduction;
                }
            }
        }

        // Zero-sum check: balances + compost should equal 0
        let balance_sum: i32 = balances.iter().sum();
        // Note: in the real system, compost goes to commons pool (positive)
        // and is deducted from individual balances. So:
        // sum(balances) + total_compost = 0 (approximately, due to rounding)
        // Actually, initially sum(balances) = 0 (zero-sum exchanges).
        // After demurrage: positive balances decrease, but negative balances don't.
        // So sum(balances) becomes negative by total_compost.
        assert_eq!(
            balance_sum + total_compost,
            0,
            "Minted currency zero-sum violated: balances={}, compost={}",
            balance_sum,
            total_compost
        );
    }

    // =========================================================================
    // Simulation 6: Blended Oracle Rate Stability
    // =========================================================================

    /// Simulate oracle rate blending over 100 price updates.
    /// Verify the blended rate is smoother than the raw external feed.
    #[test]
    fn sim_oracle_blending_smoothness() {
        let community_rate = 100.0; // stable community consensus
        let mut rng = Rng::new(555);

        let mut blended_rates = Vec::new();
        let mut raw_rates = Vec::new();

        for _ in 0..100 {
            // External rate: noisy, ±20% around community rate
            let noise = (rng.next_f64() - 0.5) * 40.0; // -20 to +20
            let external = community_rate + noise;
            let confidence = 0.6 + rng.next_f64() * 0.3; // 0.6-0.9

            let blended = compute_blended_oracle_rate(community_rate, Some(external), confidence);
            blended_rates.push(blended);
            raw_rates.push(external);
        }

        // Compute variance of blended vs raw
        let raw_mean: f64 = raw_rates.iter().sum::<f64>() / raw_rates.len() as f64;
        let raw_variance: f64 = raw_rates
            .iter()
            .map(|r| (r - raw_mean).powi(2))
            .sum::<f64>()
            / raw_rates.len() as f64;

        let blended_mean: f64 = blended_rates.iter().sum::<f64>() / blended_rates.len() as f64;
        let blended_variance: f64 = blended_rates
            .iter()
            .map(|r| (r - blended_mean).powi(2))
            .sum::<f64>()
            / blended_rates.len() as f64;

        // Blended should have significantly lower variance
        assert!(
            blended_variance < raw_variance * 0.5,
            "Blended variance ({:.2}) should be < 50% of raw variance ({:.2})",
            blended_variance,
            raw_variance
        );

        // Blended mean should be closer to community rate
        assert!(
            (blended_mean - community_rate).abs() < (raw_mean - community_rate).abs() + 1.0,
            "Blended mean ({:.2}) should be closer to community rate ({}) than raw mean ({:.2})",
            blended_mean,
            community_rate,
            raw_mean
        );
    }

    // =========================================================================
    // Simulation 7: Fee Tier Distribution
    // =========================================================================

    /// Verify fee tier boundaries produce expected distributions.
    #[test]
    fn sim_fee_tier_distribution() {
        let mut newcomer = 0u32;
        let mut member = 0u32;
        let mut steward = 0u32;

        // Simulate 1000 agents with uniformly distributed MYCEL scores
        for i in 0..1000 {
            let score = i as f64 / 1000.0;
            match FeeTier::from_mycel(score) {
                FeeTier::Newcomer => newcomer += 1,
                FeeTier::Member => member += 1,
                FeeTier::Steward => steward += 1,
            }
        }

        // Newcomer: score < 0.3 → 30% of agents
        // Member: 0.3 <= score <= 0.7 → 40% of agents
        // Steward: score > 0.7 → 30% of agents (actually 29.9% due to exclusive >)
        assert_eq!(newcomer, 300); // 0.000 to 0.299
        assert_eq!(member, 401); // 0.300 to 0.700
        assert_eq!(steward, 299); // 0.701 to 0.999

        // Verify fee rates are monotonically decreasing
        assert!(FeeTier::Newcomer.base_fee_rate() > FeeTier::Member.base_fee_rate());
        assert!(FeeTier::Member.base_fee_rate() > FeeTier::Steward.base_fee_rate());
    }
}
