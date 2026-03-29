// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Resilience tests for Mycelix finance types.
//!
//! Tests failure modes, boundary conditions, and degraded operation scenarios.
//! Pure Rust — no HDK dependency.

#[cfg(test)]
mod resilience_tests {
    use crate::*;

    // =========================================================================
    // Oracle Failure Scenarios
    // =========================================================================

    /// When external oracle returns NaN/Inf/negative, blended rate must fall back to community.
    #[test]
    fn resilience_oracle_malicious_external_feed() {
        let community = 100.0;
        let malicious_values = [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -1.0,
            0.0,
            -0.001,
        ];

        for &val in &malicious_values {
            let rate = compute_blended_oracle_rate(community, Some(val), 0.95);
            assert_eq!(
                rate, community,
                "Malicious external value {:?} should fall back to community rate",
                val
            );
        }
    }

    /// When external oracle has zero confidence, community rate is sole authority.
    #[test]
    fn resilience_oracle_zero_confidence() {
        let rate = compute_blended_oracle_rate(100.0, Some(999.0), 0.0);
        assert_eq!(
            rate, 100.0,
            "Zero confidence should ignore external feed entirely"
        );
    }

    /// When community rate itself is extreme, blended rate shouldn't amplify it.
    #[test]
    fn resilience_oracle_extreme_community_rate() {
        // Very high community rate
        let rate = compute_blended_oracle_rate(1_000_000.0, Some(500_000.0), 0.8);
        assert!(
            rate > 500_000.0 && rate <= 1_000_000.0,
            "Blended rate {} should be between external and community",
            rate
        );

        // Very small community rate
        let rate = compute_blended_oracle_rate(0.001, Some(0.002), 0.8);
        assert!(
            rate > 0.0 && rate.is_finite(),
            "Small community rate should still produce valid blended rate"
        );
    }

    // =========================================================================
    // Demurrage Edge Cases
    // =========================================================================

    /// Demurrage with maximum possible elapsed time (100 years).
    #[test]
    fn resilience_demurrage_century_elapsed() {
        let balance = u64::MAX / 2;
        let century_seconds = 100 * 365 * 86_400;
        let deduction = compute_demurrage_deduction(
            balance,
            DEMURRAGE_EXEMPT_FLOOR,
            DEMURRAGE_RATE,
            century_seconds,
        );
        let eligible = balance - DEMURRAGE_EXEMPT_FLOOR;
        assert!(
            deduction <= eligible,
            "Century deduction {} exceeds eligible {}",
            deduction,
            eligible
        );
        // After 100 years at 2%, should converge very close to eligible
        assert!(
            deduction as f64 / eligible as f64 > 0.85,
            "Century decay should remove most of eligible balance"
        );
    }

    /// Demurrage at exact exempt floor boundary.
    #[test]
    fn resilience_demurrage_exact_floor() {
        // Balance == floor: no deduction
        assert_eq!(
            compute_demurrage_deduction(
                DEMURRAGE_EXEMPT_FLOOR,
                DEMURRAGE_EXEMPT_FLOOR,
                DEMURRAGE_RATE,
                86_400
            ),
            0
        );
        // Balance == floor + 1: tiny deduction possible
        let d = compute_demurrage_deduction(
            DEMURRAGE_EXEMPT_FLOOR + 1,
            DEMURRAGE_EXEMPT_FLOOR,
            DEMURRAGE_RATE,
            86_400,
        );
        assert!(d <= 1, "1 micro-SAP above floor should deduct at most 1");
    }

    /// Demurrage with 1-second elapsed time.
    #[test]
    fn resilience_demurrage_one_second() {
        let balance = 10_000_000_000u64; // 10,000 SAP
        let d = compute_demurrage_deduction(balance, DEMURRAGE_EXEMPT_FLOOR, DEMURRAGE_RATE, 1);
        // 2% annual on 9000 SAP for 1 second: ~0.00057 SAP = 570 micro-SAP
        assert!(d < 1000, "1-second demurrage should be tiny, got {}", d);
        assert!(
            d > 0,
            "1-second demurrage on large balance should be non-zero"
        );
    }

    // =========================================================================
    // Mint Cap Resilience
    // =========================================================================

    /// Mint cap counter handles overflow gracefully.
    #[test]
    fn resilience_mint_cap_near_overflow() {
        let counter = SapMintCapCounter {
            period_start_micros: 0,
            cumulative_minted: u64::MAX - 10,
            mint_count: u32::MAX,
            last_updated_micros: 0,
        };
        // Would overflow without saturating_add
        assert!(counter.would_exceed_cap(100));
        // Remaining capacity should be tiny
        assert_eq!(
            counter.remaining_capacity(),
            SAP_MINT_ANNUAL_MAX.saturating_sub(u64::MAX - 10)
        );
    }

    /// Period expiry at timestamp boundaries.
    #[test]
    fn resilience_mint_cap_timestamp_edge() {
        let counter = SapMintCapCounter {
            period_start_micros: i64::MAX - 1000,
            cumulative_minted: 0,
            mint_count: 0,
            last_updated_micros: 0,
        };
        // now < period_start (wrapping): should not be expired
        assert!(!counter.is_period_expired(0));
        // Handles gracefully without panic
        assert!(!counter.is_period_expired(i64::MAX));
    }

    // =========================================================================
    // Collateral Health Under Stress
    // =========================================================================

    /// LTV at exact threshold boundaries.
    #[test]
    fn resilience_ltv_exact_boundaries() {
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.80),
            CollateralHealthStatus::Healthy
        );
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.800001),
            CollateralHealthStatus::Warning
        );
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.90),
            CollateralHealthStatus::Warning
        );
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.900001),
            CollateralHealthStatus::MarginCall
        );
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.95),
            CollateralHealthStatus::MarginCall
        );
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.950001),
            CollateralHealthStatus::Liquidation
        );
    }

    /// LTV with extreme values.
    #[test]
    fn resilience_ltv_extreme_values() {
        // Negative LTV (shouldn't happen, but handle gracefully)
        assert_eq!(
            CollateralHealthStatus::from_ltv(-1.0),
            CollateralHealthStatus::Healthy
        );
        // Zero LTV (no obligation)
        assert_eq!(
            CollateralHealthStatus::from_ltv(0.0),
            CollateralHealthStatus::Healthy
        );
        // Over 100% LTV (underwater)
        assert_eq!(
            CollateralHealthStatus::from_ltv(1.5),
            CollateralHealthStatus::Liquidation
        );
        // Extreme LTV
        assert_eq!(
            CollateralHealthStatus::from_ltv(100.0),
            CollateralHealthStatus::Liquidation
        );
    }

    /// LTV with NaN/Inf (should still return a valid status).
    #[test]
    fn resilience_ltv_nan_inf() {
        // NaN: 0.95 < NaN is false, 0.90 < NaN is false, 0.80 < NaN is false
        // So from_ltv(NaN) = Healthy (all comparisons return false)
        let status = CollateralHealthStatus::from_ltv(f64::NAN);
        // NaN behavior: all > comparisons return false, so falls through to Healthy
        assert_eq!(status, CollateralHealthStatus::Healthy);

        // Infinity: all comparisons true → Liquidation
        assert_eq!(
            CollateralHealthStatus::from_ltv(f64::INFINITY),
            CollateralHealthStatus::Liquidation
        );
    }

    // =========================================================================
    // Diversification Edge Cases
    // =========================================================================

    /// Diversification with 100 components of the same class.
    #[test]
    fn resilience_diversification_many_same_class() {
        let components: Vec<CollateralComponent> = (0..100)
            .map(|i| CollateralComponent {
                asset_id: format!("asset_{}", i),
                asset_class: CollateralAssetClass::RealEstate,
                weight: 0.01,
                current_value: 10_000,
            })
            .collect();

        assert_eq!(
            compute_diversification_bonus(&components),
            0.0,
            "100 assets of same class should get zero bonus"
        );
    }

    /// Diversification with all 8 asset classes.
    #[test]
    fn resilience_diversification_all_classes() {
        let classes = vec![
            CollateralAssetClass::RealEstate,
            CollateralAssetClass::Cryptocurrency,
            CollateralAssetClass::EnergyCertificate,
            CollateralAssetClass::AgriculturalAsset,
            CollateralAssetClass::CarbonCredit,
            CollateralAssetClass::Vehicle,
            CollateralAssetClass::Equipment,
            CollateralAssetClass::Other,
        ];
        let components: Vec<CollateralComponent> = classes
            .into_iter()
            .map(|c| CollateralComponent {
                asset_id: "test".into(),
                asset_class: c,
                weight: 0.125,
                current_value: 1000,
            })
            .collect();

        // 8 classes → 7 bonus classes * 5% = 35%, capped at 20%
        assert!(
            (compute_diversification_bonus(&components) - DIVERSIFICATION_BONUS_CAP).abs()
                < f64::EPSILON
        );
    }

    // =========================================================================
    // Currency Status Transition Resilience
    // =========================================================================

    /// Verify all invalid transitions are correctly rejected.
    #[test]
    fn resilience_currency_status_invalid_transitions() {
        // Retired is terminal
        assert!(!CurrencyStatus::Retired.can_transition_to(&CurrencyStatus::Active));
        assert!(!CurrencyStatus::Retired.can_transition_to(&CurrencyStatus::Suspended));
        assert!(!CurrencyStatus::Retired.can_transition_to(&CurrencyStatus::Draft));

        // Draft can only go to Active
        assert!(!CurrencyStatus::Draft.can_transition_to(&CurrencyStatus::Suspended));
        assert!(!CurrencyStatus::Draft.can_transition_to(&CurrencyStatus::Retired));

        // Self-transitions are invalid
        assert!(!CurrencyStatus::Active.can_transition_to(&CurrencyStatus::Active));
        assert!(!CurrencyStatus::Draft.can_transition_to(&CurrencyStatus::Draft));
    }

    /// Verify all valid transitions work.
    #[test]
    fn resilience_currency_status_valid_transitions() {
        assert!(CurrencyStatus::Draft.can_transition_to(&CurrencyStatus::Active));
        assert!(CurrencyStatus::Active.can_transition_to(&CurrencyStatus::Suspended));
        assert!(CurrencyStatus::Active.can_transition_to(&CurrencyStatus::Retired));
        assert!(CurrencyStatus::Suspended.can_transition_to(&CurrencyStatus::Active));
        assert!(CurrencyStatus::Suspended.can_transition_to(&CurrencyStatus::Retired));
    }

    // =========================================================================
    // TEND Limit Tier Resilience
    // =========================================================================

    /// Verify TEND limits are monotonically increasing with stress.
    #[test]
    fn resilience_tend_limits_monotonic() {
        assert!(TendLimitTier::Normal.limit() < TendLimitTier::Elevated.limit());
        assert!(TendLimitTier::Elevated.limit() < TendLimitTier::High.limit());
        assert!(TendLimitTier::High.limit() < TendLimitTier::Emergency.limit());
    }

    /// Verify TEND tier transitions at all vitality boundaries.
    #[test]
    fn resilience_tend_tier_boundaries() {
        // Emergency: 0-10
        for v in 0..=10 {
            assert_eq!(TendLimitTier::from_vitality(v), TendLimitTier::Emergency);
        }
        // High: 11-20
        for v in 11..=20 {
            assert_eq!(TendLimitTier::from_vitality(v), TendLimitTier::High);
        }
        // Elevated: 21-40
        for v in 21..=40 {
            assert_eq!(TendLimitTier::from_vitality(v), TendLimitTier::Elevated);
        }
        // Normal: 41+
        for v in [41, 50, 100, 1000, u32::MAX] {
            assert_eq!(TendLimitTier::from_vitality(v), TendLimitTier::Normal);
        }
    }

    // =========================================================================
    // Minted Currency Parameter Validation Resilience
    // =========================================================================

    /// Verify that extreme parameter combinations are caught.
    #[test]
    fn resilience_minted_params_extreme_values() {
        let mut params = MintedCurrencyParams {
            name: "Test".into(),
            symbol: "TST".into(),
            description: "A test currency".into(),
            credit_limit: MINTED_CREDIT_LIMIT_MIN,
            demurrage_rate: 0.0,
            max_service_hours: 1,
            min_service_minutes: MINTED_MIN_SERVICE_MINUTES_MIN,
            requires_confirmation: false,
            confirmation_timeout_hours: 0,
            max_exchanges_per_day: 0,
        };
        assert!(
            params.validate().is_ok(),
            "Minimum valid params should pass"
        );

        // Max everything
        params.credit_limit = MINTED_CREDIT_LIMIT_MAX;
        params.demurrage_rate = MINTED_DEMURRAGE_RATE_MAX;
        params.max_service_hours = MINTED_MAX_SERVICE_HOURS_MAX;
        params.min_service_minutes = 60;
        params.confirmation_timeout_hours = MINTED_CONFIRMATION_TIMEOUT_MAX;
        params.max_exchanges_per_day = MINTED_MAX_EXCHANGES_PER_DAY_MAX;
        assert!(
            params.validate().is_ok(),
            "Maximum valid params should pass"
        );
    }

    /// NaN demurrage rate should be rejected.
    #[test]
    fn resilience_minted_params_nan_demurrage() {
        let params = MintedCurrencyParams {
            name: "Test".into(),
            symbol: "TST".into(),
            description: "".into(),
            credit_limit: 50,
            demurrage_rate: f64::NAN,
            max_service_hours: 8,
            min_service_minutes: 15,
            requires_confirmation: false,
            confirmation_timeout_hours: 0,
            max_exchanges_per_day: 0,
        };
        assert!(
            params.validate().is_err(),
            "NaN demurrage should be rejected"
        );
    }

    // =========================================================================
    // Fee Tier Resilience
    // =========================================================================

    /// Fee tier with NaN/Inf MYCEL scores.
    #[test]
    fn resilience_fee_tier_nan_inf() {
        // NaN: 0.7 < NaN is false, 0.3 <= NaN is false → Newcomer
        assert_eq!(FeeTier::from_mycel(f64::NAN), FeeTier::Newcomer);
        // Infinity: > 0.7 is true → Steward
        assert_eq!(FeeTier::from_mycel(f64::INFINITY), FeeTier::Steward);
        // Negative: < 0.3 → Newcomer
        assert_eq!(FeeTier::from_mycel(-1.0), FeeTier::Newcomer);
    }

    /// Fee rates are always positive and finite.
    #[test]
    fn resilience_fee_rates_always_valid() {
        for score_thousandths in 0..=1000 {
            let score = score_thousandths as f64 / 1000.0;
            let tier = FeeTier::from_mycel(score);
            let rate = tier.base_fee_rate();
            assert!(
                rate > 0.0 && rate.is_finite(),
                "Fee rate for score {} should be positive finite, got {}",
                score,
                rate
            );
        }
    }
}
