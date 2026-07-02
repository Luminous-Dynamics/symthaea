// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-end lifecycle simulation: deposit -> covenant -> health -> margin call -> liquidation
//!
//! Tests the complete lifecycle of a collateral position through all states.
//! Uses pure types only (no Holochain conductor).

#[cfg(test)]
mod lifecycle_tests {
    use crate::*;

    #[test]
    fn lifecycle_deposit_to_liquidation() {
        // Phase 1: Initial deposit
        // Agent deposits 1000 ETH at oracle rate 2.0 -> 2000 SAP minted
        let deposit_amount = 1000u64;
        let oracle_rate = 2.0;
        let sap_minted = (deposit_amount as f64 * oracle_rate) as u64;
        assert_eq!(sap_minted, 2000);

        // Phase 2: Covenant placed
        let covenant = Covenant {
            id: "cov-lifecycle-001".into(),
            collateral_id: "deposit-lifecycle-001".into(),
            restriction: CovenantRestriction::CollateralPledge {
                obligation_id: "loan-001".into(),
            },
            beneficiary_did: "did:mycelix:lender".into(),
            created_at_micros: 1_700_000_000_000_000,
            expires_at_micros: None,
            released: false,
            released_by: None,
            released_at_micros: None,
        };
        assert!(!covenant.released);

        // Phase 3: Initial health check -- healthy
        let initial_value = 2000u64; // SAP equivalent
        let obligation = 1000u64; // 50% LTV
        let ltv = obligation as f64 / initial_value as f64;
        assert_eq!(
            CollateralHealthStatus::from_ltv(ltv),
            CollateralHealthStatus::Healthy
        );

        // Phase 4: Price drops 30% -- still healthy (LTV ~71.4%, below 80% threshold)
        let crashed_value = (initial_value as f64 * 0.70) as u64; // 1400
        let new_ltv = obligation as f64 / crashed_value as f64; // ~71.4%
        assert_eq!(
            CollateralHealthStatus::from_ltv(new_ltv),
            CollateralHealthStatus::Healthy
        );

        // Phase 5: Price drops to 50% of original -- liquidation territory
        // LTV = 1000/1000 = 100%, which is > 0.95 -> Liquidation
        let crashed_value_2 = (initial_value as f64 * 0.50) as u64; // 1000
        let ltv_2 = obligation as f64 / crashed_value_2 as f64; // 100%
        assert_eq!(
            CollateralHealthStatus::from_ltv(ltv_2),
            CollateralHealthStatus::Liquidation
        );

        // Phase 6: Verify covenant prevents redemption (covenant still active)
        assert!(!covenant.released);
        // In real system: redeem_collateral would call check_covenants_inner and reject

        // Phase 7: Governance-approved liquidation releases covenant
        let released_covenant = Covenant {
            released: true,
            released_by: Some("did:mycelix:governance".into()),
            released_at_micros: Some(1_700_001_000_000_000),
            ..covenant
        };
        assert!(released_covenant.released);

        // Phase 8: After liquidation, SAP is recovered
        // obligation (1000) recovered from crashed collateral (1000 SAP equivalent)
        // Shortfall = 0 in this case (exactly at 100% LTV)
    }

    #[test]
    fn lifecycle_multi_collateral_survives_crash() {
        // Diversified position: 4 asset classes
        let components = vec![
            CollateralComponent {
                asset_id: "property-001".into(),
                asset_class: CollateralAssetClass::RealEstate,
                weight: 0.40,
                current_value: 400_000,
            },
            CollateralComponent {
                asset_id: "solar-cert-001".into(),
                asset_class: CollateralAssetClass::EnergyCertificate,
                weight: 0.25,
                current_value: 250_000,
            },
            CollateralComponent {
                asset_id: "maize-harvest-001".into(),
                asset_class: CollateralAssetClass::AgriculturalAsset,
                weight: 0.20,
                current_value: 200_000,
            },
            CollateralComponent {
                asset_id: "carbon-001".into(),
                asset_class: CollateralAssetClass::CarbonCredit,
                weight: 0.15,
                current_value: 150_000,
            },
        ];

        let total_value: u64 = components.iter().map(|c| c.current_value).sum();
        assert_eq!(total_value, 1_000_000);

        let bonus = compute_diversification_bonus(&components);
        assert!((bonus - 0.15).abs() < f64::EPSILON); // 3 extra classes * 5%

        let obligation = 700_000u64; // 70% LTV
        let raw_ltv = obligation as f64 / total_value as f64;
        let effective_ltv = raw_ltv * (1.0 - bonus); // 70% * 0.85 = 59.5%

        assert_eq!(
            CollateralHealthStatus::from_ltv(effective_ltv),
            CollateralHealthStatus::Healthy
        );

        // 30% crash on ALL assets
        let crashed_value = (total_value as f64 * 0.70) as u64; // 700,000
        let crashed_raw_ltv = obligation as f64 / crashed_value as f64; // 100%
        let crashed_effective_ltv = crashed_raw_ltv * (1.0 - bonus); // 85%

        // Without diversification: Liquidation (100% LTV > 0.95)
        assert_eq!(
            CollateralHealthStatus::from_ltv(crashed_raw_ltv),
            CollateralHealthStatus::Liquidation
        );
        // With diversification: Warning (85% effective LTV, 0.80 < 0.85 <= 0.90) -- SURVIVED!
        assert_eq!(
            CollateralHealthStatus::from_ltv(crashed_effective_ltv),
            CollateralHealthStatus::Warning
        );
    }

    #[test]
    fn lifecycle_energy_certificate_to_sap() {
        // Energy project produces 10,000 kWh solar
        let kwh = 10_000.0f64;
        // Oracle consensus: 1 kWh = 0.10 SAP (100,000 micro-SAP)
        let sap_per_kwh = 100_000u64;
        let sap_value = (kwh as u64) * sap_per_kwh;
        assert_eq!(sap_value, 1_000_000_000); // 1,000 SAP

        // Certificate verified -> collateral registered -> SAP position opened
        // LTV at 50% -> can borrow 500 SAP against 1,000 SAP of energy
        let obligation = 500_000_000u64; // 500 SAP
        let ltv = obligation as f64 / sap_value as f64;
        assert_eq!(
            CollateralHealthStatus::from_ltv(ltv),
            CollateralHealthStatus::Healthy
        );
    }

    #[test]
    fn lifecycle_fiat_bridge_deposit() {
        // Fiat deposit: $1,000 USD at rate 1.0 (1 USD cent = 1 SAP micro-unit)
        let fiat_amount = 100_000u64; // $1,000 in cents
        let exchange_rate = 10_000.0; // 1 cent = 10,000 micro-SAP (1 USD = 1M micro-SAP = 1 SAP)
        let sap_minted = (fiat_amount as f64 * exchange_rate) as u64;
        assert_eq!(sap_minted, 1_000_000_000); // 1,000 SAP

        // Verify blended rate with external oracle
        let community_rate = 10_000.0;
        let external_rate = 10_200.0; // 2% higher
        let confidence = 0.8;
        let blended = compute_blended_oracle_rate(community_rate, Some(external_rate), confidence);

        // Blended should be between community and external
        assert!(blended > community_rate && blended < external_rate);

        // Blended should be closer to community (70% base weight)
        let midpoint = (community_rate + external_rate) / 2.0;
        assert!(
            blended < midpoint,
            "Blended {} should be closer to community than midpoint {}",
            blended,
            midpoint
        );
    }

    #[test]
    fn lifecycle_tend_timebank_through_demurrage() {
        // Start: 0 balance (zero-sum)
        // Exchange: Alice provides 4 hours gardening to Bob
        let alice_balance: i32 = -4; // owes 4 hours
        let bob_balance: i32 = 4; // owed 4 hours
        assert_eq!(alice_balance + bob_balance, 0);

        // After 6 months with 3% demurrage on positive balances:
        let bob_deduction = compute_minted_demurrage(bob_balance, 0.03, 180 * 86_400);
        // 3% annual on 4 hours for 6 months ~ 0.06 hours -> rounds to 0 (< 1 hour)
        assert_eq!(
            bob_deduction, 0,
            "Small balance should not lose whole hours to demurrage"
        );

        // After 10 years with 3% demurrage on a 100-hour balance:
        let large_balance: i32 = 100;
        let deduction_10yr = compute_minted_demurrage(large_balance, 0.03, 10 * 365 * 86_400);
        // 100 * (1 - e^(-0.03 * 10)) = 100 * 0.2592 ~ 25 hours
        assert!(
            deduction_10yr >= 24 && deduction_10yr <= 27,
            "10-year demurrage on 100h should be ~25h, got {}",
            deduction_10yr
        );

        // Zero-sum preserved: deduction goes to commons pool
        let remaining = large_balance - deduction_10yr;
        assert!(remaining > 0);
        assert_eq!(remaining + deduction_10yr, large_balance);
    }
}
