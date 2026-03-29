// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Safety-invariant property tests for symthaea-nurture.
//!
//! Tests that developmental safety bounds hold:
//! - CriticalPeriodModel plasticity is always in [0, 1]
//! - Plasticity is monotonically non-increasing after peak for all domains
//! - AttachmentSystem security_score remains in [0, 1]
//! - DevelopmentalAge clamps negative months to zero

use proptest::prelude::*;
use symthaea_nurture::{
    AttachmentSystem, CriticalPeriodDomain, CriticalPeriodModel, DevelopmentalAge,
};

fn arb_domain() -> impl Strategy<Value = CriticalPeriodDomain> {
    prop_oneof![
        Just(CriticalPeriodDomain::Attachment),
        Just(CriticalPeriodDomain::Language),
        Just(CriticalPeriodDomain::Vision),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Safety: analytical_plasticity must always return a value in [0, 1] for any
    /// non-negative age and any domain. Out-of-bounds plasticity would corrupt
    /// learning rate modulation and could cause unbounded weight updates.
    #[test]
    fn prop_plasticity_bounded(
        domain in arb_domain(),
        age_months in 0.0_f64..600.0,  // 0 to 50 years
    ) {
        let p = CriticalPeriodModel::analytical_plasticity(domain, age_months);

        prop_assert!(p >= 0.0 && p <= 1.0,
            "{:?} plasticity at {}mo = {} out of [0,1]", domain, age_months, p);
    }

    /// Safety: plasticity must be monotonically non-increasing after the peak period.
    /// The peak ends at 12mo for Attachment/Vision, 36mo for Language.
    /// A plasticity increase after peak closure would violate the critical period
    /// model and could re-open sensitive learning windows inappropriately.
    #[test]
    fn prop_plasticity_monotonic_after_peak(
        domain in arb_domain(),
        base_age in 0.0_f64..200.0,
        delta in 0.001_f64..100.0,
    ) {
        let peak_end = match domain {
            CriticalPeriodDomain::Attachment => 12.0,
            CriticalPeriodDomain::Language => 36.0,
            CriticalPeriodDomain::Vision => 12.0,
        };

        // Only test after peak
        let age_early = peak_end + base_age;
        let age_later = age_early + delta;

        let p_early = CriticalPeriodModel::analytical_plasticity(domain, age_early);
        let p_later = CriticalPeriodModel::analytical_plasticity(domain, age_later);

        prop_assert!(p_later <= p_early + 1e-12,
            "{:?} plasticity must not increase after peak: {}mo={} -> {}mo={}",
            domain, age_early, p_early, age_later, p_later);
    }

    /// Safety: AttachmentSystem security_score must remain in [0, 1] after any
    /// sequence of interactions. Security score drives downstream neuromodulator
    /// gating; out-of-bounds values would corrupt the emotional regulation system.
    #[test]
    fn prop_security_score_bounded(
        responsiveness in 0.0_f64..1.0,
        warmth in 0.0_f64..1.0,
        attunement in 0.0_f64..1.0,
        n_interactions in 1_usize..200,
    ) {
        let mut sys = AttachmentSystem::new();
        sys.advance_age(12.0); // Past forming stage

        let action = symthaea_nurture::CaregiverAction {
            responsiveness,
            warmth,
            attunement,
            contact: true,
            vocalizing: true,
        };

        for _ in 0..n_interactions {
            sys.process_interaction(&action);
        }

        prop_assert!(sys.security_score >= 0.0 && sys.security_score <= 1.0,
            "security_score {} out of [0,1] after {} interactions (resp={}, warm={}, att={})",
            sys.security_score, n_interactions, responsiveness, warmth, attunement);
    }

    /// Safety: DevelopmentalAge clamps negative months to zero. Negative ages
    /// are biologically meaningless and would produce invalid plasticity lookups
    /// and milestone evaluations.
    #[test]
    fn prop_developmental_age_non_negative(months in -1000.0_f64..1000.0) {
        let age = DevelopmentalAge::new(months);
        prop_assert!(age.months >= 0.0,
            "DevelopmentalAge.months must be >= 0, got {} from input {}", age.months, months);

        if months >= 0.0 {
            prop_assert!((age.months - months).abs() < 1e-10,
                "positive months should be preserved: {} != {}", age.months, months);
        }
    }

    /// Safety: critical_period_plasticity on AttachmentSystem must stay in [0, 1]
    /// as age advances. This value modulates learning rate, so out-of-bounds
    /// would cause unbounded or negative learning.
    #[test]
    fn prop_attachment_plasticity_bounded_through_age(
        increments in prop::collection::vec(0.1_f64..24.0, 1..20),
    ) {
        let mut sys = AttachmentSystem::new();

        prop_assert!(sys.critical_period_plasticity >= 0.0 && sys.critical_period_plasticity <= 1.0,
            "initial plasticity {} out of [0,1]", sys.critical_period_plasticity);

        for inc in &increments {
            sys.advance_age(*inc);
            prop_assert!(sys.critical_period_plasticity >= 0.0 && sys.critical_period_plasticity <= 1.0,
                "plasticity {} out of [0,1] at total_months={}",
                sys.critical_period_plasticity, sys.total_months);
        }
    }
}
