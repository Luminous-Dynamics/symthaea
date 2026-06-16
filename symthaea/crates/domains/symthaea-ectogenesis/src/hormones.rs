// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gestational hormone curve modeling.
//!
//! Provides physiologically accurate hormone trajectories across 40 weeks
//! for hCG, progesterone, estradiol, cortisol, and thyroid T4.

use crate::types::{GestationalWeek, HormoneProfile};

/// hCG curve: peaks at ~week 10 (~100,000 mIU/mL), then declines to plateau ~20,000.
pub fn hcg_curve(week: GestationalWeek) -> f64 {
    let w = week.0 as f64;
    if w <= 10.0 {
        // Quadratic rise from 0 to 100,000 at week 10
        100_000.0 * (w / 10.0).powf(2.0)
    } else {
        // Exponential decay from 100,000 toward plateau of 20,000
        (100_000.0 - 20_000.0) * (-0.05 * (w - 10.0)).exp() + 20_000.0
    }
}

/// Progesterone curve: rises from ~10 to ~300 ng/mL over gestation.
pub fn progesterone_curve(week: GestationalWeek) -> f64 {
    let w = week.0 as f64;
    10.0 + 290.0 * (1.0 - (-0.08 * w).exp())
}

/// Estradiol curve: rises exponentially, especially in third trimester.
pub fn estradiol_curve(week: GestationalWeek) -> f64 {
    let w = week.0 as f64;
    50.0 * (0.1 * w).exp()
}

/// Cortisol curve: gradual rise from ~15 mcg/dL with gestation.
pub fn cortisol_curve(week: GestationalWeek) -> f64 {
    15.0 + (week.0 as f64 * 0.2)
}

/// Thyroid T4 curve: gradual rise from ~8 mcg/dL with gestation.
pub fn thyroid_t4_curve(week: GestationalWeek) -> f64 {
    8.0 + (week.0 as f64 * 0.05)
}

/// Build a complete target hormone profile for a given gestational week.
pub fn target_hormones(week: GestationalWeek) -> HormoneProfile {
    HormoneProfile {
        hcg: hcg_curve(week),
        progesterone: progesterone_curve(week),
        estradiol: estradiol_curve(week),
        cortisol: cortisol_curve(week),
        thyroid_t4: thyroid_t4_curve(week),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hcg_peaks_around_week_10() {
        let hcg_8 = hcg_curve(GestationalWeek::new(8));
        let hcg_10 = hcg_curve(GestationalWeek::new(10));
        let hcg_15 = hcg_curve(GestationalWeek::new(15));
        let hcg_30 = hcg_curve(GestationalWeek::new(30));

        assert!(
            hcg_10 > hcg_8,
            "hCG should rise to peak: {hcg_10} > {hcg_8}"
        );
        assert!(
            hcg_10 > hcg_15,
            "hCG should decline after peak: {hcg_10} > {hcg_15}"
        );
        assert!(
            hcg_30 < hcg_10,
            "hCG declines from peak: {hcg_30} < {hcg_10}"
        );
    }

    #[test]
    fn test_hcg_peak_value() {
        let peak = hcg_curve(GestationalWeek::new(10));
        // At w=10: 100_000 * (10/10)^2 = 100,000 (continuous with decay branch)
        assert!(
            peak > 5000.0,
            "hCG at week 10 should be significant: {peak}"
        );
    }

    #[test]
    fn test_progesterone_rises_monotonically() {
        for w in 1..40 {
            let p_w = progesterone_curve(GestationalWeek::new(w));
            let p_next = progesterone_curve(GestationalWeek::new(w + 1));
            assert!(
                p_next > p_w,
                "Progesterone should rise: week {w}={p_w} < week {}={p_next}",
                w + 1
            );
        }
    }

    #[test]
    fn test_estradiol_rises() {
        let e10 = estradiol_curve(GestationalWeek::new(10));
        let e30 = estradiol_curve(GestationalWeek::new(30));
        let e40 = estradiol_curve(GestationalWeek::new(40));
        assert!(e30 > e10, "Estradiol should rise");
        assert!(e40 > e30, "Estradiol should continue rising");
    }

    #[test]
    fn test_cortisol_rises() {
        let c10 = cortisol_curve(GestationalWeek::new(10));
        let c30 = cortisol_curve(GestationalWeek::new(30));
        assert!(c30 > c10, "Cortisol should rise with gestation");
    }

    #[test]
    fn test_target_hormones_all_positive() {
        for w in 0..=40 {
            let h = target_hormones(GestationalWeek::new(w));
            assert!(h.hcg >= 0.0, "hCG must be non-negative at week {w}");
            assert!(
                h.progesterone > 0.0,
                "Progesterone must be positive at week {w}"
            );
            assert!(h.estradiol > 0.0, "Estradiol must be positive at week {w}");
            assert!(h.cortisol > 0.0, "Cortisol must be positive at week {w}");
            assert!(h.thyroid_t4 > 0.0, "T4 must be positive at week {w}");
        }
    }

    #[test]
    fn test_hormone_profile_serde() {
        let h = target_hormones(GestationalWeek::new(20));
        let json = serde_json::to_string(&h).unwrap();
        let deser: crate::types::HormoneProfile = serde_json::from_str(&json).unwrap();
        assert!((deser.hcg - h.hcg).abs() < 1e-6);
        assert!((deser.progesterone - h.progesterone).abs() < 1e-6);
    }
}
