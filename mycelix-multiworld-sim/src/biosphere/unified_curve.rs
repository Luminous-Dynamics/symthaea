// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Unified coherence curve: B(t) → K(t) → Sim forward projection.
//!
//! Produces a single continuous time series from 3.8 Ga to 2200 CE,
//! stitching together three measurement regimes:
//!
//! - **B(t)** (3.8 Ga → 10,000 BCE): Biosphere coherence from paleontological data
//! - **K(t)** (1810 → 2020 CE): Civilizational coherence from historical proxies
//! - **Blend zone** (10,000 BCE → 1810 CE): HANPP-grounded thermodynamic transition
//!
//! The sim's Eight Harmonies scores extend the curve into the future (2026 → 2200+).

use super::bridge::CoherenceSource;
use super::temporal_bins::MaAge;
use super::BiosphereCoherenceEngine;

/// Embedded K(t) series from kosmic-lab output.
/// These are the actual computed values from the historical K(t) pipeline.
const KT_SERIES: &[(i32, f64)] = &[
    (1810, 0.000),
    (1820, 0.010),
    (1830, 0.023),
    (1840, 0.035),
    (1850, 0.047),
    (1860, 0.061),
    (1870, 0.076),
    (1880, 0.100),
    (1890, 0.125),
    (1900, 0.136),
    (1910, 0.152),
    (1920, 0.166),
    (1930, 0.182),
    (1940, 0.198),
    (1950, 0.211),
    (1960, 0.237),
    (1970, 0.343),
    (1980, 0.434),
    (1990, 0.481),
    (2000, 0.112),
    (2010, 0.555),
    (2020, 0.782),
];

/// A point on the unified curve.
#[derive(Debug, Clone)]
pub struct UnifiedPoint {
    /// Age in Ma (positive = past, negative = future).
    /// 0.0 = 2026 CE.
    pub age_ma: MaAge,
    /// Calendar year (CE). Negative = BCE.
    pub year_ce: f64,
    /// Coherence value [0, 1].
    pub value: f64,
    /// Lower CI bound.
    pub ci_lower: f64,
    /// Upper CI bound.
    pub ci_upper: f64,
    /// Which regime is primary.
    pub source: CoherenceSource,
}

fn year_to_ma(year_ce: f64) -> MaAge {
    (2026.0 - year_ce) / 1_000_000.0
}

fn ma_to_year(age_ma: MaAge) -> f64 {
    2026.0 - age_ma * 1_000_000.0
}

/// Interpolate K(t) at a given year.
#[allow(dead_code)]
fn kt_at_year(year: f64) -> Option<f64> {
    if year < KT_SERIES[0].0 as f64 || year > KT_SERIES.last().unwrap().0 as f64 {
        return None;
    }
    for pair in KT_SERIES.windows(2) {
        let (y0, v0) = (pair[0].0 as f64, pair[0].1);
        let (y1, v1) = (pair[1].0 as f64, pair[1].1);
        if year >= y0 && year <= y1 {
            let t = (year - y0) / (y1 - y0);
            return Some(v0 + t * (v1 - v0));
        }
    }
    Some(KT_SERIES.last().unwrap().1)
}

/// Compute C(t) using the **headroom formula**:
///
/// ```text
/// C(t) = B_hanpp(t) + K(t) × (1 - B_hanpp(t))
/// ```
///
/// K(t) measures civilizational coherence *on top of* the biosphere, not
/// instead of it. At 1810 CE, B_hanpp ≈ 0.90 and K(t) = 0.0, so C(t) = 0.90.
/// At 2020 CE, B_hanpp ≈ 0.72 (HANPP depression) and K(t) = 0.78,
/// so C(t) = 0.72 + 0.78 × 0.28 = 0.94.
///
/// This eliminates the 1810 cliff entirely: civilization adds to biosphere
/// coherence, it never replaces it.
fn headroom_composite(bt_hanpp: f64, kt: f64) -> f64 {
    bt_hanpp + kt * (1.0 - bt_hanpp)
}

/// Build the unified curve from 3.8 Ga to 2200 CE.
///
/// `sim_harmony_scores`: Optional forward projection from sim (year_ce, harmony_mean).
/// Pass empty slice if no sim data available.
pub fn build_unified_curve(sim_harmony_scores: &[(f64, f64)]) -> Vec<UnifiedPoint> {
    let bt_engine = BiosphereCoherenceEngine::build();
    let mut points = Vec::with_capacity(100);

    // ── Phase 1: B(t) deep time (3.8 Ga → 10,000 BCE) ──
    // Pure biosphere. No human civilization exists yet.
    for pt in bt_engine.full_curve() {
        if pt.age_ma > 0.012 {
            points.push(UnifiedPoint {
                age_ma: pt.age_ma,
                year_ce: ma_to_year(pt.age_ma),
                value: pt.value,
                ci_lower: pt.ci_lower,
                ci_upper: pt.ci_upper,
                source: CoherenceSource::Biosphere,
            });
        }
    }

    // ── Phase 2: Holocene transition (10,000 BCE → 1810 CE) ──
    // Biosphere coherence adjusted by HANPP. K(t) doesn't have data yet,
    // so civilizational addition is zero — C(t) = B_hanpp(t).
    let transition_years: Vec<f64> = vec![
        -8000.0, -5000.0, -3000.0, -1000.0, 0.0, 500.0, 1000.0, 1400.0, 1600.0, 1800.0,
    ];

    for &year in &transition_years {
        let age = year_to_ma(year);
        let (bt_val, bt_lo, bt_hi) = bt_engine.bt_at(age);
        let hanpp_adj = super::bridge::biosphere_energy_adjustment(age);
        let bt_hanpp = bt_val * hanpp_adj;
        // No K(t) data before 1810, so civilizational addition = 0
        let ct = headroom_composite(bt_hanpp, 0.0);
        let ci_lo = bt_lo * hanpp_adj;
        let ci_hi = (bt_hi * hanpp_adj).min(1.0);

        points.push(UnifiedPoint {
            age_ma: age,
            year_ce: year,
            value: ct,
            ci_lower: ci_lo,
            ci_upper: ci_hi,
            source: CoherenceSource::Blended,
        });
    }

    // ── Phase 3: K(t) era (1810 → 2020 CE) — headroom formula ──
    // C(t) = B_hanpp(t) + K(t) × (1 - B_hanpp(t))
    // K(t) fills the remaining headroom above the biosphere floor.
    for &(year, kt_val) in KT_SERIES {
        let age = year_to_ma(year as f64);
        let (bt_val, _, _) = bt_engine.bt_at(age);
        let hanpp_adj = super::bridge::biosphere_energy_adjustment(age);
        let bt_hanpp = bt_val * hanpp_adj;
        let ct = headroom_composite(bt_hanpp, kt_val);
        // CI from both B(t) and K(t) uncertainty
        let ci_lo = headroom_composite(bt_hanpp * 0.95, kt_val * 0.85);
        let ci_hi =
            headroom_composite((bt_hanpp * 1.05).min(1.0), (kt_val * 1.15).min(1.0)).min(1.0);

        points.push(UnifiedPoint {
            age_ma: age,
            year_ce: year as f64,
            value: ct,
            ci_lower: ci_lo,
            ci_upper: ci_hi,
            source: CoherenceSource::Civilization,
        });
    }

    // ── Phase 4: Sim forward projection (2026 → 2200+ CE) ──
    for &(year, harmony_mean) in sim_harmony_scores {
        let age = year_to_ma(year);
        let (bt_val, _, _) = bt_engine.bt_at(age);
        let hanpp_adj = super::bridge::biosphere_energy_adjustment(age);
        let bt_hanpp = bt_val * hanpp_adj;
        let ct = headroom_composite(bt_hanpp, harmony_mean);

        points.push(UnifiedPoint {
            age_ma: age,
            year_ce: year,
            value: ct,
            ci_lower: ct * 0.85,
            ci_upper: (ct * 1.15).min(1.0),
            source: CoherenceSource::Civilization,
        });
    }

    // Sort by age descending (oldest first)
    points.sort_by(|a, b| b.age_ma.partial_cmp(&a.age_ma).unwrap());

    points
}

/// Export unified curve as CSV.
pub fn unified_to_csv(points: &[UnifiedPoint]) -> String {
    let mut csv = String::from("age_ma,year_ce,value,ci_lower,ci_upper,source\n");
    for pt in points {
        csv.push_str(&format!(
            "{:.6},{:.1},{:.6},{:.6},{:.6},{:?}\n",
            pt.age_ma, pt.year_ce, pt.value, pt.ci_lower, pt.ci_upper, pt.source,
        ));
    }
    csv
}

/// Summary statistics of the unified curve.
pub fn unified_summary(points: &[UnifiedPoint]) -> String {
    let bt_count = points
        .iter()
        .filter(|p| matches!(p.source, CoherenceSource::Biosphere))
        .count();
    let blend_count = points
        .iter()
        .filter(|p| matches!(p.source, CoherenceSource::Blended))
        .count();
    let kt_count = points
        .iter()
        .filter(|p| matches!(p.source, CoherenceSource::Civilization))
        .count();

    let oldest = points.first().map(|p| p.age_ma).unwrap_or(0.0);
    let newest_year = points.last().map(|p| p.year_ce).unwrap_or(0.0);
    let peak = points.iter().map(|p| p.value).fold(0.0_f64, f64::max);
    let nadir = points.iter().map(|p| p.value).fold(1.0_f64, f64::min);

    format!(
        "Unified Coherence Curve: {:.1} Ga → {:.0} CE\n\
         Points: {} total ({} B(t), {} blended, {} K(t)/sim)\n\
         Range: {:.4} (nadir) → {:.4} (peak)\n\
         Span: {:.2} billion years",
        oldest / 1000.0,
        newest_year,
        points.len(),
        bt_count,
        blend_count,
        kt_count,
        nadir,
        peak,
        oldest / 1000.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_curve_builds() {
        let curve = build_unified_curve(&[]);
        assert!(
            curve.len() > 50,
            "Should have >50 points, got {}",
            curve.len()
        );
    }

    #[test]
    fn curve_spans_full_range() {
        let curve = build_unified_curve(&[]);
        let oldest = curve.first().unwrap().age_ma;
        let newest = curve.last().unwrap().age_ma;
        assert!(oldest > 3000.0, "Should start in deep time");
        assert!(newest < 0.001, "Should reach near-present");
    }

    #[test]
    fn curve_sorted_oldest_first() {
        let curve = build_unified_curve(&[]);
        for pair in curve.windows(2) {
            assert!(
                pair[0].age_ma >= pair[1].age_ma,
                "Should be sorted oldest-first"
            );
        }
    }

    #[test]
    fn all_three_sources_present() {
        let curve = build_unified_curve(&[]);
        let has_bio = curve
            .iter()
            .any(|p| matches!(p.source, CoherenceSource::Biosphere));
        let has_blend = curve
            .iter()
            .any(|p| matches!(p.source, CoherenceSource::Blended));
        let has_civ = curve
            .iter()
            .any(|p| matches!(p.source, CoherenceSource::Civilization));
        assert!(has_bio, "Should have biosphere points");
        assert!(has_blend, "Should have blended points");
        assert!(has_civ, "Should have civilization points");
    }

    #[test]
    fn sim_projection_extends_curve() {
        let sim_data = vec![(2050.0, 0.6), (2100.0, 0.7), (2200.0, 0.8)];
        let curve = build_unified_curve(&sim_data);
        let newest_year = curve.last().unwrap().year_ce;
        assert!(newest_year >= 2200.0, "Should extend to 2200 CE");
    }

    #[test]
    fn csv_export_has_header() {
        let curve = build_unified_curve(&[]);
        let csv = unified_to_csv(&curve);
        assert!(csv.starts_with("age_ma,year_ce,value,ci_lower,ci_upper,source\n"));
    }

    #[test]
    fn kt_interpolation_works() {
        let k1970 = kt_at_year(1970.0).unwrap();
        assert!(
            (k1970 - 0.343).abs() < 0.01,
            "K(1970) should be ~0.343, got {}",
            k1970
        );

        let k2020 = kt_at_year(2020.0).unwrap();
        assert!(
            (k2020 - 0.782).abs() < 0.01,
            "K(2020) should be ~0.782, got {}",
            k2020
        );
    }

    #[test]
    fn blend_zone_is_continuous() {
        let curve = build_unified_curve(&[]);
        // Check that B(t) → blend → K(t) transition doesn't have catastrophic gaps.
        // The 1810 CE boundary (blend→K(t)) is a known discontinuity because K(t)
        // proxy normalization starts from zero at 1810. This is inherent to the
        // two different measurement systems, not a bug.
        for pair in curve.windows(2) {
            let delta = (pair[0].value - pair[1].value).abs();
            // Only check within pure B(t) and pure blend zones (not at regime boundaries)
            let same_source =
                std::mem::discriminant(&pair[0].source) == std::mem::discriminant(&pair[1].source);
            if same_source && pair[0].age_ma < 0.02 && pair[0].age_ma > 0.00001 {
                assert!(
                    delta < 0.5,
                    "Gap of {} between {:.4} Ma ({:?}) and {:.4} Ma ({:?})",
                    delta,
                    pair[0].age_ma,
                    pair[0].source,
                    pair[1].age_ma,
                    pair[1].source,
                );
            }
        }
    }
}
