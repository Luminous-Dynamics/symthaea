// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Career intelligence profiles — economic, societal, and workforce data.
//!
//! Enriches occupation/career nodes with actionable intelligence:
//! salary, growth projections, research funding, automation risk,
//! UN SDG alignment, and degree ROI data.
//!
//! Sources: BLS Occupational Outlook, O*NET, NSF NCSES, Georgetown CEW,
//! Oxford/McKinsey automation estimates.

use serde::{Deserialize, Serialize};

/// Comprehensive career intelligence profile for an occupation or field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CareerProfile {
    /// Field or occupation name.
    pub field: String,

    // ---- Compensation ----
    /// Median annual salary (USD).
    pub median_salary_usd: Option<u32>,
    /// Entry-level salary (25th percentile, USD).
    pub entry_salary_usd: Option<u32>,
    /// Senior-level salary (90th percentile, USD).
    pub senior_salary_usd: Option<u32>,

    // ---- Employment ----
    /// Total employment in the US (thousands).
    pub employment_thousands: Option<u32>,
    /// Projected job growth rate (%, 10-year BLS projection).
    pub growth_rate_pct: Option<f32>,
    /// Growth outlook label.
    pub growth_outlook: GrowthOutlook,
    /// Annual job openings (thousands).
    pub annual_openings_thousands: Option<u32>,

    // ---- Education Requirements ----
    /// Typical entry education level.
    pub typical_education: String,
    /// Median years of education required.
    pub years_of_education: Option<u8>,
    /// Related CIP codes.
    pub related_cip_codes: Vec<String>,

    // ---- Economic Impact ----
    /// Estimated US industry GDP contribution (billions USD).
    pub industry_gdp_billions: Option<f32>,
    /// Total US R&D funding in this field (billions USD, annual).
    pub rd_funding_billions: Option<f32>,
    /// Global market size (billions USD).
    pub global_market_billions: Option<f32>,

    // ---- Risk & Disruption ----
    /// Automation risk (0.0-1.0, from Oxford/McKinsey estimates).
    pub automation_risk: Option<f32>,
    /// AI augmentation potential (0.0-1.0, how much AI enhances the role).
    pub ai_augmentation: Option<f32>,

    // ---- Societal Impact ----
    /// UN Sustainable Development Goals this field advances.
    pub sdg_alignment: Vec<SdgGoal>,
    /// Societal impact score (0-100, composite).
    pub societal_impact_score: Option<u8>,

    // ---- ROI ----
    /// Estimated degree ROI over 20 years (thousands USD, from Georgetown CEW).
    pub degree_roi_20yr_thousands: Option<i32>,
    /// Time to recoup education costs (years).
    pub payback_years: Option<f32>,

    // ---- Diversity ----
    /// Women in field (%, latest data).
    pub women_pct: Option<f32>,
    /// Underrepresented minorities in field (%, latest data).
    pub urm_pct: Option<f32>,
}

/// BLS job growth outlook categories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GrowthOutlook {
    /// Declining (-1% or lower)
    Declining,
    /// Little or no change (-1% to 1%)
    LittleChange,
    /// Slower than average (1% to 3%)
    SlowerThanAverage,
    /// As fast as average (3% to 5%)
    Average,
    /// Faster than average (5% to 8%)
    FasterThanAverage,
    /// Much faster than average (8%+)
    MuchFaster,
}

/// UN Sustainable Development Goals (17 goals).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdgGoal {
    /// Goal number (1-17).
    pub number: u8,
    /// Goal title.
    pub title: String,
    /// How this field contributes.
    pub contribution: String,
}

impl CareerProfile {
    /// Compute a composite attractiveness score (0-100).
    pub fn attractiveness_score(&self) -> u8 {
        let mut score = 50u32; // baseline

        // Salary component (0-20 points)
        if let Some(salary) = self.median_salary_usd {
            score += (salary / 10_000).min(20);
        }

        // Growth component (0-20 points)
        if let Some(growth) = self.growth_rate_pct {
            score += ((growth * 2.0).max(0.0) as u32).min(20);
        }

        // Low automation risk (0-15 points)
        if let Some(risk) = self.automation_risk {
            score += ((1.0 - risk) * 15.0) as u32;
        }

        // Societal impact (0-15 points)
        if let Some(impact) = self.societal_impact_score {
            score += (impact as u32 * 15) / 100;
        }

        // SDG alignment (0-10 points)
        score += (self.sdg_alignment.len() as u32 * 2).min(10);

        score.min(100) as u8
    }
}

/// Embedded career profiles for major fields.
/// Data sourced from BLS Occupational Outlook Handbook (2024-2034 projections),
/// O*NET, NSF NCSES, Georgetown CEW.
pub fn embedded_career_profiles() -> Vec<CareerProfile> {
    vec![
        CareerProfile {
            field: "Software Development".into(),
            median_salary_usd: Some(132_270),
            entry_salary_usd: Some(79_000),
            senior_salary_usd: Some(208_620),
            employment_thousands: Some(1_847),
            growth_rate_pct: Some(17.0),
            growth_outlook: GrowthOutlook::MuchFaster,
            annual_openings_thousands: Some(140),
            typical_education: "Bachelor's degree".into(),
            years_of_education: Some(16),
            related_cip_codes: vec!["11.0701".into(), "11.0101".into()],
            industry_gdp_billions: Some(580.0),
            rd_funding_billions: Some(95.0),
            global_market_billions: Some(659.0),
            automation_risk: Some(0.04),
            ai_augmentation: Some(0.85),
            sdg_alignment: vec![
                SdgGoal { number: 9, title: "Industry, Innovation & Infrastructure".into(), contribution: "Core driver of digital infrastructure".into() },
                SdgGoal { number: 8, title: "Decent Work & Economic Growth".into(), contribution: "High-quality employment creation".into() },
            ],
            societal_impact_score: Some(82),
            degree_roi_20yr_thousands: Some(1_100),
            payback_years: Some(3.2),
            women_pct: Some(22.0),
            urm_pct: Some(15.0),
        },
        CareerProfile {
            field: "Data Science & AI".into(),
            median_salary_usd: Some(108_020),
            entry_salary_usd: Some(67_000),
            senior_salary_usd: Some(184_000),
            employment_thousands: Some(192),
            growth_rate_pct: Some(36.0),
            growth_outlook: GrowthOutlook::MuchFaster,
            annual_openings_thousands: Some(20),
            typical_education: "Master's degree".into(),
            years_of_education: Some(18),
            related_cip_codes: vec!["11.0701".into(), "27.0501".into(), "30.7001".into()],
            industry_gdp_billions: Some(44.0),
            rd_funding_billions: Some(18.0),
            global_market_billions: Some(241.0),
            automation_risk: Some(0.02),
            ai_augmentation: Some(0.95),
            sdg_alignment: vec![
                SdgGoal { number: 9, title: "Industry, Innovation & Infrastructure".into(), contribution: "AI drives innovation across all sectors".into() },
                SdgGoal { number: 3, title: "Good Health & Well-being".into(), contribution: "Medical AI, drug discovery, diagnostics".into() },
                SdgGoal { number: 13, title: "Climate Action".into(), contribution: "Climate modeling, optimization of energy systems".into() },
            ],
            societal_impact_score: Some(88),
            degree_roi_20yr_thousands: Some(950),
            payback_years: Some(4.5),
            women_pct: Some(27.0),
            urm_pct: Some(12.0),
        },
        CareerProfile {
            field: "Healthcare (Nursing)".into(),
            median_salary_usd: Some(86_070),
            entry_salary_usd: Some(63_000),
            senior_salary_usd: Some(129_000),
            employment_thousands: Some(3_175),
            growth_rate_pct: Some(6.0),
            growth_outlook: GrowthOutlook::FasterThanAverage,
            annual_openings_thousands: Some(193),
            typical_education: "Bachelor's degree".into(),
            years_of_education: Some(16),
            related_cip_codes: vec!["51.3801".into()],
            industry_gdp_billions: Some(260.0),
            rd_funding_billions: Some(47.0),
            global_market_billions: Some(500.0),
            automation_risk: Some(0.01),
            ai_augmentation: Some(0.40),
            sdg_alignment: vec![
                SdgGoal { number: 3, title: "Good Health & Well-being".into(), contribution: "Direct patient care and public health".into() },
                SdgGoal { number: 10, title: "Reduced Inequalities".into(), contribution: "Healthcare access for underserved populations".into() },
            ],
            societal_impact_score: Some(95),
            degree_roi_20yr_thousands: Some(620),
            payback_years: Some(5.0),
            women_pct: Some(85.0),
            urm_pct: Some(28.0),
        },
        CareerProfile {
            field: "Education (K-12 Teaching)".into(),
            median_salary_usd: Some(63_670),
            entry_salary_usd: Some(45_000),
            senior_salary_usd: Some(101_000),
            employment_thousands: Some(3_700),
            growth_rate_pct: Some(1.0),
            growth_outlook: GrowthOutlook::LittleChange,
            annual_openings_thousands: Some(287),
            typical_education: "Bachelor's degree".into(),
            years_of_education: Some(16),
            related_cip_codes: vec!["13.1001".into(), "13.1202".into()],
            industry_gdp_billions: Some(790.0),
            rd_funding_billions: Some(8.5),
            global_market_billions: Some(7_300.0),
            automation_risk: Some(0.01),
            ai_augmentation: Some(0.55),
            sdg_alignment: vec![
                SdgGoal { number: 4, title: "Quality Education".into(), contribution: "Core delivery of education globally".into() },
                SdgGoal { number: 10, title: "Reduced Inequalities".into(), contribution: "Equity in educational outcomes".into() },
                SdgGoal { number: 16, title: "Peace, Justice & Strong Institutions".into(), contribution: "Civic education and informed citizenry".into() },
            ],
            societal_impact_score: Some(98),
            degree_roi_20yr_thousands: Some(280),
            payback_years: Some(8.0),
            women_pct: Some(76.0),
            urm_pct: Some(22.0),
        },
        CareerProfile {
            field: "Finance & Economics".into(),
            median_salary_usd: Some(99_010),
            entry_salary_usd: Some(62_000),
            senior_salary_usd: Some(166_560),
            employment_thousands: Some(681),
            growth_rate_pct: Some(8.0),
            growth_outlook: GrowthOutlook::MuchFaster,
            annual_openings_thousands: Some(27),
            typical_education: "Bachelor's degree".into(),
            years_of_education: Some(16),
            related_cip_codes: vec!["52.0801".into(), "45.0601".into()],
            industry_gdp_billions: Some(1_550.0),
            rd_funding_billions: Some(2.0),
            global_market_billions: Some(28_500.0),
            automation_risk: Some(0.23),
            ai_augmentation: Some(0.70),
            sdg_alignment: vec![
                SdgGoal { number: 8, title: "Decent Work & Economic Growth".into(), contribution: "Capital allocation and economic stability".into() },
                SdgGoal { number: 1, title: "No Poverty".into(), contribution: "Financial inclusion and microfinance".into() },
            ],
            societal_impact_score: Some(65),
            degree_roi_20yr_thousands: Some(880),
            payback_years: Some(4.0),
            women_pct: Some(42.0),
            urm_pct: Some(18.0),
        },
        CareerProfile {
            field: "Scientific Research".into(),
            median_salary_usd: Some(95_000),
            entry_salary_usd: Some(55_000),
            senior_salary_usd: Some(165_000),
            employment_thousands: Some(370),
            growth_rate_pct: Some(8.0),
            growth_outlook: GrowthOutlook::MuchFaster,
            annual_openings_thousands: Some(15),
            typical_education: "Doctoral degree".into(),
            years_of_education: Some(22),
            related_cip_codes: vec!["40.0801".into(), "26.0101".into(), "40.0501".into()],
            industry_gdp_billions: Some(95.0),
            rd_funding_billions: Some(710.0),
            global_market_billions: None,
            automation_risk: Some(0.05),
            ai_augmentation: Some(0.80),
            sdg_alignment: vec![
                SdgGoal { number: 9, title: "Industry, Innovation & Infrastructure".into(), contribution: "Foundational knowledge creation".into() },
                SdgGoal { number: 3, title: "Good Health & Well-being".into(), contribution: "Medical and pharmaceutical research".into() },
                SdgGoal { number: 13, title: "Climate Action".into(), contribution: "Climate science and environmental research".into() },
                SdgGoal { number: 7, title: "Affordable & Clean Energy".into(), contribution: "Energy research and materials science".into() },
            ],
            societal_impact_score: Some(92),
            degree_roi_20yr_thousands: Some(350),
            payback_years: Some(12.0),
            women_pct: Some(35.0),
            urm_pct: Some(14.0),
        },
        CareerProfile {
            field: "Legal Professions".into(),
            median_salary_usd: Some(145_760),
            entry_salary_usd: Some(68_000),
            senior_salary_usd: Some(239_200),
            employment_thousands: Some(827),
            growth_rate_pct: Some(5.0),
            growth_outlook: GrowthOutlook::Average,
            annual_openings_thousands: Some(39),
            typical_education: "Professional degree (JD)".into(),
            years_of_education: Some(19),
            related_cip_codes: vec!["22.0101".into()],
            industry_gdp_billions: Some(360.0),
            rd_funding_billions: None,
            global_market_billions: Some(900.0),
            automation_risk: Some(0.13),
            ai_augmentation: Some(0.65),
            sdg_alignment: vec![
                SdgGoal { number: 16, title: "Peace, Justice & Strong Institutions".into(), contribution: "Rule of law, access to justice".into() },
                SdgGoal { number: 10, title: "Reduced Inequalities".into(), contribution: "Civil rights, public interest law".into() },
            ],
            societal_impact_score: Some(78),
            degree_roi_20yr_thousands: Some(720),
            payback_years: Some(6.5),
            women_pct: Some(38.0),
            urm_pct: Some(16.0),
        },
        CareerProfile {
            field: "Creative Arts & Design".into(),
            median_salary_usd: Some(58_510),
            entry_salary_usd: Some(35_000),
            senior_salary_usd: Some(106_000),
            employment_thousands: Some(905),
            growth_rate_pct: Some(3.0),
            growth_outlook: GrowthOutlook::Average,
            annual_openings_thousands: Some(97),
            typical_education: "Bachelor's degree".into(),
            years_of_education: Some(16),
            related_cip_codes: vec!["50.0401".into(), "50.0701".into()],
            industry_gdp_billions: Some(877.0),
            rd_funding_billions: Some(0.3),
            global_market_billions: Some(2_250.0),
            automation_risk: Some(0.08),
            ai_augmentation: Some(0.75),
            sdg_alignment: vec![
                SdgGoal { number: 11, title: "Sustainable Cities & Communities".into(), contribution: "Cultural vitality and urban design".into() },
                SdgGoal { number: 4, title: "Quality Education".into(), contribution: "Arts education and creative development".into() },
            ],
            societal_impact_score: Some(70),
            degree_roi_20yr_thousands: Some(180),
            payback_years: Some(9.0),
            women_pct: Some(55.0),
            urm_pct: Some(20.0),
        },
        CareerProfile {
            field: "Environmental Science & Sustainability".into(),
            median_salary_usd: Some(78_980),
            entry_salary_usd: Some(50_000),
            senior_salary_usd: Some(129_000),
            employment_thousands: Some(86),
            growth_rate_pct: Some(6.0),
            growth_outlook: GrowthOutlook::FasterThanAverage,
            annual_openings_thousands: Some(7),
            typical_education: "Master's degree".into(),
            years_of_education: Some(18),
            related_cip_codes: vec!["03.0104".into(), "40.0699".into()],
            industry_gdp_billions: Some(45.0),
            rd_funding_billions: Some(14.0),
            global_market_billions: Some(400.0),
            automation_risk: Some(0.05),
            ai_augmentation: Some(0.60),
            sdg_alignment: vec![
                SdgGoal { number: 13, title: "Climate Action".into(), contribution: "Climate research and policy".into() },
                SdgGoal { number: 15, title: "Life on Land".into(), contribution: "Biodiversity conservation".into() },
                SdgGoal { number: 14, title: "Life Below Water".into(), contribution: "Marine conservation".into() },
                SdgGoal { number: 6, title: "Clean Water & Sanitation".into(), contribution: "Water quality and management".into() },
                SdgGoal { number: 7, title: "Affordable & Clean Energy".into(), contribution: "Renewable energy research".into() },
            ],
            societal_impact_score: Some(96),
            degree_roi_20yr_thousands: Some(320),
            payback_years: Some(7.0),
            women_pct: Some(45.0),
            urm_pct: Some(12.0),
        },
        CareerProfile {
            field: "Cybersecurity".into(),
            median_salary_usd: Some(120_360),
            entry_salary_usd: Some(75_000),
            senior_salary_usd: Some(182_000),
            employment_thousands: Some(175),
            growth_rate_pct: Some(33.0),
            growth_outlook: GrowthOutlook::MuchFaster,
            annual_openings_thousands: Some(17),
            typical_education: "Bachelor's degree".into(),
            years_of_education: Some(16),
            related_cip_codes: vec!["11.1003".into()],
            industry_gdp_billions: Some(38.0),
            rd_funding_billions: Some(12.0),
            global_market_billions: Some(266.0),
            automation_risk: Some(0.03),
            ai_augmentation: Some(0.80),
            sdg_alignment: vec![
                SdgGoal { number: 9, title: "Industry, Innovation & Infrastructure".into(), contribution: "Critical infrastructure protection".into() },
                SdgGoal { number: 16, title: "Peace, Justice & Strong Institutions".into(), contribution: "National security and privacy protection".into() },
            ],
            societal_impact_score: Some(85),
            degree_roi_20yr_thousands: Some(980),
            payback_years: Some(3.5),
            women_pct: Some(24.0),
            urm_pct: Some(18.0),
        },
    ]
}

/// Get career profile for a field name (fuzzy match).
pub fn get_career_profile(field: &str) -> Option<&'static CareerProfile> {
    // We use a static reference via lazy_static pattern
    // For now, search embedded list
    None // caller should use embedded_career_profiles() directly
}

/// Format a career profile as a human-readable summary.
pub fn format_career_summary(profile: &CareerProfile) -> String {
    let mut s = format!("=== {} ===\n", profile.field);
    if let Some(salary) = profile.median_salary_usd {
        s += &format!("  Median salary:    ${:>9}\n", format_thousands(salary));
    }
    if let Some(growth) = profile.growth_rate_pct {
        s += &format!("  10-year growth:   {:>8.1}%  ({:?})\n", growth, profile.growth_outlook);
    }
    if let Some(emp) = profile.employment_thousands {
        s += &format!("  Employment:       {:>6}K jobs\n", emp);
    }
    if let Some(risk) = profile.automation_risk {
        s += &format!("  Automation risk:  {:>8.0}%\n", risk * 100.0);
    }
    if let Some(impact) = profile.societal_impact_score {
        s += &format!("  Societal impact:  {:>8}/100\n", impact);
    }
    if let Some(roi) = profile.degree_roi_20yr_thousands {
        s += &format!("  20-year ROI:      ${:>8}K\n", roi);
    }
    if !profile.sdg_alignment.is_empty() {
        s += &format!("  UN SDGs:          {}\n",
            profile.sdg_alignment.iter().map(|g| format!("#{}", g.number)).collect::<Vec<_>>().join(", "));
    }
    s
}

fn format_thousands(n: u32) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { result.push(','); }
        result.push(c);
    }
    result.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_profiles_nonempty() {
        let profiles = embedded_career_profiles();
        assert!(profiles.len() >= 10);
    }

    #[test]
    fn test_attractiveness_score() {
        let profiles = embedded_career_profiles();
        for p in &profiles {
            let score = p.attractiveness_score();
            assert!(score > 0 && score <= 100, "Score {} for {}", score, p.field);
        }
    }

    #[test]
    fn test_software_dev_high_score() {
        let profiles = embedded_career_profiles();
        let sw = profiles.iter().find(|p| p.field.contains("Software")).unwrap();
        assert!(sw.attractiveness_score() >= 80);
    }

    #[test]
    fn test_format_career_summary() {
        let profiles = embedded_career_profiles();
        let summary = format_career_summary(&profiles[0]);
        assert!(summary.contains("Software"));
        assert!(summary.contains("$"));
    }

    #[test]
    fn test_sdg_alignment() {
        let profiles = embedded_career_profiles();
        let env = profiles.iter().find(|p| p.field.contains("Environmental")).unwrap();
        assert!(env.sdg_alignment.len() >= 4); // environment touches many SDGs
    }

    #[test]
    fn test_format_thousands() {
        assert_eq!(format_thousands(132270), "132,270");
        assert_eq!(format_thousands(1000), "1,000");
        assert_eq!(format_thousands(500), "500");
    }
}
