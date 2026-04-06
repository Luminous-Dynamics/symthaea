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

    // ---- Enriched Data (O*NET-sourced, optional) ----
    /// Top work activities (what you actually do day-to-day).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub work_activities: Vec<WorkActivity>,
    /// Top required skills with importance ratings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_skills: Vec<RequiredSkill>,
    /// Key technologies, software, and equipment.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub technology_tools: Vec<TechnologyTool>,
    /// Work value ratings (6 dimensions).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub work_values: Option<WorkValues>,
    /// Work-life balance characteristics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub work_life_balance: Option<WorkLifeBalance>,
    /// Typical career progression stages with salary.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub career_stages: Vec<CareerStage>,
    /// Related occupations you could transition to.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub career_transitions: Vec<CareerTransition>,
    /// If this is an emerging field, its metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emerging_field: Option<EmergingField>,
    /// O*NET SOC code (e.g., "15-1252.00").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub onet_soc_code: Option<String>,
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

// ============================================================
// Enriched Career Intelligence Types
// ============================================================

/// What you actually do day-to-day (O*NET Generalized Work Activities).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkActivity {
    pub name: String,
    /// Importance (1.0-5.0).
    pub importance: f32,
}

/// A skill required for the occupation (O*NET Skills taxonomy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequiredSkill {
    pub name: String,
    /// Importance (1.0-5.0).
    pub importance: f32,
    /// Category.
    pub category: String,
}

/// Specific technology, software, or equipment used.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnologyTool {
    pub name: String,
    /// "ProgrammingLanguage", "Software", "Platform", "Framework", "Equipment"
    pub category: String,
}

/// O*NET Work Values (6 dimensions, each 1.0-5.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkValues {
    pub achievement: f32,
    pub independence: f32,
    pub recognition: f32,
    pub relationships: f32,
    pub support: f32,
    pub working_conditions: f32,
}

/// Work-life balance characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkLifeBalance {
    pub hours_per_week: Option<f32>,
    /// 0.0-1.0 (1.0 = fully flexible).
    pub schedule_flexibility: Option<f32>,
    /// 0.0-1.0 (1.0 = fully remote).
    pub remote_availability: Option<f32>,
    /// 1-5 (1=sedentary, 5=very heavy).
    pub physical_demands: Option<u8>,
    /// 1-5 subjective stress level.
    pub stress_level: Option<u8>,
}

/// A stage in a career progression timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CareerStage {
    pub title: String,
    pub years_experience: u8,
    pub typical_salary_usd: u32,
    pub key_responsibilities: Vec<String>,
}

/// A related occupation you could transition to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CareerTransition {
    pub target_field: String,
    /// Skill overlap (0-100%).
    pub skill_overlap_pct: u8,
    pub gap_skills: Vec<String>,
    pub transition_months: Option<u8>,
    /// "Higher", "Similar", "Lower"
    pub salary_direction: String,
}

/// An emerging career field not yet in traditional databases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergingField {
    pub parent_fields: Vec<String>,
    /// "Nascent", "Emerging", "Establishing", "Maturing"
    pub maturity: String,
    pub estimated_practitioners: Option<u32>,
    pub emergence_year: Option<u16>,
    /// Projected 5-year growth multiplier (e.g., 3.0 = triple).
    pub growth_multiplier_5yr: Option<f32>,
    pub key_organizations: Vec<String>,
    pub foundation_skills: Vec<String>,
    /// Connection to Luminous Dynamics / Symthaea.
    pub luminous_connection: Option<String>,
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
/// Default values for the enriched fields (used by profiles that haven't been enriched yet).
fn enriched_defaults() -> (Vec<WorkActivity>, Vec<RequiredSkill>, Vec<TechnologyTool>, Option<WorkValues>, Option<WorkLifeBalance>, Vec<CareerStage>, Vec<CareerTransition>, Option<EmergingField>, Option<String>) {
    (vec![], vec![], vec![], None, None, vec![], vec![], None, None)
}

/// All career profiles — established + emerging fields.
pub fn all_career_profiles() -> Vec<CareerProfile> {
    let mut all = embedded_career_profiles();
    all.extend(embedded_emerging_fields());
    all
}

pub fn embedded_career_profiles() -> Vec<CareerProfile> {
    let d = enriched_defaults;
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
            // ---- Enriched ----
            work_activities: vec![
                WorkActivity { name: "Analyzing Data or Information".into(), importance: 4.5 },
                WorkActivity { name: "Making Decisions and Solving Problems".into(), importance: 4.3 },
                WorkActivity { name: "Thinking Creatively".into(), importance: 4.2 },
                WorkActivity { name: "Updating and Using Relevant Knowledge".into(), importance: 4.5 },
                WorkActivity { name: "Interacting With Computers".into(), importance: 4.8 },
            ],
            required_skills: vec![
                RequiredSkill { name: "Programming".into(), importance: 4.8, category: "Technical".into() },
                RequiredSkill { name: "Complex Problem Solving".into(), importance: 4.5, category: "CrossFunctional".into() },
                RequiredSkill { name: "Critical Thinking".into(), importance: 4.3, category: "Basic".into() },
                RequiredSkill { name: "Systems Analysis".into(), importance: 4.0, category: "Technical".into() },
                RequiredSkill { name: "Active Learning".into(), importance: 4.2, category: "Basic".into() },
            ],
            technology_tools: vec![
                TechnologyTool { name: "Python".into(), category: "ProgrammingLanguage".into() },
                TechnologyTool { name: "JavaScript".into(), category: "ProgrammingLanguage".into() },
                TechnologyTool { name: "Git".into(), category: "Software".into() },
                TechnologyTool { name: "Docker".into(), category: "Platform".into() },
                TechnologyTool { name: "AWS/Azure/GCP".into(), category: "Platform".into() },
                TechnologyTool { name: "SQL".into(), category: "ProgrammingLanguage".into() },
            ],
            work_values: Some(WorkValues {
                achievement: 4.2, independence: 3.8, recognition: 3.5,
                relationships: 3.0, support: 3.3, working_conditions: 4.5,
            }),
            work_life_balance: Some(WorkLifeBalance {
                hours_per_week: Some(40.0), schedule_flexibility: Some(0.8),
                remote_availability: Some(0.85), physical_demands: Some(1),
                stress_level: Some(3),
            }),
            career_stages: vec![
                CareerStage { title: "Junior Developer".into(), years_experience: 0, typical_salary_usd: 79_000, key_responsibilities: vec!["Write code under guidance".into(), "Fix bugs".into(), "Write tests".into()] },
                CareerStage { title: "Mid-Level Developer".into(), years_experience: 3, typical_salary_usd: 110_000, key_responsibilities: vec!["Own features end-to-end".into(), "Code review".into(), "Mentor juniors".into()] },
                CareerStage { title: "Senior Developer".into(), years_experience: 6, typical_salary_usd: 150_000, key_responsibilities: vec!["Architecture decisions".into(), "Technical leadership".into(), "Cross-team collaboration".into()] },
                CareerStage { title: "Staff/Principal Engineer".into(), years_experience: 10, typical_salary_usd: 200_000, key_responsibilities: vec!["Org-wide technical strategy".into(), "Solve ambiguous problems".into(), "Define standards".into()] },
            ],
            career_transitions: vec![
                CareerTransition { target_field: "Data Science & AI".into(), skill_overlap_pct: 72, gap_skills: vec!["Statistics".into(), "ML algorithms".into()], transition_months: Some(6), salary_direction: "Similar".into() },
                CareerTransition { target_field: "Product Management".into(), skill_overlap_pct: 55, gap_skills: vec!["Business strategy".into(), "User research".into()], transition_months: Some(12), salary_direction: "Higher".into() },
                CareerTransition { target_field: "Cybersecurity".into(), skill_overlap_pct: 65, gap_skills: vec!["Network security".into(), "Threat modeling".into()], transition_months: Some(6), salary_direction: "Similar".into() },
            ],
            emerging_field: None,
            onet_soc_code: Some("15-1252.00".into()),
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
            work_activities: vec![], required_skills: vec![], technology_tools: vec![],
            work_values: None, work_life_balance: None, career_stages: vec![],
            career_transitions: vec![], emerging_field: None, onet_soc_code: None,
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
            work_activities: vec![], required_skills: vec![], technology_tools: vec![],
            work_values: None, work_life_balance: None, career_stages: vec![],
            career_transitions: vec![], emerging_field: None, onet_soc_code: None,
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
            work_activities: vec![], required_skills: vec![], technology_tools: vec![],
            work_values: None, work_life_balance: None, career_stages: vec![],
            career_transitions: vec![], emerging_field: None, onet_soc_code: None,
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
            work_activities: vec![], required_skills: vec![], technology_tools: vec![],
            work_values: None, work_life_balance: None, career_stages: vec![],
            career_transitions: vec![], emerging_field: None, onet_soc_code: None,
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
            work_activities: vec![], required_skills: vec![], technology_tools: vec![],
            work_values: None, work_life_balance: None, career_stages: vec![],
            career_transitions: vec![], emerging_field: None, onet_soc_code: None,
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
            work_activities: vec![], required_skills: vec![], technology_tools: vec![],
            work_values: None, work_life_balance: None, career_stages: vec![],
            career_transitions: vec![], emerging_field: None, onet_soc_code: None,
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
            work_activities: vec![], required_skills: vec![], technology_tools: vec![],
            work_values: None, work_life_balance: None, career_stages: vec![],
            career_transitions: vec![], emerging_field: None, onet_soc_code: None,
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
            work_activities: vec![], required_skills: vec![], technology_tools: vec![],
            work_values: None, work_life_balance: None, career_stages: vec![],
            career_transitions: vec![], emerging_field: None, onet_soc_code: None,
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
            work_activities: vec![], required_skills: vec![], technology_tools: vec![],
            work_values: None, work_life_balance: None, career_stages: vec![],
            career_transitions: vec![], emerging_field: None, onet_soc_code: None,
        },
    ]
}

/// Emerging career fields not yet in traditional databases.
pub fn embedded_emerging_fields() -> Vec<CareerProfile> {
    vec![
        CareerProfile {
            field: "AI Safety & Alignment".into(),
            median_salary_usd: Some(145_000), entry_salary_usd: Some(90_000), senior_salary_usd: Some(250_000),
            employment_thousands: Some(5), growth_rate_pct: Some(50.0), growth_outlook: GrowthOutlook::MuchFaster,
            annual_openings_thousands: Some(2), typical_education: "Master's/PhD".into(), years_of_education: Some(20),
            related_cip_codes: vec!["11.0701".into(), "42.0101".into()],
            industry_gdp_billions: None, rd_funding_billions: Some(2.0), global_market_billions: Some(8.0),
            automation_risk: Some(0.01), ai_augmentation: Some(0.90),
            sdg_alignment: vec![SdgGoal { number: 16, title: "Peace, Justice & Strong Institutions".into(), contribution: "Ensuring AI systems are safe and aligned with human values".into() }],
            societal_impact_score: Some(95), degree_roi_20yr_thousands: Some(1_200), payback_years: Some(4.0),
            women_pct: Some(25.0), urm_pct: Some(10.0),
            work_activities: vec![WorkActivity { name: "Analyzing Data or Information".into(), importance: 4.8 }, WorkActivity { name: "Thinking Creatively".into(), importance: 4.5 }],
            required_skills: vec![RequiredSkill { name: "Machine Learning".into(), importance: 4.8, category: "Technical".into() }, RequiredSkill { name: "Ethics/Philosophy".into(), importance: 4.5, category: "CrossFunctional".into() }],
            technology_tools: vec![TechnologyTool { name: "Python".into(), category: "ProgrammingLanguage".into() }, TechnologyTool { name: "PyTorch".into(), category: "Framework".into() }],
            work_values: Some(WorkValues { achievement: 4.8, independence: 4.0, recognition: 3.5, relationships: 3.0, support: 3.2, working_conditions: 4.0 }),
            work_life_balance: Some(WorkLifeBalance { hours_per_week: Some(45.0), schedule_flexibility: Some(0.7), remote_availability: Some(0.8), physical_demands: Some(1), stress_level: Some(3) }),
            career_stages: vec![CareerStage { title: "Research Engineer".into(), years_experience: 0, typical_salary_usd: 90_000, key_responsibilities: vec!["Implement safety evaluations".into()] }, CareerStage { title: "Senior Researcher".into(), years_experience: 5, typical_salary_usd: 180_000, key_responsibilities: vec!["Lead alignment research".into()] }],
            career_transitions: vec![CareerTransition { target_field: "Data Science & AI".into(), skill_overlap_pct: 85, gap_skills: vec![], transition_months: Some(3), salary_direction: "Similar".into() }],
            emerging_field: Some(EmergingField { parent_fields: vec!["Data Science & AI".into(), "Philosophy".into()], maturity: "Nascent".into(), estimated_practitioners: Some(5_000), emergence_year: Some(2016), growth_multiplier_5yr: Some(5.0), key_organizations: vec!["Anthropic".into(), "OpenAI".into(), "DeepMind".into(), "MIRI".into()], foundation_skills: vec!["Machine Learning".into(), "Ethics".into(), "Mathematics".into()], luminous_connection: Some("Core concern of Symthaea's moral algebra and ethics engine".into()) }),
            onet_soc_code: None,
        },
        CareerProfile {
            field: "Consciousness Computing".into(),
            median_salary_usd: Some(120_000), entry_salary_usd: Some(75_000), senior_salary_usd: Some(200_000),
            employment_thousands: Some(1), growth_rate_pct: Some(100.0), growth_outlook: GrowthOutlook::MuchFaster,
            annual_openings_thousands: Some(1), typical_education: "PhD".into(), years_of_education: Some(22),
            related_cip_codes: vec!["11.0701".into(), "42.0101".into(), "38.0101".into()],
            industry_gdp_billions: None, rd_funding_billions: Some(0.5), global_market_billions: Some(2.0),
            automation_risk: Some(0.01), ai_augmentation: Some(0.95),
            sdg_alignment: vec![SdgGoal { number: 9, title: "Industry, Innovation & Infrastructure".into(), contribution: "Next-generation computing paradigm".into() }, SdgGoal { number: 3, title: "Good Health & Well-being".into(), contribution: "Consciousness measurement for clinical applications".into() }],
            societal_impact_score: Some(90), degree_roi_20yr_thousands: None, payback_years: None,
            women_pct: Some(20.0), urm_pct: Some(8.0),
            work_activities: vec![WorkActivity { name: "Thinking Creatively".into(), importance: 5.0 }, WorkActivity { name: "Analyzing Data or Information".into(), importance: 4.8 }],
            required_skills: vec![RequiredSkill { name: "HDC/Hyperdimensional Computing".into(), importance: 5.0, category: "Technical".into() }, RequiredSkill { name: "IIT/Phi Measurement".into(), importance: 4.8, category: "Technical".into() }, RequiredSkill { name: "Rust Programming".into(), importance: 4.5, category: "Technical".into() }],
            technology_tools: vec![TechnologyTool { name: "Rust".into(), category: "ProgrammingLanguage".into() }, TechnologyTool { name: "Python/PyPhi".into(), category: "Framework".into() }, TechnologyTool { name: "NixOS".into(), category: "Platform".into() }],
            work_values: Some(WorkValues { achievement: 5.0, independence: 4.5, recognition: 3.0, relationships: 3.5, support: 2.5, working_conditions: 3.5 }),
            work_life_balance: Some(WorkLifeBalance { hours_per_week: Some(50.0), schedule_flexibility: Some(0.6), remote_availability: Some(0.9), physical_demands: Some(1), stress_level: Some(3) }),
            career_stages: vec![CareerStage { title: "Research Associate".into(), years_experience: 0, typical_salary_usd: 75_000, key_responsibilities: vec!["Implement HDC models".into()] }, CareerStage { title: "Lead Researcher".into(), years_experience: 5, typical_salary_usd: 150_000, key_responsibilities: vec!["Design consciousness architectures".into()] }],
            career_transitions: vec![],
            emerging_field: Some(EmergingField { parent_fields: vec!["Computer Science".into(), "Neuroscience".into(), "Philosophy".into()], maturity: "Nascent".into(), estimated_practitioners: Some(500), emergence_year: Some(2020), growth_multiplier_5yr: Some(10.0), key_organizations: vec!["Luminous Dynamics".into(), "Allen Institute".into(), "Tononi Lab (UW)".into()], foundation_skills: vec!["HDC".into(), "IIT".into(), "CfC/LTC".into(), "Rust".into(), "Nix".into()], luminous_connection: Some("This IS Symthaea — the Holographic Liquid Brain. Our core domain.".into()) }),
            onet_soc_code: None,
        },
        CareerProfile {
            field: "Climate Technology".into(),
            median_salary_usd: Some(95_000), entry_salary_usd: Some(60_000), senior_salary_usd: Some(170_000),
            employment_thousands: Some(50), growth_rate_pct: Some(25.0), growth_outlook: GrowthOutlook::MuchFaster,
            annual_openings_thousands: Some(10), typical_education: "Bachelor's/Master's".into(), years_of_education: Some(18),
            related_cip_codes: vec!["03.0104".into(), "14.0101".into()],
            industry_gdp_billions: Some(45.0), rd_funding_billions: Some(25.0), global_market_billions: Some(700.0),
            automation_risk: Some(0.05), ai_augmentation: Some(0.70),
            sdg_alignment: vec![SdgGoal { number: 13, title: "Climate Action".into(), contribution: "Direct climate intervention".into() }, SdgGoal { number: 7, title: "Affordable & Clean Energy".into(), contribution: "Renewable energy technology".into() }],
            societal_impact_score: Some(98), degree_roi_20yr_thousands: Some(500), payback_years: Some(6.0),
            women_pct: Some(35.0), urm_pct: Some(15.0),
            work_activities: vec![], required_skills: vec![], technology_tools: vec![],
            work_values: None, work_life_balance: None, career_stages: vec![],
            career_transitions: vec![], emerging_field: Some(EmergingField { parent_fields: vec!["Environmental Science".into(), "Engineering".into()], maturity: "Emerging".into(), estimated_practitioners: Some(200_000), emergence_year: Some(2015), growth_multiplier_5yr: Some(3.0), key_organizations: vec!["Tesla".into(), "Breakthrough Energy".into(), "IEA".into()], foundation_skills: vec!["Engineering".into(), "Data Science".into(), "Policy".into()], luminous_connection: Some("Mycelix climate cluster tracks carbon and renewable energy".into()) }),
            onet_soc_code: None,
        },
        CareerProfile {
            field: "Quantum Computing".into(),
            median_salary_usd: Some(140_000), entry_salary_usd: Some(85_000), senior_salary_usd: Some(230_000),
            employment_thousands: Some(10), growth_rate_pct: Some(30.0), growth_outlook: GrowthOutlook::MuchFaster,
            annual_openings_thousands: Some(3), typical_education: "PhD".into(), years_of_education: Some(22),
            related_cip_codes: vec!["40.0801".into(), "11.0701".into()],
            industry_gdp_billions: None, rd_funding_billions: Some(5.0), global_market_billions: Some(65.0),
            automation_risk: Some(0.02), ai_augmentation: Some(0.80),
            sdg_alignment: vec![SdgGoal { number: 9, title: "Industry, Innovation & Infrastructure".into(), contribution: "Next-generation computing".into() }],
            societal_impact_score: Some(80), degree_roi_20yr_thousands: Some(900), payback_years: Some(5.0),
            women_pct: Some(20.0), urm_pct: Some(10.0),
            work_activities: vec![], required_skills: vec![], technology_tools: vec![],
            work_values: None, work_life_balance: None, career_stages: vec![],
            career_transitions: vec![], emerging_field: Some(EmergingField { parent_fields: vec!["Physics".into(), "Computer Science".into()], maturity: "Emerging".into(), estimated_practitioners: Some(15_000), emergence_year: Some(2019), growth_multiplier_5yr: Some(4.0), key_organizations: vec!["IBM".into(), "Google Quantum AI".into(), "IonQ".into(), "Rigetti".into()], foundation_skills: vec!["Quantum Mechanics".into(), "Linear Algebra".into(), "Programming".into()], luminous_connection: Some("Symthaea's SubstrateType::QuantumComputer models quantum consciousness substrates".into()) }),
            onet_soc_code: None,
        },
        CareerProfile {
            field: "Decentralized Governance".into(),
            median_salary_usd: Some(100_000), entry_salary_usd: Some(65_000), senior_salary_usd: Some(180_000),
            employment_thousands: Some(20), growth_rate_pct: Some(20.0), growth_outlook: GrowthOutlook::MuchFaster,
            annual_openings_thousands: Some(5), typical_education: "Bachelor's".into(), years_of_education: Some(16),
            related_cip_codes: vec!["11.0701".into(), "45.1001".into()],
            industry_gdp_billions: None, rd_funding_billions: Some(1.0), global_market_billions: Some(30.0),
            automation_risk: Some(0.05), ai_augmentation: Some(0.75),
            sdg_alignment: vec![SdgGoal { number: 16, title: "Peace, Justice & Strong Institutions".into(), contribution: "Decentralized democratic infrastructure".into() }, SdgGoal { number: 10, title: "Reduced Inequalities".into(), contribution: "Community-owned governance systems".into() }],
            societal_impact_score: Some(92), degree_roi_20yr_thousands: Some(600), payback_years: Some(5.0),
            women_pct: Some(28.0), urm_pct: Some(18.0),
            work_activities: vec![], required_skills: vec![], technology_tools: vec![],
            work_values: None, work_life_balance: None, career_stages: vec![],
            career_transitions: vec![], emerging_field: Some(EmergingField { parent_fields: vec!["Computer Science".into(), "Political Science".into()], maturity: "Emerging".into(), estimated_practitioners: Some(50_000), emergence_year: Some(2017), growth_multiplier_5yr: Some(3.0), key_organizations: vec!["Holochain/Holo".into(), "Ethereum Foundation".into(), "RadicalxChange".into()], foundation_skills: vec!["Distributed Systems".into(), "Cryptography".into(), "Governance Theory".into()], luminous_connection: Some("Mycelix IS decentralized governance — 16 Holochain cluster architecture".into()) }),
            onet_soc_code: None,
        },
        CareerProfile {
            field: "Space Economy".into(),
            median_salary_usd: Some(115_000), entry_salary_usd: Some(70_000), senior_salary_usd: Some(190_000),
            employment_thousands: Some(80), growth_rate_pct: Some(15.0), growth_outlook: GrowthOutlook::MuchFaster,
            annual_openings_thousands: Some(8), typical_education: "Bachelor's/Master's".into(), years_of_education: Some(18),
            related_cip_codes: vec!["14.0201".into(), "40.0801".into()],
            industry_gdp_billions: Some(469.0), rd_funding_billions: Some(30.0), global_market_billions: Some(630.0),
            automation_risk: Some(0.08), ai_augmentation: Some(0.65),
            sdg_alignment: vec![SdgGoal { number: 9, title: "Industry, Innovation & Infrastructure".into(), contribution: "Space-based infrastructure".into() }],
            societal_impact_score: Some(75), degree_roi_20yr_thousands: Some(700), payback_years: Some(5.0),
            women_pct: Some(24.0), urm_pct: Some(12.0),
            work_activities: vec![], required_skills: vec![], technology_tools: vec![],
            work_values: None, work_life_balance: None, career_stages: vec![],
            career_transitions: vec![], emerging_field: Some(EmergingField { parent_fields: vec!["Aerospace Engineering".into(), "Physics".into()], maturity: "Establishing".into(), estimated_practitioners: Some(300_000), emergence_year: Some(2012), growth_multiplier_5yr: Some(2.0), key_organizations: vec!["SpaceX".into(), "Blue Origin".into(), "NASA".into(), "ESA".into()], foundation_skills: vec!["Aerospace Engineering".into(), "Physics".into(), "Systems Engineering".into()], luminous_connection: Some("Mycelix space cluster + multiworld-sim for interplanetary infrastructure".into()) }),
            onet_soc_code: None,
        },
        CareerProfile {
            field: "Longevity & Biotech".into(),
            median_salary_usd: Some(105_000), entry_salary_usd: Some(65_000), senior_salary_usd: Some(200_000),
            employment_thousands: Some(30), growth_rate_pct: Some(20.0), growth_outlook: GrowthOutlook::MuchFaster,
            annual_openings_thousands: Some(5), typical_education: "PhD".into(), years_of_education: Some(22),
            related_cip_codes: vec!["26.0101".into(), "26.1201".into()],
            industry_gdp_billions: None, rd_funding_billions: Some(15.0), global_market_billions: Some(110.0),
            automation_risk: Some(0.03), ai_augmentation: Some(0.85),
            sdg_alignment: vec![SdgGoal { number: 3, title: "Good Health & Well-being".into(), contribution: "Extending healthy human lifespan".into() }],
            societal_impact_score: Some(88), degree_roi_20yr_thousands: Some(450), payback_years: Some(8.0),
            women_pct: Some(45.0), urm_pct: Some(12.0),
            work_activities: vec![], required_skills: vec![], technology_tools: vec![],
            work_values: None, work_life_balance: None, career_stages: vec![],
            career_transitions: vec![], emerging_field: Some(EmergingField { parent_fields: vec!["Biology".into(), "Medicine".into()], maturity: "Emerging".into(), estimated_practitioners: Some(50_000), emergence_year: Some(2013), growth_multiplier_5yr: Some(3.5), key_organizations: vec!["Altos Labs".into(), "Calico (Alphabet)".into(), "Unity Biotechnology".into()], foundation_skills: vec!["Molecular Biology".into(), "Bioinformatics".into(), "Statistics".into()], luminous_connection: None }),
            onet_soc_code: None,
        },
        CareerProfile {
            field: "Synthetic Biology".into(),
            median_salary_usd: Some(100_000), entry_salary_usd: Some(60_000), senior_salary_usd: Some(180_000),
            employment_thousands: Some(15), growth_rate_pct: Some(25.0), growth_outlook: GrowthOutlook::MuchFaster,
            annual_openings_thousands: Some(3), typical_education: "PhD".into(), years_of_education: Some(22),
            related_cip_codes: vec!["26.0101".into(), "14.0501".into()],
            industry_gdp_billions: None, rd_funding_billions: Some(8.0), global_market_billions: Some(35.0),
            automation_risk: Some(0.05), ai_augmentation: Some(0.80),
            sdg_alignment: vec![SdgGoal { number: 2, title: "Zero Hunger".into(), contribution: "Engineered crops and food production".into() }, SdgGoal { number: 3, title: "Good Health & Well-being".into(), contribution: "Engineered therapeutics and diagnostics".into() }],
            societal_impact_score: Some(85), degree_roi_20yr_thousands: Some(400), payback_years: Some(9.0),
            women_pct: Some(40.0), urm_pct: Some(14.0),
            work_activities: vec![], required_skills: vec![], technology_tools: vec![],
            work_values: None, work_life_balance: None, career_stages: vec![],
            career_transitions: vec![], emerging_field: Some(EmergingField { parent_fields: vec!["Biology".into(), "Engineering".into()], maturity: "Emerging".into(), estimated_practitioners: Some(30_000), emergence_year: Some(2010), growth_multiplier_5yr: Some(3.0), key_organizations: vec!["Ginkgo Bioworks".into(), "Zymergen".into(), "iGEM Foundation".into()], foundation_skills: vec!["Molecular Biology".into(), "Genetic Engineering".into(), "Bioinformatics".into()], luminous_connection: None }),
            onet_soc_code: None,
        },
    ]
}

/// Load O*NET enrichment data (bundled at compile time) and merge into profiles.
pub fn enrich_profiles_with_onet(profiles: &mut [CareerProfile]) {
    let onet_json = include_str!("../data/onet_enrichment.json");
    let onet_data: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_str(onet_json).unwrap_or_default();

    for profile in profiles.iter_mut() {
        if let Some(data) = onet_data.get(&profile.field) {
            // Only enrich if not already populated
            if profile.required_skills.is_empty() {
                if let Some(skills) = data.get("skills").and_then(|v| v.as_array()) {
                    profile.required_skills = skills
                        .iter()
                        .filter_map(|s| {
                            Some(RequiredSkill {
                                name: s.get("name")?.as_str()?.to_string(),
                                importance: s.get("importance")?.as_f64()? as f32,
                                category: "O*NET".to_string(),
                            })
                        })
                        .collect();
                }
            }

            if profile.work_activities.is_empty() {
                if let Some(acts) = data.get("activities").and_then(|v| v.as_array()) {
                    profile.work_activities = acts
                        .iter()
                        .filter_map(|a| {
                            Some(WorkActivity {
                                name: a.get("name")?.as_str()?.to_string(),
                                importance: a.get("importance")?.as_f64()? as f32,
                            })
                        })
                        .collect();
                }
            }

            if profile.technology_tools.is_empty() {
                if let Some(tech) = data.get("tech").and_then(|v| v.as_array()) {
                    profile.technology_tools = tech
                        .iter()
                        .filter_map(|t| {
                            Some(TechnologyTool {
                                name: t.get("name")?.as_str()?.to_string(),
                                category: t.get("category")?.as_str().unwrap_or("Software").to_string(),
                            })
                        })
                        .collect();
                }
            }

            if profile.work_values.is_none() {
                if let Some(wv) = data.get("work_values").and_then(|v| v.as_object()) {
                    if !wv.is_empty() {
                        profile.work_values = Some(WorkValues {
                            achievement: wv.get("Achievement").and_then(|v| v.as_f64()).unwrap_or(3.0) as f32,
                            independence: wv.get("Independence").and_then(|v| v.as_f64()).unwrap_or(3.0) as f32,
                            recognition: wv.get("Recognition").and_then(|v| v.as_f64()).unwrap_or(3.0) as f32,
                            relationships: wv.get("Relationships").and_then(|v| v.as_f64()).unwrap_or(3.0) as f32,
                            support: wv.get("Support").and_then(|v| v.as_f64()).unwrap_or(3.0) as f32,
                            working_conditions: wv.get("Working Conditions").and_then(|v| v.as_f64()).unwrap_or(3.0) as f32,
                        });
                    }
                }
            }

            // Set O*NET SOC code
            if profile.onet_soc_code.is_none() {
                if let Some(soc) = data.get("soc").and_then(|v| v.as_str()) {
                    profile.onet_soc_code = Some(soc.to_string());
                }
            }
        }
    }
}

/// All career profiles, enriched with real O*NET data.
pub fn all_career_profiles_enriched() -> Vec<CareerProfile> {
    let mut profiles = all_career_profiles();
    enrich_profiles_with_onet(&mut profiles);
    profiles
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

/// Format a detailed career profile (includes O*NET enriched data).
pub fn format_career_detail(profile: &CareerProfile) -> String {
    let mut s = format_career_summary(profile);

    if let Some(ref ef) = profile.emerging_field {
        s += &format!("\n  [EMERGING FIELD] Maturity: {}, Est. practitioners: {}\n",
            ef.maturity, ef.estimated_practitioners.map(|n| format_thousands(n)).unwrap_or("unknown".into()));
        if !ef.key_organizations.is_empty() {
            s += &format!("  Key orgs: {}\n", ef.key_organizations.join(", "));
        }
        if let Some(ref lc) = ef.luminous_connection {
            s += &format!("  Luminous: {}\n", lc);
        }
    }

    if !profile.work_activities.is_empty() {
        s += "\n  Day-to-day activities:\n";
        for a in &profile.work_activities {
            let bar = "#".repeat((a.importance * 4.0) as usize);
            s += &format!("    {:<45} {:.1} {}\n", a.name, a.importance, bar);
        }
    }

    if !profile.required_skills.is_empty() {
        s += "\n  Required skills:\n";
        for sk in &profile.required_skills {
            let bar = "#".repeat((sk.importance * 4.0) as usize);
            s += &format!("    {:<45} {:.1} {}\n", sk.name, sk.importance, bar);
        }
    }

    if !profile.technology_tools.is_empty() {
        s += "\n  Tools & Technologies:\n";
        for t in &profile.technology_tools {
            s += &format!("    {} ({})\n", t.name, t.category);
        }
    }

    if let Some(ref wv) = profile.work_values {
        s += &format!("\n  Work values: Achievement {:.1} | Independence {:.1} | Recognition {:.1} | Relationships {:.1} | Support {:.1} | Conditions {:.1}\n",
            wv.achievement, wv.independence, wv.recognition, wv.relationships, wv.support, wv.working_conditions);
    }

    if let Some(ref wlb) = profile.work_life_balance {
        s += "\n  Work-life balance:\n";
        if let Some(h) = wlb.hours_per_week { s += &format!("    Hours/week: {:.0}\n", h); }
        if let Some(r) = wlb.remote_availability { s += &format!("    Remote:     {:.0}%\n", r * 100.0); }
        if let Some(f) = wlb.schedule_flexibility { s += &format!("    Flexibility:{:.0}%\n", f * 100.0); }
        if let Some(stress) = wlb.stress_level { s += &format!("    Stress:     {}/5\n", stress); }
    }

    if !profile.career_stages.is_empty() {
        s += "\n  Career progression:\n";
        for stage in &profile.career_stages {
            s += &format!("    {}yr+ {:30} ${}\n",
                stage.years_experience, stage.title, format_thousands(stage.typical_salary_usd));
        }
    }

    if !profile.career_transitions.is_empty() {
        s += "\n  Career transitions:\n";
        for t in &profile.career_transitions {
            s += &format!("    -> {} ({}% skill overlap, {} salary, ~{}mo)\n",
                t.target_field, t.skill_overlap_pct, t.salary_direction,
                t.transition_months.unwrap_or(0));
        }
    }

    s
}

/// Compare two career profiles side-by-side.
pub fn format_career_comparison(a: &CareerProfile, b: &CareerProfile) -> String {
    let mut s = format!("{:>30} vs {}\n", a.field, b.field);
    s += &format!("{:>30}    {}\n", "=".repeat(a.field.len().min(30)), "=".repeat(b.field.len().min(30)));

    let fmt_opt_u32 = |v: Option<u32>| v.map(|n| format!("${}", format_thousands(n))).unwrap_or("-".into());
    let fmt_opt_f32 = |v: Option<f32>| v.map(|n| format!("{:.1}%", n)).unwrap_or("-".into());
    let fmt_opt_u8 = |v: Option<u8>| v.map(|n| format!("{}", n)).unwrap_or("-".into());

    s += &format!("{:>30}    {}\n", fmt_opt_u32(a.median_salary_usd), fmt_opt_u32(b.median_salary_usd));
    s += &format!("{:>30}    {}\n", fmt_opt_f32(a.growth_rate_pct), fmt_opt_f32(b.growth_rate_pct));
    s += &format!("{:>30}    {}\n",
        a.automation_risk.map(|r| format!("{:.0}% automation risk", r * 100.0)).unwrap_or("-".into()),
        b.automation_risk.map(|r| format!("{:.0}% automation risk", r * 100.0)).unwrap_or("-".into()));
    s += &format!("{:>30}    {}\n",
        fmt_opt_u8(a.societal_impact_score).to_string() + "/100 impact",
        fmt_opt_u8(b.societal_impact_score).to_string() + "/100 impact");

    // Shared skills
    let a_skills: std::collections::HashSet<_> = a.required_skills.iter().map(|s| s.name.clone()).collect();
    let b_skills: std::collections::HashSet<_> = b.required_skills.iter().map(|s| s.name.clone()).collect();
    let shared: Vec<_> = a_skills.intersection(&b_skills).collect();
    let a_only: Vec<_> = a_skills.difference(&b_skills).collect();
    let b_only: Vec<_> = b_skills.difference(&a_skills).collect();

    if !shared.is_empty() {
        s += &format!("\n  Shared skills: {}\n", shared.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "));
    }
    if !a_only.is_empty() {
        s += &format!("  Only {}: {}\n", a.field, a_only.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "));
    }
    if !b_only.is_empty() {
        s += &format!("  Only {}: {}\n", b.field, b_only.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "));
    }

    s += &format!("\n  Attractiveness: {} ({}) vs {} ({})\n",
        a.attractiveness_score(), a.field,
        b.attractiveness_score(), b.field);

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

    #[test]
    fn test_all_profiles_includes_emerging() {
        let all = all_career_profiles();
        assert!(all.len() > embedded_career_profiles().len());
        assert!(all.iter().any(|p| p.emerging_field.is_some()));
    }

    #[test]
    fn test_software_dev_enriched() {
        let profiles = embedded_career_profiles();
        let sw = profiles.iter().find(|p| p.field.contains("Software")).unwrap();
        assert!(!sw.work_activities.is_empty());
        assert!(!sw.required_skills.is_empty());
        assert!(!sw.technology_tools.is_empty());
        assert!(sw.work_values.is_some());
        assert!(sw.work_life_balance.is_some());
        assert!(!sw.career_stages.is_empty());
        assert!(!sw.career_transitions.is_empty());
        assert!(sw.onet_soc_code.is_some());
    }

    #[test]
    fn test_emerging_field_consciousness() {
        let emerging = embedded_emerging_fields();
        let cc = emerging.iter().find(|p| p.field.contains("Consciousness")).unwrap();
        assert!(cc.emerging_field.is_some());
        let ef = cc.emerging_field.as_ref().unwrap();
        assert!(ef.luminous_connection.is_some());
        assert!(ef.maturity == "Nascent");
    }

    #[test]
    fn test_emerging_fields_count() {
        let emerging = embedded_emerging_fields();
        assert_eq!(emerging.len(), 8);
    }
}
