//! Design-stage power and precision planning at the independent model-pair level.
//!
//! The independent observations are independently trained treatment/placebo
//! model pairs. Tasks and generations within a pair are measurements, not
//! additional training replicates.
//!
//! For each scenario the estimated pair-level standard deviation is modeled as
//! sqrt(between_pair_sd^2 + within_pair_item_sd^2 / items_per_pair). The
//! noncentral-t power calculation uses n independent pair effects and df=n-1.
//! Power is computed by deterministic numerical integration of the noncentral-t
//! representation, not by Monte Carlo pseudo-replication over tasks.

use serde::{Deserialize, Serialize};

pub const CAUSAL_POWER_REQUEST_SCHEMA: &str = "symthaea.causal_power_plan_request.v1";
pub const CAUSAL_POWER_REPORT_SCHEMA: &str = "symthaea.causal_power_plan_report.v1";
pub const MAX_POWER_SCENARIOS: usize = 32;
pub const MAX_ITEMS_PER_PAIR_SCENARIOS: usize = 32;
pub const MAX_ATTRITION_SCENARIOS: usize = 20;
pub const MAX_INDEPENDENT_MODEL_PAIRS: u32 = 500;
pub const MAX_CONTRASTS: u32 = 100;
/// Bound output size and numerical integration work before starting the plan.
pub const MAX_POWER_CURVE_POINTS: usize = 5_000;
const NONCENTRAL_T_SIMPSON_STEPS: usize = 4_096;
const BETA_MAX_ITERATIONS: usize = 256;
const BETA_EPSILON: f64 = 3.0e-14;
const BETA_FPMIN: f64 = 1.0e-300;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TestTailV1 {
    TwoSided,
    Greater,
    Less,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MultiplicityPolicyV1 {
    None,
    Bonferroni,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PowerScenarioV1 {
    pub scenario_id: String,
    /// Expected mean treatment-minus-placebo effect for one independent pair.
    pub true_effect_permille: f64,
    /// Between-pair standard deviation after pair-level aggregation.
    pub between_pair_sd_permille: f64,
    /// Within-pair item measurement standard deviation before averaging items.
    pub within_pair_item_sd_permille: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalPowerPlanRequestV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub primary_estimand_id: String,
    pub independent_unit: String,
    pub alpha_permille: u16,
    pub test_tail: TestTailV1,
    pub multiplicity_policy: MultiplicityPolicyV1,
    pub contrast_count: u32,
    pub target_power_permille: u16,
    pub target_ci_half_width_permille: f64,
    pub max_gross_model_pair_replicates: u32,
    pub attrition_rates_permille: Vec<u16>,
    pub items_per_pair_scenarios: Vec<u32>,
    pub effect_scenarios: Vec<PowerScenarioV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PairCountPowerPointV1 {
    pub gross_independent_model_pairs: u32,
    pub expected_completed_model_pairs: u32,
    pub degrees_of_freedom: Option<u32>,
    pub effective_pair_effect_sd_permille: Option<f64>,
    pub noncentrality_parameter: Option<f64>,
    pub power_permille: Option<f64>,
    pub expected_two_sided_95_ci_half_width_permille: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PowerScenarioResultV1 {
    pub scenario_id: String,
    pub true_effect_permille: f64,
    pub between_pair_sd_permille: f64,
    pub within_pair_item_sd_permille: f64,
    pub attrition_rate_permille: u16,
    pub items_per_pair: u32,
    pub adjusted_alpha_permille: f64,
    pub minimum_gross_pairs_for_target_power: Option<u32>,
    pub minimum_gross_pairs_for_target_precision: Option<u32>,
    pub curve: Vec<PairCountPowerPointV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CausalPowerPlanReportV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub primary_estimand_id: String,
    pub independent_unit: String,
    pub request_sha256: String,
    pub multiplicity_policy: MultiplicityPolicyV1,
    pub contrast_count: u32,
    pub familywise_alpha_permille: u16,
    pub adjusted_alpha_policy: String,
    pub target_power_permille: u16,
    pub target_ci_half_width_permille: f64,
    pub max_gross_model_pair_replicates: u32,
    pub attrition_scenarios_permille: Vec<u16>,
    pub items_per_pair_scenarios: Vec<u32>,
    pub independent_degrees_of_freedom_rule: String,
    pub within_pair_measurement_rule: String,
    pub numerical_method: String,
    pub results: Vec<PowerScenarioResultV1>,
    pub threshold_selection_used_evaluation_data: bool,
    pub questions_counted_as_independent_training_replicates: bool,
    pub independently_authenticated: bool,
    pub confirmatory_eligible: bool,
    pub analysis_status: String,
    pub issues: Vec<String>,
    pub claim_boundary: String,
}

fn valid_opaque_id(value: &str, max_len: usize) -> bool {
    !value.trim().is_empty()
        && value.len() <= max_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
}

fn valid_number(value: f64, label: &str) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{label} must be finite"))
    }
}

/// Lanczos approximation for log(Gamma(x)); all planner arguments use x > 0.
fn log_gamma(value: f64) -> f64 {
    const COEFFICIENTS: [f64; 8] = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if value < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * value).sin().ln()
            - log_gamma(1.0 - value);
    }
    let z = value - 1.0;
    let mut x = 0.999_999_999_999_809_9;
    for (index, coefficient) in COEFFICIENTS.iter().enumerate() {
        x += coefficient / (z + index as f64 + 1.0);
    }
    let t = z + COEFFICIENTS.len() as f64 - 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < BETA_FPMIN {
        d = BETA_FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=BETA_MAX_ITERATIONS {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        let mut aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < BETA_FPMIN {
            d = BETA_FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < BETA_FPMIN {
            c = BETA_FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < BETA_FPMIN {
            d = BETA_FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < BETA_FPMIN {
            c = BETA_FPMIN;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() <= BETA_EPSILON {
            break;
        }
    }
    h
}

fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let log_bt = log_gamma(a + b) - log_gamma(a) - log_gamma(b)
        + a * x.ln()
        + b * (1.0 - x).ln();
    let bt = log_bt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        (bt * beta_continued_fraction(a, b, x) / a).clamp(0.0, 1.0)
    } else {
        (1.0 - bt * beta_continued_fraction(b, a, 1.0 - x) / b).clamp(0.0, 1.0)
    }
}

fn student_t_cdf(value: f64, degrees_of_freedom: u32) -> f64 {
    if value == 0.0 {
        return 0.5;
    }
    let nu = degrees_of_freedom as f64;
    let x = nu / (nu + value * value);
    let beta = regularized_incomplete_beta(nu / 2.0, 0.5, x);
    if value > 0.0 {
        1.0 - 0.5 * beta
    } else {
        0.5 * beta
    }
}

/// Central Student-t critical value by monotone bisection.
fn student_t_critical_value(alpha: f64, tail: TestTailV1, df: u32) -> Result<f64, String> {
    if !(0.0..1.0).contains(&alpha) || df == 0 {
        return Err("critical-value alpha and degrees of freedom are invalid".to_owned());
    }
    let probability = match tail {
        TestTailV1::TwoSided => 1.0 - alpha / 2.0,
        TestTailV1::Greater | TestTailV1::Less => 1.0 - alpha,
    };
    let mut low = 0.0;
    let mut high = 1.0;
    while student_t_cdf(high, df) < probability && high < 1.0e8 {
        high *= 2.0;
    }
    if high >= 1.0e8 {
        return Err("could not bracket Student-t critical value".to_owned());
    }
    for _ in 0..100 {
        let mid = (low + high) / 2.0;
        if student_t_cdf(mid, df) < probability {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok((low + high) / 2.0)
}

/// Abramowitz-Stegun normal CDF approximation; maximum absolute error is
/// approximately 7.5e-8 across the real line.
fn normal_cdf(value: f64) -> f64 {
    let absolute = value.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * absolute);
    let density = 0.398_942_280_401_432_7 * (-0.5 * absolute * absolute).exp();
    let polynomial = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let positive = 1.0 - density * polynomial;
    if value >= 0.0 {
        positive
    } else {
        1.0 - positive
    }
}

fn noncentral_t_power(
    noncentrality: f64,
    df: u32,
    alpha: f64,
    tail: TestTailV1,
) -> Result<f64, String> {
    if df == 0 || !noncentrality.is_finite() {
        return Err("noncentral-t parameters are invalid".to_owned());
    }
    let critical = student_t_critical_value(alpha, tail, df)?;
    let nu = df as f64;
    let x_max = ((nu + 14.0 * (2.0 * nu).sqrt() + 200.0) / nu).sqrt();
    let steps = NONCENTRAL_T_SIMPSON_STEPS;
    let step = x_max / steps as f64;
    let log_constant = 2.0_f64.ln() + (nu / 2.0) * nu.ln()
        - (nu / 2.0) * 2.0_f64.ln()
        - log_gamma(nu / 2.0);

    let integrand = |x: f64| -> f64 {
        let log_density = if x == 0.0 {
            if df == 1 {
                log_constant
            } else {
                f64::NEG_INFINITY
            }
        } else {
            log_constant + (nu - 1.0) * x.ln() - 0.5 * nu * x * x
        };
        let density = log_density.exp();
        let threshold = critical * x;
        let conditional_power = match tail {
            TestTailV1::TwoSided => {
                normal_cdf(noncentrality - threshold) + normal_cdf(-noncentrality - threshold)
            }
            TestTailV1::Greater => normal_cdf(noncentrality - threshold),
            TestTailV1::Less => normal_cdf(-noncentrality - threshold),
        };
        density * conditional_power
    };

    let mut sum = integrand(0.0) + integrand(x_max);
    for index in 1..steps {
        let x = index as f64 * step;
        sum += if index.is_multiple_of(2) {
            2.0 * integrand(x)
        } else {
            4.0 * integrand(x)
        };
    }
    let power = sum * step / 3.0;
    if power.is_finite() {
        Ok(power.clamp(0.0, 1.0))
    } else {
        Err("noncentral-t power integration produced a non-finite result".to_owned())
    }
}

fn effective_pair_sd(
    between_pair_sd_permille: f64,
    within_pair_item_sd_permille: f64,
    items_per_pair: u32,
) -> Result<f64, String> {
    if items_per_pair == 0 {
        return Err("items_per_pair must be positive".to_owned());
    }
    let between = between_pair_sd_permille;
    let within = within_pair_item_sd_permille / (items_per_pair as f64).sqrt();
    let standard_deviation = (between * between + within * within).sqrt();
    if standard_deviation.is_finite() && standard_deviation > 0.0 {
        Ok(standard_deviation)
    } else {
        Err("effective paired-effect standard deviation must be positive and finite".to_owned())
    }
}

fn adjusted_alpha(request: &CausalPowerPlanRequestV1) -> f64 {
    let familywise_alpha = f64::from(request.alpha_permille) / 1_000.0;
    match request.multiplicity_policy {
        MultiplicityPolicyV1::None => familywise_alpha,
        MultiplicityPolicyV1::Bonferroni => familywise_alpha / f64::from(request.contrast_count),
    }
}

fn expected_completed_pairs(gross: u32, attrition_permille: u16) -> u32 {
    (u64::from(gross) * u64::from(1_000 - attrition_permille) / 1_000) as u32
}

fn analyze_one_scenario(
    request: &CausalPowerPlanRequestV1,
    scenario: &PowerScenarioV1,
    attrition_permille: u16,
    items_per_pair: u32,
    alpha: f64,
) -> Result<PowerScenarioResultV1, String> {
    let pair_sd = effective_pair_sd(
        scenario.between_pair_sd_permille,
        scenario.within_pair_item_sd_permille,
        items_per_pair,
    )?;
    let mut curve = Vec::with_capacity(request.max_gross_model_pair_replicates as usize);
    let mut minimum_power_gross = None;
    let mut minimum_precision_gross = None;

    for gross in 2..=request.max_gross_model_pair_replicates {
        let completed = expected_completed_pairs(gross, attrition_permille);
        if completed < 2 {
            curve.push(PairCountPowerPointV1 {
                gross_independent_model_pairs: gross,
                expected_completed_model_pairs: completed,
                degrees_of_freedom: None,
                effective_pair_effect_sd_permille: Some(pair_sd),
                noncentrality_parameter: None,
                power_permille: None,
                expected_two_sided_95_ci_half_width_permille: None,
            });
            continue;
        }
        let df = completed - 1;
        let standard_error = pair_sd / (completed as f64).sqrt();
        let noncentrality = scenario.true_effect_permille / standard_error;
        let power = noncentral_t_power(
            noncentrality,
            df,
            alpha,
            request.test_tail,
        )?;
        let precision_critical = student_t_critical_value(0.05, TestTailV1::TwoSided, df)?;
        let ci_half_width = precision_critical * standard_error;

        if minimum_power_gross.is_none()
            && power * 1_000.0 >= f64::from(request.target_power_permille)
        {
            minimum_power_gross = Some(gross);
        }
        if minimum_precision_gross.is_none()
            && ci_half_width <= request.target_ci_half_width_permille
        {
            minimum_precision_gross = Some(gross);
        }
        curve.push(PairCountPowerPointV1 {
            gross_independent_model_pairs: gross,
            expected_completed_model_pairs: completed,
            degrees_of_freedom: Some(df),
            effective_pair_effect_sd_permille: Some(pair_sd),
            noncentrality_parameter: Some(noncentrality),
            power_permille: Some(power * 1_000.0),
            expected_two_sided_95_ci_half_width_permille: Some(ci_half_width),
        });
    }

    Ok(PowerScenarioResultV1 {
        scenario_id: scenario.scenario_id.clone(),
        true_effect_permille: scenario.true_effect_permille,
        between_pair_sd_permille: scenario.between_pair_sd_permille,
        within_pair_item_sd_permille: scenario.within_pair_item_sd_permille,
        attrition_rate_permille: attrition_permille,
        items_per_pair,
        adjusted_alpha_permille: alpha * 1_000.0,
        minimum_gross_pairs_for_target_power: minimum_power_gross,
        minimum_gross_pairs_for_target_precision: minimum_precision_gross,
        curve,
    })
}

fn validate_request(request: &CausalPowerPlanRequestV1) -> Result<(), String> {
    if request.schema != CAUSAL_POWER_REQUEST_SCHEMA {
        return Err(format!("request schema must be {CAUSAL_POWER_REQUEST_SCHEMA}"));
    }
    for (value, label) in [
        (&request.study_id, "study_id"),
        (&request.benchmark_id, "benchmark_id"),
        (&request.benchmark_version, "benchmark_version"),
        (&request.primary_estimand_id, "primary_estimand_id"),
        (&request.independent_unit, "independent_unit"),
    ] {
        if !valid_opaque_id(value, 128) {
            return Err(format!("{label} must be a nonempty opaque ASCII identifier"));
        }
    }
    if request.independent_unit != "independent_model_pair" {
        return Err("independent_unit must be independent_model_pair; item-level replication is prohibited".to_owned());
    }
    if !(1..=200).contains(&request.alpha_permille) {
        return Err("alpha_permille must be in 1..=200".to_owned());
    }
    if !(500..=990).contains(&request.target_power_permille) {
        return Err("target_power_permille must be in 500..=990".to_owned());
    }
    valid_number(request.target_ci_half_width_permille, "target_ci_half_width_permille")?;
    if request.target_ci_half_width_permille <= 0.0 {
        return Err("target_ci_half_width_permille must be positive".to_owned());
    }
    if !(2..=MAX_INDEPENDENT_MODEL_PAIRS).contains(&request.max_gross_model_pair_replicates) {
        return Err(format!(
            "max_gross_model_pair_replicates must be in 2..={MAX_INDEPENDENT_MODEL_PAIRS}"
        ));
    }
    if !(1..=MAX_CONTRASTS).contains(&request.contrast_count) {
        return Err(format!("contrast_count must be in 1..={MAX_CONTRASTS}"));
    }
    if request.multiplicity_policy == MultiplicityPolicyV1::None && request.contrast_count != 1 {
        return Err("multiplicity_policy=none requires exactly one confirmatory contrast".to_owned());
    }
    if request.attrition_rates_permille.is_empty()
        || request.attrition_rates_permille.len() > MAX_ATTRITION_SCENARIOS
        || request.attrition_rates_permille.iter().any(|rate| *rate > 900)
    {
        return Err("attrition rates must contain 1..=20 values in 0..=900 permille".to_owned());
    }
    if request.items_per_pair_scenarios.is_empty()
        || request.items_per_pair_scenarios.len() > MAX_ITEMS_PER_PAIR_SCENARIOS
        || request.items_per_pair_scenarios.contains(&0)
    {
        return Err("items_per_pair_scenarios must contain 1..=32 positive values".to_owned());
    }
    if request.effect_scenarios.is_empty() || request.effect_scenarios.len() > MAX_POWER_SCENARIOS {
        return Err("effect_scenarios must contain 1..=32 scenarios".to_owned());
    }
    let curve_points = request
        .effect_scenarios
        .len()
        .checked_mul(request.attrition_rates_permille.len())
        .and_then(|count| count.checked_mul(request.items_per_pair_scenarios.len()))
        .and_then(|count| {
            count.checked_mul(request.max_gross_model_pair_replicates.saturating_sub(1) as usize)
        })
        .ok_or_else(|| "power-plan curve size overflows".to_owned())?;
    if curve_points > MAX_POWER_CURVE_POINTS {
        return Err(format!(
            "power plan requests {curve_points} curve points, above the bounded maximum {MAX_POWER_CURVE_POINTS}"
        ));
    }
    let unique_attrition: std::collections::BTreeSet<u16> =
        request.attrition_rates_permille.iter().copied().collect();
    if unique_attrition.len() != request.attrition_rates_permille.len() {
        return Err("attrition-rate scenarios must be unique".to_owned());
    }
    let unique_items: std::collections::BTreeSet<u32> =
        request.items_per_pair_scenarios.iter().copied().collect();
    if unique_items.len() != request.items_per_pair_scenarios.len() {
        return Err("items-per-pair scenarios must be unique".to_owned());
    }
    let mut scenario_ids = std::collections::BTreeSet::new();
    for scenario in &request.effect_scenarios {
        if !valid_opaque_id(&scenario.scenario_id, 64)
            || !scenario_ids.insert(scenario.scenario_id.as_str())
        {
            return Err("scenario IDs must be unique nonempty opaque identifiers".to_owned());
        }
        for (value, label) in [
            (scenario.true_effect_permille, "true_effect_permille"),
            (scenario.between_pair_sd_permille, "between_pair_sd_permille"),
            (scenario.within_pair_item_sd_permille, "within_pair_item_sd_permille"),
        ] {
            valid_number(value, label)?;
        }
        if scenario.between_pair_sd_permille < 0.0
            || scenario.within_pair_item_sd_permille < 0.0
            || (scenario.between_pair_sd_permille == 0.0
                && scenario.within_pair_item_sd_permille == 0.0)
        {
            return Err("scenario SD components must be non-negative and not both zero".to_owned());
        }
        if (request.test_tail == TestTailV1::Greater && scenario.true_effect_permille < 0.0)
            || (request.test_tail == TestTailV1::Less && scenario.true_effect_permille > 0.0)
        {
            return Err("one-sided test direction must agree with the declared treatment-minus-placebo effect direction".to_owned());
        }
    }
    Ok(())
}

/// Build power and expected-precision curves without treating task examples as
/// independent training replicates. Thresholds are not fit and no model is run.
pub fn plan_causal_power(
    request: &CausalPowerPlanRequestV1,
) -> Result<CausalPowerPlanReportV1, String> {
    validate_request(request)?;
    let bytes = serde_json::to_vec(request)
        .map_err(|error| format!("cannot serialize power-plan request: {error}"))?;
    let request_sha256 = crate::hash_domain(b"causal-power-plan-request.v1", &bytes);
    let alpha = adjusted_alpha(request);
    let mut results = Vec::new();
    for scenario in &request.effect_scenarios {
        for attrition in &request.attrition_rates_permille {
            for items_per_pair in &request.items_per_pair_scenarios {
                results.push(analyze_one_scenario(
                    request,
                    scenario,
                    *attrition,
                    *items_per_pair,
                    alpha,
                )?);
            }
        }
    }
    Ok(CausalPowerPlanReportV1 {
        schema: CAUSAL_POWER_REPORT_SCHEMA.to_owned(),
        study_id: request.study_id.clone(),
        benchmark_id: request.benchmark_id.clone(),
        benchmark_version: request.benchmark_version.clone(),
        primary_estimand_id: request.primary_estimand_id.clone(),
        independent_unit: request.independent_unit.clone(),
        request_sha256,
        multiplicity_policy: request.multiplicity_policy,
        contrast_count: request.contrast_count,
        familywise_alpha_permille: request.alpha_permille,
        adjusted_alpha_policy: match request.multiplicity_policy {
            MultiplicityPolicyV1::None => "single confirmatory contrast; no adjustment".to_owned(),
            MultiplicityPolicyV1::Bonferroni => {
                "Bonferroni familywise control; per-contrast alpha = familywise alpha / contrast_count".to_owned()
            }
        },
        target_power_permille: request.target_power_permille,
        target_ci_half_width_permille: request.target_ci_half_width_permille,
        max_gross_model_pair_replicates: request.max_gross_model_pair_replicates,
        attrition_scenarios_permille: request.attrition_rates_permille.clone(),
        items_per_pair_scenarios: request.items_per_pair_scenarios.clone(),
        independent_degrees_of_freedom_rule: "df = expected completed independently trained treatment/placebo model pairs - 1; questions, source clusters and repeated generations are not independent training replicates".to_owned(),
        within_pair_measurement_rule: "effective pair-level SD = sqrt(between_pair_SD^2 + within_pair_item_SD^2 / items_per_pair); increasing items reduces the declared within-pair measurement component but never increases independent model-pair degrees of freedom".to_owned(),
        numerical_method: "noncentral-t power integrated over the chi-square scale mixture using fixed-step Simpson quadrature; central-t critical values use regularized incomplete beta and bisection; normal CDF uses Abramowitz-Stegun approximation".to_owned(),
        results,
        threshold_selection_used_evaluation_data: false,
        questions_counted_as_independent_training_replicates: false,
        independently_authenticated: false,
        confirmatory_eligible: false,
        analysis_status: "design_sensitivity_only".to_owned(),
        issues: vec![
            "power is conditional on the declared mean effect, between-pair SD, within-pair measurement SD and paired-normal model; model-family heterogeneity must be represented in the scenario SD or a separate stratified design".to_owned(),
            "expected completed pairs are the deterministic floor of gross planned pairs times one minus attrition; random attrition variance and informative missing-pair mechanisms are not simulated in v1".to_owned(),
            "Bonferroni is conservative; Holm's step-down procedure is not represented as a single fixed per-contrast alpha in this planner".to_owned(),
            "numerical integration and the normal CDF approximation introduce small numerical error; the report is planning evidence, not an inferential guarantee".to_owned(),
            "the within-pair item noise decomposition is a sensitivity model; estimate its components from pilot evidence instead of assuming item count eliminates dependence".to_owned(),
        ],
        claim_boundary: "This is a design-stage power and precision sensitivity planner for independent treatment/placebo model-pair effects. It does not train models, verify model-pair independence, authenticate intervention receipts, establish the causal estimand, prove external validity, or create confirmatory evidence. Results are conditional on user-supplied effect and variance scenarios, pair attrition and the paired-normal model. Increasing task items changes the assumed within-pair measurement component only; it never increases independent model-pair degrees of freedom.".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario() -> PowerScenarioV1 {
        PowerScenarioV1 {
            scenario_id: "central".to_owned(),
            true_effect_permille: 100.0,
            between_pair_sd_permille: 180.0,
            within_pair_item_sd_permille: 300.0,
        }
    }

    fn request() -> CausalPowerPlanRequestV1 {
        CausalPowerPlanRequestV1 {
            schema: CAUSAL_POWER_REQUEST_SCHEMA.to_owned(),
            study_id: "study-001".to_owned(),
            benchmark_id: "contamination-panel".to_owned(),
            benchmark_version: "v1".to_owned(),
            primary_estimand_id: "treatment-minus-placebo".to_owned(),
            independent_unit: "independent_model_pair".to_owned(),
            alpha_permille: 50,
            test_tail: TestTailV1::TwoSided,
            multiplicity_policy: MultiplicityPolicyV1::None,
            contrast_count: 1,
            target_power_permille: 800,
            target_ci_half_width_permille: 100.0,
            max_gross_model_pair_replicates: 40,
            attrition_rates_permille: vec![0, 100, 200],
            items_per_pair_scenarios: vec![10, 100],
            effect_scenarios: vec![scenario()],
        }
    }

    #[test]
    fn hand_computable_json_fixture_round_trips_and_plans() {
        let request: CausalPowerPlanRequestV1 = serde_json::from_str(include_str!(
            "../fixtures/causal-power-plan-request.hand.v1.json"
        ))
        .unwrap();
        let report = plan_causal_power(&request).unwrap();
        assert_eq!(report.study_id, "causal-power-fixture-001");
        assert_eq!(report.independent_unit, "independent_model_pair");
        assert_eq!(report.results.len(), 3 * 2 * 2);
        assert!(!report.questions_counted_as_independent_training_replicates);
        assert!(!report.confirmatory_eligible);
    }

    #[test]
    fn student_t_cdf_is_symmetric_and_centered() {
        for df in [1, 2, 5, 10, 30, 100] {
            assert!((student_t_cdf(0.0, df) - 0.5).abs() < 1.0e-12);
            for value in [0.1, 1.0, 2.5, 5.0] {
                let sum = student_t_cdf(value, df) + student_t_cdf(-value, df);
                assert!((sum - 1.0).abs() < 1.0e-10);
            }
        }
    }

    #[test]
    fn student_t_quantiles_invert_the_cdf() {
        for (df, alpha) in [(1, 0.05), (5, 0.05), (10, 0.01), (30, 0.05), (100, 0.05)] {
            let critical = student_t_critical_value(alpha, TestTailV1::TwoSided, df).unwrap();
            let tail = 2.0 * (1.0 - student_t_cdf(critical, df));
            assert!((tail - alpha).abs() < 1.0e-9);
        }
    }

    #[test]
    fn noncentral_t_power_increases_with_effect_and_sample_size() {
        let small = noncentral_t_power(0.5, 19, 0.05, TestTailV1::TwoSided).unwrap();
        let large_effect = noncentral_t_power(1.0, 19, 0.05, TestTailV1::TwoSided).unwrap();
        let large_n = noncentral_t_power(0.5, 39, 0.05, TestTailV1::TwoSided).unwrap();
        assert!(large_effect > small);
        assert!(large_n > small);
    }

    #[test]
    fn paired_normal_power_is_near_the_normal_approximation_at_large_df() {
        let df = 300;
        let power = noncentral_t_power(2.0, df, 0.05, TestTailV1::TwoSided).unwrap();
        let normal_approx = normal_cdf(2.0 - 1.959_963_984_54)
            + normal_cdf(-2.0 - 1.959_963_984_54);
        assert!((power - normal_approx).abs() < 0.015);
    }

    #[test]
    fn more_items_reduce_measurement_component_not_degrees_of_freedom() {
        let mut request = request();
        request.max_gross_model_pair_replicates = 20;
        request.attrition_rates_permille = vec![0];
        request.items_per_pair_scenarios = vec![10, 100];
        let report = plan_causal_power(&request).unwrap();
        let low_items = &report.results[0].curve[18];
        let high_items = &report.results[1].curve[18];
        assert_eq!(low_items.degrees_of_freedom, high_items.degrees_of_freedom);
        assert!(
            high_items.effective_pair_effect_sd_permille
                < low_items.effective_pair_effect_sd_permille
        );
    }

    #[test]
    fn attrition_reduces_completed_pairs_and_degrees_of_freedom() {
        let mut request = request();
        request.max_gross_model_pair_replicates = 20;
        request.attrition_rates_permille = vec![0, 200];
        request.items_per_pair_scenarios = vec![10];
        let report = plan_causal_power(&request).unwrap();
        let no_attrition = &report.results[0].curve[18];
        let attrition = &report.results[1].curve[18];
        assert!(attrition.expected_completed_model_pairs < no_attrition.expected_completed_model_pairs);
        assert!(attrition.degrees_of_freedom < no_attrition.degrees_of_freedom);
    }

    #[test]
    fn bonferroni_multiplicity_reduces_per_contrast_alpha() {
        let mut request = request();
        request.multiplicity_policy = MultiplicityPolicyV1::Bonferroni;
        request.contrast_count = 6;
        let report = plan_causal_power(&request).unwrap();
        assert_eq!(report.results[0].adjusted_alpha_permille, 50.0 / 6.0);
    }

    #[test]
    fn item_counts_never_change_independent_unit_or_replication_rule() {
        let mut request = request();
        request.items_per_pair_scenarios = vec![1, 10, 100, 10_000];
        let report = plan_causal_power(&request).unwrap();
        assert!(!report.questions_counted_as_independent_training_replicates);
        assert!(report.independent_degrees_of_freedom_rule.contains("model pairs - 1"));
    }

    #[test]
    fn planning_rejects_item_level_independent_units() {
        let mut request = request();
        request.independent_unit = "individual_task_item".to_owned();
        assert!(plan_causal_power(&request).is_err());
    }

    #[test]
    fn no_multiplicity_policy_requires_one_primary_contrast() {
        let mut request = request();
        request.contrast_count = 6;
        assert!(plan_causal_power(&request).is_err());
    }

    #[test]
    fn excessive_power_grid_is_rejected_before_integration() {
        let mut request = request();
        request.max_gross_model_pair_replicates = MAX_INDEPENDENT_MODEL_PAIRS;
        request.attrition_rates_permille = vec![0, 100, 200, 300, 400, 500, 600];
        request.items_per_pair_scenarios = (1..=20).collect();
        request.effect_scenarios = (0..32)
            .map(|index| PowerScenarioV1 {
                scenario_id: format!("scenario-{index}"),
                true_effect_permille: 100.0,
                between_pair_sd_permille: 180.0,
                within_pair_item_sd_permille: 300.0,
            })
            .collect();
        assert!(plan_causal_power(&request).is_err());
    }

    #[test]
    fn duplicate_attrition_or_item_scenarios_are_rejected() {
        let mut request = request();
        request.attrition_rates_permille = vec![0, 0];
        assert!(plan_causal_power(&request).is_err());
        let mut request = request();
        request.items_per_pair_scenarios = vec![10, 10];
        assert!(plan_causal_power(&request).is_err());
    }

    #[test]
    fn one_sided_test_direction_must_match_the_effect() {
        let mut request = request();
        request.test_tail = TestTailV1::Greater;
        request.effect_scenarios[0].true_effect_permille = -100.0;
        assert!(plan_causal_power(&request).is_err());
    }

    #[test]
    fn zero_variance_scenario_is_rejected() {
        let mut request = request();
        request.effect_scenarios[0].between_pair_sd_permille = 0.0;
        request.effect_scenarios[0].within_pair_item_sd_permille = 0.0;
        assert!(plan_causal_power(&request).is_err());
    }

    #[test]
    fn power_plan_digest_binds_assumptions() {
        let first = plan_causal_power(&request()).unwrap();
        let mut changed = request();
        changed.effect_scenarios[0].true_effect_permille = 150.0;
        let second = plan_causal_power(&changed).unwrap();
        assert_ne!(first.request_sha256, second.request_sha256);
    }
}
