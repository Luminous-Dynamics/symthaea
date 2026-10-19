use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;

const INPUT_SCHEMA: &str = "cropfoundry-trial-power-input-v0";
const RECEIPT_SCHEMA: &str = "cropfoundry-trial-power-receipt-v0";

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PowerInput {
    schema_version: String,
    study_id: String,
    protocol_uri: String,
    variance_evidence_uri: String,
    endpoint: EndpointInput,
    design: DesignInput,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EndpointInput {
    name: String,
    unit: String,
    minimum_detectable_effect_abs: f64,
    residual_sd: f64,
    familywise_alpha: f64,
    target_power: f64,
    co_primary_endpoint_count: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DesignInput {
    design_type: String,
    allocation_unit: String,
    plots_per_cluster: usize,
    intracluster_correlation: f64,
    attrition_fraction: f64,
    planned_site_season_count: usize,
}

#[derive(Debug, Serialize)]
struct TrialPowerPlan {
    receipt_schema: String,
    study_id: String,
    protocol_uri: String,
    variance_evidence_uri: String,
    input_sha256: String,
    method: String,
    endpoint_name: String,
    endpoint_unit: String,
    minimum_detectable_effect_abs: f64,
    residual_sd: f64,
    familywise_alpha: f64,
    co_primary_endpoint_count: usize,
    bonferroni_alpha_per_endpoint: f64,
    target_power: f64,
    design_type: String,
    allocation_unit: String,
    intracluster_correlation: f64,
    z_critical_two_sided: f64,
    z_power: f64,
    independent_plots_required_per_arm_before_cluster_and_attrition_adjustment: usize,
    design_effect: f64,
    attrition_fraction: f64,
    required_recruited_plots_per_arm_before_cluster_rounding: usize,
    plots_per_cluster: usize,
    clusters_required_per_arm: usize,
    planned_plots_per_arm_after_cluster_rounding: usize,
    total_planned_plots_two_arms: usize,
    planned_site_season_count: usize,
    approximate_effective_retained_plots_per_arm: f64,
    warnings: Vec<String>,
    field_efficacy_claim: bool,
    training_authorized: bool,
    notes: Vec<String>,
}

fn inverse_normal_cdf(p: f64) -> Result<f64> {
    if !(0.0 < p && p < 1.0) {
        return Err(anyhow!("normal quantile probability must be strictly between zero and one"));
    }

    // Peter J. Acklam's rational approximation to the inverse standard-normal CDF.
    // Accuracy is ample for a screening planner; the receipt explicitly disclaims a final analysis.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const LOW: f64 = 0.02425;
    const HIGH: f64 = 1.0 - LOW;

    if p < LOW {
        let q = (-2.0 * p.ln()).sqrt();
        return Ok(
            (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0),
        );
    }
    if p > HIGH {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return Ok(
            -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0),
        );
    }

    let q = p - 0.5;
    let r = q * q;
    Ok(
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q)
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0),
    )
}

fn validate(input: &PowerInput) -> Result<()> {
    if input.schema_version != INPUT_SCHEMA {
        return Err(anyhow!("schema_version must equal {INPUT_SCHEMA}"));
    }
    for (name, value) in [
        ("study_id", input.study_id.as_str()),
        ("protocol_uri", input.protocol_uri.as_str()),
        ("variance_evidence_uri", input.variance_evidence_uri.as_str()),
        ("endpoint.name", input.endpoint.name.as_str()),
        ("endpoint.unit", input.endpoint.unit.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(anyhow!("{name} must be non-empty"));
        }
    }
    if input.design.design_type != "parallel_two_arm_clustered" {
        return Err(anyhow!(
            "v0 supports only design_type=parallel_two_arm_clustered; use a design-specific method otherwise"
        ));
    }
    if !matches!(input.design.allocation_unit.as_str(), "plot" | "farm" | "greenhouse_cluster") {
        return Err(anyhow!(
            "allocation_unit must be plot, farm, or greenhouse_cluster"
        ));
    }
    if input.design.plots_per_cluster == 0 {
        return Err(anyhow!("plots_per_cluster must be at least one"));
    }
    if input.design.allocation_unit == "plot" && input.design.plots_per_cluster != 1 {
        return Err(anyhow!(
            "plot-level randomization requires plots_per_cluster=1 in v0"
        ));
    }
    if input.design.allocation_unit != "plot" && input.design.plots_per_cluster < 2 {
        return Err(anyhow!(
            "cluster allocation requires plots_per_cluster >= 2 in v0"
        ));
    }
    let endpoint = &input.endpoint;
    if !endpoint.minimum_detectable_effect_abs.is_finite()
        || endpoint.minimum_detectable_effect_abs <= 0.0
    {
        return Err(anyhow!("minimum_detectable_effect_abs must be finite and positive"));
    }
    if !endpoint.residual_sd.is_finite() || endpoint.residual_sd <= 0.0 {
        return Err(anyhow!("residual_sd must be finite and positive"));
    }
    if !endpoint.familywise_alpha.is_finite()
        || endpoint.familywise_alpha <= 0.0
        || endpoint.familywise_alpha >= 0.2
    {
        return Err(anyhow!("familywise_alpha must be in (0, 0.2)"));
    }
    if !endpoint.target_power.is_finite()
        || endpoint.target_power <= 0.5
        || endpoint.target_power >= 0.999
    {
        return Err(anyhow!("target_power must be in (0.5, 0.999)"));
    }
    if endpoint.co_primary_endpoint_count == 0 {
        return Err(anyhow!("co_primary_endpoint_count must be at least one"));
    }
    let alpha_per_endpoint =
        endpoint.familywise_alpha / endpoint.co_primary_endpoint_count as f64;
    if alpha_per_endpoint <= 0.0 || alpha_per_endpoint >= 0.2 {
        return Err(anyhow!("Bonferroni-adjusted alpha is outside the supported numerical range"));
    }
    if !input.design.intracluster_correlation.is_finite()
        || !(0.0..=1.0).contains(&input.design.intracluster_correlation)
    {
        return Err(anyhow!("intracluster_correlation must be in [0, 1]"));
    }
    if !input.design.attrition_fraction.is_finite()
        || !(0.0..1.0).contains(&input.design.attrition_fraction)
    {
        return Err(anyhow!("attrition_fraction must be in [0, 1)"));
    }
    if input.design.planned_site_season_count == 0 {
        return Err(anyhow!("planned_site_season_count must be at least one"));
    }
    Ok(())
}

fn calculate(input: &PowerInput, input_sha256: String) -> Result<TrialPowerPlan> {
    validate(input)?;
    let endpoint = &input.endpoint;
    let design = &input.design;
    let alpha_per_endpoint =
        endpoint.familywise_alpha / endpoint.co_primary_endpoint_count as f64;
    let z_critical = inverse_normal_cdf(1.0 - alpha_per_endpoint / 2.0)?;
    let z_power = inverse_normal_cdf(endpoint.target_power)?;
    let effect = endpoint.minimum_detectable_effect_abs;
    let sigma = endpoint.residual_sd;
    let raw_n = (2.0 * (z_critical + z_power).powi(2) * sigma.powi(2) / effect.powi(2)).ceil();
    if !raw_n.is_finite() || raw_n > (usize::MAX / 4) as f64 {
        return Err(anyhow!("computed sample size exceeds supported integer range"));
    }
    let independent_n = (raw_n as usize).max(2);
    let design_effect = 1.0
        + (design.plots_per_cluster.saturating_sub(1) as f64)
            * design.intracluster_correlation;
    let attrition_adjusted =
        ((independent_n as f64 * design_effect) / (1.0 - design.attrition_fraction)).ceil();
    if !attrition_adjusted.is_finite() || attrition_adjusted > (usize::MAX / 4) as f64 {
        return Err(anyhow!("adjusted sample size exceeds supported integer range"));
    }
    let required_plots = (attrition_adjusted as usize).max(2);
    let clusters_per_arm = required_plots.div_ceil(design.plots_per_cluster);
    let planned_plots_per_arm = clusters_per_arm
        .checked_mul(design.plots_per_cluster)
        .ok_or_else(|| anyhow!("planned per-arm plot count exceeds supported integer range"))?;
    let total_planned_plots_two_arms = planned_plots_per_arm
        .checked_mul(2)
        .ok_or_else(|| anyhow!("two-arm total plot count exceeds supported integer range"))?;
    let effective_retained = planned_plots_per_arm as f64
        * (1.0 - design.attrition_fraction)
        / design_effect;

    let mut warnings = vec![
        "Normal-approximation preflight only; not a final design-specific power calculation."
            .to_owned(),
        "Variance evidence and protocol URIs are recorded, not independently verified by this command."
            .to_owned(),
        "This calculation does not model blocking gains, spatial autocorrelation, repeated measures, unequal cluster sizes, non-normal outcomes, interactions, or non-inferiority margins."
            .to_owned(),
        "Confirm assumptions and analysis with an agricultural statistician before preregistration."
            .to_owned(),
    ];
    if clusters_per_arm < 10 && design.allocation_unit != "plot" {
        warnings.push(
            "Fewer than 10 randomized clusters per arm: normal design-effect approximation may be optimistic; use cluster-aware simulation or small-sample correction."
                .to_owned(),
        );
    }
    if design.planned_site_season_count < 2 {
        warnings.push(
            "Only one site-season is planned; do not claim cross-site or cross-season transfer."
                .to_owned(),
        );
    }
    if design.attrition_fraction > 0.2 {
        warnings.push(
            "Attrition allowance exceeds 20%; justify feasibility and report missingness by arm."
                .to_owned(),
        );
    }
    if endpoint.co_primary_endpoint_count > 1 {
        warnings.push(
            "Bonferroni adjustment is applied to co-primary endpoints; predefine the endpoint family and report each outcome."
                .to_owned(),
        );
    }

    Ok(TrialPowerPlan {
        receipt_schema: RECEIPT_SCHEMA.to_owned(),
        study_id: input.study_id.clone(),
        protocol_uri: input.protocol_uri.clone(),
        variance_evidence_uri: input.variance_evidence_uri.clone(),
        input_sha256,
        method: "two_arm_normal_approximation_with_bonferroni_and_cluster_design_effect".to_owned(),
        endpoint_name: endpoint.name.clone(),
        endpoint_unit: endpoint.unit.clone(),
        minimum_detectable_effect_abs: effect,
        residual_sd: sigma,
        familywise_alpha: endpoint.familywise_alpha,
        co_primary_endpoint_count: endpoint.co_primary_endpoint_count,
        bonferroni_alpha_per_endpoint: alpha_per_endpoint,
        target_power: endpoint.target_power,
        design_type: design.design_type.clone(),
        allocation_unit: design.allocation_unit.clone(),
        intracluster_correlation: design.intracluster_correlation,
        z_critical_two_sided: z_critical,
        z_power,
        independent_plots_required_per_arm_before_cluster_and_attrition_adjustment: independent_n,
        design_effect,
        attrition_fraction: design.attrition_fraction,
        required_recruited_plots_per_arm_before_cluster_rounding: required_plots,
        plots_per_cluster: design.plots_per_cluster,
        clusters_required_per_arm: clusters_per_arm,
        planned_plots_per_arm_after_cluster_rounding: planned_plots_per_arm,
        total_planned_plots_two_arms,
        planned_site_season_count: design.planned_site_season_count,
        approximate_effective_retained_plots_per_arm: (effective_retained * 1000.0).round() / 1000.0,
        warnings,
        field_efficacy_claim: false,
        training_authorized: false,
        notes: vec![
            "The smallest practically important effect and residual standard deviation must use the same declared unit.".to_owned(),
            "Use pilot or historical variance that is independent of the confirmatory trial outcomes.".to_owned(),
            "A study may be underpowered for site- or crop-specific heterogeneity even when it is powered for an overall mean.".to_owned(),
            "Post-hoc observed power is not a substitute for effect estimates and confidence intervals.".to_owned(),
        ],
    })
}

/// Create a deterministic, fail-closed field-trial power preflight receipt; no crop outcomes are scored.
pub fn run(input_path: PathBuf, out: Option<PathBuf>) -> Result<()> {
    let bytes = fs::read(&input_path)
        .with_context(|| format!("cannot read trial-power input {}", input_path.display()))?;
    let input: PowerInput = serde_json::from_slice(&bytes)
        .with_context(|| format!("cannot parse trial-power JSON {}", input_path.display()))?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let plan = calculate(&input, digest)?;
    let rendered = serde_json::to_string_pretty(&plan)?;
    println!("{rendered}");
    if let Some(path) = out {
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("cannot create output directory {}", parent.display()))?;
        }
        fs::write(&path, format!("{rendered}\n"))
            .with_context(|| format!("cannot write trial-power receipt {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PowerInput {
        PowerInput {
            schema_version: INPUT_SCHEMA.to_owned(),
            study_id: "synthetic-trial".to_owned(),
            protocol_uri: "urn:cropfoundry:synthetic-protocol".to_owned(),
            variance_evidence_uri: "urn:cropfoundry:synthetic-variance-source".to_owned(),
            endpoint: EndpointInput {
                name: "marketable_yield".to_owned(),
                unit: "kg_per_plot".to_owned(),
                minimum_detectable_effect_abs: 0.5,
                residual_sd: 1.0,
                familywise_alpha: 0.05,
                target_power: 0.8,
                co_primary_endpoint_count: 1,
            },
            design: DesignInput {
                design_type: "parallel_two_arm_clustered".to_owned(),
                allocation_unit: "plot".to_owned(),
                plots_per_cluster: 1,
                intracluster_correlation: 0.0,
                attrition_fraction: 0.0,
                planned_site_season_count: 2,
            },
        }
    }

    #[test]
    fn unsupported_input_fields_fail_closed() {
        let mut value = serde_json::to_value(fixture()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("unrecognized_field".to_owned(), serde_json::json!(true));
        assert!(serde_json::from_value::<PowerInput>(value).is_err());
    }

    #[test]
    fn inverse_normal_cdf_rejects_boundary_probabilities() {
        assert!(inverse_normal_cdf(0.0).is_err());
        assert!(inverse_normal_cdf(1.0).is_err());
    }

    #[test]
    fn inverse_normal_cdf_matches_reference_quantiles() {
        assert!((inverse_normal_cdf(0.975).unwrap() - 1.959_963_986).abs() < 1e-6);
        assert!((inverse_normal_cdf(0.9).unwrap() - 1.281_551_566).abs() < 1e-6);
        assert!((inverse_normal_cdf(0.5).unwrap()).abs() < 1e-8);
    }

    #[test]
    fn basic_two_arm_plan_reproduces_standard_normal_approximation() {
        let plan = calculate(&fixture(), "a".repeat(64)).unwrap();
        assert_eq!(plan.independent_plots_required_per_arm_before_cluster_and_attrition_adjustment, 63);
        assert_eq!(plan.total_planned_plots_two_arms, 126);
        assert_eq!(plan.design_type, "parallel_two_arm_clustered");
        assert_eq!(plan.allocation_unit, "plot");
        assert_eq!(plan.intracluster_correlation, 0.0);
        assert!((plan.design_effect - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cluster_size_icc_and_attrition_increase_planned_enrollment() {
        let mut input = fixture();
        input.design.allocation_unit = "farm".to_owned();
        input.design.plots_per_cluster = 5;
        input.design.intracluster_correlation = 0.1;
        input.design.attrition_fraction = 0.1;
        let plan = calculate(&input, "b".repeat(64)).unwrap();
        assert_eq!(plan.design_effect, 1.4);
        assert_eq!(plan.clusters_required_per_arm, 20);
        assert_eq!(plan.planned_plots_per_arm_after_cluster_rounding, 100);
        assert_eq!(plan.total_planned_plots_two_arms, 200);
        assert!(plan.approximate_effective_retained_plots_per_arm >= 63.0);
    }

    #[test]
    fn multiple_primary_endpoints_adjust_alpha_and_increase_n() {
        let mut input = fixture();
        input.endpoint.co_primary_endpoint_count = 2;
        let one_endpoint = calculate(&fixture(), "c".repeat(64)).unwrap();
        let two_endpoints = calculate(&input, "d".repeat(64)).unwrap();
        assert_eq!(two_endpoints.bonferroni_alpha_per_endpoint, 0.025);
        assert!(
            two_endpoints
                .independent_plots_required_per_arm_before_cluster_and_attrition_adjustment
                > one_endpoint.independent_plots_required_per_arm_before_cluster_and_attrition_adjustment
        );
    }

    #[test]
    fn invalid_icc_and_invalid_alpha_fail_closed() {
        let mut input = fixture();
        input.design.intracluster_correlation = 1.2;
        assert!(validate(&input).is_err());
        let mut input = fixture();
        input.endpoint.familywise_alpha = 0.0;
        assert!(validate(&input).is_err());
    }

    #[test]
    fn cluster_randomization_requires_multiple_plots_per_cluster() {
        let mut input = fixture();
        input.design.allocation_unit = "farm".to_owned();
        input.design.plots_per_cluster = 1;
        assert!(validate(&input).is_err());
    }

    #[test]
    fn integer_overflow_in_cluster_rounding_fails_closed() {
        let mut input = fixture();
        input.design.allocation_unit = "farm".to_owned();
        input.design.plots_per_cluster = usize::MAX;
        input.design.intracluster_correlation = 0.0;
        assert!(calculate(&input, "f".repeat(64)).is_err());
    }

    #[test]
    fn plan_receipt_never_claims_field_efficacy_or_training_authority() {
        let plan = calculate(&fixture(), "e".repeat(64)).unwrap();
        assert_eq!(plan.receipt_schema, RECEIPT_SCHEMA);
        assert!(!plan.field_efficacy_claim);
        assert!(!plan.training_authorized);
        assert!(plan.warnings.iter().any(|warning| warning.contains("preflight only")));
    }
}
