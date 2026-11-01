use std::env;
use std::error::Error;
use std::fs;

use connectome_cluster_iv_dgp::{
    ClusterIvDgpConfig, ClusterIvScenarioPreset, generate_cluster_iv_dataset, scenario_preset,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let prefix = args
        .next()
        .unwrap_or_else(|| "connectome-cluster-iv-dgp-example".to_owned());
    let scenario = args.next().unwrap_or_else(|| "example".to_owned());
    let config = match scenario.as_str() {
        "example" => ClusterIvDgpConfig::example(),
        "strong-exact-exclusion" => scenario_preset(ClusterIvScenarioPreset::StrongExactExclusion, 20_261_101),
        "weak-exact-exclusion" => scenario_preset(ClusterIvScenarioPreset::WeakExactExclusion, 20_261_101),
        "zero-relevance" => scenario_preset(ClusterIvScenarioPreset::ZeroRelevance, 20_261_101),
        "direct-effect-inside-support" => scenario_preset(ClusterIvScenarioPreset::DirectEffectInsideSupport, 20_261_101),
        "direct-effect-outside-support" => scenario_preset(ClusterIvScenarioPreset::DirectEffectOutsideSupport, 20_261_101),
        "direct-effect-at-boundary" => scenario_preset(ClusterIvScenarioPreset::DirectEffectAtBoundary, 20_261_101),
        "cluster-specific-direct-effects" => scenario_preset(ClusterIvScenarioPreset::ClusterSpecificDirectEffects, 20_261_101),
        "many-correlated-instruments" => scenario_preset(ClusterIvScenarioPreset::ManyCorrelatedInstruments, 20_261_101),
        "few-large-clusters" => scenario_preset(ClusterIvScenarioPreset::FewLargeClusters, 20_261_101),
        "many-small-clusters" => scenario_preset(ClusterIvScenarioPreset::ManySmallClusters, 20_261_101),
        "strength-concentrated-in-few-clusters" => scenario_preset(ClusterIvScenarioPreset::StrengthConcentratedInFewClusters, 20_261_101),
        "cluster-level-instruments" => scenario_preset(ClusterIvScenarioPreset::ClusterLevelInstruments, 20_261_101),
        "unequal-heteroskedastic-clusters" => scenario_preset(ClusterIvScenarioPreset::UnequalHeteroskedasticClusters, 20_261_101),
        other => return Err(format!("unknown scenario {other:?}; see README for supported presets").into()),
    };
    let dataset = generate_cluster_iv_dataset(config)?;
    let csv_path = format!("{prefix}.csv");
    let manifest_path = format!("{prefix}.manifest.json");
    let oracle_path = format!("{prefix}.oracle.json");

    fs::write(&csv_path, dataset.to_csv())?;
    fs::write(&manifest_path, dataset.manifest_json())?;
    fs::write(&oracle_path, dataset.oracle_json())?;

    println!("wrote {} observations across {} clusters", dataset.rows.len(), dataset.config.cluster_count());
    println!("observed CSV: {csv_path}");
    println!("public manifest: {manifest_path}");
    println!("RESTRICTED oracle truth: {oracle_path}");
    println!("Hash all three files separately. Keep the oracle path outside the estimator environment until predictions are frozen.");
    Ok(())
}
