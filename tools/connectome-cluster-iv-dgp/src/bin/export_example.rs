use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use connectome_cluster_iv_dgp::{
    ClusterIvDgpConfig, ClusterIvScenarioPreset, generate_cluster_iv_dataset, scenario_preset,
};

fn validate_oracle_destination(
    public_csv: &Path,
    oracle_output: &Path,
) -> Result<(), &'static str> {
    let public_parent = public_csv.parent().unwrap_or_else(|| Path::new("."));
    let oracle_parent = oracle_output.parent().unwrap_or_else(|| Path::new("."));
    if public_parent == oracle_parent {
        return Err("oracle output must be in a different directory from public artifacts");
    }
    if public_csv == oracle_output {
        return Err("oracle output must not overwrite the observed CSV");
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let prefix = args
        .next()
        .unwrap_or_else(|| "connectome-cluster-iv-dgp-example".to_owned());
    let scenario = args.next().unwrap_or_else(|| "example".to_owned());
    let mut oracle_output: Option<PathBuf> = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--oracle-out" => {
                if oracle_output.is_some() {
                    return Err("--oracle-out may be specified only once".into());
                }
                let path = args
                    .next()
                    .ok_or("--oracle-out requires a separate evaluator-controlled path")?;
                if path.trim().is_empty() {
                    return Err("--oracle-out path must not be empty".into());
                }
                oracle_output = Some(PathBuf::from(path));
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }

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

    fs::write(&csv_path, dataset.to_csv())?;
    fs::write(&manifest_path, dataset.manifest_json())?;

    println!("wrote {} observations across {} clusters", dataset.rows.len(), dataset.config.cluster_count());
    println!("observed CSV: {csv_path}");
    println!("public manifest: {manifest_path}");

    if let Some(path) = oracle_output {
        validate_oracle_destination(Path::new(&csv_path), &path)
            .map_err(str::to_owned)?;
        if path == PathBuf::from(&manifest_path) {
            return Err("--oracle-out must not overwrite the public manifest".into());
        }
        fs::write(&path, dataset.oracle_json())?;
        eprintln!(
            "WARNING: evaluator truth written to {}. Keep this file outside the estimator/model environment until predictions are frozen.",
            path.display()
        );
    } else {
        println!("oracle truth was not written; use --oracle-out only in an evaluator-controlled environment after arranging separate custody.");
    }

    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_output_in_public_directory_is_rejected() {
        assert!(validate_oracle_destination(
            Path::new("/tmp/public/data.csv"),
            Path::new("/tmp/public/oracle.json"),
        ).is_err());
    }

    #[test]
    fn oracle_output_in_separate_evaluator_directory_is_allowed() {
        assert_eq!(
            validate_oracle_destination(
                Path::new("/tmp/public/data.csv"),
                Path::new("/secure/evaluator/oracle.json"),
            ),
            Ok(())
        );
    }

    #[test]
    fn oracle_cannot_overwrite_observed_csv() {
        assert!(validate_oracle_destination(
            Path::new("/tmp/public/data.csv"),
            Path::new("/tmp/public/data.csv"),
        ).is_err());
    }
}
