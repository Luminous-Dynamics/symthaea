use std::env;
use std::error::Error;
use std::fs;

use connectome_cluster_iv_dgp::{ClusterIvDgpConfig, generate_cluster_iv_dataset};

fn main() -> Result<(), Box<dyn Error>> {
    let prefix = env::args()
        .nth(1)
        .unwrap_or_else(|| "connectome-cluster-iv-dgp-example".to_owned());
    let dataset = generate_cluster_iv_dataset(ClusterIvDgpConfig::example())?;
    let csv_path = format!("{prefix}.csv");
    let manifest_path = format!("{prefix}.manifest.json");

    fs::write(&csv_path, dataset.to_csv())?;
    fs::write(&manifest_path, dataset.manifest_json())?;

    println!("wrote {} observations across {} clusters", dataset.rows.len(), dataset.config.cluster_count());
    println!("CSV: {csv_path}");
    println!("manifest: {manifest_path}");
    println!("SHA-256 hash both files before running blinded inference.");
    Ok(())
}
