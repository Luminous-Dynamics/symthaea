use std::{env, fs, io};

use mel_temporal_grounding_qualifier::music_performance_identity_inference::PairedMusicalIdentityComparisonCaseV1;
use mel_temporal_grounding_qualifier::music_performance_identity_sampling_design::{
    PerformanceIdentitySamplingDesignV1, audit_paired_comparison_with_sampling_design,
    audit_performance_identity_sampling_design,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if !(1..=2).contains(&args.len()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: mel-music-performance-identity-sampling-design <design.json> [comparison.json]",
        )
        .into());
    }

    let design_text = fs::read_to_string(&args[0])?;
    let design: PerformanceIdentitySamplingDesignV1 = serde_json::from_str(&design_text)?;
    let output = if let Some(comparison_path) = args.get(1) {
        let comparison_text = fs::read_to_string(comparison_path)?;
        let comparison: PairedMusicalIdentityComparisonCaseV1 =
            serde_json::from_str(&comparison_text)?;
        serde_json::to_value(
            audit_paired_comparison_with_sampling_design(&comparison, &design)
                .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))?,
        )?
    } else {
        serde_json::to_value(
            audit_performance_identity_sampling_design(&design)
                .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))?,
        )?
    };

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
