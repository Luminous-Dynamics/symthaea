use symfp_spectral_core_reference::{
    analyze_spectra, classify_rank_template, target_leakage, template_by_name,
    SpectrumBlockV1, SymFpProfileV1, CRYSTALLOGRAPHIC_TEMPLATES_V1,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profile = SymFpProfileV1::paper_degree_9();
    let c2 = template_by_name("C2").ok_or("missing C2 template")?;
    let spectra: Vec<SpectrumBlockV1> = (0..=9)
        .map(|degree| SpectrumBlockV1 {
            degree,
            singular_values: (0..=2 * degree)
                .map(|index| {
                    if index < c2.fixed_dimensions[degree] as usize {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            svd_sweeps: None,
            svd_residual: None,
        })
        .collect();

    let receipt = analyze_spectra(&spectra, profile)?;
    let classification = classify_rank_template(
        &receipt,
        &CRYSTALLOGRAPHIC_TEMPLATES_V1,
    )?;
    let cs_leakage = target_leakage(
        &receipt,
        template_by_name("Cs").ok_or("missing Cs template")?,
    )?;

    println!("rank sequence: {:?}", receipt.rank_sequence);
    println!("degree-zero-scaled threshold: {:.3e}", receipt.tau);
    println!(
        "nearest rank-template matches at L=9: {:?} (L1 distance {})",
        classification.matched_schoenflies,
        classification.minimum_l1_distance
    );
    println!(
        "pLS against Cs: {:.6} (leakage {:.6})",
        cs_leakage.p_ls,
        cs_leakage.pooled_leakage_score
    );
    println!(
        "authority: {}",
        cs_leakage.authority
    );
    Ok(())
}
