use symmetry_harmonic_rank_oracle::{
    GroupHypothesisV1, MatchStatusV1, TemplateBudgetV1, match_signature, rank_template,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let budget = TemplateBudgetV1::default();
    let c3 = GroupHypothesisV1::Cyclic { order: 3 };

    let truncated = rank_template(c3, 2, budget)?;
    let ambiguous = match_signature(&truncated, 6, budget)?;
    println!("degree cutoff = {}", ambiguous.maximum_degree);
    println!("template = {truncated:?}");
    println!("status = {:?}", ambiguous.status);
    for hypothesis in &ambiguous.matching_hypotheses {
        println!("candidate = {}", hypothesis.canonical_label());
    }
    println!(
        "first family divergence degree = {:?}",
        ambiguous.first_family_divergence_degree
    );

    let extended = rank_template(c3, 3, budget)?;
    let resolved = match_signature(&extended, 6, budget)?;
    println!("\nextended degree cutoff = {}", resolved.maximum_degree);
    println!("template = {extended:?}");
    println!("status = {:?}", resolved.status);
    for hypothesis in &resolved.matching_hypotheses {
        println!("candidate = {}", hypothesis.canonical_label());
    }

    assert_eq!(ambiguous.status, MatchStatusV1::Ambiguous);
    assert_eq!(resolved.status, MatchStatusV1::Unique);
    assert_eq!(resolved.matching_hypotheses, vec![c3]);
    Ok(())
}
