use physical_storage_layout_search::{
    cuboid_fault_masks, greedy_pair_swap_search, morton_order, row_major_order,
    score_mapping, seeded_random_order, serpentine_order, Assignment, Grid3,
    LayoutProblem, Mapping,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let problem = LayoutProblem::new(
        Grid3::new(4, 4, 3)?,
        vec![0, 15, 16, 31, 32, 47],
        7,
        6,
        2,
    )?;
    let masks = cuboid_fault_masks(
        problem.grid,
        &[(2, 2, 1), (3, 3, 1), (2, 2, 2), (3, 3, 2), (4, 4, 1)],
    )?;

    let row = row_major_order(&problem);
    let snake = serpentine_order(&problem);
    let morton = morton_order(&problem);
    let random = seeded_random_order(&problem, 2026);
    let candidates = vec![
        Mapping::from_order(&problem, "row-major-contiguous", &row, Assignment::Contiguous)?,
        Mapping::from_order(&problem, "row-major-interleaved", &row, Assignment::Interleaved)?,
        Mapping::from_order(&problem, "serpentine-contiguous", &snake, Assignment::Contiguous)?,
        Mapping::from_order(&problem, "serpentine-interleaved", &snake, Assignment::Interleaved)?,
        Mapping::from_order(&problem, "morton-contiguous", &morton, Assignment::Contiguous)?,
        Mapping::from_order(&problem, "morton-interleaved", &morton, Assignment::Interleaved)?,
        Mapping::from_order(&problem, "random-2026-contiguous", &random, Assignment::Contiguous)?,
        Mapping::from_order(&problem, "random-2026-interleaved", &random, Assignment::Interleaved)?,
    ];

    println!("{{");
    println!("  \"evidence_class\": \"synthetic_deterministic_geometry\",");
    println!("  \"grid_xyz\": [4, 4, 3],");
    println!("  \"stripe_count\": 7,");
    println!("  \"shards_per_stripe\": 6,");
    println!("  \"known_erasure_budget_per_stripe\": 2,");
    println!("  \"fault_set_count\": {},", masks.len());
    println!("  \"candidates\": [");
    for (index, mapping) in candidates.iter().enumerate() {
        let score = score_mapping(&problem, mapping, &masks)?;
        let comma = if index + 1 == candidates.len() { "" } else { "," };
        println!(
            "    {{\"name\":\"{}\",\"layout_fingerprint_fnv1a64\":\"{}\",\"unrecoverable_fault_sets\":{},\"worst_erased_shards_in_one_stripe\":{},\"total_excess_erasures\":{},\"candidate_evaluations\":1}}{}",
            mapping.name,
            mapping.fingerprint_fnv1a64(),
            score.unrecoverable_fault_sets,
            score.worst_erased_shards_in_one_stripe,
            score.total_excess_erasures,
            comma
        );
    }

    let search = greedy_pair_swap_search(&problem, &masks, &candidates[0], 500)?;
    println!("  ],");
    println!("  \"greedy_pair_swap\": {{");
    println!("    \"candidate_evaluations\": {},", search.candidate_evaluations);
    println!("    \"termination\": \"{}\",", search.termination);
    println!("    \"unrecoverable_fault_sets\": {},", search.score.unrecoverable_fault_sets);
    println!("    \"worst_erased_shards_in_one_stripe\": {},", search.score.worst_erased_shards_in_one_stripe);
    println!("    \"layout_fingerprint_fnv1a64\": \"{}\"", search.mapping.fingerprint_fnv1a64());
    println!("  }}");
    println!("}}");
    Ok(())
}
