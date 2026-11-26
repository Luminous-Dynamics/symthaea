use dna_channel_benchmark::{
    edit_counts, exact_payload_match, progressive_erasure_sets, rank_soft_erasures,
    sequence_constraints, summarize_read_batch,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let reference = b"ACGTACGTACGT";
    let reads: [&[u8]; 4] = [
        b"ACGTACGTACGT",
        b"ACGTTCGTACGT",
        b"ACGTTACGTACGT",
        b"ACGTACGACGT",
    ];
    let edit_summaries = reads
        .iter()
        .map(|read| edit_counts(reference, read))
        .collect::<Result<Vec<_>, _>>()?;
    let ranked = rank_soft_erasures(&[0.01, 0.12, 0.83, 0.41, 0.03], 0.10)?;
    let erasure_trials = progressive_erasure_sets(&ranked, 4);
    let read_summary = summarize_read_batch(&[1, 3, 2, 1], false);
    let constraints_ok = sequence_constraints(b"ACGTACGT", 0.4, 0.6, 2)?;
    let exact_match = exact_payload_match(b"immutable-payload", b"immutable-payload");

    println!("{{");
    println!("  \"evidence_class\": \"synthetic_deterministic_sequence_channel\",");
    println!("  \"read_count\": {},", edit_summaries.len());
    println!("  \"edit_counts\": [");
    for (index, edit) in edit_summaries.iter().enumerate() {
        let comma = if index + 1 == edit_summaries.len() { "" } else { "," };
        println!(
            "    {{\"distance\":{},\"substitutions\":{},\"insertions\":{},\"deletions\":{},\"matches\":{}}}{}",
            edit.distance, edit.substitutions, edit.insertions, edit.deletions, edit.matches, comma
        );
    }
    println!("  ],");
    println!("  \"ranked_soft_erasure_positions\": {:?},", ranked.iter().map(|item| item.position).collect::<Vec<_>>());
    println!("  \"progressive_erasure_trials\": {:?},", erasure_trials);
    println!("  \"read_batch\": {{\"distinct_records\":{},\"read_observations\":{},\"additional_read_copies\":{}}},",
        read_summary.distinct_read_records, read_summary.read_observations, read_summary.additional_read_copies);
    println!("  \"sequence_constraints_pass\": {},", constraints_ok);
    println!("  \"exact_payload_match\": {}", exact_match);
    println!("}}");
    Ok(())
}
