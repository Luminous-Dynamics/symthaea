use morphology_continuity_oracle::{canonical_templates, evaluate_episode, template, TargetEvidence};

fn main() {
    let templates = canonical_templates();
    let observed = template(&templates, "table")
        .expect("table fixture exists")
        .cells
        .clone();
    let report = evaluate_episode("chair", &observed, &templates, 4, TargetEvidence::Missing);
    let current = report.current_exact_matches.join(",");
    let histories = report.compatible_histories.join(",");
    let observed_iou = report.observed_vs_target.as_ref().map_or(0, |metrics| metrics.iou_ppm);

    // All string values are fixed fixture labels. Integer metrics make the receipt deterministic.
    println!(
        "{{\"schema\":\"morphology-continuity-oracle/v0.1\",\"fixture\":\"chair-backrest-removed\",\"current_exact_matches\":[\"{}\"],\"compatible_histories\":[\"{}\",\"{}\"],\"ground_truth_target\":\"chair\",\"target_evidence\":\"missing\",\"disposition\":\"{:?}\",\"observed_target_iou_ppm\":{},\"restoration_oracle\":false}}",
        current,
        report.compatible_histories[0],
        report.compatible_histories[1],
        report.disposition,
        observed_iou
    );
}
