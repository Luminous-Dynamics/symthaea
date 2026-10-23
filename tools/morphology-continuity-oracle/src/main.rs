use morphology_continuity_oracle::{
    REQUEST_SHA256, TargetEvidence, canonical_templates, evaluate_episode, template,
};
use std::env;

fn verified_sha_or_local(name: &str) -> String {
    match env::var(name) {
        Ok(value) if value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) => {
            value
        }
        _ => "unbound-local".to_owned(),
    }
}

fn main() {
    let templates = canonical_templates();
    let observed = template(&templates, "table")
        .expect("table fixture exists")
        .cells
        .clone();
    let report = evaluate_episode("chair", &observed, &templates, 4, TargetEvidence::Missing);
    let current = report.current_exact_matches.join(",");
    let observed_iou = report
        .observed_vs_target
        .as_ref()
        .map_or(0, |metrics| metrics.iou_ppm);
    let source_commit = verified_sha_or_local("SOURCE_COMMIT");
    let source_tree_sha = verified_sha_or_local("SOURCE_TREE_SHA");

    // All emitted labels are fixed fixture strings or validated hexadecimal identifiers.
    println!(
        concat!(
            "{{\"schema\":\"morphology-continuity-oracle/v0.1\",",
            "\"source_commit\":\"{}\",",
            "\"source_tree_sha\":\"{}\",",
            "\"request_sha256\":\"{}\",",
            "\"fixture\":\"chair-backrest-removed\",",
            "\"current_exact_matches\":[\"{}\"],",
            "\"compatible_histories\":[\"{}\",\"{}\"],",
            "\"ground_truth_target\":\"chair\",",
            "\"target_evidence\":\"missing\",",
            "\"disposition\":\"{:?}\",",
            "\"observed_target_iou_ppm\":{},",
            "\"restoration_oracle\":false}}"
        ),
        source_commit,
        source_tree_sha,
        REQUEST_SHA256,
        current,
        report.compatible_histories[0],
        report.compatible_histories[1],
        report.disposition,
        observed_iou
    );
}
