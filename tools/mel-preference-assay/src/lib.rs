use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: &str = "mel-blind-preference-assay-v1";
const ASSIGNMENT_ALGORITHM: &str = "sha256-presentation-v1";
const ASSIGNMENT_DOMAIN: &[u8] = b"MEL-BLIND-PREFERENCE-ASSAY-V1\0";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssayV1 {
    pub schema_version: String,
    pub assay_id: String,
    pub frozen_at: String,
    pub assignment_algorithm: String,
    pub assignment_seed_commitment_sha256: String,
    pub assignment_seed_revealed: String,
    pub minimum_per_order_per_stratum: usize,
    pub minimum_identical_controls_for_bias_flag: usize,
    pub presentations: Vec<PresentationV1>,
    pub claim_ceiling: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresentationV1 {
    pub presentation_id: String,
    pub case_id: String,
    pub stratum: String,
    pub judge_pseudonym_sha256: String,
    pub candidate_a_sha256: String,
    pub candidate_b_sha256: String,
    pub left_candidate: String,
    pub control_kind: String,
    pub choice: String,
    pub judgment_protocol_sha256: String,
}

#[derive(Debug, Default, Serialize)]
pub struct ChoiceCounts {
    pub left: usize,
    pub right: usize,
    pub tie: usize,
    pub neither: usize,
    pub skip: usize,
}

#[derive(Debug, Default, Serialize)]
pub struct OrderCounts {
    pub a_left: usize,
    pub b_left: usize,
}

#[derive(Debug, Serialize)]
pub struct StratumBalance {
    pub presentations: usize,
    pub a_left: usize,
    pub b_left: usize,
    pub balance_requirement_met: bool,
}

#[derive(Debug, Serialize)]
pub struct SideBiasReport {
    pub identical_control_presentations: usize,
    pub decisive_control_count: usize,
    pub left_choice_count: usize,
    pub right_choice_count: usize,
    pub tie_count: usize,
    pub neither_count: usize,
    pub skip_count: usize,
    pub left_choice_rate: Option<f64>,
    pub wilson_95: Option<[f64; 2]>,
    pub directional_bias_detected: bool,
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct CaseOrderEffect {
    pub case_id: String,
    pub a_left_decisive: usize,
    pub a_right_decisive: usize,
    pub a_win_rate_when_a_left: f64,
    pub a_win_rate_when_a_right: f64,
    pub signed_a_win_rate_gap: f64,
    pub absolute_a_win_rate_gap: f64,
}

#[derive(Debug, Serialize)]
pub struct OrderEffectReport {
    pub matched_case_count: usize,
    pub mean_signed_a_win_rate_gap: Option<f64>,
    pub mean_absolute_a_win_rate_gap: Option<f64>,
    pub cases: Vec<CaseOrderEffect>,
}

#[derive(Debug, Serialize)]
pub struct AuditReport {
    pub schema_version: String,
    pub assay_id: String,
    pub valid: bool,
    pub presentation_count: usize,
    pub unique_judge_count: usize,
    pub order_counts: OrderCounts,
    pub stratum_balance: BTreeMap<String, StratumBalance>,
    pub choice_counts: ChoiceCounts,
    pub candidate_a_win_rate: Option<f64>,
    pub side_bias: SideBiasReport,
    pub order_effect: OrderEffectReport,
    pub claim_ceiling: Vec<String>,
}

#[derive(Debug, Default)]
struct StratumCounter {
    presentations: usize,
    a_left: usize,
    b_left: usize,
}

#[derive(Debug, Default)]
struct CaseCounter {
    a_left_decisive: usize,
    a_left_wins: usize,
    a_right_decisive: usize,
    a_right_wins: usize,
}

pub fn parse_and_audit(input: &str) -> Result<AuditReport, String> {
    let assay: AssayV1 =
        serde_json::from_str(input).map_err(|error| format!("invalid assay JSON: {error}"))?;
    audit_assay(&assay)
}

pub fn audit_assay(assay: &AssayV1) -> Result<AuditReport, String> {
    let mut errors = Vec::new();

    if assay.schema_version != SCHEMA_VERSION {
        errors.push(format!(
            "unsupported schema_version: {}",
            assay.schema_version
        ));
    }
    if assay.assay_id.trim().is_empty() {
        errors.push("assay_id must not be empty".to_owned());
    }
    if assay.frozen_at.trim().is_empty() {
        errors.push("frozen_at must not be empty".to_owned());
    }
    if assay.assignment_algorithm != ASSIGNMENT_ALGORITHM {
        errors.push(format!(
            "unsupported assignment_algorithm: {}",
            assay.assignment_algorithm
        ));
    }
    if !is_lower_hex_64(&assay.assignment_seed_revealed) {
        errors.push(
            "assignment_seed_revealed must be 64 lowercase hexadecimal characters".to_owned(),
        );
    } else {
        let commitment = sha256_hex(assay.assignment_seed_revealed.as_bytes());
        if commitment != assay.assignment_seed_commitment_sha256 {
            errors.push("assignment seed does not match its SHA-256 commitment".to_owned());
        }
    }
    if !is_lower_hex_64(&assay.assignment_seed_commitment_sha256) {
        errors.push(
            "assignment_seed_commitment_sha256 must be 64 lowercase hexadecimal characters"
                .to_owned(),
        );
    }
    if assay.minimum_per_order_per_stratum == 0 {
        errors.push("minimum_per_order_per_stratum must be at least 1".to_owned());
    }
    if assay.minimum_identical_controls_for_bias_flag == 0 {
        errors.push("minimum_identical_controls_for_bias_flag must be at least 1".to_owned());
    }
    if assay.presentations.is_empty() {
        errors.push("presentations must not be empty".to_owned());
    }
    if assay.claim_ceiling.is_empty()
        || assay
            .claim_ceiling
            .iter()
            .any(|claim| claim.trim().is_empty())
    {
        errors.push("claim_ceiling must contain non-empty limits".to_owned());
    }
    if !errors.is_empty() {
        return Err(errors.join("; "));
    }

    let mut presentation_ids = BTreeSet::new();
    let mut judge_case_ids = BTreeSet::new();
    let mut judge_pair_ids = BTreeSet::new();
    let mut case_identities: BTreeMap<String, (String, String, String, String)> = BTreeMap::new();
    let mut strata: BTreeMap<String, StratumCounter> = BTreeMap::new();
    let mut cases: BTreeMap<String, CaseCounter> = BTreeMap::new();
    let mut judges = BTreeSet::new();
    let mut order_counts = OrderCounts::default();
    let mut choice_counts = ChoiceCounts::default();

    let mut controls_total = 0usize;
    let mut controls_left = 0usize;
    let mut controls_right = 0usize;
    let mut controls_tie = 0usize;
    let mut controls_neither = 0usize;
    let mut controls_skip = 0usize;
    let mut candidate_a_wins = 0usize;
    let mut candidate_pair_decisive = 0usize;

    for row in &assay.presentations {
        if row.presentation_id.trim().is_empty()
            || row.case_id.trim().is_empty()
            || row.stratum.trim().is_empty()
        {
            errors.push("presentation_id, case_id and stratum must not be empty".to_owned());
            continue;
        }
        if !presentation_ids.insert(row.presentation_id.clone()) {
            errors.push(format!(
                "duplicate presentation_id: {}",
                row.presentation_id
            ));
        }
        if !is_lower_hex_64(&row.judge_pseudonym_sha256)
            || !is_lower_hex_64(&row.candidate_a_sha256)
            || !is_lower_hex_64(&row.candidate_b_sha256)
            || !is_lower_hex_64(&row.judgment_protocol_sha256)
        {
            errors.push(format!(
                "presentation {} has a malformed SHA-256 field",
                row.presentation_id
            ));
            continue;
        }

        judges.insert(row.judge_pseudonym_sha256.clone());

        let judge_case_key = format!("{}:{}", row.judge_pseudonym_sha256, row.case_id);
        if !judge_case_ids.insert(judge_case_key) {
            errors.push(format!(
                "rater saw the same case more than once: {}",
                row.presentation_id
            ));
        }

        let (first, second) = if row.candidate_a_sha256 <= row.candidate_b_sha256 {
            (&row.candidate_a_sha256, &row.candidate_b_sha256)
        } else {
            (&row.candidate_b_sha256, &row.candidate_a_sha256)
        };
        let judge_pair_key = format!("{}:{}:{}", row.judge_pseudonym_sha256, first, second);
        if !judge_pair_ids.insert(judge_pair_key) {
            errors.push(format!(
                "rater saw the same candidate pair more than once: {}",
                row.presentation_id
            ));
        }

        let identity = (
            row.candidate_a_sha256.clone(),
            row.candidate_b_sha256.clone(),
            row.stratum.clone(),
            row.control_kind.clone(),
        );
        if let Some(existing) = case_identities.get(&row.case_id) {
            if existing != &identity {
                errors.push(format!(
                    "case_id maps to inconsistent candidates or stratum: {}",
                    row.case_id
                ));
            }
        } else {
            case_identities.insert(row.case_id.clone(), identity);
        }

        match row.control_kind.as_str() {
            "candidate_pair" if row.candidate_a_sha256 == row.candidate_b_sha256 => {
                errors.push(format!(
                    "candidate_pair requires distinct candidate hashes: {}",
                    row.presentation_id
                ));
            }
            "identical_duplicate" if row.candidate_a_sha256 != row.candidate_b_sha256 => {
                errors.push(format!(
                    "identical_duplicate requires identical candidate hashes: {}",
                    row.presentation_id
                ));
            }
            "candidate_pair" | "identical_duplicate" => {}
            other => errors.push(format!(
                "unsupported control_kind {other} in {}",
                row.presentation_id
            )),
        }

        let expected_left = if assignment_selects_a_left(
            &assay.assignment_seed_revealed,
            &assay.assay_id,
            &row.presentation_id,
        ) {
            "a"
        } else {
            "b"
        };
        if row.left_candidate != expected_left {
            errors.push(format!(
                "presentation {} violates its precommitted display assignment",
                row.presentation_id
            ));
        }
        if row.left_candidate != "a" && row.left_candidate != "b" {
            errors.push(format!(
                "presentation {} has invalid left_candidate",
                row.presentation_id
            ));
            continue;
        }

        match row.choice.as_str() {
            "left" => choice_counts.left += 1,
            "right" => choice_counts.right += 1,
            "tie" => choice_counts.tie += 1,
            "neither" => choice_counts.neither += 1,
            "skip" => choice_counts.skip += 1,
            other => {
                errors.push(format!(
                    "invalid choice {other} in {}",
                    row.presentation_id
                ));
                continue;
            }
        }

        let stratum = strata.entry(row.stratum.clone()).or_default();
        stratum.presentations += 1;
        if row.left_candidate == "a" {
            order_counts.a_left += 1;
            stratum.a_left += 1;
        } else {
            order_counts.b_left += 1;
            stratum.b_left += 1;
        }

        if row.control_kind == "identical_duplicate" {
            controls_total += 1;
            match row.choice.as_str() {
                "left" => controls_left += 1,
                "right" => controls_right += 1,
                "tie" => controls_tie += 1,
                "neither" => controls_neither += 1,
                "skip" => controls_skip += 1,
                _ => {}
            }
        } else if row.control_kind == "candidate_pair" {
            match row.choice.as_str() {
                "left" | "right" => {
                    candidate_pair_decisive += 1;
                    let selected_a = (row.left_candidate == "a" && row.choice == "left")
                        || (row.left_candidate == "b" && row.choice == "right");
                    if selected_a {
                        candidate_a_wins += 1;
                    }
                    let counter = cases.entry(row.case_id.clone()).or_default();
                    if row.left_candidate == "a" {
                        counter.a_left_decisive += 1;
                        if selected_a {
                            counter.a_left_wins += 1;
                        }
                    } else {
                        counter.a_right_decisive += 1;
                        if selected_a {
                            counter.a_right_wins += 1;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    for (stratum_name, counts) in &strata {
        let minimum = assay.minimum_per_order_per_stratum;
        if counts.a_left < minimum || counts.b_left < minimum {
            errors.push(format!(
                "stratum {stratum_name} is not counterbalanced: A-left={}, B-left={}, minimum={minimum}",
                counts.a_left, counts.b_left
            ));
        }
    }
    if !errors.is_empty() {
        return Err(errors.join("; "));
    }

    let stratum_balance = strata
        .into_iter()
        .map(|(name, counts)| {
            let balance_requirement_met = counts.a_left >= assay.minimum_per_order_per_stratum
                && counts.b_left >= assay.minimum_per_order_per_stratum;
            (
                name,
                StratumBalance {
                    presentations: counts.presentations,
                    a_left: counts.a_left,
                    b_left: counts.b_left,
                    balance_requirement_met,
                },
            )
        })
        .collect();

    let controls_decisive = controls_left + controls_right;
    let left_choice_rate = rate(controls_left, controls_decisive);
    let controls_interval = wilson_95(controls_left, controls_decisive);
    let directional_bias_detected = controls_decisive
        >= assay.minimum_identical_controls_for_bias_flag
        && controls_interval.is_some_and(|interval| interval[0] > 0.5 || interval[1] < 0.5);
    let side_bias_status = if controls_decisive < assay.minimum_identical_controls_for_bias_flag {
        "insufficient_identical_controls"
    } else if directional_bias_detected {
        "directional_side_bias_detected"
    } else {
        "no_directional_bias_detected_not_equivalence_proof"
    };

    let mut effect_cases = Vec::new();
    for (case_id, counts) in cases {
        if counts.a_left_decisive > 0 && counts.a_right_decisive > 0 {
            let left_rate = counts.a_left_wins as f64 / counts.a_left_decisive as f64;
            let right_rate = counts.a_right_wins as f64 / counts.a_right_decisive as f64;
            let signed_gap = left_rate - right_rate;
            effect_cases.push(CaseOrderEffect {
                case_id,
                a_left_decisive: counts.a_left_decisive,
                a_right_decisive: counts.a_right_decisive,
                a_win_rate_when_a_left: left_rate,
                a_win_rate_when_a_right: right_rate,
                signed_a_win_rate_gap: signed_gap,
                absolute_a_win_rate_gap: signed_gap.abs(),
            });
        }
    }
    let mean_signed_gap = if effect_cases.is_empty() {
        None
    } else {
        Some(
            effect_cases
                .iter()
                .map(|case| case.signed_a_win_rate_gap)
                .sum::<f64>()
                / effect_cases.len() as f64,
        )
    };
    let mean_absolute_gap = if effect_cases.is_empty() {
        None
    } else {
        Some(
            effect_cases
                .iter()
                .map(|case| case.absolute_a_win_rate_gap)
                .sum::<f64>()
                / effect_cases.len() as f64,
        )
    };

    Ok(AuditReport {
        schema_version: SCHEMA_VERSION.to_owned(),
        assay_id: assay.assay_id.clone(),
        valid: true,
        presentation_count: assay.presentations.len(),
        unique_judge_count: judges.len(),
        order_counts,
        stratum_balance,
        choice_counts,
        candidate_a_win_rate: rate(candidate_a_wins, candidate_pair_decisive),
        side_bias: SideBiasReport {
            identical_control_presentations: controls_total,
            decisive_control_count: controls_decisive,
            left_choice_count: controls_left,
            right_choice_count: controls_right,
            tie_count: controls_tie,
            neither_count: controls_neither,
            skip_count: controls_skip,
            left_choice_rate,
            wilson_95: controls_interval,
            directional_bias_detected,
            status: side_bias_status.to_owned(),
        },
        order_effect: OrderEffectReport {
            matched_case_count: effect_cases.len(),
            mean_signed_a_win_rate_gap: mean_signed_gap,
            mean_absolute_a_win_rate_gap: mean_absolute_gap,
            cases: effect_cases,
        },
        claim_ceiling: assay.claim_ceiling.clone(),
    })
}

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(input: &[u8]) -> String {
    use std::fmt::Write as _;

    let digest = Sha256::digest(input);
    let mut encoded = String::with_capacity(64);
    for byte in digest.iter() {
        write!(&mut encoded, "{:02x}", *byte).expect("writing to String cannot fail");
    }
    encoded
}

fn assignment_selects_a_left(seed: &str, assay_id: &str, presentation_id: &str) -> bool {
    let mut hasher = Sha256::new();
    hasher.update(ASSIGNMENT_DOMAIN);
    hasher.update(seed.as_bytes());
    hasher.update(b"\0");
    hasher.update(assay_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(presentation_id.as_bytes());
    let digest = hasher.finalize();
    (digest[0] & 0x80) == 0
}

fn rate(successes: usize, total: usize) -> Option<f64> {
    if total == 0 {
        None
    } else {
        Some(successes as f64 / total as f64)
    }
}

fn wilson_95(successes: usize, total: usize) -> Option<[f64; 2]> {
    if total == 0 {
        return None;
    }
    let z = 1.959_963_984_540_054_f64;
    let n = total as f64;
    let p = successes as f64 / n;
    let z_squared = z * z;
    let denominator = 1.0 + z_squared / n;
    let centre = (p + z_squared / (2.0 * n)) / denominator;
    let half_width =
        z * ((p * (1.0 - p) / n) + (z_squared / (4.0 * n * n))).sqrt() / denominator;
    Some([(centre - half_width).max(0.0), (centre + half_width).min(1.0)])
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../fixtures/valid-assay-v1.json");

    fn fixture() -> AssayV1 {
        serde_json::from_str(FIXTURE).expect("fixture must deserialize")
    }

    fn digest(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }

    #[test]
    fn valid_fixture_checks_assignment_and_counterbalancing() {
        let report = parse_and_audit(FIXTURE).expect("fixture should be valid");
        assert!(report.valid);
        assert_eq!(report.presentation_count, 4);
        assert_eq!(report.unique_judge_count, 4);
        assert_eq!(report.order_effect.matched_case_count, 1);
        assert_eq!(report.order_effect.mean_absolute_a_win_rate_gap, Some(0.0));
        assert_eq!(report.side_bias.status, "insufficient_identical_controls");
    }

    #[test]
    fn injected_identical_control_bias_is_reported() {
        let mut assay = fixture();
        for row in assay
            .presentations
            .iter_mut()
            .filter(|row| row.control_kind == "identical_duplicate")
        {
            let index = row
                .presentation_id
                .rsplit('-')
                .next()
                .and_then(|part| part.parse::<usize>().ok())
                .expect("control index");
            row.choice = if index < 18 { "left" } else { "right" }.to_owned();
        }
        for index in 2..24 {
            let presentation_id = format!("control-{index:02}");
            let left_candidate = if assignment_selects_a_left(
                &assay.assignment_seed_revealed,
                &assay.assay_id,
                &presentation_id,
            ) {
                "a"
            } else {
                "b"
            };
            let audio = digest(&format!("identical-audio-{index:02}"));
            assay.presentations.push(PresentationV1 {
                presentation_id: presentation_id.clone(),
                case_id: format!("identical-control-{index:02}"),
                stratum: "identical-control".to_owned(),
                judge_pseudonym_sha256: digest(&format!("judge-control-{index:02}")),
                candidate_a_sha256: audio.clone(),
                candidate_b_sha256: audio,
                left_candidate: left_candidate.to_owned(),
                control_kind: "identical_duplicate".to_owned(),
                choice: if index < 18 { "left" } else { "right" }.to_owned(),
                judgment_protocol_sha256: digest("mel-blind-preference-protocol-v1"),
            });
        }

        let report = audit_assay(&assay).expect("injected-control assay remains structurally valid");
        assert_eq!(report.side_bias.decisive_control_count, 24);
        assert!(report.side_bias.directional_bias_detected);
        assert_eq!(report.side_bias.status, "directional_side_bias_detected");
    }

    #[test]
    fn tampered_display_order_is_rejected() {
        let mut assay = fixture();
        assay.presentations[0].left_candidate = "b".to_owned();
        let error = audit_assay(&assay).expect_err("tampered order must fail");
        assert!(error.contains("precommitted display assignment"));
    }

    #[test]
    fn seed_commitment_mismatch_is_rejected() {
        let mut assay = fixture();
        assay.assignment_seed_revealed = "0".repeat(64);
        let error = audit_assay(&assay).expect_err("changed seed must fail");
        assert!(error.contains("does not match its SHA-256 commitment"));
    }

    #[test]
    fn repeated_pair_for_one_rater_is_rejected() {
        let mut assay = fixture();
        let mut duplicate = assay.presentations[2].clone();
        duplicate.presentation_id = "pair-00-r3".to_owned();
        duplicate.left_candidate = if assignment_selects_a_left(
            &assay.assignment_seed_revealed,
            &assay.assay_id,
            &duplicate.presentation_id,
        ) {
            "a"
        } else {
            "b"
        }
        .to_owned();
        assay.presentations.push(duplicate);
        let error = audit_assay(&assay).expect_err("same rater/pair exposure must fail");
        assert!(error.contains("rater saw the same candidate pair more than once"));
    }

    #[test]
    fn unbalanced_stratum_is_rejected() {
        let mut assay = fixture();
        assay.minimum_per_order_per_stratum = 2;
        let error = audit_assay(&assay).expect_err("underpowered stratum must fail");
        assert!(error.contains("is not counterbalanced"));
    }

    #[test]
    fn identical_control_cannot_claim_different_candidate_bytes() {
        let mut assay = fixture();
        assay.presentations[0].candidate_b_sha256 = digest("different-audio");
        let error = audit_assay(&assay).expect_err("non-identical control must fail");
        assert!(error.contains("identical_duplicate requires identical candidate hashes"));
    }
}
