use anyhow::{ensure, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

const LEDGER_SCHEMA_VERSION: &str = "sld-oracle-ledger-0.1";

#[derive(Debug, Parser)]
#[command(
    name = "sld-oracle-audit",
    about = "Validate independent, blinded oracle-adjudication evidence before SLD comparisons"
)]
struct Args {
    /// Immutable adjudication ledger JSON.
    #[arg(long)]
    ledger: PathBuf,
    /// JSON array containing every scored case ID, frozen before adjudication.
    #[arg(long)]
    case_ids: PathBuf,
    /// Output receipt-compatible oracle evidence summary.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Reviewer {
    reviewer_id: String,
    expertise_categories: Vec<String>,
    case_authorship_conflicts: Vec<String>,
    conflict_disclosure: String,
    independence_attested: bool,
}

#[derive(Debug, Deserialize)]
struct Judgment {
    reviewer_id: String,
    disposition: String,
    critical_invariant_flags: Vec<String>,
    confidence: f64,
    rationale: String,
    assumptions: Vec<String>,
    ambiguity_flags: Vec<String>,
    submitted_at: String,
}

#[derive(Debug, Deserialize)]
struct Adjudication {
    status: String,
    final_disposition: Option<String>,
    adjudicator_id: Option<String>,
    rationale: String,
    reviewed_at: String,
}

#[derive(Debug, Deserialize)]
struct Dissent {
    reviewer_id: String,
    reason: String,
}

#[derive(Debug, Deserialize)]
struct CaseReview {
    case_id: String,
    risk_level: String,
    initial_labels_blinded: bool,
    initial_judgments: Vec<Judgment>,
    adjudication: Adjudication,
    dissent: Vec<Dissent>,
}

#[derive(Debug, Deserialize)]
struct DeclaredSummary {
    expected_case_count: usize,
    cases_with_two_independent_initial_reviews: usize,
    cases_adjudicated: usize,
    cases_indeterminate: usize,
    unresolved_disagreements: usize,
    critical_cases_reviewed: usize,
    reviewer_count: usize,
    blind_initial_reviews: bool,
    case_assignment_randomized: bool,
    case_authors_excluded_from_adjudication: bool,
    conflicts_disclosed: bool,
}

#[derive(Debug, Deserialize)]
struct Ledger {
    schema_version: String,
    status: String,
    case_id_inventory_sha256: String,
    rubric_sha256: String,
    protocol_sha256: String,
    assignment_seed: u64,
    randomized_case_order: Vec<String>,
    reviewers: Vec<Reviewer>,
    cases: Vec<CaseReview>,
    summary: DeclaredSummary,
    completed_at: Option<String>,
    limitations: Vec<String>,
}

#[derive(Debug, Serialize)]
struct OracleEvidenceSummary {
    status: String,
    ledger_sha256: String,
    case_id_inventory_sha256: String,
    protocol_sha256: String,
    rubric_sha256: String,
    reviewer_count: usize,
    expected_case_count: usize,
    cases_with_two_independent_initial_reviews: usize,
    cases_adjudicated: usize,
    cases_indeterminate: usize,
    unresolved_disagreements: usize,
    critical_cases_reviewed: usize,
    blind_initial_reviews: bool,
    case_assignment_randomized: bool,
    case_authors_excluded_from_adjudication: bool,
    conflicts_disclosed: bool,
    initial_label_pairwise_agreement_rate: f64,
    initial_label_unanimous_case_rate: f64,
    completed_at: Option<String>,
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn index(&mut self, upper: usize) -> usize {
        (((self.next_u64() as u128) * (upper as u128)) >> 64) as usize
    }
}

fn seeded_shuffle(mut items: Vec<String>, seed: u64) -> Vec<String> {
    items.sort();
    let mut rng = SplitMix64::new(seed);
    for index in (1..items.len()).rev() {
        let selected = rng.index(index + 1);
        items.swap(index, selected);
    }
    items
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &PathBuf) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn audit(ledger: &Ledger, expected_ids: &[String], ledger_bytes: &[u8]) -> Result<OracleEvidenceSummary> {
    ensure!(
        ledger.schema_version == LEDGER_SCHEMA_VERSION,
        "unsupported oracle ledger schema version"
    );
    ensure!(ledger.status == "COMPLETE", "oracle ledger status must be COMPLETE");
    ensure!(
        ledger.completed_at.as_ref().is_some_and(|value| !value.trim().is_empty()),
        "completed oracle ledger must carry completed_at"
    );
    ensure!(
        ledger.limitations.iter().any(|value| !value.trim().is_empty()),
        "ledger must declare at least one limitation"
    );
    ensure!(
        is_lower_hex(&ledger.case_id_inventory_sha256, 64)
            && is_lower_hex(&ledger.rubric_sha256, 64)
            && is_lower_hex(&ledger.protocol_sha256, 64),
        "corpus, rubric, and protocol digests must be lowercase SHA-256"
    );

    let mut expected: Vec<String> = expected_ids.to_vec();
    ensure!(!expected.is_empty(), "frozen case-ID inventory is empty");
    ensure!(
        expected.iter().all(|id| !id.trim().is_empty()),
        "frozen case-ID inventory contains blank IDs"
    );
    expected.sort();
    ensure!(
        expected.windows(2).all(|pair| pair[0] != pair[1]),
        "frozen case-ID inventory contains duplicate IDs"
    );
    let canonical_inventory = serde_json::to_vec(&expected)?;
    let inventory_digest = sha256_hex(&canonical_inventory);
    ensure!(
        inventory_digest == ledger.case_id_inventory_sha256,
        "ledger case_id_inventory_sha256 does not match canonical sorted case-ID inventory"
    );

    let mut reviewers = BTreeMap::new();
    for reviewer in &ledger.reviewers {
        ensure!(
            !reviewer.reviewer_id.trim().is_empty(),
            "reviewer identifier is blank"
        );
        ensure!(
            !reviewer.expertise_categories.is_empty(),
            "reviewer {} has no declared expertise category",
            reviewer.reviewer_id
        );
        ensure!(
            reviewer.independence_attested,
            "reviewer {} has not attested to independence",
            reviewer.reviewer_id
        );
        ensure!(
            reviewer.conflict_disclosure != "UNRESOLVED",
            "reviewer {} has unresolved conflicts of interest",
            reviewer.reviewer_id
        );
        ensure!(
            matches!(
                reviewer.conflict_disclosure.as_str(),
                "NONE_DECLARED" | "DISCLOSED_AND_MANAGED" | "UNRESOLVED"
            ),
            "reviewer {} has invalid conflict-disclosure state",
            reviewer.reviewer_id
        );
        ensure!(
            reviewers.insert(reviewer.reviewer_id.clone(), reviewer).is_none(),
            "duplicate reviewer identifier {}",
            reviewer.reviewer_id
        );
    }
    ensure!(reviewers.len() >= 2, "at least two reviewers are required");
    ensure!(
        ledger.summary.reviewer_count == reviewers.len(),
        "summary reviewer_count does not match reviewer registry"
    );

    let expected_set: BTreeSet<&str> = expected.iter().map(String::as_str).collect();
    let expected_order = seeded_shuffle(expected.clone(), ledger.assignment_seed);
    ensure!(
        ledger.randomized_case_order == expected_order,
        "randomized_case_order does not match the frozen seeded shuffle"
    );
    let mut case_map = BTreeMap::new();
    let mut reviewed_count = 0usize;
    let mut adjudicated_count = 0usize;
    let mut indeterminate_count = 0usize;
    let mut unresolved_count = 0usize;
    let mut critical_reviewed = 0usize;
    let mut initial_pairwise_agreement_rates = Vec::new();
    let mut initial_unanimous_cases = 0usize;
    let mut all_blinded = true;
    let mut all_reviewed_twice = true;

    for case in &ledger.cases {
        ensure!(
            expected_set.contains(case.case_id.as_str()),
            "ledger contains unexpected case {}",
            case.case_id
        );
        ensure!(
            case_map.insert(case.case_id.clone(), case).is_none(),
            "ledger contains duplicate case {}",
            case.case_id
        );
        ensure!(
            matches!(case.risk_level.as_str(), "critical" | "high" | "medium" | "low" | "other"),
            "case {} has invalid risk_level {}",
            case.case_id,
            case.risk_level
        );
        all_blinded &= case.initial_labels_blinded;
        ensure!(
            case.initial_labels_blinded,
            "case {} initial labels were not recorded as blinded",
            case.case_id
        );

        let mut initial_reviewers = BTreeSet::new();
        for judgment in &case.initial_judgments {
            ensure!(
                initial_reviewers.insert(judgment.reviewer_id.clone()),
                "case {} has duplicate initial judgments from reviewer {}",
                case.case_id,
                judgment.reviewer_id
            );
            let reviewer = reviewers
                .get(&judgment.reviewer_id)
                .with_context(|| format!("case {} refers to undeclared reviewer {}", case.case_id, judgment.reviewer_id))?;
            ensure!(
                !reviewer.case_authorship_conflicts.iter().any(|id| id == &case.case_id),
                "case author {} cannot provide independent initial judgment for {}",
                judgment.reviewer_id,
                case.case_id
            );
            ensure!(
                judgment.confidence.is_finite()
                    && (0.0..=1.0).contains(&judgment.confidence),
                "case {} contains invalid confidence from {}",
                case.case_id,
                judgment.reviewer_id
            );
            ensure!(
                !judgment.disposition.trim().is_empty()
                    && !judgment.rationale.trim().is_empty()
                    && !judgment.submitted_at.trim().is_empty()
                    && judgment.critical_invariant_flags.iter().all(|flag| !flag.trim().is_empty())
                    && judgment.assumptions.iter().all(|item| !item.trim().is_empty())
                    && judgment.ambiguity_flags.iter().all(|item| !item.trim().is_empty()),
                "case {} contains incomplete initial judgment from {}",
                case.case_id,
                judgment.reviewer_id
            );
        }
        ensure!(
            initial_reviewers.len() >= 2,
            "case {} needs at least two distinct independent initial judgments",
            case.case_id
        );
        reviewed_count += 1;
        all_reviewed_twice &= initial_reviewers.len() >= 2;
        if initial_dispositions.len() == 1 {
            initial_unanimous_cases += 1;
        }
        let judgments = &case.initial_judgments;
        let mut pair_count = 0usize;
        let mut agreeing_pairs = 0usize;
        for left in 0..judgments.len() {
            for right in (left + 1)..judgments.len() {
                pair_count += 1;
                if judgments[left].disposition == judgments[right].disposition {
                    agreeing_pairs += 1;
                }
            }
        }
        ensure!(pair_count > 0, "case {} has no initial-review pairs", case.case_id);
        initial_pairwise_agreement_rates.push(agreeing_pairs as f64 / pair_count as f64);

        let adjudication = &case.adjudication;
        ensure!(
            !adjudication.rationale.trim().is_empty() && !adjudication.reviewed_at.trim().is_empty(),
            "case {} has incomplete adjudication rationale/timestamp",
            case.case_id
        );
        ensure!(
            matches!(adjudication.status.as_str(), "ACCEPTED" | "REVISED" | "INDETERMINATE" | "REJECTED"),
            "case {} has invalid adjudication status {}",
            case.case_id,
            adjudication.status
        );
        let initial_dispositions: BTreeSet<&str> =
            case.initial_judgments.iter().map(|judgment| judgment.disposition.as_str()).collect();
        if matches!(adjudication.status.as_str(), "ACCEPTED" | "REVISED") {
            let final_disposition = adjudication.final_disposition.as_deref()
                .context(format!("case {} lacks a final disposition", case.case_id))?;
            ensure!(!final_disposition.trim().is_empty(), "case {} final disposition is blank", case.case_id);

            let requires_adjudicator =
                initial_dispositions.len() > 1 || adjudication.status == "REVISED";
            if requires_adjudicator {
                let adjudicator_id = adjudication.adjudicator_id.as_deref()
                    .context(format!("case {} requires an independent adjudicator", case.case_id))?;
                ensure!(
                    !initial_reviewers.contains(adjudicator_id),
                    "case {} adjudicator must differ from initial reviewers",
                    case.case_id
                );
            }
            if let Some(adjudicator_id) = adjudication.adjudicator_id.as_deref() {
                ensure!(
                    !initial_reviewers.contains(adjudicator_id),
                    "case {} adjudicator cannot also be an initial reviewer",
                    case.case_id
                );
                let adjudicator = reviewers.get(adjudicator_id)
                    .with_context(|| format!("case {} refers to undeclared adjudicator {}", case.case_id, adjudicator_id))?;
                ensure!(
                    !adjudicator.case_authorship_conflicts.iter().any(|id| id == &case.case_id)
                        && adjudicator.independence_attested
                        && adjudicator.conflict_disclosure != "UNRESOLVED",
                    "case {} adjudicator is not independent or has an unmanaged conflict",
                    case.case_id
                );
            }
            if adjudication.status == "ACCEPTED" && initial_dispositions.len() == 1 {
                ensure!(
                    initial_dispositions.iter().next().copied() == Some(final_disposition),
                    "case {} accepted disposition differs from unanimous initial judgments",
                    case.case_id
                );
            }
            adjudicated_count += 1;
        } else {
            indeterminate_count += if adjudication.status == "INDETERMINATE" { 1 } else { 0 };
            unresolved_count += 1;
        }
        if case.risk_level == "critical" {
            critical_reviewed += if initial_reviewers.len() >= 2 { 1 } else { 0 };
        }
        for dissent in &case.dissent {
            ensure!(
                !dissent.reviewer_id.trim().is_empty() && !dissent.reason.trim().is_empty(),
                "case {} has incomplete dissent record",
                case.case_id
            );
            ensure!(
                initial_reviewers.contains(&dissent.reviewer_id),
                "case {} dissent must reference an initial reviewer",
                case.case_id
            );
        }
    }

    ensure!(
        case_map.len() == expected_set.len()
            && expected_set.iter().all(|id| case_map.contains_key(*id)),
        "ledger does not cover the exact frozen case-ID inventory"
    );
    ensure!(
        ledger.summary.expected_case_count == expected.len(),
        "summary expected_case_count mismatch"
    );
    ensure!(
        ledger.summary.cases_with_two_independent_initial_reviews == reviewed_count
            && all_reviewed_twice,
        "summary initial-review count mismatch"
    );
    ensure!(
        ledger.summary.cases_adjudicated == adjudicated_count,
        "summary cases_adjudicated mismatch"
    );
    ensure!(
        ledger.summary.cases_indeterminate == indeterminate_count,
        "summary cases_indeterminate mismatch"
    );
    ensure!(
        ledger.summary.unresolved_disagreements == unresolved_count,
        "summary unresolved_disagreements mismatch"
    );
    ensure!(
        ledger.summary.critical_cases_reviewed == critical_reviewed,
        "summary critical_cases_reviewed mismatch"
    );
    ensure!(
        ledger.summary.blind_initial_reviews && all_blinded,
        "summary claims blinded review but case records contradict it"
    );
    ensure!(
        ledger.summary.case_assignment_randomized
            && ledger.randomized_case_order.len() == expected.len()
            && ledger.randomized_case_order.iter().collect::<BTreeSet<_>>().len() == expected.len(),
        "summary claims randomized assignment but order is incomplete/duplicated"
    );
    ensure!(
        ledger.summary.case_authors_excluded_from_adjudication,
        "summary cannot claim case-author exclusion is false"
    );
    ensure!(
        ledger.summary.conflicts_disclosed,
        "summary must attest that reviewer conflicts were disclosed"
    );
    ensure!(
        adjudicated_count == expected.len() && indeterminate_count == 0 && unresolved_count == 0,
        "COMPLETE oracle ledger cannot contain unadjudicated, rejected, or indeterminate cases"
    );

    Ok(OracleEvidenceSummary {
        status: "COMPLETE".to_owned(),
        ledger_sha256: sha256_hex(ledger_bytes),
        case_id_inventory_sha256: inventory_digest,
        protocol_sha256: ledger.protocol_sha256.clone(),
        rubric_sha256: ledger.rubric_sha256.clone(),
        reviewer_count: reviewers.len(),
        expected_case_count: expected.len(),
        cases_with_two_independent_initial_reviews: reviewed_count,
        cases_adjudicated: adjudicated_count,
        cases_indeterminate: indeterminate_count,
        unresolved_disagreements: unresolved_count,
        critical_cases_reviewed: critical_reviewed,
        blind_initial_reviews: true,
        case_assignment_randomized: true,
        case_authors_excluded_from_adjudication: true,
        conflicts_disclosed: true,
        initial_label_pairwise_agreement_rate: initial_pairwise_agreement_rates.iter().sum::<f64>()
            / initial_pairwise_agreement_rates.len() as f64,
        initial_label_unanimous_case_rate: initial_unanimous_cases as f64 / expected.len() as f64,
        completed_at: ledger.completed_at.clone(),
    })
}

fn run(args: Args) -> Result<()> {
    let ledger_bytes = fs::read(&args.ledger).with_context(|| format!("read {}", args.ledger.display()))?;
    let ledger: Ledger = serde_json::from_slice(&ledger_bytes)
        .with_context(|| format!("parse ledger {}", args.ledger.display()))?;
    let case_ids_value: serde_json::Value = read_json(&args.case_ids)?;
    let case_ids = case_ids_value
        .as_array()
        .context("case_ids input must be a JSON array of strings")?
        .iter()
        .map(|value| value.as_str().map(str::to_owned).context("case IDs must be strings"))
        .collect::<Result<Vec<_>>>()?;
    let summary = audit(&ledger, &case_ids, &ledger_bytes)?;
    if let Some(parent) = args.out.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let bytes = serde_json::to_vec_pretty(&summary)?;
    fs::write(&args.out, bytes).with_context(|| format!("write {}", args.out.display()))?;
    println!(
        "Oracle audit COMPLETE: {} cases, {} reviewers, {} critical cases; immutable ledger {}",
        summary.expected_case_count,
        summary.reviewer_count,
        summary.critical_cases_reviewed,
        summary.ledger_sha256
    );
    Ok(())
}

fn main() -> Result<()> {
    if let Err(error) = run(Args::parse()) {
        eprintln!("SLD oracle audit failed closed: {error:#}");
        std::process::exit(2);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn digest(bytes: &[u8]) -> String { sha256_hex(bytes) }

    fn fixture() -> (Ledger, Vec<String>, Vec<u8>) {
        let ids = vec!["CASE-A".to_string(), "CASE-B".to_string()];
        let sorted = serde_json::to_vec(&ids).expect("inventory JSON");
        let corpus_digest = digest(&sorted);
        let reviewers = vec![
            Reviewer {
                reviewer_id: "REV-A".to_string(),
                expertise_categories: vec!["AI_SAFETY".to_string()],
                case_authorship_conflicts: vec![],
                conflict_disclosure: "NONE_DECLARED".to_string(),
                independence_attested: true,
            },
            Reviewer {
                reviewer_id: "REV-B".to_string(),
                expertise_categories: vec!["MEASUREMENT".to_string()],
                case_authorship_conflicts: vec![],
                conflict_disclosure: "NONE_DECLARED".to_string(),
                independence_attested: true,
            },
            Reviewer {
                reviewer_id: "REV-C".to_string(),
                expertise_categories: vec!["DOMAIN_ETHICS".to_string()],
                case_authorship_conflicts: vec![],
                conflict_disclosure: "NONE_DECLARED".to_string(),
                independence_attested: true,
            },
        ];
        let mk_case = |id: &str, risk_level: &str| CaseReview {
            case_id: id.to_string(),
            risk_level: risk_level.to_string(),
            initial_labels_blinded: true,
            initial_judgments: vec![
                Judgment {
                    reviewer_id: "REV-A".to_string(),
                    disposition: "HONOR_EXIT".to_string(),
                    critical_invariant_flags: vec![],
                    confidence: 0.9,
                    rationale: "Respect explicit stop request.".to_string(),
                    assumptions: vec![],
                    ambiguity_flags: vec![],
                    submitted_at: "2026-10-22T11:00:00Z".to_string(),
                },
                Judgment {
                    reviewer_id: "REV-B".to_string(),
                    disposition: "HONOR_EXIT".to_string(),
                    critical_invariant_flags: vec![],
                    confidence: 0.8,
                    rationale: "No override condition is stated.".to_string(),
                    assumptions: vec![],
                    ambiguity_flags: vec![],
                    submitted_at: "2026-10-22T11:01:00Z".to_string(),
                },
            ],
            adjudication: Adjudication {
                status: "ACCEPTED".to_string(),
                final_disposition: Some("HONOR_EXIT".to_string()),
                adjudicator_id: None,
                rationale: "Independent judgments agree and satisfy the rubric.".to_string(),
                reviewed_at: "2026-10-22T11:02:00Z".to_string(),
            },
            dissent: vec![],
        };
        let cases = vec![mk_case("CASE-A", "critical"), mk_case("CASE-B", "high")];
        let ledger = Ledger {
            schema_version: LEDGER_SCHEMA_VERSION.to_string(),
            status: "COMPLETE".to_string(),
            case_id_inventory_sha256: corpus_digest,
            rubric_sha256: "a".repeat(64),
            protocol_sha256: "b".repeat(64),
            assignment_seed: 7,
            randomized_case_order: seeded_shuffle(ids.clone(), 7),
            reviewers,
            cases,
            summary: DeclaredSummary {
                expected_case_count: 2,
                cases_with_two_independent_initial_reviews: 2,
                cases_adjudicated: 2,
                cases_indeterminate: 0,
                unresolved_disagreements: 0,
                critical_cases_reviewed: 1,
                reviewer_count: 3,
                blind_initial_reviews: true,
                case_assignment_randomized: true,
                case_authors_excluded_from_adjudication: true,
                conflicts_disclosed: true,
            },
            completed_at: Some("2026-10-22T11:03:00Z".to_string()),
            limitations: vec!["Synthetic unit test only.".to_string()],
        };
        let bytes = serde_json::to_vec(&json!({"fixture": true})).expect("ledger bytes");
        (ledger, ids, bytes)
    }

    #[test]
    fn accepts_complete_fully_covered_independent_ledger() {
        let (ledger, ids, bytes) = fixture();
        let result = audit(&ledger, &ids, &bytes).expect("valid fixture");
        assert_eq!(result.status, "COMPLETE");
        assert_eq!(result.expected_case_count, 2);
        assert_eq!(result.critical_cases_reviewed, 1);
        assert_eq!(result.initial_label_pairwise_agreement_rate, 1.0);
        assert_eq!(result.initial_label_unanimous_case_rate, 1.0);
        assert_eq!(result.ledger_sha256, sha256_hex(&bytes));
    }

    #[test]
    fn rejects_duplicate_reviewer_on_same_case() {
        let (mut ledger, ids, bytes) = fixture();
        ledger.cases[0].initial_judgments[1].reviewer_id = "REV-A".to_string();
        assert!(audit(&ledger, &ids, &bytes).is_err());
    }

    #[test]
    fn rejects_case_authorship_conflict() {
        let (mut ledger, ids, bytes) = fixture();
        ledger.reviewers[0].case_authorship_conflicts.push("CASE-A".to_string());
        assert!(audit(&ledger, &ids, &bytes).is_err());
    }

    #[test]
    fn rejects_unresolved_critical_disagreement() {
        let (mut ledger, ids, bytes) = fixture();
        ledger.cases[0].initial_judgments[1].disposition = "ASK_CLARIFICATION".to_string();
        assert!(audit(&ledger, &ids, &bytes).is_err());
    }

    #[test]
    fn rejects_wrong_frozen_case_inventory() {
        let (ledger, mut ids, bytes) = fixture();
        ids.pop();
        assert!(audit(&ledger, &ids, &bytes).is_err());
    }

    #[test]
    fn resolved_disagreement_is_preserved_in_reliability_metrics() {
        let (mut ledger, ids, bytes) = fixture();
        ledger.cases[0].initial_judgments[1].disposition = "ASK_CLARIFICATION".to_string();
        ledger.cases[0].adjudication.status = "REVISED".to_string();
        ledger.cases[0].adjudication.final_disposition = Some("HONOR_EXIT".to_string());
        ledger.cases[0].adjudication.adjudicator_id = Some("REV-C".to_string());
        ledger.cases[0].adjudication.rationale = "Third reviewer adjudicated the disagreement.".to_string();

        let result = audit(&ledger, &ids, &bytes).expect("independently resolved disagreement");
        assert_eq!(result.initial_label_pairwise_agreement_rate, 0.5);
        assert_eq!(result.initial_label_unanimous_case_rate, 0.5);
        assert_eq!(result.status, "COMPLETE");
    }

}
