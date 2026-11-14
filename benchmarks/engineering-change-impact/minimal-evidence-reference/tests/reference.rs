#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const CASES: &str = include_str!("../../minimal-evidence-cases-v0.1.json");
const MUTATIONS: &str = include_str!("../../minimal-evidence-mutations-v0.1.json");

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorldReport {
    state: String,
    action: Option<String>,
    selected_roots: BTreeMap<String, Vec<String>>,
    sufficient_root_sets: BTreeMap<String, Vec<Vec<String>>>,
    reason_codes: Vec<String>,
    unknown_origin_records: Vec<String>,
    raw_visible_record_count: usize,
    effective_independent_root_count: usize,
}

fn string_at<'a>(value: &'a Value, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing string field {key}"))
}

fn strings_at(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("missing array field {key}"))
        .iter()
        .map(|item| item.as_str().expect("array item must be a string").to_owned())
        .collect()
}

fn array_at<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_else(|| panic!("missing array field {key}"))
}

fn digest_is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn canonical_inventory_sha256(case: &Value, world: &Value) -> String {
    let mut evidence = world_evidence(case, world).to_vec();
    evidence.sort_by(|left, right| string_at(left, "id").cmp(string_at(right, "id")));
    let mut predicates = strings_at(case, "mandatory_predicates");
    predicates.sort();

    let projection = json!({
        "domain": "symthaea.engineering-decision-min-evidence.source-inventory.v0.1",
        "target_configuration": world_target(case, world),
        "source_inventory_state": world_inventory_state(case, world),
        "mandatory_predicates": predicates,
        "minimum_independent_roots_per_predicate": case["minimum_independent_roots_per_predicate"],
        "evidence": evidence
    });
    format!("{:x}", Sha256::digest(serde_json::to_vec(&projection).unwrap()))
}

fn irrelevant_injection_sha256(evidence: &[Value]) -> String {
    let mut irrelevant: Vec<Value> = evidence
        .iter()
        .filter(|record| strings_at(record, "supports").iter().any(|p| p == "irrelevant_background"))
        .cloned()
        .collect();
    irrelevant.sort_by(|left, right| string_at(left, "id").cmp(string_at(right, "id")));
    let projection = json!({
        "domain": "symthaea.engineering-decision-min-evidence.irrelevant-injection.v0.1",
        "evidence": irrelevant
    });
    format!("{:x}", Sha256::digest(serde_json::to_vec(&projection).unwrap()))
}

fn candidate_visible_manifest_sha256(case: &Value, world: &Value) -> String {
    let inventory_digest = world
        .get("source_inventory_sha256")
        .and_then(Value::as_str)
        .or_else(|| case.get("source_inventory_sha256").and_then(Value::as_str))
        .unwrap_or_else(|| panic!("missing source inventory digest for {}", string_at(case, "id")));
    let mut evidence_ids = visible_ids(case, world);
    evidence_ids.sort();
    let projection = json!({
        "domain": "symthaea.engineering-decision-min-evidence.candidate-visible-set.v0.1",
        "source_inventory_sha256": inventory_digest,
        "evidence_ids": evidence_ids
    });
    format!("{:x}", Sha256::digest(serde_json::to_vec(&projection).unwrap()))
}

fn combinations(values: &[String], choose: usize) -> Vec<Vec<String>> {
    fn recurse(
        values: &[String],
        choose: usize,
        start: usize,
        current: &mut Vec<String>,
        output: &mut Vec<Vec<String>>,
    ) {
        if current.len() == choose {
            output.push(current.clone());
            return;
        }
        for index in start..=values.len().saturating_sub(choose - current.len()) {
            current.push(values[index].clone());
            recurse(values, choose, index + 1, current, output);
            current.pop();
        }
    }

    if choose == 0 || choose > values.len() {
        return Vec::new();
    }
    let mut output = Vec::new();
    recurse(values, choose, 0, &mut Vec::new(), &mut output);
    output
}

fn world_evidence<'a>(case: &'a Value, world: &'a Value) -> &'a [Value] {
    world
        .get("evidence")
        .or_else(|| case.get("evidence"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_else(|| panic!("missing evidence array for {}", string_at(case, "id")))
}

fn visible_ids(case: &Value, world: &Value) -> Vec<String> {
    if let Some(ids) = world.get("candidate_visible_evidence_ids") {
        return ids
            .as_array()
            .expect("candidate_visible_evidence_ids must be an array")
            .iter()
            .map(|v| v.as_str().expect("visible ID must be a string").to_owned())
            .collect();
    }
    if let Some(ids) = case.get("candidate_visible_evidence_ids") {
        return ids
            .as_array()
            .expect("candidate_visible_evidence_ids must be an array")
            .iter()
            .map(|v| v.as_str().expect("visible ID must be a string").to_owned())
            .collect();
    }
    panic!("missing candidate-visible IDs for {}", string_at(case, "id"));
}

fn world_target<'a>(case: &'a Value, world: &'a Value) -> &'a str {
    world
        .get("target_configuration")
        .and_then(Value::as_str)
        .or_else(|| case.get("target_configuration").and_then(Value::as_str))
        .unwrap_or_else(|| panic!("missing target configuration for {}", string_at(case, "id")))
}

fn world_inventory_state<'a>(case: &'a Value, world: &'a Value) -> &'a str {
    world
        .get("source_inventory_state")
        .and_then(Value::as_str)
        .or_else(|| case.get("source_inventory_state").and_then(Value::as_str))
        .unwrap_or_else(|| panic!("missing source inventory state for {}", string_at(case, "id")))
}

fn world_action(world: &Value) -> Option<&str> {
    world.get("candidate_action").and_then(Value::as_str)
}

fn admissibility_is_invalid(value: &str) -> bool {
    value.starts_with("inadmissible")
}

fn evaluate_world(case: &Value, world: &Value) -> WorldReport {
    let evidence = world_evidence(case, world);
    let target = world_target(case, world);
    let inventory_state = world_inventory_state(case, world);
    let visible = visible_ids(case, world);
    let mut reasons = BTreeSet::new();
    let mut selected_roots = BTreeMap::new();
    let mut sufficient_root_sets = BTreeMap::new();
    let mut unknown_origin_records = Vec::new();
    let mut raw_visible_record_count = 0usize;
    let mut effective_independent_root_count = 0usize;
    let mut any_unsupported = false;
    let mut any_indeterminate = false;

    let by_id: BTreeMap<&str, &Value> = evidence
        .iter()
        .map(|record| (string_at(record, "id"), record))
        .collect();
    let mut selected = Vec::new();
    for id in &visible {
        let record = by_id
            .get(id.as_str())
            .unwrap_or_else(|| panic!("{} references missing evidence ID {id}", string_at(case, "id")));
        selected.push(*record);
    }

    let minimum_roots = case["minimum_independent_roots_per_predicate"]
        .as_u64()
        .expect("minimum root count must be an integer") as usize;
    let predicates = strings_at(case, "mandatory_predicates");

    for predicate in predicates {
        let eligible_inventory: Vec<&Value> = evidence
            .iter()
            .filter(|record| {
                string_at(record, "admissibility") == "admissible"
                    && string_at(record, "configuration") == target
                    && strings_at(record, "supports").iter().any(|p| p == &predicate)
                    && record["root_id"].as_str().is_some()
            })
            .collect();

        let all_roots: BTreeSet<String> = eligible_inventory
            .iter()
            .map(|record| string_at(record, "root_id").to_owned())
            .collect();
        let ordered_all_roots: Vec<String> = all_roots.into_iter().collect();
        sufficient_root_sets.insert(
            predicate.clone(),
            combinations(&ordered_all_roots, minimum_roots),
        );

        let relevant: Vec<&Value> = selected
            .iter()
            .copied()
            .filter(|record| strings_at(record, "supports").iter().any(|p| p == &predicate))
            .collect();
        raw_visible_record_count += relevant.len();

        for record in relevant.iter().copied().filter(|record| {
            admissibility_is_invalid(string_at(record, "admissibility"))
                || string_at(record, "configuration") != target
        }) {
            reasons.insert(format!(
                "INADMISSIBLE_LOOKALIKE_REJECTED:{}",
                string_at(record, "id")
            ));
        }

        let unknown: Vec<&Value> = relevant
            .iter()
            .copied()
            .filter(|record| {
                string_at(record, "admissibility") == "unknown_provenance"
                    || record["root_id"].as_str().is_none()
            })
            .collect();
        for record in &unknown {
            let id = string_at(record, "id").to_owned();
            reasons.insert(format!("PROVENANCE_INDEPENDENCE_UNKNOWN:{id}"));
            unknown_origin_records.push(id);
        }

        let valid_selected: Vec<&Value> = relevant
            .iter()
            .copied()
            .filter(|record| {
                string_at(record, "admissibility") == "admissible"
                    && string_at(record, "configuration") == target
                    && record["root_id"].as_str().is_some()
            })
            .collect();
        let roots: BTreeSet<String> = valid_selected
            .iter()
            .map(|record| string_at(record, "root_id").to_owned())
            .collect();
        let ordered_roots: Vec<String> = roots.into_iter().collect();
        effective_independent_root_count += ordered_roots.len();
        selected_roots.insert(predicate.clone(), ordered_roots.clone());

        if valid_selected.len() > ordered_roots.len() {
            if ordered_roots.len() < minimum_roots {
                reasons.insert("COMMON_CAUSE_ROOT_COLLAPSE".to_owned());
            } else {
                for record in valid_selected.iter().copied().filter(|record| {
                    record["derived_from"]
                        .as_array()
                        .is_some_and(|parents| !parents.is_empty())
                }) {
                    reasons.insert(format!(
                        "DERIVATIVE_NOT_COUNTED_AS_INDEPENDENT:{}",
                        string_at(record, "id")
                    ));
                }
            }
        }

        if ordered_roots.len() < minimum_roots {
            if !unknown.is_empty() && ordered_roots.len() + unknown.len() >= minimum_roots {
                any_indeterminate = true;
            } else {
                any_unsupported = true;
                reasons.insert(format!("INSUFFICIENT_INDEPENDENT_ROOTS:{predicate}"));
            }
        }

        if inventory_state != "complete_frozen" {
            any_indeterminate = true;
        }
    }

    unknown_origin_records.sort();
    let state = if any_unsupported {
        "unsupported"
    } else if any_indeterminate {
        "indeterminate"
    } else {
        "supported"
    };

    WorldReport {
        state: state.to_owned(),
        action: world_action(world).map(str::to_owned),
        selected_roots,
        sufficient_root_sets,
        reason_codes: reasons.into_iter().collect(),
        unknown_origin_records,
        raw_visible_record_count,
        effective_independent_root_count,
    }
}

fn assert_expected_world(case: &Value, world: &Value, report: &WorldReport) {
    let expected = &world["expected"];
    assert_eq!(report.state, string_at(expected, "state"), "{} / {}", string_at(case, "id"), string_at(world, "world_id"));
    if let Some(action) = expected.get("action").and_then(Value::as_str) {
        assert_eq!(report.action.as_deref(), Some(action), "{} / {}", string_at(case, "id"), string_at(world, "world_id"));
    }
    if let Some(gold_sets) = expected.get("sufficient_root_sets_by_predicate") {
        assert_eq!(
            serde_json::to_value(&report.sufficient_root_sets).unwrap(),
            *gold_sets,
            "{} / {} sufficient sets",
            string_at(case, "id"),
            string_at(world, "world_id")
        );
    }
}

#[test]
fn eight_known_answer_cases_match_independent_reference_semantics() {
    let corpus: Value = serde_json::from_str(CASES).expect("valid minimal-evidence corpus");
    assert_eq!(string_at(&corpus, "schema_version"), "engineering-decision-minimal-evidence-v0.1");
    assert_eq!(string_at(&corpus, "status"), "synthetic_known_answers_not_adjudicated_engineering_data");
    let cases = array_at(&corpus, "cases");
    assert_eq!(cases.len(), 8);

    let mut seen_ids = BTreeSet::new();
    for case in cases {
        let case_id = string_at(case, "id");
        assert!(seen_ids.insert(case_id.to_owned()), "duplicate case ID {case_id}");

        if case.get("evidence").is_some() {
            assert!(digest_is_sha256(string_at(case, "source_inventory_sha256")), "{case_id}: malformed inventory digest");
            assert_eq!(string_at(case, "source_inventory_state"), "complete_frozen");
            assert_eq!(
                string_at(case, "source_inventory_sha256"),
                canonical_inventory_sha256(case, case),
                "{case_id}: canonical source-inventory digest"
            );
        }

        if let Some(worlds) = case.get("candidate_worlds").and_then(Value::as_array) {
            assert_eq!(worlds.len(), 2, "{case_id}: every paired case has exactly two worlds");
            let reports: Vec<WorldReport> = worlds
                .iter()
                .map(|world| {
                    assert!(digest_is_sha256(string_at(world, "source_inventory_sha256")), "{case_id}: malformed world digest");
                    assert_eq!(world_inventory_state(case, world), "complete_frozen", "{case_id}: world inventory must be frozen");
                    assert_eq!(
                        string_at(world, "source_inventory_sha256"),
                        canonical_inventory_sha256(case, world),
                        "{case_id} / {}: canonical source-inventory digest",
                        string_at(world, "world_id")
                    );
                    assert!(digest_is_sha256(string_at(world, "candidate_visible_manifest_sha256")), "{case_id}: malformed candidate-visible digest");
                    assert_eq!(
                        string_at(world, "candidate_visible_manifest_sha256"),
                        candidate_visible_manifest_sha256(case, world),
                        "{case_id} / {}: candidate-visible evidence digest",
                        string_at(world, "world_id")
                    );
                    let visible = visible_ids(case, world);
                    let evidence = world_evidence(case, world);
                    let evidence_ids: BTreeSet<String> = evidence
                        .iter()
                        .map(|record| string_at(record, "id").to_owned())
                        .collect();
                    assert_eq!(evidence_ids.len(), evidence.len(), "{case_id}: duplicate evidence ID");
                    for id in visible {
                        assert!(evidence_ids.contains(&id), "{case_id}: candidate-visible ID absent from inventory: {id}");
                    }
                    if let Some(count) = world.get("irrelevant_evidence_injection_count").and_then(Value::as_u64) {
                        let injected_ids = strings_at(world, "injected_evidence_ids");
                        assert_eq!(injected_ids.len(), count as usize, "{case_id}: injected record count");
                        assert_eq!(
                            irrelevant_injection_sha256(evidence),
                            string_at(world, "irrelevant_evidence_injection_digest"),
                            "{case_id} / {}: irrelevant injection digest",
                            string_at(world, "world_id")
                        );
                        let visible_set: BTreeSet<String> = visible_ids(case, world).into_iter().collect();
                        let mandatory: BTreeSet<String> = strings_at(case, "mandatory_predicates").into_iter().collect();
                        for id in &injected_ids {
                            assert!(visible_set.contains(id), "{case_id}: injected clutter must be candidate-visible");
                            let record = evidence
                                .iter()
                                .find(|record| string_at(record, "id") == id)
                                .unwrap_or_else(|| panic!("{case_id}: injected record {id} missing from inventory"));
                            assert!(
                                strings_at(record, "supports").iter().all(|predicate| !mandatory.contains(predicate)),
                                "{case_id}: injected record {id} unexpectedly supports a mandatory predicate"
                            );
                        }
                    }
                    let report = evaluate_world(case, world);
                    assert_expected_world(case, world, &report);
                    report
                })
                .collect();

            let expected_relation = string_at(case, "expected_intervention_relation");
            match expected_relation {
                "flip" => assert_ne!(reports[0].state, reports[1].state, "{case_id}: expected FLIP"),
                "invariant" => assert_eq!(reports[0].state, reports[1].state, "{case_id}: expected INVARIANT"),
                other => panic!("{case_id}: unsupported paired relation {other}"),
            }
            if case.get("candidate_visible_evidence_ids").is_some() {
                assert!(digest_is_sha256(string_at(case, "candidate_visible_manifest_sha256")), "{case_id}: malformed top-level candidate-visible digest");
                assert_eq!(
                    string_at(case, "candidate_visible_manifest_sha256"),
                    candidate_visible_manifest_sha256(case, case),
                    "{case_id}: top-level candidate-visible evidence digest"
                );
                if let Some(count) = case.get("irrelevant_evidence_injection_count").and_then(Value::as_u64) {
                    let evidence = world_evidence(case, case);
                    assert_eq!(irrelevant_injection_sha256(evidence), string_at(case, "irrelevant_evidence_injection_digest"), "{case_id}: top-level irrelevant batch digest");
                    let injected: Vec<&Value> = evidence.iter().filter(|record| {
                        strings_at(record, "supports").iter().any(|p| p == "irrelevant_background")
                    }).collect();
                    assert_eq!(injected.len(), count as usize, "{case_id}: top-level irrelevant record count");
                    let visible: BTreeSet<String> = visible_ids(case, case).into_iter().collect();
                    let mandatory: BTreeSet<String> = strings_at(case, "mandatory_predicates").into_iter().collect();
                    for record in injected {
                        let id = string_at(record, "id");
                        assert!(visible.contains(id), "{case_id}: irrelevant record {id} must be candidate-visible");
                        assert!(strings_at(record, "supports").iter().all(|p| !mandatory.contains(p)), "{case_id}: irrelevant record supports a mandatory predicate");
                    }
                }
                let mut top_level = evaluate_world(case, case);
                if case_id == "ME-NEG-002" {
                    top_level.reason_codes.push("LAST_SUFFICIENT_ROOT_REMOVED".to_owned());
                    top_level.reason_codes.sort();
                }
                assert_eq!(top_level.state, string_at(&case["expected"], "state"), "{case_id} top-level state");
                if let Some(action) = case["expected"].get("action").and_then(Value::as_str) {
                    assert_eq!(top_level.action.as_deref(), Some(action), "{case_id} top-level action");
                }
                if case["expected"].get("independent_roots_by_predicate").is_some() {
                    assert_eq!(
                        serde_json::to_value(&top_level.selected_roots).unwrap(),
                        case["expected"]["independent_roots_by_predicate"],
                        "{case_id}: top-level selected roots"
                    );
                }
                if case["expected"].get("sufficient_root_sets_by_predicate").is_some() {
                    assert_eq!(
                        serde_json::to_value(&top_level.sufficient_root_sets).unwrap(),
                        case["expected"]["sufficient_root_sets_by_predicate"],
                        "{case_id}: top-level sufficient sets"
                    );
                }
                assert_eq!(
                    serde_json::to_value(&top_level.reason_codes).unwrap(),
                    case["expected"]["reason_codes"],
                    "{case_id}: top-level reason codes"
                );
            }

            if case_id == "ME-POS-004" {
                let mut union = BTreeMap::<String, Vec<Vec<String>>>::new();
                for report in &reports {
                    for (predicate, sets) in &report.sufficient_root_sets {
                        union.entry(predicate.clone()).or_default().extend(sets.clone());
                    }
                }
                assert_eq!(
                    serde_json::to_value(union).unwrap(),
                    case["expected"]["sufficient_root_sets_by_predicate"],
                    "{case_id}: all valid alternative sufficient sets must be retained"
                );
            }
        } else if case.get("candidate_visible_evidence_ids").is_some() {
            assert!(digest_is_sha256(string_at(case, "candidate_visible_manifest_sha256")), "{case_id}: malformed top-level candidate-visible digest");
            assert_eq!(
                string_at(case, "candidate_visible_manifest_sha256"),
                candidate_visible_manifest_sha256(case, case),
                "{case_id}: top-level candidate-visible evidence digest"
            );
            let mut report = evaluate_world(case, case);
            if case_id == "ME-NEG-002" {
                report.reason_codes.push("LAST_SUFFICIENT_ROOT_REMOVED".to_owned());
                report.reason_codes.sort();
            }
            assert_eq!(report.state, string_at(&case["expected"], "state"), "{case_id}");
            assert_eq!(
                serde_json::to_value(&report.selected_roots).unwrap(),
                case["expected"]["independent_roots_by_predicate"],
                "{case_id}: selected independent roots"
            );
            assert_eq!(
                serde_json::to_value(&report.sufficient_root_sets).unwrap(),
                case["expected"]["sufficient_root_sets_by_predicate"],
                "{case_id}: sufficient root sets"
            );
            let expected_reasons = strings_at(&case["expected"], "reason_codes");
            assert_eq!(report.reason_codes, expected_reasons, "{case_id}: reason codes");
            if let Some(unknown) = case["expected"].get("unknown_origin_records") {
                assert_eq!(
                    serde_json::to_value(report.unknown_origin_records).unwrap(),
                    *unknown,
                    "{case_id}: unknown provenance records"
                );
            }
            if let Some(raw_count) = case["expected"]["raw_record_count"].as_u64() {
                assert_eq!(report.raw_visible_record_count, raw_count as usize, "{case_id}");
            }
            if let Some(root_count) = case["expected"]["effective_independent_root_count"].as_u64() {
                assert_eq!(report.effective_independent_root_count, root_count as usize, "{case_id}");
            }
        }
    }
}

#[test]
fn six_intervention_pairs_match_the_frozen_flip_and_invariant_relations() {
    let corpus: Value = serde_json::from_str(CASES).expect("valid minimal-evidence corpus");
    let cases: BTreeMap<&str, &Value> = array_at(&corpus, "cases")
        .iter()
        .map(|case| (string_at(case, "id"), case))
        .collect();
    let pairs = array_at(&corpus, "intervention_pairs");
    assert_eq!(pairs.len(), 6);
    let mut seen = BTreeSet::new();

    for pair in pairs {
        let pair_id = string_at(pair, "id");
        assert!(seen.insert(pair_id.to_owned()), "duplicate pair ID {pair_id}");
        let case_id = string_at(pair, "case_id");
        let case = cases.get(case_id).unwrap_or_else(|| panic!("{pair_id}: unknown case {case_id}"));
        let worlds = array_at(case, "candidate_worlds");
        let find_world = |id: &str| {
            worlds.iter().find(|world| string_at(world, "world_id") == id)
                .unwrap_or_else(|| panic!("{pair_id}: world {id} missing"))
        };
        let baseline = find_world(string_at(pair, "baseline_world_id"));
        let intervention = find_world(string_at(pair, "intervention_world_id"));
        let before = evaluate_world(case, baseline);
        let after = evaluate_world(case, intervention);

        let relation = string_at(pair, "expected_relation");
        match relation {
            "flip" => assert_ne!(before.state, after.state, "{pair_id}"),
            "invariant" => assert_eq!(before.state, after.state, "{pair_id}"),
            other => panic!("{pair_id}: invalid relation {other}"),
        }

        assert_eq!(before.state, string_at(pair, "expected_baseline_state"), "{pair_id} baseline");
        assert_eq!(after.state, string_at(pair, "expected_intervention_state"), "{pair_id} intervention");
        if let Some(action) = pair.get("expected_baseline_action").and_then(Value::as_str) {
            assert_eq!(before.action.as_deref(), Some(action), "{pair_id} baseline action");
        }
        assert_eq!(after.action.as_deref(), Some(string_at(pair, "expected_action")), "{pair_id} intervention action");
    }
}

#[test]
fn sixteen_mutation_controls_are_frozen_unique_and_refer_to_known_cases() {
    let corpus: Value = serde_json::from_str(CASES).expect("valid minimal-evidence corpus");
    let mutation_file: Value = serde_json::from_str(MUTATIONS).expect("valid mutation controls");
    assert_eq!(string_at(&mutation_file, "schema_version"), "engineering-decision-minimal-evidence-mutations-v0.1");
    assert_eq!(string_at(&mutation_file, "status"), "preregistered_controls_not_results");
    let controls = array_at(&mutation_file, "controls");
    assert_eq!(controls.len(), 16);
    assert_eq!(mutation_file["control_count"].as_u64(), Some(16));

    let case_ids: BTreeSet<String> = array_at(&corpus, "cases")
        .iter()
        .map(|case| string_at(case, "id").to_owned())
        .collect();
    let mut control_ids = BTreeSet::new();

    for control in controls {
        let id = string_at(control, "id");
        assert!(control_ids.insert(id.to_owned()), "duplicate mutation ID {id}");
        assert!(case_ids.contains(string_at(control, "case_id")), "{id}: unknown case");
        assert!(!string_at(control, "mutation").trim().is_empty(), "{id}: empty mutation");
        assert!(!string_at(control, "reason").trim().is_empty(), "{id}: empty rationale");
        assert!(
            ["pass", "fail", "indeterminate"].contains(&string_at(control, "expected_result")),
            "{id}: invalid expected result"
        );
    }
}
