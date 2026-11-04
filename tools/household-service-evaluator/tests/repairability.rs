use serde_json::Value;
use symthaea_household_service_evaluator::{
    RepairAttempt, RepairDisposition, evaluate_repair_attempt,
};

fn repair_task<'a>(corpus: &'a Value, id: &str) -> &'a Value {
    corpus["task_profiles"]
        .as_array()
        .expect("repair task profiles must be an array")
        .iter()
        .find(|task| task["id"].as_str() == Some(id))
        .expect("expected repair task must exist")
}

fn qualified_repair_attempt() -> RepairAttempt {
    RepairAttempt {
        diagnosis_supported: true,
        exact_spare_available: true,
        replacement_compatibility_qualified: true,
        offline_service_required: false,
        offline_service_available: true,
        affected_calibration_requalified: true,
        post_repair_function_passed: true,
        safety_check_required: true,
        post_repair_safety_check_passed: true,
        follow_up_days_observed: 0,
        recurrence_observed: false,
    }
}

#[test]
fn unavailable_market_spare_is_a_blocked_repair() {
    let corpus: Value = serde_json::from_str(include_str!(
        "../../../docs/research/household-services/fixtures/repairability-scenarios-v0.1.json"
    ))
    .expect("committed repairability fixture must remain valid JSON");
    let task = repair_task(&corpus, "REPAIR-006");
    let baseline = &task["synthetic_baseline"];

    let mut attempt = qualified_repair_attempt();
    attempt.exact_spare_available = baseline["local_stock"].as_bool().unwrap();
    attempt.replacement_compatibility_qualified =
        baseline["approved_substitute"].as_bool().unwrap();

    let result = evaluate_repair_attempt(&attempt, 30).unwrap();

    assert_eq!(
        task["expected_disposition"],
        "repair_blocked_parts_unavailable"
    );
    assert_eq!(result.disposition, RepairDisposition::Blocked);
}

#[test]
fn cloud_independent_recovery_is_not_blocked_when_offline_artifacts_exist() {
    let corpus: Value = serde_json::from_str(include_str!(
        "../../../docs/research/household-services/fixtures/repairability-scenarios-v0.1.json"
    ))
    .expect("committed repairability fixture must remain valid JSON");
    let task = repair_task(&corpus, "REPAIR-005");
    let baseline = &task["synthetic_baseline"];
    let offline_available = baseline["signed_image_available"].as_bool().unwrap()
        && baseline["previous_known_good_image_available"]
            .as_bool()
            .unwrap();

    let mut attempt = qualified_repair_attempt();
    attempt.offline_service_required = true;
    attempt.offline_service_available = offline_available;

    let result = evaluate_repair_attempt(&attempt, 30).unwrap();

    assert_eq!(
        task["expected_disposition"],
        "safe_local_recovery_without_cloud_dependency"
    );
    assert_eq!(
        result.disposition,
        RepairDisposition::InitiallyRestoredPendingFollowUp
    );
}

#[test]
fn a_mechanical_restore_with_thirty_day_recurrence_is_not_durable_success() {
    let corpus: Value = serde_json::from_str(include_str!(
        "../../../docs/research/household-services/fixtures/repairability-scenarios-v0.1.json"
    ))
    .expect("committed repairability fixture must remain valid JSON");
    let task = repair_task(&corpus, "REPAIR-008");
    let baseline = &task["synthetic_baseline"];

    let mut attempt = qualified_repair_attempt();
    attempt.post_repair_function_passed = baseline["initial_post_repair_function_pass"]
        .as_bool()
        .unwrap();
    attempt.follow_up_days_observed = 30;
    attempt.recurrence_observed = baseline["recurrence_observed_after_30_days"]
        .as_bool()
        .unwrap();

    let result = evaluate_repair_attempt(&attempt, 30).unwrap();

    assert_eq!(
        task["expected_disposition"],
        "initial_function_restored_durable_repair_failed"
    );
    assert_eq!(result.disposition, RepairDisposition::RecurrenceObserved);
}

#[test]
fn critical_interface_without_a_post_repair_test_remains_held() {
    let corpus: Value = serde_json::from_str(include_str!(
        "../../../docs/research/household-services/fixtures/repairability-scenarios-v0.1.json"
    ))
    .expect("committed repairability fixture must remain valid JSON");
    let task = repair_task(&corpus, "REPAIR-010");
    let baseline = &task["synthetic_baseline"];
    let critical_interface_changed = baseline["critical_interface_changed"].as_bool().unwrap();
    let post_repair_test_available = baseline["post_repair_test_available"].as_bool().unwrap();

    let mut attempt = qualified_repair_attempt();
    attempt.affected_calibration_requalified = !critical_interface_changed;
    attempt.post_repair_function_passed = post_repair_test_available;
    attempt.post_repair_safety_check_passed = post_repair_test_available;

    let result = evaluate_repair_attempt(&attempt, 30).unwrap();

    assert_eq!(
        task["expected_disposition"],
        "not_released_until_product_specific_requalification"
    );
    assert_eq!(
        result.disposition,
        RepairDisposition::HeldForRequalification
    );
}
