use serde_json::Value;
use symthaea_household_service_evaluator::{
    evaluate_repair_attempt, RepairAttempt, RepairDisposition,
};

fn repair_task<'a>(corpus: &'a Value, id: &str) -> &'a Value {
    corpus["task_profiles"]
        .as_array()
        .expect("repair task profiles must be an array")
        .iter()
        .find(|task| task["id"].as_str() == Some(id))
        .expect("expected repair task must exist")
}

fn repair_attempt(
    exact_spare_available: bool,
    replacement_compatibility_qualified: bool,
    offline_service_required: bool,
    offline_service_available: bool,
    affected_calibration_requalified: bool,
    post_repair_function_passed: bool,
    post_repair_safety_check_passed: bool,
    follow_up_days_observed: u32,
    recurrence_observed: bool,
) -> RepairAttempt {
    RepairAttempt {
        diagnosis_supported: true,
        exact_spare_available,
        replacement_compatibility_qualified,
        offline_service_required,
        offline_service_available,
        affected_calibration_requalified,
        post_repair_function_passed,
        safety_check_required: true,
        post_repair_safety_check_passed,
        follow_up_days_observed,
        recurrence_observed,
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

    let result = evaluate_repair_attempt(
        &repair_attempt(
            baseline["local_stock"].as_bool().unwrap(),
            baseline["approved_substitute"].as_bool().unwrap(),
            false,
            true,
            true,
            true,
            true,
            0,
            false,
        ),
        30,
    )
    .unwrap();

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

    let result = evaluate_repair_attempt(
        &repair_attempt(true, true, true, offline_available, true, true, true, 0, false),
        30,
    )
    .unwrap();

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

    let result = evaluate_repair_attempt(
        &repair_attempt(
            true,
            true,
            false,
            true,
            true,
            baseline["initial_post_repair_function_pass"]
                .as_bool()
                .unwrap(),
            true,
            30,
            baseline["recurrence_observed_after_30_days"]
                .as_bool()
                .unwrap(),
        ),
        30,
    )
    .unwrap();

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
    let critical_interface_changed = baseline["critical_interface_changed"]
        .as_bool()
        .unwrap();
    let post_repair_test_available = baseline["post_repair_test_available"]
        .as_bool()
        .unwrap();

    let result = evaluate_repair_attempt(
        &repair_attempt(
            true,
            true,
            false,
            true,
            !critical_interface_changed,
            post_repair_test_available,
            post_repair_test_available,
            0,
            false,
        ),
        30,
    )
    .unwrap();

    assert_eq!(
        task["expected_disposition"],
        "not_released_until_product_specific_requalification"
    );
    assert_eq!(
        result.disposition,
        RepairDisposition::HeldForRequalification
    );
}
