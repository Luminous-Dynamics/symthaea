#[path = "../src/language/inference_contract.rs"]
mod inference_contract;
#[path = "../src/language/inference_permit.rs"]
mod inference_permit;
#[path = "../src/language/inference_binding.rs"]
mod inference_binding;
#[path = "../src/language/inference_receipt.rs"]
mod inference_receipt;
#[path = "../src/language/inference_resource_guard.rs"]
mod inference_resource_guard;

use inference_receipt::{InferenceRateLimitEvidence, InferenceWireEvidence};
use inference_resource_guard::{
    InferenceResourceAuthority, InferenceResourceGuard, InferenceResourceGuardConfig,
    InferenceResourceGuardError,
};

fn guard_with_id(
    requests: Option<u64>,
    tokens: Option<u64>,
    instance_byte: u8,
) -> InferenceResourceGuard {
    InferenceResourceGuard::new(
        InferenceResourceAuthority::new(1, 1, requests, tokens),
        InferenceResourceGuardConfig::default(),
        [instance_byte; 32],
    )
    .unwrap()
}

fn guard(requests: Option<u64>, tokens: Option<u64>) -> InferenceResourceGuard {
    guard_with_id(requests, tokens, 1)
}

fn evidence(
    remaining_requests: Option<u64>,
    remaining_tokens: Option<u64>,
    retry_after_millis: Option<u64>,
    request_reset_after_millis: Option<u64>,
) -> InferenceWireEvidence {
    InferenceWireEvidence::new(
        None,
        None,
        Some("reasoner-v1"),
        None,
        Some("stop"),
        None,
        InferenceRateLimitEvidence {
            retry_after_millis,
            request_limit: None,
            request_remaining: remaining_requests,
            request_reset_after_millis,
            token_limit: None,
            token_remaining: remaining_tokens,
            token_reset_after_millis: None,
        },
        5,
        false,
        false,
    )
    .unwrap()
}

#[test]
fn zero_guard_instance_identity_is_rejected() {
    let err = InferenceResourceGuard::new(
        InferenceResourceAuthority::new(1, 1, Some(1), Some(1)),
        InferenceResourceGuardConfig::default(),
        [0; 32],
    )
    .unwrap_err();
    assert_eq!(err, InferenceResourceGuardError::InvalidInstanceId);
}

#[test]
fn reservation_prevents_concurrent_request_oversubscription() {
    let mut guard = guard(Some(1), Some(1_000));
    let first = guard.reserve(100, 0).unwrap();
    assert_eq!(guard.effective_remaining_requests(), Some(0));
    assert_eq!(guard.active_reservation_count(), 1);

    let second = guard.reserve(100, 0).unwrap_err();
    assert_eq!(second, InferenceResourceGuardError::RequestBudgetExhausted);

    guard
        .settle_success(first, &evidence(Some(0), Some(900), None, None))
        .unwrap();
    assert_eq!(guard.active_reservation_count(), 0);
}

#[test]
fn reservation_cannot_settle_a_different_guard_with_same_generation() {
    let mut first_guard = guard_with_id(Some(2), Some(1_000), 1);
    let mut second_guard = guard_with_id(Some(2), Some(1_000), 2);
    let first_reservation = first_guard.reserve(100, 0).unwrap();
    let second_reservation = second_guard.reserve(100, 0).unwrap();

    let err = second_guard
        .settle_success(first_reservation, &evidence(None, None, None, None))
        .unwrap_err();
    assert_eq!(err, InferenceResourceGuardError::ReservationGuardMismatch);
    assert_eq!(second_guard.active_reservation_count(), 1);

    second_guard
        .settle_success(second_reservation, &evidence(None, None, None, None))
        .unwrap();
    assert_eq!(second_guard.active_reservation_count(), 0);
}

#[test]
fn abandoned_reservation_stays_charged_but_stops_blocking_epoch_change() {
    let mut guard = guard(Some(3), Some(1_000));
    let reservation = guard.reserve(200, 0).unwrap();
    assert_eq!(guard.effective_remaining_requests(), Some(2));
    assert_eq!(guard.effective_remaining_tokens(), Some(800));
    assert_eq!(guard.active_reservation_count(), 1);

    guard.abandon(reservation).unwrap();
    assert_eq!(guard.active_reservation_count(), 0);
    assert_eq!(guard.effective_remaining_requests(), Some(2));
    assert_eq!(guard.effective_remaining_tokens(), Some(800));

    guard
        .advance_quota_epoch(2, Some(5), Some(2_000))
        .unwrap();
    assert_eq!(guard.effective_remaining_requests(), Some(5));
}

#[test]
fn provider_observations_can_tighten_but_not_replenish_same_epoch() {
    let mut guard = guard(Some(100), Some(100_000));
    let first = guard.reserve(1_000, 0).unwrap();
    guard
        .settle_success(first, &evidence(Some(10), Some(50_000), None, Some(60_000)))
        .unwrap();
    assert_eq!(guard.effective_remaining_requests(), Some(10));
    assert_eq!(guard.effective_remaining_tokens(), Some(50_000));

    let second = guard.reserve(1_000, 1).unwrap();
    assert_eq!(guard.effective_remaining_requests(), Some(9));
    assert_eq!(guard.effective_remaining_tokens(), Some(49_000));

    guard
        .settle_success(
            second,
            &evidence(Some(99), Some(99_000), None, Some(30_000)),
        )
        .unwrap();
    assert_eq!(guard.effective_remaining_requests(), Some(9));
    assert_eq!(guard.effective_remaining_tokens(), Some(49_000));
    assert_eq!(guard.last_observed_reset_after_millis(), Some(30_000));
}

#[test]
fn provider_retry_after_opens_bounded_cooldown_without_refilling_quota() {
    let mut config = InferenceResourceGuardConfig::default();
    config.max_backoff_millis = 10_000;
    let mut guard = InferenceResourceGuard::new(
        InferenceResourceAuthority::new(1, 1, Some(5), Some(10_000)),
        config,
        [2; 32],
    )
    .unwrap();

    let reservation = guard.reserve(100, 1_000).unwrap();
    guard
        .settle_rate_limited(
            reservation,
            &evidence(Some(0), Some(9_900), Some(60_000), Some(2_000)),
            1_100,
        )
        .unwrap();

    assert_eq!(guard.cooldown_until_millis(), Some(11_100));
    assert_eq!(guard.effective_remaining_requests(), Some(0));
    assert_eq!(
        guard.reserve(1, 2_000).unwrap_err(),
        InferenceResourceGuardError::CoolingDown
    );
    assert_eq!(
        guard.reserve(1, 12_000).unwrap_err(),
        InferenceResourceGuardError::RequestBudgetExhausted
    );
}

#[test]
fn only_explicit_quota_epoch_advance_replenishes_budget() {
    let mut guard = guard(Some(1), Some(100));
    let reservation = guard.reserve(100, 0).unwrap();
    guard
        .settle_success(reservation, &evidence(Some(0), Some(0), None, Some(1)))
        .unwrap();
    assert_eq!(guard.effective_remaining_requests(), Some(0));
    assert_eq!(guard.effective_remaining_tokens(), Some(0));

    assert_eq!(
        guard.advance_quota_epoch(1, Some(10), Some(1_000)).unwrap_err(),
        InferenceResourceGuardError::QuotaEpochNotAdvanced
    );
    guard
        .advance_quota_epoch(2, Some(10), Some(1_000))
        .unwrap();
    assert_eq!(guard.effective_remaining_requests(), Some(10));
    assert_eq!(guard.effective_remaining_tokens(), Some(1_000));
    assert_eq!(guard.cooldown_until_millis(), None);
}

#[test]
fn credential_rejection_requires_new_credential_epoch_and_does_not_refill_budget() {
    let mut guard = guard(Some(3), Some(1_000));
    let reservation = guard.reserve(100, 0).unwrap();
    guard.settle_auth_rejected(reservation).unwrap();
    assert!(guard.is_credential_blocked());
    assert_eq!(guard.effective_remaining_requests(), Some(2));
    assert_eq!(
        guard.reserve(100, 1).unwrap_err(),
        InferenceResourceGuardError::CredentialBlocked
    );

    assert_eq!(
        guard.advance_credential_epoch(1).unwrap_err(),
        InferenceResourceGuardError::CredentialEpochNotAdvanced
    );
    guard.advance_credential_epoch(2).unwrap();
    assert!(!guard.is_credential_blocked());
    assert_eq!(guard.effective_remaining_requests(), Some(2));
}

#[test]
fn credential_epoch_change_is_blocked_while_old_epoch_reservation_is_active() {
    let mut guard = guard(Some(3), Some(1_000));
    let reservation = guard.reserve(100, 0).unwrap();
    assert_eq!(
        guard.advance_credential_epoch(2).unwrap_err(),
        InferenceResourceGuardError::ActiveReservationsPresent
    );
    guard.abandon(reservation).unwrap();
    guard.advance_credential_epoch(2).unwrap();
}

#[test]
fn repeated_failures_open_exponential_but_bounded_circuit() {
    let mut config = InferenceResourceGuardConfig::default();
    config.failure_threshold = 2;
    config.failure_base_backoff_millis = 100;
    config.max_backoff_millis = 350;
    let mut guard = InferenceResourceGuard::new(
        InferenceResourceAuthority::new(1, 1, Some(10), Some(10_000)),
        config,
        [3; 32],
    )
    .unwrap();

    let first = guard.reserve(10, 0).unwrap();
    guard.settle_failure(first, 0).unwrap();
    assert_eq!(guard.cooldown_until_millis(), None);

    let second = guard.reserve(10, 0).unwrap();
    guard.settle_failure(second, 1_000).unwrap();
    assert_eq!(guard.cooldown_until_millis(), Some(1_100));

    let third = guard.reserve(10, 1_101).unwrap();
    guard.settle_failure(third, 1_101).unwrap();
    assert_eq!(guard.cooldown_until_millis(), Some(1_301));

    let fourth = guard.reserve(10, 1_302).unwrap();
    guard.settle_failure(fourth, 1_302).unwrap();
    assert_eq!(guard.cooldown_until_millis(), Some(1_652));
}

#[test]
fn active_reservation_blocks_quota_epoch_replacement() {
    let mut guard = guard(Some(5), Some(1_000));
    let reservation = guard.reserve(100, 0).unwrap();
    assert_eq!(
        guard.advance_quota_epoch(2, Some(10), Some(2_000)).unwrap_err(),
        InferenceResourceGuardError::ActiveReservationsPresent
    );
    guard.abandon(reservation).unwrap();
}

#[test]
fn no_automatic_refund_from_provider_usage_claims() {
    let mut guard = guard(Some(5), Some(1_000));
    let reservation = guard.reserve(500, 0).unwrap();
    let evidence = InferenceWireEvidence::new(
        None,
        None,
        Some("reasoner-v1"),
        None,
        Some("stop"),
        Some(inference_receipt::InferenceTokenUsageEvidence {
            prompt_tokens: Some(10),
            completion_tokens: Some(10),
            total_tokens: Some(20),
        }),
        InferenceRateLimitEvidence::default(),
        1,
        false,
        false,
    )
    .unwrap();
    guard.settle_success(reservation, &evidence).unwrap();
    assert_eq!(guard.effective_remaining_tokens(), Some(500));
}
