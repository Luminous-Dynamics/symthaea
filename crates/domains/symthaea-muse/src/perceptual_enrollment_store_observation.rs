// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1EILR-F: read-only observation of the validated current allocator state.
//!
//! The durable allocator intentionally keeps `inspect_current` as an exact-head
//! assertion. Crash recovery, however, begins from the independently confirmed
//! predecessor and may legitimately find exactly one newer durable allocation.
//! This module discovers that current head without weakening allocator mutation
//! CAS semantics or adding a second filesystem reader.
//!
//! The recovery observer performs at most two fully validated `inspect_current`
//! calls:
//!
//! ```text
//! confirmed head still current -> return that validated state
//! confirmed head is stale      -> use the observed head from the mismatch
//!                                 for exactly one re-read
//! second mismatch              -> fail closed
//! ```
//!
//! P1EILR-C remains responsible for deciding whether the returned state is
//! synchronized, exactly one allocation ahead, rolled back, or impossibly far
//! ahead. This module only observes durable truth; it grants no witness or
//! scored-collection authority.

use crate::evidence_digest::{
    perceptual_enrollment_store::{
        DurableEnrollmentAllocationErrorV1, DurableEnrollmentAllocationStateV1,
        DurableEnrollmentAllocationStoreV1,
    },
    perceptual_enrollment_lifecycle::FrozenPerceptualEnrollmentPolicyV1,
    perceptual_participant_identity::{
        FrozenParticipantIdentityBoundaryPolicyV1,
        FrozenPerceptualParticipantTokenGenerationReceiptV1,
    },
    perceptual_participant_schedule::{
        PerceptualCohortSlotsV1, PerceptualParticipantScheduleBookV1,
    },
    perceptual_stimulus_pack::{
        FrozenC6fRenderSubjectBindingV1, FrozenPerceptualStimulusPackV1,
    },
    perceptual_study_protocol::FrozenPerceptualStudyProtocolV1,
};

/// Observe the allocator's fully validated current durable state starting from
/// one independently confirmed predecessor head.
///
/// This function does not mutate allocator state and does not relax
/// `allocate_next` or exact-head `inspect_current` semantics. If the allocator
/// advances again between the stale-head observation and the exact re-read, the
/// second mismatch is returned rather than chased indefinitely.
#[allow(clippy::too_many_arguments)]
pub fn inspect_validated_current_from_confirmed_head(
    store: &DurableEnrollmentAllocationStoreV1,
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
    confirmed_ledger_sha256: &str,
) -> Result<DurableEnrollmentAllocationStateV1, DurableEnrollmentAllocationErrorV1> {
    observe_current_once(confirmed_ledger_sha256, |expected| {
        store.inspect_current(
            protocol,
            stimulus_pack,
            render_binding,
            cohort,
            token_receipt,
            identity_policy,
            schedule,
            enrollment_policy,
            expected,
        )
    })
}

fn observe_current_once<F>(
    confirmed_ledger_sha256: &str,
    mut inspect: F,
) -> Result<DurableEnrollmentAllocationStateV1, DurableEnrollmentAllocationErrorV1>
where
    F: FnMut(
        &str,
    ) -> Result<DurableEnrollmentAllocationStateV1, DurableEnrollmentAllocationErrorV1>,
{
    match inspect(confirmed_ledger_sha256) {
        Ok(state) => Ok(state),
        Err(DurableEnrollmentAllocationErrorV1::ExpectedLedgerHeadMismatch {
            found, ..
        }) => inspect(&found),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::perceptual_enrollment_lifecycle::FrozenPerceptualEnrollmentAllocationLedgerV1;
    use crate::evidence_digest::perceptual_enrollment_store::DURABLE_ENROLLMENT_ALLOCATION_STATE_VERSION;

    fn state(ledger_sha256: &str) -> DurableEnrollmentAllocationStateV1 {
        DurableEnrollmentAllocationStateV1 {
            state_version: DURABLE_ENROLLMENT_ALLOCATION_STATE_VERSION.into(),
            enrollment_policy_sha256: "a".repeat(64),
            token_generation_receipt_sha256: "b".repeat(64),
            participant_schedule_sha256: "c".repeat(64),
            eligibility_gates: Vec::new(),
            ledger: FrozenPerceptualEnrollmentAllocationLedgerV1 {
                ledger_version: "test".into(),
                enrollment_policy_sha256: "a".repeat(64),
                token_generation_receipt_sha256: "b".repeat(64),
                participant_schedule_sha256: "c".repeat(64),
                allocations: Vec::new(),
                final_allocation_head_sha256: "0".repeat(64),
                ledger_sha256: ledger_sha256.into(),
            },
            state_sha256: "d".repeat(64),
        }
    }

    #[test]
    fn confirmed_current_head_needs_one_read() {
        let head = "1".repeat(64);
        let mut calls = 0usize;
        let observed = observe_current_once(&head, |expected| {
            calls += 1;
            assert_eq!(expected, head);
            Ok(state(&head))
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(observed.ledger.ledger_sha256, head);
    }

    #[test]
    fn one_advanced_head_is_revalidated_exactly_once() {
        let confirmed = "1".repeat(64);
        let current = "2".repeat(64);
        let mut calls = 0usize;
        let observed = observe_current_once(&confirmed, |expected| {
            calls += 1;
            if calls == 1 {
                assert_eq!(expected, confirmed);
                Err(DurableEnrollmentAllocationErrorV1::ExpectedLedgerHeadMismatch {
                    expected: confirmed.clone(),
                    found: current.clone(),
                })
            } else {
                assert_eq!(expected, current);
                Ok(state(&current))
            }
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(observed.ledger.ledger_sha256, current);
    }

    #[test]
    fn allocator_moving_again_during_observation_fails_closed() {
        let confirmed = "1".repeat(64);
        let first_observed = "2".repeat(64);
        let second_observed = "3".repeat(64);
        let mut calls = 0usize;
        let error = observe_current_once(&confirmed, |expected| {
            calls += 1;
            if calls == 1 {
                Err(DurableEnrollmentAllocationErrorV1::ExpectedLedgerHeadMismatch {
                    expected: expected.into(),
                    found: first_observed.clone(),
                })
            } else {
                assert_eq!(expected, first_observed);
                Err(DurableEnrollmentAllocationErrorV1::ExpectedLedgerHeadMismatch {
                    expected: expected.into(),
                    found: second_observed.clone(),
                })
            }
        })
        .unwrap_err();
        assert_eq!(calls, 2);
        assert!(matches!(
            error,
            DurableEnrollmentAllocationErrorV1::ExpectedLedgerHeadMismatch { ref found, .. }
                if found == &second_observed
        ));
    }

    #[test]
    fn non_head_validation_error_is_not_retried() {
        let confirmed = "1".repeat(64);
        let mut calls = 0usize;
        let error = observe_current_once(&confirmed, |_| {
            calls += 1;
            Err(DurableEnrollmentAllocationErrorV1::StateDigestMismatch)
        })
        .unwrap_err();
        assert_eq!(calls, 1);
        assert!(matches!(
            error,
            DurableEnrollmentAllocationErrorV1::StateDigestMismatch
        ));
    }
}
