use proptest::prelude::*;
use symthaea_ai_assurance::{
    EffectAdmissionCommitment, EffectEntryDomain, EffectEntryError, EffectEntryPermit,
    EffectEntryTicket,
};

const SLOTS: usize = 4;

#[derive(Clone, Copy)]
struct TicketModel {
    epoch: u64,
    commitment: EffectAdmissionCommitment,
}

fn commitment(tag: u8) -> EffectAdmissionCommitment {
    EffectAdmissionCommitment::new(
        [tag; 32],
        [tag.wrapping_add(1); 32],
        [tag.wrapping_add(2); 32],
    )
}

fn guaranteed_different(commitment: EffectAdmissionCommitment) -> EffectAdmissionCommitment {
    let mut authority = commitment.authority_snapshot_digest();
    authority[0] ^= 1;
    EffectAdmissionCommitment::new(
        commitment.action_binding(),
        authority,
        commitment.adapter_semantics_digest(),
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn public_effect_entry_state_machine_matches_model(
        operations in prop::collection::vec((0_u8..6, 0_usize..SLOTS, any::<u8>(), any::<bool>()), 1..140)
    ) {
        let domain = EffectEntryDomain::new();
        let mut tickets: Vec<Option<EffectEntryTicket>> =
            (0..SLOTS).map(|_| None).collect();
        let mut ticket_models: Vec<Option<TicketModel>> = vec![None; SLOTS];
        let mut permits: Vec<Option<EffectEntryPermit>> =
            (0..SLOTS).map(|_| None).collect();
        let mut permit_sequences: Vec<Option<u64>> = vec![None; SLOTS];

        let mut model_epoch = 0_u64;
        let mut model_sequence = 0_u64;
        let mut model_open = true;
        let mut model_outstanding = 0_u64;

        for (operation, slot, tag, use_correct_commitment) in operations {
            match operation {
                0 => {
                    let commitment = commitment(tag);
                    let result = domain.issue_ticket(commitment);
                    if model_open {
                        let ticket = result.expect("running model must issue ticket");
                        prop_assert_eq!(ticket.commitment(), commitment);
                        tickets[slot] = Some(ticket);
                        ticket_models[slot] = Some(TicketModel {
                            epoch: model_epoch,
                            commitment,
                        });
                    } else {
                        prop_assert!(matches!(
                            result,
                            Err(EffectEntryError::AdmissionStopped { .. })
                        ));
                    }
                }
                1 => {
                    let was_open = model_open;
                    let receipt = domain.revoke_all().unwrap();
                    model_epoch += 1;
                    model_sequence += 1;
                    model_open = false;
                    prop_assert_eq!(receipt.current_epoch().get(), model_epoch);
                    prop_assert_eq!(receipt.revocation_sequence().get(), model_sequence);
                    prop_assert_eq!(
                        receipt.admitted_activity().outstanding_permits(),
                        model_outstanding
                    );
                    prop_assert_eq!(receipt.admitted_activity().in_flight_effects(), 0);
                    if !was_open {
                        prop_assert!(domain.is_stopped());
                    }
                }
                2 => {
                    if let Some(ticket) = tickets[slot].take() {
                        let model = ticket_models[slot]
                            .take()
                            .expect("ticket model accompanies public ticket");
                        let expected_commitment = if use_correct_commitment {
                            model.commitment
                        } else {
                            guaranteed_different(model.commitment)
                        };
                        let result = domain.acquire(ticket, expected_commitment);

                        if expected_commitment.digest() != model.commitment.digest() {
                            prop_assert!(matches!(
                                result,
                                Err(EffectEntryError::CommitmentMismatch)
                            ));
                        } else if !model_open {
                            prop_assert!(matches!(
                                result,
                                Err(EffectEntryError::AdmissionStopped { .. })
                            ));
                        } else if model.epoch != model_epoch {
                            prop_assert!(matches!(result, Err(EffectEntryError::Revoked { .. })));
                        } else {
                            let permit = result.expect("current exact ticket must acquire");
                            model_sequence += 1;
                            model_outstanding += 1;
                            prop_assert_eq!(permit.acquisition_sequence().get(), model_sequence);
                            prop_assert_eq!(permit.commitment(), model.commitment);

                            if let Some(previous) = permits[slot].take() {
                                drop(previous);
                                model_outstanding -= 1;
                            }
                            permits[slot] = Some(permit);
                            permit_sequences[slot] = Some(model_sequence);
                        }
                    }
                }
                3 => {
                    if let Some(permit) = permits[slot].take() {
                        drop(permit);
                        permit_sequences[slot] = None;
                        model_outstanding -= 1;
                    }
                }
                4 => {
                    if let Some(permit) = permits[slot].take() {
                        let permit_sequence = permit_sequences[slot]
                            .take()
                            .expect("permit sequence accompanies public permit");
                        let permit_commitment = permit.commitment();
                        model_outstanding -= 1;
                        let expected_outstanding = model_outstanding;
                        let (receipt, activity_during_callback) = permit
                            .enter(|| domain.activity())
                            .expect("valid acquired permit must enter");
                        prop_assert_eq!(receipt.acquisition_sequence().get(), permit_sequence);
                        prop_assert_eq!(receipt.commitment(), permit_commitment);
                        prop_assert_eq!(
                            activity_during_callback.outstanding_permits(),
                            expected_outstanding
                        );
                        prop_assert_eq!(activity_during_callback.in_flight_effects(), 1);
                    }
                }
                5 => {
                    let before_sequence = model_sequence;
                    let before_activity = domain.activity();
                    let result = domain.resume();
                    if model_open {
                        prop_assert!(matches!(result, Err(EffectEntryError::AlreadyRunning)));
                        prop_assert_eq!(domain.current_sequence().get(), before_sequence);
                        prop_assert_eq!(domain.activity(), before_activity);
                    } else if model_outstanding != 0 {
                        prop_assert!(matches!(
                            result,
                            Err(EffectEntryError::ResumeWhileActive { .. })
                        ));
                        prop_assert_eq!(domain.current_sequence().get(), before_sequence);
                        prop_assert_eq!(domain.activity(), before_activity);
                        prop_assert!(domain.is_stopped());
                    } else {
                        let receipt = result.expect("quiescent stopped model must resume");
                        model_sequence += 1;
                        model_open = true;
                        prop_assert_eq!(receipt.epoch().get(), model_epoch);
                        prop_assert_eq!(receipt.resume_sequence().get(), model_sequence);
                        prop_assert!(domain.activity().is_quiescent());
                    }
                }
                _ => unreachable!(),
            }

            prop_assert_eq!(domain.current_epoch().get(), model_epoch);
            prop_assert_eq!(domain.current_sequence().get(), model_sequence);
            prop_assert_eq!(domain.is_stopped(), !model_open);
            prop_assert_eq!(domain.activity().outstanding_permits(), model_outstanding);
            prop_assert_eq!(domain.activity().in_flight_effects(), 0);
        }

        for permit in permits.into_iter().flatten() {
            drop(permit);
            model_outstanding -= 1;
        }
        prop_assert_eq!(model_outstanding, 0);
        prop_assert!(domain.activity().is_quiescent());
        prop_assert_eq!(domain.is_stopped(), !model_open);
    }
}
