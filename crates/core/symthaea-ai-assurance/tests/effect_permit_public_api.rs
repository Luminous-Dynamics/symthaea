use std::sync::{mpsc, Arc, Barrier};
use std::thread;

use symthaea_ai_assurance::{EffectEntryDomain, EffectEntryError};

#[test]
fn public_revocation_before_acquisition_prevents_entry() {
    let domain = EffectEntryDomain::new();
    let binding = [11; 32];
    let ticket = domain.issue_ticket(binding);
    let revocation = domain.revoke_all().unwrap();

    let result = domain.acquire(ticket, binding);
    assert!(matches!(result, Err(EffectEntryError::Revoked { .. })));
    assert_eq!(revocation.previous_epoch().get(), 0);
    assert_eq!(revocation.current_epoch().get(), 1);
    assert!(revocation.admitted_activity().is_quiescent());
}

#[test]
fn public_acquisition_before_revocation_preserves_one_admitted_effect() {
    let domain = EffectEntryDomain::new();
    let binding = [12; 32];
    let ticket = domain.issue_ticket(binding);
    let permit = domain.acquire(ticket, binding).unwrap();
    let acquisition = permit.acquisition_sequence();

    let revocation = domain.revoke_all().unwrap();
    assert_eq!(revocation.admitted_activity().outstanding_permits(), 1);
    assert_eq!(revocation.admitted_activity().in_flight_effects(), 0);

    let (receipt, effect_result) = permit.enter(|| "entered").unwrap();
    assert_eq!(effect_result, "entered");
    assert_eq!(receipt.action_binding(), binding);
    assert_eq!(receipt.acquisition_sequence(), acquisition);
    assert!(acquisition < revocation.revocation_sequence());
    assert!(domain.activity().is_quiescent());
}

#[test]
fn public_effect_callback_does_not_hold_revocation_lock() {
    let domain = Arc::new(EffectEntryDomain::new());
    let binding = [13; 32];
    let ticket = domain.issue_ticket(binding);
    let permit = domain.acquire(ticket, binding).unwrap();
    let acquisition = permit.acquisition_sequence();

    let (entered_tx, entered_rx) = mpsc::channel();
    let (continue_tx, continue_rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        permit.enter(|| {
            entered_tx.send(()).unwrap();
            continue_rx.recv().unwrap();
            13_u64
        })
    });

    entered_rx.recv().unwrap();
    let revocation = domain.revoke_all().unwrap();
    assert!(acquisition < revocation.revocation_sequence());
    assert_eq!(revocation.admitted_activity().outstanding_permits(), 0);
    assert_eq!(revocation.admitted_activity().in_flight_effects(), 1);

    continue_tx.send(()).unwrap();
    let (receipt, value) = worker.join().unwrap().unwrap();
    assert_eq!(value, 13);
    assert_eq!(receipt.acquisition_sequence(), acquisition);
    assert!(domain.activity().is_quiescent());
}

#[test]
fn public_concurrent_race_has_only_admit_before_revoke_or_revoke_before_admit() {
    for tag in 0_u8..32 {
        let domain = Arc::new(EffectEntryDomain::new());
        let binding = [tag; 32];
        let ticket = domain.issue_ticket(binding);
        let barrier = Arc::new(Barrier::new(3));

        let acquire_domain = Arc::clone(&domain);
        let acquire_barrier = Arc::clone(&barrier);
        let acquire_thread = thread::spawn(move || {
            acquire_barrier.wait();
            acquire_domain.acquire(ticket, binding)
        });

        let revoke_domain = Arc::clone(&domain);
        let revoke_barrier = Arc::clone(&barrier);
        let revoke_thread = thread::spawn(move || {
            revoke_barrier.wait();
            revoke_domain.revoke_all().unwrap()
        });

        barrier.wait();
        let acquisition = acquire_thread.join().unwrap();
        let revocation = revoke_thread.join().unwrap();

        match acquisition {
            Ok(permit) => {
                assert!(permit.acquisition_sequence() < revocation.revocation_sequence());
                assert_eq!(revocation.admitted_activity().outstanding_permits(), 1);
                let (_, entered) = permit.enter(|| true).unwrap();
                assert!(entered);
                assert!(domain.activity().is_quiescent());
            }
            Err(EffectEntryError::Revoked {
                ticket_epoch,
                current_epoch,
                current_sequence,
            }) => {
                assert!(ticket_epoch < current_epoch);
                assert_eq!(current_sequence, revocation.revocation_sequence());
                assert!(revocation.admitted_activity().is_quiescent());
            }
            Err(other) => panic!("unexpected linearization result: {other}"),
        }
    }
}
