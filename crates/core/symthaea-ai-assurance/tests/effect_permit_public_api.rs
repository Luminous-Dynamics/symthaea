use std::sync::{mpsc, Arc, Barrier};
use std::thread;

use symthaea_ai_assurance::{EffectEntryDomain, EffectEntryError};

#[test]
fn public_revocation_latches_admission_until_explicit_resume() {
    let domain = EffectEntryDomain::new();
    let binding = [11; 32];
    let ticket = domain.issue_ticket(binding).unwrap();
    let stale_after_resume = domain.issue_ticket(binding).unwrap();
    let revocation = domain.revoke_all().unwrap();

    assert!(domain.is_stopped());
    assert!(matches!(
        domain.issue_ticket(binding),
        Err(EffectEntryError::AdmissionStopped { .. })
    ));
    assert!(matches!(
        domain.acquire(ticket, binding),
        Err(EffectEntryError::AdmissionStopped { .. })
    ));
    assert!(revocation.admitted_activity().is_quiescent());

    let resume = domain.resume().unwrap();
    assert!(!domain.is_stopped());
    assert_eq!(resume.epoch(), revocation.current_epoch());
    assert!(revocation.revocation_sequence() < resume.resume_sequence());
    assert!(matches!(
        domain.acquire(stale_after_resume, binding),
        Err(EffectEntryError::Revoked { .. })
    ));
}

#[test]
fn public_acquisition_before_revocation_preserves_one_admitted_effect() {
    let domain = EffectEntryDomain::new();
    let binding = [12; 32];
    let ticket = domain.issue_ticket(binding).unwrap();
    let permit = domain.acquire(ticket, binding).unwrap();
    let acquisition = permit.acquisition_sequence();

    let revocation = domain.revoke_all().unwrap();
    assert!(domain.is_stopped());
    assert_eq!(revocation.admitted_activity().outstanding_permits(), 1);
    assert_eq!(revocation.admitted_activity().in_flight_effects(), 0);
    assert!(matches!(
        domain.resume(),
        Err(EffectEntryError::ResumeWhileActive { .. })
    ));

    let (receipt, effect_result) = permit.enter(|| "entered").unwrap();
    assert_eq!(effect_result, "entered");
    assert_eq!(receipt.action_binding(), binding);
    assert_eq!(receipt.acquisition_sequence(), acquisition);
    assert!(acquisition < revocation.revocation_sequence());
    assert!(domain.activity().is_quiescent());

    domain.resume().unwrap();
    assert!(!domain.is_stopped());
}

#[test]
fn public_effect_callback_does_not_hold_revocation_lock() {
    let domain = Arc::new(EffectEntryDomain::new());
    let binding = [13; 32];
    let ticket = domain.issue_ticket(binding).unwrap();
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
    assert!(matches!(
        domain.resume(),
        Err(EffectEntryError::ResumeWhileActive { .. })
    ));

    continue_tx.send(()).unwrap();
    let (receipt, value) = worker.join().unwrap().unwrap();
    assert_eq!(value, 13);
    assert_eq!(receipt.acquisition_sequence(), acquisition);
    assert!(domain.activity().is_quiescent());

    domain.resume().unwrap();
    assert!(!domain.is_stopped());
}

#[test]
fn public_new_work_requires_resume_after_emergency_stop() {
    let domain = EffectEntryDomain::new();
    let binding = [14; 32];
    domain.revoke_all().unwrap();

    assert!(matches!(
        domain.issue_ticket(binding),
        Err(EffectEntryError::AdmissionStopped { .. })
    ));
    let stopped_sequence = domain.current_sequence();
    let resume = domain.resume().unwrap();
    assert!(stopped_sequence < resume.resume_sequence());

    let ticket = domain.issue_ticket(binding).unwrap();
    let permit = domain.acquire(ticket, binding).unwrap();
    drop(permit);
    assert!(domain.activity().is_quiescent());
}

#[test]
fn public_concurrent_race_has_only_admit_before_stop_or_stop_before_admit() {
    for tag in 0_u8..32 {
        let domain = Arc::new(EffectEntryDomain::new());
        let binding = [tag; 32];
        let ticket = domain.issue_ticket(binding).unwrap();
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
        assert!(domain.is_stopped());

        match acquisition {
            Ok(permit) => {
                assert!(permit.acquisition_sequence() < revocation.revocation_sequence());
                assert_eq!(revocation.admitted_activity().outstanding_permits(), 1);
                assert!(matches!(
                    domain.resume(),
                    Err(EffectEntryError::ResumeWhileActive { .. })
                ));
                let (_, entered) = permit.enter(|| true).unwrap();
                assert!(entered);
                assert!(domain.activity().is_quiescent());
                domain.resume().unwrap();
            }
            Err(EffectEntryError::AdmissionStopped {
                current_epoch,
                current_sequence,
            }) => {
                assert_eq!(current_epoch, revocation.current_epoch());
                assert_eq!(current_sequence, revocation.revocation_sequence());
                assert!(revocation.admitted_activity().is_quiescent());
                domain.resume().unwrap();
            }
            Err(other) => panic!("unexpected linearization result: {other}"),
        }
        assert!(!domain.is_stopped());
    }
}
