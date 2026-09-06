use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use symthaea_authority::{Digest32, Operation, ResourceRef};
use symthaea_iot_actuation_guard_admission_reservation::{
    AdmissionReservationHead, CurrentAdmissionReservationFenceError,
    DurableAdmissionReservationStore,
};
use symthaea_iot_authority::{InclusiveRangeI64, SAFETY_ENVELOPE_SCHEMA_VERSION, SafetyEnvelope};
use symthaea_iot_device_protocol::{
    DEVICE_ENFORCEMENT_CONFIG_SCHEMA_VERSION, DeviceEnforcementConfigV1,
};

fn d(byte: u8) -> Digest32 {
    Digest32([byte; 32])
}

fn temp_root(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "symthaea-admission-current-{label}-{}-{nanos}",
        std::process::id()
    ))
}

fn config() -> DeviceEnforcementConfigV1 {
    DeviceEnforcementConfigV1 {
        schema_version: DEVICE_ENFORCEMENT_CONFIG_SCHEMA_VERSION,
        device: ResourceRef("iot:valve:72".into()),
        operation: Operation("valve.open".into()),
        exact_policy_digest: d(20),
        minimum_policy_registry_sequence: 5,
        safety: SafetyEnvelope {
            schema_version: SAFETY_ENVELOPE_SCHEMA_VERSION,
            policy_id: "device-local-safe-open".into(),
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("valve.open".into()),
            allowed_firmware: BTreeSet::from([d(7)]),
            parameter_ranges: BTreeMap::from([(
                "duration_ms".into(),
                InclusiveRangeI64 {
                    min: 1_000,
                    max: 120_000,
                },
            )]),
            required_observations: BTreeMap::from([(
                "pressure_x100".into(),
                InclusiveRangeI64 {
                    min: 100,
                    max: 350_000,
                },
            )]),
        },
        maximum_envelope_lifetime_s: 5,
    }
}

#[test]
fn current_admission_fence_rejects_wrong_composed_head() {
    let root = temp_root("wrong-head");
    let store = DurableAdmissionReservationStore::open(&root, config()).unwrap();
    let current = store.current_head().unwrap();
    let wrong = AdmissionReservationHead {
        generation: current.generation,
        digest: d(0xFE),
    };

    assert!(matches!(
        store.fence_current(wrong),
        Err(CurrentAdmissionReservationFenceError::HeadMismatch {
            expected,
            current: observed,
        }) if expected == wrong && observed == current
    ));

    drop(store);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn held_admission_fence_blocks_competing_store_on_same_kernel_lock() {
    let root = temp_root("blocking");
    let cfg = config();
    let store_a = DurableAdmissionReservationStore::open(&root, cfg.clone()).unwrap();
    let store_b = DurableAdmissionReservationStore::open(&root, cfg).unwrap();
    let head = store_a.current_head().unwrap();
    let fence = store_a.fence_current(head).unwrap();

    assert_eq!(fence.head(), head);
    assert_eq!(fence.checkpoint().head().unwrap(), head);

    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        started_tx.send(()).unwrap();
        let observed = store_b.current_head().unwrap();
        done_tx.send(observed).unwrap();
    });

    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(done_rx.recv_timeout(Duration::from_millis(150)).is_err());

    drop(fence);
    assert_eq!(done_rx.recv_timeout(Duration::from_secs(2)).unwrap(), head);
    worker.join().unwrap();

    drop(store_a);
    fs::remove_dir_all(root).unwrap();
}
