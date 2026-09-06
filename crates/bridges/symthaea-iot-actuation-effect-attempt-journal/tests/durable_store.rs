use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use symthaea_authority::ResourceRef;
use symthaea_iot_actuation_effect_attempt_journal::{
    DurableEffectAttemptJournalCheckpointV1, DurableEffectAttemptJournalStore,
    EffectAttemptJournalError,
};

fn temp_root(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "symthaea-effect-attempt-{label}-{}-{nanos}",
        std::process::id()
    ))
}

#[test]
fn deterministic_genesis_reopens_against_exact_retained_head() {
    let root = temp_root("genesis");
    let device = ResourceRef("iot:valve:72".into());
    let genesis = DurableEffectAttemptJournalCheckpointV1::genesis(&device).unwrap();
    let head = genesis.head().unwrap();

    let store = DurableEffectAttemptJournalStore::open(&root, &device, head).unwrap();
    assert_eq!(store.current_checkpoint().unwrap(), genesis);
    assert_eq!(store.trusted_current_head(), head);

    drop(store);
    let reopened = DurableEffectAttemptJournalStore::open(&root, &device, head).unwrap();
    assert_eq!(reopened.current_checkpoint().unwrap().head().unwrap(), head);

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn another_devices_retained_head_is_rejected() {
    let root = temp_root("wrong-head");
    let device = ResourceRef("iot:valve:72".into());
    let other = ResourceRef("iot:valve:73".into());
    let wrong_head = DurableEffectAttemptJournalCheckpointV1::genesis(&other)
        .unwrap()
        .head()
        .unwrap();

    let error = match DurableEffectAttemptJournalStore::open(&root, &device, wrong_head) {
        Ok(_) => panic!("wrong retained head unexpectedly opened journal"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        EffectAttemptJournalError::TrustedJournalHeadMismatch
    ));

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn malformed_persisted_bytes_fail_closed() {
    let root = temp_root("malformed");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("effect-attempt.state"), b"not-a-canonical-checkpoint").unwrap();

    let device = ResourceRef("iot:valve:72".into());
    let head = DurableEffectAttemptJournalCheckpointV1::genesis(&device)
        .unwrap()
        .head()
        .unwrap();
    let error = match DurableEffectAttemptJournalStore::open(&root, &device, head) {
        Ok(_) => panic!("malformed journal unexpectedly opened"),
        Err(error) => error,
    };
    assert!(matches!(error, EffectAttemptJournalError::StateEncoding));

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn symlinked_root_is_rejected() {
    use std::os::unix::fs::symlink;

    let target = temp_root("symlink-target");
    let link = temp_root("symlink-link");
    fs::create_dir_all(&target).unwrap();
    symlink(&target, &link).unwrap();

    let device = ResourceRef("iot:valve:72".into());
    let head = DurableEffectAttemptJournalCheckpointV1::genesis(&device)
        .unwrap()
        .head()
        .unwrap();
    let error = match DurableEffectAttemptJournalStore::open(&link, &device, head) {
        Ok(_) => panic!("symlinked journal root unexpectedly opened"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        EffectAttemptJournalError::InvalidRootDirectory
    ));

    fs::remove_file(link).unwrap();
    fs::remove_dir_all(target).unwrap();
}
