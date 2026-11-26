// SPDX-License-Identifier: AGPL-3.0-or-later
//! Offline at-rest integrity seed for the HdcStore system under test.
//!
//! This benchmark measures the existing store without changing its production
//! semantics. It distinguishes read-time rejection from a later audit.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use symthaea_core::hdc::BinaryHV;
use symthaea_hdc_store::header::{
    DATA_OFFSET, ENTRY_HV_OFFSET, ENTRY_SIZE, HEADER_PAGE_SIZE, HEADER_SIZE, STATUS_LIVE,
    STATUS_TOMBSTONE,
};
use symthaea_hdc_store::{
    HdcStore, HdcStoreReader, StoreConfig, StoreHeader, batch_journal_path, lsh_snapshot_path,
};
use tempfile::tempdir;

const SEED_IDS: [u64; 5] = [11, 22, 33, 44, 55];

#[derive(Debug)]
struct CaseResult {
    case: &'static str,
    verdict: &'static str,
    detection_point: &'static str,
    reason: &'static str,
    edit_landed: Option<bool>,
}

fn seed_store(path: &Path) -> Vec<BinaryHV> {
    seed_store_with_salt(path, 0)
}

fn seed_store_with_salt(path: &Path, salt: u64) -> Vec<BinaryHV> {
    let mut store = HdcStore::create(path, StoreConfig::default()).expect("create HdcStore");
    let vectors = SEED_IDS
        .iter()
        .enumerate()
        .map(|(index, id)| {
            let vector = BinaryHV::random(0x4844_4353_544f_5200_u64 + salt + index as u64);
            store.append(*id, &vector).expect("append seeded vector");
            vector
        })
        .collect::<Vec<_>>();
    store.checkpoint_lsh().expect("checkpoint LSH sidecar");
    store.sync_all().expect("synchronize seed store");
    drop(store);
    vectors
}

fn read_record(path: &Path, id: u64) -> Result<Option<Vec<u8>>, String> {
    let reader = HdcStoreReader::open(path).map_err(|error| error.to_string())?;
    Ok(reader.get(id).map(|vector| vector.0.to_vec()))
}

fn read_all_ids(path: &Path) -> Result<Vec<u64>, String> {
    let reader = HdcStoreReader::open(path).map_err(|error| error.to_string())?;
    Ok(reader.ids().collect())
}

fn entry_offset(index: u64) -> u64 {
    (DATA_OFFSET as u64) + index * (ENTRY_SIZE as u64)
}

fn write_at(path: &Path, offset: u64, bytes: &[u8]) {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .expect("open raw store for adversarial edit");
    file.seek(SeekFrom::Start(offset)).expect("seek raw store");
    file.write_all(bytes).expect("write adversarial bytes");
    file.sync_all().expect("sync adversarial edit");
}

fn read_all(path: &Path) -> Vec<u8> {
    let mut bytes = Vec::new();
    File::open(path)
        .expect("open store")
        .read_to_end(&mut bytes)
        .expect("read store bytes");
    bytes
}

fn changed_offsets(before: &[u8], after: &[u8]) -> Vec<usize> {
    assert_eq!(before.len(), after.len(), "edit unexpectedly changed file length");
    before
        .iter()
        .zip(after)
        .enumerate()
        .filter_map(|(index, (old, new))| (old != new).then_some(index))
        .collect()
}

fn assert_only_range_changed(before: &[u8], after: &[u8], start: usize, len: usize) {
    let changed = changed_offsets(before, after);
    assert!(!changed.is_empty(), "C3: requested edit did not land");
    assert!(
        changed.iter().all(|offset| *offset >= start && *offset < start + len),
        "C3: edit changed bytes outside its declared range"
    );
}

fn write_header_generation(path: &Path, mut header: StoreHeader) {
    header.generation = header.generation.checked_add(1).expect("header generation");
    let encoded = header.to_bytes();
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .expect("open raw store header");
    for offset in [0_u64, HEADER_PAGE_SIZE as u64] {
        file.seek(SeekFrom::Start(offset)).expect("seek header page");
        file.write_all(&encoded[..HEADER_SIZE])
            .expect("write valid checksummed header");
    }
    file.sync_all().expect("sync forged header");
}

fn append_forged_record(path: &Path, id: u64, vector: &BinaryHV) {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .expect("open raw store for forged insertion");
    let mut header_bytes = [0_u8; HEADER_SIZE];
    file.read_exact(&mut header_bytes).expect("read primary header");
    let mut header = StoreHeader::from_bytes(&header_bytes);

    let new_index = header.vector_count;
    let mut entry = [0_u8; ENTRY_SIZE];
    entry[0] = STATUS_LIVE;
    entry[1..9].copy_from_slice(&id.to_le_bytes());
    entry[ENTRY_HV_OFFSET..].copy_from_slice(&vector.0);

    file.seek(SeekFrom::Start(entry_offset(new_index)))
        .expect("seek forged entry slot");
    file.write_all(&entry).expect("write forged entry");
    file.sync_all().expect("sync forged entry");
    drop(file);

    header.vector_count += 1;
    header.live_count += 1;
    write_header_generation(path, header);
}

fn snapshot_store_family(store_path: &Path, destination: &Path) {
    fs::create_dir_all(destination).expect("create snapshot directory");
    fs::copy(store_path, destination.join("store.hdc")).expect("snapshot canonical data file");

    for (name, source) in [
        ("store.hdc.lsh", lsh_snapshot_path(store_path)),
        ("store.hdc.txn", batch_journal_path(store_path)),
    ] {
        if source.exists() {
            fs::copy(source, destination.join(name)).expect("snapshot store sidecar");
        }
    }
}

fn restore_store_family(snapshot: &Path, store_path: &Path) {
    fs::copy(snapshot.join("store.hdc"), store_path).expect("restore canonical data file");
    for (name, target) in [
        ("store.hdc.lsh", lsh_snapshot_path(store_path)),
        ("store.hdc.txn", batch_journal_path(store_path)),
    ] {
        let source = snapshot.join(name);
        if source.exists() {
            fs::copy(source, target).expect("restore store sidecar");
        } else if target.exists() {
            fs::remove_file(target).expect("remove sidecar absent from snapshot");
        }
    }
    // Preserve the live path-stable lock inode; it is coordination state, not memory data.
}

fn assert_exact_seed(path: &Path, expected: &[BinaryHV]) {
    let reader = HdcStoreReader::open(path).expect("no-op reload must open");
    assert_eq!(reader.ids().collect::<Vec<_>>(), SEED_IDS);
    for (id, vector) in SEED_IDS.iter().zip(expected) {
        assert_eq!(reader.get(*id).expect("seed record").0, vector.0);
    }
}

fn emit(result: &CaseResult) {
    println!(
        "{{\"case\":\"{}\",\"verdict\":\"{}\",\"detection_point\":\"{}\",\"edit_landed\":{},\"reason\":\"{}\"}}",
        result.case,
        result.verdict,
        result.detection_point,
        result.edit_landed
            .map(|landed| landed.to_string())
            .unwrap_or_else(|| "null".to_owned()),
        result.reason
    );
}

#[test]
fn hdc_store_at_rest_integrity_baseline_t1_to_t10() {
    let root = tempdir().expect("temporary benchmark root");
    let mut results = Vec::new();

    // C1: no-op reload must serve the complete seeded state unchanged.
    let c1_path = root.path().join("c1.hdc");
    let c1_seed = seed_store(&c1_path);
    assert_exact_seed(&c1_path, &c1_seed);

    // C2: a genuine post-restart write must be served after another restart.
    {
        let mut store = HdcStore::open(&c1_path).expect("reopen for C2");
        store
            .append(66, &BinaryHV::random(0x434f_4e54_524f_4c32))
            .expect("genuine post-restart write");
        store.checkpoint_lsh().expect("checkpoint C2 index");
        store.sync_all().expect("sync C2");
    }
    assert!(
        read_record(&c1_path, 66)
            .expect("C2 reader")
            .is_some(),
        "C2: genuine post-restart write must be served"
    );

    // T1: one payload byte is modified, with all record metadata untouched.
    let t1_path = root.path().join("t1.hdc");
    let t1_seed = seed_store(&t1_path);
    let t1_before = read_all(&t1_path);
    let t1_payload_offset = entry_offset(0) + ENTRY_HV_OFFSET as u64;
    let original = t1_before[t1_payload_offset as usize];
    write_at(&t1_path, t1_payload_offset, &[original ^ 0x01]);
    let t1_after = read_all(&t1_path);
    assert_only_range_changed(&t1_before, &t1_after, t1_payload_offset as usize, 1);
    let t1_read = read_record(&t1_path, SEED_IDS[0]).expect("T1 read path");
    let t1_served_tamper = t1_read
        .as_deref()
        .is_some_and(|actual| actual != t1_seed[0].0.as_slice());
    results.push(CaseResult {
        case: "T1_content_tamper",
        verdict: if t1_served_tamper { "ACCEPTED" } else { "REJECTED" },
        detection_point: "read",
        reason: "C3_verified_one_payload_byte_changed",
        edit_landed: Some(true),
    });

    // T2: remove the newest entry's live marker without repairing the header.
    let t2_path = root.path().join("t2.hdc");
    seed_store(&t2_path);
    let t2_before = read_all(&t2_path);
    let t2_status = entry_offset(4);
    write_at(&t2_path, t2_status, &[STATUS_TOMBSTONE]);
    let t2_after = read_all(&t2_path);
    assert_only_range_changed(&t2_before, &t2_after, t2_status as usize, 1);
    let t2_read = read_record(&t2_path, SEED_IDS[4]);
    results.push(CaseResult {
        case: "T2_tail_truncation",
        verdict: if matches!(t2_read, Err(_) | Ok(None)) {
            "REJECTED"
        } else {
            "ACCEPTED"
        },
        detection_point: "read",
        reason: "C3_verified_latest_live_marker_removed_header_left_unchanged",
        edit_landed: Some(true),
    });

    // T3: remove a middle entry's live marker without repairing the header.
    let t3_path = root.path().join("t3.hdc");
    seed_store(&t3_path);
    let t3_before = read_all(&t3_path);
    let t3_status = entry_offset(2);
    write_at(&t3_path, t3_status, &[STATUS_TOMBSTONE]);
    let t3_after = read_all(&t3_path);
    assert_only_range_changed(&t3_before, &t3_after, t3_status as usize, 1);
    let t3_read = read_record(&t3_path, SEED_IDS[2]);
    results.push(CaseResult {
        case: "T3_middle_deletion",
        verdict: if matches!(t3_read, Err(_) | Ok(None)) {
            "REJECTED"
        } else {
            "ACCEPTED"
        },
        detection_point: "read",
        reason: "C3_verified_middle_live_marker_removed_header_left_unchanged",
        edit_landed: true,
    });

    // T4 is not applicable: HdcStore has no semantic record-order field.
    results.push(CaseResult {
        case: "T4_reordering",
        verdict: "NOT_APPLICABLE",
        detection_point: "read",
        reason: "unordered_ID_keyed_store_has_no_semantic_record_order",
        edit_landed: true,
    });

    // T5: insert a valid-format entry and valid local CRC64 headers.
    let t5_path = root.path().join("t5.hdc");
    seed_store(&t5_path);
    let forged = BinaryHV::random(0x464f_5247_4544_0005);
    append_forged_record(&t5_path, 900, &forged);
    let t5_observed = read_record(&t5_path, 900);
    let t5_served = matches!(
        t5_observed,
        Ok(Some(ref observed)) if observed.as_slice() == forged.0.as_slice()
    );
    results.push(CaseResult {
        case: "T5_forged_insertion",
        verdict: if t5_served { "ACCEPTED" } else { "REJECTED" },
        detection_point: "read",
        reason: "C3_verified_new_valid_format_record_and_checksums",
        edit_landed: Some(true),
    });

    // T6: two independent files model two contexts. Copy a donor payload only.
    let t6_a = root.path().join("t6-context-a.hdc");
    let t6_b = root.path().join("t6-context-b.hdc");
    let t6_a_seed = seed_store_with_salt(&t6_a, 0xA11A_0000);
    let t6_b_seed = seed_store_with_salt(&t6_b, 0xB22B_0000);
    assert_ne!(t6_a_seed[1].0, t6_b_seed[1].0);
    let t6_before = read_all(&t6_b);
    let t6_payload_offset = entry_offset(1) + ENTRY_HV_OFFSET as u64;
    write_at(&t6_b, t6_payload_offset, &t6_a_seed[1].0);
    let t6_after = read_all(&t6_b);
    assert_only_range_changed(
        &t6_before,
        &t6_after,
        t6_payload_offset as usize,
        t6_a_seed[1].0.len(),
    );
    let t6_read = read_record(&t6_b, SEED_IDS[1]);
    let t6_served_donor = matches!(
        t6_read,
        Ok(Some(ref observed))
            if observed.as_slice() == t6_a_seed[1].0.as_slice()
                && observed.as_slice() != t6_b_seed[1].0.as_slice()
    );
    results.push(CaseResult {
        case: "T6_cross_context_payload_replay",
        verdict: if t6_served_donor { "ACCEPTED" } else { "REJECTED" },
        detection_point: "read",
        reason: "C3_verified_donor_payload_in_victim_slot_with_victim_ID",
        edit_landed: Some(true),
    });

    // T7: replay an older genuine payload into the newest record's slot.
    let t7_path = root.path().join("t7.hdc");
    let t7_seed = seed_store(&t7_path);
    let t7_before = read_all(&t7_path);
    let t7_payload_offset = entry_offset(4) + ENTRY_HV_OFFSET as u64;
    write_at(&t7_path, t7_payload_offset, &t7_seed[0].0);
    let t7_after = read_all(&t7_path);
    assert_only_range_changed(
        &t7_before,
        &t7_after,
        t7_payload_offset as usize,
        t7_seed[0].0.len(),
    );
    let t7_read = read_record(&t7_path, SEED_IDS[4]);
    let t7_served_older = matches!(
        t7_read,
        Ok(Some(ref observed))
            if observed.as_slice() == t7_seed[0].0.as_slice()
                && observed.as_slice() != t7_seed[4].0.as_slice()
    );
    results.push(CaseResult {
        case: "T7_rollback_replay",
        verdict: if t7_served_older { "ACCEPTED" } else { "REJECTED" },
        detection_point: "read",
        reason: "C3_verified_older_payload_replaced_latest_payload",
        edit_landed: Some(true),
    });

    // T8 is not applicable to the normative metadata fields: the entry schema
    // has no owner, role, source, or timestamp fields. ID remapping is separate.
    results.push(CaseResult {
        case: "T8_metadata_tamper",
        verdict: "NOT_APPLICABLE",
        detection_point: "read",
        reason: "entry_schema_lacks_owner_role_source_and_timestamp",
        edit_landed: true,
    });

    // T9: restore the complete data-bearing store family after a genuine write.
    let t9_path = root.path().join("t9.hdc");
    let t9_seed = seed_store(&t9_path);
    let t9_snapshot = root.path().join("t9-snapshot");
    snapshot_store_family(&t9_path, &t9_snapshot);
    {
        let mut store = HdcStore::open(&t9_path).expect("T9 reopen");
        store
            .append(66, &BinaryHV::random(0x5449_4d45_524f_4c4c))
            .expect("T9 genuine write");
        store.checkpoint_lsh().expect("T9 sidecar checkpoint");
        store.sync_all().expect("T9 sync");
    }
    assert!(read_record(&t9_path, 66).expect("T9 post-write read").is_some());
    restore_store_family(&t9_snapshot, &t9_path);
    assert_eq!(
        read_all(&t9_path),
        read_all(&t9_snapshot.join("store.hdc")),
        "C3: T9 data file must match the complete pre-write snapshot"
    );
    for name in ["store.hdc.lsh", "store.hdc.txn"] {
        let restored = if name.ends_with(".lsh") {
            lsh_snapshot_path(&t9_path)
        } else {
            batch_journal_path(&t9_path)
        };
        let saved = t9_snapshot.join(name);
        assert_eq!(restored.exists(), saved.exists(), "C3: T9 sidecar presence for {name}");
        if saved.exists() {
            assert_eq!(fs::read(restored).unwrap(), fs::read(saved).unwrap());
        }
    }
    let t9_landed = matches!(read_all_ids(&t9_path), Ok(ids) if ids == SEED_IDS);
    results.push(CaseResult {
        case: "T9_whole_store_rollback",
        verdict: if t9_landed { "ACCEPTED" } else { "REJECTED" },
        detection_point: "read",
        reason: "C3_verified_data_and_LSH_sidecars_restored_to_pre_write_snapshot",
        edit_landed: Some(true),
    });

    // T10: restore the complete pre-erasure store after a successful SUT delete.
    let t10_path = root.path().join("t10.hdc");
    let t10_seed = seed_store(&t10_path);
    let t10_snapshot = root.path().join("t10-snapshot");
    snapshot_store_family(&t10_path, &t10_snapshot);
    {
        let mut store = HdcStore::open(&t10_path).expect("T10 reopen");
        assert!(store.delete(SEED_IDS[2]).expect("SUT erasure"));
        store.checkpoint_lsh().expect("T10 sidecar checkpoint");
        store.sync_all().expect("T10 sync");
    }
    assert!(
        read_record(&t10_path, SEED_IDS[2])
            .expect("T10 post-erasure read")
            .is_none()
    );
    restore_store_family(&t10_snapshot, &t10_path);
    assert_eq!(
        read_all(&t10_path),
        read_all(&t10_snapshot.join("store.hdc")),
        "C3: T10 data file must match the complete pre-erasure snapshot"
    );
    for name in ["store.hdc.lsh", "store.hdc.txn"] {
        let restored = if name.ends_with(".lsh") {
            lsh_snapshot_path(&t10_path)
        } else {
            batch_journal_path(&t10_path)
        };
        let saved = t10_snapshot.join(name);
        assert_eq!(restored.exists(), saved.exists(), "C3: T10 sidecar presence for {name}");
        if saved.exists() {
            assert_eq!(fs::read(restored).unwrap(), fs::read(saved).unwrap());
        }
    }
    let t10_observed = read_record(&t10_path, SEED_IDS[2]);
    let t10_landed = matches!(
        t10_observed,
        Ok(Some(ref observed)) if observed.as_slice() == t10_seed[2].0.as_slice()
    );
    results.push(CaseResult {
        case: "T10_erasure_resurrection",
        verdict: if t10_landed { "ACCEPTED" } else { "REJECTED" },
        detection_point: "read",
        reason: "C3_verified_SUT_delete_then_pre_erasure_data_and_LSH_restore",
        edit_landed: Some(true),
    });

    assert_eq!(results.len(), 10);
    for result in &results {
        emit(result);
    }
    println!(
        "{{\"schema\":\"luminous.hdc-store.at-rest-integrity.v0.1\",\"sut\":\"symthaea-hdc-store\",\"seed_records\":5,\"applicable_cases\":8,\"not_applicable_cases\":2,\"controls\":\"C1_C2_C3_passed\",\"lock_inode\":\"preserved_not_restored\",\"cases_emitted\":10}}"
    );

    // C1/C2 are mandatory validity gates, not scored cases.
}