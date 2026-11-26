// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Synthetic, file-backed reference lab for the SAT memory-at-rest method.
//!
//! This crate exercises the T1-T10 mutation procedures, C1-C3 controls,
//! denominator rules and native score calculation against three deliberately
//! small reference implementations. It is NOT a test of the Symthaea memory
//! backend or any production agent. The protected profile is an illustrative
//! design: record HMACs use a test key held outside the store, the rollback
//! anchor is held outside the restored store bundle, and erasure tombstones
//! survive a store-only backup restore.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const GENESIS: &str = "SAT-MEMORY-GENESIS-v0.1";
const RECORD_DOMAIN: &str = "SAT-MEMORY-RECORD-v0.1\n";
const LAB_KEY: &[u8] = b"synthetic-only-key-not-for-production-use";
const RECORDS_PER_CONTEXT: usize = 5;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProfileMode {
    Vulnerable,
    ProtectedRead,
    ProtectedAudit,
}

impl ProfileMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "vulnerable" => Ok(Self::Vulnerable),
            "protected-read" => Ok(Self::ProtectedRead),
            "protected-audit" => Ok(Self::ProtectedAudit),
            _ => Err(format!(
                "unknown mode {value:?}; use vulnerable, protected-read, or protected-audit"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NativeVerdict {
    Rejected,
    Reported,
    Accepted,
    Error,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DetectionPoint {
    Read,
    Audit,
    None,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObservationPointClass {
    A,
    B,
    C,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CaseStatus {
    Applicable,
    Error,
    NotApplicable,
    NotRun,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CampaignStatus {
    Evaluable,
    NotEvaluable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Record {
    pub context_id: String,
    pub record_id: String,
    pub sequence: u64,
    pub owner: String,
    pub role: String,
    pub source: String,
    pub timestamp: String,
    pub fact: String,
    pub previous_mac: String,
    pub record_mac: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct StoreFile {
    generation: u64,
    records: Vec<Record>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct IntegrityAnchor {
    generation: u64,
    head_mac: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
struct ErasureLedger {
    erased_record_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RecordPayload<'a> {
    context_id: &'a str,
    record_id: &'a str,
    sequence: u64,
    owner: &'a str,
    role: &'a str,
    source: &'a str,
    timestamp: &'a str,
    fact: &'a str,
    previous_mac: &'a str,
}

#[derive(Debug, Clone)]
struct LabPaths {
    store: PathBuf,
    anchor: PathBuf,
    erasure: PathBuf,
    results: PathBuf,
}

impl LabPaths {
    fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            store: root.join("store").join("memory-store.json"),
            anchor: root.join("trust-domain").join("rollback-anchor.json"),
            erasure: root.join("trust-domain").join("erasure-ledger.json"),
            results: root.join("results"),
        }
    }
}

#[derive(Debug)]
struct Sut {
    paths: LabPaths,
    mode: ProfileMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlResult {
    pub case_id: String,
    pub passed: bool,
    pub detail: String,
    pub result_artifact_ref: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseResult {
    pub case_id: String,
    pub case_status: CaseStatus,
    pub edit_landed: bool,
    pub native_verdict: NativeVerdict,
    pub detection_point: DetectionPoint,
    pub observation_point_class: ObservationPointClass,
    pub affected_context: String,
    pub c3_detail: String,
    pub result_detail: String,
    pub result_artifact_ref: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeScore {
    pub denominator_n: usize,
    pub rejected: usize,
    pub reported_at_read: usize,
    pub reported_at_audit: usize,
    pub accepted: usize,
    pub errors_excluded: usize,
    pub not_applicable_excluded: usize,
    pub score: Option<f64>,
    pub category_scores: BTreeMap<String, Option<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignReport {
    pub schema_version: String,
    pub campaign_id: String,
    pub profile_mode: ProfileMode,
    pub evidence_level: String,
    pub real_sut_evaluated: bool,
    pub campaign_status: CampaignStatus,
    pub c1_no_op_reload: ControlResult,
    pub c2_genuine_write_after_restart: ControlResult,
    pub cases: Vec<CaseResult>,
    pub native_score: NativeScore,
    pub limitations: Vec<String>,
}

#[derive(Debug)]
enum IntegrityFailure {
    MissingAnchor,
    AnchorGenerationMismatch,
    AnchorHeadMismatch,
    StoreGenerationMismatch,
    SequenceMismatch,
    PreviousMacMismatch,
    RecordMacMismatch,
    DuplicateRecordId,
    ErasureResurrection(String),
}

impl std::fmt::Display for IntegrityFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingAnchor => write!(f, "rollback anchor is missing"),
            Self::AnchorGenerationMismatch => write!(f, "store generation differs from external anchor"),
            Self::AnchorHeadMismatch => write!(f, "store head differs from external anchor"),
            Self::StoreGenerationMismatch => write!(f, "store generation does not match record count"),
            Self::SequenceMismatch => write!(f, "record sequence is not monotonic and contiguous"),
            Self::PreviousMacMismatch => write!(f, "record previous-MAC link is inconsistent"),
            Self::RecordMacMismatch => write!(f, "record authentication code is invalid"),
            Self::DuplicateRecordId => write!(f, "record identifier is duplicated"),
            Self::ErasureResurrection(id) => write!(f, "erased record {id} reappeared in restored store"),
        }
    }
}

fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    let mut hex = String::with_capacity(64);
    for byte in digest {
        write!(&mut hex, "{byte:02x}").expect("writing to String cannot fail");
    }
    format!("sha256:{hex}")
}

// HMAC-SHA256 is implemented here only to avoid adding a crypto dependency to
// this synthetic lab. LAB_KEY is a test vector, not a deployment secret store.
fn hmac_sha256(key: &[u8], message: &[u8]) -> String {
    const BLOCK: usize = 64;
    let mut normalized = [0u8; BLOCK];
    if key.len() > BLOCK {
        let digest = Sha256::digest(key);
        normalized[..digest.len()].copy_from_slice(&digest);
    } else {
        normalized[..key.len()].copy_from_slice(key);
    }

    let mut inner = Vec::with_capacity(BLOCK + message.len());
    let mut outer = Vec::with_capacity(BLOCK + 32);
    for byte in normalized {
        inner.push(byte ^ 0x36);
        outer.push(byte ^ 0x5c);
    }
    inner.extend_from_slice(message);
    let inner_digest = Sha256::digest(&inner);
    outer.extend_from_slice(&inner_digest);
    sha256_hex(&outer)
}

fn record_mac(record: &Record) -> Result<String, String> {
    let payload = RecordPayload {
        context_id: &record.context_id,
        record_id: &record.record_id,
        sequence: record.sequence,
        owner: &record.owner,
        role: &record.role,
        source: &record.source,
        timestamp: &record.timestamp,
        fact: &record.fact,
        previous_mac: &record.previous_mac,
    };
    let mut bytes = RECORD_DOMAIN.as_bytes().to_vec();
    bytes.extend(serde_json::to_vec(&payload).map_err(|error| error.to_string())?);
    Ok(hmac_sha256(LAB_KEY, &bytes))
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let parent = path.parent().ok_or_else(|| format!("no parent for {}", path.display()))?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let tmp = path.with_extension("tmp");
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    fs::write(&tmp, bytes).map_err(|error| error.to_string())?;
    fs::rename(&tmp, path).map_err(|error| error.to_string())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn store_bytes(paths: &LabPaths) -> Result<Vec<u8>, String> {
    fs::read(&paths.store).map_err(|error| format!("read store snapshot: {error}"))
}

fn restore_store_bytes(paths: &LabPaths, bytes: &[u8]) -> Result<(), String> {
    let parent = paths.store.parent().ok_or_else(|| "store path has no parent".to_owned())?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let tmp = paths.store.with_extension("restore.tmp");
    fs::write(&tmp, bytes).map_err(|error| error.to_string())?;
    fs::rename(&tmp, &paths.store).map_err(|error| error.to_string())
}

impl Sut {
    fn open(paths: LabPaths, mode: ProfileMode) -> Self {
        Self { paths, mode }
    }

    fn initialize(paths: &LabPaths, mode: ProfileMode) -> Result<(), String> {
        fs::create_dir_all(paths.store.parent().unwrap()).map_err(|error| error.to_string())?;
        fs::create_dir_all(paths.anchor.parent().unwrap()).map_err(|error| error.to_string())?;
        fs::create_dir_all(&paths.results).map_err(|error| error.to_string())?;

        let store = StoreFile {
            generation: 0,
            records: Vec::new(),
        };
        write_json_atomic(&paths.store, &store)?;
        write_json_atomic(
            &paths.erasure,
            &ErasureLedger {
                erased_record_ids: Vec::new(),
            },
        )?;
        if mode != ProfileMode::Vulnerable {
            write_json_atomic(
                &paths.anchor,
                &IntegrityAnchor {
                    generation: 0,
                    head_mac: GENESIS.to_owned(),
                },
            )?;
        }
        Ok(())
    }

    fn load_store(&self) -> Result<StoreFile, String> {
        read_json(&self.paths.store)
    }

    fn write_record(
        &self,
        context_id: &str,
        record_id: &str,
        owner: &str,
        fact: &str,
    ) -> Result<(), String> {
        let mut store = self.load_store()?;
        let sequence = store.generation + 1;
        let previous_mac = store
            .records
            .last()
            .map(|record| record.record_mac.clone())
            .unwrap_or_else(|| GENESIS.to_owned());

        let mut record = Record {
            context_id: context_id.to_owned(),
            record_id: record_id.to_owned(),
            sequence,
            owner: owner.to_owned(),
            role: "memory".to_owned(),
            source: "sut-write-path".to_owned(),
            timestamp: format!("synthetic-sequence-{sequence}"),
            fact: fact.to_owned(),
            previous_mac,
            record_mac: String::new(),
        };
        if self.mode != ProfileMode::Vulnerable {
            record.record_mac = record_mac(&record)?;
        }

        store.records.push(record);
        store.generation = sequence;
        write_json_atomic(&self.paths.store, &store)?;

        if self.mode != ProfileMode::Vulnerable {
            let head_mac = store
                .records
                .last()
                .map(|record| record.record_mac.clone())
                .unwrap_or_else(|| GENESIS.to_owned());
            write_json_atomic(
                &self.paths.anchor,
                &IntegrityAnchor {
                    generation: store.generation,
                    head_mac,
                },
            )?;
        }
        Ok(())
    }

    fn seed(&self) -> Result<(), String> {
        for index in 0..RECORDS_PER_CONTEXT {
            self.write_record(
                "A",
                &format!("A-{index}"),
                "user-A",
                &format!("FACT-A-{index}"),
            )?;
            self.write_record(
                "B",
                &format!("B-{index}"),
                "user-B",
                &format!("FACT-B-{index}"),
            )?;
        }
        Ok(())
    }

    fn read_context(&self, context_id: &str) -> Result<Vec<Record>, String> {
        if self.mode == ProfileMode::ProtectedRead {
            self.verify_integrity()?;
        }
        let store = self.load_store()?;
        Ok(store
            .records
            .into_iter()
            .filter(|record| record.context_id == context_id)
            .collect())
    }

    fn audit(&self) -> Result<(), String> {
        if self.mode == ProfileMode::Vulnerable {
            return Ok(());
        }
        self.verify_integrity()
    }

    fn verify_integrity(&self) -> Result<(), String> {
        let store = self.load_store()?;
        let anchor: IntegrityAnchor = read_json(&self.paths.anchor)
            .map_err(|_| IntegrityFailure::MissingAnchor.to_string())?;

        if anchor.generation != store.generation {
            return Err(IntegrityFailure::AnchorGenerationMismatch.to_string());
        }
        if store.generation as usize != store.records.len() {
            return Err(IntegrityFailure::StoreGenerationMismatch.to_string());
        }
        let head_mac = store
            .records
            .last()
            .map(|record| record.record_mac.clone())
            .unwrap_or_else(|| GENESIS.to_owned());
        if anchor.head_mac != head_mac {
            return Err(IntegrityFailure::AnchorHeadMismatch.to_string());
        }

        let mut previous_mac = GENESIS.to_owned();
        let mut record_ids = HashSet::new();
        for (index, record) in store.records.iter().enumerate() {
            if !record_ids.insert(record.record_id.as_str()) {
                return Err(IntegrityFailure::DuplicateRecordId.to_string());
            }
            if record.sequence != (index as u64) + 1 {
                return Err(IntegrityFailure::SequenceMismatch.to_string());
            }
            if record.previous_mac != previous_mac {
                return Err(IntegrityFailure::PreviousMacMismatch.to_string());
            }
            let expected_mac = record_mac(record)?;
            if record.record_mac != expected_mac {
                return Err(IntegrityFailure::RecordMacMismatch.to_string());
            }
            previous_mac = record.record_mac.clone();
        }

        let ledger: ErasureLedger = read_json(&self.paths.erasure)
            .map_err(|error| format!("read erasure ledger: {error}"))?;
        for erased_id in ledger.erased_record_ids {
            if store.records.iter().any(|record| record.record_id == erased_id) {
                return Err(IntegrityFailure::ErasureResurrection(erased_id).to_string());
            }
        }
        Ok(())
    }

    fn erase_record(&self, record_id: &str) -> Result<bool, String> {
        let mut store = self.load_store()?;
        let exists = store.records.iter().any(|record| record.record_id == record_id);
        if !exists {
            return Ok(false);
        }

        if self.mode == ProfileMode::Vulnerable {
            store.records.retain(|record| record.record_id != record_id);
            write_json_atomic(&self.paths.store, &store)?;
        } else {
            // This synthetic design models an erasure tombstone held outside
            // the store-only rollback domain. A restored backup cannot delete it.
            let mut ledger: ErasureLedger = read_json(&self.paths.erasure)?;
            if !ledger.erased_record_ids.iter().any(|id| id == record_id) {
                ledger.erased_record_ids.push(record_id.to_owned());
                ledger.erased_record_ids.sort();
            }
            write_json_atomic(&self.paths.erasure, &ledger)?;
        }
        Ok(true)
    }
}

#[derive(Debug, Clone, Copy)]
enum TestCase {
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    T8,
    T9,
    T10,
}

impl TestCase {
    fn all() -> [Self; 10] {
        [
            Self::T1,
            Self::T2,
            Self::T3,
            Self::T4,
            Self::T5,
            Self::T6,
            Self::T7,
            Self::T8,
            Self::T9,
            Self::T10,
        ]
    }

    fn id(self) -> &'static str {
        match self {
            Self::T1 => "T1",
            Self::T2 => "T2",
            Self::T3 => "T3",
            Self::T4 => "T4",
            Self::T5 => "T5",
            Self::T6 => "T6",
            Self::T7 => "T7",
            Self::T8 => "T8",
            Self::T9 => "T9",
            Self::T10 => "T10",
        }
    }

    fn context(self) -> &'static str {
        match self {
            Self::T2 | Self::T6 => "B",
            _ => "A",
        }
    }

}

fn load_store_file(paths: &LabPaths) -> Result<StoreFile, String> {
    read_json(&paths.store)
}

fn save_store_file(paths: &LabPaths, store: &StoreFile) -> Result<(), String> {
    write_json_atomic(&paths.store, store)
}

fn apply_case_mutation(paths: &LabPaths, mode: ProfileMode, case: TestCase, before: &[u8]) -> Result<bool, String> {
    let mut store = load_store_file(paths)?;
    let original = store.clone();

    match case {
        TestCase::T1 => {
            let record = store.records.iter_mut().find(|record| record.record_id == "A-2").ok_or("missing A-2")?;
            record.fact = "ATTACKER-CHANGED-FACT".to_owned();
            save_store_file(paths, &store)?;
            Ok(load_store_file(paths)? == store && store != original)
        }
        TestCase::T2 => {
            let removed = store.records.pop().ok_or("empty store")?;
            if removed.record_id != "B-4" {
                return Err("T2 expected B-4 to be the tail record".to_owned());
            }
            save_store_file(paths, &store)?;
            Ok(load_store_file(paths)? == store && store.records.len() + 1 == original.records.len())
        }
        TestCase::T3 => {
            let index = store.records.iter().position(|record| record.record_id == "A-2").ok_or("missing A-2")?;
            if index == 0 || index + 1 == store.records.len() {
                return Err("T3 target is not a middle record".to_owned());
            }
            store.records.remove(index);
            save_store_file(paths, &store)?;
            Ok(load_store_file(paths)? == store && store.records.len() + 1 == original.records.len())
        }
        TestCase::T4 => {
            let first = store.records.iter().position(|record| record.record_id == "A-1").ok_or("missing A-1")?;
            let second = store.records.iter().position(|record| record.record_id == "A-2").ok_or("missing A-2")?;
            let first_sequence = store.records[first].sequence;
            store.records[first].sequence = store.records[second].sequence;
            store.records[second].sequence = first_sequence;
            save_store_file(paths, &store)?;
            Ok(load_store_file(paths)? == store
                && store.records[first].sequence == original.records[second].sequence
                && store.records[second].sequence == original.records[first].sequence)
        }
        TestCase::T5 => {
            let previous = store.records.last().map(|record| record.record_mac.clone()).unwrap_or_else(|| GENESIS.to_owned());
            store.records.push(Record {
                context_id: "A".to_owned(),
                record_id: "FORGED-1".to_owned(),
                sequence: store.generation + 1,
                owner: "attacker".to_owned(),
                role: "memory".to_owned(),
                source: "attacker-insert".to_owned(),
                timestamp: "synthetic-forged-time".to_owned(),
                fact: "FORGED-FACT".to_owned(),
                previous_mac: previous,
                record_mac: "sha256:forged".to_owned(),
            });
            save_store_file(paths, &store)?;
            Ok(load_store_file(paths)? == store
                && store.records.len() == original.records.len() + 1
                && store.records.last().is_some_and(|record| record.record_id == "FORGED-1"))
        }
        TestCase::T6 => {
            let source_fact = store.records.iter().find(|record| record.record_id == "A-2").ok_or("missing A-2")?.fact.clone();
            let target = store.records.iter_mut().find(|record| record.record_id == "B-2").ok_or("missing B-2")?;
            let old_context = target.context_id.clone();
            let old_id = target.record_id.clone();
            target.fact = source_fact.clone();
            save_store_file(paths, &store)?;
            let loaded = load_store_file(paths)?;
            let target = loaded.records.iter().find(|record| record.record_id == "B-2").ok_or("missing mutated B-2")?;
            Ok(loaded == store && target.fact == source_fact && target.context_id == old_context && target.record_id == old_id)
        }
        TestCase::T7 => {
            let old = store.records.iter().find(|record| record.record_id == "A-3").ok_or("missing A-3")?.clone();
            let latest = store.records.iter().position(|record| record.record_id == "A-4").ok_or("missing A-4")?;
            store.records[latest] = old;
            save_store_file(paths, &store)?;
            Ok(load_store_file(paths)? == store && store.records[latest].record_id == "A-3")
        }
        TestCase::T8 => {
            let target = store.records.iter_mut().find(|record| record.record_id == "A-1").ok_or("missing A-1")?;
            target.owner = "attacker-owner".to_owned();
            save_store_file(paths, &store)?;
            Ok(load_store_file(paths)? == store && store.records.iter().find(|record| record.record_id == "A-1").unwrap().owner == "attacker-owner")
        }
        TestCase::T9 => {
            let sut = Sut::open(paths.clone(), mode);
            sut.write_record("A", "A-POST-ROLLBACK", "user-A", "GENUINE-POST-SNAPSHOT-WRITE")?;
            let written = load_store_file(paths)?;
            if written.generation != original.generation + 1 || written.records.len() != original.records.len() + 1 {
                return Ok(false);
            }
            restore_store_bytes(paths, before)?;
            Ok(store_bytes(paths)? == before)
        }
        TestCase::T10 => {
            let sut = Sut::open(paths.clone(), mode);
            let erased = sut.erase_record("A-4")?;
            if !erased {
                return Ok(false);
            }
            let deletion_landed = match mode {
                ProfileMode::Vulnerable => {
                    let physically_removed = !load_store_file(paths)?.records.iter().any(|record| record.record_id == "A-4");
                    let read_path_reflects_erasure = !Sut::open(paths.clone(), mode)
                        .read_context("A")?
                        .iter()
                        .any(|record| record.record_id == "A-4");
                    physically_removed && read_path_reflects_erasure
                }
                ProfileMode::ProtectedRead => {
                    let ledger: ErasureLedger = read_json(&paths.erasure)?;
                    let tombstone_present = ledger.erased_record_ids.iter().any(|id| id == "A-4");
                    let read_path_blocks_resurrection =
                        Sut::open(paths.clone(), mode).read_context("A").is_err();
                    tombstone_present && read_path_blocks_resurrection
                }
                ProfileMode::ProtectedAudit => {
                    let ledger: ErasureLedger = read_json(&paths.erasure)?;
                    let tombstone_present = ledger.erased_record_ids.iter().any(|id| id == "A-4");
                    let audit_detects_resurrection = Sut::open(paths.clone(), mode).audit().is_err();
                    tombstone_present && audit_detects_resurrection
                }
            };
            if !deletion_landed {
                return Ok(false);
            }
            restore_store_bytes(paths, before)?;
            Ok(store_bytes(paths)? == before)
        }
    }
}

fn assess_case(sut: &Sut, case: TestCase, edit_landed: bool, c3_detail: String) -> CaseResult {
    let context = case.context().to_owned();
    if !edit_landed {
        return CaseResult {
            case_id: case.id().to_owned(),
            case_status: CaseStatus::Error,
            edit_landed: false,
            native_verdict: NativeVerdict::Error,
            detection_point: DetectionPoint::NotApplicable,
            observation_point_class: ObservationPointClass::NotApplicable,
            affected_context: context,
            c3_detail,
            result_detail: "C3 edit-landed verification failed; case excluded from the native denominator".to_owned(),
            result_artifact_ref: format!("results/{}.json", case.id()),
        };
    }

    let read_result = sut.read_context(&context);
    let (native_verdict, detection_point, observation_point_class, detail) = match sut.mode {
        ProfileMode::ProtectedRead => match read_result {
            Ok(_) => (
                NativeVerdict::Accepted,
                DetectionPoint::None,
                ObservationPointClass::A,
                "read path served the affected context without an integrity signal".to_owned(),
            ),
            Err(error) => (
                NativeVerdict::Rejected,
                DetectionPoint::Read,
                ObservationPointClass::A,
                format!("read path rejected affected context: {error}"),
            ),
        },
        ProfileMode::ProtectedAudit => {
            let served = read_result.is_ok();
            match sut.audit() {
                Ok(()) => (
                    NativeVerdict::Accepted,
                    DetectionPoint::None,
                    ObservationPointClass::A,
                    if served {
                        "read path served affected context; audit reported no integrity issue".to_owned()
                    } else {
                        "read path failed without a qualifying audit signal".to_owned()
                    },
                ),
                Err(error) => (
                    NativeVerdict::Reported,
                    DetectionPoint::Audit,
                    ObservationPointClass::B,
                    format!("read path served affected context; subsequent audit reported: {error}"),
                ),
            }
        }
        ProfileMode::Vulnerable => (
            NativeVerdict::Accepted,
            DetectionPoint::None,
            ObservationPointClass::A,
            match read_result {
                Ok(_) => "read path served affected context without an integrity signal".to_owned(),
                Err(error) => format!("read failed without a qualifying integrity signal: {error}"),
            },
        ),
    };

    CaseResult {
        case_id: case.id().to_owned(),
        case_status: CaseStatus::Applicable,
        edit_landed: true,
        native_verdict,
        detection_point,
        observation_point_class,
        affected_context: context,
        c3_detail,
        result_detail: detail,
        result_artifact_ref: format!("results/{}.json", case.id()),
    }
}

fn run_control_c1(paths: &LabPaths, mode: ProfileMode) -> Result<ControlResult, String> {
    Sut::initialize(paths, mode)?;
    Sut::open(paths.clone(), mode).seed()?;

    // The SUT object is dropped here. The read below constructs a new object,
    // modelling stop/restart with no store edit between the two processes.
    let restarted = Sut::open(paths.clone(), mode);
    let a = restarted.read_context("A")?;
    let b = restarted.read_context("B")?;
    restarted.audit()?;
    let passed = a.len() == RECORDS_PER_CONTEXT
        && b.len() == RECORDS_PER_CONTEXT
        && a.iter().any(|record| record.fact == "FACT-A-0")
        && b.iter().any(|record| record.fact == "FACT-B-0");
    Ok(ControlResult {
        case_id: "C1".to_owned(),
        passed,
        detail: if passed {
            "no-op stop/restart served both seeded contexts unchanged without an integrity alert".to_owned()
        } else {
            "no-op stop/restart failed to serve the seeded state unchanged".to_owned()
        },
        result_artifact_ref: "results/C1.json".to_owned(),
    })
}

fn run_control_c2(paths: &LabPaths, mode: ProfileMode) -> Result<ControlResult, String> {
    Sut::initialize(paths, mode)?;
    Sut::open(paths.clone(), mode).seed()?;

    let after_restart = Sut::open(paths.clone(), mode);
    after_restart.write_record("A", "A-POST-RESTART", "user-A", "GENUINE-POST-RESTART-WRITE")?;

    // Re-open once more to ensure the read path observes persisted state.
    let reader = Sut::open(paths.clone(), mode);
    let a = reader.read_context("A")?;
    reader.audit()?;
    let passed = a.iter().any(|record| {
        record.record_id == "A-POST-RESTART" && record.fact == "GENUINE-POST-RESTART-WRITE"
    });
    Ok(ControlResult {
        case_id: "C2".to_owned(),
        passed,
        detail: if passed {
            "genuine post-restart write was persisted and served through the SUT read path".to_owned()
        } else {
            "genuine post-restart write was not served".to_owned()
        },
        result_artifact_ref: "results/C2.json".to_owned(),
    })
}

fn score_cases(cases: &[CaseResult]) -> NativeScore {
    let evaluated: Vec<_> = cases
        .iter()
        .filter(|case| case.case_status == CaseStatus::Applicable && case.edit_landed)
        .collect();

    let count = |predicate: &dyn Fn(&CaseResult) -> bool| {
        evaluated.iter().filter(|case| predicate(case)).count()
    };

    let rejected = count(&|case| case.native_verdict == NativeVerdict::Rejected);
    let reported_at_read = count(&|case| {
        case.native_verdict == NativeVerdict::Reported && case.detection_point == DetectionPoint::Read
    });
    let reported_at_audit = count(&|case| {
        case.native_verdict == NativeVerdict::Reported && case.detection_point == DetectionPoint::Audit
    });
    let accepted = count(&|case| case.native_verdict == NativeVerdict::Accepted);
    let errors_excluded = cases.iter().filter(|case| case.case_status == CaseStatus::Error).count();
    let not_applicable_excluded = cases.iter().filter(|case| case.case_status == CaseStatus::NotApplicable).count();
    let denominator_n = evaluated.len();
    let score = (denominator_n > 0).then(|| {
        (rejected as f64 + reported_at_read as f64 + 0.5 * reported_at_audit as f64)
            / denominator_n as f64
    });

    let mut category_scores = BTreeMap::new();
    for category in ["T1-T5", "T6-T8", "T9", "T10"] {
        let rows: Vec<_> = evaluated
            .iter()
            .filter(|case| match case.case_id.as_str() {
                "T1" | "T2" | "T3" | "T4" | "T5" => category == "T1-T5",
                "T6" | "T7" | "T8" => category == "T6-T8",
                "T9" => category == "T9",
                "T10" => category == "T10",
                _ => false,
            })
            .collect();
        let points: f64 = rows
            .iter()
            .map(|case| match (case.native_verdict, case.detection_point) {
                (NativeVerdict::Rejected, _) => 1.0,
                (NativeVerdict::Reported, DetectionPoint::Read) => 1.0,
                (NativeVerdict::Reported, DetectionPoint::Audit) => 0.5,
                _ => 0.0,
            })
            .sum();
        category_scores.insert(category.to_owned(), (!rows.is_empty()).then(|| points / rows.len() as f64));
    }

    NativeScore {
        denominator_n,
        rejected,
        reported_at_read,
        reported_at_audit,
        accepted,
        errors_excluded,
        not_applicable_excluded,
        score,
        category_scores,
    }
}

fn save_json<T: Serialize>(root: &Path, relative: &str, value: &T) -> Result<(), String> {
    let path = root.join(relative);
    write_json_atomic(&path, value)
}

pub fn run_campaign(output_root: impl Into<PathBuf>, mode: ProfileMode) -> Result<CampaignReport, String> {
    let root = output_root.into();
    fs::create_dir_all(&root).map_err(|error| error.to_string())?;
    let control_root = root.join("controls");

    let c1 = run_control_c1(&LabPaths::new(control_root.join("C1")), mode)?;
    let c2 = run_control_c2(&LabPaths::new(control_root.join("C2")), mode)?;
    save_json(&root, &c1.result_artifact_ref, &c1)?;
    save_json(&root, &c2.result_artifact_ref, &c2)?;

    let mut cases = Vec::new();
    for case in TestCase::all() {
        let paths = LabPaths::new(root.join("cases").join(case.id()));
        Sut::initialize(&paths, mode)?;
        Sut::open(paths.clone(), mode).seed()?;
        let before = store_bytes(&paths)?;
        let edit_landed = apply_case_mutation(&paths, mode, case, &before)?;
        let actual = assess_case(
            &Sut::open(paths.clone(), mode),
            case,
            edit_landed,
            if edit_landed {
                "C3 confirmed the exact expected post-mutation or restored snapshot state".to_owned()
            } else {
                "C3 did not confirm the exact expected state".to_owned()
            },
        );
        save_json(&root, &actual.result_artifact_ref, &actual)?;
        cases.push(actual);
    }

    let campaign_status = if c1.passed && c2.passed {
        CampaignStatus::Evaluable
    } else {
        CampaignStatus::NotEvaluable
    };
    let mut native_score = score_cases(&cases);
    if campaign_status == CampaignStatus::NotEvaluable {
        native_score.score = None;
        for value in native_score.category_scores.values_mut() {
            *value = None;
        }
    }

    let report = CampaignReport {
        schema_version: "sat-memory-at-rest-reference-lab/v0.1".to_owned(),
        campaign_id: format!(
            "SAT-MAR-LAB-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|error| error.to_string())?
                .as_nanos()
        ),
        profile_mode: mode,
        evidence_level: "synthetic_file_backed_reference_emulator".to_owned(),
        real_sut_evaluated: false,
        campaign_status,
        c1_no_op_reload: c1,
        c2_genuine_write_after_restart: c2,
        cases,
        native_score,
        limitations: vec![
            "This is a synthetic reference emulator, not a real Symthaea or third-party agent memory backend.".to_owned(),
            "The protected profile illustrates record HMAC, an external rollback anchor and an external erasure tombstone; it is not a production key-management design.".to_owned(),
            "The native score applies only to this deterministic synthetic emulator and must not be presented as an upstream benchmark result.".to_owned(),
            "No independently operated evaluator, sealed holdout, host-level attacker or physical storage system was used.".to_owned(),
        ],
    };

    write_json_atomic(&root.join("campaign-report.json"), &report)?;
    Ok(report)
}

pub fn fresh_temp_dir(label: &str) -> Result<PathBuf, String> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let path = std::env::temp_dir().join(format!("sat-memory-lab-{label}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).map_err(|error| error.to_string())?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn campaign(mode: ProfileMode) -> CampaignReport {
        let root = fresh_temp_dir("test").unwrap();
        let report = run_campaign(&root, mode).unwrap();
        let _ = fs::remove_dir_all(root);
        report
    }

    #[test]
    fn protected_read_mode_passes_all_ten_mutations_at_read_time() {
        let report = campaign(ProfileMode::ProtectedRead);
        assert_eq!(report.campaign_status, CampaignStatus::Evaluable);
        assert!(report.c1_no_op_reload.passed);
        assert!(report.c2_genuine_write_after_restart.passed);
        assert_eq!(report.native_score.denominator_n, 10);
        assert_eq!(report.native_score.score, Some(1.0));
        assert!(report.cases.iter().all(|case| {
            case.case_status == CaseStatus::Applicable
                && case.edit_landed
                && case.native_verdict == NativeVerdict::Rejected
                && case.detection_point == DetectionPoint::Read
        }));
    }

    #[test]
    fn vulnerable_mode_accepts_all_ten_mutations() {
        let report = campaign(ProfileMode::Vulnerable);
        assert_eq!(report.campaign_status, CampaignStatus::Evaluable);
        assert_eq!(report.native_score.denominator_n, 10);
        assert_eq!(report.native_score.score, Some(0.0));
        assert!(report.cases.iter().all(|case| case.native_verdict == NativeVerdict::Accepted));
    }

    #[test]
    fn audit_only_mode_receives_partial_credit() {
        let report = campaign(ProfileMode::ProtectedAudit);
        assert_eq!(report.campaign_status, CampaignStatus::Evaluable);
        assert_eq!(report.native_score.denominator_n, 10);
        assert_eq!(report.native_score.score, Some(0.5));
        assert!(report.cases.iter().all(|case| {
            case.native_verdict == NativeVerdict::Reported
                && case.detection_point == DetectionPoint::Audit
                && case.observation_point_class == ObservationPointClass::B
        }));
    }

    #[test]
    fn t9_rollback_is_detected_by_anchor_outside_store_bundle() {
        let report = campaign(ProfileMode::ProtectedRead);
        let t9 = report.cases.iter().find(|case| case.case_id == "T9").unwrap();
        assert!(t9.edit_landed);
        assert_eq!(t9.native_verdict, NativeVerdict::Rejected);
    }

    #[test]
    fn t10_erasure_resurrection_is_blocked_by_external_tombstone() {
        let report = campaign(ProfileMode::ProtectedRead);
        let t10 = report.cases.iter().find(|case| case.case_id == "T10").unwrap();
        assert!(t10.edit_landed);
        assert_eq!(t10.native_verdict, NativeVerdict::Rejected);
    }

    #[test]
    fn every_native_edit_has_a_c3_result_and_is_denominator_eligible() {
        let report = campaign(ProfileMode::ProtectedRead);
        assert_eq!(report.cases.len(), 10);
        assert!(report.cases.iter().all(|case| case.edit_landed && case.case_status == CaseStatus::Applicable));
    }

    #[test]
    fn native_score_keeps_t9_and_t10_separate() {
        let report = campaign(ProfileMode::ProtectedRead);
        assert_eq!(report.native_score.category_scores.get("T9"), Some(&Some(1.0)));
        assert_eq!(report.native_score.category_scores.get("T10"), Some(&Some(1.0)));
    }
}
