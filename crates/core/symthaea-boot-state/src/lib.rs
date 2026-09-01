// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persistent lifecycle state for Spore boot ecology.
//!
//! This crate keeps the persistent facts required to derive the next boot's
//! `BootStateReceipt`. It deliberately does not parse user journals, process
//! names, filenames, or other personal content. The system integration supplies
//! coarse operational facts and this crate persists them atomically.

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use symthaea_boot_ecology::{
    BootEcologyComposer, BootOutcome, BootStateReceipt, GenerationHealth,
    GenerationTransition, MorphologyLineage, PreviousTermination, StorageState,
};

pub const STATE_SCHEMA_VERSION: u16 = 1;
pub const STATE_FILE: &str = "state.json";
pub const LINEAGE_FILE: &str = "lineage.json";
pub const CLEAN_MARKER_FILE: &str = "clean-shutdown.json";
pub const VISUAL_SEED_FILE: &str = "visual-seed.bin";
pub const RECEIPT_FILE: &str = "boot-state.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PersistentBootState {
    pub schema_version: u16,
    pub boot_counter: u64,
    pub active_generation: Option<String>,
    pub last_boot_blessed: bool,
    pub previous_hardware_fingerprint: Option<String>,
    pub last_boot_started_unix_ms: Option<u64>,
    pub last_boot_ended_unix_ms: Option<u64>,
}

impl Default for PersistentBootState {
    fn default() -> Self {
        Self {
            schema_version: STATE_SCHEMA_VERSION,
            boot_counter: 0,
            active_generation: None,
            last_boot_blessed: false,
            previous_hardware_fingerprint: None,
            last_boot_started_unix_ms: None,
            last_boot_ended_unix_ms: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CleanShutdownMarker {
    pub schema_version: u16,
    pub termination: PreviousTermination,
    pub uptime_secs: u64,
    pub generation: Option<String>,
    pub hardware_fingerprint: Option<String>,
    pub written_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrepareInput {
    pub current_generation: String,
    pub hardware_fingerprint: Option<String>,
    pub storage_state: StorageState,
    pub oom_events: u32,
    pub thermal_events: u32,
    pub mesh_enabled: bool,
    pub mesh_peers_last_seen: u32,
}

impl PrepareInput {
    pub fn minimal(current_generation: impl Into<String>) -> Self {
        Self {
            current_generation: current_generation.into(),
            hardware_fingerprint: None,
            storage_state: StorageState::Unknown,
            oom_events: 0,
            thermal_events: 0,
            mesh_enabled: false,
            mesh_peers_last_seen: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrepareResult {
    pub receipt: BootStateReceipt,
    pub lineage: MorphologyLineage,
    pub inferred_interrupted_previous_boot: bool,
}

#[derive(Debug, Clone)]
pub struct BootStateStore {
    persistent_dir: PathBuf,
    runtime_dir: PathBuf,
}

impl BootStateStore {
    pub fn new(persistent_dir: impl Into<PathBuf>, runtime_dir: impl Into<PathBuf>) -> Self {
        Self {
            persistent_dir: persistent_dir.into(),
            runtime_dir: runtime_dir.into(),
        }
    }

    pub fn persistent_dir(&self) -> &Path {
        &self.persistent_dir
    }

    pub fn runtime_dir(&self) -> &Path {
        &self.runtime_dir
    }

    pub fn receipt_path(&self) -> PathBuf {
        self.runtime_dir.join(RECEIPT_FILE)
    }

    pub fn lineage_path(&self) -> PathBuf {
        self.persistent_dir.join(LINEAGE_FILE)
    }

    pub fn state_path(&self) -> PathBuf {
        self.persistent_dir.join(STATE_FILE)
    }

    pub fn clean_marker_path(&self) -> PathBuf {
        self.persistent_dir.join(CLEAN_MARKER_FILE)
    }

    pub fn ensure_dirs(&self) -> Result<(), String> {
        fs::create_dir_all(&self.persistent_dir)
            .map_err(|e| format!("create {}: {e}", self.persistent_dir.display()))?;
        fs::create_dir_all(&self.runtime_dir)
            .map_err(|e| format!("create {}: {e}", self.runtime_dir.display()))?;
        Ok(())
    }

    pub fn prepare(&self, input: &PrepareInput) -> Result<PrepareResult, String> {
        self.ensure_dirs()?;

        let mut state = self.load_state()?;
        let lineage = self.load_lineage()?;
        let visual_seed = self.load_or_create_visual_seed()?;
        let clean_marker = self.load_clean_marker()?;

        let is_first_boot = state.active_generation.is_none() && state.boot_counter == 0;
        let inferred_interrupted_previous_boot = !is_first_boot && clean_marker.is_none();
        let previous_termination = if is_first_boot {
            PreviousTermination::FirstBoot
        } else if let Some(marker) = &clean_marker {
            marker.termination.clone()
        } else {
            // We know the previous session ended without our clean marker, but we
            // do not pretend to know whether that was power loss, panic, reset,
            // or another interruption. Recovery is expressed through health.
            PreviousTermination::Unknown
        };

        let previous_uptime_secs = clean_marker.as_ref().map(|m| m.uptime_secs).unwrap_or(0);
        let generation_transition = generation_transition(
            state.active_generation.as_deref(),
            &input.current_generation,
            lineage.last_known_good_generation.as_deref(),
            state.last_boot_blessed,
        );

        let generation_health = if inferred_interrupted_previous_boot {
            GenerationHealth::Recovery
        } else if lineage.last_known_good_generation.as_deref()
            == Some(input.current_generation.as_str())
        {
            GenerationHealth::KnownGood
        } else if !state.last_boot_blessed && state.active_generation.is_some() {
            GenerationHealth::PreviousBootIncomplete
        } else {
            GenerationHealth::Unknown
        };

        let receipt = BootStateReceipt {
            schema_version: symthaea_boot_ecology::BOOT_ECOLOGY_SCHEMA_VERSION,
            machine_visual_seed: visual_seed,
            boot_counter: state.boot_counter,
            previous_termination,
            previous_uptime_secs,
            generation_transition,
            generation_health,
            storage_state: input.storage_state.clone(),
            oom_events: input.oom_events,
            thermal_events: input.thermal_events,
            hardware_fingerprint: input.hardware_fingerprint.clone(),
            previous_hardware_fingerprint: state.previous_hardware_fingerprint.clone(),
            mesh_enabled: input.mesh_enabled,
            mesh_peers_last_seen: input.mesh_peers_last_seen,
        };

        write_json_atomic(&self.receipt_path(), &receipt)?;
        // Keep a runtime copy of the lineage paired with the receipt so preview
        // and live rendering consume a consistent snapshot.
        write_json_atomic(&self.runtime_dir.join(LINEAGE_FILE), &lineage)?;

        state.boot_counter = state.boot_counter.saturating_add(1);
        state.active_generation = Some(input.current_generation.clone());
        state.last_boot_blessed = false;
        state.previous_hardware_fingerprint = input.hardware_fingerprint.clone();
        state.last_boot_started_unix_ms = Some(now_unix_ms());
        state.last_boot_ended_unix_ms = None;
        write_json_atomic(&self.state_path(), &state)?;

        // Consuming the marker makes a missing marker on the next boot
        // meaningful. Ignore a race where a marker disappeared after read.
        match fs::remove_file(self.clean_marker_path()) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("remove clean marker: {e}")),
        }

        Ok(PrepareResult {
            receipt,
            lineage,
            inferred_interrupted_previous_boot,
        })
    }

    pub fn bless(&self, generation: &str) -> Result<MorphologyLineage, String> {
        self.ensure_dirs()?;
        let receipt: BootStateReceipt = read_json(&self.receipt_path())?
            .ok_or_else(|| format!("missing runtime receipt {}", self.receipt_path().display()))?;

        // Validate authority before mutating lineage. A stale or mismatched
        // health-gate invocation must never advance Last Known Good.
        let mut state = self.load_state()?;
        if state.active_generation.as_deref() != Some(generation) {
            return Err(format!(
                "refusing to bless generation {generation}; active generation is {:?}",
                state.active_generation
            ));
        }

        let mut lineage = self.load_lineage()?;
        let genome = BootEcologyComposer::compose(&receipt, &lineage);
        lineage.record_outcome(
            &genome,
            &BootOutcome::ReachedGraphicalTarget {
                generation: generation.to_string(),
            },
        );
        write_json_atomic(&self.lineage_path(), &lineage)?;

        state.last_boot_blessed = true;
        write_json_atomic(&self.state_path(), &state)?;
        Ok(lineage)
    }

    pub fn mark_shutdown(
        &self,
        termination: PreviousTermination,
        uptime_secs: u64,
        generation: Option<String>,
        hardware_fingerprint: Option<String>,
    ) -> Result<(), String> {
        if !matches!(
            termination,
            PreviousTermination::CleanPoweroff
                | PreviousTermination::CleanReboot
                | PreviousTermination::Suspend
                | PreviousTermination::Hibernate
        ) {
            return Err("shutdown marker only accepts clean lifecycle terminations".into());
        }
        self.ensure_dirs()?;
        let marker = CleanShutdownMarker {
            schema_version: STATE_SCHEMA_VERSION,
            termination,
            uptime_secs,
            generation,
            hardware_fingerprint,
            written_unix_ms: now_unix_ms(),
        };
        write_json_atomic(&self.clean_marker_path(), &marker)?;

        let mut state = self.load_state()?;
        state.last_boot_ended_unix_ms = Some(marker.written_unix_ms);
        write_json_atomic(&self.state_path(), &state)
    }

    pub fn load_state(&self) -> Result<PersistentBootState, String> {
        Ok(read_json(&self.state_path())?.unwrap_or_default())
    }

    pub fn load_lineage(&self) -> Result<MorphologyLineage, String> {
        Ok(read_json(&self.lineage_path())?.unwrap_or_default())
    }

    fn load_clean_marker(&self) -> Result<Option<CleanShutdownMarker>, String> {
        read_json(&self.clean_marker_path())
    }

    fn load_or_create_visual_seed(&self) -> Result<[u8; 32], String> {
        let path = self.persistent_dir.join(VISUAL_SEED_FILE);
        if path.exists() {
            let mut bytes = [0u8; 32];
            let mut file = File::open(&path)
                .map_err(|e| format!("open visual seed {}: {e}", path.display()))?;
            file.read_exact(&mut bytes)
                .map_err(|e| format!("read visual seed {}: {e}", path.display()))?;
            return Ok(bytes);
        }

        let mut bytes = [0u8; 32];
        let mut random = File::open("/dev/urandom")
            .map_err(|e| format!("open /dev/urandom: {e}"))?;
        random
            .read_exact(&mut bytes)
            .map_err(|e| format!("read /dev/urandom: {e}"))?;
        write_bytes_atomic(&path, &bytes)?;
        Ok(bytes)
    }
}

fn generation_transition(
    previous_generation: Option<&str>,
    current_generation: &str,
    last_known_good: Option<&str>,
    previous_boot_blessed: bool,
) -> GenerationTransition {
    let Some(previous) = previous_generation else {
        return GenerationTransition::Unknown;
    };
    if previous == current_generation {
        return GenerationTransition::Same;
    }
    if !previous_boot_blessed && last_known_good == Some(current_generation) {
        return GenerationTransition::RolledBack {
            attempted: previous.to_string(),
            restored: current_generation.to_string(),
        };
    }
    GenerationTransition::Updated {
        from: previous.to_string(),
        to: current_generation.to_string(),
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<Option<T>, String> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|e| format!("parse {}: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("read {}: {e}", path.display())),
    }
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|e| format!("serialize {}: {e}", path.display()))?;
    write_bytes_atomic(path, &bytes)
}

fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    let tmp = parent.join(format!(
        ".{}.tmp-{}-{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("spore"),
        std::process::id(),
        now_unix_ms()
    ));
    {
        let mut file = File::create(&tmp)
            .map_err(|e| format!("create temporary {}: {e}", tmp.display()))?;
        file.write_all(bytes)
            .map_err(|e| format!("write temporary {}: {e}", tmp.display()))?;
        file.sync_all()
            .map_err(|e| format!("sync temporary {}: {e}", tmp.display()))?;
    }
    fs::rename(&tmp, path)
        .map_err(|e| format!("rename {} -> {}: {e}", tmp.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store(name: &str) -> BootStateStore {
        let root = std::env::temp_dir().join(format!(
            "spore-boot-state-{name}-{}-{}",
            std::process::id(),
            now_unix_ms()
        ));
        BootStateStore::new(root.join("persist"), root.join("run"))
    }

    #[test]
    fn first_prepare_is_germination_and_persists_seed() {
        let store = temp_store("first");
        let result = store.prepare(&PrepareInput::minimal("generation-1")).unwrap();
        assert_eq!(result.receipt.previous_termination, PreviousTermination::FirstBoot);
        assert!(!result.inferred_interrupted_previous_boot);
        let seed = result.receipt.machine_visual_seed;

        store
            .mark_shutdown(
                PreviousTermination::CleanReboot,
                42,
                Some("generation-1".into()),
                None,
            )
            .unwrap();
        let second = store.prepare(&PrepareInput::minimal("generation-1")).unwrap();
        assert_eq!(second.receipt.machine_visual_seed, seed);
        assert_eq!(second.receipt.previous_termination, PreviousTermination::CleanReboot);
        assert_eq!(second.receipt.previous_uptime_secs, 42);
    }

    #[test]
    fn missing_clean_marker_becomes_recovery_without_guessing_cause() {
        let store = temp_store("interrupted");
        store.prepare(&PrepareInput::minimal("generation-1")).unwrap();
        let next = store.prepare(&PrepareInput::minimal("generation-1")).unwrap();
        assert!(next.inferred_interrupted_previous_boot);
        assert_eq!(next.receipt.previous_termination, PreviousTermination::Unknown);
        assert_eq!(next.receipt.generation_health, GenerationHealth::Recovery);
        assert!(next.receipt.needs_repair_visual());
    }

    #[test]
    fn health_gate_is_required_before_known_good_advances() {
        let store = temp_store("bless");
        store.prepare(&PrepareInput::minimal("generation-1")).unwrap();
        assert!(store.load_lineage().unwrap().last_known_good_generation.is_none());
        let lineage = store.bless("generation-1").unwrap();
        assert_eq!(lineage.last_known_good_generation.as_deref(), Some("generation-1"));
        assert!(store.load_state().unwrap().last_boot_blessed);
    }

    #[test]
    fn rollback_is_inferred_when_unblessed_candidate_returns_to_known_good() {
        let store = temp_store("rollback");
        store.prepare(&PrepareInput::minimal("generation-1")).unwrap();
        store.bless("generation-1").unwrap();
        store
            .mark_shutdown(
                PreviousTermination::CleanReboot,
                100,
                Some("generation-1".into()),
                None,
            )
            .unwrap();
        store.prepare(&PrepareInput::minimal("generation-2")).unwrap();
        // No bless for generation-2: simulate failed candidate, then restore 1.
        let restored = store.prepare(&PrepareInput::minimal("generation-1")).unwrap();
        assert!(matches!(
            restored.receipt.generation_transition,
            GenerationTransition::RolledBack { ref attempted, ref restored }
                if attempted == "generation-2" && restored == "generation-1"
        ));
    }

    #[test]
    fn refuses_to_bless_generation_other_than_active_without_mutating_lineage() {
        let store = temp_store("wrong-bless");
        store.prepare(&PrepareInput::minimal("generation-1")).unwrap();
        let before = store.load_lineage().unwrap();
        assert!(store.bless("generation-2").is_err());
        let after = store.load_lineage().unwrap();
        assert_eq!(after, before);
        assert!(!store.load_state().unwrap().last_boot_blessed);
    }
}
