// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Epistemic Auditor — DuckDB-backed Consciousness Telemetry Audit Trail
//!
//! Buffers per-cycle [`AuditRecord`]s extracted from [`CycleMetadata`] and
//! periodically flushes them to DuckDB for retrospective analysis.
//!
//! ## Design
//!
//! - **Zero runtime overhead when disabled**: The entire module is gated behind
//!   the `epistemic_auditor` feature. When the feature is enabled but
//!   `CognitiveLoopConfig::epistemic_auditor_db_path` is `None`, the
//!   `Option<EpistemicAuditor>` field on `CognitiveLoopService` is `None`.
//! - **Non-blocking flush**: [`EpistemicAuditor::flush_background`] drains the
//!   in-memory buffer and spawns a `std::thread` (not tokio) to write rows.
//! - **AtomicBool guard**: Skips a new flush if a previous one is still
//!   in progress (prevents pile-up under slow I/O).
//! - **Cadence**: Flush every [`super::thresholds::EPISTEMIC_AUDITOR_FLUSH_CADENCE`]
//!   cycles (prime: 1009). Wired in `cycle_phase_output.rs`.
//!
//! ## Schema
//!
//! Six tables: `phi_trajectory`, `graduation_log`, `moral_audit`,
//! `neuromod_history`, `energy_audit`, `substrate_audit`.
//! Each row corresponds to one cognitive cycle.

#![cfg(feature = "epistemic_auditor")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

// ── Schema ────────────────────────────────────────────────────────────────────

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS phi_trajectory (
    cycle_id          UBIGINT NOT NULL,
    timestamp_us      BIGINT  NOT NULL,
    phi               DOUBLE  NOT NULL,
    consciousness_level DOUBLE NOT NULL,
    micro_phi         DOUBLE  NOT NULL,
    meso_phi          DOUBLE  NOT NULL,
    macro_phi         DOUBLE  NOT NULL,
    emergence_ratio   DOUBLE  NOT NULL,
    limiting_component TEXT   NOT NULL
);

CREATE TABLE IF NOT EXISTS graduation_log (
    cycle_id                   UBIGINT NOT NULL,
    timestamp_us               BIGINT  NOT NULL,
    graduations_processed      UBIGINT NOT NULL,
    graduations_rejected       UBIGINT NOT NULL,
    semantic_evictions         UBIGINT NOT NULL,
    episodic_memory_count      UINTEGER NOT NULL,
    episodic_avg_psi           DOUBLE  NOT NULL,
    codebook_diversity         FLOAT   NOT NULL,
    codebook_utilization_rate  FLOAT   NOT NULL,
    memory_db_flushed          BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS moral_audit (
    cycle_id                   UBIGINT  NOT NULL,
    timestamp_us               BIGINT   NOT NULL,
    moral_score                FLOAT    NOT NULL,
    moral_anomaly_score        DOUBLE   NOT NULL,
    topo_unity                 DOUBLE   NOT NULL,
    topo_completeness          DOUBLE   NOT NULL,
    topo_circularity           DOUBLE   NOT NULL,
    topo_free_energy           DOUBLE   NOT NULL,
    topo_beta_0                UINTEGER NOT NULL,
    topo_beta_1                UINTEGER NOT NULL,
    value_inversion            BOOLEAN  NOT NULL,
    free_energy_spike          BOOLEAN  NOT NULL,
    drift_alert                BOOLEAN  NOT NULL,
    fragmentation_increase     BOOLEAN  NOT NULL,
    trajectory_convergence     BOOLEAN  NOT NULL,
    convergence_severity       DOUBLE   NOT NULL,
    escalation_level           TEXT     NOT NULL,
    harmony_entropy            DOUBLE   NOT NULL,
    fingerprint_velocity       DOUBLE   NOT NULL
);

CREATE TABLE IF NOT EXISTS neuromod_history (
    cycle_id                   UBIGINT NOT NULL,
    timestamp_us               BIGINT  NOT NULL,
    dopamine_effective         FLOAT   NOT NULL,
    noradrenaline_effective    FLOAT   NOT NULL,
    serotonin_effective        FLOAT   NOT NULL,
    acetylcholine_effective    FLOAT   NOT NULL,
    gaba_effective             FLOAT   NOT NULL,
    oxytocin_effective         FLOAT   NOT NULL,
    glutamate_effective        FLOAT   NOT NULL,
    endocannabinoid_effective  FLOAT   NOT NULL,
    adenosine_effective        FLOAT   NOT NULL,
    da_phasic_burst            FLOAT   NOT NULL,
    ne_phasic_spike            FLOAT   NOT NULL,
    consciousness_mod          FLOAT   NOT NULL,
    allostatic_load            FLOAT   NOT NULL,
    ei_balance_ratio           FLOAT   NOT NULL,
    bath_entropy               FLOAT   NOT NULL,
    sleep_pressure             FLOAT   NOT NULL,
    circadian_hour             FLOAT   NOT NULL
);

CREATE TABLE IF NOT EXISTS energy_audit (
    cycle_id                   UBIGINT NOT NULL,
    timestamp_us               BIGINT  NOT NULL,
    total_energy_spent         DOUBLE  NOT NULL,
    energy_this_cycle          DOUBLE  NOT NULL,
    throughput_multiplier      FLOAT   NOT NULL,
    thermodynamic_load         FLOAT   NOT NULL,
    cycle_duration_us          UBIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS substrate_audit (
    cycle_id                   UBIGINT NOT NULL,
    timestamp_us               BIGINT  NOT NULL,
    feasibility_raw            DOUBLE  NOT NULL,
    honest_confidence          DOUBLE  NOT NULL,
    effective_feasibility      DOUBLE  NOT NULL,
    tau_factor                 FLOAT   NOT NULL,
    scale_pressure             FLOAT   NOT NULL,
    effective_dim_fraction     FLOAT   NOT NULL,
    transition_count           UINTEGER NOT NULL
);
";

// ── AuditRecord ───────────────────────────────────────────────────────────────

/// Compact telemetry snapshot extracted from one [`super::types::CycleMetadata`].
///
/// ~80 fields covering Phi, graduation, moral, neuromod, energy, and substrate
/// telemetry. Stored as one row per table per flush.
#[derive(Debug, Clone)]
pub struct AuditRecord {
    // ── Identity ──────────────────────────────────────────────────────────
    pub cycle_id: u64,
    /// Wall-clock timestamp in microseconds (from `std::time::SystemTime`).
    pub timestamp_us: i64,

    // ── Phi / Consciousness ────────────────────────────────────────────────
    pub phi: f64,
    pub consciousness_level: f64,
    pub micro_phi: f64,
    pub meso_phi: f64,
    pub macro_phi: f64,
    pub emergence_ratio: f64,
    pub limiting_component: String,

    // ── Graduation ────────────────────────────────────────────────────────
    pub graduations_processed: u64,
    pub graduations_rejected: u64,
    pub semantic_evictions: u64,
    pub episodic_memory_count: u32,
    pub episodic_avg_psi: f64,
    pub codebook_diversity: f32,
    pub codebook_utilization_rate: f32,
    pub memory_db_flushed: bool,

    // ── Moral ─────────────────────────────────────────────────────────────
    pub moral_score: f32,
    pub moral_anomaly_score: f64,
    pub topo_unity: f64,
    pub topo_completeness: f64,
    pub topo_circularity: f64,
    pub topo_free_energy: f64,
    pub topo_beta_0: u32,
    pub topo_beta_1: u32,
    pub value_inversion: bool,
    pub free_energy_spike: bool,
    pub drift_alert: bool,
    pub fragmentation_increase: bool,
    pub trajectory_convergence: bool,
    pub convergence_severity: f64,
    pub escalation_level: String,
    pub harmony_entropy: f64,
    pub fingerprint_velocity: f64,

    // ── Neuromod ──────────────────────────────────────────────────────────
    pub dopamine_effective: f32,
    pub noradrenaline_effective: f32,
    pub serotonin_effective: f32,
    pub acetylcholine_effective: f32,
    pub gaba_effective: f32,
    pub oxytocin_effective: f32,
    pub glutamate_effective: f32,
    pub endocannabinoid_effective: f32,
    pub adenosine_effective: f32,
    pub da_phasic_burst: f32,
    pub ne_phasic_spike: f32,
    pub consciousness_mod: f32,
    pub allostatic_load: f32,
    pub ei_balance_ratio: f32,
    pub bath_entropy: f32,
    pub sleep_pressure: f32,
    pub circadian_hour: f32,

    // ── Energy ────────────────────────────────────────────────────────────
    pub total_energy_spent: f64,
    pub energy_this_cycle: f64,
    pub throughput_multiplier: f32,
    pub thermodynamic_load: f32,
    pub cycle_duration_us: u64,

    // ── Substrate ─────────────────────────────────────────────────────────
    pub feasibility_raw: f64,
    pub honest_confidence: f64,
    pub effective_feasibility: f64,
    pub tau_factor: f32,
    pub scale_pressure: f32,
    pub effective_dim_fraction: f32,
    pub transition_count: u32,
}

impl AuditRecord {
    /// Extract an [`AuditRecord`] from a [`super::types::CycleMetadata`].
    pub fn from_metadata(cycle_id: u64, m: &super::types::CycleMetadata) -> Self {
        let timestamp_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0);

        // Escalation level as string
        let escalation_level = format!("{:?}", m.ethics.moral_escalation_level);

        Self {
            cycle_id,
            timestamp_us,

            // Phi / Consciousness
            phi: m.quality.equation_v2_consciousness,
            consciousness_level: m.consciousness.consciousness_level,
            micro_phi: m.structural.structural_micro_phi,
            meso_phi: m.structural.structural_meso_phi,
            macro_phi: m.structural.structural_macro_phi,
            emergence_ratio: m.structural.structural_emergence_ratio,
            limiting_component: m.eq_v2_limiting_component.clone(),

            // Graduation
            graduations_processed: m.memory.graduations_processed,
            graduations_rejected: m.memory.graduations_rejected,
            semantic_evictions: m.memory.semantic_evictions,
            episodic_memory_count: m.memory.episodic_memory_count as u32,
            episodic_avg_psi: m.memory.episodic_avg_psi,
            codebook_diversity: m.memory.codebook_diversity,
            codebook_utilization_rate: m.memory.codebook_utilization_rate,
            memory_db_flushed: m.memory.memory_db_flushed,

            // Moral
            moral_score: m.ethics.moral_score,
            moral_anomaly_score: m.ethics.moral_anomaly_score,
            topo_unity: m.ethics.moral_topo_unity,
            topo_completeness: m.ethics.moral_topo_completeness,
            topo_circularity: m.ethics.moral_topo_circularity,
            topo_free_energy: m.ethics.moral_topo_free_energy,
            topo_beta_0: m.ethics.moral_topo_beta_0 as u32,
            topo_beta_1: m.ethics.moral_topo_beta_1 as u32,
            value_inversion: m.ethics.moral_value_inversion,
            free_energy_spike: m.ethics.moral_free_energy_spike,
            drift_alert: m.ethics.moral_drift_alert,
            fragmentation_increase: m.ethics.moral_fragmentation_increase,
            trajectory_convergence: m.ethics.moral_trajectory_convergence,
            convergence_severity: m.ethics.moral_convergence_severity,
            escalation_level,
            harmony_entropy: m.ethics.harmony_entropy,
            fingerprint_velocity: m.ethics.moral_fingerprint_velocity,

            // Neuromod
            dopamine_effective: m.neuromod.dopamine_effective,
            noradrenaline_effective: m.neuromod.noradrenaline_effective,
            serotonin_effective: m.neuromod.serotonin_effective,
            acetylcholine_effective: m.neuromod.acetylcholine_effective,
            gaba_effective: m.neuromod.neuromod_gaba_effective,
            oxytocin_effective: m.neuromod.neuromod_oxytocin_effective,
            glutamate_effective: m.neuromod.neuromod_glutamate_effective,
            endocannabinoid_effective: m.neuromod.neuromod_endocannabinoid_effective,
            adenosine_effective: m.neuromod.neuromod_adenosine_effective,
            da_phasic_burst: m.neuromod.neuromod_da_phasic,
            ne_phasic_spike: m.neuromod.neuromod_ne_phasic,
            consciousness_mod: m.neuromod.neuromod_consciousness_mod,
            allostatic_load: m.neuromod.neuromod_allostatic_load,
            ei_balance_ratio: m.neuromod.neuromod_ei_ratio,
            bath_entropy: m.neuromod.neuromod_bath_entropy,
            sleep_pressure: m.neuromod.neuromod_sleep_pressure,
            circadian_hour: m.neuromod.circadian_hour,

            // Energy
            total_energy_spent: m.substrate.total_energy_spent,
            energy_this_cycle: m.substrate.energy_this_cycle,
            throughput_multiplier: m.substrate.energy_throughput_multiplier,
            thermodynamic_load: m.thermodynamic_load,
            cycle_duration_us: m.adaptive.cycle_duration_us,

            // Substrate
            feasibility_raw: m.substrate.substrate_feasibility_raw,
            honest_confidence: m.substrate.substrate_honest_confidence,
            effective_feasibility: m.substrate.substrate_effective_feasibility,
            tau_factor: m.substrate.substrate_tau_factor,
            scale_pressure: m.substrate.substrate_scale_pressure,
            effective_dim_fraction: m.substrate.effective_dim_fraction,
            transition_count: m.substrate.transition_count as u32,
        }
    }
}

// ── EpistemicAuditor ──────────────────────────────────────────────────────────

/// DuckDB-backed consciousness telemetry audit trail.
///
/// Buffers [`AuditRecord`]s in memory and periodically flushes them to DuckDB
/// via a background thread. Thread-safe via `Arc<Mutex<duckdb::Connection>>`.
pub(crate) struct EpistemicAuditor {
    buffer: Vec<AuditRecord>,
    conn: Arc<Mutex<duckdb::Connection>>,
    pub total_flushed: u64,
    pub flush_count: u32,
    pub flush_errors: u64,
    flush_in_progress: Arc<AtomicBool>,
}

impl EpistemicAuditor {
    /// Open (or create) a DuckDB database at `db_path` (or `:memory:` when
    /// `db_path` is `None`) and create the audit schema.
    pub fn new(db_path: Option<&str>) -> Result<Self, String> {
        let conn = match db_path {
            Some(path) => duckdb::Connection::open(path)
                .map_err(|e| format!("EpistemicAuditor: open {path}: {e}"))?,
            None => duckdb::Connection::open_in_memory()
                .map_err(|e| format!("EpistemicAuditor: open in-memory: {e}"))?,
        };

        conn.execute_batch(SCHEMA_SQL)
            .map_err(|e| format!("EpistemicAuditor: schema init: {e}"))?;

        Ok(Self {
            buffer: Vec::with_capacity(1024),
            conn: Arc::new(Mutex::new(conn)),
            total_flushed: 0,
            flush_count: 0,
            flush_errors: 0,
            flush_in_progress: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Push a record into the in-memory buffer.
    pub fn record(&mut self, record: AuditRecord) {
        self.buffer.push(record);
    }

    /// Drain the buffer and spawn a background thread to write to DuckDB.
    ///
    /// Skips the flush if a previous one is still in progress.
    pub fn flush_background(&mut self) {
        if self
            .flush_in_progress
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            // Previous flush still running — skip.
            return;
        }

        if self.buffer.is_empty() {
            self.flush_in_progress.store(false, Ordering::Release);
            return;
        }

        let records: Vec<AuditRecord> = self.buffer.drain(..).collect();
        let n = records.len() as u64;
        self.total_flushed += n;
        self.flush_count += 1;

        let conn: Arc<Mutex<duckdb::Connection>> = Arc::clone(&self.conn);
        let flag = Arc::clone(&self.flush_in_progress);

        std::thread::spawn(move || {
            if let Ok(guard) = conn.lock() {
                Self::do_flush(&guard, &records);
            }
            flag.store(false, Ordering::Release);
        });
    }

    /// Flush synchronously (blocking). Used in tests and final shutdown.
    pub fn flush_sync(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        let records: Vec<AuditRecord> = self.buffer.drain(..).collect();
        let n = records.len() as u64;
        self.total_flushed += n;
        self.flush_count += 1;

        if let Ok(guard) = self.conn.lock() {
            self.flush_errors += Self::do_flush(&guard, &records);
        }
    }

    /// Write a batch of records to all six audit tables via parameterized INSERT.
    /// Returns the number of INSERT errors encountered.
    fn do_flush(conn: &duckdb::Connection, records: &[AuditRecord]) -> u64 {
        let mut errors = 0u64;
        // Helper: execute INSERT and count/log errors.
        macro_rules! audit_insert {
            ($table:expr, $sql:expr, $params:expr) => {
                if let Err(e) = conn.execute($sql, $params) {
                    errors += 1;
                    tracing::warn!(table = $table, err = %e, "audit INSERT failed");
                }
            };
        }

        for r in records {
            // phi_trajectory
            audit_insert!(
                "phi_trajectory",
                "INSERT INTO phi_trajectory VALUES (?,?,?,?,?,?,?,?,?)",
                duckdb::params![
                    r.cycle_id as i64,
                    r.timestamp_us,
                    r.phi,
                    r.consciousness_level,
                    r.micro_phi,
                    r.meso_phi,
                    r.macro_phi,
                    r.emergence_ratio,
                    r.limiting_component.as_str(),
                ]
            );

            // graduation_log
            audit_insert!(
                "graduation_log",
                "INSERT INTO graduation_log VALUES (?,?,?,?,?,?,?,?,?,?)",
                duckdb::params![
                    r.cycle_id as i64,
                    r.timestamp_us,
                    r.graduations_processed as i64,
                    r.graduations_rejected as i64,
                    r.semantic_evictions as i64,
                    r.episodic_memory_count as i32,
                    r.episodic_avg_psi,
                    r.codebook_diversity,
                    r.codebook_utilization_rate,
                    r.memory_db_flushed,
                ]
            );

            // moral_audit
            audit_insert!(
                "moral_audit",
                "INSERT INTO moral_audit VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                duckdb::params![
                    r.cycle_id as i64,
                    r.timestamp_us,
                    r.moral_score,
                    r.moral_anomaly_score,
                    r.topo_unity,
                    r.topo_completeness,
                    r.topo_circularity,
                    r.topo_free_energy,
                    r.topo_beta_0 as i32,
                    r.topo_beta_1 as i32,
                    r.value_inversion,
                    r.free_energy_spike,
                    r.drift_alert,
                    r.fragmentation_increase,
                    r.trajectory_convergence,
                    r.convergence_severity,
                    r.escalation_level.as_str(),
                    r.harmony_entropy,
                    r.fingerprint_velocity,
                ]
            );

            // neuromod_history
            audit_insert!(
                "neuromod_history",
                "INSERT INTO neuromod_history VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                duckdb::params![
                    r.cycle_id as i64,
                    r.timestamp_us,
                    r.dopamine_effective,
                    r.noradrenaline_effective,
                    r.serotonin_effective,
                    r.acetylcholine_effective,
                    r.gaba_effective,
                    r.oxytocin_effective,
                    r.glutamate_effective,
                    r.endocannabinoid_effective,
                    r.adenosine_effective,
                    r.da_phasic_burst,
                    r.ne_phasic_spike,
                    r.consciousness_mod,
                    r.allostatic_load,
                    r.ei_balance_ratio,
                    r.bath_entropy,
                    r.sleep_pressure,
                    r.circadian_hour,
                ]
            );

            // energy_audit
            audit_insert!(
                "energy_audit",
                "INSERT INTO energy_audit VALUES (?,?,?,?,?,?,?)",
                duckdb::params![
                    r.cycle_id as i64,
                    r.timestamp_us,
                    r.total_energy_spent,
                    r.energy_this_cycle,
                    r.throughput_multiplier,
                    r.thermodynamic_load,
                    r.cycle_duration_us as i64,
                ]
            );

            // substrate_audit
            audit_insert!(
                "substrate_audit",
                "INSERT INTO substrate_audit VALUES (?,?,?,?,?,?,?,?,?)",
                duckdb::params![
                    r.cycle_id as i64,
                    r.timestamp_us,
                    r.feasibility_raw,
                    r.honest_confidence,
                    r.effective_feasibility,
                    r.tau_factor,
                    r.scale_pressure,
                    r.effective_dim_fraction,
                    r.transition_count as i32,
                ]
            );
        }
        errors
    }

    /// Total records written to DuckDB plus records still in buffer.
    pub fn total_records(&self) -> u64 {
        self.total_flushed + self.buffer.len() as u64
    }

    /// Current buffer length (unflushed records).
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }
}

// ── Query result types ────────────────────────────────────────────────────────

/// Phi statistics summary over a cycle range.
pub struct PhiStatistics {
    pub count: u64,
    pub mean: f64,
    pub stddev: f64,
    pub min: f64,
    pub max: f64,
    pub mean_consciousness: f64,
}

/// High-level audit summary over a cycle range.
pub struct AuditSummary {
    pub total_cycles: u64,
    pub phi: PhiStatistics,
    pub total_graduations: u64,
    pub total_rejections: u64,
    pub moral_anomaly_count: u64,
    pub total_energy: f64,
}

impl EpistemicAuditor {
    /// Phi statistics for cycles in `[from, to)`.
    pub fn phi_statistics(&self, from: u64, to: u64) -> Result<PhiStatistics, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("phi_statistics: lock: {e}"))?;

        let mut stmt = conn
            .prepare(
                "SELECT COUNT(*), AVG(phi), STDDEV_POP(phi), MIN(phi), MAX(phi), \
                 AVG(consciousness_level) \
                 FROM phi_trajectory \
                 WHERE cycle_id >= ? AND cycle_id < ?",
            )
            .map_err(|e| format!("phi_statistics: prepare: {e}"))?;

        let row = stmt
            .query_row(
                duckdb::params![from as i64, to as i64],
                |r: &duckdb::Row<'_>| {
                    Ok(PhiStatistics {
                        count: r.get::<_, i64>(0)? as u64,
                        mean: r.get::<_, f64>(1).unwrap_or(0.0),
                        stddev: r.get::<_, f64>(2).unwrap_or(0.0),
                        min: r.get::<_, f64>(3).unwrap_or(0.0),
                        max: r.get::<_, f64>(4).unwrap_or(0.0),
                        mean_consciousness: r.get::<_, f64>(5).unwrap_or(0.0),
                    })
                },
            )
            .map_err(|e| format!("phi_statistics: query: {e}"))?;

        Ok(row)
    }

    /// Count of moral anomaly events (moral_anomaly_score > 0) in `[from, to)`.
    pub fn moral_anomaly_count(&self, from: u64, to: u64) -> Result<u64, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("moral_anomaly_count: lock: {e}"))?;

        let mut stmt = conn
            .prepare(
                "SELECT COUNT(*) FROM moral_audit \
                 WHERE cycle_id >= ? AND cycle_id < ? AND moral_anomaly_score > 0.0",
            )
            .map_err(|e| format!("moral_anomaly_count: prepare: {e}"))?;

        let count: i64 = stmt
            .query_row(
                duckdb::params![from as i64, to as i64],
                |r: &duckdb::Row<'_>| r.get(0),
            )
            .map_err(|e| format!("moral_anomaly_count: query: {e}"))?;

        Ok(count as u64)
    }

    /// Aggregated audit summary for cycles in `[from, to)`.
    pub fn audit_summary(&self, from: u64, to: u64) -> Result<AuditSummary, String> {
        let phi = self.phi_statistics(from, to)?;
        let moral_anomaly_count = self.moral_anomaly_count(from, to)?;

        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("audit_summary: lock: {e}"))?;

        // graduation totals
        let mut stmt = conn
            .prepare(
                "SELECT COALESCE(SUM(graduations_processed), 0), \
                        COALESCE(SUM(graduations_rejected), 0) \
                 FROM graduation_log WHERE cycle_id >= ? AND cycle_id < ?",
            )
            .map_err(|e| format!("audit_summary: graduation prepare: {e}"))?;

        let (total_graduations, total_rejections) = stmt
            .query_row(
                duckdb::params![from as i64, to as i64],
                |r: &duckdb::Row<'_>| Ok((r.get::<_, i64>(0)? as u64, r.get::<_, i64>(1)? as u64)),
            )
            .map_err(|e| format!("audit_summary: graduation query: {e}"))?;

        // total energy
        let mut stmt2 = conn
            .prepare(
                "SELECT COALESCE(SUM(energy_this_cycle), 0.0) \
                 FROM energy_audit WHERE cycle_id >= ? AND cycle_id < ?",
            )
            .map_err(|e| format!("audit_summary: energy prepare: {e}"))?;

        let total_energy: f64 = stmt2
            .query_row(
                duckdb::params![from as i64, to as i64],
                |r: &duckdb::Row<'_>| r.get(0),
            )
            .map_err(|e| format!("audit_summary: energy query: {e}"))?;

        Ok(AuditSummary {
            total_cycles: phi.count,
            phi,
            total_graduations,
            total_rejections,
            moral_anomaly_count,
            total_energy,
        })
    }

    /// Total cycles recorded in the phi_trajectory table.
    pub fn total_cycles_audited(&self) -> u64 {
        let Ok(conn) = self.conn.lock() else {
            return 0;
        };
        let Ok(mut stmt) = conn.prepare("SELECT COUNT(*) FROM phi_trajectory") else {
            return 0;
        };
        stmt.query_row([], |r: &duckdb::Row<'_>| r.get::<_, i64>(0))
            .map(|n| n as u64)
            .unwrap_or(0)
    }

    /// Full audit summary over all recorded cycles.
    pub fn audit_summary_all(&self) -> Result<AuditSummary, String> {
        let total = self.total_cycles_audited();
        if total == 0 {
            return Err(format!(
                "EpistemicAuditor: no cycles audited yet (history capacity: {})",
                self.history.capacity()
            ));
        }
        self.audit_summary(0, total + 1)
    }

    /// Generate a formatted audit report over all recorded cycles.
    pub fn generate_report_all(&self) -> Result<String, String> {
        let total = self.total_cycles_audited();
        if total == 0 {
            return Err(format!(
                "EpistemicAuditor: no cycles audited yet (history capacity: {})",
                self.history.capacity()
            ));
        }
        self.generate_report(0, total + 1)
    }

    // ── Export ────────────────────────────────────────────────────────────────

    /// Export all audit tables to files in the given directory.
    ///
    /// `format` can be `"parquet"`, `"csv"`, or `"json"`.
    /// Creates one file per table (e.g., `phi_trajectory.parquet`).
    /// Returns the number of files written.
    pub fn export(&self, dir: &str, format: &str) -> Result<usize, String> {
        let fmt = match format.to_lowercase().as_str() {
            "parquet" => "PARQUET",
            "csv" => "CSV",
            "json" => "JSON",
            other => {
                return Err(format!(
                    "Unsupported export format: {other}. Use parquet, csv, or json."
                ));
            }
        };

        let conn = self.conn.lock().map_err(|e| format!("export: lock: {e}"))?;

        let tables = [
            "phi_trajectory",
            "graduation_log",
            "moral_audit",
            "neuromod_history",
            "energy_audit",
            "substrate_audit",
        ];

        let mut written = 0;
        for table in &tables {
            let ext = format.to_lowercase();
            let path = format!("{dir}/{table}.{ext}");
            let sql = format!("COPY {table} TO '{path}' (FORMAT {fmt})");
            conn.execute_batch(&sql)
                .map_err(|e| format!("export {table}: {e}"))?;
            written += 1;
        }

        Ok(written)
    }

    // ── Formatted Report ─────────────────────────────────────────────────────

    /// Generate a human-readable audit report for cycles in `[from, to)`.
    ///
    /// Returns a multi-line string suitable for logging, CLI output, or
    /// inclusion in documents. Covers Phi trajectory, memory graduation,
    /// moral integrity, neuromodulator balance, and energy consumption.
    pub fn generate_report(&self, from: u64, to: u64) -> Result<String, String> {
        let summary = self.audit_summary(from, to)?;
        let conn = self.conn.lock().map_err(|e| format!("report: lock: {e}"))?;

        // Neuromodulator averages
        let neuromod = {
            let mut stmt = conn
                .prepare(
                    "SELECT AVG(dopamine_effective), AVG(serotonin_effective), \
                        AVG(noradrenaline_effective), AVG(acetylcholine_effective), \
                        AVG(gaba_effective), AVG(ei_balance_ratio), \
                        AVG(allostatic_load), AVG(sleep_pressure) \
                 FROM neuromod_history WHERE cycle_id >= ? AND cycle_id < ?",
                )
                .map_err(|e| format!("report neuromod: {e}"))?;
            stmt.query_row(
                duckdb::params![from as i64, to as i64],
                |r: &duckdb::Row<'_>| {
                    Ok((
                        r.get::<_, f64>(0).unwrap_or(0.0),
                        r.get::<_, f64>(1).unwrap_or(0.0),
                        r.get::<_, f64>(2).unwrap_or(0.0),
                        r.get::<_, f64>(3).unwrap_or(0.0),
                        r.get::<_, f64>(4).unwrap_or(0.0),
                        r.get::<_, f64>(5).unwrap_or(0.0),
                        r.get::<_, f64>(6).unwrap_or(0.0),
                        r.get::<_, f64>(7).unwrap_or(0.0),
                    ))
                },
            )
            .map_err(|e| format!("report neuromod query: {e}"))?
        };

        // Moral drift alerts
        let moral_drift_count: i64 = {
            let mut stmt = conn
                .prepare(
                    "SELECT COUNT(*) FROM moral_audit \
                 WHERE cycle_id >= ? AND cycle_id < ? AND drift_alert = true",
                )
                .map_err(|e| format!("report moral_drift: {e}"))?;
            stmt.query_row(
                duckdb::params![from as i64, to as i64],
                |r: &duckdb::Row<'_>| r.get(0),
            )
            .map_err(|e| format!("report moral_drift query: {e}"))?
        };

        // Substrate transitions
        let substrate_transitions: i64 = {
            let mut stmt = conn
                .prepare(
                    "SELECT COUNT(*) FROM substrate_audit \
                 WHERE cycle_id >= ? AND cycle_id < ? \
                   AND transition_count > 0",
                )
                .map_err(|e| format!("report substrate: {e}"))?;
            stmt.query_row(
                duckdb::params![from as i64, to as i64],
                |r: &duckdb::Row<'_>| r.get(0),
            )
            .map_err(|e| format!("report substrate query: {e}"))?
        };

        // Energy stats
        let energy = {
            let mut stmt = conn
                .prepare(
                    "SELECT AVG(energy_this_cycle), AVG(cycle_duration_us), \
                        AVG(thermodynamic_load) \
                 FROM energy_audit WHERE cycle_id >= ? AND cycle_id < ?",
                )
                .map_err(|e| format!("report energy: {e}"))?;
            stmt.query_row(
                duckdb::params![from as i64, to as i64],
                |r: &duckdb::Row<'_>| {
                    Ok((
                        r.get::<_, f64>(0).unwrap_or(0.0),
                        r.get::<_, f64>(1).unwrap_or(0.0),
                        r.get::<_, f64>(2).unwrap_or(0.0),
                    ))
                },
            )
            .map_err(|e| format!("report energy query: {e}"))?
        };

        let grad_rate = if summary.total_graduations + summary.total_rejections > 0 {
            summary.total_graduations as f64
                / (summary.total_graduations + summary.total_rejections) as f64
                * 100.0
        } else {
            0.0
        };

        Ok(format!(
            "\
================================================================
  EPISTEMIC AUDIT REPORT  (cycles {from}..{to})
================================================================

  CONSCIOUSNESS
    Cycles audited:     {total_cycles}
    Phi mean:           {phi_mean:.6}  (stddev {phi_std:.6})
    Phi range:          [{phi_min:.6}, {phi_max:.6}]
    Consciousness mean: {cons:.6}

  MEMORY GRADUATION
    Graduated:          {grad}
    Rejected:           {rej}
    Acceptance rate:    {grad_rate:.1}%

  MORAL INTEGRITY
    Anomaly events:     {anomalies}
    Drift alerts:       {drifts}

  NEUROMODULATOR BALANCE
    Dopamine:           {da:.4}
    Serotonin:          {ser:.4}
    Noradrenaline:      {ne:.4}
    Acetylcholine:      {ach:.4}
    GABA:               {gaba:.4}
    E/I ratio:          {ei:.4}
    Allostatic load:    {allo:.4}
    Sleep pressure:     {sleep:.4}

  ENERGY & PERFORMANCE
    Total energy:       {energy_total:.6} J
    Energy/cycle:       {energy_cycle:.6} J
    Cycle duration:     {dur:.0} us
    Thermodynamic load: {thermo:.4}
    Substrate switches: {switches}

================================================================",
            total_cycles = summary.total_cycles,
            phi_mean = summary.phi.mean,
            phi_std = summary.phi.stddev,
            phi_min = summary.phi.min,
            phi_max = summary.phi.max,
            cons = summary.phi.mean_consciousness,
            grad = summary.total_graduations,
            rej = summary.total_rejections,
            anomalies = summary.moral_anomaly_count,
            drifts = moral_drift_count,
            da = neuromod.0,
            ser = neuromod.1,
            ne = neuromod.2,
            ach = neuromod.3,
            gaba = neuromod.4,
            ei = neuromod.5,
            allo = neuromod.6,
            sleep = neuromod.7,
            energy_total = summary.total_energy,
            energy_cycle = energy.0,
            dur = energy.1,
            thermo = energy.2,
            switches = substrate_transitions,
        ))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "epistemic_auditor")]
mod tests {
    use super::*;

    fn make_test_record(cycle: u64) -> AuditRecord {
        AuditRecord {
            cycle_id: cycle,
            timestamp_us: (cycle * 1_000_000) as i64,
            // Phi / Consciousness
            phi: 0.5 + (cycle as f64) * 0.001,
            consciousness_level: 0.6,
            micro_phi: 0.2,
            meso_phi: 0.15,
            macro_phi: 0.35,
            emergence_ratio: 1.05,
            limiting_component: "Integration".to_string(),
            // Graduation
            graduations_processed: cycle * 3,
            graduations_rejected: cycle,
            semantic_evictions: 0,
            episodic_memory_count: (cycle as u32).min(1000),
            episodic_avg_psi: 0.45,
            codebook_diversity: 0.7,
            codebook_utilization_rate: 0.4,
            memory_db_flushed: false,
            // Moral
            moral_score: 0.3,
            moral_anomaly_score: if cycle % 5 == 0 { 0.2 } else { 0.0 },
            topo_unity: 1.0,
            topo_completeness: 0.8,
            topo_circularity: 0.1,
            topo_free_energy: 0.05,
            topo_beta_0: 1,
            topo_beta_1: 0,
            value_inversion: false,
            free_energy_spike: false,
            drift_alert: false,
            fragmentation_increase: false,
            trajectory_convergence: false,
            convergence_severity: 0.0,
            escalation_level: "Log".to_string(),
            harmony_entropy: 1.2,
            fingerprint_velocity: 0.01,
            // Neuromod
            dopamine_effective: 1.0,
            noradrenaline_effective: 0.8,
            serotonin_effective: 0.9,
            acetylcholine_effective: 1.1,
            gaba_effective: 0.7,
            oxytocin_effective: 0.6,
            glutamate_effective: 1.2,
            endocannabinoid_effective: 0.5,
            adenosine_effective: 0.4,
            da_phasic_burst: 0.1,
            ne_phasic_spike: 0.05,
            consciousness_mod: 1.0,
            allostatic_load: 0.2,
            ei_balance_ratio: 1.1,
            bath_entropy: 0.8,
            sleep_pressure: 0.3,
            circadian_hour: 14.0,
            // Energy
            total_energy_spent: (cycle as f64) * 1e-9,
            energy_this_cycle: 1e-9,
            throughput_multiplier: 1.0,
            thermodynamic_load: 0.1,
            cycle_duration_us: 32_000,
            // Substrate
            feasibility_raw: 0.55,
            honest_confidence: 0.10,
            effective_feasibility: 0.28,
            tau_factor: 1.0,
            scale_pressure: 0.0,
            effective_dim_fraction: 1.0,
            transition_count: 0,
        }
    }

    #[test]
    fn test_auditor_creation() {
        let auditor = EpistemicAuditor::new(None).expect("in-memory auditor");
        assert_eq!(auditor.buffer_len(), 0);
        assert_eq!(auditor.total_flushed, 0);
        assert_eq!(auditor.total_records(), 0);
    }

    #[test]
    fn test_record_and_flush() {
        let mut auditor = EpistemicAuditor::new(None).expect("in-memory auditor");
        for i in 0..10 {
            auditor.record(make_test_record(i));
        }
        assert_eq!(auditor.buffer_len(), 10);
        assert_eq!(auditor.total_records(), 10);

        auditor.flush_sync();

        assert_eq!(auditor.buffer_len(), 0);
        assert_eq!(auditor.total_flushed, 10);
        assert_eq!(auditor.flush_count, 1);
    }

    #[test]
    fn test_phi_statistics_query() {
        let mut auditor = EpistemicAuditor::new(None).expect("in-memory auditor");
        for i in 0..20 {
            auditor.record(make_test_record(i));
        }
        auditor.flush_sync();

        let stats = auditor.phi_statistics(0, 20).expect("phi stats");
        assert_eq!(stats.count, 20);
        assert!(stats.mean > 0.5, "mean phi should be > 0.5");
        assert!(stats.min >= 0.5, "min phi should be >= 0.5");
        assert!(stats.max <= 1.0, "max phi should be <= 1.0");
        assert!(
            stats.mean_consciousness > 0.0,
            "mean consciousness should be > 0"
        );
    }

    #[test]
    fn test_audit_summary() {
        let mut auditor = EpistemicAuditor::new(None).expect("in-memory auditor");
        for i in 0..50 {
            auditor.record(make_test_record(i));
        }
        auditor.flush_sync();

        let summary = auditor.audit_summary(0, 50).expect("audit summary");
        assert_eq!(summary.total_cycles, 50);
        // cycles 0,5,10,...,45 have anomaly_score > 0 → 10 entries
        assert_eq!(summary.moral_anomaly_count, 10);
        assert!(summary.total_energy > 0.0);
        // graduations_processed = sum of i*3 for i in 0..50
        let expected_grad: u64 = (0u64..50).map(|i| i * 3).sum();
        assert_eq!(summary.total_graduations, expected_grad);

        // total_cycles_audited should also return 50
        assert_eq!(auditor.total_cycles_audited(), 50);
    }
}
