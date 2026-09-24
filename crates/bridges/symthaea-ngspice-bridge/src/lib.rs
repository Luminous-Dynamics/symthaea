// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! ngspice adapter boundary.
//!
//! ENG-SPICE-001 admits a deliberately narrow real-solver path: an explicit
//! configured, self-contained netlist is executed in ngspice batch mode and
//! requested scalar `.measure` results are parsed from a dedicated log artifact.
//! Arbitrary circuit topology is **not** inferred from `SimulationRequest`
//! scalar parameters.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use symthaea_sim_bridge::{
    CommandSolver, EngineeringDomain, ExecutionMode, SimulationBackend, SimulationError,
    SimulationEvidence, SimulationRequest, SimulationResult, SolverKind,
};

const PARSER_VERSION: &str = "ngspice-measure-parser-v1";
const REAL_RESULT_CONFIDENCE: f64 = 0.5;
static LOG_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// ngspice backend descriptor.
#[derive(Debug, Clone)]
pub struct NgspiceBridge {
    /// When true, return deterministic placeholder metrics for orchestration tests.
    pub dry_run: bool,
    /// Command used to invoke the solver (e.g. `ngspice`).
    pub solver_cmd: String,
    /// Exact netlist artifact used by the real solver path.
    pub netlist_path: PathBuf,
    /// Units for requested `.measure` scalars, keyed case-insensitively.
    pub metric_units: BTreeMap<String, String>,
}

impl Default for NgspiceBridge {
    fn default() -> Self {
        Self {
            dry_run: false,
            solver_cmd: "ngspice".to_string(),
            netlist_path: PathBuf::from("input.sp"),
            metric_units: BTreeMap::new(),
        }
    }
}

impl NgspiceBridge {
    /// Create a dry-run ngspice bridge.
    pub fn dry_run() -> Self {
        Self {
            dry_run: true,
            ..Self::default()
        }
    }

    /// Bind the exact netlist artifact used by subsequent real executions.
    pub fn with_netlist_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.netlist_path = path.into();
        self
    }

    /// Declare the physical unit for one requested `.measure` scalar.
    pub fn with_metric_unit(mut self, metric: impl AsRef<str>, unit: impl Into<String>) -> Self {
        self.metric_units
            .insert(normalize_metric(metric.as_ref()), unit.into());
        self
    }

    fn run_real(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
        request.validate()?;
        if request.requested_metrics.is_empty() {
            return Err(SimulationError::InvalidRequest(
                "real ngspice execution requires at least one requested .measure metric".into(),
            ));
        }
        let requested = canonical_requested_metrics(request)?;
        for normalized in requested.keys() {
            match self.metric_units.get(normalized) {
                Some(unit) if !unit.trim().is_empty() => {}
                _ => {
                    return Err(SimulationError::InvalidRequest(format!(
                        "ngspice requested metric {normalized:?} requires an explicit unit"
                    )));
                }
            }
        }

        let netlist_bytes = fs::read(&self.netlist_path).map_err(|error| {
            SimulationError::Adapter(format!(
                "failed to read ngspice netlist {:?}: {error}",
                self.netlist_path
            ))
        })?;
        if netlist_bytes.is_empty() {
            return Err(SimulationError::InvalidRequest(
                "ngspice netlist artifact cannot be empty".into(),
            ));
        }
        let netlist_text = std::str::from_utf8(&netlist_bytes).map_err(|error| {
            SimulationError::InvalidRequest(format!(
                "qualified ngspice netlist must be UTF-8 text: {error}"
            ))
        })?;
        validate_qualified_netlist(netlist_text, &requested)?;

        let input_digest = blake3::hash(&netlist_bytes).to_hex().to_string();
        let solver_version = self.solver_version()?;
        let log_path = temporary_log_path(&input_digest);

        let command = CommandSolver::new(&self.solver_cmd)
            .arg("-b")
            // Ignore user .spiceinit so local interactive preferences cannot
            // silently change a qualified batch execution.
            .arg("-n")
            .arg("-o")
            .arg(log_path.to_string_lossy().to_string())
            .arg(self.netlist_path.to_string_lossy().to_string());

        if let Err(error) = command.execute() {
            let _ = fs::remove_file(&log_path);
            return Err(error);
        }

        let log_bytes = fs::read(&log_path).map_err(|error| {
            let _ = fs::remove_file(&log_path);
            SimulationError::Adapter(format!(
                "ngspice exited successfully but log artifact {:?} could not be read: {error}",
                log_path
            ))
        })?;
        let _ = fs::remove_file(&log_path);
        let output_digest = blake3::hash(&log_bytes).to_hex().to_string();
        let log = std::str::from_utf8(&log_bytes).map_err(|error| {
            SimulationError::Adapter(format!("ngspice log is not valid UTF-8: {error}"))
        })?;

        reject_known_failure_markers(log)?;
        let parsed = parse_measurements(log, &requested)?;

        let mut result = SimulationResult::converged(&request.id, REAL_RESULT_CONFIDENCE);
        for metric in &request.requested_metrics {
            let normalized = normalize_metric(metric);
            let value = *parsed.get(&normalized).ok_or_else(|| {
                SimulationError::Adapter(format!(
                    "requested ngspice .measure metric {metric:?} was not present in the parsed log"
                ))
            })?;
            let unit = self.metric_units.get(&normalized).expect("validated above");
            result = result.with_metric(metric, value, unit);
        }
        result.warnings.push(
            "ngspice process/log/metric parsing succeeded; this does not establish circuit-model or physical validity"
                .into(),
        );
        result = result.with_external_evidence(SimulationEvidence {
            mode: ExecutionMode::ExternalSolver,
            backend: Some(self.name().into()),
            solver_version: Some(solver_version),
            input_digest: Some(input_digest),
            output_digest: Some(output_digest),
            parser_version: Some(PARSER_VERSION.into()),
        });
        result.validate()?;
        if !result.is_engineering_evidence() {
            return Err(SimulationError::Adapter(
                "parsed ngspice result did not satisfy complete external-evidence provenance"
                    .into(),
            ));
        }
        Ok(result)
    }

    fn solver_version(&self) -> Result<String, SimulationError> {
        let output = CommandSolver::new(&self.solver_cmd)
            .arg("-n")
            .arg("--version")
            .execute()?;
        output
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .ok_or_else(|| {
                SimulationError::Adapter(
                    "ngspice --version succeeded but returned no version text".into(),
                )
            })
    }
}

impl SimulationBackend for NgspiceBridge {
    fn name(&self) -> &'static str {
        "ngspice"
    }

    fn supported_solvers(&self) -> &[SolverKind] {
        &[SolverKind::Circuit]
    }

    fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
        if request.solver != SolverKind::Circuit {
            return Err(SimulationError::InvalidRequest(format!(
                "ngspice cannot satisfy {:?}",
                request.solver
            )));
        }
        if !matches!(
            request.domain,
            EngineeringDomain::Electrical | EngineeringDomain::Systems
        ) {
            return Err(SimulationError::InvalidRequest(format!(
                "ngspice expected electrical/systems request, got {:?}",
                request.domain
            )));
        }

        if self.dry_run {
            return Ok(SimulationResult::dry_run(&request.id, self.name(), 0.55)
                .with_metric("peak_voltage", 12.1, "V")
                .with_metric("settling_time", 0.032, "s"));
        }

        self.run_real(request)
    }
}

fn canonical_requested_metrics(
    request: &SimulationRequest,
) -> Result<BTreeMap<String, String>, SimulationError> {
    let mut requested = BTreeMap::new();
    for metric in &request.requested_metrics {
        let normalized = normalize_metric(metric);
        if normalized.is_empty() {
            return Err(SimulationError::InvalidRequest(
                "ngspice requested metric cannot be empty".into(),
            ));
        }
        if requested.insert(normalized.clone(), metric.clone()).is_some() {
            return Err(SimulationError::InvalidRequest(format!(
                "duplicate ngspice requested metric after case normalization: {metric:?}"
            )));
        }
    }
    Ok(requested)
}

fn normalize_metric(metric: &str) -> String {
    metric.trim().to_ascii_lowercase()
}

/// Restrict the first qualified execution lane to declarative self-contained
/// netlists. Includes/libraries and the interactive control language need a
/// later explicit trusted-artifact/authority design rather than implicit file or
/// process authority.
fn validate_qualified_netlist(
    netlist: &str,
    requested: &BTreeMap<String, String>,
) -> Result<(), SimulationError> {
    let forbidden = [
        ".control", ".endc", ".include", ".inc", ".lib", "source", "load", "shell",
    ];
    let mut measures = BTreeSet::new();
    let mut has_tabulated_output = false;

    for raw_line in netlist.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('*') {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        let tokens: Vec<_> = lower.split_whitespace().collect();
        let Some(first) = tokens.first().copied() else {
            continue;
        };
        if forbidden.contains(&first) {
            return Err(SimulationError::InvalidRequest(format!(
                "qualified ngspice netlist uses forbidden directive/command {first:?}; first tranche permits only self-contained declarative netlists"
            )));
        }
        if first == ".print" || first == ".plot" {
            has_tabulated_output = true;
        }
        if first == ".measure" || first == ".meas" {
            if tokens.len() < 3 {
                return Err(SimulationError::InvalidRequest(
                    "ngspice .measure line is missing analysis/name fields".into(),
                ));
            }
            let name = normalize_metric(tokens[2]);
            if !measures.insert(name.clone()) {
                return Err(SimulationError::InvalidRequest(format!(
                    "ngspice netlist declares duplicate .measure name {name:?}"
                )));
            }
        }
    }

    if !has_tabulated_output {
        return Err(SimulationError::InvalidRequest(
            "qualified ngspice batch netlist requires a .print or .plot directive so .measure data remains available without a rawfile"
                .into(),
        ));
    }
    for normalized in requested.keys() {
        if !measures.contains(normalized) {
            return Err(SimulationError::InvalidRequest(format!(
                "requested metric {normalized:?} is not declared by a .measure/.meas statement in the bound netlist"
            )));
        }
    }
    Ok(())
}

/// Parse scalar `.measure` output lines of the form `name = value ...`.
fn parse_measurements(
    log: &str,
    requested: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, f64>, SimulationError> {
    let wanted: BTreeSet<_> = requested.keys().cloned().collect();
    let mut values = BTreeMap::new();
    for line in log.lines() {
        let Some((left, right)) = line.split_once('=') else {
            continue;
        };
        let Some(name) = left.split_whitespace().next() else {
            continue;
        };
        let normalized = normalize_metric(name);
        if !wanted.contains(&normalized) {
            continue;
        }
        if values.contains_key(&normalized) {
            return Err(SimulationError::Adapter(format!(
                "ngspice log contains duplicate requested .measure result {normalized:?}"
            )));
        }
        let token = right.split_whitespace().next().ok_or_else(|| {
            SimulationError::Adapter(format!(
                "ngspice .measure result {normalized:?} has no scalar value"
            ))
        })?;
        let value = token.parse::<f64>().map_err(|_| {
            SimulationError::Adapter(format!(
                "ngspice .measure result {normalized:?} is not a finite scalar: {token:?}"
            ))
        })?;
        if !value.is_finite() {
            return Err(SimulationError::Adapter(format!(
                "ngspice .measure result {normalized:?} is non-finite"
            )));
        }
        values.insert(normalized, value);
    }
    for normalized in wanted {
        if !values.contains_key(&normalized) {
            return Err(SimulationError::Adapter(format!(
                "requested ngspice .measure result {normalized:?} is missing"
            )));
        }
    }
    Ok(values)
}

/// Process exit success is not convergence. Reject known ngspice analysis-failure
/// diagnostics before any parsed scalar is promoted as a converged result.
fn reject_known_failure_markers(log: &str) -> Result<(), SimulationError> {
    for line in log.lines() {
        let normalized = line.trim().to_ascii_lowercase();
        let hard_failure = normalized.contains("fatal error")
            || normalized.starts_with("error:")
            || normalized.contains("doanalyses: error")
            || normalized.contains("timestep too small")
            || normalized.contains("singular matrix")
            || normalized.contains("no convergence")
            || normalized.contains("convergence failed")
            || normalized.contains("analysis aborted")
            || normalized.contains("simulation interrupted")
            || (normalized.contains("measure") && normalized.contains("failed"));
        if hard_failure {
            return Err(SimulationError::Adapter(format!(
                "ngspice log reports numerical/analysis failure: {}",
                line.trim()
            )));
        }
    }
    Ok(())
}

fn temporary_log_path(input_digest: &str) -> PathBuf {
    let sequence = LOG_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "symthaea-ngspice-{}-{}-{}.log",
        std::process::id(),
        sequence,
        &input_digest[..input_digest.len().min(16)]
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(metrics: &[&str]) -> SimulationRequest {
        let mut request = SimulationRequest::new(
            "spice-1",
            EngineeringDomain::Electrical,
            SolverKind::Circuit,
            "screen transient response",
        );
        request.requested_metrics = metrics.iter().map(|metric| (*metric).into()).collect();
        request
    }

    fn safe_netlist() -> &'static str {
        "* fixture\nV1 in 0 1\nR1 in out 1k\nR2 out 0 1k\n.tran 1u 10u\n.measure tran vmax MAX v(out)\n.print tran v(out)\n.end\n"
    }

    #[test]
    fn dry_run_returns_non_external_fixture_metrics() {
        let backend = NgspiceBridge::dry_run();
        let result = backend.run(&request(&[])).unwrap();
        assert!(result.converged);
        assert_eq!(result.evidence.mode, ExecutionMode::DryRun);
        assert!(!result.is_engineering_evidence());
    }

    #[test]
    fn qualified_netlist_binds_requested_measure() {
        let requested = canonical_requested_metrics(&request(&["vmax"])).unwrap();
        assert!(validate_qualified_netlist(safe_netlist(), &requested).is_ok());
    }

    #[test]
    fn control_language_is_rejected_from_first_qualified_lane() {
        let requested = canonical_requested_metrics(&request(&["vmax"])).unwrap();
        let netlist = ".control\nshell echo nope\n.endc\n.measure tran vmax MAX v(out)\n.print tran v(out)\n";
        assert!(matches!(
            validate_qualified_netlist(netlist, &requested),
            Err(SimulationError::InvalidRequest(message)) if message.contains("forbidden")
        ));
    }

    #[test]
    fn includes_are_rejected_until_artifact_graph_is_explicit() {
        let requested = canonical_requested_metrics(&request(&["vmax"])).unwrap();
        let netlist = ".include vendor.lib\n.measure tran vmax MAX v(out)\n.print tran v(out)\n";
        assert!(validate_qualified_netlist(netlist, &requested).is_err());
    }

    #[test]
    fn missing_measure_declaration_fails_before_execution() {
        let requested = canonical_requested_metrics(&request(&["missing"])).unwrap();
        assert!(validate_qualified_netlist(safe_netlist(), &requested).is_err());
    }

    #[test]
    fn measure_parser_accepts_scientific_scalar_and_ignores_at_clause() {
        let requested = canonical_requested_metrics(&request(&["vmax", "settle"])).unwrap();
        let log = "\nvmax = 1.234000e+01 at= 2.0e-03\nsettle = 3.200000e-02\n";
        let parsed = parse_measurements(log, &requested).unwrap();
        assert_eq!(parsed["vmax"], 12.34);
        assert_eq!(parsed["settle"], 0.032);
    }

    #[test]
    fn duplicate_measurement_is_rejected_as_ambiguous() {
        let requested = canonical_requested_metrics(&request(&["vmax"])).unwrap();
        let log = "vmax = 1.0\nvmax = 2.0\n";
        assert!(matches!(
            parse_measurements(log, &requested),
            Err(SimulationError::Adapter(message)) if message.contains("duplicate")
        ));
    }

    #[test]
    fn missing_measurement_fails_closed() {
        let requested = canonical_requested_metrics(&request(&["vmax"])).unwrap();
        assert!(matches!(
            parse_measurements("other = 1.0\n", &requested),
            Err(SimulationError::Adapter(message)) if message.contains("missing")
        ));
    }

    #[test]
    fn convergence_failure_marker_blocks_promotion() {
        let log = "Warning: trouble with node\ndoAnalyses: TRAN: Timestep too small; trouble with x1\n";
        assert!(reject_known_failure_markers(log).is_err());
    }

    #[test]
    fn duplicate_requested_metrics_are_case_insensitively_rejected() {
        assert!(matches!(
            canonical_requested_metrics(&request(&["VMAX", "vmax"])),
            Err(SimulationError::InvalidRequest(message)) if message.contains("duplicate")
        ));
    }

    #[test]
    fn real_path_requires_metric_units_before_execution() {
        let backend = NgspiceBridge::default();
        assert!(matches!(
            backend.run(&request(&["vmax"])),
            Err(SimulationError::InvalidRequest(message)) if message.contains("explicit unit")
        ));
    }

    #[test]
    fn configured_metric_units_are_case_insensitive() {
        let bridge = NgspiceBridge::default().with_metric_unit("VMAX", "V");
        assert_eq!(bridge.metric_units.get("vmax").map(String::as_str), Some("V"));
    }

    #[test]
    fn temp_log_paths_do_not_collide_within_process() {
        let a = temporary_log_path("0123456789abcdef0123");
        let b = temporary_log_path("0123456789abcdef0123");
        assert_ne!(a, b);
        assert!(a.starts_with(std::env::temp_dir()));
    }

    #[test]
    fn explicit_netlist_path_is_preserved() {
        let bridge = NgspiceBridge::default()
            .with_netlist_path(std::path::Path::new("fixture.cir"));
        assert_eq!(bridge.netlist_path, PathBuf::from("fixture.cir"));
    }
}
