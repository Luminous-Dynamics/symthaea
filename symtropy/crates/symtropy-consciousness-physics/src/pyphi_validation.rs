// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! IIT 4.0 validation via PyPhi FFI.
//!
//! This module builds empirical Transition Probability Matrices (TPMs) from
//! small agent groups in the physics simulation, sends them to PyPhi (Python)
//! for exact IIT integrated information (Φ) computation, and compares the
//! result against our fast softmin bottleneck approximation.
//!
//! # Architecture
//!
//! The bridge uses subprocess communication (JSON over stdin/stdout) rather
//! than PyO3 embedding. This keeps the game engine free of Python runtime
//! dependencies while allowing validation in research workflows.
//!
//! # Limitations
//!
//! - PyPhi has O(2^N) complexity; practical for N ≤ 8 agents.
//! - TPM is estimated empirically from observed state transitions; accuracy
//!   depends on having visited all 2^N states (requires sufficient ticks).
//! - Discretization (continuous Φ → binary) loses information.
//!
//! # Usage
//!
//! ```bash
//! # Requires: pip install pyphi (in nix develop shell)
//! cargo test -p symtropy-consciousness-physics --features pyphi-validation
//! ```

use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};

use crate::coupling::{ConsciousnessField, EntityConsciousness};
use symtropy_physics::BodyHandle;

// ────────────────────────────────────────────────────────────
//  TPM Construction
// ────────────────────────────────────────────────────────────

/// Threshold for discretizing Φ to binary (on/off).
/// Agents with Φ ≥ this value are considered "integrated" (state = 1).
/// Set low (0.1) because the softmin bottleneck typically produces Φ ∈ [0.05, 0.25].
pub const PHI_BINARY_THRESHOLD: f64 = 0.1;

/// A binary snapshot of agent states at one tick.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BinaryState(pub Vec<u8>);

impl BinaryState {
    /// Convert to a state index (little-endian binary → usize).
    pub fn to_index(&self) -> usize {
        self.0
            .iter()
            .enumerate()
            .fold(0, |acc, (i, &bit)| acc | ((bit as usize) << i))
    }

    /// Create from a state index.
    pub fn from_index(index: usize, n: usize) -> Self {
        let bits: Vec<u8> = (0..n).map(|i| ((index >> i) & 1) as u8).collect();
        Self(bits)
    }
}

/// Empirical Transition Probability Matrix for N agents.
///
/// Each row corresponds to a system state (2^N rows), each column to a node.
/// Entry `tpm[state][node]` = probability that `node` is ON given the system
/// is in `state`. This is PyPhi's "state-by-node" format.
pub struct EmpiricalTpm {
    /// Number of agents (nodes).
    pub n: usize,
    /// Transition counts: `counts[from_state][to_state]`.
    counts: Vec<Vec<u64>>,
    /// Total transitions observed from each state.
    totals: Vec<u64>,
}

impl EmpiricalTpm {
    /// Create a new empty TPM for `n` agents.
    pub fn new(n: usize) -> Self {
        let num_states = 1 << n;
        Self {
            n,
            counts: vec![vec![0u64; num_states]; num_states],
            totals: vec![0; num_states],
        }
    }

    /// Record a state transition.
    pub fn record_transition(&mut self, from: &BinaryState, to: &BinaryState) {
        let fi = from.to_index();
        let ti = to.to_index();
        self.counts[fi][ti] += 1;
        self.totals[fi] += 1;
    }

    /// Number of distinct states visited (states with at least one outgoing transition).
    pub fn states_visited(&self) -> usize {
        self.totals.iter().filter(|&&t| t > 0).count()
    }

    /// Total number of states (2^N).
    pub fn num_states(&self) -> usize {
        1 << self.n
    }

    /// Coverage: fraction of states visited.
    pub fn coverage(&self) -> f64 {
        self.states_visited() as f64 / self.num_states() as f64
    }

    /// Convert to state-by-node format for PyPhi.
    ///
    /// Returns a `2^N × N` matrix where `tpm[state][node]` is the probability
    /// that `node` is ON in the next tick given the current system `state`.
    ///
    /// Unvisited states are filled with uniform probability (0.5) — this is
    /// the maximum-entropy prior, which is conservative for Φ estimation.
    pub fn to_state_by_node(&self) -> Vec<Vec<f64>> {
        let num_states = self.num_states();
        let mut tpm = vec![vec![0.5f64; self.n]; num_states];

        for from_idx in 0..num_states {
            if self.totals[from_idx] == 0 {
                continue; // Leave as uniform (0.5)
            }

            // For each node, compute P(node=1 | from_state)
            for node in 0..self.n {
                let mut on_count = 0u64;
                for to_idx in 0..num_states {
                    if (to_idx >> node) & 1 == 1 {
                        on_count += self.counts[from_idx][to_idx];
                    }
                }
                tpm[from_idx][node] = on_count as f64 / self.totals[from_idx] as f64;
            }
        }

        tpm
    }
}

// ────────────────────────────────────────────────────────────
//  Agent State Discretization
// ────────────────────────────────────────────────────────────

/// Discretize agent Φ values to a binary state vector.
pub fn discretize_agents(
    entities: &BTreeMap<BodyHandle, EntityConsciousness>,
    handles: &[BodyHandle],
    threshold: f64,
) -> BinaryState {
    let bits: Vec<u8> = handles
        .iter()
        .map(|h| {
            entities
                .get(h)
                .map(|e| if e.phi() >= threshold { 1 } else { 0 })
                .unwrap_or(0)
        })
        .collect();
    BinaryState(bits)
}

// ────────────────────────────────────────────────────────────
//  TPM Collection from Simulation
// ────────────────────────────────────────────────────────────

/// Collector that accumulates state transitions from a running simulation.
pub struct TpmCollector {
    handles: Vec<BodyHandle>,
    tpm: EmpiricalTpm,
    threshold: f64,
    prev_state: Option<BinaryState>,
    ticks_recorded: u64,
}

impl TpmCollector {
    /// Create a collector for the given agent handles.
    ///
    /// # Panics
    /// Panics if `handles.len() > 12` (PyPhi complexity limit).
    pub fn new(handles: Vec<BodyHandle>, threshold: f64) -> Self {
        assert!(
            handles.len() <= 12,
            "PyPhi validation limited to ≤12 agents (got {})",
            handles.len()
        );
        let n = handles.len();
        Self {
            handles,
            tpm: EmpiricalTpm::new(n),
            threshold,
            prev_state: None,
            ticks_recorded: 0,
        }
    }

    /// Record one tick of the simulation.
    pub fn tick<const D: usize>(&mut self, field: &ConsciousnessField<D>) {
        let state = discretize_agents(&field.entities, &self.handles, self.threshold);
        if let Some(ref prev) = self.prev_state {
            self.tpm.record_transition(prev, &state);
            self.ticks_recorded += 1;
        }
        self.prev_state = Some(state);
    }

    /// Current TPM coverage (fraction of 2^N states visited).
    pub fn coverage(&self) -> f64 {
        self.tpm.coverage()
    }

    /// Number of ticks recorded.
    pub fn ticks_recorded(&self) -> u64 {
        self.ticks_recorded
    }

    /// Consume the collector and return the empirical TPM.
    pub fn into_tpm(self) -> EmpiricalTpm {
        self.tpm
    }

    /// Reference to the current TPM.
    pub fn tpm(&self) -> &EmpiricalTpm {
        &self.tpm
    }

    /// The handles being tracked.
    pub fn handles(&self) -> &[BodyHandle] {
        &self.handles
    }
}

// ────────────────────────────────────────────────────────────
//  PyPhi Bridge (subprocess)
// ────────────────────────────────────────────────────────────

/// Result from PyPhi computation.
#[derive(Debug, Clone)]
pub struct PyPhiResult {
    /// Big Phi (IIT integrated information).
    pub phi: Option<f64>,
    /// Number of concepts in the constellation.
    pub num_concepts: usize,
    /// Size of the major complex.
    pub major_complex_size: usize,
    /// Error message if computation failed.
    pub error: Option<String>,
}

/// Call PyPhi to compute exact IIT Φ for a given TPM and state.
///
/// This shells out to Python via subprocess. Requires:
/// - Python 3.8+ with `pyphi` installed
/// - The `scripts/pyphi_compute.py` script accessible at the expected path
///
/// Returns `Err` if the subprocess fails (Python not found, script missing, etc.).
pub fn call_pyphi(
    tpm: &[Vec<f64>],
    state: &[u8],
    node_labels: Option<&[String]>,
) -> Result<PyPhiResult, String> {
    // Build JSON request
    let labels: Vec<String> = node_labels
        .map(|l| l.to_vec())
        .unwrap_or_else(|| (0..state.len()).map(|i| format!("A{i}")).collect());

    let request = serde_json_minimal(tpm, state, &labels);

    // Find script path
    let script_path = find_pyphi_script()?;

    // Shell out to Python — check PYPHI_PYTHON env var first, then common venv paths
    let python = std::env::var("PYPHI_PYTHON").unwrap_or_else(|_| {
        for candidate in &["/tmp/pyphi-venv/bin/python3", "/tmp/pyphi310/bin/python3"] {
            if std::path::Path::new(candidate).exists() {
                return candidate.to_string();
            }
        }
        "python3".to_string()
    });

    let mut child = Command::new(&python)
        .arg(&script_path)
        .env("PYPHI_WELCOME_OFF", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn python3: {e}"))?;

    // Write request to stdin
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(request.as_bytes())
            .map_err(|e| format!("Failed to write to python3 stdin: {e}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("Failed to wait for python3: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("python3 exited with {}: {}", output.status, stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_pyphi_response(&stdout)
}

/// Minimal JSON serialization without serde dependency.
fn serde_json_minimal(tpm: &[Vec<f64>], state: &[u8], labels: &[String]) -> String {
    let tpm_str: Vec<String> = tpm
        .iter()
        .map(|row| {
            let vals: Vec<String> = row.iter().map(|v| format!("{v}")).collect();
            format!("[{}]", vals.join(","))
        })
        .collect();

    let state_str: Vec<String> = state.iter().map(|s| format!("{s}")).collect();
    let labels_str: Vec<String> = labels.iter().map(|l| format!("\"{l}\"")).collect();

    format!(
        r#"{{"tpm":[{}],"state":[{}],"node_labels":[{}]}}"#,
        tpm_str.join(","),
        state_str.join(","),
        labels_str.join(",")
    )
}

/// Find the pyphi_compute.py script by checking known paths.
fn find_pyphi_script() -> Result<String, String> {
    // Check relative to crate root
    let candidates = [
        "scripts/pyphi_compute.py",
        "crates/symtropy-consciousness-physics/scripts/pyphi_compute.py",
        // From workspace root
        concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/pyphi_compute.py"),
    ];

    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return Ok(path.to_string());
        }
    }

    Err(format!(
        "pyphi_compute.py not found. Checked: {:?}",
        candidates
    ))
}

/// Parse PyPhi's JSON response.
fn parse_pyphi_response(json: &str) -> Result<PyPhiResult, String> {
    // Minimal JSON parsing without serde
    let json = json.trim();

    let phi = extract_f64(json, "phi");
    let num_concepts = extract_usize(json, "num_concepts").unwrap_or(0);
    let major_complex_size = extract_usize(json, "major_complex_size").unwrap_or(0);
    let error = extract_string(json, "error");

    Ok(PyPhiResult {
        phi,
        num_concepts,
        major_complex_size,
        error,
    })
}

fn extract_f64(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{key}\":");
    let start = json.find(&pattern)? + pattern.len();
    let rest = json[start..].trim_start();
    if rest.starts_with("null") {
        return None;
    }
    let end = rest
        .find(|c: char| c == ',' || c == '}')
        .unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

fn extract_usize(json: &str, key: &str) -> Option<usize> {
    extract_f64(json, key).map(|f| f as usize)
}

fn extract_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\":");
    let start = json.find(&pattern)? + pattern.len();
    let rest = json[start..].trim_start();
    if rest.starts_with("null") {
        return None;
    }
    if !rest.starts_with('"') {
        return None;
    }
    let rest = &rest[1..]; // skip opening quote
    let end = rest.find('"').unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

// ────────────────────────────────────────────────────────────
//  Validation: Compare Symtropy Φ vs PyPhi Φ
// ────────────────────────────────────────────────────────────

/// Validation result comparing our approximation against PyPhi ground truth.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Our engine's mean Φ across the agent group.
    pub symtropy_phi: f64,
    /// PyPhi's exact IIT Φ.
    pub pyphi_phi: Option<f64>,
    /// Absolute error (|symtropy - pyphi|).
    pub absolute_error: Option<f64>,
    /// Number of agents in the group.
    pub group_size: usize,
    /// TPM coverage (fraction of 2^N states visited).
    pub tpm_coverage: f64,
    /// Number of ticks used to build the TPM.
    pub ticks_used: u64,
    /// Error message if validation failed.
    pub error: Option<String>,
}

/// Run full validation: collect TPM, call PyPhi, compare.
///
/// The `field` should have been simulated for enough ticks that the
/// `TpmCollector` has reasonable coverage (ideally > 50%).
pub fn validate_against_pyphi<const D: usize>(
    collector: &TpmCollector,
    field: &ConsciousnessField<D>,
) -> ValidationResult {
    let handles = collector.handles();

    // Compute our engine's mean Φ for this group
    let symtropy_phi = handles
        .iter()
        .filter_map(|h| field.entities.get(h).map(|e| e.phi()))
        .sum::<f64>()
        / handles.len().max(1) as f64;

    // Get the current binary state
    let current_state = discretize_agents(&field.entities, handles, collector.threshold);

    // Build TPM in state-by-node format
    let tpm = collector.tpm().to_state_by_node();

    // Call PyPhi
    match call_pyphi(&tpm, &current_state.0, None) {
        Ok(result) => {
            let pyphi_phi = result.phi;
            let absolute_error = pyphi_phi.map(|pp| (symtropy_phi - pp).abs());

            ValidationResult {
                symtropy_phi,
                pyphi_phi,
                absolute_error,
                group_size: handles.len(),
                tpm_coverage: collector.coverage(),
                ticks_used: collector.ticks_recorded(),
                error: result.error,
            }
        }
        Err(e) => ValidationResult {
            symtropy_phi,
            pyphi_phi: None,
            absolute_error: None,
            group_size: handles.len(),
            tpm_coverage: collector.coverage(),
            ticks_used: collector.ticks_recorded(),
            error: Some(e),
        },
    }
}

// ────────────────────────────────────────────────────────────
//  Tests (always compiled, PyPhi calls gated by availability)
// ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_state_roundtrip() {
        for n in 2..=5 {
            for idx in 0..(1 << n) {
                let state = BinaryState::from_index(idx, n);
                assert_eq!(
                    state.to_index(),
                    idx,
                    "Roundtrip failed for n={n}, idx={idx}"
                );
            }
        }
    }

    #[test]
    fn binary_state_examples() {
        // [0, 0, 0] = index 0
        assert_eq!(BinaryState(vec![0, 0, 0]).to_index(), 0);
        // [1, 0, 0] = index 1 (bit 0 set)
        assert_eq!(BinaryState(vec![1, 0, 0]).to_index(), 1);
        // [0, 1, 0] = index 2 (bit 1 set)
        assert_eq!(BinaryState(vec![0, 1, 0]).to_index(), 2);
        // [1, 1, 0] = index 3
        assert_eq!(BinaryState(vec![1, 1, 0]).to_index(), 3);
        // [1, 1, 1] = index 7
        assert_eq!(BinaryState(vec![1, 1, 1]).to_index(), 7);
    }

    #[test]
    fn empty_tpm_uniform() {
        let tpm = EmpiricalTpm::new(3);
        let sbn = tpm.to_state_by_node();
        // Unvisited states should have uniform 0.5
        for row in &sbn {
            for &val in row {
                assert!((val - 0.5).abs() < 1e-10, "Unvisited state should be 0.5");
            }
        }
        assert_eq!(sbn.len(), 8); // 2^3
        assert_eq!(sbn[0].len(), 3);
    }

    #[test]
    fn tpm_deterministic_transition() {
        let mut tpm = EmpiricalTpm::new(2);
        // Always: [0,0] → [1,1], [1,1] → [0,0]
        for _ in 0..100 {
            let from = BinaryState(vec![0, 0]);
            let to = BinaryState(vec![1, 1]);
            tpm.record_transition(&from, &to);

            let from2 = BinaryState(vec![1, 1]);
            let to2 = BinaryState(vec![0, 0]);
            tpm.record_transition(&from2, &to2);
        }
        let sbn = tpm.to_state_by_node();
        // State [0,0] (index 0): always goes to [1,1] → both nodes ON
        assert!((sbn[0][0] - 1.0).abs() < 1e-10);
        assert!((sbn[0][1] - 1.0).abs() < 1e-10);
        // State [1,1] (index 3): always goes to [0,0] → both nodes OFF
        assert!((sbn[3][0] - 0.0).abs() < 1e-10);
        assert!((sbn[3][1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn tpm_coverage_tracking() {
        let mut tpm = EmpiricalTpm::new(2); // 4 states
        assert_eq!(tpm.coverage(), 0.0);

        tpm.record_transition(&BinaryState(vec![0, 0]), &BinaryState(vec![1, 0]));
        assert!((tpm.coverage() - 0.25).abs() < 1e-10); // 1/4

        tpm.record_transition(&BinaryState(vec![1, 0]), &BinaryState(vec![1, 1]));
        assert!((tpm.coverage() - 0.5).abs() < 1e-10); // 2/4
    }

    #[test]
    fn discretize_agents_threshold() {
        let mut entities = BTreeMap::new();
        let h0 = BodyHandle(0);
        let h1 = BodyHandle(1);
        let h2 = BodyHandle(2);

        let mut e0 = EntityConsciousness::new(100.0);
        e0.result = Some(symthaea_consciousness_equation::ConsciousnessResult {
            consciousness_level: 0.5,
            bottleneck_factor: 0.5,
            weighted_sum: 0.5,
            embodiment_factor: 1.0,
            narrative_coherence: 1.0,
            social_embedding: 1.0,
            temporal_stability: 1.0,
            bottleneck_name: "phi".to_string(),
            factors: symthaea_consciousness_equation::ConsciousnessInputs::default(),
        }); // above threshold
        let mut e1 = EntityConsciousness::new(100.0);
        e1.result = Some(symthaea_consciousness_equation::ConsciousnessResult {
            consciousness_level: 0.1,
            bottleneck_factor: 0.1,
            weighted_sum: 0.1,
            embodiment_factor: 1.0,
            narrative_coherence: 1.0,
            social_embedding: 1.0,
            temporal_stability: 1.0,
            bottleneck_name: "phi".to_string(),
            factors: symthaea_consciousness_equation::ConsciousnessInputs::default(),
        }); // below threshold
        let mut e2 = EntityConsciousness::new(100.0);
        e2.result = Some(symthaea_consciousness_equation::ConsciousnessResult {
            consciousness_level: 0.3,
            bottleneck_factor: 0.3,
            weighted_sum: 0.3,
            embodiment_factor: 1.0,
            narrative_coherence: 1.0,
            social_embedding: 1.0,
            temporal_stability: 1.0,
            bottleneck_name: "phi".to_string(),
            factors: symthaea_consciousness_equation::ConsciousnessInputs::default(),
        }); // at threshold (≥ means ON)

        entities.insert(h0, e0);
        entities.insert(h1, e1);
        entities.insert(h2, e2);

        let state = discretize_agents(&entities, &[h0, h1, h2], 0.3);
        assert_eq!(state.0, vec![1, 0, 1]);
    }

    #[test]
    fn json_serialization_format() {
        let tpm = vec![
            vec![0.0, 0.5],
            vec![1.0, 0.0],
            vec![0.5, 0.5],
            vec![0.0, 1.0],
        ];
        let state = vec![1u8, 0];
        let labels = vec!["A0".to_string(), "A1".to_string()];

        let json = serde_json_minimal(&tpm, &state, &labels);
        assert!(json.contains("\"tpm\":"));
        assert!(json.contains("\"state\":[1,0]"));
        assert!(json.contains("\"node_labels\":[\"A0\",\"A1\"]"));
    }

    #[test]
    fn parse_pyphi_response_success() {
        let json = r#"{"phi": 0.3125, "num_concepts": 3, "major_complex_size": 3, "error": null}"#;
        let result = parse_pyphi_response(json).unwrap();
        assert!((result.phi.unwrap() - 0.3125).abs() < 1e-10);
        assert_eq!(result.num_concepts, 3);
        assert!(result.error.is_none());
    }

    #[test]
    fn parse_pyphi_response_error() {
        let json = r#"{"phi": null, "num_concepts": 0, "major_complex_size": 0, "error": "pyphi not installed"}"#;
        let result = parse_pyphi_response(json).unwrap();
        assert!(result.phi.is_none());
        assert_eq!(result.error.as_deref(), Some("pyphi not installed"));
    }

    #[test]
    fn collector_basic() {
        use nalgebra::SVector;
        use symtropy_math::Point;
        use symtropy_physics::PhysicsWorld;

        let mut world = PhysicsWorld::<2>::new(SVector::zeros());
        let mut field = ConsciousnessField::<2>::new();

        let h0 = world.add_sphere(Point::new([0.0, 0.0]), 1.0, 1.0);
        let h1 = world.add_sphere(Point::new([3.0, 0.0]), 1.0, 1.0);
        let h2 = world.add_sphere(Point::new([6.0, 0.0]), 1.0, 1.0);

        field.register(h0, 100.0, 2.0);
        field.register(h1, 100.0, 2.0);
        field.register(h2, 100.0, 2.0);

        let mut collector = TpmCollector::new(vec![h0, h1, h2], PHI_BINARY_THRESHOLD);
        assert_eq!(collector.ticks_recorded(), 0);
        assert_eq!(collector.coverage(), 0.0);

        // Record some ticks
        for _ in 0..5 {
            collector.tick(&field);
        }
        // First tick has no prev_state, so 4 transitions recorded
        assert_eq!(collector.ticks_recorded(), 4);
    }
}
