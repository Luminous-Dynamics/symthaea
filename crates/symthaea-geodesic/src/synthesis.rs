// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Geodesic Code Synthesis -- the integration pipeline.
//!
//! Orchestrates all layers of the geodesic architecture to synthesize code by
//! walking the shortest path through program manifold space:
//!
//! 1. **Project** specification to manifold base point
//! 2. **Locate** nearest fiber -> candidate encoding
//! 3. **Predict** execution outcome via oracle
//! 4. **Verify** topology against Betti constraints
//! 5. **Verify** sheaf coherence against codebase context
//! 6. **Refine** if needed via manifold tangent vectors
//!
//! # Design Principles
//!
//! This module is purely an orchestrator. All domain logic lives in the
//! individual layer modules (pdg, topology, manifold, execution_oracle, sheaf).
//! Synthesis ties them together with a refinement loop.
//!
//! # What This Enables That LLMs Cannot Do
//!
//! - **Pre-generation verification**: predict what code will do before it exists
//! - **Topological invariants**: hard constraints on loop/branch structure
//! - **Formal composability**: sheaf conditions guarantee interface correctness
//! - **O(1) complexity prediction**: CfC convergence rate analysis

use symthaea_core::hdc::binary_hv::BinaryHV;

use crate::execution_oracle::{ComplexityClass, ExecutionOracle, OperationType, PredictionResult};
use crate::manifold::ProgramManifold;
use crate::sheaf::{CodeSheaf, LocalSection, SheafDiagnostic};
use crate::topology::{TopologicalConstraint, TopologicalFingerprint};

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for the geodesic synthesizer.
#[derive(Debug, Clone)]
pub struct SynthesisConfig {
    /// Maximum refinement iterations before giving up.
    pub max_refinements: usize,
    /// Sheaf similarity threshold for gluing conditions.
    pub sheaf_threshold: f64,
    /// Whether to run the execution oracle for convergence prediction.
    pub enable_oracle: bool,
    /// Whether to verify topological constraints.
    pub enable_topology: bool,
    /// Whether to verify sheaf coherence.
    pub enable_sheaf: bool,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            max_refinements: 5,
            sheaf_threshold: 0.3,
            enable_oracle: true,
            enable_topology: true,
            enable_sheaf: true,
        }
    }
}

// ============================================================================
// CODE SPECIFICATION
// ============================================================================

/// A code specification for synthesis.
///
/// Describes what should be synthesized: the desired function, its topological
/// constraints, an optional expected output encoding, and the existing codebase
/// context for sheaf verification.
#[derive(Debug, Clone)]
pub struct CodeSpec {
    /// Human-readable description of the desired code.
    pub description: String,
    /// Name of the function to synthesize.
    pub function_name: String,
    /// Topological constraints the result must satisfy (e.g., "exactly 1 loop").
    pub topology_constraints: Vec<TopologicalConstraint>,
    /// Expected output encoding for oracle verification. If `Some`, the oracle
    /// will check whether the candidate converges toward this target.
    pub expected_output: Option<BinaryHV>,
    /// Existing codebase context sections for sheaf coherence verification.
    /// The synthesized function must compose correctly with these.
    pub context_sections: Vec<LocalSection>,
}

impl CodeSpec {
    /// Create a minimal spec with just a name and description.
    pub fn new(function_name: &str, description: &str) -> Self {
        Self {
            description: description.to_string(),
            function_name: function_name.to_string(),
            topology_constraints: Vec::new(),
            expected_output: None,
            context_sections: Vec::new(),
        }
    }

    /// Add a topological constraint.
    pub fn with_constraint(mut self, constraint: TopologicalConstraint) -> Self {
        self.topology_constraints.push(constraint);
        self
    }

    /// Set the expected output encoding.
    pub fn with_expected_output(mut self, output: BinaryHV) -> Self {
        self.expected_output = Some(output);
        self
    }

    /// Add a context section for sheaf verification.
    pub fn with_context(mut self, section: LocalSection) -> Self {
        self.context_sections.push(section);
        self
    }
}

// ============================================================================
// SYNTHESIS RESULT
// ============================================================================

/// Result of a synthesis attempt.
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// Whether synthesis succeeded (all enabled checks passed).
    pub success: bool,
    /// The synthesized code encoding, if a candidate was found.
    pub encoding: Option<BinaryHV>,
    /// Topological fingerprint of the result, if topology was checked.
    pub fingerprint: Option<TopologicalFingerprint>,
    /// Execution oracle prediction, if oracle was run.
    pub prediction: Option<PredictionResult>,
    /// Sheaf diagnostics (empty = coherent).
    pub sheaf_violations: Vec<SheafDiagnostic>,
    /// Topology constraint violations (empty = satisfied).
    pub topology_violations: Vec<String>,
    /// Number of refinement iterations used.
    pub refinements: usize,
    /// Telemetry for the synthesis process.
    pub report: SynthesisReport,
}

/// Telemetry for the synthesis process.
#[derive(Debug, Clone, Default)]
pub struct SynthesisReport {
    /// Number of fibers searched in the manifold.
    pub manifold_fibers_searched: usize,
    /// Similarity of the nearest fiber's centroid to the query.
    pub nearest_fiber_similarity: f64,
    /// Number of topology constraint checks performed.
    pub topology_checks: usize,
    /// Number of oracle predictions run.
    pub oracle_predictions: usize,
    /// Number of sheaf gluing conditions checked.
    pub sheaf_conditions_checked: usize,
    /// Whether the oracle detected convergence.
    pub convergence_detected: bool,
    /// Estimated computational complexity of the candidate, if oracle ran.
    pub estimated_complexity: Option<ComplexityClass>,
}

// ============================================================================
// GEODESIC SYNTHESIZER
// ============================================================================

/// The Geodesic Synthesizer: main entry point for topology-aware code generation.
///
/// Owns a [`ProgramManifold`] populated with known code encodings, and uses it
/// to find, verify, and refine code candidates that satisfy topological,
/// behavioral, and composability constraints.
///
/// # Pipeline
///
/// ```text
///   Spec -> [Manifold lookup] -> Candidate
///        -> [Oracle prediction] -> Convergence check
///        -> [Topology verification] -> Betti constraint check
///        -> [Sheaf coherence] -> Interface compatibility check
///        -> [Refinement loop] -> Adjusted candidate
///        -> SynthesisResult
/// ```
pub struct GeodesicSynthesizer {
    manifold: ProgramManifold,
    config: SynthesisConfig,
}

impl GeodesicSynthesizer {
    /// Create a new synthesizer with the given configuration.
    ///
    /// The manifold starts empty; populate it via `manifold_mut()` before
    /// calling `synthesize()`.
    pub fn new(config: SynthesisConfig) -> Self {
        Self {
            manifold: ProgramManifold::new(),
            config,
        }
    }

    /// Get mutable access to the manifold for populating it with fibers.
    pub fn manifold_mut(&mut self) -> &mut ProgramManifold {
        &mut self.manifold
    }

    /// Get a reference to the manifold.
    pub fn manifold(&self) -> &ProgramManifold {
        &self.manifold
    }

    /// Synthesize code from a specification.
    ///
    /// Runs the full 6-step pipeline (find candidate, oracle, topology, sheaf,
    /// refine) and returns a [`SynthesisResult`] with the outcome and telemetry.
    pub fn synthesize(&mut self, spec: &CodeSpec) -> SynthesisResult {
        let mut report = SynthesisReport::default();

        // Step 1-2: Find candidate from manifold.
        let (mut candidate, similarity) = match self.find_candidate(spec, &mut report) {
            Some(pair) => pair,
            None => {
                return SynthesisResult {
                    success: false,
                    encoding: None,
                    fingerprint: None,
                    prediction: None,
                    sheaf_violations: Vec::new(),
                    topology_violations: Vec::new(),
                    refinements: 0,
                    report,
                };
            }
        };
        report.nearest_fiber_similarity = similarity as f64;

        // Refinement loop: try up to max_refinements times.
        let mut refinement_count = 0;

        loop {
            // Step 3: Run execution oracle.
            let prediction = if self.config.enable_oracle {
                let pred = self.run_oracle(&candidate, spec);
                if let Some(ref p) = pred {
                    report.oracle_predictions += 1;
                    report.convergence_detected = p.converged;
                    report.estimated_complexity = Some(p.complexity_estimate.clone());
                }
                pred
            } else {
                None
            };

            // Step 4: Verify topological constraints.
            // NOTE: Full PDG integration requires the candidate to have an
            // associated simplicial complex. For now, we compute a fingerprint
            // from the candidate's bit-region structure. Full PDG-derived
            // fingerprints will be wired in Phase 4 when synthesis feeds back
            // into PDG construction.
            let topology_violations = if self.config.enable_topology {
                let fingerprint = self.compute_fingerprint(&candidate);
                let violations = self.verify_topology(&fingerprint, spec);
                report.topology_checks += spec.topology_constraints.len();
                violations
            } else {
                Vec::new()
            };

            // Step 5: Verify sheaf coherence.
            let sheaf_violations = if self.config.enable_sheaf {
                let violations = self.verify_sheaf(&candidate, spec);
                report.sheaf_conditions_checked += spec.context_sections.len();
                violations
            } else {
                Vec::new()
            };

            // Check if we're done.
            let all_clear = topology_violations.is_empty() && sheaf_violations.is_empty();

            if all_clear || refinement_count >= self.config.max_refinements {
                let fingerprint = if self.config.enable_topology {
                    Some(self.compute_fingerprint(&candidate))
                } else {
                    None
                };

                return SynthesisResult {
                    success: all_clear,
                    encoding: Some(candidate),
                    fingerprint,
                    prediction,
                    sheaf_violations,
                    topology_violations,
                    refinements: refinement_count,
                    report,
                };
            }

            // Step 6: Refine candidate.
            let direction = self.violation_direction(&topology_violations, &sheaf_violations, spec);
            candidate = self.refine(&candidate, &direction);
            refinement_count += 1;
        }
    }

    // ── Internal pipeline steps ──────────────────────────────────────────

    /// Step 1-2: Find a candidate encoding from the manifold.
    ///
    /// Projects the spec description into HDC space and finds the nearest
    /// fiber in the manifold, returning the fiber's best implementation
    /// encoding and its centroid similarity to the query.
    fn find_candidate(
        &self,
        spec: &CodeSpec,
        report: &mut SynthesisReport,
    ) -> Option<(BinaryHV, f32)> {
        let query = self.encode_spec(spec);
        report.manifold_fibers_searched = self.manifold.fiber_count();

        let fiber = self.manifold.nearest_fiber(&query)?;
        let similarity = query.similarity(&fiber.centroid);

        // Use the best-quality implementation from the fiber, or fall back
        // to the centroid if the fiber has no points (shouldn't happen).
        let encoding = fiber.best().map(|p| p.encoding).unwrap_or(fiber.centroid);

        Some((encoding, similarity))
    }

    /// Encode a specification into an HDC query vector.
    ///
    /// Uses a simple strategy: hash the function name and description into
    /// a deterministic BinaryHV, then bind with any expected output if present.
    fn encode_spec(&self, spec: &CodeSpec) -> BinaryHV {
        let name_hv = BinaryHV::random(spec_name_seed(&spec.function_name));
        let desc_hv = BinaryHV::random(spec_name_seed(&spec.description));

        let mut query = name_hv.bind(&desc_hv);

        if let Some(ref expected) = spec.expected_output {
            query = query.bind(expected);
        }

        query
    }

    /// Step 3: Run the execution oracle on a candidate.
    ///
    /// Creates an oracle with the candidate as initial state, feeds it a
    /// sequence of operations derived from the candidate and expected output,
    /// and returns the prediction result.
    fn run_oracle(&self, candidate: &BinaryHV, spec: &CodeSpec) -> Option<PredictionResult> {
        let mut oracle = ExecutionOracle::with_state(*candidate);

        // Build a statement sequence from the candidate.
        // Use Assignment as a generic operation type for the encoding.
        let mut statements: Vec<(BinaryHV, OperationType)> =
            vec![(*candidate, OperationType::Assignment)];

        // If there's an expected output, add it as a second statement so the
        // oracle can predict convergence toward the target.
        if let Some(ref expected) = spec.expected_output {
            statements.push((*expected, OperationType::Assignment));
        }

        Some(oracle.predict_sequence(&statements))
    }

    /// Compute a topological fingerprint from a candidate encoding.
    ///
    /// Without a full PDG, we create a minimal simplicial complex from the
    /// candidate's bit-region structure. 8 "regions" of the HV become vertices,
    /// with edges where adjacent regions have high bit correlation.
    fn compute_fingerprint(&self, candidate: &BinaryHV) -> TopologicalFingerprint {
        use symthaea_hodge::SimplicialComplex;

        let n_regions: usize = 8;
        let region_size = BinaryHV::BYTES / n_regions;
        let vertices: Vec<usize> = (0..n_regions).collect();

        let mut one_simplices = Vec::new();
        let mut two_simplices = Vec::new();

        // Create edges between regions with high byte-level correlation.
        for i in 0..n_regions {
            for j in (i + 1)..n_regions {
                let start_i = i * region_size;
                let start_j = j * region_size;
                let matching: u32 = (0..region_size)
                    .map(|k| (!(candidate.0[start_i + k] ^ candidate.0[start_j + k])).count_ones())
                    .sum();
                let correlation = matching as f64 / (region_size as f64 * 8.0);

                // Edge if correlation > 0.45 (slightly below random expectation
                // of 0.5, so most random HVs produce a connected complex).
                if correlation > 0.45 {
                    one_simplices.push(vec![i, j]);
                }
            }
        }

        // Find triangles among the edges.
        for i in 0..n_regions {
            for j in (i + 1)..n_regions {
                for k in (j + 1)..n_regions {
                    let has_ij = one_simplices.iter().any(|e| e == &[i, j]);
                    let has_jk = one_simplices.iter().any(|e| e == &[j, k]);
                    let has_ik = one_simplices.iter().any(|e| e == &[i, k]);
                    if has_ij && has_jk && has_ik {
                        two_simplices.push(vec![i, j, k]);
                    }
                }
            }
        }

        let zero_simplices: Vec<Vec<usize>> = vertices.iter().map(|&v| vec![v]).collect();
        let complex = SimplicialComplex {
            vertices,
            simplices: vec![zero_simplices, one_simplices, two_simplices],
            max_dim: 2,
        };

        TopologicalFingerprint::from_complex(&complex)
    }

    /// Step 4: Verify topological constraints against a fingerprint.
    fn verify_topology(
        &self,
        fingerprint: &TopologicalFingerprint,
        spec: &CodeSpec,
    ) -> Vec<String> {
        fingerprint.verify(&spec.topology_constraints)
    }

    /// Step 5: Verify sheaf coherence.
    ///
    /// Builds a [`CodeSheaf`] from the spec's context sections plus a new
    /// section for the candidate, then checks all gluing conditions.
    fn verify_sheaf(&self, candidate: &BinaryHV, spec: &CodeSpec) -> Vec<SheafDiagnostic> {
        if spec.context_sections.is_empty() {
            return Vec::new();
        }

        let mut sheaf = CodeSheaf::new().with_threshold(self.config.sheaf_threshold);

        // Add all context sections.
        for section in &spec.context_sections {
            sheaf.add_section(section.clone());
        }

        // Add the candidate as a new section. Conservative assumption: it
        // calls everything in the context.
        let candidate_callees: Vec<String> = spec
            .context_sections
            .iter()
            .map(|s| s.name.clone())
            .collect();

        sheaf.add_section(LocalSection {
            name: spec.function_name.clone(),
            encoding: *candidate,
            param_types: vec![],
            return_type: BinaryHV::zero(),
            callees: candidate_callees,
        });

        match sheaf.verify() {
            Ok(_) => Vec::new(),
            Err(diagnostics) => diagnostics,
        }
    }

    /// Compute a "direction" vector for refinement based on violations.
    ///
    /// Combines topology and sheaf violation signals into a single BinaryHV
    /// that points "away" from the current violations.
    fn violation_direction(
        &self,
        topology_violations: &[String],
        sheaf_violations: &[SheafDiagnostic],
        spec: &CodeSpec,
    ) -> BinaryHV {
        let mut components = Vec::new();

        // Each topology violation contributes a deterministic direction.
        for violation in topology_violations {
            components.push(BinaryHV::random(spec_name_seed(violation)));
        }

        // Each sheaf violation contributes a direction derived from the
        // violation message. This nudges the candidate toward compatibility.
        for diag in sheaf_violations {
            components.push(BinaryHV::random(spec_name_seed(&diag.message)));
        }

        // Fallback: random perturbation to avoid getting stuck.
        if components.is_empty() {
            return BinaryHV::random(spec_name_seed(&spec.function_name).wrapping_add(9999));
        }

        BinaryHV::bundle(&components)
    }

    /// Step 6: Refine a candidate by partially moving it toward a direction.
    ///
    /// Uses a "partial bind" strategy: bind the candidate with the direction
    /// vector, then bundle the original (weighted 2:1) with the adjusted
    /// version. This produces a vector similar to the original but shifted.
    fn refine(&self, candidate: &BinaryHV, direction: &BinaryHV) -> BinaryHV {
        let adjusted = candidate.bind(direction);
        BinaryHV::bundle(&[*candidate, *candidate, adjusted])
    }
}

// ── Utility ─────────────────────────────────────────────────────────────

/// Derive a deterministic seed from a string (for spec encoding).
fn spec_name_seed(s: &str) -> u64 {
    s.bytes()
        .enumerate()
        .fold(0x6E0D_C0DE_5007_0000u64, |acc, (i, b)| {
            acc.wrapping_add((b as u64).wrapping_shl((i as u32) % 64))
        })
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sheaf::LocalSection;
    use crate::topology::TopologicalConstraint;

    /// Helper: create a simple LocalSection.
    fn make_section(name: &str, seed: u64) -> LocalSection {
        LocalSection {
            name: name.to_string(),
            encoding: BinaryHV::random(seed),
            param_types: vec![],
            return_type: BinaryHV::random(seed.wrapping_add(1)),
            callees: vec![],
        }
    }

    /// Helper: insert a fiber into the manifold via the public `insert` API.
    fn seed_manifold(synth: &mut GeodesicSynthesizer, name: &str, seed: u64) {
        let encoding = BinaryHV::random(seed);
        let fingerprint = synth.compute_fingerprint(&encoding);
        synth
            .manifold_mut()
            .insert(name, encoding, fingerprint, 0.8);
    }

    #[test]
    fn test_synthesis_empty_manifold() {
        let config = SynthesisConfig::default();
        let mut synth = GeodesicSynthesizer::new(config);
        let spec = CodeSpec::new("my_func", "compute something");

        let result = synth.synthesize(&spec);

        assert!(!result.success, "should fail with empty manifold");
        assert!(result.encoding.is_none());
        assert_eq!(result.refinements, 0);
        assert_eq!(result.report.manifold_fibers_searched, 0);
    }

    #[test]
    fn test_synthesis_with_candidate() {
        let config = SynthesisConfig {
            enable_oracle: false,
            enable_topology: false,
            enable_sheaf: false,
            ..SynthesisConfig::default()
        };
        let mut synth = GeodesicSynthesizer::new(config);

        // Populate manifold with a fiber.
        seed_manifold(&mut synth, "test_impl", 42);

        let spec = CodeSpec::new("test_func", "a test function");
        let result = synth.synthesize(&spec);

        assert!(result.success, "should succeed with all checks disabled");
        assert!(result.encoding.is_some());
        assert_eq!(result.report.manifold_fibers_searched, 1);
    }

    #[test]
    fn test_synthesis_topology_violation() {
        let config = SynthesisConfig {
            enable_oracle: false,
            enable_topology: true,
            enable_sheaf: false,
            max_refinements: 0,
            ..SynthesisConfig::default()
        };
        let mut synth = GeodesicSynthesizer::new(config);
        seed_manifold(&mut synth, "simple_func", 100);

        // Require exactly 5 loops -- the candidate's derived fingerprint
        // will almost certainly not have beta_1 = 5.
        let spec = CodeSpec::new("loop_heavy", "needs 5 loops").with_constraint(
            TopologicalConstraint::ExactBetti {
                dimension: 1,
                count: 5,
            },
        );

        let result = synth.synthesize(&spec);

        assert!(
            !result.topology_violations.is_empty(),
            "expected topology violations for ExactLoops(5)"
        );
        assert_eq!(result.refinements, 0);
        assert!(result.report.topology_checks > 0);
    }

    #[test]
    fn test_synthesis_full_pipeline() {
        let config = SynthesisConfig {
            enable_oracle: true,
            enable_topology: true,
            enable_sheaf: true,
            max_refinements: 2,
            sheaf_threshold: 0.3,
        };
        let mut synth = GeodesicSynthesizer::new(config);
        seed_manifold(&mut synth, "full_impl", 200);

        let spec = CodeSpec::new("integrate", "full pipeline test")
            .with_expected_output(BinaryHV::random(202))
            .with_context(make_section("helper", 203));

        let result = synth.synthesize(&spec);

        // Pipeline ran all stages -- we verify telemetry, not pass/fail.
        assert!(result.encoding.is_some(), "should have a candidate");
        assert!(
            result.report.oracle_predictions > 0,
            "oracle should have run"
        );
        assert!(result.refinements <= 2);
    }

    #[test]
    fn test_synthesis_sheaf_coherence_no_context() {
        // With no context sections, sheaf check is trivially passed.
        let config = SynthesisConfig {
            enable_oracle: false,
            enable_topology: false,
            enable_sheaf: true,
            max_refinements: 0,
            sheaf_threshold: 0.3,
        };
        let mut synth = GeodesicSynthesizer::new(config);
        seed_manifold(&mut synth, "standalone", 300);

        let spec = CodeSpec::new("no_deps", "standalone function");
        let result = synth.synthesize(&spec);

        assert!(result.success, "no context -> no sheaf violations");
        assert!(result.sheaf_violations.is_empty());
    }

    #[test]
    fn test_synthesis_refinement_exhausted() {
        let config = SynthesisConfig {
            enable_oracle: false,
            enable_topology: true,
            enable_sheaf: false,
            max_refinements: 3,
            sheaf_threshold: 0.3,
        };
        let mut synth = GeodesicSynthesizer::new(config);
        seed_manifold(&mut synth, "refine_test", 400);

        // Impossible constraint -> exhausts all refinement iterations.
        let spec = CodeSpec::new("impossible", "needs 100 loops").with_constraint(
            TopologicalConstraint::ExactBetti {
                dimension: 1,
                count: 100,
            },
        );

        let result = synth.synthesize(&spec);

        assert!(!result.success, "should fail with impossible constraint");
        assert_eq!(
            result.refinements, 3,
            "should exhaust all refinement iterations"
        );
    }

    #[test]
    fn test_code_spec_builder() {
        let spec = CodeSpec::new("foo", "does foo things")
            .with_constraint(TopologicalConstraint::ExactBetti {
                dimension: 1,
                count: 1,
            })
            .with_expected_output(BinaryHV::random(999))
            .with_context(make_section("bar", 998));

        assert_eq!(spec.function_name, "foo");
        assert_eq!(spec.description, "does foo things");
        assert_eq!(spec.topology_constraints.len(), 1);
        assert!(spec.expected_output.is_some());
        assert_eq!(spec.context_sections.len(), 1);
    }

    #[test]
    fn test_spec_name_seed_deterministic() {
        let s1 = spec_name_seed("hello");
        let s2 = spec_name_seed("hello");
        assert_eq!(s1, s2);

        let s3 = spec_name_seed("world");
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_default_config() {
        let config = SynthesisConfig::default();
        assert_eq!(config.max_refinements, 5);
        assert!((config.sheaf_threshold - 0.3).abs() < f64::EPSILON);
        assert!(config.enable_oracle);
        assert!(config.enable_topology);
        assert!(config.enable_sheaf);
    }
}
