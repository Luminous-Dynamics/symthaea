//! Python bindings for WisdomEngine via PyO3
//!
//! This module exposes the WisdomEngine to Python, enabling:
//! - Pattern learning and tracking
//! - HDC-based associative learning
//! - Trust scoring and recommendations
//!
//! Usage from Python:
//! ```python
//! from mycelix_wisdom import PyWisdomBridge
//!
//! bridge = PyWisdomBridge()
//! pattern_id = bridge.learn_pattern("nixos/packages", "install firefox", "nix-env -iA nixpkgs.firefox")
//! bridge.record_outcome(pattern_id, True, "Successfully installed")
//! ```

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{
    SymthaeaCausalBridge, SymthaeaPattern, PatternEpistemics,
    OracleVerificationLevel,
    PatternId, SymthaeaId,
};

// ==============================================================================
// Pattern Epistemics (E/N/M 3D Classification)
// ==============================================================================

/// Python wrapper for PatternEpistemics - the 3D Epistemic Cube
#[pyclass(name = "PyPatternEpistemics")]
#[derive(Clone)]
pub struct PyPatternEpistemics {
    inner: PatternEpistemics,
}

#[pymethods]
impl PyPatternEpistemics {
    /// Create new epistemics with E/N/M values
    ///
    /// Args:
    ///     empirical: E-axis (0-4) - How verifiable is this?
    ///     normative: N-axis (0-3) - Who agrees this is binding?
    ///     materiality: M-axis (0-3) - How long does this matter?
    #[new]
    #[pyo3(signature = (empirical=2, normative=1, materiality=1))]
    pub fn new(empirical: u8, normative: u8, materiality: u8) -> Self {
        Self {
            inner: PatternEpistemics {
                e_level: empirical,
                n_level: normative,
                m_level: materiality,
            },
        }
    }

    /// Get the empirical level (E-axis: 0=Null, 1=Testimonial, 2=Private, 3=ZKP, 4=Public)
    #[getter]
    pub fn empirical(&self) -> u8 {
        self.inner.e_level
    }

    /// Get the normative level (N-axis: 0=Personal, 1=Communal, 2=Network, 3=Axiomatic)
    #[getter]
    pub fn normative(&self) -> u8 {
        self.inner.n_level
    }

    /// Get the materiality level (M-axis: 0=Ephemeral, 1=Temporal, 2=Persistent, 3=Foundational)
    #[getter]
    pub fn materiality(&self) -> u8 {
        self.inner.m_level
    }

    fn __repr__(&self) -> String {
        format!(
            "PyPatternEpistemics(E{}, N{}, M{})",
            self.inner.e_level,
            self.inner.n_level,
            self.inner.m_level
        )
    }
}

// ==============================================================================
// Symthaea Pattern
// ==============================================================================

/// Python wrapper for a learned pattern
#[pyclass(name = "PyPattern")]
#[derive(Clone)]
pub struct PyPattern {
    inner: SymthaeaPattern,
}

#[pymethods]
impl PyPattern {
    /// Get the pattern ID
    #[getter]
    pub fn pattern_id(&self) -> u64 {
        self.inner.pattern_id
    }

    /// Get the problem domain
    #[getter]
    pub fn problem_domain(&self) -> String {
        self.inner.problem_domain.clone()
    }

    /// Get the solution
    #[getter]
    pub fn solution(&self) -> String {
        self.inner.solution.clone()
    }

    /// Get the success rate (0.0 - 1.0)
    #[getter]
    pub fn success_rate(&self) -> f32 {
        self.inner.success_rate
    }

    /// Get the usage count
    #[getter]
    pub fn usage_count(&self) -> u64 {
        self.inner.usage_count
    }

    /// Get the success count
    #[getter]
    pub fn success_count(&self) -> u64 {
        self.inner.success_count
    }

    /// Get the description
    #[getter]
    pub fn description(&self) -> String {
        self.inner.description.clone()
    }

    /// Get the epistemic classification
    #[getter]
    pub fn epistemics(&self) -> PyPatternEpistemics {
        PyPatternEpistemics {
            inner: self.inner.epistemic_classification.clone(),
        }
    }

    /// Whether validated in production
    #[getter]
    pub fn validated_in_production(&self) -> bool {
        self.inner.validated_in_production
    }

    fn __repr__(&self) -> String {
        format!(
            "PyPattern(id={}, domain='{}', success_rate={:.2}%, uses={})",
            self.inner.pattern_id,
            self.inner.problem_domain,
            self.inner.success_rate * 100.0,
            self.inner.usage_count
        )
    }
}

// ==============================================================================
// Action Prediction (from HDC Learner)
// ==============================================================================

/// Python wrapper for HDC action predictions
#[pyclass(name = "PyActionPrediction")]
#[derive(Clone)]
pub struct PyActionPrediction {
    /// The recommended action
    #[pyo3(get)]
    pub action: String,
    /// Confidence score (0.0 - 1.0)
    #[pyo3(get)]
    pub confidence: f32,
    /// Whether this is a novel prediction (no direct experience)
    #[pyo3(get)]
    pub is_novel: bool,
}

#[pymethods]
impl PyActionPrediction {
    fn __repr__(&self) -> String {
        format!(
            "PyActionPrediction(action='{}', confidence={:.2}%, novel={})",
            self.action,
            self.confidence * 100.0,
            self.is_novel
        )
    }
}

// ==============================================================================
// Main Wisdom Bridge
// ==============================================================================

/// Python wrapper for SymthaeaCausalBridge - the main WisdomEngine interface
///
/// This is the primary entry point for Python code to interact with the
/// WisdomEngine. It provides pattern learning, outcome recording, and
/// HDC-based associative learning.
#[pyclass(name = "PyWisdomBridge")]
pub struct PyWisdomBridge {
    inner: SymthaeaCausalBridge,
    next_pattern_id: PatternId,
    symthaea_id: SymthaeaId,
}

#[pymethods]
impl PyWisdomBridge {
    /// Create a new WisdomBridge
    ///
    /// Args:
    ///     symthaea_id: Optional identifier for this Symthaea instance
    #[new]
    #[pyo3(signature = (symthaea_id=None))]
    pub fn new(symthaea_id: Option<u64>) -> Self {
        let mut bridge = SymthaeaCausalBridge::new();
        let sid = symthaea_id.unwrap_or(1);
        bridge.symthaea_id = Some(sid);

        Self {
            inner: bridge,
            next_pattern_id: 1,
            symthaea_id: sid,
        }
    }

    /// Learn a new pattern from experience
    ///
    /// Args:
    ///     problem_domain: The domain/category (e.g., "nixos/packages")
    ///     description: Human-readable description
    ///     solution: The solution/action
    ///     phi: Optional consciousness level (0.0-1.0)
    ///
    /// Returns:
    ///     Pattern ID for future reference
    #[pyo3(signature = (problem_domain, description, solution, phi=None))]
    pub fn learn_pattern(
        &mut self,
        problem_domain: &str,
        description: &str,
        solution: &str,
        phi: Option<f32>,
    ) -> u64 {
        let pattern_id = self.next_pattern_id;
        self.next_pattern_id += 1;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut pattern = SymthaeaPattern::new(
            pattern_id,
            problem_domain,
            solution,
            phi.unwrap_or(0.5),
            timestamp,
            self.symthaea_id,
        );
        pattern.description = description.to_string();

        self.inner.on_pattern_learned(pattern)
    }

    /// Record an outcome for a pattern usage
    ///
    /// Args:
    ///     pattern_id: The pattern that was used
    ///     success: Whether it succeeded
    ///     context: Optional context string
    #[pyo3(signature = (pattern_id, success, context=None))]
    pub fn record_outcome(
        &mut self,
        pattern_id: u64,
        success: bool,
        context: Option<&str>,
    ) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        self.inner.on_pattern_used(
            pattern_id,
            success,
            timestamp,
            context.unwrap_or(""),
            OracleVerificationLevel::Testimonial,
        );
    }

    /// Get a pattern by ID
    ///
    /// Args:
    ///     pattern_id: The pattern ID
    ///
    /// Returns:
    ///     PyPattern if found, None otherwise
    pub fn get_pattern(&self, pattern_id: u64) -> Option<PyPattern> {
        self.inner.patterns.get(&pattern_id).map(|p| PyPattern {
            inner: p.clone(),
        })
    }

    /// List all patterns in a domain
    ///
    /// Args:
    ///     domain: The domain to filter by (or None for all)
    ///
    /// Returns:
    ///     List of patterns
    #[pyo3(signature = (domain=None))]
    pub fn list_patterns(&self, domain: Option<&str>) -> Vec<PyPattern> {
        self.inner
            .patterns
            .values()
            .filter(|p| {
                domain
                    .map(|d| p.problem_domain.contains(d))
                    .unwrap_or(true)
            })
            .map(|p| PyPattern { inner: p.clone() })
            .collect()
    }

    /// Get patterns sorted by success rate
    ///
    /// Args:
    ///     domain: Optional domain filter
    ///     limit: Maximum number of patterns to return
    ///
    /// Returns:
    ///     List of patterns sorted by success rate (descending)
    #[pyo3(signature = (domain=None, limit=10))]
    pub fn top_patterns(&self, domain: Option<&str>, limit: usize) -> Vec<PyPattern> {
        let mut patterns: Vec<_> = self
            .inner
            .patterns
            .values()
            .filter(|p| {
                domain
                    .map(|d| p.problem_domain.contains(d))
                    .unwrap_or(true)
            })
            .cloned()
            .collect();

        patterns.sort_by(|a, b| {
            b.success_rate
                .partial_cmp(&a.success_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        patterns
            .into_iter()
            .take(limit)
            .map(|p| PyPattern { inner: p })
            .collect()
    }

    /// Get the total number of patterns
    #[getter]
    pub fn pattern_count(&self) -> usize {
        self.inner.patterns.len()
    }

    /// Get overall statistics
    pub fn stats(&self) -> PyObject {
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);

            let total_patterns = self.inner.patterns.len();
            let total_uses: u64 = self.inner.patterns.values().map(|p| p.usage_count).sum();
            let total_successes: u64 = self.inner.patterns.values().map(|p| p.success_count).sum();
            let avg_success_rate = if total_uses > 0 {
                total_successes as f32 / total_uses as f32
            } else {
                0.0
            };

            let _ = dict.set_item("total_patterns", total_patterns);
            let _ = dict.set_item("total_uses", total_uses);
            let _ = dict.set_item("total_successes", total_successes);
            let _ = dict.set_item("average_success_rate", avg_success_rate);

            dict.into()
        })
    }

    // ==========================================================================
    // HDC Associative Learner Methods
    // ==========================================================================

    /// Register an action for HDC learning
    ///
    /// Args:
    ///     action: The action identifier to register
    pub fn register_action(&mut self, action: &str) {
        self.inner.associative_learner.register_action(action);
    }

    /// Learn from an experience using HDC
    ///
    /// Args:
    ///     context: List of [key, value] pairs describing the context
    ///     action: The action taken
    ///     outcome: The outcome value (0.0 = failure, 1.0 = success)
    pub fn hdc_learn(&mut self, context: Vec<(String, String)>, action: &str, outcome: f32) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Convert to borrowed form for the API
        let ctx: Vec<(&str, &str)> = context.iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        self.inner.associative_learner.learn(&ctx, action, outcome, timestamp);
    }

    /// Predict the best action for a given context using HDC
    ///
    /// Args:
    ///     context: List of [key, value] pairs describing the current context
    ///
    /// Returns:
    ///     List of action predictions sorted by confidence
    pub fn hdc_predict(&mut self, context: Vec<(String, String)>) -> Vec<PyActionPrediction> {
        // Convert to borrowed form for the API
        let ctx: Vec<(&str, &str)> = context.iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let predictions = self.inner.associative_learner.predict(&ctx);

        predictions
            .into_iter()
            .map(|p| PyActionPrediction {
                action: p.action,
                confidence: p.confidence,
                is_novel: p.is_novel,
            })
            .collect()
    }

    // ==========================================================================
    // Trust-Weighted Scoring
    // ==========================================================================

    /// Get trust score for an agent
    ///
    /// Args:
    ///     agent_id: The agent identifier
    ///
    /// Returns:
    ///     Trust score (0.0 - 1.0), defaults to 0.5 for unknown agents
    pub fn get_trust(&self, agent_id: u64) -> f32 {
        self.inner.trust_registry.trust_score(agent_id)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyWisdomBridge(patterns={}, symthaea_id={})",
            self.inner.patterns.len(),
            self.symthaea_id
        )
    }
}

// ==============================================================================
// Module Registration
// ==============================================================================

/// Register all Python bindings
///
/// This function is called from lib.rs to register all Python classes.
/// The #[pymodule] is defined in lib.rs to avoid duplicate symbol errors.
pub fn register_wisdom_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWisdomBridge>()?;
    m.add_class::<PyPattern>()?;
    m.add_class::<PyPatternEpistemics>()?;
    m.add_class::<PyActionPrediction>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wisdom_bridge_creation() {
        let bridge = PyWisdomBridge::new(Some(42));
        assert_eq!(bridge.symthaea_id, 42);
        assert_eq!(bridge.pattern_count(), 0);
    }

    #[test]
    fn test_pattern_learning() {
        let mut bridge = PyWisdomBridge::new(None);

        let pattern_id = bridge.learn_pattern(
            "nixos/packages",
            "Install Firefox browser",
            "nix-env -iA nixpkgs.firefox",
            Some(0.8),
        );

        assert_eq!(pattern_id, 1);
        assert_eq!(bridge.pattern_count(), 1);

        let pattern = bridge.get_pattern(pattern_id).unwrap();
        assert_eq!(pattern.problem_domain(), "nixos/packages");
        assert_eq!(pattern.solution(), "nix-env -iA nixpkgs.firefox");
    }

    #[test]
    fn test_outcome_recording() {
        let mut bridge = PyWisdomBridge::new(None);

        let pattern_id = bridge.learn_pattern(
            "nixos/packages",
            "Install vim",
            "nix-env -iA nixpkgs.vim",
            None,
        );

        // Record some outcomes
        bridge.record_outcome(pattern_id, true, Some("test context"));
        bridge.record_outcome(pattern_id, true, None);
        bridge.record_outcome(pattern_id, false, None);

        let pattern = bridge.get_pattern(pattern_id).unwrap();
        assert_eq!(pattern.usage_count(), 3);
        assert_eq!(pattern.success_count(), 2);
    }

    #[test]
    fn test_epistemics() {
        let epistemics = PyPatternEpistemics::new(3, 2, 1);
        assert_eq!(epistemics.inner.e_level, 3);
        assert_eq!(epistemics.inner.n_level, 2);
        assert_eq!(epistemics.inner.m_level, 1);
    }
}
