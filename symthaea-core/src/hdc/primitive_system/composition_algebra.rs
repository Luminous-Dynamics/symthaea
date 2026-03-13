use super::PrimitiveSystem;
use crate::hdc::binary_hv::BinaryHV;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Algebra for defining and evaluating named compositions.
///
/// Allows users to define reusable compositions like:
/// - `CAUSALITY = CAUSE ⊗ EFFECT` (bind)
/// - `TEMPORAL_FLOW = BEFORE → DURING → AFTER` (sequence)
/// - `PHYSICS_CORE = MASS + ENERGY + FORCE` (bundle)
/// - `MOSTLY_CAUSE = CAUSE:3 + EFFECT:1` (weighted bundle)
///
/// # Example
/// ```ignore
/// let system = PrimitiveSystem::global();
/// let mut algebra = CompositionAlgebra::new();
///
/// // Define compositions
/// algebra.define("CAUSALITY", "CAUSE ⊗ EFFECT", system)?;
/// algebra.define("TEMPORAL", "BEFORE → DURING → AFTER", system)?;
///
/// // Use in new expressions
/// algebra.define("CAUSAL_TIME", "CAUSALITY ⊗ TEMPORAL", system)?;
///
/// // Evaluate
/// let result = algebra.get("CAUSAL_TIME")?;
/// ```
#[derive(Debug, Clone)]
pub struct CompositionAlgebra {
    /// Named compositions: name -> (expression, encoding)
    compositions: HashMap<String, NamedComposition>,
}

/// A named composition with its expression and computed encoding
#[derive(Debug, Clone)]
pub struct NamedComposition {
    /// The name of this composition
    pub name: String,
    /// The expression that defines it (e.g., "CAUSE ⊗ EFFECT")
    pub expression: String,
    /// The computed BinaryHV encoding
    pub encoding: BinaryHV,
    /// Source primitives/compositions used
    pub sources: Vec<String>,
}

impl CompositionAlgebra {
    /// Create a new empty algebra.
    pub fn new() -> Self {
        Self {
            compositions: HashMap::new(),
        }
    }

    /// Define a new named composition from an expression.
    ///
    /// Expression syntax:
    /// - `A ⊗ B` or `A ^ B` - Bind (XOR)
    /// - `A + B + C` - Bundle (majority vote)
    /// - `A → B → C` or `A > B > C` - Sequence (position-aware)
    /// - `A:2 + B:1` - Weighted bundle
    ///
    /// Names can reference primitives or previously defined compositions.
    pub fn define(
        &mut self,
        name: &str,
        expression: &str,
        system: &PrimitiveSystem,
    ) -> Result<(), CompositionAlgebraError> {
        if name.is_empty() {
            return Err(CompositionAlgebraError::InvalidName(
                "name cannot be empty".to_string(),
            ));
        }

        let (encoding, sources) = self.evaluate_expression(expression, system)?;

        self.compositions.insert(
            name.to_string(),
            NamedComposition {
                name: name.to_string(),
                expression: expression.to_string(),
                encoding,
                sources,
            },
        );

        Ok(())
    }

    /// Get a named composition by name.
    pub fn get(&self, name: &str) -> Option<&NamedComposition> {
        self.compositions.get(name)
    }

    /// Get encoding for a name (composition or primitive).
    pub fn get_encoding(&self, name: &str, system: &PrimitiveSystem) -> Option<BinaryHV> {
        // First check compositions
        if let Some(comp) = self.compositions.get(name) {
            return Some(comp.encoding);
        }
        // Then check primitives
        system.get(name).map(|p| p.encoding)
    }

    /// List all defined compositions.
    pub fn list(&self) -> Vec<&NamedComposition> {
        self.compositions.values().collect()
    }

    /// Remove a composition.
    pub fn remove(&mut self, name: &str) -> Option<NamedComposition> {
        self.compositions.remove(name)
    }

    /// Clear all compositions.
    pub fn clear(&mut self) {
        self.compositions.clear();
    }

    /// Evaluate an expression and return the encoding.
    fn evaluate_expression(
        &self,
        expression: &str,
        system: &PrimitiveSystem,
    ) -> Result<(BinaryHV, Vec<String>), CompositionAlgebraError> {
        let expr = expression.trim();

        // Check for sequence operator (→ or >)
        if expr.contains('→') || expr.contains('>') {
            return self.evaluate_sequence(expr, system);
        }

        // Check for bind operator (⊗ or ^)
        if expr.contains('⊗') || expr.contains('^') {
            return self.evaluate_bind(expr, system);
        }

        // Check for weighted bundle (contains :)
        if expr.contains(':') && expr.contains('+') {
            return self.evaluate_weighted_bundle(expr, system);
        }

        // Check for bundle operator (+)
        if expr.contains('+') {
            return self.evaluate_bundle(expr, system);
        }

        // Single name - look up directly
        let name = expr.trim();
        if let Some(enc) = self.get_encoding(name, system) {
            return Ok((enc, vec![name.to_string()]));
        }

        Err(CompositionAlgebraError::NotFound(name.to_string()))
    }

    fn evaluate_bind(
        &self,
        expr: &str,
        system: &PrimitiveSystem,
    ) -> Result<(BinaryHV, Vec<String>), CompositionAlgebraError> {
        // Split on ⊗ or ^
        let parts: Vec<&str> = if expr.contains('⊗') {
            expr.split('⊗').collect()
        } else {
            expr.split('^').collect()
        };

        if parts.len() < 2 {
            return Err(CompositionAlgebraError::ParseError(
                "bind requires at least 2 operands".to_string(),
            ));
        }

        let mut sources = Vec::new();
        let first_name = parts[0].trim();
        let mut result = self
            .get_encoding(first_name, system)
            .ok_or_else(|| CompositionAlgebraError::NotFound(first_name.to_string()))?;
        sources.push(first_name.to_string());

        for part in &parts[1..] {
            let name = part.trim();
            let enc = self
                .get_encoding(name, system)
                .ok_or_else(|| CompositionAlgebraError::NotFound(name.to_string()))?;
            result = result.bind(&enc);
            sources.push(name.to_string());
        }

        Ok((result, sources))
    }

    fn evaluate_bundle(
        &self,
        expr: &str,
        system: &PrimitiveSystem,
    ) -> Result<(BinaryHV, Vec<String>), CompositionAlgebraError> {
        let parts: Vec<&str> = expr.split('+').collect();

        if parts.len() < 2 {
            return Err(CompositionAlgebraError::ParseError(
                "bundle requires at least 2 operands".to_string(),
            ));
        }

        let mut sources = Vec::new();
        let mut encodings = Vec::new();

        for part in &parts {
            let name = part.trim();
            let enc = self
                .get_encoding(name, system)
                .ok_or_else(|| CompositionAlgebraError::NotFound(name.to_string()))?;
            encodings.push(enc);
            sources.push(name.to_string());
        }

        let result = BinaryHV::bundle(&encodings);
        Ok((result, sources))
    }

    fn evaluate_weighted_bundle(
        &self,
        expr: &str,
        system: &PrimitiveSystem,
    ) -> Result<(BinaryHV, Vec<String>), CompositionAlgebraError> {
        let parts: Vec<&str> = expr.split('+').collect();

        let mut sources = Vec::new();
        let mut encodings = Vec::new();
        let mut weights = Vec::new();

        for part in &parts {
            let part = part.trim();
            let kv: Vec<&str> = part.split(':').collect();

            let (name, weight) = if kv.len() == 2 {
                let w: f32 = kv[1].trim().parse().map_err(|_| {
                    CompositionAlgebraError::ParseError(format!("invalid weight: {}", kv[1]))
                })?;
                (kv[0].trim(), w)
            } else {
                (part, 1.0)
            };

            let enc = self
                .get_encoding(name, system)
                .ok_or_else(|| CompositionAlgebraError::NotFound(name.to_string()))?;
            encodings.push(enc);
            weights.push(weight);
            sources.push(name.to_string());
        }

        // Normalize weights
        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            return Err(CompositionAlgebraError::ParseError(
                "weights must sum to positive value".to_string(),
            ));
        }
        let weights: Vec<f32> = weights.iter().map(|w| w / total).collect();

        // Weighted bundling
        let mut result_bytes = [0u8; 2048];
        for byte_idx in 0..2048 {
            let mut byte_val: u8 = 0;
            for bit_in_byte in 0..8 {
                let mut weighted_sum: f32 = 0.0;
                for (enc, w) in encodings.iter().zip(weights.iter()) {
                    let enc_byte = enc.0[byte_idx];
                    let bit = (enc_byte >> bit_in_byte) & 1;
                    weighted_sum += if bit == 1 { *w } else { -*w };
                }
                if weighted_sum > 0.0 {
                    byte_val |= 1u8 << bit_in_byte;
                }
            }
            result_bytes[byte_idx] = byte_val;
        }

        Ok((BinaryHV(result_bytes), sources))
    }

    fn evaluate_sequence(
        &self,
        expr: &str,
        system: &PrimitiveSystem,
    ) -> Result<(BinaryHV, Vec<String>), CompositionAlgebraError> {
        // Split on → or >
        let parts: Vec<&str> = if expr.contains('→') {
            expr.split('→').collect()
        } else {
            expr.split('>').collect()
        };

        if parts.len() < 2 {
            return Err(CompositionAlgebraError::ParseError(
                "sequence requires at least 2 elements".to_string(),
            ));
        }

        let mut sources = Vec::new();
        let first_name = parts[0].trim();
        let mut result = self
            .get_encoding(first_name, system)
            .ok_or_else(|| CompositionAlgebraError::NotFound(first_name.to_string()))?;
        sources.push(first_name.to_string());

        for (i, part) in parts[1..].iter().enumerate() {
            let name = part.trim();
            let enc = self
                .get_encoding(name, system)
                .ok_or_else(|| CompositionAlgebraError::NotFound(name.to_string()))?;
            let permuted = enc.permute(i + 1);
            result = result.bind(&permuted);
            sources.push(name.to_string());
        }

        Ok((result, sources))
    }

    /// Export all compositions to a serializable format.
    pub fn export(&self) -> Vec<CompositionExport> {
        self.compositions
            .values()
            .map(|c| CompositionExport {
                name: c.name.clone(),
                expression: c.expression.clone(),
            })
            .collect()
    }

    /// Import compositions from exported format.
    pub fn import(
        &mut self,
        exports: &[CompositionExport],
        system: &PrimitiveSystem,
    ) -> Result<usize, CompositionAlgebraError> {
        let mut count = 0;
        for exp in exports {
            self.define(&exp.name, &exp.expression, system)?;
            count += 1;
        }
        Ok(count)
    }

    /// Query what happens when a component is removed from a composite.
    ///
    /// Unbinds the `removed` component from `composite` via XOR (self-inverse),
    /// then finds the nearest named axiom to the residual. Returns the name of
    /// the nearest axiom, the similarity score, and the residual encoding.
    ///
    /// # Example
    /// ```ignore
    /// // "What happens when you remove LEGITIMACY from LEGITIMATE_GOVERNANCE?"
    /// let (nearest, sim, _) = algebra.query_transition(
    ///     "LEGITIMATE_GOVERNANCE", "LEGITIMACY", system
    /// ).unwrap();
    /// // nearest ≈ "REVOLUTION" (authority without legitimacy)
    /// ```
    pub fn query_transition(
        &self,
        composite: &str,
        removed: &str,
        system: &PrimitiveSystem,
    ) -> Result<(String, f32, BinaryHV), CompositionAlgebraError> {
        let composite_hv = self
            .get_encoding(composite, system)
            .ok_or_else(|| CompositionAlgebraError::NotFound(composite.to_string()))?;
        let removed_hv = self
            .get_encoding(removed, system)
            .ok_or_else(|| CompositionAlgebraError::NotFound(removed.to_string()))?;

        // XOR is its own inverse: A ^ B ^ B = A
        // So (A ^ B ^ C) ^ B = A ^ C — removes B from the composite.
        let residual = composite_hv.bind(&removed_hv);

        // Find nearest axiom/composition (excluding the composite itself)
        let mut best_name = String::new();
        let mut best_sim: f32 = -1.0;

        for (name, comp) in &self.compositions {
            if name == composite {
                continue;
            }
            let sim = residual.similarity(&comp.encoding);
            if sim > best_sim {
                best_sim = sim;
                best_name = name.clone();
            }
        }

        if best_name.is_empty() {
            return Err(CompositionAlgebraError::ParseError(
                "no compositions to compare against".to_string(),
            ));
        }

        Ok((best_name, best_sim, residual))
    }

    /// Simulate a chain of institutional state transitions.
    ///
    /// Each step either removes (`Remove`) or adds (`Add`) a component.
    /// Returns the trajectory: at each step, the current encoding, its nearest
    /// axiom, and the similarity to that axiom.
    ///
    /// # Example
    /// ```ignore
    /// let trajectory = algebra.query_chain(
    ///     "NATION_STATE",
    ///     &[
    ///         TransitionStep::Remove("ENFORCEMENT"),
    ///         TransitionStep::Add("LEGITIMACY"),
    ///     ],
    ///     &system,
    /// ).unwrap();
    /// // Step 0: NATION_STATE minus ENFORCEMENT → nearest FAILED_STATE
    /// // Step 1: ... plus LEGITIMACY → nearest ???
    /// ```
    pub fn query_chain(
        &self,
        start: &str,
        steps: &[TransitionStep<'_>],
        system: &PrimitiveSystem,
    ) -> Result<Vec<TransitionResult>, CompositionAlgebraError> {
        let mut current = self
            .get_encoding(start, system)
            .ok_or_else(|| CompositionAlgebraError::NotFound(start.to_string()))?;

        let mut trajectory = Vec::with_capacity(steps.len());

        for step in steps {
            let component_name = step.component();
            let component_hv = self
                .get_encoding(component_name, system)
                .ok_or_else(|| CompositionAlgebraError::NotFound(component_name.to_string()))?;

            // XOR for both add and remove (XOR is its own inverse in BinaryHV)
            current = current.bind(&component_hv);

            // Find nearest axiom
            let mut best_name = String::new();
            let mut best_sim: f32 = -1.0;

            for (name, comp) in &self.compositions {
                let sim = current.similarity(&comp.encoding);
                if sim > best_sim {
                    best_sim = sim;
                    best_name = name.clone();
                }
            }

            let action = match step {
                TransitionStep::Remove(name) => format!("-{name}"),
                TransitionStep::Add(name) => format!("+{name}"),
            };
            trajectory.push(TransitionResult {
                action,
                nearest_axiom: best_name,
                similarity: best_sim,
                encoding: current,
            });
        }

        Ok(trajectory)
    }

    /// Rank all compositions by similarity to a given encoding.
    ///
    /// Returns a sorted list of (name, similarity) pairs, most similar first.
    /// Useful for understanding what a residual "looks like" after unbinding.
    pub fn rank_by_similarity(&self, target: &BinaryHV) -> Vec<(String, f32)> {
        let mut ranked: Vec<(String, f32)> = self
            .compositions
            .iter()
            .map(|(name, comp)| (name.clone(), target.similarity(&comp.encoding)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

impl Default for CompositionAlgebra {
    fn default() -> Self {
        Self::new()
    }
}

impl CompositionAlgebra {
    /// Load pre-defined institutional causal axioms into the algebra.
    ///
    /// These encode state transitions and decomposition analyses for
    /// institutional entities. Each axiom captures a "what happens when
    /// you remove/add a component" relationship:
    ///
    /// - **REVOLUTION**: Authority without legitimacy (AUTHORITY with LEGITIMACY removed)
    /// - **FAILED_STATE**: Nation-state with enforcement collapsed
    /// - **BORDER_DISPUTE**: Overlapping sovereignty claims
    /// - **LEGITIMATE_GOVERNANCE**: Authority grounded in consent
    /// - **REGULATORY_CAPTURE**: When regulated entities capture the regulator
    /// - **TRADE_AGREEMENT**: Bilateral treaty with economic exchange
    /// - **ECONOMIC_SANCTION**: Punitive constraint on economic exchange
    /// - **CIVIL_DISOBEDIENCE**: Population rejecting compliance without enforcement collapse
    ///
    /// Returns the number of axioms successfully loaded (skips any whose
    /// parent primitives are missing from the system).
    pub fn load_institutional_axioms(&mut self, system: &PrimitiveSystem) -> usize {
        let axioms = [
            // State transitions: "what happens when you remove a component?"
            ("REVOLUTION", "AUTHORITY ^ ENFORCEMENT ^ PROHIBITION"),
            // Authority maintained through force and prohibition rather than consent.
            // This is what remains when LEGITIMACY is absent from governance.
            ("FAILED_STATE", "SOVEREIGNTY ^ POPULATION"),
            // Sovereignty claim persists but enforcement has collapsed.
            // Territory remains claimed but ungoverned.
            ("BORDER_DISPUTE", "SOVEREIGNTY ^ OVERLAPS"),
            // Two sovereignty claims with topological overlap.
            // Uses OVERLAPS (Tier 3 topology) — shared spatial parts.
            ("LEGITIMATE_GOVERNANCE", "AUTHORITY ^ LEGITIMACY ^ TRUST"),
            // Full authority stack: recognized, accepted, trusted.
            // Uses TRUST (Tier 4 social) — belief in cooperation.
            ("REGULATORY_CAPTURE", "REGULATION ^ DEFECT"),
            // Regulatory body subverted by regulated entity.
            // Uses DEFECT (Tier 4 game theory) — self-interested deviation.
            ("TRADE_AGREEMENT", "TREATY ^ EXCHANGE ^ RECIPROCATE"),
            // Bilateral trade treaty with reciprocal obligations.
            ("ECONOMIC_SANCTION", "SANCTION ^ EXCHANGE ^ PROHIBITION"),
            // Punitive restriction on economic exchange.
            // Uses PROHIBITION (morality domain) — forbidden action.
            ("CIVIL_DISOBEDIENCE", "POPULATION ^ PROHIBITION ^ COOPERATE"),
            // Collective refusal to comply without violent enforcement challenge.
            // Population + prohibition + cooperate = organized peaceful resistance.
        ];

        let mut loaded = 0;
        for (name, expr) in &axioms {
            if self.define(name, expr, system).is_ok() {
                loaded += 1;
            }
        }
        loaded
    }
}

/// A single step in a causal transition chain.
#[derive(Debug, Clone, Copy)]
pub enum TransitionStep<'a> {
    /// Remove a component (unbind via XOR)
    Remove(&'a str),
    /// Add a component (bind via XOR)
    Add(&'a str),
}

impl<'a> TransitionStep<'a> {
    /// Get the component name referenced by this step.
    pub fn component(&self) -> &'a str {
        match self {
            TransitionStep::Remove(name) | TransitionStep::Add(name) => name,
        }
    }
}

/// Result of one step in a causal transition chain.
#[derive(Debug, Clone)]
pub struct TransitionResult {
    /// Whether this was an Add or Remove, and the component name.
    pub action: String,
    /// Nearest known axiom after this step
    pub nearest_axiom: String,
    /// Similarity to the nearest axiom
    pub similarity: f32,
    /// The encoding after this step
    pub encoding: BinaryHV,
}

/// Exportable composition (without encoding)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionExport {
    pub name: String,
    pub expression: String,
}

/// Errors from composition algebra operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositionAlgebraError {
    /// Name not found (primitive or composition)
    NotFound(String),
    /// Invalid composition name
    InvalidName(String),
    /// Expression parse error
    ParseError(String),
}

impl std::fmt::Display for CompositionAlgebraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompositionAlgebraError::NotFound(name) => write!(f, "not found: {name}"),
            CompositionAlgebraError::InvalidName(msg) => write!(f, "invalid name: {msg}"),
            CompositionAlgebraError::ParseError(msg) => write!(f, "parse error: {msg}"),
        }
    }
}

impl std::error::Error for CompositionAlgebraError {}
