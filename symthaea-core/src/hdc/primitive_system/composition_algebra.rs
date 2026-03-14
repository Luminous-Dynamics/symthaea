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

    /// Simulate a chain of institutional state transitions from a starting point.
    ///
    /// Each `TransitionStep` either removes a component (XOR unbinding) or adds one
    /// (majority bundle). After each step, finds the nearest named axiom and records
    /// the result as a `TransitionResult`.
    ///
    /// Returns the full trajectory — one `TransitionResult` per step.
    pub fn query_chain(
        &self,
        start: &str,
        steps: &[TransitionStep<'_>],
        system: &PrimitiveSystem,
    ) -> Result<Vec<TransitionResult>, CompositionAlgebraError> {
        let mut current = self
            .get_encoding(start, system)
            .ok_or_else(|| CompositionAlgebraError::NotFound(start.to_string()))?;

        let mut results = Vec::with_capacity(steps.len());

        for step in steps {
            match step {
                TransitionStep::Remove(component) => {
                    let comp_hv = self
                        .get_encoding(component, system)
                        .ok_or_else(|| CompositionAlgebraError::NotFound(component.to_string()))?;
                    // XOR is self-inverse: A ^ B ^ B = A
                    current = current.bind(&comp_hv);
                }
                TransitionStep::Add(component) => {
                    let comp_hv = self
                        .get_encoding(component, system)
                        .ok_or_else(|| CompositionAlgebraError::NotFound(component.to_string()))?;
                    // bind_temporal makes Add distinguishable from Remove
                    current = current.bind_temporal(&comp_hv);
                }
            }

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

            results.push(TransitionResult {
                nearest: best_name,
                similarity: best_sim,
            });
        }

        Ok(results)
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
    /// Uses majority-vote **bundling** (`+`) so that composites remain
    /// similar to their source components — enabling decomposition by
    /// re-bundling the remaining sources after removal.
    ///
    /// Returns the number of axioms successfully loaded (skips any whose
    /// parent primitives are missing from the system).
    pub fn load_institutional_axioms(&mut self, system: &PrimitiveSystem) -> usize {
        let axioms = [
            ("REVOLUTION", "AUTHORITY + ENFORCEMENT + PROHIBITION"),
            ("FAILED_STATE", "SOVEREIGNTY + POPULATION"),
            ("BORDER_DISPUTE", "SOVEREIGNTY + OVERLAPS"),
            ("LEGITIMATE_GOVERNANCE", "AUTHORITY + LEGITIMACY + TRUST"),
            ("REGULATORY_CAPTURE", "REGULATION + DEFECT"),
            ("TRADE_AGREEMENT", "TREATY + EXCHANGE + RECIPROCATE"),
            ("ECONOMIC_SANCTION", "SANCTION + EXCHANGE + PROHIBITION"),
            ("CIVIL_DISOBEDIENCE", "POPULATION + PROHIBITION + COOPERATE"),
            (
                "DEMOCRATIC_ELECTION",
                "AUTHORITY + LEGITIMACY + POPULATION + COOPERATE",
            ),
            ("ARMS_EMBARGO", "SANCTION + ENFORCEMENT + PROHIBITION"),
            (
                "SOCIAL_CONTRACT",
                "SOVEREIGNTY + LEGITIMACY + OBLIGATION + COOPERATE",
            ),
            ("CORRUPTION", "AUTHORITY + DEFECT + EXCHANGE"),
        ];

        let mut loaded = 0;
        for (name, expr) in &axioms {
            if self.define(name, expr, system).is_ok() {
                loaded += 1;
            }
        }
        loaded
    }

    /// Load institutional axioms with causal-salience weights.
    ///
    /// Higher weight = more salient in the composite's identity.
    pub fn load_institutional_axioms_weighted(&mut self, system: &PrimitiveSystem) -> usize {
        let axioms = [
            ("REVOLUTION", "AUTHORITY:3 + ENFORCEMENT:3 + PROHIBITION:1"),
            ("FAILED_STATE", "SOVEREIGNTY:2 + POPULATION:1"),
            ("BORDER_DISPUTE", "SOVEREIGNTY:3 + OVERLAPS:2"),
            (
                "LEGITIMATE_GOVERNANCE",
                "AUTHORITY:2 + LEGITIMACY:3 + TRUST:2",
            ),
            ("REGULATORY_CAPTURE", "REGULATION:2 + DEFECT:3"),
            ("TRADE_AGREEMENT", "TREATY:2 + EXCHANGE:3 + RECIPROCATE:2"),
            (
                "ECONOMIC_SANCTION",
                "SANCTION:3 + EXCHANGE:2 + PROHIBITION:2",
            ),
            (
                "CIVIL_DISOBEDIENCE",
                "POPULATION:3 + PROHIBITION:2 + COOPERATE:2",
            ),
            (
                "DEMOCRATIC_ELECTION",
                "AUTHORITY:2 + LEGITIMACY:3 + POPULATION:2 + COOPERATE:1",
            ),
            ("ARMS_EMBARGO", "SANCTION:3 + ENFORCEMENT:2 + PROHIBITION:2"),
            (
                "SOCIAL_CONTRACT",
                "SOVEREIGNTY:2 + LEGITIMACY:3 + OBLIGATION:2 + COOPERATE:1",
            ),
            ("CORRUPTION", "AUTHORITY:2 + DEFECT:3 + EXCHANGE:2"),
        ];

        let mut loaded = 0;
        for (name, expr) in &axioms {
            if self.define(name, expr, system).is_ok() {
                loaded += 1;
            }
        }
        loaded
    }

    /// Decompose a bundled composite by removing a component.
    ///
    /// Unlike XOR unbinding, this re-bundles the remaining source
    /// components (all sources except `removed`), then finds the nearest
    /// axiom to the result.
    pub fn query_decomposition(
        &self,
        composite: &str,
        removed: &str,
        system: &PrimitiveSystem,
    ) -> Result<(String, f32, BinaryHV), CompositionAlgebraError> {
        let comp = self
            .compositions
            .get(composite)
            .ok_or_else(|| CompositionAlgebraError::NotFound(composite.to_string()))?;

        // Collect remaining source encodings (exclude `removed`)
        let remaining: Vec<BinaryHV> = comp
            .sources
            .iter()
            .filter(|s| *s != removed)
            .filter_map(|s| self.get_encoding(s, system))
            .collect();

        if remaining.is_empty() {
            return Err(CompositionAlgebraError::ParseError(
                "no sources remain after removal".to_string(),
            ));
        }

        let residual = if remaining.len() == 1 {
            remaining[0]
        } else {
            BinaryHV::bundle(&remaining)
        };

        // Find nearest axiom (excluding the composite itself)
        let mut best_name = String::new();
        let mut best_sim: f32 = -1.0;

        for (name, c) in &self.compositions {
            if name == composite {
                continue;
            }
            let sim = residual.similarity(&c.encoding);
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

    /// Solve an analogy: A is to B as ??? is to D.
    ///
    /// Computes the structural transformation from A→B as the set-difference
    /// of their source components. Components in A but not B are "removed";
    /// components in B but not A are "added" (with permutation for
    /// directionality). This transformation is applied to D's sources,
    /// and the nearest axiom to the result is returned.
    pub fn query_analogy(
        &self,
        a: &str,
        b: &str,
        d: &str,
        system: &PrimitiveSystem,
    ) -> Result<(String, f32, BinaryHV), CompositionAlgebraError> {
        let comp_a = self
            .compositions
            .get(a)
            .ok_or_else(|| CompositionAlgebraError::NotFound(a.to_string()))?;
        let comp_b = self
            .compositions
            .get(b)
            .ok_or_else(|| CompositionAlgebraError::NotFound(b.to_string()))?;
        let comp_d = self
            .compositions
            .get(d)
            .ok_or_else(|| CompositionAlgebraError::NotFound(d.to_string()))?;

        // Compute transformation: removed = in A not B, added = in B not A
        let removed: Vec<&String> = comp_a
            .sources
            .iter()
            .filter(|s| !comp_b.sources.contains(s))
            .collect();
        let added: Vec<&String> = comp_b
            .sources
            .iter()
            .filter(|s| !comp_a.sources.contains(s))
            .collect();

        // Apply transformation to D's sources
        let mut result_sources: Vec<BinaryHV> = comp_d
            .sources
            .iter()
            .filter(|s| !removed.iter().any(|r| r == s))
            .filter_map(|s| self.get_encoding(s, system))
            .collect();

        // Add new components with permutation for directionality
        for (i, added_name) in added.iter().enumerate() {
            if let Some(hv) = self.get_encoding(added_name, system) {
                result_sources.push(hv.bind_temporal(&comp_d.encoding));
            }
        }

        if result_sources.is_empty() {
            return Err(CompositionAlgebraError::ParseError(
                "analogy produced empty result".to_string(),
            ));
        }

        let result_hv = if result_sources.len() == 1 {
            result_sources[0]
        } else {
            BinaryHV::bundle(&result_sources)
        };

        // Find nearest axiom
        let mut best_name = String::new();
        let mut best_sim: f32 = -1.0;

        for (name, c) in &self.compositions {
            let sim = result_hv.similarity(&c.encoding);
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

        Ok((best_name, best_sim, result_hv))
    }
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

/// A single step in an institutional state transition chain.
///
/// Each step either removes a component from the current state
/// (simulating loss of institutional capacity) or adds one back
/// (simulating reconstruction or reform).
#[derive(Debug, Clone)]
pub enum TransitionStep<'a> {
    /// Remove the named component from the current state via XOR unbinding.
    Remove(&'a str),
    /// Add (bundle) the named component into the current state.
    Add(&'a str),
}

/// Result of a single transition step: the nearest named axiom after the step.
#[derive(Debug, Clone)]
pub struct TransitionResult {
    /// Name of the nearest axiom/composition after this step.
    pub nearest: String,
    /// Similarity score to the nearest axiom [0.0, 1.0].
    pub similarity: f32,
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
