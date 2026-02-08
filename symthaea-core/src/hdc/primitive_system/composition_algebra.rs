use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::hdc::binary_hv::BinaryHV;
use super::PrimitiveSystem;

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
            return Err(CompositionAlgebraError::InvalidName("name cannot be empty".to_string()));
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
            return Err(CompositionAlgebraError::ParseError("bind requires at least 2 operands".to_string()));
        }

        let mut sources = Vec::new();
        let first_name = parts[0].trim();
        let mut result = self.get_encoding(first_name, system)
            .ok_or_else(|| CompositionAlgebraError::NotFound(first_name.to_string()))?;
        sources.push(first_name.to_string());

        for part in &parts[1..] {
            let name = part.trim();
            let enc = self.get_encoding(name, system)
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
            return Err(CompositionAlgebraError::ParseError("bundle requires at least 2 operands".to_string()));
        }

        let mut sources = Vec::new();
        let mut encodings = Vec::new();

        for part in &parts {
            let name = part.trim();
            let enc = self.get_encoding(name, system)
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
                let w: f32 = kv[1].trim().parse()
                    .map_err(|_| CompositionAlgebraError::ParseError(format!("invalid weight: {}", kv[1])))?;
                (kv[0].trim(), w)
            } else {
                (part, 1.0)
            };

            let enc = self.get_encoding(name, system)
                .ok_or_else(|| CompositionAlgebraError::NotFound(name.to_string()))?;
            encodings.push(enc);
            weights.push(weight);
            sources.push(name.to_string());
        }

        // Normalize weights
        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            return Err(CompositionAlgebraError::ParseError("weights must sum to positive value".to_string()));
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
            return Err(CompositionAlgebraError::ParseError("sequence requires at least 2 elements".to_string()));
        }

        let mut sources = Vec::new();
        let first_name = parts[0].trim();
        let mut result = self.get_encoding(first_name, system)
            .ok_or_else(|| CompositionAlgebraError::NotFound(first_name.to_string()))?;
        sources.push(first_name.to_string());

        for (i, part) in parts[1..].iter().enumerate() {
            let name = part.trim();
            let enc = self.get_encoding(name, system)
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
}

impl Default for CompositionAlgebra {
    fn default() -> Self {
        Self::new()
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

impl std::fmt::Display for CompositionAlgebraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompositionAlgebraError::NotFound(name) => write!(f, "not found: {}", name),
            CompositionAlgebraError::InvalidName(msg) => write!(f, "invalid name: {}", msg),
            CompositionAlgebraError::ParseError(msg) => write!(f, "parse error: {}", msg),
        }
    }
}

impl std::error::Error for CompositionAlgebraError {}
