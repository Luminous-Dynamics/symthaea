//! # Physics Pattern Library
//!
//! Known physics patterns encoded as HDC prototypes. Used for:
//! - Recognizing known phenomena in data
//! - Cross-domain transfer (applying patterns from one field to another)
//! - Guiding symbolic regression toward physically meaningful forms

use super::encoders::PhysicalQuantity;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A known physics pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsPattern {
    /// Pattern name
    pub name: String,
    /// Mathematical form (e.g., "y = A * exp(-E/kT)")
    pub mathematical_form: String,
    /// Domain where this pattern is commonly found
    pub domain: PatternDomain,
    /// Pattern type
    pub pattern_type: PatternType,
    /// Prototype vector (HDC representation)
    #[serde(skip)]
    pub prototype: Vec<f32>,
    /// Example instances
    pub examples: Vec<String>,
    /// Physical interpretation
    pub interpretation: String,
    /// Key parameters
    pub parameters: Vec<PatternParameter>,
}

/// Domain of physics where pattern appears.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternDomain {
    /// Classical mechanics
    Mechanics,
    /// Thermodynamics
    Thermodynamics,
    /// Electromagnetism
    Electromagnetism,
    /// Quantum mechanics
    Quantum,
    /// Nuclear physics
    Nuclear,
    /// Plasma physics
    Plasma,
    /// Condensed matter
    CondensedMatter,
    /// Statistical mechanics
    Statistical,
    /// General / cross-domain
    General,
}

/// Type of physics pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternType {
    /// Exponential decay/growth
    Exponential,
    /// Power law (y ∝ x^n)
    PowerLaw,
    /// Periodic/oscillatory
    Periodic,
    /// Resonance
    Resonance,
    /// Phase transition
    PhaseTransition,
    /// Conservation law
    Conservation,
    /// Tunneling/barrier penetration
    Tunneling,
    /// Threshold behavior
    Threshold,
    /// Saturation
    Saturation,
    /// Inverse square
    InverseSquare,
}

/// Parameter in a physics pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternParameter {
    /// Parameter name
    pub name: String,
    /// Physical meaning
    pub meaning: String,
    /// Typical units
    pub units: String,
    /// Typical range (log10 of SI value)
    pub typical_log_range: (f64, f64),
}

/// Match result when comparing data to a pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    /// Pattern that matched
    pub pattern_name: String,
    /// Similarity score (0-1)
    pub similarity: f64,
    /// Fitted parameters
    pub fitted_params: HashMap<String, f64>,
    /// Residual (how well the fit explains the data)
    pub residual: f64,
    /// Suggested physical interpretation
    pub interpretation: String,
    /// Confidence in the match
    pub confidence: f64,
}

/// Library of known physics patterns.
pub struct PatternLibrary {
    patterns: Vec<PhysicsPattern>,
}

impl PatternLibrary {
    /// Create library with standard physics patterns.
    pub fn standard() -> Self {
        let mut library = Self {
            patterns: Vec::new(),
        };
        library.add_standard_patterns();
        library
    }

    /// Create empty library.
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Add a pattern to the library.
    pub fn add_pattern(&mut self, pattern: PhysicsPattern) {
        self.patterns.push(pattern);
    }

    /// Get all patterns.
    pub fn patterns(&self) -> &[PhysicsPattern] {
        &self.patterns
    }

    /// Get patterns by domain.
    pub fn patterns_for_domain(&self, domain: PatternDomain) -> Vec<&PhysicsPattern> {
        self.patterns.iter().filter(|p| p.domain == domain).collect()
    }

    /// Get patterns by type.
    pub fn patterns_of_type(&self, pattern_type: PatternType) -> Vec<&PhysicsPattern> {
        self.patterns.iter().filter(|p| p.pattern_type == pattern_type).collect()
    }

    fn add_standard_patterns(&mut self) {
        // Arrhenius / Boltzmann
        self.patterns.push(PhysicsPattern {
            name: "Arrhenius".to_string(),
            mathematical_form: "k = A * exp(-Ea / kT)".to_string(),
            domain: PatternDomain::Thermodynamics,
            pattern_type: PatternType::Exponential,
            prototype: Vec::new(),
            examples: vec![
                "Chemical reaction rates".to_string(),
                "Diffusion coefficients".to_string(),
                "Semiconductor carrier concentration".to_string(),
            ],
            interpretation: "Thermally activated process with activation energy barrier".to_string(),
            parameters: vec![
                PatternParameter {
                    name: "A".to_string(),
                    meaning: "Pre-exponential factor".to_string(),
                    units: "varies".to_string(),
                    typical_log_range: (0.0, 15.0),
                },
                PatternParameter {
                    name: "Ea".to_string(),
                    meaning: "Activation energy".to_string(),
                    units: "eV or J/mol".to_string(),
                    typical_log_range: (-2.0, 2.0),
                },
            ],
        });

        // Gamow tunneling
        self.patterns.push(PhysicsPattern {
            name: "Gamow".to_string(),
            mathematical_form: "P = exp(-2π * η) where η = Z1*Z2*e²/(ℏv)".to_string(),
            domain: PatternDomain::Nuclear,
            pattern_type: PatternType::Tunneling,
            prototype: Vec::new(),
            examples: vec![
                "Nuclear fusion cross sections".to_string(),
                "Alpha decay rates".to_string(),
                "Proton capture reactions".to_string(),
            ],
            interpretation: "Quantum tunneling through Coulomb barrier".to_string(),
            parameters: vec![
                PatternParameter {
                    name: "EG".to_string(),
                    meaning: "Gamow energy".to_string(),
                    units: "keV".to_string(),
                    typical_log_range: (1.0, 4.0),
                },
                PatternParameter {
                    name: "S".to_string(),
                    meaning: "Astrophysical S-factor".to_string(),
                    units: "keV·barn".to_string(),
                    typical_log_range: (-2.0, 4.0),
                },
            ],
        });

        // Power law
        self.patterns.push(PhysicsPattern {
            name: "PowerLaw".to_string(),
            mathematical_form: "y = A * x^n".to_string(),
            domain: PatternDomain::General,
            pattern_type: PatternType::PowerLaw,
            prototype: Vec::new(),
            examples: vec![
                "Gravitational force (n=-2)".to_string(),
                "Stefan-Boltzmann law (n=4)".to_string(),
                "Kepler's third law (n=3/2)".to_string(),
                "Critical phenomena near phase transitions".to_string(),
            ],
            interpretation: "Scale-free relationship, often indicates underlying symmetry".to_string(),
            parameters: vec![
                PatternParameter {
                    name: "A".to_string(),
                    meaning: "Coefficient".to_string(),
                    units: "varies".to_string(),
                    typical_log_range: (-10.0, 10.0),
                },
                PatternParameter {
                    name: "n".to_string(),
                    meaning: "Power exponent".to_string(),
                    units: "dimensionless".to_string(),
                    typical_log_range: (-3.0, 3.0),
                },
            ],
        });

        // Exponential decay
        self.patterns.push(PhysicsPattern {
            name: "ExponentialDecay".to_string(),
            mathematical_form: "N(t) = N0 * exp(-t/τ)".to_string(),
            domain: PatternDomain::General,
            pattern_type: PatternType::Exponential,
            prototype: Vec::new(),
            examples: vec![
                "Radioactive decay".to_string(),
                "RC circuit discharge".to_string(),
                "Atmospheric pressure with altitude".to_string(),
            ],
            interpretation: "Constant fractional rate of change".to_string(),
            parameters: vec![
                PatternParameter {
                    name: "N0".to_string(),
                    meaning: "Initial value".to_string(),
                    units: "varies".to_string(),
                    typical_log_range: (0.0, 30.0),
                },
                PatternParameter {
                    name: "τ".to_string(),
                    meaning: "Time constant".to_string(),
                    units: "s".to_string(),
                    typical_log_range: (-15.0, 18.0),
                },
            ],
        });

        // Harmonic oscillator
        self.patterns.push(PhysicsPattern {
            name: "HarmonicOscillator".to_string(),
            mathematical_form: "x(t) = A * cos(ωt + φ)".to_string(),
            domain: PatternDomain::Mechanics,
            pattern_type: PatternType::Periodic,
            prototype: Vec::new(),
            examples: vec![
                "Pendulum motion".to_string(),
                "Spring-mass system".to_string(),
                "LC circuit oscillations".to_string(),
                "Phonons in crystals".to_string(),
            ],
            interpretation: "Linear restoring force, energy oscillates between forms".to_string(),
            parameters: vec![
                PatternParameter {
                    name: "ω".to_string(),
                    meaning: "Angular frequency".to_string(),
                    units: "rad/s".to_string(),
                    typical_log_range: (-3.0, 15.0),
                },
                PatternParameter {
                    name: "A".to_string(),
                    meaning: "Amplitude".to_string(),
                    units: "varies".to_string(),
                    typical_log_range: (-15.0, 10.0),
                },
            ],
        });

        // Lorentzian resonance
        self.patterns.push(PhysicsPattern {
            name: "LorentzianResonance".to_string(),
            mathematical_form: "I(ω) = I0 * Γ² / ((ω - ω0)² + Γ²)".to_string(),
            domain: PatternDomain::General,
            pattern_type: PatternType::Resonance,
            prototype: Vec::new(),
            examples: vec![
                "Atomic spectral lines".to_string(),
                "Mechanical resonance".to_string(),
                "RLC circuit response".to_string(),
            ],
            interpretation: "Driven oscillator response near natural frequency".to_string(),
            parameters: vec![
                PatternParameter {
                    name: "ω0".to_string(),
                    meaning: "Resonance frequency".to_string(),
                    units: "rad/s".to_string(),
                    typical_log_range: (0.0, 18.0),
                },
                PatternParameter {
                    name: "Γ".to_string(),
                    meaning: "Line width (HWHM)".to_string(),
                    units: "rad/s".to_string(),
                    typical_log_range: (-3.0, 15.0),
                },
            ],
        });

        // Fermi-Dirac distribution
        self.patterns.push(PhysicsPattern {
            name: "FermiDirac".to_string(),
            mathematical_form: "f(E) = 1 / (exp((E - μ)/kT) + 1)".to_string(),
            domain: PatternDomain::Statistical,
            pattern_type: PatternType::Saturation,
            prototype: Vec::new(),
            examples: vec![
                "Electron distribution in metals".to_string(),
                "Semiconductor carrier statistics".to_string(),
                "Neutron star degeneracy".to_string(),
            ],
            interpretation: "Quantum statistics for fermions (Pauli exclusion)".to_string(),
            parameters: vec![
                PatternParameter {
                    name: "μ".to_string(),
                    meaning: "Chemical potential / Fermi energy".to_string(),
                    units: "eV".to_string(),
                    typical_log_range: (-3.0, 1.0),
                },
                PatternParameter {
                    name: "T".to_string(),
                    meaning: "Temperature".to_string(),
                    units: "K".to_string(),
                    typical_log_range: (0.0, 8.0),
                },
            ],
        });

        // Phase transition (order parameter)
        self.patterns.push(PhysicsPattern {
            name: "PhaseTransition".to_string(),
            mathematical_form: "m ∝ |T - Tc|^β for T < Tc, m = 0 for T > Tc".to_string(),
            domain: PatternDomain::CondensedMatter,
            pattern_type: PatternType::PhaseTransition,
            prototype: Vec::new(),
            examples: vec![
                "Ferromagnetic transition".to_string(),
                "Superconducting transition".to_string(),
                "Liquid-gas critical point".to_string(),
                "Bose-Einstein condensation".to_string(),
            ],
            interpretation: "Spontaneous symmetry breaking at critical temperature".to_string(),
            parameters: vec![
                PatternParameter {
                    name: "Tc".to_string(),
                    meaning: "Critical temperature".to_string(),
                    units: "K".to_string(),
                    typical_log_range: (0.0, 4.0),
                },
                PatternParameter {
                    name: "β".to_string(),
                    meaning: "Critical exponent".to_string(),
                    units: "dimensionless".to_string(),
                    typical_log_range: (-1.0, 0.0),
                },
            ],
        });

        // Inverse square law
        self.patterns.push(PhysicsPattern {
            name: "InverseSquare".to_string(),
            mathematical_form: "F = k / r²".to_string(),
            domain: PatternDomain::General,
            pattern_type: PatternType::InverseSquare,
            prototype: Vec::new(),
            examples: vec![
                "Gravitational force".to_string(),
                "Coulomb force".to_string(),
                "Light intensity from point source".to_string(),
                "Sound intensity in 3D".to_string(),
            ],
            interpretation: "Conservation of flux through sphere in 3D space".to_string(),
            parameters: vec![
                PatternParameter {
                    name: "k".to_string(),
                    meaning: "Coupling constant".to_string(),
                    units: "varies".to_string(),
                    typical_log_range: (-30.0, 30.0),
                },
            ],
        });

        // Screening (Debye-Huckel)
        self.patterns.push(PhysicsPattern {
            name: "DebyeScreening".to_string(),
            mathematical_form: "φ(r) = (q/r) * exp(-r/λD)".to_string(),
            domain: PatternDomain::Plasma,
            pattern_type: PatternType::Exponential,
            prototype: Vec::new(),
            examples: vec![
                "Plasma shielding".to_string(),
                "Electrolyte screening".to_string(),
                "Electron screening in metals".to_string(),
            ],
            interpretation: "Charge screening by mobile carriers".to_string(),
            parameters: vec![
                PatternParameter {
                    name: "λD".to_string(),
                    meaning: "Debye length".to_string(),
                    units: "m".to_string(),
                    typical_log_range: (-10.0, -4.0),
                },
            ],
        });
    }
}

impl Default for PatternLibrary {
    fn default() -> Self {
        Self::new()
    }
}

/// Pattern matching engine.
pub struct PatternMatcher {
    library: PatternLibrary,
    /// Minimum similarity to consider a match
    match_threshold: f64,
}

impl PatternMatcher {
    /// Create matcher with standard library.
    pub fn new() -> Self {
        Self {
            library: PatternLibrary::standard(),
            match_threshold: 0.5,
        }
    }

    /// Create matcher with custom library.
    pub fn with_library(library: PatternLibrary) -> Self {
        Self {
            library,
            match_threshold: 0.5,
        }
    }

    /// Set match threshold.
    pub fn set_threshold(&mut self, threshold: f64) {
        self.match_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Find patterns matching the given data.
    pub fn find_matches(
        &mut self,
        x: &[f64],
        y: &[f64],
        x_quantity: PhysicalQuantity,
        y_quantity: PhysicalQuantity,
    ) -> Vec<PatternMatch> {
        if x.len() != y.len() || x.is_empty() {
            return Vec::new();
        }

        let mut matches = Vec::new();

        // Try each pattern type
        for pattern in &self.library.patterns {
            if let Some(m) = self.try_pattern(pattern, x, y, x_quantity, y_quantity) {
                if m.similarity >= self.match_threshold {
                    matches.push(m);
                }
            }
        }

        // Sort by similarity
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }

    /// Try to match a specific pattern to data.
    fn try_pattern(
        &self,
        pattern: &PhysicsPattern,
        x: &[f64],
        y: &[f64],
        _x_quantity: PhysicalQuantity,
        _y_quantity: PhysicalQuantity,
    ) -> Option<PatternMatch> {
        let (similarity, params, residual) = match pattern.pattern_type {
            PatternType::Exponential => self.fit_exponential(x, y),
            PatternType::PowerLaw => self.fit_power_law(x, y),
            PatternType::Periodic => self.fit_periodic(x, y),
            PatternType::InverseSquare => self.fit_inverse_square(x, y),
            _ => return None, // Other types not yet implemented
        };

        Some(PatternMatch {
            pattern_name: pattern.name.clone(),
            similarity,
            fitted_params: params,
            residual,
            interpretation: pattern.interpretation.clone(),
            confidence: similarity * (1.0 - residual).max(0.0),
        })
    }

    fn fit_exponential(&self, x: &[f64], y: &[f64]) -> (f64, HashMap<String, f64>, f64) {
        // Fit y = A * exp(B * x)
        // Take log: ln(y) = ln(A) + B*x
        let n = x.len() as f64;

        let valid: Vec<_> = x.iter().zip(y.iter())
            .filter(|(_, &yi)| yi > 0.0)
            .map(|(&xi, &yi)| (xi, yi.ln()))
            .collect();

        if valid.len() < 2 {
            return (0.0, HashMap::new(), 1.0);
        }

        let n_valid = valid.len() as f64;
        let sum_x: f64 = valid.iter().map(|(xi, _)| xi).sum();
        let sum_ln_y: f64 = valid.iter().map(|(_, ln_yi)| ln_yi).sum();
        let sum_x2: f64 = valid.iter().map(|(xi, _)| xi * xi).sum();
        let sum_x_ln_y: f64 = valid.iter().map(|(xi, ln_yi)| xi * ln_yi).sum();

        let denom = n_valid * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return (0.0, HashMap::new(), 1.0);
        }

        let b = (n_valid * sum_x_ln_y - sum_x * sum_ln_y) / denom;
        let ln_a = (sum_ln_y - b * sum_x) / n_valid;
        let a = ln_a.exp();

        // Compute R²
        let y_mean: f64 = y.iter().sum::<f64>() / n;
        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = x.iter().zip(y.iter())
            .map(|(&xi, &yi)| {
                let pred = a * (b * xi).exp();
                (yi - pred).powi(2)
            })
            .sum();

        let r_squared = if ss_tot > 0.0 { (1.0 - ss_res / ss_tot).max(0.0) } else { 0.0 };

        let mut params = HashMap::new();
        params.insert("A".to_string(), a);
        params.insert("B".to_string(), b);

        (r_squared, params, 1.0 - r_squared)
    }

    fn fit_power_law(&self, x: &[f64], y: &[f64]) -> (f64, HashMap<String, f64>, f64) {
        // Fit y = A * x^n
        // Take log: ln(y) = ln(A) + n*ln(x)
        let valid: Vec<_> = x.iter().zip(y.iter())
            .filter(|(&xi, &yi)| xi > 0.0 && yi > 0.0)
            .map(|(&xi, &yi)| (xi.ln(), yi.ln()))
            .collect();

        if valid.len() < 2 {
            return (0.0, HashMap::new(), 1.0);
        }

        let n_valid = valid.len() as f64;
        let sum_ln_x: f64 = valid.iter().map(|(ln_xi, _)| ln_xi).sum();
        let sum_ln_y: f64 = valid.iter().map(|(_, ln_yi)| ln_yi).sum();
        let sum_ln_x2: f64 = valid.iter().map(|(ln_xi, _)| ln_xi * ln_xi).sum();
        let sum_ln_x_ln_y: f64 = valid.iter().map(|(ln_xi, ln_yi)| ln_xi * ln_yi).sum();

        let denom = n_valid * sum_ln_x2 - sum_ln_x * sum_ln_x;
        if denom.abs() < 1e-10 {
            return (0.0, HashMap::new(), 1.0);
        }

        let n_exp = (n_valid * sum_ln_x_ln_y - sum_ln_x * sum_ln_y) / denom;
        let ln_a = (sum_ln_y - n_exp * sum_ln_x) / n_valid;
        let a = ln_a.exp();

        // Compute R²
        let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = x.iter().zip(y.iter())
            .filter(|(&xi, _)| xi > 0.0)
            .map(|(&xi, &yi)| {
                let pred = a * xi.powf(n_exp);
                (yi - pred).powi(2)
            })
            .sum();

        let r_squared = if ss_tot > 0.0 { (1.0 - ss_res / ss_tot).max(0.0) } else { 0.0 };

        let mut params = HashMap::new();
        params.insert("A".to_string(), a);
        params.insert("n".to_string(), n_exp);

        (r_squared, params, 1.0 - r_squared)
    }

    fn fit_periodic(&self, x: &[f64], y: &[f64]) -> (f64, HashMap<String, f64>, f64) {
        // Simple periodicity detection via autocorrelation
        if y.len() < 10 {
            return (0.0, HashMap::new(), 1.0);
        }

        let n = y.len();
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
        let y_centered: Vec<f64> = y.iter().map(|yi| yi - y_mean).collect();

        // Autocorrelation at various lags
        let mut best_lag = 1;
        let mut best_autocorr = 0.0f64;

        let variance: f64 = y_centered.iter().map(|yi| yi * yi).sum();
        if variance < 1e-10 {
            return (0.0, HashMap::new(), 1.0);
        }

        for lag in 1..n/2 {
            let autocorr: f64 = y_centered.iter().take(n - lag)
                .zip(y_centered.iter().skip(lag))
                .map(|(a, b)| a * b)
                .sum::<f64>() / variance;

            if autocorr > best_autocorr {
                best_autocorr = autocorr;
                best_lag = lag;
            }
        }

        // Estimate period from lag and x spacing
        let x_span = x.last().unwrap_or(&1.0) - x.first().unwrap_or(&0.0);
        let period = x_span * best_lag as f64 / (n - 1) as f64;
        let omega = 2.0 * std::f64::consts::PI / period;

        let mut params = HashMap::new();
        params.insert("period".to_string(), period);
        params.insert("omega".to_string(), omega);

        // Use autocorrelation as similarity measure
        let similarity = best_autocorr.max(0.0);

        (similarity, params, 1.0 - similarity)
    }

    fn fit_inverse_square(&self, x: &[f64], y: &[f64]) -> (f64, HashMap<String, f64>, f64) {
        // Fit y = k / x²
        // This is a power law with n = -2
        let (similarity, mut params, residual) = self.fit_power_law(x, y);

        // Check if exponent is close to -2
        let n = params.get("n").copied().unwrap_or(0.0);
        let deviation = (n + 2.0).abs();

        // Penalize similarity if exponent far from -2
        let adjusted_similarity = similarity * (-deviation / 0.5).exp();

        params.insert("k".to_string(), params.get("A").copied().unwrap_or(1.0));

        (adjusted_similarity, params, residual)
    }

    /// Suggest patterns that might apply to a given physical situation.
    pub fn suggest_patterns(&self, quantities: &[PhysicalQuantity]) -> Vec<&PhysicsPattern> {
        let mut suggestions = Vec::new();

        // Check if quantities suggest certain pattern types
        let has_temperature = quantities.contains(&PhysicalQuantity::Temperature);
        let has_time = quantities.contains(&PhysicalQuantity::Time);
        let has_energy = quantities.contains(&PhysicalQuantity::Energy);
        let has_length = quantities.contains(&PhysicalQuantity::Length);
        let has_reaction_rate = quantities.contains(&PhysicalQuantity::ReactionRate);

        for pattern in &self.library.patterns {
            let relevant = match pattern.pattern_type {
                PatternType::Exponential => has_temperature || has_time,
                PatternType::Tunneling => has_energy || has_reaction_rate,
                PatternType::InverseSquare => has_length,
                PatternType::PhaseTransition => has_temperature,
                _ => true, // General patterns always potentially relevant
            };

            if relevant {
                suggestions.push(pattern);
            }
        }

        suggestions
    }
}

impl Default for PatternMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_creation() {
        let library = PatternLibrary::standard();
        assert!(!library.patterns().is_empty());
    }

    #[test]
    fn test_pattern_by_domain() {
        let library = PatternLibrary::standard();
        let nuclear = library.patterns_for_domain(PatternDomain::Nuclear);
        assert!(nuclear.iter().any(|p| p.name == "Gamow"));
    }

    #[test]
    fn test_pattern_by_type() {
        let library = PatternLibrary::standard();
        let exponential = library.patterns_of_type(PatternType::Exponential);
        assert!(exponential.iter().any(|p| p.name == "Arrhenius"));
    }

    #[test]
    fn test_fit_exponential() {
        let mut matcher = PatternMatcher::new();
        matcher.set_threshold(0.5);

        // y = 2 * exp(-0.1 * x)
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * (-0.1 * xi).exp()).collect();

        let matches = matcher.find_matches(&x, &y, PhysicalQuantity::Time, PhysicalQuantity::Other);

        // Should find exponential pattern
        let exp_match = matches.iter().find(|m| m.pattern_name.contains("Decay") || m.pattern_name.contains("Arrhenius"));
        assert!(exp_match.is_some());
    }

    #[test]
    fn test_fit_power_law() {
        let mut matcher = PatternMatcher::new();
        matcher.set_threshold(0.5);

        // y = 3 * x^2
        let x: Vec<f64> = (1..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi.powi(2)).collect();

        let matches = matcher.find_matches(&x, &y, PhysicalQuantity::Length, PhysicalQuantity::Other);

        // Should find power law pattern
        let power_match = matches.iter().find(|m| m.pattern_name == "PowerLaw");
        assert!(power_match.is_some());
        if let Some(m) = power_match {
            assert!(m.similarity > 0.9);
            let n = m.fitted_params.get("n").copied().unwrap_or(0.0);
            assert!((n - 2.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_suggest_patterns() {
        let matcher = PatternMatcher::new();

        let suggestions = matcher.suggest_patterns(&[
            PhysicalQuantity::Temperature,
            PhysicalQuantity::ReactionRate,
        ]);

        // Should suggest Arrhenius and Gamow
        let names: Vec<_> = suggestions.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"Arrhenius"));
        assert!(names.contains(&"Gamow"));
    }
}
