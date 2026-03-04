//! Pre-encoded catalog of landmark physics equations.
//!
//! ~30 equations from electromagnetism, gravity, quantum mechanics,
//! field theory, fluids, cosmology, and the Spark Engine. Each entry
//! stores 4 pre-computed ContinuousHVs:
//!
//! 1. **full** — complete equation encoding (with names)
//! 2. **skeleton** — structural encoding (names stripped)
//! 3. **symmetry** — symmetry group encoding
//! 4. **dimensional** — SI dimensional encoding

use crate::dimensional::DimensionalEncoder;
use crate::equation_ast::{
    make_const, make_diffop, make_equals, make_field, make_product, make_sum, EquationEncoder,
};
use crate::symmetry::SymmetryEncoder;
use crate::types::*;
use symthaea_core::hdc::ContinuousHV;

/// A single catalog entry with pre-computed HVs.
pub struct CatalogEntry {
    /// Equation metadata.
    pub equation: PhysicsEquation,
    /// Full equation HV (includes names).
    pub full_hv: ContinuousHV,
    /// Skeleton HV (structure only, names stripped).
    pub skeleton_hv: ContinuousHV,
    /// Symmetry group HV.
    pub symmetry_hv: ContinuousHV,
    /// SI dimensional HV.
    pub dimensional_hv: ContinuousHV,
}

/// The complete physics equation catalog.
pub struct PhysicsCatalog {
    entries: Vec<CatalogEntry>,
}

impl PhysicsCatalog {
    /// Build the catalog, encoding all landmark equations.
    pub fn new() -> Self {
        let eq_encoder = EquationEncoder::new();
        let sym_encoder = SymmetryEncoder::new();
        let dim_encoder = DimensionalEncoder::new();

        let equations = build_all_equations();

        let entries = equations
            .into_iter()
            .map(|eq| {
                let full_hv = eq_encoder.encode(&eq.ast);
                let skeleton_hv = eq_encoder.encode_skeleton(&eq.ast);
                let symmetry_hv = sym_encoder.encode(&eq.symmetries);
                let dimensional_hv = dim_encoder.encode(&eq.dimensions);
                CatalogEntry {
                    equation: eq,
                    full_hv,
                    skeleton_hv,
                    symmetry_hv,
                    dimensional_hv,
                }
            })
            .collect();

        Self { entries }
    }

    /// Get all catalog entries.
    pub fn entries(&self) -> &[CatalogEntry] {
        &self.entries
    }

    /// Number of equations in the catalog.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find an entry by name (exact match).
    pub fn find_by_name(&self, name: &str) -> Option<&CatalogEntry> {
        self.entries.iter().find(|e| e.equation.name == name)
    }

    /// Get all entries in a given domain.
    pub fn entries_in_domain(&self, domain: PhysicsDomain) -> Vec<&CatalogEntry> {
        self.entries
            .iter()
            .filter(|e| e.equation.domain == domain)
            .collect()
    }
}

impl Default for PhysicsCatalog {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EQUATION BUILDERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Build all landmark equations for the catalog.
fn build_all_equations() -> Vec<PhysicsEquation> {
    let mut eqs = Vec::with_capacity(32);

    // Electromagnetism
    eqs.push(maxwell_gauss_law());
    eqs.push(maxwell_gauss_magnetism());
    eqs.push(maxwell_faraday());
    eqs.push(maxwell_ampere());

    // General Relativity
    eqs.push(einstein_field_equations());
    eqs.push(einstein_field_with_lambda());
    eqs.push(schwarzschild_metric());
    eqs.push(kerr_metric());
    eqs.push(flrw_metric());
    eqs.push(reissner_nordstrom_metric());
    eqs.push(de_sitter_metric());
    eqs.push(alcubierre_metric());

    // Quantum Mechanics
    eqs.push(schrodinger_equation());
    eqs.push(dirac_equation());
    eqs.push(klein_gordon_equation());

    // Field Theory
    eqs.push(yang_mills_equation());
    eqs.push(euler_lagrange_equation());
    eqs.push(hamilton_equations());

    // Fluids
    eqs.push(navier_stokes_equation());
    eqs.push(wave_equation());
    eqs.push(heat_equation());

    // Cosmology
    eqs.push(friedmann_first());
    eqs.push(friedmann_second());

    // Spark Engine
    eqs.push(gamow_peak_integral());
    eqs.push(coulomb_screening());
    eqs.push(dd_branching_ratio());
    eqs.push(thermal_gamow_coupling());

    eqs
}

// ── Electromagnetism ─────────────────────────────────────────────────────────

/// ∇·E = ρ/ε₀ (Gauss's law for electricity)
fn maxwell_gauss_law() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    PhysicsEquation {
        name: "Maxwell Gauss Law".to_string(),
        domain: PhysicsDomain::Electromagnetism,
        ast: make_equals(
            make_diffop(
                DiffOperator::Divergence,
                make_field("E", TensorDescriptor::vector(lor4)),
            ),
            make_product(vec![
                make_field("ρ", TensorDescriptor::scalar(lor4)),
                EquationNode::Power {
                    base: Box::new(make_const("ε₀")),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::U(1)],
            vec![DiscreteSymmetry::C, DiscreteSymmetry::P, DiscreteSymmetry::T],
            true,
        ),
        dimensions: DimensionalSignature::ELECTRIC_FIELD,
        tensor: Some(TensorDescriptor::vector(lor4)),
    }
}

/// ∇·B = 0 (Gauss's law for magnetism — no magnetic monopoles)
fn maxwell_gauss_magnetism() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    PhysicsEquation {
        name: "Maxwell Gauss Magnetism".to_string(),
        domain: PhysicsDomain::Electromagnetism,
        ast: make_equals(
            make_diffop(
                DiffOperator::Divergence,
                make_field("B", TensorDescriptor::vector(lor4)),
            ),
            EquationNode::Scalar(0.0),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::U(1)],
            vec![DiscreteSymmetry::C, DiscreteSymmetry::P, DiscreteSymmetry::T],
            true,
        ),
        dimensions: DimensionalSignature::MAGNETIC_FIELD,
        tensor: Some(TensorDescriptor::vector(lor4)),
    }
}

/// ∇×E = -∂B/∂t (Faraday's law)
fn maxwell_faraday() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    PhysicsEquation {
        name: "Maxwell Faraday Law".to_string(),
        domain: PhysicsDomain::Electromagnetism,
        ast: make_equals(
            make_diffop(
                DiffOperator::Curl,
                make_field("E", TensorDescriptor::vector(lor4)),
            ),
            EquationNode::Negate(Box::new(make_diffop(
                DiffOperator::TimeDerivative,
                make_field("B", TensorDescriptor::vector(lor4)),
            ))),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::U(1)],
            vec![DiscreteSymmetry::C, DiscreteSymmetry::P, DiscreteSymmetry::T],
            true,
        ),
        dimensions: DimensionalSignature::ELECTRIC_FIELD,
        tensor: Some(TensorDescriptor::vector(lor4)),
    }
}

/// ∇×B = μ₀J + μ₀ε₀ ∂E/∂t (Ampère-Maxwell law)
fn maxwell_ampere() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    PhysicsEquation {
        name: "Maxwell Ampere Law".to_string(),
        domain: PhysicsDomain::Electromagnetism,
        ast: make_equals(
            make_diffop(
                DiffOperator::Curl,
                make_field("B", TensorDescriptor::vector(lor4)),
            ),
            make_sum(vec![
                make_product(vec![
                    make_const("μ₀"),
                    make_field("J", TensorDescriptor::vector(lor4)),
                ]),
                make_product(vec![
                    make_const("μ₀"),
                    make_const("ε₀"),
                    make_diffop(
                        DiffOperator::TimeDerivative,
                        make_field("E", TensorDescriptor::vector(lor4)),
                    ),
                ]),
            ]),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::U(1)],
            vec![DiscreteSymmetry::C, DiscreteSymmetry::P, DiscreteSymmetry::T],
            true,
        ),
        dimensions: DimensionalSignature::MAGNETIC_FIELD,
        tensor: Some(TensorDescriptor::vector(lor4)),
    }
}

// ── General Relativity ───────────────────────────────────────────────────────

/// G_μν = 8πG/c⁴ T_μν (Einstein field equations)
fn einstein_field_equations() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    PhysicsEquation {
        name: "Einstein Field Equations".to_string(),
        domain: PhysicsDomain::GeneralRelativity,
        ast: make_equals(
            make_field("G", TensorDescriptor::symmetric_2(lor4)),
            make_product(vec![
                make_const("8πG/c⁴"),
                make_field("T", TensorDescriptor::symmetric_2(lor4)),
            ]),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::GL(4)],
            vec![],
            false, // Diffeomorphism invariance, not a gauge theory in the Yang-Mills sense
        ),
        dimensions: DimensionalSignature::INVERSE_LENGTH,
        tensor: Some(TensorDescriptor::symmetric_2(lor4)),
    }
}

/// G_μν + Λg_μν = 8πG/c⁴ T_μν (with cosmological constant)
fn einstein_field_with_lambda() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    PhysicsEquation {
        name: "Einstein Field Equations with Λ".to_string(),
        domain: PhysicsDomain::GeneralRelativity,
        ast: make_equals(
            make_sum(vec![
                make_field("G", TensorDescriptor::symmetric_2(lor4)),
                make_product(vec![
                    make_const("Λ"),
                    make_field("g", TensorDescriptor::symmetric_2(lor4)),
                ]),
            ]),
            make_product(vec![
                make_const("8πG/c⁴"),
                make_field("T", TensorDescriptor::symmetric_2(lor4)),
            ]),
        ),
        symmetries: SymmetryDescriptor::new(vec![LieGroup::GL(4)], vec![], false),
        dimensions: DimensionalSignature::INVERSE_LENGTH,
        tensor: Some(TensorDescriptor::symmetric_2(lor4)),
    }
}

/// Schwarzschild metric: vacuum, spherically symmetric, stationary
fn schwarzschild_metric() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    let mut tensor = TensorDescriptor::symmetric_2(lor4);
    tensor.solution = Some(MetricSolution {
        singularity: true,
        horizon: true,
        vacuum: true,
        stationary: true,
        axisymmetric: true,
        spherically_symmetric: true,
        cosmological: false,
        has_cosmological_constant: false,
        has_charge: false,
    });
    PhysicsEquation {
        name: "Schwarzschild Metric".to_string(),
        domain: PhysicsDomain::GeneralRelativity,
        ast: make_equals(
            make_field("g", tensor.clone()),
            make_product(vec![
                make_const("ds²"),
                make_sum(vec![
                    make_product(vec![
                        EquationNode::Negate(Box::new(make_sum(vec![
                            EquationNode::Scalar(1.0),
                            EquationNode::Negate(Box::new(make_product(vec![
                                make_const("2GM/c²"),
                                EquationNode::Power {
                                    base: Box::new(make_const("r")),
                                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                                },
                            ]))),
                        ]))),
                        make_const("dt²"),
                    ]),
                    make_product(vec![
                        EquationNode::Power {
                            base: Box::new(make_sum(vec![
                                EquationNode::Scalar(1.0),
                                EquationNode::Negate(Box::new(make_product(vec![
                                    make_const("2GM/c²"),
                                    EquationNode::Power {
                                        base: Box::new(make_const("r")),
                                        exponent: Box::new(EquationNode::Scalar(-1.0)),
                                    },
                                ]))),
                            ])),
                            exponent: Box::new(EquationNode::Scalar(-1.0)),
                        },
                        make_const("dr²"),
                    ]),
                    make_product(vec![
                        EquationNode::Power {
                            base: Box::new(make_const("r")),
                            exponent: Box::new(EquationNode::Scalar(2.0)),
                        },
                        make_const("dΩ²"),
                    ]),
                ]),
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature::AREA,
        tensor: Some(tensor),
    }
}

/// Kerr metric: rotating black hole (axisymmetric, not spherically symmetric)
fn kerr_metric() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    let mut tensor = TensorDescriptor::symmetric_2(lor4);
    tensor.solution = Some(MetricSolution {
        singularity: true,
        horizon: true,
        vacuum: true,
        stationary: true,
        axisymmetric: true,
        spherically_symmetric: false,
        cosmological: false,
        has_cosmological_constant: false,
        has_charge: false,
    });
    // Simplified AST — captures the structural essence
    PhysicsEquation {
        name: "Kerr Metric".to_string(),
        domain: PhysicsDomain::GeneralRelativity,
        ast: make_equals(
            make_field("g", tensor.clone()),
            make_product(vec![
                make_const("ds²"),
                make_sum(vec![
                    make_product(vec![
                        EquationNode::Negate(Box::new(make_sum(vec![
                            EquationNode::Scalar(1.0),
                            EquationNode::Negate(Box::new(make_product(vec![
                                make_const("2GMr"),
                                EquationNode::Power {
                                    base: Box::new(make_const("Σ")),
                                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                                },
                            ]))),
                        ]))),
                        make_const("dt²"),
                    ]),
                    make_product(vec![
                        make_const("Σ/Δ"),
                        make_const("dr²"),
                    ]),
                    make_product(vec![make_const("Σ"), make_const("dθ²")]),
                    make_product(vec![
                        make_const("cross_term"),
                        make_const("dtdφ"),
                    ]),
                ]),
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(
            vec![LieGroup::U(1)], // Axial symmetry only
            false,
        ),
        dimensions: DimensionalSignature::AREA,
        tensor: Some(tensor),
    }
}

/// FLRW metric: homogeneous + isotropic cosmological solution
fn flrw_metric() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    let mut tensor = TensorDescriptor::symmetric_2(lor4);
    tensor.solution = Some(MetricSolution {
        singularity: false,
        horizon: false,
        vacuum: false,
        stationary: false,
        axisymmetric: false,
        spherically_symmetric: false,
        cosmological: true,
        has_cosmological_constant: false,
        has_charge: false,
    });
    PhysicsEquation {
        name: "FLRW Metric".to_string(),
        domain: PhysicsDomain::Cosmology,
        ast: make_equals(
            make_field("g", tensor.clone()),
            make_product(vec![
                make_const("ds²"),
                make_sum(vec![
                    EquationNode::Negate(Box::new(make_const("dt²"))),
                    make_product(vec![
                        EquationNode::Power {
                            base: Box::new(make_const("a(t)")),
                            exponent: Box::new(EquationNode::Scalar(2.0)),
                        },
                        make_const("dΣ²"),
                    ]),
                ]),
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature::AREA,
        tensor: Some(tensor),
    }
}

/// Reissner-Nordström metric: charged, non-rotating black hole
fn reissner_nordstrom_metric() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    let mut tensor = TensorDescriptor::symmetric_2(lor4);
    tensor.solution = Some(MetricSolution {
        singularity: true,
        horizon: true,
        vacuum: false, // Electromagnetic field present
        stationary: true,
        axisymmetric: true,
        spherically_symmetric: true,
        cosmological: false,
        has_cosmological_constant: false,
        has_charge: true,
    });
    PhysicsEquation {
        name: "Reissner-Nordström Metric".to_string(),
        domain: PhysicsDomain::GeneralRelativity,
        ast: make_equals(
            make_field("g", tensor.clone()),
            make_product(vec![
                make_const("ds²"),
                make_sum(vec![
                    make_product(vec![
                        EquationNode::Negate(Box::new(make_sum(vec![
                            EquationNode::Scalar(1.0),
                            EquationNode::Negate(Box::new(make_product(vec![
                                make_const("2GM/c²r"),
                            ]))),
                            make_product(vec![
                                make_const("GQ²/(4πε₀c⁴r²)"),
                            ]),
                        ]))),
                        make_const("dt²"),
                    ]),
                    make_const("spatial_terms"),
                ]),
            ]),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::SO(3), LieGroup::U(1)],
            vec![],
            true, // U(1) gauge symmetry for EM
        ),
        dimensions: DimensionalSignature::AREA,
        tensor: Some(tensor),
    }
}

/// de Sitter metric: empty universe with positive cosmological constant
fn de_sitter_metric() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    let mut tensor = TensorDescriptor::symmetric_2(lor4);
    tensor.solution = Some(MetricSolution {
        singularity: false,
        horizon: true,
        vacuum: true,
        stationary: true,
        axisymmetric: true,
        spherically_symmetric: true,
        cosmological: true,
        has_cosmological_constant: true,
        has_charge: false,
    });
    PhysicsEquation {
        name: "de Sitter Metric".to_string(),
        domain: PhysicsDomain::Cosmology,
        ast: make_equals(
            make_field("g", tensor.clone()),
            make_product(vec![
                make_const("ds²"),
                make_sum(vec![
                    make_product(vec![
                        EquationNode::Negate(Box::new(make_sum(vec![
                            EquationNode::Scalar(1.0),
                            EquationNode::Negate(Box::new(make_product(vec![
                                make_const("Λr²/3"),
                            ]))),
                        ]))),
                        make_const("dt²"),
                    ]),
                    make_const("spatial_terms"),
                ]),
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature::AREA,
        tensor: Some(tensor),
    }
}

/// Alcubierre warp drive metric
fn alcubierre_metric() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    let mut tensor = TensorDescriptor::symmetric_2(lor4);
    tensor.solution = Some(MetricSolution {
        singularity: false,
        horizon: false,
        vacuum: false,
        stationary: false,
        axisymmetric: true,
        spherically_symmetric: false,
        cosmological: false,
        has_cosmological_constant: false,
        has_charge: false,
    });
    PhysicsEquation {
        name: "Alcubierre Metric".to_string(),
        domain: PhysicsDomain::GeneralRelativity,
        ast: make_equals(
            make_field("g", tensor.clone()),
            make_product(vec![
                make_const("ds²"),
                make_sum(vec![
                    EquationNode::Negate(Box::new(make_const("dt²"))),
                    make_product(vec![
                        make_const("f(r_s)"),
                        make_const("v_s"),
                        make_const("dxdt"),
                    ]),
                    make_const("spatial_flat"),
                ]),
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::AREA,
        tensor: Some(tensor),
    }
}

// ── Quantum Mechanics ────────────────────────────────────────────────────────

/// iℏ ∂ψ/∂t = Ĥψ (time-dependent Schrödinger equation)
fn schrodinger_equation() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "Schrödinger Equation".to_string(),
        domain: PhysicsDomain::QuantumMechanics,
        ast: make_equals(
            make_product(vec![
                make_const("iℏ"),
                make_diffop(
                    DiffOperator::TimeDerivative,
                    make_field("ψ", TensorDescriptor::scalar(euc3)),
                ),
            ]),
            make_product(vec![
                make_const("Ĥ"),
                make_field("ψ", TensorDescriptor::scalar(euc3)),
            ]),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::U(1)],
            vec![DiscreteSymmetry::T],
            true,
        ),
        dimensions: DimensionalSignature::ENERGY,
        tensor: Some(TensorDescriptor::scalar(euc3)),
    }
}

/// (iγ^μ∂_μ - m)ψ = 0 (Dirac equation)
fn dirac_equation() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    // Dirac spinor is a 4-component object, approximated as vector for structure
    PhysicsEquation {
        name: "Dirac Equation".to_string(),
        domain: PhysicsDomain::QuantumFieldTheory,
        ast: make_equals(
            make_product(vec![
                make_sum(vec![
                    make_product(vec![
                        make_const("iγ^μ"),
                        make_diffop(
                            DiffOperator::Partial,
                            make_field("ψ", TensorDescriptor::vector(lor4)),
                        ),
                    ]),
                    EquationNode::Negate(Box::new(make_product(vec![
                        make_const("m"),
                        make_field("ψ", TensorDescriptor::vector(lor4)),
                    ]))),
                ]),
            ]),
            EquationNode::Scalar(0.0),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::Poincare, LieGroup::U(1)],
            vec![DiscreteSymmetry::C, DiscreteSymmetry::P, DiscreteSymmetry::T],
            true,
        ),
        dimensions: DimensionalSignature::ENERGY,
        tensor: Some(TensorDescriptor::vector(lor4)),
    }
}

/// (□ + m²)φ = 0 (Klein-Gordon equation)
fn klein_gordon_equation() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    PhysicsEquation {
        name: "Klein-Gordon Equation".to_string(),
        domain: PhysicsDomain::QuantumFieldTheory,
        ast: make_equals(
            make_sum(vec![
                make_diffop(
                    DiffOperator::DAlembertian,
                    make_field("φ", TensorDescriptor::scalar(lor4)),
                ),
                make_product(vec![
                    EquationNode::Power {
                        base: Box::new(make_const("m")),
                        exponent: Box::new(EquationNode::Scalar(2.0)),
                    },
                    make_field("φ", TensorDescriptor::scalar(lor4)),
                ]),
            ]),
            EquationNode::Scalar(0.0),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::Poincare],
            vec![DiscreteSymmetry::C, DiscreteSymmetry::P, DiscreteSymmetry::T],
            false,
        ),
        dimensions: DimensionalSignature::ENERGY,
        tensor: Some(TensorDescriptor::scalar(lor4)),
    }
}

// ── Field Theory ─────────────────────────────────────────────────────────────

/// D_μ F^μν = J^ν (Yang-Mills equations)
fn yang_mills_equation() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    PhysicsEquation {
        name: "Yang-Mills Equation".to_string(),
        domain: PhysicsDomain::QuantumFieldTheory,
        ast: make_equals(
            make_diffop(
                DiffOperator::CovariantDerivative,
                make_field("F", TensorDescriptor::antisymmetric_2(lor4)),
            ),
            make_field("J", TensorDescriptor::vector(lor4)),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(
            vec![LieGroup::SU(3)], // Non-abelian gauge
            true,
        ),
        dimensions: DimensionalSignature::ENERGY_DENSITY,
        tensor: Some(TensorDescriptor::antisymmetric_2(lor4)),
    }
}

/// ∂L/∂φ - d/dt(∂L/∂φ̇) = 0 (Euler-Lagrange)
fn euler_lagrange_equation() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "Euler-Lagrange Equation".to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast: make_equals(
            make_sum(vec![
                make_diffop(
                    DiffOperator::Partial,
                    make_field("L", TensorDescriptor::scalar(euc3)),
                ),
                EquationNode::Negate(Box::new(make_diffop(
                    DiffOperator::TimeDerivative,
                    make_diffop(
                        DiffOperator::Partial,
                        make_field("L", TensorDescriptor::scalar(euc3)),
                    ),
                ))),
            ]),
            EquationNode::Scalar(0.0),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::GL(3)],
            vec![DiscreteSymmetry::T],
            false,
        ),
        dimensions: DimensionalSignature::FORCE,
        tensor: None,
    }
}

/// q̇ = ∂H/∂p, ṗ = -∂H/∂q (Hamilton's equations)
fn hamilton_equations() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "Hamilton Equations".to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast: make_equals(
            make_diffop(
                DiffOperator::TimeDerivative,
                make_field("q", TensorDescriptor::vector(euc3)),
            ),
            make_diffop(
                DiffOperator::Partial,
                make_field("H", TensorDescriptor::scalar(euc3)),
            ),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::Sp(3)], // Symplectic structure
            vec![DiscreteSymmetry::T],
            false,
        ),
        dimensions: DimensionalSignature::VELOCITY,
        tensor: None,
    }
}

// ── Fluids ───────────────────────────────────────────────────────────────────

/// ρ(∂v/∂t + (v·∇)v) = -∇p + μ∇²v + f (Navier-Stokes)
fn navier_stokes_equation() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "Navier-Stokes Equation".to_string(),
        domain: PhysicsDomain::FluidDynamics,
        ast: make_equals(
            make_product(vec![
                make_const("ρ"),
                make_sum(vec![
                    make_diffop(
                        DiffOperator::TimeDerivative,
                        make_field("v", TensorDescriptor::vector(euc3)),
                    ),
                    make_product(vec![
                        make_field("v", TensorDescriptor::vector(euc3)),
                        make_diffop(
                            DiffOperator::Gradient,
                            make_field("v", TensorDescriptor::vector(euc3)),
                        ),
                    ]),
                ]),
            ]),
            make_sum(vec![
                EquationNode::Negate(Box::new(make_diffop(
                    DiffOperator::Gradient,
                    make_field("p", TensorDescriptor::scalar(euc3)),
                ))),
                make_product(vec![
                    make_const("μ"),
                    make_diffop(
                        DiffOperator::Laplacian,
                        make_field("v", TensorDescriptor::vector(euc3)),
                    ),
                ]),
                make_field("f", TensorDescriptor::vector(euc3)),
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature::FORCE,
        tensor: Some(TensorDescriptor::vector(euc3)),
    }
}

/// □ψ = 0 (wave equation)
fn wave_equation() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    PhysicsEquation {
        name: "Wave Equation".to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast: make_equals(
            make_diffop(
                DiffOperator::DAlembertian,
                make_field("ψ", TensorDescriptor::scalar(lor4)),
            ),
            EquationNode::Scalar(0.0),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::Poincare], false),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: Some(TensorDescriptor::scalar(lor4)),
    }
}

/// ∂T/∂t = κ∇²T (heat/diffusion equation)
fn heat_equation() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "Heat Equation".to_string(),
        domain: PhysicsDomain::Thermodynamics,
        ast: make_equals(
            make_diffop(
                DiffOperator::TimeDerivative,
                make_field("T", TensorDescriptor::scalar(euc3)),
            ),
            make_product(vec![
                make_const("κ"),
                make_diffop(
                    DiffOperator::Laplacian,
                    make_field("T", TensorDescriptor::scalar(euc3)),
                ),
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature {
            mass: 0,
            length: 0,
            time: -1,
            current: 0,
            temperature: 1,
            amount: 0,
            luminous: 0,
        },
        tensor: Some(TensorDescriptor::scalar(euc3)),
    }
}

// ── Cosmology ────────────────────────────────────────────────────────────────

/// (ȧ/a)² = 8πGρ/3 - k/a² + Λ/3 (first Friedmann equation)
fn friedmann_first() -> PhysicsEquation {
    PhysicsEquation {
        name: "Friedmann First Equation".to_string(),
        domain: PhysicsDomain::Cosmology,
        ast: make_equals(
            EquationNode::Power {
                base: Box::new(make_product(vec![
                    make_diffop(
                        DiffOperator::TimeDerivative,
                        make_const("a"),
                    ),
                    EquationNode::Power {
                        base: Box::new(make_const("a")),
                        exponent: Box::new(EquationNode::Scalar(-1.0)),
                    },
                ])),
                exponent: Box::new(EquationNode::Scalar(2.0)),
            },
            make_sum(vec![
                make_product(vec![make_const("8πG/3"), make_const("ρ")]),
                EquationNode::Negate(Box::new(make_product(vec![
                    make_const("k"),
                    EquationNode::Power {
                        base: Box::new(make_const("a")),
                        exponent: Box::new(EquationNode::Scalar(-2.0)),
                    },
                ]))),
                make_product(vec![make_const("Λ/3")]),
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature {
            mass: 0,
            length: 0,
            time: -2,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    }
}

/// ä/a = -4πG(ρ + 3p)/3 + Λ/3 (second Friedmann equation)
fn friedmann_second() -> PhysicsEquation {
    PhysicsEquation {
        name: "Friedmann Second Equation".to_string(),
        domain: PhysicsDomain::Cosmology,
        ast: make_equals(
            make_product(vec![
                make_diffop(
                    DiffOperator::TimeDerivative,
                    make_diffop(DiffOperator::TimeDerivative, make_const("a")),
                ),
                EquationNode::Power {
                    base: Box::new(make_const("a")),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
            make_sum(vec![
                EquationNode::Negate(Box::new(make_product(vec![
                    make_const("4πG/3"),
                    make_sum(vec![make_const("ρ"), make_product(vec![
                        EquationNode::Scalar(3.0),
                        make_const("p"),
                    ])]),
                ]))),
                make_product(vec![make_const("Λ/3")]),
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature {
            mass: 0,
            length: 0,
            time: -2,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    }
}

// ── Spark Engine ─────────────────────────────────────────────────────────────

/// <σv> = √(8/(πμ)) (kT)^(-3/2) ∫ S(E) exp(-E/kT - B_G/√E) dE
/// Gamow peak integral for fusion reaction rate
fn gamow_peak_integral() -> PhysicsEquation {
    PhysicsEquation {
        name: "Gamow Peak Integral".to_string(),
        domain: PhysicsDomain::NuclearPhysics,
        ast: make_equals(
            make_const("⟨σv⟩"),
            make_product(vec![
                EquationNode::Sqrt(Box::new(make_product(vec![
                    EquationNode::Scalar(8.0),
                    EquationNode::Power {
                        base: Box::new(make_product(vec![
                            make_const("π"),
                            make_const("μ"),
                        ])),
                        exponent: Box::new(EquationNode::Scalar(-1.0)),
                    },
                ]))),
                EquationNode::Power {
                    base: Box::new(make_const("kT")),
                    exponent: Box::new(EquationNode::Scalar(-1.5)),
                },
                make_diffop(
                    DiffOperator::Integral,
                    make_product(vec![
                        make_const("S(E)"),
                        EquationNode::Exponential(Box::new(EquationNode::Negate(Box::new(
                            make_sum(vec![
                                make_product(vec![
                                    make_const("E"),
                                    EquationNode::Power {
                                        base: Box::new(make_const("kT")),
                                        exponent: Box::new(EquationNode::Scalar(-1.0)),
                                    },
                                ]),
                                make_product(vec![
                                    make_const("B_G"),
                                    EquationNode::Power {
                                        base: Box::new(make_const("E")),
                                        exponent: Box::new(EquationNode::Scalar(-0.5)),
                                    },
                                ]),
                            ]),
                        )))),
                        make_const("dE"),
                    ]),
                ),
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature {
            // <σv> has dimensions of m³/s
            mass: 0,
            length: 3,
            time: -1,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    }
}

/// f = [E/(E+Ue)] × exp(B_G × [1/√E - 1/√(E+Ue)])
/// Assenbaum screening enhancement
fn coulomb_screening() -> PhysicsEquation {
    PhysicsEquation {
        name: "Coulomb Screening Enhancement".to_string(),
        domain: PhysicsDomain::NuclearPhysics,
        ast: make_equals(
            make_const("f"),
            make_product(vec![
                make_product(vec![
                    make_const("E"),
                    EquationNode::Power {
                        base: Box::new(make_sum(vec![make_const("E"), make_const("Ue")])),
                        exponent: Box::new(EquationNode::Scalar(-1.0)),
                    },
                ]),
                EquationNode::Exponential(Box::new(make_product(vec![
                    make_const("B_G"),
                    make_sum(vec![
                        EquationNode::Power {
                            base: Box::new(make_const("E")),
                            exponent: Box::new(EquationNode::Scalar(-0.5)),
                        },
                        EquationNode::Negate(Box::new(EquationNode::Power {
                            base: Box::new(make_sum(vec![
                                make_const("E"),
                                make_const("Ue"),
                            ])),
                            exponent: Box::new(EquationNode::Scalar(-0.5)),
                        })),
                    ]),
                ]))),
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// R_neutron(E) = 0.5 × (1 - 0.003 × E_keV) for D-D branching
fn dd_branching_ratio() -> PhysicsEquation {
    PhysicsEquation {
        name: "D-D Branching Ratio".to_string(),
        domain: PhysicsDomain::NuclearPhysics,
        ast: make_equals(
            make_const("R_n"),
            make_product(vec![
                EquationNode::Scalar(0.5),
                make_sum(vec![
                    EquationNode::Scalar(1.0),
                    EquationNode::Negate(Box::new(make_product(vec![
                        EquationNode::Scalar(0.003),
                        make_const("E_keV"),
                    ]))),
                ]),
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// E_0 = (B_G² · kT² / 4)^(1/3) — Gamow peak energy
fn thermal_gamow_coupling() -> PhysicsEquation {
    PhysicsEquation {
        name: "Thermal-Gamow Coupling".to_string(),
        domain: PhysicsDomain::NuclearPhysics,
        ast: make_equals(
            make_const("E_0"),
            EquationNode::Power {
                base: Box::new(make_product(vec![
                    EquationNode::Power {
                        base: Box::new(make_const("B_G")),
                        exponent: Box::new(EquationNode::Scalar(2.0)),
                    },
                    EquationNode::Power {
                        base: Box::new(make_const("kT")),
                        exponent: Box::new(EquationNode::Scalar(2.0)),
                    },
                    EquationNode::Power {
                        base: Box::new(EquationNode::Scalar(4.0)),
                        exponent: Box::new(EquationNode::Scalar(-1.0)),
                    },
                ])),
                exponent: Box::new(make_product(vec![
                    EquationNode::Scalar(1.0),
                    EquationNode::Power {
                        base: Box::new(EquationNode::Scalar(3.0)),
                        exponent: Box::new(EquationNode::Scalar(-1.0)),
                    },
                ])),
            },
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::HDC_DIMENSION;

    #[test]
    fn catalog_builds_successfully() {
        let catalog = PhysicsCatalog::new();
        assert!(catalog.len() >= 27, "Expected >= 27 entries, got {}", catalog.len());
    }

    #[test]
    fn all_entries_have_valid_hvs() {
        let catalog = PhysicsCatalog::new();
        for entry in catalog.entries() {
            assert_eq!(entry.full_hv.dim(), HDC_DIMENSION);
            assert_eq!(entry.skeleton_hv.dim(), HDC_DIMENSION);
            assert_eq!(entry.symmetry_hv.dim(), HDC_DIMENSION);
            assert_eq!(entry.dimensional_hv.dim(), HDC_DIMENSION);
            assert!(
                entry.full_hv.values.iter().all(|v| v.is_finite()),
                "Non-finite values in full_hv for {}",
                entry.equation.name
            );
            assert!(
                entry.skeleton_hv.values.iter().all(|v| v.is_finite()),
                "Non-finite values in skeleton_hv for {}",
                entry.equation.name
            );
        }
    }

    #[test]
    fn find_by_name_works() {
        let catalog = PhysicsCatalog::new();
        let entry = catalog.find_by_name("Maxwell Gauss Law");
        assert!(entry.is_some());
        assert_eq!(
            entry.unwrap().equation.domain,
            PhysicsDomain::Electromagnetism
        );
    }

    #[test]
    fn entries_in_domain() {
        let catalog = PhysicsCatalog::new();
        let em = catalog.entries_in_domain(PhysicsDomain::Electromagnetism);
        assert_eq!(em.len(), 4, "Expected 4 Maxwell equations");
    }

    #[test]
    fn maxwell_equations_cluster() {
        let catalog = PhysicsCatalog::new();
        let gauss = catalog.find_by_name("Maxwell Gauss Law").unwrap();
        let faraday = catalog.find_by_name("Maxwell Faraday Law").unwrap();
        let einstein = catalog.find_by_name("Einstein Field Equations").unwrap();

        let maxwell_sim = gauss.symmetry_hv.similarity(&faraday.symmetry_hv);
        let maxwell_einstein_sim = gauss.symmetry_hv.similarity(&einstein.symmetry_hv);

        assert!(
            maxwell_sim > maxwell_einstein_sim,
            "Maxwell eqs should cluster: Maxwell-Maxwell ({maxwell_sim}) > Maxwell-Einstein ({maxwell_einstein_sim})"
        );
    }

    #[test]
    fn schwarzschild_kerr_more_similar_than_schwarzschild_flrw() {
        let catalog = PhysicsCatalog::new();
        let schw = catalog.find_by_name("Schwarzschild Metric").unwrap();
        let kerr = catalog.find_by_name("Kerr Metric").unwrap();
        let flrw = catalog.find_by_name("FLRW Metric").unwrap();

        let schw_kerr = schw.full_hv.similarity(&kerr.full_hv);
        let schw_flrw = schw.full_hv.similarity(&flrw.full_hv);

        assert!(
            schw_kerr > schw_flrw,
            "Schwarzschild ≈ Kerr ({schw_kerr}) > Schwarzschild ≈ FLRW ({schw_flrw})"
        );
    }

    #[test]
    fn nuclear_physics_equations_cluster() {
        let catalog = PhysicsCatalog::new();
        let gamow = catalog.find_by_name("Gamow Peak Integral").unwrap();
        let screening = catalog.find_by_name("Coulomb Screening Enhancement").unwrap();
        let wave = catalog.find_by_name("Wave Equation").unwrap();

        let nuclear_sim = gamow.full_hv.similarity(&screening.full_hv);
        let nuclear_wave_sim = gamow.full_hv.similarity(&wave.full_hv);

        assert!(
            nuclear_sim > nuclear_wave_sim,
            "Nuclear eqs should cluster: Gamow-Screening ({nuclear_sim}) > Gamow-Wave ({nuclear_wave_sim})"
        );
    }

    #[test]
    fn spark_engine_equations_present() {
        let catalog = PhysicsCatalog::new();
        assert!(catalog.find_by_name("Gamow Peak Integral").is_some());
        assert!(catalog.find_by_name("Coulomb Screening Enhancement").is_some());
        assert!(catalog.find_by_name("D-D Branching Ratio").is_some());
        assert!(catalog.find_by_name("Thermal-Gamow Coupling").is_some());
    }

    #[test]
    fn catalog_deterministic() {
        let c1 = PhysicsCatalog::new();
        let c2 = PhysicsCatalog::new();
        for (e1, e2) in c1.entries().iter().zip(c2.entries().iter()) {
            assert_eq!(e1.full_hv.values, e2.full_hv.values);
            assert_eq!(e1.equation.name, e2.equation.name);
        }
    }
}
