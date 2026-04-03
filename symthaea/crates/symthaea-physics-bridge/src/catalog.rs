// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    let mut eqs = Vec::with_capacity(48);

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

    // Nuclear Forces
    eqs.push(yukawa_potential());
    eqs.push(one_pion_exchange());
    eqs.push(nuclear_radius_formula());
    eqs.push(geiger_nuttall_law());
    eqs.push(bateman_equations());

    // Modified Gravity
    eqs.push(f_r_gravity());
    eqs.push(mond_milgrom());
    eqs.push(brans_dicke());
    eqs.push(proca_equation());

    // Statistical Mechanics
    eqs.push(boltzmann_distribution());
    eqs.push(ising_hamiltonian());
    eqs.push(partition_function());

    // Particle Physics
    eqs.push(running_coupling_alpha_s());
    eqs.push(casimir_force());
    eqs.push(higgs_potential());

    // Optics
    eqs.push(snell_law());
    eqs.push(fresnel_equations());

    // Condensed Matter
    eqs.push(bcs_gap_equation());

    // Additional Physics (filling gaps)
    eqs.push(waveguide_dispersion());
    eqs.push(drude_model());
    eqs.push(schwarzschild_radius());
    eqs.push(hawking_temperature());
    eqs.push(landauer_principle());
    eqs.push(ideal_gas_law());
    eqs.push(stefan_boltzmann_law());

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
            vec![
                DiscreteSymmetry::C,
                DiscreteSymmetry::P,
                DiscreteSymmetry::T,
            ],
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
            vec![
                DiscreteSymmetry::C,
                DiscreteSymmetry::P,
                DiscreteSymmetry::T,
            ],
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
            vec![
                DiscreteSymmetry::C,
                DiscreteSymmetry::P,
                DiscreteSymmetry::T,
            ],
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
            vec![
                DiscreteSymmetry::C,
                DiscreteSymmetry::P,
                DiscreteSymmetry::T,
            ],
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
                    make_product(vec![make_const("Σ/Δ"), make_const("dr²")]),
                    make_product(vec![make_const("Σ"), make_const("dθ²")]),
                    make_product(vec![make_const("cross_term"), make_const("dtdφ")]),
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
                            EquationNode::Negate(Box::new(make_product(vec![make_const(
                                "2GM/c²r",
                            )]))),
                            make_product(vec![make_const("GQ²/(4πε₀c⁴r²)")]),
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
                            EquationNode::Negate(Box::new(make_product(vec![make_const("Λr²/3")]))),
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
        symmetries: SymmetryDescriptor::new(vec![LieGroup::U(1)], vec![DiscreteSymmetry::T], true),
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
            make_product(vec![make_sum(vec![
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
            ])]),
            EquationNode::Scalar(0.0),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::Poincare, LieGroup::U(1)],
            vec![
                DiscreteSymmetry::C,
                DiscreteSymmetry::P,
                DiscreteSymmetry::T,
            ],
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
            vec![
                DiscreteSymmetry::C,
                DiscreteSymmetry::P,
                DiscreteSymmetry::T,
            ],
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
                    make_diffop(DiffOperator::TimeDerivative, make_const("a")),
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
                    make_sum(vec![
                        make_const("ρ"),
                        make_product(vec![EquationNode::Scalar(3.0), make_const("p")]),
                    ]),
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
                        base: Box::new(make_product(vec![make_const("π"), make_const("μ")])),
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
                            base: Box::new(make_sum(vec![make_const("E"), make_const("Ue")])),
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

// ══════════════════════════════════════════════════════════════════════════════
// NUCLEAR FORCES
// ══════════════════════════════════════════════════════════════════════════════

/// V(r) = -g²·exp(-μr)/(4πr) — Yukawa meson-exchange potential
fn yukawa_potential() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "Yukawa Potential".to_string(),
        domain: PhysicsDomain::NuclearPhysics,
        ast: make_equals(
            make_field("V", TensorDescriptor::scalar(euc3)),
            EquationNode::Negate(Box::new(make_product(vec![
                make_const("g²/4π"),
                EquationNode::Exponential(Box::new(EquationNode::Negate(Box::new(
                    make_product(vec![make_const("μ"), make_field("r", TensorDescriptor::scalar(euc3))]),
                )))),
                EquationNode::Power {
                    base: Box::new(make_field("r", TensorDescriptor::scalar(euc3))),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]))),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::SO(3)],
            vec![DiscreteSymmetry::P, DiscreteSymmetry::T],
            false,
        ),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    }
}

/// V_OPEP(r) ∝ (σ₁·σ₂)(τ₁·τ₂) exp(-m_π r)/(m_π r) — One-Pion Exchange
fn one_pion_exchange() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "One-Pion Exchange Potential".to_string(),
        domain: PhysicsDomain::NuclearPhysics,
        ast: make_equals(
            make_field("V_OPEP", TensorDescriptor::scalar(euc3)),
            make_product(vec![
                make_const("f²_πNN/4π"),
                make_const("m_π/3"),
                make_const("σ₁·σ₂"),
                make_const("τ₁·τ₂"),
                EquationNode::Exponential(Box::new(EquationNode::Negate(Box::new(
                    make_product(vec![make_const("m_π"), make_field("r", TensorDescriptor::scalar(euc3))]),
                )))),
                EquationNode::Power {
                    base: Box::new(make_product(vec![
                        make_const("m_π"),
                        make_field("r", TensorDescriptor::scalar(euc3)),
                    ])),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::SU(2), LieGroup::SU(2)], // Spin × Isospin
            vec![DiscreteSymmetry::P, DiscreteSymmetry::T],
            false,
        ),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    }
}

/// R = r₀·A^(1/3) — Nuclear radius formula
fn nuclear_radius_formula() -> PhysicsEquation {
    PhysicsEquation {
        name: "Nuclear Radius Formula".to_string(),
        domain: PhysicsDomain::NuclearPhysics,
        ast: make_equals(
            make_const("R"),
            make_product(vec![
                make_const("r₀"),
                EquationNode::Power {
                    base: Box::new(make_const("A")),
                    exponent: Box::new(make_product(vec![
                        EquationNode::Scalar(1.0),
                        EquationNode::Power {
                            base: Box::new(EquationNode::Scalar(3.0)),
                            exponent: Box::new(EquationNode::Scalar(-1.0)),
                        },
                    ])),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::LENGTH,
        tensor: None,
    }
}

/// log₁₀(t½) = aZ/√Q + b — Geiger-Nuttall law for alpha decay
fn geiger_nuttall_law() -> PhysicsEquation {
    PhysicsEquation {
        name: "Geiger-Nuttall Law".to_string(),
        domain: PhysicsDomain::NuclearPhysics,
        ast: make_equals(
            make_const("log₁₀(t½)"),
            make_sum(vec![
                make_product(vec![
                    make_const("a"),
                    make_const("Z"),
                    EquationNode::Power {
                        base: Box::new(make_const("Q_α")),
                        exponent: Box::new(EquationNode::Scalar(-0.5)),
                    },
                ]),
                make_const("b"),
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// dN_i/dt = Σ λ_j·N_j - λ_i·N_i — Bateman radioactive decay chain
fn bateman_equations() -> PhysicsEquation {
    PhysicsEquation {
        name: "Bateman Decay Equations".to_string(),
        domain: PhysicsDomain::NuclearPhysics,
        ast: make_equals(
            make_diffop(
                DiffOperator::TimeDerivative,
                make_const("N_i"),
            ),
            make_sum(vec![
                make_product(vec![make_const("λ_parent"), make_const("N_parent")]),
                EquationNode::Negate(Box::new(make_product(vec![
                    make_const("λ_i"),
                    make_const("N_i"),
                ]))),
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature {
            mass: 0,
            length: 0,
            time: -1,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// MODIFIED GRAVITY
// ══════════════════════════════════════════════════════════════════════════════

/// f(R) gravity: f'(R)R_μν - f(R)g_μν/2 + g_μν□f'(R) - ∇_μ∇_νf'(R) = κT_μν
fn f_r_gravity() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    PhysicsEquation {
        name: "f(R) Modified Gravity".to_string(),
        domain: PhysicsDomain::ModifiedGravity,
        ast: make_equals(
            make_sum(vec![
                make_product(vec![
                    make_const("f'(R)"),
                    make_field("R_μν", TensorDescriptor::symmetric_2(lor4)),
                ]),
                EquationNode::Negate(Box::new(make_product(vec![
                    EquationNode::Scalar(0.5),
                    make_const("f(R)"),
                    make_field("g_μν", TensorDescriptor::symmetric_2(lor4)),
                ]))),
            ]),
            make_product(vec![
                make_const("κ"),
                make_field("T_μν", TensorDescriptor::symmetric_2(lor4)),
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::GL(4)], false),
        dimensions: DimensionalSignature::INVERSE_LENGTH,
        tensor: Some(TensorDescriptor::symmetric_2(lor4)),
    }
}

/// μ(|a|/a₀)·a = a_N — MOND (Modified Newtonian Dynamics)
fn mond_milgrom() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "MOND Milgrom Law".to_string(),
        domain: PhysicsDomain::ModifiedGravity,
        ast: make_equals(
            make_product(vec![
                make_const("μ(|a|/a₀)"),
                make_field("a", TensorDescriptor::vector(euc3)),
            ]),
            make_field("a_N", TensorDescriptor::vector(euc3)),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature::ACCELERATION,
        tensor: Some(TensorDescriptor::vector(euc3)),
    }
}

/// □φ = 8πT/(3+2ω) — Brans-Dicke scalar-tensor gravity
fn brans_dicke() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    PhysicsEquation {
        name: "Brans-Dicke Equation".to_string(),
        domain: PhysicsDomain::ModifiedGravity,
        ast: make_equals(
            make_diffop(
                DiffOperator::DAlembertian,
                make_field("φ", TensorDescriptor::scalar(lor4)),
            ),
            make_product(vec![
                make_const("8πT/(3+2ω)"),
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::GL(4)], false),
        dimensions: DimensionalSignature::INVERSE_LENGTH,
        tensor: Some(TensorDescriptor::scalar(lor4)),
    }
}

/// ∂_μF^μν + m²A^ν = J^ν — Proca equation (massive vector boson → finite-range force)
fn proca_equation() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    PhysicsEquation {
        name: "Proca Equation".to_string(),
        domain: PhysicsDomain::ParticlePhysics,
        ast: make_equals(
            make_sum(vec![
                make_diffop(
                    DiffOperator::Partial,
                    make_field("F^μν", TensorDescriptor::antisymmetric_2(lor4)),
                ),
                make_product(vec![
                    EquationNode::Power {
                        base: Box::new(make_const("m")),
                        exponent: Box::new(EquationNode::Scalar(2.0)),
                    },
                    make_field("A^ν", TensorDescriptor::vector(lor4)),
                ]),
            ]),
            make_field("J^ν", TensorDescriptor::vector(lor4)),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::Lorentz],
            vec![DiscreteSymmetry::C, DiscreteSymmetry::P, DiscreteSymmetry::T],
            false, // NOT gauge invariant (mass breaks gauge symmetry)
        ),
        dimensions: DimensionalSignature::INVERSE_LENGTH,
        tensor: Some(TensorDescriptor::vector(lor4)),
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// STATISTICAL MECHANICS
// ══════════════════════════════════════════════════════════════════════════════

/// P(E) ∝ exp(-E/kT) — Boltzmann distribution
fn boltzmann_distribution() -> PhysicsEquation {
    PhysicsEquation {
        name: "Boltzmann Distribution".to_string(),
        domain: PhysicsDomain::StatisticalMechanics,
        ast: make_equals(
            make_const("P(E)"),
            EquationNode::Exponential(Box::new(EquationNode::Negate(Box::new(
                make_product(vec![
                    make_const("E"),
                    EquationNode::Power {
                        base: Box::new(make_const("kT")),
                        exponent: Box::new(EquationNode::Scalar(-1.0)),
                    },
                ]),
            )))),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// H = -J·Σ s_i·s_j - h·Σ s_i — Ising model Hamiltonian
fn ising_hamiltonian() -> PhysicsEquation {
    PhysicsEquation {
        name: "Ising Hamiltonian".to_string(),
        domain: PhysicsDomain::StatisticalMechanics,
        ast: make_equals(
            make_const("H"),
            EquationNode::Negate(Box::new(make_sum(vec![
                make_product(vec![make_const("J"), make_const("Σ_⟨ij⟩ s_i·s_j")]),
                make_product(vec![make_const("h"), make_const("Σ_i s_i")]),
            ]))),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![],
            vec![DiscreteSymmetry::Z2], // Z₂ spin-flip symmetry (when h=0)
            false,
        ),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    }
}

/// Z = Σ_n exp(-βE_n) — Canonical partition function
fn partition_function() -> PhysicsEquation {
    PhysicsEquation {
        name: "Canonical Partition Function".to_string(),
        domain: PhysicsDomain::StatisticalMechanics,
        ast: make_equals(
            make_const("Z"),
            make_const("Σ_n exp(-βE_n)"),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// PARTICLE PHYSICS
// ══════════════════════════════════════════════════════════════════════════════

/// α_s(Q²) = α_s(M_Z²) / (1 + b₀·α_s·ln(Q²/M_Z²)/(2π)) — QCD running coupling
fn running_coupling_alpha_s() -> PhysicsEquation {
    PhysicsEquation {
        name: "QCD Running Coupling".to_string(),
        domain: PhysicsDomain::ParticlePhysics,
        ast: make_equals(
            make_const("α_s(Q²)"),
            make_product(vec![
                make_const("α_s(M_Z²)"),
                EquationNode::Power {
                    base: Box::new(make_sum(vec![
                        EquationNode::Scalar(1.0),
                        make_product(vec![
                            make_const("b₀·α_s/(2π)"),
                            make_const("ln(Q²/M_Z²)"),
                        ]),
                    ])),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SU(3)], true),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// F/A = -π²ℏc/(240d⁴) — Casimir force between parallel plates
fn casimir_force() -> PhysicsEquation {
    PhysicsEquation {
        name: "Casimir Force".to_string(),
        domain: PhysicsDomain::ParticlePhysics,
        ast: make_equals(
            make_const("F/A"),
            EquationNode::Negate(Box::new(make_product(vec![
                make_const("π²ℏc/240"),
                EquationNode::Power {
                    base: Box::new(make_const("d")),
                    exponent: Box::new(EquationNode::Scalar(-4.0)),
                },
            ]))),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::U(1)],
            vec![DiscreteSymmetry::P, DiscreteSymmetry::T],
            true,
        ),
        dimensions: DimensionalSignature::PRESSURE,
        tensor: None,
    }
}

/// V(φ) = μ²|φ|² + λ|φ|⁴ — Higgs potential (Mexican hat)
fn higgs_potential() -> PhysicsEquation {
    let lor4 = MetricSignature::Lorentzian(4);
    PhysicsEquation {
        name: "Higgs Potential".to_string(),
        domain: PhysicsDomain::ParticlePhysics,
        ast: make_equals(
            make_const("V(φ)"),
            make_sum(vec![
                make_product(vec![
                    make_const("μ²"),
                    EquationNode::Power {
                        base: Box::new(make_field("φ", TensorDescriptor::scalar(lor4))),
                        exponent: Box::new(EquationNode::Scalar(2.0)),
                    },
                ]),
                make_product(vec![
                    make_const("λ"),
                    EquationNode::Power {
                        base: Box::new(make_field("φ", TensorDescriptor::scalar(lor4))),
                        exponent: Box::new(EquationNode::Scalar(4.0)),
                    },
                ]),
            ]),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::SU(2), LieGroup::U(1)],
            vec![],
            true, // Gauge symmetry (spontaneously broken)
        ),
        dimensions: DimensionalSignature::ENERGY_DENSITY,
        tensor: None,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// OPTICS
// ══════════════════════════════════════════════════════════════════════════════

/// n₁sin(θ₁) = n₂sin(θ₂) — Snell's law of refraction
fn snell_law() -> PhysicsEquation {
    PhysicsEquation {
        name: "Snell Law".to_string(),
        domain: PhysicsDomain::Optics,
        ast: make_equals(
            make_product(vec![make_const("n₁"), make_const("sin(θ₁)")]),
            make_product(vec![make_const("n₂"), make_const("sin(θ₂)")]),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![],
            vec![DiscreteSymmetry::T], // Time-reversible
            false,
        ),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// r_s = (n₁cos(θ₁) - n₂cos(θ₂))/(n₁cos(θ₁) + n₂cos(θ₂)) — Fresnel s-polarization
fn fresnel_equations() -> PhysicsEquation {
    PhysicsEquation {
        name: "Fresnel Equations".to_string(),
        domain: PhysicsDomain::Optics,
        ast: make_equals(
            make_const("r_s"),
            make_product(vec![
                make_sum(vec![
                    make_product(vec![make_const("n₁"), make_const("cos(θ₁)")]),
                    EquationNode::Negate(Box::new(make_product(vec![
                        make_const("n₂"),
                        make_const("cos(θ₂)"),
                    ]))),
                ]),
                EquationNode::Power {
                    base: Box::new(make_sum(vec![
                        make_product(vec![make_const("n₁"), make_const("cos(θ₁)")]),
                        make_product(vec![make_const("n₂"), make_const("cos(θ₂)")]),
                    ])),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::U(1)],
            vec![DiscreteSymmetry::T],
            false,
        ),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// CONDENSED MATTER
// ══════════════════════════════════════════════════════════════════════════════

/// Δ = 2ℏω_D·exp(-1/(N(0)V)) — BCS superconducting gap equation
fn bcs_gap_equation() -> PhysicsEquation {
    PhysicsEquation {
        name: "BCS Gap Equation".to_string(),
        domain: PhysicsDomain::CondensedMatter,
        ast: make_equals(
            make_const("Δ"),
            make_product(vec![
                EquationNode::Scalar(2.0),
                make_const("ℏω_D"),
                EquationNode::Exponential(Box::new(EquationNode::Negate(Box::new(
                    EquationNode::Power {
                        base: Box::new(make_product(vec![
                            make_const("N(0)"),
                            make_const("V"),
                        ])),
                        exponent: Box::new(EquationNode::Scalar(-1.0)),
                    },
                )))),
            ]),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::U(1)],
            vec![],
            true, // U(1) gauge symmetry broken by Cooper pairs
        ),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// LAZAR STRUCTURAL QUERY
// ══════════════════════════════════════════════════════════════════════════════

/// Lazar "Gravity-A" wave: claimed extension of strong nuclear force beyond
/// the nucleus via Element 115. Structurally, this is a Yukawa potential with
/// the range parameter μ→0 (infinite range), making it a 1/r potential
/// coupled to a massive vector field.
///
/// This is NOT a real physics equation — it's a query designed to find
/// structural analogs in the catalog via HDC skeleton similarity.
pub fn lazar_gravity_a_query() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "Lazar Gravity-A Wave (Query)".to_string(),
        domain: PhysicsDomain::ModifiedGravity,
        ast: make_equals(
            make_field("F_gravity_A", TensorDescriptor::vector(euc3)),
            make_product(vec![
                make_const("coupling_115"),
                // Key structural claim: 1/r potential (Yukawa with μ→0)
                // exp(-μr) → exp(0) = 1 when μ=0, leaving pure 1/r
                EquationNode::Power {
                    base: Box::new(make_field("r", TensorDescriptor::scalar(euc3))),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
                // Directional amplitude from 3-amplifier focusing
                make_const("amplitude(θ,φ)"),
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature::FORCE,
        tensor: Some(TensorDescriptor::vector(euc3)),
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// ADDITIONAL PHYSICS (FILLING GAPS)
// ══════════════════════════════════════════════════════════════════════════════

/// ω² = (c/n)²k² + ω_c² — Waveguide dispersion relation
///
/// The cutoff frequency ω_c depends on waveguide geometry and mode number.
/// Below cutoff, waves are evanescent (exponentially decaying).
fn waveguide_dispersion() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "Waveguide Dispersion Relation".to_string(),
        domain: PhysicsDomain::Optics,
        ast: make_equals(
            EquationNode::Power {
                base: Box::new(make_const("ω")),
                exponent: Box::new(EquationNode::Scalar(2.0)),
            },
            make_sum(vec![
                make_product(vec![
                    EquationNode::Power {
                        base: Box::new(make_const("c/n")),
                        exponent: Box::new(EquationNode::Scalar(2.0)),
                    },
                    EquationNode::Power {
                        base: Box::new(make_field("k", TensorDescriptor::scalar(euc3))),
                        exponent: Box::new(EquationNode::Scalar(2.0)),
                    },
                ]),
                EquationNode::Power {
                    base: Box::new(make_const("ω_cutoff")),
                    exponent: Box::new(EquationNode::Scalar(2.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![],
            vec![DiscreteSymmetry::T],
            false,
        ),
        dimensions: DimensionalSignature {
            mass: 0, length: 0, time: -2,
            current: 0, temperature: 0, amount: 0, luminous: 0,
        },
        tensor: None,
    }
}

/// σ(ω) = ne²/(m(γ - iω)) — Drude model for metallic conductivity
///
/// Describes free electron response to EM fields. Foundation of metamaterial
/// design — negative permittivity below plasma frequency enables waveguiding.
fn drude_model() -> PhysicsEquation {
    PhysicsEquation {
        name: "Drude Conductivity Model".to_string(),
        domain: PhysicsDomain::CondensedMatter,
        ast: make_equals(
            make_const("σ(ω)"),
            make_product(vec![
                make_const("ne²/m"),
                EquationNode::Power {
                    base: Box::new(make_sum(vec![
                        make_const("γ"),
                        EquationNode::Negate(Box::new(make_product(vec![
                            make_const("i"),
                            make_const("ω"),
                        ]))),
                    ])),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::U(1)],
            vec![DiscreteSymmetry::T],
            false,
        ),
        dimensions: DimensionalSignature {
            mass: -1, length: -3, time: 3,
            current: 2, temperature: 0, amount: 0, luminous: 0,
        },
        tensor: None,
    }
}

/// r_s = 2GM/c² — Schwarzschild radius
fn schwarzschild_radius() -> PhysicsEquation {
    PhysicsEquation {
        name: "Schwarzschild Radius".to_string(),
        domain: PhysicsDomain::GeneralRelativity,
        ast: make_equals(
            make_const("r_s"),
            make_product(vec![
                EquationNode::Scalar(2.0),
                make_const("G"),
                make_const("M"),
                EquationNode::Power {
                    base: Box::new(make_const("c")),
                    exponent: Box::new(EquationNode::Scalar(-2.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature::LENGTH,
        tensor: None,
    }
}

/// T_H = ℏc³/(8πGMk_B) — Hawking temperature
fn hawking_temperature() -> PhysicsEquation {
    PhysicsEquation {
        name: "Hawking Temperature".to_string(),
        domain: PhysicsDomain::GeneralRelativity,
        ast: make_equals(
            make_const("T_H"),
            make_product(vec![
                make_const("ℏc³"),
                EquationNode::Power {
                    base: Box::new(make_product(vec![
                        make_const("8πG"),
                        make_const("M"),
                        make_const("k_B"),
                    ])),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature {
            mass: 0, length: 0, time: 0,
            current: 0, temperature: 1, amount: 0, luminous: 0,
        },
        tensor: None,
    }
}

/// E_min = k_B T ln(2) — Landauer's principle (minimum energy to erase one bit)
///
/// Already used in Symthaea's thermodynamic physics bridge for memory cost accounting.
fn landauer_principle() -> PhysicsEquation {
    PhysicsEquation {
        name: "Landauer Principle".to_string(),
        domain: PhysicsDomain::Thermodynamics,
        ast: make_equals(
            make_const("E_min"),
            make_product(vec![
                make_const("k_B"),
                make_const("T"),
                make_const("ln(2)"),
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    }
}

/// PV = nRT — Ideal gas law
fn ideal_gas_law() -> PhysicsEquation {
    PhysicsEquation {
        name: "Ideal Gas Law".to_string(),
        domain: PhysicsDomain::Thermodynamics,
        ast: make_equals(
            make_product(vec![make_const("P"), make_const("V")]),
            make_product(vec![make_const("n"), make_const("R"), make_const("T")]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    }
}

/// P = σ T⁴ — Stefan-Boltzmann law (blackbody radiation power per unit area)
fn stefan_boltzmann_law() -> PhysicsEquation {
    PhysicsEquation {
        name: "Stefan-Boltzmann Law".to_string(),
        domain: PhysicsDomain::Thermodynamics,
        ast: make_equals(
            make_const("P/A"),
            make_product(vec![
                make_const("σ_SB"),
                EquationNode::Power {
                    base: Box::new(make_const("T")),
                    exponent: Box::new(EquationNode::Scalar(4.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature {
            mass: 1, length: 0, time: -3,
            current: 0, temperature: 0, amount: 0, luminous: 0,
        },
        tensor: None,
    }
}

/// "Art's Parts" terahertz waveguide claim: alternating Bi/Mg layers
/// act as a metamaterial waveguide at THz frequencies, enabling
/// anti-gravity propulsion.
///
/// Structurally, this is a guided wave dispersion relation in a
/// layered dielectric medium: ω² = c²k² + ω_cutoff², where the
/// cutoff frequency depends on layer thickness and refractive indices.
///
/// ORNL (2022) found: impure Bi, multiple Bi layers, terrestrial isotopes.
pub fn arts_parts_waveguide_query() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "Art's Parts THz Waveguide (Query)".to_string(),
        domain: PhysicsDomain::Optics,
        ast: make_equals(
            // ω² dispersion relation
            EquationNode::Power {
                base: Box::new(make_const("ω")),
                exponent: Box::new(EquationNode::Scalar(2.0)),
            },
            make_sum(vec![
                // c²k² (free-space dispersion)
                make_product(vec![
                    EquationNode::Power {
                        base: Box::new(make_const("c/n_eff")),
                        exponent: Box::new(EquationNode::Scalar(2.0)),
                    },
                    EquationNode::Power {
                        base: Box::new(make_field("k", TensorDescriptor::scalar(euc3))),
                        exponent: Box::new(EquationNode::Scalar(2.0)),
                    },
                ]),
                // ω_cutoff² (from layer structure)
                EquationNode::Power {
                    base: Box::new(make_const("ω_cutoff")),
                    exponent: Box::new(EquationNode::Scalar(2.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![],
            vec![DiscreteSymmetry::T, DiscreteSymmetry::P],
            false,
        ),
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

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::HDC_DIMENSION;

    #[test]
    fn catalog_builds_successfully() {
        let catalog = PhysicsCatalog::new();
        assert!(
            catalog.len() >= 52,
            "Expected >= 52 entries, got {}",
            catalog.len()
        );
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
        let screening = catalog
            .find_by_name("Coulomb Screening Enhancement")
            .unwrap();
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
        assert!(catalog
            .find_by_name("Coulomb Screening Enhancement")
            .is_some());
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

    // ── Phase 2a expansion tests ──

    #[test]
    fn new_domains_present() {
        let catalog = PhysicsCatalog::new();
        assert!(
            !catalog
                .entries_in_domain(PhysicsDomain::ModifiedGravity)
                .is_empty(),
            "Modified gravity equations should exist"
        );
        assert!(
            !catalog
                .entries_in_domain(PhysicsDomain::ParticlePhysics)
                .is_empty(),
            "Particle physics equations should exist"
        );
        assert!(
            !catalog
                .entries_in_domain(PhysicsDomain::CondensedMatter)
                .is_empty(),
            "Condensed matter equations should exist"
        );
        assert!(
            !catalog
                .entries_in_domain(PhysicsDomain::Optics)
                .is_empty(),
            "Optics equations should exist"
        );
    }

    #[test]
    fn yukawa_present_and_valid() {
        let catalog = PhysicsCatalog::new();
        let yukawa = catalog.find_by_name("Yukawa Potential");
        assert!(yukawa.is_some(), "Yukawa potential should be in catalog");
        let y = yukawa.unwrap();
        assert_eq!(y.equation.domain, PhysicsDomain::NuclearPhysics);
        assert!(y.full_hv.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn yukawa_clusters_with_opep() {
        let catalog = PhysicsCatalog::new();
        let yukawa = catalog.find_by_name("Yukawa Potential").unwrap();
        let opep = catalog.find_by_name("One-Pion Exchange Potential").unwrap();
        let snell = catalog.find_by_name("Snell Law").unwrap();

        // Yukawa and OPEP share the same exp(-μr)/r skeleton
        let nuclear_sim = yukawa.skeleton_hv.similarity(&opep.skeleton_hv);
        let cross_sim = yukawa.skeleton_hv.similarity(&snell.skeleton_hv);

        assert!(
            nuclear_sim > cross_sim,
            "Yukawa-OPEP skeleton similarity ({}) should exceed Yukawa-Snell ({})",
            nuclear_sim,
            cross_sim
        );
    }

    #[test]
    fn lazar_query_structural_search() {
        use crate::query::PhysicsSearchEngine;

        let engine = PhysicsSearchEngine::new();
        let query = lazar_gravity_a_query();

        let results = engine.search_equation(&query, 5);

        // The Lazar query should find SOME structural analogs
        assert!(
            !results.is_empty(),
            "Lazar query should return at least one result"
        );

        // Report what the nearest neighbors are (informational)
        for r in &results {
            eprintln!(
                "  Lazar neighbor: {} (score={:.3}, domain={:?})",
                r.name, r.score, r.domain
            );
        }
    }

    #[test]
    fn all_new_entries_have_finite_hvs() {
        let catalog = PhysicsCatalog::new();
        let new_names = [
            "Yukawa Potential",
            "One-Pion Exchange Potential",
            "Nuclear Radius Formula",
            "Geiger-Nuttall Law",
            "Bateman Decay Equations",
            "f(R) Modified Gravity",
            "MOND Milgrom Law",
            "Brans-Dicke Equation",
            "Proca Equation",
            "Boltzmann Distribution",
            "Ising Hamiltonian",
            "Canonical Partition Function",
            "QCD Running Coupling",
            "Casimir Force",
            "Higgs Potential",
            "Snell Law",
            "Fresnel Equations",
            "BCS Gap Equation",
            "Waveguide Dispersion Relation",
            "Drude Conductivity Model",
            "Schwarzschild Radius",
            "Hawking Temperature",
            "Landauer Principle",
            "Ideal Gas Law",
            "Stefan-Boltzmann Law",
        ];

        for name in &new_names {
            let entry = catalog.find_by_name(name);
            assert!(entry.is_some(), "Missing equation: {}", name);
            let e = entry.unwrap();
            assert!(
                e.full_hv.values.iter().all(|v| v.is_finite()),
                "Non-finite HV for {}",
                name
            );
        }
    }

    #[test]
    fn arts_parts_waveguide_search() {
        use crate::query::PhysicsSearchEngine;

        let engine = PhysicsSearchEngine::new();
        let query = arts_parts_waveguide_query();
        let results = engine.search_equation(&query, 5);

        assert!(!results.is_empty(), "Art's Parts query should return results");

        // Report neighbors (informational)
        for r in &results {
            eprintln!(
                "  Art's Parts neighbor: {} (score={:.3}, domain={:?})",
                r.name, r.score, r.domain
            );
        }
    }
}
