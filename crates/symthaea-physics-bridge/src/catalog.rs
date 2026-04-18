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
use crate::recognize::expr_to_equation_node;
use crate::symmetry::SymmetryEncoder;
use crate::types::*;
use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};
use symthaea_core::hdc::ContinuousHV;

/// Build a complete catalog entry `ast` (LHS + RHS wrapped in Equals) from a
/// ConjectureEngine `Expr`, routing through the same `expr_to_equation_node`
/// conversion that autonomous discovery uses at recognition time.
///
/// This eliminates the entire class of invisible structural-mismatch bugs
/// where a hand-constructed catalog AST differs in subtle ways (nested vs
/// flat Sum, `Const(-0.5)` vs `Negate(Const(0.5))`, literal naming
/// conventions) from whatever the discovery path produces.
///
/// The caller supplies a unique `lhs_name` — we deliberately avoid sharing
/// a single LHS token (like `"result"`) across all dogfooded entries,
/// because that would inject a common HV component into every entry's
/// full encoding and create false 99% matches between unrelated entries
/// whose RHS share any atomic structure. Per-entry unique names keep the
/// full axis honest: its job is to tiebreak among otherwise-identical
/// skeletons, not to dominate.
///
/// **Use this** for any catalog entry whose canonical form is an invariant
/// the ConjectureEngine is expected to rediscover autonomously.
fn expr_to_catalog_ast(lhs_name: &str, expr: &Expr) -> EquationNode {
    make_equals(
        EquationNode::Constant { name: lhs_name.to_string() },
        expr_to_equation_node(expr),
    )
}

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

/// Lightweight equation builder for equations that use a simple name+description encoding.
///
/// Creates an equation with a single constant-name AST node. The HDC encoding
/// captures the equation's name, domain, and dimensional signature without needing
/// a full AST tree. This is useful for filling out the catalog rapidly.
fn simple_eq(
    name: &str,
    domain: PhysicsDomain,
    description: &str,
    dimensions: DimensionalSignature,
) -> PhysicsEquation {
    PhysicsEquation {
        name: name.to_string(),
        domain,
        ast: make_equals(make_const(name), make_const(description)),
        symmetries: SymmetryDescriptor::none(),
        dimensions,
        tensor: None,
    }
}

/// Build all landmark equations for the catalog.
fn build_all_equations() -> Vec<PhysicsEquation> {
    let mut eqs = Vec::with_capacity(160);

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

    // ── Phase A1: Foundational Equations ──
    eqs.push(newton_second_law());
    eqs.push(newton_gravitation());
    eqs.push(kepler_third_law());
    eqs.push(hooke_law());
    eqs.push(centripetal_acceleration());
    eqs.push(coulomb_law());
    eqs.push(lorentz_force());
    eqs.push(ohm_law());
    eqs.push(planck_einstein_relation());
    eqs.push(de_broglie_wavelength());
    eqs.push(heisenberg_uncertainty());
    eqs.push(photoelectric_equation());
    eqs.push(mass_energy_equivalence());
    eqs.push(lorentz_factor());
    eqs.push(planck_radiation_law());
    eqs.push(wien_displacement());
    eqs.push(rayleigh_scattering());

    // ── Phase A2: Information Theory ──
    eqs.push(shannon_entropy());
    eqs.push(kl_divergence());
    eqs.push(mutual_information());
    eqs.push(cross_entropy());
    eqs.push(fisher_information());
    eqs.push(variational_free_energy());
    eqs.push(integrated_information_phi());
    eqs.push(boltzmann_entropy());
    eqs.push(bekenstein_hawking_entropy());

    // ── Phase A3: Named Equations + Biophysics ──
    eqs.push(bernoulli_equation());
    eqs.push(clausius_clapeyron());
    eqs.push(gibbs_free_energy());
    eqs.push(van_der_waals());
    eqs.push(continuity_equation());
    eqs.push(fourier_heat_law());
    eqs.push(fermi_golden_rule());
    eqs.push(compton_scattering());
    eqs.push(poisson_equation());
    eqs.push(rydberg_formula());
    eqs.push(hodgkin_huxley());
    eqs.push(lotka_volterra());
    eqs.push(bayes_theorem());
    eqs.push(arrhenius_equation());
    eqs.push(hubble_law());

    // ── Phase A4: Orbital Mechanics + Hydrogen Spectrum ──
    // Fills the gap between Kepler's third law (periods) and Rydberg (wavelengths)
    // with the direct ENERGY forms that autonomous discovery actually produces.
    eqs.push(hydrogen_energy_levels());
    eqs.push(kepler_orbital_energy());
    eqs.push(gravitational_potential_energy());
    eqs.push(harmonic_oscillator_energy());
    eqs.push(inverse_square_force());

    // ── Phase A5: Invariant-form entries (Ramanujan Protocol showcase targets) ──
    // These mirror the *shapes* autonomous conservation-law discovery actually
    // produces (natural-units forms, Cartesian components, transcendental
    // invariants), so recognition can route discoveries directly to their true
    // catalog cousins instead of to nearest-neighbor noise.
    eqs.push(harmonic_oscillator_invariant());
    eqs.push(lotka_volterra_invariant());
    eqs.push(angular_momentum_2d_cartesian());
    eqs.push(henon_heiles_hamiltonian());

    // ── Combinatorics (Mathematics domain) ──
    // These give sequence-discovery targets a direct catalog home instead of
    // routing to nearest-neighbor nuclear physics. Closes the Ramanujan
    // Protocol showcase's last weak match (triangular → Coulomb Screening 0.70).
    //
    // NOTE: Sum of Cubes (`n²(n+1)²/4`) was tried but removed — its shape
    // was too generic and produced false-positive matches at 99% against
    // any discovered formula containing nested power-of-variable subtrees
    // (e.g. PCR3BP's garbage `cos(y/e)^(x³)`). Its function `sum_of_cubes()`
    // is kept for reference but no longer pushed. If the similarity metric
    // is ever tightened to weight top-level operator agreement more heavily,
    // it can be re-added.
    eqs.push(triangular_numbers());
    eqs.push(square_pyramidal_numbers());
    eqs.push(tetrahedral_numbers());
    eqs.push(harmonic_numbers());

    // ── Phase 1B: Expand to 150 ──
    // Classical Mechanics
    eqs.push(simple_eq(
        "Simple Harmonic Oscillator",
        PhysicsDomain::ClassicalMechanics,
        "x = A cos(ωt + φ)",
        DimensionalSignature::LENGTH,
    ));
    // Angular momentum: L = Iω — product structure (clusters with p=mv)
    eqs.push(PhysicsEquation {
        name: "Angular Momentum".to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast: make_equals(
            make_const("L"),
            make_product(vec![make_const("I"), make_const("ω")]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature::ACTION,
        tensor: None,
    });
    eqs.push(simple_eq(
        "Torque",
        PhysicsDomain::ClassicalMechanics,
        "τ = r × F",
        DimensionalSignature {
            mass: 1,
            length: 2,
            time: -2,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "Work-Energy Theorem",
        PhysicsDomain::ClassicalMechanics,
        "W = ΔKE",
        DimensionalSignature::ENERGY,
    ));
    eqs.push(simple_eq(
        "Moment of Inertia",
        PhysicsDomain::ClassicalMechanics,
        "I = Σ mᵢrᵢ²",
        DimensionalSignature {
            mass: 1,
            length: 2,
            time: 0,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    // Electromagnetism
    eqs.push(simple_eq(
        "Biot-Savart Law",
        PhysicsDomain::Electromagnetism,
        "dB = (μ₀/4π)(Idl × r̂)/r²",
        DimensionalSignature::MAGNETIC_FIELD,
    ));
    // Larmor radiation: P = q²a²/(6πε₀c³) — power law in charge and acceleration
    eqs.push(PhysicsEquation {
        name: "Larmor Radiation Formula".to_string(),
        domain: PhysicsDomain::Electromagnetism,
        ast: make_equals(
            make_const("P"),
            make_product(vec![
                EquationNode::Power {
                    base: Box::new(make_const("q")),
                    exponent: Box::new(EquationNode::Scalar(2.0)),
                },
                EquationNode::Power {
                    base: Box::new(make_const("a")),
                    exponent: Box::new(EquationNode::Scalar(2.0)),
                },
                EquationNode::Power {
                    base: Box::new(make_product(vec![
                        make_const("6πε₀"),
                        EquationNode::Power {
                            base: Box::new(make_const("c")),
                            exponent: Box::new(EquationNode::Scalar(3.0)),
                        },
                    ])),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::new(vec![LieGroup::U(1)], vec![], true),
        dimensions: DimensionalSignature {
            mass: 1,
            length: 2,
            time: -3,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    });
    eqs.push(simple_eq(
        "Poynting Vector",
        PhysicsDomain::Electromagnetism,
        "S = (1/μ₀)(E × B)",
        DimensionalSignature {
            mass: 1,
            length: 0,
            time: -3,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "Magnetic Dipole Moment",
        PhysicsDomain::Electromagnetism,
        "m = NIA",
        DimensionalSignature {
            mass: 0,
            length: 2,
            time: 0,
            current: 1,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    // Thermodynamics
    // Helmholtz: F = U - TS (same structure as Gibbs G = H - TS)
    eqs.push(PhysicsEquation {
        name: "Helmholtz Free Energy".to_string(),
        domain: PhysicsDomain::Thermodynamics,
        ast: make_equals(
            make_const("F"),
            make_sum(vec![
                make_const("U"),
                EquationNode::Negate(Box::new(make_product(vec![
                    make_const("T"),
                    make_const("S"),
                ]))),
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    });
    eqs.push(simple_eq(
        "Entropy of Mixing",
        PhysicsDomain::Thermodynamics,
        "ΔS_mix = -nR Σ xᵢ ln xᵢ",
        DimensionalSignature {
            mass: 1,
            length: 2,
            time: -2,
            current: 0,
            temperature: -1,
            amount: 0,
            luminous: 0,
        },
    ));
    // Equipartition: ⟨E⟩ = (f/2)kT — proportionality (clusters with ideal gas)
    eqs.push(PhysicsEquation {
        name: "Equipartition Theorem".to_string(),
        domain: PhysicsDomain::Thermodynamics,
        ast: make_equals(
            make_const("⟨E⟩"),
            make_product(vec![
                EquationNode::Scalar(0.5),
                make_const("f"),
                make_const("k_B"),
                make_const("T"),
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    });
    // Carnot: η = 1 - T_cold/T_hot — ratio structure
    eqs.push(PhysicsEquation {
        name: "Carnot Efficiency".to_string(),
        domain: PhysicsDomain::Thermodynamics,
        ast: make_equals(
            make_const("η"),
            make_sum(vec![
                EquationNode::Scalar(1.0),
                EquationNode::Negate(Box::new(make_product(vec![
                    make_const("T_cold"),
                    EquationNode::Power {
                        base: Box::new(make_const("T_hot")),
                        exponent: Box::new(EquationNode::Scalar(-1.0)),
                    },
                ]))),
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    });
    // Quantum Mechanics
    eqs.push(simple_eq(
        "Pauli Exclusion Principle",
        PhysicsDomain::QuantumMechanics,
        "ψ(1,2) = -ψ(2,1)",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "WKB Approximation",
        PhysicsDomain::QuantumMechanics,
        "ψ ≈ exp(±i∫p dx/ℏ)/√p",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Born Approximation",
        PhysicsDomain::QuantumMechanics,
        "f(θ) = -(m/2πℏ²)∫V(r')exp(iq·r')d³r'",
        DimensionalSignature::LENGTH,
    ));
    eqs.push(simple_eq(
        "Time-Independent Perturbation",
        PhysicsDomain::QuantumMechanics,
        "E_n^(1) = ⟨n⁰|V|n⁰⟩",
        DimensionalSignature::ENERGY,
    ));
    // Nuclear/Particle
    eqs.push(simple_eq(
        "Bethe-Bloch Energy Loss",
        PhysicsDomain::NuclearPhysics,
        "-dE/dx = Kz²Z/A·(1/β²)[ln(2meβ²γ²/I) - β²]",
        DimensionalSignature {
            mass: 1,
            length: -1,
            time: -2,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    // Gamow tunneling: T = exp(-2πη) — exponential barrier penetration
    eqs.push(PhysicsEquation {
        name: "Gamow Tunneling Factor".to_string(),
        domain: PhysicsDomain::NuclearPhysics,
        ast: make_equals(
            make_const("T"),
            EquationNode::Exponential(Box::new(EquationNode::Negate(Box::new(make_product(
                vec![make_const("2π"), make_const("η")],
            ))))),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    });
    // NMR: ω₀ = γB₀ — linear proportionality (clusters with Larmor, cyclotron)
    eqs.push(PhysicsEquation {
        name: "Nuclear Magnetic Resonance".to_string(),
        domain: PhysicsDomain::NuclearPhysics,
        ast: make_equals(
            make_const("ω₀"),
            make_product(vec![make_const("γ"), make_const("B₀")]),
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
    });
    // Fluid Dynamics
    // Reynolds: Re = ρvL/μ — dimensionless ratio (clusters with Mach, fissility)
    eqs.push(PhysicsEquation {
        name: "Reynolds Number".to_string(),
        domain: PhysicsDomain::FluidDynamics,
        ast: make_equals(
            make_const("Re"),
            make_product(vec![
                make_const("ρ"),
                make_const("v"),
                make_const("L"),
                EquationNode::Power {
                    base: Box::new(make_const("μ")),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    });
    // Stokes drag: F = 6πμRv — linear force (clusters with Hooke, Ohm)
    eqs.push(PhysicsEquation {
        name: "Stokes Drag".to_string(),
        domain: PhysicsDomain::FluidDynamics,
        ast: make_equals(
            make_const("F"),
            make_product(vec![
                make_const("6π"),
                make_const("μ"),
                make_const("R"),
                make_const("v"),
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::FORCE,
        tensor: None,
    });
    // Poiseuille: Q = πR⁴ΔP/(8μL) — R⁴ power law (flow rate)
    eqs.push(PhysicsEquation {
        name: "Poiseuille Flow".to_string(),
        domain: PhysicsDomain::FluidDynamics,
        ast: make_equals(
            make_const("Q"),
            make_product(vec![
                make_const("π"),
                EquationNode::Power {
                    base: Box::new(make_const("R")),
                    exponent: Box::new(EquationNode::Scalar(4.0)),
                },
                make_const("ΔP"),
                EquationNode::Power {
                    base: Box::new(make_product(vec![
                        EquationNode::Scalar(8.0),
                        make_const("μ"),
                        make_const("L"),
                    ])),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature {
            mass: 0,
            length: 3,
            time: -1,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    });
    // Optics
    eqs.push(simple_eq(
        "Brewster Angle",
        PhysicsDomain::Optics,
        "tan(θ_B) = n₂/n₁",
        DimensionalSignature::DIMENSIONLESS,
    ));
    // Diffraction grating: d sin(θ) = mλ — product (clusters with Bragg, Snell)
    eqs.push(PhysicsEquation {
        name: "Diffraction Grating".to_string(),
        domain: PhysicsDomain::Optics,
        ast: make_equals(
            make_product(vec![make_const("d"), make_const("sin(θ)")]),
            make_product(vec![make_const("m"), make_const("λ")]),
        ),
        symmetries: SymmetryDescriptor::new(vec![], vec![DiscreteSymmetry::T], false),
        dimensions: DimensionalSignature::LENGTH,
        tensor: None,
    });
    eqs.push(simple_eq(
        "Abbe Diffraction Limit",
        PhysicsDomain::Optics,
        "d = λ/(2n sin α)",
        DimensionalSignature::LENGTH,
    ));
    // Cosmology
    eqs.push(simple_eq(
        "Hubble Parameter Evolution",
        PhysicsDomain::Cosmology,
        "H² = H₀²[Ω_r/a⁴ + Ω_m/a³ + Ω_k/a² + Ω_Λ]",
        DimensionalSignature {
            mass: 0,
            length: 0,
            time: -2,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "CMB Temperature Redshift",
        PhysicsDomain::Cosmology,
        "T(z) = T₀(1+z)",
        DimensionalSignature {
            mass: 0,
            length: 0,
            time: 0,
            current: 0,
            temperature: 1,
            amount: 0,
            luminous: 0,
        },
    ));
    // Acoustics (new domain)
    // Speed of sound: v = √(γP/ρ) — sqrt (clusters with Debye length, Alfvén)
    eqs.push(PhysicsEquation {
        name: "Speed of Sound".to_string(),
        domain: PhysicsDomain::Acoustics,
        ast: make_equals(
            make_const("v"),
            EquationNode::Power {
                base: Box::new(make_product(vec![
                    make_const("γ"),
                    make_const("P"),
                    EquationNode::Power {
                        base: Box::new(make_const("ρ")),
                        exponent: Box::new(EquationNode::Scalar(-1.0)),
                    },
                ])),
                exponent: Box::new(EquationNode::Scalar(0.5)),
            },
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::VELOCITY,
        tensor: None,
    });
    eqs.push(simple_eq(
        "Doppler Effect",
        PhysicsDomain::Acoustics,
        "f' = f(v ± v_obs)/(v ∓ v_src)",
        DimensionalSignature {
            mass: 0,
            length: 0,
            time: -1,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "Sound Intensity Level",
        PhysicsDomain::Acoustics,
        "β = 10 log₁₀(I/I₀) dB",
        DimensionalSignature::DIMENSIONLESS,
    ));
    // Plasma Physics (new domain)
    // Debye length: λ_D = √(ε₀kT/(ne²)) — square root structure
    eqs.push(PhysicsEquation {
        name: "Debye Length".to_string(),
        domain: PhysicsDomain::PlasmaPhysics,
        ast: make_equals(
            make_const("λ_D"),
            EquationNode::Power {
                base: Box::new(make_product(vec![
                    make_const("ε₀kT"),
                    EquationNode::Power {
                        base: Box::new(make_product(vec![make_const("n"), make_const("e²")])),
                        exponent: Box::new(EquationNode::Scalar(-1.0)),
                    },
                ])),
                exponent: Box::new(EquationNode::Scalar(0.5)),
            },
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::LENGTH,
        tensor: None,
    });
    // Plasma frequency: ω_p = √(ne²/(mε₀)) — dual of Debye length
    eqs.push(PhysicsEquation {
        name: "Plasma Frequency".to_string(),
        domain: PhysicsDomain::PlasmaPhysics,
        ast: make_equals(
            make_const("ω_p"),
            EquationNode::Power {
                base: Box::new(make_product(vec![
                    make_const("ne²"),
                    EquationNode::Power {
                        base: Box::new(make_product(vec![make_const("m"), make_const("ε₀")])),
                        exponent: Box::new(EquationNode::Scalar(-1.0)),
                    },
                ])),
                exponent: Box::new(EquationNode::Scalar(0.5)),
            },
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
    });
    eqs.push(simple_eq(
        "MHD Force Balance",
        PhysicsDomain::PlasmaPhysics,
        "J × B = ∇P",
        DimensionalSignature {
            mass: 1,
            length: -2,
            time: -2,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    // Information Theory (additional)
    eqs.push(simple_eq(
        "Shannon-Hartley Channel Capacity",
        PhysicsDomain::InformationTheory,
        "C = B log₂(1 + S/N)",
        DimensionalSignature {
            mass: 0,
            length: 0,
            time: -1,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "Data Processing Inequality",
        PhysicsDomain::InformationTheory,
        "I(X;Z) ≤ I(X;Y) for X→Y→Z",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Rate-Distortion Function",
        PhysicsDomain::InformationTheory,
        "R(D) = min I(X;X̂) s.t. E[d(X,X̂)] ≤ D",
        DimensionalSignature::DIMENSIONLESS,
    ));
    // General Relativity (additional)
    eqs.push(simple_eq(
        "Geodesic Equation",
        PhysicsDomain::GeneralRelativity,
        "d²xᵘ/dτ² + Γᵘᵥₛ(dxᵥ/dτ)(dxˢ/dτ) = 0",
        DimensionalSignature::ACCELERATION,
    ));
    eqs.push(simple_eq(
        "Raychaudhuri Equation",
        PhysicsDomain::GeneralRelativity,
        "dθ/dτ = -θ²/3 - σ² + ω² - R_μν uᵘuᵛ",
        DimensionalSignature {
            mass: 0,
            length: 0,
            time: -2,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "Gravitational Wave Strain",
        PhysicsDomain::GeneralRelativity,
        "h = 4GM_c^(5/3)(πf)^(2/3)/(c⁴d)",
        DimensionalSignature::DIMENSIONLESS,
    ));
    // Mathematics (new domain)
    eqs.push(simple_eq(
        "Euler Identity",
        PhysicsDomain::Mathematics,
        "e^(iπ) + 1 = 0",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Fourier Transform",
        PhysicsDomain::Mathematics,
        "F(ω) = ∫f(t)e^(-iωt)dt",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Laplace Transform",
        PhysicsDomain::Mathematics,
        "F(s) = ∫₀^∞ f(t)e^(-st)dt",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Cauchy-Schwarz Inequality",
        PhysicsDomain::Mathematics,
        "|⟨u,v⟩|² ≤ ⟨u,u⟩⟨v,v⟩",
        DimensionalSignature::DIMENSIONLESS,
    ));
    // Condensed Matter (additional)
    eqs.push(simple_eq(
        "Hall Effect",
        PhysicsDomain::CondensedMatter,
        "V_H = IB/(nqt)",
        DimensionalSignature {
            mass: 1,
            length: 2,
            time: -3,
            current: -1,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "Bragg Diffraction",
        PhysicsDomain::CondensedMatter,
        "2d sin(θ) = nλ",
        DimensionalSignature::LENGTH,
    ));
    eqs.push(simple_eq(
        "Debye Model Heat Capacity",
        PhysicsDomain::CondensedMatter,
        "C_V = 9Nk(T/Θ_D)³∫₀^(Θ_D/T) x⁴eˣ/(eˣ-1)²dx",
        DimensionalSignature {
            mass: 1,
            length: 2,
            time: -2,
            current: 0,
            temperature: -1,
            amount: 0,
            luminous: 0,
        },
    ));
    // Biophysics (additional)
    // Michaelis-Menten: v = V_max[S]/(K_m + [S]) — saturation kinetics (clusters with Hill, Langmuir)
    eqs.push(PhysicsEquation {
        name: "Michaelis-Menten Kinetics".to_string(),
        domain: PhysicsDomain::Biophysics,
        ast: make_equals(
            make_const("v"),
            make_product(vec![
                make_const("V_max"),
                make_const("[S]"),
                EquationNode::Power {
                    base: Box::new(make_sum(vec![make_const("K_m"), make_const("[S]")])),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature {
            mass: 0,
            length: 0,
            time: -1,
            current: 0,
            temperature: 0,
            amount: 1,
            luminous: 0,
        },
        tensor: None,
    });
    // Hill: θ = [L]^n/(K_d^n + [L]^n) — cooperative binding (same skeleton as Michaelis-Menten with power)
    eqs.push(PhysicsEquation {
        name: "Hill Equation".to_string(),
        domain: PhysicsDomain::Biophysics,
        ast: make_equals(
            make_const("θ"),
            make_product(vec![
                EquationNode::Power {
                    base: Box::new(make_const("[L]")),
                    exponent: Box::new(make_const("n")),
                },
                EquationNode::Power {
                    base: Box::new(make_sum(vec![
                        EquationNode::Power {
                            base: Box::new(make_const("K_d")),
                            exponent: Box::new(make_const("n")),
                        },
                        EquationNode::Power {
                            base: Box::new(make_const("[L]")),
                            exponent: Box::new(make_const("n")),
                        },
                    ])),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    });
    // Nernst: E = E° - (RT/nF)ln(Q) — logarithmic (clusters with Henderson-Hasselbalch, Shannon-Hartley)
    eqs.push(PhysicsEquation {
        name: "Nernst Equation".to_string(),
        domain: PhysicsDomain::Biophysics,
        ast: make_equals(
            make_const("E"),
            make_sum(vec![
                make_const("E°"),
                EquationNode::Negate(Box::new(make_product(vec![
                    make_const("RT"),
                    EquationNode::Power {
                        base: Box::new(make_product(vec![make_const("n"), make_const("F")])),
                        exponent: Box::new(EquationNode::Scalar(-1.0)),
                    },
                    make_const("ln(Q)"),
                ]))),
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature {
            mass: 1,
            length: 2,
            time: -3,
            current: -1,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    });
    // Chemical
    eqs.push(simple_eq(
        "Henderson-Hasselbalch",
        PhysicsDomain::Thermodynamics,
        "pH = pK_a + log₁₀([A⁻]/[HA])",
        DimensionalSignature::DIMENSIONLESS,
    ));
    // Logistic: dN/dt = rN(1 - N/K) — nonlinear ODE (clusters with Lotka-Volterra)
    eqs.push(PhysicsEquation {
        name: "Logistic Growth".to_string(),
        domain: PhysicsDomain::Biophysics,
        ast: make_equals(
            make_const("dN/dt"),
            make_product(vec![
                make_const("r"),
                make_const("N"),
                make_sum(vec![
                    EquationNode::Scalar(1.0),
                    EquationNode::Negate(Box::new(make_product(vec![
                        make_const("N"),
                        EquationNode::Power {
                            base: Box::new(make_const("K")),
                            exponent: Box::new(EquationNode::Scalar(-1.0)),
                        },
                    ]))),
                ]),
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
    });
    // Statistical Mechanics (additional)
    // Fermi-Dirac: f(E) = 1/(exp((E-μ)/kT) + 1) — full AST for exponential clustering
    eqs.push(PhysicsEquation {
        name: "Fermi-Dirac Distribution".to_string(),
        domain: PhysicsDomain::StatisticalMechanics,
        ast: make_equals(
            make_const("f(E)"),
            EquationNode::Power {
                base: Box::new(make_sum(vec![
                    EquationNode::Exponential(Box::new(make_product(vec![
                        make_sum(vec![
                            make_const("E"),
                            EquationNode::Negate(Box::new(make_const("μ"))),
                        ]),
                        EquationNode::Power {
                            base: Box::new(make_const("kT")),
                            exponent: Box::new(EquationNode::Scalar(-1.0)),
                        },
                    ]))),
                    EquationNode::Scalar(1.0),
                ])),
                exponent: Box::new(EquationNode::Scalar(-1.0)),
            },
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    });
    // Bose-Einstein: n(E) = 1/(exp((E-μ)/kT) - 1) — same skeleton as Fermi-Dirac
    eqs.push(PhysicsEquation {
        name: "Bose-Einstein Distribution".to_string(),
        domain: PhysicsDomain::StatisticalMechanics,
        ast: make_equals(
            make_const("n(E)"),
            EquationNode::Power {
                base: Box::new(make_sum(vec![
                    EquationNode::Exponential(Box::new(make_product(vec![
                        make_sum(vec![
                            make_const("E"),
                            EquationNode::Negate(Box::new(make_const("μ"))),
                        ]),
                        EquationNode::Power {
                            base: Box::new(make_const("kT")),
                            exponent: Box::new(EquationNode::Scalar(-1.0)),
                        },
                    ]))),
                    EquationNode::Scalar(-1.0),
                ])),
                exponent: Box::new(EquationNode::Scalar(-1.0)),
            },
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    });
    // Saha: exponential + power law (should cluster with Boltzmann)
    eqs.push(PhysicsEquation {
        name: "Saha Ionization Equation".to_string(),
        domain: PhysicsDomain::StatisticalMechanics,
        ast: make_equals(
            make_product(vec![
                make_const("n_i"),
                make_const("n_e"),
                EquationNode::Power {
                    base: Box::new(make_const("n_0")),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
            make_product(vec![
                EquationNode::Power {
                    base: Box::new(make_product(vec![
                        make_const("2πm_ekT"),
                        EquationNode::Power {
                            base: Box::new(make_const("h²")),
                            exponent: Box::new(EquationNode::Scalar(-1.0)),
                        },
                    ])),
                    exponent: Box::new(EquationNode::Scalar(1.5)),
                },
                EquationNode::Exponential(Box::new(EquationNode::Negate(Box::new(make_product(
                    vec![
                        make_const("χ"),
                        EquationNode::Power {
                            base: Box::new(make_const("kT")),
                            exponent: Box::new(EquationNode::Scalar(-1.0)),
                        },
                    ],
                ))))),
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    });

    // ── Phase: Exhaustive Expansion to 210+ ──

    // Fusion / Plasma
    eqs.push(simple_eq(
        "Lawson Criterion",
        PhysicsDomain::PlasmaPhysics,
        "nTτ > 1.5×10²⁰ m⁻³·keV·s",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Grad-Shafranov Equation",
        PhysicsDomain::PlasmaPhysics,
        "R∂/∂R(1/R·∂ψ/∂R) + ∂²ψ/∂Z² = -μ₀R²dp/dψ - F dF/dψ",
        DimensionalSignature {
            mass: 1,
            length: 0,
            time: -2,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "Bremsstrahlung Radiation",
        PhysicsDomain::PlasmaPhysics,
        "P_br ∝ n²Z²T^(1/2)",
        DimensionalSignature {
            mass: 1,
            length: -1,
            time: -3,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    // Alfvén: v_A = B/√(μ₀ρ) — square root (clusters with Debye, plasma freq, speed of sound)
    eqs.push(PhysicsEquation {
        name: "Alfven Wave Speed".to_string(),
        domain: PhysicsDomain::PlasmaPhysics,
        ast: make_equals(
            make_const("v_A"),
            make_product(vec![
                make_const("B"),
                EquationNode::Power {
                    base: Box::new(make_product(vec![make_const("μ₀"), make_const("ρ")])),
                    exponent: Box::new(EquationNode::Scalar(-0.5)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::VELOCITY,
        tensor: None,
    });
    eqs.push(simple_eq(
        "Magnetic Mirror Ratio",
        PhysicsDomain::PlasmaPhysics,
        "R = B_max/B_min",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Spitzer Resistivity",
        PhysicsDomain::PlasmaPhysics,
        "η ∝ Z·ln(Λ)/T^(3/2)",
        DimensionalSignature {
            mass: 1,
            length: 3,
            time: -3,
            current: -2,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "Cyclotron Frequency",
        PhysicsDomain::PlasmaPhysics,
        "ω_c = qB/m",
        DimensionalSignature {
            mass: 0,
            length: 0,
            time: -1,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    // Larmor radius: r_L = mv⊥/(qB) — ratio structure
    eqs.push(PhysicsEquation {
        name: "Larmor Radius".to_string(),
        domain: PhysicsDomain::PlasmaPhysics,
        ast: make_equals(
            make_const("r_L"),
            make_product(vec![
                make_const("m"),
                make_const("v⊥"),
                EquationNode::Power {
                    base: Box::new(make_product(vec![make_const("q"), make_const("B")])),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::LENGTH,
        tensor: None,
    });

    // Superconductors / Condensed Matter
    eqs.push(simple_eq(
        "Ginzburg-Landau Equation",
        PhysicsDomain::CondensedMatter,
        "F = α|ψ|² + β|ψ|⁴ + |(-iℏ∇ - 2eA)ψ|²/(2m*)",
        DimensionalSignature::ENERGY_DENSITY,
    ));
    eqs.push(simple_eq(
        "London Penetration Depth",
        PhysicsDomain::CondensedMatter,
        "λ_L = √(m/(μ₀ne²))",
        DimensionalSignature::LENGTH,
    ));
    eqs.push(simple_eq(
        "Eliashberg Equation",
        PhysicsDomain::CondensedMatter,
        "Δ(iω_n) = πT Σ_m λ(ω_n-ω_m)Δ(iω_m)/√(ω_m² + Δ²(iω_m))",
        DimensionalSignature::ENERGY,
    ));
    eqs.push(simple_eq(
        "Josephson Effect",
        PhysicsDomain::CondensedMatter,
        "I = I_c sin(Δφ)",
        DimensionalSignature {
            mass: 0,
            length: 0,
            time: 0,
            current: 1,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "Curie-Weiss Law",
        PhysicsDomain::CondensedMatter,
        "χ = C/(T - T_c)",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Meissner Effect",
        PhysicsDomain::CondensedMatter,
        "B_internal = 0 (perfect diamagnetism)",
        DimensionalSignature::MAGNETIC_FIELD,
    ));
    eqs.push(simple_eq(
        "Anderson Localization",
        PhysicsDomain::CondensedMatter,
        "ψ(r) ~ exp(-|r-r₀|/ξ)",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Wiedemann-Franz Law",
        PhysicsDomain::CondensedMatter,
        "κ/(σT) = L₀ = π²k_B²/(3e²)",
        DimensionalSignature::DIMENSIONLESS,
    ));

    // Materials / Chemistry
    eqs.push(simple_eq(
        "Kohn-Sham DFT",
        PhysicsDomain::CondensedMatter,
        "[-ℏ²∇²/(2m) + V_eff(r)]φ_i(r) = ε_iφ_i(r)",
        DimensionalSignature::ENERGY,
    ));
    // Lennard-Jones: V(r) = 4ε[(σ/r)^12 - (σ/r)^6] — full AST for 1/r clustering with Yukawa
    eqs.push(PhysicsEquation {
        name: "Lennard-Jones Potential".to_string(),
        domain: PhysicsDomain::Thermodynamics,
        ast: make_equals(
            make_const("V(r)"),
            make_product(vec![
                make_const("4ε"),
                make_sum(vec![
                    EquationNode::Power {
                        base: Box::new(make_product(vec![
                            make_const("σ"),
                            EquationNode::Power {
                                base: Box::new(make_const("r")),
                                exponent: Box::new(EquationNode::Scalar(-1.0)),
                            },
                        ])),
                        exponent: Box::new(EquationNode::Scalar(12.0)),
                    },
                    EquationNode::Negate(Box::new(EquationNode::Power {
                        base: Box::new(make_product(vec![
                            make_const("σ"),
                            EquationNode::Power {
                                base: Box::new(make_const("r")),
                                exponent: Box::new(EquationNode::Scalar(-1.0)),
                            },
                        ])),
                        exponent: Box::new(EquationNode::Scalar(6.0)),
                    })),
                ]),
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    });
    eqs.push(simple_eq(
        "Born-Oppenheimer Approximation",
        PhysicsDomain::QuantumMechanics,
        "Ψ(r,R) = ψ_e(r;R)·χ_n(R)",
        DimensionalSignature::DIMENSIONLESS,
    ));
    // Fick I: J = -D∇c — gradient-driven flux (clusters with Fourier heat, Ohm)
    {
        let euc3 = MetricSignature::Euclidean(3);
        eqs.push(PhysicsEquation {
            name: "Fick First Law".to_string(),
            domain: PhysicsDomain::Thermodynamics,
            ast: make_equals(
                make_field("J", TensorDescriptor::vector(euc3)),
                EquationNode::Negate(Box::new(make_product(vec![
                    make_const("D"),
                    make_diffop(
                        DiffOperator::Gradient,
                        make_field("c", TensorDescriptor::scalar(euc3)),
                    ),
                ]))),
            ),
            symmetries: SymmetryDescriptor::none(),
            dimensions: DimensionalSignature {
                mass: 0,
                length: -2,
                time: -1,
                current: 0,
                temperature: 0,
                amount: 1,
                luminous: 0,
            },
            tensor: Some(TensorDescriptor::vector(euc3)),
        });
    }
    // Fick II: ∂c/∂t = D∇²c — diffusion equation (clusters with Heat Equation)
    {
        let euc3 = MetricSignature::Euclidean(3);
        eqs.push(PhysicsEquation {
            name: "Fick Second Law".to_string(),
            domain: PhysicsDomain::Thermodynamics,
            ast: make_equals(
                make_diffop(
                    DiffOperator::TimeDerivative,
                    make_field("c", TensorDescriptor::scalar(euc3)),
                ),
                make_product(vec![
                    make_const("D"),
                    make_diffop(
                        DiffOperator::Laplacian,
                        make_field("c", TensorDescriptor::scalar(euc3)),
                    ),
                ]),
            ),
            symmetries: SymmetryDescriptor::none(),
            dimensions: DimensionalSignature {
                mass: 0,
                length: -3,
                time: -1,
                current: 0,
                temperature: 0,
                amount: 1,
                luminous: 0,
            },
            tensor: None,
        });
    }
    eqs.push(simple_eq(
        "Beer-Lambert Law",
        PhysicsDomain::Optics,
        "A = εlc",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Vegard Law",
        PhysicsDomain::CondensedMatter,
        "a(x) = (1-x)a_A + x·a_B",
        DimensionalSignature::LENGTH,
    ));
    eqs.push(simple_eq(
        "Hess Law",
        PhysicsDomain::Thermodynamics,
        "ΔH_rxn = Σ ΔH_f(products) - Σ ΔH_f(reactants)",
        DimensionalSignature::ENERGY,
    ));

    // Quantum Information
    eqs.push(simple_eq(
        "Von Neumann Entropy",
        PhysicsDomain::InformationTheory,
        "S = -Tr(ρ ln ρ)",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Quantum Fidelity",
        PhysicsDomain::InformationTheory,
        "F(ρ,σ) = (Tr√(√ρ σ √ρ))²",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Lindblad Master Equation",
        PhysicsDomain::QuantumMechanics,
        "dρ/dt = -i[H,ρ]/ℏ + Σ(LρL† - {L†L,ρ}/2)",
        DimensionalSignature {
            mass: 0,
            length: 0,
            time: -1,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "Bell CHSH Inequality",
        PhysicsDomain::InformationTheory,
        "|S| ≤ 2 (classical), ≤ 2√2 (quantum)",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "No-Cloning Theorem",
        PhysicsDomain::InformationTheory,
        "∄ U: |ψ⟩|0⟩ → |ψ⟩|ψ⟩ ∀|ψ⟩",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Rabi Oscillation",
        PhysicsDomain::QuantumMechanics,
        "P(t) = sin²(Ωt/2)",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Jaynes-Cummings Model",
        PhysicsDomain::QuantumMechanics,
        "H = ℏω_a a†a + ℏω_b σ_z/2 + ℏg(a†σ₋ + aσ₊)",
        DimensionalSignature::ENERGY,
    ));
    eqs.push(simple_eq(
        "Bloch Sphere",
        PhysicsDomain::QuantumMechanics,
        "|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩",
        DimensionalSignature::DIMENSIONLESS,
    ));

    // Climate / Atmosphere
    eqs.push(simple_eq(
        "Radiative Transfer Equation",
        PhysicsDomain::Thermodynamics,
        "dI/ds = -κI + j",
        DimensionalSignature {
            mass: 1,
            length: 0,
            time: -3,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "Greenhouse Radiative Forcing",
        PhysicsDomain::Thermodynamics,
        "ΔF = 5.35 ln(C/C₀) W/m²",
        DimensionalSignature {
            mass: 1,
            length: 0,
            time: -3,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "Clausius-Mossotti Relation",
        PhysicsDomain::Electromagnetism,
        "(ε-1)/(ε+2) = nα/(3ε₀)",
        DimensionalSignature::DIMENSIONLESS,
    ));

    // Longevity / Biology
    // Gompertz: μ(x) = a·exp(bx) — exponential growth (clusters with Arrhenius, Boltzmann)
    eqs.push(PhysicsEquation {
        name: "Gompertz Mortality Law".to_string(),
        domain: PhysicsDomain::Biophysics,
        ast: make_equals(
            make_const("μ(x)"),
            make_product(vec![
                make_const("a"),
                EquationNode::Exponential(Box::new(make_product(vec![
                    make_const("b"),
                    make_const("x"),
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
    });
    eqs.push(simple_eq(
        "Goldman-Hodgkin-Katz Voltage",
        PhysicsDomain::Biophysics,
        "V = (RT/F)ln((P_K[K⁺]_o + P_Na[Na⁺]_o)/(P_K[K⁺]_i + P_Na[Na⁺]_i))",
        DimensionalSignature {
            mass: 1,
            length: 2,
            time: -3,
            current: -1,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    // Monod: μ = μ_max·S/(K_s + S) — same skeleton as Michaelis-Menten
    eqs.push(PhysicsEquation {
        name: "Monod Equation".to_string(),
        domain: PhysicsDomain::Biophysics,
        ast: make_equals(
            make_const("μ"),
            make_product(vec![
                make_const("μ_max"),
                make_const("S"),
                EquationNode::Power {
                    base: Box::new(make_sum(vec![make_const("K_s"), make_const("S")])),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
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
    });

    // Astrophysics
    eqs.push(simple_eq(
        "Chandrasekhar Mass Limit",
        PhysicsDomain::Astrophysics,
        "M_Ch ≈ 1.4 M_☉",
        DimensionalSignature::MASS,
    ));
    eqs.push(simple_eq(
        "Eddington Luminosity",
        PhysicsDomain::Astrophysics,
        "L_Edd = 4πGMc/κ",
        DimensionalSignature {
            mass: 1,
            length: 2,
            time: -3,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "Tolman-Oppenheimer-Volkoff Equation",
        PhysicsDomain::Astrophysics,
        "dP/dr = -(ρ+P/c²)(m+4πr³P/c²)G/(r(r-2Gm/c²))",
        DimensionalSignature {
            mass: 1,
            length: -1,
            time: -2,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "Jeans Instability",
        PhysicsDomain::Astrophysics,
        "λ_J = c_s√(π/(Gρ))",
        DimensionalSignature::LENGTH,
    ));
    eqs.push(simple_eq(
        "Tsiolkovsky Rocket Equation",
        PhysicsDomain::Astrophysics,
        "Δv = v_e ln(m₀/m_f)",
        DimensionalSignature::VELOCITY,
    ));
    eqs.push(simple_eq(
        "Drake Equation",
        PhysicsDomain::Astrophysics,
        "N = R* × f_p × n_e × f_l × f_i × f_c × L",
        DimensionalSignature::DIMENSIONLESS,
    ));

    // Nonequilibrium / Statistical
    eqs.push(simple_eq(
        "Langevin Equation",
        PhysicsDomain::StatisticalMechanics,
        "m dv/dt = -γv + F(t) + η(t)",
        DimensionalSignature::FORCE,
    ));
    eqs.push(simple_eq(
        "Fokker-Planck Equation",
        PhysicsDomain::StatisticalMechanics,
        "∂P/∂t = -∂(μP)/∂x + ∂²(DP)/∂x²",
        DimensionalSignature {
            mass: 0,
            length: 0,
            time: -1,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "Boltzmann Transport Equation",
        PhysicsDomain::StatisticalMechanics,
        "∂f/∂t + v·∇f + F·∇_p f = (∂f/∂t)_coll",
        DimensionalSignature {
            mass: 0,
            length: 0,
            time: -1,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
    ));
    eqs.push(simple_eq(
        "Fluctuation-Dissipation Theorem",
        PhysicsDomain::StatisticalMechanics,
        "S(ω) = 2k_BT·Re[χ(ω)]/ω",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Jarzynski Equality",
        PhysicsDomain::StatisticalMechanics,
        "⟨e^(-W/kT)⟩ = e^(-ΔF/kT)",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Crooks Fluctuation Theorem",
        PhysicsDomain::StatisticalMechanics,
        "P_F(W)/P_R(-W) = e^((W-ΔF)/kT)",
        DimensionalSignature::DIMENSIONLESS,
    ));

    // Mathematical Physics
    eqs.push(simple_eq(
        "Noether Theorem",
        PhysicsDomain::Mathematics,
        "continuous symmetry ↔ conservation law",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Green Function",
        PhysicsDomain::Mathematics,
        "LG(x,x') = δ(x-x')",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Legendre Transform",
        PhysicsDomain::Mathematics,
        "f*(p) = sup_x(px - f(x))",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Virial Theorem",
        PhysicsDomain::ClassicalMechanics,
        "2⟨KE⟩ = -⟨r·∇V⟩",
        DimensionalSignature::ENERGY,
    ));
    eqs.push(simple_eq(
        "Pendulum Period",
        PhysicsDomain::ClassicalMechanics,
        "T = 2π√(L/g)",
        DimensionalSignature::TIME,
    ));
    eqs.push(simple_eq(
        "Gravitational Potential Energy",
        PhysicsDomain::ClassicalMechanics,
        "U = -GMm/r",
        DimensionalSignature::ENERGY,
    ));

    // QFT / Symmetry
    eqs.push(simple_eq(
        "Goldstone Theorem",
        PhysicsDomain::QuantumFieldTheory,
        "spontaneous symmetry breaking → massless boson",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Anomalous Magnetic Moment",
        PhysicsDomain::ParticlePhysics,
        "a = (g-2)/2 ≈ α/(2π) + ...",
        DimensionalSignature::DIMENSIONLESS,
    ));
    eqs.push(simple_eq(
        "Lamb Shift",
        PhysicsDomain::ParticlePhysics,
        "ΔE(2S₁/₂ - 2P₁/₂) ≈ 1057 MHz",
        DimensionalSignature::ENERGY,
    ));
    eqs.push(simple_eq(
        "Weinberg Angle",
        PhysicsDomain::ParticlePhysics,
        "sin²θ_W ≈ 0.231",
        DimensionalSignature::DIMENSIONLESS,
    ));

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
                EquationNode::Power {
                    base: Box::new(make_product(vec![
                        EquationNode::Scalar(8.0),
                        EquationNode::Power {
                            base: Box::new(make_product(vec![make_const("π"), make_const("μ")])),
                            exponent: Box::new(EquationNode::Scalar(-1.0)),
                        },
                    ])),
                    exponent: Box::new(EquationNode::Scalar(0.5)),
                },
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
                EquationNode::Exponential(Box::new(EquationNode::Negate(Box::new(make_product(
                    vec![
                        make_const("μ"),
                        make_field("r", TensorDescriptor::scalar(euc3)),
                    ],
                ))))),
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
                EquationNode::Exponential(Box::new(EquationNode::Negate(Box::new(make_product(
                    vec![
                        make_const("m_π"),
                        make_field("r", TensorDescriptor::scalar(euc3)),
                    ],
                ))))),
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
            make_diffop(DiffOperator::TimeDerivative, make_const("N_i")),
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
            make_product(vec![make_const("8πT/(3+2ω)")]),
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
            vec![
                DiscreteSymmetry::C,
                DiscreteSymmetry::P,
                DiscreteSymmetry::T,
            ],
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
            EquationNode::Exponential(Box::new(EquationNode::Negate(Box::new(make_product(
                vec![
                    make_const("E"),
                    EquationNode::Power {
                        base: Box::new(make_const("kT")),
                        exponent: Box::new(EquationNode::Scalar(-1.0)),
                    },
                ],
            ))))),
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
        ast: make_equals(make_const("Z"), make_const("Σ_n exp(-βE_n)")),
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
                        make_product(vec![make_const("b₀·α_s/(2π)"), make_const("ln(Q²/M_Z²)")]),
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
        symmetries: SymmetryDescriptor::new(vec![LieGroup::U(1)], vec![DiscreteSymmetry::T], false),
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
                        base: Box::new(make_product(vec![make_const("N(0)"), make_const("V")])),
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
        symmetries: SymmetryDescriptor::new(vec![], vec![DiscreteSymmetry::T], false),
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
        symmetries: SymmetryDescriptor::new(vec![LieGroup::U(1)], vec![DiscreteSymmetry::T], false),
        dimensions: DimensionalSignature {
            mass: -1,
            length: -3,
            time: 3,
            current: 2,
            temperature: 0,
            amount: 0,
            luminous: 0,
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
            mass: 0,
            length: 0,
            time: 0,
            current: 0,
            temperature: 1,
            amount: 0,
            luminous: 0,
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
            mass: 1,
            length: 0,
            time: -3,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// PHASE A1: FOUNDATIONAL EQUATIONS
// ══════════════════════════════════════════════════════════════════════════════

/// F = ma — Newton's second law of motion
fn newton_second_law() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "Newton Second Law".to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast: make_equals(
            make_field("F", TensorDescriptor::vector(euc3)),
            make_product(vec![
                make_const("m"),
                make_field("a", TensorDescriptor::vector(euc3)),
            ]),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::SO(3)],
            vec![DiscreteSymmetry::T],
            false,
        ),
        dimensions: DimensionalSignature::FORCE,
        tensor: Some(TensorDescriptor::vector(euc3)),
    }
}

/// F = GMm/r² — Newton's law of universal gravitation
fn newton_gravitation() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "Newton Gravitation".to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast: make_equals(
            make_const("F"),
            make_product(vec![
                make_const("G"),
                make_const("M"),
                make_const("m"),
                EquationNode::Power {
                    base: Box::new(make_const("r")),
                    exponent: Box::new(EquationNode::Scalar(-2.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature::FORCE,
        tensor: None,
    }
}

/// T² = (4π²/GM)a³ — Kepler's third law
fn kepler_third_law() -> PhysicsEquation {
    PhysicsEquation {
        name: "Kepler Third Law".to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast: make_equals(
            EquationNode::Power {
                base: Box::new(make_const("T")),
                exponent: Box::new(EquationNode::Scalar(2.0)),
            },
            make_product(vec![
                make_const("4π²/GM"),
                EquationNode::Power {
                    base: Box::new(make_const("a")),
                    exponent: Box::new(EquationNode::Scalar(3.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature {
            mass: 0,
            length: 0,
            time: 2,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    }
}

/// F = -kx — Hooke's law
fn hooke_law() -> PhysicsEquation {
    PhysicsEquation {
        name: "Hooke Law".to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast: make_equals(
            make_const("F"),
            EquationNode::Negate(Box::new(make_product(vec![
                make_const("k"),
                make_const("x"),
            ]))),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::FORCE,
        tensor: None,
    }
}

/// a = v²/r — Centripetal acceleration
fn centripetal_acceleration() -> PhysicsEquation {
    PhysicsEquation {
        name: "Centripetal Acceleration".to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast: make_equals(
            make_const("a"),
            make_product(vec![
                EquationNode::Power {
                    base: Box::new(make_const("v")),
                    exponent: Box::new(EquationNode::Scalar(2.0)),
                },
                EquationNode::Power {
                    base: Box::new(make_const("r")),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(2)], false),
        dimensions: DimensionalSignature::ACCELERATION,
        tensor: None,
    }
}

/// F = kq₁q₂/r² — Coulomb's law
fn coulomb_law() -> PhysicsEquation {
    PhysicsEquation {
        name: "Coulomb Law".to_string(),
        domain: PhysicsDomain::Electromagnetism,
        ast: make_equals(
            make_const("F"),
            make_product(vec![
                make_const("k_e"),
                make_const("q₁"),
                make_const("q₂"),
                EquationNode::Power {
                    base: Box::new(make_const("r")),
                    exponent: Box::new(EquationNode::Scalar(-2.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::new(
            vec![LieGroup::U(1), LieGroup::SO(3)],
            vec![
                DiscreteSymmetry::C,
                DiscreteSymmetry::P,
                DiscreteSymmetry::T,
            ],
            true,
        ),
        dimensions: DimensionalSignature::FORCE,
        tensor: None,
    }
}

/// F = q(E + v × B) — Lorentz force law
fn lorentz_force() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "Lorentz Force".to_string(),
        domain: PhysicsDomain::Electromagnetism,
        ast: make_equals(
            make_field("F", TensorDescriptor::vector(euc3)),
            make_product(vec![
                make_const("q"),
                make_sum(vec![
                    make_field("E", TensorDescriptor::vector(euc3)),
                    make_product(vec![make_const("v×B")]),
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
        dimensions: DimensionalSignature::FORCE,
        tensor: Some(TensorDescriptor::vector(euc3)),
    }
}

/// V = IR — Ohm's law
fn ohm_law() -> PhysicsEquation {
    PhysicsEquation {
        name: "Ohm Law".to_string(),
        domain: PhysicsDomain::Electromagnetism,
        ast: make_equals(
            make_const("V"),
            make_product(vec![make_const("I"), make_const("R")]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature {
            mass: 1,
            length: 2,
            time: -3,
            current: -1,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    }
}

/// E = hf — Planck-Einstein relation
fn planck_einstein_relation() -> PhysicsEquation {
    PhysicsEquation {
        name: "Planck-Einstein Relation".to_string(),
        domain: PhysicsDomain::QuantumMechanics,
        ast: make_equals(
            make_const("E"),
            make_product(vec![make_const("h"), make_const("f")]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::U(1)], false),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    }
}

/// λ = h/p — de Broglie wavelength
fn de_broglie_wavelength() -> PhysicsEquation {
    PhysicsEquation {
        name: "De Broglie Wavelength".to_string(),
        domain: PhysicsDomain::QuantumMechanics,
        ast: make_equals(
            make_const("λ"),
            make_product(vec![
                make_const("h"),
                EquationNode::Power {
                    base: Box::new(make_const("p")),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::U(1)], false),
        dimensions: DimensionalSignature::LENGTH,
        tensor: None,
    }
}

/// ΔxΔp ≥ ℏ/2 — Heisenberg uncertainty principle
fn heisenberg_uncertainty() -> PhysicsEquation {
    PhysicsEquation {
        name: "Heisenberg Uncertainty Principle".to_string(),
        domain: PhysicsDomain::QuantumMechanics,
        ast: make_equals(
            make_product(vec![make_const("Δx"), make_const("Δp")]),
            make_const("ℏ/2"),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::ACTION,
        tensor: None,
    }
}

/// KE = hf - φ — Photoelectric equation
fn photoelectric_equation() -> PhysicsEquation {
    PhysicsEquation {
        name: "Photoelectric Equation".to_string(),
        domain: PhysicsDomain::QuantumMechanics,
        ast: make_equals(
            make_const("KE_max"),
            make_sum(vec![
                make_product(vec![make_const("h"), make_const("f")]),
                EquationNode::Negate(Box::new(make_const("φ"))),
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    }
}

/// E = mc² — Mass-energy equivalence
fn mass_energy_equivalence() -> PhysicsEquation {
    PhysicsEquation {
        name: "Mass-Energy Equivalence".to_string(),
        domain: PhysicsDomain::GeneralRelativity,
        ast: make_equals(
            make_const("E"),
            make_product(vec![
                make_const("m"),
                EquationNode::Power {
                    base: Box::new(make_const("c")),
                    exponent: Box::new(EquationNode::Scalar(2.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::Lorentz], false),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    }
}

/// γ = 1/√(1 - v²/c²) — Lorentz factor
fn lorentz_factor() -> PhysicsEquation {
    PhysicsEquation {
        name: "Lorentz Factor".to_string(),
        domain: PhysicsDomain::GeneralRelativity,
        ast: make_equals(
            make_const("γ"),
            EquationNode::Power {
                base: Box::new(make_sum(vec![
                    EquationNode::Scalar(1.0),
                    EquationNode::Negate(Box::new(make_product(vec![
                        EquationNode::Power {
                            base: Box::new(make_const("v")),
                            exponent: Box::new(EquationNode::Scalar(2.0)),
                        },
                        EquationNode::Power {
                            base: Box::new(make_const("c")),
                            exponent: Box::new(EquationNode::Scalar(-2.0)),
                        },
                    ]))),
                ])),
                exponent: Box::new(EquationNode::Scalar(-0.5)),
            },
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::Lorentz], false),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// B(ν,T) = (2hν³/c²)/(exp(hν/kT) - 1) — Planck radiation law
fn planck_radiation_law() -> PhysicsEquation {
    PhysicsEquation {
        name: "Planck Radiation Law".to_string(),
        domain: PhysicsDomain::QuantumMechanics,
        ast: make_equals(
            make_const("B(ν,T)"),
            make_product(vec![
                make_const("2hν³/c²"),
                EquationNode::Power {
                    base: Box::new(make_sum(vec![
                        EquationNode::Exponential(Box::new(make_const("hν/kT"))),
                        EquationNode::Scalar(-1.0),
                    ])),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature {
            mass: 1,
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

/// λ_max = b/T — Wien's displacement law
fn wien_displacement() -> PhysicsEquation {
    PhysicsEquation {
        name: "Wien Displacement Law".to_string(),
        domain: PhysicsDomain::Thermodynamics,
        ast: make_equals(
            make_const("λ_max"),
            make_product(vec![
                make_const("b"),
                EquationNode::Power {
                    base: Box::new(make_const("T")),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::LENGTH,
        tensor: None,
    }
}

/// I ∝ 1/λ⁴ — Rayleigh scattering
fn rayleigh_scattering() -> PhysicsEquation {
    PhysicsEquation {
        name: "Rayleigh Scattering".to_string(),
        domain: PhysicsDomain::Optics,
        ast: make_equals(
            make_const("I"),
            make_product(vec![
                make_const("I₀"),
                EquationNode::Power {
                    base: Box::new(make_const("λ")),
                    exponent: Box::new(EquationNode::Scalar(-4.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// PHASE A2: INFORMATION THEORY
// ══════════════════════════════════════════════════════════════════════════════

/// H(X) = -Σ p(x) log p(x) — Shannon entropy
fn shannon_entropy() -> PhysicsEquation {
    PhysicsEquation {
        name: "Shannon Entropy".to_string(),
        domain: PhysicsDomain::InformationTheory,
        ast: make_equals(
            make_const("H(X)"),
            EquationNode::Negate(Box::new(make_const("Σ p(x) log p(x)"))),
        ),
        symmetries: SymmetryDescriptor::new(vec![], vec![DiscreteSymmetry::Permutation(0)], false),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// D_KL(P||Q) = Σ P ln(P/Q) — Kullback-Leibler divergence
fn kl_divergence() -> PhysicsEquation {
    PhysicsEquation {
        name: "KL Divergence".to_string(),
        domain: PhysicsDomain::InformationTheory,
        ast: make_equals(make_const("D_KL(P||Q)"), make_const("Σ P ln(P/Q)")),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// I(X;Y) = H(X) - H(X|Y) — Mutual information
fn mutual_information() -> PhysicsEquation {
    PhysicsEquation {
        name: "Mutual Information".to_string(),
        domain: PhysicsDomain::InformationTheory,
        ast: make_equals(
            make_const("I(X;Y)"),
            make_sum(vec![
                make_const("H(X)"),
                EquationNode::Negate(Box::new(make_const("H(X|Y)"))),
            ]),
        ),
        symmetries: SymmetryDescriptor::new(vec![], vec![DiscreteSymmetry::Permutation(2)], false),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// H(P,Q) = -Σ P ln Q — Cross-entropy
fn cross_entropy() -> PhysicsEquation {
    PhysicsEquation {
        name: "Cross-Entropy".to_string(),
        domain: PhysicsDomain::InformationTheory,
        ast: make_equals(
            make_const("H(P,Q)"),
            EquationNode::Negate(Box::new(make_const("Σ P ln Q"))),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// I(θ) = E[(∂/∂θ ln f)²] — Fisher information
fn fisher_information() -> PhysicsEquation {
    PhysicsEquation {
        name: "Fisher Information".to_string(),
        domain: PhysicsDomain::InformationTheory,
        ast: make_equals(make_const("I(θ)"), make_const("E[(∂ ln f/∂θ)²]")),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// F = E_q[ln q(z) - ln p(o,z)] — Friston's variational free energy
fn variational_free_energy() -> PhysicsEquation {
    PhysicsEquation {
        name: "Variational Free Energy (Friston)".to_string(),
        domain: PhysicsDomain::InformationTheory,
        ast: make_equals(make_const("F"), make_const("E_q[ln q(z) - ln p(o,z)]")),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// Φ = min_partition D_KL(p(X) || Π p(X_i)) — Integrated Information
fn integrated_information_phi() -> PhysicsEquation {
    PhysicsEquation {
        name: "Integrated Information (Phi)".to_string(),
        domain: PhysicsDomain::InformationTheory,
        ast: make_equals(
            make_const("Φ"),
            make_const("min_partition D_KL(p || Π p_i)"),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// S = k_B ln W — Boltzmann entropy
fn boltzmann_entropy() -> PhysicsEquation {
    PhysicsEquation {
        name: "Boltzmann Entropy".to_string(),
        domain: PhysicsDomain::StatisticalMechanics,
        ast: make_equals(
            make_const("S"),
            make_product(vec![make_const("k_B"), make_const("ln(W)")]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature {
            mass: 1,
            length: 2,
            time: -2,
            current: 0,
            temperature: -1,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    }
}

/// S = kA/(4ℓ_P²) — Bekenstein-Hawking entropy
fn bekenstein_hawking_entropy() -> PhysicsEquation {
    PhysicsEquation {
        name: "Bekenstein-Hawking Entropy".to_string(),
        domain: PhysicsDomain::GeneralRelativity,
        ast: make_equals(
            make_const("S_BH"),
            make_product(vec![make_const("k_B·c³/(4Gℏ)"), make_const("A")]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature {
            mass: 1,
            length: 2,
            time: -2,
            current: 0,
            temperature: -1,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// PHASE A3: NAMED EQUATIONS + BIOPHYSICS
// ══════════════════════════════════════════════════════════════════════════════

/// P + ½ρv² + ρgh = const — Bernoulli's equation
fn bernoulli_equation() -> PhysicsEquation {
    PhysicsEquation {
        name: "Bernoulli Equation".to_string(),
        domain: PhysicsDomain::FluidDynamics,
        ast: make_equals(
            make_sum(vec![
                make_const("P"),
                make_product(vec![
                    EquationNode::Scalar(0.5),
                    make_const("ρ"),
                    EquationNode::Power {
                        base: Box::new(make_const("v")),
                        exponent: Box::new(EquationNode::Scalar(2.0)),
                    },
                ]),
                make_product(vec![make_const("ρ"), make_const("g"), make_const("h")]),
            ]),
            make_const("constant"),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::PRESSURE,
        tensor: None,
    }
}

/// dP/dT = ΔH/(TΔV) — Clausius-Clapeyron
fn clausius_clapeyron() -> PhysicsEquation {
    PhysicsEquation {
        name: "Clausius-Clapeyron Equation".to_string(),
        domain: PhysicsDomain::Thermodynamics,
        ast: make_equals(
            make_const("dP/dT"),
            make_product(vec![
                make_const("ΔH"),
                EquationNode::Power {
                    base: Box::new(make_product(vec![make_const("T"), make_const("ΔV")])),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature {
            mass: 1,
            length: -1,
            time: -2,
            current: 0,
            temperature: -1,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    }
}

/// G = H - TS — Gibbs free energy
fn gibbs_free_energy() -> PhysicsEquation {
    PhysicsEquation {
        name: "Gibbs Free Energy".to_string(),
        domain: PhysicsDomain::Thermodynamics,
        ast: make_equals(
            make_const("G"),
            make_sum(vec![
                make_const("H"),
                EquationNode::Negate(Box::new(make_product(vec![
                    make_const("T"),
                    make_const("S"),
                ]))),
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    }
}

/// (P + a/V²)(V - b) = RT — Van der Waals equation
fn van_der_waals() -> PhysicsEquation {
    PhysicsEquation {
        name: "Van der Waals Equation".to_string(),
        domain: PhysicsDomain::Thermodynamics,
        ast: make_equals(
            make_product(vec![
                make_sum(vec![
                    make_const("P"),
                    make_product(vec![
                        make_const("a"),
                        EquationNode::Power {
                            base: Box::new(make_const("V")),
                            exponent: Box::new(EquationNode::Scalar(-2.0)),
                        },
                    ]),
                ]),
                make_sum(vec![
                    make_const("V"),
                    EquationNode::Negate(Box::new(make_const("b"))),
                ]),
            ]),
            make_product(vec![make_const("R"), make_const("T")]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    }
}

/// ∂ρ/∂t + ∇·(ρv) = 0 — Continuity equation
fn continuity_equation() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "Continuity Equation".to_string(),
        domain: PhysicsDomain::FluidDynamics,
        ast: make_equals(
            make_sum(vec![
                make_diffop(
                    DiffOperator::TimeDerivative,
                    make_field("ρ", TensorDescriptor::scalar(euc3)),
                ),
                make_diffop(
                    DiffOperator::Divergence,
                    make_product(vec![
                        make_field("ρ", TensorDescriptor::scalar(euc3)),
                        make_field("v", TensorDescriptor::vector(euc3)),
                    ]),
                ),
            ]),
            EquationNode::Scalar(0.0),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature {
            mass: 1,
            length: -3,
            time: -1,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    }
}

/// q = -k∇T — Fourier's law of heat conduction
fn fourier_heat_law() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "Fourier Heat Law".to_string(),
        domain: PhysicsDomain::Thermodynamics,
        ast: make_equals(
            make_field("q", TensorDescriptor::vector(euc3)),
            EquationNode::Negate(Box::new(make_product(vec![
                make_const("k"),
                make_diffop(
                    DiffOperator::Gradient,
                    make_field("T", TensorDescriptor::scalar(euc3)),
                ),
            ]))),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature {
            mass: 1,
            length: 0,
            time: -3,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
        tensor: Some(TensorDescriptor::vector(euc3)),
    }
}

/// Γ = (2π/ℏ)|⟨f|V|i⟩|²ρ(E) — Fermi's golden rule
fn fermi_golden_rule() -> PhysicsEquation {
    PhysicsEquation {
        name: "Fermi Golden Rule".to_string(),
        domain: PhysicsDomain::QuantumMechanics,
        ast: make_equals(
            make_const("Γ"),
            make_product(vec![
                make_const("2π/ℏ"),
                EquationNode::Power {
                    base: Box::new(make_const("|⟨f|V|i⟩|")),
                    exponent: Box::new(EquationNode::Scalar(2.0)),
                },
                make_const("ρ(E)"),
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

/// Δλ = (h/mc)(1 - cosθ) — Compton scattering
fn compton_scattering() -> PhysicsEquation {
    PhysicsEquation {
        name: "Compton Scattering".to_string(),
        domain: PhysicsDomain::QuantumMechanics,
        ast: make_equals(
            make_const("Δλ"),
            make_product(vec![
                make_const("h/(m_e·c)"),
                make_sum(vec![
                    EquationNode::Scalar(1.0),
                    EquationNode::Negate(Box::new(make_const("cos(θ)"))),
                ]),
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::Lorentz], false),
        dimensions: DimensionalSignature::LENGTH,
        tensor: None,
    }
}

/// ∇²φ = -ρ/ε₀ — Poisson's equation
fn poisson_equation() -> PhysicsEquation {
    let euc3 = MetricSignature::Euclidean(3);
    PhysicsEquation {
        name: "Poisson Equation".to_string(),
        domain: PhysicsDomain::Electromagnetism,
        ast: make_equals(
            make_diffop(
                DiffOperator::Laplacian,
                make_field("φ", TensorDescriptor::scalar(euc3)),
            ),
            EquationNode::Negate(Box::new(make_product(vec![
                make_field("ρ", TensorDescriptor::scalar(euc3)),
                EquationNode::Power {
                    base: Box::new(make_const("ε₀")),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]))),
        ),
        symmetries: SymmetryDescriptor::new(vec![LieGroup::U(1), LieGroup::SO(3)], vec![], true),
        dimensions: DimensionalSignature {
            mass: 1,
            length: 1,
            time: -3,
            current: -1,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    }
}

/// 1/λ = R(1/n₁² - 1/n₂²) — Rydberg formula
fn rydberg_formula() -> PhysicsEquation {
    PhysicsEquation {
        name: "Rydberg Formula".to_string(),
        domain: PhysicsDomain::QuantumMechanics,
        ast: make_equals(
            EquationNode::Power {
                base: Box::new(make_const("λ")),
                exponent: Box::new(EquationNode::Scalar(-1.0)),
            },
            make_product(vec![
                make_const("R_∞"),
                make_sum(vec![
                    EquationNode::Power {
                        base: Box::new(make_const("n₁")),
                        exponent: Box::new(EquationNode::Scalar(-2.0)),
                    },
                    EquationNode::Negate(Box::new(EquationNode::Power {
                        base: Box::new(make_const("n₂")),
                        exponent: Box::new(EquationNode::Scalar(-2.0)),
                    })),
                ]),
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature::INVERSE_LENGTH,
        tensor: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE A4: Orbital Mechanics + Hydrogen Spectrum
// ═══════════════════════════════════════════════════════════════════════════
// These entries fill the gap between Kepler's third law (periods) and the
// Rydberg wavelength formula with the direct ENERGY forms that autonomous
// discovery from numerical data actually produces.

/// Hydrogen energy levels (Bohr model energy form): E_n = -13.6 eV / n²
///
/// The autonomous discoverer converges on this form when fed hydrogen spectral
/// data or the Bohr-model ionization energies. Distinct from the Rydberg
/// wavelength formula `1/λ = R(1/n₁² - 1/n₂²)` because it's the energy of a
/// single orbital, not the transition wavelength between two.
fn hydrogen_energy_levels() -> PhysicsEquation {
    PhysicsEquation {
        name: "Hydrogen Energy Levels".to_string(),
        domain: PhysicsDomain::QuantumMechanics,
        ast: make_equals(
            make_const("E_n"),
            make_product(vec![
                EquationNode::Negate(Box::new(make_const("R_H"))),
                EquationNode::Power {
                    base: Box::new(make_const("n")),
                    exponent: Box::new(EquationNode::Scalar(-2.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    }
}

/// Kepler orbital total energy: E = ½mv² - GMm/r
///
/// The conserved total mechanical energy for two-body gravitational orbits.
/// Shape: kinetic (positive quadratic in velocity) minus potential (negative
/// inverse distance). This is the Sum-of-(Product, Negate-of-Product) that
/// the Kepler two-body autonomous discovery produces.
fn kepler_orbital_energy() -> PhysicsEquation {
    PhysicsEquation {
        name: "Kepler Orbital Energy".to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast: make_equals(
            make_const("E"),
            make_sum(vec![
                // ½ m v²
                make_product(vec![
                    EquationNode::Scalar(0.5),
                    make_const("m"),
                    EquationNode::Power {
                        base: Box::new(make_const("v")),
                        exponent: Box::new(EquationNode::Scalar(2.0)),
                    },
                ]),
                // -GMm/r
                EquationNode::Negate(Box::new(make_product(vec![
                    make_const("G"),
                    make_const("M"),
                    make_const("m"),
                    EquationNode::Power {
                        base: Box::new(make_const("r")),
                        exponent: Box::new(EquationNode::Scalar(-1.0)),
                    },
                ]))),
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(
            // SO(4) captures the Kepler hidden symmetry (Laplace-Runge-Lenz)
            vec![LieGroup::SO(4)],
            false,
        ),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    }
}

/// Gravitational potential energy: U = -GMm/r
///
/// The pure inverse-distance potential. Discovery engines that see only
/// potential-energy data (not kinetic) will converge on this form.
fn gravitational_potential_energy() -> PhysicsEquation {
    PhysicsEquation {
        name: "Gravitational Potential Energy".to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast: make_equals(
            make_const("U"),
            EquationNode::Negate(Box::new(make_product(vec![
                make_const("G"),
                make_const("M"),
                make_const("m"),
                EquationNode::Power {
                    base: Box::new(make_const("r")),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]))),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    }
}

/// Harmonic oscillator total energy: E = ½kx² + ½mv²
///
/// Shape: Sum of two quadratic kinetic/potential terms — the canonical
/// `x² + v²` invariant the engine discovers for the harmonic oscillator.
fn harmonic_oscillator_energy() -> PhysicsEquation {
    PhysicsEquation {
        name: "Harmonic Oscillator Energy".to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast: make_equals(
            make_const("E"),
            make_sum(vec![
                // ½ k x²
                make_product(vec![
                    EquationNode::Scalar(0.5),
                    make_const("k"),
                    EquationNode::Power {
                        base: Box::new(make_const("x")),
                        exponent: Box::new(EquationNode::Scalar(2.0)),
                    },
                ]),
                // ½ m v²
                make_product(vec![
                    EquationNode::Scalar(0.5),
                    make_const("m"),
                    EquationNode::Power {
                        base: Box::new(make_const("v")),
                        exponent: Box::new(EquationNode::Scalar(2.0)),
                    },
                ]),
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(2)], false),
        dimensions: DimensionalSignature::ENERGY,
        tensor: None,
    }
}

/// Harmonic oscillator invariant in natural units: `x² + v²`
///
/// This is the shape autonomous discovery produces for a harmonic oscillator
/// with `k = m = 1` — the raw Hamiltonian minus the ½ prefactor and named
/// constants. Stored separately from `harmonic_oscillator_energy()` so
/// recognition matches the discovered skeleton directly, not the physical
/// formula with dimensioned coefficients.
///
/// Dimensions: declared as DIMENSIONLESS because natural units have already
/// absorbed the physical scales. The dimensional inference layer likewise
/// returns DIMENSIONLESS (via `Inconsistent → or_dimensionless()`) for the
/// raw `x² + v²` query when `x` is length and `v` is velocity, so the two
/// align along the dimensional axis.
fn harmonic_oscillator_invariant() -> PhysicsEquation {
    // Dogfood via expr_to_catalog_ast so the catalog shape is guaranteed
    // identical to what autonomous discovery produces for this invariant.
    let v = |n: &str| Expr::Var(n.into());
    let pow2 = |e: Expr| Expr::BinOp(BinOp::Pow, Box::new(e), Box::new(Expr::Const(2.0)));
    let add = |a: Expr, b: Expr| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b));
    let expr = add(pow2(v("x")), pow2(v("v")));
    PhysicsEquation {
        name: "Harmonic Oscillator Invariant".to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast: expr_to_catalog_ast("E", &expr),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(2)], false),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// Lotka-Volterra first integral: `(x - ln(x)) + (y - ln(y))`
///
/// The canonical transcendental conservation law of predator-prey dynamics
/// with rate constants set to unity. Stored as a dedicated catalog entry
/// (separate from `lotka_volterra()` which encodes the ODE) so that when
/// autonomous discovery finds the invariant, it has a direct recognition
/// target instead of matching nearest-neighbor noise in nuclear physics.
///
/// Dimensions: DIMENSIONLESS — populations are counts and `ln` requires a
/// dimensionless argument, so the entire expression is dimensionless.
fn lotka_volterra_invariant() -> PhysicsEquation {
    // Dogfood via expr_to_catalog_ast. The Expr is the exact form produced
    // by the Lotka-Volterra template seed in `build_invariant_templates`:
    //     (x - ln(x)) + (y - ln(y))
    let v = |n: &str| Expr::Var(n.into());
    let ln = |e: Expr| Expr::Func(UnaryFn::Log, Box::new(e));
    let sub = |a: Expr, b: Expr| Expr::BinOp(BinOp::Sub, Box::new(a), Box::new(b));
    let add = |a: Expr, b: Expr| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b));
    let expr = add(sub(v("x"), ln(v("x"))), sub(v("y"), ln(v("y"))));
    PhysicsEquation {
        name: "Lotka-Volterra Invariant".to_string(),
        domain: PhysicsDomain::Biophysics,
        ast: expr_to_catalog_ast("V", &expr),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// 2D angular momentum in Cartesian form: `L_z = x·vy - y·vx`
///
/// The natural shape for conservation laws discovered from 4D orbital
/// mechanics trajectories, where the state is `(x, y, vx, vy)`. The existing
/// `Angular Momentum` catalog entry encodes the scalar form `L = I·ω`, which
/// is structurally unrelated to what discovery produces. Stored separately
/// so Kepler-like orbital invariants get a direct catalog cousin instead of
/// matching nuclear-physics nearest neighbors.
///
/// Dimensions: `length × velocity = L²/T`. Since we've declared mass out of
/// the equation (unit mass convention), the remaining dimensions are
/// `[L² T⁻¹]`.
fn angular_momentum_2d_cartesian() -> PhysicsEquation {
    // Dogfood via expr_to_catalog_ast. Expr: x·vy - y·vx.
    let v = |n: &str| Expr::Var(n.into());
    let mul = |a: Expr, b: Expr| Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b));
    let sub = |a: Expr, b: Expr| Expr::BinOp(BinOp::Sub, Box::new(a), Box::new(b));
    let expr = sub(mul(v("x"), v("vy")), mul(v("y"), v("vx")));
    PhysicsEquation {
        name: "Angular Momentum (2D Cartesian)".to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast: expr_to_catalog_ast("L_z", &expr),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(2)], false),
        dimensions: DimensionalSignature {
            mass: 0,
            length: 2,
            time: -1,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    }
}

/// Hénon-Heiles Hamiltonian: `H = ½(px² + py²) + ½(x² + y²) + x²y − (1/3)y³`
///
/// The canonical 4D chaotic Hamiltonian from stellar dynamics (Hénon &
/// Heiles 1964). Energy is the only first integral — there is no second
/// isolating integral in the chaotic regime.
///
/// Built from an `Expr` via `rhs_from_expr` so the catalog AST is
/// guaranteed to exactly match what autonomous discovery produces. The
/// earlier hand-constructed version used `Constant { name: "c_0.5000" }`
/// but the discovery path emits `Constant { name: "c_0.5000" }` via a
/// slightly different Product/Sum flattening order, and the shapes didn't
/// align — so queries for the discovered HH form routed to the nearest
/// nuclear-physics neighbor instead of this entry.
fn henon_heiles_hamiltonian() -> PhysicsEquation {
    // Construct the exact Expr tree the GP produces when the HH template
    // seed wins. This mirrors `build_invariant_templates` in the
    // ConjectureEngine 4D branch.
    let v = |n: &str| Expr::Var(n.into());
    let pow2 = |e: Expr| Expr::BinOp(BinOp::Pow, Box::new(e), Box::new(Expr::Const(2.0)));
    let pow3 = |e: Expr| Expr::BinOp(BinOp::Pow, Box::new(e), Box::new(Expr::Const(3.0)));
    let mul = |a: Expr, b: Expr| Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b));
    let add = |a: Expr, b: Expr| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b));

    let half_p2 = mul(Expr::Const(0.5), add(pow2(v("px")), pow2(v("py"))));
    let half_q2 = mul(Expr::Const(0.5), add(pow2(v("x")), pow2(v("y"))));
    let coupling = mul(pow2(v("x")), v("y"));
    let cubic = mul(Expr::Const(-1.0 / 3.0), pow3(v("y")));

    // ((((half_p2 + half_q2) + coupling) + cubic)
    let hh_expr = add(add(add(half_p2, half_q2), coupling), cubic);

    PhysicsEquation {
        name: "Hénon-Heiles Hamiltonian".to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast: expr_to_catalog_ast("H", &hh_expr),
        // No continuous rotational symmetry — the cubic cross-coupling
        // `x²y − y³/3` breaks rotational invariance. `symmetry_inference`
        // returns `none()` for a sum containing cubic terms (it only
        // matches pure sum-of-squares), which aligns with this entry.
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// COMBINATORICS — Ramanujan Showcase routing fix
// ═══════════════════════════════════════════════════════════════════════════
//
// The Ramanujan Protocol showcase discovers triangular numbers `n(n+1)/2` as
// a Z3-proven closed form, but until these entries existed the nearest
// neighbor in the catalog was "Coulomb Screening Enhancement" (similarity
// ~0.70) — a nonsense match because there were zero combinatorics entries
// in a 216-equation catalog dominated by physics.
//
// These five entries give canonical combinatorial sequences a direct
// catalog home in the `Mathematics` domain. They're constructed via
// `expr_to_catalog_ast` so the AST shape exactly matches what autonomous
// discovery produces for each closed form.

/// Triangular numbers: `T(n) = n(n+1)/2`
///
/// The 1st order polygonal number. Canonical closed form for Σk from 1 to n.
/// Discovered as `(n * (n + 1)) / 2` by the ConjectureEngine's recurrence
/// solver, so the AST is built from that exact shape.
fn triangular_numbers() -> PhysicsEquation {
    let v = |n: &str| Expr::Var(n.into());
    let mul = |a: Expr, b: Expr| Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b));
    let add = |a: Expr, b: Expr| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b));
    let div = |a: Expr, b: Expr| Expr::BinOp(BinOp::Div, Box::new(a), Box::new(b));

    // (n * (n + 1)) / 2 — exact shape produced by solve_recurrence
    let expr = div(mul(v("n"), add(v("n"), Expr::Const(1.0))), Expr::Const(2.0));

    PhysicsEquation {
        name: "Triangular Numbers".to_string(),
        domain: PhysicsDomain::Mathematics,
        ast: expr_to_catalog_ast("T", &expr),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// Square pyramidal numbers: `P(n) = n(n+1)(2n+1)/6`
///
/// Σk² from 1 to n. Classic stepping stone in combinatorial identities;
/// the natural closed form for sum-of-squares sequences. Built from the
/// exact factored shape that GP recurrence solving produces.
fn square_pyramidal_numbers() -> PhysicsEquation {
    let v = |n: &str| Expr::Var(n.into());
    let mul = |a: Expr, b: Expr| Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b));
    let add = |a: Expr, b: Expr| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b));
    let div = |a: Expr, b: Expr| Expr::BinOp(BinOp::Div, Box::new(a), Box::new(b));

    // (n * (n + 1) * (2n + 1)) / 6
    let two_n_plus_1 = add(mul(Expr::Const(2.0), v("n")), Expr::Const(1.0));
    let expr = div(
        mul(mul(v("n"), add(v("n"), Expr::Const(1.0))), two_n_plus_1),
        Expr::Const(6.0),
    );

    PhysicsEquation {
        name: "Square Pyramidal Numbers".to_string(),
        domain: PhysicsDomain::Mathematics,
        ast: expr_to_catalog_ast("P", &expr),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// Tetrahedral numbers: `Te(n) = n(n+1)(n+2)/6`
///
/// Σ T(k) from 1 to n — the 3D analogue of triangular numbers. Included
/// because sequences like pyramid counts and layer sums naturally produce
/// this shape, and it's structurally distinct from square pyramidal
/// (different factor ordering).
fn tetrahedral_numbers() -> PhysicsEquation {
    let v = |n: &str| Expr::Var(n.into());
    let mul = |a: Expr, b: Expr| Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b));
    let add = |a: Expr, b: Expr| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b));
    let div = |a: Expr, b: Expr| Expr::BinOp(BinOp::Div, Box::new(a), Box::new(b));

    // n(n+1)(n+2) / 6
    let expr = div(
        mul(
            mul(v("n"), add(v("n"), Expr::Const(1.0))),
            add(v("n"), Expr::Const(2.0)),
        ),
        Expr::Const(6.0),
    );

    PhysicsEquation {
        name: "Tetrahedral Numbers".to_string(),
        domain: PhysicsDomain::Mathematics,
        ast: expr_to_catalog_ast("Te", &expr),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// Sum of cubes: `Σk³ = (n(n+1)/2)² = T(n)²`
///
/// The famous Nicomachus identity — sum of the first n cubes equals the
/// square of the nth triangular number. Stored in its explicit closed form
/// `n²(n+1)²/4`.
///
/// **Currently not registered in the catalog list** because its AST shape
/// is too generic for the similarity metric — it scored 99% matches against
/// arbitrary formulas containing nested power-of-variable subtrees (e.g.
/// `cos(y/e)^(x³)` from PCR3BP's garbage output). Kept as a reference
/// implementation for future re-addition once the similarity metric
/// weights top-level operator agreement more heavily.
#[allow(dead_code)]
fn sum_of_cubes() -> PhysicsEquation {
    let v = |n: &str| Expr::Var(n.into());
    let pow2 = |e: Expr| Expr::BinOp(BinOp::Pow, Box::new(e), Box::new(Expr::Const(2.0)));
    let mul = |a: Expr, b: Expr| Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b));
    let add = |a: Expr, b: Expr| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b));
    let div = |a: Expr, b: Expr| Expr::BinOp(BinOp::Div, Box::new(a), Box::new(b));

    // n² * (n+1)² / 4
    let expr = div(
        mul(pow2(v("n")), pow2(add(v("n"), Expr::Const(1.0)))),
        Expr::Const(4.0),
    );

    PhysicsEquation {
        name: "Sum of Cubes (Nicomachus)".to_string(),
        domain: PhysicsDomain::Mathematics,
        ast: expr_to_catalog_ast("C", &expr),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// Harmonic numbers: `H(n) ≈ ln(n) + γ`
///
/// Asymptotic closed form for the partial sums of the harmonic series.
/// Stored as the log approximation (the exact H(n) has no elementary
/// closed form) because that's what GP regression converges to on the
/// harmonic sequence — the `ln(n) + 0.577...` shape.
///
/// Constant named for the Euler-Mascheroni constant γ ≈ 0.5772.
fn harmonic_numbers() -> PhysicsEquation {
    let v = |n: &str| Expr::Var(n.into());
    let ln = |e: Expr| Expr::Func(UnaryFn::Log, Box::new(e));
    let add = |a: Expr, b: Expr| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b));

    // ln(n) + γ  (γ as literal 0.5772156649)
    let expr = add(ln(v("n")), Expr::Const(0.577_215_664_901_532_9));

    PhysicsEquation {
        name: "Harmonic Numbers (asymptotic)".to_string(),
        domain: PhysicsDomain::Mathematics,
        ast: expr_to_catalog_ast("H", &expr),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

// Keep silencing unused-import warning in the rare case UnaryFn isn't
// referenced elsewhere in catalog.rs — it's imported for future catalog
// entries that use transcendental functions.
#[allow(dead_code)]
fn _unused_unary(_: UnaryFn) {}

/// Inverse-square force law (general form): F = k/r²
///
/// The universal shape of Coulomb, Newton gravity, and the Yukawa limit.
/// Distinct from the specific Newton gravity and Coulomb entries because
/// it's domain-agnostic — discovery engines that find `f(r) ~ 1/r²` in
/// unknown contexts (e.g., unknown force fields in simulation data) should
/// match here even when the specific constant hasn't been named.
fn inverse_square_force() -> PhysicsEquation {
    PhysicsEquation {
        name: "Inverse Square Force Law".to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast: make_equals(
            make_const("F"),
            make_product(vec![
                make_const("k"),
                EquationNode::Power {
                    base: Box::new(make_const("r")),
                    exponent: Box::new(EquationNode::Scalar(-2.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature::FORCE,
        tensor: None,
    }
}

/// C dV/dt = -g_Na·m³h(V-E_Na) - g_K·n⁴(V-E_K) - g_L(V-E_L) + I — Hodgkin-Huxley
fn hodgkin_huxley() -> PhysicsEquation {
    PhysicsEquation {
        name: "Hodgkin-Huxley Equation".to_string(),
        domain: PhysicsDomain::Biophysics,
        ast: make_equals(
            make_product(vec![make_const("C"), make_const("dV/dt")]),
            make_sum(vec![
                EquationNode::Negate(Box::new(make_product(vec![
                    make_const("g_Na"),
                    make_const("m³h"),
                    make_sum(vec![
                        make_const("V"),
                        EquationNode::Negate(Box::new(make_const("E_Na"))),
                    ]),
                ]))),
                EquationNode::Negate(Box::new(make_product(vec![
                    make_const("g_K"),
                    make_const("n⁴"),
                    make_sum(vec![
                        make_const("V"),
                        EquationNode::Negate(Box::new(make_const("E_K"))),
                    ]),
                ]))),
                EquationNode::Negate(Box::new(make_product(vec![
                    make_const("g_L"),
                    make_sum(vec![
                        make_const("V"),
                        EquationNode::Negate(Box::new(make_const("E_L"))),
                    ]),
                ]))),
                make_const("I_ext"),
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature {
            mass: 0,
            length: -2,
            time: -3,
            current: 1,
            temperature: 0,
            amount: 0,
            luminous: 0,
        },
        tensor: None,
    }
}

/// dx/dt = αx - βxy, dy/dt = δxy - γy — Lotka-Volterra
fn lotka_volterra() -> PhysicsEquation {
    PhysicsEquation {
        name: "Lotka-Volterra Equations".to_string(),
        domain: PhysicsDomain::Biophysics,
        ast: make_equals(
            make_const("dx/dt"),
            make_sum(vec![
                make_product(vec![make_const("α"), make_const("x")]),
                EquationNode::Negate(Box::new(make_product(vec![
                    make_const("β"),
                    make_const("x"),
                    make_const("y"),
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

/// P(A|B) = P(B|A)P(A)/P(B) — Bayes' theorem
fn bayes_theorem() -> PhysicsEquation {
    PhysicsEquation {
        name: "Bayes Theorem".to_string(),
        domain: PhysicsDomain::InformationTheory,
        ast: make_equals(
            make_const("P(A|B)"),
            make_product(vec![
                make_const("P(B|A)"),
                make_const("P(A)"),
                EquationNode::Power {
                    base: Box::new(make_const("P(B)")),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ]),
        ),
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    }
}

/// k = A·exp(-E_a/RT) — Arrhenius equation
fn arrhenius_equation() -> PhysicsEquation {
    PhysicsEquation {
        name: "Arrhenius Equation".to_string(),
        domain: PhysicsDomain::Thermodynamics,
        ast: make_equals(
            make_const("k"),
            make_product(vec![
                make_const("A"),
                EquationNode::Exponential(Box::new(EquationNode::Negate(Box::new(make_product(
                    vec![
                        make_const("E_a"),
                        EquationNode::Power {
                            base: Box::new(make_product(vec![make_const("R"), make_const("T")])),
                            exponent: Box::new(EquationNode::Scalar(-1.0)),
                        },
                    ],
                ))))),
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

/// v = H₀d — Hubble's law
fn hubble_law() -> PhysicsEquation {
    PhysicsEquation {
        name: "Hubble Law".to_string(),
        domain: PhysicsDomain::Cosmology,
        ast: make_equals(
            make_const("v"),
            make_product(vec![make_const("H₀"), make_const("d")]),
        ),
        symmetries: SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(3)], false),
        dimensions: DimensionalSignature::VELOCITY,
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
            catalog.len() >= 205,
            "Expected >= 205 entries, got {}",
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
        assert!(
            em.len() >= 4,
            "Expected >= 4 EM equations, got {}",
            em.len()
        );
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
            !catalog.entries_in_domain(PhysicsDomain::Optics).is_empty(),
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
            // Phase A1
            "Newton Second Law",
            "Newton Gravitation",
            "Kepler Third Law",
            "Hooke Law",
            "Coulomb Law",
            "Lorentz Force",
            "Ohm Law",
            "Planck-Einstein Relation",
            "De Broglie Wavelength",
            "Heisenberg Uncertainty Principle",
            "Mass-Energy Equivalence",
            "Lorentz Factor",
            "Planck Radiation Law",
            "Wien Displacement Law",
            // Phase A2
            "Shannon Entropy",
            "KL Divergence",
            "Mutual Information",
            "Variational Free Energy (Friston)",
            "Integrated Information (Phi)",
            "Boltzmann Entropy",
            "Bekenstein-Hawking Entropy",
            "Bayes Theorem",
            // Phase A3
            "Bernoulli Equation",
            "Gibbs Free Energy",
            "Hodgkin-Huxley Equation",
            "Lotka-Volterra Equations",
            "Hubble Law",
            "Arrhenius Equation",
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

        assert!(
            !results.is_empty(),
            "Art's Parts query should return results"
        );

        // Report neighbors (informational)
        for r in &results {
            eprintln!(
                "  Art's Parts neighbor: {} (score={:.3}, domain={:?})",
                r.name, r.score, r.domain
            );
        }
    }
}
