// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Domain types for the physics bridge.
//!
//! All enums, structs, and descriptors used across the crate.
//! No HDC dependency — pure data types with serde support.

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// SI DIMENSIONAL ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════════

/// 7 SI base dimensions with integer exponents.
///
/// Examples:
/// - Velocity: M⁰L¹T⁻¹ → `DimensionalSignature { length: 1, time: -1, ..zero }`
/// - Energy:   M¹L²T⁻² → `DimensionalSignature { mass: 1, length: 2, time: -2, ..zero }`
/// - Torque:   M¹L²T⁻² → identical to energy (correct — same dimensions)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DimensionalSignature {
    /// Mass (kg) exponent
    pub mass: i8,
    /// Length (m) exponent
    pub length: i8,
    /// Time (s) exponent
    pub time: i8,
    /// Electric current (A) exponent
    pub current: i8,
    /// Temperature (K) exponent
    pub temperature: i8,
    /// Amount of substance (mol) exponent
    pub amount: i8,
    /// Luminous intensity (cd) exponent
    pub luminous: i8,
}

impl DimensionalSignature {
    /// Dimensionless quantity.
    pub const DIMENSIONLESS: Self = Self {
        mass: 0,
        length: 0,
        time: 0,
        current: 0,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };

    /// Return exponents as a 7-element array [M, L, T, I, Θ, N, J].
    pub fn as_array(&self) -> [i8; 7] {
        [
            self.mass,
            self.length,
            self.time,
            self.current,
            self.temperature,
            self.amount,
            self.luminous,
        ]
    }

    /// Whether all exponents are zero (dimensionless).
    pub fn is_dimensionless(&self) -> bool {
        *self == Self::DIMENSIONLESS
    }

    // Common physical quantities as constants
    pub const ENERGY: Self = Self {
        mass: 1,
        length: 2,
        time: -2,
        current: 0,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };
    pub const FORCE: Self = Self {
        mass: 1,
        length: 1,
        time: -2,
        current: 0,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };
    pub const VELOCITY: Self = Self {
        mass: 0,
        length: 1,
        time: -1,
        current: 0,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };
    pub const ACCELERATION: Self = Self {
        mass: 0,
        length: 1,
        time: -2,
        current: 0,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };
    pub const MOMENTUM: Self = Self {
        mass: 1,
        length: 1,
        time: -1,
        current: 0,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };
    pub const PRESSURE: Self = Self {
        mass: 1,
        length: -1,
        time: -2,
        current: 0,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };
    pub const CHARGE: Self = Self {
        mass: 0,
        length: 0,
        time: 1,
        current: 1,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };
    pub const ELECTRIC_FIELD: Self = Self {
        mass: 1,
        length: 1,
        time: -3,
        current: -1,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };
    pub const MAGNETIC_FIELD: Self = Self {
        mass: 1,
        length: 0,
        time: -2,
        current: -1,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };
    pub const ACTION: Self = Self {
        mass: 1,
        length: 2,
        time: -1,
        current: 0,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };
    pub const MASS: Self = Self {
        mass: 1,
        length: 0,
        time: 0,
        current: 0,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };
    pub const LENGTH: Self = Self {
        mass: 0,
        length: 1,
        time: 0,
        current: 0,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };
    pub const TIME: Self = Self {
        mass: 0,
        length: 0,
        time: 1,
        current: 0,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };
    /// Inverse length (curvature, wavenumber).
    pub const INVERSE_LENGTH: Self = Self {
        mass: 0,
        length: -1,
        time: 0,
        current: 0,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };
    /// Energy density (J/m³).
    pub const ENERGY_DENSITY: Self = Self {
        mass: 1,
        length: -1,
        time: -2,
        current: 0,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };
    /// Cross-section (area, m²).
    pub const AREA: Self = Self {
        mass: 0,
        length: 2,
        time: 0,
        current: 0,
        temperature: 0,
        amount: 0,
        luminous: 0,
    };

    // ─── Dimensional arithmetic ─────────────────────────────────────────
    //
    // These operations let dimensional inference walk an expression tree.
    // Multiplication of two quantities adds their dimensions; division
    // subtracts; raising to an integer power scales.

    /// Construct from a 7-element exponent array `[M, L, T, I, Θ, N, J]`.
    pub fn from_array(exponents: [i8; 7]) -> Self {
        Self {
            mass: exponents[0],
            length: exponents[1],
            time: exponents[2],
            current: exponents[3],
            temperature: exponents[4],
            amount: exponents[5],
            luminous: exponents[6],
        }
    }

    /// Add two dimensional signatures element-wise (corresponds to multiplying
    /// quantities: `m·v` has dimensions M + (LT⁻¹) = M¹L¹T⁻¹).
    ///
    /// Returns a new signature; does not mutate `self`.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            mass: self.mass + other.mass,
            length: self.length + other.length,
            time: self.time + other.time,
            current: self.current + other.current,
            temperature: self.temperature + other.temperature,
            amount: self.amount + other.amount,
            luminous: self.luminous + other.luminous,
        }
    }

    /// Subtract two dimensional signatures (corresponds to dividing
    /// quantities: `F/m` has dimensions (MLT⁻²) − M = LT⁻²).
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            mass: self.mass - other.mass,
            length: self.length - other.length,
            time: self.time - other.time,
            current: self.current - other.current,
            temperature: self.temperature - other.temperature,
            amount: self.amount - other.amount,
            luminous: self.luminous - other.luminous,
        }
    }

    /// Scale a dimensional signature by an integer factor (corresponds to
    /// raising a quantity to a power: `v²` has dimensions 2·(LT⁻¹) = L²T⁻²).
    ///
    /// Returns `None` if any resulting exponent would overflow `i8`.
    pub fn scale(&self, factor: i8) -> Option<Self> {
        let arr = self.as_array();
        let mut out = [0i8; 7];
        for i in 0..7 {
            out[i] = arr[i].checked_mul(factor)?;
        }
        Some(Self::from_array(out))
    }

    /// Whether two signatures are equal (alias for `==`, but spelled
    /// out for use sites that prefer explicit method calls).
    pub fn matches(&self, other: &Self) -> bool {
        self == other
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TENSOR STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════════

/// Position of a tensor index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexPosition {
    /// Upper index (contravariant) — transforms inversely to basis.
    Contravariant,
    /// Lower index (covariant) — transforms with basis.
    Covariant,
}

/// Symmetry properties of tensor indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorSymmetry {
    /// No symmetry constraint.
    None,
    /// Symmetric under index exchange: T_ij = T_ji.
    Symmetric,
    /// Antisymmetric under index exchange: T_ij = -T_ji.
    Antisymmetric,
    /// Mixed symmetry (e.g., Riemann tensor).
    Mixed,
}

/// Metric signature of the space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricSignature {
    /// Positive-definite (all +), n spatial dimensions.
    Euclidean(u8),
    /// One time + (n-1) spatial dimensions, signature (-,+,...,+).
    Lorentzian(u8),
    /// General signature (p positive, q negative).
    General(u8, u8),
}

impl MetricSignature {
    /// Total number of dimensions.
    pub fn total_dims(&self) -> u8 {
        match self {
            MetricSignature::Euclidean(n) => *n,
            MetricSignature::Lorentzian(n) => *n,
            MetricSignature::General(p, q) => p + q,
        }
    }
}

/// Properties of a spacetime metric solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MetricSolution {
    /// Has a curvature singularity.
    pub singularity: bool,
    /// Has an event horizon.
    pub horizon: bool,
    /// Vacuum solution (T_μν = 0).
    pub vacuum: bool,
    /// Stationary (time-independent).
    pub stationary: bool,
    /// Axisymmetric (rotational symmetry about an axis).
    pub axisymmetric: bool,
    /// Spherically symmetric.
    pub spherically_symmetric: bool,
    /// Cosmological (homogeneous + isotropic).
    pub cosmological: bool,
    /// Has cosmological constant Λ.
    pub has_cosmological_constant: bool,
    /// Has electric charge.
    pub has_charge: bool,
}

impl MetricSolution {
    pub const FLAT: Self = Self {
        singularity: false,
        horizon: false,
        vacuum: true,
        stationary: true,
        axisymmetric: true,
        spherically_symmetric: true,
        cosmological: false,
        has_cosmological_constant: false,
        has_charge: false,
    };
}

/// Full tensor descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorDescriptor {
    /// Tensor rank (number of indices).
    pub rank: u8,
    /// Position of each index (co/contravariant).
    pub index_positions: Vec<IndexPosition>,
    /// Symmetry type.
    pub symmetry: TensorSymmetry,
    /// Metric signature of the ambient space.
    pub metric: MetricSignature,
    /// Spatial dimensionality of each index.
    pub index_dims: u8,
    /// Optional metric solution properties (for metric tensors).
    pub solution: Option<MetricSolution>,
}

impl TensorDescriptor {
    /// Scalar (rank 0).
    pub fn scalar(metric: MetricSignature) -> Self {
        Self {
            rank: 0,
            index_positions: vec![],
            symmetry: TensorSymmetry::None,
            metric,
            index_dims: metric.total_dims(),
            solution: None,
        }
    }

    /// Vector (rank 1, contravariant).
    pub fn vector(metric: MetricSignature) -> Self {
        Self {
            rank: 1,
            index_positions: vec![IndexPosition::Contravariant],
            symmetry: TensorSymmetry::None,
            metric,
            index_dims: metric.total_dims(),
            solution: None,
        }
    }

    /// Covector / 1-form (rank 1, covariant).
    pub fn covector(metric: MetricSignature) -> Self {
        Self {
            rank: 1,
            index_positions: vec![IndexPosition::Covariant],
            symmetry: TensorSymmetry::None,
            metric,
            index_dims: metric.total_dims(),
            solution: None,
        }
    }

    /// Rank-2 symmetric (e.g., metric, stress-energy).
    pub fn symmetric_2(metric: MetricSignature) -> Self {
        Self {
            rank: 2,
            index_positions: vec![IndexPosition::Covariant, IndexPosition::Covariant],
            symmetry: TensorSymmetry::Symmetric,
            metric,
            index_dims: metric.total_dims(),
            solution: None,
        }
    }

    /// Rank-2 antisymmetric (e.g., field strength tensor).
    pub fn antisymmetric_2(metric: MetricSignature) -> Self {
        Self {
            rank: 2,
            index_positions: vec![IndexPosition::Covariant, IndexPosition::Covariant],
            symmetry: TensorSymmetry::Antisymmetric,
            metric,
            index_dims: metric.total_dims(),
            solution: None,
        }
    }

    /// Rank-4 mixed symmetry (e.g., Riemann tensor).
    pub fn riemann(metric: MetricSignature) -> Self {
        Self {
            rank: 4,
            index_positions: vec![
                IndexPosition::Contravariant,
                IndexPosition::Covariant,
                IndexPosition::Covariant,
                IndexPosition::Covariant,
            ],
            symmetry: TensorSymmetry::Mixed,
            metric,
            index_dims: metric.total_dims(),
            solution: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EQUATION AST
// ═══════════════════════════════════════════════════════════════════════════════

/// Differential and integral operators.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiffOperator {
    /// Partial derivative ∂/∂x.
    Partial,
    /// Gradient ∇.
    Gradient,
    /// Divergence ∇·.
    Divergence,
    /// Curl ∇×.
    Curl,
    /// Laplacian ∇².
    Laplacian,
    /// d'Alembertian □ = ∂²/∂t² - ∇² (wave operator).
    DAlembertian,
    /// Lie derivative L_X.
    LieDerivative,
    /// Exterior derivative d.
    ExteriorDerivative,
    /// Hodge star ⋆.
    HodgeStar,
    /// Integral ∫.
    Integral,
    /// Total time derivative d/dt.
    TimeDerivative,
    /// Covariant derivative ∇_μ.
    CovariantDerivative,
}

/// Recursive equation AST node.
///
/// Encodes the syntactic structure of physics equations for HDC similarity search.
/// Not for numerical computation — purely structural.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EquationNode {
    /// Named tensor field (e.g., "F_μν", "g_ab").
    Field {
        name: String,
        tensor: TensorDescriptor,
    },
    /// Named physical/mathematical constant (e.g., "c", "ℏ", "G").
    Constant { name: String },
    /// Dimensionless scalar value.
    Scalar(f64),
    /// Differential operator applied to an operand.
    DiffOp {
        operator: DiffOperator,
        operand: Box<EquationNode>,
    },
    /// Summation (commutative — order doesn't matter).
    Sum(Vec<EquationNode>),
    /// Product (non-commutative — order matters for operators).
    Product(Vec<EquationNode>),
    /// Equation (lhs = rhs).
    Equals(Box<EquationNode>, Box<EquationNode>),
    /// Commutator [A, B] = AB - BA.
    Commutator(Box<EquationNode>, Box<EquationNode>),
    /// Tensor contraction (index summation).
    Contraction {
        tensor: Box<EquationNode>,
        /// Pairs of contracted index positions.
        contracted_indices: Vec<(usize, usize)>,
    },
    /// Exponential function exp(x).
    Exponential(Box<EquationNode>),
    /// Power x^n.
    Power {
        base: Box<EquationNode>,
        exponent: Box<EquationNode>,
    },
    /// Negation -x.
    Negate(Box<EquationNode>),
    /// Square root √x.
    Sqrt(Box<EquationNode>),
}

// ═══════════════════════════════════════════════════════════════════════════════
// SYMMETRY GROUPS
// ═══════════════════════════════════════════════════════════════════════════════

/// Lie groups relevant to physics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LieGroup {
    /// Special orthogonal group SO(n) — rotations in n dimensions.
    SO(u8),
    /// Special unitary group SU(n) — quantum gauge symmetry.
    SU(u8),
    /// Unitary group U(n).
    U(u8),
    /// Symplectic group Sp(n) — Hamiltonian mechanics.
    Sp(u8),
    /// General linear group GL(n).
    GL(u8),
    /// Lorentz group SO(3,1) — special relativity.
    Lorentz,
    /// Poincaré group — Lorentz + translations.
    Poincare,
    /// Conformal group — angle-preserving transformations.
    Conformal,
    /// Standard Model gauge group SU(3)×SU(2)×U(1).
    StandardModel,
    /// Exceptional group E(n).
    E(u8),
    /// Exceptional group G₂.
    G2,
}

impl LieGroup {
    /// Dimension of the Lie algebra (number of generators).
    pub fn algebra_dimension(&self) -> u32 {
        match self {
            LieGroup::SO(n) => {
                let n = *n as u32;
                n * (n - 1) / 2
            }
            LieGroup::SU(n) => {
                let n = *n as u32;
                n * n - 1
            }
            LieGroup::U(n) => {
                let n = *n as u32;
                n * n
            }
            LieGroup::Sp(n) => {
                let n = *n as u32;
                n * (2 * n + 1)
            }
            LieGroup::GL(n) => {
                let n = *n as u32;
                n * n
            }
            LieGroup::Lorentz => 6,        // SO(3,1) — 3 boosts + 3 rotations
            LieGroup::Poincare => 10,      // 6 Lorentz + 4 translations
            LieGroup::Conformal => 15,     // SO(4,2)
            LieGroup::StandardModel => 12, // 8 + 3 + 1
            LieGroup::E(6) => 78,
            LieGroup::E(7) => 133,
            LieGroup::E(8) => 248,
            LieGroup::E(_) => 0,
            LieGroup::G2 => 14,
        }
    }
}

/// Discrete symmetries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiscreteSymmetry {
    /// Parity (spatial inversion).
    P,
    /// Time reversal.
    T,
    /// Charge conjugation.
    C,
    /// Combined CPT.
    CPT,
    /// Z₂ gauge symmetry.
    Z2,
    /// Permutation group S_n.
    Permutation(u8),
}

/// Complete symmetry descriptor for a physics system.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SymmetryDescriptor {
    /// Continuous (Lie) symmetries.
    pub lie_groups: Vec<LieGroup>,
    /// Discrete symmetries.
    pub discrete: Vec<DiscreteSymmetry>,
    /// Total algebra dimension (sum of Lie algebra dims).
    pub algebra_dimension: u32,
    /// Whether the symmetry is a gauge symmetry (local, not global).
    pub is_gauge: bool,
}

impl SymmetryDescriptor {
    /// Empty symmetry (no known symmetries).
    pub fn none() -> Self {
        Self {
            lie_groups: vec![],
            discrete: vec![],
            algebra_dimension: 0,
            is_gauge: false,
        }
    }

    /// Create from Lie groups, auto-computing algebra dimension.
    pub fn from_lie_groups(groups: Vec<LieGroup>, is_gauge: bool) -> Self {
        let algebra_dimension = groups.iter().map(|g| g.algebra_dimension()).sum();
        Self {
            lie_groups: groups,
            discrete: vec![],
            algebra_dimension,
            is_gauge,
        }
    }

    /// Create with both Lie and discrete symmetries.
    pub fn new(lie_groups: Vec<LieGroup>, discrete: Vec<DiscreteSymmetry>, is_gauge: bool) -> Self {
        let algebra_dimension = lie_groups.iter().map(|g| g.algebra_dimension()).sum();
        Self {
            lie_groups,
            discrete,
            algebra_dimension,
            is_gauge,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHYSICS DOMAIN & CATALOG
// ═══════════════════════════════════════════════════════════════════════════════

/// Physics domain classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhysicsDomain {
    ClassicalMechanics,
    Electromagnetism,
    GeneralRelativity,
    QuantumMechanics,
    QuantumFieldTheory,
    StatisticalMechanics,
    FluidDynamics,
    Cosmology,
    NuclearPhysics,
    Thermodynamics,
    Optics,
    CondensedMatter,
    ParticlePhysics,
    ModifiedGravity,
    InformationTheory,
    Biophysics,
    Acoustics,
    PlasmaPhysics,
    Mathematics,
    Astrophysics,
}

/// A fully described physics equation with all metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicsEquation {
    /// Human-readable name (e.g., "Maxwell's Gauss Law").
    pub name: String,
    /// Physics domain.
    pub domain: PhysicsDomain,
    /// AST representation.
    pub ast: EquationNode,
    /// Symmetries of the equation.
    pub symmetries: SymmetryDescriptor,
    /// Dimensional signature (of the equation as a whole, or LHS).
    pub dimensions: DimensionalSignature,
    /// Tensor structure of the primary field.
    pub tensor: Option<TensorDescriptor>,
}

/// Search result from the physics catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Name of the matched equation.
    pub name: String,
    /// Physics domain.
    pub domain: PhysicsDomain,
    /// Overall weighted similarity score.
    pub score: f32,
    /// Per-aspect similarity breakdown.
    pub breakdown: SimilarityBreakdown,
}

/// Per-aspect similarity scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityBreakdown {
    /// Structural (skeleton) similarity.
    pub structural: f32,
    /// Symmetry group similarity.
    pub symmetry: f32,
    /// Dimensional analysis similarity.
    pub dimensional: f32,
    /// Full equation similarity (with names).
    pub full: f32,
}

/// Telemetry from a physics bridge query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsBridgeTelemetry {
    /// Number of catalog entries searched.
    pub catalog_size: usize,
    /// Number of results returned.
    pub results_returned: usize,
    /// Top match name.
    pub top_match: Option<String>,
    /// Top match score.
    pub top_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensional_signature_basics() {
        assert!(DimensionalSignature::DIMENSIONLESS.is_dimensionless());
        assert!(!DimensionalSignature::ENERGY.is_dimensionless());

        // Energy and torque have identical dimensions
        let torque = DimensionalSignature {
            mass: 1,
            length: 2,
            time: -2,
            current: 0,
            temperature: 0,
            amount: 0,
            luminous: 0,
        };
        assert_eq!(DimensionalSignature::ENERGY, torque);
    }

    #[test]
    fn dimensional_as_array() {
        let energy = DimensionalSignature::ENERGY;
        assert_eq!(energy.as_array(), [1, 2, -2, 0, 0, 0, 0]);
    }

    #[test]
    fn metric_signature_dims() {
        assert_eq!(MetricSignature::Euclidean(3).total_dims(), 3);
        assert_eq!(MetricSignature::Lorentzian(4).total_dims(), 4);
        assert_eq!(MetricSignature::General(3, 1).total_dims(), 4);
    }

    #[test]
    fn tensor_descriptor_constructors() {
        let s = TensorDescriptor::scalar(MetricSignature::Euclidean(3));
        assert_eq!(s.rank, 0);
        assert!(s.index_positions.is_empty());

        let v = TensorDescriptor::vector(MetricSignature::Lorentzian(4));
        assert_eq!(v.rank, 1);
        assert_eq!(v.index_positions, vec![IndexPosition::Contravariant]);

        let r = TensorDescriptor::riemann(MetricSignature::Lorentzian(4));
        assert_eq!(r.rank, 4);
        assert_eq!(r.symmetry, TensorSymmetry::Mixed);
    }

    #[test]
    fn lie_group_algebra_dimensions() {
        assert_eq!(LieGroup::SO(3).algebra_dimension(), 3);
        assert_eq!(LieGroup::SU(2).algebra_dimension(), 3);
        assert_eq!(LieGroup::SU(3).algebra_dimension(), 8);
        assert_eq!(LieGroup::U(1).algebra_dimension(), 1);
        assert_eq!(LieGroup::Lorentz.algebra_dimension(), 6);
        assert_eq!(LieGroup::Poincare.algebra_dimension(), 10);
        assert_eq!(LieGroup::StandardModel.algebra_dimension(), 12);
        assert_eq!(LieGroup::E(8).algebra_dimension(), 248);
        assert_eq!(LieGroup::G2.algebra_dimension(), 14);
    }

    #[test]
    fn symmetry_descriptor_construction() {
        let sm = SymmetryDescriptor::from_lie_groups(
            vec![LieGroup::SU(3), LieGroup::SU(2), LieGroup::U(1)],
            true,
        );
        assert_eq!(sm.algebra_dimension, 12); // 8 + 3 + 1
        assert!(sm.is_gauge);
    }

    #[test]
    fn equation_node_construction() {
        // ∇²ψ = -k²ψ (Helmholtz equation structure)
        let psi = EquationNode::Field {
            name: "ψ".to_string(),
            tensor: TensorDescriptor::scalar(MetricSignature::Euclidean(3)),
        };
        let laplacian_psi = EquationNode::DiffOp {
            operator: DiffOperator::Laplacian,
            operand: Box::new(psi.clone()),
        };
        let k_sq = EquationNode::Product(vec![
            EquationNode::Negate(Box::new(EquationNode::Power {
                base: Box::new(EquationNode::Constant {
                    name: "k".to_string(),
                }),
                exponent: Box::new(EquationNode::Scalar(2.0)),
            })),
            psi,
        ]);
        let helmholtz = EquationNode::Equals(Box::new(laplacian_psi), Box::new(k_sq));

        // Should be constructible
        match &helmholtz {
            EquationNode::Equals(lhs, _rhs) => match lhs.as_ref() {
                EquationNode::DiffOp { operator, .. } => {
                    assert_eq!(*operator, DiffOperator::Laplacian);
                }
                _ => panic!("Expected DiffOp"),
            },
            _ => panic!("Expected Equals"),
        }
    }

    #[test]
    fn physics_domain_classification() {
        let eq = PhysicsEquation {
            name: "Test".to_string(),
            domain: PhysicsDomain::Electromagnetism,
            ast: EquationNode::Scalar(0.0),
            symmetries: SymmetryDescriptor::none(),
            dimensions: DimensionalSignature::DIMENSIONLESS,
            tensor: None,
        };
        assert_eq!(eq.domain, PhysicsDomain::Electromagnetism);
    }

    #[test]
    fn search_result_construction() {
        let result = SearchResult {
            name: "Maxwell Gauss".to_string(),
            domain: PhysicsDomain::Electromagnetism,
            score: 0.85,
            breakdown: SimilarityBreakdown {
                structural: 0.9,
                symmetry: 0.8,
                dimensional: 0.85,
                full: 0.7,
            },
        };
        assert!(result.score > 0.8);
    }

    #[test]
    fn serde_roundtrip() {
        let sig = DimensionalSignature::ENERGY;
        let json = serde_json::to_string(&sig).unwrap();
        let back: DimensionalSignature = serde_json::from_str(&json).unwrap();
        assert_eq!(sig, back);

        let group = LieGroup::SU(3);
        let json = serde_json::to_string(&group).unwrap();
        let back: LieGroup = serde_json::from_str(&json).unwrap();
        assert_eq!(group, back);
    }
}
