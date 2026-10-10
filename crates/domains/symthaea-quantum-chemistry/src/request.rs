// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Fallible, resource-bounded preflight for quantum-chemistry requests.
//!
//! This module is not a solver and does not imply convergence or chemical
//! accuracy. It validates request preconditions before invoking lower-level
//! APIs that currently contain assertions, builds the selected basis once,
//! and retains that exact basis in the immutable validated request.
//!
//! Supported basis data is intentionally explicit: STO-3G supports Z=1..=54;
//! the current 6-31G provider contains data only for H, C, N, and O.
//!
//! The overlap orthogonality gate validates Xᵀ S X, but the existing Jacobi
//! routine does not expose its convergence status. Solver-wide residual-based
//! convergence reporting is tracked separately in issue #7325.

use crate::basis::basis_631g::Basis631G;
use crate::basis::sto3g::Sto3g;
use crate::basis::{BasisSet, BasisSetProvider};
use crate::integrals::overlap::overlap_matrix;
use crate::molecule::Molecule;
use crate::scf::generalized_eigen::canonical_orthogonalization;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Current serialized request schema version.
pub const QC_REQUEST_SCHEMA_VERSION: u16 = 1;
/// Hard maximum nuclei in a request.
pub const MAX_QC_ATOMS: usize = 256;
/// Hard basis-function cap for bounded request admission.
pub const MAX_QC_BASIS_FUNCTIONS: usize = 64;
/// Hard maximum SCF iterations permitted by a request.
pub const MAX_QC_SCF_ITERATIONS: usize = 1_000;
/// Maximum absolute residual in Xᵀ S X - I accepted during request preflight.
pub const MAX_ORTHOGONALIZATION_RESIDUAL: f64 = 1e-6;

/// Explicit coordinate convention for the native QC engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinateUnit {
    /// Atomic units of length: Bohr radii.
    Bohr,
}

/// Environment assumptions supported by this request API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QcEnvironment {
    /// Isolated molecule; no implicit solvation, embedding, or periodicity.
    GasPhase,
}

/// Electronic-structure method requested from the native backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QcMethod {
    /// Restricted Hartree-Fock; closed-shell singlets only.
    RestrictedHartreeFock,
    /// Unrestricted Hartree-Fock; charge and multiplicity are validated.
    UnrestrictedHartreeFock,
    /// Current self-consistent Kohn-Sham implementation with LDA only.
    KohnShamLda,
}

/// Basis providers currently implemented by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QcBasis {
    /// Minimal STO-3G basis, with data for hydrogen through xenon.
    Sto3g,
    /// 6-31G basis, currently implemented for H/C/N/O only.
    Six31G,
}

/// Numerical SCF controls. These govern termination, not physical accuracy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScfSettings {
    /// Maximum self-consistent-field iterations.
    pub max_iterations: usize,
    /// Energy-change threshold, in Hartree.
    pub energy_convergence_hartree: f64,
    /// RMS density-matrix change threshold.
    pub density_rms_convergence: f64,
    /// Use DIIS acceleration where supported.
    pub use_diis: bool,
    /// Recompute ERIs on demand instead of materializing a dense tensor.
    /// Not supported by the current LDA DFT path.
    pub direct_integrals: bool,
}

impl Default for ScfSettings {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            energy_convergence_hartree: 1e-8,
            density_rms_convergence: 1e-6,
            use_diis: true,
            direct_integrals: false,
        }
    }
}

/// Explicit request-level resource budget. The basis limit is the primary
/// memory guard for the current dense ERI implementation; host-level resource
/// isolation is still required.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QcResourceLimits {
    /// Maximum number of nuclei.
    pub max_atoms: usize,
    /// Maximum number of basis functions after basis construction.
    pub max_basis_functions: usize,
    /// Maximum permitted SCF iterations.
    pub max_scf_iterations: usize,
}

impl Default for QcResourceLimits {
    fn default() -> Self {
        Self {
            max_atoms: 64,
            max_basis_functions: MAX_QC_BASIS_FUNCTIONS,
            max_scf_iterations: 100,
        }
    }
}

/// Serializable request. Call validate to obtain an immutable preflight result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QcRequest {
    /// Schema version used when this request is serialized.
    pub schema_version: u16,
    /// Explicit molecular geometry; all positions must be in Bohr.
    pub molecule: Molecule,
    /// Coordinate-unit declaration, never inferred from magnitude.
    pub coordinate_unit: CoordinateUnit,
    /// Requested electronic method.
    pub method: QcMethod,
    /// Requested basis provider.
    pub basis: QcBasis,
    /// Explicit environment assumptions.
    pub environment: QcEnvironment,
    /// Numerical solver controls.
    pub scf: ScfSettings,
    /// Resource bounds for this request.
    pub resource_limits: QcResourceLimits,
}

impl QcRequest {
    /// Default isolated-molecule RHF/STO-3G single-point request.
    pub fn gas_phase_rhf_sto3g(molecule: Molecule) -> Self {
        Self {
            schema_version: QC_REQUEST_SCHEMA_VERSION,
            molecule,
            coordinate_unit: CoordinateUnit::Bohr,
            method: QcMethod::RestrictedHartreeFock,
            basis: QcBasis::Sto3g,
            environment: QcEnvironment::GasPhase,
            scf: ScfSettings::default(),
            resource_limits: QcResourceLimits::default(),
        }
    }

    /// Validate preconditions and build the exact basis once.
    ///
    /// Success means only that the input is structurally valid for this
    /// implementation and within declared limits. It does not run SCF,
    /// establish convergence, compare a reference, or qualify an energy.
    pub fn validate(self) -> Result<ValidatedQcRequest, QcPreflightError> {
        if self.schema_version != QC_REQUEST_SCHEMA_VERSION {
            return Err(QcPreflightError::UnsupportedSchemaVersion {
                received: self.schema_version,
                supported: QC_REQUEST_SCHEMA_VERSION,
            });
        }
        if self.molecule.atoms.is_empty() {
            return Err(QcPreflightError::EmptyMolecule);
        }

        let limits = &self.resource_limits;
        if limits.max_atoms == 0 || limits.max_atoms > MAX_QC_ATOMS {
            return Err(QcPreflightError::InvalidResourceLimit {
                name: "max_atoms",
                maximum: MAX_QC_ATOMS,
            });
        }
        if limits.max_basis_functions == 0 || limits.max_basis_functions > MAX_QC_BASIS_FUNCTIONS {
            return Err(QcPreflightError::InvalidResourceLimit {
                name: "max_basis_functions",
                maximum: MAX_QC_BASIS_FUNCTIONS,
            });
        }
        if limits.max_scf_iterations == 0 || limits.max_scf_iterations > MAX_QC_SCF_ITERATIONS {
            return Err(QcPreflightError::InvalidResourceLimit {
                name: "max_scf_iterations",
                maximum: MAX_QC_SCF_ITERATIONS,
            });
        }
        if self.molecule.atoms.len() > limits.max_atoms {
            return Err(QcPreflightError::AtomLimitExceeded {
                actual: self.molecule.atoms.len(),
                limit: limits.max_atoms,
            });
        }

        if self.scf.max_iterations == 0 || self.scf.max_iterations > limits.max_scf_iterations {
            return Err(QcPreflightError::InvalidScfSetting { name: "max_iterations" });
        }
        if !self.scf.energy_convergence_hartree.is_finite()
            || self.scf.energy_convergence_hartree <= 0.0
        {
            return Err(QcPreflightError::InvalidScfSetting { name: "energy_convergence_hartree" });
        }
        if !self.scf.density_rms_convergence.is_finite()
            || self.scf.density_rms_convergence <= 0.0
        {
            return Err(QcPreflightError::InvalidScfSetting { name: "density_rms_convergence" });
        }
        if self.method == QcMethod::KohnShamLda && self.scf.direct_integrals {
            return Err(QcPreflightError::UnsupportedSolverSetting {
                method: self.method,
                setting: "direct_integrals",
            });
        }
        if matches!(self.method, QcMethod::RestrictedHartreeFock | QcMethod::KohnShamLda)
            && self.molecule.multiplicity != 1
        {
            return Err(QcPreflightError::MethodRequiresSinglet {
                method: self.method,
                multiplicity: self.molecule.multiplicity,
            });
        }

        let mut nuclear_charge = 0_i64;
        for (atom_index, atom) in self.molecule.atoms.iter().enumerate() {
            let z = atom.atomic_number;
            if !(1..=118).contains(&z) {
                return Err(QcPreflightError::UnsupportedAtomicNumber {
                    atom_index,
                    atomic_number: z,
                });
            }
            if !basis_supports(self.basis, z) {
                return Err(QcPreflightError::UnsupportedBasisElement {
                    basis: self.basis,
                    atom_index,
                    atomic_number: z,
                });
            }
            nuclear_charge += i64::from(z);
            for axis in 0..3 {
                if !atom.position[axis].is_finite() {
                    return Err(QcPreflightError::NonFiniteCoordinate { atom_index, axis });
                }
            }
        }

        // Avoid Molecule::n_electrons() here: it panics on an invalid charge.
        let electrons_signed = nuclear_charge - i64::from(self.molecule.charge);
        if electrons_signed < 0 {
            return Err(QcPreflightError::ChargeExceedsNuclearCharge {
                charge: self.molecule.charge,
                nuclear_charge,
            });
        }
        if electrons_signed == 0 {
            return Err(QcPreflightError::NoElectrons);
        }
        if self.molecule.multiplicity == 0 {
            return Err(QcPreflightError::InvalidMultiplicity);
        }
        let spin_difference = i64::from(self.molecule.multiplicity) - 1;
        if electrons_signed < spin_difference || (electrons_signed - spin_difference) % 2 != 0 {
            return Err(QcPreflightError::InconsistentSpinState {
                electron_count: electrons_signed as usize,
                multiplicity: self.molecule.multiplicity,
            });
        }

        // Reject coincident or numerically overflowing centers before lower
        // level nuclear-repulsion and integral routines are called.
        for i in 0..self.molecule.atoms.len() {
            for j in (i + 1)..self.molecule.atoms.len() {
                let a = self.molecule.atoms[i].position;
                let b = self.molecule.atoms[j].position;
                let distance = (a[0] - b[0]).hypot(a[1] - b[1]).hypot(a[2] - b[2]);
                if !distance.is_finite() {
                    return Err(QcPreflightError::NonFiniteInteratomicDistance {
                        atom_i: i,
                        atom_j: j,
                    });
                }
                if distance <= 1e-12 {
                    return Err(QcPreflightError::CoincidentNuclei { atom_i: i, atom_j: j });
                }
            }
        }

        let electron_count = electrons_signed as usize;
        let multiplicity = self.molecule.multiplicity;
        // Do the spin split in i64: adding electron_count + multiplicity in
        // usize can overflow on 32-bit/WASM targets for adversarial charges.
        // The parity/range checks above guarantee these results are nonnegative.
        let n_alpha = ((electrons_signed + spin_difference) / 2) as usize;
        let n_beta = ((electrons_signed - spin_difference) / 2) as usize;
        let basis = match self.basis {
            QcBasis::Sto3g => Sto3g::build(&self.molecule),
            QcBasis::Six31G => Basis631G::build(&self.molecule),
        };
        let n_basis = basis.n_basis();
        if n_basis == 0 {
            return Err(QcPreflightError::EmptyBasis);
        }
        if n_basis > limits.max_basis_functions {
            return Err(QcPreflightError::BasisFunctionLimitExceeded {
                actual: n_basis,
                limit: limits.max_basis_functions,
            });
        }
        let overlap = overlap_matrix(&basis.functions);
        if overlap.iter().any(|value| !value.is_finite()) {
            return Err(QcPreflightError::NonFiniteOverlapMatrix);
        }
        let (orthogonalization, n_independent, _) =
            canonical_orthogonalization(&overlap, n_basis);
        if orthogonalization.iter().any(|value| !value.is_finite()) {
            return Err(QcPreflightError::NonFiniteOrthogonalization);
        }
        let orthogonality_residual = max_orthogonality_residual(
            &overlap,
            &orthogonalization,
            n_basis,
            n_independent,
        );
        if !orthogonality_residual.is_finite() {
            return Err(QcPreflightError::NonFiniteOrthogonalization);
        }
        if orthogonality_residual > MAX_ORTHOGONALIZATION_RESIDUAL {
            return Err(QcPreflightError::OrthogonalizationResidualExceeded);
        }
        if n_independent == 0 {
            return Err(QcPreflightError::NoIndependentBasisFunctions);
        }
        if n_alpha > n_independent || n_beta > n_independent {
            return Err(QcPreflightError::ElectronCountExceedsEffectiveBasisCapacity {
                electron_count,
                multiplicity,
                basis_functions: n_basis,
                independent_basis_functions: n_independent,
            });
        }

        Ok(ValidatedQcRequest {
            molecule: self.molecule,
            method: self.method,
            basis_choice: self.basis,
            basis,
            n_independent,
            environment: self.environment,
            scf: self.scf,
            electron_count,
            n_alpha,
            n_beta,
        })
    }
}

/// Maximum element-wise residual in Xᵀ S X - I for row-major S (n×n) and
/// X (n×m). This checks the transform actually returned by the existing
/// eigensolver; it does not independently certify eigenvalue convergence.
fn max_orthogonality_residual(
    overlap: &[f64],
    transform: &[f64],
    n_basis: usize,
    n_independent: usize,
) -> f64 {
    let mut max_residual = 0.0_f64;
    for i in 0..n_independent {
        for j in 0..n_independent {
            let mut value = 0.0;
            for mu in 0..n_basis {
                for nu in 0..n_basis {
                    value += transform[mu * n_independent + i]
                        * overlap[mu * n_basis + nu]
                        * transform[nu * n_independent + j];
                }
            }
            if !value.is_finite() {
                return f64::INFINITY;
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            max_residual = max_residual.max((value - expected).abs());
        }
    }
    max_residual
}

fn basis_supports(basis: QcBasis, atomic_number: u8) -> bool {
    match basis {
        QcBasis::Sto3g => (1..=54).contains(&atomic_number),
        // These are the only element match arms in the current 6-31G provider.
        QcBasis::Six31G => matches!(atomic_number, 1 | 6 | 7 | 8),
    }
}

/// Immutable input admitted by preflight. It contains no calculated energy.
#[derive(Debug, Clone)]
pub struct ValidatedQcRequest {
    molecule: Molecule,
    method: QcMethod,
    basis_choice: QcBasis,
    basis: BasisSet,
    n_independent: usize,
    environment: QcEnvironment,
    scf: ScfSettings,
    electron_count: usize,
    n_alpha: usize,
    n_beta: usize,
}

impl ValidatedQcRequest {
    /// Molecular geometry in Bohr, exposed immutably.
    pub fn molecule(&self) -> &Molecule {
        &self.molecule
    }
    /// Requested electronic method.
    pub fn method(&self) -> QcMethod {
        self.method
    }
    /// Selected basis provider.
    pub fn basis_choice(&self) -> QcBasis {
        self.basis_choice
    }
    /// Exact basis set built and retained by preflight.
    pub fn basis(&self) -> &BasisSet {
        &self.basis
    }
    /// Basis-function count after overlap-based linear-dependence removal.
    pub fn n_independent_basis_functions(&self) -> usize {
        self.n_independent
    }
    /// Explicit environment convention.
    pub fn environment(&self) -> QcEnvironment {
        self.environment
    }
    /// Numerical SCF controls.
    pub fn scf_settings(&self) -> &ScfSettings {
        &self.scf
    }
    /// Electron count after net charge is applied.
    pub fn electron_count(&self) -> usize {
        self.electron_count
    }
    /// Alpha electron count implied by multiplicity.
    pub fn n_alpha(&self) -> usize {
        self.n_alpha
    }
    /// Beta electron count implied by multiplicity.
    pub fn n_beta(&self) -> usize {
        self.n_beta
    }
}

/// Typed input rejection reason. None of these errors describes energy
/// accuracy or reaction feasibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QcPreflightError {
    UnsupportedSchemaVersion { received: u16, supported: u16 },
    EmptyMolecule,
    InvalidResourceLimit { name: &'static str, maximum: usize },
    AtomLimitExceeded { actual: usize, limit: usize },
    InvalidScfSetting { name: &'static str },
    UnsupportedSolverSetting { method: QcMethod, setting: &'static str },
    MethodRequiresSinglet { method: QcMethod, multiplicity: u32 },
    UnsupportedAtomicNumber { atom_index: usize, atomic_number: u8 },
    UnsupportedBasisElement { basis: QcBasis, atom_index: usize, atomic_number: u8 },
    NonFiniteCoordinate { atom_index: usize, axis: usize },
    NonFiniteInteratomicDistance { atom_i: usize, atom_j: usize },
    NonFiniteOverlapMatrix,
    NonFiniteOrthogonalization,
    OrthogonalizationResidualExceeded,
    NoIndependentBasisFunctions,
    CoincidentNuclei { atom_i: usize, atom_j: usize },
    ChargeExceedsNuclearCharge { charge: i32, nuclear_charge: i64 },
    NoElectrons,
    InvalidMultiplicity,
    InconsistentSpinState { electron_count: usize, multiplicity: u32 },
    EmptyBasis,
    BasisFunctionLimitExceeded { actual: usize, limit: usize },
    ElectronCountExceedsEffectiveBasisCapacity {
        electron_count: usize,
        multiplicity: u32,
        basis_functions: usize,
        independent_basis_functions: usize,
    },
}

impl fmt::Display for QcPreflightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { received, supported } => {
                write!(
                    f,
                    "unsupported QC request schema {received}; supported schema is {supported}"
                )
            }
            Self::EmptyMolecule => write!(f, "molecule has no atoms"),
            Self::InvalidResourceLimit { name, maximum } => write!(
                f,
                "resource limit {name} must be positive and no greater than {maximum}"
            ),
            Self::AtomLimitExceeded { actual, limit } => {
                write!(f, "molecule has {actual} atoms; request limit is {limit}")
            }
            Self::InvalidScfSetting { name } => write!(f, "invalid SCF setting: {name}"),
            Self::UnsupportedSolverSetting { method, setting } => {
                write!(f, "method {method:?} does not support setting {setting}")
            }
            Self::MethodRequiresSinglet { method, multiplicity } => {
                write!(f, "method {method:?} requires multiplicity 1; got {multiplicity}")
            }
            Self::UnsupportedAtomicNumber { atom_index, atomic_number } => {
                write!(f, "atom {atom_index} has unsupported atomic number {atomic_number}")
            }
            Self::UnsupportedBasisElement { basis, atom_index, atomic_number } => {
                write!(
                    f,
                    "basis {basis:?} has no declared data for atom {atom_index} (Z={atomic_number})"
                )
            }
            Self::NonFiniteCoordinate { atom_index, axis } => {
                write!(f, "atom {atom_index} coordinate axis {axis} is non-finite")
            }
            Self::NonFiniteInteratomicDistance { atom_i, atom_j } => {
                write!(
                    f,
                    "interatomic distance between atoms {atom_i} and {atom_j} is non-finite"
                )
            }
            Self::NonFiniteOverlapMatrix => write!(f, "overlap matrix contains non-finite values"),
            Self::NonFiniteOrthogonalization => {
                write!(f, "basis orthogonalization produced non-finite values")
            }
            Self::OrthogonalizationResidualExceeded => write!(
                f,
                "basis transform violates Xᵀ S X = I beyond the preflight residual limit"
            ),
            Self::NoIndependentBasisFunctions => {
                write!(f, "overlap matrix has no independent basis functions")
            }
            Self::CoincidentNuclei { atom_i, atom_j } => {
                write!(f, "atoms {atom_i} and {atom_j} occupy coincident nuclear centers")
            }
            Self::ChargeExceedsNuclearCharge { charge, nuclear_charge } => {
                write!(f, "charge {charge} exceeds total nuclear charge {nuclear_charge}")
            }
            Self::NoElectrons => {
                write!(f, "zero-electron systems are unsupported by this molecular request API")
            }
            Self::InvalidMultiplicity => write!(f, "spin multiplicity must be at least 1"),
            Self::InconsistentSpinState { electron_count, multiplicity } => {
                write!(
                    f,
                    "electron count {electron_count} is inconsistent with multiplicity {multiplicity}"
                )
            }
            Self::EmptyBasis => write!(f, "selected basis generated no basis functions"),
            Self::BasisFunctionLimitExceeded { actual, limit } => {
                write!(f, "basis has {actual} functions; request limit is {limit}")
            }
            Self::ElectronCountExceedsEffectiveBasisCapacity {
                electron_count,
                multiplicity,
                basis_functions,
                independent_basis_functions,
            } => write!(
                f,
                "{electron_count} electrons at multiplicity {multiplicity} exceed the capacity of {independent_basis_functions} independent orbitals ({basis_functions} nominal basis functions)"
            ),
        }
    }
}

impl std::error::Error for QcPreflightError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::Atom;

    fn h2() -> Molecule {
        Molecule::new(vec![Atom::new(1, 0.0, 0.0, 0.0), Atom::new(1, 0.0, 0.0, 1.4)])
    }

    #[test]
    fn accepts_h2_and_retains_the_built_basis() {
        let valid = QcRequest::gas_phase_rhf_sto3g(h2()).validate().unwrap();
        assert_eq!(valid.electron_count(), 2);
        assert_eq!((valid.n_alpha(), valid.n_beta()), (1, 1));
        assert_eq!(valid.basis().n_basis(), 2);
    }

    #[test]
    fn rejects_empty_or_non_finite_geometry() {
        let mut req = QcRequest::gas_phase_rhf_sto3g(h2());
        req.molecule.atoms.clear();
        assert_eq!(req.validate().unwrap_err(), QcPreflightError::EmptyMolecule);

        let mut req = QcRequest::gas_phase_rhf_sto3g(h2());
        req.molecule.atoms[1].position[2] = f64::NAN;
        assert_eq!(
            req.validate().unwrap_err(),
            QcPreflightError::NonFiniteCoordinate {
                atom_index: 1,
                axis: 2
            }
        );
    }

    #[test]
    fn rejects_coincident_nuclei_and_invalid_charge() {
        let mut req = QcRequest::gas_phase_rhf_sto3g(h2());
        req.molecule.atoms[1].position = req.molecule.atoms[0].position;
        assert_eq!(
            req.validate().unwrap_err(),
            QcPreflightError::CoincidentNuclei {
                atom_i: 0,
                atom_j: 1
            }
        );

        let mut req = QcRequest::gas_phase_rhf_sto3g(h2());
        req.molecule.charge = 3;
        assert!(matches!(req.validate(), Err(QcPreflightError::ChargeExceedsNuclearCharge { .. })));
    }

    #[test]
    fn rejects_inconsistent_spin_and_restricted_open_shell() {
        let mut req = QcRequest::gas_phase_rhf_sto3g(h2());
        req.molecule.multiplicity = 2;
        assert!(matches!(req.validate(), Err(QcPreflightError::MethodRequiresSinglet { .. })));

        let mut req = QcRequest::gas_phase_rhf_sto3g(h2());
        req.method = QcMethod::UnrestrictedHartreeFock;
        req.molecule.multiplicity = 2;
        assert_eq!(
            req.validate().unwrap_err(),
            QcPreflightError::InconsistentSpinState {
                electron_count: 2,
                multiplicity: 2
            }
        );
    }

    #[test]
    fn rejects_basis_without_implemented_element_data() {
        let lih = Molecule::new(vec![Atom::new(3, 0.0, 0.0, 0.0), Atom::new(1, 0.0, 0.0, 3.0)]);
        let mut req = QcRequest::gas_phase_rhf_sto3g(lih);
        req.basis = QcBasis::Six31G;
        assert_eq!(
            req.validate().unwrap_err(),
            QcPreflightError::UnsupportedBasisElement {
                basis: QcBasis::Six31G,
                atom_index: 0,
                atomic_number: 3
            }
        );
    }

    #[test]
    fn enforces_basis_function_budget() {
        let atoms = (0..34).map(|i| Atom::new(1, i as f64 * 2.0, 0.0, 0.0)).collect();
        let mut req = QcRequest::gas_phase_rhf_sto3g(Molecule::new(atoms));
        req.basis = QcBasis::Six31G;
        assert_eq!(
            req.validate().unwrap_err(),
            QcPreflightError::BasisFunctionLimitExceeded {
                actual: 68,
                limit: 64
            }
        );
    }

    #[test]
    fn rejects_unsupported_method_settings_and_schema() {
        let mut req = QcRequest::gas_phase_rhf_sto3g(h2());
        req.method = QcMethod::KohnShamLda;
        req.scf.direct_integrals = true;
        assert!(matches!(req.validate(), Err(QcPreflightError::UnsupportedSolverSetting { .. })));

        let mut req = QcRequest::gas_phase_rhf_sto3g(h2());
        req.schema_version += 1;
        assert!(matches!(req.validate(), Err(QcPreflightError::UnsupportedSchemaVersion { .. })));
    }

    #[test]
    fn rejects_non_positive_convergence_threshold() {
        let mut req = QcRequest::gas_phase_rhf_sto3g(h2());
        req.scf.energy_convergence_hartree = 0.0;
        assert_eq!(
            req.validate().unwrap_err(),
            QcPreflightError::InvalidScfSetting {
                name: "energy_convergence_hartree"
            }
        );
    }

    #[test]
    fn orthogonality_residual_is_zero_for_identity_transform() {
        let identity = [1.0, 0.0, 0.0, 1.0];
        assert_eq!(
            max_orthogonality_residual(&identity, &identity, 2, 2),
            0.0
        );
    }

    #[test]
    fn orthogonality_residual_detects_invalid_transform() {
        let identity = [1.0, 0.0, 0.0, 1.0];
        let invalid = [0.5, 0.0, 0.0, 1.0];
        assert!(
            max_orthogonality_residual(&identity, &invalid, 2, 2)
                > MAX_ORTHOGONALIZATION_RESIDUAL
        );
    }

    #[test]
    fn rejects_electron_count_beyond_effective_basis_capacity() {
        let atoms = (0..4).map(|i| Atom::new(1, i as f64 * 2.0, 0.0, 0.0)).collect();
        let req = QcRequest::gas_phase_rhf_sto3g(Molecule::with_charge(atoms, -6, 1));
        assert!(matches!(
            req.validate(),
            Err(QcPreflightError::ElectronCountExceedsEffectiveBasisCapacity { .. })
        ));
    }

    #[test]
    fn rejects_nearly_dependent_basis_that_cannot_hold_declared_spin_state() {
        let atoms = vec![
            Atom::new(1, 0.0, 0.0, 0.0),
            Atom::new(1, 1e-8, 0.0, 0.0),
            Atom::new(1, 2e-8, 0.0, 0.0),
        ];
        let req = QcRequest {
            schema_version: QC_REQUEST_SCHEMA_VERSION,
            molecule: Molecule::with_charge(atoms, 0, 2),
            coordinate_unit: CoordinateUnit::Bohr,
            method: QcMethod::UnrestrictedHartreeFock,
            basis: QcBasis::Sto3g,
            environment: QcEnvironment::GasPhase,
            scf: ScfSettings::default(),
            resource_limits: QcResourceLimits::default(),
        };
        assert!(matches!(
            req.validate(),
            Err(QcPreflightError::ElectronCountExceedsEffectiveBasisCapacity {
                electron_count: 3,
                multiplicity: 2,
                independent_basis_functions: 1,
                ..
            })
        ));
    }
}
