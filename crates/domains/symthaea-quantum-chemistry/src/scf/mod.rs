// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Self-Consistent Field (SCF) infrastructure.
//!
//! Provides the Restricted Hartree-Fock solver with DIIS convergence
//! acceleration, canonical orthogonalization, and all supporting machinery.

pub mod density;
pub mod diis;
pub mod fock;
pub mod generalized_eigen;
pub mod rhf;
pub mod uhf;

pub use generalized_eigen::{
    CheckedGeneralizedEigenError, CheckedGeneralizedEigenResult, GeneralizedEigenResult,
    ScfSolveError, ScfSolverDiagnostics, canonical_orthogonalization,
    canonical_orthogonalization_checked, solve_generalized_eigen,
    solve_generalized_eigen_checked,
};
pub use rhf::{RhfConfig, RhfResult, restricted_hartree_fock, try_restricted_hartree_fock};
pub use uhf::{UhfConfig, UhfResult, try_unrestricted_hartree_fock, unrestricted_hartree_fock};

use crate::molecule::Molecule;

/// Return the electron count without calling the legacy panicking accessor.
///
/// Fallible RHF/UHF entry points use this before constructing densities or integrals,
/// so malformed charge/empty-molecule input stays inside the typed `Result` contract.
pub(crate) fn checked_electron_count(molecule: &Molecule) -> Result<usize, ScfSolveError> {
    if molecule.atoms.is_empty() {
        return Err(ScfSolveError::EmptyMolecule);
    }

    let nuclear_charge = molecule.atoms.iter().try_fold(0_i64, |sum, atom| {
        sum.checked_add(i64::from(atom.atomic_number))
    }).ok_or(ScfSolveError::ElectronCountOverflow)?;

    let electrons = nuclear_charge - i64::from(molecule.charge);
    if electrons < 0 {
        return Err(ScfSolveError::InvalidCharge {
            charge: molecule.charge,
            nuclear_charge,
        });
    }
    if electrons == 0 {
        return Err(ScfSolveError::NoElectrons);
    }

    usize::try_from(electrons).map_err(|_| ScfSolveError::ElectronCountOverflow)
}
