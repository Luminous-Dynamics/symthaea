// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Emit reproducible JSON fixtures for independent quantum-chemistry comparison.
//!
//! Run from a clean repository root:
//! `test -z "$(git status --porcelain)" || { echo "clean worktree required" >&2; exit 2; }`
//! `cargo run -p symthaea-quantum-chemistry --example qc_native_reference_fixtures > /tmp/qc-native.json`
//!
//! The paired PySCF comparer is `scripts/qc_reference_compare.py`. This
//! example records exact native geometries and computed results; it does not
//! call PySCF and does not qualify the native implementation. Legacy reference
//! targets are included for context and are never used as the cross-backend
//! pass/fail threshold.

use serde::Serialize;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::Command;
use symthaea_quantum_chemistry::basis::BasisSet;
use symthaea_quantum_chemistry::basis::BasisSetProvider;
use symthaea_quantum_chemistry::basis::basis_631g::Basis631G;
use symthaea_quantum_chemistry::basis::sto3g::Sto3g;
use symthaea_quantum_chemistry::integrals::overlap::overlap_matrix;
use symthaea_quantum_chemistry::molecule::Molecule;
use symthaea_quantum_chemistry::scf::rhf::{RhfConfig, restricted_hartree_fock};
use symthaea_quantum_chemistry::validation::{
    benchmark_molecules, hf_631g_references, hf_sto3g_references,
};

const REPORT_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u16,
    producer: &'static str,
    producer_version: &'static str,
    source_revision: String,
    source_tree_sha: String,
    worktree_clean: bool,
    coordinate_unit: &'static str,
    method: &'static str,
    cases: Vec<CaseRecord>,
}

#[derive(Debug, Serialize)]
struct CaseRecord {
    case_id: String,
    molecule_name: String,
    basis: &'static str,
    coordinate_unit: &'static str,
    molecule: MoleculeRecord,
    historical_reference: HistoricalReference,
    native_result: NativeResult,
}

#[derive(Debug, Serialize)]
struct MoleculeRecord {
    charge: i32,
    multiplicity: u32,
    atoms: Vec<AtomRecord>,
}

#[derive(Debug, Serialize)]
struct AtomRecord {
    atomic_number: u8,
    symbol: &'static str,
    position_bohr: [f64; 3],
}

#[derive(Debug, Serialize)]
struct HistoricalReference {
    total_energy_hartree: f64,
    tolerance_hartree: f64,
    provenance: &'static str,
    note: &'static str,
}

#[derive(Debug, Serialize)]
struct NativeResult {
    status: &'static str,
    total_energy_hartree: Option<f64>,
    electronic_energy_hartree: Option<f64>,
    nuclear_repulsion_hartree: Option<f64>,
    iterations: Option<usize>,
    basis_functions: Option<usize>,
    independent_basis_functions: Option<usize>,
    /// Row-major native AO overlap matrix S_ij = <chi_i|chi_j>, dimension n_basis².
    /// Missing when any overlap entry is non-finite.
    overlap_matrix_row_major: Option<Vec<f64>>,
    panic_message: Option<String>,
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else {
        "non-string panic payload".to_owned()
    }
}

fn record_molecule(molecule: &Molecule) -> MoleculeRecord {
    MoleculeRecord {
        charge: molecule.charge,
        multiplicity: molecule.multiplicity,
        atoms: molecule
            .atoms
            .iter()
            .map(|atom| AtomRecord {
                atomic_number: atom.atomic_number,
                symbol: atom.symbol(),
                position_bohr: atom.position,
            })
            .collect(),
    }
}

fn git_output(args: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("git").args(args).output()?;
    if !output.status.success() {
        return Err(format!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn source_provenance() -> Result<(String, String), Box<dyn std::error::Error>> {
    let status = git_output(&["status", "--porcelain", "--untracked-files=all"])?;
    if !status.is_empty() {
        return Err("refusing to emit QC fixtures from a dirty Git worktree".into());
    }
    let revision = git_output(&["rev-parse", "--verify", "HEAD^{commit}"])?;
    let tree = git_output(&["rev-parse", "--verify", "HEAD^{tree}"])?;
    for (kind, sha) in [("commit", &revision), ("tree", &tree)] {
        if sha.len() != 40 || !sha.chars().all(|ch| ch.is_ascii_hexdigit()) {
            return Err(format!("Git returned an invalid {kind} SHA").into());
        }
    }
    Ok((revision, tree))
}

fn run_case(
    name: &str,
    molecule: Molecule,
    basis_name: &'static str,
    reference_energy: f64,
    tolerance: f64,
    build_basis: fn(&Molecule) -> BasisSet,
    reference_note: &'static str,
) -> CaseRecord {
    let molecule_record = record_molecule(&molecule);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let basis = build_basis(&molecule);
        let n_basis = basis.n_basis();
        let overlap = overlap_matrix(&basis.functions);
        let overlap_is_finite = overlap.len() == n_basis * n_basis
            && overlap.iter().all(|value| value.is_finite());
        let overlap = overlap_is_finite.then_some(overlap);
        let scf = restricted_hartree_fock(&molecule, &basis, &RhfConfig::default());
        (n_basis, overlap, scf)
    }));

    let native_result = match result {
        Ok((n_basis, overlap, scf)) => {
            let finite_outputs = scf.total_energy.is_finite()
                && scf.electronic_energy.is_finite()
                && scf.nuclear_repulsion.is_finite();
            let status = if overlap.is_none() {
                "non_finite_overlap"
            } else if !finite_outputs {
                "non_finite_result"
            } else if scf.converged {
                "converged"
            } else {
                "non_converged"
            };
            NativeResult {
                status,
                total_energy_hartree: finite(scf.total_energy),
                electronic_energy_hartree: finite(scf.electronic_energy),
                nuclear_repulsion_hartree: finite(scf.nuclear_repulsion),
                iterations: Some(scf.n_iterations),
                basis_functions: Some(n_basis),
                independent_basis_functions: Some(scf.n_independent),
                overlap_matrix_row_major: overlap,
                panic_message: None,
            }
        }
        Err(payload) => NativeResult {
            status: "solver_panicked",
            total_energy_hartree: None,
            electronic_energy_hartree: None,
            nuclear_repulsion_hartree: None,
            iterations: None,
            basis_functions: None,
            independent_basis_functions: None,
            overlap_matrix_row_major: None,
            panic_message: Some(panic_message(payload.as_ref())),
        },
    };

    CaseRecord {
        case_id: format!("{name}/{basis_name}/RHF"),
        molecule_name: name.to_owned(),
        basis: basis_name,
        coordinate_unit: "bohr",
        molecule: molecule_record,
        historical_reference: HistoricalReference {
            total_energy_hartree: reference_energy,
            tolerance_hartree: tolerance,
            provenance: "Existing src/validation.rs reference table; consult its comments for primary references and known limitations.",
            note: reference_note,
        },
        native_result,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (source_revision, source_tree_sha) = source_provenance()?;
    let molecules = benchmark_molecules();
    let sto3g_references = hf_sto3g_references();
    let six_31g_references = hf_631g_references();
    let mut cases = Vec::with_capacity(sto3g_references.len() + six_31g_references.len());

    for (ref_name, reference_energy, tolerance) in &sto3g_references {
        if let Some((_, molecule)) = molecules.iter().find(|(name, _)| name == ref_name) {
            let note = if *ref_name == "N2" {
                "Known native N2/STO-3G discrepancy remains unresolved; the historical target is context only."
            } else {
                "Historical reference is contextual only; cross-backend pass/fail uses numerical agreement for the same serialized geometry and basis."
            };
            cases.push(run_case(
                ref_name,
                molecule.clone(),
                "STO-3G",
                *reference_energy,
                *tolerance,
                Sto3g::build,
                note,
            ));
        }
    }

    for (ref_name, reference_energy, tolerance) in &six_31g_references {
        if let Some((_, molecule)) = molecules.iter().find(|(name, _)| name == ref_name) {
            let note = match *ref_name {
                "H2O" => "Known native H2O/6-31G discrepancy remains unresolved; the historical target is context only.",
                "CH4" => "Known native CH4/6-31G discrepancy remains unresolved; the historical target is context only.",
                _ => "Historical reference is contextual only; cross-backend pass/fail uses numerical agreement for the same serialized geometry and basis.",
            };
            cases.push(run_case(
                ref_name,
                molecule.clone(),
                "6-31G",
                *reference_energy,
                *tolerance,
                Basis631G::build,
                note,
            ));
        }
    }

    let (ending_revision, ending_tree_sha) = source_provenance()?;
    if ending_revision != source_revision || ending_tree_sha != source_tree_sha {
        return Err("repository HEAD/tree changed while QC fixtures were running".into());
    }
    let report = Report {
        schema_version: REPORT_SCHEMA_VERSION,
        producer: "symthaea-quantum-chemistry",
        producer_version: env!("CARGO_PKG_VERSION"),
        source_revision,
        source_tree_sha,
        worktree_clean: true,
        coordinate_unit: "bohr",
        method: "RHF",
        cases,
    };
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    println!();
    Ok(())
}
