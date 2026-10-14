// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Produce source-bound closed-shell DFT evidence across the grid presets.
//!
//! This executable deliberately reports SCF convergence, the checked
//! eigensolver receipt, and electron-count integration separately. It does
//! not claim agreement with PySCF or chemical accuracy.
//!
//! Example:
//!   cargo run --locked -p symthaea-quantum-chemistry --example qc_dft_molecular_grid_evidence
//!
//! The workflow sets SYMTHAEA_SOURCE_SHA / SYMTHAEA_SOURCE_TREE after checking
//! the exact checkout. When run locally without those variables, the JSON
//! records null rather than inventing source identity.

use serde_json::{Value, json};
use std::io::{self, Write};
use symthaea_quantum_chemistry::basis::{BasisSetProvider, sto3g::Sto3g};
use symthaea_quantum_chemistry::dft::grid::GridQuality;
use symthaea_quantum_chemistry::dft::xc::{DftConfig, XcFunctional, try_kohn_sham_dft};
use symthaea_quantum_chemistry::molecule::{Atom, Molecule};

const GRID_QUALITIES: [(&str, GridQuality); 4] = [
    ("coarse", GridQuality::Coarse),
    ("medium", GridQuality::Medium),
    ("fine", GridQuality::Fine),
    ("high_accuracy", GridQuality::HighAccuracy),
];

fn source_value(name: &str) -> Value {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(Value::String)
        .unwrap_or(Value::Null)
}

fn geometry_json(molecule: &Molecule) -> Value {
    json!({
        "coordinates_unit": "bohr",
        "atoms": molecule.atoms.iter().map(|atom| json!({
            "atomic_number": atom.atomic_number,
            "position": atom.position,
        })).collect::<Vec<_>>(),
        "charge": molecule.charge,
        "multiplicity": molecule.multiplicity,
        "electron_count": molecule.n_electrons(),
    })
}

fn diagnostics_json(result: &symthaea_quantum_chemistry::dft::xc::DftResult) -> Value {
    match &result.solver_diagnostics {
        Some(receipt) => json!({
            "generalized_eigensolve_count": receipt.generalized_eigensolve_count,
            "total_jacobi_rotations": receipt.total_jacobi_rotations,
            "max_transformed_off_diagonal_hartree":
                receipt.max_transformed_off_diagonal_hartree,
            "max_transformed_eigenpair_residual":
                receipt.max_transformed_eigenpair_residual,
            "max_eigenvector_orthogonality_residual":
                receipt.max_eigenvector_orthogonality_residual,
            "max_generalized_eigenpair_residual":
                receipt.max_generalized_eigenpair_residual,
            "overlap_orthogonality_residual":
                receipt.overlap_orthogonality_residual,
        }),
        None => Value::Null,
    }
}

fn main() {
    let molecules = vec![
        (
            "He",
            Molecule::new(vec![Atom::new(2, 0.0, 0.0, 0.0)]),
            "single_atom_closed_shell",
        ),
        ("H2", Molecule::h2(), "symmetric_singlet_r_1_4_bohr"),
        ("H2O", Molecule::water(), "water_0_9572_angstrom_104_52_deg"),
    ];

    let mut records = Vec::new();
    let mut all_solver_receipts_present = true;
    let mut all_scf_converged = true;
    let mut all_electron_integrals_finite = true;

    for (molecule_name, molecule, geometry_id) in &molecules {
        let basis = Sto3g::build(molecule);
        for (grid_name, grid_quality) in GRID_QUALITIES {
            let mut config = DftConfig::default();
            config.functional = XcFunctional::Lda;
            config.grid_quality = grid_quality;
            // Use strict benchmark-only SCF tolerances. Production defaults
            // remain unchanged; tighter settings reduce SCF error before
            // comparing energies across independently implemented engines.
            config.max_iterations = 200;
            config.energy_convergence = 1e-10;
            config.density_convergence = 1e-8;

            let common = json!({
                "molecule": molecule_name,
                "geometry_id": geometry_id,
                "geometry": geometry_json(molecule),
                "basis": "STO-3G",
                "functional": "Slater exchange + VWN5 correlation (LDA)",
                "grid_quality": grid_name,
                "scf_energy_convergence": config.energy_convergence,
                "scf_density_convergence": config.density_convergence,
                "scf_max_iterations": config.max_iterations,
                "scf_diis_enabled": config.use_diis,
                "expected_grid_points_per_atom": match grid_name {
                    "coarse" => 20 * 6,
                    "medium" => 35 * 26,
                    "fine" => 50 * 50,
                    "high_accuracy" => 75 * 302,
                    _ => unreachable!(),
                },
            });

            match try_kohn_sham_dft(molecule, &basis, &config) {
                Ok(result) => {
                    let receipt_present = result.solver_diagnostics.is_some();
                    let electron_integral_finite = result.integrated_electron_count.is_finite()
                        && result.integrated_electron_count_abs_error.is_finite();
                    all_solver_receipts_present &= receipt_present;
                    all_scf_converged &= result.converged;
                    all_electron_integrals_finite &= electron_integral_finite;

                    let mut record = common;
                    record["status"] = json!(if result.converged {
                        "scf_converged"
                    } else {
                        "scf_not_converged"
                    });
                    record["total_energy_hartree"] = json!(result.total_energy);
                    record["electronic_energy_hartree"] = json!(result.electronic_energy);
                    record["exchange_correlation_energy_hartree"] = json!(result.xc_energy);
                    record["nuclear_repulsion_hartree"] = json!(result.nuclear_repulsion);
                    record["n_basis"] = json!(result.n_basis);
                    record["n_grid_points"] = json!(result.n_grid_points);
                    record["scf_iterations"] = json!(result.n_iterations);
                    record["integrated_electron_count"] = json!(result.integrated_electron_count);
                    record["expected_electron_count"] = json!(molecule.n_electrons());
                    record["integrated_electron_count_abs_error"] =
                        json!(result.integrated_electron_count_abs_error);
                    record["solver_receipt_present"] = json!(receipt_present);
                    record["solver_diagnostics"] = diagnostics_json(&result);
                    records.push(record);
                }
                Err(error) => {
                    all_solver_receipts_present = false;
                    all_scf_converged = false;
                    all_electron_integrals_finite = false;
                    let mut record = common;
                    record["status"] = json!("checked_solver_error");
                    record["error"] = json!(error.to_string());
                    record["total_energy_hartree"] = Value::Null;
                    record["integrated_electron_count"] = Value::Null;
                    record["integrated_electron_count_abs_error"] = Value::Null;
                    record["solver_receipt_present"] = json!(false);
                    records.push(record);
                }
            }
        }
    }

    let pass = all_solver_receipts_present
        && all_scf_converged
        && all_electron_integrals_finite
        && records.len() == molecules.len() * GRID_QUALITIES.len();
    let report = json!({
        "schema": "symthaea.qc-dft-molecular-grid-evidence.v1",
        "source_commit": source_value("SYMTHAEA_SOURCE_SHA"),
        "source_tree": source_value("SYMTHAEA_SOURCE_TREE"),
        "basis": "STO-3G",
        "functional": "LDA (Slater exchange + canonical VWN5 correlation)",
        "energy_unit": "hartree",
        "grid_weight_unit": "bohr^3",
        "grid_electron_count_note":
            "Quadrature diagnostic only; reported error is not treated as zero or as a chemical-accuracy certificate.",
        "evidence_scope":
            "Native checked solver and grid evidence; not yet an independent PySCF molecular-energy comparison.",
        "summary": {
            "cases": records.len(),
            "solver_receipts_present_all_cases": all_solver_receipts_present,
            "scf_converged_all_cases": all_scf_converged,
            "electron_count_integrals_finite_all_cases": all_electron_integrals_finite,
            "all_cases_passed_execution_gate": pass,
        },
        "records": records,
    });

    let mut stdout = io::stdout().lock();
    if serde_json::to_writer_pretty(&mut stdout, &report).is_err()
        || writeln!(&mut stdout).is_err()
        || stdout.flush().is_err()
    {
        eprintln!("failed to write DFT molecular-grid evidence JSON");
        std::process::exit(2);
    }

    if !pass {
        std::process::exit(1);
    }
}
