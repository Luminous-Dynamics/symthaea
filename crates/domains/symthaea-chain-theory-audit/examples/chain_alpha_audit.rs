// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_chain_theory_audit::reproduce;

fn main() {
    let audit = reproduce();
    let sectors = audit.sectors;
    println!("Symthaea Chain Theory audit — combinatorial reproduction");
    println!("epistemic_status=REPRODUCED_MODEL_CALCULATION_NOT_PHYSICAL_VALIDATION");
    println!("product_assignments={}", audit.product_assignments);
    println!("product_records={}", audit.product_records);
    println!("commutator_records={}", audit.commutator_records);
    println!("total_records={}", audit.total_records);
    println!(
        "sectors_x_plus_y_plus={}",
        sectors.plus_plus
    );
    println!(
        "sectors_x_plus_y_minus={}",
        sectors.plus_minus
    );
    println!(
        "sectors_x_minus_y_plus={}",
        sectors.minus_plus
    );
    println!(
        "sectors_x_minus_y_minus={}",
        sectors.minus_minus
    );
    println!("radical_coefficients=79*sqrt(1)+48*sqrt(2)+42*sqrt(3)+3*sqrt(6)");
    println!("readout_dimension={}", audit.readout_dimension);
    println!("alpha_candidate={:.15}", audit.alpha_candidate);
    println!("inverse_alpha_candidate={:.15}", audit.inverse_alpha_candidate);
    println!("codata_2022_inverse={:.12}", audit.codata_2022_inverse);
    println!(
        "difference_in_codata_2022_standard_uncertainties={:.6}",
        audit.difference_in_standard_uncertainties
    );
    println!("This reproduces the stated formula, not its identification with electromagnetism or photons.");
}
