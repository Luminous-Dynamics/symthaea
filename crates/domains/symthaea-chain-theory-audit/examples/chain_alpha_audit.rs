// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_chain_theory_audit::{
    pair_electric_charge_audit, readout_dimension_matches_running_ladder, record_rows, reproduce,
    unbroken_generator_decomposition, weak_angle_normalization_audit,
};

fn main() {
    let audit = reproduce();
    let sectors = audit.sectors;
    println!("Symthaea Chain Theory audit — combinatorial reproduction");
    println!("epistemic_status=REPRODUCED_MODEL_CALCULATION_NOT_PHYSICAL_VALIDATION");
    println!("product_assignments={}", audit.product_assignments);
    println!("product_records={}", audit.product_records);
    println!("commutator_records={}", audit.commutator_records);
    println!("total_records={}", audit.total_records);
    println!("sectors_x_plus_y_plus={}", sectors.plus_plus);
    println!("sectors_x_plus_y_minus={}", sectors.plus_minus);
    println!("sectors_x_minus_y_plus={}", sectors.minus_plus);
    println!("sectors_x_minus_y_minus={}", sectors.minus_minus);
    println!("radical_coefficients=79*sqrt(1)+48*sqrt(2)+42*sqrt(3)+3*sqrt(6)");
    println!("readout_dimension={}", audit.readout_dimension);
    println!("alpha_candidate={:.15}", audit.alpha_candidate);
    println!(
        "inverse_alpha_candidate={:.15}",
        audit.inverse_alpha_candidate
    );
    println!("codata_2022_inverse={:.12}", audit.codata_2022_inverse);
    println!(
        "difference_in_codata_2022_standard_uncertainties={:.6}",
        audit.difference_in_standard_uncertainties
    );
    println!(
        "This reproduces the stated formula, not its identification with electromagnetism or photons."
    );
    println!();
    println!("Physical-interpretation consistency checks (conditional on the paper's assignments)");
    let generator = unbroken_generator_decomposition();
    println!(
        "unbroken_generator=1/3*(6Y)+{}/3*(2T3)+{}/3*colour exact={}",
        generator.weak_isospin_coefficient_numerator,
        generator.colour_coefficient_numerator,
        generator.decomposition_exact
    );
    println!(
        "generator_has_weak_component={} generator_has_colour_component={}",
        !generator.weak_isospin_coefficient_is_zero, generator.colour_coefficient_is_nonzero
    );
    let pair = pair_electric_charge_audit();
    println!(
        "pair_term_electric_charges={}/6,{}/6 pair_ket_is_charge_eigenstate={}",
        pair.first_term_charge_sixths,
        pair.second_term_charge_sixths,
        pair.pair_ket_is_electric_charge_eigenstate
    );
    let weak_angle = weak_angle_normalization_audit();
    println!(
        "implied_inverse_alpha_em_per_Y={:.9} per_2Y={:.9} per_6Y={:.9} GUT={:.9}",
        weak_angle.inverse_alpha_em_per_y,
        weak_angle.inverse_alpha_em_per_two_y,
        weak_angle.inverse_alpha_em_per_six_y,
        weak_angle.inverse_alpha_em_gut_normalized
    );
    println!(
        "required_hypercharge_scale_sqrt={:.9} readout_dimension_is_power_of_319={}",
        weak_angle.required_alpha_y_scale_sqrt,
        readout_dimension_matches_running_ladder()
    );
    println!(
        "independent_audit_source=SproutSeeds/chain-theory-audit@95cd02da5955450397a848c25f8287c1a037d9f2"
    );
    println!(
        "physical_scope=CONDITIONAL_GENERATOR_AND_NORMALIZATION_CHECKS_NOT_A_COMPLETE_QED_ADJUDICATION"
    );

    println!();
    println!("record,construction,slot_kinds,X,Y,ends_passed,k,weight");
    for row in record_rows() {
        let weight = match row.weight_squared {
            1 => "1",
            2 => "sqrt(2)",
            3 => "sqrt(3)",
            6 => "sqrt(6)",
            _ => unreachable!("the four exchange sectors define k"),
        };
        println!(
            "R{:03},{},{},{:+},{:+},{},{},{}",
            row.ordinal,
            row.construction,
            row.slot_kinds,
            row.signature_x,
            row.signature_y,
            row.ends_passed,
            row.weight_squared,
            weight,
        );
    }
}
