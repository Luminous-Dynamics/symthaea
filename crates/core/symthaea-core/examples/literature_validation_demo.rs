// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Literature Validation Demo
//!
//! Validates Spark Engine physics parameters against published literature.
//!
//! ## Data Sources
//!
//! - **ENDF/B-VIII.0**: Evaluated Nuclear Data File for neutron cross-sections
//! - **ASM Handbook**: Materials properties
//! - **CRC Handbook**: Chemistry and physics constants
//! - **Miracle & Senkov (2017)**: HEA properties
//! - **Was (2017)**: Radiation materials science
//! - **NRL Plasma Formulary**: Fusion parameters
//!
//! Run with: cargo run --example literature_validation_demo

use symthaea_core::physics::LiteratureValidation;

fn main() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              SPARK ENGINE PHYSICS VALIDATION                         ║");
    println!("║              Cross-checking Against Published Literature             ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Run all validations
    let validation = LiteratureValidation::validate_all();

    // Print detailed results
    validation.print_summary();

    // Print references
    println!();
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│                          LITERATURE SOURCES                            │");
    println!("├────────────────────────────────────────────────────────────────────────┤");
    println!("│                                                                        │");
    println!("│  NEUTRON CROSS-SECTIONS:                                               │");
    println!("│  [1] Brown et al., Nuclear Data Sheets 148 (2018) 1-142               │");
    println!("│      \"ENDF/B-VIII.0: The 8th Major Release\"                           │");
    println!("│      DOI: 10.1016/j.nds.2018.02.001                                    │");
    println!("│                                                                        │");
    println!("│  [2] IAEA Technical Report Series No. 283 (1988)                       │");
    println!("│      \"Neutron Fluence-to-Dose Conversion Factors\"                     │");
    println!("│                                                                        │");
    println!("│  THERMAL PROPERTIES:                                                   │");
    println!("│  [3] ASM Handbook Vol. 2: Properties and Selection (2020)              │");
    println!("│                                                                        │");
    println!("│  [4] CRC Handbook of Chemistry and Physics, 104th ed. (2023-2024)      │");
    println!("│                                                                        │");
    println!("│  HIGH-ENTROPY ALLOYS:                                                  │");
    println!("│  [5] Miracle & Senkov, Acta Materialia 122 (2017) 448-511             │");
    println!("│      \"A critical review of high entropy alloys\"                       │");
    println!("│      DOI: 10.1016/j.actamat.2016.08.081                                │");
    println!("│                                                                        │");
    println!("│  RADIATION DAMAGE:                                                     │");
    println!("│  [6] Was, Fundamentals of Radiation Materials Science, 2nd ed. (2017)  │");
    println!("│      Springer, DOI: 10.1007/978-1-4939-3438-6                          │");
    println!("│                                                                        │");
    println!("│  [7] Zinkle & Snead, Annu. Rev. Mater. Res. 44 (2014) 241-267         │");
    println!("│      \"Structural materials for fission & fusion energy\"               │");
    println!("│      DOI: 10.1146/annurev-matsci-070813-113627                         │");
    println!("│                                                                        │");
    println!("│  [8] Stoller et al., J. Nucl. Mater. 438 (2013) S218-S222             │");
    println!("│      \"On the use of SRIM for computing radiation damage exposure\"     │");
    println!("│      DOI: 10.1016/j.jnucmat.2013.01.034                                │");
    println!("│                                                                        │");
    println!("│  FUSION PHYSICS:                                                       │");
    println!("│  [9] NRL Plasma Formulary, Naval Research Laboratory (2019)            │");
    println!("│                                                                        │");
    println!("│  [10] ITER Physics Basis, Nucl. Fusion 39 (1999) 2137-2638            │");
    println!("│       DOI: 10.1088/0029-5515/39/12/301                                 │");
    println!("│                                                                        │");
    println!("└────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Final assessment
    let pass_rate = validation.overall_pass_rate();

    if pass_rate >= 0.95 {
        println!("ASSESSMENT: Physics parameters are well-calibrated against literature.");
        println!("            Spark Engine simulations use validated, peer-reviewed values.");
    } else if pass_rate >= 0.85 {
        println!("ASSESSMENT: Most physics parameters validated. Some approximations used");
        println!("            for complex phenomena (see individual parameter notes).");
    } else {
        println!("ASSESSMENT: Some parameters need review. Check parameters marked ✗.");
    }

    println!();
}
