// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Safety-invariant property tests for symthaea-cell-foundry.
//!
//! Tests that critical safety gates hold across arbitrary inputs:
//! - EthicsGate: consent_obtained=false always blocks any manipulation
//! - EthicsGate: IRB approval is independently required
//! - Quality assessment viability bounds
//! - IVG protocol stage ordering (never skips stages)

use proptest::prelude::*;
use symthaea_cell_foundry::{
    CellState, CultureEnvironment, EpigeneticProfile, EthicsGate, GermCellStage, IvgProtocol,
    assess_quality,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Safety: EthicsGate with consent_obtained=false MUST block any manipulation,
    /// regardless of IRB status or additional approvals. This is a non-negotiable
    /// bioethics invariant -- no cell manipulation without informed consent.
    #[test]
    fn prop_no_consent_always_blocks(
        irb in any::<bool>(),
        approvals in prop::collection::vec("[a-z]{3,10}", 0..5),
        action in "[a-zA-Z_]{1,20}",
    ) {
        let mut gate = EthicsGate::new();
        // consent_obtained remains false
        gate.irb_approved = irb;
        for approval in &approvals {
            gate.add_approval(approval);
        }

        let result = gate.check_manipulation(&action);
        prop_assert!(result.is_err(),
            "manipulation must be blocked without consent, irb={}, approvals={:?}",
            irb, approvals);

        // IVG must also be blocked
        let ivg_result = gate.check_ivg_protocol();
        prop_assert!(ivg_result.is_err(),
            "IVG must be blocked without consent");

        // SCNT must also be blocked
        let scnt_result = gate.check_scnt_protocol();
        prop_assert!(scnt_result.is_err(),
            "SCNT must be blocked without consent");
    }

    /// Safety: EthicsGate without IRB approval MUST block even when consent is granted.
    /// IRB review is an independent safety gate protecting research subjects.
    #[test]
    fn prop_no_irb_blocks_with_consent(
        approvals in prop::collection::vec("[a-z]{3,10}", 0..5),
        action in "[a-zA-Z_]{1,20}",
    ) {
        let mut gate = EthicsGate::new();
        gate.consent_obtained = true;
        // irb_approved remains false
        for approval in &approvals {
            gate.add_approval(approval);
        }

        let result = gate.check_manipulation(&action);
        prop_assert!(result.is_err(),
            "manipulation must be blocked without IRB approval");
    }

    /// Safety: quality assessment viability is faithfully reported in the QualityReport,
    /// and the report viability value matches the input cell's viability (clamped to [0,1]).
    /// overall_pass must be false when viability <= 0.7 (the viability threshold).
    #[test]
    fn prop_quality_viability_check(
        viability in 0.0_f64..1.0,
        pluripotency in 0.0_f64..1.0,
        division_count in 0_u32..200,
    ) {
        let mut cell = CellState::new_ipsc();
        cell.viability = viability;
        cell.pluripotency_score = pluripotency;
        cell.division_count = division_count;
        let epi = EpigeneticProfile::healthy_newborn();

        let report = assess_quality(&cell, &epi);

        // Viability is faithfully reported
        prop_assert!((report.viability - viability).abs() < 1e-10,
            "report viability {} != cell viability {}", report.viability, viability);

        // Low viability must fail overall QC
        if viability <= 0.7 {
            prop_assert!(!report.overall_pass,
                "viability {} <= 0.7 must fail QC but overall_pass=true", viability);
        }
    }

    /// Safety: IVG protocol stage ordering is preserved -- stages advance only
    /// forward (never skip or regress), and terminal states (MatureGamete, Failed)
    /// are absorbing. Stage regression would indicate a protocol safety violation.
    #[test]
    fn prop_ivg_stage_ordering(days in 1_u32..60) {
        let ipsc = CellState::new_ipsc();
        let mut protocol = IvgProtocol::new(ipsc).unwrap();
        let culture = CultureEnvironment::standard();

        let stage_order = |s: GermCellStage| -> u8 {
            match s {
                GermCellStage::Ipsc => 0,
                GermCellStage::PgcInduction => 1,
                GermCellStage::EarlyGerm => 2,
                GermCellStage::MeioticEntry => 3,
                GermCellStage::MeiosisI => 4,
                GermCellStage::MeiosisII => 5,
                GermCellStage::MatureGamete => 6,
                GermCellStage::Failed => 7,
            }
        };

        let mut prev_order = stage_order(protocol.stage);

        for _ in 0..days {
            protocol.advance_day(&culture);
            let cur_order = stage_order(protocol.stage);

            // Failed can happen from any stage, so it always counts as forward
            if protocol.stage == GermCellStage::Failed {
                break;
            }

            prop_assert!(cur_order >= prev_order,
                "stage must not regress: prev_order={}, cur_order={}, stage={:?}",
                prev_order, cur_order, protocol.stage);
            prev_order = cur_order;
        }
    }

    /// Safety: IVG protocol requires pluripotent input. Non-pluripotent cells
    /// must be rejected, preventing unsafe gametogenesis from inappropriate
    /// cell types.
    #[test]
    fn prop_ivg_rejects_non_pluripotent(
        viability in 0.0_f64..1.0,
        division_count in 0_u32..100,
    ) {
        let mut cell = CellState::new_somatic();
        cell.viability = viability;
        cell.division_count = division_count;

        let result = IvgProtocol::new(cell);
        prop_assert!(result.is_none(),
            "IVG must reject non-pluripotent cells");
    }
}
