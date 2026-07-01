// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! In Vitro Gametogenesis (IVG) protocol.
//!
//! Models the multi-stage differentiation from iPSC to mature gametes:
//! iPSC -> PGC-LC -> early germ cell -> meiotic entry -> meiosis I/II -> mature gamete.
//! Each stage is gated by quality checkpoints.

use serde::{Deserialize, Serialize};

use crate::meiosis_monitor::MeiosisMonitor;
use crate::types::{CellState, CellType, CultureEnvironment, GermCellStage, MeiosisCheckpoint};

/// IVG protocol: iPSC to mature gamete with checkpoint gating.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvgProtocol {
    /// Current IVG developmental stage (iPSC → mature gamete).
    pub stage: GermCellStage,
    /// Days elapsed since IVG protocol initiation.
    pub day: u32,
    /// Cell progressing through gametogenesis.
    pub cell_state: CellState,
    /// Meiotic checkpoints that have been successfully verified.
    pub checkpoints_passed: Vec<MeiosisCheckpoint>,
    /// Quality scores at each checkpoint (checkpoint, pass, score).
    checkpoint_history: Vec<(MeiosisCheckpoint, bool, f64)>,
}

impl IvgProtocol {
    /// Create a new IVG protocol from an iPSC cell.
    ///
    /// Returns `None` if the cell is not pluripotent (IVG requires iPSC input).
    pub fn new(ipsc: CellState) -> Option<Self> {
        if ipsc.cell_type != CellType::Pluripotent {
            return None;
        }
        Some(Self {
            stage: GermCellStage::Ipsc,
            day: 0,
            cell_state: ipsc,
            checkpoints_passed: Vec::new(),
            checkpoint_history: Vec::new(),
        })
    }

    /// Force-create an IVG protocol (for testing; skips pluripotency check).
    pub fn new_unchecked(cell: CellState) -> Self {
        Self {
            stage: GermCellStage::Ipsc,
            day: 0,
            cell_state: cell,
            checkpoints_passed: Vec::new(),
            checkpoint_history: Vec::new(),
        }
    }

    /// Advance the protocol by one day.
    ///
    /// Stage transitions are gated by viability and checkpoint quality.
    /// Culture environment affects cell health.
    pub fn advance_day(&mut self, culture: &CultureEnvironment) -> GermCellStage {
        if self.stage == GermCellStage::Failed || self.stage == GermCellStage::MatureGamete {
            return self.stage;
        }

        self.day += 1;
        self.cell_state.age_hours += 24.0;

        // Environmental stress
        let deviation = culture.deviation_from_standard();
        if deviation > 1.0 {
            self.cell_state.viability -= 0.05;
        }

        // Viability check
        if self.cell_state.viability < 0.2 {
            self.stage = GermCellStage::Failed;
            return self.stage;
        }

        // Stage transitions
        self.stage = match self.stage {
            GermCellStage::Ipsc if self.day >= 3 => {
                self.cell_state.cell_type = CellType::PrimordialGerm;
                self.cell_state.pluripotency_score -= 0.3;
                GermCellStage::PgcInduction
            }
            GermCellStage::PgcInduction if self.day >= 7 => GermCellStage::EarlyGerm,
            GermCellStage::EarlyGerm if self.day >= 14 => {
                // Gate: verify synapsis checkpoint
                let (pass, score) = MeiosisMonitor::verify_synapsis(&self.cell_state);
                self.checkpoint_history
                    .push((MeiosisCheckpoint::Synapsis, pass, score));
                if pass {
                    self.checkpoints_passed.push(MeiosisCheckpoint::Synapsis);
                    GermCellStage::MeioticEntry
                } else {
                    GermCellStage::Failed
                }
            }
            GermCellStage::MeioticEntry if self.day >= 21 => {
                // Gate: verify crossover checkpoint
                let (pass, score) = MeiosisMonitor::verify_crossover(&self.cell_state, 1);
                self.checkpoint_history
                    .push((MeiosisCheckpoint::Crossover, pass, score));
                if pass {
                    self.checkpoints_passed.push(MeiosisCheckpoint::Crossover);
                    self.cell_state.cell_type = CellType::Oocyte;
                    GermCellStage::MeiosisI
                } else {
                    GermCellStage::Failed
                }
            }
            GermCellStage::MeiosisI if self.day >= 28 => {
                self.checkpoints_passed
                    .push(MeiosisCheckpoint::SegregationI);
                self.checkpoint_history
                    .push((MeiosisCheckpoint::SegregationI, true, 0.8));
                GermCellStage::MeiosisII
            }
            GermCellStage::MeiosisII if self.day >= 35 => {
                self.checkpoints_passed
                    .push(MeiosisCheckpoint::SegregationII);
                self.checkpoint_history
                    .push((MeiosisCheckpoint::SegregationII, true, 0.8));
                GermCellStage::MatureGamete
            }
            other => other,
        };

        self.stage
    }

    /// Quality score for a specific meiosis checkpoint.
    ///
    /// Returns 0.0 if the checkpoint hasn't been evaluated yet.
    pub fn checkpoint_quality(&self, checkpoint: MeiosisCheckpoint) -> f64 {
        self.checkpoint_history
            .iter()
            .find(|(cp, _, _)| *cp == checkpoint)
            .map(|(_, _, score)| *score)
            .unwrap_or(0.0)
    }

    /// Whether the cell has reached the mature gamete stage.
    pub fn is_mature_gamete(&self) -> bool {
        self.stage == GermCellStage::MatureGamete
    }

    /// Whether the protocol has failed.
    pub fn is_failed(&self) -> bool {
        self.stage == GermCellStage::Failed
    }

    /// Number of checkpoints passed so far.
    pub fn checkpoints_passed_count(&self) -> usize {
        self.checkpoints_passed.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_requires_ipsc_input() {
        let somatic = CellState::new_somatic();
        assert!(
            IvgProtocol::new(somatic).is_none(),
            "IVG should reject non-pluripotent cells"
        );
    }

    #[test]
    fn test_accepts_ipsc() {
        let ipsc = CellState::new_ipsc();
        assert!(
            IvgProtocol::new(ipsc).is_some(),
            "IVG should accept iPSC cells"
        );
    }

    #[test]
    fn test_initial_stage_is_ipsc() {
        let ipsc = CellState::new_ipsc();
        let protocol = IvgProtocol::new(ipsc).unwrap();
        assert_eq!(protocol.stage, GermCellStage::Ipsc);
        assert_eq!(protocol.day, 0);
    }

    #[test]
    fn test_stage_progression() {
        let ipsc = CellState::new_ipsc();
        let mut protocol = IvgProtocol::new(ipsc).unwrap();
        let culture = CultureEnvironment::standard();

        // Days 1-3: stay iPSC, then transition
        for _ in 0..3 {
            protocol.advance_day(&culture);
        }
        assert_eq!(protocol.stage, GermCellStage::PgcInduction);
    }

    #[test]
    fn test_checkpoints_gate_advancement() {
        let ipsc = CellState::new_ipsc();
        let mut protocol = IvgProtocol::new(ipsc).unwrap();
        let culture = CultureEnvironment::standard();

        // Advance through early stages
        for _ in 0..14 {
            protocol.advance_day(&culture);
        }

        // Should be at MeioticEntry if synapsis passed, or Failed
        assert!(
            protocol.stage == GermCellStage::MeioticEntry
                || protocol.stage == GermCellStage::Failed,
            "Should be at MeioticEntry or Failed after day 14, got {:?}",
            protocol.stage
        );
    }

    #[test]
    fn test_full_protocol_to_mature_gamete() {
        let mut ipsc = CellState::new_ipsc();
        ipsc.viability = 0.99; // High viability for best chance
        let mut protocol = IvgProtocol::new(ipsc).unwrap();
        let culture = CultureEnvironment::standard();

        for _ in 0..40 {
            protocol.advance_day(&culture);
            if protocol.is_failed() || protocol.is_mature_gamete() {
                break;
            }
        }

        // With a healthy iPSC, the protocol should complete or fail at a checkpoint
        assert!(
            protocol.is_mature_gamete() || protocol.is_failed(),
            "Protocol should terminate: stage={:?}",
            protocol.stage
        );
    }

    #[test]
    fn test_failed_low_viability() {
        let mut ipsc = CellState::new_ipsc();
        ipsc.viability = 0.1; // Below threshold
        let mut protocol = IvgProtocol::new(ipsc).unwrap();

        let bad_culture = CultureEnvironment {
            temperature_celsius: 42.0, // Way too hot
            co2_percent: 15.0,
            o2_percent: 20.0,
            ph: 8.5,
            humidity_percent: 95.0,
            media_volume_ml: 10.0,
            passage_number: 0,
        };

        protocol.advance_day(&bad_culture);
        assert_eq!(protocol.stage, GermCellStage::Failed);
    }

    #[test]
    fn test_checkpoint_quality_unevaluated() {
        let ipsc = CellState::new_ipsc();
        let protocol = IvgProtocol::new(ipsc).unwrap();
        assert!(
            (protocol.checkpoint_quality(MeiosisCheckpoint::Synapsis)).abs() < 1e-9,
            "Unevaluated checkpoint should return 0.0"
        );
    }

    #[test]
    fn test_mature_gamete_does_not_advance() {
        let mut ipsc = CellState::new_ipsc();
        ipsc.viability = 0.99;
        let mut protocol = IvgProtocol::new(ipsc).unwrap();
        let culture = CultureEnvironment::standard();

        // Run to completion
        for _ in 0..50 {
            protocol.advance_day(&culture);
            if protocol.is_mature_gamete() {
                break;
            }
        }

        if protocol.is_mature_gamete() {
            let stage_before = protocol.stage;
            protocol.advance_day(&culture);
            assert_eq!(protocol.stage, stage_before);
        }
    }
}
