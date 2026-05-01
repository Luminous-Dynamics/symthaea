// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-resolution temporal binning for deep-time biosphere modeling.
//!
//! Covers 3,800 Ma (origin of life) to 0.003 Ma (3000 BCE) with variable
//! resolution: coarse for the Precambrian, ICS stage-level for the Phanerozoic,
//! and millennium-level for the Holocene.

use serde::{Deserialize, Serialize};

/// Age in millions of years before present.
pub type MaAge = f64;

/// Resolution category, determines uncertainty scaling and data expectations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeResolution {
    /// 3800-541 Ma: ~100 Ma bins. Very sparse fossil record.
    Precambrian,
    /// 541-0.0117 Ma: ICS stages, ~5-20 Ma bins. Good fossil record.
    PhanerozoicStage,
    /// 0.0117-0.003 Ma (11,700 BP - 3000 BCE): millennium-level bins.
    Holocene,
}

/// A temporal bin with variable resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalBin {
    /// Start age (Ma, older boundary).
    pub start_ma: MaAge,
    /// End age (Ma, younger boundary).
    pub end_ma: MaAge,
    /// Midpoint age for plotting/interpolation.
    pub midpoint_ma: MaAge,
    /// Resolution category.
    pub resolution: TimeResolution,
    /// ICS stage name or informal era name.
    pub name: String,
}

impl TemporalBin {
    pub fn new(start_ma: f64, end_ma: f64, name: &str, resolution: TimeResolution) -> Self {
        Self {
            start_ma,
            end_ma,
            midpoint_ma: (start_ma + end_ma) / 2.0,
            resolution,
            name: name.to_string(),
        }
    }

    /// Duration of this bin in Ma.
    pub fn duration_ma(&self) -> f64 {
        self.start_ma - self.end_ma
    }
}

/// Build the canonical sequence of temporal bins from 3800 Ma to 0.003 Ma.
///
/// Returns approximately 80 bins sorted oldest-first (descending age).
pub fn build_deep_time_bins() -> Vec<TemporalBin> {
    let mut bins = Vec::with_capacity(85);

    // ── Precambrian (3800 - 541 Ma) ──
    // Hadean
    bins.push(TemporalBin::new(
        3800.0,
        3500.0,
        "Early Hadean",
        TimeResolution::Precambrian,
    ));
    bins.push(TemporalBin::new(
        3500.0,
        3200.0,
        "Late Hadean/Early Archean",
        TimeResolution::Precambrian,
    ));

    // Archean
    bins.push(TemporalBin::new(
        3200.0,
        2800.0,
        "Mesoarchean",
        TimeResolution::Precambrian,
    ));
    bins.push(TemporalBin::new(
        2800.0,
        2500.0,
        "Neoarchean",
        TimeResolution::Precambrian,
    ));

    // Proterozoic
    bins.push(TemporalBin::new(
        2500.0,
        2400.0,
        "Pre-GOE Paleoproterozoic",
        TimeResolution::Precambrian,
    ));
    bins.push(TemporalBin::new(
        2400.0,
        2300.0,
        "Great Oxidation Event",
        TimeResolution::Precambrian,
    ));
    bins.push(TemporalBin::new(
        2300.0,
        2050.0,
        "Post-GOE/Huronian Glaciation",
        TimeResolution::Precambrian,
    ));
    bins.push(TemporalBin::new(
        2050.0,
        1800.0,
        "Late Paleoproterozoic",
        TimeResolution::Precambrian,
    ));
    bins.push(TemporalBin::new(
        1800.0,
        1600.0,
        "Early Mesoproterozoic",
        TimeResolution::Precambrian,
    ));
    bins.push(TemporalBin::new(
        1600.0,
        1400.0,
        "Mid Mesoproterozoic",
        TimeResolution::Precambrian,
    ));
    bins.push(TemporalBin::new(
        1400.0,
        1200.0,
        "Late Mesoproterozoic",
        TimeResolution::Precambrian,
    ));
    bins.push(TemporalBin::new(
        1200.0,
        1000.0,
        "Stenian",
        TimeResolution::Precambrian,
    ));
    bins.push(TemporalBin::new(
        1000.0,
        850.0,
        "Tonian (early)",
        TimeResolution::Precambrian,
    ));
    bins.push(TemporalBin::new(
        850.0,
        720.0,
        "Tonian (late)",
        TimeResolution::Precambrian,
    ));
    bins.push(TemporalBin::new(
        720.0,
        635.0,
        "Cryogenian (Snowball Earth)",
        TimeResolution::Precambrian,
    ));
    bins.push(TemporalBin::new(
        635.0,
        541.0,
        "Ediacaran",
        TimeResolution::Precambrian,
    ));

    // ── Phanerozoic (541 - 0.0117 Ma): ICS stages ──

    // Cambrian
    bins.push(TemporalBin::new(
        541.0,
        521.0,
        "Terreneuvian",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        521.0,
        509.0,
        "Cambrian Series 2",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        509.0,
        497.0,
        "Miaolingian",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        497.0,
        485.4,
        "Furongian",
        TimeResolution::PhanerozoicStage,
    ));

    // Ordovician
    bins.push(TemporalBin::new(
        485.4,
        470.0,
        "Early Ordovician",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        470.0,
        458.4,
        "Middle Ordovician",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        458.4,
        443.8,
        "Late Ordovician",
        TimeResolution::PhanerozoicStage,
    ));

    // Silurian
    bins.push(TemporalBin::new(
        443.8,
        433.4,
        "Llandovery",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        433.4,
        427.4,
        "Wenlock",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        427.4,
        419.2,
        "Ludlow-Pridoli",
        TimeResolution::PhanerozoicStage,
    ));

    // Devonian
    bins.push(TemporalBin::new(
        419.2,
        393.3,
        "Early Devonian",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        393.3,
        382.7,
        "Middle Devonian",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        382.7,
        358.9,
        "Late Devonian",
        TimeResolution::PhanerozoicStage,
    ));

    // Carboniferous
    bins.push(TemporalBin::new(
        358.9,
        323.2,
        "Mississippian",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        323.2,
        298.9,
        "Pennsylvanian",
        TimeResolution::PhanerozoicStage,
    ));

    // Permian
    bins.push(TemporalBin::new(
        298.9,
        272.95,
        "Cisuralian",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        272.95,
        259.51,
        "Guadalupian",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        259.51,
        251.9,
        "Lopingian",
        TimeResolution::PhanerozoicStage,
    ));

    // Triassic
    bins.push(TemporalBin::new(
        251.9,
        247.2,
        "Early Triassic",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        247.2,
        237.0,
        "Middle Triassic",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        237.0,
        201.4,
        "Late Triassic",
        TimeResolution::PhanerozoicStage,
    ));

    // Jurassic
    bins.push(TemporalBin::new(
        201.4,
        174.7,
        "Early Jurassic",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        174.7,
        161.5,
        "Middle Jurassic",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        161.5,
        145.0,
        "Late Jurassic",
        TimeResolution::PhanerozoicStage,
    ));

    // Cretaceous
    bins.push(TemporalBin::new(
        145.0,
        100.5,
        "Early Cretaceous",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        100.5,
        66.0,
        "Late Cretaceous",
        TimeResolution::PhanerozoicStage,
    ));

    // Paleogene
    bins.push(TemporalBin::new(
        66.0,
        56.0,
        "Paleocene",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        56.0,
        33.9,
        "Eocene",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        33.9,
        23.03,
        "Oligocene",
        TimeResolution::PhanerozoicStage,
    ));

    // Neogene
    bins.push(TemporalBin::new(
        23.03,
        5.333,
        "Miocene",
        TimeResolution::PhanerozoicStage,
    ));
    bins.push(TemporalBin::new(
        5.333,
        2.58,
        "Pliocene",
        TimeResolution::PhanerozoicStage,
    ));

    // Quaternary (pre-Holocene)
    bins.push(TemporalBin::new(
        2.58,
        0.0117,
        "Pleistocene",
        TimeResolution::PhanerozoicStage,
    ));

    // ── Holocene (11,700 BP - 3,000 BCE = 0.0117 - 0.005 Ma) ──
    bins.push(TemporalBin::new(
        0.0117,
        0.009,
        "Early Holocene (Greenlandian)",
        TimeResolution::Holocene,
    ));
    bins.push(TemporalBin::new(
        0.009,
        0.007,
        "Middle Holocene (Northgrippian)",
        TimeResolution::Holocene,
    ));
    bins.push(TemporalBin::new(
        0.007,
        0.005,
        "Late Holocene (Meghalayan)",
        TimeResolution::Holocene,
    ));

    bins
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bins_cover_full_range() {
        let bins = build_deep_time_bins();
        let oldest = bins.first().unwrap().start_ma;
        let youngest = bins.last().unwrap().end_ma;
        assert!((oldest - 3800.0).abs() < 0.01, "Should start at 3800 Ma");
        assert!(youngest < 0.006, "Should end near 3000 BCE (0.005 Ma)");
    }

    #[test]
    fn bins_sorted_oldest_first() {
        let bins = build_deep_time_bins();
        for pair in bins.windows(2) {
            assert!(
                pair[0].start_ma >= pair[1].start_ma,
                "Bins must be sorted oldest-first: {} before {}",
                pair[0].name,
                pair[1].name
            );
        }
    }

    #[test]
    fn no_gaps_between_bins() {
        let bins = build_deep_time_bins();
        for pair in bins.windows(2) {
            let gap = (pair[0].end_ma - pair[1].start_ma).abs();
            assert!(
                gap < 0.5,
                "Gap of {} Ma between {} and {}",
                gap,
                pair[0].name,
                pair[1].name
            );
        }
    }

    #[test]
    fn bin_count_reasonable() {
        let bins = build_deep_time_bins();
        assert!(
            bins.len() >= 50,
            "Expected at least 50 bins, got {}",
            bins.len()
        );
        assert!(
            bins.len() <= 100,
            "Expected at most 100 bins, got {}",
            bins.len()
        );
    }

    #[test]
    fn all_bins_have_positive_duration() {
        for bin in build_deep_time_bins() {
            assert!(
                bin.duration_ma() > 0.0,
                "Bin {} has non-positive duration",
                bin.name
            );
        }
    }
}
