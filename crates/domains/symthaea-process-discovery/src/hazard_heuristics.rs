// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic structural hazard heuristics.
//!
//! Deliberately contains **no list of scheduled/controlled-substance
//! structures** -- see `symthaea/CHEMICAL_PROCESS_DISCOVERY_PLAN_2026-07-12.md`
//! Phase 1's Context section for why. What's here is the same category of
//! technique published cheminformatics hazard-screening tools use ("explosophore
//! counting" -- nitro-group density, peroxide motifs, extreme unsaturation) --
//! generic structural signals, not a targeting list. `ExternalScopeConfig`
//! is the extension point for anyone who wants to vendor in an authoritative
//! external reference (e.g. a curated SMARTS list); this module ships with
//! that extension point wired but unpopulated.
//!
//! These are heuristics, not a certification: a molecule scoring low here is
//! not "confirmed safe," only "not caught by these specific generic signals."
//! `AllowlistOnlyPolicy` (`policy.rs`) is the actually-strong guarantee; this
//! module exists for the two policies that don't have that guarantee.

use std::path::PathBuf;
use symthaea_organic_chemistry::smiles::{BondOrder, Molecule};

/// Extension point for an externally-curated reference (not populated here).
#[derive(Debug, Clone, Default)]
pub struct ExternalScopeConfig {
    /// Path to a user-vendored SMARTS/structure reference file. `None` by
    /// default -- populating this and wiring a real matcher against it is
    /// explicitly left to whoever wants CWC/DEA-list-grade coverage; this
    /// crate does not ship one.
    pub extra_patterns_path: Option<PathBuf>,
}

/// Per-signal hazard score, each roughly in `[0.0, 1.0]` (unbounded above for
/// pathological inputs) so a policy can combine or threshold them individually
/// rather than being handed one opaque number.
#[derive(Debug, Clone, Copy, Default)]
pub struct HazardScore {
    /// Nitro groups (N with one N=O and one N-O), raw count. Gates on its
    /// own -- **any** nitro motif triggers review regardless of molecule
    /// size. A density-only rule (fixed 2026-07-12, second pass) missed a
    /// single nitro group on a 7+ heavy-atom molecule, which directly
    /// contradicted the "deliberately conservative, loud not precise"
    /// design intent -- a real bug an external review caught.
    pub nitro_count: u32,
    /// Nitro groups per heavy atom -- informational severity signal only,
    /// does not gate on its own.
    pub nitro_density: f64,
    /// O-O single-bond (peroxide) motifs, raw count.
    pub peroxide_count: u32,
    /// Degree-of-unsaturation per heavy atom (very high values correlate
    /// with high-energy-density structures, though most high values are
    /// just ordinary aromatics/polyenes -- a weak signal on its own).
    pub unsaturation_ratio: f64,
}

impl HazardScore {
    /// True if any individual signal crosses its conservative threshold.
    /// Deliberately conservative (low thresholds, more false positives) --
    /// this gate is meant to be loud, not precise; a human reviews every
    /// surviving certificate anyway.
    pub fn exceeds_conservative_threshold(&self) -> bool {
        self.nitro_count > 0 || self.peroxide_count > 0 || self.unsaturation_ratio > 0.9
    }
}

/// Count nitro groups: an N atom with exactly one N=O double bond and one
/// N-O single bond (organic-chemistry's `groups::detect` has no nitro
/// variant -- this is a real, separate gap found while building this
/// module, not a duplicate of existing coverage).
fn count_nitro_groups(m: &Molecule) -> u32 {
    let mut count = 0;
    for (i, atom) in m.atoms.iter().enumerate() {
        if atom.element != "N" {
            continue;
        }
        let neighbors = m.neighbors(i);
        let double_o = neighbors
            .iter()
            .filter(|(j, o)| *o == BondOrder::Double && m.atoms[*j].element == "O")
            .count();
        let single_o = neighbors
            .iter()
            .filter(|(j, o)| *o == BondOrder::Single && m.atoms[*j].element == "O")
            .count();
        // Two common depictions: neutral pentavalent N(=O)=O, or
        // charge-separated [N+](=O)[O-] -- both flagged.
        if double_o >= 2 || (double_o >= 1 && single_o >= 1) {
            count += 1;
        }
    }
    count
}

/// Count O-O single-bond (peroxide) motifs.
fn count_peroxide_motifs(m: &Molecule) -> u32 {
    m.bonds
        .iter()
        .filter(|b| {
            b.order == BondOrder::Single
                && m.atoms[b.a].element == "O"
                && m.atoms[b.b].element == "O"
        })
        .count() as u32
}

/// Score a single molecule against the generic heuristics.
pub fn score(m: &Molecule) -> HazardScore {
    let heavy_atoms = m.atoms.len().max(1) as f64;
    let nitro_count = count_nitro_groups(m);
    HazardScore {
        nitro_count,
        nitro_density: nitro_count as f64 / heavy_atoms,
        peroxide_count: count_peroxide_motifs(m),
        unsaturation_ratio: m.degree_of_unsaturation() as f64 / heavy_atoms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benign_feedstocks_score_low() {
        for smiles in ["C=C", "CCO", "CC(=O)O", "c1ccccc1"] {
            let m = Molecule::from_smiles(smiles).unwrap();
            let s = score(&m);
            assert!(
                !s.exceeds_conservative_threshold(),
                "{smiles} unexpectedly flagged: {s:?}"
            );
        }
    }

    #[test]
    fn nitro_group_detected() {
        // Nitromethane: CH3-NO2, written as C[N+](=O)[O-] or the neutral
        // shorthand C-N(=O)=O is not valid Lewis structure SMILES here; use
        // the bracket-charge form the parser supports.
        let m = Molecule::from_smiles("CN(=O)=O").unwrap();
        assert_eq!(count_nitro_groups(&m), 1);
        assert!(score(&m).exceeds_conservative_threshold());
    }

    #[test]
    fn single_nitro_group_on_large_molecule_still_flagged() {
        // Regression test for the density-only blind spot an external
        // review caught: 1 nitro group / 8 heavy atoms = density 0.125,
        // below the old 0.15 threshold -- the old rule would have silently
        // let this through, contradicting "loud, not precise."
        let m = Molecule::from_smiles("CCCCCCCN(=O)=O").unwrap(); // 1-nitroheptane
        let s = score(&m);
        assert_eq!(s.nitro_count, 1);
        assert!(
            s.nitro_density < 0.15,
            "test setup check: {}",
            s.nitro_density
        );
        assert!(
            s.exceeds_conservative_threshold(),
            "a single nitro group must always trigger review regardless of density: {s:?}"
        );
    }

    #[test]
    fn peroxide_detected() {
        // Hydrogen peroxide-like motif: C-O-O-C (a peroxide linkage).
        let m = Molecule::from_smiles("COOC").unwrap();
        assert_eq!(count_peroxide_motifs(&m), 1);
        assert!(score(&m).exceeds_conservative_threshold());
    }

    #[test]
    fn external_config_defaults_unpopulated() {
        let cfg = ExternalScopeConfig::default();
        assert!(cfg.extra_patterns_path.is_none());
    }

    // "Hazard motifs placed in structures designed to defeat simple local
    // detection" (Phase 1.2): the detectors only inspect an atom's own
    // element/bond-order neighborhood, so they should be unaffected by what
    // ELSE is attached to that neighborhood (a ring, a longer chain, an
    // aromatic system) -- these tests exercise exactly that.

    #[test]
    fn nitro_group_on_aromatic_ring_still_detected() {
        // Nitrobenzene-shaped: nitro group substituent on an aromatic ring
        // carbon. The ring's aromatic bonds are irrelevant to nitro
        // detection (which only inspects the N's own O-neighbors) -- must
        // not be silently missed just because the attachment point is
        // aromatic rather than aliphatic.
        let m = Molecule::from_smiles("c1ccc(cc1)N(=O)=O").unwrap();
        assert_eq!(count_nitro_groups(&m), 1);
        assert!(score(&m).exceeds_conservative_threshold());
    }

    #[test]
    fn peroxide_in_longer_chain_still_detected() {
        // Diethyl peroxide: peroxide linkage embedded in a longer chain
        // (not the minimal COOC case already covered) -- the motif must
        // still be found regardless of what's on either side of it.
        let m = Molecule::from_smiles("CCOOCC").unwrap();
        assert_eq!(count_peroxide_motifs(&m), 1);
        assert!(score(&m).exceeds_conservative_threshold());
    }

    #[test]
    fn nitro_group_on_ring_carbon_that_is_also_in_a_ring_bond() {
        // A nitro-substituted ring carbon has THREE relevant bonds: two
        // aromatic ring bonds plus the single bond to the nitro nitrogen.
        // Confirms the ring bonds (Aromatic order) don't get miscounted as
        // part of the nitro motif itself, and don't prevent detection.
        let m = Molecule::from_smiles("c1cc(ccc1N(=O)=O)N(=O)=O").unwrap(); // dinitrobenzene-shaped
        assert_eq!(count_nitro_groups(&m), 2);
        assert!(score(&m).exceeds_conservative_threshold());
    }
}
