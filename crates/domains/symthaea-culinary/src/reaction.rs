//! Phase 2 — browning chemistry, done honestly on two axes:
//!
//! - **Feasibility** (`symthaea-organic-chemistry`): the Maillard reaction needs a
//!   *reducing* carbonyl (an aldehyde/ketone sugar) **and** an amine (an amino
//!   acid). [`maillard_feasible`] checks exactly that from molecular structure, so
//!   it correctly rejects an amino acid alone (its acid group is a *carboxyl*, not
//!   a reducing carbonyl) and non-reducing sugars (sucrose has no free carbonyl —
//!   which is precisely why sucrose caramelizes but does not Maillard directly).
//!
//! - **Kinetics** (Arrhenius): [`arrhenius_rate`] / [`reaction_extent`] give the
//!   *temperature sensitivity* of browning. The robustly-falsifiable content of
//!   Arrhenius is the sensitivity (Q10, low-vs-high-temperature contrast) — which
//!   follows from the literature activation energy alone. Absolute onset time
//!   needs the pre-exponential fitted to a measured browning rate; we expose
//!   [`onset_temperature`] as a utility but do not assert absolute onsets (that
//!   calibration is deferred rather than faked).

use crate::thermal::ThermalTrajectory;
use crate::thresholds::GAS_CONSTANT;
use symthaea_organic_chemistry::{FunctionalGroup, Molecule, ParseError, detect};

/// Arrhenius rate k(T) = A·exp(−Eₐ / R·T), with `temp_c` in °C, `ea` in J·mol⁻¹.
/// `pre_exp_a` sets the (reaction-specific) absolute scale; it cancels in any ratio.
pub fn arrhenius_rate(temp_c: f64, ea_j_per_mol: f64, pre_exp_a: f64) -> f64 {
    let t_kelvin = temp_c + 273.15;
    pre_exp_a * (-ea_j_per_mol / (GAS_CONSTANT * t_kelvin)).exp()
}

/// Q10 — the factor by which the rate changes for a +10 °C rise at `temp_c`.
/// Independent of the pre-exponential, so it is a pure prediction of Eₐ.
pub fn q10(temp_c: f64, ea_j_per_mol: f64) -> f64 {
    arrhenius_rate(temp_c + 10.0, ea_j_per_mol, 1.0) / arrhenius_rate(temp_c, ea_j_per_mol, 1.0)
}

/// Reaction extent ∫ k(T(t)) dt over a thermal trajectory (arbitrary units when
/// `pre_exp_a` is unfitted; meaningful in ratios). Sub-samples each linear segment.
pub fn reaction_extent(trajectory: &ThermalTrajectory, ea_j_per_mol: f64, pre_exp_a: f64) -> f64 {
    let mut total = 0.0;
    const SUBSTEPS: usize = 64;
    for w in trajectory.points.windows(2) {
        let (t0, temp0) = w[0];
        let (t1, temp1) = w[1];
        let dt = t1 - t0;
        if dt <= 0.0 {
            continue;
        }
        let h = dt / SUBSTEPS as f64;
        for k in 0..SUBSTEPS {
            let f0 = k as f64 / SUBSTEPS as f64;
            let f1 = (k as f64 + 1.0) / SUBSTEPS as f64;
            let a = arrhenius_rate(temp0 + (temp1 - temp0) * f0, ea_j_per_mol, pre_exp_a);
            let b = arrhenius_rate(temp0 + (temp1 - temp0) * f1, ea_j_per_mol, pre_exp_a);
            total += 0.5 * (a + b) * h;
        }
    }
    total
}

/// Temperature (°C) at which the Arrhenius rate equals `threshold_rate`, given a
/// fitted pre-exponential. Utility for calibrated models; not asserted here.
pub fn onset_temperature(ea_j_per_mol: f64, pre_exp_a: f64, threshold_rate: f64) -> Option<f64> {
    if pre_exp_a <= 0.0 || threshold_rate <= 0.0 || threshold_rate > pre_exp_a {
        return None;
    }
    // k = A·exp(−Ea/RT) ⇒ T = Ea / (R·ln(A/k))
    let t_kelvin = ea_j_per_mol / (GAS_CONSTANT * (pre_exp_a / threshold_rate).ln());
    Some(t_kelvin - 273.15)
}

/// Does a set of reactant SMILES contain the pair Maillard requires — a reducing
/// carbonyl (aldehyde/ketone) donor *and* an amine donor?
pub fn maillard_feasible(reactant_smiles: &[&str]) -> Result<bool, ParseError> {
    let mut has_reducing_carbonyl = false;
    let mut has_amine = false;
    for s in reactant_smiles {
        let m = Molecule::from_smiles(s)?;
        let groups = detect(&m);
        // `Carbonyl` here excludes carboxyl/ester/amide by construction in the
        // organic-chemistry crate — so it means a genuine reducing carbonyl.
        if groups.contains(&FunctionalGroup::Carbonyl) {
            has_reducing_carbonyl = true;
        }
        if groups.contains(&FunctionalGroup::Amine) {
            has_amine = true;
        }
    }
    Ok(has_reducing_carbonyl && has_amine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thresholds::{CARAMELIZATION_EA_J_PER_MOL, MAILLARD_EA_J_PER_MOL};

    #[test]
    fn maillard_q10_matches_empirical_browning() {
        // Browning is famously ~2–3× faster per 10 °C. That falls straight out of
        // the literature activation energy — no fitting.
        let q = q10(140.0, MAILLARD_EA_J_PER_MOL);
        assert!((2.0..=3.0).contains(&q), "Maillard Q10 = {q}");
    }

    #[test]
    fn caramelization_is_more_temperature_sensitive_than_maillard() {
        // Higher Ea ⇒ steeper temperature dependence ⇒ larger Q10.
        assert!(q10(160.0, CARAMELIZATION_EA_J_PER_MOL) > q10(160.0, MAILLARD_EA_J_PER_MOL));
    }

    #[test]
    fn browning_negligible_when_simmering_rapid_when_searing() {
        // 10 min at 100 °C (simmer) vs 10 min at 160 °C (sear). Same units.
        let simmer = reaction_extent(
            &ThermalTrajectory::hold(100.0, 10.0),
            MAILLARD_EA_J_PER_MOL,
            1e12,
        );
        let sear = reaction_extent(
            &ThermalTrajectory::hold(160.0, 10.0),
            MAILLARD_EA_J_PER_MOL,
            1e12,
        );
        assert!(sear > simmer * 100.0, "sear={sear} simmer={simmer}");
    }

    #[test]
    fn reducing_sugar_plus_amino_acid_can_maillard() {
        // glycolaldehyde (simplest reducing sugar, a real Maillard reactant) + glycine.
        assert!(maillard_feasible(&["OCC=O", "NCC(=O)O"]).unwrap());
    }

    #[test]
    fn amino_acid_alone_cannot_maillard() {
        // Glycine's only carbonyl is a carboxyl — no reducing sugar present.
        assert!(!maillard_feasible(&["NCC(=O)O"]).unwrap());
    }

    #[test]
    fn reducing_sugar_alone_cannot_maillard() {
        assert!(!maillard_feasible(&["OCC=O"]).unwrap());
    }
}
