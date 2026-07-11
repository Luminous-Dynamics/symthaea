//! `CulinarySpec` — a recipe/process description as *data*. The validators in
//! [`crate::validate`] hold veto power over the physical invariants a spec must
//! respect; a spec that violates one is rejected with the reason, before a stove
//! is lit.
//!
//! `Serialize`/`Deserialize` on every type here is what makes the pitch's "edit
//! the JSON to make a new culinary world" real: [`crate::presets`] loads named
//! style presets from embedded JSON, and any spec — preset or hand-edited — goes
//! through the exact same [`crate::validate::validate`] the Rust-constructed
//! specs in `tests/invariants.rs` do. The validator is fixed; only the spec
//! (and which physically-valid point within its safe ranges a style prefers) is
//! data-driven.

use crate::nutrition::NutrientProfile;
use crate::thermal::ThermalTrajectory;
use serde::{Deserialize, Serialize};

/// A full culinary specification: a named style plus whichever physical
/// components it constrains. Every field is optional — a spec only carries the
/// invariants relevant to the dish.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CulinarySpec {
    pub name: String,
    #[serde(default)]
    pub emulsion: Option<Emulsion>,
    #[serde(default)]
    pub coagulation: Option<Coagulation>,
    #[serde(default)]
    pub pasteurization: Option<Pasteurization>,
    #[serde(default)]
    pub hydration: Option<Hydration>,
    #[serde(default)]
    pub candy: Option<Candy>,
    #[serde(default)]
    pub dairy: Option<DairyAcidification>,
    #[serde(default)]
    pub frying: Option<FryingFat>,
    #[serde(default)]
    pub nutrition: Option<NutrientProfile>,
}

impl CulinarySpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }
    pub fn with_emulsion(mut self, e: Emulsion) -> Self {
        self.emulsion = Some(e);
        self
    }
    pub fn with_coagulation(mut self, c: Coagulation) -> Self {
        self.coagulation = Some(c);
        self
    }
    pub fn with_pasteurization(mut self, p: Pasteurization) -> Self {
        self.pasteurization = Some(p);
        self
    }
    pub fn with_hydration(mut self, h: Hydration) -> Self {
        self.hydration = Some(h);
        self
    }
    pub fn with_candy(mut self, c: Candy) -> Self {
        self.candy = Some(c);
        self
    }
    pub fn with_dairy(mut self, d: DairyAcidification) -> Self {
        self.dairy = Some(d);
        self
    }
    pub fn with_frying(mut self, f: FryingFat) -> Self {
        self.frying = Some(f);
        self
    }
    pub fn with_nutrition(mut self, n: NutrientProfile) -> Self {
        self.nutrition = Some(n);
        self
    }
}

/// An oil-in-water (or water-in-oil) emulsion, described by its dispersed-phase
/// volume fraction φ ∈ [0, 1] — e.g. a mayonnaise is oil dispersed in egg-yolk water.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Emulsion {
    pub dispersed_phase_fraction: f64,
}

/// A protein whose coagulation window a thermal process must respect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Protein {
    EggWhite,
    EggYolk,
    /// An egg-thickened stirred custard (crème anglaise): sets, then curdles if too hot.
    Custard,
}

/// "Cook this protein along this thermal trajectory." The validator checks the
/// trajectory both *reaches* the set point and *stays below* the ruin point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Coagulation {
    pub protein: Protein,
    pub trajectory: ThermalTrajectory,
}

/// A pathogen-reduction requirement met by holding a thermal trajectory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pathogen {
    /// Salmonella in poultry.
    SalmonellaPoultry,
}

/// "This trajectory must deliver at least `required_log_reduction` decimal
/// reductions of `pathogen`." If `None`, the pathogen's standard target is used.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pasteurization {
    pub pathogen: Pathogen,
    pub trajectory: ThermalTrajectory,
    #[serde(default)]
    pub required_log_reduction: Option<f64>,
}

/// A dough/batter class with its expected baker's-percentage hydration window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DoughClass {
    Bread,
    Pastry,
    Batter,
}

/// Flour and water masses (grams). Hydration = water / flour.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hydration {
    pub dough: DoughClass,
    pub flour_g: f64,
    pub water_g: f64,
}

impl Hydration {
    /// Baker's-percentage hydration as a fraction (water mass / flour mass).
    pub fn ratio(&self) -> f64 {
        if self.flour_g <= 0.0 {
            f64::INFINITY
        } else {
            self.water_g / self.flour_g
        }
    }
}

/// A named candy-making sugar-syrup stage — each a real boiling-point-elevation
/// window of a sucrose/water solution (see `thresholds.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SugarStage {
    Thread,
    SoftBall,
    FirmBall,
    HardBall,
    SoftCrack,
    HardCrack,
}

/// "Cook this sugar syrup until it reaches `stage`." The validator checks the
/// trajectory's peak temperature falls within that stage's real window —
/// neither undercooked (too low) nor overshot into the next stage (too high).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candy {
    pub stage: SugarStage,
    pub trajectory: ThermalTrajectory,
}

/// Does this recipe want its dairy to curdle (fresh cheese/paneer-style
/// acidification) or stay smooth (a cream sauce)? Whether it actually will is
/// physics (the casein isoelectric point), not intent — the validator checks
/// `ph` against that threshold and flags a mismatch either direction.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DairyAcidification {
    pub ph: f64,
    pub should_curdle: bool,
}

/// A cooking fat with a real, citable smoke point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fat {
    ExtraVirginOliveOil,
    Butter,
    CanolaOil,
    RefinedPeanutOil,
}

/// "Heat this fat along this trajectory." The validator checks the trajectory's
/// peak temperature never exceeds the fat's smoke point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FryingFat {
    pub fat: Fat,
    pub trajectory: ThermalTrajectory,
}
