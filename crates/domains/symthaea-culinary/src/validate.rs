//! The invariant validators — the differentiated core of the crate. Each holds
//! veto power over one physical/chemical invariant and, on violation, returns a
//! [`CulinaryViolation`] whose `Display` *names the physics that fails*. This is
//! what a flat recipe web-scraper cannot do: reject the chemically impossible
//! before a stove is lit.

use crate::nutrition::NutrientProfile;
use crate::spec::{
    Candy, Coagulation, CulinarySpec, DairyAcidification, DoughClass, Emulsion, Fat, FryingFat,
    Hydration, Pasteurization, Pathogen, Protein, SugarStage,
};
use crate::thresholds as th;

/// A rejected invariant, with a human-readable, physics-grounded reason.
#[derive(Clone, Debug, PartialEq)]
pub enum CulinaryViolation {
    EmulsionBreaks {
        phi: f64,
        limit: f64,
    },
    Undercooked {
        protein: Protein,
        peak_c: f64,
        set_c: f64,
    },
    Overcoagulated {
        peak_c: f64,
        curdle_c: f64,
    },
    InsufficientPasteurization {
        delivered_log: f64,
        required_log: f64,
    },
    HydrationOutOfRange {
        dough: DoughClass,
        ratio: f64,
        min: f64,
        max: f64,
    },
    CandyUndercooked {
        stage: SugarStage,
        peak_c: f64,
        min_c: f64,
    },
    CandyOvercooked {
        stage: SugarStage,
        peak_c: f64,
        max_c: f64,
    },
    DairyWontCurdle {
        ph: f64,
        threshold: f64,
    },
    DairyWillCurdle {
        ph: f64,
        threshold: f64,
    },
    FatSmoking {
        fat: Fat,
        peak_c: f64,
        smoke_point_c: f64,
    },
    SodiumExceedsLimit {
        sodium_mg: f64,
        limit_mg: f64,
    },
    AddedSugarExceedsLimit {
        added_sugar_g: f64,
        limit_g: f64,
    },
}

impl std::fmt::Display for CulinaryViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CulinaryViolation::EmulsionBreaks { phi, limit } => write!(
                f,
                "emulsion will invert/break: dispersed-phase fraction φ={phi:.3} exceeds the \
                 random-close-packing limit {limit:.4} — droplets can no longer pack without coalescing"
            ),
            CulinaryViolation::Undercooked {
                protein,
                peak_c,
                set_c,
            } => write!(
                f,
                "undercooked {protein:?}: peak {peak_c:.1} °C never reaches the coagulation \
                 set point {set_c:.1} °C"
            ),
            CulinaryViolation::Overcoagulated { peak_c, curdle_c } => write!(
                f,
                "custard will curdle: peak {peak_c:.1} °C exceeds the {curdle_c:.1} °C over-coagulation \
                 point — egg proteins squeeze out water and scramble"
            ),
            CulinaryViolation::InsufficientPasteurization {
                delivered_log,
                required_log,
            } => write!(
                f,
                "unsafe: trajectory delivers only {delivered_log:.2}-log reduction, below the required \
                 {required_log:.1}-log kill"
            ),
            CulinaryViolation::HydrationOutOfRange {
                dough,
                ratio,
                min,
                max,
            } => write!(
                f,
                "hydration out of range for {dough:?}: {:.0}% is outside the {:.0}–{:.0}% window",
                ratio * 100.0,
                min * 100.0,
                max * 100.0
            ),
            CulinaryViolation::CandyUndercooked {
                stage,
                peak_c,
                min_c,
            } => write!(
                f,
                "syrup undercooked for {stage:?}: peak {peak_c:.1} °C is below the {min_c:.1} °C \
                 stage window"
            ),
            CulinaryViolation::CandyOvercooked {
                stage,
                peak_c,
                max_c,
            } => write!(
                f,
                "syrup overcooked past {stage:?}: peak {peak_c:.1} °C is above the {max_c:.1} °C \
                 stage window"
            ),
            CulinaryViolation::DairyWontCurdle { ph, threshold } => write!(
                f,
                "dairy won't curdle as intended: pH {ph:.2} is above the casein isoelectric point \
                 {threshold:.1} — proteins keep their charge repulsion and stay suspended"
            ),
            CulinaryViolation::DairyWillCurdle { ph, threshold } => write!(
                f,
                "dairy will curdle unintentionally: pH {ph:.2} is at or below the casein isoelectric \
                 point {threshold:.1} — proteins lose charge repulsion and coagulate"
            ),
            CulinaryViolation::FatSmoking {
                fat,
                peak_c,
                smoke_point_c,
            } => write!(
                f,
                "{fat:?} will smoke: peak {peak_c:.1} °C exceeds its {smoke_point_c:.1} °C smoke point"
            ),
            CulinaryViolation::SodiumExceedsLimit {
                sodium_mg,
                limit_mg,
            } => write!(
                f,
                "sodium {sodium_mg:.0} mg exceeds the FDA Daily Value of {limit_mg:.0} mg — a \
                 published population-level threshold, not personalized advice"
            ),
            CulinaryViolation::AddedSugarExceedsLimit {
                added_sugar_g,
                limit_g,
            } => write!(
                f,
                "added sugar {added_sugar_g:.0} g exceeds the FDA Daily Value of {limit_g:.0} g \
                 (<10% of calories on a 2,000-kcal reference diet)"
            ),
        }
    }
}

impl std::error::Error for CulinaryViolation {}

/// φ ≤ random close packing, else the emulsion inverts/breaks.
pub fn validate_emulsion(e: &Emulsion) -> Result<(), CulinaryViolation> {
    if e.dispersed_phase_fraction > th::RANDOM_CLOSE_PACKING {
        Err(CulinaryViolation::EmulsionBreaks {
            phi: e.dispersed_phase_fraction,
            limit: th::RANDOM_CLOSE_PACKING,
        })
    } else {
        Ok(())
    }
}

/// The trajectory must reach the protein's set point and (for custard) stay below
/// its curdle point.
pub fn validate_coagulation(c: &Coagulation) -> Result<(), CulinaryViolation> {
    let peak = c.trajectory.peak_temp();
    let set_c = match c.protein {
        Protein::EggWhite => th::EGG_WHITE_SET_C,
        Protein::EggYolk | Protein::Custard => th::EGG_YOLK_SET_C,
    };
    if peak < set_c {
        return Err(CulinaryViolation::Undercooked {
            protein: c.protein,
            peak_c: peak,
            set_c,
        });
    }
    if c.protein == Protein::Custard && peak > th::CUSTARD_CURDLE_C {
        return Err(CulinaryViolation::Overcoagulated {
            peak_c: peak,
            curdle_c: th::CUSTARD_CURDLE_C,
        });
    }
    Ok(())
}

/// The trajectory must deliver at least the required log-reduction of the pathogen.
pub fn validate_pasteurization(p: &Pasteurization) -> Result<(), CulinaryViolation> {
    let (d_ref, t_ref, z, default_target) = match p.pathogen {
        Pathogen::SalmonellaPoultry => (
            th::SALMONELLA_D_REF_MIN,
            th::SALMONELLA_D_REF_TEMP_C,
            th::SALMONELLA_Z_C,
            th::POULTRY_TARGET_LOG_REDUCTION,
        ),
    };
    let required = p.required_log_reduction.unwrap_or(default_target);
    let delivered = p.trajectory.log_reduction(d_ref, t_ref, z);
    if delivered + 1e-9 < required {
        Err(CulinaryViolation::InsufficientPasteurization {
            delivered_log: delivered,
            required_log: required,
        })
    } else {
        Ok(())
    }
}

/// Baker's-percentage hydration must fall in the dough class's window.
pub fn validate_hydration(h: &Hydration) -> Result<(), CulinaryViolation> {
    let (min, max) = match h.dough {
        DoughClass::Bread => (th::BREAD_HYDRATION_MIN, th::BREAD_HYDRATION_MAX),
        DoughClass::Pastry => (th::PASTRY_HYDRATION_MIN, th::PASTRY_HYDRATION_MAX),
        DoughClass::Batter => (th::BATTER_HYDRATION_MIN, th::BATTER_HYDRATION_MAX),
    };
    let ratio = h.ratio();
    if ratio < min || ratio > max {
        Err(CulinaryViolation::HydrationOutOfRange {
            dough: h.dough,
            ratio,
            min,
            max,
        })
    } else {
        Ok(())
    }
}

/// The real temperature window (°C) for a candy-making sugar-syrup stage.
pub fn sugar_stage_window(stage: SugarStage) -> (f64, f64) {
    match stage {
        SugarStage::Thread => (th::SUGAR_THREAD_MIN_C, th::SUGAR_THREAD_MAX_C),
        SugarStage::SoftBall => (th::SUGAR_SOFT_BALL_MIN_C, th::SUGAR_SOFT_BALL_MAX_C),
        SugarStage::FirmBall => (th::SUGAR_FIRM_BALL_MIN_C, th::SUGAR_FIRM_BALL_MAX_C),
        SugarStage::HardBall => (th::SUGAR_HARD_BALL_MIN_C, th::SUGAR_HARD_BALL_MAX_C),
        SugarStage::SoftCrack => (th::SUGAR_SOFT_CRACK_MIN_C, th::SUGAR_SOFT_CRACK_MAX_C),
        SugarStage::HardCrack => (th::SUGAR_HARD_CRACK_MIN_C, th::SUGAR_HARD_CRACK_MAX_C),
    }
}

/// The trajectory's peak temperature must fall within the target stage's real window.
pub fn validate_candy(c: &Candy) -> Result<(), CulinaryViolation> {
    let (min_c, max_c) = sugar_stage_window(c.stage);
    let peak = c.trajectory.peak_temp();
    if peak < min_c {
        Err(CulinaryViolation::CandyUndercooked {
            stage: c.stage,
            peak_c: peak,
            min_c,
        })
    } else if peak > max_c {
        Err(CulinaryViolation::CandyOvercooked {
            stage: c.stage,
            peak_c: peak,
            max_c,
        })
    } else {
        Ok(())
    }
}

/// Whether the recipe's stated intent (curdle or stay smooth) matches what the
/// pH will actually do at the casein isoelectric point.
pub fn validate_dairy(d: &DairyAcidification) -> Result<(), CulinaryViolation> {
    let will_curdle = d.ph <= th::CASEIN_ISOELECTRIC_PH;
    if d.should_curdle && !will_curdle {
        Err(CulinaryViolation::DairyWontCurdle {
            ph: d.ph,
            threshold: th::CASEIN_ISOELECTRIC_PH,
        })
    } else if !d.should_curdle && will_curdle {
        Err(CulinaryViolation::DairyWillCurdle {
            ph: d.ph,
            threshold: th::CASEIN_ISOELECTRIC_PH,
        })
    } else {
        Ok(())
    }
}

/// The real smoke point (°C) of a cooking fat.
pub fn smoke_point_c(fat: Fat) -> f64 {
    match fat {
        Fat::ExtraVirginOliveOil => th::SMOKE_POINT_EXTRA_VIRGIN_OLIVE_OIL_C,
        Fat::Butter => th::SMOKE_POINT_BUTTER_C,
        Fat::CanolaOil => th::SMOKE_POINT_CANOLA_OIL_C,
        Fat::RefinedPeanutOil => th::SMOKE_POINT_REFINED_PEANUT_OIL_C,
    }
}

/// The trajectory's peak temperature must never exceed the fat's smoke point.
pub fn validate_frying(fr: &FryingFat) -> Result<(), CulinaryViolation> {
    let peak = fr.trajectory.peak_temp();
    let smoke = smoke_point_c(fr.fat);
    if peak > smoke {
        Err(CulinaryViolation::FatSmoking {
            fat: fr.fat,
            peak_c: peak,
            smoke_point_c: smoke,
        })
    } else {
        Ok(())
    }
}

/// Sodium and added-sugar totals must not exceed the FDA's published Daily
/// Values (2016 Nutrition Facts label rule) — a population-level threshold,
/// not personalized advice. Unlike this crate's other validators, these are
/// two genuinely independent checks over the same profile, so (unlike, say,
/// dairy's mutually-exclusive curdle/won't-curdle) both can fire from a
/// single call — hence `Vec` rather than a single `Result`.
pub fn validate_nutrition(n: &NutrientProfile) -> Vec<CulinaryViolation> {
    let mut v = Vec::new();
    if n.sodium_mg > th::FDA_SODIUM_DAILY_VALUE_MG {
        v.push(CulinaryViolation::SodiumExceedsLimit {
            sodium_mg: n.sodium_mg,
            limit_mg: th::FDA_SODIUM_DAILY_VALUE_MG,
        });
    }
    if n.added_sugar_g > th::FDA_ADDED_SUGAR_DAILY_VALUE_G {
        v.push(CulinaryViolation::AddedSugarExceedsLimit {
            added_sugar_g: n.added_sugar_g,
            limit_g: th::FDA_ADDED_SUGAR_DAILY_VALUE_G,
        });
    }
    v
}

/// Validate every invariant a spec carries. Returns all violations found (empty
/// ⇒ the spec is physically consistent). The engine keeps the veto: a spec with
/// any violation would be rejected before cooking.
pub fn validate(spec: &CulinarySpec) -> Vec<CulinaryViolation> {
    let mut v = Vec::new();
    if let Some(e) = &spec.emulsion
        && let Err(err) = validate_emulsion(e)
    {
        v.push(err);
    }
    if let Some(c) = &spec.coagulation
        && let Err(err) = validate_coagulation(c)
    {
        v.push(err);
    }
    if let Some(p) = &spec.pasteurization
        && let Err(err) = validate_pasteurization(p)
    {
        v.push(err);
    }
    if let Some(h) = &spec.hydration
        && let Err(err) = validate_hydration(h)
    {
        v.push(err);
    }
    if let Some(c) = &spec.candy
        && let Err(err) = validate_candy(c)
    {
        v.push(err);
    }
    if let Some(d) = &spec.dairy
        && let Err(err) = validate_dairy(d)
    {
        v.push(err);
    }
    if let Some(fr) = &spec.frying
        && let Err(err) = validate_frying(fr)
    {
        v.push(err);
    }
    if let Some(n) = &spec.nutrition {
        v.extend(validate_nutrition(n));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thermal::ThermalTrajectory;

    #[test]
    fn empty_spec_is_consistent() {
        assert!(validate(&CulinarySpec::new("plain")).is_empty());
    }

    #[test]
    fn violation_message_names_the_physics() {
        let err = validate_emulsion(&Emulsion {
            dispersed_phase_fraction: 0.80,
        })
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("random-close-packing"), "got: {msg}");
    }

    #[test]
    fn egg_white_undercooked_is_flagged() {
        let c = Coagulation {
            protein: Protein::EggWhite,
            trajectory: ThermalTrajectory::hold(55.0, 5.0),
        };
        assert!(matches!(
            validate_coagulation(&c),
            Err(CulinaryViolation::Undercooked { .. })
        ));
    }

    #[test]
    fn candy_below_stage_window_is_undercooked() {
        let c = Candy {
            stage: SugarStage::SoftBall,
            trajectory: ThermalTrajectory::hold(108.0, 2.0),
        };
        assert!(matches!(
            validate_candy(&c),
            Err(CulinaryViolation::CandyUndercooked { .. })
        ));
    }

    #[test]
    fn candy_within_stage_window_passes() {
        let c = Candy {
            stage: SugarStage::SoftBall,
            trajectory: ThermalTrajectory::hold(114.0, 2.0),
        };
        assert!(validate_candy(&c).is_ok());
    }

    #[test]
    fn candy_overshot_past_stage_window_is_overcooked() {
        let c = Candy {
            stage: SugarStage::SoftBall,
            trajectory: ThermalTrajectory::hold(160.0, 2.0),
        };
        assert!(matches!(
            validate_candy(&c),
            Err(CulinaryViolation::CandyOvercooked { .. })
        ));
    }

    #[test]
    fn dairy_intent_mismatch_is_flagged_both_directions() {
        // Fresh milk (pH ~6.5), wanting it to curdle into paneer: won't happen.
        assert!(matches!(
            validate_dairy(&DairyAcidification {
                ph: 6.5,
                should_curdle: true
            }),
            Err(CulinaryViolation::DairyWontCurdle { .. })
        ));
        // Over-acidified (pH 4.2) but wanting a smooth sauce: it'll curdle anyway.
        assert!(matches!(
            validate_dairy(&DairyAcidification {
                ph: 4.2,
                should_curdle: false
            }),
            Err(CulinaryViolation::DairyWillCurdle { .. })
        ));
    }

    #[test]
    fn dairy_intent_matching_reality_passes() {
        assert!(
            validate_dairy(&DairyAcidification {
                ph: 6.5,
                should_curdle: false
            })
            .is_ok()
        );
        assert!(
            validate_dairy(&DairyAcidification {
                ph: 4.2,
                should_curdle: true
            })
            .is_ok()
        );
    }

    #[test]
    fn extra_virgin_olive_oil_at_searing_heat_smokes() {
        let fr = FryingFat {
            fat: Fat::ExtraVirginOliveOil,
            trajectory: ThermalTrajectory::hold(220.0, 3.0),
        };
        assert!(matches!(
            validate_frying(&fr),
            Err(CulinaryViolation::FatSmoking { .. })
        ));
    }

    #[test]
    fn canola_oil_at_the_same_heat_does_not_smoke() {
        let fr = FryingFat {
            fat: Fat::CanolaOil,
            trajectory: ThermalTrajectory::hold(190.0, 3.0),
        };
        assert!(validate_frying(&fr).is_ok());
    }

    #[test]
    fn sodium_over_the_fda_daily_value_is_flagged() {
        let n = NutrientProfile {
            sodium_mg: 3000.0,
            ..Default::default()
        };
        let v = validate_nutrition(&n);
        assert!(matches!(v[0], CulinaryViolation::SodiumExceedsLimit { .. }));
    }

    #[test]
    fn added_sugar_over_the_fda_daily_value_is_flagged() {
        let n = NutrientProfile {
            added_sugar_g: 75.0,
            ..Default::default()
        };
        let v = validate_nutrition(&n);
        assert!(matches!(
            v[0],
            CulinaryViolation::AddedSugarExceedsLimit { .. }
        ));
    }

    #[test]
    fn sodium_and_added_sugar_can_both_fire_independently() {
        let n = NutrientProfile {
            sodium_mg: 3000.0,
            added_sugar_g: 75.0,
            ..Default::default()
        };
        assert_eq!(validate_nutrition(&n).len(), 2);
    }

    #[test]
    fn nutrition_within_daily_values_passes() {
        let n = NutrientProfile {
            sodium_mg: 600.0,
            added_sugar_g: 10.0,
            ..Default::default()
        };
        assert!(validate_nutrition(&n).is_empty());
    }
}
