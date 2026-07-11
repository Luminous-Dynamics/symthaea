//! Physical/chemical constants used by the invariant validators, each with a
//! literature citation. Changing one of these is changing a claim about the
//! physical world — keep the citation next to the number.

/// Random close packing fraction of equal spheres. Above this an emulsion's
/// dispersed droplets can no longer pack without deforming/coalescing, so the
/// emulsion inverts or breaks.
/// Berryman, "Random close packing of hard spheres and disks", Phys. Rev. A 27 (1983).
pub const RANDOM_CLOSE_PACKING: f64 = 0.7405;

/// Egg-white (ovalbumin/ovotransferrin) coagulation onset, °C.
/// McGee, *On Food and Cooking* (2004), ch. 2; ovotransferrin ~62 °C, ovalbumin ~80 °C.
pub const EGG_WHITE_SET_C: f64 = 63.0;
/// Egg-yolk coagulation onset, °C.
pub const EGG_YOLK_SET_C: f64 = 68.0;
/// A stirred egg custard curdles (proteins over-coagulate and squeeze out water)
/// above roughly this temperature. McGee (2004): custards curdle near 82–85 °C.
pub const CUSTARD_CURDLE_C: f64 = 82.0;

/// Salmonella in poultry — decimal reduction time D at the reference temperature.
/// D_60°C = 0.396 min, z = 5.56 °C (Murphy et al., "Thermal inactivation of
/// Salmonella and Listeria in ground chicken breast", J. Food Prot. 67 (2004)).
pub const SALMONELLA_D_REF_MIN: f64 = 0.396;
pub const SALMONELLA_D_REF_TEMP_C: f64 = 60.0;
pub const SALMONELLA_Z_C: f64 = 5.56;
/// FSIS Appendix A target lethality for poultry: 7-log reduction.
pub const POULTRY_TARGET_LOG_REDUCTION: f64 = 7.0;

/// Universal gas constant, J·mol⁻¹·K⁻¹.
pub const GAS_CONSTANT: f64 = 8.314;

/// Apparent activation energy of the Maillard browning reaction, J·mol⁻¹.
/// Literature range ~100–160 kJ/mol; 125 kJ/mol is a representative mid-value
/// (e.g. Martins & van Boekel, *Food Chem.* 90 (2005) on Maillard kinetics).
pub const MAILLARD_EA_J_PER_MOL: f64 = 125_000.0;

/// Apparent activation energy of sucrose caramelization, J·mol⁻¹ — higher than
/// Maillard, consistent with its higher onset temperature (~160 °C vs ~140 °C).
/// Representative value from caramelization kinetics literature (~150–170 kJ/mol).
pub const CARAMELIZATION_EA_J_PER_MOL: f64 = 160_000.0;

/// Baker's-percentage hydration windows (water mass / flour mass, as a fraction).
/// Ranges from standard baking references (e.g. Suas, *Advanced Bread and Pastry*).
pub const BREAD_HYDRATION_MIN: f64 = 0.60;
pub const BREAD_HYDRATION_MAX: f64 = 0.85;
pub const PASTRY_HYDRATION_MIN: f64 = 0.45;
pub const PASTRY_HYDRATION_MAX: f64 = 0.60;
/// Pourable batters (crêpe/pancake) run far wetter than doughs.
pub const BATTER_HYDRATION_MIN: f64 = 1.00;
pub const BATTER_HYDRATION_MAX: f64 = 2.00;

/// Classic candy-making sugar-syrup stages, °C (converted from the standard
/// °F ranges used in confectionery references, e.g. McGee 2004 ch. 15,
/// CookWise/Corriher). Each is a boiling-point-elevation window of a
/// sucrose/water solution — higher stages correspond to less residual water.
pub const SUGAR_THREAD_MIN_C: f64 = 110.0;
pub const SUGAR_THREAD_MAX_C: f64 = 112.0;
pub const SUGAR_SOFT_BALL_MIN_C: f64 = 112.0;
pub const SUGAR_SOFT_BALL_MAX_C: f64 = 116.0;
pub const SUGAR_FIRM_BALL_MIN_C: f64 = 118.0;
pub const SUGAR_FIRM_BALL_MAX_C: f64 = 120.0;
pub const SUGAR_HARD_BALL_MIN_C: f64 = 121.0;
pub const SUGAR_HARD_BALL_MAX_C: f64 = 130.0;
pub const SUGAR_SOFT_CRACK_MIN_C: f64 = 132.0;
pub const SUGAR_SOFT_CRACK_MAX_C: f64 = 143.0;
pub const SUGAR_HARD_CRACK_MIN_C: f64 = 149.0;
pub const SUGAR_HARD_CRACK_MAX_C: f64 = 154.0;

/// Casein isoelectric point — the pH at which milk proteins lose their
/// mutual charge repulsion and coagulate (curdle). McGee, *On Food and
/// Cooking* (2004), ch. 1; standard value cited across dairy chemistry
/// references is pH 4.6.
pub const CASEIN_ISOELECTRIC_PH: f64 = 4.6;

/// Smoke points of common cooking fats, °C — the temperature at which a fat
/// begins to visibly smoke and break down, producing acrid off-flavors and
/// (at higher temperatures still) the flash point fire risk. Representative
/// values from standard culinary-science references (McGee 2004 ch. 12;
/// USDA/extension-service smoke-point tables); real oils vary by refinement
/// grade, these are the commonly-cited midpoints.
pub const SMOKE_POINT_EXTRA_VIRGIN_OLIVE_OIL_C: f64 = 191.0;
pub const SMOKE_POINT_BUTTER_C: f64 = 150.0;
pub const SMOKE_POINT_CANOLA_OIL_C: f64 = 204.0;
pub const SMOKE_POINT_REFINED_PEANUT_OIL_C: f64 = 232.0;

/// Atwater general energy factors, kcal per gram — the same system printed on
/// every nutrition label. Merrill & Watt, "Energy Value of Foods," USDA
/// Agriculture Handbook No. 74 (1955), codifying Atwater's original (1900) system.
pub const PROTEIN_KCAL_PER_G: f64 = 4.0;
pub const CARB_KCAL_PER_G: f64 = 4.0;
pub const FAT_KCAL_PER_G: f64 = 9.0;
pub const ALCOHOL_KCAL_PER_G: f64 = 7.0;

/// FDA Nutrition Facts label Daily Values (2016 final rule, 21 CFR 101.9),
/// referenced to a 2,000-kcal diet. The added-sugar DV is exactly the Dietary
/// Guidelines for Americans' "<10% of calories from added sugar" recommendation
/// expressed in grams at 4 kcal/g (200 kcal / 4 kcal/g = 50 g).
pub const FDA_SODIUM_DAILY_VALUE_MG: f64 = 2300.0;
pub const FDA_ADDED_SUGAR_DAILY_VALUE_G: f64 = 50.0;

/// Apparent activation energy of thermal ascorbic-acid (vitamin C) degradation,
/// J·mol⁻¹. Literature range is wide (~40–100+ kJ/mol depending on food matrix,
/// oxygen exposure, and pH); 75 kJ/mol is used here as a representative
/// mid-value (e.g. Van den Broeck et al., "Kinetics for isobaric-isothermal
/// degradation of L-ascorbic acid," J. Agric. Food Chem. 46 (1998), 2001–2006).
pub const VITAMIN_C_DEGRADATION_EA_J_PER_MOL: f64 = 75_000.0;
