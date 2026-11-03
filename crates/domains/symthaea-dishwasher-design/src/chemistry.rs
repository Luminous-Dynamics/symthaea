#![forbid(unsafe_code)]

//! Evidence-bounded, phase-resolved detergent dosing contract.
//!
//! This module does not formulate detergents, select an unvalidated dose, or
//! predict cleaning. It verifies that an externally selected dose schedule is
//! inside a versioned physical calibration envelope for the exact formulation,
//! soil, surface, water hardness and temperature. It fails closed when evidence
//! or compatibility data are missing.

use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WashStage {
    Prewash,
    MainWash,
    IntermediateRinse,
    FinalRinse,
    Drying,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentKind {
    EnzymeBlend,
    Surfactant,
    Builder,
    OxidizingBleach,
    RinseAid,
    WaterSoftenerSalt,
    MachineCare,
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StageWindow {
    pub stage: WashStage,
    pub start_s: u32,
    pub end_s: u32,
    /// Water temperature at the release point; drying has no liquid-water temperature.
    pub water_temperature_c: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoseEvent {
    pub event_id: String,
    pub component_id: String,
    pub stage: WashStage,
    /// Elapsed seconds from the beginning of the complete cycle.
    pub elapsed_s: u32,
    /// Duration of physical release; interactions are measured after delivery ends.
    pub release_duration_s: u32,
    /// Commanded component mass in integer milligrams. Volumetric ingredients
    /// must be converted using a measured batch density and archived in the protocol.
    pub dose_mg: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComponentCalibration {
    pub component_id: String,
    pub kind: ComponentKind,
    pub calibration_id: String,
    pub calibration_evidence_sha256: String,
    pub allowed_stages: Vec<WashStage>,
    pub supported_soil_classes: Vec<String>,
    pub supported_surface_materials: Vec<String>,
    pub minimum_temperature_c: f64,
    pub maximum_temperature_c: f64,
    pub minimum_water_hardness_ppm_as_caco3: u16,
    pub maximum_water_hardness_ppm_as_caco3: u16,
    pub minimum_release_mg: u64,
    pub maximum_release_mg: u64,
    pub maximum_cycle_mg: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComponentInteraction {
    pub first_component_id: String,
    pub second_component_id: String,
    /// Minimum interval between any release events from these two components.
    /// Must be supported by the exact-formulation compatibility study.
    pub minimum_separation_s: u32,
    pub compatibility_evidence_sha256: String,
}

/// Caller-supplied calibration metadata. Digests are identifiers only; this
/// structure is not cryptographically authenticated or independently validated.
#[derive(Debug, Clone, PartialEq)]
pub struct FormulationCalibrationProfile {
    pub formulation_id: String,
    pub formulation_evidence_sha256: String,
    pub maximum_total_dose_mg: u64,
    pub components: Vec<ComponentCalibration>,
    pub interactions: Vec<ComponentInteraction>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DosePlanRequest {
    pub soil_class: String,
    pub surface_material: String,
    pub water_hardness_ppm_as_caco3: u16,
    pub cycle_duration_s: u32,
    pub stage_windows: Vec<StageWindow>,
    pub events: Vec<DoseEvent>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DosePlanCheckSummary {
    pub event_count: usize,
    pub component_count: usize,
    pub total_dose_mg: u64,
    pub cycle_duration_s: u32,
}

/// Calibrated tolerance for comparing commanded and reported dispenser delivery.
/// The calibration must be established for the exact dispenser/channel and
/// formulation viscosity/temperature envelope; a digest alone does not authenticate it.
#[derive(Debug, Clone, PartialEq)]
pub struct DoseDeliveryCalibration {
    pub component_id: String,
    pub dispenser_channel_id: String,
    pub calibration_id: String,
    pub calibration_evidence_sha256: String,
    /// Allowed under-delivery in basis points; must be below 10,000 (100%).
    pub maximum_underdelivery_bps: u16,
    /// Allowed over-delivery in basis points.
    pub maximum_overdelivery_bps: u16,
    /// Maximum deviation for both release start and delivery duration.
    pub maximum_timing_deviation_s: u32,
}

/// Reported telemetry for one physical dispenser event. Evidence references and
/// digests are checked for presence/shape only; they are not authenticated here.
#[derive(Debug, Clone, PartialEq)]
pub struct DoseDeliveryMeasurement {
    pub event_id: String,
    pub component_id: String,
    pub dispenser_channel_id: String,
    pub calibration_id: String,
    pub actual_dose_mg: u64,
    pub actual_elapsed_s: u32,
    pub actual_duration_s: u32,
    pub measurement_evidence_sha256: String,
    pub calibration_bundle_sha256: String,
}

/// Result of comparing reported telemetry with a calibrated command tolerance.
/// This is not proof that the sensor reading is true or that cleaning succeeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DoseDeliveryCheckSummary {
    pub commanded_dose_mg: u64,
    pub actual_dose_mg: u64,
    pub minimum_allowed_dose_mg: u64,
    pub maximum_allowed_dose_mg: u64,
    pub actual_elapsed_s: u32,
    pub actual_duration_s: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChemistryError {
    InvalidFormulationId,
    InvalidEvidenceDigest(&'static str),
    InvalidMaximumTotalDose,
    InvalidDeliveryCalibration,
    InvalidDeliveryEvidence,
    DeliveryCalibrationMismatch,
    DeliveryEventMismatch,
    DeliveryStageMismatch,
    ActualDeliveryOutsideStageWindow,
    DeliveryComponentMismatch,
    DeliveryChannelMismatch,
    ActualDoseOutsideTolerance,
    ActualDeliveryTimingOutsideTolerance,
    ComponentDoseOverflow {
        component_id: String,
    },
    TotalDoseOverflow,
    NoCalibratedComponents,
    DuplicateComponentId(String),
    InvalidComponentCalibration {
        component_id: String,
        field: &'static str,
    },
    DuplicateInteraction(String, String),
    UnknownInteractionComponent(String),
    InvalidInteractionEvidence,
    ZeroInteractionSeparation,
    InvalidRequestScope(&'static str),
    InvalidStageWindow,
    DuplicateStageWindow(WashStage),
    OverlappingStageWindows,
    StageOrderViolation,
    EmptyDosePlan,
    InvalidEventId,
    DuplicateEventId(String),
    InvalidEventDuration {
        event_id: String,
    },
    UnknownComponent(String),
    UnsupportedSoil {
        component_id: String,
        soil_class: String,
    },
    UnsupportedSurface {
        component_id: String,
        surface_material: String,
    },
    TemperatureOutsideCalibration {
        component_id: String,
    },
    WaterHardnessOutsideCalibration {
        component_id: String,
    },
    StageNotCalibrated {
        component_id: String,
        stage: WashStage,
    },
    EventOutsideStageWindow {
        event_id: String,
    },
    DoseOutsideCalibration {
        event_id: String,
    },
    ComponentCycleLimitExceeded {
        component_id: String,
    },
    TotalDoseLimitExceeded,
    IncompatibleReleaseTiming {
        first_event_id: String,
        second_event_id: String,
        required_separation_s: u32,
    },
}

impl fmt::Display for ChemistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ChemistryError {}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn canonical_non_empty(value: &str) -> bool {
    !value.trim().is_empty() && value.trim() == value
}

fn validate_formulation(formulation: &FormulationCalibrationProfile) -> Result<(), ChemistryError> {
    if !canonical_non_empty(&formulation.formulation_id) {
        return Err(ChemistryError::InvalidFormulationId);
    }
    if !valid_digest(&formulation.formulation_evidence_sha256) {
        return Err(ChemistryError::InvalidEvidenceDigest("formulation"));
    }
    if formulation.maximum_total_dose_mg == 0 {
        return Err(ChemistryError::InvalidMaximumTotalDose);
    }
    if formulation.components.is_empty() {
        return Err(ChemistryError::NoCalibratedComponents);
    }

    let mut component_ids: HashSet<String> =
        HashSet::with_capacity(formulation.components.len());
    for component in &formulation.components {
        if !canonical_non_empty(&component.component_id) {
            return Err(ChemistryError::InvalidComponentCalibration {
                component_id: component.component_id.clone(),
                field: "component_id",
            });
        }
        if !component_ids.insert(component.component_id.clone()) {
            return Err(ChemistryError::DuplicateComponentId(
                component.component_id.clone(),
            ));
        }
        if !canonical_non_empty(&component.calibration_id) {
            return Err(ChemistryError::InvalidComponentCalibration {
                component_id: component.component_id.clone(),
                field: "calibration_id",
            });
        }
        if !valid_digest(&component.calibration_evidence_sha256) {
            return Err(ChemistryError::InvalidEvidenceDigest("component calibration"));
        }
        let unique_stages: HashSet<WashStage> =
            component.allowed_stages.iter().copied().collect();
        let unique_soils: HashSet<&str> = component
            .supported_soil_classes
            .iter()
            .map(String::as_str)
            .collect();
        let unique_surfaces: HashSet<&str> = component
            .supported_surface_materials
            .iter()
            .map(String::as_str)
            .collect();
        if component.allowed_stages.is_empty()
            || unique_stages.len() != component.allowed_stages.len()
            || component.supported_soil_classes.is_empty()
            || unique_soils.len() != component.supported_soil_classes.len()
            || component
                .supported_soil_classes
                .iter()
                .any(|value| !canonical_non_empty(value))
            || component.supported_surface_materials.is_empty()
            || unique_surfaces.len() != component.supported_surface_materials.len()
            || component
                .supported_surface_materials
                .iter()
                .any(|value| !canonical_non_empty(value))
        {
            return Err(ChemistryError::InvalidComponentCalibration {
                component_id: component.component_id.clone(),
                field: "empty, non-canonical or duplicate supported scope",
            });
        }
        if !component.minimum_temperature_c.is_finite()
            || !component.maximum_temperature_c.is_finite()
            || component.minimum_temperature_c < 0.0
            || component.maximum_temperature_c < component.minimum_temperature_c
        {
            return Err(ChemistryError::InvalidComponentCalibration {
                component_id: component.component_id.clone(),
                field: "temperature range",
            });
        }
        if component.minimum_water_hardness_ppm_as_caco3
            > component.maximum_water_hardness_ppm_as_caco3
        {
            return Err(ChemistryError::InvalidComponentCalibration {
                component_id: component.component_id.clone(),
                field: "water hardness range",
            });
        }
        if component.minimum_release_mg == 0
            || component.maximum_release_mg < component.minimum_release_mg
            || component.maximum_cycle_mg < component.minimum_release_mg
        {
            return Err(ChemistryError::InvalidComponentCalibration {
                component_id: component.component_id.clone(),
                field: "dose bounds",
            });
        }
    }

    let mut interaction_keys: HashSet<(String, String)> = HashSet::new();
    for interaction in &formulation.interactions {
        if !canonical_non_empty(&interaction.first_component_id) {
            return Err(ChemistryError::UnknownInteractionComponent(
                interaction.first_component_id.clone(),
            ));
        }
        if !canonical_non_empty(&interaction.second_component_id) {
            return Err(ChemistryError::UnknownInteractionComponent(
                interaction.second_component_id.clone(),
            ));
        }
        if !component_ids.contains(&interaction.first_component_id) {
            return Err(ChemistryError::UnknownInteractionComponent(
                interaction.first_component_id.clone(),
            ));
        }
        if !component_ids.contains(&interaction.second_component_id) {
            return Err(ChemistryError::UnknownInteractionComponent(
                interaction.second_component_id.clone(),
            ));
        }
        if interaction.first_component_id == interaction.second_component_id {
            return Err(ChemistryError::DuplicateInteraction(
                interaction.first_component_id.clone(),
                interaction.second_component_id.clone(),
            ));
        }
        if interaction.minimum_separation_s == 0 {
            return Err(ChemistryError::ZeroInteractionSeparation);
        }
        if !valid_digest(&interaction.compatibility_evidence_sha256) {
            return Err(ChemistryError::InvalidInteractionEvidence);
        }
        let (left, right) = if interaction.first_component_id
            <= interaction.second_component_id
        {
            (
                interaction.first_component_id.clone(),
                interaction.second_component_id.clone(),
            )
        } else {
            (
                interaction.second_component_id.clone(),
                interaction.first_component_id.clone(),
            )
        };
        if !interaction_keys.insert((left.clone(), right.clone())) {
            return Err(ChemistryError::DuplicateInteraction(left, right));
        }
    }

    Ok(())
}

fn validate_stage_windows(
    windows: &[StageWindow],
    cycle_duration_s: u32,
) -> Result<HashMap<WashStage, &StageWindow>, ChemistryError> {
    if windows.is_empty() || cycle_duration_s == 0 {
        return Err(ChemistryError::InvalidStageWindow);
    }

    let mut map = HashMap::with_capacity(windows.len());
    let mut ordered: Vec<&StageWindow> = windows.iter().collect();
    ordered.sort_by_key(|window| (window.start_s, window.end_s));

    let mut previous_order: Option<u8> = None;
    for window in &ordered {
        if window.start_s >= window.end_s || window.end_s > cycle_duration_s {
            return Err(ChemistryError::InvalidStageWindow);
        }
        if map.contains_key(&window.stage) {
            return Err(ChemistryError::DuplicateStageWindow(window.stage));
        }
        let stage_order = match window.stage {
            WashStage::Prewash => 0,
            WashStage::MainWash => 1,
            WashStage::IntermediateRinse => 2,
            WashStage::FinalRinse => 3,
            WashStage::Drying => 4,
        };
        if previous_order.is_some_and(|previous| stage_order <= previous) {
            return Err(ChemistryError::StageOrderViolation);
        }
        previous_order = Some(stage_order);
        match (window.stage, window.water_temperature_c) {
            (WashStage::Drying, None) => {}
            (WashStage::Drying, Some(_)) => return Err(ChemistryError::InvalidStageWindow),
            (_, Some(temperature)) if temperature.is_finite() && temperature >= 0.0 => {}
            _ => return Err(ChemistryError::InvalidStageWindow),
        }
        map.insert(window.stage, *window);
    }
    if ordered
        .windows(2)
        .any(|pair| pair[0].end_s > pair[1].start_s)
    {
        return Err(ChemistryError::OverlappingStageWindows);
    }
    Ok(map)
}

/// Validate a proposed phase-resolved dose plan against physical calibration.
///
/// This function verifies bounds and timing only. It does not decide that a
/// recipe is chemically optimal, safe for human contact, or sufficient to clean.
/// Those conclusions must be established by the formulation owner and independent
/// physical acceptance tests. If a condition is outside scope, the plan rejects.
/// Compare reported dispenser telemetry with a separately calibrated delivery
/// tolerance. This checks quantities and timing only; it does not authenticate
/// the sensor, calibration record or evidence, and does not establish cleaning.
pub fn check_reported_dose_delivery(
    commanded: &DoseEvent,
    stage_window: &StageWindow,
    calibration: &DoseDeliveryCalibration,
    measurement: &DoseDeliveryMeasurement,
) -> Result<DoseDeliveryCheckSummary, ChemistryError> {
    if !canonical_non_empty(&commanded.event_id)
        || commanded.dose_mg == 0
        || commanded.release_duration_s == 0
    {
        return Err(ChemistryError::DeliveryEventMismatch);
    }
    if stage_window.stage != commanded.stage || stage_window.start_s >= stage_window.end_s {
        return Err(ChemistryError::DeliveryStageMismatch);
    }
    let commanded_end_s =
        u64::from(commanded.elapsed_s) + u64::from(commanded.release_duration_s);
    if commanded.elapsed_s < stage_window.start_s
        || commanded_end_s > u64::from(stage_window.end_s)
    {
        return Err(ChemistryError::DeliveryStageMismatch);
    }
    if !canonical_non_empty(&calibration.component_id)
        || !canonical_non_empty(&calibration.dispenser_channel_id)
        || !canonical_non_empty(&calibration.calibration_id)
        || calibration.maximum_underdelivery_bps >= 10_000
        || !valid_digest(&calibration.calibration_evidence_sha256)
    {
        return Err(ChemistryError::InvalidDeliveryCalibration);
    }
    if !canonical_non_empty(&measurement.event_id)
        || !canonical_non_empty(&measurement.component_id)
        || !canonical_non_empty(&measurement.dispenser_channel_id)
        || !valid_digest(&measurement.measurement_evidence_sha256)
        || !valid_digest(&measurement.calibration_bundle_sha256)
    {
        return Err(ChemistryError::InvalidDeliveryEvidence);
    }
    if measurement.event_id != commanded.event_id {
        return Err(ChemistryError::DeliveryEventMismatch);
    }
    if calibration.component_id != commanded.component_id
        || measurement.component_id != commanded.component_id
    {
        return Err(ChemistryError::DeliveryComponentMismatch);
    }
    if measurement.dispenser_channel_id != calibration.dispenser_channel_id {
        return Err(ChemistryError::DeliveryChannelMismatch);
    }
    if measurement.calibration_id != calibration.calibration_id {
        return Err(ChemistryError::DeliveryCalibrationMismatch);
    }
    if measurement.actual_duration_s == 0 {
        return Err(ChemistryError::ActualDeliveryTimingOutsideTolerance);
    }
    let actual_end_s =
        u64::from(measurement.actual_elapsed_s) + u64::from(measurement.actual_duration_s);
    if measurement.actual_elapsed_s < stage_window.start_s
        || actual_end_s > u64::from(stage_window.end_s)
    {
        return Err(ChemistryError::ActualDeliveryOutsideStageWindow);
    }

    const BASIS_POINTS: u128 = 10_000;
    let commanded_mg = u128::from(commanded.dose_mg);
    let minimum_numerator = commanded_mg
        * (BASIS_POINTS - u128::from(calibration.maximum_underdelivery_bps));
    // Round the allowed minimum upward so the check never silently widens
    // the permitted under-delivery tolerance.
    let minimum_allowed_dose_mg =
        ((minimum_numerator + BASIS_POINTS - 1) / BASIS_POINTS) as u64;
    let maximum_numerator = commanded_mg
        * (BASIS_POINTS + u128::from(calibration.maximum_overdelivery_bps));
    let maximum_allowed_dose_mg = (maximum_numerator / BASIS_POINTS)
        .try_into()
        .unwrap_or(u64::MAX);

    if measurement.actual_dose_mg < minimum_allowed_dose_mg
        || measurement.actual_dose_mg > maximum_allowed_dose_mg
    {
        return Err(ChemistryError::ActualDoseOutsideTolerance);
    }
    if measurement.actual_elapsed_s.abs_diff(commanded.elapsed_s)
        > calibration.maximum_timing_deviation_s
        || measurement.actual_duration_s.abs_diff(commanded.release_duration_s)
            > calibration.maximum_timing_deviation_s
    {
        return Err(ChemistryError::ActualDeliveryTimingOutsideTolerance);
    }

    Ok(DoseDeliveryCheckSummary {
        commanded_dose_mg: commanded.dose_mg,
        actual_dose_mg: measurement.actual_dose_mg,
        minimum_allowed_dose_mg,
        maximum_allowed_dose_mg,
        actual_elapsed_s: measurement.actual_elapsed_s,
        actual_duration_s: measurement.actual_duration_s,
    })
}

pub fn validate_dose_plan(
    formulation: &FormulationCalibrationProfile,
    request: &DosePlanRequest,
) -> Result<DosePlanCheckSummary, ChemistryError> {
    validate_formulation(formulation)?;

    if !canonical_non_empty(&request.soil_class) {
        return Err(ChemistryError::InvalidRequestScope("soil_class"));
    }
    if !canonical_non_empty(&request.surface_material) {
        return Err(ChemistryError::InvalidRequestScope("surface_material"));
    }
    if request.events.is_empty() {
        return Err(ChemistryError::EmptyDosePlan);
    }

    let stage_windows = validate_stage_windows(&request.stage_windows, request.cycle_duration_s)?;
    let calibration_by_id: HashMap<String, &ComponentCalibration> = formulation
        .components
        .iter()
        .map(|component| (component.component_id.clone(), component))
        .collect();

    let mut event_ids: HashSet<String> = HashSet::with_capacity(request.events.len());
    let mut component_totals: HashMap<String, u64> = HashMap::new();
    let mut total_dose_mg = 0_u64;

    for event in &request.events {
        if !canonical_non_empty(&event.event_id) {
            return Err(ChemistryError::InvalidEventId);
        }
        if !event_ids.insert(event.event_id.clone()) {
            return Err(ChemistryError::DuplicateEventId(event.event_id.clone()));
        }
        if event.release_duration_s == 0 {
            return Err(ChemistryError::InvalidEventDuration {
                event_id: event.event_id.clone(),
            });
        }
        let calibration = calibration_by_id
            .get(&event.component_id)
            .copied()
            .ok_or_else(|| ChemistryError::UnknownComponent(event.component_id.clone()))?;

        if !calibration
            .supported_soil_classes
            .iter()
            .any(|soil| soil == &request.soil_class)
        {
            return Err(ChemistryError::UnsupportedSoil {
                component_id: event.component_id.clone(),
                soil_class: request.soil_class.clone(),
            });
        }
        if !calibration
            .supported_surface_materials
            .iter()
            .any(|surface| surface == &request.surface_material)
        {
            return Err(ChemistryError::UnsupportedSurface {
                component_id: event.component_id.clone(),
                surface_material: request.surface_material.clone(),
            });
        }
        if request.water_hardness_ppm_as_caco3
            < calibration.minimum_water_hardness_ppm_as_caco3
            || request.water_hardness_ppm_as_caco3
                > calibration.maximum_water_hardness_ppm_as_caco3
        {
            return Err(ChemistryError::WaterHardnessOutsideCalibration {
                component_id: event.component_id.clone(),
            });
        }
        if !calibration.allowed_stages.contains(&event.stage) {
            return Err(ChemistryError::StageNotCalibrated {
                component_id: event.component_id.clone(),
                stage: event.stage,
            });
        }
        let window = stage_windows
            .get(&event.stage)
            .ok_or(ChemistryError::EventOutsideStageWindow {
                event_id: event.event_id.clone(),
            })?;
        let release_end_s =
            u64::from(event.elapsed_s) + u64::from(event.release_duration_s);
        if event.elapsed_s < window.start_s || release_end_s > u64::from(window.end_s) {
            return Err(ChemistryError::EventOutsideStageWindow {
                event_id: event.event_id.clone(),
            });
        }
        let Some(stage_temperature_c) = window.water_temperature_c else {
            return Err(ChemistryError::TemperatureOutsideCalibration {
                component_id: event.component_id.clone(),
            });
        };
        if stage_temperature_c < calibration.minimum_temperature_c
            || stage_temperature_c > calibration.maximum_temperature_c
        {
            return Err(ChemistryError::TemperatureOutsideCalibration {
                component_id: event.component_id.clone(),
            });
        }
        if event.dose_mg < calibration.minimum_release_mg
            || event.dose_mg > calibration.maximum_release_mg
        {
            return Err(ChemistryError::DoseOutsideCalibration {
                event_id: event.event_id.clone(),
            });
        }

        let component_total = component_totals
            .entry(calibration.component_id.clone())
            .or_insert(0_u64);
        *component_total = (*component_total)
            .checked_add(event.dose_mg)
            .ok_or_else(|| ChemistryError::ComponentDoseOverflow {
                component_id: calibration.component_id.clone(),
            })?;
        total_dose_mg = total_dose_mg
            .checked_add(event.dose_mg)
            .ok_or(ChemistryError::TotalDoseOverflow)?;
        if *component_total > calibration.maximum_cycle_mg {
            return Err(ChemistryError::ComponentCycleLimitExceeded {
                component_id: calibration.component_id.clone(),
            });
        }
        if total_dose_mg > formulation.maximum_total_dose_mg {
            return Err(ChemistryError::TotalDoseLimitExceeded);
        }
    }

    for interaction in &formulation.interactions {
        let first: Vec<&DoseEvent> = request
            .events
            .iter()
            .filter(|event| event.component_id == interaction.first_component_id)
            .collect();
        let second: Vec<&DoseEvent> = request
            .events
            .iter()
            .filter(|event| event.component_id == interaction.second_component_id)
            .collect();

        for left in &first {
            for right in &second {
                let left_start = u64::from(left.elapsed_s);
                let left_end = left_start + u64::from(left.release_duration_s);
                let right_start = u64::from(right.elapsed_s);
                let right_end = right_start + u64::from(right.release_duration_s);
                let separation_s = if left_end <= right_start {
                    right_start - left_end
                } else if right_end <= left_start {
                    left_start - right_end
                } else {
                    0
                };
                if separation_s < u64::from(interaction.minimum_separation_s) {
                    return Err(ChemistryError::IncompatibleReleaseTiming {
                        first_event_id: left.event_id.clone(),
                        second_event_id: right.event_id.clone(),
                        required_separation_s: interaction.minimum_separation_s,
                    });
                }
            }
        }
    }

    Ok(DosePlanCheckSummary {
        event_count: request.events.len(),
        component_count: component_totals.len(),
        total_dose_mg,
        cycle_duration_s: request.cycle_duration_s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prewash_window() -> StageWindow {
        StageWindow {
            stage: WashStage::Prewash,
            start_s: 0,
            end_s: 300,
            water_temperature_c: Some(40.0),
        }
    }

    fn delivery_calibration() -> DoseDeliveryCalibration {
        DoseDeliveryCalibration {
            component_id: "enzyme".to_owned(),
            dispenser_channel_id: "channel-a".to_owned(),
            calibration_id: "dispenser-channel-a-v1".to_owned(),
            calibration_evidence_sha256: "d".repeat(64),
            maximum_underdelivery_bps: 500,
            maximum_overdelivery_bps: 500,
            maximum_timing_deviation_s: 2,
        }
    }

    fn delivery_measurement(event: &DoseEvent) -> DoseDeliveryMeasurement {
        DoseDeliveryMeasurement {
            event_id: event.event_id.clone(),
            component_id: event.component_id.clone(),
            dispenser_channel_id: "channel-a".to_owned(),
            calibration_id: "dispenser-channel-a-v1".to_owned(),
            actual_dose_mg: event.dose_mg + 10,
            actual_elapsed_s: event.elapsed_s + 1,
            actual_duration_s: event.release_duration_s,
            measurement_evidence_sha256: "e".repeat(64),
            calibration_bundle_sha256: "f".repeat(64),
        }
    }

    #[test]
    fn accepts_reported_delivery_inside_calibrated_integer_mass_and_timing_tolerance() {
        let event = request().events.into_iter().next().unwrap();
        let summary = check_reported_dose_delivery(
            &event,
            &prewash_window(),
            &delivery_calibration(),
            &delivery_measurement(&event),
        )
        .unwrap();
        assert_eq!(summary.commanded_dose_mg, 500);
        assert_eq!(summary.actual_dose_mg, 510);
        assert_eq!(summary.minimum_allowed_dose_mg, 475);
        assert_eq!(summary.maximum_allowed_dose_mg, 525);
    }

    #[test]
    fn rejects_reported_under_and_over_delivery() {
        let event = request().events.remove(0);
        let mut under = delivery_measurement(&event);
        under.actual_dose_mg = 474;
        assert!(matches!(
            check_reported_dose_delivery(&event, &prewash_window(), &delivery_calibration(), &under),
            Err(ChemistryError::ActualDoseOutsideTolerance)
        ));

        let mut over = delivery_measurement(&event);
        over.actual_dose_mg = 526;
        assert!(matches!(
            check_reported_dose_delivery(&event, &prewash_window(), &delivery_calibration(), &over),
            Err(ChemistryError::ActualDoseOutsideTolerance)
        ));
    }

    #[test]
    fn rejects_actual_delivery_that_spills_past_phase_boundary() {
        let mut event = request().events.into_iter().next().unwrap();
        event.elapsed_s = 295;
        event.release_duration_s = 5;
        let mut actual = delivery_measurement(&event);
        actual.actual_elapsed_s = 296;
        actual.actual_duration_s = 5;
        assert!(matches!(
            check_reported_dose_delivery(
                &event,
                &prewash_window(),
                &delivery_calibration(),
                &actual
            ),
            Err(ChemistryError::ActualDeliveryOutsideStageWindow)
        ));
    }

    #[test]
    fn rejects_reported_delivery_timing_deviation() {
        let event = request().events.remove(0);
        let mut measurement = delivery_measurement(&event);
        measurement.actual_elapsed_s += 3;
        assert!(matches!(
            check_reported_dose_delivery(&event, &prewash_window(), &delivery_calibration(), &measurement),
            Err(ChemistryError::ActualDeliveryTimingOutsideTolerance)
        ));
    }

    #[test]
    fn rejects_delivery_from_wrong_component_or_dispenser_channel() {
        let event = request().events.into_iter().next().unwrap();

        let mut wrong_component = delivery_measurement(&event);
        wrong_component.component_id = "bleach".to_owned();
        assert!(matches!(
            check_reported_dose_delivery(
                &event,
                &prewash_window(),
                &delivery_calibration(),
                &wrong_component
            ),
            Err(ChemistryError::DeliveryComponentMismatch)
        ));

        let mut wrong_channel = delivery_measurement(&event);
        wrong_channel.dispenser_channel_id = "channel-b".to_owned();
        assert!(matches!(
            check_reported_dose_delivery(
                &event,
                &prewash_window(),
                &delivery_calibration(),
                &wrong_channel
            ),
            Err(ChemistryError::DeliveryChannelMismatch)
        ));
    }

    #[test]
    fn rejects_delivery_event_calibration_and_evidence_mismatch() {
        let event = request().events.remove(0);
        let mut wrong_event = delivery_measurement(&event);
        wrong_event.event_id = "another-event".to_owned();
        assert!(matches!(
            check_reported_dose_delivery(&event, &prewash_window(), &delivery_calibration(), &wrong_event),
            Err(ChemistryError::DeliveryEventMismatch)
        ));

        let mut wrong_calibration = delivery_measurement(&event);
        wrong_calibration.calibration_id = "other-channel".to_owned();
        assert!(matches!(
            check_reported_dose_delivery(&event, &prewash_window(), &delivery_calibration(), &wrong_calibration),
            Err(ChemistryError::DeliveryCalibrationMismatch)
        ));

        let mut bad_evidence = delivery_measurement(&event);
        bad_evidence.measurement_evidence_sha256 = "missing".to_owned();
        assert!(matches!(
            check_reported_dose_delivery(&event, &prewash_window(), &delivery_calibration(), &bad_evidence),
            Err(ChemistryError::InvalidDeliveryEvidence)
        ));
    }

    #[test]
    fn rejects_invalid_delivery_tolerance_profile() {
        let event = request().events.remove(0);
        let mut calibration = delivery_calibration();
        calibration.maximum_underdelivery_bps = 10_000;
        assert!(matches!(
            check_reported_dose_delivery(&event, &prewash_window(), &calibration, &delivery_measurement(&event)),
            Err(ChemistryError::InvalidDeliveryCalibration)
        ));
    }

    fn calibration(
        component_id: &str,
        kind: ComponentKind,
        stages: Vec<WashStage>,
        max_cycle: u64,
    ) -> ComponentCalibration {
        ComponentCalibration {
            component_id: component_id.to_owned(),
            kind,
            calibration_id: format!("calibration-{component_id}-v1"),
            calibration_evidence_sha256: "a".repeat(64),
            allowed_stages: stages,
            supported_soil_classes: vec!["starch".to_owned(), "protein".to_owned()],
            supported_surface_materials: vec![
                "ceramic".to_owned(),
                "glass".to_owned(),
                "stainless_steel".to_owned(),
            ],
            minimum_temperature_c: 35.0,
            maximum_temperature_c: 65.0,
            minimum_water_hardness_ppm_as_caco3: 50,
            maximum_water_hardness_ppm_as_caco3: 250,
            minimum_release_mg: 100,
            maximum_release_mg: 2_000,
            maximum_cycle_mg: max_cycle,
        }
    }

    fn formulation() -> FormulationCalibrationProfile {
        FormulationCalibrationProfile {
            formulation_id: "test-formulation-v1".to_owned(),
            formulation_evidence_sha256: "b".repeat(64),
            maximum_total_dose_mg: 3_000,
            components: vec![
                calibration(
                    "enzyme",
                    ComponentKind::EnzymeBlend,
                    vec![WashStage::Prewash, WashStage::MainWash],
                    2_000,
                ),
                calibration(
                    "bleach",
                    ComponentKind::OxidizingBleach,
                    vec![WashStage::MainWash],
                    1_000,
                ),
                calibration(
                    "rinse-aid",
                    ComponentKind::RinseAid,
                    vec![WashStage::FinalRinse],
                    500,
                ),
            ],
            interactions: vec![ComponentInteraction {
                first_component_id: "enzyme".to_owned(),
                second_component_id: "bleach".to_owned(),
                minimum_separation_s: 120,
                compatibility_evidence_sha256: "c".repeat(64),
            }],
        }
    }

    fn request() -> DosePlanRequest {
        DosePlanRequest {
            soil_class: "starch".to_owned(),
            surface_material: "ceramic".to_owned(),
            water_hardness_ppm_as_caco3: 120,
            cycle_duration_s: 1_800,
            stage_windows: vec![
                StageWindow { stage: WashStage::Prewash, start_s: 0, end_s: 300, water_temperature_c: Some(40.0) },
                StageWindow { stage: WashStage::MainWash, start_s: 300, end_s: 1_200, water_temperature_c: Some(55.0) },
                StageWindow { stage: WashStage::FinalRinse, start_s: 1_200, end_s: 1_500, water_temperature_c: Some(45.0) },
                StageWindow { stage: WashStage::Drying, start_s: 1_500, end_s: 1_800, water_temperature_c: None },
            ],
            events: vec![
                DoseEvent {
                    event_id: "enzyme-pre".to_owned(),
                    component_id: "enzyme".to_owned(),
                    stage: WashStage::Prewash,
                    elapsed_s: 120,
                    release_duration_s: 5,
                    dose_mg: 500,
                },
                DoseEvent {
                    event_id: "bleach-main".to_owned(),
                    component_id: "bleach".to_owned(),
                    stage: WashStage::MainWash,
                    elapsed_s: 600,
                    release_duration_s: 5,
                    dose_mg: 400,
                },
                DoseEvent {
                    event_id: "rinse-aid-final".to_owned(),
                    component_id: "rinse-aid".to_owned(),
                    stage: WashStage::FinalRinse,
                    elapsed_s: 1_350,
                    release_duration_s: 5,
                    dose_mg: 200,
                },
            ],
        }
    }

    #[test]
    fn validates_a_fully_scoped_calibrated_schedule() {
        let result = validate_dose_plan(&formulation(), &request()).unwrap();
        assert_eq!(result.event_count, 3);
        assert_eq!(result.component_count, 3);
        assert_eq!(result.total_dose_mg, 1_100);
    }

    #[test]
    fn validates_temperature_at_each_dosing_phase_not_one_global_value() {
        let mut input = request();
        input.stage_windows[0].water_temperature_c = Some(70.0);
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::TemperatureOutsideCalibration { .. })
        ));
    }

    #[test]
    fn rejects_a_stage_order_that_runs_backward_in_time() {
        let mut input = request();
        input.stage_windows[1].start_s = 1_000;
        input.stage_windows[1].end_s = 1_100;
        input.stage_windows[2].start_s = 500;
        input.stage_windows[2].end_s = 800;
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::StageOrderViolation)
                | Err(ChemistryError::OverlappingStageWindows)
        ));
    }

    #[test]
    fn rejects_dosing_at_the_exclusive_stage_end() {
        let mut input = request();
        input.events[0].elapsed_s = 300;
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::EventOutsideStageWindow { .. })
        ));
    }

    #[test]
    fn rejects_component_outside_soil_envelope() {
        let mut profile = formulation();
        profile.components[0].supported_soil_classes = vec!["protein".to_owned()];
        assert!(matches!(
            validate_dose_plan(&profile, &request()),
            Err(ChemistryError::UnsupportedSoil { .. })
        ));
    }

    #[test]
    fn rejects_temperature_outside_calibration() {
        let mut input = request();
        input.stage_windows[2].water_temperature_c = Some(70.0);
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::TemperatureOutsideCalibration { .. })
        ));
    }

    #[test]
    fn rejects_water_hardness_outside_calibration() {
        let mut input = request();
        input.water_hardness_ppm_as_caco3 = 300;
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::WaterHardnessOutsideCalibration { .. })
        ));
    }

    #[test]
    fn rejects_component_in_unsupported_stage() {
        let mut input = request();
        input.events[2].stage = WashStage::MainWash;
        input.events[2].elapsed_s = 1_000;
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::StageNotCalibrated { .. })
        ));
    }

    #[test]
    fn rejects_release_outside_stage_window() {
        let mut input = request();
        input.events[0].elapsed_s = 600;
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::EventOutsideStageWindow { .. })
        ));
    }

    #[test]
    fn rejects_per_event_and_total_component_overdose() {
        let mut input = request();
        input.events[0].dose_mg = 2_100;
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::DoseOutsideCalibration { .. })
        ));

        let mut input = request();
        input.events[0].dose_mg = 1_100;
        input.events.push(DoseEvent {
            event_id: "enzyme-pre-2".to_owned(),
            component_id: "enzyme".to_owned(),
            stage: WashStage::Prewash,
            elapsed_s: 240,
            release_duration_s: 5,
            dose_mg: 1_000,
        });
        input.events[1].elapsed_s = 700;
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::ComponentCycleLimitExceeded { .. })
        ));
    }

    #[test]
    fn rejects_integer_dose_accumulator_overflow() {
        let mut profile = formulation();
        profile.maximum_total_dose_mg = u64::MAX;
        profile.components[0].maximum_release_mg = u64::MAX;
        profile.components[0].maximum_cycle_mg = u64::MAX;
        let mut input = request();
        input.events = vec![
            DoseEvent {
                event_id: "max-one".to_owned(),
                component_id: "enzyme".to_owned(),
                stage: WashStage::Prewash,
                elapsed_s: 10,
                release_duration_s: 1,
                dose_mg: u64::MAX,
            },
            DoseEvent {
                event_id: "max-two".to_owned(),
                component_id: "enzyme".to_owned(),
                stage: WashStage::Prewash,
                elapsed_s: 20,
                release_duration_s: 1,
                dose_mg: 1,
            },
        ];
        assert!(matches!(
            validate_dose_plan(&profile, &input),
            Err(ChemistryError::ComponentDoseOverflow { .. })
        ));
    }

    #[test]
    fn rejects_total_dose_accumulator_overflow_across_components() {
        let mut profile = formulation();
        profile.maximum_total_dose_mg = u64::MAX;
        profile.components[0].maximum_release_mg = u64::MAX;
        profile.components[0].maximum_cycle_mg = u64::MAX;
        profile.components[1].maximum_release_mg = u64::MAX;
        profile.components[1].maximum_cycle_mg = u64::MAX;
        let mut input = request();
        input.events = vec![
            DoseEvent {
                event_id: "max-enzyme".to_owned(),
                component_id: "enzyme".to_owned(),
                stage: WashStage::Prewash,
                elapsed_s: 10,
                release_duration_s: 1,
                dose_mg: u64::MAX,
            },
            DoseEvent {
                event_id: "one-mg-bleach".to_owned(),
                component_id: "bleach".to_owned(),
                stage: WashStage::MainWash,
                elapsed_s: 600,
                release_duration_s: 1,
                dose_mg: 1,
            },
        ];
        assert!(matches!(
            validate_dose_plan(&profile, &input),
            Err(ChemistryError::TotalDoseOverflow)
        ));
    }

    #[test]
    fn rejects_release_that_starts_inside_but_finishes_after_phase() {
        let mut input = request();
        input.events[0].elapsed_s = 299;
        input.events[0].release_duration_s = 2;
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::EventOutsideStageWindow { .. })
        ));
    }

    #[test]
    fn interaction_separation_is_measured_after_release_completion() {
        let mut input = request();
        input.events[0].stage = WashStage::MainWash;
        input.events[0].elapsed_s = 400;
        input.events[0].release_duration_s = 100;
        input.events[1].elapsed_s = 600;
        input.events[1].release_duration_s = 5;
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::IncompatibleReleaseTiming { .. })
        ));
    }

    #[test]
    fn invalid_zero_duration_release_is_rejected() {
        let mut input = request();
        input.events[0].release_duration_s = 0;
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::InvalidEventDuration { .. })
        ));
    }

    #[test]
    fn rejects_incompatible_component_timing() {
        let mut input = request();
        input.events[0].stage = WashStage::MainWash;
        input.events[0].elapsed_s = 590;
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::IncompatibleReleaseTiming { .. })
        ));
    }

    #[test]
    fn rejects_total_formulation_limit() {
        let mut profile = formulation();
        profile.maximum_total_dose_mg = 800;
        assert!(matches!(
            validate_dose_plan(&profile, &request()),
            Err(ChemistryError::TotalDoseLimitExceeded)
        ));
    }

    #[test]
    fn rejects_duplicate_or_noncanonical_calibration_scope_entries() {
        let mut profile = formulation();
        profile.components[0]
            .supported_soil_classes
            .push("starch".to_owned());
        assert!(matches!(
            validate_dose_plan(&profile, &request()),
            Err(ChemistryError::InvalidComponentCalibration { .. })
        ));

        let mut profile = formulation();
        profile.components[0]
            .allowed_stages
            .push(WashStage::Prewash);
        assert!(matches!(
            validate_dose_plan(&profile, &request()),
            Err(ChemistryError::InvalidComponentCalibration { .. })
        ));

        let mut profile = formulation();
        profile.components[0].component_id = " enzyme".to_owned();
        assert!(matches!(
            validate_dose_plan(&profile, &request()),
            Err(ChemistryError::InvalidComponentCalibration { .. })
        ));
    }

    #[test]
    fn rejects_missing_evidence_or_unknown_component() {
        let mut profile = formulation();
        profile.components[0].calibration_evidence_sha256 = "missing".to_owned();
        assert!(matches!(
            validate_dose_plan(&profile, &request()),
            Err(ChemistryError::InvalidEvidenceDigest("component calibration"))
        ));

        let mut input = request();
        input.events[0].component_id = "unvalidated".to_owned();
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::UnknownComponent(_))
        ));
    }

    #[test]
    fn rejects_duplicate_events_and_empty_plans() {
        let mut input = request();
        input.events[1].event_id = input.events[0].event_id.clone();
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::DuplicateEventId(_))
        ));

        input.events.clear();
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::EmptyDosePlan)
        ));
    }

    #[test]
    fn rejects_duplicate_stage_windows() {
        let mut input = request();
        input.stage_windows.push(StageWindow {
            stage: WashStage::Prewash,
            start_s: 1_800,
            end_s: 1_900,
            water_temperature_c: Some(40.0),
        });
        input.cycle_duration_s = 2_000;
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::DuplicateStageWindow(WashStage::Prewash))
        ));
    }

    #[test]
    fn rejects_invalid_stage_windows() {
        let mut input = request();
        input.stage_windows[1].start_s = 200;
        assert!(matches!(
            validate_dose_plan(&formulation(), &input),
            Err(ChemistryError::OverlappingStageWindows)
        ));
    }
}
