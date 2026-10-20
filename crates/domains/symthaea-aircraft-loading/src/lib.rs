//! Deterministic, fail-closed arithmetic audit for a declared aircraft loading state.
//!
//! This crate is a research prototype. It does not issue a load release, dispatch a flight,
//! authenticate an aircraft flight manual, prove cargo restraint, or replace an operator's
//! approved weight-and-balance programme. All masses are kilograms and all arms/moments are
//! metres and kilogram-metres relative to the aircraft profile's declared datum.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

/// A bounded aircraft configuration and its declared loading limits.
///
/// The source and approval references are provenance fields supplied by the caller. The
/// auditor validates their presence and the SHA-256 syntax; it does not authenticate the
/// referenced documents, signatures, approval status, or applicability to a real airframe.
#[derive(Debug, Clone, PartialEq)]
pub struct AircraftProfile {
    /// Stable aircraft model/configuration identifier.
    pub aircraft_id: String,
    /// Exact revision of the configuration used for this audit.
    pub configuration_revision: String,
    /// Reference to the applicable approved aircraft/weight-and-balance data.
    pub source_reference: String,
    /// SHA-256 digest claimed for the source bytes (64 hexadecimal characters).
    pub source_sha256: String,
    /// Reference to operator/authority approval evidence; presence is not authentication.
    pub approval_reference: String,
    /// Basic empty aircraft mass in kilograms.
    pub empty_mass_kg: f64,
    /// Basic empty aircraft moment in kilogram-metres about the declared datum.
    pub empty_moment_kg_m: f64,
    /// Maximum take-off mass in kilograms for this configuration.
    pub maximum_takeoff_mass_kg: f64,
    /// Loading stations with declared arms and payload-mass caps.
    pub stations: Vec<LoadingStation>,
    /// Declared center-of-gravity envelope slices keyed by total aircraft mass.
    pub cg_envelope: Vec<CgEnvelopeBand>,
}

/// A loading station and its declared payload-mass limit.
///
/// This is a simple station cap, not a substitute for floor pressure, running loads,
/// local structural limits, restraint geometry, or cargo-bay spatial fit.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadingStation {
    /// Unique station identifier within an aircraft profile.
    pub station_id: String,
    /// Station arm in metres from the profile's datum.
    pub arm_m: f64,
    /// Maximum aggregate loaded-item mass assigned to this station, in kilograms.
    pub maximum_station_load_kg: f64,
}

/// One mass-dependent slice of the aircraft's approved CG envelope.
///
/// Both mass endpoints are inclusive. If two slices overlap at the audited mass, the result
/// is deliberately reported as not assessable rather than choosing one arbitrarily.
#[derive(Debug, Clone, PartialEq)]
pub struct CgEnvelopeBand {
    /// Minimum total aircraft mass covered by this slice, in kilograms.
    pub minimum_mass_kg: f64,
    /// Maximum total aircraft mass covered by this slice, in kilograms.
    pub maximum_mass_kg: f64,
    /// Forward CG bound in metres from the declared datum.
    pub forward_cg_m: f64,
    /// Aft CG bound in metres from the declared datum.
    pub aft_cg_m: f64,
}

/// A loading-phase label for a declared weight-and-balance state.
///
/// The prototype computes results only for Takeoff. Other phases are present in the enum so
/// callers cannot silently relabel a ramp/taxi/landing state as a take-off audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadingPhase {
    /// Aircraft state at take-off; the only phase currently evaluated.
    Takeoff,
    /// Aircraft state on the ramp; not yet supported by the prototype.
    Ramp,
    /// Aircraft state during taxi; not yet supported by the prototype.
    Taxi,
    /// Aircraft state at landing; not yet supported by the prototype.
    Landing,
}

/// A loading scenario evaluated against exactly one aircraft profile revision.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadingScenario {
    /// Stable manifest/scenario identifier.
    pub scenario_id: String,
    /// Explicit mass/operating phase for the supplied item list.
    pub phase: LoadingPhase,
    /// Aircraft identifier copied from the source profile.
    pub aircraft_id: String,
    /// Exact configuration revision copied from the source profile.
    pub configuration_revision: String,
    /// Loaded items for this one declared mass state.
    pub items: Vec<LoadedItem>,
}

/// Type of an item contributing mass at a station.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LoadedItemKind {
    /// Freight, baggage, or other cargo.
    Cargo,
    /// Passenger or crew mass represented at a declared station.
    Occupant,
    /// Fuel currently present at a declared tank/station.
    Fuel,
    /// Explicit ballast mass.
    Ballast,
}

/// One item assigned to one station in the loading scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedItem {
    /// Unique item identifier within the scenario.
    pub item_id: String,
    /// Item category.
    pub kind: LoadedItemKind,
    /// Target loading station identifier.
    pub station_id: String,
    /// Item mass in kilograms.
    pub mass_kg: f64,
    /// Locator for separate securement/restraint evidence, when applicable.
    ///
    /// The locator is not checked for authenticity or engineering adequacy. A missing locator
    /// for cargo, occupants, or ballast leaves the scenario not assessable for human review.
    pub securement_evidence_reference: Option<String>,
}

/// Outcome of the arithmetic and declared-limit checks.
///
/// None of these outcomes authorizes dispatch, flight, loading, or release to service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditDisposition {
    /// At least one declared hard limit is exceeded.
    BlockedByDeclaredLimit,
    /// A required check could not be assessed with the supplied scenario/profile.
    NotAssessable,
    /// Arithmetic fits the supplied declarations; independent human approval is still required.
    WithinDeclaredEnvelopeForHumanReview,
}

/// Severity of a loading audit finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FindingSeverity {
    /// A declared numeric or station limit is violated.
    Blocking,
    /// Required evidence or a uniquely applicable envelope is absent/ambiguous.
    Unknown,
}

/// Stable code for a report finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FindingCode {
    /// Aggregate mass at a station exceeds its declared maximum.
    StationLoadExceeded,
    /// Total aircraft mass exceeds the declared maximum take-off mass.
    MaximumTakeoffMassExceeded,
    /// Calculated CG falls outside the uniquely applicable declared envelope.
    CenterOfGravityOutsideEnvelope,
    /// No declared CG envelope band covers the calculated mass.
    CenterOfGravityEnvelopeMissing,
    /// Multiple envelope bands cover the same calculated mass.
    CenterOfGravityEnvelopeAmbiguous,
    /// Securement evidence locator is missing for an item that requires it.
    SecurementEvidenceMissing,
}

/// One finding; deterministic ordering is applied before a report is returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditFinding {
    /// Stable finding code.
    pub code: FindingCode,
    /// Whether this finding blocks the declared scenario or leaves it unknown.
    pub severity: FindingSeverity,
    /// Item or station identifier most directly associated with this finding.
    pub subject_id: Option<String>,
    /// Human-readable explanation with units where relevant.
    pub detail: String,
}

/// Aggregate mass at a station in the audited scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct StationLoadSummary {
    /// Station identifier.
    pub station_id: String,
    /// Aggregate loaded-item mass assigned to this station, in kilograms.
    pub loaded_mass_kg: f64,
    /// Declared station cap in kilograms.
    pub maximum_station_load_kg: f64,
}

/// Result of deterministic mass, moment, station, and declared-CG-envelope checks.
#[derive(Debug, Clone, PartialEq)]
pub struct AuditReport {
    /// Scenario identifier supplied by the caller.
    pub scenario_id: String,
    /// Aircraft identifier used for the audit.
    pub aircraft_id: String,
    /// Exact configuration revision used for the audit.
    pub configuration_revision: String,
    /// Computed total mass, including basic empty mass, in kilograms.
    pub total_mass_kg: f64,
    /// Computed total moment, including basic empty moment, in kilogram-metres.
    pub total_moment_kg_m: f64,
    /// Computed center of gravity in metres from the declared datum.
    pub center_of_gravity_m: f64,
    /// Station masses, sorted by station identifier.
    pub station_loads: Vec<StationLoadSummary>,
    /// Overall disposition; never a dispatch or flight authorization.
    pub disposition: AuditDisposition,
    /// Findings sorted by severity, code, and subject identifier.
    pub findings: Vec<AuditFinding>,
    /// Always false: this prototype checks provenance syntax, not source authenticity.
    pub profile_source_authenticated: bool,
}


/// Declared bounded uncertainty for one loaded item and its effective station arm.
///
/// Bounds must contain the nominal item mass and the profile's nominal station arm. Arm
/// uncertainty can represent approved compartment placement tolerances or another externally
/// justified bounded variation; this type does not establish that the bounds are physically valid.
#[derive(Debug, Clone, PartialEq)]
pub struct ItemLoadUncertainty {
    /// Item identifier matching exactly one item in the scenario.
    pub item_id: String,
    /// Lowest possible item mass, in kilograms; must be positive.
    pub minimum_mass_kg: f64,
    /// Highest possible item mass, in kilograms.
    pub maximum_mass_kg: f64,
    /// Lowest possible effective longitudinal arm, in metres from the profile datum.
    pub minimum_station_arm_m: f64,
    /// Highest possible effective longitudinal arm, in metres from the profile datum.
    pub maximum_station_arm_m: f64,
}

/// Bounded empty-aircraft and per-item uncertainty for a single declared take-off scenario.
///
/// Empty mass and moment ranges must contain the values in the profile. Every scenario item must
/// have exactly one matching item range. This is an input model only; bounds and source evidence
/// are not authenticated by the calculation kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadingUncertaintyBounds {
    /// Lower bound on basic empty aircraft mass, in kilograms.
    pub minimum_empty_mass_kg: f64,
    /// Upper bound on basic empty aircraft mass, in kilograms.
    pub maximum_empty_mass_kg: f64,
    /// Lower bound on basic empty aircraft moment, in kilogram-metres.
    pub minimum_empty_moment_kg_m: f64,
    /// Upper bound on basic empty aircraft moment, in kilogram-metres.
    pub maximum_empty_moment_kg_m: f64,
    /// One bounded uncertainty declaration per scenario item.
    pub items: Vec<ItemLoadUncertainty>,
}

/// Conservative mass, moment and CG interval propagated from declared input bounds.
///
/// The interval arithmetic deliberately ignores correlations between total mass and total moment,
/// so the CG interval may be wider than the set of physically attainable CG values. It is intended
/// to expose uncertainty for further review, not to establish compliance with a CG envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadingUncertaintyEnvelope {
    /// Scenario identity associated with the analysis.
    pub scenario_id: String,
    /// Aircraft identity associated with the analysis.
    pub aircraft_id: String,
    /// Exact configuration revision used by the analysis.
    pub configuration_revision: String,
    /// Conservative lower bound on total mass, in kilograms.
    pub minimum_total_mass_kg: f64,
    /// Conservative upper bound on total mass, in kilograms.
    pub maximum_total_mass_kg: f64,
    /// Conservative lower bound on total moment, in kilogram-metres.
    pub minimum_total_moment_kg_m: f64,
    /// Conservative upper bound on total moment, in kilogram-metres.
    pub maximum_total_moment_kg_m: f64,
    /// Conservative lower bound on center of gravity, in metres from the profile datum.
    pub minimum_center_of_gravity_m: f64,
    /// Conservative upper bound on center of gravity, in metres from the profile datum.
    pub maximum_center_of_gravity_m: f64,
}

/// Input/configuration error that prevents a meaningful arithmetic audit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditInputError {
    /// A source profile has missing or invalid required fields.
    InvalidProfile(&'static str),
    /// Scenario aircraft identity or revision differs from the supplied profile.
    ProfileIdentityMismatch,
    /// A station identifier occurs more than once in the profile.
    DuplicateStationId(String),
    /// An item identifier occurs more than once in the scenario.
    DuplicateItemId(String),
    /// An item refers to a station absent from the profile.
    UnknownStationId(String),
    /// The requested loading phase has no qualified profile/calculation in this prototype.
    UnsupportedLoadingPhase(LoadingPhase),
    /// A loaded item has a missing identifier or non-positive/non-finite mass.
    InvalidLoadedItem(String),
    /// A finite-input arithmetic operation produced a non-finite result.
    NumericOverflow,
    /// A non-zero contribution was lost to floating-point underflow or addition rounding.
    NumericPrecisionLoss,
    /// An uncertainty range is malformed, incomplete, or does not contain its nominal value.
    InvalidUncertainty(&'static str),
}

impl fmt::Display for AuditInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProfile(reason) => write!(f, "invalid aircraft profile: {reason}"),
            Self::ProfileIdentityMismatch => {
                write!(
                    f,
                    "scenario aircraft identity/revision does not match profile"
                )
            }
            Self::DuplicateStationId(id) => write!(f, "duplicate loading station id: {id}"),
            Self::DuplicateItemId(id) => write!(f, "duplicate loaded item id: {id}"),
            Self::UnknownStationId(id) => write!(f, "unknown loading station id: {id}"),
            Self::UnsupportedLoadingPhase(phase) => {
                write!(f, "unsupported loading phase for aircraft audit: {phase:?}")
            }
            Self::InvalidLoadedItem(id) => write!(f, "invalid loaded item: {id}"),
            Self::NumericOverflow => write!(f, "numeric overflow during loading audit"),
            Self::NumericPrecisionLoss => {
                write!(f, "numeric precision loss during loading audit")
            }
            Self::InvalidUncertainty(reason) => write!(f, "invalid loading uncertainty: {reason}"),
        }
    }
}

impl Error for AuditInputError {}


/// Analyze uncertainty bounds for one take-off loading scenario.
///
/// The nominal profile/scenario are first checked by `audit_loading`; this function then propagates
/// supplied mass and effective-arm intervals using outward-rounded interval arithmetic. The result
/// is intentionally only an uncertainty envelope: it does not decide whether every possible
/// mass/CG combination remains inside the aircraft's mass-dependent approved envelope, and it is
/// not a loading/dispatch authorization.
pub fn analyze_loading_uncertainty(
    profile: &AircraftProfile,
    scenario: &LoadingScenario,
    uncertainty: &LoadingUncertaintyBounds,
) -> Result<LoadingUncertaintyEnvelope, AuditInputError> {
    let _nominal_report = audit_loading(profile, scenario)?;
    if !uncertainty.minimum_empty_mass_kg.is_finite()
        || !uncertainty.maximum_empty_mass_kg.is_finite()
        || uncertainty.minimum_empty_mass_kg <= 0.0
        || uncertainty.minimum_empty_mass_kg > profile.empty_mass_kg
        || profile.empty_mass_kg > uncertainty.maximum_empty_mass_kg
        || !uncertainty.minimum_empty_moment_kg_m.is_finite()
        || !uncertainty.maximum_empty_moment_kg_m.is_finite()
        || uncertainty.minimum_empty_moment_kg_m > profile.empty_moment_kg_m
        || profile.empty_moment_kg_m > uncertainty.maximum_empty_moment_kg_m
    {
        return Err(AuditInputError::InvalidUncertainty(
            "empty mass/moment intervals must be finite, ordered, and contain the profile values",
        ));
    }

    let mut bounds_by_item = BTreeMap::<String, &ItemLoadUncertainty>::new();
    for bounds in &uncertainty.items {
        if bounds.item_id.trim().is_empty()
            || !bounds.minimum_mass_kg.is_finite()
            || !bounds.maximum_mass_kg.is_finite()
            || bounds.minimum_mass_kg <= 0.0
            || bounds.minimum_mass_kg > bounds.maximum_mass_kg
            || !bounds.minimum_station_arm_m.is_finite()
            || !bounds.maximum_station_arm_m.is_finite()
            || bounds.minimum_station_arm_m > bounds.maximum_station_arm_m
        {
            return Err(AuditInputError::InvalidUncertainty(
                "item mass and arm intervals must be finite, positive/ordered as applicable",
            ));
        }
        if bounds_by_item.insert(bounds.item_id.clone(), bounds).is_some() {
            return Err(AuditInputError::InvalidUncertainty(
                "item uncertainty identifiers must be unique",
            ));
        }
    }

    let stations: BTreeMap<String, &LoadingStation> = profile
        .stations
        .iter()
        .map(|station| (station.station_id.clone(), station))
        .collect();
    let mut joined =
        BTreeMap::<String, (&LoadedItem, &LoadingStation, &ItemLoadUncertainty)>::new();
    for item in &scenario.items {
        let bounds = bounds_by_item.remove(&item.item_id).ok_or(
            AuditInputError::InvalidUncertainty(
                "every scenario item must have exactly one uncertainty declaration",
            ),
        )?;
        let station = stations
            .get(&item.station_id)
            .copied()
            .ok_or_else(|| AuditInputError::UnknownStationId(item.station_id.clone()))?;
        if bounds.minimum_mass_kg > item.mass_kg
            || item.mass_kg > bounds.maximum_mass_kg
            || bounds.minimum_station_arm_m > station.arm_m
            || station.arm_m > bounds.maximum_station_arm_m
        {
            return Err(AuditInputError::InvalidUncertainty(
                "item bounds must contain nominal mass and station arm",
            ));
        }
        joined.insert(item.item_id.clone(), (item, station, bounds));
    }
    if !bounds_by_item.is_empty() {
        return Err(AuditInputError::InvalidUncertainty(
            "uncertainty declarations must not name items absent from the scenario",
        ));
    }

    let mut total_mass_min = uncertainty.minimum_empty_mass_kg;
    let mut total_mass_max = uncertainty.maximum_empty_mass_kg;
    let mut total_moment_min = uncertainty.minimum_empty_moment_kg_m;
    let mut total_moment_max = uncertainty.maximum_empty_moment_kg_m;
    for (_item_id, (_item, _station, bounds)) in joined {
        total_mass_min = add_outward_lower(total_mass_min, bounds.minimum_mass_kg)?;
        total_mass_max = add_outward_upper(total_mass_max, bounds.maximum_mass_kg)?;
        let (item_moment_min, item_moment_max) = product_interval(
            bounds.minimum_mass_kg,
            bounds.maximum_mass_kg,
            bounds.minimum_station_arm_m,
            bounds.maximum_station_arm_m,
        )?;
        total_moment_min = add_outward_lower(total_moment_min, item_moment_min)?;
        total_moment_max = add_outward_upper(total_moment_max, item_moment_max)?;
    }
    if total_mass_min <= 0.0 || total_mass_min > total_mass_max {
        return Err(AuditInputError::NumericOverflow);
    }

    // Interval division: denominator is strictly positive, so the quotient is bounded by the
    // extrema of all four numerator/denominator endpoint combinations. This is conservative even
    // though total moment and total mass are correlated through the same loaded items.
    let quotient_corners = [
        total_moment_min / total_mass_min,
        total_moment_min / total_mass_max,
        total_moment_max / total_mass_min,
        total_moment_max / total_mass_max,
    ];
    if quotient_corners.iter().any(|value| !value.is_finite()) {
        return Err(AuditInputError::NumericOverflow);
    }
    let cg_min = quotient_corners.iter().copied().fold(f64::INFINITY, f64::min);
    let cg_max = quotient_corners
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    Ok(LoadingUncertaintyEnvelope {
        scenario_id: scenario.scenario_id.clone(),
        aircraft_id: profile.aircraft_id.clone(),
        configuration_revision: profile.configuration_revision.clone(),
        minimum_total_mass_kg: total_mass_min,
        maximum_total_mass_kg: total_mass_max,
        minimum_total_moment_kg_m: total_moment_min,
        maximum_total_moment_kg_m: total_moment_max,
        minimum_center_of_gravity_m: outward_down(cg_min)?,
        maximum_center_of_gravity_m: outward_up(cg_max)?,
    })
}

fn add_outward_lower(left: f64, right: f64) -> Result<f64, AuditInputError> {
    let sum = left + right;
    if !sum.is_finite() {
        return Err(AuditInputError::NumericOverflow);
    }
    outward_down(sum)
}

fn add_outward_upper(left: f64, right: f64) -> Result<f64, AuditInputError> {
    let sum = left + right;
    if !sum.is_finite() {
        return Err(AuditInputError::NumericOverflow);
    }
    outward_up(sum)
}

fn product_interval(
    mass_min: f64,
    mass_max: f64,
    arm_min: f64,
    arm_max: f64,
) -> Result<(f64, f64), AuditInputError> {
    let corners = [
        mass_min * arm_min,
        mass_min * arm_max,
        mass_max * arm_min,
        mass_max * arm_max,
    ];
    if corners.iter().any(|value| !value.is_finite()) {
        return Err(AuditInputError::NumericOverflow);
    }
    let minimum = corners.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = corners
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    Ok((outward_down(minimum)?, outward_up(maximum)?))
}

fn outward_down(value: f64) -> Result<f64, AuditInputError> {
    let result = value.next_down();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(AuditInputError::NumericOverflow)
    }
}

fn outward_up(value: f64) -> Result<f64, AuditInputError> {
    let result = value.next_up();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(AuditInputError::NumericOverflow)
    }
}

/// Audit one declared take-off loading state against one explicit aircraft profile.
///
/// This function is deterministic and has no external dependencies or side effects. It rejects
/// malformed inputs, profile/revision mismatches, and any phase other than Takeoff. A returned
/// report checks only the values supplied by the caller; its profile references and any
/// securement references are not authenticated. Fuel burn, ramp/taxi/landing phase differences,
/// passenger movement,
/// dangerous-goods segregation, package geometry, floor pressure, restraint strength,
/// aeroelastic effects, structural substantiation, and operational authorization are out of
/// scope for this first increment.
pub fn audit_loading(
    profile: &AircraftProfile,
    scenario: &LoadingScenario,
) -> Result<AuditReport, AuditInputError> {
    validate_profile(profile)?;

    if scenario.scenario_id.trim().is_empty() {
        return Err(AuditInputError::InvalidLoadedItem(
            "scenario id is empty".to_owned(),
        ));
    }
    if scenario.phase != LoadingPhase::Takeoff {
        return Err(AuditInputError::UnsupportedLoadingPhase(scenario.phase));
    }
    if scenario.aircraft_id != profile.aircraft_id
        || scenario.configuration_revision != profile.configuration_revision
    {
        return Err(AuditInputError::ProfileIdentityMismatch);
    }

    let mut stations = BTreeMap::<String, &LoadingStation>::new();
    for station in &profile.stations {
        if stations
            .insert(station.station_id.clone(), station)
            .is_some()
        {
            return Err(AuditInputError::DuplicateStationId(
                station.station_id.clone(),
            ));
        }
    }

    let mut item_ids = BTreeSet::new();
    let mut station_mass = BTreeMap::<String, f64>::new();
    let mut total_mass_kg = profile.empty_mass_kg;
    let mut total_moment_kg_m = profile.empty_moment_kg_m;
    let mut findings = Vec::<AuditFinding>::new();

    for station_id in stations.keys() {
        station_mass.insert(station_id.clone(), 0.0);
    }

    for item in &scenario.items {
        if item.item_id.trim().is_empty() || !item.mass_kg.is_finite() || item.mass_kg <= 0.0 {
            return Err(AuditInputError::InvalidLoadedItem(item.item_id.clone()));
        }
        if !item_ids.insert(item.item_id.clone()) {
            return Err(AuditInputError::DuplicateItemId(item.item_id.clone()));
        }

        let station = stations
            .get(&item.station_id)
            .copied()
            .ok_or_else(|| AuditInputError::UnknownStationId(item.station_id.clone()))?;

        let item_moment = item.mass_kg * station.arm_m;
        if !item_moment.is_finite() {
            return Err(AuditInputError::NumericOverflow);
        }
        if item.mass_kg != 0.0 && station.arm_m != 0.0 && item_moment == 0.0 {
            return Err(AuditInputError::NumericPrecisionLoss);
        }
        total_mass_kg = checked_add(total_mass_kg, item.mass_kg)?;
        total_moment_kg_m = checked_add(total_moment_kg_m, item_moment)?;

        let current_station_mass =
            station_mass
                .get(&item.station_id)
                .copied()
                .ok_or(AuditInputError::InvalidProfile(
                    "station mass accumulator is missing",
                ))?;
        station_mass.insert(
            item.station_id.clone(),
            checked_add(current_station_mass, item.mass_kg)?,
        );

        if item.kind != LoadedItemKind::Fuel
            && item
                .securement_evidence_reference
                .as_deref()
                .is_none_or(|reference| reference.trim().is_empty())
        {
            findings.push(AuditFinding {
                code: FindingCode::SecurementEvidenceMissing,
                severity: FindingSeverity::Unknown,
                subject_id: Some(item.item_id.clone()),
                detail: "no securement evidence reference was supplied; physical restraint is not verified"
                    .to_owned(),
            });
        }
    }

    let center_of_gravity_m = total_moment_kg_m / total_mass_kg;
    if !center_of_gravity_m.is_finite() {
        return Err(AuditInputError::NumericOverflow);
    }
    if total_moment_kg_m != 0.0 && center_of_gravity_m == 0.0 {
        return Err(AuditInputError::NumericPrecisionLoss);
    }

    if total_mass_kg > profile.maximum_takeoff_mass_kg {
        findings.push(AuditFinding {
            code: FindingCode::MaximumTakeoffMassExceeded,
            severity: FindingSeverity::Blocking,
            subject_id: None,
            detail: format!(
                "total mass {:.3} kg exceeds declared maximum take-off mass {:.3} kg",
                total_mass_kg, profile.maximum_takeoff_mass_kg
            ),
        });
    }

    for (station_id, loaded_mass_kg) in &station_mass {
        let station = stations
            .get(station_id)
            .copied()
            .ok_or(AuditInputError::InvalidProfile(
                "station reference is missing from profile index",
            ))?;
        if *loaded_mass_kg > station.maximum_station_load_kg {
            findings.push(AuditFinding {
                code: FindingCode::StationLoadExceeded,
                severity: FindingSeverity::Blocking,
                subject_id: Some(station_id.clone()),
                detail: format!(
                    "station load {:.3} kg exceeds declared limit {:.3} kg",
                    loaded_mass_kg, station.maximum_station_load_kg
                ),
            });
        }
    }

    let matching_bands: Vec<&CgEnvelopeBand> = profile
        .cg_envelope
        .iter()
        .filter(|band| {
            total_mass_kg >= band.minimum_mass_kg && total_mass_kg <= band.maximum_mass_kg
        })
        .collect();

    match matching_bands.as_slice() {
        [] => findings.push(AuditFinding {
            code: FindingCode::CenterOfGravityEnvelopeMissing,
            severity: FindingSeverity::Unknown,
            subject_id: None,
            detail: format!(
                "no declared CG envelope band covers total mass {:.3} kg",
                total_mass_kg
            ),
        }),
        [band] => {
            if center_of_gravity_m < band.forward_cg_m || center_of_gravity_m > band.aft_cg_m {
                findings.push(AuditFinding {
                    code: FindingCode::CenterOfGravityOutsideEnvelope,
                    severity: FindingSeverity::Blocking,
                    subject_id: None,
                    detail: format!(
                        "calculated CG {:.6} m is outside declared bounds [{:.6}, {:.6}] m",
                        center_of_gravity_m, band.forward_cg_m, band.aft_cg_m
                    ),
                });
            }
        }
        _ => findings.push(AuditFinding {
            code: FindingCode::CenterOfGravityEnvelopeAmbiguous,
            severity: FindingSeverity::Unknown,
            subject_id: None,
            detail: format!(
                "{} declared CG envelope bands cover total mass {:.3} kg",
                matching_bands.len(),
                total_mass_kg
            ),
        }),
    }

    findings.sort_by(|a, b| {
        (a.severity, a.code, a.subject_id.as_deref()).cmp(&(
            b.severity,
            b.code,
            b.subject_id.as_deref(),
        ))
    });

    let disposition = if findings
        .iter()
        .any(|finding| finding.severity == FindingSeverity::Blocking)
    {
        AuditDisposition::BlockedByDeclaredLimit
    } else if findings
        .iter()
        .any(|finding| finding.severity == FindingSeverity::Unknown)
    {
        AuditDisposition::NotAssessable
    } else {
        AuditDisposition::WithinDeclaredEnvelopeForHumanReview
    };

    let mut station_loads = Vec::with_capacity(station_mass.len());
    for (station_id, loaded_mass_kg) in station_mass {
        let station = stations
            .get(&station_id)
            .copied()
            .ok_or(AuditInputError::InvalidProfile(
                "station reference is missing from profile index",
            ))?;
        station_loads.push(StationLoadSummary {
            station_id,
            loaded_mass_kg,
            maximum_station_load_kg: station.maximum_station_load_kg,
        });
    }

    Ok(AuditReport {
        scenario_id: scenario.scenario_id.clone(),
        aircraft_id: profile.aircraft_id.clone(),
        configuration_revision: profile.configuration_revision.clone(),
        total_mass_kg,
        total_moment_kg_m,
        center_of_gravity_m,
        station_loads,
        disposition,
        findings,
        profile_source_authenticated: false,
    })
}

fn validate_profile(profile: &AircraftProfile) -> Result<(), AuditInputError> {
    if profile.aircraft_id.trim().is_empty() {
        return Err(AuditInputError::InvalidProfile("aircraft id is empty"));
    }
    if profile.configuration_revision.trim().is_empty() {
        return Err(AuditInputError::InvalidProfile(
            "configuration revision is empty",
        ));
    }
    if profile.source_reference.trim().is_empty() {
        return Err(AuditInputError::InvalidProfile("source reference is empty"));
    }
    if profile.approval_reference.trim().is_empty() {
        return Err(AuditInputError::InvalidProfile(
            "approval reference is empty",
        ));
    }
    if !is_sha256_hex(&profile.source_sha256) {
        return Err(AuditInputError::InvalidProfile(
            "source SHA-256 must contain exactly 64 hexadecimal characters",
        ));
    }
    if !profile.empty_mass_kg.is_finite() || profile.empty_mass_kg <= 0.0 {
        return Err(AuditInputError::InvalidProfile(
            "empty mass must be positive and finite",
        ));
    }
    if !profile.empty_moment_kg_m.is_finite() {
        return Err(AuditInputError::InvalidProfile(
            "empty moment must be finite",
        ));
    }
    if !profile.maximum_takeoff_mass_kg.is_finite() || profile.maximum_takeoff_mass_kg <= 0.0 {
        return Err(AuditInputError::InvalidProfile(
            "maximum take-off mass must be positive and finite",
        ));
    }
    if profile.empty_mass_kg > profile.maximum_takeoff_mass_kg {
        return Err(AuditInputError::InvalidProfile(
            "empty mass exceeds maximum take-off mass",
        ));
    }
    if profile.stations.is_empty() {
        return Err(AuditInputError::InvalidProfile(
            "at least one loading station is required",
        ));
    }
    if profile.cg_envelope.is_empty() {
        return Err(AuditInputError::InvalidProfile(
            "at least one declared CG envelope band is required",
        ));
    }

    for station in &profile.stations {
        if station.station_id.trim().is_empty() {
            return Err(AuditInputError::InvalidProfile(
                "loading station id is empty",
            ));
        }
        if !station.arm_m.is_finite() {
            return Err(AuditInputError::InvalidProfile(
                "station arm must be finite",
            ));
        }
        if !station.maximum_station_load_kg.is_finite() || station.maximum_station_load_kg <= 0.0 {
            return Err(AuditInputError::InvalidProfile(
                "station limit must be positive and finite",
            ));
        }
    }

    for band in &profile.cg_envelope {
        if !band.minimum_mass_kg.is_finite()
            || !band.maximum_mass_kg.is_finite()
            || band.minimum_mass_kg <= 0.0
            || band.maximum_mass_kg < band.minimum_mass_kg
        {
            return Err(AuditInputError::InvalidProfile(
                "CG envelope mass interval is invalid",
            ));
        }
        if !band.forward_cg_m.is_finite()
            || !band.aft_cg_m.is_finite()
            || band.forward_cg_m > band.aft_cg_m
        {
            return Err(AuditInputError::InvalidProfile(
                "CG envelope bounds are invalid",
            ));
        }
    }

    Ok(())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn checked_add(current: f64, increment: f64) -> Result<f64, AuditInputError> {
    let result = current + increment;
    if !result.is_finite() {
        return Err(AuditInputError::NumericOverflow);
    }
    if increment != 0.0 && result == current {
        return Err(AuditInputError::NumericPrecisionLoss);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> AircraftProfile {
        AircraftProfile {
            aircraft_id: "SYNTHETIC-100".to_owned(),
            configuration_revision: "cfg-test-001".to_owned(),
            source_reference: "fixture://approved-weight-and-balance-data".to_owned(),
            source_sha256: "a".repeat(64),
            approval_reference: "synthetic-approval-reference".to_owned(),
            empty_mass_kg: 1_000.0,
            empty_moment_kg_m: 2_500.0,
            maximum_takeoff_mass_kg: 2_000.0,
            stations: vec![
                LoadingStation {
                    station_id: "FWD".to_owned(),
                    arm_m: 1.0,
                    maximum_station_load_kg: 600.0,
                },
                LoadingStation {
                    station_id: "AFT".to_owned(),
                    arm_m: 3.0,
                    maximum_station_load_kg: 600.0,
                },
            ],
            cg_envelope: vec![CgEnvelopeBand {
                minimum_mass_kg: 1_000.0,
                maximum_mass_kg: 2_000.0,
                forward_cg_m: 2.0,
                aft_cg_m: 3.0,
            }],
        }
    }

    fn scenario(items: Vec<LoadedItem>) -> LoadingScenario {
        LoadingScenario {
            scenario_id: "scenario-test-001".to_owned(),
            phase: LoadingPhase::Takeoff,
            aircraft_id: "SYNTHETIC-100".to_owned(),
            configuration_revision: "cfg-test-001".to_owned(),
            items,
        }
    }

    fn cargo(item_id: &str, station_id: &str, mass_kg: f64) -> LoadedItem {
        LoadedItem {
            item_id: item_id.to_owned(),
            kind: LoadedItemKind::Cargo,
            station_id: station_id.to_owned(),
            mass_kg,
            securement_evidence_reference: Some(format!("fixture://securement/{item_id}")),
        }
    }

    fn approx_eq(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-9, "{left} != {right}");
    }

    #[test]
    fn computes_mass_moment_and_cg_known_answer() {
        let report = audit_loading(&profile(), &scenario(vec![cargo("box-1", "AFT", 100.0)]))
            .expect("valid synthetic fixture should audit");
        approx_eq(report.total_mass_kg, 1_100.0);
        approx_eq(report.total_moment_kg_m, 2_800.0);
        approx_eq(report.center_of_gravity_m, 2_800.0 / 1_100.0);
        assert_eq!(
            report.disposition,
            AuditDisposition::WithinDeclaredEnvelopeForHumanReview
        );
        assert!(!report.profile_source_authenticated);
    }

    #[test]
    fn blocks_mass_above_declared_takeoff_limit() {
        let mut aircraft = profile();
        aircraft.maximum_takeoff_mass_kg = 1_200.0;
        aircraft.stations[0].maximum_station_load_kg = 700.0;
        let report = audit_loading(&aircraft, &scenario(vec![cargo("box-1", "FWD", 300.0)]))
            .expect("valid scenario should produce a report");
        assert_eq!(report.disposition, AuditDisposition::BlockedByDeclaredLimit);
        assert!(
            report
                .findings
                .iter()
                .any(|finding| finding.code == FindingCode::MaximumTakeoffMassExceeded)
        );
    }

    #[test]
    fn blocks_station_load_above_declared_limit() {
        let aircraft = profile();
        let report = audit_loading(&aircraft, &scenario(vec![cargo("box-1", "AFT", 700.0)]))
            .expect("valid scenario should produce a report");
        assert!(
            report
                .findings
                .iter()
                .any(|finding| finding.code == FindingCode::StationLoadExceeded)
        );
        assert_eq!(report.disposition, AuditDisposition::BlockedByDeclaredLimit);
    }

    #[test]
    fn blocks_cg_outside_the_unique_mass_band() {
        let mut aircraft = profile();
        aircraft.stations[1].arm_m = 8.0;
        let report = audit_loading(&aircraft, &scenario(vec![cargo("box-1", "AFT", 500.0)]))
            .expect("valid scenario should produce a report");
        assert!(
            report
                .findings
                .iter()
                .any(|finding| { finding.code == FindingCode::CenterOfGravityOutsideEnvelope })
        );
        assert_eq!(report.disposition, AuditDisposition::BlockedByDeclaredLimit);
    }

    #[test]
    fn missing_securement_reference_is_not_assessable() {
        let mut item = cargo("box-1", "AFT", 100.0);
        item.securement_evidence_reference = None;
        let report = audit_loading(&profile(), &scenario(vec![item]))
            .expect("mass audit remains numerically possible");
        assert_eq!(report.disposition, AuditDisposition::NotAssessable);
        assert!(report.findings.iter().any(|finding| {
            finding.code == FindingCode::SecurementEvidenceMissing
                && finding.severity == FindingSeverity::Unknown
        }));
    }

    #[test]
    fn missing_cg_band_is_not_assessable() {
        let mut aircraft = profile();
        aircraft.cg_envelope[0].maximum_mass_kg = 1_200.0;
        let report = audit_loading(&aircraft, &scenario(vec![cargo("box-1", "FWD", 300.0)]))
            .expect("unknown envelope is a report-level result");
        assert_eq!(report.disposition, AuditDisposition::NotAssessable);
        assert!(
            report
                .findings
                .iter()
                .any(|finding| { finding.code == FindingCode::CenterOfGravityEnvelopeMissing })
        );
    }

    #[test]
    fn overlapping_cg_bands_are_not_assessed_by_arbitrary_selection() {
        let mut aircraft = profile();
        aircraft.cg_envelope.push(CgEnvelopeBand {
            minimum_mass_kg: 1_050.0,
            maximum_mass_kg: 1_500.0,
            forward_cg_m: 2.1,
            aft_cg_m: 2.9,
        });
        let report = audit_loading(&aircraft, &scenario(vec![cargo("box-1", "AFT", 100.0)]))
            .expect("ambiguous envelope is a report-level result");
        assert_eq!(report.disposition, AuditDisposition::NotAssessable);
        assert!(
            report
                .findings
                .iter()
                .any(|finding| { finding.code == FindingCode::CenterOfGravityEnvelopeAmbiguous })
        );
    }

    #[test]
    fn rejects_bad_digest_syntax() {
        let mut aircraft = profile();
        aircraft.source_sha256 = "not-a-digest".to_owned();
        assert!(matches!(
            audit_loading(&aircraft, &scenario(vec![])),
            Err(AuditInputError::InvalidProfile(_))
        ));
    }

    #[test]
    fn rejects_non_takeoff_phase_until_phase_specific_profiles_exist() {
        let mut manifest = scenario(vec![]);
        manifest.phase = LoadingPhase::Landing;
        assert_eq!(
            audit_loading(&profile(), &manifest),
            Err(AuditInputError::UnsupportedLoadingPhase(
                LoadingPhase::Landing
            ))
        );
    }

    #[test]
    fn rejects_profile_revision_mismatch() {
        let mut manifest = scenario(vec![]);
        manifest.configuration_revision = "different-revision".to_owned();
        assert_eq!(
            audit_loading(&profile(), &manifest),
            Err(AuditInputError::ProfileIdentityMismatch)
        );
    }

    #[test]
    fn rejects_duplicate_item_ids() {
        let items = vec![cargo("same-id", "FWD", 10.0), cargo("same-id", "AFT", 10.0)];
        assert_eq!(
            audit_loading(&profile(), &scenario(items)),
            Err(AuditInputError::DuplicateItemId("same-id".to_owned()))
        );
    }

    #[test]
    fn rejects_unknown_stations() {
        assert_eq!(
            audit_loading(&profile(), &scenario(vec![cargo("box-1", "MISSING", 10.0)])),
            Err(AuditInputError::UnknownStationId("MISSING".to_owned()))
        );
    }

    #[test]
    fn rejects_numeric_overflow_in_moment_calculation() {
        let mut aircraft = profile();
        aircraft.stations[0].arm_m = 8.0;
        aircraft.stations[0].maximum_station_load_kg = 1.0e308;
        aircraft.maximum_takeoff_mass_kg = 1.0e308;
        aircraft.cg_envelope[0].maximum_mass_kg = 1.0e308;
        assert_eq!(
            audit_loading(&aircraft, &scenario(vec![cargo("box-1", "FWD", 1.0e308)])),
            Err(AuditInputError::NumericOverflow)
        );
    }

    #[test]
    fn rejects_mass_contribution_lost_to_addition_rounding() {
        let mut aircraft = profile();
        aircraft.empty_mass_kg = 1.0e200;
        aircraft.empty_moment_kg_m = 0.0;
        aircraft.maximum_takeoff_mass_kg = f64::MAX;
        aircraft.stations[0].maximum_station_load_kg = 1.0e100;
        aircraft.cg_envelope[0].maximum_mass_kg = f64::MAX;
        assert_eq!(
            audit_loading(&aircraft, &scenario(vec![cargo("box-1", "FWD", 1.0)])),
            Err(AuditInputError::NumericPrecisionLoss)
        );
    }

    #[test]
    fn rejects_nonzero_mass_arm_product_lost_to_underflow() {
        let mut aircraft = profile();
        aircraft.empty_mass_kg = 1.0e-200;
        aircraft.empty_moment_kg_m = 0.0;
        aircraft.maximum_takeoff_mass_kg = 1.0;
        aircraft.stations[0].arm_m = 1.0e-200;
        aircraft.stations[0].maximum_station_load_kg = 1.0;
        aircraft.cg_envelope[0].minimum_mass_kg = 1.0e-200;
        aircraft.cg_envelope[0].maximum_mass_kg = 1.0;
        let item = cargo("tiny-box", "FWD", 1.0e-200);
        assert_eq!(
            audit_loading(&aircraft, &scenario(vec![item])),
            Err(AuditInputError::NumericPrecisionLoss)
        );
    }

    #[test]
    fn item_order_does_not_change_report() {
        let first = cargo("a", "FWD", 50.0);
        let second = cargo("b", "AFT", 75.0);
        let report_a = audit_loading(&profile(), &scenario(vec![first.clone(), second.clone()]))
            .expect("valid order");
        let report_b =
            audit_loading(&profile(), &scenario(vec![second, first])).expect("reordered items");
        assert_eq!(report_a, report_b);
    }
}
