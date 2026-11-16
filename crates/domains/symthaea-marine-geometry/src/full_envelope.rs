//! G2f-a typed contract and semantic validation for full external hull envelopes.
//!
//! This is deliberately separate from the G2a/G2b keel-to-waterline
//! displacement geometry. It validates input semantics; it does not generate
//! a hull surface, B-rep, displacement shell, or engineering approval.

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeSet;

pub const FULL_ENVELOPE_SCHEMA_VERSION: u32 = 1;
pub const MAX_FULL_ENVELOPE_STATIONS: usize = 512;
const COORDINATE_TOLERANCE_M: f64 = 1.0e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SymmetryPolicy {
    SymmetricAboutY0,
    ExplicitPortAndStarboard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HullEndCondition {
    PointedStem,
    Transom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WaterlinePortSource {
    MirroredFromStarboard,
    ExplicitPortProfile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ProfilePointRole {
    Keel,
    Bilge,
    Chine,
    Bottom,
    Waterline,
    Topside,
    Sheer,
    DeckEdge,
    Stem,
    Transom,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ProfileRegion {
    Bottom,
    Topsides,
    DeckEdgeTransition,
    StemClosure,
    TransomClosure,
    Unsupported,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DeckPointRole {
    PortDeckEdge,
    DeckCamber,
    DeckCrown,
    StarboardDeckEdge,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DeckConstruction {
    LoftedDeckSections,
    PlanarDeck,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SurfaceRegion {
    Bottom,
    Topsides,
    Deck,
    BowEndClosure,
    SternEndClosure,
    ChineBreaklines,
    KeelBreakline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExteriorShellSemantics {
    ClosedShell,
    ExplicitBoundaryLoops,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OpeningStatus {
    Supported,
    UnsupportedRequiresReview,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExternalComponentType {
    Rudder,
    Propeller,
    Shaft,
    BilgeKeel,
    BowThruster,
    Skeg,
    Superstructure,
    Bulwark,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ComponentIntegrationSemantics {
    SeparateAssembly,
    AttachedButSeparatelyMeshed,
    NotIncluded,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FullEnvelopeCoordinateFrame {
    pub id: String,
    pub axis_convention: String,
    pub handedness: String,
    pub origin_description: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrincipalDimensions {
    pub length_m: f64,
    pub maximum_beam_m: f64,
    pub design_draft_m: f64,
    pub design_freeboard_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndConditions {
    pub bow: HullEndCondition,
    pub stern: HullEndCondition,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignWaterline {
    pub waterline_id: String,
    pub plane_z_m: f64,
    pub station_intersections: Vec<WaterlineStationIntersection>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaterlineStationIntersection {
    pub station_id: String,
    pub starboard_point_id: String,
    pub port_source: WaterlinePortSource,
    #[serde(deserialize_with = "deserialize_present_nullable_string")]
    pub port_point_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FullEnvelopeStation {
    pub station_id: String,
    pub x_over_length: f64,
    pub starboard_profile: Vec<ProfilePoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port_profile: Option<Vec<ProfilePoint>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfilePoint {
    pub point_id: String,
    pub y_m: f64,
    pub z_m: f64,
    pub role: ProfilePointRole,
    pub region_to_next: ProfileRegion,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeckSurface {
    pub surface_id: String,
    pub construction: DeckConstruction,
    pub sections: Vec<DeckSection>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeckSection {
    pub station_id: String,
    pub points: Vec<DeckPoint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeckPoint {
    pub point_id: String,
    pub y_m: f64,
    pub z_m: f64,
    pub role: DeckPointRole,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundarySemantics {
    pub exterior_shell: ExteriorShellSemantics,
    pub openings: Vec<OpeningBoundary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpeningBoundary {
    pub opening_id: String,
    pub boundary_point_refs: Vec<String>,
    pub status: OpeningStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalComponent {
    pub component_id: String,
    pub component_type: ExternalComponentType,
    pub source_digest_blake3: String,
    pub integration_semantics: ComponentIntegrationSemantics,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FullEnvelopeSpec {
    pub schema_version: u32,
    pub candidate_id: String,
    pub units: String,
    pub coordinate_frame: FullEnvelopeCoordinateFrame,
    pub principal_dimensions: PrincipalDimensions,
    pub symmetry: SymmetryPolicy,
    pub end_conditions: EndConditions,
    pub design_waterline: DesignWaterline,
    pub stations: Vec<FullEnvelopeStation>,
    pub deck_surface: DeckSurface,
    pub surface_inventory: Vec<SurfaceRegion>,
    pub boundary_semantics: BoundarySemantics,
    #[serde(default)]
    pub external_components: Vec<ExternalComponent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FullEnvelopeIssueLevel {
    Rejected,
    Incomplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FullEnvelopeDisposition {
    ValidInputContract,
    Rejected,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullEnvelopeIssue {
    pub code: String,
    pub level: FullEnvelopeIssueLevel,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullEnvelopeValidationReport {
    pub report_schema_version: u32,
    pub candidate_id: String,
    pub source_spec_digest_blake3: Option<String>,
    pub disposition: FullEnvelopeDisposition,
    pub issues: Vec<FullEnvelopeIssue>,
}

fn issue(
    issues: &mut Vec<FullEnvelopeIssue>,
    level: FullEnvelopeIssueLevel,
    code: &str,
    detail: impl Into<String>,
) {
    issues.push(FullEnvelopeIssue {
        code: code.to_owned(),
        level,
        detail: detail.into(),
    });
}

fn reject(issues: &mut Vec<FullEnvelopeIssue>, code: &str, detail: impl Into<String>) {
    issue(issues, FullEnvelopeIssueLevel::Rejected, code, detail);
}

fn incomplete(issues: &mut Vec<FullEnvelopeIssue>, code: &str, detail: impl Into<String>) {
    issue(issues, FullEnvelopeIssueLevel::Incomplete, code, detail);
}

fn finite(value: f64) -> bool {
    value.is_finite()
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= COORDINATE_TOLERANCE_M
}

fn deserialize_present_nullable_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer)
}

fn digest_is_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn profile_point_by_id<'a>(
    profile: &'a [ProfilePoint],
    point_id: &str,
) -> Option<&'a ProfilePoint> {
    profile.iter().find(|point| point.point_id == point_id)
}

fn validate_profile(
    station_id: &str,
    profile: &[ProfilePoint],
    starboard: bool,
    dimensions: &PrincipalDimensions,
    issues: &mut Vec<FullEnvelopeIssue>,
) {
    if profile.len() < 3 || profile.len() > 128 {
        reject(
            issues,
            "profile_point_count",
            format!("station {station_id} profile must contain 3..=128 points"),
        );
        return;
    }

    let mut ids = BTreeSet::new();
    for point in profile {
        if point.point_id.trim().is_empty() || !ids.insert(point.point_id.as_str()) {
            reject(
                issues,
                "profile_point_identity",
                format!("station {station_id} has an empty or duplicate profile point ID"),
            );
        }
        if !finite(point.y_m) || !finite(point.z_m) {
            reject(
                issues,
                "profile_non_finite_coordinate",
                format!("station {station_id} contains a non-finite profile coordinate"),
            );
        }
        if starboard && point.y_m < -COORDINATE_TOLERANCE_M {
            reject(
                issues,
                "starboard_sign",
                format!("station {station_id} starboard profile contains a negative y coordinate"),
            );
        }
        if !starboard && point.y_m > COORDINATE_TOLERANCE_M {
            reject(
                issues,
                "port_sign",
                format!("station {station_id} port profile contains a positive y coordinate"),
            );
        }
        if point.y_m.abs() > 0.5 * dimensions.maximum_beam_m + COORDINATE_TOLERANCE_M {
            reject(
                issues,
                "profile_beam_envelope",
                format!("station {station_id} exceeds the declared maximum half-beam"),
            );
        }
        if point.z_m > dimensions.design_draft_m + COORDINATE_TOLERANCE_M {
            reject(
                issues,
                "profile_draft_envelope",
                format!("station {station_id} extends below the declared design-draft envelope"),
            );
        }
        if point.region_to_next == ProfileRegion::Unsupported {
            incomplete(
                issues,
                "unsupported_profile_region",
                format!("station {station_id} contains a profile segment declared unsupported"),
            );
        }
    }

    let first = &profile[0];
    let last = &profile[profile.len() - 1];
    if first.role != ProfilePointRole::Keel || !close(first.y_m, 0.0) || first.z_m <= 0.0 {
        reject(
            issues,
            "keel_profile_endpoint",
            format!("station {station_id} profile must start at a below-waterline keel on y=0"),
        );
    }
    if !matches!(
        last.role,
        ProfilePointRole::Sheer | ProfilePointRole::DeckEdge
    ) {
        reject(
            issues,
            "sheer_profile_endpoint",
            format!("station {station_id} profile must end at a sheer/deck-edge point"),
        );
    }

    let keel_count = profile
        .iter()
        .filter(|point| point.role == ProfilePointRole::Keel)
        .count();
    let waterline_indices = profile
        .iter()
        .enumerate()
        .filter(|(_, point)| point.role == ProfilePointRole::Waterline)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let sheer_count = profile
        .iter()
        .filter(|point| {
            matches!(
                point.role,
                ProfilePointRole::Sheer | ProfilePointRole::DeckEdge
            )
        })
        .count();
    let bottom_count = profile
        .iter()
        .filter(|point| {
            matches!(
                point.role,
                ProfilePointRole::Bottom | ProfilePointRole::Bilge | ProfilePointRole::Chine
            )
        })
        .count();
    if keel_count != 1 || waterline_indices.len() != 1 || sheer_count != 1 || bottom_count == 0 {
        reject(
            issues,
            "profile_semantic_markers",
            format!(
                "station {station_id} requires one keel, one waterline, one sheer and at least one bottom/bilge/chine marker"
            ),
        );
    }
    if let Some(waterline_index) = waterline_indices.first().copied() {
        let waterline = &profile[waterline_index];
        if waterline_index == 0
            || waterline_index + 1 >= profile.len()
            || !close(waterline.z_m, 0.0)
        {
            reject(
                issues,
                "profile_waterline_position",
                format!("station {station_id} waterline must lie between keel and sheer on z=0"),
            );
        }
    }

    for pair in profile.windows(2) {
        let y_progress = if starboard {
            pair[1].y_m + COORDINATE_TOLERANCE_M >= pair[0].y_m
        } else {
            pair[1].y_m - COORDINATE_TOLERANCE_M <= pair[0].y_m
        };
        if !y_progress {
            reject(
                issues,
                "profile_transverse_order",
                format!(
                    "station {station_id} profile is not monotonically ordered from keel to sheer"
                ),
            );
        }
        if pair[1].z_m > pair[0].z_m + COORDINATE_TOLERANCE_M {
            reject(
                issues,
                "profile_vertical_order",
                format!("station {station_id} profile reverses vertical order"),
            );
        }
    }
}

fn validate_unique_station_ids(
    stations: &[FullEnvelopeStation],
    issues: &mut Vec<FullEnvelopeIssue>,
) {
    let mut ids = BTreeSet::new();
    for station in stations {
        if station.station_id.trim().is_empty() || !ids.insert(station.station_id.as_str()) {
            reject(
                issues,
                "station_identity",
                "station IDs must be non-empty and unique",
            );
        }
    }
}

fn validate_symmetry_and_port_profiles(
    spec: &FullEnvelopeSpec,
    issues: &mut Vec<FullEnvelopeIssue>,
) {
    for (station_index, station) in spec.stations.iter().enumerate() {
        match spec.symmetry {
            SymmetryPolicy::SymmetricAboutY0 => {
                if station.port_profile.is_some() {
                    reject(
                        issues,
                        "symmetric_port_profile_supplied",
                        format!(
                            "station {} supplies explicit port geometry despite symmetric-about-y0 policy",
                            station.station_id
                        ),
                    );
                }
            }
            SymmetryPolicy::ExplicitPortAndStarboard => {
                let Some(port) = station.port_profile.as_ref() else {
                    reject(
                        issues,
                        "explicit_port_profile_missing",
                        format!(
                            "station {} requires an explicit port profile",
                            station.station_id
                        ),
                    );
                    continue;
                };
                validate_profile(
                    &station.station_id,
                    port,
                    false,
                    &spec.principal_dimensions,
                    issues,
                );
                if port.len() != station.starboard_profile.len() {
                    reject(
                        issues,
                        "port_starboard_point_count",
                        format!(
                            "station {} port/starboard profile point counts differ",
                            station.station_id
                        ),
                    );
                    continue;
                }
                let pointed_end = (station_index == 0
                    && spec.end_conditions.bow == HullEndCondition::PointedStem)
                    || (station_index + 1 == spec.stations.len()
                        && spec.end_conditions.stern == HullEndCondition::PointedStem);
                for (port_point, starboard_point) in
                    port.iter().zip(station.starboard_profile.iter())
                {
                    let semantic_mismatch = port_point.role != starboard_point.role
                        || port_point.region_to_next != starboard_point.region_to_next;
                    let identity_or_geometry_mismatch = if pointed_end {
                        port_point.point_id != starboard_point.point_id
                            || !close(port_point.y_m, 0.0)
                            || !close(starboard_point.y_m, 0.0)
                            || !close(port_point.z_m, starboard_point.z_m)
                    } else {
                        port_point.point_id == starboard_point.point_id
                    };
                    if semantic_mismatch || identity_or_geometry_mismatch {
                        reject(
                            issues,
                            "port_starboard_correspondence",
                            if pointed_end {
                                format!(
                                    "station {} pointed stem requires shared port/starboard point identity and coincident centreplane coordinates for {:?}",
                                    station.station_id, starboard_point.role
                                )
                            } else {
                                format!(
                                    "station {} explicit port point {} must have a distinct identity and matching semantic role/breakline with starboard point {}",
                                    station.station_id, port_point.point_id, starboard_point.point_id
                                )
                            },
                        );
                    }
                }
            }
        }
    }
}

fn validate_end_condition(
    station: &FullEnvelopeStation,
    condition: HullEndCondition,
    end_name: &str,
    issues: &mut Vec<FullEnvelopeIssue>,
) {
    let Some(waterline) = station
        .starboard_profile
        .iter()
        .find(|point| point.role == ProfilePointRole::Waterline)
    else {
        return;
    };
    match condition {
        HullEndCondition::PointedStem => {
            if station
                .starboard_profile
                .iter()
                .any(|point| !close(point.y_m, 0.0))
            {
                reject(
                    issues,
                    "pointed_end_not_collapsed",
                    format!("{end_name} pointed-stem profile must collapse exactly to y=0"),
                );
            }
            if let Some(port) = station.port_profile.as_ref()
                && port.iter().any(|point| !close(point.y_m, 0.0))
            {
                reject(
                    issues,
                    "pointed_port_end_not_collapsed",
                    format!("{end_name} pointed-stem port profile must collapse exactly to y=0"),
                );
            }
        }
        HullEndCondition::Transom => {
            if waterline.y_m <= COORDINATE_TOLERANCE_M {
                reject(
                    issues,
                    "transom_waterline_breadth",
                    format!("{end_name} transom must have positive starboard waterline breadth"),
                );
            }
        }
    }
}

fn validate_waterline(spec: &FullEnvelopeSpec, issues: &mut Vec<FullEnvelopeIssue>) {
    let waterline = &spec.design_waterline;
    if waterline.waterline_id.trim().is_empty() {
        reject(
            issues,
            "waterline_identity",
            "design waterline ID must not be empty",
        );
    }
    if !finite(waterline.plane_z_m) || !close(waterline.plane_z_m, 0.0) {
        reject(
            issues,
            "waterline_plane",
            "G2f-a v1 requires the named design waterline to be the z=0 plane",
        );
    }
    if waterline.station_intersections.len() != spec.stations.len() {
        reject(
            issues,
            "waterline_station_coverage",
            "design waterline must reference exactly one intersection for every station",
        );
        return;
    }
    for (station, intersection) in spec
        .stations
        .iter()
        .zip(waterline.station_intersections.iter())
    {
        if intersection.station_id != station.station_id {
            reject(
                issues,
                "waterline_station_order",
                format!(
                    "waterline intersection does not match station {}",
                    station.station_id
                ),
            );
            continue;
        }
        let Some(starboard_point) =
            profile_point_by_id(&station.starboard_profile, &intersection.starboard_point_id)
        else {
            reject(
                issues,
                "waterline_starboard_reference",
                format!(
                    "station {} waterline starboard point reference does not resolve",
                    station.station_id
                ),
            );
            continue;
        };
        if starboard_point.role != ProfilePointRole::Waterline || !close(starboard_point.z_m, 0.0) {
            reject(
                issues,
                "waterline_starboard_semantics",
                format!(
                    "station {} reference is not its z=0 waterline point",
                    station.station_id
                ),
            );
        }

        match (spec.symmetry, intersection.port_source) {
            (SymmetryPolicy::SymmetricAboutY0, WaterlinePortSource::MirroredFromStarboard) => {
                if intersection.port_point_id.is_some() {
                    reject(
                        issues,
                        "mirrored_waterline_port_reference",
                        format!(
                            "station {} mirrored port intersection must not name a source port point",
                            station.station_id
                        ),
                    );
                }
            }
            (
                SymmetryPolicy::ExplicitPortAndStarboard,
                WaterlinePortSource::ExplicitPortProfile,
            ) => {
                let Some(port_id) = intersection.port_point_id.as_deref() else {
                    reject(
                        issues,
                        "explicit_waterline_port_reference",
                        format!(
                            "station {} requires an explicit port waterline point ID",
                            station.station_id
                        ),
                    );
                    continue;
                };
                let Some(port_profile) = station.port_profile.as_ref() else {
                    continue;
                };
                let Some(port_point) = profile_point_by_id(port_profile, port_id) else {
                    reject(
                        issues,
                        "waterline_port_reference",
                        format!(
                            "station {} waterline port point reference does not resolve",
                            station.station_id
                        ),
                    );
                    continue;
                };
                let pointed_end = (spec
                    .stations
                    .first()
                    .is_some_and(|first| station.station_id == first.station_id)
                    && spec.end_conditions.bow == HullEndCondition::PointedStem)
                    || (spec
                        .stations
                        .last()
                        .is_some_and(|last| station.station_id == last.station_id)
                        && spec.end_conditions.stern == HullEndCondition::PointedStem);
                let port_coordinate_invalid = if pointed_end {
                    !close(port_point.y_m, 0.0)
                        || port_point.point_id != starboard_point.point_id
                } else {
                    port_point.y_m >= -COORDINATE_TOLERANCE_M
                };
                if port_point.role != ProfilePointRole::Waterline
                    || !close(port_point.z_m, 0.0)
                    || port_coordinate_invalid
                {
                    reject(
                        issues,
                        "waterline_port_semantics",
                        format!(
                            "station {} explicit port reference must satisfy its pointed/transom waterline identity and coordinate semantics",
                            station.station_id
                        ),
                    );
                }
            }
            _ => reject(
                issues,
                "waterline_port_source_policy",
                format!(
                    "station {} port source conflicts with declared symmetry policy",
                    station.station_id
                ),
            ),
        }
    }
}

fn validate_deck(spec: &FullEnvelopeSpec, issues: &mut Vec<FullEnvelopeIssue>) {
    let deck = &spec.deck_surface;
    if deck.surface_id.trim().is_empty() {
        reject(
            issues,
            "deck_surface_identity",
            "deck surface ID must not be empty",
        );
    }
    if deck.construction == DeckConstruction::Unsupported {
        incomplete(
            issues,
            "unsupported_deck_construction",
            "deck construction is explicitly declared unsupported",
        );
    }
    if deck.sections.len() != spec.stations.len() {
        reject(
            issues,
            "deck_station_coverage",
            "deck must provide exactly one section for every hull station",
        );
        return;
    }
    for (station, deck_section) in spec.stations.iter().zip(deck.sections.iter()) {
        if deck_section.station_id != station.station_id {
            reject(
                issues,
                "deck_station_order",
                format!(
                    "deck section does not match hull station {}",
                    station.station_id
                ),
            );
            continue;
        }
        let is_pointed_bow = spec
            .stations
            .first()
            .is_some_and(|first| station.station_id == first.station_id)
            && spec.end_conditions.bow == HullEndCondition::PointedStem;
        let is_pointed_stern = spec
            .stations
            .last()
            .is_some_and(|last| station.station_id == last.station_id)
            && spec.end_conditions.stern == HullEndCondition::PointedStem;
        let is_pointed_end = is_pointed_bow || is_pointed_stern;
        if deck_section.points.len() > 128
            || (is_pointed_end && deck_section.points.len() != 1)
            || (!is_pointed_end && deck_section.points.len() < 2)
        {
            reject(
                issues,
                "deck_point_count",
                format!(
                    "deck section {} must contain one collapsed point at a pointed end or 2..=128 points otherwise",
                    station.station_id
                ),
            );
            continue;
        }
        if is_pointed_end {
            let point = &deck_section.points[0];
            let starboard_sheer = station.starboard_profile.last();
            if point.role != DeckPointRole::DeckCrown
                || !close(point.y_m, 0.0)
                || point.z_m >= -COORDINATE_TOLERANCE_M
                || starboard_sheer.is_none_or(|sheer| {
                    point.point_id != sheer.point_id
                        || !close(point.y_m, sheer.y_m)
                        || !close(point.z_m, sheer.z_m)
                })
            {
                reject(
                    issues,
                    "pointed_deck_endpoint",
                    format!(
                        "pointed-end deck section {} must share the exact above-waterline centreplane sheer/crown point identity and coordinates",
                        station.station_id
                    ),
                );
            }
            continue;
        }
        let mut ids = BTreeSet::new();
        for point in &deck_section.points {
            if point.point_id.trim().is_empty() || !ids.insert(point.point_id.as_str()) {
                reject(
                    issues,
                    "deck_point_identity",
                    format!(
                        "deck section {} contains an empty or duplicate point ID",
                        station.station_id
                    ),
                );
            }
            if !finite(point.y_m) || !finite(point.z_m) || point.z_m >= -COORDINATE_TOLERANCE_M {
                reject(
                    issues,
                    "deck_coordinate",
                    format!(
                        "deck section {} must have finite points above the z=0 waterline",
                        station.station_id
                    ),
                );
            }
        }
        for pair in deck_section.points.windows(2) {
            if pair[1].y_m <= pair[0].y_m + COORDINATE_TOLERANCE_M {
                reject(
                    issues,
                    "deck_transverse_order",
                    format!(
                        "deck section {} points must be strictly ordered from port to starboard",
                        station.station_id
                    ),
                );
            }
        }
        if deck_section.points.first().map(|point| point.role) != Some(DeckPointRole::PortDeckEdge)
            || deck_section.points.last().map(|point| point.role)
                != Some(DeckPointRole::StarboardDeckEdge)
        {
            reject(
                issues,
                "deck_edge_markers",
                format!(
                    "deck section {} must start/end at port/starboard deck edges",
                    station.station_id
                ),
            );
        }

        let Some(starboard_sheer) = station.starboard_profile.last() else {
            continue;
        };
        let deck_starboard = deck_section
            .points
            .last()
            .expect("deck point count checked");
        if deck_starboard.point_id != starboard_sheer.point_id
            || !close(deck_starboard.y_m, starboard_sheer.y_m)
            || !close(deck_starboard.z_m, starboard_sheer.z_m)
        {
            reject(
                issues,
                "deck_starboard_seam",
                format!(
                    "deck section {} starboard deck edge does not resolve to its sheer point",
                    station.station_id
                ),
            );
        }

        let deck_port = deck_section
            .points
            .first()
            .expect("deck point count checked");
        match spec.symmetry {
            SymmetryPolicy::SymmetricAboutY0 => {
                let expected_id = format!("MIRROR:{}", starboard_sheer.point_id);
                if deck_port.point_id != expected_id
                    || !close(deck_port.y_m, -starboard_sheer.y_m)
                    || !close(deck_port.z_m, starboard_sheer.z_m)
                {
                    reject(
                        issues,
                        "deck_port_mirror_seam",
                        format!(
                            "deck section {} port deck edge must be the exact mirrored sheer identity",
                            station.station_id
                        ),
                    );
                }
            }
            SymmetryPolicy::ExplicitPortAndStarboard => {
                let Some(port_profile) = station.port_profile.as_ref() else {
                    continue;
                };
                let Some(port_sheer) = port_profile.last() else {
                    continue;
                };
                if deck_port.point_id != port_sheer.point_id
                    || !close(deck_port.y_m, port_sheer.y_m)
                    || !close(deck_port.z_m, port_sheer.z_m)
                {
                    reject(
                        issues,
                        "deck_port_seam",
                        format!(
                            "deck section {} port deck edge does not resolve to its port sheer point",
                            station.station_id
                        ),
                    );
                }
            }
        }
    }
}

fn validate_surface_inventory(spec: &FullEnvelopeSpec, issues: &mut Vec<FullEnvelopeIssue>) {
    let mut seen = BTreeSet::new();
    for region in &spec.surface_inventory {
        if !seen.insert(*region) {
            reject(
                issues,
                "duplicate_surface_region",
                "surface inventory contains a duplicate region",
            );
        }
    }
    for required in [
        SurfaceRegion::Bottom,
        SurfaceRegion::Topsides,
        SurfaceRegion::Deck,
        SurfaceRegion::BowEndClosure,
        SurfaceRegion::SternEndClosure,
    ] {
        if !seen.contains(&required) {
            reject(
                issues,
                "missing_surface_region",
                format!("surface inventory is missing required region {required:?}"),
            );
        }
    }
}

fn validate_boundary_and_components(spec: &FullEnvelopeSpec, issues: &mut Vec<FullEnvelopeIssue>) {
    let boundary = &spec.boundary_semantics;
    match boundary.exterior_shell {
        ExteriorShellSemantics::ClosedShell if !boundary.openings.is_empty() => reject(
            issues,
            "closed_shell_has_openings",
            "a closed-shell declaration cannot include explicit openings",
        ),
        ExteriorShellSemantics::ExplicitBoundaryLoops if boundary.openings.is_empty() => reject(
            issues,
            "boundary_loops_missing",
            "explicit-boundary-loops semantics require at least one declared opening/boundary loop",
        ),
        _ => {}
    }

    let mut resolvable_point_ids = BTreeSet::<String>::new();
    for station in &spec.stations {
        for point in &station.starboard_profile {
            resolvable_point_ids.insert(point.point_id.clone());
            if spec.symmetry == SymmetryPolicy::SymmetricAboutY0 {
                resolvable_point_ids.insert(format!("MIRROR:{}", point.point_id));
            }
        }
        if let Some(port_profile) = station.port_profile.as_ref() {
            for point in port_profile {
                resolvable_point_ids.insert(point.point_id.clone());
            }
        }
    }
    for deck_section in &spec.deck_surface.sections {
        for point in &deck_section.points {
            resolvable_point_ids.insert(point.point_id.clone());
        }
    }

    let mut opening_ids = BTreeSet::new();
    for opening in &boundary.openings {
        if opening.opening_id.trim().is_empty() || !opening_ids.insert(opening.opening_id.as_str())
        {
            reject(
                issues,
                "opening_identity",
                "opening IDs must be non-empty and unique",
            );
        }
        if opening.boundary_point_refs.len() < 3 {
            reject(
                issues,
                "opening_boundary_size",
                format!(
                    "opening {} requires at least three boundary point references",
                    opening.opening_id
                ),
            );
        }
        let mut refs = BTreeSet::new();
        if opening
            .boundary_point_refs
            .iter()
            .any(|point_id| point_id.trim().is_empty() || !refs.insert(point_id.as_str()))
        {
            reject(
                issues,
                "opening_boundary_references",
                format!(
                    "opening {} has empty or duplicate point references",
                    opening.opening_id
                ),
            );
        }
        for point_id in &opening.boundary_point_refs {
            if !resolvable_point_ids.contains(point_id) {
                reject(
                    issues,
                    "opening_boundary_point_reference",
                    format!(
                        "opening {} references unknown point {}",
                        opening.opening_id, point_id
                    ),
                );
            }
        }
        if opening.status == OpeningStatus::UnsupportedRequiresReview {
            incomplete(
                issues,
                "unsupported_opening",
                format!(
                    "opening {} requires a geometry/topology implementation not supported by G2f-a",
                    opening.opening_id
                ),
            );
        }
    }

    let mut component_ids = BTreeSet::new();
    for component in &spec.external_components {
        if component.component_id.trim().is_empty()
            || !component_ids.insert(component.component_id.as_str())
        {
            reject(
                issues,
                "external_component_identity",
                "external component IDs must be non-empty and unique",
            );
        }
        if !digest_is_hex(&component.source_digest_blake3) {
            reject(
                issues,
                "external_component_digest",
                format!(
                    "component {} source digest must be 64 hexadecimal characters",
                    component.component_id
                ),
            );
        }
    }
}

pub fn compute_full_envelope_spec_digest(
    spec: &FullEnvelopeSpec,
) -> Result<String, serde_json::Error> {
    let canonical = serde_json::to_vec(spec)?;
    Ok(blake3::hash(&canonical).to_hex().to_string())
}

/// Validate the typed G2f-a v1 full-envelope input contract.
///
/// A ValidInputContract result is a statement about schema-level and
/// cross-field input semantics only. It is not a generated surface, B-rep,
/// hydrostatic result, structural assessment, or vessel approval.
pub fn validate_full_envelope_spec(spec: &FullEnvelopeSpec) -> FullEnvelopeValidationReport {
    let mut issues = Vec::new();

    if spec.schema_version != FULL_ENVELOPE_SCHEMA_VERSION {
        reject(
            &mut issues,
            "schema_version",
            format!(
                "unsupported full-envelope schema {}; expected {}",
                spec.schema_version, FULL_ENVELOPE_SCHEMA_VERSION
            ),
        );
    }
    if spec.candidate_id.trim().is_empty() {
        reject(
            &mut issues,
            "candidate_id",
            "candidate ID must not be empty",
        );
    }
    if spec.units != "m" {
        reject(
            &mut issues,
            "units",
            "G2f-a v1 accepts metres only; no implicit conversion is performed",
        );
    }
    let frame = &spec.coordinate_frame;
    if frame.id.trim().is_empty()
        || frame.axis_convention != "x-forward,y-starboard,z-down"
        || frame.handedness != "right-handed"
        || frame.origin_description.trim().len() < 4
    {
        reject(
            &mut issues,
            "coordinate_frame",
            "requires explicit right-handed x-forward/y-starboard/z-down frame and origin description",
        );
    }

    let dimensions = &spec.principal_dimensions;
    if !finite(dimensions.length_m)
        || !finite(dimensions.maximum_beam_m)
        || !finite(dimensions.design_draft_m)
        || !finite(dimensions.design_freeboard_m)
        || dimensions.length_m <= 0.0
        || dimensions.maximum_beam_m <= 0.0
        || dimensions.design_draft_m <= 0.0
        || dimensions.design_freeboard_m <= 0.0
    {
        reject(
            &mut issues,
            "principal_dimensions",
            "principal dimensions must be finite and positive",
        );
    }

    validate_unique_station_ids(&spec.stations, &mut issues);
    if spec.stations.len() < 2 || spec.stations.len() > MAX_FULL_ENVELOPE_STATIONS {
        reject(
            &mut issues,
            "station_count",
            format!("full-envelope station count must be within 2..={MAX_FULL_ENVELOPE_STATIONS}"),
        );
    } else {
        if !finite(spec.stations[0].x_over_length) || !close(spec.stations[0].x_over_length, 0.0) {
            reject(
                &mut issues,
                "first_station_extent",
                "first station must be at x/L=0 (bow)",
            );
        }
        let last = spec.stations.last().expect("station count checked");
        if !finite(last.x_over_length) || !close(last.x_over_length, 1.0) {
            reject(
                &mut issues,
                "last_station_extent",
                "last station must be at x/L=1 (stern)",
            );
        }
        for (index, station) in spec.stations.iter().enumerate() {
            if !finite(station.x_over_length) || !(0.0..=1.0).contains(&station.x_over_length) {
                reject(
                    &mut issues,
                    "station_coordinate",
                    format!(
                        "station {} x/L must be finite and within [0,1]",
                        station.station_id
                    ),
                );
            }
            if index > 0 && station.x_over_length <= spec.stations[index - 1].x_over_length {
                reject(
                    &mut issues,
                    "station_order",
                    format!(
                        "station {} is not strictly after its predecessor",
                        station.station_id
                    ),
                );
            }
            validate_profile(
                &station.station_id,
                &station.starboard_profile,
                true,
                dimensions,
                &mut issues,
            );
            if station.starboard_profile.last().is_some_and(|point| {
                -point.z_m + COORDINATE_TOLERANCE_M < dimensions.design_freeboard_m
            }) {
                reject(
                    &mut issues,
                    "design_freeboard",
                    format!(
                        "station {} sheer is below the declared design freeboard",
                        station.station_id
                    ),
                );
            }
        }
        validate_end_condition(
            &spec.stations[0],
            spec.end_conditions.bow,
            "bow",
            &mut issues,
        );
        validate_end_condition(
            spec.stations.last().expect("station count checked"),
            spec.end_conditions.stern,
            "stern",
            &mut issues,
        );
    }

    validate_symmetry_and_port_profiles(spec, &mut issues);
    validate_waterline(spec, &mut issues);
    validate_deck(spec, &mut issues);
    validate_surface_inventory(spec, &mut issues);
    validate_boundary_and_components(spec, &mut issues);

    let source_digest = compute_full_envelope_spec_digest(spec).ok();
    if source_digest.is_none() {
        reject(
            &mut issues,
            "source_spec_digest",
            "full-envelope specification cannot be serialized canonically",
        );
    }

    let disposition = if issues
        .iter()
        .any(|item| item.level == FullEnvelopeIssueLevel::Rejected)
    {
        FullEnvelopeDisposition::Rejected
    } else if issues
        .iter()
        .any(|item| item.level == FullEnvelopeIssueLevel::Incomplete)
    {
        FullEnvelopeDisposition::Incomplete
    } else {
        FullEnvelopeDisposition::ValidInputContract
    };

    FullEnvelopeValidationReport {
        report_schema_version: 1,
        candidate_id: spec.candidate_id.clone(),
        source_spec_digest_blake3: source_digest,
        disposition,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_fixture() -> FullEnvelopeSpec {
        serde_json::from_str(include_str!(
            "../../../../docs/research/marine/fixtures/marine-full-envelope-box-001.json"
        ))
        .unwrap()
    }

    fn assert_rejected(spec: &FullEnvelopeSpec, code: &str) {
        let report = validate_full_envelope_spec(spec);
        assert_eq!(report.disposition, FullEnvelopeDisposition::Rejected);
        assert!(report.issues.iter().any(|item| item.code == code));
    }

    #[test]
    fn synthetic_box_fixture_satisfies_full_envelope_input_contract() {
        let spec = box_fixture();
        let report = validate_full_envelope_spec(&spec);
        assert_eq!(
            report.disposition,
            FullEnvelopeDisposition::ValidInputContract
        );
        assert!(report.issues.is_empty());
        assert_eq!(
            report.source_spec_digest_blake3.as_deref().unwrap().len(),
            64
        );
    }

    #[test]
    fn source_digest_binds_semantic_metadata_and_geometry() {
        let mut spec = box_fixture();
        let original = compute_full_envelope_spec_digest(&spec).unwrap();
        spec.end_conditions.bow = HullEndCondition::PointedStem;
        let changed = compute_full_envelope_spec_digest(&spec).unwrap();
        assert_ne!(original, changed);
    }

    #[test]
    fn unsupported_deck_construction_is_incomplete_not_a_pass() {
        let mut spec = box_fixture();
        spec.deck_surface.construction = DeckConstruction::Unsupported;
        let report = validate_full_envelope_spec(&spec);
        assert_eq!(report.disposition, FullEnvelopeDisposition::Incomplete);
        assert!(
            report
                .issues
                .iter()
                .any(|item| item.code == "unsupported_deck_construction")
        );
    }

    #[test]
    fn wrong_station_order_is_rejected() {
        let mut spec = box_fixture();
        spec.stations.swap(0, 1);
        assert_rejected(&spec, "station_order");
    }

    #[test]
    fn duplicate_station_identity_is_rejected() {
        let mut spec = box_fixture();
        spec.stations[1].station_id = spec.stations[0].station_id.clone();
        assert_rejected(&spec, "station_identity");
    }

    #[test]
    fn station_must_have_exactly_one_waterline_reference() {
        let mut spec = box_fixture();
        spec.design_waterline.station_intersections.pop();
        assert_rejected(&spec, "waterline_station_coverage");
    }

    #[test]
    fn waterline_reference_must_resolve_to_a_waterline_point() {
        let mut spec = box_fixture();
        spec.design_waterline.station_intersections[0].starboard_point_id = "NO-SUCH-POINT".into();
        assert_rejected(&spec, "waterline_starboard_reference");
    }

    #[test]
    fn waterline_plane_must_be_z_zero_in_v1() {
        let mut spec = box_fixture();
        spec.design_waterline.plane_z_m = 0.25;
        assert_rejected(&spec, "waterline_plane");
    }

    #[test]
    fn deck_starboard_seam_must_match_sheer_identity_and_coordinates() {
        let mut spec = box_fixture();
        spec.deck_surface.sections[0].points[2].z_m += 0.01;
        assert_rejected(&spec, "deck_starboard_seam");
    }

    #[test]
    fn deck_port_mirror_identity_is_required_for_symmetric_hulls() {
        let mut spec = box_fixture();
        spec.deck_surface.sections[0].points[0].point_id = "WRONG-MIRROR-ID".into();
        assert_rejected(&spec, "deck_port_mirror_seam");
    }

    #[test]
    fn symmetric_hull_must_not_supply_a_separate_port_profile() {
        let mut spec = box_fixture();
        let port_profile = spec.stations[0].starboard_profile.clone();
        spec.stations[0].port_profile = Some(port_profile);
        assert_rejected(&spec, "symmetric_port_profile_supplied");
    }

    #[test]
    fn explicit_asymmetric_policy_requires_port_profiles_and_references() {
        let mut spec = box_fixture();
        spec.symmetry = SymmetryPolicy::ExplicitPortAndStarboard;
        for station in &mut spec.stations {
            let mut port = station.starboard_profile.clone();
            for point in &mut port {
                point.point_id = format!("PORT-{}", point.point_id);
                point.y_m = -0.9 * point.y_m;
            }
            station.port_profile = Some(port);
        }
        for intersection in &mut spec.design_waterline.station_intersections {
            intersection.port_source = WaterlinePortSource::ExplicitPortProfile;
            intersection.port_point_id = Some(format!("PORT-{}", intersection.starboard_point_id));
        }
        for (station, deck_section) in spec
            .stations
            .iter()
            .zip(spec.deck_surface.sections.iter_mut())
        {
            let port_sheer = station.port_profile.as_ref().unwrap().last().unwrap();
            let deck_port = deck_section.points.first_mut().unwrap();
            deck_port.point_id = port_sheer.point_id.clone();
            deck_port.y_m = port_sheer.y_m;
            deck_port.z_m = port_sheer.z_m;
        }
        let report = validate_full_envelope_spec(&spec);
        assert_eq!(
            report.disposition,
            FullEnvelopeDisposition::ValidInputContract
        );
        assert!(report.issues.is_empty());
    }


    #[test]
    fn explicit_asymmetric_waterline_does_not_require_equal_port_starboard_breadths() {
        let mut spec = box_fixture();
        spec.symmetry = SymmetryPolicy::ExplicitPortAndStarboard;
        for station in &mut spec.stations {
            let mut port = station.starboard_profile.clone();
            for point in &mut port {
                point.point_id = format!("PORT-{}", point.point_id);
                point.y_m = -0.8 * point.y_m;
            }
            station.port_profile = Some(port);
        }
        for intersection in &mut spec.design_waterline.station_intersections {
            intersection.port_source = WaterlinePortSource::ExplicitPortProfile;
            intersection.port_point_id = Some(format!("PORT-{}", intersection.starboard_point_id));
        }
        for (station, deck_section) in spec
            .stations
            .iter()
            .zip(spec.deck_surface.sections.iter_mut())
        {
            let port_sheer = station.port_profile.as_ref().unwrap().last().unwrap();
            let deck_port = deck_section.points.first_mut().unwrap();
            deck_port.point_id = port_sheer.point_id.clone();
            deck_port.y_m = port_sheer.y_m;
            deck_port.z_m = port_sheer.z_m;
        }
        let report = validate_full_envelope_spec(&spec);
        assert_eq!(report.disposition, FullEnvelopeDisposition::ValidInputContract);
        assert!(report.issues.is_empty());
        for intersection in &spec.design_waterline.station_intersections {
            let station = spec
                .stations
                .iter()
                .find(|item| item.station_id == intersection.station_id)
                .unwrap();
            let starboard = profile_point_by_id(
                &station.starboard_profile,
                &intersection.starboard_point_id,
            )
            .unwrap();
            let port = profile_point_by_id(
                station.port_profile.as_ref().unwrap(),
                intersection.port_point_id.as_deref().unwrap(),
            )
            .unwrap();
            assert_ne!(starboard.y_m.abs(), port.y_m.abs());
        }
    }

    #[test]
    fn explicit_asymmetric_policy_cannot_silently_mirror_missing_port() {
        let mut spec = box_fixture();
        spec.symmetry = SymmetryPolicy::ExplicitPortAndStarboard;
        assert_rejected(&spec, "explicit_port_profile_missing");
    }


    #[test]
    fn explicit_port_and_starboard_profiles_share_point_ids_at_pointed_stem_only() {
        let mut spec = box_fixture();
        spec.symmetry = SymmetryPolicy::ExplicitPortAndStarboard;
        spec.end_conditions.bow = HullEndCondition::PointedStem;

        for point in &mut spec.stations[0].starboard_profile {
            point.y_m = 0.0;
        }
        let bow_port = spec.stations[0].starboard_profile.clone();
        spec.stations[0].port_profile = Some(bow_port);

        let mut stern_port = spec.stations[1].starboard_profile.clone();
        for point in &mut stern_port {
            point.point_id = format!("PORT-{}", point.point_id);
            point.y_m = -0.9 * point.y_m;
        }
        spec.stations[1].port_profile = Some(stern_port);

        let bow_station_id = spec.stations[0].station_id.clone();
        for intersection in &mut spec.design_waterline.station_intersections {
            intersection.port_source = WaterlinePortSource::ExplicitPortProfile;
            if intersection.station_id == bow_station_id {
                intersection.port_point_id = Some(intersection.starboard_point_id.clone());
            } else {
                intersection.port_point_id = Some(format!("PORT-{}", intersection.starboard_point_id));
            }
        }

        spec.deck_surface.sections[0].points = vec![DeckPoint {
            point_id: "BOW-SHEER-S".into(),
            y_m: 0.0,
            z_m: -4.0,
            role: DeckPointRole::DeckCrown,
        }];
        let stern_port_sheer = spec.stations[1].port_profile.as_ref().unwrap().last().unwrap().clone();
        let stern_deck_port = spec.deck_surface.sections[1].points.first_mut().unwrap();
        stern_deck_port.point_id = stern_port_sheer.point_id;
        stern_deck_port.y_m = stern_port_sheer.y_m;
        stern_deck_port.z_m = stern_port_sheer.z_m;

        let report = validate_full_envelope_spec(&spec);
        assert_eq!(report.disposition, FullEnvelopeDisposition::ValidInputContract);
        assert!(report.issues.is_empty());
    }

    #[test]
    fn pointed_bow_must_have_exactly_collapsed_transverse_profile() {
        let mut spec = box_fixture();
        spec.end_conditions.bow = HullEndCondition::PointedStem;
        assert_rejected(&spec, "pointed_end_not_collapsed");
    }

    #[test]
    fn closed_shell_cannot_claim_an_unaccounted_opening() {
        let mut spec = box_fixture();
        spec.boundary_semantics.openings.push(OpeningBoundary {
            opening_id: "DECK-HATCH-001".into(),
            boundary_point_refs: vec!["A".into(), "B".into(), "C".into()],
            status: OpeningStatus::Supported,
        });
        assert_rejected(&spec, "closed_shell_has_openings");
    }

    #[test]
    fn unsupported_opening_is_incomplete() {
        let mut spec = box_fixture();
        spec.boundary_semantics.exterior_shell = ExteriorShellSemantics::ExplicitBoundaryLoops;
        spec.boundary_semantics.openings.push(OpeningBoundary {
            opening_id: "UNMODELED-OPENING".into(),
            boundary_point_refs: vec!["BOW-KEEL".into(), "BOW-WL-S".into(), "BOW-SHEER-S".into()],
            status: OpeningStatus::UnsupportedRequiresReview,
        });
        let report = validate_full_envelope_spec(&spec);
        assert_eq!(report.disposition, FullEnvelopeDisposition::Incomplete);
        assert!(
            report
                .issues
                .iter()
                .any(|item| item.code == "unsupported_opening")
        );
    }

    #[test]
    fn duplicate_surface_region_is_rejected() {
        let mut spec = box_fixture();
        spec.surface_inventory.push(SurfaceRegion::Deck);
        assert_rejected(&spec, "duplicate_surface_region");
    }

    #[test]
    fn missing_stern_closure_region_is_rejected() {
        let mut spec = box_fixture();
        spec.surface_inventory
            .retain(|region| *region != SurfaceRegion::SternEndClosure);
        assert_rejected(&spec, "missing_surface_region");
    }

    #[test]
    fn external_component_digest_must_be_a_blake3_hex_digest() {
        let mut spec = box_fixture();
        spec.external_components.push(ExternalComponent {
            component_id: "rudder-001".into(),
            component_type: ExternalComponentType::Rudder,
            source_digest_blake3: "not-a-digest".into(),
            integration_semantics: ComponentIntegrationSemantics::SeparateAssembly,
        });
        assert_rejected(&spec, "external_component_digest");
    }

    #[test]
    fn schema_version_is_fail_closed() {
        let mut spec = box_fixture();
        spec.schema_version = 2;
        assert_rejected(&spec, "schema_version");
    }

    #[test]
    fn non_finite_profile_coordinates_are_rejected() {
        let mut spec = box_fixture();
        spec.stations[0].starboard_profile[1].y_m = f64::NAN;
        assert_rejected(&spec, "profile_non_finite_coordinate");
    }

    #[test]
    fn design_freeboard_must_be_supported_by_each_station_sheer() {
        let mut spec = box_fixture();
        spec.principal_dimensions.design_freeboard_m = 5.0;
        assert_rejected(&spec, "design_freeboard");
    }

    #[test]
    fn unknown_component_and_profile_fields_are_not_accepted() {
        let mut value = serde_json::to_value(box_fixture()).unwrap();
        value["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<FullEnvelopeSpec>(value).is_err());
    }

    #[test]
    fn pointed_bow_deck_may_collapse_to_one_centreplane_point() {
        let mut spec = box_fixture();
        spec.end_conditions.bow = HullEndCondition::PointedStem;
        for point in &mut spec.stations[0].starboard_profile {
            point.y_m = 0.0;
        }
        spec.deck_surface.sections[0].points = vec![DeckPoint {
            point_id: "BOW-SHEER-S".into(),
            y_m: 0.0,
            z_m: -4.0,
            role: DeckPointRole::DeckCrown,
        }];
        let report = validate_full_envelope_spec(&spec);
        assert_eq!(
            report.disposition,
            FullEnvelopeDisposition::ValidInputContract
        );
        assert!(report.issues.is_empty());
    }

    #[test]
    fn opening_boundary_point_references_must_resolve() {
        let mut spec = box_fixture();
        spec.boundary_semantics.exterior_shell = ExteriorShellSemantics::ExplicitBoundaryLoops;
        spec.boundary_semantics.openings.push(OpeningBoundary {
            opening_id: "HATCH-001".into(),
            boundary_point_refs: vec!["UNKNOWN-A".into(), "UNKNOWN-B".into(), "UNKNOWN-C".into()],
            status: OpeningStatus::Supported,
        });
        assert_rejected(&spec, "opening_boundary_point_reference");
    }

    #[test]
    fn empty_profile_is_rejected_without_panicking() {
        let mut spec = box_fixture();
        spec.stations[0].starboard_profile.clear();
        let report = validate_full_envelope_spec(&spec);
        assert_eq!(report.disposition, FullEnvelopeDisposition::Rejected);
        assert!(
            report
                .issues
                .iter()
                .any(|item| item.code == "profile_point_count")
        );
    }

    #[test]
    fn unsupported_profile_segment_is_incomplete_not_valid() {
        let mut spec = box_fixture();
        spec.stations[0].starboard_profile[0].region_to_next = ProfileRegion::Unsupported;
        let report = validate_full_envelope_spec(&spec);
        assert_eq!(report.disposition, FullEnvelopeDisposition::Incomplete);
        assert!(
            report
                .issues
                .iter()
                .any(|item| item.code == "unsupported_profile_region")
        );
    }

    #[test]
    fn required_nullable_port_point_id_must_be_present_in_json() {
        let mut value = serde_json::to_value(box_fixture()).unwrap();
        value["design_waterline"]["station_intersections"][0]
            .as_object_mut()
            .unwrap()
            .remove("port_point_id");
        assert!(serde_json::from_value::<FullEnvelopeSpec>(value).is_err());
    }

    #[test]
    fn missing_openings_array_is_not_silently_defaulted() {
        let mut value = serde_json::to_value(box_fixture()).unwrap();
        value["boundary_semantics"]
            .as_object_mut()
            .unwrap()
            .remove("openings");
        assert!(serde_json::from_value::<FullEnvelopeSpec>(value).is_err());
    }

    #[test]
    fn schema_deserializer_rejects_unknown_nested_fields() {
        let mut value = serde_json::to_value(box_fixture()).unwrap();
        value["stations"][0]["starboard_profile"][0]["mystery"] = serde_json::json!(9);
        assert!(serde_json::from_value::<FullEnvelopeSpec>(value).is_err());
    }

    #[test]
    fn station_budget_is_bounded() {
        let mut spec = box_fixture();
        let original = spec.stations[1].clone();
        spec.stations = (0..=MAX_FULL_ENVELOPE_STATIONS)
            .map(|index| {
                let mut station = original.clone();
                station.station_id = format!("S-{index}");
                station.x_over_length = index as f64 / MAX_FULL_ENVELOPE_STATIONS as f64;
                station
            })
            .collect();
        let report = validate_full_envelope_spec(&spec);
        assert_eq!(report.disposition, FullEnvelopeDisposition::Rejected);
        assert!(
            report
                .issues
                .iter()
                .any(|item| item.code == "station_count")
        );
    }
}
