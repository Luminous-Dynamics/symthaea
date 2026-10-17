// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic checks for modular-interior configuration manifests.
//!
//! This module checks declared envelopes, IDs, service ports and arithmetic only.
//! It does not establish seaworthiness, structural adequacy, stability, fire compliance,
//! accessibility, habitability or approval to build.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

pub const VALIDATOR_VERSION: &str = "marine-interior-validator-v1";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Configuration {
    pub schema_version: String,
    pub configuration_id: String,
    pub configuration_name: String,
    pub customization_intent: String,
    pub platform: Platform,
    pub coordinate_frame: CoordinateFrame,
    pub module_catalog: Vec<ModuleDefinition>,
    pub placements: Vec<Placement>,
    pub platform_services: Vec<PlatformService>,
    pub connections: Vec<Connection>,
    pub reserved_regions: Vec<ReservedRegion>,
    pub safety_gates: Vec<SafetyGate>,
    pub assumptions: Vec<Assumption>,
    pub provenance: Vec<Provenance>,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Platform {
    pub vessel_concept_ref: String,
    pub vessel_revision: String,
    pub archetype: String,
    pub interior_bounds_mm: EnvelopeMm,
    pub deck_bounds: Vec<DeckBounds>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CoordinateFrame {
    pub unit: String,
    pub origin: String,
    pub x_axis: String,
    pub y_axis: String,
    pub z_axis: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnvelopeMm {
    pub length_mm: i64,
    pub width_mm: i64,
    pub height_mm: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BoundsMm {
    pub x_min_mm: i64,
    pub x_max_mm: i64,
    pub y_min_mm: i64,
    pub y_max_mm: i64,
    pub z_min_mm: i64,
    pub z_max_mm: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RatedValue {
    pub value: Option<f64>,
    pub unit: String,
    pub status: String,
    pub source_ref: Option<String>,
    pub rationale: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeckBounds {
    pub deck_id: String,
    pub bounds_mm: BoundsMm,
    pub max_area_load_kg_m2: RatedValue,
    pub max_point_load_kg: RatedValue,
    pub source_ref: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CgMm {
    pub x_mm: Option<f64>,
    pub y_mm: Option<f64>,
    pub z_mm: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MassProperties {
    pub status: String,
    pub mass_kg: Option<f64>,
    pub local_cg_mm: CgMm,
    pub source_ref: Option<String>,
    pub rationale: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Port {
    pub port_id: String,
    pub medium: String,
    pub direction: String,
    pub rating: RatedValue,
    pub connector_profile: String,
    pub criticality: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceClearanceMm {
    pub forward: i64,
    pub aft: i64,
    pub port: i64,
    pub starboard: i64,
    pub above: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModuleDefinition {
    pub module_id: String,
    pub revision: u32,
    pub name: String,
    pub function: String,
    pub envelope_mm: EnvelopeMm,
    pub mass_properties: MassProperties,
    pub ports: Vec<Port>,
    pub service_clearance_mm: ServiceClearanceMm,
    pub minimum_accessibility: String,
    pub fire_material_status: String,
    pub mounting_status: String,
    pub electrical_status: String,
    pub sanitary_status: String,
    pub provenance: Vec<Provenance>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PositionMm {
    pub x_mm: i64,
    pub y_mm: i64,
    pub z_mm: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    pub instance_id: String,
    pub module_id: String,
    pub module_revision: u32,
    pub deck_id: String,
    pub position_mm: PositionMm,
    pub rotation_z_deg: i16,
    pub status: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlatformService {
    pub service_id: String,
    pub medium: String,
    pub direction: String,
    pub capacity: RatedValue,
    pub connector_profile: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub kind: String,
    pub instance_id: Option<String>,
    pub port_id: Option<String>,
    pub service_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouteLength {
    pub value: Option<f64>,
    pub status: String,
    pub source_ref: Option<String>,
    pub rationale: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Connection {
    pub connection_id: String,
    pub medium: String,
    pub endpoint_a: Endpoint,
    pub endpoint_b: Endpoint,
    pub route_length_m: RouteLength,
    pub compatibility_status: String,
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReservedRegion {
    pub region_id: String,
    pub deck_id: String,
    pub bounds_mm: BoundsMm,
    pub purpose: String,
    pub clearance_rule: String,
    pub evidence_status: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SafetyGate {
    pub gate_id: String,
    pub category: String,
    pub disposition: String,
    pub evidence_refs: Vec<String>,
    pub notes: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Assumption {
    pub assumption_id: String,
    pub category: String,
    pub statement: String,
    pub status: String,
    pub falsification_method: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub source_id: String,
    pub title: String,
    pub uri: String,
    pub scope: String,
    pub source_type: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Error,
    Indeterminate,
    Warning,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Finding {
    pub code: String,
    pub severity: Severity,
    pub subject: String,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Point3Mm {
    pub x_mm: f64,
    pub y_mm: f64,
    pub z_mm: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct GeometrySummary {
    pub checked_placements: usize,
    pub physical_overlaps: usize,
    pub out_of_bounds_placements: usize,
    pub clearance_out_of_bounds: usize,
    pub reserved_region_conflicts: usize,
    pub blocked_service_clearances: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct WeightLedger {
    pub known_subtotal_mass_kg: f64,
    pub total_mass_kg: Option<f64>,
    pub mass_weighted_cg_mm: Option<Point3Mm>,
    pub all_mass_values_present: bool,
    pub all_local_cg_values_present: bool,
    pub all_mass_inputs_measured_or_source_verified: bool,
    pub disposition: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ServiceLoadSummary {
    pub service_id: String,
    pub medium: String,
    pub declared_load: Option<f64>,
    pub declared_capacity: Option<f64>,
    pub unit: String,
    pub connection_count: usize,
    pub disposition: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ReceiptBody {
    pub configuration_id: String,
    pub validator_version: String,
    pub configuration_sha256: String,
    pub overall_disposition: String,
    pub findings: Vec<Finding>,
    pub geometry: GeometrySummary,
    pub weight_ledger: WeightLedger,
    pub service_loads: Vec<ServiceLoadSummary>,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ValidationReceipt {
    pub body: ReceiptBody,
    pub receipt_sha256: String,
}

#[derive(Clone, Copy, Debug)]
struct Box3 {
    x0: i128,
    x1: i128,
    y0: i128,
    y1: i128,
    z0: i128,
    z1: i128,
}

impl Box3 {
    fn intersects(self, other: Self) -> bool {
        self.x0 < other.x1 && self.x1 > other.x0
            && self.y0 < other.y1 && self.y1 > other.y0
            && self.z0 < other.z1 && self.z1 > other.z0
    }

    fn within(self, bounds: &BoundsMm) -> bool {
        self.x0 >= i128::from(bounds.x_min_mm) && self.x1 <= i128::from(bounds.x_max_mm)
            && self.y0 >= i128::from(bounds.y_min_mm) && self.y1 <= i128::from(bounds.y_max_mm)
            && self.z0 >= i128::from(bounds.z_min_mm) && self.z1 <= i128::from(bounds.z_max_mm)
    }

    fn from_bounds(bounds: &BoundsMm) -> Self {
        Self {
            x0: i128::from(bounds.x_min_mm), x1: i128::from(bounds.x_max_mm),
            y0: i128::from(bounds.y_min_mm), y1: i128::from(bounds.y_max_mm),
            z0: i128::from(bounds.z_min_mm), z1: i128::from(bounds.z_max_mm),
        }
    }
}

fn add_finding(
    findings: &mut Vec<Finding>,
    code: &str,
    severity: Severity,
    subject: impl Into<String>,
    message: impl Into<String>,
) {
    findings.push(Finding {
        code: code.to_owned(),
        severity,
        subject: subject.into(),
        message: message.into(),
    });
}

fn check_unique<'a>(
    ids: impl Iterator<Item = &'a str>,
    label: &str,
    findings: &mut Vec<Finding>,
) {
    let mut seen = HashSet::new();
    for id in ids {
        if !seen.insert(id.to_owned()) {
            add_finding(findings, "DUPLICATE_IDENTIFIER", Severity::Error, id,
                format!("duplicate {label} identifier"));
        }
    }
}

fn valid_bounds(bounds: &BoundsMm) -> bool {
    bounds.x_min_mm < bounds.x_max_mm
        && bounds.y_min_mm < bounds.y_max_mm
        && bounds.z_min_mm < bounds.z_max_mm
}

fn validate_rated(value: &RatedValue, subject: &str, findings: &mut Vec<Finding>) {
    if !matches!(
        value.status.as_str(),
        "unknown" | "design_assumption" | "computed" | "measured" | "source_verified"
    ) {
        add_finding(findings, "INVALID_RATING_STATUS", Severity::Error, subject,
            "unsupported rating evidence status");
    }
    match value.value {
        Some(number) if !number.is_finite() || number < 0.0 =>
            add_finding(findings, "INVALID_RATING_VALUE", Severity::Error, subject,
                "ratings must be finite and non-negative"),
        Some(_) if value.status == "unknown" =>
            add_finding(findings, "UNKNOWN_RATING_HAS_VALUE", Severity::Error, subject,
                "unknown status cannot carry a numeric rating"),
        None if value.status != "unknown" =>
            add_finding(findings, "RATING_VALUE_MISSING", Severity::Indeterminate, subject,
                "rating is missing despite a non-unknown status"),
        _ => {}
    }
}

fn physical_box(placement: &Placement, module: &ModuleDefinition) -> Option<Box3> {
    let length = module.envelope_mm.length_mm;
    let width = module.envelope_mm.width_mm;
    let height = module.envelope_mm.height_mm;
    if length <= 0 || width <= 0 || height <= 0 {
        return None;
    }
    let (length, width) = match placement.rotation_z_deg {
        0 | 180 => (length, width),
        90 | 270 => (width, length),
        _ => return None,
    };
    let x0 = i128::from(placement.position_mm.x_mm);
    let y0 = i128::from(placement.position_mm.y_mm);
    let z0 = i128::from(placement.position_mm.z_mm);
    Some(Box3 {
        x0, x1: x0 + i128::from(length),
        y0, y1: y0 + i128::from(width),
        z0, z1: z0 + i128::from(height),
    })
}

fn clearance_box(placement: &Placement, module: &ModuleDefinition) -> Option<Box3> {
    let physical = physical_box(placement, module)?;
    let c = &module.service_clearance_mm;
    if [c.forward, c.aft, c.port, c.starboard, c.above].iter().any(|v| *v < 0) {
        return None;
    }
    let (xm, xp, ym, yp) = match placement.rotation_z_deg {
        0 => (c.aft, c.forward, c.port, c.starboard),
        90 => (c.starboard, c.port, c.aft, c.forward),
        180 => (c.forward, c.aft, c.starboard, c.port),
        270 => (c.port, c.starboard, c.forward, c.aft),
        _ => return None,
    };
    Some(Box3 {
        x0: physical.x0 - i128::from(xm),
        x1: physical.x1 + i128::from(xp),
        y0: physical.y0 - i128::from(ym),
        y1: physical.y1 + i128::from(yp),
        z0: physical.z0,
        z1: physical.z1 + i128::from(c.above),
    })
}

fn resolve_port<'a>(
    endpoint: &'a Endpoint,
    placements: &HashMap<&'a str, &'a Placement>,
    modules: &HashMap<&'a str, &'a ModuleDefinition>,
) -> Option<(&'a str, &'a Port)> {
    let instance_id = endpoint.instance_id.as_deref()?;
    let port_id = endpoint.port_id.as_deref()?;
    let placement = placements.get(instance_id)?;
    let module = modules.get(placement.module_id.as_str())?;
    let port = module.ports.iter().find(|p| p.port_id == port_id)?;
    Some((instance_id, port))
}

fn resolve_service<'a>(
    endpoint: &Endpoint,
    services: &HashMap<&'a str, &'a PlatformService>,
) -> Option<&'a PlatformService> {
    services.get(endpoint.service_id.as_deref()?).copied()
}

fn endpoint_shape_valid(endpoint: &Endpoint) -> bool {
    match endpoint.kind.as_str() {
        "module_port" => endpoint.instance_id.as_deref().is_some_and(|v| !v.trim().is_empty())
            && endpoint.port_id.as_deref().is_some_and(|v| !v.trim().is_empty())
            && endpoint.service_id.is_none(),
        "platform_service" => endpoint.instance_id.is_none()
            && endpoint.port_id.is_none()
            && endpoint.service_id.as_deref().is_some_and(|v| !v.trim().is_empty()),
        _ => false,
    }
}

fn stable_id(value: &str) -> bool {
    (3..=64).contains(&value.len())
        && value.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

fn compatible_port_direction(port: &str, service: &str) -> bool {
    match service {
        "source" => matches!(port, "in" | "bidirectional"),
        "sink" => matches!(port, "out" | "bidirectional"),
        "bidirectional" => matches!(port, "in" | "out" | "bidirectional"),
        _ => false,
    }
}

fn compatible_pair(a: &str, b: &str) -> bool {
    (a == "out" && b == "in") || (a == "in" && b == "out")
        || a == "bidirectional" || b == "bidirectional"
}

fn rotate_cg(x: f64, y: f64, module: &ModuleDefinition, rotation: i16) -> Option<(f64, f64)> {
    let l = module.envelope_mm.length_mm as f64;
    let w = module.envelope_mm.width_mm as f64;
    match rotation {
        0 => Some((x, y)),
        90 => Some((w - y, x)),
        180 => Some((l - x, w - y)),
        270 => Some((y, l - x)),
        _ => None,
    }
}

fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

fn compute_weight_ledger(
    configuration: &Configuration,
    modules: &HashMap<&str, &ModuleDefinition>,
    findings: &mut Vec<Finding>,
) -> WeightLedger {
    let mut known_total = 0.0;
    let mut mass_complete = true;
    let mut cg_complete = true;
    let mut inputs_verified = true;
    let mut wx = 0.0;
    let mut wy = 0.0;
    let mut wz = 0.0;

    for placement in &configuration.placements {
        let Some(module) = modules.get(placement.module_id.as_str()).copied() else {
            mass_complete = false;
            cg_complete = false;
            inputs_verified = false;
            continue;
        };
        let mp = &module.mass_properties;
        let mass = if mp.status == "unknown" {
            None
        } else {
            mp.mass_kg.filter(|v| v.is_finite() && *v >= 0.0)
        };
        if let Some(m) = mass {
            known_total += m;
            if !known_total.is_finite() {
                mass_complete = false;
                add_finding(findings, "MASS_SUM_OVERFLOW", Severity::Error,
                    &placement.instance_id, "known mass subtotal overflowed numeric range");
            }
        } else {
            mass_complete = false;
            inputs_verified = false;
            add_finding(findings, "MASS_INPUT_UNKNOWN", Severity::Indeterminate,
                &placement.instance_id, "mass is unknown; total configuration mass cannot be closed");
        }
        if !matches!(mp.status.as_str(), "measured" | "source_verified") {
            inputs_verified = false;
            add_finding(findings, "MASS_INPUT_NOT_VERIFIED", Severity::Indeterminate,
                &placement.instance_id, "mass is not measured or source-verified");
        }
        let cg = &mp.local_cg_mm;
        let local = match (cg.x_mm, cg.y_mm, cg.z_mm) {
            (Some(x), Some(y), Some(z)) if x.is_finite() && y.is_finite() && z.is_finite() =>
                Some((x, y, z)),
            _ => None,
        };
        if local.is_none() {
            cg_complete = false;
            inputs_verified = false;
            add_finding(findings, "CG_INPUT_UNKNOWN", Severity::Indeterminate,
                &placement.instance_id, "module-local centre of gravity is incomplete");
        }
        if let Some((x, y, z)) = local {
            if x < 0.0 || x > module.envelope_mm.length_mm as f64
                || y < 0.0 || y > module.envelope_mm.width_mm as f64
                || z < 0.0 || z > module.envelope_mm.height_mm as f64
            {
                add_finding(findings, "CG_OUTSIDE_MODULE_ENVELOPE", Severity::Error,
                    &placement.instance_id, "module-local CG lies outside the declared module envelope");
            }
            if let (Some(m), Some((rx, ry))) = (mass, rotate_cg(x, y, module, placement.rotation_z_deg)) {
                wx += m * (placement.position_mm.x_mm as f64 + rx);
                wy += m * (placement.position_mm.y_mm as f64 + ry);
                wz += m * (placement.position_mm.z_mm as f64 + z);
            }
        }
    }

    let total_mass = (mass_complete).then_some(known_total);
    let cg = if mass_complete && cg_complete && known_total > 0.0 {
        Some(Point3Mm {
            x_mm: wx / known_total,
            y_mm: wy / known_total,
            z_mm: wz / known_total,
        })
    } else {
        None
    };
    let disposition = if !mass_complete || !cg_complete {
        "indeterminate"
    } else if inputs_verified {
        "arithmetic_from_measured_or_source_verified_inputs"
    } else {
        "assumption_based_not_weight_closure"
    };
    if disposition != "arithmetic_from_measured_or_source_verified_inputs" {
        add_finding(findings, "WEIGHT_LEDGER_NOT_CLOSED", Severity::Indeterminate,
            &configuration.configuration_id,
            "partial or assumption-based weight arithmetic is not a vessel stability result");
    }
    WeightLedger {
        known_subtotal_mass_kg: known_total,
        total_mass_kg: total_mass,
        mass_weighted_cg_mm: cg,
        all_mass_values_present: mass_complete,
        all_local_cg_values_present: cg_complete,
        all_mass_inputs_measured_or_source_verified: inputs_verified,
        disposition: disposition.to_owned(),
    }
}

/// Validates the declared configuration and creates a deterministic receipt.
/// The result is bookkeeping evidence only and grants no design, safety, or operating approval.
pub fn validate_configuration(
    configuration: &Configuration,
) -> Result<ValidationReceipt, serde_json::Error> {
    let input_bytes = serde_json::to_vec(configuration)?;
    let input_digest = sha256_hex(&input_bytes);
    let mut findings = Vec::new();
    let mut geometry = GeometrySummary {
        checked_placements: configuration.placements.len(),
        physical_overlaps: 0,
        out_of_bounds_placements: 0,
        clearance_out_of_bounds: 0,
        reserved_region_conflicts: 0,
        blocked_service_clearances: 0,
    };

    if configuration.schema_version != "1.0.0" {
        add_finding(&mut findings, "UNSUPPORTED_SCHEMA_VERSION", Severity::Error,
            &configuration.configuration_id, "schema version is not supported");
    }
    if !stable_id(&configuration.configuration_id) || configuration.configuration_name.trim().is_empty()
        || configuration.customization_intent.trim().is_empty() || configuration.claim_ceiling.trim().is_empty()
    {
        add_finding(&mut findings, "INVALID_CONFIGURATION_IDENTITY", Severity::Error,
            &configuration.configuration_id, "configuration ID or required descriptive text is invalid");
    }
    if configuration.module_catalog.is_empty() || configuration.placements.is_empty()
        || configuration.platform.deck_bounds.is_empty() || configuration.safety_gates.is_empty()
        || configuration.provenance.is_empty() || configuration.assumptions.is_empty()
    {
        add_finding(&mut findings, "INCOMPLETE_CONFIGURATION_MANIFEST", Severity::Error,
            &configuration.configuration_id,
            "module catalog, placements, deck bounds, safety gates, assumptions and provenance must be populated");
    }
    if configuration.coordinate_frame.unit != "mm"
        || configuration.coordinate_frame.x_axis != "forward"
        || configuration.coordinate_frame.y_axis != "starboard"
        || configuration.coordinate_frame.z_axis != "up"
    {
        add_finding(&mut findings, "UNSUPPORTED_COORDINATE_FRAME", Severity::Error,
            &configuration.configuration_id, "coordinate frame must use declared ship axes in mm");
    }

    check_unique(configuration.module_catalog.iter().map(|x| x.module_id.as_str()), "module", &mut findings);
    check_unique(configuration.placements.iter().map(|x| x.instance_id.as_str()), "instance", &mut findings);
    check_unique(configuration.platform.deck_bounds.iter().map(|x| x.deck_id.as_str()), "deck", &mut findings);
    check_unique(configuration.platform_services.iter().map(|x| x.service_id.as_str()), "platform service", &mut findings);
    check_unique(configuration.connections.iter().map(|x| x.connection_id.as_str()), "connection", &mut findings);
    check_unique(configuration.reserved_regions.iter().map(|x| x.region_id.as_str()), "reserved region", &mut findings);

    let modules: HashMap<&str, &ModuleDefinition> = configuration.module_catalog.iter()
        .map(|m| (m.module_id.as_str(), m)).collect();
    let placements: HashMap<&str, &Placement> = configuration.placements.iter()
        .map(|p| (p.instance_id.as_str(), p)).collect();
    let services: HashMap<&str, &PlatformService> = configuration.platform_services.iter()
        .map(|s| (s.service_id.as_str(), s)).collect();
    let decks: HashMap<&str, &DeckBounds> = configuration.platform.deck_bounds.iter()
        .map(|d| (d.deck_id.as_str(), d)).collect();

    for deck in &configuration.platform.deck_bounds {
        if !valid_bounds(&deck.bounds_mm) {
            add_finding(&mut findings, "INVALID_DECK_BOUNDS", Severity::Error,
                &deck.deck_id, "deck minimum coordinates must be below maximum coordinates");
        }
        validate_rated(&deck.max_area_load_kg_m2, &deck.deck_id, &mut findings);
        validate_rated(&deck.max_point_load_kg, &deck.deck_id, &mut findings);
    }

    for module in &configuration.module_catalog {
        let e = &module.envelope_mm;
        if !stable_id(&module.module_id) || module.name.trim().is_empty() || module.function.trim().is_empty()
            || module.provenance.is_empty() || module.ports.is_empty()
        {
            add_finding(&mut findings, "INCOMPLETE_MODULE_DEFINITION", Severity::Error,
                &module.module_id, "module ID, name, function, ports and provenance must be populated");
        }
        if module.revision == 0 || e.length_mm <= 0 || e.width_mm <= 0 || e.height_mm <= 0 {
            add_finding(&mut findings, "INVALID_MODULE_ENVELOPE", Severity::Error,
                &module.module_id, "revision and all envelope dimensions must be positive");
        }
        if !matches!(module.mass_properties.status.as_str(),
            "unknown" | "design_assumption" | "computed" | "measured" | "source_verified")
        {
            add_finding(&mut findings, "INVALID_MASS_STATUS", Severity::Error,
                &module.module_id, "unsupported mass evidence status");
        }
        if module.mass_properties.status == "unknown"
            && (module.mass_properties.mass_kg.is_some()
                || module.mass_properties.local_cg_mm.x_mm.is_some()
                || module.mass_properties.local_cg_mm.y_mm.is_some()
                || module.mass_properties.local_cg_mm.z_mm.is_some())
        {
            add_finding(&mut findings, "UNKNOWN_MASS_HAS_VALUE", Severity::Error,
                &module.module_id, "unknown mass status cannot carry a mass or local CG");
        }
        let mut port_ids = HashSet::new();
        for port in &module.ports {
            if !port_ids.insert(port.port_id.as_str()) {
                add_finding(&mut findings, "DUPLICATE_PORT_ID", Severity::Error,
                    &module.module_id, format!("duplicate port ID {}", port.port_id));
            }
            validate_rated(&port.rating, &format!("{}.{}", module.module_id, port.port_id), &mut findings);
            if port.rating.value.is_some()
                && port.rating.status != "unknown"
                && !matches!(port.rating.status.as_str(), "measured" | "source_verified")
            {
                add_finding(&mut findings, "PORT_RATING_ASSUMED", Severity::Indeterminate,
                    format!("{}.{}", module.module_id, port.port_id), "port rating is not verified");
            }
            if !matches!(port.direction.as_str(), "in" | "out" | "bidirectional") {
                add_finding(&mut findings, "INVALID_PORT_DIRECTION", Severity::Error,
                    format!("{}.{}", module.module_id, port.port_id), "unsupported port direction");
            }
            if module.service_clearance_mm.forward < 0 || module.service_clearance_mm.aft < 0
                || module.service_clearance_mm.port < 0 || module.service_clearance_mm.starboard < 0
                || module.service_clearance_mm.above < 0
            {
                add_finding(&mut findings, "NEGATIVE_SERVICE_CLEARANCE", Severity::Error,
                    &module.module_id, "service clearance cannot be negative");
                break;
            }
        }
    }

    for service in &configuration.platform_services {
        validate_rated(&service.capacity, &service.service_id, &mut findings);
        if service.capacity.value.is_some()
            && service.capacity.status != "unknown"
            && !matches!(service.capacity.status.as_str(), "measured" | "source_verified")
        {
            add_finding(&mut findings, "SERVICE_CAPACITY_ASSUMED", Severity::Indeterminate,
                &service.service_id, "platform service capacity is not verified");
        }
    }

    let mut boxes: Vec<(&Placement, Box3, Box3)> = Vec::new();
    for placement in &configuration.placements {
        let Some(module) = modules.get(placement.module_id.as_str()).copied() else {
            add_finding(&mut findings, "MODULE_REFERENCE_MISSING", Severity::Error,
                &placement.instance_id, "placement references a missing catalogue module");
            continue;
        };
        if module.revision != placement.module_revision {
            add_finding(&mut findings, "MODULE_REVISION_MISMATCH", Severity::Error,
                &placement.instance_id, "placement revision does not match the pinned module revision");
        }
        let Some(deck) = decks.get(placement.deck_id.as_str()).copied() else {
            add_finding(&mut findings, "DECK_REFERENCE_MISSING", Severity::Error,
                &placement.instance_id, "placement references a missing deck");
            continue;
        };
        let Some(physical) = physical_box(placement, module) else {
            add_finding(&mut findings, "INVALID_PLACEMENT_GEOMETRY", Severity::Error,
                &placement.instance_id, "unsupported rotation or invalid module envelope");
            continue;
        };
        let Some(clearance) = clearance_box(placement, module) else {
            add_finding(&mut findings, "INVALID_CLEARANCE_GEOMETRY", Severity::Error,
                &placement.instance_id, "invalid rotation or service-clearance envelope");
            continue;
        };
        if !physical.within(&deck.bounds_mm) {
            geometry.out_of_bounds_placements += 1;
            add_finding(&mut findings, "PLACEMENT_OUT_OF_BOUNDS", Severity::Error,
                &placement.instance_id, "physical envelope lies outside the declared deck bounds");
        }
        if !clearance.within(&deck.bounds_mm) {
            geometry.clearance_out_of_bounds += 1;
            add_finding(&mut findings, "CLEARANCE_OUT_OF_BOUNDS", Severity::Error,
                &placement.instance_id, "service-clearance envelope extends outside deck bounds");
        }
        for region in configuration.reserved_regions.iter().filter(|r| r.deck_id == placement.deck_id) {
            if !valid_bounds(&region.bounds_mm) {
                add_finding(&mut findings, "INVALID_RESERVED_REGION_BOUNDS", Severity::Error,
                    &region.region_id, "reserved-region bounds are invalid");
                continue;
            }
            let rb = Box3::from_bounds(&region.bounds_mm);
            let requires_clear = matches!(region.clearance_rule.as_str(), "keep_clear" | "access_required");
            if requires_clear && clearance.intersects(rb) {
                geometry.reserved_region_conflicts += 1;
                add_finding(&mut findings, "RESERVED_REGION_CONFLICT", Severity::Error,
                    format!("{} / {}", placement.instance_id, region.region_id),
                    "module or service-clearance envelope intersects a required-clear region");
            } else if region.clearance_rule == "boundary_must_not_be_penetrated" && physical.intersects(rb) {
                geometry.reserved_region_conflicts += 1;
                add_finding(&mut findings, "BOUNDARY_PENETRATION", Severity::Error,
                    format!("{} / {}", placement.instance_id, region.region_id),
                    "module envelope intersects a declared fire/watertight boundary");
            } else if region.clearance_rule == "requires_review" && clearance.intersects(rb) {
                add_finding(&mut findings, "RESERVED_REGION_REQUIRES_REVIEW", Severity::Indeterminate,
                    format!("{} / {}", placement.instance_id, region.region_id),
                    "reserved-region intersection requires specialist review");
            }
        }
        for (prior, prior_physical, prior_clearance) in &boxes {
            if prior.deck_id != placement.deck_id {
                continue;
            }
            if physical.intersects(*prior_physical) {
                geometry.physical_overlaps += 1;
                add_finding(&mut findings, "MODULE_PHYSICAL_OVERLAP", Severity::Error,
                    format!("{} / {}", prior.instance_id, placement.instance_id),
                    "module physical envelopes intersect");
            } else if clearance.intersects(*prior_physical) || prior_clearance.intersects(physical) {
                geometry.blocked_service_clearances += 1;
                add_finding(&mut findings, "SERVICE_CLEARANCE_BLOCKED", Severity::Error,
                    format!("{} / {}", prior.instance_id, placement.instance_id),
                    "one module occupies another module's service-clearance envelope");
            }
        }
        boxes.push((placement, physical, clearance));
    }

    let mut port_use: HashMap<String, usize> = HashMap::new();
    let mut service_loads = Vec::new();
    for connection in &configuration.connections {
        if !stable_id(&connection.connection_id) || !endpoint_shape_valid(&connection.endpoint_a)
            || !endpoint_shape_valid(&connection.endpoint_b)
        {
            add_finding(&mut findings, "INVALID_CONNECTION_ENDPOINT_SHAPE", Severity::Error,
                &connection.connection_id, "connection endpoints must match their declared module-port or platform-service shape");
            continue;
        }
        if connection.route_length_m.value.is_none() || connection.route_length_m.status == "unknown" {
            add_finding(&mut findings, "SERVICE_ROUTE_UNKNOWN", Severity::Indeterminate,
                &connection.connection_id, "service route has not been laid out or measured");
        }
        match connection.compatibility_status.as_str() {
            "verified_incompatible" => add_finding(&mut findings, "CONNECTION_INCOMPATIBLE", Severity::Error,
                &connection.connection_id, "connection is marked incompatible"),
            "indeterminate" => add_finding(&mut findings, "CONNECTION_INDETERMINATE", Severity::Indeterminate,
                &connection.connection_id, "connection compatibility is indeterminate"),
            "declared_unverified" => add_finding(&mut findings, "CONNECTION_UNVERIFIED", Severity::Indeterminate,
                &connection.connection_id, "interface compatibility has not been verified"),
            "verified_compatible" if connection.evidence_refs.is_empty() =>
                add_finding(&mut findings, "VERIFIED_CONNECTION_WITHOUT_EVIDENCE", Severity::Error,
                    &connection.connection_id, "verified connection lacks evidence references"),
            "verified_compatible" => {}
            _ => add_finding(&mut findings, "INVALID_CONNECTION_STATUS", Severity::Error,
                &connection.connection_id, "unsupported connection status"),
        }

        let ap = resolve_port(&connection.endpoint_a, &placements, &modules);
        let bp = resolve_port(&connection.endpoint_b, &placements, &modules);
        let asvc = resolve_service(&connection.endpoint_a, &services);
        let bsvc = resolve_service(&connection.endpoint_b, &services);

        if connection.endpoint_a.kind == "module_port" && ap.is_none()
            || connection.endpoint_b.kind == "module_port" && bp.is_none()
        {
            add_finding(&mut findings, "CONNECTION_PORT_UNRESOLVED", Severity::Error,
                &connection.connection_id, "connection refers to a missing placement/port");
        }
        if connection.endpoint_a.kind == "platform_service" && asvc.is_none()
            || connection.endpoint_b.kind == "platform_service" && bsvc.is_none()
        {
            add_finding(&mut findings, "CONNECTION_SERVICE_UNRESOLVED", Severity::Error,
                &connection.connection_id, "connection refers to a missing platform service");
        }
        let endpoint_a_valid = (connection.endpoint_a.kind == "module_port" && ap.is_some())
            || (connection.endpoint_a.kind == "platform_service" && asvc.is_some());
        let endpoint_b_valid = (connection.endpoint_b.kind == "module_port" && bp.is_some())
            || (connection.endpoint_b.kind == "platform_service" && bsvc.is_some());
        if !endpoint_a_valid || !endpoint_b_valid {
            continue;
        }

        if let (Some((instance, port)), None, None, Some(service)) = (ap, bp, asvc, bsvc) {
            *port_use.entry(format!("{}.{}", instance, port.port_id)).or_default() += 1;
            validate_module_service_link(&connection.medium, port, service, &connection.connection_id, &mut findings);
            add_service_load(&mut service_loads, service, port, &connection.connection_id, &mut findings);
        } else if let (None, Some((instance, port)), Some(service), None) = (ap, bp, asvc, bsvc) {
            *port_use.entry(format!("{}.{}", instance, port.port_id)).or_default() += 1;
            validate_module_service_link(&connection.medium, port, service, &connection.connection_id, &mut findings);
            add_service_load(&mut service_loads, service, port, &connection.connection_id, &mut findings);
        } else if let (Some((ai, pa)), Some((bi, pb)), None, None) = (ap, bp, asvc, bsvc) {
            *port_use.entry(format!("{}.{}", ai, pa.port_id)).or_default() += 1;
            *port_use.entry(format!("{}.{}", bi, pb.port_id)).or_default() += 1;
            if connection.medium != pa.medium || connection.medium != pb.medium
                || pa.connector_profile != pb.connector_profile
                || !compatible_pair(&pa.direction, &pb.direction)
            {
                add_finding(&mut findings, "MODULE_LINK_INCOMPATIBLE", Severity::Error,
                    &connection.connection_id, "module-to-module interfaces differ by medium, profile or direction");
            } else if pa.connector_profile.contains("PROVISIONAL") {
                add_finding(&mut findings, "CONNECTOR_PROFILE_PROVISIONAL", Severity::Indeterminate,
                    &connection.connection_id, "provisional profile is not evidence of a proven physical connector standard");
            }
        } else {
            add_finding(&mut findings, "UNSUPPORTED_CONNECTION_TOPOLOGY", Severity::Error,
                &connection.connection_id, "unsupported endpoint pairing; use an explicit adapter module");
        }
    }

    for placement in &configuration.placements {
        if let Some(module) = modules.get(placement.module_id.as_str()) {
            for port in &module.ports {
                let count = *port_use.get(&format!("{}.{}", placement.instance_id, port.port_id)).unwrap_or(&0);
                if count > 1 {
                    add_finding(&mut findings, "PORT_CONNECTED_MULTIPLE_TIMES", Severity::Error,
                        format!("{}.{}", placement.instance_id, port.port_id),
                        "port has multiple connections; model a splitter/adapter explicitly");
                } else if count == 0 && port.criticality == "safety_critical" {
                    add_finding(&mut findings, "SAFETY_CRITICAL_PORT_UNCONNECTED", Severity::Error,
                        format!("{}.{}", placement.instance_id, port.port_id), "safety-critical port is unconnected");
                }
            }
        }
    }

    for gate in &configuration.safety_gates {
        match gate.disposition.as_str() {
            "fail" => add_finding(&mut findings, "SAFETY_GATE_FAILED", Severity::Error,
                &gate.gate_id, "hard safety gate is marked fail"),
            "not_assessed" | "indeterminate" => add_finding(&mut findings, "SAFETY_GATE_NOT_CLOSED",
                Severity::Indeterminate, &gate.gate_id, "hard safety gate is not closed"),
            "pass" if gate.evidence_refs.is_empty() => add_finding(&mut findings,
                "SAFETY_GATE_PASS_WITHOUT_EVIDENCE", Severity::Error, &gate.gate_id,
                "safety gate cannot pass without evidence references"),
            "not_applicable" if gate.evidence_refs.is_empty() =>
                add_finding(&mut findings, "SAFETY_GATE_NA_WITHOUT_EVIDENCE", Severity::Indeterminate,
                    &gate.gate_id, "not-applicable disposition lacks supporting evidence"),
            "pass" | "not_applicable" => {}
            _ => add_finding(&mut findings, "INVALID_SAFETY_GATE_DISPOSITION", Severity::Error,
                &gate.gate_id, "unsupported safety-gate disposition"),
        }
    }

    let weight_ledger = compute_weight_ledger(configuration, &modules, &mut findings);
    findings.sort_by(|a, b| a.code.cmp(&b.code).then(a.subject.cmp(&b.subject)).then(a.message.cmp(&b.message)));
    service_loads.sort_by(|a, b| a.service_id.cmp(&b.service_id));
    let overall = if findings.iter().any(|x| x.severity == Severity::Error) {
        "rejected"
    } else if findings.iter().any(|x| x.severity == Severity::Indeterminate) {
        "indeterminate"
    } else {
        "candidate_only"
    };

    let body = ReceiptBody {
        configuration_id: configuration.configuration_id.clone(),
        validator_version: VALIDATOR_VERSION.to_owned(),
        configuration_sha256: input_digest,
        overall_disposition: overall.to_owned(),
        findings,
        geometry,
        weight_ledger,
        service_loads,
        claim_ceiling: configuration.claim_ceiling.clone(),
    };
    let body_bytes = serde_json::to_vec(&body)?;
    Ok(ValidationReceipt { body, receipt_sha256: sha256_hex(&body_bytes) })
}

fn validate_module_service_link(
    medium: &str,
    port: &Port,
    service: &PlatformService,
    subject: &str,
    findings: &mut Vec<Finding>,
) {
    if medium != port.medium || medium != service.medium {
        add_finding(findings, "CONNECTION_MEDIUM_MISMATCH", Severity::Error, subject,
            "connection, module port and platform service must declare the same medium");
    }
    if !compatible_port_direction(&port.direction, &service.direction) {
        add_finding(findings, "CONNECTION_DIRECTION_MISMATCH", Severity::Error, subject,
            "module port direction conflicts with the platform source/sink direction");
    }
    if port.connector_profile != service.connector_profile {
        add_finding(findings, "CONNECTOR_PROFILE_MISMATCH", Severity::Error, subject,
            "connector profiles differ; model and review an adapter explicitly");
    } else if port.connector_profile.contains("PROVISIONAL") {
        add_finding(findings, "CONNECTOR_PROFILE_PROVISIONAL", Severity::Indeterminate, subject,
            "provisional profile is not evidence of a proven physical connector standard");
    }
    if port.rating.unit != service.capacity.unit {
        add_finding(findings, "SERVICE_CAPACITY_UNIT_MISMATCH", Severity::Error, subject,
            "port rating and platform capacity units differ");
    }
}

fn add_service_load(
    summaries: &mut Vec<ServiceLoadSummary>,
    service: &PlatformService,
    port: &Port,
    _connection_id: &str,
    findings: &mut Vec<Finding>,
) {
    let index = if let Some(i) = summaries.iter().position(|s| s.service_id == service.service_id) {
        i
    } else {
        summaries.push(ServiceLoadSummary {
            service_id: service.service_id.clone(),
            medium: service.medium.clone(),
            declared_load: Some(0.0),
            declared_capacity: service.capacity.value,
            unit: port.rating.unit.clone(),
            connection_count: 0,
            disposition: "indeterminate".to_owned(),
        });
        summaries.len() - 1
    };
    let summary = &mut summaries[index];
    summary.connection_count += 1;
    if summary.unit != port.rating.unit || service.capacity.unit != port.rating.unit {
        add_finding(findings, "MIXED_SERVICE_LOAD_UNITS", Severity::Error,
            &service.service_id, "connected port ratings and service capacity must use the same unit");
        summary.declared_load = None;
        summary.disposition = "indeterminate_mixed_units".to_owned();
        return;
    }
    if port.rating.status == "unknown" {
        summary.declared_load = None;
    } else if let (Some(sofar), Some(value)) = (summary.declared_load, port.rating.value) {
        if value.is_finite() && value >= 0.0 && sofar.is_finite() {
            summary.declared_load = Some(sofar + value);
        } else {
            summary.declared_load = None;
        }
    } else {
        summary.declared_load = None;
    }
    match (summary.declared_load, service.capacity.value) {
        (Some(load), Some(capacity))
            if service.capacity.status != "unknown"
                && load > capacity + f64::EPSILON * capacity.abs().max(1.0) => {
            summary.disposition = "exceeded_declared_capacity".to_owned();
            add_finding(findings, "SERVICE_CAPACITY_EXCEEDED", Severity::Error,
                &service.service_id, format!("declared load {load} {} exceeds capacity {capacity} {}",
                    port.rating.unit, service.capacity.unit));
        }
        (Some(_), Some(_)) if matches!(service.capacity.status.as_str(), "measured" | "source_verified")
            && matches!(port.rating.status.as_str(), "measured" | "source_verified") => {
            summary.disposition = "within_verified_declared_capacity_only".to_owned();
        }
        (Some(_), Some(_)) => {
            summary.disposition = "within_assumed_declared_capacity".to_owned();
        }
        _ => {
            summary.disposition = "indeterminate".to_owned();
            add_finding(findings, "SERVICE_CAPACITY_INDETERMINATE", Severity::Indeterminate,
                &service.service_id, "load or capacity input is incomplete");
        }
    }
}

/// Checks that a receipt belongs to this exact input and has not been modified.
/// This is integrity checking, not identity signing or proof of source authenticity.
pub fn verify_validation_receipt(configuration: &Configuration, receipt: &ValidationReceipt) -> bool {
    let Ok(input_bytes) = serde_json::to_vec(configuration) else { return false; };
    let Ok(body_bytes) = serde_json::to_vec(&receipt.body) else { return false; };
    receipt.body.configuration_id == configuration.configuration_id
        && receipt.body.validator_version == VALIDATOR_VERSION
        && receipt.body.configuration_sha256 == sha256_hex(&input_bytes)
        && receipt.receipt_sha256 == sha256_hex(&body_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIVE: &str = include_str!("../fixtures/modular-interiors/liveaboard-custom-layout-v1.json");
    const EXPEDITION: &str = include_str!("../fixtures/modular-interiors/expedition-research-layout-v1.json");

    fn live() -> Configuration { serde_json::from_str(LIVE).expect("live fixture parses") }
    fn expedition() -> Configuration { serde_json::from_str(EXPEDITION).expect("expedition fixture parses") }
    fn has(receipt: &ValidationReceipt, code: &str) -> bool {
        receipt.body.findings.iter().any(|finding| finding.code == code)
    }

    #[test]
    fn fixtures_parse_and_remain_explicitly_indeterminate() {
        for config in [live(), expedition()] {
            let receipt = validate_configuration(&config).expect("receipt serializes");
            assert_eq!(receipt.body.overall_disposition, "indeterminate");
            assert_eq!(receipt.body.geometry.physical_overlaps, 0);
            assert_eq!(receipt.body.geometry.out_of_bounds_placements, 0);
            assert_eq!(receipt.body.geometry.reserved_region_conflicts, 0);
            assert!(receipt.body.weight_ledger.mass_weighted_cg_mm.is_some());
            assert_eq!(receipt.body.weight_ledger.disposition, "assumption_based_not_weight_closure");
            assert!(has(&receipt, "SAFETY_GATE_NOT_CLOSED"));
            assert!(verify_validation_receipt(&config, &receipt));
        }
    }

    #[test]
    fn digest_is_deterministic_and_binds_input_and_receipt() {
        let config = live();
        let a = validate_configuration(&config).expect("receipt serializes");
        let b = validate_configuration(&config).expect("receipt serializes");
        assert_eq!(a.receipt_sha256, b.receipt_sha256);
        assert!(verify_validation_receipt(&config, &a));
        let mut changed = config.clone();
        changed.configuration_name.push('x');
        assert!(!verify_validation_receipt(&changed, &a));
        let mut tampered = a.clone();
        tampered.body.claim_ceiling.push('x');
        assert!(!verify_validation_receipt(&config, &tampered));
    }

    #[test]
    fn overlap_out_of_bounds_and_reserved_conflict_fail_closed() {
        let mut config = live();
        config.placements[1].position_mm.x_mm = 1000;
        let receipt = validate_configuration(&config).expect("receipt serializes");
        assert_eq!(receipt.body.overall_disposition, "rejected");
        assert!(has(&receipt, "MODULE_PHYSICAL_OVERLAP"));

        let mut config = live();
        config.placements[0].position_mm.x_mm = 13_000;
        assert!(has(&validate_configuration(&config).expect("receipt serializes"), "PLACEMENT_OUT_OF_BOUNDS"));

        let mut config = live();
        config.placements[0].position_mm.y_mm = 600;
        assert!(has(&validate_configuration(&config).expect("receipt serializes"), "RESERVED_REGION_CONFLICT"));
    }

    #[test]
    fn connection_mismatch_and_overload_fail_closed() {
        let mut config = live();
        config.connections[0].medium = "data_network".to_owned();
        let receipt = validate_configuration(&config).expect("receipt serializes");
        assert_eq!(receipt.body.overall_disposition, "rejected");
        assert!(has(&receipt, "CONNECTION_MEDIUM_MISMATCH"));

        let mut config = live();
        config.platform_services.iter_mut().find(|s| s.service_id == "AC-BUS")
            .expect("AC bus exists").capacity.value = Some(1.0);
        assert!(has(&validate_configuration(&config).expect("receipt serializes"), "SERVICE_CAPACITY_EXCEEDED"));
    }

    #[test]
    fn unknown_mass_prevents_weight_closure() {
        let mut config = live();
        let module = config.module_catalog.iter_mut().find(|m| m.module_id == "SLEEP-POD")
            .expect("sleep pod exists");
        module.mass_properties.status = "unknown".to_owned();
        module.mass_properties.mass_kg = None;
        module.mass_properties.local_cg_mm = CgMm { x_mm: None, y_mm: None, z_mm: None };
        let ledger = validate_configuration(&config).expect("receipt serializes").body.weight_ledger;
        assert_eq!(ledger.disposition, "indeterminate");
        assert!(ledger.total_mass_kg.is_none());
        assert!(ledger.mass_weighted_cg_mm.is_none());
    }

    #[test]
    fn module_revision_mismatch_and_duplicate_port_are_detected() {
        let mut config = live();
        config.placements[0].module_revision = 2;
        assert!(has(&validate_configuration(&config).expect("receipt serializes"), "MODULE_REVISION_MISMATCH"));

        let mut config = live();
        let mut duplicate = config.connections[0].clone();
        duplicate.connection_id = "CONN-DUPLICATE-TEST".to_owned();
        config.connections.push(duplicate);
        assert!(has(&validate_configuration(&config).expect("receipt serializes"), "PORT_CONNECTED_MULTIPLE_TIMES"));
    }
}
