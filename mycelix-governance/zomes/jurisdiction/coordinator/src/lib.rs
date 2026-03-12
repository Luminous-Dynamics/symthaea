//! Jurisdiction Coordinator Zome
//! Business logic for governance jurisdiction constraints
//!
//! Provides CRUD operations for `JurisdictionConstraintEntry` records stored on
//! the DHT, with anchor-based O(1) lookups by zone_id, regulatory tag, and
//! authority DID, plus a point-in-polygon spatial query.

use hdk::prelude::*;
use jurisdiction_integrity::*;

// ============================================================================
// REAL-TIME SIGNALS
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "payload")]
pub enum JurisdictionSignal {
    ZoneCreated {
        zone_id: String,
        regulatory_tags: Vec<String>,
    },
    ZoneUpdated {
        zone_id: String,
        version: u32,
    },
}

// ============================================================================
// INPUT TYPES
// ============================================================================

/// Input for point-in-polygon spatial queries.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PointQuery {
    pub lat: f64,
    pub lon: f64,
}

/// Input for updating an existing jurisdiction zone.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateJurisdictionInput {
    /// The zone_id of the jurisdiction to update (used for lookup).
    pub zone_id: String,
    /// The updated jurisdiction constraint data.
    pub entry: JurisdictionConstraintEntry,
}

// ============================================================================
// HELPERS
// ============================================================================

/// Compute the entry hash for an anchor string.
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Ray-casting algorithm for point-in-polygon test.
///
/// Polygon vertices are `(lat, lon)` pairs. Returns `true` if the point
/// `(lat, lon)` lies inside the polygon.
fn point_in_polygon(lat: f64, lon: f64, polygon: &[(f64, f64)]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (yi, xi) = polygon[i];
        let (yj, xj) = polygon[j];
        if ((yi > lon) != (yj > lon)) && (lat < (xj - xi) * (lon - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Resolve links from an anchor to `JurisdictionRecord` records.
fn records_from_anchor(
    anchor_str: &str,
    link_type: LinkTypes,
) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(anchor_str)?, link_type)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

// ============================================================================
// EXTERN FUNCTIONS
// ============================================================================

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    // Pre-create the active_zones anchor so queries never fail on empty DNA
    let anchor = Anchor("active_zones".to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    Ok(InitCallbackResult::Pass)
}

/// Create a new jurisdiction zone.
///
/// Validates input, creates the DHT entry, and establishes anchor links for
/// zone_id lookup, regulatory tag queries, active zones listing, and
/// (optionally) authority DID lookup.
#[hdk_extern]
pub fn create_jurisdiction(entry: JurisdictionConstraintEntry) -> ExternResult<Record> {
    // Input validation — coordinator-level length/finiteness checks
    if entry.zone_id.is_empty() || entry.zone_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Zone ID must be 1-256 characters".into()
        )));
    }
    if !entry.enforcement_risk.is_finite() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "enforcement_risk must be finite".into()
        )));
    }
    for (lat, lon) in &entry.zone_polygon {
        if !lat.is_finite() || !lon.is_finite() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Polygon coordinates must be finite".into()
            )));
        }
    }
    if entry.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be 4096 characters or fewer".into()
        )));
    }

    let signal_zone_id = entry.zone_id.clone();
    let signal_tags = entry.regulatory_tags.clone();
    let authority = entry.authority_did.clone();
    let tags = entry.regulatory_tags.clone();

    let record = JurisdictionRecord { entry, version: 1 };
    let action_hash = create_entry(&EntryTypes::Jurisdiction(record))?;

    // --- Anchor links ---

    // O(1) lookup by zone_id
    let zid_anchor = format!("zone:{}", signal_zone_id);
    create_entry(&EntryTypes::Anchor(Anchor(zid_anchor.clone())))?;
    create_link(
        anchor_hash(&zid_anchor)?,
        action_hash.clone(),
        LinkTypes::ZoneById,
        (),
    )?;

    // Regulatory tag links
    for tag in &tags {
        let tag_anchor = format!("tag:{}", tag);
        create_entry(&EntryTypes::Anchor(Anchor(tag_anchor.clone())))?;
        create_link(
            anchor_hash(&tag_anchor)?,
            action_hash.clone(),
            LinkTypes::TagToZone,
            (),
        )?;
    }

    // Active zones link
    create_entry(&EntryTypes::Anchor(Anchor("active_zones".to_string())))?;
    create_link(
        anchor_hash("active_zones")?,
        action_hash.clone(),
        LinkTypes::ActiveZones,
        (),
    )?;

    // Authority DID link (if present)
    if let Some(ref did) = authority {
        let auth_anchor = format!("authority:{}", did);
        create_entry(&EntryTypes::Anchor(Anchor(auth_anchor.clone())))?;
        create_link(
            anchor_hash(&auth_anchor)?,
            action_hash.clone(),
            LinkTypes::AuthorityToZone,
            (),
        )?;
    }

    // Emit signal
    let _ = emit_signal(&JurisdictionSignal::ZoneCreated {
        zone_id: signal_zone_id,
        regulatory_tags: signal_tags,
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created jurisdiction".into()
    )))
}

/// Get a jurisdiction zone by its zone_id (O(1) anchor-link lookup).
#[hdk_extern]
pub fn get_jurisdiction(zone_id: String) -> ExternResult<Option<Record>> {
    if zone_id.is_empty() || zone_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Zone ID must be 1-256 characters".into()
        )));
    }

    let zid_anchor = format!("zone:{}", zone_id);
    if let Ok(entry_hash) = anchor_hash(&zid_anchor) {
        if let Ok(links) = get_links(
            LinkQuery::try_new(entry_hash, LinkTypes::ZoneById)?,
            GetStrategy::default(),
        ) {
            // Take the most recent link (latest create wins)
            if let Some(link) = links.into_iter().max_by_key(|l| l.timestamp) {
                if let Ok(ah) = ActionHash::try_from(link.target) {
                    if let Some(record) = get(ah, GetOptions::default())? {
                        return Ok(Some(record));
                    }
                }
            }
        }
    }

    Ok(None)
}

/// List all active jurisdiction zones.
#[hdk_extern]
pub fn list_active_jurisdictions(_: ()) -> ExternResult<Vec<Record>> {
    records_from_anchor("active_zones", LinkTypes::ActiveZones)
}

/// List all jurisdiction zones tagged with a given regulatory tag.
#[hdk_extern]
pub fn list_by_regulatory_tag(tag: String) -> ExternResult<Vec<Record>> {
    if tag.is_empty() || tag.len() > 128 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Tag must be 1-128 characters".into()
        )));
    }

    let tag_anchor = format!("tag:{}", tag);
    records_from_anchor(&tag_anchor, LinkTypes::TagToZone)
}

/// Find all jurisdiction zones whose polygon contains a given point.
///
/// Uses the ray-casting algorithm for point-in-polygon testing.
/// Only zones with non-empty polygons are candidates.
#[hdk_extern]
pub fn zones_containing_point(input: PointQuery) -> ExternResult<Vec<Record>> {
    if !input.lat.is_finite() || !input.lon.is_finite() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Latitude and longitude must be finite".into()
        )));
    }
    if input.lat < -90.0 || input.lat > 90.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Latitude must be in [-90, 90]".into()
        )));
    }
    if input.lon < -180.0 || input.lon > 180.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Longitude must be in [-180, 180]".into()
        )));
    }

    let all_zones = records_from_anchor("active_zones", LinkTypes::ActiveZones)?;

    let mut matching = Vec::new();
    for record in all_zones {
        if let Some(jr) = record
            .entry()
            .to_app_option::<JurisdictionRecord>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if !jr.entry.zone_polygon.is_empty()
                && point_in_polygon(input.lat, input.lon, &jr.entry.zone_polygon)
            {
                matching.push(record);
            }
        }
    }

    Ok(matching)
}

/// Update an existing jurisdiction zone.
///
/// Looks up the current record by zone_id, increments the version, and
/// creates a DHT update. Re-links regulatory tags and authority DID if changed.
#[hdk_extern]
pub fn update_jurisdiction(input: UpdateJurisdictionInput) -> ExternResult<Record> {
    // Input validation
    if input.zone_id.is_empty() || input.zone_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Zone ID must be 1-256 characters".into()
        )));
    }
    if !input.entry.enforcement_risk.is_finite() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "enforcement_risk must be finite".into()
        )));
    }
    for (lat, lon) in &input.entry.zone_polygon {
        if !lat.is_finite() || !lon.is_finite() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Polygon coordinates must be finite".into()
            )));
        }
    }

    // zone_id in the input must match the entry's zone_id
    if input.entry.zone_id != input.zone_id {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "zone_id in entry must match the lookup zone_id".into()
        )));
    }

    // Fetch current record
    let current_record = get_jurisdiction(input.zone_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Jurisdiction zone not found".into())
    ))?;

    let current: JurisdictionRecord = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid jurisdiction record".into()
        )))?;

    let old_tags = current.entry.regulatory_tags.clone();
    let old_authority = current.entry.authority_did.clone();
    let new_version = current.version + 1;

    let updated_record = JurisdictionRecord {
        entry: input.entry,
        version: new_version,
    };

    let new_tags = updated_record.entry.regulatory_tags.clone();
    let new_authority = updated_record.entry.authority_did.clone();
    let signal_zone_id = updated_record.entry.zone_id.clone();

    let action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::Jurisdiction(updated_record),
    )?;

    // Update zone_id link to point to the new action
    let zid_anchor = format!("zone:{}", signal_zone_id);
    create_entry(&EntryTypes::Anchor(Anchor(zid_anchor.clone())))?;
    create_link(
        anchor_hash(&zid_anchor)?,
        action_hash.clone(),
        LinkTypes::ZoneById,
        (),
    )?;

    // Update active_zones link
    create_entry(&EntryTypes::Anchor(Anchor("active_zones".to_string())))?;
    create_link(
        anchor_hash("active_zones")?,
        action_hash.clone(),
        LinkTypes::ActiveZones,
        (),
    )?;

    // Add links for any new regulatory tags
    for tag in &new_tags {
        if !old_tags.contains(tag) {
            let tag_anchor = format!("tag:{}", tag);
            create_entry(&EntryTypes::Anchor(Anchor(tag_anchor.clone())))?;
            create_link(
                anchor_hash(&tag_anchor)?,
                action_hash.clone(),
                LinkTypes::TagToZone,
                (),
            )?;
        }
    }

    // Add authority link if changed
    if new_authority != old_authority {
        if let Some(ref did) = new_authority {
            let auth_anchor = format!("authority:{}", did);
            create_entry(&EntryTypes::Anchor(Anchor(auth_anchor.clone())))?;
            create_link(
                anchor_hash(&auth_anchor)?,
                action_hash.clone(),
                LinkTypes::AuthorityToZone,
                (),
            )?;
        }
    }

    // Emit signal
    let _ = emit_signal(&JurisdictionSignal::ZoneUpdated {
        zone_id: signal_zone_id,
        version: new_version,
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated jurisdiction".into()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_in_polygon_inside() {
        // Simple square: (0,0), (10,0), (10,10), (0,10)
        let polygon = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(point_in_polygon(5.0, 5.0, &polygon));
    }

    #[test]
    fn test_point_in_polygon_outside() {
        let polygon = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(!point_in_polygon(15.0, 5.0, &polygon));
    }

    #[test]
    fn test_point_in_polygon_degenerate() {
        // Less than 3 vertices — always false
        assert!(!point_in_polygon(0.0, 0.0, &[]));
        assert!(!point_in_polygon(0.0, 0.0, &[(0.0, 0.0)]));
        assert!(!point_in_polygon(0.0, 0.0, &[(0.0, 0.0), (1.0, 1.0)]));
    }

    #[test]
    fn test_point_in_polygon_triangle() {
        // Triangle with vertices at (0,0), (10,5), (0,10)
        // The algorithm treats polygon[i] = (yi, xi) where yi is the
        // "vertical" axis and xi is the "horizontal" axis.
        let triangle = vec![(0.0, 0.0), (10.0, 5.0), (0.0, 10.0)];
        // Centroid ~(3.3, 5.0) — inside
        assert!(point_in_polygon(3.0, 5.0, &triangle));
        // Far outside
        assert!(!point_in_polygon(20.0, 20.0, &triangle));
    }

    #[test]
    fn test_point_in_polygon_geo_box() {
        // Axis-aligned box from (0, 0) to (50, 50) in (lat, lon) space
        // The ray-casting algorithm works correctly on axis-aligned rectangles
        // when vertices are ordered consistently.
        let geo_box = vec![
            (0.0, 0.0),
            (0.0, 50.0),
            (50.0, 50.0),
            (50.0, 0.0),
        ];
        // Inside
        assert!(point_in_polygon(25.0, 25.0, &geo_box));
        // Outside
        assert!(!point_in_polygon(60.0, 25.0, &geo_box));
        assert!(!point_in_polygon(25.0, 60.0, &geo_box));
        assert!(!point_in_polygon(-5.0, 25.0, &geo_box));
    }
}
