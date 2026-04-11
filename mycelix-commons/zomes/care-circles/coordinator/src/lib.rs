// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Circles Coordinator Zome
//! Business logic for care circle creation, membership, and discovery.

use care_circles_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::civic_requirement_basic;
use mycelix_zome_helpers::records_from_links;


fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    anchor_hash(anchor_str)
}

/// Create a new care circle. The creator automatically becomes an Organizer member.
#[hdk_extern]
pub fn create_circle(circle: CareCircle) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "create_circle")?;
    let action_hash = create_entry(&EntryTypes::CareCircle(circle.clone()))?;

    // Link to all circles
    let all_anchor = ensure_anchor("all_circles")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllCircles, ())?;

    // Link by type
    let type_anchor = ensure_anchor(&format!("circle_type:{}", circle.circle_type.anchor_key()))?;
    create_link(
        type_anchor,
        action_hash.clone(),
        LinkTypes::TypeToCircle,
        (),
    )?;

    // Link creator to circle
    let creator_anchor = ensure_anchor(&format!("agent_created_circles:{}", circle.created_by))?;
    create_link(
        creator_anchor,
        action_hash.clone(),
        LinkTypes::AgentToCreatedCircle,
        (),
    )?;

    // Auto-join creator as Organizer
    let now = sys_time()?;
    let membership = CircleMembership {
        circle_hash: action_hash.clone(),
        member: circle.created_by.clone(),
        role: MemberRole::Organizer,
        joined_at: now,
        active: true,
    };
    let membership_hash = create_entry(&EntryTypes::CircleMembership(membership))?;

    // Link circle to membership
    let circle_members_anchor = ensure_anchor(&format!("circle_members:{}", action_hash))?;
    create_link(
        circle_members_anchor,
        membership_hash.clone(),
        LinkTypes::CircleToMembership,
        (),
    )?;

    // Link agent to membership
    let agent_membership_anchor =
        ensure_anchor(&format!("agent_memberships:{}", circle.created_by))?;
    create_link(
        agent_membership_anchor,
        membership_hash,
        LinkTypes::AgentToMembership,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created circle".into()
    )))
}

/// Input for joining a circle
#[derive(Serialize, Deserialize, Debug)]
pub struct JoinCircleInput {
    pub circle_hash: ActionHash,
    pub role: MemberRole,
}

/// Join an existing care circle
#[hdk_extern]
pub fn join_circle(input: JoinCircleInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "join_circle")?;
    let caller = agent_info()?.agent_initial_pubkey;

    // Verify circle exists
    let circle_record = get(input.circle_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Circle not found".into())),
    )?;

    let circle: CareCircle = circle_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid circle entry".into()
        )))?;

    if !circle.active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot join an inactive circle".into()
        )));
    }

    // Check member count
    let circle_members_anchor = anchor_hash(&format!("circle_members:{}", input.circle_hash))?;
    let existing_links = get_links(
        LinkQuery::try_new(circle_members_anchor.clone(), LinkTypes::CircleToMembership)?,
        GetStrategy::default(),
    )?;

    // Count active members
    let mut active_count = 0u32;
    for link in &existing_links {
        let ah = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(rec) = get(ah, GetOptions::default())? {
            if let Some(m) = rec
                .entry()
                .to_app_option::<CircleMembership>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if m.active {
                    if m.member == caller {
                        return Err(wasm_error!(WasmErrorInner::Guest(
                            "Already a member of this circle".into()
                        )));
                    }
                    active_count += 1;
                }
            }
        }
    }

    if active_count >= circle.max_members {
        return Err(wasm_error!(WasmErrorInner::Guest("Circle is full".into())));
    }

    // Cannot self-assign Organizer role
    let role = if input.role == MemberRole::Organizer {
        MemberRole::Member
    } else {
        input.role
    };

    let now = sys_time()?;
    let membership = CircleMembership {
        circle_hash: input.circle_hash.clone(),
        member: caller.clone(),
        role,
        joined_at: now,
        active: true,
    };

    let membership_hash = create_entry(&EntryTypes::CircleMembership(membership))?;

    let cm_anchor = ensure_anchor(&format!("circle_members:{}", input.circle_hash))?;
    create_link(
        cm_anchor,
        membership_hash.clone(),
        LinkTypes::CircleToMembership,
        (),
    )?;

    let am_anchor = ensure_anchor(&format!("agent_memberships:{}", caller))?;
    create_link(
        am_anchor,
        membership_hash.clone(),
        LinkTypes::AgentToMembership,
        (),
    )?;

    get(membership_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created membership".into()
    )))
}

/// Leave a care circle by deactivating membership
#[hdk_extern]
pub fn leave_circle(circle_hash: ActionHash) -> ExternResult<bool> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "leave_circle")?;
    let caller = agent_info()?.agent_initial_pubkey;

    let cm_anchor = anchor_hash(&format!("circle_members:{}", circle_hash))?;
    let links = get_links(
        LinkQuery::try_new(cm_anchor, LinkTypes::CircleToMembership)?,
        GetStrategy::default(),
    )?;

    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(rec) = get(ah.clone(), GetOptions::default())? {
            if let Some(m) = rec
                .entry()
                .to_app_option::<CircleMembership>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if m.member == caller && m.active {
                    let updated = CircleMembership {
                        circle_hash: m.circle_hash,
                        member: m.member,
                        role: m.role,
                        joined_at: m.joined_at,
                        active: false,
                    };
                    update_entry(ah, &EntryTypes::CircleMembership(updated))?;
                    return Ok(true);
                }
            }
        }
    }

    Ok(false)
}

/// Get circles the calling agent is a member of
#[hdk_extern]
pub fn get_my_circles(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;
    let am_anchor = anchor_hash(&format!("agent_memberships:{}", caller))?;
    let links = get_links(
        LinkQuery::try_new(am_anchor, LinkTypes::AgentToMembership)?,
        GetStrategy::default(),
    )?;

    let mut circles = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(rec) = get(ah, GetOptions::default())? {
            if let Some(m) = rec
                .entry()
                .to_app_option::<CircleMembership>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if m.active {
                    if let Some(circle_rec) = get(m.circle_hash, GetOptions::default())? {
                        circles.push(circle_rec);
                    }
                }
            }
        }
    }

    Ok(circles)
}

/// Get all members of a circle
#[hdk_extern]
pub fn get_circle_members(circle_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let cm_anchor = anchor_hash(&format!("circle_members:{}", circle_hash))?;
    let links = get_links(
        LinkQuery::try_new(cm_anchor, LinkTypes::CircleToMembership)?,
        GetStrategy::default(),
    )?;

    let mut members = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(rec) = get(ah, GetOptions::default())? {
            if let Some(m) = rec
                .entry()
                .to_app_option::<CircleMembership>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if m.active {
                    members.push(rec);
                }
            }
        }
    }

    Ok(members)
}

/// Get all circles
#[hdk_extern]
pub fn get_all_circles(_: ()) -> ExternResult<Vec<Record>> {
    let all_anchor = anchor_hash("all_circles")?;
    let links = get_links(
        LinkQuery::try_new(all_anchor, LinkTypes::AllCircles)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get circles by type
#[hdk_extern]
pub fn get_circles_by_type(circle_type: CircleType) -> ExternResult<Vec<Record>> {
    let type_anchor = anchor_hash(&format!("circle_type:{}", circle_type.anchor_key()))?;
    let links = get_links(
        LinkQuery::try_new(type_anchor, LinkTypes::TypeToCircle)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ── TEND MUTUAL CREDIT ─────────────────────────────────────────────────

/// Best-effort cross-cluster call to TEND zome to record the exchange.
/// Returns the TEND exchange ID if successful, or None if the finance
/// cluster is unreachable (the local exchange record is still created).
fn bridge_tend_exchange(
    receiver_did: &str,
    hours: f32,
    description: &str,
) -> Option<String> {
    #[derive(Serialize, Debug)]
    struct TendPayload {
        receiver_did: String,
        hours: f32,
        service_description: String,
        service_category: String,
        cultural_alias: Option<String>,
        dao_did: String,
        service_date: Option<Timestamp>,
    }
    let payload = TendPayload {
        receiver_did: receiver_did.to_string(),
        hours,
        service_description: description.to_string(),
        service_category: "CareCircle".to_string(),
        cultural_alias: None,
        dao_did: "did:mycelix:commons".to_string(),
        service_date: None,
    };
    match call(
        CallTargetCell::OtherRole("finance".into()),
        ZomeName::from("tend"),
        FunctionName::from("record_exchange"),
        None,
        payload,
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            #[derive(Deserialize, Debug)]
            struct TendResult { id: String }
            result.decode::<TendResult>().ok().map(|r| r.id)
        }
        _ => None,
    }
}

/// Input for recording a care exchange within a circle.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RecordCareExchangeInput {
    pub circle_hash: ActionHash,
    pub receiver: AgentPubKey,
    pub hours: f32,
    pub service_description: String,
}

/// Aggregate TEND balance for a circle.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CircleTendBalance {
    pub circle_hash: ActionHash,
    pub total_exchanges: u32,
    pub total_hours: f32,
}

/// Record a care exchange within a circle. Caller is the provider.
/// Makes a best-effort cross-cluster call to the TEND zome.
#[hdk_extern]
pub fn record_care_exchange(input: RecordCareExchangeInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "record_care_exchange")?;
    let provider = agent_info()?.agent_initial_pubkey;
    if provider == input.receiver {
        return Err(wasm_error!("Cannot record exchange with yourself"));
    }
    let receiver_did = format!("did:mycelix:{}", input.receiver);
    let tend_id = bridge_tend_exchange(&receiver_did, input.hours, &input.service_description);
    let status = if tend_id.is_some() { CircleTendStatus::Confirmed } else { CircleTendStatus::Proposed };
    let exchange = CircleTendExchange {
        circle_hash: input.circle_hash.clone(),
        provider,
        receiver: input.receiver,
        hours: input.hours,
        service_description: input.service_description,
        tend_exchange_id: tend_id,
        status,
        created_at: sys_time()?,
    };
    let action_hash = create_entry(&EntryTypes::CircleTendExchange(exchange))?;
    create_link(input.circle_hash, action_hash.clone(), LinkTypes::CircleToTendExchange, ())?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!("Failed to get created exchange record"))
}

/// Get the aggregate TEND balance for a circle.
#[hdk_extern]
pub fn get_circle_tend_balance(circle_hash: ActionHash) -> ExternResult<CircleTendBalance> {
    let links = get_links(
        LinkQuery::try_new(circle_hash.clone(), LinkTypes::CircleToTendExchange)?,
        GetStrategy::default(),
    )?;
    let mut total_exchanges = 0u32;
    let mut total_hours = 0.0f32;
    for link in links {
        let target = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!("Invalid link target"))?;
        if let Some(record) = get(target, GetOptions::default())? {
            if let Some(exchange) = record.entry().to_app_option::<CircleTendExchange>().ok().flatten() {
                if exchange.status == CircleTendStatus::Confirmed {
                    total_exchanges += 1;
                    total_hours += exchange.hours;
                }
            }
        }
    }
    Ok(CircleTendBalance { circle_hash, total_exchanges, total_hours })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ──────────────────────────────────────────────────────────

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    // ── JoinCircleInput serde roundtrip ─────────────────────────────────

    #[test]
    fn join_circle_input_serde_roundtrip() {
        let input = JoinCircleInput {
            circle_hash: fake_action_hash(),
            role: MemberRole::Member,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: JoinCircleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.circle_hash, input.circle_hash);
        assert_eq!(decoded.role, MemberRole::Member);
    }

    #[test]
    fn join_circle_input_serde_organizer_role() {
        let input = JoinCircleInput {
            circle_hash: fake_action_hash(),
            role: MemberRole::Organizer,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: JoinCircleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.role, MemberRole::Organizer);
    }

    #[test]
    fn join_circle_input_serde_observer_role() {
        let input = JoinCircleInput {
            circle_hash: fake_action_hash(),
            role: MemberRole::Observer,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: JoinCircleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.role, MemberRole::Observer);
    }

    // ── MemberRole serde roundtrip (all variants) ───────────────────────

    #[test]
    fn member_role_serde_all_variants() {
        let roles = vec![
            MemberRole::Organizer,
            MemberRole::Member,
            MemberRole::Observer,
        ];
        for role in &roles {
            let json = serde_json::to_string(role).unwrap();
            let decoded: MemberRole = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, role);
        }
    }

    // ── CircleType serde roundtrip (all variants) ───────────────────────

    #[test]
    fn circle_type_serde_all_variants() {
        let types = vec![
            CircleType::Neighborhood,
            CircleType::Workplace,
            CircleType::Faith,
            CircleType::Family,
            CircleType::School,
            CircleType::Custom("Book Club".to_string()),
        ];
        for ct in &types {
            let json = serde_json::to_string(ct).unwrap();
            let decoded: CircleType = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, ct);
        }
    }

    // ── CircleType::anchor_key pure function tests ──────────────────────

    #[test]
    fn circle_type_anchor_key_known_variants() {
        assert_eq!(CircleType::Neighborhood.anchor_key(), "neighborhood");
        assert_eq!(CircleType::Workplace.anchor_key(), "workplace");
        assert_eq!(CircleType::Faith.anchor_key(), "faith");
        assert_eq!(CircleType::Family.anchor_key(), "family");
        assert_eq!(CircleType::School.anchor_key(), "school");
    }

    #[test]
    fn circle_type_anchor_key_custom_lowercases_and_replaces_spaces() {
        let ct = CircleType::Custom("Book Club".to_string());
        assert_eq!(ct.anchor_key(), "custom_book_club");
    }

    #[test]
    fn circle_type_anchor_key_custom_already_lowercase() {
        let ct = CircleType::Custom("garden".to_string());
        assert_eq!(ct.anchor_key(), "custom_garden");
    }

    #[test]
    fn circle_type_anchor_key_custom_empty_string() {
        let ct = CircleType::Custom(String::new());
        assert_eq!(ct.anchor_key(), "custom_");
    }

    // ── CareCircle serde roundtrip ──────────────────────────────────────

    #[test]
    fn care_circle_serde_roundtrip() {
        let circle = CareCircle {
            name: "Helpers United".to_string(),
            description: "A neighborhood support circle".to_string(),
            location: "Downtown".to_string(),
            max_members: 20,
            created_by: fake_agent(),
            circle_type: CircleType::Neighborhood,
            active: true,
            created_at: Timestamp::from_micros(1000),
        };
        let json = serde_json::to_string(&circle).unwrap();
        let decoded: CareCircle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, circle);
    }

    // ── CircleMembership serde roundtrip ────────────────────────────────

    #[test]
    fn circle_membership_serde_roundtrip() {
        let membership = CircleMembership {
            circle_hash: fake_action_hash(),
            member: fake_agent(),
            role: MemberRole::Organizer,
            joined_at: Timestamp::from_micros(5000),
            active: true,
        };
        let json = serde_json::to_string(&membership).unwrap();
        let decoded: CircleMembership = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, membership);
    }

    #[test]
    fn circle_membership_inactive_serde_roundtrip() {
        let membership = CircleMembership {
            circle_hash: fake_action_hash(),
            member: fake_agent(),
            role: MemberRole::Observer,
            joined_at: Timestamp::from_micros(0),
            active: false,
        };
        let json = serde_json::to_string(&membership).unwrap();
        let decoded: CircleMembership = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.active, false);
        assert_eq!(decoded.role, MemberRole::Observer);
    }

    // ── CareCircle inactive serde roundtrip ──────────────────────────────

    #[test]
    fn care_circle_inactive_serde_roundtrip() {
        let circle = CareCircle {
            name: "Closed Circle".to_string(),
            description: "A deactivated circle".to_string(),
            location: "Remote".to_string(),
            max_members: 5,
            created_by: fake_agent(),
            circle_type: CircleType::Family,
            active: false,
            created_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&circle).unwrap();
        let decoded: CareCircle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.active, false);
        assert_eq!(decoded.name, "Closed Circle");
    }

    // ── CareCircle boundary conditions ───────────────────────────────────

    #[test]
    fn care_circle_max_members_boundary_min() {
        let circle = CareCircle {
            name: "Tiny Circle".to_string(),
            description: "Just two people".to_string(),
            location: "Anywhere".to_string(),
            max_members: 2,
            created_by: fake_agent(),
            circle_type: CircleType::Custom("Pair".to_string()),
            active: true,
            created_at: Timestamp::from_micros(100),
        };
        let json = serde_json::to_string(&circle).unwrap();
        let decoded: CareCircle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.max_members, 2);
    }

    #[test]
    fn care_circle_max_members_boundary_max() {
        let circle = CareCircle {
            name: "Huge Circle".to_string(),
            description: "Maximum capacity".to_string(),
            location: "City-wide".to_string(),
            max_members: 500,
            created_by: fake_agent(),
            circle_type: CircleType::Neighborhood,
            active: true,
            created_at: Timestamp::from_micros(100),
        };
        let json = serde_json::to_string(&circle).unwrap();
        let decoded: CareCircle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.max_members, 500);
    }

    #[test]
    fn care_circle_empty_strings_serde() {
        let circle = CareCircle {
            name: "X".to_string(),
            description: "Y".to_string(),
            location: "Z".to_string(),
            max_members: 3,
            created_by: fake_agent(),
            circle_type: CircleType::Custom(String::new()),
            active: true,
            created_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&circle).unwrap();
        let decoded: CareCircle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.circle_type, CircleType::Custom(String::new()));
    }

    // ── CircleType custom edge cases ─────────────────────────────────────

    #[test]
    fn circle_type_anchor_key_custom_special_characters() {
        let ct = CircleType::Custom("Art & Craft".to_string());
        let key = ct.anchor_key();
        assert!(key.starts_with("custom_"));
        assert!(key.contains("art"));
    }

    #[test]
    fn circle_type_anchor_key_custom_unicode() {
        let ct = CircleType::Custom("Cafe Reseau".to_string());
        let key = ct.anchor_key();
        assert_eq!(key, "custom_cafe_reseau");
    }

    // ── Membership role transitions ──────────────────────────────────────

    #[test]
    fn membership_role_change_member_to_organizer_serde() {
        let membership = CircleMembership {
            circle_hash: fake_action_hash(),
            member: fake_agent(),
            role: MemberRole::Organizer,
            joined_at: Timestamp::from_micros(1000),
            active: true,
        };
        let json = serde_json::to_string(&membership).unwrap();
        let decoded: CircleMembership = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.role, MemberRole::Organizer);
        assert!(decoded.active);
    }

    #[test]
    fn membership_role_change_organizer_to_observer_serde() {
        let membership = CircleMembership {
            circle_hash: fake_action_hash(),
            member: fake_agent(),
            role: MemberRole::Observer,
            joined_at: Timestamp::from_micros(500),
            active: true,
        };
        let json = serde_json::to_string(&membership).unwrap();
        let decoded: CircleMembership = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.role, MemberRole::Observer);
    }

    // ── Membership deactivation (leave circle) ──────────────────────────

    #[test]
    fn membership_deactivation_preserves_role() {
        let membership = CircleMembership {
            circle_hash: fake_action_hash(),
            member: fake_agent(),
            role: MemberRole::Organizer,
            joined_at: Timestamp::from_micros(1000),
            active: false,
        };
        let json = serde_json::to_string(&membership).unwrap();
        let decoded: CircleMembership = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.role, MemberRole::Organizer);
        assert!(!decoded.active);
    }

    #[test]
    fn membership_deactivation_preserves_joined_at() {
        let membership = CircleMembership {
            circle_hash: fake_action_hash(),
            member: fake_agent(),
            role: MemberRole::Member,
            joined_at: Timestamp::from_micros(42_000),
            active: false,
        };
        let json = serde_json::to_string(&membership).unwrap();
        let decoded: CircleMembership = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.joined_at, Timestamp::from_micros(42_000));
        assert!(!decoded.active);
    }

    // ── CareCircle with all CircleType variants ──────────────────────────

    #[test]
    fn care_circle_serde_all_circle_types() {
        let types = vec![
            CircleType::Neighborhood,
            CircleType::Workplace,
            CircleType::Faith,
            CircleType::Family,
            CircleType::School,
            CircleType::Custom("Quilting Bee".to_string()),
        ];
        for ct in types {
            let circle = CareCircle {
                name: "Test".to_string(),
                description: "Test desc".to_string(),
                location: "Test loc".to_string(),
                max_members: 10,
                created_by: fake_agent(),
                circle_type: ct.clone(),
                active: true,
                created_at: Timestamp::from_micros(0),
            };
            let json = serde_json::to_string(&circle).unwrap();
            let decoded: CareCircle = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.circle_type, ct);
        }
    }

    // ── JoinCircleInput with different agent hashes ──────────────────────

    #[test]
    fn join_circle_input_distinct_hashes() {
        let hash_a = ActionHash::from_raw_36(vec![0xAA; 36]);
        let hash_b = ActionHash::from_raw_36(vec![0xBB; 36]);
        let input_a = JoinCircleInput {
            circle_hash: hash_a.clone(),
            role: MemberRole::Member,
        };
        let input_b = JoinCircleInput {
            circle_hash: hash_b.clone(),
            role: MemberRole::Observer,
        };
        let json_a = serde_json::to_string(&input_a).unwrap();
        let json_b = serde_json::to_string(&input_b).unwrap();
        assert_ne!(json_a, json_b);
        let decoded_a: JoinCircleInput = serde_json::from_str(&json_a).unwrap();
        let decoded_b: JoinCircleInput = serde_json::from_str(&json_b).unwrap();
        assert_eq!(decoded_a.circle_hash, hash_a);
        assert_eq!(decoded_b.circle_hash, hash_b);
    }

    // ── CircleMembership different agents ─────────────────────────────────

    #[test]
    fn circle_membership_different_agents() {
        let agent_a = AgentPubKey::from_raw_36(vec![0x01; 36]);
        let agent_b = AgentPubKey::from_raw_36(vec![0x02; 36]);
        let mem_a = CircleMembership {
            circle_hash: fake_action_hash(),
            member: agent_a.clone(),
            role: MemberRole::Organizer,
            joined_at: Timestamp::from_micros(1000),
            active: true,
        };
        let mem_b = CircleMembership {
            circle_hash: fake_action_hash(),
            member: agent_b.clone(),
            role: MemberRole::Member,
            joined_at: Timestamp::from_micros(2000),
            active: true,
        };
        let json_a = serde_json::to_string(&mem_a).unwrap();
        let json_b = serde_json::to_string(&mem_b).unwrap();
        let decoded_a: CircleMembership = serde_json::from_str(&json_a).unwrap();
        let decoded_b: CircleMembership = serde_json::from_str(&json_b).unwrap();
        assert_ne!(decoded_a.member, decoded_b.member);
        assert_eq!(decoded_a.member, agent_a);
        assert_eq!(decoded_b.member, agent_b);
    }
}
