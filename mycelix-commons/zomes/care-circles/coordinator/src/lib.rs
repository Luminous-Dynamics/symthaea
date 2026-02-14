//! Circles Coordinator Zome
//! Business logic for care circle creation, membership, and discovery.

use care_circles_integrity::*;
use hdk::prelude::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    anchor_hash(anchor_str)
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
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

/// Create a new care circle. The creator automatically becomes an Organizer member.
#[hdk_extern]
pub fn create_circle(circle: CareCircle) -> ExternResult<Record> {
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
}
