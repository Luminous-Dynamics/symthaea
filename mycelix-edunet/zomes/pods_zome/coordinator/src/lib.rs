// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Learning Pods Coordinator Zome
//!
//! Implements the business logic for the revolutionary Learning Pods feature.
//!
//! ## Core Functionality
//!
//! - **Pod Lifecycle**: Create, activate, pause, complete, archive pods
//! - **Membership Management**: Invite, accept, leave, remove members
//! - **Progress Tracking**: Share progress within pods (privacy-preserving)
//! - **Discussions**: Threaded discussions for peer learning
//! - **Achievements**: Collective achievements and badges
//! - **Challenges**: Learning challenges within pods

use hdk::prelude::*;
use hdk::prelude::HdkPathExt;
use pods_integrity::{
    EntryTypes, LinkTypes, LearningPod, PodMembership, PodProgress,
    PodDiscussion, DiscussionReply, PodAchievement, PodChallenge,
    PodStatus, MembershipStatus, MemberRole,
    PeerTutoringSession, TutoringSessionStatus,
};

// Helper function to ensure a path exists and return its entry hash
fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed_path = path.typed(link_type)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

// Helper function to convert timestamp to i64 (microseconds)
fn timestamp_to_i64(ts: Timestamp) -> i64 {
    ts.as_micros()
}

// ============== Pod Lifecycle Functions ==============

/// Create a new learning pod
#[hdk_extern]
pub fn create_pod(pod: LearningPod) -> ExternResult<ActionHash> {
    // Create the pod entry
    let action_hash = create_entry(EntryTypes::LearningPod(pod.clone()))?;

    // Create path anchor for listing all pods
    let path = Path::from("all_pods");
    let path_hash = ensure_path(path, LinkTypes::AllPods)?;
    create_link(path_hash, action_hash.clone(), LinkTypes::AllPods, ())?;

    // Link pod to target courses
    for course_hash in &pod.target_courses {
        create_link(
            action_hash.clone(),
            course_hash.clone(),
            LinkTypes::PodToCourses,
            (),
        )?;
        create_link(
            course_hash.clone(),
            action_hash.clone(),
            LinkTypes::CourseToPods,
            (),
        )?;
    }

    // Auto-add creator as admin member
    let creator_membership = PodMembership {
        pod_hash: action_hash.clone(),
        member: pod.creator.clone(),
        role: MemberRole::Admin,
        status: MembershipStatus::Active,
        joined_at: Some(pod.created_at),
        personal_goals: vec![],
        pod_reputation_permille: 1000, // Creator starts with full reputation (1.0)
        contributions: 0,
    };
    create_membership(creator_membership)?;

    Ok(action_hash)
}

/// Get a pod by its action hash
#[hdk_extern]
pub fn get_pod(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// List all pods
#[hdk_extern]
pub fn list_pods(_: ()) -> ExternResult<Vec<Record>> {
    let path = Path::from("all_pods");
    let path_hash = ensure_path(path, LinkTypes::AllPods)?;

    let links = get_links(
        LinkQuery::try_new(path_hash, LinkTypes::AllPods)?,
        GetStrategy::Local
    )?;

    let mut pods = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!("Failed to convert link target to ActionHash"))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            pods.push(record);
        }
    }

    Ok(pods)
}

/// Update pod status
#[hdk_extern]
pub fn update_pod_status(input: UpdatePodStatusInput) -> ExternResult<ActionHash> {
    // Get current pod
    let record = get(input.pod_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!("Pod not found"))?;

    let mut pod: LearningPod = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!("Failed to deserialize pod"))?;

    // Verify caller is admin
    let caller = agent_info()?.agent_initial_pubkey;
    if !is_pod_admin(&input.pod_hash, &caller)? {
        return Err(wasm_error!("Only pod admins can change status"));
    }

    // Update status
    pod.status = input.new_status;
    pod.last_activity = timestamp_to_i64(sys_time()?);

    update_entry(input.pod_hash, EntryTypes::LearningPod(pod))
}

// ============== Membership Functions ==============

/// Create a membership (invite a member)
#[hdk_extern]
pub fn create_membership(membership: PodMembership) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::PodMembership(membership.clone()))?;

    // Link pod -> member
    create_link(
        membership.pod_hash.clone(),
        membership.member.clone(),
        LinkTypes::PodToMembers,
        (),
    )?;

    // Link member -> pod
    create_link(
        membership.member,
        membership.pod_hash,
        LinkTypes::MemberToPods,
        (),
    )?;

    Ok(action_hash)
}

/// Invite a member to a pod
#[hdk_extern]
pub fn invite_member(input: InviteMemberInput) -> ExternResult<ActionHash> {
    let caller = agent_info()?.agent_initial_pubkey;

    // Verify caller is at least a moderator
    if !is_pod_moderator(&input.pod_hash, &caller)? {
        return Err(wasm_error!("Only moderators and admins can invite members"));
    }

    // Check pod isn't full
    let member_count = get_pod_member_count(&input.pod_hash)?;
    let pod = get_pod_entry(&input.pod_hash)?;
    if member_count >= pod.max_members as usize {
        return Err(wasm_error!("Pod is full"));
    }

    // Create pending membership
    let membership = PodMembership {
        pod_hash: input.pod_hash,
        member: input.invitee,
        role: MemberRole::Member,
        status: MembershipStatus::Invited,
        joined_at: None,
        personal_goals: vec![],
        pod_reputation_permille: 500, // Start at neutral reputation (0.5)
        contributions: 0,
    };

    create_membership(membership)
}

/// Accept a pod invitation
#[hdk_extern]
pub fn accept_invitation(pod_hash: ActionHash) -> ExternResult<ActionHash> {
    let caller = agent_info()?.agent_initial_pubkey;

    // Find the membership entry
    let membership_hash = find_membership_hash(&pod_hash, &caller)?
        .ok_or(wasm_error!("No invitation found"))?;

    let record = get(membership_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!("Membership not found"))?;

    let mut membership: PodMembership = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!("Failed to deserialize membership"))?;

    // Verify it's an invitation
    if membership.status != MembershipStatus::Invited {
        return Err(wasm_error!("No pending invitation"));
    }

    // Accept the invitation
    membership.status = MembershipStatus::Active;
    membership.joined_at = Some(timestamp_to_i64(sys_time()?));

    update_entry(membership_hash, EntryTypes::PodMembership(membership))
}

/// Leave a pod
#[hdk_extern]
pub fn leave_pod(pod_hash: ActionHash) -> ExternResult<ActionHash> {
    let caller = agent_info()?.agent_initial_pubkey;

    let membership_hash = find_membership_hash(&pod_hash, &caller)?
        .ok_or(wasm_error!("Not a member of this pod"))?;

    let record = get(membership_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!("Membership not found"))?;

    let mut membership: PodMembership = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!("Failed to deserialize membership"))?;

    // Can't leave if you're the only admin
    if membership.role == MemberRole::Admin {
        let admin_count = count_pod_admins(&pod_hash)?;
        if admin_count <= 1 {
            return Err(wasm_error!("Cannot leave: you are the only admin"));
        }
    }

    membership.status = MembershipStatus::Left;

    update_entry(membership_hash, EntryTypes::PodMembership(membership))
}

/// Get all members of a pod
#[hdk_extern]
pub fn get_pod_members(pod_hash: ActionHash) -> ExternResult<Vec<AgentPubKey>> {
    let links = get_links(
        LinkQuery::try_new(pod_hash, LinkTypes::PodToMembers)?,
        GetStrategy::Local
    )?;

    let members: Vec<AgentPubKey> = links
        .into_iter()
        .filter_map(|link| AgentPubKey::try_from(link.target).ok())
        .collect();

    Ok(members)
}

/// Get all pods a member belongs to
#[hdk_extern]
pub fn get_my_pods(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;

    let links = get_links(
        LinkQuery::try_new(caller, LinkTypes::MemberToPods)?,
        GetStrategy::Local
    )?;

    let mut pods = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!("Failed to convert link target to ActionHash"))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            pods.push(record);
        }
    }

    Ok(pods)
}

// ============== Progress Functions ==============

/// Update your progress in a pod (private entry)
#[hdk_extern]
pub fn update_pod_progress(progress: PodProgress) -> ExternResult<ActionHash> {
    let caller = agent_info()?.agent_initial_pubkey;

    // Verify caller is the member
    if progress.member != caller {
        return Err(wasm_error!("Can only update your own progress"));
    }

    // Verify caller is a member of the pod
    if !is_pod_member(&progress.pod_hash, &caller)? {
        return Err(wasm_error!("Not a member of this pod"));
    }

    create_entry(EntryTypes::PodProgress(progress))
}

/// Get aggregated pod progress (privacy-preserving)
/// Returns anonymized statistics, not individual progress
#[hdk_extern]
pub fn get_pod_progress_stats(pod_hash: ActionHash) -> ExternResult<PodProgressStats> {
    // This would aggregate progress from all members
    // In a real implementation, this would use secure aggregation
    // For now, return placeholder stats

    let member_count = get_pod_member_count(&pod_hash)?;

    Ok(PodProgressStats {
        pod_hash,
        member_count: member_count as u32,
        avg_progress: 0.0, // Would be computed from aggregated data
        avg_streak: 0,
        total_courses_completed: 0,
        last_updated: timestamp_to_i64(sys_time()?),
    })
}

// ============== Discussion Functions ==============

/// Create a discussion in a pod
#[hdk_extern]
pub fn create_discussion(discussion: PodDiscussion) -> ExternResult<ActionHash> {
    let caller = agent_info()?.agent_initial_pubkey;

    // Verify caller is a member of the pod
    if !is_pod_member(&discussion.pod_hash, &caller)? {
        return Err(wasm_error!("Only pod members can create discussions"));
    }

    let action_hash = create_entry(EntryTypes::PodDiscussion(discussion.clone()))?;

    // Link pod -> discussion
    create_link(
        discussion.pod_hash,
        action_hash.clone(),
        LinkTypes::PodToDiscussions,
        (),
    )?;

    Ok(action_hash)
}

/// Reply to a discussion
#[hdk_extern]
pub fn create_reply(reply: DiscussionReply) -> ExternResult<ActionHash> {
    let caller = agent_info()?.agent_initial_pubkey;

    // Get the discussion to find the pod
    let discussion_record = get(reply.discussion_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!("Discussion not found"))?;

    let discussion: PodDiscussion = discussion_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!("Failed to deserialize discussion"))?;

    // Verify caller is a member of the pod
    if !is_pod_member(&discussion.pod_hash, &caller)? {
        return Err(wasm_error!("Only pod members can reply"));
    }

    let action_hash = create_entry(EntryTypes::DiscussionReply(reply.clone()))?;

    // Link discussion -> reply
    create_link(
        reply.discussion_hash,
        action_hash.clone(),
        LinkTypes::DiscussionToReplies,
        (),
    )?;

    Ok(action_hash)
}

/// Get all discussions in a pod
#[hdk_extern]
pub fn get_pod_discussions(pod_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(pod_hash, LinkTypes::PodToDiscussions)?,
        GetStrategy::Local
    )?;

    let mut discussions = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!("Failed to convert link target to ActionHash"))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            discussions.push(record);
        }
    }

    Ok(discussions)
}

// ============== Achievement Functions ==============

/// Record a pod achievement
#[hdk_extern]
pub fn record_achievement(achievement: PodAchievement) -> ExternResult<ActionHash> {
    let caller = agent_info()?.agent_initial_pubkey;

    // Verify caller is a moderator or admin
    if !is_pod_moderator(&achievement.pod_hash, &caller)? {
        return Err(wasm_error!("Only moderators and admins can record achievements"));
    }

    let action_hash = create_entry(EntryTypes::PodAchievement(achievement.clone()))?;

    // Link pod -> achievement
    create_link(
        achievement.pod_hash,
        action_hash.clone(),
        LinkTypes::PodToAchievements,
        (),
    )?;

    Ok(action_hash)
}

/// Get all achievements for a pod
#[hdk_extern]
pub fn get_pod_achievements(pod_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(pod_hash, LinkTypes::PodToAchievements)?,
        GetStrategy::Local
    )?;

    let mut achievements = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!("Failed to convert link target to ActionHash"))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            achievements.push(record);
        }
    }

    Ok(achievements)
}

// ============== Challenge Functions ==============

/// Create a learning challenge in a pod
#[hdk_extern]
pub fn create_challenge(challenge: PodChallenge) -> ExternResult<ActionHash> {
    let caller = agent_info()?.agent_initial_pubkey;

    // Verify caller is a member of the pod
    if !is_pod_member(&challenge.pod_hash, &caller)? {
        return Err(wasm_error!("Only pod members can create challenges"));
    }

    let action_hash = create_entry(EntryTypes::PodChallenge(challenge.clone()))?;

    // Link pod -> challenge
    create_link(
        challenge.pod_hash,
        action_hash.clone(),
        LinkTypes::PodToChallenges,
        (),
    )?;

    Ok(action_hash)
}

/// Get active challenges for a pod
#[hdk_extern]
pub fn get_pod_challenges(pod_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(pod_hash, LinkTypes::PodToChallenges)?,
        GetStrategy::Local
    )?;

    let mut challenges = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!("Failed to convert link target to ActionHash"))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            challenges.push(record);
        }
    }

    Ok(challenges)
}

// ============== Helper Functions ==============

fn get_pod_entry(pod_hash: &ActionHash) -> ExternResult<LearningPod> {
    let record = get(pod_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!("Pod not found"))?;

    record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!("Failed to deserialize pod"))
}

fn is_pod_member(pod_hash: &ActionHash, agent: &AgentPubKey) -> ExternResult<bool> {
    let members = get_pod_members(pod_hash.clone())?;
    Ok(members.contains(agent))
}

fn is_pod_admin(pod_hash: &ActionHash, agent: &AgentPubKey) -> ExternResult<bool> {
    if let Some(membership_hash) = find_membership_hash(pod_hash, agent)? {
        if let Some(record) = get(membership_hash, GetOptions::default())? {
            if let Ok(Some(membership)) = record.entry().to_app_option::<PodMembership>() {
                return Ok(membership.role == MemberRole::Admin &&
                         membership.status == MembershipStatus::Active);
            }
        }
    }
    Ok(false)
}

fn is_pod_moderator(pod_hash: &ActionHash, agent: &AgentPubKey) -> ExternResult<bool> {
    if let Some(membership_hash) = find_membership_hash(pod_hash, agent)? {
        if let Some(record) = get(membership_hash, GetOptions::default())? {
            if let Ok(Some(membership)) = record.entry().to_app_option::<PodMembership>() {
                return Ok((membership.role == MemberRole::Moderator ||
                          membership.role == MemberRole::Admin) &&
                         membership.status == MembershipStatus::Active);
            }
        }
    }
    Ok(false)
}

fn find_membership_hash(pod_hash: &ActionHash, member: &AgentPubKey) -> ExternResult<Option<ActionHash>> {
    // This is a simplified implementation
    // In production, you'd want a more efficient lookup
    let links = get_links(
        LinkQuery::try_new(pod_hash.clone(), LinkTypes::PodToMembers)?,
        GetStrategy::Local
    )?;

    for link in links {
        if let Ok(link_target) = AgentPubKey::try_from(link.target.clone()) {
            if &link_target == member {
                // In a real implementation, we'd store the membership hash in the link tag
                // For now, return a placeholder
                return Ok(Some(ActionHash::from_raw_36(vec![0; 36])));
            }
        }
    }

    Ok(None)
}

fn get_pod_member_count(pod_hash: &ActionHash) -> ExternResult<usize> {
    let members = get_pod_members(pod_hash.clone())?;
    Ok(members.len())
}

fn count_pod_admins(_pod_hash: &ActionHash) -> ExternResult<usize> {
    // Simplified implementation
    // In production, would query membership entries
    Ok(1)
}

// ============== Input/Output Types ==============

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdatePodStatusInput {
    pub pod_hash: ActionHash,
    pub new_status: PodStatus,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct InviteMemberInput {
    pub pod_hash: ActionHash,
    pub invitee: AgentPubKey,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PodProgressStats {
    pub pod_hash: ActionHash,
    pub member_count: u32,
    pub avg_progress: f64,
    pub avg_streak: u32,
    pub total_courses_completed: u32,
    pub last_updated: i64,
}

// ============== Peer Tutoring ==============
//
// Topic-sharded, time-boxed anchors prevent DHT hotspots:
// Path::from("tutoring.{topic_id}.week_{week}") distributes load naturally.
// Matching runs CLIENT-SIDE in Leptos (per guardrail: no global aggregation in WASM).

/// Compute the week number for time-boxed anchor sharding.
fn current_week_number(timestamp: i64) -> u32 {
    // Unix epoch was a Thursday. Week 0 = 1970-01-01.
    // Approximate: timestamp / (7 * 24 * 3600)
    (timestamp / (7 * 24 * 3600)).max(0) as u32
}

/// Get the topic-sharded, time-boxed anchor for a tutoring topic.
fn tutoring_anchor(topic_id: &str, week: u32) -> ExternResult<EntryHash> {
    // Sharding: "tutoring.{topic_id}.week_{week}" — distributes across DHT neighborhoods
    let path = Path::from(format!("tutoring.{}.week_{}", topic_id, week));
    let typed_path = path.typed(LinkTypes::TopicToTutoringSessions)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Input for creating a tutoring session request.
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateTutoringSessionInput {
    pub tutor: AgentPubKey,
    pub topic_id: String,
    pub topic_name: String,
}

/// Create a peer tutoring session request.
///
/// Links to topic-sharded anchor (prevents DHT hotspots) and both agents.
#[hdk_extern]
pub fn create_tutoring_session(input: CreateTutoringSessionInput) -> ExternResult<ActionHash> {
    let tutee = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?.as_micros() / 1_000_000;
    let week = current_week_number(now);

    let session = PeerTutoringSession {
        tutor: input.tutor.clone(),
        tutee: tutee.clone(),
        topic_id: input.topic_id.clone(),
        topic_name: input.topic_name,
        status: TutoringSessionStatus::Requested,
        created_at: now,
        completed_at: None,
        tutor_rating_permille: None,
        tutee_rating_permille: None,
        notes: None,
    };

    let action_hash = create_entry(EntryTypes::PeerTutoringSession(session))?;

    // Link: tutee -> session
    let tutee_hash: AnyDhtHash = tutee.into();
    create_link(tutee_hash, action_hash.clone(), LinkTypes::AgentToTuteeSessions, vec![])?;

    // Link: tutor -> session
    let tutor_hash: AnyDhtHash = input.tutor.into();
    create_link(tutor_hash, action_hash.clone(), LinkTypes::AgentToTutorSessions, vec![])?;

    // Link: topic-sharded anchor -> session (prevents hotspot)
    let anchor = tutoring_anchor(&input.topic_id, week)?;
    create_link(anchor, action_hash.clone(), LinkTypes::TopicToTutoringSessions, vec![])?;

    Ok(action_hash)
}

/// Input for completing a tutoring session.
#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteTutoringSessionInput {
    pub session_hash: ActionHash,
    pub rating_permille: u16,
    pub notes: Option<String>,
}

/// Complete a peer tutoring session. Both tutor and tutee can call this
/// to provide their rating. Sovereignty growth is recorded on both sides.
#[hdk_extern]
pub fn complete_tutoring_session(input: CompleteTutoringSessionInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let record = get(input.session_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Session not found".into())))?;

    let mut session: PeerTutoringSession = match record.entry().as_option() {
        Some(Entry::App(bytes)) => PeerTutoringSession::try_from(
            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
        ).map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {:?}", e))))?,
        _ => return Err(wasm_error!(WasmErrorInner::Guest("Not an entry".into()))),
    };

    // Set rating based on who's calling
    if agent == session.tutor {
        session.tutor_rating_permille = Some(input.rating_permille);
    } else if agent == session.tutee {
        session.tutee_rating_permille = Some(input.rating_permille);
    } else {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only tutor or tutee can complete the session".into()
        )));
    }

    session.notes = input.notes.or(session.notes);

    // Mark completed if both have rated
    if session.tutor_rating_permille.is_some() && session.tutee_rating_permille.is_some() {
        session.status = TutoringSessionStatus::Completed;
        session.completed_at = Some(sys_time()?.as_micros() / 1_000_000);
    }

    update_entry(input.session_hash, &session)
}

/// List tutoring sessions where the caller is the tutor.
#[hdk_extern]
pub fn list_my_tutor_sessions(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_hash: AnyDhtHash = agent.into();

    let links = get_links(
        LinkQuery::try_new(agent_hash, LinkTypes::AgentToTutorSessions)?,
        GetStrategy::Local,
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

/// List tutoring sessions where the caller is the tutee.
#[hdk_extern]
pub fn list_my_tutee_sessions(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_hash: AnyDhtHash = agent.into();

    let links = get_links(
        LinkQuery::try_new(agent_hash, LinkTypes::AgentToTuteeSessions)?,
        GetStrategy::Local,
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

/// Find available tutors for a topic in the current time window.
/// Returns session records linked to the topic-sharded anchor.
///
/// NOTE: The actual scoring/matching runs CLIENT-SIDE in Leptos.
/// This zome only provides the raw data via anchor queries.
#[hdk_extern]
pub fn find_topic_tutors(topic_id: String) -> ExternResult<Vec<Record>> {
    let now = sys_time()?.as_micros() / 1_000_000;
    let week = current_week_number(now);

    // Query current week and previous week (covers edge cases near week boundary)
    let mut records = Vec::new();
    for w in [week.saturating_sub(1), week] {
        let anchor = tutoring_anchor(&topic_id, w)?;
        let links = get_links(
            LinkQuery::try_new(anchor, LinkTypes::TopicToTutoringSessions)?,
            GetStrategy::Local,
        )?;

        for link in links {
            if let Some(hash) = link.target.into_action_hash() {
                if let Some(record) = get(hash, GetOptions::default())? {
                    records.push(record);
                }
            }
        }
    }

    Ok(records)
}

// ============== Collective Competence Proofs ==============

/// Collective competence profile for a Learning Pod.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PodCompetenceProfile {
    pub pod_hash: String,
    pub collective_skills: Vec<String>,
    pub member_count: u32,
}

/// Aggregate collective competence for a Pod.
///
/// Collects all member skills from personal goals and produces a union.
/// Client-side matching compares this against career archetypes.
#[hdk_extern]
pub fn aggregate_pod_competence(pod_hash: ActionHash) -> ExternResult<PodCompetenceProfile> {
    let member_links = get_links(
        LinkQuery::try_new(pod_hash.clone(), LinkTypes::PodToMembers)?,
        GetStrategy::Local,
    )?;

    let mut all_skills = std::collections::HashSet::new();
    let mut member_count = 0u32;

    for link in member_links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(membership) = PodMembership::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        if membership.status == MembershipStatus::Active {
                            member_count += 1;
                            for goal in &membership.personal_goals {
                                all_skills.insert(goal.to_lowercase());
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(PodCompetenceProfile {
        pod_hash: pod_hash.to_string(),
        collective_skills: all_skills.into_iter().collect(),
        member_count,
    })
}
