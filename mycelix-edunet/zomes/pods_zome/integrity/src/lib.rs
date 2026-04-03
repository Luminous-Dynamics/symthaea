// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Learning Pods Integrity Zome
//!
//! Defines entry types for the revolutionary "Learning Pods" feature.
//!
//! ## What are Learning Pods?
//!
//! Learning Pods are small groups (3-7 people) that form cohort-based learning circles.
//! They enable:
//! - **Peer Accountability**: Members support each other's learning journey
//! - **Social Learning**: Shared discussions, challenges, and achievements
//! - **Privacy-Preserving Competition**: Anonymous progress comparison within pod
//! - **Collective Credentials**: Pod-level achievements and certifications
//! - **Federated Learning Cohorts**: Pods can participate in FL rounds together
//!
//! ## Pod Lifecycle
//!
//! 1. **Formation**: Creator invites members, sets learning goals
//! 2. **Active**: Members learn, share progress, collaborate
//! 3. **Achievement**: Pod completes goals, earns collective credentials
//! 4. **Evolution**: Pod can split, merge, or graduate

use hdi::prelude::*;

/// Pod status lifecycle
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PodStatus {
    /// Pod is forming, accepting new members
    Forming,
    /// Pod is active with learning in progress
    Active,
    /// Pod is paused (e.g., break, holidays)
    Paused,
    /// Pod has completed its learning goals
    Completed,
    /// Pod has been archived
    Archived,
}

impl Default for PodStatus {
    fn default() -> Self {
        PodStatus::Forming
    }
}

/// Learning Pod - a small group cohort for social learning
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct LearningPod {
    /// Human-readable name for the pod
    pub name: String,
    /// Description of the pod's purpose and goals
    pub description: String,
    /// The agent who created the pod
    pub creator: AgentPubKey,
    /// Target course(s) the pod is learning (action hashes)
    pub target_courses: Vec<ActionHash>,
    /// Minimum members required for pod to become active
    pub min_members: u8,
    /// Maximum members allowed (3-7 recommended)
    pub max_members: u8,
    /// Current status of the pod
    pub status: PodStatus,
    /// Learning goals for the pod (free-form text)
    pub goals: Vec<String>,
    /// Target completion date (optional)
    pub target_completion: Option<i64>,
    /// Whether progress is shared within the pod
    pub share_progress: bool,
    /// Whether discussions are public or pod-only
    pub private_discussions: bool,
    /// Creation timestamp
    pub created_at: i64,
    /// Last activity timestamp
    pub last_activity: i64,
}

/// Pod membership status
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MembershipStatus {
    /// Invited but not yet accepted
    Invited,
    /// Active member
    Active,
    /// Left the pod voluntarily
    Left,
    /// Removed by pod governance
    Removed,
    /// Graduated (completed pod goals)
    Graduated,
}

impl Default for MembershipStatus {
    fn default() -> Self {
        MembershipStatus::Invited
    }
}

/// Pod membership role
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemberRole {
    /// Regular member
    Member,
    /// Can invite new members, moderate discussions
    Moderator,
    /// Full control over pod settings
    Admin,
}

impl Default for MemberRole {
    fn default() -> Self {
        MemberRole::Member
    }
}

/// Pod Membership - links an agent to a pod with role and status
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct PodMembership {
    /// The pod this membership is for
    pub pod_hash: ActionHash,
    /// The member agent
    pub member: AgentPubKey,
    /// Role in the pod
    pub role: MemberRole,
    /// Current membership status
    pub status: MembershipStatus,
    /// When the member joined (accepted invitation)
    pub joined_at: Option<i64>,
    /// Personal commitment/goals for this pod
    pub personal_goals: Vec<String>,
    /// Reputation score as permille (0-1000 representing 0.0-1.0)
    pub pod_reputation_permille: u16,
    /// Number of contributions (discussions, help, etc.)
    pub contributions: u64,
}

/// Pod Progress - aggregated progress for a pod member (private entry)
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct PodProgress {
    /// The pod this progress is for
    pub pod_hash: ActionHash,
    /// The member this progress belongs to
    pub member: AgentPubKey,
    /// Overall progress as permille (0-1000 representing 0.0-1.0)
    pub progress_permille: u16,
    /// Courses completed
    pub courses_completed: Vec<ActionHash>,
    /// Current course progress map (course_hash -> progress_permille)
    pub course_progress: Vec<(ActionHash, u16)>,
    /// Weekly learning time in minutes
    pub weekly_time_minutes: u32,
    /// Streak (consecutive days of learning)
    pub streak_days: u32,
    /// Last activity timestamp
    pub last_active: i64,
}

/// Pod Discussion - a discussion thread within a pod
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct PodDiscussion {
    /// The pod this discussion belongs to
    pub pod_hash: ActionHash,
    /// Discussion title
    pub title: String,
    /// Initial message content
    pub content: String,
    /// Author of the discussion
    pub author: AgentPubKey,
    /// Related course (optional)
    pub related_course: Option<ActionHash>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Whether this is pinned
    pub pinned: bool,
    /// Creation timestamp
    pub created_at: i64,
}

/// Discussion Reply - a reply in a pod discussion
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct DiscussionReply {
    /// The discussion this reply belongs to
    pub discussion_hash: ActionHash,
    /// Parent reply (if nested)
    pub parent_reply: Option<ActionHash>,
    /// Reply content
    pub content: String,
    /// Author of the reply
    pub author: AgentPubKey,
    /// Helpful votes received
    pub helpful_votes: u32,
    /// Creation timestamp
    pub created_at: i64,
}

/// Pod Achievement - a collective achievement earned by the pod
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct PodAchievement {
    /// The pod that earned this achievement
    pub pod_hash: ActionHash,
    /// Achievement type/name
    pub achievement_type: String,
    /// Description of what was achieved
    pub description: String,
    /// Members who contributed to this achievement
    pub contributing_members: Vec<AgentPubKey>,
    /// Evidence/proof (e.g., credential hashes, course completions)
    pub evidence: Vec<String>,
    /// When the achievement was earned
    pub earned_at: i64,
    /// Points/XP value of this achievement
    pub points: u32,
}

/// Pod Challenge - a learning challenge within the pod
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct PodChallenge {
    /// The pod this challenge belongs to
    pub pod_hash: ActionHash,
    /// Challenge title
    pub title: String,
    /// Challenge description and rules
    pub description: String,
    /// Creator of the challenge
    pub creator: AgentPubKey,
    /// Challenge type
    pub challenge_type: ChallengeType,
    /// Start time
    pub start_time: i64,
    /// End time
    pub end_time: i64,
    /// Reward for completing (points, badges, etc.)
    pub reward: String,
    /// Whether the challenge is active
    pub active: bool,
}

/// Types of challenges
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChallengeType {
    /// Complete specific courses
    CourseCompletion,
    /// Maintain a learning streak
    StreakChallenge,
    /// Learn for X hours
    TimeChallenge,
    /// Help other pod members
    PeerHelp,
    /// Custom challenge
    Custom,
}

// ============== Peer Tutoring ==============

/// A 1:1 peer tutoring session between a tutor and tutee.
///
/// Linked via topic-sharded, time-boxed anchors to prevent DHT hotspots:
/// `Path::from("tutoring.{topic}.week_{week_number}")`
/// This distributes load naturally and lets clients fetch localized subsets.
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct PeerTutoringSession {
    /// Agent who is tutoring (has mastered the topic)
    pub tutor: AgentPubKey,
    /// Agent being tutored (struggling with the topic)
    pub tutee: AgentPubKey,
    /// Topic being tutored (e.g., "CAPS.Mathematics.Gr12.P1.CALC")
    pub topic_id: String,
    /// Human-readable topic name
    pub topic_name: String,
    /// Session status
    pub status: TutoringSessionStatus,
    /// When the session was created/requested
    pub created_at: i64,
    /// When the session was completed (if finished)
    pub completed_at: Option<i64>,
    /// Tutor's rating of the session (0-1000 permille)
    pub tutor_rating_permille: Option<u16>,
    /// Tutee's rating of the session (0-1000 permille)
    pub tutee_rating_permille: Option<u16>,
    /// Optional notes about the session
    pub notes: Option<String>,
}

/// Status of a peer tutoring session
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TutoringSessionStatus {
    /// Tutor has been matched/requested
    Requested,
    /// Both parties have confirmed
    Active,
    /// Session completed successfully
    Completed,
    /// Session was cancelled
    Cancelled,
}

// ============== Entry and Link Type Definitions ==============

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(required_validations = 3, visibility = "public")]
    LearningPod(LearningPod),
    #[entry_type(required_validations = 3, visibility = "public")]
    PodMembership(PodMembership),
    #[entry_type(required_validations = 1, visibility = "private")]
    PodProgress(PodProgress),
    #[entry_type(required_validations = 3, visibility = "public")]
    PodDiscussion(PodDiscussion),
    #[entry_type(required_validations = 3, visibility = "public")]
    DiscussionReply(DiscussionReply),
    #[entry_type(required_validations = 3, visibility = "public")]
    PodAchievement(PodAchievement),
    #[entry_type(required_validations = 3, visibility = "public")]
    PodChallenge(PodChallenge),
    #[entry_type(required_validations = 1, visibility = "public")]
    PeerTutoringSession(PeerTutoringSession),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Path anchor for all pods
    AllPods,
    /// Pod -> Members
    PodToMembers,
    /// Member -> Pods they're in
    MemberToPods,
    /// Pod -> Discussions
    PodToDiscussions,
    /// Discussion -> Replies
    DiscussionToReplies,
    /// Pod -> Achievements
    PodToAchievements,
    /// Pod -> Challenges
    PodToChallenges,
    /// Pod -> Target Courses
    PodToCourses,
    /// Course -> Pods studying it
    CourseToPods,
    /// Agent -> tutoring sessions as tutor
    AgentToTutorSessions,
    /// Agent -> tutoring sessions as tutee
    AgentToTuteeSessions,
    /// Topic-sharded anchor -> tutoring sessions (e.g., "tutoring.calculus.week_14")
    TopicToTutoringSessions,
}

// ============== Validation Functions ==============

/// Validate pod creation
pub fn validate_create_pod(pod: &LearningPod) -> ExternResult<ValidateCallbackResult> {
    // Pod name must not be empty
    if pod.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Pod name cannot be empty".to_string(),
        ));
    }

    // Pod name length limit
    if pod.name.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Pod name must be 100 characters or less".to_string(),
        ));
    }

    // Validate member limits (3-7 recommended, 2-10 allowed)
    if pod.min_members < 2 || pod.min_members > 10 {
        return Ok(ValidateCallbackResult::Invalid(
            "Minimum members must be between 2 and 10".to_string(),
        ));
    }

    if pod.max_members < pod.min_members || pod.max_members > 15 {
        return Ok(ValidateCallbackResult::Invalid(
            "Maximum members must be >= minimum and <= 15".to_string(),
        ));
    }

    // Must have at least one goal
    if pod.goals.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Pod must have at least one learning goal".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate membership creation
pub fn validate_create_membership(membership: &PodMembership) -> ExternResult<ValidateCallbackResult> {
    // Reputation must be valid (0-1000 permille)
    if membership.pod_reputation_permille > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Pod reputation must be between 0 and 1000 (permille)".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate peer tutoring session
pub fn validate_tutoring_session(session: &PeerTutoringSession) -> ExternResult<ValidateCallbackResult> {
    if session.tutor == session.tutee {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot tutor yourself".to_string(),
        ));
    }
    if session.topic_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Topic ID cannot be empty".to_string(),
        ));
    }
    if let Some(rating) = session.tutor_rating_permille {
        if rating > 1000 {
            return Ok(ValidateCallbackResult::Invalid(
                "Rating must be 0-1000 permille".to_string(),
            ));
        }
    }
    if let Some(rating) = session.tutee_rating_permille {
        if rating > 1000 {
            return Ok(ValidateCallbackResult::Invalid(
                "Rating must be 0-1000 permille".to_string(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Validate callback dispatcher
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::LearningPod(pod) => validate_create_pod(&pod),
                EntryTypes::PodMembership(membership) => validate_create_membership(&membership),
                EntryTypes::PeerTutoringSession(session) => validate_tutoring_session(&session),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::LearningPod(pod) => validate_create_pod(&pod),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pod_status_default() {
        assert_eq!(PodStatus::default(), PodStatus::Forming);
    }

    #[test]
    fn test_membership_status_default() {
        assert_eq!(MembershipStatus::default(), MembershipStatus::Invited);
    }

    #[test]
    fn test_member_role_default() {
        assert_eq!(MemberRole::default(), MemberRole::Member);
    }
}
