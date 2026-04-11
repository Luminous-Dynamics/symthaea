// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared Inboxes & Team Collaboration for Mycelix Mail
//!
//! Enables team-based email management with shared inboxes,
//! assignment workflows, internal comments, and collaboration features.

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A team organization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Team {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub avatar_url: Option<String>,
    pub owner_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub settings: TeamSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TeamSettings {
    pub allow_member_invites: bool,
    pub require_assignment: bool,
    pub auto_assign_round_robin: bool,
    pub sla_warning_hours: Option<u32>,
    pub sla_critical_hours: Option<u32>,
    pub notify_on_assignment: bool,
    pub notify_on_mention: bool,
}

/// Team membership with role
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamMember {
    pub id: Uuid,
    pub team_id: Uuid,
    pub user_id: Uuid,
    pub role: TeamRole,
    pub status: MemberStatus,
    pub joined_at: DateTime<Utc>,
    pub invited_by: Option<Uuid>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TeamRole {
    Owner,
    Admin,
    Member,
    Guest,
}

impl TeamRole {
    pub fn can_manage_members(&self) -> bool {
        matches!(self, TeamRole::Owner | TeamRole::Admin)
    }

    pub fn can_manage_inboxes(&self) -> bool {
        matches!(self, TeamRole::Owner | TeamRole::Admin)
    }

    pub fn can_assign_emails(&self) -> bool {
        !matches!(self, TeamRole::Guest)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum MemberStatus {
    Active,
    Inactive,
    Pending,
}

/// A shared inbox accessible by team members
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedInbox {
    pub id: Uuid,
    pub team_id: Uuid,
    pub name: String,
    pub email_address: String,
    pub description: Option<String>,
    pub color: Option<String>,
    pub icon: Option<String>,
    pub created_at: DateTime<Utc>,
    pub settings: SharedInboxSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SharedInboxSettings {
    pub auto_assign: bool,
    pub assignment_strategy: AssignmentStrategy,
    pub allowed_members: Option<Vec<Uuid>>, // None = all team members
    pub notify_all_on_new: bool,
    pub require_resolution: bool,
    pub categories_enabled: bool,
    pub canned_responses_enabled: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum AssignmentStrategy {
    #[default]
    Manual,
    RoundRobin,
    LeastBusy,
    Random,
}

/// Email assignment to a team member
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailAssignment {
    pub id: Uuid,
    pub email_id: Uuid,
    pub inbox_id: Uuid,
    pub assignee_id: Option<Uuid>,
    pub assigned_by: Option<Uuid>,
    pub status: AssignmentStatus,
    pub priority: AssignmentPriority,
    pub due_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub resolution_note: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AssignmentStatus {
    Unassigned,
    Assigned,
    InProgress,
    WaitingForCustomer,
    WaitingForThirdParty,
    Resolved,
    Closed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Ord, PartialOrd, Eq)]
pub enum AssignmentPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Urgent = 4,
}

/// Internal comment on an email
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalComment {
    pub id: Uuid,
    pub email_id: Uuid,
    pub author_id: Uuid,
    pub author_name: String,
    pub content: String,
    pub mentions: Vec<Uuid>,
    pub is_resolution_note: bool,
    pub created_at: DateTime<Utc>,
    pub edited_at: Option<DateTime<Utc>>,
}

/// Canned response template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CannedResponse {
    pub id: Uuid,
    pub inbox_id: Option<Uuid>, // None = team-wide
    pub team_id: Uuid,
    pub name: String,
    pub subject: Option<String>,
    pub body: String,
    pub shortcut: Option<String>,
    pub variables: Vec<String>, // Placeholder variables like {{customer_name}}
    pub created_by: Uuid,
    pub created_at: DateTime<Utc>,
}

/// Team activity log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityLogEntry {
    pub id: Uuid,
    pub team_id: Uuid,
    pub actor_id: Uuid,
    pub action: ActivityAction,
    pub target_type: TargetType,
    pub target_id: Uuid,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivityAction {
    Created,
    Updated,
    Deleted,
    Assigned,
    Unassigned,
    Claimed,
    Resolved,
    Reopened,
    Commented,
    Mentioned,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TargetType {
    Email,
    SharedInbox,
    Team,
    Member,
    CannedResponse,
}

/// Service for team collaboration
pub struct TeamCollaborationService {
    teams: HashMap<Uuid, Team>,
    members: HashMap<Uuid, Vec<TeamMember>>,
    shared_inboxes: HashMap<Uuid, SharedInbox>,
    assignments: HashMap<Uuid, EmailAssignment>,
    comments: HashMap<Uuid, Vec<InternalComment>>,
    canned_responses: HashMap<Uuid, Vec<CannedResponse>>,
    activity_log: Vec<ActivityLogEntry>,
    round_robin_index: HashMap<Uuid, usize>, // inbox_id -> current index
}

impl TeamCollaborationService {
    pub fn new() -> Self {
        Self {
            teams: HashMap::new(),
            members: HashMap::new(),
            shared_inboxes: HashMap::new(),
            assignments: HashMap::new(),
            comments: HashMap::new(),
            canned_responses: HashMap::new(),
            activity_log: Vec::new(),
            round_robin_index: HashMap::new(),
        }
    }

    // Team management

    /// Create a new team
    pub fn create_team(&mut self, name: String, owner_id: Uuid) -> Team {
        let team = Team {
            id: Uuid::new_v4(),
            name,
            description: None,
            avatar_url: None,
            owner_id,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            settings: TeamSettings::default(),
        };

        let team_id = team.id;
        self.teams.insert(team_id, team.clone());

        // Add owner as member
        let owner_member = TeamMember {
            id: Uuid::new_v4(),
            team_id,
            user_id: owner_id,
            role: TeamRole::Owner,
            status: MemberStatus::Active,
            joined_at: Utc::now(),
            invited_by: None,
        };

        self.members.entry(team_id).or_default().push(owner_member);

        self.log_activity(team_id, owner_id, ActivityAction::Created, TargetType::Team, team_id);

        team
    }

    /// Add a member to a team
    pub fn add_member(
        &mut self,
        team_id: Uuid,
        user_id: Uuid,
        role: TeamRole,
        invited_by: Uuid,
    ) -> Result<TeamMember, TeamError> {
        if !self.teams.contains_key(&team_id) {
            return Err(TeamError::TeamNotFound(team_id));
        }

        // Check if already a member
        if let Some(members) = self.members.get(&team_id) {
            if members.iter().any(|m| m.user_id == user_id) {
                return Err(TeamError::AlreadyMember(user_id));
            }
        }

        let member = TeamMember {
            id: Uuid::new_v4(),
            team_id,
            user_id,
            role,
            status: MemberStatus::Active,
            joined_at: Utc::now(),
            invited_by: Some(invited_by),
        };

        self.members.entry(team_id).or_default().push(member.clone());

        self.log_activity(team_id, invited_by, ActivityAction::Created, TargetType::Member, member.id);

        Ok(member)
    }

    /// Remove a member from a team
    pub fn remove_member(&mut self, team_id: Uuid, user_id: Uuid, removed_by: Uuid) -> Result<(), TeamError> {
        if let Some(members) = self.members.get_mut(&team_id) {
            let original_len = members.len();
            members.retain(|m| m.user_id != user_id);

            if members.len() == original_len {
                return Err(TeamError::MemberNotFound(user_id));
            }

            self.log_activity(team_id, removed_by, ActivityAction::Deleted, TargetType::Member, user_id);
        }

        Ok(())
    }

    // Shared inbox management

    /// Create a shared inbox
    pub fn create_shared_inbox(
        &mut self,
        team_id: Uuid,
        name: String,
        email_address: String,
        created_by: Uuid,
    ) -> Result<SharedInbox, TeamError> {
        if !self.teams.contains_key(&team_id) {
            return Err(TeamError::TeamNotFound(team_id));
        }

        let inbox = SharedInbox {
            id: Uuid::new_v4(),
            team_id,
            name,
            email_address,
            description: None,
            color: None,
            icon: None,
            created_at: Utc::now(),
            settings: SharedInboxSettings::default(),
        };

        self.shared_inboxes.insert(inbox.id, inbox.clone());

        self.log_activity(team_id, created_by, ActivityAction::Created, TargetType::SharedInbox, inbox.id);

        Ok(inbox)
    }

    /// Get shared inboxes for a team
    pub fn get_team_inboxes(&self, team_id: Uuid) -> Vec<&SharedInbox> {
        self.shared_inboxes
            .values()
            .filter(|i| i.team_id == team_id)
            .collect()
    }

    // Email assignment

    /// Assign an email to a team member
    pub fn assign_email(
        &mut self,
        email_id: Uuid,
        inbox_id: Uuid,
        assignee_id: Uuid,
        assigned_by: Uuid,
    ) -> Result<EmailAssignment, TeamError> {
        let inbox = self.shared_inboxes.get(&inbox_id)
            .ok_or(TeamError::InboxNotFound(inbox_id))?;

        let assignment = EmailAssignment {
            id: Uuid::new_v4(),
            email_id,
            inbox_id,
            assignee_id: Some(assignee_id),
            assigned_by: Some(assigned_by),
            status: AssignmentStatus::Assigned,
            priority: AssignmentPriority::Normal,
            due_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            resolved_at: None,
            resolution_note: None,
        };

        self.assignments.insert(assignment.id, assignment.clone());

        self.log_activity(inbox.team_id, assigned_by, ActivityAction::Assigned, TargetType::Email, email_id);

        Ok(assignment)
    }

    /// Claim an unassigned email
    pub fn claim_email(&mut self, assignment_id: Uuid, user_id: Uuid) -> Result<(), TeamError> {
        let assignment = self.assignments.get_mut(&assignment_id)
            .ok_or(TeamError::AssignmentNotFound(assignment_id))?;

        if assignment.assignee_id.is_some() {
            return Err(TeamError::AlreadyAssigned(assignment_id));
        }

        assignment.assignee_id = Some(user_id);
        assignment.status = AssignmentStatus::Assigned;
        assignment.updated_at = Utc::now();

        let inbox = self.shared_inboxes.get(&assignment.inbox_id).unwrap();
        self.log_activity(inbox.team_id, user_id, ActivityAction::Claimed, TargetType::Email, assignment.email_id);

        Ok(())
    }

    /// Auto-assign using the configured strategy
    pub fn auto_assign(&mut self, email_id: Uuid, inbox_id: Uuid) -> Result<EmailAssignment, TeamError> {
        let inbox = self.shared_inboxes.get(&inbox_id)
            .ok_or(TeamError::InboxNotFound(inbox_id))?;

        let members = self.get_inbox_members(inbox_id)?;
        if members.is_empty() {
            return Err(TeamError::NoAvailableAssignee);
        }

        let assignee_id = match inbox.settings.assignment_strategy {
            AssignmentStrategy::Manual => {
                return Err(TeamError::ManualAssignmentRequired);
            }
            AssignmentStrategy::RoundRobin => {
                let idx = self.round_robin_index.entry(inbox_id).or_insert(0);
                let member = &members[*idx % members.len()];
                *idx = (*idx + 1) % members.len();
                member.user_id
            }
            AssignmentStrategy::LeastBusy => {
                // Find member with fewest active assignments
                let mut workloads: HashMap<Uuid, usize> = HashMap::new();
                for member in &members {
                    workloads.insert(member.user_id, 0);
                }

                for assignment in self.assignments.values() {
                    if assignment.inbox_id == inbox_id &&
                       !matches!(assignment.status, AssignmentStatus::Resolved | AssignmentStatus::Closed) {
                        if let Some(assignee) = assignment.assignee_id {
                            *workloads.entry(assignee).or_insert(0) += 1;
                        }
                    }
                }

                workloads.into_iter()
                    .min_by_key(|(_, count)| *count)
                    .map(|(id, _)| id)
                    .unwrap_or(members[0].user_id)
            }
            AssignmentStrategy::Random => {
                use rand::Rng;
                let idx = rand::thread_rng().gen_range(0..members.len());
                members[idx].user_id
            }
        };

        self.assign_email(email_id, inbox_id, assignee_id, Uuid::nil())
    }

    /// Get members who can access an inbox
    fn get_inbox_members(&self, inbox_id: Uuid) -> Result<Vec<&TeamMember>, TeamError> {
        let inbox = self.shared_inboxes.get(&inbox_id)
            .ok_or(TeamError::InboxNotFound(inbox_id))?;

        let team_members = self.members.get(&inbox.team_id)
            .ok_or(TeamError::TeamNotFound(inbox.team_id))?;

        let members: Vec<&TeamMember> = team_members
            .iter()
            .filter(|m| m.status == MemberStatus::Active)
            .filter(|m| {
                if let Some(ref allowed) = inbox.settings.allowed_members {
                    allowed.contains(&m.user_id)
                } else {
                    true
                }
            })
            .collect();

        Ok(members)
    }

    /// Resolve an assignment
    pub fn resolve_assignment(
        &mut self,
        assignment_id: Uuid,
        user_id: Uuid,
        resolution_note: Option<String>,
    ) -> Result<(), TeamError> {
        let assignment = self.assignments.get_mut(&assignment_id)
            .ok_or(TeamError::AssignmentNotFound(assignment_id))?;

        assignment.status = AssignmentStatus::Resolved;
        assignment.resolved_at = Some(Utc::now());
        assignment.resolution_note = resolution_note;
        assignment.updated_at = Utc::now();

        let inbox = self.shared_inboxes.get(&assignment.inbox_id).unwrap();
        self.log_activity(inbox.team_id, user_id, ActivityAction::Resolved, TargetType::Email, assignment.email_id);

        Ok(())
    }

    // Comments

    /// Add an internal comment to an email
    pub fn add_comment(
        &mut self,
        email_id: Uuid,
        author_id: Uuid,
        author_name: String,
        content: String,
        mentions: Vec<Uuid>,
    ) -> InternalComment {
        let comment = InternalComment {
            id: Uuid::new_v4(),
            email_id,
            author_id,
            author_name,
            content,
            mentions: mentions.clone(),
            is_resolution_note: false,
            created_at: Utc::now(),
            edited_at: None,
        };

        self.comments.entry(email_id).or_default().push(comment.clone());

        // Log mentions as separate activities
        for mentioned_user in mentions {
            // In real impl, would need team_id from assignment
            // self.log_activity(team_id, author_id, ActivityAction::Mentioned, TargetType::Email, email_id);
        }

        comment
    }

    /// Get comments for an email
    pub fn get_comments(&self, email_id: Uuid) -> Vec<&InternalComment> {
        self.comments
            .get(&email_id)
            .map(|c| c.iter().collect())
            .unwrap_or_default()
    }

    // Canned responses

    /// Create a canned response
    pub fn create_canned_response(
        &mut self,
        team_id: Uuid,
        inbox_id: Option<Uuid>,
        name: String,
        body: String,
        created_by: Uuid,
    ) -> CannedResponse {
        let response = CannedResponse {
            id: Uuid::new_v4(),
            inbox_id,
            team_id,
            name,
            subject: None,
            body,
            shortcut: None,
            variables: Vec::new(),
            created_by,
            created_at: Utc::now(),
        };

        self.canned_responses.entry(team_id).or_default().push(response.clone());

        response
    }

    /// Get canned responses for a team/inbox
    pub fn get_canned_responses(&self, team_id: Uuid, inbox_id: Option<Uuid>) -> Vec<&CannedResponse> {
        self.canned_responses
            .get(&team_id)
            .map(|responses| {
                responses
                    .iter()
                    .filter(|r| inbox_id.is_none() || r.inbox_id == inbox_id || r.inbox_id.is_none())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Apply variables to a canned response
    pub fn apply_canned_response(
        response: &CannedResponse,
        variables: &HashMap<String, String>,
    ) -> (Option<String>, String) {
        let mut subject = response.subject.clone();
        let mut body = response.body.clone();

        for (key, value) in variables {
            let placeholder = format!("{{{{{}}}}}", key);
            if let Some(ref mut s) = subject {
                *s = s.replace(&placeholder, value);
            }
            body = body.replace(&placeholder, value);
        }

        (subject, body)
    }

    // Activity logging

    fn log_activity(
        &mut self,
        team_id: Uuid,
        actor_id: Uuid,
        action: ActivityAction,
        target_type: TargetType,
        target_id: Uuid,
    ) {
        let entry = ActivityLogEntry {
            id: Uuid::new_v4(),
            team_id,
            actor_id,
            action,
            target_type,
            target_id,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        };

        self.activity_log.push(entry);
    }

    /// Get activity log for a team
    pub fn get_activity_log(&self, team_id: Uuid, limit: usize) -> Vec<&ActivityLogEntry> {
        self.activity_log
            .iter()
            .rev()
            .filter(|e| e.team_id == team_id)
            .take(limit)
            .collect()
    }
}

impl Default for TeamCollaborationService {
    fn default() -> Self {
        Self::new()
    }
}

/// Team-related errors
#[derive(Debug, Clone)]
pub enum TeamError {
    TeamNotFound(Uuid),
    MemberNotFound(Uuid),
    InboxNotFound(Uuid),
    AssignmentNotFound(Uuid),
    AlreadyMember(Uuid),
    AlreadyAssigned(Uuid),
    NoAvailableAssignee,
    ManualAssignmentRequired,
    PermissionDenied(String),
}

impl std::fmt::Display for TeamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TeamNotFound(id) => write!(f, "Team not found: {}", id),
            Self::MemberNotFound(id) => write!(f, "Member not found: {}", id),
            Self::InboxNotFound(id) => write!(f, "Inbox not found: {}", id),
            Self::AssignmentNotFound(id) => write!(f, "Assignment not found: {}", id),
            Self::AlreadyMember(id) => write!(f, "User {} is already a member", id),
            Self::AlreadyAssigned(id) => write!(f, "Assignment {} is already assigned", id),
            Self::NoAvailableAssignee => write!(f, "No available team member for assignment"),
            Self::ManualAssignmentRequired => write!(f, "Manual assignment is required"),
            Self::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
        }
    }
}

impl std::error::Error for TeamError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_team() {
        let mut service = TeamCollaborationService::new();
        let owner_id = Uuid::new_v4();

        let team = service.create_team("Engineering".to_string(), owner_id);

        assert_eq!(team.name, "Engineering");
        assert_eq!(team.owner_id, owner_id);

        // Owner should be automatically added as member
        let members = service.members.get(&team.id).unwrap();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].role, TeamRole::Owner);
    }

    #[test]
    fn test_shared_inbox() {
        let mut service = TeamCollaborationService::new();
        let owner_id = Uuid::new_v4();

        let team = service.create_team("Support".to_string(), owner_id);
        let inbox = service.create_shared_inbox(
            team.id,
            "Support Queue".to_string(),
            "support@company.com".to_string(),
            owner_id,
        ).unwrap();

        assert_eq!(inbox.team_id, team.id);
        assert_eq!(inbox.email_address, "support@company.com");
    }

    #[test]
    fn test_email_assignment() {
        let mut service = TeamCollaborationService::new();
        let owner_id = Uuid::new_v4();
        let member_id = Uuid::new_v4();
        let email_id = Uuid::new_v4();

        let team = service.create_team("Support".to_string(), owner_id);
        service.add_member(team.id, member_id, TeamRole::Member, owner_id).unwrap();

        let inbox = service.create_shared_inbox(
            team.id,
            "Queue".to_string(),
            "queue@example.com".to_string(),
            owner_id,
        ).unwrap();

        let assignment = service.assign_email(email_id, inbox.id, member_id, owner_id).unwrap();

        assert_eq!(assignment.assignee_id, Some(member_id));
        assert_eq!(assignment.status, AssignmentStatus::Assigned);
    }
}
