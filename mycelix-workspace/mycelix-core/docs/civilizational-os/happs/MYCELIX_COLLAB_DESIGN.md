# Mycelix-Collab: Decentralized Project Collaboration

## Vision & Design Document

**Version**: 1.0.0
**Created**: December 30, 2025
**Status**: Design Phase
**Priority**: Tier 1 - Ecosystem Expansion

---

## Executive Summary

Mycelix-Collab is a decentralized project coordination platform that enables teams to collaborate on complex projects with transparent contribution tracking, milestone-based payments, and reputation-aware team formation. It bridges Marketplace (for project-based work), Praxis (for skill verification), and MATL (for trust-based collaboration).

### Why Collab?

Traditional project management fails at:
- **Attribution**: Who really did what?
- **Payment fairness**: Equal pay for unequal work
- **Trust**: Working with strangers is risky
- **Coordination**: Distributed teams struggle
- **Learning**: Skills aren't recognized

Collab brings transparency and trust to collaborative work.

---

## Core Principles

### 1. Contribution Transparency
Every contribution is recorded, verified, and visible.

### 2. Fair Compensation
Pay is proportional to verified contribution, not politics.

### 3. Skill Discovery
Work reveals and builds skills, recognized via Praxis.

### 4. Reputation Staking
Team members stake reputation on project success.

### 5. Emergent Coordination
Teams self-organize around shared goals.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Collab hApp                                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     Coordinator Zomes                            ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │ Project  │ │  Team    │ │  Task    │ │Milestone │           ││
│  │  │ Manager  │ │ Manager  │ │ Manager  │ │ Manager  │           ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │Contribut │ │ Escrow   │ │ Review   │ │ Skills   │           ││
│  │  │ Tracker  │ │ Manager  │ │ System   │ │ Tracker  │           ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│                     Cross-hApp Integration                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │Marketplace│  │ Praxis   │  │ MATL/    │  │ Arbiter  │            │
│  │(Funding)  │  │(Skills)  │  │ Bridge   │  │(Disputes)│            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### Core Entry Types

```rust
/// A collaborative project
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Project {
    /// Unique project ID
    pub project_id: String,
    /// Project title
    pub title: String,
    /// Description
    pub description: String,
    /// Project type
    pub project_type: ProjectType,
    /// Current status
    pub status: ProjectStatus,
    /// Creator
    pub creator: AgentPubKey,
    /// Team configuration
    pub team_config: TeamConfig,
    /// Compensation model
    pub compensation: CompensationModel,
    /// Required skills
    pub required_skills: Vec<RequiredSkill>,
    /// Milestones
    pub milestones: Vec<Milestone>,
    /// Budget (if funded)
    pub budget: Option<Budget>,
    /// Timeline
    pub timeline: ProjectTimeline,
    /// Governance model
    pub governance: ProjectGovernance,
    /// Created at
    pub created_at: Timestamp,
    /// Tags
    pub tags: Vec<String>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ProjectType {
    /// Software development
    Software { repo_url: Option<String> },
    /// Research project
    Research { field: String },
    /// Creative project
    Creative { medium: String },
    /// Event organization
    Event { date: Option<String>, location: Option<String> },
    /// Content creation
    Content { content_type: String },
    /// Community initiative
    Community { scope: String },
    /// Other
    Other(String),
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ProjectStatus {
    Draft,
    Recruiting,
    Active,
    Paused,
    Completed,
    Cancelled,
    Archived,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct TeamConfig {
    /// Maximum team size
    pub max_size: Option<u32>,
    /// Minimum team size to start
    pub min_size: u32,
    /// Roles needed
    pub roles: Vec<ProjectRole>,
    /// How team members are selected
    pub selection: TeamSelection,
    /// Can members leave during project?
    pub allow_leave: bool,
    /// Can new members join during project?
    pub allow_join: bool,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectRole {
    pub name: String,
    pub description: String,
    pub required_skills: Vec<String>,
    pub min_reputation: f64,
    pub min_identity_level: IdentityLevel,
    pub count_needed: u32,
    pub compensation_share: f64, // Percentage of role allocation
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum TeamSelection {
    /// Creator approves all
    CreatorApproval,
    /// First come, first served (if qualified)
    FirstCome,
    /// Team votes on new members
    TeamVote { threshold: f64 },
    /// Automatic based on credentials
    Automatic,
    /// Hybrid
    Hybrid,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct CompensationModel {
    /// How compensation is divided
    pub distribution: CompensationDistribution,
    /// Payment schedule
    pub schedule: PaymentSchedule,
    /// Currency/token
    pub currency: String,
    /// Minimum payout
    pub min_payout: u64,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum CompensationDistribution {
    /// Equal split among team
    Equal,
    /// Based on role allocation
    RoleBased,
    /// Based on tracked contributions
    ContributionBased { algorithm: String },
    /// Fixed amounts per role
    FixedPerRole(HashMap<String, u64>),
    /// Hybrid: base + contribution bonus
    Hybrid { base_percentage: f64 },
    /// Team decides at end
    TeamDecision,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum PaymentSchedule {
    /// On project completion
    OnCompletion,
    /// Per milestone
    PerMilestone,
    /// Regular intervals
    Periodic { interval_days: u32 },
    /// Continuous streaming
    Streaming,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Milestone {
    pub milestone_id: String,
    pub title: String,
    pub description: String,
    pub deliverables: Vec<Deliverable>,
    pub deadline: Timestamp,
    pub budget_allocation: f64, // Percentage
    pub status: MilestoneStatus,
    pub review_criteria: Vec<ReviewCriterion>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Deliverable {
    pub name: String,
    pub description: String,
    pub deliverable_type: DeliverableType,
    pub required: bool,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum DeliverableType {
    Document,
    Code { repository: Option<String> },
    Design,
    Video,
    Data,
    Event,
    Report,
    Other(String),
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum MilestoneStatus {
    NotStarted,
    InProgress,
    InReview,
    Approved,
    Rejected { reason: String },
    Paid,
}

/// Team member in a project
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TeamMember {
    /// Project
    pub project: ActionHash,
    /// Member
    pub member: AgentPubKey,
    /// Role in project
    pub role: String,
    /// Joined at
    pub joined_at: Timestamp,
    /// Status
    pub status: MemberStatus,
    /// Verified skills
    pub verified_skills: Vec<ActionHash>, // Praxis credentials
    /// Reputation at join time
    pub reputation_at_join: f64,
    /// Contribution score (updated)
    pub contribution_score: f64,
    /// Custom terms (if any)
    pub custom_terms: Option<String>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum MemberStatus {
    Pending,
    Active,
    OnLeave,
    Removed,
    Left,
    Completed,
}

/// A recorded contribution
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Contribution {
    /// Project
    pub project: ActionHash,
    /// Contributor
    pub contributor: AgentPubKey,
    /// Task (if task-based)
    pub task: Option<ActionHash>,
    /// Contribution type
    pub contribution_type: ContributionType,
    /// Description
    pub description: String,
    /// Evidence
    pub evidence: Vec<ContributionEvidence>,
    /// Self-reported hours
    pub hours_claimed: Option<f64>,
    /// Timestamp
    pub contributed_at: Timestamp,
    /// Verification status
    pub verification: VerificationStatus,
    /// Value score (calculated)
    pub value_score: f64,
    /// Skills demonstrated
    pub skills_demonstrated: Vec<String>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ContributionType {
    Code { commit_hash: Option<String>, lines_added: u32, lines_removed: u32 },
    Documentation,
    Design { asset_type: String },
    Research,
    Management,
    Communication,
    Testing,
    Review,
    Mentoring,
    Other(String),
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ContributionEvidence {
    pub evidence_type: String,
    pub reference: String, // URL, hash, etc.
    pub description: String,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum VerificationStatus {
    SelfReported,
    PeerVerified { verifier: AgentPubKey },
    AutoVerified { method: String },
    Disputed { dispute: ActionHash },
    Rejected { reason: String },
}

/// A task within a project
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Task {
    pub task_id: String,
    pub project: ActionHash,
    pub milestone: Option<ActionHash>,
    pub title: String,
    pub description: String,
    pub status: TaskStatus,
    pub assignee: Option<AgentPubKey>,
    pub created_by: AgentPubKey,
    pub created_at: Timestamp,
    pub deadline: Option<Timestamp>,
    pub priority: Priority,
    pub estimated_hours: Option<f64>,
    pub actual_hours: Option<f64>,
    pub dependencies: Vec<ActionHash>,
    pub skills_required: Vec<String>,
    pub value_points: u32,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum TaskStatus {
    Open,
    Assigned,
    InProgress,
    InReview,
    Completed,
    Blocked,
    Cancelled,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
    Optional,
}

/// Peer review of work
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PeerReview {
    /// What is being reviewed
    pub subject: ReviewSubject,
    /// Reviewer
    pub reviewer: AgentPubKey,
    /// Rating
    pub rating: ReviewRating,
    /// Detailed feedback
    pub feedback: String,
    /// Criteria scores
    pub criteria_scores: HashMap<String, u32>,
    /// Reviewed at
    pub reviewed_at: Timestamp,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ReviewSubject {
    Contribution(ActionHash),
    Milestone(ActionHash),
    Task(ActionHash),
    TeamMember(AgentPubKey),
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ReviewRating {
    Outstanding, // 5
    ExceedsExpectations, // 4
    MeetsExpectations, // 3
    BelowExpectations, // 2
    Unacceptable, // 1
}
```

---

## Zome Specifications

### 1. Project Manager Zome

```rust
// project_manager/src/lib.rs

/// Create a new project
#[hdk_extern]
pub fn create_project(input: CreateProjectInput) -> ExternResult<ActionHash> {
    let creator = agent_info()?.agent_latest_pubkey;

    // Validate required skills exist in Praxis
    for skill in &input.required_skills {
        verify_skill_exists(&skill.skill_id)?;
    }

    // Create milestones
    let milestones: Vec<Milestone> = input.milestones
        .iter()
        .map(|m| Milestone {
            milestone_id: generate_milestone_id(),
            title: m.title.clone(),
            description: m.description.clone(),
            deliverables: m.deliverables.clone(),
            deadline: m.deadline,
            budget_allocation: m.budget_allocation,
            status: MilestoneStatus::NotStarted,
            review_criteria: m.review_criteria.clone(),
        })
        .collect();

    let project = Project {
        project_id: generate_project_id(),
        title: input.title,
        description: input.description,
        project_type: input.project_type,
        status: ProjectStatus::Draft,
        creator: creator.clone(),
        team_config: input.team_config,
        compensation: input.compensation,
        required_skills: input.required_skills,
        milestones,
        budget: input.budget,
        timeline: input.timeline,
        governance: input.governance.unwrap_or_default(),
        created_at: sys_time()?,
        tags: input.tags,
    };

    let hash = create_entry(&EntryTypes::Project(project.clone()))?;

    // Creator automatically joins as team lead
    join_project_internal(&hash, &creator, "project_lead".to_string())?;

    // Index by type and tags
    create_project_indexes(&hash, &project)?;

    Ok(hash)
}

/// Open project for recruitment
#[hdk_extern]
pub fn open_for_recruitment(project_hash: ActionHash) -> ExternResult<()> {
    let caller = agent_info()?.agent_latest_pubkey;

    let mut project = get_project(&project_hash)?;

    // Verify caller is creator
    if project.creator != caller {
        return Err(WasmError::Guest("Only creator can open recruitment".into()));
    }

    // Verify project has funding if required
    if project.budget.is_some() {
        verify_funding_escrowed(&project_hash)?;
    }

    project.status = ProjectStatus::Recruiting;
    update_project(&project_hash, &project)?;

    // Post to Marketplace for visibility
    if project.budget.is_some() {
        bridge_call::<()>(
            "marketplace",
            "list_project_opportunity",
            ProjectListing {
                project: project_hash.clone(),
                title: project.title.clone(),
                skills_needed: project.required_skills.clone(),
                budget: project.budget.clone(),
            },
        )?;
    }

    Ok(())
}

/// Start the project (after team is formed)
#[hdk_extern]
pub fn start_project(project_hash: ActionHash) -> ExternResult<()> {
    let caller = agent_info()?.agent_latest_pubkey;

    let mut project = get_project(&project_hash)?;

    // Verify caller has authority
    verify_project_authority(&caller, &project)?;

    // Check minimum team size
    let team = get_project_team(&project_hash)?;
    if team.len() < project.team_config.min_size as usize {
        return Err(WasmError::Guest("Team too small to start".into()));
    }

    // Check required roles are filled
    for role in &project.team_config.roles {
        let role_count = team.iter().filter(|m| m.role == role.name).count();
        if role_count < role.count_needed as usize {
            return Err(WasmError::Guest(format!("Role {} not filled", role.name)));
        }
    }

    project.status = ProjectStatus::Active;
    project.timeline.started_at = Some(sys_time()?);

    update_project(&project_hash, &project)?;

    // Notify all team members
    for member in &team {
        send_signal_to_agent(&member.member, Signal::ProjectStarted(project_hash.clone()))?;
    }

    Ok(())
}
```

### 2. Team Manager Zome

```rust
// team_manager/src/lib.rs

/// Apply to join a project
#[hdk_extern]
pub fn apply_to_project(input: ApplyInput) -> ExternResult<ActionHash> {
    let applicant = agent_info()?.agent_latest_pubkey;

    let project = get_project(&input.project)?;

    // Verify project is recruiting
    if project.status != ProjectStatus::Recruiting {
        return Err(WasmError::Guest("Project not recruiting".into()));
    }

    // Verify role exists
    let role = project.team_config.roles
        .iter()
        .find(|r| r.name == input.role)
        .ok_or(WasmError::Guest("Role not found".into()))?;

    // Verify applicant meets requirements
    verify_role_requirements(&applicant, role)?;

    // Check if role is still available
    let team = get_project_team(&input.project)?;
    let role_count = team.iter().filter(|m| m.role == input.role).count();
    if role_count >= role.count_needed as usize {
        return Err(WasmError::Guest("Role is full".into()));
    }

    // Create application
    let application = TeamApplication {
        project: input.project,
        applicant: applicant.clone(),
        role: input.role,
        cover_letter: input.cover_letter,
        relevant_credentials: input.credentials,
        portfolio: input.portfolio,
        applied_at: sys_time()?,
        status: ApplicationStatus::Pending,
    };

    let hash = create_entry(&EntryTypes::TeamApplication(application))?;

    // Process based on selection method
    match project.team_config.selection {
        TeamSelection::FirstCome => {
            // Auto-approve if qualified
            approve_application(&hash)?;
        }
        TeamSelection::Automatic => {
            // Check credentials automatically
            if verify_automatic_eligibility(&applicant, role)? {
                approve_application(&hash)?;
            }
        }
        _ => {
            // Notify project creator / team for review
            send_signal_to_agent(&project.creator, Signal::NewApplication(hash.clone()))?;
        }
    }

    Ok(hash)
}

/// Approve a team application
#[hdk_extern]
pub fn approve_application(application_hash: ActionHash) -> ExternResult<()> {
    let caller = agent_info()?.agent_latest_pubkey;

    let mut application = get_application(&application_hash)?;
    let project = get_project(&application.project)?;

    // Verify caller has approval authority
    match project.team_config.selection {
        TeamSelection::CreatorApproval => {
            if caller != project.creator {
                return Err(WasmError::Guest("Only creator can approve".into()));
            }
        }
        TeamSelection::TeamVote { threshold } => {
            // Check if vote threshold met
            let approvals = get_application_votes(&application_hash)?;
            let team = get_project_team(&application.project)?;
            let approval_rate = approvals.len() as f64 / team.len() as f64;
            if approval_rate < threshold {
                // Record this vote instead
                record_application_vote(&application_hash, &caller, true)?;
                return Ok(());
            }
        }
        _ => {}
    }

    // Create team member entry
    let member = TeamMember {
        project: application.project.clone(),
        member: application.applicant.clone(),
        role: application.role.clone(),
        joined_at: sys_time()?,
        status: MemberStatus::Active,
        verified_skills: application.relevant_credentials.clone(),
        reputation_at_join: get_reputation(&application.applicant)?,
        contribution_score: 0.0,
        custom_terms: None,
    };

    create_entry(&EntryTypes::TeamMember(member))?;

    // Update application status
    application.status = ApplicationStatus::Approved;
    update_application(&application_hash, &application)?;

    // Notify applicant
    send_signal_to_agent(
        &application.applicant,
        Signal::ApplicationApproved(application.project.clone()),
    )?;

    Ok(())
}

/// Leave a project
#[hdk_extern]
pub fn leave_project(project_hash: ActionHash) -> ExternResult<()> {
    let member_agent = agent_info()?.agent_latest_pubkey;

    let project = get_project(&project_hash)?;

    // Check if leaving is allowed
    if !project.team_config.allow_leave && project.status == ProjectStatus::Active {
        return Err(WasmError::Guest("Cannot leave active project".into()));
    }

    // Get member record
    let mut member = get_team_member(&project_hash, &member_agent)?;

    // Update status
    member.status = MemberStatus::Left;
    update_team_member(&member)?;

    // Reputation impact for leaving
    let impact = if project.status == ProjectStatus::Active {
        -0.02 // Negative for leaving active project
    } else {
        0.0
    };

    bridge_call::<()>(
        "bridge",
        "apply_reputation_impact",
        ReputationImpact {
            agent: member_agent.clone(),
            happ: "collab".to_string(),
            impact,
            reason: "Left project".to_string(),
        },
    )?;

    // Notify project
    notify_project_team(&project_hash, Signal::MemberLeft(member_agent))?;

    Ok(())
}
```

### 3. Contribution Tracker Zome

```rust
// contribution_tracker/src/lib.rs

/// Record a contribution
#[hdk_extern]
pub fn record_contribution(input: ContributionInput) -> ExternResult<ActionHash> {
    let contributor = agent_info()?.agent_latest_pubkey;

    // Verify contributor is team member
    let member = get_team_member(&input.project, &contributor)?;
    if member.status != MemberStatus::Active {
        return Err(WasmError::Guest("Not an active team member".into()));
    }

    // Validate evidence
    for evidence in &input.evidence {
        validate_evidence(evidence)?;
    }

    // Calculate initial value score
    let value_score = calculate_contribution_value(&input)?;

    let contribution = Contribution {
        project: input.project,
        contributor: contributor.clone(),
        task: input.task,
        contribution_type: input.contribution_type,
        description: input.description,
        evidence: input.evidence,
        hours_claimed: input.hours_claimed,
        contributed_at: sys_time()?,
        verification: VerificationStatus::SelfReported,
        value_score,
        skills_demonstrated: input.skills,
    };

    let hash = create_entry(&EntryTypes::Contribution(contribution.clone()))?;

    // If task-based, update task
    if let Some(task_hash) = &input.task {
        add_contribution_to_task(task_hash, &hash)?;
    }

    // Update member's contribution score
    update_member_contribution_score(&input.project, &contributor, value_score)?;

    // Auto-verify if possible (e.g., git commits)
    if can_auto_verify(&contribution) {
        auto_verify_contribution(&hash, &contribution)?;
    }

    Ok(hash)
}

/// Peer verify a contribution
#[hdk_extern]
pub fn verify_contribution(input: VerifyContributionInput) -> ExternResult<()> {
    let verifier = agent_info()?.agent_latest_pubkey;

    let mut contribution = get_contribution(&input.contribution)?;

    // Verify verifier is team member
    let verifier_member = get_team_member(&contribution.project, &verifier)?;
    if verifier_member.status != MemberStatus::Active {
        return Err(WasmError::Guest("Not an active team member".into()));
    }

    // Can't verify own contribution
    if contribution.contributor == verifier {
        return Err(WasmError::Guest("Cannot verify own contribution".into()));
    }

    // Update verification status
    contribution.verification = VerificationStatus::PeerVerified { verifier: verifier.clone() };

    // Adjust value score based on verification
    if input.accurate {
        contribution.value_score *= 1.1; // Bonus for verified
    } else {
        contribution.value_score *= 0.5; // Penalty for disputed
    }

    update_contribution(&input.contribution, &contribution)?;

    Ok(())
}

/// Dispute a contribution
#[hdk_extern]
pub fn dispute_contribution(input: DisputeContributionInput) -> ExternResult<ActionHash> {
    let disputer = agent_info()?.agent_latest_pubkey;

    let mut contribution = get_contribution(&input.contribution)?;

    // Verify disputer is team member
    get_team_member(&contribution.project, &disputer)?;

    // File dispute with Arbiter
    let dispute = bridge_call::<ActionHash>(
        "arbiter",
        "file_dispute",
        DisputeInput {
            source_happ: "collab".to_string(),
            source_reference: input.contribution.clone(),
            claimant: disputer,
            respondent: contribution.contributor.clone(),
            dispute_type: DisputeType::ContributionDispute,
            claim: input.reason,
            remedy_sought: input.remedy,
            stake_value: StakeValue {
                reputation: contribution.value_score * 0.01,
                ..Default::default()
            },
        },
    )?;

    // Update contribution status
    contribution.verification = VerificationStatus::Disputed { dispute: dispute.clone() };
    update_contribution(&input.contribution, &contribution)?;

    Ok(dispute)
}

/// Calculate contribution value
fn calculate_contribution_value(input: &ContributionInput) -> f64 {
    let base_value = match &input.contribution_type {
        ContributionType::Code { lines_added, lines_removed, .. } => {
            // Lines changed, weighted
            (*lines_added as f64 * 0.1 + *lines_removed as f64 * 0.05).min(100.0)
        }
        ContributionType::Documentation => 10.0,
        ContributionType::Design { .. } => 20.0,
        ContributionType::Research => 15.0,
        ContributionType::Management => 10.0,
        ContributionType::Review => 5.0,
        ContributionType::Mentoring => 15.0,
        _ => 5.0,
    };

    // Adjust for hours claimed
    let hours_multiplier = input.hours_claimed
        .map(|h| (h / 4.0).min(2.0).max(0.5))
        .unwrap_or(1.0);

    // Adjust for evidence quality
    let evidence_multiplier = if input.evidence.is_empty() {
        0.5
    } else {
        1.0 + (input.evidence.len() as f64 * 0.1).min(0.5)
    };

    base_value * hours_multiplier * evidence_multiplier
}

/// Auto-verify if possible
fn can_auto_verify(contribution: &Contribution) -> bool {
    match &contribution.contribution_type {
        ContributionType::Code { commit_hash: Some(_), .. } => true,
        _ => false,
    }
}

fn auto_verify_contribution(hash: &ActionHash, contribution: &Contribution) -> ExternResult<()> {
    if let ContributionType::Code { commit_hash: Some(commit), .. } = &contribution.contribution_type {
        // Verify commit exists and matches contributor
        // This would query a git integration service
        let verified = verify_git_commit(commit, &contribution.contributor)?;

        if verified {
            let mut updated = contribution.clone();
            updated.verification = VerificationStatus::AutoVerified {
                method: "git_commit_signature".to_string(),
            };
            update_contribution(hash, &updated)?;
        }
    }

    Ok(())
}
```

### 4. Milestone Manager Zome

```rust
// milestone_manager/src/lib.rs

/// Submit milestone for review
#[hdk_extern]
pub fn submit_milestone(input: MilestoneSubmissionInput) -> ExternResult<ActionHash> {
    let submitter = agent_info()?.agent_latest_pubkey;

    let project = get_project(&input.project)?;

    // Verify submitter has authority
    verify_project_authority(&submitter, &project)?;

    // Get milestone
    let milestone = project.milestones
        .iter()
        .find(|m| m.milestone_id == input.milestone_id)
        .ok_or(WasmError::Guest("Milestone not found".into()))?;

    // Verify all deliverables are present
    for deliverable in &milestone.deliverables {
        if deliverable.required {
            if !input.deliverables.iter().any(|d| d.name == deliverable.name) {
                return Err(WasmError::Guest(
                    format!("Missing required deliverable: {}", deliverable.name)
                ));
            }
        }
    }

    // Create submission
    let submission = MilestoneSubmission {
        project: input.project,
        milestone_id: input.milestone_id,
        submitted_by: submitter,
        submitted_at: sys_time()?,
        deliverables: input.deliverables,
        notes: input.notes,
        status: SubmissionStatus::Pending,
    };

    let hash = create_entry(&EntryTypes::MilestoneSubmission(submission))?;

    // Trigger review process
    initiate_milestone_review(&hash, &project)?;

    Ok(hash)
}

/// Review and approve milestone
#[hdk_extern]
pub fn review_milestone(input: MilestoneReviewInput) -> ExternResult<()> {
    let reviewer = agent_info()?.agent_latest_pubkey;

    let submission = get_milestone_submission(&input.submission)?;
    let project = get_project(&submission.project)?;

    // Verify reviewer has authority (client, or governance)
    verify_review_authority(&reviewer, &project)?;

    if input.approved {
        // Mark milestone as approved
        update_milestone_status(&submission.project, &submission.milestone_id, MilestoneStatus::Approved)?;

        // Trigger payment
        if let Some(budget) = &project.budget {
            let milestone = project.milestones
                .iter()
                .find(|m| m.milestone_id == submission.milestone_id)
                .unwrap();

            let payout = (budget.total as f64 * milestone.budget_allocation) as u64;
            trigger_milestone_payout(&submission.project, payout)?;
        }

        // Update contributor reputation
        let contributions = get_milestone_contributions(&submission.project, &submission.milestone_id)?;
        for contribution in contributions {
            bridge_call::<()>(
                "bridge",
                "apply_reputation_impact",
                ReputationImpact {
                    agent: contribution.contributor.clone(),
                    happ: "collab".to_string(),
                    impact: 0.01,
                    reason: "Milestone completed".to_string(),
                },
            )?;
        }
    } else {
        // Reject with feedback
        update_milestone_status(
            &submission.project,
            &submission.milestone_id,
            MilestoneStatus::Rejected { reason: input.feedback.unwrap_or_default() },
        )?;
    }

    Ok(())
}

/// Calculate payout distribution for milestone
fn calculate_payout_distribution(
    project: &Project,
    milestone_id: &str,
    total_payout: u64,
) -> ExternResult<Vec<(AgentPubKey, u64)>> {
    let contributions = get_milestone_contributions(&project.project_id.parse()?, milestone_id)?;
    let team = get_project_team(&project.project_id.parse()?)?;

    match &project.compensation.distribution {
        CompensationDistribution::Equal => {
            let per_member = total_payout / team.len() as u64;
            Ok(team.iter().map(|m| (m.member.clone(), per_member)).collect())
        }

        CompensationDistribution::ContributionBased { .. } => {
            // Calculate based on contribution scores
            let total_score: f64 = contributions.iter().map(|c| c.value_score).sum();

            Ok(contributions
                .iter()
                .map(|c| {
                    let share = if total_score > 0.0 {
                        c.value_score / total_score
                    } else {
                        1.0 / contributions.len() as f64
                    };
                    (c.contributor.clone(), (total_payout as f64 * share) as u64)
                })
                .collect())
        }

        CompensationDistribution::RoleBased => {
            // Based on role allocations
            let mut payouts: HashMap<AgentPubKey, u64> = HashMap::new();

            for member in &team {
                let role = project.team_config.roles
                    .iter()
                    .find(|r| r.name == member.role);

                if let Some(r) = role {
                    let role_total = (total_payout as f64 * r.compensation_share) as u64;
                    let role_members: Vec<_> = team.iter()
                        .filter(|m| m.role == r.name)
                        .collect();
                    let per_member = role_total / role_members.len() as u64;
                    payouts.insert(member.member.clone(), per_member);
                }
            }

            Ok(payouts.into_iter().collect())
        }

        CompensationDistribution::Hybrid { base_percentage } => {
            // Base + contribution bonus
            let base_pool = (total_payout as f64 * base_percentage) as u64;
            let bonus_pool = total_payout - base_pool;

            let per_member_base = base_pool / team.len() as u64;

            let total_score: f64 = contributions.iter().map(|c| c.value_score).sum();

            let mut payouts: HashMap<AgentPubKey, u64> = HashMap::new();

            for member in &team {
                payouts.insert(member.member.clone(), per_member_base);
            }

            for contribution in &contributions {
                let bonus = if total_score > 0.0 {
                    (bonus_pool as f64 * (contribution.value_score / total_score)) as u64
                } else {
                    0
                };
                *payouts.entry(contribution.contributor.clone()).or_insert(0) += bonus;
            }

            Ok(payouts.into_iter().collect())
        }

        _ => {
            // Default to equal
            let per_member = total_payout / team.len() as u64;
            Ok(team.iter().map(|m| (m.member.clone(), per_member)).collect())
        }
    }
}
```

---

## Cross-hApp Integration

### Praxis Integration
```rust
// Issue skill credential based on project work
if project.status == ProjectStatus::Completed {
    for member in &team {
        let demonstrated_skills = aggregate_demonstrated_skills(&member.member, &project_hash)?;

        for skill in demonstrated_skills {
            bridge_call::<ActionHash>(
                "praxis",
                "issue_skill_endorsement",
                SkillEndorsement {
                    recipient: member.member.clone(),
                    skill: skill.clone(),
                    evidence: vec![project_hash.clone()],
                    endorser: project.creator.clone(),
                },
            )?;
        }
    }
}
```

### Marketplace Integration
```rust
// Fund project via Marketplace
let funding = bridge_call::<FundingResult>(
    "marketplace",
    "create_project_funding",
    ProjectFunding {
        project: project_hash,
        amount: budget.total,
        currency: budget.currency.clone(),
        milestones: project.milestones.iter()
            .map(|m| (m.milestone_id.clone(), m.budget_allocation))
            .collect(),
    },
)?;
```

---

## Success Metrics

| Metric | 1 Year | 3 Years |
|--------|--------|---------|
| Projects created | 500 | 10,000 |
| Team members | 2,000 | 50,000 |
| Contributions tracked | 50,000 | 2,000,000 |
| Total payout distributed | $100K | $10M |
| Skills credentials issued | 1,000 | 50,000 |
| Dispute rate | <5% | <2% |

---

*"Collaboration is the art of aligning individual genius with collective purpose."*
