// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Knowledge Roots Integrity Zome
//!
//! Defines the decentralized curriculum graph - a community-built knowledge structure.
//!
//! ## What are Knowledge Roots?
//!
//! Knowledge Roots is a decentralized prerequisite graph that maps learning pathways.
//! Unlike traditional curricula controlled by institutions, Knowledge Roots is:
//!
//! - **Community-Built**: Anyone can propose connections between concepts
//! - **DAO-Governed**: Community votes on curriculum structure changes
//! - **AI-Enhanced**: ML models suggest optimal learning paths
//! - **Skill-Aligned**: Maps to industry skill frameworks (ESCO, O*NET, etc.)
//! - **Credential-Gated**: Advanced topics unlock with demonstrated mastery
//!
//! ## Core Concepts
//!
//! - **Knowledge Node**: A concept, skill, or topic that can be learned
//! - **Learning Edge**: A relationship between nodes (prerequisite, enhances, etc.)
//! - **Learning Path**: A sequence of nodes representing a curriculum
//! - **Skill Tree**: A visual representation of related knowledge areas

use hdi::prelude::*;

/// Types of knowledge nodes
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    /// A fundamental concept (e.g., "Variables in Programming")
    Concept,
    /// A practical skill (e.g., "Write unit tests")
    Skill,
    /// A knowledge topic (e.g., "Machine Learning")
    Topic,
    /// A course or module (e.g., "Introduction to Python")
    Course,
    /// An assessment/certification (e.g., "Python Developer Certification")
    Assessment,
    /// A project or practical application
    Project,
}

impl Default for NodeType {
    fn default() -> Self {
        NodeType::Concept
    }
}

/// Difficulty levels
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifficultyLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

impl Default for DifficultyLevel {
    fn default() -> Self {
        DifficultyLevel::Beginner
    }
}

/// Grade level for PreK through post-doctoral
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradeLevel {
    PreK,
    Kindergarten,
    Grade1, Grade2, Grade3, Grade4, Grade5, Grade6,
    Grade7, Grade8, Grade9, Grade10, Grade11, Grade12,
    College,
    /// Undergraduate (associate/bachelor's degree)
    Undergraduate,
    /// Graduate (master's degree)
    Graduate,
    /// Doctoral (PhD)
    Doctoral,
    /// Post-doctoral research
    PostDoctoral,
    /// Professional degree (JD, MD, PharmD, etc.)
    Professional,
    Adult,
}

impl GradeLevel {
    /// Numeric value for ordering (PreK=0, K=1, Grade1=2, ..., Grade12=13)
    pub fn ordinal(&self) -> u8 {
        match self {
            GradeLevel::PreK => 0,
            GradeLevel::Kindergarten => 1,
            GradeLevel::Grade1 => 2,
            GradeLevel::Grade2 => 3,
            GradeLevel::Grade3 => 4,
            GradeLevel::Grade4 => 5,
            GradeLevel::Grade5 => 6,
            GradeLevel::Grade6 => 7,
            GradeLevel::Grade7 => 8,
            GradeLevel::Grade8 => 9,
            GradeLevel::Grade9 => 10,
            GradeLevel::Grade10 => 11,
            GradeLevel::Grade11 => 12,
            GradeLevel::Grade12 => 13,
            GradeLevel::College => 14,
            GradeLevel::Undergraduate => 15,
            GradeLevel::Graduate => 16,
            GradeLevel::Doctoral => 17,
            GradeLevel::PostDoctoral => 18,
            GradeLevel::Professional => 19,
            GradeLevel::Adult => 20,
        }
    }

    /// Age range for this grade level
    pub fn age_range(&self) -> (u8, u8) {
        match self {
            GradeLevel::PreK => (3, 5),
            GradeLevel::Kindergarten => (5, 6),
            GradeLevel::Grade1 => (6, 7),
            GradeLevel::Grade2 => (7, 8),
            GradeLevel::Grade3 => (8, 9),
            GradeLevel::Grade4 => (9, 10),
            GradeLevel::Grade5 => (10, 11),
            GradeLevel::Grade6 => (11, 12),
            GradeLevel::Grade7 => (12, 13),
            GradeLevel::Grade8 => (13, 14),
            GradeLevel::Grade9 => (14, 15),
            GradeLevel::Grade10 => (15, 16),
            GradeLevel::Grade11 => (16, 17),
            GradeLevel::Grade12 => (17, 18),
            GradeLevel::College => (18, 22),
            GradeLevel::Undergraduate => (18, 22),
            GradeLevel::Graduate => (22, 26),
            GradeLevel::Doctoral => (22, 30),
            GradeLevel::PostDoctoral => (28, 40),
            GradeLevel::Professional => (22, 30),
            GradeLevel::Adult => (18, 99),
        }
    }
}

impl Default for GradeLevel {
    fn default() -> Self { GradeLevel::Adult }
}

/// Bloom's Taxonomy level
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BloomLevel {
    Remember,
    Understand,
    Apply,
    Analyze,
    Evaluate,
    Create,
}

impl Default for BloomLevel {
    fn default() -> Self { BloomLevel::Understand }
}

/// Academic standards framework
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StandardFramework {
    CommonCore,
    NGSS,
    StateSpecific(String),
    InternationalBaccalaureate,
    Cambridge,
    /// ACM/IEEE Computing Curricula (e.g., "CS2013", "CC2020")
    ACM(String),
    /// ABET accreditation criteria (e.g., "EAC", "CAC")
    ABET(String),
    /// NCES Classification of Instructional Programs
    CIP,
    /// Professional certification (e.g., "CompTIA-Security+", "AWS-SAA")
    ProfessionalCertification(String),
    Custom(String),
}

/// Standards alignment for a knowledge node
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcademicStandard {
    pub framework: StandardFramework,
    pub code: String,
    pub description: String,
    pub grade_level: GradeLevel,
}

/// Subject area classification
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubjectArea {
    Mathematics,
    EnglishLanguageArts,
    Science,
    SocialStudies,
    ForeignLanguage,
    Arts,
    PhysicalEducation,
    Technology,
    Custom(String),
}

impl Default for SubjectArea {
    fn default() -> Self { SubjectArea::Custom("General".to_string()) }
}

/// Knowledge Node - a single concept/skill in the knowledge graph
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct KnowledgeNode {
    /// Human-readable title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Type of knowledge node
    pub node_type: NodeType,
    /// Difficulty level
    pub difficulty: DifficultyLevel,
    /// Domain/category (e.g., "Programming", "Data Science")
    pub domain: String,
    /// Sub-domain (e.g., "Python", "Neural Networks")
    pub subdomain: Option<String>,
    /// Tags for search and categorization
    pub tags: Vec<String>,
    /// Estimated time to learn (in hours)
    pub estimated_hours: u32,
    /// External skill framework alignments (ESCO, O*NET codes)
    pub skill_alignments: Vec<SkillAlignment>,
    /// Related courses (action hashes)
    pub related_courses: Vec<ActionHash>,
    /// Creator of this node
    pub creator: AgentPubKey,
    /// Current status
    pub status: NodeStatus,
    /// Creation timestamp
    pub created_at: i64,
    /// Last modified timestamp
    pub modified_at: i64,
    /// Version number (for updates)
    pub version: u32,
    /// Grade levels this node targets
    #[serde(default)]
    pub grade_levels: Vec<GradeLevel>,
    /// Bloom's taxonomy level
    #[serde(default)]
    pub bloom_level: Option<BloomLevel>,
    /// Subject area
    #[serde(default)]
    pub subject_area: Option<SubjectArea>,
    /// Academic standards this node aligns to
    #[serde(default)]
    pub academic_standards: Vec<AcademicStandard>,
}

/// External skill framework alignment
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillAlignment {
    /// Framework name (e.g., "ESCO", "O*NET", "SFIA")
    pub framework: String,
    /// Code or identifier in the framework
    pub code: String,
    /// Name in the framework
    pub name: String,
    /// Confidence of alignment as permille (0-1000 representing 0.0-1.0)
    /// Use confidence_permille / 1000.0 to get the float value
    pub confidence_permille: u16,
}

/// Node status in the knowledge graph
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Proposed, awaiting community review
    Proposed,
    /// Approved and active
    Active,
    /// Deprecated (superseded by another node)
    Deprecated,
    /// Archived (no longer relevant)
    Archived,
}

impl Default for NodeStatus {
    fn default() -> Self {
        NodeStatus::Proposed
    }
}

/// Types of edges between knowledge nodes
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Target requires completion of source (strict prerequisite)
    Requires,
    /// Target is enhanced by knowledge of source (soft prerequisite)
    Recommends,
    /// Nodes cover related topics
    RelatedTo,
    /// Target is a part of source (composition)
    PartOf,
    /// Source leads to target in a sequence
    LeadsTo,
    /// Nodes are alternatives (choose one)
    AlternativeTo,
    /// Source builds upon target (specialization)
    Specializes,
    /// Source is applied in target (practical application)
    AppliedIn,
}

impl Default for EdgeType {
    fn default() -> Self {
        EdgeType::Recommends
    }
}

/// Learning Edge - a relationship between two knowledge nodes
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct LearningEdge {
    /// Source node (from)
    pub source_node: ActionHash,
    /// Target node (to)
    pub target_node: ActionHash,
    /// Type of relationship
    pub edge_type: EdgeType,
    /// Strength of the relationship as permille (0-1000 for 0.0-1.0)
    pub strength_permille: u16,
    /// Description of why this connection exists
    pub rationale: String,
    /// Who proposed this edge
    pub proposer: AgentPubKey,
    /// Current status
    pub status: EdgeStatus,
    /// Community votes (for governance)
    pub upvotes: u32,
    pub downvotes: u32,
    /// Creation timestamp
    pub created_at: i64,
}

/// Edge status
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeStatus {
    /// Proposed, awaiting votes
    Proposed,
    /// Approved by community
    Approved,
    /// Rejected by community
    Rejected,
    /// Under dispute
    Disputed,
}

impl Default for EdgeStatus {
    fn default() -> Self {
        EdgeStatus::Proposed
    }
}

/// Learning Path - a curated sequence through the knowledge graph
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct LearningPath {
    /// Path title
    pub title: String,
    /// Path description
    pub description: String,
    /// Ordered sequence of nodes
    pub nodes: Vec<ActionHash>,
    /// Target skill/outcome
    pub target_outcome: String,
    /// Target difficulty level at completion
    pub target_level: DifficultyLevel,
    /// Total estimated hours
    pub total_hours: u32,
    /// Creator of the path
    pub creator: AgentPubKey,
    /// Whether this is an official/curated path
    pub official: bool,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Number of learners who completed this path
    pub completions: u32,
    /// Average rating as permille (0-5000 for 0.0-5.0 scale)
    pub avg_rating_permille: u16,
    /// Creation timestamp
    pub created_at: i64,
}

/// Skill Tree - a visual grouping of related knowledge
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct SkillTree {
    /// Tree name
    pub name: String,
    /// Description
    pub description: String,
    /// Root domain (e.g., "Web Development")
    pub domain: String,
    /// Tree structure (serialized for flexibility)
    pub structure: SkillTreeStructure,
    /// Creator
    pub creator: AgentPubKey,
    /// Version
    pub version: u32,
    /// Creation timestamp
    pub created_at: i64,
}

/// Skill tree structure (hierarchical)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillTreeStructure {
    /// Root nodes (entry points)
    pub roots: Vec<ActionHash>,
    /// Tiers/levels in the tree
    pub tiers: Vec<SkillTreeTier>,
}

/// A tier in the skill tree
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillTreeTier {
    /// Tier name (e.g., "Fundamentals", "Intermediate", "Advanced")
    pub name: String,
    /// Nodes at this tier
    pub nodes: Vec<ActionHash>,
    /// Order in the tree (0 = first tier)
    pub order: u32,
}

/// Learner Progress on a node - tracks individual mastery
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct NodeProgress {
    /// The knowledge node
    pub node_hash: ActionHash,
    /// The learner
    pub learner: AgentPubKey,
    /// Mastery level as permille (0-1000 for 0.0-1.0)
    pub mastery_permille: u16,
    /// Status of this node for the learner
    pub progress_status: ProgressStatus,
    /// Time spent learning (minutes)
    pub time_spent: u32,
    /// Number of attempts (for assessments)
    pub attempts: u32,
    /// Best score as permille (0-1000 for 0.0-1.0)
    pub best_score_permille: Option<u16>,
    /// Evidence of completion (credential hashes, etc.)
    pub evidence: Vec<String>,
    /// When the node was started
    pub started_at: i64,
    /// When the node was completed (if applicable)
    pub completed_at: Option<i64>,
}

/// Progress status on a node
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgressStatus {
    /// Not started
    NotStarted,
    /// Currently learning
    InProgress,
    /// Completed but not mastered
    Completed,
    /// Fully mastered
    Mastered,
    /// Locked (prerequisites not met)
    Locked,
}

impl Default for ProgressStatus {
    fn default() -> Self {
        ProgressStatus::NotStarted
    }
}

/// Edge Vote - community governance on curriculum structure
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct EdgeVote {
    /// The edge being voted on
    pub edge_hash: ActionHash,
    /// The voter
    pub voter: AgentPubKey,
    /// Vote direction
    pub vote: VoteDirection,
    /// Reasoning for the vote
    pub reason: Option<String>,
    /// Voter's expertise level in this domain (self-reported)
    pub expertise_level: DifficultyLevel,
    /// Timestamp
    pub created_at: i64,
}

/// Vote direction
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoteDirection {
    Up,
    Down,
    Abstain,
}

/// Path Recommendation - AI-suggested learning path for a learner
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct PathRecommendation {
    /// The learner this recommendation is for
    pub learner: AgentPubKey,
    /// Target skill/outcome
    pub target: String,
    /// Recommended nodes in order
    pub recommended_nodes: Vec<ActionHash>,
    /// Confidence score as permille (0-1000 for 0.0-1.0)
    pub confidence_permille: u16,
    /// Reasoning for this recommendation
    pub reasoning: String,
    /// Based on learner's current progress
    pub current_progress: Vec<ActionHash>,
    /// Estimated time to complete
    pub estimated_hours: u32,
    /// When this recommendation was generated
    pub generated_at: i64,
    /// Model/algorithm version used
    pub model_version: String,
}

// ============== Entry and Link Type Definitions ==============

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(required_validations = 3, visibility = "public")]
    KnowledgeNode(KnowledgeNode),
    #[entry_type(required_validations = 3, visibility = "public")]
    LearningEdge(LearningEdge),
    #[entry_type(required_validations = 3, visibility = "public")]
    LearningPath(LearningPath),
    #[entry_type(required_validations = 3, visibility = "public")]
    SkillTree(SkillTree),
    #[entry_type(required_validations = 1, visibility = "private")]
    NodeProgress(NodeProgress),
    #[entry_type(required_validations = 3, visibility = "public")]
    EdgeVote(EdgeVote),
    #[entry_type(required_validations = 1, visibility = "private")]
    PathRecommendation(PathRecommendation),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Path anchor for all nodes
    AllNodes,
    /// Domain -> Nodes in that domain
    DomainToNodes,
    /// Node -> Outgoing edges
    NodeToEdges,
    /// Node -> Incoming edges (prerequisites of this node)
    NodeToPrerequisites,
    /// Node -> Learner progress entries
    NodeToProgress,
    /// Learner -> Their progress entries
    LearnerToProgress,
    /// Node -> Votes on edges from this node
    NodeToVotes,
    /// Path anchor for all paths
    AllPaths,
    /// Path -> Nodes in the path
    PathToNodes,
    /// Skill Tree anchor
    AllSkillTrees,
    /// Skill Tree -> Nodes
    SkillTreeToNodes,
    /// Node -> Related courses
    NodeToCourses,
    /// Grade level -> Nodes at that grade
    GradeToNodes,
    /// Subject -> Nodes in that subject
    SubjectToNodes,
    /// Node -> Mentor profiles (for PhD milestones and advanced topics)
    NodeToMentors,
    /// ISCED-F field -> Nodes in that broad field
    IscedFieldToNodes,
    /// Career -> Nodes (links career pathway nodes to educational prerequisites)
    CareerToNodes,
}

// ============== Validation Functions ==============

/// Validate knowledge node creation
pub fn validate_create_node(node: &KnowledgeNode) -> ExternResult<ValidateCallbackResult> {
    // Title must not be empty
    if node.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Node title cannot be empty".to_string(),
        ));
    }

    // Title length limit
    if node.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Node title must be 200 characters or less".to_string(),
        ));
    }

    // Estimated hours must be reasonable
    if node.estimated_hours > 10000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Estimated hours seems unreasonable (>10000)".to_string(),
        ));
    }

    if node.description.len() > 10000 { return Ok(ValidateCallbackResult::Invalid("Node description too long (max 10000 characters)".to_string())); }
    if node.domain.len() > 200 { return Ok(ValidateCallbackResult::Invalid("Domain name too long (max 200 characters)".to_string())); }
    if let Some(ref sub) = node.subdomain { if sub.len() > 200 { return Ok(ValidateCallbackResult::Invalid("Subdomain name too long (max 200 characters)".to_string())); } }
    if node.tags.len() > 50 { return Ok(ValidateCallbackResult::Invalid("Too many tags (max 50)".to_string())); }
    for tag in &node.tags { if tag.len() > 100 { return Ok(ValidateCallbackResult::Invalid("Tag too long (max 100 characters)".to_string())); } }
    if node.related_courses.len() > 100 { return Ok(ValidateCallbackResult::Invalid("Too many related courses (max 100)".to_string())); }
    if node.skill_alignments.len() > 50 { return Ok(ValidateCallbackResult::Invalid("Too many skill alignments (max 50)".to_string())); }

    // Grade levels must not be excessive
    if node.grade_levels.len() > 21 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many grade levels (max 21)".to_string(),
        ));
    }

    // Academic standards must be bounded
    if node.academic_standards.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many academic standards (max 50)".to_string(),
        ));
    }

    for std in &node.academic_standards {
        if std.code.len() > 100 {
            return Ok(ValidateCallbackResult::Invalid(
                "Standard code too long (max 100 characters)".to_string(),
            ));
        }
        if std.description.len() > 500 {
            return Ok(ValidateCallbackResult::Invalid(
                "Standard description too long (max 500 characters)".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate learning edge creation
pub fn validate_create_edge(edge: &LearningEdge) -> ExternResult<ValidateCallbackResult> {
    // Can't create self-loops
    if edge.source_node == edge.target_node {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot create edge from node to itself".to_string(),
        ));
    }

    // Strength must be valid (0-1000 permille)
    if edge.strength_permille > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Edge strength must be between 0 and 1000 (permille)".to_string(),
        ));
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
                EntryTypes::KnowledgeNode(node) => validate_create_node(&node),
                EntryTypes::LearningEdge(edge) => validate_create_edge(&edge),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::KnowledgeNode(node) => validate_create_node(&node),
                EntryTypes::LearningEdge(edge) => validate_create_edge(&edge),
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
    fn test_node_type_default() {
        assert_eq!(NodeType::default(), NodeType::Concept);
    }

    #[test]
    fn test_difficulty_level_default() {
        assert_eq!(DifficultyLevel::default(), DifficultyLevel::Beginner);
    }

    #[test]
    fn test_edge_type_default() {
        assert_eq!(EdgeType::default(), EdgeType::Recommends);
    }

    #[test]
    fn test_node_valid() {
        let node = KnowledgeNode { title: "Rust".to_string(), description: "Learn Rust".to_string(), node_type: NodeType::Course, difficulty: DifficultyLevel::Beginner, domain: "Programming".to_string(), subdomain: None, tags: vec!["rust".to_string()], estimated_hours: 40, skill_alignments: vec![], related_courses: vec![], creator: AgentPubKey::from_raw_36(vec![0u8; 36]), status: NodeStatus::Proposed, created_at: 0, modified_at: 0, version: 1, grade_levels: vec![], bloom_level: None, subject_area: None, academic_standards: vec![] };
        assert!(matches!(validate_create_node(&node).unwrap(), ValidateCallbackResult::Valid));
    }
    #[test]
    fn test_node_description_too_long() {
        let node = KnowledgeNode { title: "T".to_string(), description: "x".repeat(10001), node_type: NodeType::Concept, difficulty: DifficultyLevel::Beginner, domain: "T".to_string(), subdomain: None, tags: vec![], estimated_hours: 10, skill_alignments: vec![], related_courses: vec![], creator: AgentPubKey::from_raw_36(vec![0u8; 36]), status: NodeStatus::Proposed, created_at: 0, modified_at: 0, version: 1, grade_levels: vec![], bloom_level: None, subject_area: None, academic_standards: vec![] };
        assert!(matches!(validate_create_node(&node).unwrap(), ValidateCallbackResult::Invalid(_)));
    }
    #[test]
    fn test_node_too_many_tags() {
        let node = KnowledgeNode { title: "T".to_string(), description: "T".to_string(), node_type: NodeType::Concept, difficulty: DifficultyLevel::Beginner, domain: "T".to_string(), subdomain: None, tags: (0..51).map(|i| format!("t{}", i)).collect(), estimated_hours: 10, skill_alignments: vec![], related_courses: vec![], creator: AgentPubKey::from_raw_36(vec![0u8; 36]), status: NodeStatus::Proposed, created_at: 0, modified_at: 0, version: 1, grade_levels: vec![], bloom_level: None, subject_area: None, academic_standards: vec![] };
        assert!(matches!(validate_create_node(&node).unwrap(), ValidateCallbackResult::Invalid(_)));
    }
    #[test]
    fn test_edge_self_loop() {
        let h = ActionHash::from_raw_36(vec![1u8; 36]);
        let edge = LearningEdge { source_node: h.clone(), target_node: h, edge_type: EdgeType::Requires, strength_permille: 800, rationale: "T".to_string(), proposer: AgentPubKey::from_raw_36(vec![0u8; 36]), status: EdgeStatus::Proposed, upvotes: 0, downvotes: 0, created_at: 0 };
        assert!(matches!(validate_create_edge(&edge).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_grade_level_ordinal() {
        assert_eq!(GradeLevel::PreK.ordinal(), 0);
        assert_eq!(GradeLevel::Kindergarten.ordinal(), 1);
        assert_eq!(GradeLevel::Grade3.ordinal(), 4);
        assert_eq!(GradeLevel::Grade12.ordinal(), 13);
        assert_eq!(GradeLevel::College.ordinal(), 14);
        assert_eq!(GradeLevel::Undergraduate.ordinal(), 15);
        assert_eq!(GradeLevel::Graduate.ordinal(), 16);
        assert_eq!(GradeLevel::Doctoral.ordinal(), 17);
        assert_eq!(GradeLevel::PostDoctoral.ordinal(), 18);
        assert_eq!(GradeLevel::Professional.ordinal(), 19);
        assert_eq!(GradeLevel::Adult.ordinal(), 20);
    }

    #[test]
    fn test_grade_level_age_range() {
        assert_eq!(GradeLevel::Grade3.age_range(), (8, 9));
        assert_eq!(GradeLevel::PreK.age_range(), (3, 5));
        assert_eq!(GradeLevel::College.age_range(), (18, 22));
    }

    #[test]
    fn test_bloom_level_default() {
        assert_eq!(BloomLevel::default(), BloomLevel::Understand);
    }

    #[test]
    fn test_grade_level_default() {
        assert_eq!(GradeLevel::default(), GradeLevel::Adult);
    }

    #[test]
    fn test_subject_area_default() {
        assert_eq!(SubjectArea::default(), SubjectArea::Custom("General".to_string()));
    }

    #[test]
    fn test_node_with_grade_levels_valid() {
        let node = KnowledgeNode {
            title: "Fractions".to_string(), description: "Learn fractions".to_string(),
            node_type: NodeType::Concept, difficulty: DifficultyLevel::Beginner,
            domain: "Math".to_string(), subdomain: None, tags: vec![],
            estimated_hours: 10, skill_alignments: vec![], related_courses: vec![],
            creator: AgentPubKey::from_raw_36(vec![0u8; 36]),
            status: NodeStatus::Proposed, created_at: 0, modified_at: 0, version: 1,
            grade_levels: vec![GradeLevel::Grade3, GradeLevel::Grade4, GradeLevel::Grade5],
            bloom_level: Some(BloomLevel::Understand),
            subject_area: Some(SubjectArea::Mathematics),
            academic_standards: vec![AcademicStandard {
                framework: StandardFramework::CommonCore,
                code: "3.NF.A.1".to_string(),
                description: "Understand a fraction 1/b".to_string(),
                grade_level: GradeLevel::Grade3,
            }],
        };
        assert!(matches!(validate_create_node(&node).unwrap(), ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_node_too_many_grade_levels() {
        let node = KnowledgeNode {
            title: "T".to_string(), description: "T".to_string(),
            node_type: NodeType::Concept, difficulty: DifficultyLevel::Beginner,
            domain: "T".to_string(), subdomain: None, tags: vec![],
            estimated_hours: 1, skill_alignments: vec![], related_courses: vec![],
            creator: AgentPubKey::from_raw_36(vec![0u8; 36]),
            status: NodeStatus::Proposed, created_at: 0, modified_at: 0, version: 1,
            grade_levels: vec![GradeLevel::PreK; 22],
            bloom_level: None, subject_area: None, academic_standards: vec![],
        };
        assert!(matches!(validate_create_node(&node).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_node_standard_code_too_long() {
        let node = KnowledgeNode {
            title: "T".to_string(), description: "T".to_string(),
            node_type: NodeType::Concept, difficulty: DifficultyLevel::Beginner,
            domain: "T".to_string(), subdomain: None, tags: vec![],
            estimated_hours: 1, skill_alignments: vec![], related_courses: vec![],
            creator: AgentPubKey::from_raw_36(vec![0u8; 36]),
            status: NodeStatus::Proposed, created_at: 0, modified_at: 0, version: 1,
            grade_levels: vec![], bloom_level: None, subject_area: None,
            academic_standards: vec![AcademicStandard {
                framework: StandardFramework::CommonCore,
                code: "x".repeat(101),
                description: "T".to_string(),
                grade_level: GradeLevel::Grade1,
            }],
        };
        assert!(matches!(validate_create_node(&node).unwrap(), ValidateCallbackResult::Invalid(_)));
    }
}
