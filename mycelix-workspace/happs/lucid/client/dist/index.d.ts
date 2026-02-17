import { AgentPubKey, Timestamp, ActionHash, AppClient, Record as Record$1 } from '@holochain/client';

/**
 * LUCID Type Definitions
 *
 * TypeScript types matching the Rust entry types in the LUCID hApp.
 */

/** Empirical Level (E0-E4) - Degree of empirical validation */
declare enum EmpiricalLevel {
    /** E0: Unverified - Personal intuition */
    E0 = "E0",
    /** E1: Anecdotal - Single observation */
    E1 = "E1",
    /** E2: Tested - Multiple observations */
    E2 = "E2",
    /** E3: Verified - Systematic verification */
    E3 = "E3",
    /** E4: Established - Consensus or proof */
    E4 = "E4"
}
/** Normative Level (N0-N3) - Normative/ethical coherence */
declare enum NormativeLevel {
    /** N0: Personal - Meaningful to self only */
    N0 = "N0",
    /** N1: Contested - Recognized but disputed */
    N1 = "N1",
    /** N2: Emerging - Growing acceptance */
    N2 = "N2",
    /** N3: Endorsed - Broad agreement */
    N3 = "N3"
}
/** Materiality Level (M0-M3) - Practical/material significance */
declare enum MaterialityLevel {
    /** M0: Abstract - Purely theoretical */
    M0 = "M0",
    /** M1: Potential - May have implications */
    M1 = "M1",
    /** M2: Applicable - Practical applications */
    M2 = "M2",
    /** M3: Transformative - Significant impact */
    M3 = "M3"
}
/** Harmonic Level (H0-H4) - Alignment with higher purpose */
declare enum HarmonicLevel {
    /** H0: Discordant - Potentially harmful */
    H0 = "H0",
    /** H1: Neutral - No particular alignment */
    H1 = "H1",
    /** H2: Resonant - Aligns with one harmony */
    H2 = "H2",
    /** H3: Harmonic - Aligns with multiple harmonies */
    H3 = "H3",
    /** H4: Transcendent - Serves universal flourishing */
    H4 = "H4"
}
/** Full E/N/M/H epistemic classification */
interface EpistemicClassification {
    empirical: EmpiricalLevel;
    normative: NormativeLevel;
    materiality: MaterialityLevel;
    harmonic: HarmonicLevel;
}
/** Type of thought/knowledge unit */
declare enum ThoughtType {
    Claim = "Claim",
    Note = "Note",
    Question = "Question",
    Insight = "Insight",
    Definition = "Definition",
    Prediction = "Prediction",
    Hypothesis = "Hypothesis",
    Reflection = "Reflection",
    Quote = "Quote",
    Task = "Task"
}
/** A thought - the primary knowledge unit in LUCID */
interface Thought {
    id: string;
    content: string;
    thought_type: ThoughtType;
    epistemic: EpistemicClassification;
    confidence: number;
    tags: string[];
    domain: string | null;
    related_thoughts: string[];
    source_hashes: ActionHash[];
    parent_thought: string | null;
    created_at: Timestamp;
    updated_at: Timestamp;
    version: number;
}
/** Input for creating a thought */
interface CreateThoughtInput {
    content: string;
    thought_type?: ThoughtType;
    epistemic?: EpistemicClassification;
    confidence?: number;
    tags?: string[];
    domain?: string;
    related_thoughts?: string[];
    parent_thought?: string;
}
/** Input for updating a thought */
interface UpdateThoughtInput {
    thought_id: string;
    content?: string;
    thought_type?: ThoughtType;
    epistemic?: EpistemicClassification;
    confidence?: number;
    tags?: string[];
    domain?: string;
    related_thoughts?: string[];
}
/** Search filters for thoughts */
interface SearchThoughtsInput {
    tags?: string[];
    domain?: string;
    thought_type?: ThoughtType;
    min_empirical?: EmpiricalLevel;
    min_confidence?: number;
    content_contains?: string;
    limit?: number;
    offset?: number;
}
/** A tag for organizing thoughts */
interface Tag {
    name: string;
    description: string | null;
    color: string | null;
    parent_tag: string | null;
    created_at: Timestamp;
}
/** Input for creating a tag */
interface CreateTagInput {
    name: string;
    description?: string;
    color?: string;
    parent_tag?: string;
}
/** A knowledge domain */
interface Domain {
    name: string;
    description: string | null;
    parent_domain: string | null;
    related_domains: string[];
    created_at: Timestamp;
}
/** Input for creating a domain */
interface CreateDomainInput {
    name: string;
    description?: string;
    parent_domain?: string;
    related_domains?: string[];
}
/** Type of relationship between thoughts */
declare enum RelationshipType {
    Supports = "Supports",
    Contradicts = "Contradicts",
    Implies = "Implies",
    ImpliedBy = "ImpliedBy",
    Refines = "Refines",
    ExampleOf = "ExampleOf",
    RelatedTo = "RelatedTo",
    DependsOn = "DependsOn",
    Supersedes = "Supersedes",
    RespondsTo = "RespondsTo"
}
/** A relationship between two thoughts */
interface Relationship {
    id: string;
    from_thought_id: string;
    to_thought_id: string;
    relationship_type: RelationshipType;
    strength: number;
    description: string | null;
    bidirectional: boolean;
    created_at: Timestamp;
    active: boolean;
}
/** Input for creating a relationship */
interface CreateRelationshipInput {
    from_thought_id: string;
    to_thought_id: string;
    relationship_type: RelationshipType;
    strength?: number;
    description?: string;
    bidirectional?: boolean;
}
/** Type of source */
declare enum SourceType {
    AcademicPaper = "AcademicPaper",
    Book = "Book",
    NewsArticle = "NewsArticle",
    BlogPost = "BlogPost",
    WebPage = "WebPage",
    Video = "Video",
    Audio = "Audio",
    SocialMedia = "SocialMedia",
    Conversation = "Conversation",
    PersonalExperience = "PersonalExperience",
    OfficialDocument = "OfficialDocument",
    Dataset = "Dataset",
    LucidThought = "LucidThought",
    KnowledgeBase = "KnowledgeBase",
    Other = "Other"
}
/** A source of knowledge */
interface Source {
    id: string;
    source_type: SourceType;
    title: string;
    url: string | null;
    authors: string[];
    publication_date: string | null;
    publisher: string | null;
    doi: string | null;
    isbn: string | null;
    description: string | null;
    content_hash: string | null;
    credibility: number;
    notes: string | null;
    tags: string[];
    added_at: Timestamp;
    updated_at: Timestamp;
}
/** Input for creating a source */
interface CreateSourceInput {
    source_type: SourceType;
    title: string;
    url?: string;
    authors?: string[];
    publication_date?: string;
    publisher?: string;
    doi?: string;
    isbn?: string;
    description?: string;
    credibility?: number;
    notes?: string;
    tags?: string[];
}
/** How a source relates to a thought */
declare enum CitationRelationship {
    Supports = "Supports",
    Refutes = "Refutes",
    Context = "Context",
    Origin = "Origin",
    Alternative = "Alternative",
    Methodology = "Methodology",
    Definition = "Definition",
    Reference = "Reference"
}
/** A citation linking a thought to a source */
interface Citation {
    id: string;
    thought_id: string;
    source_id: string;
    location: string | null;
    quote: string | null;
    relationship: CitationRelationship;
    strength: number;
    notes: string | null;
    created_at: Timestamp;
}
/** Input for creating a citation */
interface CreateCitationInput {
    thought_id: string;
    source_id: string;
    location?: string;
    quote?: string;
    relationship?: CitationRelationship;
    strength?: number;
    notes?: string;
}
/** A snapshot of a thought at a point in time */
interface BeliefVersion {
    thought_id: string;
    version: number;
    content: string;
    confidence: number;
    epistemic_code: string;
    change_reason: string | null;
    previous_version: ActionHash | null;
    timestamp: Timestamp;
}
/** Input for recording a version */
interface RecordVersionInput {
    thought_id: string;
    version: number;
    content: string;
    confidence: number;
    epistemic_code: string;
    change_reason?: string;
    previous_version?: ActionHash;
}
/** A knowledge graph snapshot */
interface GraphSnapshot {
    id: string;
    name: string;
    description: string | null;
    timestamp: Timestamp;
    thought_count: number;
    avg_confidence: number;
    tags: string[];
}
/** Input for creating a snapshot */
interface CreateSnapshotInput {
    name: string;
    description?: string;
    thought_count: number;
    avg_confidence: number;
    tags?: string[];
}
/** Input for getting belief at a specific time */
interface BeliefAtTimeInput {
    thought_id: string;
    timestamp: Timestamp;
}
/** Visibility level for a thought */
type Visibility = {
    type: 'Private';
} | {
    type: 'SharedWith';
    agents: AgentPubKey[];
} | {
    type: 'GroupMembers';
    group_id: string;
} | {
    type: 'Federated';
    happs: string[];
} | {
    type: 'Public';
};
/** Sharing policy for a thought */
interface SharingPolicy {
    thought_id: string;
    visibility: Visibility;
    expires_at: Timestamp | null;
    delegatable: boolean;
    created_at: Timestamp;
    updated_at: Timestamp;
}
/** Input for setting a sharing policy */
interface SetPolicyInput {
    thought_id: string;
    visibility: Visibility;
    expires_at?: Timestamp;
    delegatable?: boolean;
}
/** An access grant */
interface AccessGrant {
    id: string;
    thought_id: string;
    grantee: AgentPubKey;
    grantor: AgentPubKey;
    permissions: string[];
    expires_at: Timestamp | null;
    can_delegate: boolean;
    revoked: boolean;
    created_at: Timestamp;
}
/** Input for granting access */
interface GrantAccessInput {
    thought_id: string;
    grantee: AgentPubKey;
    permissions: string[];
    expires_at?: Timestamp;
    can_delegate?: boolean;
}
/** Input for checking access */
interface CheckAccessInput {
    thought_id: string;
    agent: AgentPubKey;
}
/** Type of contradiction */
declare enum ContradictionType {
    Logical = "Logical",
    Factual = "Factual",
    Temporal = "Temporal",
    SourceConflict = "SourceConflict",
    ConfidenceInconsistency = "ConfidenceInconsistency"
}
/** A detected contradiction */
interface Contradiction {
    id: string;
    thought_a_id: string;
    thought_b_id: string;
    contradiction_type: ContradictionType;
    severity: number;
    explanation: string;
    suggested_resolution: string | null;
    resolved: boolean;
    resolution_notes: string | null;
    detected_at: Timestamp;
    resolved_at: Timestamp | null;
}
/** Input for recording a contradiction */
interface RecordContradictionInput {
    thought_a_id: string;
    thought_b_id: string;
    contradiction_type: ContradictionType;
    severity: number;
    explanation: string;
    suggested_resolution?: string;
}
/** A coherence report */
interface CoherenceReport {
    id: string;
    thought_ids: string[];
    coherence_score: number;
    phi_estimate: number | null;
    contradiction_count: number;
    knowledge_gaps: string[];
    suggestions: string[];
    created_at: Timestamp;
}
/** Input for creating a coherence report */
interface CreateReportInput {
    thought_ids: string[];
    coherence_score: number;
    phi_estimate?: number;
    contradiction_count: number;
    knowledge_gaps: string[];
    suggestions: string[];
}
/** Type of inference */
declare enum InferenceType {
    Deduction = "Deduction",
    Induction = "Induction",
    Abduction = "Abduction",
    Analogy = "Analogy"
}
/** An inference derived from existing thoughts */
interface Inference {
    id: string;
    content: string;
    premise_ids: string[];
    inference_type: InferenceType;
    confidence: number;
    accepted: boolean;
    created_at: Timestamp;
}
/** Input for recording an inference */
interface RecordInferenceInput {
    content: string;
    premise_ids: string[];
    inference_type: InferenceType;
    confidence: number;
}
/** Statistics about the knowledge graph */
interface KnowledgeGraphStats {
    total_thoughts: number;
    by_type: [string, number][];
    by_domain: [string, number][];
    average_confidence: number;
    average_epistemic_strength: number;
    top_tags: [string, number][];
}
/** Federation status */
declare enum FederationStatus {
    Pending = "Pending",
    Active = "Active",
    Revoked = "Revoked",
    Failed = "Failed"
}
/** Federation record */
interface FederationRecord {
    thought_id: string;
    target_happ: string;
    external_id: string | null;
    status: FederationStatus;
    federated_at: Timestamp;
}
/** External reputation score */
interface ExternalReputation {
    agent: AgentPubKey;
    k_vector: [number, number, number, number, number, number, number, number];
    trust_score: number;
    source_happ: string;
    fetched_at: Timestamp;
}
/** Coherence check result from Symthaea */
interface CoherenceResult {
    coherent: boolean;
    phi_estimate: number;
    suggestions: string[];
}
/** Stance on a belief */
declare enum BeliefStance {
    StronglyAgree = "StronglyAgree",
    Agree = "Agree",
    Neutral = "Neutral",
    Disagree = "Disagree",
    StronglyDisagree = "StronglyDisagree"
}
/** A shared belief in the collective */
interface BeliefShare {
    content_hash: string;
    content: string;
    belief_type: string;
    epistemic_code: string;
    confidence: number;
    tags: string[];
    shared_at: Timestamp;
    evidence_hashes: string[];
    embedding: number[];
    stance: BeliefStance | null;
}
/** Input for sharing a belief */
interface ShareBeliefInput {
    content_hash: string;
    content: string;
    belief_type: string;
    epistemic_code: string;
    confidence: number;
    tags: string[];
    evidence_hashes: string[];
    embedding?: number[];
    stance?: BeliefStance;
}
/** Vote types for belief validation */
declare enum ValidationVoteType {
    Corroborate = "Corroborate",
    Contradict = "Contradict",
    Plausible = "Plausible",
    Implausible = "Implausible",
    Abstain = "Abstain"
}
/** A validation vote on a shared belief */
interface ValidationVote {
    belief_share_hash: ActionHash;
    vote_type: ValidationVoteType;
    evidence: string | null;
    voter_weight: number;
    voted_at: Timestamp;
}
/** Input for casting a vote */
interface CastVoteInput {
    belief_share_hash: ActionHash;
    vote_type: ValidationVoteType;
    evidence?: string;
}
/** Types of consensus */
declare enum ConsensusType {
    StrongConsensus = "StrongConsensus",
    ModerateConsensus = "ModerateConsensus",
    WeakConsensus = "WeakConsensus",
    Contested = "Contested",
    Insufficient = "Insufficient"
}
/** A consensus record for a belief */
interface ConsensusRecord {
    belief_share_hash: ActionHash;
    consensus_type: ConsensusType;
    validator_count: number;
    agreement_score: number;
    summary: string;
    reached_at: Timestamp;
}
/** Types of emergent patterns */
declare enum PatternType {
    Convergence = "Convergence",
    Divergence = "Divergence",
    Trend = "Trend",
    Cluster = "Cluster",
    ContradictionCluster = "ContradictionCluster"
}
/** An emergent pattern detected in the collective */
interface EmergentPattern {
    pattern_id: string;
    description: string;
    belief_hashes: ActionHash[];
    pattern_type: PatternType;
    confidence: number;
    detected_at: Timestamp;
}
/** Input for recording a pattern */
interface RecordPatternInput {
    pattern_id: string;
    description: string;
    belief_hashes: ActionHash[];
    pattern_type: PatternType;
    confidence: number;
}
/** Input for detecting patterns */
interface DetectPatternsInput {
    similarity_threshold?: number;
    min_cluster_size?: number;
}
/** A pattern cluster from detection */
interface PatternCluster {
    pattern_id: string;
    representative_content: string;
    representative_hash: ActionHash;
    member_hashes: ActionHash[];
    member_count: number;
    pattern_type: PatternType;
    coherence: number;
    tags: string[];
}
/** Epistemic reputation for a validator */
interface EpistemicReputation {
    agent: AgentPubKey;
    domains: string[];
    accuracy_score: number;
    calibration_score: number;
    validation_count: number;
    updated_at: Timestamp;
}
/** Input for updating reputation */
interface UpdateReputationInput {
    agent: AgentPubKey;
    domains: string[];
    accuracy_score: number;
    calibration_score: number;
    validation_count: number;
}
/** Relationship stages */
declare enum RelationshipStage {
    NoRelation = "NoRelation",
    Acquaintance = "Acquaintance",
    Collaborator = "Collaborator",
    TrustedPeer = "TrustedPeer",
    PartnerInTruth = "PartnerInTruth"
}
/** Relationship between agents */
interface AgentRelationship {
    other_agent: AgentPubKey;
    trust_score: number;
    interaction_count: number;
    last_interaction: Timestamp;
    relationship_stage: RelationshipStage;
    shared_domains: string[];
    agreement_ratio: number;
}
/** Input for updating a relationship */
interface UpdateRelationshipInput {
    other_agent: AgentPubKey;
    trust_delta?: number;
    relationship_stage?: RelationshipStage;
    new_domains?: string[];
    agreement_ratio?: number;
}
/** Weighted consensus result */
interface WeightedConsensusResult {
    weighted_value: number;
    confidence: number;
    total_weight: number;
    voter_count: number;
    breakdown: ConsensusBreakdown;
}
/** Consensus breakdown by vote type */
interface ConsensusBreakdown {
    corroborate: number;
    plausible: number;
    abstain: number;
    implausible: number;
    contradict: number;
}
/** Collective statistics */
interface CollectiveStats {
    total_belief_shares: number;
    total_patterns: number;
    active_validators: number;
}

/**
 * LUCID Zome Client
 *
 * Core operations for thoughts, tags, domains, and relationships.
 */

declare class LucidZomeClient {
    private client;
    private roleName;
    private zomeName;
    constructor(client: AppClient, roleName?: string, zomeName?: string);
    private callZome;
    /** Create a new thought */
    createThought(input: CreateThoughtInput): Promise<Thought>;
    /** Get a thought by ID */
    getThought(thoughtId: string): Promise<Thought | null>;
    /** Get a thought by action hash */
    getThoughtByHash(actionHash: ActionHash): Promise<Thought | null>;
    /** Update an existing thought */
    updateThought(input: UpdateThoughtInput): Promise<Thought>;
    /** Delete a thought */
    deleteThought(thoughtId: string): Promise<ActionHash>;
    /** Get all thoughts for the current user */
    getMyThoughts(): Promise<Thought[]>;
    /** Get thoughts by tag */
    getThoughtsByTag(tag: string): Promise<Thought[]>;
    /** Get thoughts by domain */
    getThoughtsByDomain(domain: string): Promise<Thought[]>;
    /** Get thoughts by type */
    getThoughtsByType(thoughtType: ThoughtType): Promise<Thought[]>;
    /** Get child thoughts of a parent */
    getChildThoughts(parentThoughtId: string): Promise<Thought[]>;
    /** Search thoughts with filters */
    searchThoughts(input: SearchThoughtsInput): Promise<Thought[]>;
    /** Create a new tag */
    createTag(input: CreateTagInput): Promise<Tag>;
    /** Get all tags for the current user */
    getMyTags(): Promise<Tag[]>;
    /** Create a new domain */
    createDomain(input: CreateDomainInput): Promise<Domain>;
    /** Get all domains for the current user */
    getMyDomains(): Promise<Domain[]>;
    /** Create a relationship between thoughts */
    createRelationship(input: CreateRelationshipInput): Promise<Relationship>;
    /** Get relationships from a thought */
    getThoughtRelationships(thoughtId: string): Promise<Relationship[]>;
    /** Get thoughts related to a given thought */
    getRelatedThoughts(thoughtId: string): Promise<Thought[]>;
    /** Get statistics about the knowledge graph */
    getStats(): Promise<KnowledgeGraphStats>;
}

/**
 * Sources Zome Client
 *
 * Source tracking and citation operations.
 */

declare class SourcesZomeClient {
    private client;
    private roleName;
    private zomeName;
    constructor(client: AppClient, roleName?: string, zomeName?: string);
    private callZome;
    /** Create a new source */
    createSource(input: CreateSourceInput): Promise<Source>;
    /** Get a source by ID */
    getSource(sourceId: string): Promise<Source | null>;
    /** Get a source by URL */
    getSourceByUrl(url: string): Promise<Source | null>;
    /** Get all sources for the current user */
    getMySources(): Promise<Source[]>;
    /** Get sources by type */
    getSourcesByType(sourceType: SourceType): Promise<Source[]>;
    /** Create a citation linking a thought to a source */
    createCitation(input: CreateCitationInput): Promise<Citation>;
    /** Get citations for a thought */
    getThoughtCitations(thoughtId: string): Promise<Citation[]>;
    /** Get citations for a source */
    getSourceCitations(sourceId: string): Promise<Citation[]>;
    /** Get all sources for a thought (via citations) */
    getThoughtSources(thoughtId: string): Promise<Source[]>;
}

/**
 * Temporal Zome Client
 *
 * Belief versioning and time-travel queries.
 */

declare class TemporalZomeClient {
    private client;
    private roleName;
    private zomeName;
    constructor(client: AppClient, roleName?: string, zomeName?: string);
    private callZome;
    /** Record a new version of a thought */
    recordVersion(input: RecordVersionInput): Promise<BeliefVersion>;
    /** Get all versions of a thought (history) */
    getThoughtHistory(thoughtId: string): Promise<BeliefVersion[]>;
    /** Get belief at a specific timestamp */
    getBeliefAtTime(thoughtId: string, timestamp: Timestamp): Promise<BeliefVersion | null>;
    /** Create a snapshot of the current knowledge graph */
    createSnapshot(input: CreateSnapshotInput): Promise<GraphSnapshot>;
    /** Get all snapshots */
    getMySnapshots(): Promise<GraphSnapshot[]>;
}

/**
 * Privacy Zome Client
 *
 * Sharing policies and access control.
 */

declare class PrivacyZomeClient {
    private client;
    private roleName;
    private zomeName;
    constructor(client: AppClient, roleName?: string, zomeName?: string);
    private callZome;
    /** Set sharing policy for a thought */
    setSharingPolicy(input: SetPolicyInput): Promise<SharingPolicy>;
    /** Get sharing policy for a thought */
    getSharingPolicy(thoughtId: string): Promise<SharingPolicy | null>;
    /** Grant access to a specific agent */
    grantAccess(input: GrantAccessInput): Promise<AccessGrant>;
    /** Get grants for a thought */
    getThoughtGrants(thoughtId: string): Promise<AccessGrant[]>;
    /** Check if an agent has access to a thought */
    checkAccess(thoughtId: string, agent: AgentPubKey): Promise<boolean>;
    /** Log an access event */
    logAccess(thoughtId: string, accessType: string): Promise<ActionHash>;
}

/**
 * Reasoning Zome Client
 *
 * Coherence checking, contradiction detection, and inference.
 */

declare class ReasoningZomeClient {
    private client;
    private roleName;
    private zomeName;
    constructor(client: AppClient, roleName?: string, zomeName?: string);
    private callZome;
    /** Record a detected contradiction */
    recordContradiction(input: RecordContradictionInput): Promise<Contradiction>;
    /** Get contradictions for a thought */
    getThoughtContradictions(thoughtId: string): Promise<Contradiction[]>;
    /** Create a coherence report */
    createCoherenceReport(input: CreateReportInput): Promise<CoherenceReport>;
    /** Record an inference */
    recordInference(input: RecordInferenceInput): Promise<Inference>;
    /** Get all inferences */
    getMyInferences(): Promise<Inference[]>;
}

/**
 * Bridge Zome Client
 *
 * Cross-hApp integration with Identity, Knowledge DKG, and Symthaea.
 */

declare class BridgeZomeClient {
    private client;
    private roleName;
    private zomeName;
    constructor(client: AppClient, roleName?: string, zomeName?: string);
    private callZome;
    /** Record federation of a thought to external hApp */
    recordFederation(thoughtId: string, targetHapp: string, externalId?: string): Promise<FederationRecord>;
    /** Get federation records for a thought */
    getThoughtFederations(thoughtId: string): Promise<FederationRecord[]>;
    /** Cache external reputation score */
    cacheReputation(agent: AgentPubKey, kVector: [number, number, number, number, number, number, number, number], trustScore: number, sourceHapp: string): Promise<ExternalReputation>;
    /** Get cached reputation for an agent */
    getCachedReputation(agent: AgentPubKey): Promise<ExternalReputation | null>;
    /** Check coherence with Symthaea consciousness engine */
    checkCoherenceWithSymthaea(thoughtIds: string[]): Promise<CoherenceResult>;
}

/**
 * Collective Sensemaking Zome Client
 *
 * Distributed belief sharing, validation, consensus discovery, and emergent truth.
 */

declare class CollectiveZomeClient {
    private client;
    private roleName;
    private zomeName;
    constructor(client: AppClient, roleName?: string, zomeName?: string);
    private callZome;
    /** Share a belief to the collective */
    shareBelief(input: ShareBeliefInput): Promise<BeliefShare>;
    /** Get all shared beliefs */
    getAllBeliefShares(): Promise<BeliefShare[]>;
    /** Get beliefs by tag */
    getBeliefsByTag(tag: string): Promise<BeliefShare[]>;
    /** Cast a vote on a shared belief */
    castVote(input: CastVoteInput): Promise<ValidationVote>;
    /** Get votes for a belief share */
    getBeliefVotes(beliefHash: ActionHash): Promise<ValidationVote[]>;
    /** Get consensus for a belief */
    getBeliefConsensus(beliefHash: ActionHash): Promise<ConsensusRecord | null>;
    /** Calculate weighted consensus using relationships */
    calculateWeightedConsensus(beliefHash: ActionHash): Promise<WeightedConsensusResult>;
    /** Detect emergent patterns across beliefs */
    detectPatterns(input?: DetectPatternsInput): Promise<PatternCluster[]>;
    /** Record an emergent pattern */
    recordPattern(input: RecordPatternInput): Promise<EmergentPattern>;
    /** Get all recorded patterns */
    getRecordedPatterns(): Promise<EmergentPattern[]>;
    /** Update agent's epistemic reputation */
    updateReputation(input: UpdateReputationInput): Promise<EpistemicReputation>;
    /** Update or create a relationship with another agent */
    updateRelationship(input: UpdateRelationshipInput): Promise<AgentRelationship>;
    /** Get relationship with a specific agent */
    getRelationship(otherAgent: AgentPubKey): Promise<AgentRelationship | null>;
    /** Get all my relationships */
    getMyRelationships(): Promise<AgentRelationship[]>;
    /** Get collective statistics */
    getCollectiveStats(): Promise<CollectiveStats>;
}

/**
 * LUCID Client Utilities
 */

/**
 * Decode entry from a Holochain record
 */
declare function decodeRecord<T>(record: Record$1): T;
/**
 * Decode entries from multiple Holochain records
 */
declare function decodeRecords<T>(records: Record$1[]): T[];
/**
 * Generate epistemic code string from classification
 */
declare function epistemicCode(e: number, n: number, m: number, h: number): string;
/**
 * Calculate overall epistemic strength
 * Weights: E=40%, N=25%, M=20%, H=15%
 */
declare function calculateEpistemicStrength(e: number, n: number, m: number, h: number): number;
/**
 * Format timestamp for display
 */
declare function formatTimestamp(timestamp: number): string;
/**
 * Parse ISO date string to Holochain timestamp (microseconds)
 */
declare function parseToTimestamp(isoDate: string): number;
/**
 * Source type quality weights for confidence calculation
 */
declare const SOURCE_QUALITY_WEIGHTS: Record<string, number>;
/**
 * Get quality weight for a source type
 */
declare function getSourceQualityWeight(sourceType: string): number;

/**
 * Main LUCID client providing access to all zome functionality.
 */
declare class LucidClient {
    private client;
    private roleName;
    /** Core thought operations */
    readonly lucid: LucidZomeClient;
    /** Source and citation operations */
    readonly sources: SourcesZomeClient;
    /** Temporal versioning operations */
    readonly temporal: TemporalZomeClient;
    /** Privacy and sharing operations */
    readonly privacy: PrivacyZomeClient;
    /** Reasoning and coherence operations */
    readonly reasoning: ReasoningZomeClient;
    /** Cross-hApp bridge operations */
    readonly bridge: BridgeZomeClient;
    /** Collective sensemaking operations */
    readonly collective: CollectiveZomeClient;
    constructor(client: AppClient, roleName?: string);
    /** Create a new thought */
    createThought(...args: Parameters<LucidZomeClient['createThought']>): ReturnType<LucidZomeClient['createThought']>;
    /** Get a thought by ID */
    getThought(...args: Parameters<LucidZomeClient['getThought']>): ReturnType<LucidZomeClient['getThought']>;
    /** Update an existing thought */
    updateThought(...args: Parameters<LucidZomeClient['updateThought']>): ReturnType<LucidZomeClient['updateThought']>;
    /** Delete a thought */
    deleteThought(...args: Parameters<LucidZomeClient['deleteThought']>): ReturnType<LucidZomeClient['deleteThought']>;
    /** Get all thoughts for the current user */
    getMyThoughts(): ReturnType<LucidZomeClient['getMyThoughts']>;
    /** Search thoughts with filters */
    searchThoughts(...args: Parameters<LucidZomeClient['searchThoughts']>): ReturnType<LucidZomeClient['searchThoughts']>;
    /** Get statistics about the knowledge graph */
    getStats(): ReturnType<LucidZomeClient['getStats']>;
    /**
     * Create a thought with a source citation in one call.
     */
    createThoughtWithSource(thoughtInput: Parameters<LucidZomeClient['createThought']>[0], sourceInput: Parameters<SourcesZomeClient['createSource']>[0], citationOptions?: {
        location?: string;
        quote?: string;
        relationship?: CitationRelationship;
    }): Promise<{
        thought: Thought;
        source: Source;
        citation: Citation;
    }>;
    /**
     * Get a thought with its full context (sources, relationships, history).
     */
    getThoughtWithContext(thoughtId: string): Promise<{
        thought: Thought | null;
        sources: Source[];
        relationships: Relationship[];
        history: BeliefVersion[];
        contradictions: Contradiction[];
    }>;
    /**
     * Share a thought with specific agents.
     */
    shareThought(thoughtId: string, agents: Parameters<PrivacyZomeClient['grantAccess']>[0]['grantee'][], permissions?: string[]): Promise<AccessGrant[]>;
    /**
     * Make a thought public.
     */
    makePublic(thoughtId: string): Promise<SharingPolicy>;
    /**
     * Make a thought private.
     */
    makePrivate(thoughtId: string): Promise<SharingPolicy>;
}

export { type AccessGrant, type AgentRelationship, type BeliefAtTimeInput, type BeliefShare, BeliefStance, type BeliefVersion, BridgeZomeClient, type CastVoteInput, type CheckAccessInput, type Citation, CitationRelationship, type CoherenceReport, type CoherenceResult, type CollectiveStats, CollectiveZomeClient, type ConsensusBreakdown, type ConsensusRecord, ConsensusType, type Contradiction, ContradictionType, type CreateCitationInput, type CreateDomainInput, type CreateRelationshipInput, type CreateReportInput, type CreateSnapshotInput, type CreateSourceInput, type CreateTagInput, type CreateThoughtInput, type DetectPatternsInput, type Domain, type EmergentPattern, EmpiricalLevel, type EpistemicClassification, type EpistemicReputation, type ExternalReputation, type FederationRecord, FederationStatus, type GrantAccessInput, type GraphSnapshot, HarmonicLevel, type Inference, InferenceType, type KnowledgeGraphStats, LucidClient, LucidZomeClient, MaterialityLevel, NormativeLevel, type PatternCluster, PatternType, PrivacyZomeClient, ReasoningZomeClient, type RecordContradictionInput, type RecordInferenceInput, type RecordPatternInput, type RecordVersionInput, type Relationship, RelationshipStage, RelationshipType, SOURCE_QUALITY_WEIGHTS, type SearchThoughtsInput, type SetPolicyInput, type ShareBeliefInput, type SharingPolicy, type Source, SourceType, SourcesZomeClient, type Tag, TemporalZomeClient, type Thought, ThoughtType, type UpdateRelationshipInput, type UpdateReputationInput, type UpdateThoughtInput, type ValidationVote, ValidationVoteType, type Visibility, type WeightedConsensusResult, calculateEpistemicStrength, decodeRecord, decodeRecords, LucidClient as default, epistemicCode, formatTimestamp, getSourceQualityWeight, parseToTimestamp };
