/**
 * Symthaea Agent Types
 *
 * Core type definitions for the Symthaea agent system.
 * Symthaea = "growing together" - AI that grows with citizens, not above them.
 *
 * @module symthaea/types
 */

/**
 * Agent domains for civic applications
 */
export type CivicAgentDomain =
  | 'benefits'      // SNAP, Medicaid, housing, etc.
  | 'permits'       // Business, building, vehicle permits
  | 'tax'           // Tax questions, deductions, filing
  | 'voting'        // Voter registration, ballot info
  | 'justice'       // Court dates, legal aid, expungement
  | 'housing'       // Affordable housing, tenant rights
  | 'employment'    // Job training, unemployment, workers comp
  | 'education'     // School enrollment, financial aid
  | 'health'        // Medicaid, clinics, mental health
  | 'emergency'     // Crisis resources, disaster relief
  | 'general';      // General civic questions

/**
 * Agent capability levels
 */
export type AgentCapabilityLevel =
  | 'informational'   // Can provide information only
  | 'navigational'    // Can help navigate processes
  | 'transactional'   // Can help complete transactions
  | 'advisory';       // Can provide personalized advice

/**
 * Conversation context for agent interactions
 */
export interface ConversationContext {
  /** Unique conversation ID */
  conversationId: string;
  /** Citizen DID (if authenticated) */
  citizenDid?: string;
  /** Channel the conversation is happening on */
  channel: 'sms' | 'web' | 'voice' | 'dashboard' | 'api';
  /** Conversation history */
  history: ConversationTurn[];
  /** Extracted entities from conversation */
  entities: ExtractedEntity[];
  /** Current intent (if identified) */
  currentIntent?: Intent;
  /** Agent domain handling the conversation */
  domain: CivicAgentDomain;
  /** Metadata */
  metadata: Record<string, unknown>;
  /** Started at */
  startedAt: Date;
  /** Last activity */
  lastActivityAt: Date;
}

/**
 * A single turn in a conversation
 */
export interface ConversationTurn {
  /** Turn ID */
  id: string;
  /** Who sent this turn */
  role: 'citizen' | 'agent' | 'system';
  /** Message content */
  content: string;
  /** Timestamp */
  timestamp: Date;
  /** Confidence in understanding (for agent turns) */
  confidence?: number;
  /** Sources used for response */
  sources?: KnowledgeSource[];
  /** Suggested actions */
  suggestedActions?: SuggestedAction[];
}

/**
 * Extracted entity from conversation
 */
export interface ExtractedEntity {
  /** Entity type */
  type: EntityType;
  /** Entity value */
  value: string;
  /** Confidence */
  confidence: number;
  /** Which turn this was extracted from */
  turnId: string;
}

/**
 * Entity types that can be extracted
 */
export type EntityType =
  | 'zip_code'
  | 'income'
  | 'household_size'
  | 'date'
  | 'case_number'
  | 'permit_type'
  | 'program_name'
  | 'phone_number'
  | 'email'
  | 'address'
  | 'name'
  | 'date_of_birth'
  | 'ssn_last_four';

/**
 * Identified intent
 */
export interface Intent {
  /** Intent name */
  name: string;
  /** Confidence */
  confidence: number;
  /** Required entities for this intent */
  requiredEntities: EntityType[];
  /** Entities we have */
  providedEntities: EntityType[];
  /** Whether we have all required entities */
  complete: boolean;
}

/**
 * Knowledge source used in response
 */
export interface KnowledgeSource {
  /** Source type */
  type: 'regulation' | 'faq' | 'form' | 'policy' | 'website' | 'dkg';
  /** Source ID */
  id: string;
  /** Source title */
  title: string;
  /** Source URL (if applicable) */
  url?: string;
  /** Relevance score */
  relevance: number;
  /** Last updated */
  lastUpdated?: Date;
}

/**
 * Suggested action for citizen
 */
export interface SuggestedAction {
  /** Action type */
  type: 'link' | 'call' | 'form' | 'location' | 'appointment';
  /** Action label */
  label: string;
  /** Action value (URL, phone, etc.) */
  value: string;
  /** Priority */
  priority: 'primary' | 'secondary';
}

/**
 * Agent response
 */
export interface AgentResponse {
  /** Response text */
  text: string;
  /** Confidence in response */
  confidence: number;
  /** Sources used */
  sources: KnowledgeSource[];
  /** Suggested actions */
  suggestedActions: SuggestedAction[];
  /** Whether agent needs more information */
  needsMoreInfo: boolean;
  /** Follow-up questions if needed */
  followUpQuestions?: string[];
  /** Whether to escalate to human */
  escalateToHuman: boolean;
  /** Escalation reason */
  escalationReason?: string;
  /** Response metadata */
  metadata: ResponseMetadata;
}

/**
 * Response metadata for auditing
 */
export interface ResponseMetadata {
  /** Model used */
  model: string;
  /** Latency in ms */
  latencyMs: number;
  /** Domain */
  domain: CivicAgentDomain;
  /** Intent matched */
  intentMatched?: string;
  /** Knowledge base queries made */
  kbQueries: number;
  /** Whether response was cached */
  cached: boolean;
}

/**
 * Agent configuration
 */
export interface AgentConfig {
  /** Agent ID */
  id: string;
  /** Agent name */
  name: string;
  /** Agent domain */
  domain: CivicAgentDomain;
  /** Capability level */
  capabilityLevel: AgentCapabilityLevel;
  /** Model to use */
  model: ModelConfig;
  /** Knowledge base configuration */
  knowledgeBase: KnowledgeBaseConfig;
  /** Escalation configuration */
  escalation: EscalationConfig;
  /** Personality/tone configuration */
  personality: PersonalityConfig;
  /** Rate limiting */
  rateLimit: RateLimitConfig;
}

/**
 * Model configuration
 */
export interface ModelConfig {
  /** Model provider */
  provider: 'ollama' | 'openai' | 'anthropic' | 'local';
  /** Model name */
  model: string;
  /** Temperature */
  temperature: number;
  /** Max tokens */
  maxTokens: number;
  /** System prompt */
  systemPrompt: string;
}

/**
 * Knowledge base configuration
 */
export interface KnowledgeBaseConfig {
  /** Enable knowledge base */
  enabled: boolean;
  /** Sources to use */
  sources: KnowledgeSourceType[];
  /** Max sources per response */
  maxSourcesPerResponse: number;
  /** Minimum relevance threshold */
  minRelevance: number;
  /** DKG integration */
  dkgEnabled: boolean;
  /** DKG namespace */
  dkgNamespace?: string;
}

/**
 * Knowledge source types
 */
export type KnowledgeSourceType =
  | 'federal_regulations'
  | 'state_regulations'
  | 'local_ordinances'
  | 'agency_faqs'
  | 'forms_library'
  | 'policy_documents'
  | 'dkg';

/**
 * Escalation configuration
 */
export interface EscalationConfig {
  /** Enable human escalation */
  enabled: boolean;
  /** Confidence threshold for escalation */
  confidenceThreshold: number;
  /** Topics that always escalate */
  alwaysEscalateTopics: string[];
  /** Max turns before suggesting escalation */
  maxTurnsBeforeEscalation: number;
  /** Escalation channels */
  channels: EscalationChannel[];
}

/**
 * Escalation channel
 */
export interface EscalationChannel {
  /** Channel type */
  type: 'phone' | 'chat' | 'email' | 'appointment';
  /** Channel value */
  value: string;
  /** Availability */
  availability: string;
  /** Average wait time */
  avgWaitTime?: string;
}

/**
 * Personality configuration
 */
export interface PersonalityConfig {
  /** Tone */
  tone: 'formal' | 'friendly' | 'empathetic' | 'neutral';
  /** Reading level target */
  readingLevel: 'simple' | 'standard' | 'technical';
  /** Include emojis */
  useEmojis: boolean;
  /** Language preferences */
  languages: string[];
  /** Greeting style */
  greetingStyle: 'full' | 'brief' | 'none';
}

/**
 * Rate limit configuration
 */
export interface RateLimitConfig {
  /** Max requests per minute */
  requestsPerMinute: number;
  /** Max requests per hour */
  requestsPerHour: number;
  /** Max conversation length (turns) */
  maxConversationLength: number;
}

/**
 * Agent statistics
 */
export interface AgentStats {
  /** Total conversations */
  totalConversations: number;
  /** Average conversation length */
  avgConversationLength: number;
  /** Resolution rate (no escalation needed) */
  resolutionRate: number;
  /** Escalation rate */
  escalationRate: number;
  /** Average confidence */
  avgConfidence: number;
  /** Average latency */
  avgLatencyMs: number;
  /** Conversations by domain */
  byDomain: Record<CivicAgentDomain, number>;
  /** Top intents */
  topIntents: Array<{ intent: string; count: number }>;
}

/**
 * Escalation request
 */
export interface EscalationRequest {
  /** Request ID */
  id: string;
  /** Conversation ID */
  conversationId: string;
  /** Citizen DID */
  citizenDid?: string;
  /** Reason */
  reason: string;
  /** Priority */
  priority: 'low' | 'normal' | 'high' | 'urgent';
  /** Domain */
  domain: CivicAgentDomain;
  /** Context summary */
  contextSummary: string;
  /** Extracted entities */
  entities: ExtractedEntity[];
  /** Requested at */
  requestedAt: Date;
  /** Status */
  status: 'pending' | 'assigned' | 'in_progress' | 'resolved';
  /** Assigned to (human agent ID) */
  assignedTo?: string;
}
