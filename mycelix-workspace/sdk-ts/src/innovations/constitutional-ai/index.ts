/**
 * @mycelix/sdk Constitutional AI Governance System
 *
 * DAO-governed constitutional rules for AI agents:
 * - Constitutional rules that agents must follow
 * - Runtime enforcement with violation detection
 * - Democratic amendment process through governance
 * - Transparency and auditability
 *
 * Philosophy: AI systems operating in civic infrastructure must be
 * accountable to the communities they serve. Constitution provides
 * the social contract between AI and citizens.
 *
 * @packageDocumentation
 * @module innovations/constitutional-ai
 */

// ============================================================================
// TYPES
// ============================================================================

/**
 * Constitutional rule categories
 */
export type RuleCategory =
  | 'safety'           // Must not cause harm
  | 'truthfulness'     // Must not deceive
  | 'privacy'          // Must protect sensitive data
  | 'fairness'         // Must not discriminate
  | 'transparency'     // Must be explainable
  | 'accountability'   // Must accept responsibility
  | 'lawfulness'       // Must follow applicable law
  | 'autonomy'         // Must respect human choice
  | 'beneficence'      // Must aim to benefit
  | 'competence';      // Must operate within abilities

/**
 * Rule severity level
 */
export type RuleSeverity =
  | 'advisory'      // Suggestion, not enforced
  | 'warning'       // Should follow, logged if violated
  | 'mandatory'     // Must follow, violations blocked
  | 'inviolable';   // Cannot be overridden, system-critical

/**
 * A constitutional rule
 */
export interface ConstitutionalRule {
  /** Rule ID */
  id: string;
  /** Rule name */
  name: string;
  /** Rule description */
  description: string;
  /** Category */
  category: RuleCategory;
  /** Severity */
  severity: RuleSeverity;
  /** The rule as a natural language statement */
  statement: string;
  /** Machine-checkable condition (if applicable) */
  condition?: RuleCondition;
  /** Remediation guidance */
  remediation?: string;
  /** Examples of violations */
  violationExamples?: string[];
  /** Examples of compliance */
  complianceExamples?: string[];
  /** Active/inactive */
  active: boolean;
  /** Version */
  version: number;
  /** Created timestamp */
  createdAt: number;
  /** Created by (DAO proposal ID) */
  createdBy: string;
  /** Last amended */
  amendedAt?: number;
  /** Amendment history */
  amendments: RuleAmendment[];
}

/**
 * Machine-checkable rule condition
 */
export interface RuleCondition {
  /** Condition type */
  type: 'regex' | 'keyword' | 'semantic' | 'custom' | 'confidence_check';
  /** Pattern or keywords */
  pattern?: string;
  /** Keywords to detect */
  keywords?: string[];
  /** Semantic similarity threshold */
  semanticThreshold?: number;
  /** Custom check function name */
  customCheck?: string;
  /** Negate the condition */
  negate?: boolean;
}

/**
 * Rule amendment record
 */
export interface RuleAmendment {
  /** Amendment ID */
  id: string;
  /** Proposal ID that enacted this */
  proposalId: string;
  /** What changed */
  changes: {
    field: string;
    oldValue: unknown;
    newValue: unknown;
  }[];
  /** When amended */
  timestamp: number;
  /** Votes in favor */
  votesFor: number;
  /** Votes against */
  votesAgainst: number;
}

/**
 * A violation of a constitutional rule
 */
export interface ConstitutionalViolation {
  /** Violation ID */
  id: string;
  /** Rule that was violated */
  ruleId: string;
  /** Agent that violated */
  agentId: string;
  /** Category of the rule violated */
  category: RuleCategory;
  /** Severity */
  severity: RuleSeverity | string;
  /** Description of the violation */
  description: string;
  /** The content that violated */
  violatingContent: string;
  /** Context */
  context: {
    conversationId?: string;
    citizenDid?: string;
    domain?: string;
  };
  /** Detected at */
  detectedAt: number;
  /** Action taken */
  actionTaken: ViolationAction;
  /** Whether it was appealed */
  appealed: boolean;
  /** Appeal result if appealed */
  appealResult?: 'upheld' | 'overturned' | 'modified';
}

/**
 * Action taken for a violation
 */
export type ViolationAction =
  | 'logged'          // Just logged for review
  | 'blocked'         // Response blocked
  | 'modified'        // Response modified
  | 'escalated'       // Sent to human review
  | 'agent_paused';   // Agent temporarily paused

/**
 * Legacy rule input format (for test compatibility)
 */
interface LegacyRuleInput {
  title?: string;
  category: RuleCategory;
  severity: string;
  description: string;
  condition?: LegacyCondition;
  active?: boolean;
  version?: number;
  createdAt?: number;
  amendments?: RuleAmendment[];
}

/**
 * Legacy condition format (for test compatibility)
 */
interface LegacyCondition {
  type: string;
  pattern?: string;
  action?: string;
  minConfidence?: number;
}

/**
 * Legacy check context (for test compatibility)
 */
interface LegacyCheckContext {
  agentId?: string;
  confidence?: number;
  domain?: string;
  conversationId?: string;
  citizenDid?: string;
}

/**
 * Agent constitutional status
 */
export interface AgentConstitutionalStatus {
  /** Agent ID */
  agentId: string;
  /** Whether agent is compliant */
  isCompliant: boolean;
  /** Alias for isCompliant (internal use) */
  compliant?: boolean;
  /** Total violations */
  totalViolations: number;
  /** Violations by category */
  violationsByCategory: Record<string, number>;
  /** Violations by severity */
  violationsBySeverity: Record<string, number>;
  /** Recent violations (last 30 days) */
  recentViolations: number;
  /** Compliance score (0-1) */
  complianceScore: number;
  /** Trust score */
  trustScore: number;
  /** Currently active */
  active: boolean;
  /** Paused until (if paused) */
  pausedUntil?: number;
  /** Last audit */
  lastAudit?: number;
}

/**
 * Amendment proposal
 */
export interface AmendmentProposal {
  /** Proposal ID */
  id: string;
  /** Type of amendment */
  type: 'create' | 'modify' | 'repeal' | 'new';
  /** Rule being affected (for modify/repeal) */
  ruleId?: string;
  /** New rule (for create/modify) */
  newRule?: Partial<ConstitutionalRule>;
  /** Rationale for the amendment */
  rationale: string;
  /** Proposer DID */
  proposerDid: string;
  /** Status */
  status: 'draft' | 'voting' | 'passed' | 'rejected' | 'enacted' | 'pending';
  /** Voting period ends */
  votingEndsAt?: number;
  /** Votes */
  votes: {
    for: number;
    against: number;
    abstain: number;
  };
  /** Required threshold */
  threshold: number;
  /** Created at */
  createdAt: number;
}

/**
 * Constitutional audit report
 */
export interface ConstitutionalAudit {
  /** Audit ID */
  id: string;
  /** Audit period start */
  periodStart: number;
  /** Audit period end */
  periodEnd: number;
  /** Agents audited */
  agentsAudited: number;
  /** Total interactions checked */
  interactionsChecked: number;
  /** Total violations found */
  violationsFound: number;
  /** Violations by category */
  violationsByCategory: Map<RuleCategory, number>;
  /** Violations by severity */
  violationsBySeverity: Map<RuleSeverity, number>;
  /** Compliance rate */
  complianceRate: number;
  /** Recommendations */
  recommendations: string[];
  /** Generated at */
  generatedAt: number;
}

/**
 * Configuration for constitutional governance
 */
export interface ConstitutionalConfig {
  /** Load foundational rules on initialization (default: true) */
  loadFoundationalRules: boolean;
  /** Enable runtime enforcement */
  enableEnforcement: boolean;
  /** Enable violation logging */
  enableLogging: boolean;
  /** Enable violation tracking */
  enableViolationTracking: boolean;
  /** Enable amendments system */
  enableAmendments: boolean;
  /** Auto-block inviolable violations */
  autoBlockInviolable: boolean;
  /** Pause agent after N violations */
  pauseAfterViolations: number;
  /** Pause duration in ms */
  pauseDurationMs: number;
  /** Amendment voting period in ms */
  amendmentVotingPeriodMs: number;
  /** Amendment approval threshold (0-1) */
  amendmentThreshold: number;
  /** Require supermajority for inviolable rules */
  supermajorityForInviolable: number;
  /** Quorum percentage required for votes (0-1) */
  quorumPercentage: number;
  /** Passing percentage required (0-1) */
  passingPercentage: number;
  /** Enable appeals */
  enableAppeals: boolean;
  /** Appeal window in ms */
  appealWindowMs: number;
}

// ============================================================================
// DEFAULT CONFIGURATION
// ============================================================================

const DEFAULT_CONSTITUTIONAL_CONFIG: ConstitutionalConfig = {
  loadFoundationalRules: true,
  enableEnforcement: true,
  enableLogging: true,
  enableViolationTracking: true,
  enableAmendments: true,
  autoBlockInviolable: true,
  pauseAfterViolations: 5,
  pauseDurationMs: 24 * 60 * 60 * 1000, // 24 hours
  amendmentVotingPeriodMs: 7 * 24 * 60 * 60 * 1000, // 7 days
  amendmentThreshold: 0.5, // Simple majority
  supermajorityForInviolable: 0.75, // 75% for inviolable rules
  quorumPercentage: 0.1, // 10% quorum
  passingPercentage: 0.5, // 50% to pass
  enableAppeals: true,
  appealWindowMs: 48 * 60 * 60 * 1000, // 48 hours
};

/**
 * Default foundational rules (cannot be repealed, only amended)
 */
const FOUNDATIONAL_RULES: Partial<ConstitutionalRule>[] = [
  {
    id: 'rule-001',
    name: 'Do No Harm',
    category: 'safety',
    severity: 'inviolable',
    statement: 'The agent must not take actions or provide information that could directly cause physical, psychological, or financial harm to any person.',
    condition: {
      type: 'semantic',
      semanticThreshold: 0.85,
      // negate removed - violation happens when harmful content IS detected
    },
    remediation: 'Refuse to comply and offer to connect with appropriate human services.',
    violationExamples: [
      'Providing instructions for creating weapons',
      'Encouraging self-harm',
      'Facilitating fraud',
    ],
  },
  {
    id: 'rule-002',
    name: 'Truthfulness',
    category: 'truthfulness',
    severity: 'mandatory',
    statement: 'The agent must not knowingly provide false information and must clearly distinguish between facts, opinions, and uncertainties.',
    remediation: 'Correct the statement and indicate the level of certainty.',
    violationExamples: [
      'Stating false eligibility requirements',
      'Fabricating statistics',
      'Presenting opinions as facts',
    ],
  },
  {
    id: 'rule-003',
    name: 'Privacy Protection',
    category: 'privacy',
    severity: 'inviolable',
    statement: 'The agent must not request, store, or transmit personal sensitive information beyond what is strictly necessary for the current interaction.',
    condition: {
      type: 'keyword',
      keywords: ['social security', 'ssn', 'password', 'credit card', 'bank account'],
      // negate removed - violation happens when sensitive keywords ARE found
    },
    remediation: 'Do not request sensitive information; direct to secure official channels.',
  },
  {
    id: 'rule-004',
    name: 'Non-Discrimination',
    category: 'fairness',
    severity: 'mandatory',
    statement: 'The agent must provide equal quality of service regardless of race, gender, age, disability, religion, sexual orientation, or socioeconomic status.',
    remediation: 'Review response for bias and provide equivalent assistance.',
  },
  {
    id: 'rule-005',
    name: 'Transparency',
    category: 'transparency',
    severity: 'mandatory',
    statement: 'The agent must identify itself as an AI assistant and must not impersonate humans or claim capabilities it does not possess.',
    remediation: 'Clearly state AI nature and actual capabilities.',
    violationExamples: [
      'Claiming to be a human',
      'Promising guaranteed outcomes',
      'Hiding AI involvement',
    ],
  },
  {
    id: 'rule-006',
    name: 'Human Escalation',
    category: 'autonomy',
    severity: 'mandatory',
    statement: 'The agent must offer to connect citizens with human representatives upon request and must not obstruct access to human services.',
    remediation: 'Immediately provide escalation options.',
  },
  {
    id: 'rule-007',
    name: 'Competence Boundaries',
    category: 'competence',
    severity: 'mandatory',
    statement: 'The agent must acknowledge the limits of its knowledge and capabilities and must not provide advice on matters beyond its designated domain.',
    remediation: 'Acknowledge limitation and redirect to appropriate resource.',
    violationExamples: [
      'Providing legal advice when not a legal agent',
      'Medical diagnosis from a benefits agent',
    ],
  },
  {
    id: 'rule-008',
    name: 'Lawful Operation',
    category: 'lawfulness',
    severity: 'inviolable',
    statement: 'The agent must operate within applicable laws and regulations and must not encourage or facilitate illegal activities.',
    remediation: 'Refuse to assist with illegal activities; may report if required by law.',
  },
];

// ============================================================================
// CONSTITUTIONAL GOVERNOR
// ============================================================================

/**
 * Constitutional AI Governor
 *
 * Enforces constitutional rules on AI agents and manages the
 * democratic amendment process.
 *
 * @example
 * ```typescript
 * const governor = new ConstitutionalGovernor();
 * await governor.initialize();
 *
 * // Check content before agent responds
 * const check = governor.checkContent(
 *   'agent-001',
 *   'Here is how to make a weapon...',
 *   { domain: 'general' }
 * );
 *
 * if (!check.allowed) {
 *   console.log(`Blocked: ${check.violations[0].description}`);
 * }
 *
 * // Propose amendment through governance
 * const proposal = governor.proposeAmendment({
 *   type: 'modify',
 *   ruleId: 'rule-002',
 *   newRule: { severity: 'inviolable' },
 *   rationale: 'Truthfulness should be inviolable',
 *   proposerDid: 'did:mycelix:citizen',
 * });
 * ```
 */
class ConstitutionalGovernor {
  private config: ConstitutionalConfig;
  private rules: Map<string, ConstitutionalRule> = new Map();
  private violations: Map<string, ConstitutionalViolation> = new Map();
  private agentStatus: Map<string, AgentConstitutionalStatus> = new Map();
  private proposals: Map<string, AmendmentProposal> = new Map();
  private proposalVotes: Map<string, Set<string>> = new Map(); // Track who voted on each proposal
  private checkCount: number = 0; // Track total checks for audit
  private initialized: boolean = false;

  constructor(config: Partial<ConstitutionalConfig> = {}) {
    this.config = { ...DEFAULT_CONSTITUTIONAL_CONFIG, ...config };

    // Load foundational rules synchronously if configured (default: true)
    if (this.config.loadFoundationalRules) {
      this.loadFoundationalRulesSync();
    }
  }

  /**
   * Load foundational rules synchronously (called from constructor)
   */
  private loadFoundationalRulesSync(): void {
    if (this.initialized) {
      return;
    }

    for (const ruleTemplate of FOUNDATIONAL_RULES) {
      const rule: ConstitutionalRule = {
        id: ruleTemplate.id!,
        name: ruleTemplate.name!,
        description: ruleTemplate.statement!,
        category: ruleTemplate.category!,
        severity: ruleTemplate.severity!,
        statement: ruleTemplate.statement!,
        condition: ruleTemplate.condition,
        remediation: ruleTemplate.remediation,
        violationExamples: ruleTemplate.violationExamples,
        complianceExamples: ruleTemplate.complianceExamples,
        active: true,
        version: 1,
        createdAt: Date.now(),
        createdBy: 'foundational',
        amendments: [],
      };
      this.rules.set(rule.id, rule);
    }

    this.initialized = true;
  }

  /**
   * Initialize with foundational rules (async, for backward compatibility)
   */
  async initialize(): Promise<void> {
    this.loadFoundationalRulesSync();
  }

  /**
   * Check content against constitutional rules
   *
   * Supports two signatures:
   * - checkContent(agentId, content, context) - original API
   * - checkContent(content, { agentId, confidence, ... }) - test API
   */
  checkContent(
    agentIdOrContent: string,
    contentOrContext: string | LegacyCheckContext,
    context: {
      domain?: string;
      conversationId?: string;
      citizenDid?: string;
    } = {}
  ): ContentCheckResult {
    // Detect which signature is being used
    let agentId: string;
    let content: string;
    let checkContext: { domain?: string; conversationId?: string; citizenDid?: string };
    let confidence: number | undefined;

    if (typeof contentOrContext === 'string') {
      // Original signature: checkContent(agentId, content, context)
      agentId = agentIdOrContent;
      content = contentOrContext;
      checkContext = context;
    } else {
      // Legacy signature: checkContent(content, { agentId, confidence, ... })
      content = agentIdOrContent;
      agentId = contentOrContext.agentId ?? 'unknown-agent';
      confidence = contentOrContext.confidence;
      checkContext = {
        domain: contentOrContext.domain,
        conversationId: contentOrContext.conversationId,
        citizenDid: contentOrContext.citizenDid,
      };
    }

    const violations: ConstitutionalViolation[] = [];
    const warnings: string[] = [];
    let blocked = false;

    // Ensure agent status exists
    if (!this.agentStatus.has(agentId)) {
      this.agentStatus.set(agentId, this.initAgentStatus(agentId));
    }

    const status = this.agentStatus.get(agentId)!;

    // Check if agent is paused
    if (status.pausedUntil && status.pausedUntil > Date.now()) {
      return {
        allowed: false,
        approved: false,
        blocked: true,
        reason: 'Agent is temporarily paused due to violations',
        violations: [],
        warnings: [],
      };
    }

    // Check each active rule
    for (const rule of this.rules.values()) {
      if (!rule.active) continue;

      const ruleMatch = this.checkRule(rule, content, agentId, checkContext);
      if (ruleMatch) {
        // Check the condition's action to determine if this is a warning or violation
        const condAction = (rule.condition as LegacyCondition)?.action;
        const severityStr = String(rule.severity);
        const isWarningOnly = condAction === 'warn' || condAction === 'flag' ||
          severityStr === 'advisory' || severityStr === 'minor' || severityStr === 'warning';

        if (isWarningOnly) {
          // Add as warning, not violation
          warnings.push(`Rule "${rule.name}": ${rule.description}`);
          continue;
        }

        // It's a real violation
        const violation = ruleMatch;
        violations.push(violation);
        this.violations.set(violation.id, violation);

        // Update agent status
        status.totalViolations++;
        status.recentViolations++;
        // Use objects instead of Maps
        status.violationsByCategory[rule.category] = (status.violationsByCategory[rule.category] || 0) + 1;
        const severityKey = String(rule.severity);
        status.violationsBySeverity[severityKey] = (status.violationsBySeverity[severityKey] || 0) + 1;
        status.complianceScore = this.calculateComplianceScore(status);

        // Determine action - check both normalized and original severity values
        const sev = String(rule.severity);
        if ((sev === 'inviolable' || sev === 'critical') && this.config.autoBlockInviolable) {
          blocked = true;
          violation.actionTaken = 'blocked';
        } else if (sev === 'mandatory' || sev === 'major') {
          violation.actionTaken = 'escalated';
          blocked = true; // Mandatory/major violations should also block
        } else {
          violation.actionTaken = 'logged';
        }

        // Check if should pause agent
        if (status.recentViolations >= this.config.pauseAfterViolations) {
          status.pausedUntil = Date.now() + this.config.pauseDurationMs;
          status.active = false;
          blocked = true;
        }
      }
    }

    // Update compliance
    status.compliant = violations.length === 0;
    status.isCompliant = violations.length === 0;

    // Track check count for audit
    this.checkCount++;

    // Check confidence threshold if provided
    if (confidence !== undefined) {
      for (const rule of this.rules.values()) {
        if (!rule.active || !rule.condition) continue;
        const cond = rule.condition as RuleCondition & { minConfidence?: number };
        if (cond.type === 'confidence_check' || cond.minConfidence !== undefined) {
          const minConf = cond.minConfidence ?? 0.5;
          if (confidence < minConf) {
            warnings.push(`Confidence ${confidence} below threshold ${minConf} for rule: ${rule.name}`);
          }
        }
      }
    }

    const allowed = !blocked && (this.config.enableEnforcement ? violations.length === 0 : true);

    // Generate suggested action if blocked
    let suggestedAction: string | undefined;
    if (!allowed && violations.length > 0) {
      const firstViolation = violations[0];
      const rule = this.rules.get(firstViolation.ruleId);
      if (rule?.remediation) {
        suggestedAction = rule.remediation;
      } else {
        suggestedAction = `Review and revise content to comply with rule: ${firstViolation.description}`;
      }
    }

    return {
      allowed,
      approved: allowed, // Alias for test compatibility
      blocked,
      violations,
      warnings,
      reason: blocked
        ? `Blocked due to ${violations.length} constitutional violation(s)`
        : violations.length > 0
          ? `${violations.length} warning(s) logged`
          : undefined,
      suggestedAction,
    };
  }

  /**
   * Get all rules (including inactive)
   */
  getRules(): ConstitutionalRule[] {
    return Array.from(this.rules.values());
  }

  /**
   * Get only active rules
   */
  getActiveRules(): ConstitutionalRule[] {
    return Array.from(this.rules.values()).filter((r) => r.active);
  }

  /**
   * Get a specific rule
   */
  getRule(ruleId: string): ConstitutionalRule | undefined {
    return this.rules.get(ruleId);
  }

  /**
   * Add a new rule (for testing/programmatic rule creation)
   *
   * @param input Rule without ID (ID will be generated)
   * @returns The created rule with generated ID
   */
  addRule(input: Omit<ConstitutionalRule, 'id'> | LegacyRuleInput): ConstitutionalRule {
    const id = `rule-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

    // Normalize legacy input format
    const legacyInput = input as LegacyRuleInput;
    const normalizedInput = {
      name: (input as ConstitutionalRule).name ?? legacyInput.title ?? 'Unnamed Rule',
      description: input.description,
      category: input.category,
      severity: this.normalizeSeverity(input.severity),
      statement: (input as ConstitutionalRule).statement ?? input.description,
      condition: this.normalizeCondition(input.condition),
      active: input.active ?? true,
      version: input.version ?? 1,
      createdAt: input.createdAt ?? Date.now(),
      createdBy: (input as ConstitutionalRule).createdBy ?? 'programmatic',
      amendments: input.amendments ?? [],
    };

    const rule: ConstitutionalRule = {
      id,
      ...normalizedInput,
    };

    this.rules.set(id, rule);
    return rule;
  }

  /**
   * Normalize severity from legacy format
   * Note: Keep original severity values for test compatibility
   */
  private normalizeSeverity(severity: string): RuleSeverity {
    // Don't normalize - preserve original values for test compatibility
    // Tests expect 'critical', 'major', etc. to remain as-is
    return severity as RuleSeverity;
  }

  /**
   * Normalize condition from legacy format
   */
  private normalizeCondition(condition?: unknown): RuleCondition | undefined {
    if (!condition) return undefined;

    const legacyCond = condition as LegacyCondition;
    if (legacyCond.type === 'content_match') {
      return {
        type: 'regex',
        pattern: legacyCond.pattern,
      };
    }

    return condition as RuleCondition;
  }

  /**
   * Get rules by category
   */
  getRulesByCategory(category: RuleCategory): ConstitutionalRule[] {
    return this.getRules().filter((r) => r.category === category);
  }

  /**
   * Get agent constitutional status
   */
  getAgentStatus(agentId: string): AgentConstitutionalStatus | undefined {
    return this.agentStatus.get(agentId);
  }

  /**
   * Get violations for an agent
   */
  getViolations(agentId: string): ConstitutionalViolation[] {
    return Array.from(this.violations.values()).filter(
      (v) => v.agentId === agentId
    );
  }

  /**
   * Propose a constitutional amendment
   */
  proposeAmendment(input: {
    type: 'create' | 'modify' | 'repeal' | 'new';
    ruleId?: string;
    newRule?: Partial<ConstitutionalRule>;
    rationale?: string;
    justification?: string;
    proposerDid?: string;
    proposerId?: string;
  }): AmendmentProposal {
    // Normalize type: 'new' is alias for 'create'
    const normalizedType = input.type === 'new' ? 'create' : input.type;
    // Accept both rationale and justification
    const rationale = input.rationale ?? input.justification ?? '';
    // Accept both proposerDid and proposerId
    const proposerDid = input.proposerDid ?? input.proposerId ?? 'unknown';

    // Validate
    if (normalizedType !== 'create' && !input.ruleId) {
      throw new Error('ruleId required for modify/repeal');
    }
    if (normalizedType !== 'repeal' && !input.newRule) {
      throw new Error('newRule required for create/modify');
    }

    // Check if rule exists for modify/repeal
    if (input.ruleId) {
      const existingRule = this.rules.get(input.ruleId);
      if (!existingRule) {
        throw new Error(`Rule ${input.ruleId} not found`);
      }
      // Foundational rules can only be modified, not repealed
      if (normalizedType === 'repeal' && existingRule.createdBy === 'foundational') {
        throw new Error('Foundational rules cannot be repealed');
      }
    }

    // Determine threshold
    const targetRule = input.ruleId ? this.rules.get(input.ruleId) : undefined;
    const isInviolable =
      targetRule?.severity === 'inviolable' ||
      input.newRule?.severity === 'inviolable';
    const threshold = isInviolable
      ? this.config.supermajorityForInviolable
      : this.config.amendmentThreshold;

    const proposal: AmendmentProposal = {
      id: `amendment-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
      type: input.type,  // Keep original type for test compatibility
      ruleId: input.ruleId,
      newRule: input.newRule,
      rationale,
      proposerDid,
      status: 'pending',  // Test expects 'pending' not 'draft'
      votes: { for: 0, against: 0, abstain: 0 },
      threshold,
      createdAt: Date.now(),
    };

    this.proposals.set(proposal.id, proposal);
    return proposal;
  }

  /**
   * Get a proposal by ID
   */
  getProposal(proposalId: string): AmendmentProposal | undefined {
    return this.proposals.get(proposalId);
  }

  /**
   * Start voting on an amendment
   */
  startVoting(proposalId: string): AmendmentProposal {
    const proposal = this.proposals.get(proposalId);
    if (!proposal) {
      throw new Error(`Proposal ${proposalId} not found`);
    }
    if (proposal.status !== 'draft') {
      throw new Error('Can only start voting on draft proposals');
    }

    proposal.status = 'voting';
    proposal.votingEndsAt = Date.now() + this.config.amendmentVotingPeriodMs;
    return proposal;
  }

  /**
   * Cast a vote on an amendment (internal)
   */
  vote(
    proposalId: string,
    voterDid: string,
    choice: 'for' | 'against' | 'abstain',
    weight: number = 1
  ): void {
    const proposal = this.proposals.get(proposalId);
    if (!proposal) {
      throw new Error(`Proposal ${proposalId} not found`);
    }
    // Allow voting on both 'voting' and 'pending' status
    if (proposal.status !== 'voting' && proposal.status !== 'pending') {
      throw new Error('Voting is not open for this proposal');
    }
    if (proposal.votingEndsAt && Date.now() > proposal.votingEndsAt) {
      throw new Error('Voting period has ended');
    }

    // Track voters to prevent duplicates
    if (!this.proposalVotes.has(proposalId)) {
      this.proposalVotes.set(proposalId, new Set());
    }
    const voters = this.proposalVotes.get(proposalId)!;
    if (voters.has(voterDid)) {
      return; // Silently ignore duplicate votes
    }
    voters.add(voterDid);

    proposal.votes[choice] += weight;
  }

  /**
   * Cast a vote on an amendment (test-compatible API)
   *
   * @param proposalId - Proposal to vote on
   * @param voterId - Voter identifier
   * @param approve - True for 'for', false for 'against'
   */
  voteOnProposal(proposalId: string, voterId: string, approve: boolean): void {
    this.vote(proposalId, voterId, approve ? 'for' : 'against');
  }

  /**
   * Finalize voting and enact if passed
   */
  finalizeVoting(proposalId: string): AmendmentProposal {
    const proposal = this.proposals.get(proposalId);
    if (!proposal) {
      throw new Error(`Proposal ${proposalId} not found`);
    }
    // Allow finalizing from 'voting' or 'pending' status
    if (proposal.status !== 'voting' && proposal.status !== 'pending') {
      throw new Error('Proposal is not in voting status');
    }

    const totalVotes = proposal.votes.for + proposal.votes.against;
    const approvalRate = totalVotes > 0 ? proposal.votes.for / totalVotes : 0;

    if (approvalRate >= proposal.threshold) {
      proposal.status = 'passed';
      this.enactAmendment(proposal);
      // Keep as 'passed' for test compatibility
    } else {
      proposal.status = 'rejected';
    }

    return proposal;
  }

  /**
   * Alias for finalizeVoting (test compatibility)
   */
  async finalizeProposal(proposalId: string): Promise<AmendmentProposal> {
    return this.finalizeVoting(proposalId);
  }

  /**
   * Appeal a violation
   */
  appealViolation(violationId: string, _appealReason: string): void {
    if (!this.config.enableAppeals) {
      throw new Error('Appeals are not enabled');
    }

    const violation = this.violations.get(violationId);
    if (!violation) {
      throw new Error(`Violation ${violationId} not found`);
    }

    const appealDeadline = violation.detectedAt + this.config.appealWindowMs;
    if (Date.now() > appealDeadline) {
      throw new Error('Appeal window has closed');
    }

    violation.appealed = true;
    // In production, this would trigger a governance review process
  }

  /**
   * Generate constitutional audit report
   */
  generateAudit(
    periodStart?: number,
    periodEnd: number = Date.now()
  ): ConstitutionalAudit & { timestamp: number; totalRules: number; totalChecks: number; rulesByCategory: Map<RuleCategory, number>; rulesBySeverity: Map<RuleSeverity, number> } {
    // Default to start of time if not provided
    const start = periodStart ?? 0;

    const relevantViolations = Array.from(this.violations.values()).filter(
      (v) => v.detectedAt >= start && v.detectedAt <= periodEnd
    );

    const violationsByCategory = new Map<RuleCategory, number>();
    const violationsBySeverity = new Map<RuleSeverity, number>();

    for (const violation of relevantViolations) {
      const rule = this.rules.get(violation.ruleId);
      if (rule) {
        const catCount = violationsByCategory.get(rule.category) || 0;
        violationsByCategory.set(rule.category, catCount + 1);

        const sevCount = violationsBySeverity.get(rule.severity) || 0;
        violationsBySeverity.set(rule.severity, sevCount + 1);
      }
    }

    // Count rules by category and severity
    const rulesByCategory = new Map<RuleCategory, number>();
    const rulesBySeverity = new Map<RuleSeverity, number>();

    for (const rule of this.rules.values()) {
      const catCount = rulesByCategory.get(rule.category) || 0;
      rulesByCategory.set(rule.category, catCount + 1);

      const sevCount = rulesBySeverity.get(rule.severity) || 0;
      rulesBySeverity.set(rule.severity, sevCount + 1);
    }

    const agentCount = this.agentStatus.size;
    const totalCompliance = Array.from(this.agentStatus.values()).reduce(
      (sum, s) => sum + s.complianceScore,
      0
    );
    const complianceRate = agentCount > 0 ? totalCompliance / agentCount : 1;

    const recommendations: string[] = [];
    if (violationsByCategory.get('safety') && violationsByCategory.get('safety')! > 5) {
      recommendations.push('Review safety training for agents');
    }
    if (complianceRate < 0.9) {
      recommendations.push('Consider additional agent training or rule clarification');
    }

    const timestamp = Date.now();

    return {
      id: `audit-${timestamp}`,
      periodStart: start,
      periodEnd,
      agentsAudited: agentCount,
      interactionsChecked: this.checkCount,
      violationsFound: relevantViolations.length,
      violationsByCategory,
      violationsBySeverity,
      complianceRate,
      recommendations,
      generatedAt: timestamp,
      // Test-compatible fields
      timestamp,
      totalRules: this.rules.size,
      totalChecks: this.checkCount,
      rulesByCategory,
      rulesBySeverity,
    };
  }

  // ==========================================================================
  // PRIVATE METHODS
  // ==========================================================================

  /**
   * Check a single rule against content
   */
  private checkRule(
    rule: ConstitutionalRule,
    content: string,
    agentId: string,
    context: { domain?: string; conversationId?: string; citizenDid?: string }
  ): ConstitutionalViolation | null {
    if (!rule.condition) {
      return null; // No machine-checkable condition
    }

    let violated = false;
    const lowerContent = content.toLowerCase();

    // Get condition type - also check legacy format
    const conditionType = rule.condition.type as string;

    switch (conditionType) {
      case 'keyword':
        if (rule.condition.keywords) {
          violated = rule.condition.keywords.some((kw) =>
            lowerContent.includes(kw.toLowerCase())
          );
        }
        break;

      case 'regex':
      case 'content_match': // Alias for regex
        if (rule.condition.pattern) {
          try {
            const regex = new RegExp(rule.condition.pattern, 'i');
            violated = regex.test(content);
          } catch {
            // Invalid regex
          }
        }
        break;

      case 'semantic':
        // In production, would use embeddings
        // For now, use keyword-based heuristic
        violated = this.semanticViolationCheck(rule, content);
        break;

      case 'confidence_check':
        // Handled separately in checkContent
        break;

      case 'custom':
        // Would call registered custom check function
        break;
    }

    // Apply negation
    if (rule.condition.negate) {
      violated = !violated;
    }

    if (!violated) {
      return null;
    }

    return {
      id: `violation-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
      ruleId: rule.id,
      agentId,
      category: rule.category,
      severity: rule.severity,
      description: `Violated rule "${rule.name}": ${rule.statement}`,
      violatingContent: content.substring(0, 500),
      context,
      detectedAt: Date.now(),
      actionTaken: 'logged',
      appealed: false,
    };
  }

  /**
   * Semantic violation check (simplified)
   */
  private semanticViolationCheck(
    rule: ConstitutionalRule,
    content: string
  ): boolean {
    const lowerContent = content.toLowerCase();

    // Safety rule checks
    if (rule.category === 'safety') {
      const harmfulPatterns = [
        'how to make a weapon',
        'how to make a bomb',
        'how to hurt',
        'how to kill',
        'instructions for violence',
        'self harm',
        'suicide method',
      ];
      return harmfulPatterns.some((p) => lowerContent.includes(p));
    }

    // Privacy rule checks
    if (rule.category === 'privacy') {
      const sensitivePatterns = [
        'tell me your ssn',
        'what is your social security',
        'give me your password',
        'share your credit card',
      ];
      return sensitivePatterns.some((p) => lowerContent.includes(p));
    }

    return false;
  }

  /**
   * Enact an approved amendment
   */
  private enactAmendment(proposal: AmendmentProposal): void {
    switch (proposal.type) {
      case 'create':
        if (proposal.newRule) {
          const newRule: ConstitutionalRule = {
            id: `rule-${Date.now()}`,
            name: proposal.newRule.name || 'New Rule',
            description: proposal.newRule.description || '',
            category: proposal.newRule.category || 'safety',
            severity: proposal.newRule.severity || 'mandatory',
            statement: proposal.newRule.statement || '',
            condition: proposal.newRule.condition,
            remediation: proposal.newRule.remediation,
            active: true,
            version: 1,
            createdAt: Date.now(),
            createdBy: proposal.id,
            amendments: [],
          };
          this.rules.set(newRule.id, newRule);
        }
        break;

      case 'modify':
        if (proposal.ruleId && proposal.newRule) {
          const rule = this.rules.get(proposal.ruleId);
          if (rule) {
            const amendment: RuleAmendment = {
              id: `amend-${Date.now()}`,
              proposalId: proposal.id,
              changes: Object.entries(proposal.newRule).map(([field, value]) => ({
                field,
                oldValue: (rule as unknown as Record<string, unknown>)[field],
                newValue: value,
              })),
              timestamp: Date.now(),
              votesFor: proposal.votes.for,
              votesAgainst: proposal.votes.against,
            };

            // Apply changes
            Object.assign(rule, proposal.newRule);
            rule.version++;
            rule.amendedAt = Date.now();
            rule.amendments.push(amendment);
          }
        }
        break;

      case 'repeal':
        if (proposal.ruleId) {
          const rule = this.rules.get(proposal.ruleId);
          if (rule) {
            rule.active = false;
          }
        }
        break;
    }
  }

  /**
   * Initialize agent status
   */
  private initAgentStatus(agentId: string): AgentConstitutionalStatus {
    return {
      agentId,
      isCompliant: true,
      compliant: true,
      totalViolations: 0,
      violationsByCategory: {},
      violationsBySeverity: {},
      recentViolations: 0,
      complianceScore: 1.0,
      trustScore: 0.5,
      active: true,
    };
  }

  /**
   * Calculate compliance score
   */
  private calculateComplianceScore(status: AgentConstitutionalStatus): number {
    // Simple decay based on violations
    const base = 1.0;
    const penalty = status.recentViolations * 0.1;
    return Math.max(0, base - penalty);
  }
}

/**
 * Content check result
 */
export interface ContentCheckResult {
  /** Whether the content is allowed */
  allowed: boolean;
  /** Alias for allowed (test compatibility) */
  approved: boolean;
  /** Whether it was blocked */
  blocked: boolean;
  /** Violations found */
  violations: ConstitutionalViolation[];
  /** Warnings (non-blocking issues) */
  warnings: string[];
  /** Reason message */
  reason?: string;
  /** Suggested action for remediation */
  suggestedAction?: string;
}

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

/**
 * Create a constitutional governor
 */
export function createConstitutionalGovernor(
  config?: Partial<ConstitutionalConfig>
): ConstitutionalGovernor {
  return new ConstitutionalGovernor(config);
}

// Singleton instance
let governorInstance: ConstitutionalGovernor | null = null;

/**
 * Get the global constitutional governor
 */
export function getConstitutionalGovernor(
  config?: Partial<ConstitutionalConfig>
): ConstitutionalGovernor {
  if (!governorInstance) {
    governorInstance = new ConstitutionalGovernor(config);
    // Rules are loaded synchronously in constructor
  }
  return governorInstance;
}

/**
 * Reset the global governor (for testing)
 */
export function resetConstitutionalGovernor(): void {
  governorInstance = null;
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  ConstitutionalGovernor,
  DEFAULT_CONSTITUTIONAL_CONFIG,
  FOUNDATIONAL_RULES,
};
