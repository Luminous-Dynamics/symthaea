// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Civic Configuration Profiles for GovOS
 *
 * GovOS IS Mycelix configured for civic use. These profiles configure
 * existing Mycelix modules for government and civic applications.
 *
 * @module config/civic
 */

import {
  type MycelixConfig,
  type MATLConfig,
  type FLConfig,
  type SecurityConfig,
  type BridgeConfig,
  type ClientConfig,
  type LogConfig,
  type PresetValidationWarning,
  type PresetValidationResult,
  DEFAULT_MATL_CONFIG,
  DEFAULT_FL_CONFIG,
  DEFAULT_SECURITY_CONFIG,
  DEFAULT_BRIDGE_CONFIG,
  DEFAULT_CLIENT_CONFIG,
  DEFAULT_LOG_CONFIG,
  initConfig,
  type ConfigManager,
} from './index.js';

// ============================================================================
// Civic Configuration Types
// ============================================================================

/**
 * Civic domain categories for configuration
 */
export type CivicDomain =
  | 'municipal'      // General city/county services
  | 'benefits'       // Social services, SNAP, housing assistance
  | 'permits'        // Building, business, vehicle permits
  | 'voting'         // Elections, proposals, civic participation
  | 'justice'        // Appeals, disputes, restorative justice
  | 'emergency'      // Crisis response, urgent decisions
  | 'healthcare'     // Health services coordination
  | 'education';     // Educational programs, credentials

/**
 * Civic verification levels (maps to E-N-M epistemic levels)
 */
export type CivicVerificationLevel =
  | 'anonymous'      // E0 - No verification, basic access
  | 'self_attested'  // E1 - User-provided information
  | 'verified'       // E2 - Third-party verification
  | 'authenticated'  // E3 - Government ID verified
  | 'institutional'; // E4 - Organizational verification

/**
 * Civic workflow types with governance parameters
 */
export interface CivicWorkflowConfig {
  /** Workflow name */
  name: string;
  /** Required verification level */
  verificationLevel: CivicVerificationLevel;
  /** Quorum requirement (0-1) */
  quorum: number;
  /** Approval threshold (0-1) */
  approvalThreshold: number;
  /** Maximum processing time in ms */
  maxProcessingTime: number;
  /** Whether appeals are allowed */
  appealsEnabled: boolean;
  /** Appeal deadline in ms after decision */
  appealDeadline: number;
  /** Human oversight required */
  humanOversightRequired: boolean;
  /** Audit level: minimal, standard, comprehensive */
  auditLevel: 'minimal' | 'standard' | 'comprehensive';
}

/**
 * Extended civic configuration
 */
export interface CivicConfig {
  /** Active civic domain */
  domain: CivicDomain;
  /** Workflow configurations by type */
  workflows: Record<string, CivicWorkflowConfig>;
  /** Trust score display (show citizens their score breakdown) */
  trustScoreTransparency: boolean;
  /** Algorithm explanation requirement */
  algorithmExplanationRequired: boolean;
  /** Multi-channel access (SMS, web, phone) */
  multiChannelAccess: boolean;
  /** Default language */
  defaultLanguage: string;
  /** Supported languages */
  supportedLanguages: string[];
  /** Accessibility compliance level */
  accessibilityLevel: 'WCAG_AA' | 'WCAG_AAA';
  /** Data retention period in days */
  dataRetentionDays: number;
  /** Enable proactive outreach */
  proactiveOutreach: boolean;
}

// ============================================================================
// Civic-Specific MATL Configurations
// ============================================================================

/**
 * Municipal services MATL config
 * Standard trust thresholds for general government operations
 */
export const MUNICIPAL_MATL_CONFIG: MATLConfig = {
  ...DEFAULT_MATL_CONFIG,
  defaultTrustThreshold: 0.5,
  maxByzantineTolerance: 0.34,
  adaptiveWindowSize: 100,
  sigmaMultiplier: 2.0,
  laplacePrior: 1,
};

/**
 * Benefits eligibility MATL config
 * Lower barriers to access, but verify accuracy
 */
export const BENEFITS_MATL_CONFIG: MATLConfig = {
  ...DEFAULT_MATL_CONFIG,
  defaultTrustThreshold: 0.4, // Lower barrier for benefits access
  maxByzantineTolerance: 0.34,
  adaptiveWindowSize: 50, // Faster adaptation for individual cases
  sigmaMultiplier: 2.5, // Higher sensitivity to anomalies (fraud detection)
  laplacePrior: 2, // More weight to prior (benefit of doubt)
};

/**
 * Voting/governance MATL config
 * High integrity requirements
 */
export const VOTING_MATL_CONFIG: MATLConfig = {
  ...DEFAULT_MATL_CONFIG,
  defaultTrustThreshold: 0.6, // Higher threshold for voting integrity
  maxByzantineTolerance: 0.33, // Strict Byzantine tolerance
  adaptiveWindowSize: 200, // Larger window for voting patterns
  sigmaMultiplier: 3.0, // Very sensitive to anomalies
  laplacePrior: 1,
};

/**
 * Emergency/crisis MATL config
 * Fast decisions, lower thresholds
 */
export const EMERGENCY_MATL_CONFIG: MATLConfig = {
  ...DEFAULT_MATL_CONFIG,
  defaultTrustThreshold: 0.3, // Lower barrier in emergencies
  maxByzantineTolerance: 0.34, // Maximum validated tolerance
  adaptiveWindowSize: 20, // Very fast adaptation
  sigmaMultiplier: 1.5, // Less strict anomaly detection
  laplacePrior: 3, // Strong benefit of doubt
};

/**
 * Justice/appeals MATL config
 * Balanced, fair, transparent
 */
export const JUSTICE_MATL_CONFIG: MATLConfig = {
  ...DEFAULT_MATL_CONFIG,
  defaultTrustThreshold: 0.5,
  maxByzantineTolerance: 0.4,
  adaptiveWindowSize: 150, // Longer-term patterns
  sigmaMultiplier: 2.0,
  laplacePrior: 2, // Presumption of good faith
};

// ============================================================================
// Civic Federated Learning Configurations
// ============================================================================

/**
 * Municipal FL config
 * Collaborative learning across departments
 */
export const MUNICIPAL_FL_CONFIG: FLConfig = {
  ...DEFAULT_FL_CONFIG,
  minParticipants: 3,
  maxParticipants: 50,
  roundTimeout: 60000,
  byzantineTolerance: 0.33,
  defaultAggregationMethod: 'trust_weighted',
  trustThreshold: 0.5,
};

/**
 * Benefits FL config
 * Privacy-preserving eligibility learning
 */
export const BENEFITS_FL_CONFIG: FLConfig = {
  ...DEFAULT_FL_CONFIG,
  minParticipants: 5, // More participants for privacy
  maxParticipants: 100,
  roundTimeout: 120000, // Longer timeout for complex eligibility
  byzantineTolerance: 0.33,
  defaultAggregationMethod: 'trimmed_mean', // Robust to outliers
  trustThreshold: 0.5,
};

/**
 * Voting FL config
 * High-integrity collaborative counting
 */
export const VOTING_FL_CONFIG: FLConfig = {
  ...DEFAULT_FL_CONFIG,
  minParticipants: 7, // Higher minimum for integrity
  maxParticipants: 21, // Odd number for tie-breaking
  roundTimeout: 30000, // Fast rounds
  byzantineTolerance: 0.25, // Strict tolerance
  defaultAggregationMethod: 'krum', // Byzantine-resilient
  trustThreshold: 0.7, // High trust requirement
};

// ============================================================================
// Civic Security Configurations
// ============================================================================

/**
 * Municipal security config
 * Standard government security
 */
export const MUNICIPAL_SECURITY_CONFIG: SecurityConfig = {
  ...DEFAULT_SECURITY_CONFIG,
  hashAlgorithm: 'SHA-256',
  rateLimitMaxRequests: 100,
  rateLimitWindowMs: 60000,
  enableConstantTime: true,
  defaultSecretTTL: 3600000, // 1 hour
  maxAuditLogSize: 50000,
  maxJsonDepth: 10,
  maxJsonSize: 2097152, // 2MB
};

/**
 * Benefits security config
 * Enhanced privacy for sensitive data
 */
export const BENEFITS_SECURITY_CONFIG: SecurityConfig = {
  ...DEFAULT_SECURITY_CONFIG,
  hashAlgorithm: 'SHA-384',
  rateLimitMaxRequests: 50, // Lower rate limit
  rateLimitWindowMs: 60000,
  enableConstantTime: true,
  defaultSecretTTL: 1800000, // 30 minutes
  maxAuditLogSize: 100000, // Comprehensive audit
  maxJsonDepth: 8,
  maxJsonSize: 1048576, // 1MB
};

/**
 * Voting security config
 * Maximum security for election integrity
 */
export const VOTING_SECURITY_CONFIG: SecurityConfig = {
  ...DEFAULT_SECURITY_CONFIG,
  hashAlgorithm: 'SHA-512',
  rateLimitMaxRequests: 30, // Strict rate limiting
  rateLimitWindowMs: 60000,
  enableConstantTime: true,
  defaultSecretTTL: 300000, // 5 minutes
  maxAuditLogSize: 200000, // Maximum audit trail
  maxJsonDepth: 5,
  maxJsonSize: 524288, // 512KB
};

// ============================================================================
// Default Civic Workflow Configurations
// ============================================================================

/**
 * Standard permit application workflow
 */
export const PERMIT_WORKFLOW: CivicWorkflowConfig = {
  name: 'permit_application',
  verificationLevel: 'verified',
  quorum: 0.5,
  approvalThreshold: 0.6,
  maxProcessingTime: 14 * 24 * 60 * 60 * 1000, // 14 days
  appealsEnabled: true,
  appealDeadline: 30 * 24 * 60 * 60 * 1000, // 30 days
  humanOversightRequired: true,
  auditLevel: 'standard',
};

/**
 * Benefits eligibility workflow
 */
export const BENEFITS_WORKFLOW: CivicWorkflowConfig = {
  name: 'benefits_eligibility',
  verificationLevel: 'self_attested', // Lower barrier to apply
  quorum: 0.0, // Individual determination
  approvalThreshold: 0.5,
  maxProcessingTime: 30 * 24 * 60 * 60 * 1000, // 30 days
  appealsEnabled: true,
  appealDeadline: 60 * 24 * 60 * 60 * 1000, // 60 days
  humanOversightRequired: true,
  auditLevel: 'comprehensive',
};

/**
 * Voting/proposal workflow
 */
export const VOTING_WORKFLOW: CivicWorkflowConfig = {
  name: 'civic_voting',
  verificationLevel: 'authenticated',
  quorum: 0.1, // 10% quorum for proposals
  approvalThreshold: 0.5, // Simple majority
  maxProcessingTime: 7 * 24 * 60 * 60 * 1000, // 7 days voting period
  appealsEnabled: false, // Votes are final
  appealDeadline: 0,
  humanOversightRequired: false,
  auditLevel: 'comprehensive',
};

/**
 * Emergency response workflow
 */
export const EMERGENCY_WORKFLOW: CivicWorkflowConfig = {
  name: 'emergency_response',
  verificationLevel: 'anonymous', // No barriers in emergency
  quorum: 0.0,
  approvalThreshold: 0.3, // Low threshold for action
  maxProcessingTime: 4 * 60 * 60 * 1000, // 4 hours max
  appealsEnabled: true,
  appealDeadline: 72 * 60 * 60 * 1000, // 72 hours
  humanOversightRequired: true,
  auditLevel: 'comprehensive',
};

/**
 * Appeal/dispute resolution workflow
 */
export const APPEAL_WORKFLOW: CivicWorkflowConfig = {
  name: 'appeal_resolution',
  verificationLevel: 'verified',
  quorum: 0.6, // Higher quorum for appeals
  approvalThreshold: 0.5,
  maxProcessingTime: 45 * 24 * 60 * 60 * 1000, // 45 days
  appealsEnabled: true, // Can appeal the appeal
  appealDeadline: 30 * 24 * 60 * 60 * 1000, // 30 days
  humanOversightRequired: true,
  auditLevel: 'comprehensive',
};

// ============================================================================
// Civic Configuration Presets
// ============================================================================

/**
 * Default civic configuration
 */
export const DEFAULT_CIVIC_CONFIG: CivicConfig = {
  domain: 'municipal',
  workflows: {
    permit: PERMIT_WORKFLOW,
    benefits: BENEFITS_WORKFLOW,
    voting: VOTING_WORKFLOW,
    emergency: EMERGENCY_WORKFLOW,
    appeal: APPEAL_WORKFLOW,
  },
  trustScoreTransparency: true,
  algorithmExplanationRequired: true,
  multiChannelAccess: true,
  defaultLanguage: 'en',
  supportedLanguages: ['en', 'es', 'zh', 'vi', 'ko', 'tl'],
  accessibilityLevel: 'WCAG_AA',
  dataRetentionDays: 2555, // 7 years for government records
  proactiveOutreach: true,
};

/**
 * Municipal services configuration preset
 * General purpose city/county government operations
 */
export const CIVIC_MUNICIPAL_CONFIG: Partial<MycelixConfig> = {
  matl: MUNICIPAL_MATL_CONFIG,
  fl: MUNICIPAL_FL_CONFIG,
  security: MUNICIPAL_SECURITY_CONFIG,
  bridge: {
    ...DEFAULT_BRIDGE_CONFIG,
    enableCrossHappReputation: true,
    defaultReputationWeight: 0.8,
  },
  client: {
    ...DEFAULT_CLIENT_CONFIG,
    connectionTimeout: 30000,
    requestTimeout: 15000,
    autoReconnect: true,
  },
  logging: {
    ...DEFAULT_LOG_CONFIG,
    level: 'info',
    structured: true,
    timestamps: true,
  },
};

/**
 * Benefits/social services configuration preset
 * Optimized for accessibility and privacy
 */
export const CIVIC_BENEFITS_CONFIG: Partial<MycelixConfig> = {
  matl: BENEFITS_MATL_CONFIG,
  fl: BENEFITS_FL_CONFIG,
  security: BENEFITS_SECURITY_CONFIG,
  bridge: {
    ...DEFAULT_BRIDGE_CONFIG,
    enableCrossHappReputation: true,
    defaultReputationWeight: 0.6, // Lower weight, benefit of doubt
  },
  client: {
    ...DEFAULT_CLIENT_CONFIG,
    connectionTimeout: 60000, // Longer timeout for slow connections
    requestTimeout: 30000,
    autoReconnect: true,
  },
  logging: {
    ...DEFAULT_LOG_CONFIG,
    level: 'info',
    structured: true,
    timestamps: true,
  },
};

/**
 * Voting/governance configuration preset
 * Maximum integrity and auditability
 */
export const CIVIC_VOTING_CONFIG: Partial<MycelixConfig> = {
  matl: VOTING_MATL_CONFIG,
  fl: VOTING_FL_CONFIG,
  security: VOTING_SECURITY_CONFIG,
  bridge: {
    ...DEFAULT_BRIDGE_CONFIG,
    enableCrossHappReputation: true,
    defaultReputationWeight: 1.0, // Full reputation weight
  },
  client: {
    ...DEFAULT_CLIENT_CONFIG,
    connectionTimeout: 15000,
    requestTimeout: 10000,
    autoReconnect: true,
    maxReconnectAttempts: 10, // More retries for voting
  },
  logging: {
    ...DEFAULT_LOG_CONFIG,
    level: 'info',
    structured: true,
    timestamps: true,
  },
};

/**
 * Emergency/crisis configuration preset
 * Fast response, accessible
 */
export const CIVIC_EMERGENCY_CONFIG: Partial<MycelixConfig> = {
  matl: EMERGENCY_MATL_CONFIG,
  fl: {
    ...DEFAULT_FL_CONFIG,
    minParticipants: 2, // Lower minimum in emergencies
    roundTimeout: 10000, // Very fast rounds
    byzantineTolerance: 0.34, // Maximum validated tolerance
  },
  security: {
    ...MUNICIPAL_SECURITY_CONFIG,
    rateLimitMaxRequests: 500, // High rate limit for emergencies
  },
  bridge: {
    ...DEFAULT_BRIDGE_CONFIG,
    defaultMessageTTL: 5000, // Short TTL for urgent messages
  },
  client: {
    ...DEFAULT_CLIENT_CONFIG,
    connectionTimeout: 5000, // Fast connections
    requestTimeout: 5000,
    autoReconnect: true,
    maxReconnectAttempts: 20, // Many retries
    reconnectDelay: 500, // Fast retry
  },
  logging: {
    ...DEFAULT_LOG_CONFIG,
    level: 'warn', // Minimal logging for speed
    structured: true,
    timestamps: true,
  },
};

/**
 * Justice/appeals configuration preset
 * Fair, transparent, auditable
 */
export const CIVIC_JUSTICE_CONFIG: Partial<MycelixConfig> = {
  matl: JUSTICE_MATL_CONFIG,
  fl: {
    ...DEFAULT_FL_CONFIG,
    minParticipants: 5,
    byzantineTolerance: 0.33,
    defaultAggregationMethod: 'median', // Robust to extremes
  },
  security: {
    ...BENEFITS_SECURITY_CONFIG,
    maxAuditLogSize: 200000, // Comprehensive audit
  },
  bridge: {
    ...DEFAULT_BRIDGE_CONFIG,
    enableCrossHappReputation: true,
  },
  client: {
    ...DEFAULT_CLIENT_CONFIG,
    connectionTimeout: 45000, // Patient connections
    requestTimeout: 30000,
  },
  logging: {
    ...DEFAULT_LOG_CONFIG,
    level: 'info',
    structured: true,
    timestamps: true,
  },
};

// ============================================================================
// Civic Configuration Validation
// ============================================================================

/**
 * Validate civic-specific configuration
 */
export function validateCivicConfig(
  config: Partial<MycelixConfig>,
  domain: CivicDomain
): PresetValidationWarning[] {
  const warnings: PresetValidationWarning[] = [];

  // Domain-specific validations
  switch (domain) {
    case 'voting':
      if ((config.matl?.maxByzantineTolerance ?? 0.34) > 0.33) {
        warnings.push({
          code: 'CIVIC_VOTING_BYZANTINE',
          message: 'Byzantine tolerance may be too high for voting integrity',
          suggestion: 'Set maxByzantineTolerance ≤ 0.33 for elections',
          severity: 'warning',
        });
      }
      if (config.security?.hashAlgorithm !== 'SHA-512') {
        warnings.push({
          code: 'CIVIC_VOTING_HASH',
          message: 'Consider SHA-512 for maximum election integrity',
          suggestion: 'Use SHA-512 hash algorithm for voting applications',
          severity: 'info',
        });
      }
      break;

    case 'benefits':
      if ((config.matl?.defaultTrustThreshold ?? 0.5) > 0.5) {
        warnings.push({
          code: 'CIVIC_BENEFITS_BARRIER',
          message: 'High trust threshold may create barriers to benefits access',
          suggestion: 'Consider lower defaultTrustThreshold for accessibility',
          severity: 'warning',
        });
      }
      break;

    case 'emergency':
      if ((config.fl?.roundTimeout ?? 30000) > 15000) {
        warnings.push({
          code: 'CIVIC_EMERGENCY_TIMEOUT',
          message: 'FL round timeout may be too long for emergency response',
          suggestion: 'Use shorter roundTimeout for faster emergency decisions',
          severity: 'warning',
        });
      }
      break;

    case 'justice':
      if (config.security?.maxAuditLogSize && config.security.maxAuditLogSize < 100000) {
        warnings.push({
          code: 'CIVIC_JUSTICE_AUDIT',
          message: 'Audit log size may be insufficient for legal requirements',
          suggestion: 'Increase maxAuditLogSize for comprehensive appeal records',
          severity: 'warning',
        });
      }
      break;
  }

  // Universal civic requirements
  if (config.logging?.level === 'silent') {
    warnings.push({
      code: 'CIVIC_NO_LOGGING',
      message: 'Silent logging not recommended for government applications',
      suggestion: 'Enable at least "info" level logging for accountability',
      severity: 'error',
    });
  }

  return warnings;
}

/**
 * Validate civic configuration preset
 */
export function validateCivicPreset(
  config: Partial<MycelixConfig>,
  domain: CivicDomain
): PresetValidationResult {
  const civicWarnings = validateCivicConfig(config, domain);
  const hasErrors = civicWarnings.some((w) => w.severity === 'error');

  return {
    valid: !hasErrors,
    warnings: civicWarnings,
    detectedType: 'custom',
  };
}

// ============================================================================
// Civic Configuration Factory
// ============================================================================

/**
 * Create a civic configuration for a specific domain
 */
export function createCivicConfig(
  domain: CivicDomain,
  overrides: Partial<MycelixConfig> = {}
): Partial<MycelixConfig> {
  const baseConfigs: Record<CivicDomain, Partial<MycelixConfig>> = {
    municipal: CIVIC_MUNICIPAL_CONFIG,
    benefits: CIVIC_BENEFITS_CONFIG,
    permits: CIVIC_MUNICIPAL_CONFIG,
    voting: CIVIC_VOTING_CONFIG,
    justice: CIVIC_JUSTICE_CONFIG,
    emergency: CIVIC_EMERGENCY_CONFIG,
    healthcare: CIVIC_BENEFITS_CONFIG, // Similar privacy requirements
    education: CIVIC_MUNICIPAL_CONFIG,
  };

  const baseConfig = baseConfigs[domain];

  // Deep merge overrides
  return {
    matl: { ...baseConfig.matl, ...overrides.matl } as MATLConfig,
    fl: { ...baseConfig.fl, ...overrides.fl } as FLConfig,
    security: { ...baseConfig.security, ...overrides.security } as SecurityConfig,
    bridge: { ...baseConfig.bridge, ...overrides.bridge } as BridgeConfig,
    client: { ...baseConfig.client, ...overrides.client } as ClientConfig,
    logging: { ...baseConfig.logging, ...overrides.logging } as LogConfig,
  };
}

/**
 * Initialize Mycelix with civic configuration
 */
export function initCivicConfig(
  domain: CivicDomain,
  overrides: Partial<MycelixConfig> = {}
): ConfigManager {
  const config = createCivicConfig(domain, overrides);
  return initConfig(config);
}

/**
 * Create a named civic preset
 */
export function createCivicPreset(
  name: string,
  domain: CivicDomain,
  overrides: Partial<MycelixConfig> = {}
): {
  name: string;
  domain: CivicDomain;
  config: Partial<MycelixConfig>;
  civic: CivicConfig;
  apply: () => ConfigManager;
  validate: () => PresetValidationResult;
} {
  const config = createCivicConfig(domain, overrides);

  return {
    name,
    domain,
    config,
    civic: { ...DEFAULT_CIVIC_CONFIG, domain },
    apply: () => initCivicConfig(domain, overrides),
    validate: () => validateCivicPreset(config, domain),
  };
}

// ============================================================================
// Pre-built Civic Presets
// ============================================================================

/**
 * MY GOV Dashboard preset
 * Unified citizen portal configuration
 */
export const MYGOV_DASHBOARD_PRESET = createCivicPreset('mygov-dashboard', 'municipal', {
  client: {
    ...DEFAULT_CLIENT_CONFIG,
    enableBatching: true, // Batch requests for dashboard
    maxBatchSize: 20,
    batchTimeout: 100,
  },
});

/**
 * SMS Gateway preset
 * Mobile-first civic access
 */
export const SMS_GATEWAY_PRESET = createCivicPreset('sms-gateway', 'municipal', {
  client: {
    ...DEFAULT_CLIENT_CONFIG,
    connectionTimeout: 10000, // Fast for SMS
    requestTimeout: 5000,
  },
  security: {
    ...MUNICIPAL_SECURITY_CONFIG,
    rateLimitMaxRequests: 200, // Higher for SMS volume
  },
});

/**
 * Civitas AI Assistant preset
 * AI-assisted civic navigation
 */
export const CIVITAS_PRESET = createCivicPreset('civitas', 'municipal', {
  matl: {
    ...MUNICIPAL_MATL_CONFIG,
    defaultTrustThreshold: 0.4, // Helpful, not paranoid
  },
  client: {
    ...DEFAULT_CLIENT_CONFIG,
    requestTimeout: 30000, // Longer for AI processing
  },
});

/**
 * Election integrity preset
 * Maximum security voting
 */
export const ELECTION_PRESET = createCivicPreset('election', 'voting', {
  fl: {
    ...VOTING_FL_CONFIG,
    minParticipants: 11, // Higher for elections
    byzantineTolerance: 0.2, // Very strict
  },
});

/**
 * Crisis response preset
 * Emergency operations
 */
export const CRISIS_PRESET = createCivicPreset('crisis', 'emergency', {});

// ============================================================================
// Symthaea Civic Agents Configuration
// ============================================================================

/**
 * Symthaea agent types for civic domains
 */
export type SymthaeaAgentType =
  | 'sophia-tax'       // Tax questions, deductions, filing
  | 'sophia-benefits'  // SNAP, Medicaid, housing assistance
  | 'sophia-permits'   // Business, building, vehicle permits
  | 'sophia-civitas'   // General civic navigation
  | 'sophia-appeals'   // Appeal and dispute guidance
  | 'sophia-emergency' // Crisis/emergency assistance
  | 'sophia-education' // Educational program guidance
  | 'sophia-health';   // Healthcare navigation

/**
 * Symthaea agent configuration
 */
export interface SymthaeaAgentConfig {
  /** Agent type identifier */
  type: SymthaeaAgentType;
  /** Human-readable name */
  name: string;
  /** Agent description */
  description: string;
  /** Primary civic domain */
  domain: CivicDomain;
  /** Secondary domains this agent can assist with */
  secondaryDomains: CivicDomain[];
  /** Knowledge base topics */
  knowledgeTopics: string[];
  /** Maximum response time in ms */
  maxResponseTime: number;
  /** When to escalate to human */
  escalationThreshold: number;
  /** Required verification level for queries */
  requiredVerificationLevel: CivicVerificationLevel;
  /** Enable proactive suggestions */
  proactiveSuggestions: boolean;
  /** Supported languages */
  supportedLanguages: string[];
  /** Tone/style of responses */
  communicationStyle: 'formal' | 'friendly' | 'empathetic' | 'urgent';
}

/**
 * Sophia-Tax Agent Configuration
 * Expert in tax questions, deductions, and filing assistance
 */
export const SOPHIA_TAX_AGENT: SymthaeaAgentConfig = {
  type: 'sophia-tax',
  name: 'Sophia Tax Assistant',
  description: 'Expert guidance on tax questions, deductions, and filing',
  domain: 'municipal',
  secondaryDomains: ['benefits'],
  knowledgeTopics: [
    'federal_income_tax',
    'state_income_tax',
    'property_tax',
    'sales_tax',
    'tax_deductions',
    'tax_credits',
    'filing_deadlines',
    'payment_plans',
    'tax_exemptions',
    'small_business_tax',
  ],
  maxResponseTime: 5000,
  escalationThreshold: 0.6, // Escalate if confidence < 60%
  requiredVerificationLevel: 'self_attested',
  proactiveSuggestions: true,
  supportedLanguages: ['en', 'es', 'zh', 'vi'],
  communicationStyle: 'friendly',
};

/**
 * Sophia-Benefits Agent Configuration
 * Expert in social services, SNAP, Medicaid, housing assistance
 */
export const SOPHIA_BENEFITS_AGENT: SymthaeaAgentConfig = {
  type: 'sophia-benefits',
  name: 'Sophia Benefits Navigator',
  description: 'Helps navigate social services, SNAP, Medicaid, and housing programs',
  domain: 'benefits',
  secondaryDomains: ['healthcare', 'municipal'],
  knowledgeTopics: [
    'snap_eligibility',
    'medicaid_enrollment',
    'housing_assistance',
    'utility_assistance',
    'child_care_subsidies',
    'unemployment_benefits',
    'disability_benefits',
    'food_pantries',
    'emergency_assistance',
    'veterans_benefits',
  ],
  maxResponseTime: 8000, // More time for complex eligibility
  escalationThreshold: 0.5, // Lower threshold - always offer human help
  requiredVerificationLevel: 'anonymous', // No barriers to inquiry
  proactiveSuggestions: true,
  supportedLanguages: ['en', 'es', 'zh', 'vi', 'ko', 'tl'],
  communicationStyle: 'empathetic',
};

/**
 * Sophia-Permits Agent Configuration
 * Expert in business, building, and vehicle permits
 */
export const SOPHIA_PERMITS_AGENT: SymthaeaAgentConfig = {
  type: 'sophia-permits',
  name: 'Sophia Permits Guide',
  description: 'Guides through permit applications for business, building, and vehicles',
  domain: 'permits',
  secondaryDomains: ['municipal'],
  knowledgeTopics: [
    'business_licenses',
    'building_permits',
    'zoning_requirements',
    'vehicle_registration',
    'food_service_permits',
    'home_occupation_permits',
    'sign_permits',
    'special_event_permits',
    'contractor_licenses',
    'inspection_scheduling',
  ],
  maxResponseTime: 5000,
  escalationThreshold: 0.7, // Higher threshold - permits are procedural
  requiredVerificationLevel: 'self_attested',
  proactiveSuggestions: true,
  supportedLanguages: ['en', 'es', 'zh'],
  communicationStyle: 'formal',
};

/**
 * Sophia-Civitas Agent Configuration
 * General civic navigation and government services
 */
export const SOPHIA_CIVITAS_AGENT: SymthaeaAgentConfig = {
  type: 'sophia-civitas',
  name: 'Civitas - Your Civic Guide',
  description: 'General assistant for all government services and civic questions',
  domain: 'municipal',
  secondaryDomains: ['benefits', 'permits', 'voting', 'justice'],
  knowledgeTopics: [
    'government_services',
    'civic_participation',
    'public_meetings',
    'elected_officials',
    'service_locations',
    'contact_information',
    'complaint_filing',
    'public_records',
    'community_resources',
    'local_events',
  ],
  maxResponseTime: 3000, // Fast for general queries
  escalationThreshold: 0.4, // Routes to specialists quickly
  requiredVerificationLevel: 'anonymous',
  proactiveSuggestions: true,
  supportedLanguages: ['en', 'es', 'zh', 'vi', 'ko', 'tl', 'ar'],
  communicationStyle: 'friendly',
};

/**
 * Sophia-Appeals Agent Configuration
 * Appeal and dispute resolution guidance
 */
export const SOPHIA_APPEALS_AGENT: SymthaeaAgentConfig = {
  type: 'sophia-appeals',
  name: 'Sophia Appeals Advocate',
  description: 'Guides citizens through appeal processes and dispute resolution',
  domain: 'justice',
  secondaryDomains: ['benefits', 'permits'],
  knowledgeTopics: [
    'appeal_rights',
    'appeal_deadlines',
    'appeal_process',
    'mediation_options',
    'dispute_resolution',
    'administrative_hearings',
    'legal_aid_resources',
    'documentation_requirements',
    'representation_rights',
    'restorative_justice',
  ],
  maxResponseTime: 6000,
  escalationThreshold: 0.5, // Always offer human help for appeals
  requiredVerificationLevel: 'verified',
  proactiveSuggestions: true,
  supportedLanguages: ['en', 'es', 'zh', 'vi'],
  communicationStyle: 'empathetic',
};

/**
 * Sophia-Emergency Agent Configuration
 * Crisis and emergency assistance
 */
export const SOPHIA_EMERGENCY_AGENT: SymthaeaAgentConfig = {
  type: 'sophia-emergency',
  name: 'Sophia Emergency Response',
  description: 'Immediate assistance during emergencies and crises',
  domain: 'emergency',
  secondaryDomains: ['benefits', 'healthcare'],
  knowledgeTopics: [
    'emergency_services',
    'evacuation_routes',
    'shelter_locations',
    'crisis_hotlines',
    'disaster_assistance',
    'emergency_supplies',
    'first_aid_guidance',
    'utility_emergencies',
    'safety_protocols',
    'community_alerts',
  ],
  maxResponseTime: 2000, // Very fast for emergencies
  escalationThreshold: 0.3, // Very low - always involve humans in emergencies
  requiredVerificationLevel: 'anonymous', // No barriers in crisis
  proactiveSuggestions: true,
  supportedLanguages: ['en', 'es', 'zh', 'vi', 'ko', 'tl', 'ar'],
  communicationStyle: 'urgent',
};

/**
 * Sophia-Education Agent Configuration
 * Educational program navigation
 */
export const SOPHIA_EDUCATION_AGENT: SymthaeaAgentConfig = {
  type: 'sophia-education',
  name: 'Sophia Education Navigator',
  description: 'Guides through educational programs, scholarships, and credentials',
  domain: 'education',
  secondaryDomains: ['benefits'],
  knowledgeTopics: [
    'school_enrollment',
    'scholarship_programs',
    'financial_aid',
    'adult_education',
    'vocational_training',
    'credential_verification',
    'continuing_education',
    'special_education',
    'library_services',
    'tutoring_resources',
  ],
  maxResponseTime: 5000,
  escalationThreshold: 0.6,
  requiredVerificationLevel: 'self_attested',
  proactiveSuggestions: true,
  supportedLanguages: ['en', 'es', 'zh', 'vi', 'ko'],
  communicationStyle: 'friendly',
};

/**
 * Sophia-Health Agent Configuration
 * Healthcare navigation assistance
 */
export const SOPHIA_HEALTH_AGENT: SymthaeaAgentConfig = {
  type: 'sophia-health',
  name: 'Sophia Health Navigator',
  description: 'Navigates healthcare services, insurance, and community health resources',
  domain: 'healthcare',
  secondaryDomains: ['benefits'],
  knowledgeTopics: [
    'healthcare_enrollment',
    'insurance_options',
    'community_health_centers',
    'vaccination_schedules',
    'mental_health_services',
    'substance_abuse_resources',
    'prescription_assistance',
    'preventive_care',
    'maternal_child_health',
    'senior_health_services',
  ],
  maxResponseTime: 6000,
  escalationThreshold: 0.5, // Health questions often need human verification
  requiredVerificationLevel: 'self_attested',
  proactiveSuggestions: true,
  supportedLanguages: ['en', 'es', 'zh', 'vi', 'ko', 'tl'],
  communicationStyle: 'empathetic',
};

/**
 * Registry of all Symthaea civic agents
 */
export const SYMTHAEA_CIVIC_AGENTS: Record<SymthaeaAgentType, SymthaeaAgentConfig> = {
  'sophia-tax': SOPHIA_TAX_AGENT,
  'sophia-benefits': SOPHIA_BENEFITS_AGENT,
  'sophia-permits': SOPHIA_PERMITS_AGENT,
  'sophia-civitas': SOPHIA_CIVITAS_AGENT,
  'sophia-appeals': SOPHIA_APPEALS_AGENT,
  'sophia-emergency': SOPHIA_EMERGENCY_AGENT,
  'sophia-education': SOPHIA_EDUCATION_AGENT,
  'sophia-health': SOPHIA_HEALTH_AGENT,
};

/**
 * Get the appropriate Symthaea agent for a civic domain
 */
export function getAgentForDomain(domain: CivicDomain): SymthaeaAgentConfig {
  const domainToAgent: Record<CivicDomain, SymthaeaAgentType> = {
    municipal: 'sophia-civitas',
    benefits: 'sophia-benefits',
    permits: 'sophia-permits',
    voting: 'sophia-civitas', // Civitas handles voting info
    justice: 'sophia-appeals',
    emergency: 'sophia-emergency',
    healthcare: 'sophia-health',
    education: 'sophia-education',
  };

  return SYMTHAEA_CIVIC_AGENTS[domainToAgent[domain]];
}

/**
 * Get all agents that can handle a specific topic
 */
export function getAgentsForTopic(topic: string): SymthaeaAgentConfig[] {
  return Object.values(SYMTHAEA_CIVIC_AGENTS).filter((agent) =>
    agent.knowledgeTopics.some((t) => t.includes(topic) || topic.includes(t))
  );
}

/**
 * Symthaea conversation context for civic interactions
 */
export interface SymthaeaCivicContext {
  /** Active agent */
  agent: SymthaeaAgentConfig;
  /** Citizen's verification level */
  verificationLevel: CivicVerificationLevel;
  /** Current civic domain */
  domain: CivicDomain;
  /** Conversation language */
  language: string;
  /** Whether human escalation is available */
  humanEscalationAvailable: boolean;
  /** Session start time */
  sessionStartTime: Date;
  /** Topics discussed in this session */
  discussedTopics: string[];
}

/**
 * Create a new Symthaea civic conversation context
 */
export function createCivicContext(
  domain: CivicDomain,
  verificationLevel: CivicVerificationLevel = 'anonymous',
  language: string = 'en'
): SymthaeaCivicContext {
  const agent = getAgentForDomain(domain);

  return {
    agent,
    verificationLevel,
    domain,
    language: agent.supportedLanguages.includes(language) ? language : 'en',
    humanEscalationAvailable: true,
    sessionStartTime: new Date(),
    discussedTopics: [],
  };
}

