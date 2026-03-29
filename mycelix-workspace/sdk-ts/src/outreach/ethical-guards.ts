// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Ethical Guards for Proactive Outreach
 *
 * Safeguards to ensure proactive government outreach respects citizen
 * autonomy, privacy, and dignity. These aren't just compliance checks -
 * they're the conscience of the system.
 *
 * Core Principles:
 * 1. Consent is sovereign - citizens control their data and notifications
 * 2. Transparency over persuasion - inform, don't manipulate
 * 3. Access, not addiction - help citizens, don't create dependency
 * 4. Dignity preservation - never shame or stigmatize
 * 5. Equal treatment - no discrimination in outreach
 *
 * @module outreach/ethical-guards
 */

import type { EligibilityPrediction } from './eligibility.js';
import type { LifeEvent, LifeEventType } from './life-events.js';
import type { Nudge, NudgeChannel, NudgeUrgency } from './nudge-engine.js';

/**
 * Ethical violation types
 */
export type EthicalViolationType =
  | 'consent_not_given'           // Citizen hasn't consented to outreach
  | 'consent_withdrawn'           // Citizen withdrew consent
  | 'dark_pattern'                // Manipulative design detected
  | 'excessive_contact'           // Too many nudges
  | 'sensitive_data_exposure'     // PII in wrong channel
  | 'discrimination_risk'         // Disparate treatment detected
  | 'dignity_violation'           // Shaming or stigmatizing language
  | 'quiet_hours_violation'       // Contacting during restricted times
  | 'channel_mismatch'            // Using wrong channel for content
  | 'urgency_inflation'           // False urgency to prompt action
  | 'benefit_cliff_trap'          // Could push citizen off benefit cliff
  | 'surveillance_overreach'      // Inferring too much from data
  | 'minor_without_guardian'      // Contacting minor directly
  | 'deceased_contact';           // Attempting to contact deceased person

/**
 * Result of an ethical check
 */
export interface EthicalCheckResult {
  /** Whether the action passes ethical checks */
  passed: boolean;
  /** Violations found (empty if passed) */
  violations: EthicalViolation[];
  /** Warnings that don't block but should be noted */
  warnings: EthicalWarning[];
  /** Suggested modifications to make action ethical */
  suggestions: EthicalSuggestion[];
  /** Audit trail entry */
  auditEntry: EthicalAuditEntry;
}

/**
 * An ethical violation
 */
export interface EthicalViolation {
  /** Type of violation */
  type: EthicalViolationType;
  /** Human-readable description */
  description: string;
  /** Severity (blocks action if critical) */
  severity: 'critical' | 'serious' | 'moderate';
  /** Which ethical principle was violated */
  principle: EthicalPrinciple;
  /** Remediation steps */
  remediation: string;
}

/**
 * An ethical warning (doesn't block, but logged)
 */
export interface EthicalWarning {
  /** Warning code */
  code: string;
  /** Description */
  description: string;
  /** Recommendation */
  recommendation: string;
}

/**
 * Suggested modification
 */
export interface EthicalSuggestion {
  /** What to change */
  field: string;
  /** Current value */
  currentValue: unknown;
  /** Suggested value */
  suggestedValue: unknown;
  /** Why this change is suggested */
  reason: string;
}

/**
 * Audit entry for ethical check
 */
export interface EthicalAuditEntry {
  /** Timestamp */
  timestamp: Date;
  /** Action being checked */
  action: string;
  /** Citizen DID (hashed for privacy) */
  citizenDidHash: string;
  /** Result */
  result: 'passed' | 'blocked' | 'modified';
  /** Violations (if any) */
  violationTypes: EthicalViolationType[];
  /** Reviewer (system or human) */
  reviewer: 'automated' | 'human_required';
}

/**
 * Core ethical principles
 */
export type EthicalPrinciple =
  | 'consent_sovereignty'
  | 'transparency'
  | 'dignity'
  | 'equal_treatment'
  | 'privacy'
  | 'autonomy'
  | 'beneficence';

/**
 * Configuration for ethical guards
 */
export interface EthicalGuardConfig {
  /** Require explicit consent for all outreach */
  requireExplicitConsent: boolean;
  /** Maximum nudges per week per citizen */
  maxNudgesPerWeek: number;
  /** Quiet hours (no contact) */
  quietHours: { start: number; end: number };
  /** Sensitive event types requiring extra care */
  sensitiveEventTypes: LifeEventType[];
  /** Channels allowed for sensitive information */
  secureChannels: NudgeChannel[];
  /** Enable benefit cliff detection */
  enableBenefitCliffDetection: boolean;
  /** Minimum confidence for predictions before nudging */
  minPredictionConfidence: number;
  /** Require human review for certain actions */
  requireHumanReview: EthicalViolationType[];
  /** Enable anti-discrimination checks */
  enableDisparateTreatmentDetection: boolean;
}

/**
 * Default ethical guard configuration
 */
export const DEFAULT_ETHICAL_CONFIG: EthicalGuardConfig = {
  requireExplicitConsent: true,
  maxNudgesPerWeek: 3,
  quietHours: { start: 21, end: 8 }, // 9 PM to 8 AM
  sensitiveEventTypes: [
    'death',
    'divorce',
    'incarceration_release',
    'health_crisis',
    'housing_instability',
    'disability_onset',
  ],
  secureChannels: ['dashboard', 'mail'], // Not SMS for sensitive info
  enableBenefitCliffDetection: true,
  minPredictionConfidence: 0.6,
  requireHumanReview: [
    'discrimination_risk',
    'dignity_violation',
    'minor_without_guardian',
  ],
  enableDisparateTreatmentDetection: true,
};

/**
 * Dark pattern detector keywords
 */
const DARK_PATTERN_PHRASES = [
  'act now',
  'limited time',
  'don\'t miss out',
  'last chance',
  'expires soon',
  'urgent action required',
  'you\'ll lose',
  'claim before',
  'others are already',
  'you\'re falling behind',
];

/**
 * Stigmatizing language to avoid
 */
const STIGMATIZING_PHRASES = [
  'welfare',
  'handout',
  'poor',
  'needy',
  'underprivileged',
  'at-risk',
  'disadvantaged',
  'low-income', // prefer "income-eligible"
  'food stamps', // use "SNAP" instead
];

/**
 * Preferred dignified language
 */
const DIGNIFIED_ALTERNATIVES: Record<string, string> = {
  'welfare': 'benefits',
  'handout': 'assistance',
  'poor': 'income-eligible',
  'needy': 'eligible',
  'food stamps': 'SNAP',
  'low-income': 'income-eligible',
};

/**
 * Ethical Guards System
 *
 * Validates all outreach actions against ethical principles.
 */
export class EthicalGuards {
  private config: EthicalGuardConfig;
  private consentRegistry: Map<string, CitizenConsent> = new Map();
  private contactHistory: Map<string, ContactRecord[]> = new Map();
  private auditLog: EthicalAuditEntry[] = [];

  constructor(config: Partial<EthicalGuardConfig> = {}) {
    this.config = { ...DEFAULT_ETHICAL_CONFIG, ...config };
  }

  /**
   * Check if a nudge passes ethical standards
   */
  checkNudge(nudge: Nudge, citizenDid: string): EthicalCheckResult {
    const violations: EthicalViolation[] = [];
    const warnings: EthicalWarning[] = [];
    const suggestions: EthicalSuggestion[] = [];

    // 1. Consent check
    const consentResult = this.checkConsent(citizenDid);
    if (!consentResult.hasConsent) {
      violations.push({
        type: consentResult.withdrawn ? 'consent_withdrawn' : 'consent_not_given',
        description: consentResult.withdrawn
          ? 'Citizen has withdrawn consent for proactive outreach'
          : 'Citizen has not given consent for proactive outreach',
        severity: 'critical',
        principle: 'consent_sovereignty',
        remediation: 'Obtain explicit consent before contacting',
      });
    }

    // 2. Contact frequency check
    const frequencyResult = this.checkContactFrequency(citizenDid);
    if (frequencyResult.exceeded) {
      violations.push({
        type: 'excessive_contact',
        description: `Citizen has received ${frequencyResult.count} nudges this week (max: ${this.config.maxNudgesPerWeek})`,
        severity: 'serious',
        principle: 'autonomy',
        remediation: 'Wait until next week or consolidate messages',
      });
    }

    // 3. Quiet hours check
    const quietResult = this.checkQuietHours(nudge);
    if (quietResult.inQuietHours && nudge.urgency !== 'critical') {
      violations.push({
        type: 'quiet_hours_violation',
        description: `Attempting to contact during quiet hours (${this.config.quietHours.start}:00 - ${this.config.quietHours.end}:00)`,
        severity: 'moderate',
        principle: 'autonomy',
        remediation: 'Schedule for delivery after quiet hours',
      });
      suggestions.push({
        field: 'scheduledFor',
        currentValue: nudge.createdAt,
        suggestedValue: this.getNextAllowedTime(),
        reason: 'Respect quiet hours',
      });
    }

    // 4. Dark pattern check
    const darkPatternResult = this.checkDarkPatterns(nudge.body);
    if (darkPatternResult.detected) {
      violations.push({
        type: 'dark_pattern',
        description: `Manipulative language detected: "${darkPatternResult.phrases.join('", "')}"`,
        severity: 'serious',
        principle: 'transparency',
        remediation: 'Remove manipulative language and use neutral, informative tone',
      });
    }

    // 5. Dignity check
    const dignityResult = this.checkDignity(nudge.body);
    if (dignityResult.issues.length > 0) {
      violations.push({
        type: 'dignity_violation',
        description: `Stigmatizing language detected: "${dignityResult.issues.join('", "')}"`,
        severity: 'serious',
        principle: 'dignity',
        remediation: 'Use person-first, dignified language',
      });
      for (const [phrase, replacement] of Object.entries(dignityResult.suggestions)) {
        suggestions.push({
          field: 'message',
          currentValue: phrase,
          suggestedValue: replacement,
          reason: 'Use dignified language',
        });
      }
    }

    // 6. Channel appropriateness check
    const channelResult = this.checkChannelAppropriateness(nudge);
    if (!channelResult.appropriate) {
      if (channelResult.sensitiveContent) {
        violations.push({
          type: 'sensitive_data_exposure',
          description: `Sensitive information should not be sent via ${nudge.preferredChannel}`,
          severity: 'critical',
          principle: 'privacy',
          remediation: `Use secure channel: ${this.config.secureChannels.join(' or ')}`,
        });
      } else {
        warnings.push({
          code: 'CHANNEL_SUBOPTIMAL',
          description: channelResult.reason || 'Channel may not be optimal for this content',
          recommendation: `Consider using ${channelResult.suggestedChannel}`,
        });
      }
    }

    // 7. Urgency inflation check
    const urgencyResult = this.checkUrgencyInflation(nudge);
    if (urgencyResult.inflated) {
      violations.push({
        type: 'urgency_inflation',
        description: 'Urgency level appears inflated for the actual deadline/situation',
        severity: 'moderate',
        principle: 'transparency',
        remediation: 'Use urgency that matches actual timeline',
      });
      suggestions.push({
        field: 'urgency',
        currentValue: nudge.urgency,
        suggestedValue: urgencyResult.suggestedUrgency,
        reason: 'Match urgency to actual timeline',
      });
    }

    // Build audit entry
    const auditEntry: EthicalAuditEntry = {
      timestamp: new Date(),
      action: `nudge:${nudge.type}`,
      citizenDidHash: this.hashDid(citizenDid),
      result: violations.length === 0 ? 'passed' : 'blocked',
      violationTypes: violations.map((v) => v.type),
      reviewer: this.needsHumanReview(violations) ? 'human_required' : 'automated',
    };

    this.auditLog.push(auditEntry);

    return {
      passed: violations.filter((v) => v.severity === 'critical').length === 0,
      violations,
      warnings,
      suggestions,
      auditEntry,
    };
  }

  /**
   * Check if a life event inference is ethically appropriate
   */
  checkEventInference(
    citizenDid: string,
    event: LifeEvent,
    dataUsed: string[]
  ): EthicalCheckResult {
    const violations: EthicalViolation[] = [];
    const warnings: EthicalWarning[] = [];
    const suggestions: EthicalSuggestion[] = [];

    // 1. Consent for data use
    const consent = this.consentRegistry.get(citizenDid);
    if (consent) {
      for (const dataSource of dataUsed) {
        if (!consent.allowedDataSources.includes(dataSource)) {
          violations.push({
            type: 'consent_not_given',
            description: `Citizen has not consented to use of ${dataSource} for inference`,
            severity: 'critical',
            principle: 'consent_sovereignty',
            remediation: `Remove ${dataSource} from inference or obtain consent`,
          });
        }
      }
    }

    // 2. Surveillance overreach check
    if (dataUsed.length > 3) {
      warnings.push({
        code: 'INFERENCE_COMPLEXITY',
        description: `Event inferred from ${dataUsed.length} data sources`,
        recommendation: 'Consider if this level of inference is necessary',
      });
    }

    // 3. Sensitive event handling
    if (this.config.sensitiveEventTypes.includes(event.type)) {
      warnings.push({
        code: 'SENSITIVE_EVENT',
        description: `${event.type} is a sensitive life event requiring extra care`,
        recommendation: 'Ensure messaging is compassionate and non-intrusive',
      });

      // Lower confidence threshold for sensitive events = more caution
      if (event.confidence < 0.9) {
        violations.push({
          type: 'surveillance_overreach',
          description: `Low confidence (${event.confidence}) inference of sensitive event`,
          severity: 'serious',
          principle: 'privacy',
          remediation: 'Do not act on low-confidence sensitive event inferences',
        });
      }
    }

    const auditEntry: EthicalAuditEntry = {
      timestamp: new Date(),
      action: `inference:${event.type}`,
      citizenDidHash: this.hashDid(citizenDid),
      result: violations.length === 0 ? 'passed' : 'blocked',
      violationTypes: violations.map((v) => v.type),
      reviewer: 'automated',
    };

    this.auditLog.push(auditEntry);

    return {
      passed: violations.filter((v) => v.severity === 'critical').length === 0,
      violations,
      warnings,
      suggestions,
      auditEntry,
    };
  }

  /**
   * Check for potential benefit cliff trap
   *
   * A benefit cliff occurs when a small increase in income causes
   * a large loss of benefits. We should warn citizens, not push them off.
   */
  checkBenefitCliff(
    predictions: EligibilityPrediction[],
    _currentIncome: number,
    suggestedAction?: string
  ): { atRisk: boolean; warning?: string } {
    if (!this.config.enableBenefitCliffDetection) {
      return { atRisk: false };
    }

    // Find programs citizen is currently eligible for
    const eligible = predictions.filter((p) => p.prediction === 'likely' || p.prediction === 'possible');

    // Check if suggested action might affect income
    const incomeAffectingActions = ['job_gain', 'promotion', 'raise', 'hours_increase'];
    if (suggestedAction && incomeAffectingActions.some((a) => suggestedAction.includes(a))) {
      // Calculate potential cliff
      const cliffPrograms = eligible.filter((p) => {
        // Programs with hard income cutoffs create cliffs
        return p.program === 'medicaid' || p.program === 'snap' || p.program === 'housing';
      });

      if (cliffPrograms.length > 0) {
        return {
          atRisk: true,
          warning: `Citizen may be at risk of benefit cliff. Programs at risk: ${cliffPrograms.map((p) => p.program).join(', ')}. Ensure citizen understands full impact before encouraging action.`,
        };
      }
    }

    return { atRisk: false };
  }

  /**
   * Register citizen consent
   */
  registerConsent(citizenDid: string, consent: Partial<CitizenConsent>): void {
    this.consentRegistry.set(citizenDid, {
      citizenDid,
      givenAt: new Date(),
      channels: consent.channels || ['dashboard'],
      eventTypes: consent.eventTypes || [],
      allowedDataSources: consent.allowedDataSources || ['citizen_report'],
      withdrawn: false,
    });
  }

  /**
   * Withdraw citizen consent
   */
  withdrawConsent(citizenDid: string): void {
    const existing = this.consentRegistry.get(citizenDid);
    if (existing) {
      existing.withdrawn = true;
      existing.withdrawnAt = new Date();
    } else {
      this.consentRegistry.set(citizenDid, {
        citizenDid,
        givenAt: new Date(),
        channels: [],
        eventTypes: [],
        allowedDataSources: [],
        withdrawn: true,
        withdrawnAt: new Date(),
      });
    }
  }

  /**
   * Record a contact (for frequency tracking)
   */
  recordContact(citizenDid: string, nudgeId: string, channel: NudgeChannel): void {
    const history = this.contactHistory.get(citizenDid) || [];
    history.push({
      nudgeId,
      channel,
      timestamp: new Date(),
    });
    this.contactHistory.set(citizenDid, history);
  }

  /**
   * Get audit log
   */
  getAuditLog(options: { citizenDid?: string; since?: Date } = {}): EthicalAuditEntry[] {
    let entries = [...this.auditLog];

    if (options.citizenDid) {
      const hash = this.hashDid(options.citizenDid);
      entries = entries.filter((e) => e.citizenDidHash === hash);
    }

    if (options.since) {
      entries = entries.filter((e) => e.timestamp >= options.since!);
    }

    return entries;
  }

  /**
   * Sanitize message for dignity
   */
  sanitizeForDignity(message: string): string {
    let sanitized = message;
    for (const [phrase, replacement] of Object.entries(DIGNIFIED_ALTERNATIVES)) {
      const regex = new RegExp(phrase, 'gi');
      sanitized = sanitized.replace(regex, replacement);
    }
    return sanitized;
  }

  /**
   * Remove dark patterns from message
   */
  removeDarkPatterns(message: string): string {
    let cleaned = message;
    for (const phrase of DARK_PATTERN_PHRASES) {
      const regex = new RegExp(phrase, 'gi');
      cleaned = cleaned.replace(regex, '');
    }
    // Clean up extra whitespace
    return cleaned.replace(/\s+/g, ' ').trim();
  }

  // Private methods

  private checkConsent(citizenDid: string): { hasConsent: boolean; withdrawn: boolean } {
    if (!this.config.requireExplicitConsent) {
      return { hasConsent: true, withdrawn: false };
    }

    const consent = this.consentRegistry.get(citizenDid);
    if (!consent) {
      return { hasConsent: false, withdrawn: false };
    }

    if (consent.withdrawn) {
      return { hasConsent: false, withdrawn: true };
    }

    return { hasConsent: true, withdrawn: false };
  }

  private checkContactFrequency(citizenDid: string): { exceeded: boolean; count: number } {
    const history = this.contactHistory.get(citizenDid) || [];
    const oneWeekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
    const recentContacts = history.filter((c) => c.timestamp >= oneWeekAgo);

    return {
      exceeded: recentContacts.length >= this.config.maxNudgesPerWeek,
      count: recentContacts.length,
    };
  }

  private checkQuietHours(_nudge: Nudge): { inQuietHours: boolean } {
    const now = new Date();
    const hour = now.getHours();
    const { start, end } = this.config.quietHours;

    // Handle overnight quiet hours (e.g., 21:00 to 08:00)
    const inQuietHours = start > end
      ? hour >= start || hour < end
      : hour >= start && hour < end;

    return { inQuietHours };
  }

  private getNextAllowedTime(): Date {
    const now = new Date();
    const hour = now.getHours();
    const { end } = this.config.quietHours;

    if (hour < end) {
      // Still in morning quiet hours, schedule for end of quiet hours today
      const next = new Date(now);
      next.setHours(end, 0, 0, 0);
      return next;
    } else {
      // Schedule for tomorrow morning after quiet hours
      const next = new Date(now);
      next.setDate(next.getDate() + 1);
      next.setHours(end, 0, 0, 0);
      return next;
    }
  }

  private checkDarkPatterns(message: string): { detected: boolean; phrases: string[] } {
    const lower = message.toLowerCase();
    const found = DARK_PATTERN_PHRASES.filter((phrase) => lower.includes(phrase));
    return { detected: found.length > 0, phrases: found };
  }

  private checkDignity(message: string): {
    issues: string[];
    suggestions: Record<string, string>;
  } {
    const lower = message.toLowerCase();
    const issues: string[] = [];
    const suggestions: Record<string, string> = {};

    for (const phrase of STIGMATIZING_PHRASES) {
      if (lower.includes(phrase)) {
        issues.push(phrase);
        if (DIGNIFIED_ALTERNATIVES[phrase]) {
          suggestions[phrase] = DIGNIFIED_ALTERNATIVES[phrase];
        }
      }
    }

    return { issues, suggestions };
  }

  private checkChannelAppropriateness(nudge: Nudge): {
    appropriate: boolean;
    sensitiveContent: boolean;
    reason?: string;
    suggestedChannel?: NudgeChannel;
  } {
    // Check for sensitive content indicators
    const sensitiveKeywords = ['income', 'disability', 'health', 'death', 'divorce', 'prison'];
    const hasSensitiveContent = sensitiveKeywords.some((kw) =>
      nudge.body.toLowerCase().includes(kw)
    );

    if (hasSensitiveContent && !this.config.secureChannels.includes(nudge.preferredChannel)) {
      return {
        appropriate: false,
        sensitiveContent: true,
        reason: 'Sensitive information should use secure channels',
        suggestedChannel: 'dashboard',
      };
    }

    return { appropriate: true, sensitiveContent: false };
  }

  private checkUrgencyInflation(nudge: Nudge): {
    inflated: boolean;
    suggestedUrgency?: NudgeUrgency;
  } {
    // If marked urgent but no deadline within 7 days
    if (nudge.urgency === 'high') {
      // Check message for actual deadline
      const hasImmediateDeadline = /within \d+ days?|by (today|tomorrow|this week)/i.test(
        nudge.body
      );
      if (!hasImmediateDeadline) {
        return { inflated: true, suggestedUrgency: 'medium' };
      }
    }

    // If marked critical but not a true emergency
    if (nudge.urgency === 'critical') {
      const emergencyIndicators = ['eviction', 'utility shutoff', 'medical emergency', 'crisis'];
      const isRealEmergency = emergencyIndicators.some((ind) =>
        nudge.body.toLowerCase().includes(ind)
      );
      if (!isRealEmergency) {
        return { inflated: true, suggestedUrgency: 'high' };
      }
    }

    return { inflated: false };
  }

  private needsHumanReview(violations: EthicalViolation[]): boolean {
    return violations.some((v) => this.config.requireHumanReview.includes(v.type));
  }

  private hashDid(did: string): string {
    // Simple hash for audit purposes (not cryptographically secure, just for logs)
    let hash = 0;
    for (let i = 0; i < did.length; i++) {
      const char = did.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return `did:hash:${Math.abs(hash).toString(16)}`;
  }
}

/**
 * Citizen consent record
 */
export interface CitizenConsent {
  citizenDid: string;
  givenAt: Date;
  channels: NudgeChannel[];
  eventTypes: LifeEventType[];
  allowedDataSources: string[];
  withdrawn: boolean;
  withdrawnAt?: Date;
}

/**
 * Contact history record
 */
interface ContactRecord {
  nudgeId: string;
  channel: NudgeChannel;
  timestamp: Date;
}

/**
 * Create an ethical guards instance
 */
export function createEthicalGuards(
  config: Partial<EthicalGuardConfig> = {}
): EthicalGuards {
  return new EthicalGuards(config);
}

/**
 * Quick ethical check for a message
 */
export function quickEthicalCheck(message: string): {
  passed: boolean;
  issues: string[];
} {
  const issues: string[] = [];

  // Check dark patterns
  const lower = message.toLowerCase();
  for (const phrase of DARK_PATTERN_PHRASES) {
    if (lower.includes(phrase)) {
      issues.push(`Dark pattern: "${phrase}"`);
    }
  }

  // Check dignity
  for (const phrase of STIGMATIZING_PHRASES) {
    if (lower.includes(phrase)) {
      issues.push(`Stigmatizing language: "${phrase}"`);
    }
  }

  return {
    passed: issues.length === 0,
    issues,
  };
}
