/**
 * Proactive Outreach Engine
 *
 * Government that finds citizens, not vice versa.
 *
 * This module enables proactive government outreach by:
 * 1. Detecting life events that may trigger benefit eligibility
 * 2. Predicting eligibility based on citizen profiles
 * 3. Delivering nudges through multiple channels
 * 4. Enforcing ethical safeguards at every step
 *
 * Core Philosophy:
 * - Citizens should not have to navigate bureaucracy to get help
 * - Government has an obligation to inform citizens of available benefits
 * - Proactive outreach must respect privacy, autonomy, and dignity
 * - Transparency over persuasion - inform, don't manipulate
 *
 * @module outreach
 */

// Life Event Detection
export {
  LifeEventDetector,
  createLifeEventDetector,
  type LifeEvent,
  type LifeEventType,
  type LifeEventSource,
  type LifeEventDetectorConfig,
  type CitizenProfile,
  type LifeEventStats,
  DEFAULT_LIFE_EVENT_CONFIG,
} from './life-events.js';

// Eligibility Prediction
export {
  EligibilityPredictor,
  createEligibilityPredictor,
  type EligibilityPrediction,
  type BenefitProgram,
  type CitizenEligibilityProfile,
  type EligibilityCriteria,
  PROGRAM_CRITERIA,
} from './eligibility.js';

// Nudge Engine
export {
  NudgeEngine,
  createNudgeEngine,
  type Nudge,
  type NudgeType,
  type NudgeChannel,
  type NudgeUrgency,
  type NudgeEngineConfig,
  type NudgeEngineStats,
  type NudgeResponse,
  type CitizenNotificationPrefs,
  type ChannelDeliveryFunctions,
  DEFAULT_NUDGE_ENGINE_CONFIG,
} from './nudge-engine.js';

// Ethical Guards
export {
  EthicalGuards,
  createEthicalGuards,
  quickEthicalCheck,
  type EthicalCheckResult,
  type EthicalViolation,
  type EthicalViolationType,
  type EthicalWarning,
  type EthicalSuggestion,
  type EthicalAuditEntry,
  type EthicalPrinciple,
  type EthicalGuardConfig,
  type CitizenConsent,
  DEFAULT_ETHICAL_CONFIG,
} from './ethical-guards.js';

/**
 * Integrated Outreach System
 *
 * Combines all outreach components into a single coordinated system.
 */
export interface OutreachSystemConfig {
  /** Life event detection configuration */
  lifeEvents?: Partial<import('./life-events.js').LifeEventDetectorConfig>;
  /** Eligibility prediction configuration */
  eligibility?: Record<string, unknown>;
  /** Nudge engine configuration */
  nudge?: Partial<import('./nudge-engine.js').NudgeEngineConfig>;
  /** Ethical guards configuration */
  ethical?: Partial<import('./ethical-guards.js').EthicalGuardConfig>;
}

/**
 * Integrated Outreach System
 *
 * Provides a unified interface for proactive citizen outreach.
 */
export class OutreachSystem {
  public readonly lifeEvents: import('./life-events.js').LifeEventDetector;
  public readonly eligibility: import('./eligibility.js').EligibilityPredictor;
  public readonly nudge: import('./nudge-engine.js').NudgeEngine;
  public readonly ethical: import('./ethical-guards.js').EthicalGuards;

  private running = false;

  constructor(config: OutreachSystemConfig = {}) {
    // Lazy imports to avoid circular dependencies
    const { LifeEventDetector } = require('./life-events.js');
    const { EligibilityPredictor } = require('./eligibility.js');
    const { NudgeEngine } = require('./nudge-engine.js');
    const { EthicalGuards } = require('./ethical-guards.js');

    this.lifeEvents = new LifeEventDetector(config.lifeEvents);
    this.eligibility = new EligibilityPredictor(config.eligibility);
    this.nudge = new NudgeEngine(config.nudge);
    this.ethical = new EthicalGuards(config.ethical);
  }

  /**
   * Start the outreach system
   */
  start(): void {
    if (this.running) return;
    this.lifeEvents.start();
    this.running = true;
    console.log('[OutreachSystem] Started');
  }

  /**
   * Stop the outreach system
   */
  stop(): void {
    if (!this.running) return;
    this.lifeEvents.stop();
    this.running = false;
    console.log('[OutreachSystem] Stopped');
  }

  /**
   * Register citizen for proactive outreach
   */
  registerCitizen(
    citizenDid: string,
    consent: Partial<import('./ethical-guards.js').CitizenConsent>,
    prefs?: Partial<import('./nudge-engine.js').CitizenNotificationPrefs>
  ): void {
    // Register consent with ethical guards
    this.ethical.registerConsent(citizenDid, consent);

    // Register consent with life event detector
    this.lifeEvents.registerConsent(citizenDid);

    // Set notification preferences if provided
    if (prefs) {
      this.nudge.setCitizenPrefs(citizenDid, prefs);
    }
  }

  /**
   * Withdraw citizen from proactive outreach
   */
  withdrawCitizen(citizenDid: string): void {
    this.ethical.withdrawConsent(citizenDid);
    this.lifeEvents.withdrawConsent(citizenDid);
  }

  /**
   * Process a life event for a citizen
   *
   * This is the main entry point for proactive outreach.
   * 1. Validates the event ethically
   * 2. Predicts eligibility based on the event
   * 3. Creates and delivers nudges if appropriate
   */
  async processLifeEvent(
    citizenDid: string,
    event: import('./life-events.js').LifeEvent,
    profile: import('./eligibility.js').CitizenEligibilityProfile
  ): Promise<ProcessingResult> {
    const result: ProcessingResult = {
      eventProcessed: false,
      predictionsGenerated: 0,
      nudgesCreated: 0,
      nudgesSent: 0,
      ethicalViolations: [],
      errors: [],
    };

    try {
      // 1. Ethical check on event inference
      const eventCheck = this.ethical.checkEventInference(
        citizenDid,
        event,
        [event.source]
      );

      if (!eventCheck.passed) {
        result.ethicalViolations.push(...eventCheck.violations);
        console.log(`[OutreachSystem] Event blocked by ethical guards: ${eventCheck.violations.map((v) => v.type).join(', ')}`);
        return result;
      }

      result.eventProcessed = true;

      // 2. Predict eligibility
      const programs = this.eligibility.getProgramsForEvent(event.type);
      const predictions = programs.map((program) => this.eligibility.predictProgram(profile, program));
      result.predictionsGenerated = predictions.length;

      if (predictions.length === 0) {
        console.log(`[OutreachSystem] No eligibility predictions for event ${event.type}`);
        return result;
      }

      // 3. Filter to eligible programs
      const eligible = predictions.filter((p: import('./eligibility.js').EligibilityPrediction) => p.prediction === 'likely' || p.prediction === 'possible');
      if (eligible.length === 0) {
        console.log(`[OutreachSystem] Citizen not eligible for any programs`);
        return result;
      }

      // 4. Create nudge
      const nudge = this.nudge.createEventNudge(citizenDid, event, eligible);
      if (!nudge) {
        console.log(`[OutreachSystem] No nudge created (citizen prefs may block)`);
        return result;
      }

      result.nudgesCreated = 1;

      // 5. Ethical check on nudge
      const nudgeCheck = this.ethical.checkNudge(nudge, citizenDid);
      if (!nudgeCheck.passed) {
        result.ethicalViolations.push(...nudgeCheck.violations);
        console.log(`[OutreachSystem] Nudge blocked by ethical guards: ${nudgeCheck.violations.map((v) => v.type).join(', ')}`);
        return result;
      }

      // 6. Queue nudge for delivery (nudge was already added to pending by createEventNudge)
      result.nudgesSent = 1;
      this.ethical.recordContact(citizenDid, nudge.id, nudge.preferredChannel);

      return result;
    } catch (error) {
      result.errors.push(error instanceof Error ? error.message : String(error));
      return result;
    }
  }

  /**
   * Run scheduled checks for a citizen
   */
  async runScheduledChecks(
    citizenDid: string,
    profile: import('./eligibility.js').CitizenEligibilityProfile & import('./life-events.js').CitizenProfile
  ): Promise<ProcessingResult[]> {
    const results: ProcessingResult[] = [];

    // Detect scheduled events (age milestones, benefit expirations)
    const events = this.lifeEvents.detectScheduledEvents(citizenDid, profile);

    for (const event of events) {
      const result = await this.processLifeEvent(citizenDid, event, profile);
      results.push(result);
    }

    return results;
  }

  /**
   * Get system statistics
   */
  getStats(): OutreachStats {
    return {
      lifeEvents: this.lifeEvents.getStats(),
      nudges: this.nudge.getStats(),
      ethicalAudit: {
        totalChecks: this.ethical.getAuditLog().length,
        violationsBlocked: this.ethical.getAuditLog().filter((e) => e.result === 'blocked').length,
      },
    };
  }
}

/**
 * Result of processing a life event
 */
export interface ProcessingResult {
  /** Whether the event was processed (passed ethical checks) */
  eventProcessed: boolean;
  /** Number of eligibility predictions generated */
  predictionsGenerated: number;
  /** Number of nudges created */
  nudgesCreated: number;
  /** Number of nudges successfully sent */
  nudgesSent: number;
  /** Ethical violations that blocked actions */
  ethicalViolations: import('./ethical-guards.js').EthicalViolation[];
  /** Errors encountered */
  errors: string[];
}

/**
 * System-wide statistics
 */
export interface OutreachStats {
  lifeEvents: import('./life-events.js').LifeEventStats;
  nudges: import('./nudge-engine.js').NudgeEngineStats;
  ethicalAudit: {
    totalChecks: number;
    violationsBlocked: number;
  };
}

/**
 * Create an integrated outreach system
 */
export function createOutreachSystem(config: OutreachSystemConfig = {}): OutreachSystem {
  return new OutreachSystem(config);
}

/**
 * Preset configurations for common scenarios
 */
export const OUTREACH_PRESETS = {
  /**
   * Conservative preset - maximum ethical safeguards
   */
  conservative: {
    lifeEvents: {
      minConfidence: 0.9,
      enabledEventTypes: ['age_milestone', 'benefit_expiration'] as import('./life-events.js').LifeEventType[],
      requireConsent: true,
    },
    eligibility: {
      minConfidence: 0.8,
    },
    nudge: {
      batchSize: 1,
    },
    ethical: {
      requireExplicitConsent: true,
      maxNudgesPerWeek: 1,
      enableBenefitCliffDetection: true,
    },
  } satisfies OutreachSystemConfig,

  /**
   * Standard preset - balanced approach
   */
  standard: {
    lifeEvents: {
      minConfidence: 0.7,
      requireConsent: true,
    },
    eligibility: {
      minConfidence: 0.6,
    },
    nudge: {
    },
    ethical: {
      requireExplicitConsent: true,
    },
  } satisfies OutreachSystemConfig,

  /**
   * Proactive preset - more aggressive outreach (still ethical)
   */
  proactive: {
    lifeEvents: {
      minConfidence: 0.6,
      enabledEventTypes: [
        'birth',
        'address_change',
        'income_change',
        'job_loss',
        'job_gain',
        'age_milestone',
        'disability_onset',
        'benefit_expiration',
      ] as import('./life-events.js').LifeEventType[],
      requireConsent: true,
    },
    eligibility: {
      minConfidence: 0.5, // Lower barrier = more outreach
    },
    nudge: {
    },
    ethical: {
      requireExplicitConsent: true, // Still require consent!
      maxNudgesPerWeek: 5,
      enableBenefitCliffDetection: true,
    },
  } satisfies OutreachSystemConfig,

  /**
   * Emergency/Crisis preset - for disaster response
   */
  emergency: {
    lifeEvents: {
      minConfidence: 0.5,
      enabledEventTypes: [
        'natural_disaster',
        'health_crisis',
        'housing_instability',
      ] as import('./life-events.js').LifeEventType[],
      requireConsent: false, // Emergency exception
    },
    eligibility: {
      minConfidence: 0.4, // Maximum reach
    },
    nudge: {
    },
    ethical: {
      requireExplicitConsent: false, // Emergency exception
      maxNudgesPerWeek: 10,
      quietHours: { start: 23, end: 6 }, // Shorter quiet hours in emergency
    },
  } satisfies OutreachSystemConfig,
};
