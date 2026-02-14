/**
 * Life Event Detection for Proactive Outreach
 *
 * Detects life events that may trigger eligibility for benefits or services.
 * Events are detected from citizen data changes, not surveillance.
 *
 * Philosophy: Government should find citizens, not vice versa.
 *
 * @module outreach/life-events
 */

/**
 * Types of life events that may trigger proactive outreach
 */
export type LifeEventType =
  | 'birth'                  // New child in household
  | 'death'                  // Loss of household member
  | 'marriage'               // Marriage or domestic partnership
  | 'divorce'                // Divorce or separation
  | 'address_change'         // Moved to new address
  | 'income_change'          // Significant income change
  | 'job_loss'               // Lost employment
  | 'job_gain'               // Gained employment
  | 'age_milestone'          // Turned 18, 62, 65, etc.
  | 'disability_onset'       // New disability status
  | 'veteran_discharge'      // Discharged from military
  | 'immigration_status'     // Immigration status change
  | 'education_completion'   // Completed degree/certification
  | 'incarceration_release'  // Released from incarceration
  | 'housing_instability'    // At risk of homelessness
  | 'health_crisis'          // Major health event
  | 'natural_disaster'       // Affected by disaster
  | 'benefit_expiration';    // Existing benefit about to expire

/**
 * Life event with context
 */
export interface LifeEvent {
  /** Event type */
  type: LifeEventType;
  /** When the event was detected */
  detectedAt: Date;
  /** When the event actually occurred (if known) */
  occurredAt?: Date;
  /** Confidence in detection (0-1) */
  confidence: number;
  /** Source of detection */
  source: LifeEventSource;
  /** Additional context */
  context: Record<string, unknown>;
  /** Citizen DID */
  citizenDid: string;
  /** Whether citizen has been notified */
  citizenNotified: boolean;
  /** Potential programs this event may qualify for */
  potentialPrograms: string[];
}

/**
 * Source of life event detection
 */
export type LifeEventSource =
  | 'citizen_report'      // Citizen reported the event
  | 'application_data'    // Detected from application submission
  | 'address_update'      // Address change in system
  | 'benefits_data'       // Cross-referenced from benefits data
  | 'vital_records'       // Birth/death records (with consent)
  | 'employment_data'     // Employment verification (with consent)
  | 'scheduled'           // Scheduled event (age milestone)
  | 'external_api';       // External data source (with consent)

/**
 * Configuration for life event detection
 */
export interface LifeEventDetectorConfig {
  /** Enable detection */
  enabled: boolean;
  /** Event types to detect */
  enabledEventTypes: LifeEventType[];
  /** Minimum confidence threshold */
  minConfidence: number;
  /** How often to check for events (ms) */
  checkIntervalMs: number;
  /** Require explicit consent for detection */
  requireConsent: boolean;
  /** Data sources to use */
  enabledSources: LifeEventSource[];
  /** Notification delay after detection (ms) */
  notificationDelayMs: number;
}

/**
 * Default configuration
 */
export const DEFAULT_LIFE_EVENT_CONFIG: LifeEventDetectorConfig = {
  enabled: true,
  enabledEventTypes: [
    'birth',
    'address_change',
    'income_change',
    'job_loss',
    'age_milestone',
    'benefit_expiration',
  ],
  minConfidence: 0.7,
  checkIntervalMs: 24 * 60 * 60 * 1000, // Daily
  requireConsent: true,
  enabledSources: ['citizen_report', 'application_data', 'address_update', 'scheduled'],
  notificationDelayMs: 4 * 60 * 60 * 1000, // 4 hours delay
};

/**
 * Event type to potential programs mapping
 */
const EVENT_PROGRAM_MAP: Record<LifeEventType, string[]> = {
  birth: ['medicaid', 'snap', 'wic', 'tanf', 'childcare'],
  death: ['survivor_benefits', 'grief_counseling', 'estate_assistance'],
  marriage: ['tax_benefits', 'insurance_options'],
  divorce: ['tanf', 'snap', 'housing', 'legal_aid'],
  address_change: ['voting_registration', 'school_enrollment', 'utilities'],
  income_change: ['snap', 'medicaid', 'housing', 'liheap', 'childcare'],
  job_loss: ['unemployment', 'snap', 'medicaid', 'cobra', 'job_training'],
  job_gain: ['tax_credits', 'childcare', 'transportation'],
  age_milestone: ['medicare', 'social_security', 'senior_services'],
  disability_onset: ['disability_benefits', 'medicaid', 'vocational_rehab'],
  veteran_discharge: ['va_benefits', 'gi_bill', 'veteran_housing'],
  immigration_status: ['citizenship_services', 'work_authorization'],
  education_completion: ['student_loan_repayment', 'job_placement'],
  incarceration_release: ['reentry_services', 'housing', 'job_training'],
  housing_instability: ['emergency_housing', 'rental_assistance', 'shelters'],
  health_crisis: ['medicaid', 'disability', 'hospital_charity'],
  natural_disaster: ['fema_assistance', 'emergency_housing', 'crisis_funds'],
  benefit_expiration: ['recertification', 'alternative_programs'],
};

/**
 * Life Event Detector
 *
 * Detects life events from citizen data and triggers proactive outreach.
 */
export class LifeEventDetector {
  private config: LifeEventDetectorConfig;
  private consentedCitizens: Set<string> = new Set();
  private detectedEvents: Map<string, LifeEvent[]> = new Map();
  private checkInterval: ReturnType<typeof setInterval> | null = null;

  constructor(config: Partial<LifeEventDetectorConfig> = {}) {
    this.config = { ...DEFAULT_LIFE_EVENT_CONFIG, ...config };
  }

  /**
   * Start the detector
   */
  start(): void {
    if (!this.config.enabled) return;

    this.checkInterval = setInterval(() => {
      this.runDetection();
    }, this.config.checkIntervalMs);

    console.log('[LifeEventDetector] Started');
  }

  /**
   * Stop the detector
   */
  stop(): void {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
    console.log('[LifeEventDetector] Stopped');
  }

  /**
   * Register citizen consent for proactive outreach
   */
  registerConsent(citizenDid: string): void {
    this.consentedCitizens.add(citizenDid);
  }

  /**
   * Withdraw citizen consent
   */
  withdrawConsent(citizenDid: string): void {
    this.consentedCitizens.delete(citizenDid);
    this.detectedEvents.delete(citizenDid);
  }

  /**
   * Check if citizen has consented
   */
  hasConsent(citizenDid: string): boolean {
    return !this.config.requireConsent || this.consentedCitizens.has(citizenDid);
  }

  /**
   * Detect life event from data change
   */
  detectFromDataChange(
    citizenDid: string,
    changeType: string,
    oldValue: unknown,
    newValue: unknown,
    source: LifeEventSource
  ): LifeEvent | null {
    if (!this.hasConsent(citizenDid)) return null;
    if (!this.config.enabledSources.includes(source)) return null;

    const eventType = this.inferEventType(changeType, oldValue, newValue);
    if (!eventType) return null;
    if (!this.config.enabledEventTypes.includes(eventType)) return null;

    const confidence = this.calculateConfidence(eventType, changeType, oldValue, newValue);
    if (confidence < this.config.minConfidence) return null;

    const event: LifeEvent = {
      type: eventType,
      detectedAt: new Date(),
      confidence,
      source,
      context: { changeType, oldValue, newValue },
      citizenDid,
      citizenNotified: false,
      potentialPrograms: EVENT_PROGRAM_MAP[eventType] || [],
    };

    this.recordEvent(event);
    return event;
  }

  /**
   * Detect scheduled events (age milestones, benefit expirations)
   */
  detectScheduledEvents(citizenDid: string, profile: CitizenProfile): LifeEvent[] {
    if (!this.hasConsent(citizenDid)) return [];

    const events: LifeEvent[] = [];

    // Age milestone detection
    if (profile.dateOfBirth) {
      const age = this.calculateAge(profile.dateOfBirth);
      const milestoneAges = [18, 21, 26, 55, 60, 62, 65, 70];

      for (const milestone of milestoneAges) {
        const daysUntil = this.daysUntilAge(profile.dateOfBirth, milestone);
        if (daysUntil >= 0 && daysUntil <= 30) {
          events.push({
            type: 'age_milestone',
            detectedAt: new Date(),
            occurredAt: this.getAgeDate(profile.dateOfBirth, milestone),
            confidence: 1.0,
            source: 'scheduled',
            context: { milestone, currentAge: age, daysUntil },
            citizenDid,
            citizenNotified: false,
            potentialPrograms: this.getProgramsForAgeMilestone(milestone),
          });
        }
      }
    }

    // Benefit expiration detection
    if (profile.activeBenefits) {
      for (const benefit of profile.activeBenefits) {
        if (benefit.expiresAt) {
          const daysUntil = this.daysUntilDate(benefit.expiresAt);
          if (daysUntil >= 0 && daysUntil <= 60) {
            events.push({
              type: 'benefit_expiration',
              detectedAt: new Date(),
              occurredAt: benefit.expiresAt,
              confidence: 1.0,
              source: 'scheduled',
              context: { benefitType: benefit.type, daysUntil },
              citizenDid,
              citizenNotified: false,
              potentialPrograms: ['recertification', benefit.type],
            });
          }
        }
      }
    }

    events.forEach((e) => this.recordEvent(e));
    return events;
  }

  /**
   * Get pending events for a citizen
   */
  getPendingEvents(citizenDid: string): LifeEvent[] {
    return (this.detectedEvents.get(citizenDid) || []).filter((e) => !e.citizenNotified);
  }

  /**
   * Mark event as notified
   */
  markNotified(citizenDid: string, eventType: LifeEventType): void {
    const events = this.detectedEvents.get(citizenDid);
    if (events) {
      const event = events.find((e) => e.type === eventType && !e.citizenNotified);
      if (event) {
        event.citizenNotified = true;
      }
    }
  }

  /**
   * Get statistics
   */
  getStats(): LifeEventStats {
    let totalEvents = 0;
    let notifiedEvents = 0;
    const eventsByType: Record<string, number> = {};

    this.detectedEvents.forEach((events) => {
      totalEvents += events.length;
      events.forEach((e) => {
        if (e.citizenNotified) notifiedEvents++;
        eventsByType[e.type] = (eventsByType[e.type] || 0) + 1;
      });
    });

    return {
      totalCitizensConsented: this.consentedCitizens.size,
      totalEventsDetected: totalEvents,
      totalEventsNotified: notifiedEvents,
      eventsByType,
    };
  }

  // Private methods

  private runDetection(): void {
    // In production, this would query citizen profiles for scheduled events
    console.log('[LifeEventDetector] Running scheduled detection');
  }

  private recordEvent(event: LifeEvent): void {
    const events = this.detectedEvents.get(event.citizenDid) || [];
    events.push(event);
    this.detectedEvents.set(event.citizenDid, events);
  }

  private inferEventType(
    changeType: string,
    _oldValue: unknown,
    _newValue: unknown
  ): LifeEventType | null {
    const inferenceMap: Record<string, LifeEventType> = {
      address: 'address_change',
      household_size_increase: 'birth',
      household_size_decrease: 'death',
      income: 'income_change',
      employment_status_lost: 'job_loss',
      employment_status_gained: 'job_gain',
      marital_status_married: 'marriage',
      marital_status_divorced: 'divorce',
      disability_status: 'disability_onset',
      veteran_status: 'veteran_discharge',
    };

    return inferenceMap[changeType] || null;
  }

  private calculateConfidence(
    eventType: LifeEventType,
    _changeType: string,
    _oldValue: unknown,
    _newValue: unknown
  ): number {
    // Simple confidence calculation - in production would be more sophisticated
    const baseConfidence: Record<LifeEventType, number> = {
      birth: 0.9,
      death: 0.9,
      marriage: 0.95,
      divorce: 0.95,
      address_change: 1.0,
      income_change: 0.85,
      job_loss: 0.9,
      job_gain: 0.9,
      age_milestone: 1.0,
      disability_onset: 0.8,
      veteran_discharge: 0.95,
      immigration_status: 0.9,
      education_completion: 0.95,
      incarceration_release: 0.9,
      housing_instability: 0.7,
      health_crisis: 0.75,
      natural_disaster: 0.9,
      benefit_expiration: 1.0,
    };

    return baseConfidence[eventType] || 0.5;
  }

  private calculateAge(dateOfBirth: Date): number {
    const today = new Date();
    let age = today.getFullYear() - dateOfBirth.getFullYear();
    const monthDiff = today.getMonth() - dateOfBirth.getMonth();
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < dateOfBirth.getDate())) {
      age--;
    }
    return age;
  }

  private daysUntilAge(dateOfBirth: Date, targetAge: number): number {
    const targetDate = new Date(dateOfBirth);
    targetDate.setFullYear(dateOfBirth.getFullYear() + targetAge);
    return this.daysUntilDate(targetDate);
  }

  private daysUntilDate(date: Date): number {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const target = new Date(date);
    target.setHours(0, 0, 0, 0);
    return Math.ceil((target.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
  }

  private getAgeDate(dateOfBirth: Date, age: number): Date {
    const date = new Date(dateOfBirth);
    date.setFullYear(dateOfBirth.getFullYear() + age);
    return date;
  }

  private getProgramsForAgeMilestone(age: number): string[] {
    const agePrograms: Record<number, string[]> = {
      18: ['voter_registration', 'financial_literacy'],
      21: ['adult_services'],
      26: ['insurance_marketplace'],
      55: ['senior_discounts', 'retirement_planning'],
      60: ['senior_services'],
      62: ['early_social_security', 'senior_housing'],
      65: ['medicare', 'social_security', 'senior_services'],
      70: ['required_minimum_distribution'],
    };
    return agePrograms[age] || ['age_related_services'];
  }
}

/**
 * Citizen profile for event detection
 */
export interface CitizenProfile {
  did: string;
  dateOfBirth?: Date;
  activeBenefits?: Array<{
    type: string;
    expiresAt?: Date;
  }>;
}

/**
 * Life event statistics
 */
export interface LifeEventStats {
  totalCitizensConsented: number;
  totalEventsDetected: number;
  totalEventsNotified: number;
  eventsByType: Record<string, number>;
}

/**
 * Create a life event detector instance
 */
export function createLifeEventDetector(
  config: Partial<LifeEventDetectorConfig> = {}
): LifeEventDetector {
  return new LifeEventDetector(config);
}
