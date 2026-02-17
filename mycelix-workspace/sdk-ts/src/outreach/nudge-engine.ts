/**
 * Nudge Engine for Proactive Outreach
 *
 * Delivers proactive suggestions to citizens through multiple channels.
 * Respects citizen preferences and ethical boundaries.
 *
 * Philosophy: Helpful, not intrusive. Inform, don't manipulate.
 *
 * @module outreach/nudge-engine
 */

import type { EligibilityPrediction, BenefitProgram } from './eligibility.js';
import type { LifeEvent, LifeEventType } from './life-events.js';

/**
 * Delivery channels for nudges
 */
export type NudgeChannel = 'sms' | 'email' | 'dashboard' | 'push' | 'mail';

/**
 * Nudge urgency levels
 */
export type NudgeUrgency = 'low' | 'medium' | 'high' | 'critical';

/**
 * Nudge types
 */
export type NudgeType =
  | 'eligibility_suggestion'    // May qualify for a benefit
  | 'deadline_reminder'         // Upcoming deadline
  | 'action_required'           // Action needed on application
  | 'status_update'             // Application status changed
  | 'recertification_notice'    // Time to recertify
  | 'new_program_available'     // New program matches profile
  | 'life_event_followup'       // Follow-up after life event
  | 'resource_suggestion';      // Local resources available

/**
 * A nudge message to be delivered
 */
export interface Nudge {
  /** Unique nudge ID */
  id: string;
  /** Nudge type */
  type: NudgeType;
  /** Citizen DID */
  citizenDid: string;
  /** Preferred delivery channel */
  preferredChannel: NudgeChannel;
  /** Fallback channels in order */
  fallbackChannels: NudgeChannel[];
  /** Urgency level */
  urgency: NudgeUrgency;
  /** Short title */
  title: string;
  /** Full message body */
  body: string;
  /** SMS-friendly short version */
  smsBody: string;
  /** Call to action */
  callToAction?: {
    text: string;
    url?: string;
    command?: string; // SMS command
  };
  /** Related program */
  relatedProgram?: BenefitProgram;
  /** Related life event */
  relatedEvent?: LifeEventType;
  /** When to deliver */
  deliverAt: Date;
  /** Expiration (don't deliver after) */
  expiresAt?: Date;
  /** Created timestamp */
  createdAt: Date;
  /** Delivery status */
  status: 'pending' | 'delivered' | 'failed' | 'expired' | 'suppressed';
  /** Delivery attempts */
  attempts: NudgeDeliveryAttempt[];
  /** Citizen response (if any) */
  response?: NudgeResponse;
}

/**
 * Nudge delivery attempt
 */
export interface NudgeDeliveryAttempt {
  channel: NudgeChannel;
  attemptedAt: Date;
  success: boolean;
  errorMessage?: string;
}

/**
 * Citizen response to a nudge
 */
export interface NudgeResponse {
  action: 'applied' | 'dismissed' | 'snoozed' | 'opted_out';
  respondedAt: Date;
  snoozeUntil?: Date;
}

/**
 * Citizen notification preferences
 */
export interface CitizenNotificationPrefs {
  /** Opted in to proactive outreach */
  optedIn: boolean;
  /** Preferred channel */
  preferredChannel: NudgeChannel;
  /** Allowed channels */
  allowedChannels: NudgeChannel[];
  /** Quiet hours (no notifications) */
  quietHours?: { start: number; end: number }; // 0-23
  /** Max nudges per week */
  maxNudgesPerWeek: number;
  /** Specific program opt-outs */
  programOptOuts: BenefitProgram[];
  /** Language preference */
  language: string;
}

/**
 * Default notification preferences
 */
export const DEFAULT_NOTIFICATION_PREFS: CitizenNotificationPrefs = {
  optedIn: false,
  preferredChannel: 'sms',
  allowedChannels: ['sms', 'dashboard'],
  quietHours: { start: 21, end: 8 }, // 9pm - 8am
  maxNudgesPerWeek: 3,
  programOptOuts: [],
  language: 'en',
};

/**
 * Nudge Engine Configuration
 */
export interface NudgeEngineConfig {
  /** Enable nudge delivery */
  enabled: boolean;
  /** Default delivery delay after event detection (ms) */
  defaultDeliveryDelay: number;
  /** Maximum delivery attempts */
  maxDeliveryAttempts: number;
  /** Retry delay between attempts (ms) */
  retryDelay: number;
  /** Batch size for processing */
  batchSize: number;
  /** Processing interval (ms) */
  processingInterval: number;
}

/**
 * Default engine configuration
 */
export const DEFAULT_NUDGE_ENGINE_CONFIG: NudgeEngineConfig = {
  enabled: true,
  defaultDeliveryDelay: 4 * 60 * 60 * 1000, // 4 hours
  maxDeliveryAttempts: 3,
  retryDelay: 2 * 60 * 60 * 1000, // 2 hours
  batchSize: 100,
  processingInterval: 5 * 60 * 1000, // 5 minutes
};

/**
 * Channel-specific delivery functions
 */
export interface ChannelDeliveryFunctions {
  sms?: (to: string, body: string) => Promise<boolean>;
  email?: (to: string, subject: string, body: string) => Promise<boolean>;
  dashboard?: (citizenDid: string, notification: DashboardNotification) => Promise<boolean>;
  push?: (citizenDid: string, notification: PushNotification) => Promise<boolean>;
  mail?: (address: MailAddress, content: MailContent) => Promise<boolean>;
}

export interface DashboardNotification {
  title: string;
  body: string;
  action?: { text: string; url: string };
}

export interface PushNotification {
  title: string;
  body: string;
  data?: Record<string, string>;
}

export interface MailAddress {
  name: string;
  street: string;
  city: string;
  state: string;
  zip: string;
}

export interface MailContent {
  subject: string;
  body: string;
}

/**
 * Nudge Engine
 *
 * Manages creation, scheduling, and delivery of proactive nudges.
 */
export class NudgeEngine {
  private config: NudgeEngineConfig;
  private pendingNudges: Map<string, Nudge> = new Map();
  private citizenPrefs: Map<string, CitizenNotificationPrefs> = new Map();
  private deliveryFunctions: ChannelDeliveryFunctions = {};
  private citizenContacts: Map<string, CitizenContactInfo> = new Map();
  private processingInterval: ReturnType<typeof setInterval> | null = null;
  private nudgeCountByWeek: Map<string, number> = new Map();

  constructor(config: Partial<NudgeEngineConfig> = {}) {
    this.config = { ...DEFAULT_NUDGE_ENGINE_CONFIG, ...config };
  }

  /**
   * Start the nudge engine
   */
  start(): void {
    if (!this.config.enabled) return;

    this.processingInterval = setInterval(() => {
      this.processNudges();
    }, this.config.processingInterval);

    console.log('[NudgeEngine] Started');
  }

  /**
   * Stop the nudge engine
   */
  stop(): void {
    if (this.processingInterval) {
      clearInterval(this.processingInterval);
      this.processingInterval = null;
    }
    console.log('[NudgeEngine] Stopped');
  }

  /**
   * Register delivery functions for channels
   */
  registerDeliveryFunctions(functions: ChannelDeliveryFunctions): void {
    this.deliveryFunctions = { ...this.deliveryFunctions, ...functions };
  }

  /**
   * Set citizen notification preferences
   */
  setCitizenPrefs(citizenDid: string, prefs: Partial<CitizenNotificationPrefs>): void {
    const current = this.citizenPrefs.get(citizenDid) || DEFAULT_NOTIFICATION_PREFS;
    this.citizenPrefs.set(citizenDid, { ...current, ...prefs });
  }

  /**
   * Set citizen contact info
   */
  setCitizenContact(citizenDid: string, contact: CitizenContactInfo): void {
    this.citizenContacts.set(citizenDid, contact);
  }

  /**
   * Create a nudge from a life event
   */
  createEventNudge(
    citizenDid: string,
    event: LifeEvent,
    predictions: EligibilityPrediction[]
  ): Nudge | null {
    const prefs = this.citizenPrefs.get(citizenDid);
    if (!prefs?.optedIn) return null;

    // Filter to likely/possible predictions not opted out
    const relevantPredictions = predictions.filter(
      (p) => p.prediction !== 'unlikely' && !prefs.programOptOuts.includes(p.program)
    );

    if (relevantPredictions.length === 0) return null;

    const topPrograms = relevantPredictions.slice(0, 3);
    const programNames = topPrograms.map((p) => this.getProgramDisplayName(p.program)).join(', ');

    const nudge: Nudge = {
      id: this.generateId(),
      type: 'life_event_followup',
      citizenDid,
      preferredChannel: prefs.preferredChannel,
      fallbackChannels: prefs.allowedChannels.filter((c) => c !== prefs.preferredChannel),
      urgency: this.getUrgencyForEvent(event.type),
      title: `You may qualify for assistance`,
      body: `Based on your recent ${event.type.replace(/_/g, ' ')}, you may be eligible for: ${programNames}.\n\nWould you like to check your eligibility? This is completely optional.`,
      smsBody: `You may qualify for ${programNames} based on recent changes. Reply BENEFITS CHECK to learn more, or STOP to opt out.`,
      callToAction: {
        text: 'Check Eligibility',
        command: 'BENEFITS CHECK',
      },
      relatedProgram: topPrograms[0].program,
      relatedEvent: event.type,
      deliverAt: new Date(Date.now() + this.config.defaultDeliveryDelay),
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // 7 days
      createdAt: new Date(),
      status: 'pending',
      attempts: [],
    };

    this.pendingNudges.set(nudge.id, nudge);
    return nudge;
  }

  /**
   * Create a deadline reminder nudge
   */
  createDeadlineNudge(
    citizenDid: string,
    program: BenefitProgram,
    deadline: Date,
    description: string
  ): Nudge | null {
    const prefs = this.citizenPrefs.get(citizenDid);
    if (!prefs?.optedIn) return null;
    if (prefs.programOptOuts.includes(program)) return null;

    const daysUntil = Math.ceil((deadline.getTime() - Date.now()) / (1000 * 60 * 60 * 24));
    const urgency: NudgeUrgency = daysUntil <= 3 ? 'critical' : daysUntil <= 7 ? 'high' : 'medium';

    const nudge: Nudge = {
      id: this.generateId(),
      type: 'deadline_reminder',
      citizenDid,
      preferredChannel: prefs.preferredChannel,
      fallbackChannels: prefs.allowedChannels.filter((c) => c !== prefs.preferredChannel),
      urgency,
      title: `Action needed: ${this.getProgramDisplayName(program)}`,
      body: `Reminder: ${description}\n\nDeadline: ${deadline.toLocaleDateString()}\n\nDon't miss this deadline - your benefits may be affected.`,
      smsBody: `Reminder: ${description} Deadline: ${deadline.toLocaleDateString()}. Reply STATUS for details.`,
      callToAction: {
        text: 'View Details',
        command: 'STATUS',
      },
      relatedProgram: program,
      deliverAt: new Date(), // Immediate
      expiresAt: deadline,
      createdAt: new Date(),
      status: 'pending',
      attempts: [],
    };

    this.pendingNudges.set(nudge.id, nudge);
    return nudge;
  }

  /**
   * Create a status update nudge
   */
  createStatusNudge(
    citizenDid: string,
    program: BenefitProgram,
    caseNumber: string,
    newStatus: string,
    nextStep?: string
  ): Nudge | null {
    const prefs = this.citizenPrefs.get(citizenDid);
    if (!prefs?.optedIn) return null;

    const nudge: Nudge = {
      id: this.generateId(),
      type: 'status_update',
      citizenDid,
      preferredChannel: prefs.preferredChannel,
      fallbackChannels: prefs.allowedChannels.filter((c) => c !== prefs.preferredChannel),
      urgency: newStatus === 'approved' ? 'high' : 'medium',
      title: `Update: ${this.getProgramDisplayName(program)}`,
      body: `Your application ${caseNumber} status: ${newStatus}\n\n${nextStep ? `Next step: ${nextStep}` : ''}`,
      smsBody: `${this.getProgramDisplayName(program)} Case ${caseNumber}: ${newStatus}. ${nextStep || 'Reply STATUS for details.'}`,
      callToAction: {
        text: 'View Details',
        command: `STATUS ${caseNumber}`,
      },
      relatedProgram: program,
      deliverAt: new Date(),
      createdAt: new Date(),
      status: 'pending',
      attempts: [],
    };

    this.pendingNudges.set(nudge.id, nudge);
    return nudge;
  }

  /**
   * Record citizen response to a nudge
   */
  recordResponse(nudgeId: string, response: NudgeResponse): void {
    const nudge = this.pendingNudges.get(nudgeId);
    if (nudge) {
      nudge.response = response;
      if (response.action === 'opted_out' && nudge.citizenDid) {
        // Handle opt-out
        const prefs = this.citizenPrefs.get(nudge.citizenDid);
        if (prefs) {
          prefs.optedIn = false;
          this.citizenPrefs.set(nudge.citizenDid, prefs);
        }
      }
    }
  }

  /**
   * Get pending nudges for a citizen
   */
  getPendingNudges(citizenDid: string): Nudge[] {
    return Array.from(this.pendingNudges.values()).filter(
      (n) => n.citizenDid === citizenDid && n.status === 'pending'
    );
  }

  /**
   * Get statistics
   */
  getStats(): NudgeEngineStats {
    const nudges = Array.from(this.pendingNudges.values());
    return {
      totalPending: nudges.filter((n) => n.status === 'pending').length,
      totalDelivered: nudges.filter((n) => n.status === 'delivered').length,
      totalFailed: nudges.filter((n) => n.status === 'failed').length,
      totalSuppressed: nudges.filter((n) => n.status === 'suppressed').length,
      byType: this.countByField(nudges, 'type'),
      byChannel: this.countByField(nudges, 'preferredChannel'),
      totalOptedIn: Array.from(this.citizenPrefs.values()).filter((p) => p.optedIn).length,
    };
  }

  // Private methods

  private async processNudges(): Promise<void> {
    const now = new Date();
    const toProcess = Array.from(this.pendingNudges.values())
      .filter((n) => n.status === 'pending' && n.deliverAt <= now)
      .slice(0, this.config.batchSize);

    for (const nudge of toProcess) {
      await this.deliverNudge(nudge);
    }
  }

  private async deliverNudge(nudge: Nudge): Promise<void> {
    // Check expiration
    if (nudge.expiresAt && nudge.expiresAt < new Date()) {
      nudge.status = 'expired';
      return;
    }

    // Check rate limiting
    const weekKey = `${nudge.citizenDid}-${this.getWeekKey()}`;
    const weekCount = this.nudgeCountByWeek.get(weekKey) || 0;
    const prefs = this.citizenPrefs.get(nudge.citizenDid);
    if (prefs && weekCount >= prefs.maxNudgesPerWeek) {
      nudge.status = 'suppressed';
      return;
    }

    // Check quiet hours
    if (prefs?.quietHours && this.isQuietHours(prefs.quietHours)) {
      // Reschedule for after quiet hours
      nudge.deliverAt = this.getAfterQuietHours(prefs.quietHours);
      return;
    }

    // Try to deliver
    const channels = [nudge.preferredChannel, ...nudge.fallbackChannels];
    let delivered = false;

    for (const channel of channels) {
      if (nudge.attempts.length >= this.config.maxDeliveryAttempts) break;

      const success = await this.deliverToChannel(nudge, channel);
      nudge.attempts.push({
        channel,
        attemptedAt: new Date(),
        success,
      });

      if (success) {
        delivered = true;
        nudge.status = 'delivered';
        this.nudgeCountByWeek.set(weekKey, weekCount + 1);
        break;
      }
    }

    if (!delivered && nudge.attempts.length >= this.config.maxDeliveryAttempts) {
      nudge.status = 'failed';
    }
  }

  private async deliverToChannel(nudge: Nudge, channel: NudgeChannel): Promise<boolean> {
    const contact = this.citizenContacts.get(nudge.citizenDid);
    if (!contact) return false;

    try {
      switch (channel) {
        case 'sms':
          if (this.deliveryFunctions.sms && contact.phone) {
            return await this.deliveryFunctions.sms(contact.phone, nudge.smsBody);
          }
          break;
        case 'email':
          if (this.deliveryFunctions.email && contact.email) {
            return await this.deliveryFunctions.email(contact.email, nudge.title, nudge.body);
          }
          break;
        case 'dashboard':
          if (this.deliveryFunctions.dashboard) {
            return await this.deliveryFunctions.dashboard(nudge.citizenDid, {
              title: nudge.title,
              body: nudge.body,
              action: nudge.callToAction
                ? { text: nudge.callToAction.text, url: nudge.callToAction.url || '#' }
                : undefined,
            });
          }
          break;
        case 'push':
          if (this.deliveryFunctions.push) {
            return await this.deliveryFunctions.push(nudge.citizenDid, {
              title: nudge.title,
              body: nudge.smsBody,
            });
          }
          break;
        case 'mail':
          if (this.deliveryFunctions.mail && contact.mailingAddress) {
            return await this.deliveryFunctions.mail(contact.mailingAddress, {
              subject: nudge.title,
              body: nudge.body,
            });
          }
          break;
      }
    } catch (error) {
      console.error(`[NudgeEngine] Delivery error for ${channel}:`, error);
    }

    return false;
  }

  private generateId(): string {
    return `nudge-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private getUrgencyForEvent(eventType: LifeEventType): NudgeUrgency {
    const urgencyMap: Partial<Record<LifeEventType, NudgeUrgency>> = {
      job_loss: 'high',
      housing_instability: 'critical',
      health_crisis: 'critical',
      natural_disaster: 'critical',
      benefit_expiration: 'high',
    };
    return urgencyMap[eventType] || 'medium';
  }

  private getProgramDisplayName(program: BenefitProgram): string {
    const names: Record<BenefitProgram, string> = {
      snap: 'SNAP',
      tanf: 'TANF',
      medicaid: 'Medicaid',
      medicare: 'Medicare',
      housing: 'Housing Assistance',
      liheap: 'Energy Assistance',
      childcare: 'Child Care',
      wic: 'WIC',
      unemployment: 'Unemployment',
      disability: 'Disability Benefits',
      social_security: 'Social Security',
      veterans: 'Veterans Benefits',
      job_training: 'Job Training',
      transportation: 'Transportation',
      legal_aid: 'Legal Aid',
    };
    return names[program] || program;
  }

  private getWeekKey(): string {
    const now = new Date();
    const year = now.getFullYear();
    const week = Math.ceil(
      (now.getTime() - new Date(year, 0, 1).getTime()) / (7 * 24 * 60 * 60 * 1000)
    );
    return `${year}-W${week}`;
  }

  private isQuietHours(quietHours: { start: number; end: number }): boolean {
    const hour = new Date().getHours();
    if (quietHours.start < quietHours.end) {
      return hour >= quietHours.start && hour < quietHours.end;
    }
    // Spans midnight
    return hour >= quietHours.start || hour < quietHours.end;
  }

  private getAfterQuietHours(quietHours: { start: number; end: number }): Date {
    const now = new Date();
    const result = new Date(now);
    result.setHours(quietHours.end, 0, 0, 0);
    if (result <= now) {
      result.setDate(result.getDate() + 1);
    }
    return result;
  }

  private countByField(nudges: Nudge[], field: keyof Nudge): Record<string, number> {
    const counts: Record<string, number> = {};
    nudges.forEach((n) => {
      const value = String(n[field]);
      counts[value] = (counts[value] || 0) + 1;
    });
    return counts;
  }
}

/**
 * Citizen contact information
 */
export interface CitizenContactInfo {
  phone?: string;
  email?: string;
  mailingAddress?: MailAddress;
}

/**
 * Nudge engine statistics
 */
export interface NudgeEngineStats {
  totalPending: number;
  totalDelivered: number;
  totalFailed: number;
  totalSuppressed: number;
  byType: Record<string, number>;
  byChannel: Record<string, number>;
  totalOptedIn: number;
}

/**
 * Create a nudge engine instance
 */
export function createNudgeEngine(config: Partial<NudgeEngineConfig> = {}): NudgeEngine {
  return new NudgeEngine(config);
}
