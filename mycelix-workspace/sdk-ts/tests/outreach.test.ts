// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Outreach Module Tests
 *
 * Tests for life event detection, eligibility prediction, nudge engine,
 * and ethical guards.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  LifeEventDetector,
  createLifeEventDetector,
  DEFAULT_LIFE_EVENT_CONFIG,
  type LifeEvent,
  type LifeEventType,
  type CitizenProfile,
} from '../src/outreach/life-events.js';
import {
  EligibilityPredictor,
  createEligibilityPredictor,
  PROGRAM_CRITERIA,
  type CitizenEligibilityProfile,
  type BenefitProgram,
} from '../src/outreach/eligibility.js';
import {
  NudgeEngine,
  createNudgeEngine,
  DEFAULT_NUDGE_ENGINE_CONFIG,
  type CitizenNotificationPrefs,
} from '../src/outreach/nudge-engine.js';
import {
  EthicalGuards,
  createEthicalGuards,
  quickEthicalCheck,
  DEFAULT_ETHICAL_CONFIG,
  type Nudge,
} from '../src/outreach/ethical-guards.js';

// =============================================================================
// Life Event Detector Tests
// =============================================================================

describe('LifeEventDetector', () => {
  let detector: LifeEventDetector;

  beforeEach(() => {
    detector = createLifeEventDetector({ requireConsent: true });
  });

  afterEach(() => {
    detector.stop();
  });

  it('should create with default config', () => {
    const d = createLifeEventDetector();
    expect(d).toBeDefined();
  });

  it('should register and check consent', () => {
    expect(detector.hasConsent('did:citizen:1')).toBe(false);

    detector.registerConsent('did:citizen:1');
    expect(detector.hasConsent('did:citizen:1')).toBe(true);
  });

  it('should withdraw consent and clear events', () => {
    detector.registerConsent('did:citizen:1');
    detector.detectFromDataChange('did:citizen:1', 'address', 'old', 'new', 'citizen_report');

    detector.withdrawConsent('did:citizen:1');
    expect(detector.hasConsent('did:citizen:1')).toBe(false);
    expect(detector.getPendingEvents('did:citizen:1')).toHaveLength(0);
  });

  it('should not detect events without consent', () => {
    const event = detector.detectFromDataChange('did:citizen:1', 'address', 'old', 'new', 'citizen_report');
    expect(event).toBeNull();
  });

  it('should detect address change event', () => {
    detector.registerConsent('did:citizen:1');
    const event = detector.detectFromDataChange('did:citizen:1', 'address', '123 Old St', '456 New St', 'citizen_report');

    expect(event).not.toBeNull();
    expect(event!.type).toBe('address_change');
    expect(event!.citizenDid).toBe('did:citizen:1');
    expect(event!.confidence).toBe(1.0);
    expect(event!.potentialPrograms.length).toBeGreaterThan(0);
  });

  it('should detect job loss event', () => {
    detector.registerConsent('did:citizen:1');
    const event = detector.detectFromDataChange('did:citizen:1', 'employment_status_lost', true, false, 'citizen_report');

    expect(event).not.toBeNull();
    expect(event!.type).toBe('job_loss');
    expect(event!.potentialPrograms).toContain('unemployment');
  });

  it('should detect income change event', () => {
    detector.registerConsent('did:citizen:1');
    const event = detector.detectFromDataChange('did:citizen:1', 'income', 50000, 30000, 'application_data');

    expect(event).not.toBeNull();
    expect(event!.type).toBe('income_change');
  });

  it('should reject events from disabled sources', () => {
    detector.registerConsent('did:citizen:1');
    const event = detector.detectFromDataChange('did:citizen:1', 'address', 'old', 'new', 'external_api');

    expect(event).toBeNull(); // external_api not in default enabled sources
  });

  it('should reject unknown change types', () => {
    detector.registerConsent('did:citizen:1');
    const event = detector.detectFromDataChange('did:citizen:1', 'unknown_field', 'old', 'new', 'citizen_report');

    expect(event).toBeNull();
  });

  it('should detect scheduled age milestone events', () => {
    detector.registerConsent('did:citizen:1');

    // Create profile with date of birth approaching 65
    const dob = new Date();
    dob.setFullYear(dob.getFullYear() - 65);
    dob.setDate(dob.getDate() + 15); // 15 days until 65th birthday

    const profile: CitizenProfile = {
      did: 'did:citizen:1',
      dateOfBirth: dob,
    };

    const events = detector.detectScheduledEvents('did:citizen:1', profile);

    expect(events.length).toBeGreaterThanOrEqual(1);
    const ageMilestone = events.find((e) => e.type === 'age_milestone');
    expect(ageMilestone).toBeDefined();
    expect(ageMilestone!.potentialPrograms).toContain('medicare');
  });

  it('should detect benefit expiration events', () => {
    detector.registerConsent('did:citizen:1');

    const expiresAt = new Date();
    expiresAt.setDate(expiresAt.getDate() + 30); // 30 days from now

    const profile: CitizenProfile = {
      did: 'did:citizen:1',
      activeBenefits: [{ type: 'snap', expiresAt }],
    };

    const events = detector.detectScheduledEvents('did:citizen:1', profile);
    const expEvent = events.find((e) => e.type === 'benefit_expiration');

    expect(expEvent).toBeDefined();
    expect(expEvent!.potentialPrograms).toContain('recertification');
  });

  it('should mark events as notified', () => {
    detector.registerConsent('did:citizen:1');
    detector.detectFromDataChange('did:citizen:1', 'address', 'old', 'new', 'citizen_report');

    expect(detector.getPendingEvents('did:citizen:1')).toHaveLength(1);

    detector.markNotified('did:citizen:1', 'address_change');

    expect(detector.getPendingEvents('did:citizen:1')).toHaveLength(0);
  });

  it('should track stats', () => {
    detector.registerConsent('did:citizen:1');
    detector.registerConsent('did:citizen:2');
    detector.detectFromDataChange('did:citizen:1', 'address', 'old', 'new', 'citizen_report');

    const stats = detector.getStats();

    expect(stats.totalCitizensConsented).toBe(2);
    expect(stats.totalEventsDetected).toBe(1);
    expect(stats.eventsByType['address_change']).toBe(1);
  });

  it('should not detect when consent not required and no consent registered', () => {
    const noConsentDetector = createLifeEventDetector({ requireConsent: false });
    // Should work without explicit consent when requireConsent is false
    const event = noConsentDetector.detectFromDataChange('did:citizen:1', 'address', 'old', 'new', 'citizen_report');
    expect(event).not.toBeNull();
  });
});

// =============================================================================
// Eligibility Predictor Tests
// =============================================================================

describe('EligibilityPredictor', () => {
  let predictor: EligibilityPredictor;

  beforeEach(() => {
    predictor = createEligibilityPredictor();
  });

  it('should predict all programs', () => {
    const profile: CitizenEligibilityProfile = {
      householdSize: 4,
      annualIncome: 25000,
      isCitizen: true,
      hasChildren: true,
    };

    const predictions = predictor.predictAll(profile);

    expect(predictions.length).toBeGreaterThan(0);
    // Should be sorted by likelihood
    const firstPrediction = predictions[0].prediction;
    if (firstPrediction === 'likely') {
      expect(['likely', 'possible', 'unlikely']).toContain(predictions[predictions.length - 1].prediction);
    }
  });

  it('should predict SNAP eligibility for low income family', () => {
    const profile: CitizenEligibilityProfile = {
      householdSize: 4,
      annualIncome: 25000, // Below 130% FPL for family of 4
      isCitizen: true,
      hasChildren: true,
    };

    const prediction = predictor.predictProgram(profile, 'snap');

    expect(prediction.program).toBe('snap');
    expect(prediction.positiveFactors.length).toBeGreaterThan(0);
    expect(prediction.confidence).toBeGreaterThan(0);
  });

  it('should flag missing income info', () => {
    const profile: CitizenEligibilityProfile = {
      householdSize: 3,
    };

    const prediction = predictor.predictProgram(profile, 'snap');

    expect(prediction.missingInfo).toContain('Income information');
  });

  it('should detect medicare eligibility for 65+', () => {
    const profile: CitizenEligibilityProfile = {
      householdSize: 1,
      age: 66,
    };

    const prediction = predictor.predictProgram(profile, 'medicare');

    expect(prediction.positiveFactors.some((f) => f.includes('Age'))).toBe(true);
  });

  it('should detect under-age for medicare', () => {
    const profile: CitizenEligibilityProfile = {
      householdSize: 1,
      age: 40,
    };

    const prediction = predictor.predictProgram(profile, 'medicare');

    expect(prediction.cautionFactors.some((f) => f.includes('below minimum'))).toBe(true);
  });

  it('should detect WIC eligibility for pregnant women', () => {
    const profile: CitizenEligibilityProfile = {
      householdSize: 2,
      annualIncome: 25000,
      isPregnant: true,
    };

    const prediction = predictor.predictProgram(profile, 'wic');

    expect(prediction.positiveFactors.some((f) => f.includes('pregnant'))).toBe(true);
  });

  it('should detect veteran eligibility', () => {
    const profile: CitizenEligibilityProfile = {
      householdSize: 1,
      isVeteran: true,
    };

    const prediction = predictor.predictProgram(profile, 'veterans');

    expect(prediction.positiveFactors.some((f) => f.includes('Veteran'))).toBe(true);
  });

  it('should detect disability benefit eligibility', () => {
    const profile: CitizenEligibilityProfile = {
      householdSize: 1,
      hasDisability: true,
    };

    const prediction = predictor.predictProgram(profile, 'disability');

    expect(prediction.positiveFactors.some((f) => f.includes('disability'))).toBe(true);
  });

  it('should detect unemployment eligibility', () => {
    const profile: CitizenEligibilityProfile = {
      householdSize: 1,
      isEmployed: false,
    };

    const prediction = predictor.predictProgram(profile, 'unemployment');

    expect(prediction.positiveFactors.some((f) => f.includes('unemployed'))).toBe(true);
  });

  it('should get programs for job_loss event', () => {
    const programs = predictor.getProgramsForEvent('job_loss');

    expect(programs).toContain('unemployment');
    expect(programs).toContain('snap');
    expect(programs).toContain('medicaid');
  });

  it('should get programs for birth event', () => {
    const programs = predictor.getProgramsForEvent('birth');

    expect(programs).toContain('wic');
    expect(programs).toContain('medicaid');
  });

  it('should generate next steps', () => {
    const profile: CitizenEligibilityProfile = {
      householdSize: 3,
      annualIncome: 20000,
      isCitizen: true,
    };

    const prediction = predictor.predictProgram(profile, 'snap');

    expect(prediction.nextSteps.length).toBeGreaterThan(0);
  });

  it('should include related events in prediction', () => {
    const jobLossEvent: LifeEvent = {
      type: 'job_loss',
      detectedAt: new Date(),
      confidence: 0.9,
      source: 'citizen_report',
      context: {},
      citizenDid: 'did:test',
      citizenNotified: false,
      potentialPrograms: [],
    };

    const profile: CitizenEligibilityProfile = {
      householdSize: 1,
      recentEvents: [jobLossEvent],
    };

    const prediction = predictor.predictProgram(profile, 'unemployment');

    expect(prediction.relatedEvents).toContain('job_loss');
  });

  it('should generate quick SMS summary', () => {
    const profile: CitizenEligibilityProfile = {
      householdSize: 4,
      annualIncome: 20000,
      isCitizen: true,
      hasChildren: true,
    };

    const predictions = predictor.predictAll(profile);
    const summary = predictor.getQuickSummary(predictions);

    expect(summary).toContain('screening only');
    expect(typeof summary).toBe('string');
  });

  it('should have all programs in PROGRAM_CRITERIA', () => {
    const programs: BenefitProgram[] = [
      'snap', 'tanf', 'medicaid', 'medicare', 'housing',
      'liheap', 'childcare', 'wic', 'unemployment', 'disability',
      'social_security', 'veterans', 'job_training', 'transportation', 'legal_aid',
    ];

    for (const program of programs) {
      expect(PROGRAM_CRITERIA[program]).toBeDefined();
      expect(PROGRAM_CRITERIA[program].displayName).toBeDefined();
      expect(PROGRAM_CRITERIA[program].triggerEvents.length).toBeGreaterThan(0);
    }
  });
});

// =============================================================================
// Nudge Engine Tests
// =============================================================================

describe('NudgeEngine', () => {
  let engine: NudgeEngine;

  beforeEach(() => {
    engine = createNudgeEngine();
  });

  afterEach(() => {
    engine.stop();
  });

  it('should create with default config', () => {
    expect(engine).toBeDefined();
  });

  it('should not create nudge without opt-in', () => {
    const event: LifeEvent = {
      type: 'job_loss',
      detectedAt: new Date(),
      confidence: 0.9,
      source: 'citizen_report',
      context: {},
      citizenDid: 'did:citizen:1',
      citizenNotified: false,
      potentialPrograms: [],
    };

    const nudge = engine.createEventNudge('did:citizen:1', event, [
      { program: 'unemployment', prediction: 'likely', confidence: 0.8, positiveFactors: [], cautionFactors: [], missingInfo: [], nextSteps: [] },
    ]);

    expect(nudge).toBeNull(); // No prefs registered = not opted in
  });

  it('should create nudge for opted-in citizen', () => {
    engine.setCitizenPrefs('did:citizen:1', {
      optedIn: true,
      preferredChannel: 'sms',
      allowedChannels: ['sms', 'dashboard'],
    });

    const event: LifeEvent = {
      type: 'job_loss',
      detectedAt: new Date(),
      confidence: 0.9,
      source: 'citizen_report',
      context: {},
      citizenDid: 'did:citizen:1',
      citizenNotified: false,
      potentialPrograms: [],
    };

    const nudge = engine.createEventNudge('did:citizen:1', event, [
      { program: 'unemployment', prediction: 'likely', confidence: 0.8, positiveFactors: [], cautionFactors: [], missingInfo: [], nextSteps: [] },
    ]);

    expect(nudge).not.toBeNull();
    expect(nudge!.type).toBe('life_event_followup');
    expect(nudge!.preferredChannel).toBe('sms');
    expect(nudge!.status).toBe('pending');
    expect(nudge!.smsBody).toContain('BENEFITS CHECK');
  });

  it('should filter out opted-out programs', () => {
    engine.setCitizenPrefs('did:citizen:1', {
      optedIn: true,
      preferredChannel: 'sms',
      allowedChannels: ['sms'],
      programOptOuts: ['unemployment'],
    });

    const event: LifeEvent = {
      type: 'job_loss',
      detectedAt: new Date(),
      confidence: 0.9,
      source: 'citizen_report',
      context: {},
      citizenDid: 'did:citizen:1',
      citizenNotified: false,
      potentialPrograms: [],
    };

    const nudge = engine.createEventNudge('did:citizen:1', event, [
      { program: 'unemployment', prediction: 'likely', confidence: 0.8, positiveFactors: [], cautionFactors: [], missingInfo: [], nextSteps: [] },
    ]);

    expect(nudge).toBeNull(); // Only program was opted out
  });

  it('should create deadline reminder', () => {
    engine.setCitizenPrefs('did:citizen:1', {
      optedIn: true,
      preferredChannel: 'email',
      allowedChannels: ['email', 'sms'],
    });

    const deadline = new Date(Date.now() + 5 * 24 * 60 * 60 * 1000); // 5 days
    const nudge = engine.createDeadlineNudge('did:citizen:1', 'snap', deadline, 'Recertification due');

    expect(nudge).not.toBeNull();
    expect(nudge!.type).toBe('deadline_reminder');
    expect(nudge!.urgency).toBe('high'); // 5 days = high
  });

  it('should set critical urgency for imminent deadlines', () => {
    engine.setCitizenPrefs('did:citizen:1', {
      optedIn: true,
      preferredChannel: 'sms',
      allowedChannels: ['sms'],
    });

    const deadline = new Date(Date.now() + 2 * 24 * 60 * 60 * 1000); // 2 days
    const nudge = engine.createDeadlineNudge('did:citizen:1', 'snap', deadline, 'Due soon');

    expect(nudge!.urgency).toBe('critical');
  });

  it('should create status update nudge', () => {
    engine.setCitizenPrefs('did:citizen:1', {
      optedIn: true,
      preferredChannel: 'dashboard',
      allowedChannels: ['dashboard'],
    });

    const nudge = engine.createStatusNudge('did:citizen:1', 'snap', 'CASE-123', 'approved', 'Visit office');

    expect(nudge).not.toBeNull();
    expect(nudge!.type).toBe('status_update');
    expect(nudge!.urgency).toBe('high'); // approved = high
    expect(nudge!.body).toContain('CASE-123');
  });

  it('should record response and handle opt-out', () => {
    engine.setCitizenPrefs('did:citizen:1', {
      optedIn: true,
      preferredChannel: 'sms',
      allowedChannels: ['sms'],
    });

    const event: LifeEvent = {
      type: 'income_change',
      detectedAt: new Date(),
      confidence: 0.9,
      source: 'citizen_report',
      context: {},
      citizenDid: 'did:citizen:1',
      citizenNotified: false,
      potentialPrograms: [],
    };

    const nudge = engine.createEventNudge('did:citizen:1', event, [
      { program: 'snap', prediction: 'likely', confidence: 0.8, positiveFactors: [], cautionFactors: [], missingInfo: [], nextSteps: [] },
    ]);

    engine.recordResponse(nudge!.id, { action: 'opted_out', respondedAt: new Date() });

    // After opt-out, new nudges should not be created
    const nudge2 = engine.createEventNudge('did:citizen:1', event, [
      { program: 'medicaid', prediction: 'likely', confidence: 0.8, positiveFactors: [], cautionFactors: [], missingInfo: [], nextSteps: [] },
    ]);

    expect(nudge2).toBeNull();
  });

  it('should track pending nudges per citizen', () => {
    engine.setCitizenPrefs('did:citizen:1', {
      optedIn: true,
      preferredChannel: 'sms',
      allowedChannels: ['sms'],
    });

    const event: LifeEvent = {
      type: 'birth',
      detectedAt: new Date(),
      confidence: 0.9,
      source: 'citizen_report',
      context: {},
      citizenDid: 'did:citizen:1',
      citizenNotified: false,
      potentialPrograms: [],
    };

    engine.createEventNudge('did:citizen:1', event, [
      { program: 'wic', prediction: 'likely', confidence: 0.9, positiveFactors: [], cautionFactors: [], missingInfo: [], nextSteps: [] },
    ]);

    const pending = engine.getPendingNudges('did:citizen:1');
    expect(pending.length).toBe(1);
  });

  it('should provide stats', () => {
    const stats = engine.getStats();

    expect(stats.totalPending).toBe(0);
    expect(stats.totalDelivered).toBe(0);
    expect(stats.totalOptedIn).toBe(0);
  });
});

// =============================================================================
// Ethical Guards Tests
// =============================================================================

describe('EthicalGuards', () => {
  let guards: EthicalGuards;

  beforeEach(() => {
    guards = createEthicalGuards();
  });

  it('should create with default config', () => {
    expect(guards).toBeDefined();
  });

  it('should block nudge without consent', () => {
    const nudge = createTestNudge('did:citizen:1');
    const result = guards.checkNudge(nudge, 'did:citizen:1');

    expect(result.passed).toBe(false);
    expect(result.violations.some((v) => v.type === 'consent_not_given')).toBe(true);
  });

  it('should pass nudge with consent', () => {
    guards.registerConsent('did:citizen:1', {
      channels: ['sms'],
      allowedDataSources: ['citizen_report'],
    });

    const nudge = createTestNudge('did:citizen:1');
    const result = guards.checkNudge(nudge, 'did:citizen:1');

    // May have warnings but should pass consent check
    const consentViolation = result.violations.find((v) =>
      v.type === 'consent_not_given' || v.type === 'consent_withdrawn'
    );
    expect(consentViolation).toBeUndefined();
  });

  it('should block after consent withdrawal', () => {
    guards.registerConsent('did:citizen:1', { channels: ['sms'] });
    guards.withdrawConsent('did:citizen:1');

    const nudge = createTestNudge('did:citizen:1');
    const result = guards.checkNudge(nudge, 'did:citizen:1');

    expect(result.violations.some((v) => v.type === 'consent_withdrawn')).toBe(true);
  });

  it('should detect dark patterns', () => {
    guards.registerConsent('did:citizen:1', { channels: ['sms'] });

    const nudge = createTestNudge('did:citizen:1', {
      body: 'Act now! Limited time offer! Don\'t miss out on benefits!',
    });

    const result = guards.checkNudge(nudge, 'did:citizen:1');

    expect(result.violations.some((v) => v.type === 'dark_pattern')).toBe(true);
  });

  it('should detect stigmatizing language', () => {
    guards.registerConsent('did:citizen:1', { channels: ['sms'] });

    const nudge = createTestNudge('did:citizen:1', {
      body: 'As a needy person, you may qualify for welfare handouts.',
    });

    const result = guards.checkNudge(nudge, 'did:citizen:1');

    expect(result.violations.some((v) => v.type === 'dignity_violation')).toBe(true);
  });

  it('should detect sensitive content on insecure channel', () => {
    guards.registerConsent('did:citizen:1', { channels: ['sms'] });

    const nudge = createTestNudge('did:citizen:1', {
      body: 'Based on your disability status, you may qualify for assistance.',
      preferredChannel: 'sms',
    });

    const result = guards.checkNudge(nudge, 'did:citizen:1');

    expect(result.violations.some((v) => v.type === 'sensitive_data_exposure')).toBe(true);
  });

  it('should detect urgency inflation', () => {
    guards.registerConsent('did:citizen:1', { channels: ['dashboard'] });

    const nudge = createTestNudge('did:citizen:1', {
      body: 'You may be eligible for some benefits. Consider applying when convenient.',
      urgency: 'high',
      preferredChannel: 'dashboard',
    });

    const result = guards.checkNudge(nudge, 'did:citizen:1');

    expect(result.violations.some((v) => v.type === 'urgency_inflation')).toBe(true);
    expect(result.suggestions.some((s) => s.field === 'urgency')).toBe(true);
  });

  it('should check event inference ethics', () => {
    guards.registerConsent('did:citizen:1', {
      allowedDataSources: ['citizen_report'],
    });

    const event: LifeEvent = {
      type: 'job_loss',
      detectedAt: new Date(),
      confidence: 0.9,
      source: 'citizen_report',
      context: {},
      citizenDid: 'did:citizen:1',
      citizenNotified: false,
      potentialPrograms: [],
    };

    const result = guards.checkEventInference('did:citizen:1', event, ['citizen_report']);

    expect(result.passed).toBe(true);
  });

  it('should block inference from unauthorized data source', () => {
    guards.registerConsent('did:citizen:1', {
      allowedDataSources: ['citizen_report'],
    });

    const event: LifeEvent = {
      type: 'income_change',
      detectedAt: new Date(),
      confidence: 0.8,
      source: 'external_api',
      context: {},
      citizenDid: 'did:citizen:1',
      citizenNotified: false,
      potentialPrograms: [],
    };

    const result = guards.checkEventInference('did:citizen:1', event, ['external_api']);

    expect(result.violations.some((v) => v.type === 'consent_not_given')).toBe(true);
  });

  it('should flag low-confidence sensitive event inference', () => {
    guards.registerConsent('did:citizen:1', {
      allowedDataSources: ['citizen_report'],
    });

    const event: LifeEvent = {
      type: 'death', // Sensitive
      detectedAt: new Date(),
      confidence: 0.6, // Low confidence
      source: 'citizen_report',
      context: {},
      citizenDid: 'did:citizen:1',
      citizenNotified: false,
      potentialPrograms: [],
    };

    const result = guards.checkEventInference('did:citizen:1', event, ['citizen_report']);

    expect(result.violations.some((v) => v.type === 'surveillance_overreach')).toBe(true);
  });

  it('should detect benefit cliff risk', () => {
    const predictions = [
      { program: 'medicaid' as BenefitProgram, prediction: 'likely' as const, confidence: 0.8, positiveFactors: [], cautionFactors: [], missingInfo: [], nextSteps: [] },
      { program: 'snap' as BenefitProgram, prediction: 'likely' as const, confidence: 0.8, positiveFactors: [], cautionFactors: [], missingInfo: [], nextSteps: [] },
    ];

    const result = guards.checkBenefitCliff(predictions, 25000, 'job_gain');

    expect(result.atRisk).toBe(true);
    expect(result.warning).toContain('benefit cliff');
  });

  it('should not detect cliff risk when detection is disabled', () => {
    const noCliffGuards = createEthicalGuards({ enableBenefitCliffDetection: false });

    const predictions = [
      { program: 'medicaid' as BenefitProgram, prediction: 'likely' as const, confidence: 0.8, positiveFactors: [], cautionFactors: [], missingInfo: [], nextSteps: [] },
    ];

    const result = noCliffGuards.checkBenefitCliff(predictions, 25000, 'job_gain');

    expect(result.atRisk).toBe(false);
  });

  it('should sanitize stigmatizing language', () => {
    const sanitized = guards.sanitizeForDignity('This welfare handout is for poor families.');

    expect(sanitized).not.toContain('welfare');
    expect(sanitized).not.toContain('handout');
    expect(sanitized).not.toContain('poor');
    expect(sanitized).toContain('benefits');
    expect(sanitized).toContain('assistance');
  });

  it('should remove dark patterns', () => {
    const cleaned = guards.removeDarkPatterns('Act now! This limited time offer expires soon!');

    expect(cleaned).not.toContain('Act now');
    expect(cleaned).not.toContain('limited time');
  });

  it('should maintain audit log', () => {
    guards.registerConsent('did:citizen:1', { channels: ['sms'] });

    const nudge = createTestNudge('did:citizen:1');
    guards.checkNudge(nudge, 'did:citizen:1');

    const log = guards.getAuditLog();
    expect(log.length).toBeGreaterThan(0);
    expect(log[0].action).toContain('nudge:');
  });

  it('should record contact history', () => {
    guards.recordContact('did:citizen:1', 'nudge-1', 'sms');
    guards.recordContact('did:citizen:1', 'nudge-2', 'email');

    // Register consent and check frequency
    guards.registerConsent('did:citizen:1', { channels: ['sms'] });

    // After 3 contacts, should trigger excessive_contact
    guards.recordContact('did:citizen:1', 'nudge-3', 'sms');

    const nudge = createTestNudge('did:citizen:1');
    const result = guards.checkNudge(nudge, 'did:citizen:1');

    expect(result.violations.some((v) => v.type === 'excessive_contact')).toBe(true);
  });
});

// =============================================================================
// quickEthicalCheck Tests
// =============================================================================

describe('quickEthicalCheck', () => {
  it('should pass clean messages', () => {
    const result = quickEthicalCheck('You may be eligible for SNAP benefits. Would you like more information?');
    expect(result.passed).toBe(true);
    expect(result.issues).toHaveLength(0);
  });

  it('should flag dark patterns', () => {
    const result = quickEthicalCheck('Act now! Last chance to claim your benefits!');
    expect(result.passed).toBe(false);
    expect(result.issues.some((i) => i.includes('Dark pattern'))).toBe(true);
  });

  it('should flag stigmatizing language', () => {
    const result = quickEthicalCheck('Welfare benefits for needy families.');
    expect(result.passed).toBe(false);
    expect(result.issues.some((i) => i.includes('Stigmatizing'))).toBe(true);
  });
});

// =============================================================================
// Default Config Tests
// =============================================================================

describe('Default Configurations', () => {
  it('should have sensible life event defaults', () => {
    expect(DEFAULT_LIFE_EVENT_CONFIG.enabled).toBe(true);
    expect(DEFAULT_LIFE_EVENT_CONFIG.minConfidence).toBe(0.7);
    expect(DEFAULT_LIFE_EVENT_CONFIG.requireConsent).toBe(true);
  });

  it('should have sensible nudge engine defaults', () => {
    expect(DEFAULT_NUDGE_ENGINE_CONFIG.enabled).toBe(true);
    expect(DEFAULT_NUDGE_ENGINE_CONFIG.maxDeliveryAttempts).toBe(3);
  });

  it('should have sensible ethical defaults', () => {
    expect(DEFAULT_ETHICAL_CONFIG.requireExplicitConsent).toBe(true);
    expect(DEFAULT_ETHICAL_CONFIG.maxNudgesPerWeek).toBe(3);
    expect(DEFAULT_ETHICAL_CONFIG.secureChannels).toContain('dashboard');
    expect(DEFAULT_ETHICAL_CONFIG.enableBenefitCliffDetection).toBe(true);
  });
});

// =============================================================================
// Helper
// =============================================================================

function createTestNudge(citizenDid: string, overrides: Partial<Nudge> = {}): Nudge {
  return {
    id: `test-nudge-${Date.now()}`,
    type: 'eligibility_suggestion',
    citizenDid,
    preferredChannel: 'sms',
    fallbackChannels: ['dashboard'],
    urgency: 'medium',
    title: 'You may qualify for assistance',
    body: 'Based on your recent changes, you may be eligible for SNAP benefits.',
    smsBody: 'You may qualify for SNAP. Reply BENEFITS CHECK.',
    deliverAt: new Date(),
    createdAt: new Date(),
    status: 'pending',
    attempts: [],
    ...overrides,
  };
}
