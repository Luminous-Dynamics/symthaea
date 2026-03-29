// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Health Client
 *
 * Unified client for the Health hApp SDK, providing access to
 * patient management, medical records, prescriptions, consent,
 * clinical trials, and insurance.
 *
 * @module @mycelix/sdk/health
 */

import {
  type AppClient,
  AppWebsocket,
 type ActionHash, type AgentPubKey, type Timestamp } from '@holochain/client';

import { RetryPolicy, RetryPolicies, type RetryOptions } from '../../common/retry';

import type {
  Patient,
  Provider,
  License,
  Encounter,
  Diagnosis,
  LabResult,
  VitalSigns,
  Prescription,
  Consent,
  ClinicalTrial,
  TrialParticipant,
  AdverseEvent,
  Allergy,
  DataCategory,
  DataPermission,
} from './index';

/**
 * Health SDK Error
 */
export class HealthSdkError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly cause?: unknown
  ) {
    super(message);
    this.name = 'HealthSdkError';
  }
}

/**
 * Configuration for the Health client
 */
export interface HealthClientConfig {
  /** Role ID for the health DNA */
  roleId: string;
  /** Retry configuration for zome calls */
  retry?: RetryOptions | RetryPolicy;
}

const DEFAULT_CONFIG: HealthClientConfig = {
  roleId: 'health',
};

/**
 * Connection options for creating a new client
 */
export interface HealthConnectionOptions {
  /** WebSocket URL to connect to */
  url: string;
  /** Optional timeout in milliseconds */
  timeout?: number;
  /** Retry configuration for connection and zome calls */
  retry?: RetryOptions | RetryPolicy;
}

// ============================================================================
// Zome Clients
// ============================================================================

/**
 * Patient Client - Patient management operations
 */
export class PatientClient {
  constructor(
    private readonly client: AppClient,
    private readonly roleId: string = 'health'
  ) {}

  async createPatient(patient: Patient): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'patient',
      fn_name: 'create_patient',
      payload: patient,
    }) as Promise<ActionHash>;
  }

  async getPatient(patientHash: ActionHash): Promise<Patient | null> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'patient',
      fn_name: 'get_patient',
      payload: patientHash,
    }) as Promise<Patient | null>;
  }

  async updatePatient(originalHash: ActionHash, updatedPatient: Patient): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'patient',
      fn_name: 'update_patient',
      payload: { original_hash: originalHash, updated_patient: updatedPatient },
    }) as Promise<ActionHash>;
  }

  async searchPatientsByName(name: string): Promise<Patient[]> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'patient',
      fn_name: 'search_patients_by_name',
      payload: name,
    }) as Promise<Patient[]>;
  }

  async getPatientByMrn(mrn: string): Promise<Patient | null> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'patient',
      fn_name: 'get_patient_by_mrn',
      payload: mrn,
    }) as Promise<Patient | null>;
  }

  async addPatientAllergy(patientHash: ActionHash, allergy: Allergy): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'patient',
      fn_name: 'add_patient_allergy',
      payload: { patient_hash: patientHash, allergy },
    }) as Promise<ActionHash>;
  }
}

/**
 * Provider Client - Healthcare provider management
 */
export class ProviderClient {
  constructor(
    private readonly client: AppClient,
    private readonly roleId: string = 'health'
  ) {}

  async createProvider(provider: Provider): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'provider',
      fn_name: 'create_provider',
      payload: provider,
    }) as Promise<ActionHash>;
  }

  async getProvider(providerHash: ActionHash): Promise<Provider | null> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'provider',
      fn_name: 'get_provider',
      payload: providerHash,
    }) as Promise<Provider | null>;
  }

  async searchProvidersBySpecialty(specialty: string): Promise<Provider[]> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'provider',
      fn_name: 'search_providers_by_specialty',
      payload: specialty,
    }) as Promise<Provider[]>;
  }

  async getProviderByNpi(npi: string): Promise<Provider | null> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'provider',
      fn_name: 'get_provider_by_npi',
      payload: npi,
    }) as Promise<Provider | null>;
  }

  async addLicense(license: License): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'provider',
      fn_name: 'add_license',
      payload: license,
    }) as Promise<ActionHash>;
  }

  async verifyProviderCredentials(providerHash: ActionHash): Promise<{
    provider_name: string;
    specialty: string;
    has_active_license: boolean;
    has_board_certification: boolean;
    matl_trust_score: number;
  }> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'provider',
      fn_name: 'verify_provider_credentials',
      payload: providerHash,
    }) as Promise<{
      provider_name: string;
      specialty: string;
      has_active_license: boolean;
      has_board_certification: boolean;
      matl_trust_score: number;
    }>;
  }
}

/**
 * Records Client - Medical records management
 */
export class RecordsClient {
  constructor(
    private readonly client: AppClient,
    private readonly roleId: string = 'health'
  ) {}

  async createEncounter(encounter: Encounter): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'records',
      fn_name: 'create_encounter',
      payload: encounter,
    }) as Promise<ActionHash>;
  }

  async getPatientEncounters(patientHash: ActionHash): Promise<Encounter[]> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'records',
      fn_name: 'get_patient_encounters',
      payload: patientHash,
    }) as Promise<Encounter[]>;
  }

  async createDiagnosis(diagnosis: Diagnosis): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'records',
      fn_name: 'create_diagnosis',
      payload: diagnosis,
    }) as Promise<ActionHash>;
  }

  async createLabResult(labResult: LabResult): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'records',
      fn_name: 'create_lab_result',
      payload: labResult,
    }) as Promise<ActionHash>;
  }

  async getPatientLabResults(patientHash: ActionHash): Promise<LabResult[]> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'records',
      fn_name: 'get_patient_lab_results',
      payload: patientHash,
    }) as Promise<LabResult[]>;
  }

  async acknowledgeCriticalResult(resultHash: ActionHash): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'records',
      fn_name: 'acknowledge_critical_result',
      payload: { result_hash: resultHash },
    }) as Promise<ActionHash>;
  }

  async recordVitalSigns(vitals: VitalSigns): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'records',
      fn_name: 'record_vital_signs',
      payload: vitals,
    }) as Promise<ActionHash>;
  }

  async getCriticalResults(): Promise<LabResult[]> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'records',
      fn_name: 'get_critical_results',
      payload: null,
    }) as Promise<LabResult[]>;
  }
}

/**
 * Prescriptions Client - Prescription management
 */
export class PrescriptionsClient {
  constructor(
    private readonly client: AppClient,
    private readonly roleId: string = 'health'
  ) {}

  async createPrescription(prescription: Prescription): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'prescriptions',
      fn_name: 'create_prescription',
      payload: prescription,
    }) as Promise<ActionHash>;
  }

  async getPatientPrescriptions(patientHash: ActionHash): Promise<Prescription[]> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'prescriptions',
      fn_name: 'get_patient_prescriptions',
      payload: patientHash,
    }) as Promise<Prescription[]>;
  }

  async getActivePrescriptions(patientHash: ActionHash): Promise<Prescription[]> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'prescriptions',
      fn_name: 'get_active_prescriptions',
      payload: patientHash,
    }) as Promise<Prescription[]>;
  }

  async fillPrescription(fill: {
    fill_id: string;
    prescription_hash: ActionHash;
    pharmacy_hash: ActionHash;
    pharmacist: AgentPubKey;
    fill_date: Timestamp;
    quantity_dispensed: number;
    days_supply_dispensed: number;
    ndc_dispensed: string;
    patient_counseled: boolean;
    drug_interactions_reviewed: boolean;
    status: 'Pending' | 'ReadyForPickup' | 'Dispensed' | 'PartialFill' | 'Cancelled' | 'Returned';
  }): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'prescriptions',
      fn_name: 'fill_prescription',
      payload: fill,
    }) as Promise<ActionHash>;
  }

  async discontinuePrescription(rxHash: ActionHash, reason: string): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'prescriptions',
      fn_name: 'discontinue_prescription',
      payload: { rx_hash: rxHash, reason },
    }) as Promise<ActionHash>;
  }
}

/**
 * Consent Client - Patient consent management
 */
export class ConsentClient {
  constructor(
    private readonly client: AppClient,
    private readonly roleId: string = 'health'
  ) {}

  async createConsent(consent: Consent): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'consent',
      fn_name: 'create_consent',
      payload: consent,
    }) as Promise<ActionHash>;
  }

  async getPatientConsents(patientHash: ActionHash): Promise<Consent[]> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'consent',
      fn_name: 'get_patient_consents',
      payload: patientHash,
    }) as Promise<Consent[]>;
  }

  async getActiveConsents(patientHash: ActionHash): Promise<Consent[]> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'consent',
      fn_name: 'get_active_consents',
      payload: patientHash,
    }) as Promise<Consent[]>;
  }

  async revokeConsent(consentHash: ActionHash, reason: string): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'consent',
      fn_name: 'revoke_consent',
      payload: { consent_hash: consentHash, reason },
    }) as Promise<ActionHash>;
  }

  async checkAuthorization(input: {
    patient_hash: ActionHash;
    requestor: AgentPubKey;
    data_category: DataCategory;
    permission: DataPermission;
    is_emergency: boolean;
  }): Promise<{
    authorized: boolean;
    consent_hash?: ActionHash;
    reason: string;
  }> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'consent',
      fn_name: 'check_authorization',
      payload: input,
    }) as Promise<{
      authorized: boolean;
      consent_hash?: ActionHash;
      reason: string;
    }>;
  }

  async getAccessLogs(patientHash: ActionHash): Promise<Array<{
    log_id: string;
    accessor: AgentPubKey;
    access_type: DataPermission;
    data_categories_accessed: DataCategory[];
    accessed_at: Timestamp;
    emergency_override: boolean;
  }>> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'consent',
      fn_name: 'get_access_logs',
      payload: patientHash,
    }) as Promise<Array<{
      log_id: string;
      accessor: AgentPubKey;
      access_type: DataPermission;
      data_categories_accessed: DataCategory[];
      accessed_at: Timestamp;
      emergency_override: boolean;
    }>>;
  }
}

/**
 * Trials Client - Clinical trial management
 */
export class TrialsClient {
  constructor(
    private readonly client: AppClient,
    private readonly roleId: string = 'health'
  ) {}

  async createTrial(trial: ClinicalTrial): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'trials',
      fn_name: 'create_trial',
      payload: trial,
    }) as Promise<ActionHash>;
  }

  async getTrial(trialHash: ActionHash): Promise<ClinicalTrial | null> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'trials',
      fn_name: 'get_trial',
      payload: trialHash,
    }) as Promise<ClinicalTrial | null>;
  }

  async getRecruitingTrials(): Promise<ClinicalTrial[]> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'trials',
      fn_name: 'get_recruiting_trials',
      payload: null,
    }) as Promise<ClinicalTrial[]>;
  }

  async checkTrialEligibility(
    trialHash: ActionHash,
    patientHash: ActionHash,
    patientAge: number
  ): Promise<{
    eligible: boolean;
    reasons: string[];
    trial_title: string;
    trial_phase: string;
  }> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'trials',
      fn_name: 'check_eligibility',
      payload: { trial_hash: trialHash, patient_hash: patientHash, patient_age: patientAge },
    }) as Promise<{
      eligible: boolean;
      reasons: string[];
      trial_title: string;
      trial_phase: string;
    }>;
  }

  async enrollParticipant(participant: TrialParticipant): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'trials',
      fn_name: 'enroll_participant',
      payload: participant,
    }) as Promise<ActionHash>;
  }

  async withdrawParticipant(participantHash: ActionHash, reason: string): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'trials',
      fn_name: 'withdraw_participant',
      payload: { participant_hash: participantHash, reason },
    }) as Promise<ActionHash>;
  }

  async reportAdverseEvent(event: AdverseEvent): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'trials',
      fn_name: 'report_adverse_event',
      payload: event,
    }) as Promise<ActionHash>;
  }

  async getSeriousAdverseEvents(trialHash: ActionHash): Promise<AdverseEvent[]> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'trials',
      fn_name: 'get_serious_adverse_events',
      payload: trialHash,
    }) as Promise<AdverseEvent[]>;
  }
}

/**
 * Insurance Client - Insurance and billing operations
 */
export class InsuranceClient {
  constructor(
    private readonly client: AppClient,
    private readonly roleId: string = 'health'
  ) {}

  async registerInsurancePlan(plan: {
    plan_id: string;
    patient_hash: ActionHash;
    payer_name: string;
    payer_id: string;
    member_id: string;
    plan_type: string;
    coverage_type: string;
    effective_date: string;
    matl_trust_score: number;
    [key: string]: unknown;
  }): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'insurance',
      fn_name: 'register_insurance_plan',
      payload: plan,
    }) as Promise<ActionHash>;
  }

  async getPatientInsurance(patientHash: ActionHash): Promise<unknown[]> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'insurance',
      fn_name: 'get_patient_insurance',
      payload: patientHash,
    }) as Promise<unknown[]>;
  }

  async submitClaim(claim: {
    claim_id: string;
    patient_hash: ActionHash;
    plan_hash: ActionHash;
    encounter_hash: ActionHash;
    billing_provider_hash: ActionHash;
    primary_diagnosis: string;
    line_items: Array<{
      line_number: number;
      procedure_code: string;
      charge_amount: number;
      [key: string]: unknown;
    }>;
    total_charges: number;
    [key: string]: unknown;
  }): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleId,
      zome_name: 'insurance',
      fn_name: 'submit_claim',
      payload: claim,
    }) as Promise<ActionHash>;
  }
}

// ============================================================================
// Unified Client
// ============================================================================

/**
 * Unified Mycelix Health Client
 *
 * Provides access to all health functionality through a single interface.
 *
 * @example
 * ```typescript
 * import { MycelixHealthClient } from '@mycelix/sdk/health';
 *
 * // Connect to Holochain
 * const health = await MycelixHealthClient.connect({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // Create a patient
 * const patientHash = await health.patient.createPatient({
 *   patient_id: 'P001',
 *   first_name: 'John',
 *   last_name: 'Doe',
 *   date_of_birth: '1990-01-15',
 *   biological_sex: 'Male',
 *   contact: { country: 'US' },
 *   primary_language: 'en',
 *   allergies: [],
 *   conditions: [],
 *   medications: [],
 *   matl_trust_score: 0.8,
 *   created_at: Date.now() * 1000,
 *   updated_at: Date.now() * 1000,
 * });
 *
 * // Search providers
 * const cardiologists = await health.provider.searchProvidersBySpecialty('Cardiology');
 *
 * // Check consent
 * const auth = await health.consent.checkAuthorization({
 *   patient_hash: patientHash,
 *   requestor: myAgentKey,
 *   data_category: 'LabResults',
 *   permission: 'Read',
 *   is_emergency: false,
 * });
 * ```
 */
export class MycelixHealthClient {
  /** Patient management */
  public readonly patient: PatientClient;

  /** Provider management */
  public readonly provider: ProviderClient;

  /** Medical records */
  public readonly records: RecordsClient;

  /** Prescriptions */
  public readonly prescriptions: PrescriptionsClient;

  /** Consent management */
  public readonly consent: ConsentClient;

  /** Clinical trials */
  public readonly trials: TrialsClient;

  /** Insurance operations */
  public readonly insurance: InsuranceClient;

  private readonly config: HealthClientConfig;

  /** Retry policy for zome calls */
  private readonly retryPolicy: RetryPolicy;

  /**
   * Create a health client from an existing Holochain client
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration overrides
   */
  constructor(
    private readonly client: AppClient,
    config: Partial<HealthClientConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    // Set up retry policy
    if (config.retry instanceof RetryPolicy) {
      this.retryPolicy = config.retry;
    } else if (config.retry) {
      this.retryPolicy = new RetryPolicy(config.retry);
    } else {
      this.retryPolicy = RetryPolicies.standard;
    }

    // Initialize sub-clients
    this.patient = new PatientClient(client, this.config.roleId);
    this.provider = new ProviderClient(client, this.config.roleId);
    this.records = new RecordsClient(client, this.config.roleId);
    this.prescriptions = new PrescriptionsClient(client, this.config.roleId);
    this.consent = new ConsentClient(client, this.config.roleId);
    this.trials = new TrialsClient(client, this.config.roleId);
    this.insurance = new InsuranceClient(client, this.config.roleId);
  }

  /**
   * Connect to Holochain and create a health client
   *
   * @param options - Connection options
   * @returns Connected health client
   */
  static async connect(
    options: HealthConnectionOptions
  ): Promise<MycelixHealthClient> {
    // Set up retry for connection
    const retryPolicy = options.retry instanceof RetryPolicy
      ? options.retry
      : options.retry
        ? new RetryPolicy(options.retry)
        : RetryPolicies.network;

    const connectFn = async () => {
      try {
        const client = await AppWebsocket.connect({
          url: new URL(options.url),
          wsClientOptions: { origin: 'mycelix-health-sdk' },
        });

        return new MycelixHealthClient(client, { retry: retryPolicy });
      } catch (error) {
        throw new HealthSdkError(
          'CONNECTION_ERROR',
          `Failed to connect to Holochain: ${error instanceof Error ? error.message : String(error)}`,
          error
        );
      }
    };

    return retryPolicy.execute(connectFn);
  }

  /**
   * Create a health client from an existing AppClient
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration overrides
   * @returns Health client
   */
  static fromClient(
    client: AppClient,
    config: Partial<HealthClientConfig> = {}
  ): MycelixHealthClient {
    return new MycelixHealthClient(client, config);
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Get complete patient summary with all related data
   *
   * @param patientHash - Patient identifier
   * @returns Complete patient information
   */
  async getPatientSummary(patientHash: ActionHash): Promise<{
    patient: Patient | null;
    encounters: Encounter[];
    activePrescriptions: Prescription[];
    recentLabResults: LabResult[];
    activeConsents: Consent[];
  } | null> {
    const patient = await this.patient.getPatient(patientHash);
    if (!patient) {
      return null;
    }

    const [encounters, activePrescriptions, recentLabResults, activeConsents] = await Promise.all([
      this.records.getPatientEncounters(patientHash),
      this.prescriptions.getActivePrescriptions(patientHash),
      this.records.getPatientLabResults(patientHash),
      this.consent.getActiveConsents(patientHash),
    ]);

    return {
      patient,
      encounters,
      activePrescriptions,
      recentLabResults: recentLabResults.slice(0, 10),
      activeConsents,
    };
  }

  /**
   * Find eligible trials for a patient
   *
   * @param patientHash - Patient identifier
   * @param patientAge - Patient's age
   * @returns Eligible clinical trials
   */
  async findEligibleTrials(
    patientHash: ActionHash,
    patientAge: number
  ): Promise<Array<{
    trial: ClinicalTrial;
    eligibility: {
      eligible: boolean;
      reasons: string[];
    };
  }>> {
    const recruitingTrials = await this.trials.getRecruitingTrials();
    const results: Array<{
      trial: ClinicalTrial;
      eligibility: { eligible: boolean; reasons: string[] };
    }> = [];

    for (const trial of recruitingTrials) {
      try {
        const eligibility = await this.trials.checkTrialEligibility(
          trial.trial_id as unknown as ActionHash,
          patientHash,
          patientAge
        );
        results.push({
          trial,
          eligibility: {
            eligible: eligibility.eligible,
            reasons: eligibility.reasons,
          },
        });
      } catch {
        // Skip trials where eligibility check fails
      }
    }

    return results;
  }

  /**
   * Create encounter with diagnosis and vitals
   *
   * @param encounter - Encounter details
   * @param vitals - Vital signs
   * @param diagnosis - Optional diagnosis
   * @returns Created record hashes
   */
  async createEncounterWithData(
    encounter: Encounter,
    vitals: VitalSigns,
    diagnosis?: Diagnosis
  ): Promise<{
    encounterHash: ActionHash;
    vitalsHash: ActionHash;
    diagnosisHash?: ActionHash;
  }> {
    const encounterHash = await this.records.createEncounter(encounter);
    const vitalsHash = await this.records.recordVitalSigns({
      ...vitals,
      encounter_hash: encounterHash,
    });

    let diagnosisHash: ActionHash | undefined;
    if (diagnosis) {
      diagnosisHash = await this.records.createDiagnosis({
        ...diagnosis,
        encounter_hash: encounterHash,
      });
    }

    return {
      encounterHash,
      vitalsHash,
      diagnosisHash,
    };
  }

  /**
   * Get critical alerts for a provider
   *
   * @returns Critical results and other alerts
   */
  async getCriticalAlerts(): Promise<{
    criticalLabResults: LabResult[];
    unacknowledgedCount: number;
  }> {
    const criticalLabResults = await this.records.getCriticalResults();
    const unacknowledgedCount = criticalLabResults.filter(r => !r.acknowledged_by).length;

    return {
      criticalLabResults,
      unacknowledgedCount,
    };
  }

  /**
   * Get the underlying Holochain client
   */
  getClient(): AppClient {
    return this.client;
  }

  /**
   * Get the current retry policy
   */
  getRetryPolicy(): RetryPolicy {
    return this.retryPolicy;
  }

  /**
   * Create a new client with a different retry policy
   *
   * @param retry - New retry configuration
   * @returns New client instance with updated retry policy
   */
  withRetry(retry: RetryOptions | RetryPolicy): MycelixHealthClient {
    return new MycelixHealthClient(this.client, { ...this.config, retry });
  }
}

export default MycelixHealthClient;
