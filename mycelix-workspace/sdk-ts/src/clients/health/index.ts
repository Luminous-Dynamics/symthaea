// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Health Client - Master SDK (Phase 4)
 *
 * Unified client for the mycelix-health hApp.
 * Provides patient management, medical records, prescriptions, consent,
 * clinical trials, and insurance operations.
 *
 * Re-exports from existing comprehensive implementation.
 *
 * @module @mycelix/sdk/clients/health
 */

// Re-export all types and clients from the existing comprehensive implementation
export * from '../../integrations/health/index.js';
export * from '../../integrations/health/client.js';

// Re-export defaults (these don't conflict with named exports)
export { default as HealthTypes } from '../../integrations/health/index.js';
export { default as HealthClientDefault } from '../../integrations/health/client.js';

// Note: Explicit re-exports removed to avoid duplicate identifier errors.
// All types are already exported via `export *` above:
// - Patient, Provider, License, Encounter, Diagnosis, LabResult, VitalSigns
// - Prescription, Consent, ClinicalTrial, TrialParticipant, AdverseEvent, Allergy
// - DataCategory, DataPermission
// - MycelixHealthClient, HealthSdkError, HealthClientConfig, HealthConnectionOptions
// - PatientClient, ProviderClient, RecordsClient, PrescriptionsClient
// - ConsentClient, TrialsClient, InsuranceClient

/**
 * Health Client - Unified client for mycelix-health hApp
 *
 * Total: ~45 functions covered across 7 zome clients
 * - patient: Create, get, update, search patients (6 functions)
 * - provider: Create, get, search, verify providers (6 functions)
 * - records: Encounters, diagnoses, labs, vitals (8 functions)
 * - prescriptions: Create, get, fill, discontinue (5 functions)
 * - consent: Create, get, revoke, check authorization (6 functions)
 * - trials: Create, get, enroll, adverse events (8 functions)
 * - insurance: Plans, claims (3 functions)
 */
import {
  PatientClient,
  ProviderClient,
  RecordsClient,
  PrescriptionsClient,
  ConsentClient,
  TrialsClient,
  InsuranceClient,
} from '../../integrations/health/client.js';

import type { AppClient } from '@holochain/client';

export class HealthClient {
  /** Patient management */
  readonly patient: PatientClient;

  /** Provider management */
  readonly provider: ProviderClient;

  /** Medical records */
  readonly records: RecordsClient;

  /** Prescriptions */
  readonly prescriptions: PrescriptionsClient;

  /** Consent management */
  readonly consent: ConsentClient;

  /** Clinical trials */
  readonly trials: TrialsClient;

  /** Insurance operations */
  readonly insurance: InsuranceClient;

  constructor(
    client: AppClient,
    roleId: string = 'health'
  ) {
    this.patient = new PatientClient(client, roleId);
    this.provider = new ProviderClient(client, roleId);
    this.records = new RecordsClient(client, roleId);
    this.prescriptions = new PrescriptionsClient(client, roleId);
    this.consent = new ConsentClient(client, roleId);
    this.trials = new TrialsClient(client, roleId);
    this.insurance = new InsuranceClient(client, roleId);
  }
}
