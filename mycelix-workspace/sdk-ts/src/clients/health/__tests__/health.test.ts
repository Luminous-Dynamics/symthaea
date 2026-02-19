/**
 * Health Client Tests
 *
 * Verifies zome call arguments and response pass-through for
 * PatientClient, ProviderClient, RecordsClient, PrescriptionsClient,
 * ConsentClient, TrialsClient, and InsuranceClient.
 *
 * Health clients use AppClient directly and return data (ActionHash, entities)
 * without HolochainRecord wrappers.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  PatientClient,
  ProviderClient,
  RecordsClient,
  PrescriptionsClient,
  ConsentClient,
  TrialsClient,
  InsuranceClient,
} from '../../../integrations/health/client';
import type { AppClient } from '@holochain/client';

// ============================================================================
// MOCK HELPERS
// ============================================================================

function createMockClient(): AppClient {
  return {
    callZome: vi.fn(),
  } as unknown as AppClient;
}

const MOCK_HASH = new Uint8Array(32);

// ============================================================================
// MOCK ENTRIES
// ============================================================================

const PATIENT = {
  patient_id: 'P001',
  first_name: 'Alice',
  last_name: 'Johnson',
  date_of_birth: '1990-03-15',
  biological_sex: 'Female',
  contact: { country: 'US', city: 'Richardson', state_province: 'TX' },
  primary_language: 'en',
  allergies: [{ allergen: 'Penicillin', reaction: 'Rash', severity: 'Moderate', verified: true }],
  conditions: ['Asthma'],
  medications: ['Albuterol'],
  matl_trust_score: 0.85,
  created_at: 1708200000,
  updated_at: 1708200000,
};

const PROVIDER = {
  provider_type: 'Physician',
  first_name: 'Dr. Bob',
  last_name: 'Smith',
  title: 'MD',
  specialty: 'Cardiology',
  sub_specialties: ['Interventional Cardiology'],
  locations: [],
  contact: { email: 'dr.smith@hospital.org', phone_office: '555-0100' },
  languages: ['en'],
  accepting_patients: true,
  telehealth_enabled: true,
  matl_trust_score: 0.92,
  epistemic_level: 'PeerReviewed',
  created_at: 1708200000,
  updated_at: 1708200000,
};

const CREDENTIAL_RESULT = {
  provider_name: 'Dr. Bob Smith',
  specialty: 'Cardiology',
  has_active_license: true,
  has_board_certification: true,
  matl_trust_score: 0.92,
};

const LAB_RESULT = {
  result_id: 'lab-001',
  patient_hash: MOCK_HASH,
  ordering_provider: MOCK_HASH,
  loinc_code: '2093-3',
  test_name: 'Total Cholesterol',
  value: '210',
  unit: 'mg/dL',
  reference_range: '< 200',
  interpretation: 'High',
  specimen_type: 'Blood',
  collection_time: 1708200000,
  result_time: 1708210000,
  performing_lab: 'Quest Diagnostics',
  is_critical: false,
};

const PRESCRIPTION = {
  prescription_id: 'rx-001',
  patient_hash: MOCK_HASH,
  prescriber_hash: MOCK_HASH,
  rxnorm_code: '197361',
  medication_name: 'Lisinopril',
  strength: '10mg',
  form: 'Tablet',
  route: 'Oral',
  dosage_instructions: 'Take 1 tablet daily',
  quantity: 30,
  quantity_unit: 'tablets',
  refills_authorized: 3,
  refills_remaining: 3,
  days_supply: 30,
  dispense_as_written: false,
  status: 'Active',
  written_date: 1708200000,
  effective_date: 1708200000,
  expiration_date: 1739736000,
  indication: 'Hypertension',
};

const AUTH_RESULT = {
  authorized: true,
  consent_hash: MOCK_HASH,
  reason: 'Active consent for treatment',
};

const ELIGIBILITY_RESULT = {
  eligible: true,
  reasons: ['Age within range', 'No exclusion criteria met'],
  trial_title: 'Novel Anticoagulant Trial Phase 3',
  trial_phase: 'Phase3',
};

// ============================================================================
// PATIENT CLIENT TESTS
// ============================================================================

describe('PatientClient', () => {
  let mockClient: AppClient;
  let patient: PatientClient;

  beforeEach(() => {
    mockClient = createMockClient();
    patient = new PatientClient(mockClient);
  });

  it('createPatient calls correct zome', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(MOCK_HASH);

    const result = await patient.createPatient(PATIENT as any);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'patient',
      fn_name: 'create_patient',
      payload: PATIENT,
    });
    expect(result).toBe(MOCK_HASH);
  });

  it('getPatient returns patient data', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(PATIENT);

    const result = await patient.getPatient(MOCK_HASH);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'patient',
      fn_name: 'get_patient',
      payload: MOCK_HASH,
    });
    expect(result!.first_name).toBe('Alice');
    expect(result!.biological_sex).toBe('Female');
  });

  it('searchPatientsByName returns patient list', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([PATIENT]);

    const result = await patient.searchPatientsByName('Johnson');

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'patient',
      fn_name: 'search_patients_by_name',
      payload: 'Johnson',
    });
    expect(result).toHaveLength(1);
    expect(result[0].last_name).toBe('Johnson');
  });
});

// ============================================================================
// PROVIDER CLIENT TESTS
// ============================================================================

describe('ProviderClient', () => {
  let mockClient: AppClient;
  let provider: ProviderClient;

  beforeEach(() => {
    mockClient = createMockClient();
    provider = new ProviderClient(mockClient);
  });

  it('createProvider calls correct zome', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(MOCK_HASH);

    const result = await provider.createProvider(PROVIDER as any);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'provider',
      fn_name: 'create_provider',
      payload: PROVIDER,
    });
    expect(result).toBe(MOCK_HASH);
  });

  it('verifyProviderCredentials returns verification result', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(CREDENTIAL_RESULT);

    const result = await provider.verifyProviderCredentials(MOCK_HASH);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'provider',
      fn_name: 'verify_provider_credentials',
      payload: MOCK_HASH,
    });
    expect(result.has_active_license).toBe(true);
    expect(result.has_board_certification).toBe(true);
    expect(result.matl_trust_score).toBe(0.92);
  });
});

// ============================================================================
// RECORDS CLIENT TESTS
// ============================================================================

describe('RecordsClient', () => {
  let mockClient: AppClient;
  let records: RecordsClient;

  beforeEach(() => {
    mockClient = createMockClient();
    records = new RecordsClient(mockClient);
  });

  it('createLabResult calls correct zome', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(MOCK_HASH);

    const result = await records.createLabResult(LAB_RESULT as any);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'records',
      fn_name: 'create_lab_result',
      payload: LAB_RESULT,
    });
    expect(result).toBe(MOCK_HASH);
  });

  it('getPatientLabResults returns result list', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([LAB_RESULT]);

    const result = await records.getPatientLabResults(MOCK_HASH);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'records',
      fn_name: 'get_patient_lab_results',
      payload: MOCK_HASH,
    });
    expect(result).toHaveLength(1);
    expect(result[0].test_name).toBe('Total Cholesterol');
    expect(result[0].interpretation).toBe('High');
  });

  it('getCriticalResults passes null payload', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);

    const result = await records.getCriticalResults();

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'records',
      fn_name: 'get_critical_results',
      payload: null,
    });
    expect(result).toHaveLength(0);
  });
});

// ============================================================================
// PRESCRIPTIONS CLIENT TESTS
// ============================================================================

describe('PrescriptionsClient', () => {
  let mockClient: AppClient;
  let prescriptions: PrescriptionsClient;

  beforeEach(() => {
    mockClient = createMockClient();
    prescriptions = new PrescriptionsClient(mockClient);
  });

  it('createPrescription calls correct zome', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(MOCK_HASH);

    const result = await prescriptions.createPrescription(PRESCRIPTION as any);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'prescriptions',
      fn_name: 'create_prescription',
      payload: PRESCRIPTION,
    });
    expect(result).toBe(MOCK_HASH);
  });

  it('getActivePrescriptions returns prescription list', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([PRESCRIPTION]);

    const result = await prescriptions.getActivePrescriptions(MOCK_HASH);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'prescriptions',
      fn_name: 'get_active_prescriptions',
      payload: MOCK_HASH,
    });
    expect(result).toHaveLength(1);
    expect(result[0].medication_name).toBe('Lisinopril');
    expect(result[0].status).toBe('Active');
  });

  it('discontinuePrescription sends hash and reason', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(MOCK_HASH);

    const result = await prescriptions.discontinuePrescription(MOCK_HASH, 'Side effects');

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'prescriptions',
      fn_name: 'discontinue_prescription',
      payload: { rx_hash: MOCK_HASH, reason: 'Side effects' },
    });
    expect(result).toBe(MOCK_HASH);
  });
});

// ============================================================================
// CONSENT CLIENT TESTS
// ============================================================================

describe('ConsentClient', () => {
  let mockClient: AppClient;
  let consent: ConsentClient;

  beforeEach(() => {
    mockClient = createMockClient();
    consent = new ConsentClient(mockClient);
  });

  it('checkAuthorization returns authorization result', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(AUTH_RESULT);

    const input = {
      patient_hash: MOCK_HASH,
      requestor: MOCK_HASH,
      data_category: 'LabResults' as const,
      permission: 'Read' as const,
      is_emergency: false,
    };
    const result = await consent.checkAuthorization(input);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'consent',
      fn_name: 'check_authorization',
      payload: input,
    });
    expect(result.authorized).toBe(true);
    expect(result.reason).toBe('Active consent for treatment');
  });

  it('revokeConsent sends hash and reason', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(MOCK_HASH);

    const result = await consent.revokeConsent(MOCK_HASH, 'Patient request');

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'consent',
      fn_name: 'revoke_consent',
      payload: { consent_hash: MOCK_HASH, reason: 'Patient request' },
    });
    expect(result).toBe(MOCK_HASH);
  });
});

// ============================================================================
// TRIALS CLIENT TESTS
// ============================================================================

describe('TrialsClient', () => {
  let mockClient: AppClient;
  let trials: TrialsClient;

  beforeEach(() => {
    mockClient = createMockClient();
    trials = new TrialsClient(mockClient);
  });

  it('checkTrialEligibility returns eligibility result', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(ELIGIBILITY_RESULT);

    const result = await trials.checkTrialEligibility(MOCK_HASH, MOCK_HASH, 45);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'trials',
      fn_name: 'check_eligibility',
      payload: { trial_hash: MOCK_HASH, patient_hash: MOCK_HASH, patient_age: 45 },
    });
    expect(result.eligible).toBe(true);
    expect(result.trial_phase).toBe('Phase3');
  });

  it('getRecruitingTrials passes null payload', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);

    const result = await trials.getRecruitingTrials();

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'trials',
      fn_name: 'get_recruiting_trials',
      payload: null,
    });
    expect(result).toHaveLength(0);
  });
});

// ============================================================================
// INSURANCE CLIENT TESTS
// ============================================================================

describe('InsuranceClient', () => {
  let mockClient: AppClient;
  let insurance: InsuranceClient;

  beforeEach(() => {
    mockClient = createMockClient();
    insurance = new InsuranceClient(mockClient);
  });

  it('registerInsurancePlan calls correct zome', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(MOCK_HASH);

    const input = {
      plan_id: 'plan-001',
      patient_hash: MOCK_HASH,
      payer_name: 'Blue Cross',
      payer_id: 'BCBS',
      member_id: 'MEM-12345',
      plan_type: 'PPO',
      coverage_type: 'Medical',
      effective_date: '2026-01-01',
      matl_trust_score: 0.88,
    };
    const result = await insurance.registerInsurancePlan(input);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'insurance',
      fn_name: 'register_insurance_plan',
      payload: input,
    });
    expect(result).toBe(MOCK_HASH);
  });

  it('getPatientInsurance returns plan list', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
      { plan_id: 'plan-001', payer_name: 'Blue Cross' },
    ]);

    const result = await insurance.getPatientInsurance(MOCK_HASH);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'health',
      zome_name: 'insurance',
      fn_name: 'get_patient_insurance',
      payload: MOCK_HASH,
    });
    expect(result).toHaveLength(1);
  });
});
