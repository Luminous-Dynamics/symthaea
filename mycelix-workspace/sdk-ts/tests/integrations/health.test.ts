// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Health Integration Tests
 *
 * Tests for HealthService -- the domain-specific SDK client for the
 * Mycelix-Health hApp. Covers patient management, provider management,
 * medical records, prescriptions, consent management, clinical trials,
 * and insurance.
 *
 * All calls are dispatched through the health role (configurable via cellId)
 * to the patient, provider, records, prescriptions, consent, trials,
 * and insurance coordinator zomes.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

import {
  HealthService,
  type Patient,
  type BiologicalSex,
  type BloodType,
  type ContactInfo,
  type EmergencyContact,
  type Allergy,
  type AllergySeverity,
  type Provider,
  type ProviderType,
  type EpistemicLevel,
  type PracticeLocation,
  type ProviderContact,
  type License,
  type LicenseType,
  type LicenseStatus,
  type Encounter,
  type EncounterType,
  type EncounterStatus,
  type RecordsEpistemicLevel,
  type Diagnosis,
  type DiagnosisType,
  type DiagnosisStatus,
  type DiagnosisSeverity,
  type ProcedurePerformed,
  type ProcedureOutcome,
  type LabResult,
  type LabInterpretation,
  type VitalSigns,
  type Prescription,
  type MedicationForm,
  type AdministrationRoute,
  type PrescriptionStatus,
  type DrugSchedule,
  type Consent,
  type ConsentGrantee,
  type ConsentScope,
  type DataCategory,
  type DataPermission,
  type ConsentPurpose,
  type ConsentStatus,
  type ClinicalTrial,
  type TrialPhase,
  type StudyType,
  type TrialStatus,
  type TrialEpistemicLevel,
  type EligibilityCriteria,
  type Intervention,
  type InterventionType,
  type Outcome,
  type TrialParticipant,
  type ParticipantStatus,
  type AdverseEvent,
  type AESeverity,
  type SeriousnessCriteria,
  type Causality,
  type AEOutcome,
  type ActionTaken,
} from '../../src/integrations/health/index.js';

// ============================================================================
// Mock Holochain client
// ============================================================================

function createMockClient() {
  return {
    callZome: vi.fn().mockResolvedValue({}),
  };
}

// ============================================================================
// Test helpers
// ============================================================================

function fakeActionHash(): Uint8Array {
  return new Uint8Array(39).fill(0xab);
}

function fakeAgentKey(): Uint8Array {
  return new Uint8Array(39).fill(0xca);
}

// ============================================================================
// Type construction tests
// ============================================================================

describe('Health Types', () => {
  describe('BiologicalSex', () => {
    it('should accept all BiologicalSex variants', () => {
      const sexes: BiologicalSex[] = ['Male', 'Female', 'Intersex', 'Unknown'];
      expect(sexes).toHaveLength(4);
    });
  });

  describe('BloodType', () => {
    it('should accept all BloodType variants', () => {
      const types: BloodType[] = [
        'APositive', 'ANegative', 'BPositive', 'BNegative',
        'ABPositive', 'ABNegative', 'OPositive', 'ONegative', 'Unknown',
      ];
      expect(types).toHaveLength(9);
    });
  });

  describe('AllergySeverity', () => {
    it('should accept all AllergySeverity variants', () => {
      const severities: AllergySeverity[] = ['Mild', 'Moderate', 'Severe', 'LifeThreatening'];
      expect(severities).toHaveLength(4);
    });
  });

  describe('ProviderType', () => {
    it('should accept string provider types', () => {
      const types: ProviderType[] = [
        'Physician', 'Nurse', 'NursePractitioner', 'PhysicianAssistant',
        'Pharmacist', 'Therapist', 'Dentist', 'Optometrist',
        'Chiropractor', 'Researcher', 'LabTechnician',
      ];
      expect(types).toHaveLength(11);
    });

    it('should accept custom provider type', () => {
      const custom: ProviderType = { Other: 'Midwife' };
      expect(custom).toEqual({ Other: 'Midwife' });
    });
  });

  describe('EpistemicLevel', () => {
    it('should accept all EpistemicLevel variants', () => {
      const levels: EpistemicLevel[] = ['Unverified', 'PeerReviewed', 'Replicated', 'Consensus'];
      expect(levels).toHaveLength(4);
    });
  });

  describe('LicenseType', () => {
    it('should accept string license types', () => {
      const types: LicenseType[] = [
        'Medical', 'Nursing', 'Pharmacy', 'Dental', 'Psychology',
        'Therapy', 'DEA', 'StateControlled', 'BoardCertification',
      ];
      expect(types).toHaveLength(9);
    });
  });

  describe('LicenseStatus', () => {
    it('should accept all LicenseStatus variants', () => {
      const statuses: LicenseStatus[] = ['Active', 'Expired', 'Suspended', 'Revoked', 'Pending', 'Restricted'];
      expect(statuses).toHaveLength(6);
    });
  });

  describe('EncounterType', () => {
    it('should accept all EncounterType variants', () => {
      const types: EncounterType[] = [
        'Office', 'Emergency', 'Inpatient', 'Outpatient', 'Telehealth',
        'HomeVisit', 'Procedure', 'Surgery', 'LabOnly', 'ImagingOnly',
      ];
      expect(types).toHaveLength(10);
    });
  });

  describe('EncounterStatus', () => {
    it('should accept all EncounterStatus variants', () => {
      const statuses: EncounterStatus[] = ['Planned', 'InProgress', 'Completed', 'Cancelled', 'NoShow'];
      expect(statuses).toHaveLength(5);
    });
  });

  describe('RecordsEpistemicLevel', () => {
    it('should accept all RecordsEpistemicLevel variants', () => {
      const levels: RecordsEpistemicLevel[] = [
        'PatientReported', 'ProviderObserved', 'TestConfirmed', 'Consensus',
      ];
      expect(levels).toHaveLength(4);
    });
  });

  describe('DiagnosisType', () => {
    it('should accept all DiagnosisType variants', () => {
      const types: DiagnosisType[] = ['Primary', 'Secondary', 'Differential', 'RuledOut', 'WorkingDiagnosis'];
      expect(types).toHaveLength(5);
    });
  });

  describe('PrescriptionStatus', () => {
    it('should accept all PrescriptionStatus variants', () => {
      const statuses: PrescriptionStatus[] = [
        'Active', 'Completed', 'Discontinued', 'OnHold',
        'Cancelled', 'Expired', 'EnteredInError',
      ];
      expect(statuses).toHaveLength(7);
    });
  });

  describe('MedicationForm', () => {
    it('should accept string medication forms', () => {
      const forms: MedicationForm[] = [
        'Tablet', 'Capsule', 'Liquid', 'Injection', 'Topical',
        'Patch', 'Inhaler', 'Drops', 'Suppository', 'Powder',
      ];
      expect(forms).toHaveLength(10);
    });
  });

  describe('DrugSchedule', () => {
    it('should accept all DrugSchedule variants', () => {
      const schedules: DrugSchedule[] = [
        'ScheduleI', 'ScheduleII', 'ScheduleIII', 'ScheduleIV', 'ScheduleV', 'NotControlled',
      ];
      expect(schedules).toHaveLength(6);
    });
  });

  describe('DataCategory', () => {
    it('should accept all DataCategory variants', () => {
      const categories: DataCategory[] = [
        'Demographics', 'Allergies', 'Medications', 'Diagnoses', 'Procedures',
        'LabResults', 'ImagingStudies', 'VitalSigns', 'Immunizations',
        'MentalHealth', 'SubstanceAbuse', 'SexualHealth', 'GeneticData',
        'FinancialData', 'All',
      ];
      expect(categories).toHaveLength(15);
    });
  });

  describe('DataPermission', () => {
    it('should accept all DataPermission variants', () => {
      const perms: DataPermission[] = ['Read', 'Write', 'Share', 'Export', 'Delete', 'Amend'];
      expect(perms).toHaveLength(6);
    });
  });

  describe('ConsentStatus', () => {
    it('should accept all ConsentStatus variants', () => {
      const statuses: ConsentStatus[] = ['Active', 'Expired', 'Revoked', 'Pending', 'Rejected'];
      expect(statuses).toHaveLength(5);
    });
  });

  describe('TrialPhase', () => {
    it('should accept all TrialPhase variants', () => {
      const phases: TrialPhase[] = [
        'EarlyPhase1', 'Phase1', 'Phase1Phase2', 'Phase2',
        'Phase2Phase3', 'Phase3', 'Phase4', 'NotApplicable',
      ];
      expect(phases).toHaveLength(8);
    });
  });

  describe('StudyType', () => {
    it('should accept all StudyType variants', () => {
      const types: StudyType[] = ['Interventional', 'Observational', 'ExpandedAccess', 'Registry'];
      expect(types).toHaveLength(4);
    });
  });

  describe('ParticipantStatus', () => {
    it('should accept all ParticipantStatus variants', () => {
      const statuses: ParticipantStatus[] = [
        'Screening', 'Enrolled', 'Active', 'FollowUp',
        'Completed', 'Withdrawn', 'ScreenFail', 'LostToFollowUp',
      ];
      expect(statuses).toHaveLength(8);
    });
  });

  describe('AESeverity', () => {
    it('should accept all AESeverity variants', () => {
      const severities: AESeverity[] = ['Mild', 'Moderate', 'Severe', 'LifeThreatening', 'Death'];
      expect(severities).toHaveLength(5);
    });
  });

  describe('Causality', () => {
    it('should accept all Causality variants', () => {
      const causalities: Causality[] = [
        'DefinitelyRelated', 'ProbablyRelated', 'PossiblyRelated',
        'UnlikelyRelated', 'NotRelated', 'Unknown',
      ];
      expect(causalities).toHaveLength(6);
    });
  });

  describe('ConsentGrantee', () => {
    it('should accept Provider variant', () => {
      const grantee: ConsentGrantee = { Provider: fakeActionHash() };
      expect(grantee).toHaveProperty('Provider');
    });

    it('should accept Organization variant', () => {
      const grantee: ConsentGrantee = { Organization: 'Mayo Clinic' };
      expect(grantee).toEqual({ Organization: 'Mayo Clinic' });
    });

    it('should accept EmergencyAccess variant', () => {
      const grantee: ConsentGrantee = 'EmergencyAccess';
      expect(grantee).toBe('EmergencyAccess');
    });

    it('should accept Public variant', () => {
      const grantee: ConsentGrantee = 'Public';
      expect(grantee).toBe('Public');
    });
  });
});

// ============================================================================
// HealthService
// ============================================================================

describe('HealthService', () => {
  let client: ReturnType<typeof createMockClient>;
  let health: HealthService;

  beforeEach(() => {
    client = createMockClient();
    health = new HealthService(client as any);
  });

  describe('constructor', () => {
    it('should create with default cellId', () => {
      expect(health).toBeInstanceOf(HealthService);
    });

    it('should accept custom cellId', () => {
      const custom = new HealthService(client as any, 'custom-health');
      expect(custom).toBeInstanceOf(HealthService);
    });
  });

  // ==========================================================================
  // Patient Management
  // ==========================================================================

  describe('Patient Management', () => {
    const mockPatient: Patient = {
      patient_id: 'P001',
      first_name: 'Jane',
      last_name: 'Doe',
      date_of_birth: '1990-01-15',
      biological_sex: 'Female',
      contact: { country: 'US' },
      primary_language: 'English',
      allergies: [],
      conditions: [],
      medications: [],
      matl_trust_score: 0.5,
      created_at: Date.now(),
      updated_at: Date.now(),
    };

    describe('createPatient', () => {
      it('should call patient.create_patient with correct params', async () => {
        const mockHash = fakeActionHash();
        client.callZome.mockResolvedValue(mockHash);

        const result = await health.createPatient(mockPatient);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'patient',
          fn_name: 'create_patient',
          payload: mockPatient,
        });
        expect(result).toEqual(mockHash);
      });
    });

    describe('getPatient', () => {
      it('should call patient.get_patient with action hash', async () => {
        const hash = fakeActionHash();
        client.callZome.mockResolvedValue(mockPatient);

        const result = await health.getPatient(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'patient',
          fn_name: 'get_patient',
          payload: hash,
        });
        expect(result).toEqual(mockPatient);
      });
    });

    describe('updatePatient', () => {
      it('should call patient.update_patient with original hash and updated patient', async () => {
        const originalHash = fakeActionHash();
        const updatedPatient = { ...mockPatient, first_name: 'Janet' };
        client.callZome.mockResolvedValue(fakeActionHash());

        await health.updatePatient(originalHash, updatedPatient);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'patient',
          fn_name: 'update_patient',
          payload: { original_hash: originalHash, updated_patient: updatedPatient },
        });
      });
    });

    describe('searchPatientsByName', () => {
      it('should call patient.search_patients_by_name with name string', async () => {
        client.callZome.mockResolvedValue([mockPatient]);

        const result = await health.searchPatientsByName('Doe');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'patient',
          fn_name: 'search_patients_by_name',
          payload: 'Doe',
        });
        expect(result).toHaveLength(1);
      });
    });

    describe('getPatientByMrn', () => {
      it('should call patient.get_patient_by_mrn with MRN string', async () => {
        client.callZome.mockResolvedValue(mockPatient);

        await health.getPatientByMrn('MRN-001');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'patient',
          fn_name: 'get_patient_by_mrn',
          payload: 'MRN-001',
        });
      });
    });

    describe('addPatientAllergy', () => {
      it('should call patient.add_patient_allergy with hash and allergy', async () => {
        const patientHash = fakeActionHash();
        const allergy: Allergy = {
          allergen: 'Penicillin',
          reaction: 'Hives',
          severity: 'Moderate',
          verified: false,
        };
        client.callZome.mockResolvedValue(fakeActionHash());

        await health.addPatientAllergy(patientHash, allergy);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'patient',
          fn_name: 'add_patient_allergy',
          payload: { patient_hash: patientHash, allergy },
        });
      });
    });
  });

  // ==========================================================================
  // Provider Management
  // ==========================================================================

  describe('Provider Management', () => {
    describe('createProvider', () => {
      it('should call provider.create_provider', async () => {
        const provider = { first_name: 'Dr.' } as Provider;
        client.callZome.mockResolvedValue(fakeActionHash());

        await health.createProvider(provider);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'provider',
          fn_name: 'create_provider',
          payload: provider,
        });
      });
    });

    describe('getProvider', () => {
      it('should call provider.get_provider', async () => {
        const hash = fakeActionHash();
        await health.getProvider(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'provider',
          fn_name: 'get_provider',
          payload: hash,
        });
      });
    });

    describe('searchProvidersBySpecialty', () => {
      it('should call provider.search_providers_by_specialty', async () => {
        await health.searchProvidersBySpecialty('Cardiology');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'provider',
          fn_name: 'search_providers_by_specialty',
          payload: 'Cardiology',
        });
      });
    });

    describe('getProviderByNpi', () => {
      it('should call provider.get_provider_by_npi', async () => {
        await health.getProviderByNpi('1234567890');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'provider',
          fn_name: 'get_provider_by_npi',
          payload: '1234567890',
        });
      });
    });

    describe('addLicense', () => {
      it('should call provider.add_license', async () => {
        const license: License = {
          provider_hash: fakeActionHash(),
          license_type: 'Medical',
          license_number: 'TX-12345',
          issuing_authority: 'Texas Medical Board',
          jurisdiction: 'TX',
          issued_date: '2020-01-01',
          expiration_date: '2026-01-01',
          status: 'Active',
        };
        client.callZome.mockResolvedValue(fakeActionHash());

        await health.addLicense(license);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'provider',
          fn_name: 'add_license',
          payload: license,
        });
      });
    });

    describe('verifyProviderCredentials', () => {
      it('should call provider.verify_provider_credentials', async () => {
        const hash = fakeActionHash();
        client.callZome.mockResolvedValue({
          provider_name: 'Dr. Smith',
          specialty: 'Internal Medicine',
          has_active_license: true,
          has_board_certification: true,
          matl_trust_score: 0.9,
        });

        const result = await health.verifyProviderCredentials(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'provider',
          fn_name: 'verify_provider_credentials',
          payload: hash,
        });
        expect(result.has_active_license).toBe(true);
      });
    });
  });

  // ==========================================================================
  // Medical Records
  // ==========================================================================

  describe('Medical Records', () => {
    describe('createEncounter', () => {
      it('should call records.create_encounter', async () => {
        const encounter = { encounter_id: 'E001' } as Encounter;
        client.callZome.mockResolvedValue(fakeActionHash());

        await health.createEncounter(encounter);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'records',
          fn_name: 'create_encounter',
          payload: encounter,
        });
      });
    });

    describe('getPatientEncounters', () => {
      it('should call records.get_patient_encounters', async () => {
        const hash = fakeActionHash();
        await health.getPatientEncounters(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'records',
          fn_name: 'get_patient_encounters',
          payload: hash,
        });
      });
    });

    describe('createDiagnosis', () => {
      it('should call records.create_diagnosis', async () => {
        const diagnosis = { diagnosis_id: 'D001' } as Diagnosis;
        await health.createDiagnosis(diagnosis);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'records',
          fn_name: 'create_diagnosis',
          payload: diagnosis,
        });
      });
    });

    describe('createLabResult', () => {
      it('should call records.create_lab_result', async () => {
        const labResult = { result_id: 'L001' } as LabResult;
        await health.createLabResult(labResult);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'records',
          fn_name: 'create_lab_result',
          payload: labResult,
        });
      });
    });

    describe('getPatientLabResults', () => {
      it('should call records.get_patient_lab_results', async () => {
        const hash = fakeActionHash();
        await health.getPatientLabResults(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'records',
          fn_name: 'get_patient_lab_results',
          payload: hash,
        });
      });
    });

    describe('acknowledgeCriticalResult', () => {
      it('should call records.acknowledge_critical_result with result_hash payload', async () => {
        const hash = fakeActionHash();
        await health.acknowledgeCriticalResult(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'records',
          fn_name: 'acknowledge_critical_result',
          payload: { result_hash: hash },
        });
      });
    });

    describe('recordVitalSigns', () => {
      it('should call records.record_vital_signs', async () => {
        const vitals: VitalSigns = {
          patient_hash: fakeActionHash(),
          recorded_at: Date.now(),
          recorded_by: fakeAgentKey(),
          heart_rate_bpm: 72,
          blood_pressure_systolic: 120,
          blood_pressure_diastolic: 80,
        };
        await health.recordVitalSigns(vitals);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'records',
          fn_name: 'record_vital_signs',
          payload: vitals,
        });
      });
    });

    describe('getCriticalResults', () => {
      it('should call records.get_critical_results with null payload', async () => {
        await health.getCriticalResults();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'records',
          fn_name: 'get_critical_results',
          payload: null,
        });
      });
    });
  });

  // ==========================================================================
  // Prescriptions
  // ==========================================================================

  describe('Prescriptions', () => {
    describe('createPrescription', () => {
      it('should call prescriptions.create_prescription', async () => {
        const rx = { prescription_id: 'RX001' } as Prescription;
        await health.createPrescription(rx);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'prescriptions',
          fn_name: 'create_prescription',
          payload: rx,
        });
      });
    });

    describe('getPatientPrescriptions', () => {
      it('should call prescriptions.get_patient_prescriptions', async () => {
        const hash = fakeActionHash();
        await health.getPatientPrescriptions(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'prescriptions',
          fn_name: 'get_patient_prescriptions',
          payload: hash,
        });
      });
    });

    describe('getActivePrescriptions', () => {
      it('should call prescriptions.get_active_prescriptions', async () => {
        const hash = fakeActionHash();
        await health.getActivePrescriptions(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'prescriptions',
          fn_name: 'get_active_prescriptions',
          payload: hash,
        });
      });
    });

    describe('fillPrescription', () => {
      it('should call prescriptions.fill_prescription', async () => {
        const fill = {
          fill_id: 'F001',
          prescription_hash: fakeActionHash(),
          pharmacy_hash: fakeActionHash(),
          pharmacist: fakeAgentKey(),
          fill_date: Date.now(),
          quantity_dispensed: 30,
          days_supply_dispensed: 30,
          ndc_dispensed: '12345-6789-01',
          patient_counseled: true,
          drug_interactions_reviewed: true,
          status: 'Dispensed' as const,
        };
        await health.fillPrescription(fill);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'prescriptions',
          fn_name: 'fill_prescription',
          payload: fill,
        });
      });
    });

    describe('discontinuePrescription', () => {
      it('should call prescriptions.discontinue_prescription with rx_hash and reason', async () => {
        const rxHash = fakeActionHash();
        await health.discontinuePrescription(rxHash, 'Patient allergic reaction');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'prescriptions',
          fn_name: 'discontinue_prescription',
          payload: { rx_hash: rxHash, reason: 'Patient allergic reaction' },
        });
      });
    });
  });

  // ==========================================================================
  // Consent Management
  // ==========================================================================

  describe('Consent Management', () => {
    describe('createConsent', () => {
      it('should call consent.create_consent', async () => {
        const consent = { consent_id: 'C001' } as Consent;
        await health.createConsent(consent);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'consent',
          fn_name: 'create_consent',
          payload: consent,
        });
      });
    });

    describe('getPatientConsents', () => {
      it('should call consent.get_patient_consents', async () => {
        const hash = fakeActionHash();
        await health.getPatientConsents(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'consent',
          fn_name: 'get_patient_consents',
          payload: hash,
        });
      });
    });

    describe('getActiveConsents', () => {
      it('should call consent.get_active_consents', async () => {
        const hash = fakeActionHash();
        await health.getActiveConsents(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'consent',
          fn_name: 'get_active_consents',
          payload: hash,
        });
      });
    });

    describe('revokeConsent', () => {
      it('should call consent.revoke_consent with consent_hash and reason', async () => {
        const consentHash = fakeActionHash();
        await health.revokeConsent(consentHash, 'Patient requested');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'consent',
          fn_name: 'revoke_consent',
          payload: { consent_hash: consentHash, reason: 'Patient requested' },
        });
      });
    });

    describe('checkAuthorization', () => {
      it('should call consent.check_authorization with correct params', async () => {
        const input = {
          patient_hash: fakeActionHash(),
          requestor: fakeAgentKey(),
          data_category: 'LabResults' as DataCategory,
          permission: 'Read' as DataPermission,
          is_emergency: false,
        };
        client.callZome.mockResolvedValue({ authorized: true, reason: 'Active consent exists' });

        const result = await health.checkAuthorization(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'consent',
          fn_name: 'check_authorization',
          payload: input,
        });
        expect(result.authorized).toBe(true);
      });
    });

    describe('getAccessLogs', () => {
      it('should call consent.get_access_logs', async () => {
        const hash = fakeActionHash();
        await health.getAccessLogs(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'consent',
          fn_name: 'get_access_logs',
          payload: hash,
        });
      });
    });
  });

  // ==========================================================================
  // Clinical Trials
  // ==========================================================================

  describe('Clinical Trials', () => {
    describe('createTrial', () => {
      it('should call trials.create_trial', async () => {
        const trial = { trial_id: 'T001' } as ClinicalTrial;
        await health.createTrial(trial);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'trials',
          fn_name: 'create_trial',
          payload: trial,
        });
      });
    });

    describe('getTrial', () => {
      it('should call trials.get_trial', async () => {
        const hash = fakeActionHash();
        await health.getTrial(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'trials',
          fn_name: 'get_trial',
          payload: hash,
        });
      });
    });

    describe('getRecruitingTrials', () => {
      it('should call trials.get_recruiting_trials with null payload', async () => {
        await health.getRecruitingTrials();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'trials',
          fn_name: 'get_recruiting_trials',
          payload: null,
        });
      });
    });

    describe('checkTrialEligibility', () => {
      it('should call trials.check_eligibility with structured payload', async () => {
        const trialHash = fakeActionHash();
        const patientHash = fakeActionHash();
        client.callZome.mockResolvedValue({
          eligible: true,
          reasons: [],
          trial_title: 'Test Trial',
          trial_phase: 'Phase2',
        });

        const result = await health.checkTrialEligibility(trialHash, patientHash, 45);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'trials',
          fn_name: 'check_eligibility',
          payload: { trial_hash: trialHash, patient_hash: patientHash, patient_age: 45 },
        });
        expect(result.eligible).toBe(true);
      });
    });

    describe('enrollParticipant', () => {
      it('should call trials.enroll_participant', async () => {
        const participant = { participant_id: 'TP001' } as TrialParticipant;
        await health.enrollParticipant(participant);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'trials',
          fn_name: 'enroll_participant',
          payload: participant,
        });
      });
    });

    describe('withdrawParticipant', () => {
      it('should call trials.withdraw_participant with hash and reason', async () => {
        const hash = fakeActionHash();
        await health.withdrawParticipant(hash, 'Patient decision');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'trials',
          fn_name: 'withdraw_participant',
          payload: { participant_hash: hash, reason: 'Patient decision' },
        });
      });
    });

    describe('reportAdverseEvent', () => {
      it('should call trials.report_adverse_event', async () => {
        const event = { event_id: 'AE001' } as AdverseEvent;
        await health.reportAdverseEvent(event);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'trials',
          fn_name: 'report_adverse_event',
          payload: event,
        });
      });
    });

    describe('getSeriousAdverseEvents', () => {
      it('should call trials.get_serious_adverse_events', async () => {
        const hash = fakeActionHash();
        await health.getSeriousAdverseEvents(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'trials',
          fn_name: 'get_serious_adverse_events',
          payload: hash,
        });
      });
    });
  });

  // ==========================================================================
  // Insurance
  // ==========================================================================

  describe('Insurance', () => {
    describe('registerInsurancePlan', () => {
      it('should call insurance.register_insurance_plan', async () => {
        const plan = {
          plan_id: 'INS001',
          patient_hash: fakeActionHash(),
          payer_name: 'Blue Cross',
          payer_id: 'BC-001',
          member_id: 'M-12345',
          plan_type: 'PPO',
          coverage_type: 'Medical',
          effective_date: '2025-01-01',
          matl_trust_score: 0.8,
        };
        await health.registerInsurancePlan(plan);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'insurance',
          fn_name: 'register_insurance_plan',
          payload: plan,
        });
      });
    });

    describe('getPatientInsurance', () => {
      it('should call insurance.get_patient_insurance', async () => {
        const hash = fakeActionHash();
        await health.getPatientInsurance(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'insurance',
          fn_name: 'get_patient_insurance',
          payload: hash,
        });
      });
    });

    describe('submitClaim', () => {
      it('should call insurance.submit_claim', async () => {
        const claim = {
          claim_id: 'CLM001',
          patient_hash: fakeActionHash(),
          plan_hash: fakeActionHash(),
          encounter_hash: fakeActionHash(),
          billing_provider_hash: fakeActionHash(),
          primary_diagnosis: 'J06.9',
          line_items: [{ line_number: 1, procedure_code: '99213', charge_amount: 150 }],
          total_charges: 150,
        };
        await health.submitClaim(claim);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'insurance',
          fn_name: 'submit_claim',
          payload: claim,
        });
      });
    });
  });

  // ==========================================================================
  // Custom cellId
  // ==========================================================================

  describe('Custom cellId', () => {
    it('should use custom cellId as role_name for all calls', async () => {
      const customHealth = new HealthService(client as any, 'my-health-cell');

      await customHealth.createPatient({} as Patient);
      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({ role_name: 'my-health-cell' }),
      );
    });
  });

  // ==========================================================================
  // Error propagation
  // ==========================================================================

  describe('Error propagation', () => {
    it('should propagate zome call errors for createPatient', async () => {
      client.callZome.mockRejectedValue(new Error('Conductor unavailable'));
      await expect(health.createPatient({} as Patient)).rejects.toThrow('Conductor unavailable');
    });

    it('should propagate zome call errors for getPatient', async () => {
      client.callZome.mockRejectedValue(new Error('Not found'));
      await expect(health.getPatient(fakeActionHash())).rejects.toThrow('Not found');
    });

    it('should propagate zome call errors for createEncounter', async () => {
      client.callZome.mockRejectedValue(new Error('Validation failed'));
      await expect(health.createEncounter({} as Encounter)).rejects.toThrow('Validation failed');
    });

    it('should propagate zome call errors for createPrescription', async () => {
      client.callZome.mockRejectedValue(new Error('Permission denied'));
      await expect(health.createPrescription({} as Prescription)).rejects.toThrow('Permission denied');
    });

    it('should propagate zome call errors for checkAuthorization', async () => {
      client.callZome.mockRejectedValue(new Error('Network error'));
      await expect(health.checkAuthorization({
        patient_hash: fakeActionHash(),
        requestor: fakeAgentKey(),
        data_category: 'All',
        permission: 'Read',
        is_emergency: false,
      })).rejects.toThrow('Network error');
    });

    it('should propagate zome call errors for createTrial', async () => {
      client.callZome.mockRejectedValue(new Error('IRB not approved'));
      await expect(health.createTrial({} as ClinicalTrial)).rejects.toThrow('IRB not approved');
    });
  });

  // ==========================================================================
  // Zome routing verification
  // ==========================================================================

  describe('Zome routing', () => {
    it('should route patient methods to patient zome', async () => {
      await health.createPatient({} as Patient);
      await health.getPatient(fakeActionHash());
      await health.searchPatientsByName('test');

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { zome_name: string }).zome_name).toBe('patient');
      }
    });

    it('should route provider methods to provider zome', async () => {
      await health.createProvider({} as Provider);
      await health.getProvider(fakeActionHash());
      await health.searchProvidersBySpecialty('test');

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { zome_name: string }).zome_name).toBe('provider');
      }
    });

    it('should route records methods to records zome', async () => {
      await health.createEncounter({} as Encounter);
      await health.createDiagnosis({} as Diagnosis);
      await health.createLabResult({} as LabResult);
      await health.getCriticalResults();

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { zome_name: string }).zome_name).toBe('records');
      }
    });

    it('should route prescription methods to prescriptions zome', async () => {
      await health.createPrescription({} as Prescription);
      await health.getPatientPrescriptions(fakeActionHash());
      await health.getActivePrescriptions(fakeActionHash());

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { zome_name: string }).zome_name).toBe('prescriptions');
      }
    });

    it('should route consent methods to consent zome', async () => {
      await health.createConsent({} as Consent);
      await health.getPatientConsents(fakeActionHash());
      await health.getActiveConsents(fakeActionHash());

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { zome_name: string }).zome_name).toBe('consent');
      }
    });

    it('should route trials methods to trials zome', async () => {
      await health.createTrial({} as ClinicalTrial);
      await health.getTrial(fakeActionHash());
      await health.getRecruitingTrials();

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { zome_name: string }).zome_name).toBe('trials');
      }
    });

    it('should route insurance methods to insurance zome', async () => {
      const plan = {
        plan_id: 'X',
        patient_hash: fakeActionHash(),
        payer_name: 'X',
        payer_id: 'X',
        member_id: 'X',
        plan_type: 'X',
        coverage_type: 'X',
        effective_date: 'X',
        matl_trust_score: 0,
      };
      await health.registerInsurancePlan(plan);
      await health.getPatientInsurance(fakeActionHash());

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { zome_name: string }).zome_name).toBe('insurance');
      }
    });
  });
});
