// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Health SDK Integration
 *
 * TypeScript SDK for interacting with the Mycelix-Health hApp.
 * Provides patient management, medical records, prescriptions,
 * consent, clinical trials, and insurance functionality.
 */

import type { MycelixClient } from '../../client';
import type { ActionHash, AgentPubKey, Timestamp } from '@holochain/client';

// =============================================================================
// Type Definitions
// =============================================================================

// Patient Types
export interface Patient {
  patient_id: string;
  mrn?: string;
  first_name: string;
  last_name: string;
  date_of_birth: string;
  biological_sex: BiologicalSex;
  gender_identity?: string;
  blood_type?: BloodType;
  contact: ContactInfo;
  emergency_contact?: EmergencyContact;
  primary_language: string;
  allergies: Allergy[];
  conditions: string[];
  medications: string[];
  mycelix_identity_hash?: ActionHash;
  matl_trust_score: number;
  created_at: Timestamp;
  updated_at: Timestamp;
}

export type BiologicalSex = 'Male' | 'Female' | 'Intersex' | 'Unknown';
export type BloodType =
  | 'APositive'
  | 'ANegative'
  | 'BPositive'
  | 'BNegative'
  | 'ABPositive'
  | 'ABNegative'
  | 'OPositive'
  | 'ONegative'
  | 'Unknown';

export interface ContactInfo {
  address_line1?: string;
  address_line2?: string;
  city?: string;
  state_province?: string;
  postal_code?: string;
  country: string;
  phone_primary?: string;
  phone_secondary?: string;
  email?: string;
}

export interface EmergencyContact {
  name: string;
  relationship: string;
  phone: string;
  email?: string;
}

export interface Allergy {
  allergen: string;
  reaction: string;
  severity: AllergySeverity;
  verified: boolean;
  verified_by?: AgentPubKey;
  verified_at?: Timestamp;
}

export type AllergySeverity = 'Mild' | 'Moderate' | 'Severe' | 'LifeThreatening';

// Provider Types
export interface Provider {
  npi?: string;
  provider_type: ProviderType;
  first_name: string;
  last_name: string;
  title: string;
  specialty: string;
  sub_specialties: string[];
  organization?: string;
  locations: PracticeLocation[];
  contact: ProviderContact;
  languages: string[];
  accepting_patients: boolean;
  telehealth_enabled: boolean;
  mycelix_identity_hash?: ActionHash;
  matl_trust_score: number;
  epistemic_level: EpistemicLevel;
  created_at: Timestamp;
  updated_at: Timestamp;
}

export type ProviderType =
  | 'Physician'
  | 'Nurse'
  | 'NursePractitioner'
  | 'PhysicianAssistant'
  | 'Pharmacist'
  | 'Therapist'
  | 'Dentist'
  | 'Optometrist'
  | 'Chiropractor'
  | 'Researcher'
  | 'LabTechnician'
  | { Other: string };

export type EpistemicLevel = 'Unverified' | 'PeerReviewed' | 'Replicated' | 'Consensus';

export interface PracticeLocation {
  name: string;
  address_line1: string;
  address_line2?: string;
  city: string;
  state_province: string;
  postal_code: string;
  country: string;
  phone: string;
  fax?: string;
  hours?: string;
  is_primary: boolean;
}

export interface ProviderContact {
  email: string;
  phone_office: string;
  phone_emergency?: string;
  website?: string;
}

export interface License {
  provider_hash: ActionHash;
  license_type: LicenseType;
  license_number: string;
  issuing_authority: string;
  jurisdiction: string;
  issued_date: string;
  expiration_date: string;
  status: LicenseStatus;
  verification_source?: string;
  verified_at?: Timestamp;
  verified_by?: AgentPubKey;
}

export type LicenseType =
  | 'Medical'
  | 'Nursing'
  | 'Pharmacy'
  | 'Dental'
  | 'Psychology'
  | 'Therapy'
  | 'DEA'
  | 'StateControlled'
  | 'BoardCertification'
  | { Other: string };

export type LicenseStatus = 'Active' | 'Expired' | 'Suspended' | 'Revoked' | 'Pending' | 'Restricted';

// Medical Records Types
export interface Encounter {
  encounter_id: string;
  patient_hash: ActionHash;
  provider_hash: ActionHash;
  encounter_type: EncounterType;
  status: EncounterStatus;
  start_time: Timestamp;
  end_time?: Timestamp;
  location?: string;
  chief_complaint: string;
  diagnoses: Diagnosis[];
  procedures: ProcedurePerformed[];
  notes: string;
  consent_hash: ActionHash;
  epistemic_level: RecordsEpistemicLevel;
  created_at: Timestamp;
  updated_at: Timestamp;
}

export type EncounterType =
  | 'Office'
  | 'Emergency'
  | 'Inpatient'
  | 'Outpatient'
  | 'Telehealth'
  | 'HomeVisit'
  | 'Procedure'
  | 'Surgery'
  | 'LabOnly'
  | 'ImagingOnly';

export type EncounterStatus = 'Planned' | 'InProgress' | 'Completed' | 'Cancelled' | 'NoShow';

export type RecordsEpistemicLevel = 'PatientReported' | 'ProviderObserved' | 'TestConfirmed' | 'Consensus';

export interface Diagnosis {
  diagnosis_id: string;
  patient_hash: ActionHash;
  encounter_hash?: ActionHash;
  icd10_code: string;
  snomed_code?: string;
  description: string;
  diagnosis_type: DiagnosisType;
  status: DiagnosisStatus;
  onset_date?: string;
  resolution_date?: string;
  diagnosing_provider: AgentPubKey;
  severity?: DiagnosisSeverity;
  notes?: string;
  epistemic_level: RecordsEpistemicLevel;
  created_at: Timestamp;
}

export type DiagnosisType = 'Primary' | 'Secondary' | 'Differential' | 'RuledOut' | 'WorkingDiagnosis';
export type DiagnosisStatus = 'Active' | 'Resolved' | 'Inactive' | 'Recurrence' | 'Remission';
export type DiagnosisSeverity = 'Mild' | 'Moderate' | 'Severe' | 'Critical';

export interface ProcedurePerformed {
  procedure_id: string;
  patient_hash: ActionHash;
  encounter_hash: ActionHash;
  cpt_code: string;
  hcpcs_code?: string;
  description: string;
  performed_by: AgentPubKey;
  performed_at: Timestamp;
  location: string;
  outcome: ProcedureOutcome;
  complications: string[];
  notes?: string;
  consent_hash: ActionHash;
}

export type ProcedureOutcome = 'Successful' | 'PartialSuccess' | 'Unsuccessful' | 'Complicated' | 'Aborted';

export interface LabResult {
  result_id: string;
  patient_hash: ActionHash;
  encounter_hash?: ActionHash;
  ordering_provider: AgentPubKey;
  loinc_code: string;
  test_name: string;
  value: string;
  unit: string;
  reference_range: string;
  interpretation: LabInterpretation;
  specimen_type: string;
  collection_time: Timestamp;
  result_time: Timestamp;
  performing_lab: string;
  notes?: string;
  is_critical: boolean;
  acknowledged_by?: AgentPubKey;
  acknowledged_at?: Timestamp;
}

export type LabInterpretation = 'Normal' | 'Abnormal' | 'High' | 'Low' | 'Critical' | 'Inconclusive';

export interface VitalSigns {
  patient_hash: ActionHash;
  encounter_hash?: ActionHash;
  recorded_at: Timestamp;
  recorded_by: AgentPubKey;
  temperature_celsius?: number;
  heart_rate_bpm?: number;
  blood_pressure_systolic?: number;
  blood_pressure_diastolic?: number;
  respiratory_rate?: number;
  oxygen_saturation?: number;
  height_cm?: number;
  weight_kg?: number;
  bmi?: number;
  pain_level?: number;
  notes?: string;
}

// Prescription Types
export interface Prescription {
  prescription_id: string;
  patient_hash: ActionHash;
  prescriber_hash: ActionHash;
  encounter_hash?: ActionHash;
  rxnorm_code: string;
  ndc_code?: string;
  medication_name: string;
  strength: string;
  form: MedicationForm;
  route: AdministrationRoute;
  dosage_instructions: string;
  quantity: number;
  quantity_unit: string;
  refills_authorized: number;
  refills_remaining: number;
  days_supply: number;
  dispense_as_written: boolean;
  status: PrescriptionStatus;
  written_date: Timestamp;
  effective_date: Timestamp;
  expiration_date: Timestamp;
  schedule?: DrugSchedule;
  dea_number?: string;
  pharmacy_hash?: ActionHash;
  notes?: string;
  indication: string;
  indication_icd10?: string;
}

export type MedicationForm =
  | 'Tablet'
  | 'Capsule'
  | 'Liquid'
  | 'Injection'
  | 'Topical'
  | 'Patch'
  | 'Inhaler'
  | 'Drops'
  | 'Suppository'
  | 'Powder'
  | { Other: string };

export type AdministrationRoute =
  | 'Oral'
  | 'Intravenous'
  | 'Intramuscular'
  | 'Subcutaneous'
  | 'Topical'
  | 'Transdermal'
  | 'Inhalation'
  | 'Ophthalmic'
  | 'Otic'
  | 'Nasal'
  | 'Rectal'
  | 'Sublingual'
  | { Other: string };

export type PrescriptionStatus =
  | 'Active'
  | 'Completed'
  | 'Discontinued'
  | 'OnHold'
  | 'Cancelled'
  | 'Expired'
  | 'EnteredInError';

export type DrugSchedule =
  | 'ScheduleI'
  | 'ScheduleII'
  | 'ScheduleIII'
  | 'ScheduleIV'
  | 'ScheduleV'
  | 'NotControlled';

// Consent Types
export interface Consent {
  consent_id: string;
  patient_hash: ActionHash;
  grantee: ConsentGrantee;
  scope: ConsentScope;
  permissions: DataPermission[];
  purpose: ConsentPurpose;
  status: ConsentStatus;
  granted_at: Timestamp;
  expires_at?: Timestamp;
  revoked_at?: Timestamp;
  revocation_reason?: string;
  document_hash?: string;
  witness?: AgentPubKey;
  legal_representative?: AgentPubKey;
  notes?: string;
}

export type ConsentGrantee =
  | { Provider: ActionHash }
  | { Organization: string }
  | { Agent: AgentPubKey }
  | { ResearchStudy: ActionHash }
  | { InsuranceCompany: ActionHash }
  | 'EmergencyAccess'
  | 'Public';

export interface ConsentScope {
  data_categories: DataCategory[];
  date_range?: { start: Timestamp; end?: Timestamp };
  encounter_hashes?: ActionHash[];
  exclusions: DataCategory[];
}

export type DataCategory =
  | 'Demographics'
  | 'Allergies'
  | 'Medications'
  | 'Diagnoses'
  | 'Procedures'
  | 'LabResults'
  | 'ImagingStudies'
  | 'VitalSigns'
  | 'Immunizations'
  | 'MentalHealth'
  | 'SubstanceAbuse'
  | 'SexualHealth'
  | 'GeneticData'
  | 'FinancialData'
  | 'All';

export type DataPermission = 'Read' | 'Write' | 'Share' | 'Export' | 'Delete' | 'Amend';

export type ConsentPurpose =
  | 'Treatment'
  | 'Payment'
  | 'HealthcareOperations'
  | 'Research'
  | 'PublicHealth'
  | 'LegalProceeding'
  | 'Marketing'
  | 'FamilyNotification'
  | { Other: string };

export type ConsentStatus = 'Active' | 'Expired' | 'Revoked' | 'Pending' | 'Rejected';

// Clinical Trials Types
export interface ClinicalTrial {
  trial_id: string;
  nct_number?: string;
  title: string;
  short_title?: string;
  description: string;
  phase: TrialPhase;
  study_type: StudyType;
  status: TrialStatus;
  principal_investigator: AgentPubKey;
  sponsor: string;
  collaborators: string[];
  target_enrollment: number;
  current_enrollment: number;
  start_date: Timestamp;
  estimated_end_date?: Timestamp;
  actual_end_date?: Timestamp;
  eligibility: EligibilityCriteria;
  interventions: Intervention[];
  primary_outcomes: Outcome[];
  secondary_outcomes: Outcome[];
  irb_approved: boolean;
  irb_approval_date?: Timestamp;
  irb_expiration_date?: Timestamp;
  desci_publication_hash?: ActionHash;
  epistemic_level: TrialEpistemicLevel;
  matl_trust_score: number;
  created_at: Timestamp;
  updated_at: Timestamp;
}

export type TrialPhase =
  | 'EarlyPhase1'
  | 'Phase1'
  | 'Phase1Phase2'
  | 'Phase2'
  | 'Phase2Phase3'
  | 'Phase3'
  | 'Phase4'
  | 'NotApplicable';

export type StudyType = 'Interventional' | 'Observational' | 'ExpandedAccess' | 'Registry';

export type TrialStatus =
  | 'NotYetRecruiting'
  | 'Recruiting'
  | 'EnrollingByInvitation'
  | 'ActiveNotRecruiting'
  | 'Suspended'
  | 'Terminated'
  | 'Completed'
  | 'Withdrawn';

export type TrialEpistemicLevel = 'E0_Preliminary' | 'E1_PeerReviewed' | 'E2_Replicated' | 'E3_Consensus';

export interface EligibilityCriteria {
  min_age?: number;
  max_age?: number;
  sex: 'All' | 'Female' | 'Male';
  healthy_volunteers: boolean;
  inclusion_criteria: string[];
  exclusion_criteria: string[];
}

export interface Intervention {
  intervention_type: InterventionType;
  name: string;
  description: string;
  arm_group?: string;
}

export type InterventionType =
  | 'Drug'
  | 'Device'
  | 'Biological'
  | 'Procedure'
  | 'Radiation'
  | 'Behavioral'
  | 'Genetic'
  | 'DietarySupplement'
  | 'Diagnostic'
  | { Other: string };

export interface Outcome {
  measure: string;
  time_frame: string;
  description: string;
}

export interface TrialParticipant {
  participant_id: string;
  trial_hash: ActionHash;
  patient_hash: ActionHash;
  consent_hash: ActionHash;
  enrollment_date: Timestamp;
  withdrawal_date?: Timestamp;
  withdrawal_reason?: string;
  arm_assignment?: string;
  status: ParticipantStatus;
  blinded: boolean;
  screening_passed: boolean;
  screening_date?: Timestamp;
  enrollment_site: string;
  primary_contact: AgentPubKey;
}

export type ParticipantStatus =
  | 'Screening'
  | 'Enrolled'
  | 'Active'
  | 'FollowUp'
  | 'Completed'
  | 'Withdrawn'
  | 'ScreenFail'
  | 'LostToFollowUp';

export interface AdverseEvent {
  event_id: string;
  participant_hash: ActionHash;
  trial_hash: ActionHash;
  event_term: string;
  description: string;
  onset_date: Timestamp;
  resolution_date?: Timestamp;
  ongoing: boolean;
  severity: AESeverity;
  seriousness: SeriousnessCriteria[];
  is_serious: boolean;
  is_unexpected: boolean;
  causality: Causality;
  outcome: AEOutcome;
  action_taken: ActionTaken[];
  reported_by: AgentPubKey;
  reported_at: Timestamp;
  medwatch_submitted: boolean;
  medwatch_date?: Timestamp;
}

export type AESeverity = 'Mild' | 'Moderate' | 'Severe' | 'LifeThreatening' | 'Death';

export type SeriousnessCriteria =
  | 'Death'
  | 'LifeThreatening'
  | 'Hospitalization'
  | 'Disability'
  | 'CongenitalAnomaly'
  | 'ImportantMedicalEvent';

export type Causality =
  | 'DefinitelyRelated'
  | 'ProbablyRelated'
  | 'PossiblyRelated'
  | 'UnlikelyRelated'
  | 'NotRelated'
  | 'Unknown';

export type AEOutcome =
  | 'Recovered'
  | 'Recovering'
  | 'NotRecovered'
  | 'RecoveredWithSequelae'
  | 'Fatal'
  | 'Unknown';

export type ActionTaken =
  | 'NoneRequired'
  | 'StudyDrugInterrupted'
  | 'StudyDrugReduced'
  | 'StudyDrugDiscontinued'
  | 'SubjectWithdrawn'
  | 'MedicationGiven'
  | 'ProcedurePerformed'
  | 'Hospitalized';

// =============================================================================
// Health Service
// =============================================================================

/**
 * Mycelix Health Service
 *
 * Provides methods for interacting with the Mycelix-Health hApp,
 * including patient management, medical records, prescriptions,
 * consent, clinical trials, and insurance.
 */
export class HealthService {
  constructor(
    private client: MycelixClient,
    private cellId: string = 'health'
  ) {}

  // ---------------------------------------------------------------------------
  // Patient Management
  // ---------------------------------------------------------------------------

  /** Create a new patient profile */
  async createPatient(patient: Patient): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'patient', fn_name: 'create_patient', payload: patient });
  }

  /** Get a patient by action hash */
  async getPatient(patientHash: ActionHash): Promise<Patient | null> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'patient', fn_name: 'get_patient', payload: patientHash });
  }

  /** Update a patient profile */
  async updatePatient(originalHash: ActionHash, updatedPatient: Patient): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'patient', fn_name: 'update_patient', payload: {
      original_hash: originalHash,
      updated_patient: updatedPatient,
    } });
  }

  /** Search patients by name */
  async searchPatientsByName(name: string): Promise<Patient[]> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'patient', fn_name: 'search_patients_by_name', payload: name });
  }

  /** Get patient by MRN */
  async getPatientByMrn(mrn: string): Promise<Patient | null> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'patient', fn_name: 'get_patient_by_mrn', payload: mrn });
  }

  /** Add allergy to patient */
  async addPatientAllergy(patientHash: ActionHash, allergy: Allergy): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'patient', fn_name: 'add_patient_allergy', payload: {
      patient_hash: patientHash,
      allergy,
    } });
  }

  // ---------------------------------------------------------------------------
  // Provider Management
  // ---------------------------------------------------------------------------

  /** Create a new provider profile */
  async createProvider(provider: Provider): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'provider', fn_name: 'create_provider', payload: provider });
  }

  /** Get a provider by action hash */
  async getProvider(providerHash: ActionHash): Promise<Provider | null> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'provider', fn_name: 'get_provider', payload: providerHash });
  }

  /** Search providers by specialty */
  async searchProvidersBySpecialty(specialty: string): Promise<Provider[]> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'provider', fn_name: 'search_providers_by_specialty', payload: specialty });
  }

  /** Get provider by NPI */
  async getProviderByNpi(npi: string): Promise<Provider | null> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'provider', fn_name: 'get_provider_by_npi', payload: npi });
  }

  /** Add license to provider */
  async addLicense(license: License): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'provider', fn_name: 'add_license', payload: license });
  }

  /** Verify provider credentials */
  async verifyProviderCredentials(providerHash: ActionHash): Promise<{
    provider_name: string;
    specialty: string;
    has_active_license: boolean;
    has_board_certification: boolean;
    matl_trust_score: number;
  }> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'provider', fn_name: 'verify_provider_credentials', payload: providerHash });
  }

  // ---------------------------------------------------------------------------
  // Medical Records
  // ---------------------------------------------------------------------------

  /** Create an encounter */
  async createEncounter(encounter: Encounter): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'records', fn_name: 'create_encounter', payload: encounter });
  }

  /** Get patient's encounters */
  async getPatientEncounters(patientHash: ActionHash): Promise<Encounter[]> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'records', fn_name: 'get_patient_encounters', payload: patientHash });
  }

  /** Create a diagnosis */
  async createDiagnosis(diagnosis: Diagnosis): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'records', fn_name: 'create_diagnosis', payload: diagnosis });
  }

  /** Create a lab result */
  async createLabResult(labResult: LabResult): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'records', fn_name: 'create_lab_result', payload: labResult });
  }

  /** Get patient's lab results */
  async getPatientLabResults(patientHash: ActionHash): Promise<LabResult[]> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'records', fn_name: 'get_patient_lab_results', payload: patientHash });
  }

  /** Acknowledge critical result */
  async acknowledgeCriticalResult(resultHash: ActionHash): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'records', fn_name: 'acknowledge_critical_result', payload: {
      result_hash: resultHash,
    } });
  }

  /** Record vital signs */
  async recordVitalSigns(vitals: VitalSigns): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'records', fn_name: 'record_vital_signs', payload: vitals });
  }

  /** Get critical results */
  async getCriticalResults(): Promise<LabResult[]> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'records', fn_name: 'get_critical_results', payload: null });
  }

  // ---------------------------------------------------------------------------
  // Prescriptions
  // ---------------------------------------------------------------------------

  /** Create a prescription */
  async createPrescription(prescription: Prescription): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'prescriptions', fn_name: 'create_prescription', payload: prescription });
  }

  /** Get patient's prescriptions */
  async getPatientPrescriptions(patientHash: ActionHash): Promise<Prescription[]> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'prescriptions', fn_name: 'get_patient_prescriptions', payload: patientHash });
  }

  /** Get active prescriptions */
  async getActivePrescriptions(patientHash: ActionHash): Promise<Prescription[]> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'prescriptions', fn_name: 'get_active_prescriptions', payload: patientHash });
  }

  /** Fill a prescription */
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
    return this.client.callZome({ role_name: this.cellId, zome_name: 'prescriptions', fn_name: 'fill_prescription', payload: fill });
  }

  /** Discontinue a prescription */
  async discontinuePrescription(rxHash: ActionHash, reason: string): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'prescriptions', fn_name: 'discontinue_prescription', payload: {
      rx_hash: rxHash,
      reason,
    } });
  }

  // ---------------------------------------------------------------------------
  // Consent Management
  // ---------------------------------------------------------------------------

  /** Create a consent directive */
  async createConsent(consent: Consent): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'consent', fn_name: 'create_consent', payload: consent });
  }

  /** Get patient's consents */
  async getPatientConsents(patientHash: ActionHash): Promise<Consent[]> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'consent', fn_name: 'get_patient_consents', payload: patientHash });
  }

  /** Get active consents */
  async getActiveConsents(patientHash: ActionHash): Promise<Consent[]> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'consent', fn_name: 'get_active_consents', payload: patientHash });
  }

  /** Revoke consent */
  async revokeConsent(consentHash: ActionHash, reason: string): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'consent', fn_name: 'revoke_consent', payload: {
      consent_hash: consentHash,
      reason,
    } });
  }

  /** Check authorization */
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
    return this.client.callZome({ role_name: this.cellId, zome_name: 'consent', fn_name: 'check_authorization', payload: input });
  }

  /** Get access logs */
  async getAccessLogs(patientHash: ActionHash): Promise<
    Array<{
      log_id: string;
      accessor: AgentPubKey;
      access_type: DataPermission;
      data_categories_accessed: DataCategory[];
      accessed_at: Timestamp;
      emergency_override: boolean;
    }>
  > {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'consent', fn_name: 'get_access_logs', payload: patientHash });
  }

  // ---------------------------------------------------------------------------
  // Clinical Trials
  // ---------------------------------------------------------------------------

  /** Create a clinical trial */
  async createTrial(trial: ClinicalTrial): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'trials', fn_name: 'create_trial', payload: trial });
  }

  /** Get a trial */
  async getTrial(trialHash: ActionHash): Promise<ClinicalTrial | null> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'trials', fn_name: 'get_trial', payload: trialHash });
  }

  /** Get recruiting trials */
  async getRecruitingTrials(): Promise<ClinicalTrial[]> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'trials', fn_name: 'get_recruiting_trials', payload: null });
  }

  /** Check eligibility for a trial */
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
    return this.client.callZome({ role_name: this.cellId, zome_name: 'trials', fn_name: 'check_eligibility', payload: {
      trial_hash: trialHash,
      patient_hash: patientHash,
      patient_age: patientAge,
    } });
  }

  /** Enroll participant in trial */
  async enrollParticipant(participant: TrialParticipant): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'trials', fn_name: 'enroll_participant', payload: participant });
  }

  /** Withdraw participant from trial */
  async withdrawParticipant(participantHash: ActionHash, reason: string): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'trials', fn_name: 'withdraw_participant', payload: {
      participant_hash: participantHash,
      reason,
    } });
  }

  /** Report adverse event */
  async reportAdverseEvent(event: AdverseEvent): Promise<ActionHash> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'trials', fn_name: 'report_adverse_event', payload: event });
  }

  /** Get serious adverse events for a trial */
  async getSeriousAdverseEvents(trialHash: ActionHash): Promise<AdverseEvent[]> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'trials', fn_name: 'get_serious_adverse_events', payload: trialHash });
  }

  // ---------------------------------------------------------------------------
  // Insurance (Abbreviated for brevity)
  // ---------------------------------------------------------------------------

  /** Register insurance plan */
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
    return this.client.callZome({ role_name: this.cellId, zome_name: 'insurance', fn_name: 'register_insurance_plan', payload: plan });
  }

  /** Get patient's insurance */
  async getPatientInsurance(patientHash: ActionHash): Promise<unknown[]> {
    return this.client.callZome({ role_name: this.cellId, zome_name: 'insurance', fn_name: 'get_patient_insurance', payload: patientHash });
  }

  /** Submit a claim */
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
    return this.client.callZome({ role_name: this.cellId, zome_name: 'insurance', fn_name: 'submit_claim', payload: claim });
  }
}

// =============================================================================
// Unified Client
// =============================================================================

export {
  MycelixHealthClient,
  HealthSdkError,
  PatientClient,
  ProviderClient,
  RecordsClient,
  PrescriptionsClient,
  ConsentClient,
  TrialsClient,
  InsuranceClient,
} from './client';
export type { HealthClientConfig, HealthConnectionOptions } from './client';

// =============================================================================
// Export
// =============================================================================

export default HealthService;
