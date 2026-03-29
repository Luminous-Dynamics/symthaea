// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Health FHIR R4 Export Module
 *
 * Provides FHIR R4 (Fast Healthcare Interoperability Resources) export
 * capabilities for Mycelix health data, enabling interoperability with
 * external EHR systems, health information exchanges, and SMART on FHIR apps.
 *
 * Supported FHIR Resources:
 * - Patient
 * - Observation (vitals, labs)
 * - Condition
 * - MedicationStatement
 * - AllergyIntolerance
 * - Immunization
 * - Procedure
 * - DiagnosticReport
 * - DocumentReference
 * - Consent
 *
 * @module @mycelix/sdk/integrations/health-fhir
 */

import type { ActionHash } from '@holochain/client';

// ============================================================================
// FHIR R4 Resource Types
// ============================================================================

/**
 * Base FHIR Resource interface
 */
export interface FHIRResource {
  resourceType: string;
  id: string;
  meta?: {
    versionId?: string;
    lastUpdated?: string;
    source?: string;
    profile?: string[];
    security?: FHIRCoding[];
    tag?: FHIRCoding[];
  };
}

/**
 * FHIR Coding
 */
export interface FHIRCoding {
  system?: string;
  version?: string;
  code?: string;
  display?: string;
  userSelected?: boolean;
}

/**
 * FHIR CodeableConcept
 */
export interface FHIRCodeableConcept {
  coding?: FHIRCoding[];
  text?: string;
}

/**
 * FHIR Identifier
 */
export interface FHIRIdentifier {
  use?: 'usual' | 'official' | 'temp' | 'secondary' | 'old';
  type?: FHIRCodeableConcept;
  system?: string;
  value?: string;
  period?: { start?: string; end?: string };
}

/**
 * FHIR HumanName
 */
export interface FHIRHumanName {
  use?: 'usual' | 'official' | 'temp' | 'nickname' | 'anonymous' | 'old' | 'maiden';
  text?: string;
  family?: string;
  given?: string[];
  prefix?: string[];
  suffix?: string[];
  period?: { start?: string; end?: string };
}

/**
 * FHIR Address
 */
export interface FHIRAddress {
  use?: 'home' | 'work' | 'temp' | 'old' | 'billing';
  type?: 'postal' | 'physical' | 'both';
  text?: string;
  line?: string[];
  city?: string;
  district?: string;
  state?: string;
  postalCode?: string;
  country?: string;
  period?: { start?: string; end?: string };
}

/**
 * FHIR ContactPoint (phone, email, etc.)
 */
export interface FHIRContactPoint {
  system?: 'phone' | 'fax' | 'email' | 'pager' | 'url' | 'sms' | 'other';
  value?: string;
  use?: 'home' | 'work' | 'temp' | 'old' | 'mobile';
  rank?: number;
  period?: { start?: string; end?: string };
}

/**
 * FHIR Reference
 */
export interface FHIRReference {
  reference?: string;
  type?: string;
  identifier?: FHIRIdentifier;
  display?: string;
}

/**
 * FHIR Quantity
 */
export interface FHIRQuantity {
  value?: number;
  comparator?: '<' | '<=' | '>=' | '>';
  unit?: string;
  system?: string;
  code?: string;
}

/**
 * FHIR Period
 */
export interface FHIRPeriod {
  start?: string;
  end?: string;
}

/**
 * FHIR Attachment
 */
export interface FHIRAttachment {
  contentType?: string;
  language?: string;
  data?: string; // Base64
  url?: string;
  size?: number;
  hash?: string;
  title?: string;
  creation?: string;
}

// ============================================================================
// FHIR Patient Resource
// ============================================================================

/**
 * FHIR R4 Patient Resource
 */
export interface FHIRPatient extends FHIRResource {
  resourceType: 'Patient';
  identifier?: FHIRIdentifier[];
  active?: boolean;
  name?: FHIRHumanName[];
  telecom?: FHIRContactPoint[];
  gender?: 'male' | 'female' | 'other' | 'unknown';
  birthDate?: string;
  deceasedBoolean?: boolean;
  deceasedDateTime?: string;
  address?: FHIRAddress[];
  maritalStatus?: FHIRCodeableConcept;
  multipleBirthBoolean?: boolean;
  multipleBirthInteger?: number;
  photo?: FHIRAttachment[];
  contact?: Array<{
    relationship?: FHIRCodeableConcept[];
    name?: FHIRHumanName;
    telecom?: FHIRContactPoint[];
    address?: FHIRAddress;
    gender?: 'male' | 'female' | 'other' | 'unknown';
    organization?: FHIRReference;
    period?: FHIRPeriod;
  }>;
  communication?: Array<{
    language: FHIRCodeableConcept;
    preferred?: boolean;
  }>;
  generalPractitioner?: FHIRReference[];
  managingOrganization?: FHIRReference;
  link?: Array<{
    other: FHIRReference;
    type: 'replaced-by' | 'replaces' | 'refer' | 'seealso';
  }>;
}

// ============================================================================
// FHIR Observation Resource
// ============================================================================

/**
 * FHIR R4 Observation Resource (vitals, labs)
 */
export interface FHIRObservation extends FHIRResource {
  resourceType: 'Observation';
  identifier?: FHIRIdentifier[];
  status: 'registered' | 'preliminary' | 'final' | 'amended' | 'corrected' | 'cancelled' | 'entered-in-error' | 'unknown';
  category?: FHIRCodeableConcept[];
  code: FHIRCodeableConcept;
  subject?: FHIRReference;
  encounter?: FHIRReference;
  effectiveDateTime?: string;
  effectivePeriod?: FHIRPeriod;
  issued?: string;
  performer?: FHIRReference[];
  valueQuantity?: FHIRQuantity;
  valueCodeableConcept?: FHIRCodeableConcept;
  valueString?: string;
  valueBoolean?: boolean;
  valueInteger?: number;
  valueRange?: { low?: FHIRQuantity; high?: FHIRQuantity };
  dataAbsentReason?: FHIRCodeableConcept;
  interpretation?: FHIRCodeableConcept[];
  note?: Array<{ text: string }>;
  bodySite?: FHIRCodeableConcept;
  method?: FHIRCodeableConcept;
  specimen?: FHIRReference;
  device?: FHIRReference;
  referenceRange?: Array<{
    low?: FHIRQuantity;
    high?: FHIRQuantity;
    type?: FHIRCodeableConcept;
    appliesTo?: FHIRCodeableConcept[];
    age?: { low?: FHIRQuantity; high?: FHIRQuantity };
    text?: string;
  }>;
  component?: Array<{
    code: FHIRCodeableConcept;
    valueQuantity?: FHIRQuantity;
    valueCodeableConcept?: FHIRCodeableConcept;
    valueString?: string;
    dataAbsentReason?: FHIRCodeableConcept;
    interpretation?: FHIRCodeableConcept[];
    referenceRange?: FHIRObservation['referenceRange'];
  }>;
}

// ============================================================================
// FHIR Condition Resource
// ============================================================================

/**
 * FHIR R4 Condition Resource
 */
export interface FHIRCondition extends FHIRResource {
  resourceType: 'Condition';
  identifier?: FHIRIdentifier[];
  clinicalStatus?: FHIRCodeableConcept;
  verificationStatus?: FHIRCodeableConcept;
  category?: FHIRCodeableConcept[];
  severity?: FHIRCodeableConcept;
  code?: FHIRCodeableConcept;
  bodySite?: FHIRCodeableConcept[];
  subject: FHIRReference;
  encounter?: FHIRReference;
  onsetDateTime?: string;
  onsetAge?: FHIRQuantity;
  onsetPeriod?: FHIRPeriod;
  onsetRange?: { low?: FHIRQuantity; high?: FHIRQuantity };
  onsetString?: string;
  abatementDateTime?: string;
  abatementAge?: FHIRQuantity;
  abatementPeriod?: FHIRPeriod;
  abatementRange?: { low?: FHIRQuantity; high?: FHIRQuantity };
  abatementString?: string;
  recordedDate?: string;
  recorder?: FHIRReference;
  asserter?: FHIRReference;
  stage?: Array<{
    summary?: FHIRCodeableConcept;
    assessment?: FHIRReference[];
    type?: FHIRCodeableConcept;
  }>;
  evidence?: Array<{
    code?: FHIRCodeableConcept[];
    detail?: FHIRReference[];
  }>;
  note?: Array<{ text: string }>;
}

// ============================================================================
// FHIR AllergyIntolerance Resource
// ============================================================================

/**
 * FHIR R4 AllergyIntolerance Resource
 */
export interface FHIRAllergyIntolerance extends FHIRResource {
  resourceType: 'AllergyIntolerance';
  identifier?: FHIRIdentifier[];
  clinicalStatus?: FHIRCodeableConcept;
  verificationStatus?: FHIRCodeableConcept;
  type?: 'allergy' | 'intolerance';
  category?: Array<'food' | 'medication' | 'environment' | 'biologic'>;
  criticality?: 'low' | 'high' | 'unable-to-assess';
  code?: FHIRCodeableConcept;
  patient: FHIRReference;
  encounter?: FHIRReference;
  onsetDateTime?: string;
  onsetAge?: FHIRQuantity;
  onsetPeriod?: FHIRPeriod;
  onsetRange?: { low?: FHIRQuantity; high?: FHIRQuantity };
  onsetString?: string;
  recordedDate?: string;
  recorder?: FHIRReference;
  asserter?: FHIRReference;
  lastOccurrence?: string;
  note?: Array<{ text: string }>;
  reaction?: Array<{
    substance?: FHIRCodeableConcept;
    manifestation: FHIRCodeableConcept[];
    description?: string;
    onset?: string;
    severity?: 'mild' | 'moderate' | 'severe';
    exposureRoute?: FHIRCodeableConcept;
    note?: Array<{ text: string }>;
  }>;
}

// ============================================================================
// FHIR MedicationStatement Resource
// ============================================================================

/**
 * FHIR R4 MedicationStatement Resource
 */
export interface FHIRMedicationStatement extends FHIRResource {
  resourceType: 'MedicationStatement';
  identifier?: FHIRIdentifier[];
  basedOn?: FHIRReference[];
  partOf?: FHIRReference[];
  status: 'active' | 'completed' | 'entered-in-error' | 'intended' | 'stopped' | 'on-hold' | 'unknown' | 'not-taken';
  statusReason?: FHIRCodeableConcept[];
  category?: FHIRCodeableConcept;
  medicationCodeableConcept?: FHIRCodeableConcept;
  medicationReference?: FHIRReference;
  subject: FHIRReference;
  context?: FHIRReference;
  effectiveDateTime?: string;
  effectivePeriod?: FHIRPeriod;
  dateAsserted?: string;
  informationSource?: FHIRReference;
  derivedFrom?: FHIRReference[];
  reasonCode?: FHIRCodeableConcept[];
  reasonReference?: FHIRReference[];
  note?: Array<{ text: string }>;
  dosage?: Array<{
    sequence?: number;
    text?: string;
    additionalInstruction?: FHIRCodeableConcept[];
    patientInstruction?: string;
    timing?: {
      repeat?: {
        frequency?: number;
        period?: number;
        periodUnit?: 's' | 'min' | 'h' | 'd' | 'wk' | 'mo' | 'a';
      };
      code?: FHIRCodeableConcept;
    };
    asNeededBoolean?: boolean;
    asNeededCodeableConcept?: FHIRCodeableConcept;
    site?: FHIRCodeableConcept;
    route?: FHIRCodeableConcept;
    method?: FHIRCodeableConcept;
    doseAndRate?: Array<{
      type?: FHIRCodeableConcept;
      doseQuantity?: FHIRQuantity;
      doseRange?: { low?: FHIRQuantity; high?: FHIRQuantity };
      rateQuantity?: FHIRQuantity;
      rateRange?: { low?: FHIRQuantity; high?: FHIRQuantity };
    }>;
    maxDosePerPeriod?: { numerator?: FHIRQuantity; denominator?: FHIRQuantity };
    maxDosePerAdministration?: FHIRQuantity;
    maxDosePerLifetime?: FHIRQuantity;
  }>;
}

// ============================================================================
// FHIR Consent Resource
// ============================================================================

/**
 * FHIR R4 Consent Resource
 */
export interface FHIRConsent extends FHIRResource {
  resourceType: 'Consent';
  identifier?: FHIRIdentifier[];
  status: 'draft' | 'proposed' | 'active' | 'rejected' | 'inactive' | 'entered-in-error';
  scope: FHIRCodeableConcept;
  category: FHIRCodeableConcept[];
  patient?: FHIRReference;
  dateTime?: string;
  performer?: FHIRReference[];
  organization?: FHIRReference[];
  sourceAttachment?: FHIRAttachment;
  sourceReference?: FHIRReference;
  policy?: Array<{
    authority?: string;
    uri?: string;
  }>;
  policyRule?: FHIRCodeableConcept;
  verification?: Array<{
    verified: boolean;
    verifiedWith?: FHIRReference;
    verificationDate?: string;
  }>;
  provision?: FHIRConsentProvision;
}

/**
 * FHIR Consent Provision
 */
export interface FHIRConsentProvision {
  type?: 'deny' | 'permit';
  period?: FHIRPeriod;
  actor?: Array<{
    role: FHIRCodeableConcept;
    reference: FHIRReference;
  }>;
  action?: FHIRCodeableConcept[];
  securityLabel?: FHIRCoding[];
  purpose?: FHIRCoding[];
  class?: FHIRCoding[];
  code?: FHIRCodeableConcept[];
  dataPeriod?: FHIRPeriod;
  data?: Array<{
    meaning: 'instance' | 'related' | 'dependents' | 'authoredby';
    reference: FHIRReference;
  }>;
  provision?: FHIRConsentProvision[];
}

// ============================================================================
// FHIR Bundle Resource
// ============================================================================

/**
 * FHIR R4 Bundle Resource (for exporting multiple resources)
 */
export interface FHIRBundle extends FHIRResource {
  resourceType: 'Bundle';
  type: 'document' | 'message' | 'transaction' | 'transaction-response' | 'batch' | 'batch-response' | 'history' | 'searchset' | 'collection';
  timestamp?: string;
  total?: number;
  link?: Array<{
    relation: string;
    url: string;
  }>;
  entry?: Array<{
    fullUrl?: string;
    resource?: FHIRResource;
    search?: {
      mode?: 'match' | 'include' | 'outcome';
      score?: number;
    };
    request?: {
      method: 'GET' | 'HEAD' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
      url: string;
      ifNoneMatch?: string;
      ifModifiedSince?: string;
      ifMatch?: string;
      ifNoneExist?: string;
    };
    response?: {
      status: string;
      location?: string;
      etag?: string;
      lastModified?: string;
      outcome?: FHIRResource;
    };
  }>;
  signature?: {
    type: FHIRCoding[];
    when: string;
    who: FHIRReference;
    onBehalfOf?: FHIRReference;
    targetFormat?: string;
    sigFormat?: string;
    data?: string; // Base64
  };
}

// ============================================================================
// FHIR Export Client
// ============================================================================

export interface ZomeCallable {
  callZome(args: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<unknown>;
}

const HEALTH_ROLE = 'health';

/**
 * Client for FHIR R4 export operations
 */
export class FHIRExportClient {
  constructor(public readonly client: ZomeCallable) {}

  /**
   * Export patient demographics as FHIR Patient resource
   */
  async exportPatient(patientHash: ActionHash): Promise<FHIRPatient> {
    const data = await this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'patient',
      fn_name: 'get_patient',
      payload: { patientHash },
    }) as Record<string, unknown>;

    // Transform Mycelix patient data to FHIR Patient
    return {
      resourceType: 'Patient',
      id: data.patientId as string || 'patient-1',
      name: [{
        family: data.lastName as string,
        given: [data.firstName as string],
      }],
      gender: (data.gender as string)?.toLowerCase() as FHIRPatient['gender'],
      birthDate: data.dateOfBirth as string,
      telecom: [
        ...(data.phone ? [{ system: 'phone' as const, value: data.phone as string }] : []),
        ...(data.email ? [{ system: 'email' as const, value: data.email as string }] : []),
      ],
      address: data.address ? [{
        line: [(data.address as Record<string, string>).street],
        city: (data.address as Record<string, string>).city,
        state: (data.address as Record<string, string>).state,
        postalCode: (data.address as Record<string, string>).zip,
      }] : undefined,
    };
  }

  /**
   * Export patient vitals as FHIR Observations
   */
  async exportVitals(patientHash: ActionHash): Promise<FHIRObservation[]> {
    const vitals = await this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'vitals',
      fn_name: 'get_vitals',
      payload: { patientHash },
    }) as Record<string, unknown>;

    const observations: FHIRObservation[] = [];
    const timestamp = vitals.recordedAt as number || Date.now();

    if (vitals.bloodPressureSystolic || vitals.bloodPressureDiastolic) {
      observations.push({
        resourceType: 'Observation',
        id: `bp-${timestamp}`,
        status: 'final',
        code: { coding: [createVitalsCoding('blood-pressure')] },
        effectiveDateTime: toFHIRDateTime(timestamp),
        component: [
          ...(vitals.bloodPressureSystolic ? [{
            code: { coding: [createVitalsCoding('blood-pressure-systolic')] },
            valueQuantity: { value: vitals.bloodPressureSystolic as number, unit: 'mmHg' },
          }] : []),
          ...(vitals.bloodPressureDiastolic ? [{
            code: { coding: [createVitalsCoding('blood-pressure-diastolic')] },
            valueQuantity: { value: vitals.bloodPressureDiastolic as number, unit: 'mmHg' },
          }] : []),
        ],
      });
    }

    if (vitals.heartRate) {
      observations.push({
        resourceType: 'Observation',
        id: `hr-${timestamp}`,
        status: 'final',
        code: { coding: [createVitalsCoding('heart-rate')] },
        effectiveDateTime: toFHIRDateTime(timestamp),
        valueQuantity: { value: vitals.heartRate as number, unit: 'beats/minute' },
      });
    }

    if (vitals.temperature) {
      observations.push({
        resourceType: 'Observation',
        id: `temp-${timestamp}`,
        status: 'final',
        code: { coding: [createVitalsCoding('body-temperature')] },
        effectiveDateTime: toFHIRDateTime(timestamp),
        valueQuantity: { value: vitals.temperature as number, unit: '°F' },
      });
    }

    return observations;
  }

  /**
   * Export patient observations (vitals, labs)
   */
  async exportObservations(
    patientHash: ActionHash,
    options?: {
      category?: 'vital-signs' | 'laboratory' | 'social-history' | 'survey';
      startDate?: string;
      endDate?: string;
      codes?: string[]; // LOINC codes
    }
  ): Promise<FHIRObservation[]> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'fhir_export',
      fn_name: 'export_observations',
      payload: { patient_hash: patientHash, options },
    }) as Promise<FHIRObservation[]>;
  }

  /**
   * Export patient conditions/diagnoses
   */
  async exportConditions(
    patientHash: ActionHash,
    options?: {
      clinicalStatus?: string;
      category?: string;
      onlyActive?: boolean;
    }
  ): Promise<FHIRCondition[]> {
    const data = await this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'records',
      fn_name: 'get_diagnoses',
      payload: { patient_hash: patientHash, options },
    }) as Array<Record<string, unknown>>;

    return data.map((dx) => ({
      resourceType: 'Condition' as const,
      id: dx.diagnosisId as string,
      clinicalStatus: {
        coding: [{ code: (dx.status as string)?.toLowerCase() || 'active' }],
      },
      code: {
        coding: [createICD10Coding(dx.icd10Code as string, dx.description as string)],
      },
      subject: { reference: `Patient/${patientHash}` },
      onsetDateTime: dx.onsetDate as string,
    }));
  }

  /**
   * Export patient allergies
   */
  async exportAllergies(patientHash: ActionHash): Promise<FHIRAllergyIntolerance[]> {
    const data = await this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'records',
      fn_name: 'get_allergies',
      payload: patientHash,
    }) as Array<Record<string, unknown>>;

    return data.map((allergy) => {
      const severityMap: Record<string, 'low' | 'high'> = {
        'mild': 'low',
        'moderate': 'low',
        'severe': 'high',
        'lifethreatening': 'high',
      };
      const severity = (allergy.severity as string)?.toLowerCase()?.replace(/\s/g, '');
      return {
        resourceType: 'AllergyIntolerance' as const,
        id: allergy.allergyId as string,
        clinicalStatus: { coding: [{ code: 'active' }] },
        criticality: severityMap[severity] || 'high',
        code: { text: allergy.allergen as string },
        patient: { reference: `Patient/${patientHash}` },
        recordedDate: allergy.recordedDate as string,
        reaction: allergy.reaction ? [{
          manifestation: [{ coding: [{ display: allergy.reaction as string }] }],
        }] : undefined,
      };
    });
  }

  /**
   * Export patient medications
   */
  async exportMedications(
    patientHash: ActionHash,
    options?: {
      status?: FHIRMedicationStatement['status'];
      activeOnly?: boolean;
    }
  ): Promise<FHIRMedicationStatement[]> {
    const data = await this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'records',
      fn_name: 'get_medications',
      payload: { patient_hash: patientHash, options },
    }) as Array<Record<string, unknown>>;

    return data.map((med) => ({
      resourceType: 'MedicationStatement' as const,
      id: med.prescriptionId as string,
      status: (med.status as string)?.toLowerCase() as FHIRMedicationStatement['status'] || 'active',
      medicationCodeableConcept: {
        coding: med.rxNormCode ? [createRxNormCoding(med.rxNormCode as string, med.medicationName as string)] : undefined,
        text: med.medicationName as string,
      },
      subject: { reference: `Patient/${patientHash}` },
      effectiveDateTime: med.startDate as string,
      dosage: [{
        text: `${med.dosage} ${med.frequency || ''}`.trim(),
      }],
    }));
  }

  /**
   * Export patient consents
   */
  async exportConsents(patientHash: ActionHash): Promise<FHIRConsent[]> {
    const data = await this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'records',
      fn_name: 'get_consents',
      payload: patientHash,
    }) as Array<Record<string, unknown>>;

    return data.map((consent) => ({
      resourceType: 'Consent' as const,
      id: consent.consentId as string,
      status: (consent.status as string)?.toLowerCase() as FHIRConsent['status'] || 'active',
      scope: { text: consent.scope as string },
      category: [{ text: consent.scope as string }],
      patient: { reference: `Patient/${patientHash}` },
      dateTime: consent.grantedDate as string,
      provision: {
        period: {
          start: consent.grantedDate as string,
          end: consent.expirationDate as string,
        },
      },
    }));
  }

  /**
   * Export complete patient record as FHIR Bundle
   */
  async exportBundle(
    patientHash: ActionHash,
    options?: {
      includePatient?: boolean;
      includeObservations?: boolean;
      includeConditions?: boolean;
      includeAllergies?: boolean;
      includeMedications?: boolean;
      includeConsents?: boolean;
      startDate?: string;
      endDate?: string;
    }
  ): Promise<FHIRBundle> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'fhir_export',
      fn_name: 'export_bundle',
      payload: { patient_hash: patientHash, options: options || { includePatient: true } },
    }) as Promise<FHIRBundle>;
  }

  /**
   * Export complete patient bundle with all resources
   */
  async exportPatientBundle(patientHash: ActionHash): Promise<FHIRBundle> {
    const entries: Array<{ resource: FHIRResource }> = [];

    // Export patient
    const patient = await this.exportPatient(patientHash);
    entries.push({ resource: patient });

    // Export vitals
    const vitals = await this.exportVitals(patientHash);
    for (const vital of vitals) {
      entries.push({ resource: vital });
    }

    // Export conditions
    const conditions = await this.exportConditions(patientHash);
    for (const condition of conditions) {
      entries.push({ resource: condition });
    }

    // Export allergies
    const allergies = await this.exportAllergies(patientHash);
    for (const allergy of allergies) {
      entries.push({ resource: allergy });
    }

    // Export medications
    const medications = await this.exportMedications(patientHash);
    for (const med of medications) {
      entries.push({ resource: med });
    }

    // Export consents
    const consents = await this.exportConsents(patientHash);
    for (const consent of consents) {
      entries.push({ resource: consent });
    }

    return {
      resourceType: 'Bundle',
      id: `bundle-${Date.now()}`,
      type: 'collection',
      timestamp: new Date().toISOString(),
      entry: entries,
    };
  }

  /**
   * Export as C-CDA document (Consolidated Clinical Document Architecture)
   */
  async exportCCDA(
    patientHash: ActionHash,
    documentType: 'CCD' | 'CCR' | 'Discharge' | 'Referral' | 'TransitionOfCare'
  ): Promise<string> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'fhir_export',
      fn_name: 'export_ccda',
      payload: { patient_hash: patientHash, document_type: documentType },
    }) as Promise<string>;
  }

  /**
   * Validate FHIR resource against profiles
   */
  async validateResource(
    resource: FHIRResource,
    profiles?: string[]
  ): Promise<{
    valid: boolean;
    errors: Array<{ path: string; message: string; severity: 'error' | 'warning' | 'info' }>;
  }> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'fhir_export',
      fn_name: 'validate_resource',
      payload: { resource, profiles },
    }) as Promise<{
      valid: boolean;
      errors: Array<{ path: string; message: string; severity: 'error' | 'warning' | 'info' }>;
    }>;
  }
}

// ============================================================================
// FHIR Import Client
// ============================================================================

/**
 * Client for FHIR R4 import operations
 */
export class FHIRImportClient {
  constructor(public readonly client: ZomeCallable) {}

  /**
   * Import a FHIR Patient
   */
  async importPatient(patient: FHIRPatient): Promise<{ patientHash: ActionHash; imported: boolean }> {
    const name = patient.name?.[0];
    const phone = patient.telecom?.find(t => t.system === 'phone')?.value;
    const email = patient.telecom?.find(t => t.system === 'email')?.value;
    const address = patient.address?.[0];

    const payload = {
      firstName: name?.given?.[0],
      lastName: name?.family,
      gender: patient.gender ? patient.gender.charAt(0).toUpperCase() + patient.gender.slice(1) : undefined,
      dateOfBirth: patient.birthDate,
      phone,
      email,
      address: address ? {
        street: address.line?.[0],
        city: address.city,
        state: address.state,
        zip: address.postalCode,
      } : undefined,
    };

    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'patient',
      fn_name: 'create_patient_from_fhir',
      payload,
    }) as Promise<{ patientHash: ActionHash; imported: boolean }>;
  }

  /**
   * Import a FHIR Observation (vitals)
   */
  async importObservation(
    observation: FHIRObservation,
    patientHash: ActionHash
  ): Promise<{ vitalRecordHash: ActionHash; imported: boolean }> {
    const code = observation.code?.coding?.[0]?.code;
    const value = observation.valueQuantity?.value;

    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'vitals',
      fn_name: 'create_vital_from_fhir',
      payload: {
        patientHash,
        loincCode: code,
        value,
        unit: observation.valueQuantity?.unit,
        effectiveDateTime: observation.effectiveDateTime,
      },
    }) as Promise<{ vitalRecordHash: ActionHash; imported: boolean }>;
  }

  /**
   * Import a FHIR Condition (diagnosis)
   */
  async importCondition(
    condition: FHIRCondition,
    patientHash: ActionHash
  ): Promise<{ diagnosisHash: ActionHash; imported: boolean }> {
    const coding = condition.code?.coding?.[0];

    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'records',
      fn_name: 'create_diagnosis_from_fhir',
      payload: {
        patientHash,
        icd10Code: coding?.code,
        description: coding?.display || condition.code?.text,
        status: condition.clinicalStatus?.coding?.[0]?.code,
        onsetDate: condition.onsetDateTime,
      },
    }) as Promise<{ diagnosisHash: ActionHash; imported: boolean }>;
  }

  /**
   * Import a FHIR AllergyIntolerance
   */
  async importAllergyIntolerance(
    allergy: FHIRAllergyIntolerance,
    patientHash: ActionHash
  ): Promise<{ allergyHash: ActionHash; imported: boolean }> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'records',
      fn_name: 'create_allergy_from_fhir',
      payload: {
        patientHash,
        allergen: allergy.code?.text || allergy.code?.coding?.[0]?.display,
        criticality: allergy.criticality,
        reaction: allergy.reaction?.[0]?.manifestation?.[0]?.coding?.[0]?.display,
        severity: allergy.reaction?.[0]?.severity,
      },
    }) as Promise<{ allergyHash: ActionHash; imported: boolean }>;
  }

  /**
   * Import a FHIR MedicationStatement
   */
  async importMedicationStatement(
    medication: FHIRMedicationStatement,
    patientHash: ActionHash
  ): Promise<{ prescriptionHash: ActionHash; imported: boolean }> {
    const coding = medication.medicationCodeableConcept?.coding?.[0];

    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'records',
      fn_name: 'create_medication_from_fhir',
      payload: {
        patientHash,
        rxNormCode: coding?.code,
        medicationName: coding?.display || medication.medicationCodeableConcept?.text,
        status: medication.status,
        dosage: medication.dosage?.[0]?.text,
        startDate: medication.effectiveDateTime,
      },
    }) as Promise<{ prescriptionHash: ActionHash; imported: boolean }>;
  }

  /**
   * Import a FHIR Bundle into Mycelix
   */
  async importBundle(
    bundle: FHIRBundle,
    patientHash?: ActionHash,
    _options?: {
      overwriteExisting?: boolean;
      validateFirst?: boolean;
      skipDuplicates?: boolean;
    }
  ): Promise<{
    totalResources: number;
    importedResources: number;
    errors: string[];
  }> {
    const entries = bundle.entry || [];
    const errors: string[] = [];
    let importedResources = 0;
    let currentPatientHash = patientHash;

    for (const entry of entries) {
      const resource = entry.resource;
      if (!resource) continue;

      try {
        switch (resource.resourceType) {
          case 'Patient': {
            const result = await this.importPatient(resource as FHIRPatient);
            if (result.imported) {
              importedResources++;
              currentPatientHash = result.patientHash;
            }
            break;
          }
          case 'Condition': {
            if (currentPatientHash) {
              const result = await this.importCondition(resource as FHIRCondition, currentPatientHash);
              if (result.imported) importedResources++;
            }
            break;
          }
          case 'Observation': {
            if (currentPatientHash) {
              const result = await this.importObservation(resource as FHIRObservation, currentPatientHash);
              if (result.imported) importedResources++;
            }
            break;
          }
          case 'AllergyIntolerance': {
            if (currentPatientHash) {
              const result = await this.importAllergyIntolerance(resource as FHIRAllergyIntolerance, currentPatientHash);
              if (result.imported) importedResources++;
            }
            break;
          }
          case 'MedicationStatement': {
            if (currentPatientHash) {
              const result = await this.importMedicationStatement(resource as FHIRMedicationStatement, currentPatientHash);
              if (result.imported) importedResources++;
            }
            break;
          }
          default:
            // Skip unsupported resource types
            break;
        }
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error);
        errors.push(`${resource.resourceType}/${resource.id}: ${errorMsg}`);
      }
    }

    return {
      totalResources: entries.length,
      importedResources,
      errors,
    };
  }

  /**
   * Import a single FHIR resource
   */
  async importResource(
    resource: FHIRResource,
    patientHash: ActionHash
  ): Promise<{ success: boolean; actionHash?: ActionHash; error?: string }> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'fhir_import',
      fn_name: 'import_resource',
      payload: { resource, patient_hash: patientHash },
    }) as Promise<{ success: boolean; actionHash?: ActionHash; error?: string }>;
  }

  /**
   * Import from external FHIR server
   */
  async importFromServer(
    serverUrl: string,
    patientId: string,
    targetPatientHash: ActionHash,
    accessToken?: string
  ): Promise<{
    imported: number;
    resourceTypes: string[];
    errors: string[];
  }> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'fhir_import',
      fn_name: 'import_from_server',
      payload: {
        server_url: serverUrl,
        patient_id: patientId,
        target_patient_hash: targetPatientHash,
        access_token: accessToken,
      },
    }) as Promise<{
      imported: number;
      resourceTypes: string[];
      errors: string[];
    }>;
  }
}

// ============================================================================
// Unified FHIR Bridge
// ============================================================================

/**
 * Unified interface for FHIR R4 interoperability
 */
export class FHIRBridge {
  readonly export: FHIRExportClient;
  readonly import: FHIRImportClient;

  private static readonly VALID_RESOURCE_TYPES = new Set([
    'Patient', 'Observation', 'Condition', 'AllergyIntolerance',
    'MedicationStatement', 'Consent', 'Bundle', 'Immunization',
    'Procedure', 'DiagnosticReport', 'DocumentReference',
  ]);

  constructor(client: ZomeCallable) {
    this.export = new FHIRExportClient(client);
    this.import = new FHIRImportClient(client);
  }

  /**
   * Validate a FHIR resource
   */
  validateResource(resource: FHIRResource): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Check resourceType
    if (!resource.resourceType) {
      errors.push('Missing resourceType');
    } else if (!FHIRBridge.VALID_RESOURCE_TYPES.has(resource.resourceType)) {
      errors.push(`Invalid resourceType: ${resource.resourceType}`);
    }

    // Check id for non-Patient resources
    if (!resource.id && resource.resourceType !== 'Bundle') {
      errors.push('Missing id');
    }

    // Resource-specific validation
    const fieldValidation = validateRequiredFields(resource);
    if (!fieldValidation.valid) {
      errors.push(...fieldValidation.missingFields.map(f => `Missing required field: ${f}`));
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  /**
   * Create a downloadable JSON file of patient data
   */
  async createPatientExport(patientHash: ActionHash): Promise<{
    bundle: FHIRBundle;
    json: string;
    resourceCount: number;
  }> {
    const bundle = await this.export.exportBundle(patientHash, {
      includePatient: true,
      includeObservations: true,
      includeConditions: true,
      includeAllergies: true,
      includeMedications: true,
      includeConsents: true,
    });

    return {
      bundle,
      json: JSON.stringify(bundle, null, 2),
      resourceCount: bundle.entry?.length || 0,
    };
  }

  /**
   * Transfer patient record to external FHIR server
   */
  async transferToExternalServer(
    patientHash: ActionHash,
    serverUrl: string,
    accessToken: string,
    options?: {
      resourceTypes?: string[];
      validateBeforeSend?: boolean;
    }
  ): Promise<{
    success: boolean;
    transferred: number;
    errors: string[];
  }> {
    return this.export.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'fhir_transfer',
      fn_name: 'transfer_to_server',
      payload: {
        patient_hash: patientHash,
        server_url: serverUrl,
        access_token: accessToken,
        options,
      },
    }) as Promise<{
      success: boolean;
      transferred: number;
      errors: string[];
    }>;
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a FHIR bridge instance
 */
export function createFHIRBridge(client: ZomeCallable): FHIRBridge {
  return new FHIRBridge(client);
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Convert timestamp to FHIR datetime string
 * Handles both milliseconds (JS Date.getTime()) and microseconds (Mycelix internal)
 */
export function toFHIRDateTime(timestamp: number | Date): string {
  // Handle Date objects
  if (timestamp instanceof Date) {
    return timestamp.toISOString();
  }

  // Detect if timestamp is in microseconds (> 1e14) or milliseconds
  // Timestamps after 2001 in microseconds are > 1e15
  // Timestamps after 2001 in milliseconds are < 2e12
  const ms = timestamp > 1e14 ? timestamp / 1000 : timestamp;
  return new Date(ms).toISOString();
}

/**
 * Convert FHIR datetime string to timestamp (milliseconds)
 */
export function fromFHIRDateTime(datetime: string): number {
  return new Date(datetime).getTime();
}

/**
 * Create LOINC coding for common vitals
 * Returns FHIRCoding with system, code, display
 */
export function createVitalsCoding(vitalType: string): FHIRCoding {
  const loincCodes: Record<string, { code: string; display: string }> = {
    'heart-rate': { code: '8867-4', display: 'Heart rate' },
    'blood-pressure': { code: '85354-9', display: 'Blood pressure panel' },
    'blood-pressure-systolic': { code: '8480-6', display: 'Systolic blood pressure' },
    'blood-pressure-diastolic': { code: '8462-4', display: 'Diastolic blood pressure' },
    'temperature': { code: '8310-5', display: 'Body temperature' },
    'body-temperature': { code: '8310-5', display: 'Body temperature' },
    'respiratory-rate': { code: '9279-1', display: 'Respiratory rate' },
    'oxygen-saturation': { code: '2708-6', display: 'Oxygen saturation' },
    'weight': { code: '29463-7', display: 'Body weight' },
    'body-weight': { code: '29463-7', display: 'Body weight' },
    'height': { code: '8302-2', display: 'Body height' },
    'body-height': { code: '8302-2', display: 'Body height' },
    'bmi': { code: '39156-5', display: 'Body mass index' },
  };

  const loinc = loincCodes[vitalType];
  if (!loinc) {
    return { display: vitalType };
  }

  return {
    system: 'http://loinc.org',
    code: loinc.code,
    display: loinc.display,
  };
}

/**
 * Create SNOMED coding for conditions
 */
export function createSNOMEDCoding(
  code: string,
  display?: string
): FHIRCoding {
  return {
    system: 'http://snomed.info/sct',
    code,
    display,
  };
}

/**
 * Create RxNorm coding for medications
 */
export function createRxNormCoding(
  code: string,
  display?: string
): FHIRCoding {
  return {
    system: 'http://www.nlm.nih.gov/research/umls/rxnorm',
    code,
    display,
  };
}

/**
 * Create ICD-10-CM coding for diagnoses
 */
export function createICD10Coding(
  code: string,
  display?: string
): FHIRCoding {
  return {
    system: 'http://hl7.org/fhir/sid/icd-10-cm',
    code,
    display,
  };
}

/**
 * Validate a FHIR resource has required fields
 * @param resource - The FHIR resource to validate
 * @param requiredFields - Optional array of field names to check (if not provided, uses resource-specific defaults)
 */
export function validateRequiredFields(
  resource: FHIRResource,
  requiredFields?: string[]
): { valid: boolean; missingFields: string[] } {
  const missingFields: string[] = [];

  // If specific fields provided, check those
  if (requiredFields) {
    for (const field of requiredFields) {
      if (!((resource as unknown) as Record<string, unknown>)[field]) {
        missingFields.push(field);
      }
    }
    return {
      valid: missingFields.length === 0,
      missingFields,
    };
  }

  // Default validation based on resource type
  if (!resource.resourceType) missingFields.push('resourceType');
  if (!resource.id) missingFields.push('id');

  // Resource-specific required fields
  switch (resource.resourceType) {
    case 'Patient':
      // Patient has no required fields beyond base
      break;
    case 'Observation': {
      const obs = resource as FHIRObservation;
      if (!obs.status) missingFields.push('status');
      if (!obs.code) missingFields.push('code');
      break;
    }
    case 'Condition': {
      const cond = resource as FHIRCondition;
      if (!cond.subject) missingFields.push('subject');
      break;
    }
    case 'AllergyIntolerance': {
      const allergy = resource as FHIRAllergyIntolerance;
      if (!allergy.patient) missingFields.push('patient');
      break;
    }
    case 'MedicationStatement': {
      const med = resource as FHIRMedicationStatement;
      if (!med.status) missingFields.push('status');
      if (!med.subject) missingFields.push('subject');
      if (!med.medicationCodeableConcept && !med.medicationReference) {
        missingFields.push('medication[x]');
      }
      break;
    }
    case 'Consent': {
      const consent = resource as FHIRConsent;
      if (!consent.status) missingFields.push('status');
      if (!consent.scope) missingFields.push('scope');
      if (!consent.category || consent.category.length === 0) {
        missingFields.push('category');
      }
      break;
    }
  }

  return {
    valid: missingFields.length === 0,
    missingFields,
  };
}

// ============================================================================
// Exports
// ============================================================================

export default {
  FHIRBridge,
  FHIRExportClient,
  FHIRImportClient,
  createFHIRBridge,
  toFHIRDateTime,
  fromFHIRDateTime,
  createVitalsCoding,
  createSNOMEDCoding,
  createRxNormCoding,
  createICD10Coding,
  validateRequiredFields,
};
