/**
 * Health-FHIR Integration Tests
 *
 * Tests for the health-fhir integration module including:
 * - FHIR R4 resource export
 * - FHIR R4 resource import
 * - Utility functions for coding systems
 * - Type validation
 * - Cross-system interoperability workflows
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  FHIRBridge,
  FHIRExportClient,
  FHIRImportClient,
  toFHIRDateTime,
  fromFHIRDateTime,
  createVitalsCoding,
  createSNOMEDCoding,
  createRxNormCoding,
  createICD10Coding,
  validateRequiredFields,
  type FHIRResource,
  type FHIRPatient,
  type FHIRObservation,
  type FHIRCondition,
  type FHIRAllergyIntolerance,
  type FHIRMedicationStatement,
  type FHIRConsent,
  type FHIRBundle,
  type FHIRCoding,
  type FHIRCodeableConcept,
  type FHIRIdentifier,
  type FHIRHumanName,
  type FHIRAddress,
  type FHIRContactPoint,
  type FHIRReference,
  type FHIRQuantity,
  type FHIRPeriod,
} from '../../src/integrations/health-fhir/index.js';
import type { MycelixClient } from '../../src/client/index.js';

describe('Health-FHIR Integration', () => {
  describe('toFHIRDateTime', () => {
    it('should convert timestamp to FHIR datetime format', () => {
      const timestamp = new Date('2024-06-15T14:30:00Z').getTime();
      const fhirDateTime = toFHIRDateTime(timestamp);
      expect(fhirDateTime).toBe('2024-06-15T14:30:00.000Z');
    });

    it('should handle Date objects', () => {
      const date = new Date('2024-01-01T00:00:00Z');
      const fhirDateTime = toFHIRDateTime(date);
      expect(fhirDateTime).toBe('2024-01-01T00:00:00.000Z');
    });

    it('should handle current timestamp', () => {
      const now = Date.now();
      const fhirDateTime = toFHIRDateTime(now);
      expect(fhirDateTime).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
    });
  });

  describe('fromFHIRDateTime', () => {
    it('should convert FHIR datetime to timestamp', () => {
      const fhirDateTime = '2024-06-15T14:30:00.000Z';
      const timestamp = fromFHIRDateTime(fhirDateTime);
      expect(timestamp).toBe(new Date('2024-06-15T14:30:00Z').getTime());
    });

    it('should handle date-only format', () => {
      const fhirDate = '2024-06-15';
      const timestamp = fromFHIRDateTime(fhirDate);
      expect(timestamp).toBe(new Date('2024-06-15').getTime());
    });

    it('should handle datetime without milliseconds', () => {
      const fhirDateTime = '2024-06-15T14:30:00Z';
      const timestamp = fromFHIRDateTime(fhirDateTime);
      expect(typeof timestamp).toBe('number');
    });
  });

  describe('createVitalsCoding', () => {
    it('should create blood pressure coding', () => {
      const coding = createVitalsCoding('blood-pressure');
      expect(coding.system).toBe('http://loinc.org');
      expect(coding.code).toBe('85354-9');
      expect(coding.display).toBe('Blood pressure panel');
    });

    it('should create heart rate coding', () => {
      const coding = createVitalsCoding('heart-rate');
      expect(coding.code).toBe('8867-4');
      expect(coding.display).toBe('Heart rate');
    });

    it('should create body temperature coding', () => {
      const coding = createVitalsCoding('body-temperature');
      expect(coding.code).toBe('8310-5');
      expect(coding.display).toBe('Body temperature');
    });

    it('should create respiratory rate coding', () => {
      const coding = createVitalsCoding('respiratory-rate');
      expect(coding.code).toBe('9279-1');
    });

    it('should create oxygen saturation coding', () => {
      const coding = createVitalsCoding('oxygen-saturation');
      expect(coding.code).toBe('2708-6');
    });

    it('should create body weight coding', () => {
      const coding = createVitalsCoding('body-weight');
      expect(coding.code).toBe('29463-7');
    });

    it('should create body height coding', () => {
      const coding = createVitalsCoding('body-height');
      expect(coding.code).toBe('8302-2');
    });

    it('should create BMI coding', () => {
      const coding = createVitalsCoding('bmi');
      expect(coding.code).toBe('39156-5');
    });
  });

  describe('createSNOMEDCoding', () => {
    it('should create SNOMED CT coding with system', () => {
      const coding = createSNOMEDCoding('38341003', 'Hypertensive disorder');
      expect(coding.system).toBe('http://snomed.info/sct');
      expect(coding.code).toBe('38341003');
      expect(coding.display).toBe('Hypertensive disorder');
    });

    it('should handle code without display', () => {
      const coding = createSNOMEDCoding('73211009');
      expect(coding.code).toBe('73211009');
      expect(coding.display).toBeUndefined();
    });
  });

  describe('createRxNormCoding', () => {
    it('should create RxNorm coding', () => {
      const coding = createRxNormCoding('197361', 'Lisinopril 10 MG Oral Tablet');
      expect(coding.system).toBe('http://www.nlm.nih.gov/research/umls/rxnorm');
      expect(coding.code).toBe('197361');
      expect(coding.display).toBe('Lisinopril 10 MG Oral Tablet');
    });
  });

  describe('createICD10Coding', () => {
    it('should create ICD-10-CM coding', () => {
      const coding = createICD10Coding('I10', 'Essential hypertension');
      expect(coding.system).toBe('http://hl7.org/fhir/sid/icd-10-cm');
      expect(coding.code).toBe('I10');
      expect(coding.display).toBe('Essential hypertension');
    });
  });

  describe('validateRequiredFields', () => {
    it('should pass for valid patient resource', () => {
      const patient: FHIRPatient = {
        resourceType: 'Patient',
        id: 'patient-1',
        name: [{ family: 'Smith', given: ['John'] }],
      };

      const result = validateRequiredFields(patient, ['resourceType', 'id']);
      expect(result.valid).toBe(true);
      expect(result.missingFields).toHaveLength(0);
    });

    it('should fail for missing required fields', () => {
      const patient: Partial<FHIRPatient> = {
        resourceType: 'Patient',
      };

      const result = validateRequiredFields(patient as FHIRPatient, ['resourceType', 'id', 'name']);
      expect(result.valid).toBe(false);
      expect(result.missingFields).toContain('id');
      expect(result.missingFields).toContain('name');
    });

    it('should handle empty resource', () => {
      const result = validateRequiredFields({} as FHIRResource, ['resourceType']);
      expect(result.valid).toBe(false);
      expect(result.missingFields).toContain('resourceType');
    });
  });

  describe('FHIRExportClient', () => {
    let mockClient: MycelixClient;
    let exportClient: FHIRExportClient;

    beforeEach(() => {
      mockClient = {
        callZome: vi.fn(),
      } as unknown as MycelixClient;
      exportClient = new FHIRExportClient(mockClient);
    });

    it('should export patient to FHIR format', async () => {
      const patientHash = new Uint8Array(39);
      vi.mocked(mockClient.callZome).mockResolvedValue({
        patientId: 'patient-1',
        firstName: 'John',
        lastName: 'Smith',
        dateOfBirth: '1980-05-15',
        gender: 'Male',
        phone: '+1-555-123-4567',
        email: 'john.smith@email.com',
        address: {
          street: '123 Main St',
          city: 'Los Angeles',
          state: 'CA',
          zip: '90210',
        },
      });

      const fhirPatient = await exportClient.exportPatient(patientHash);

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'health',
        zome_name: 'patient',
        fn_name: 'get_patient',
        payload: { patientHash },
      });

      expect(fhirPatient.resourceType).toBe('Patient');
      expect(fhirPatient.name?.[0]?.family).toBe('Smith');
      expect(fhirPatient.name?.[0]?.given).toContain('John');
      expect(fhirPatient.gender).toBe('male');
    });

    it('should export vitals as FHIR Observation', async () => {
      const patientHash = new Uint8Array(39);
      vi.mocked(mockClient.callZome).mockResolvedValue({
        recordedAt: Date.now(),
        bloodPressureSystolic: 120,
        bloodPressureDiastolic: 80,
        heartRate: 72,
        temperature: 98.6,
        respiratoryRate: 16,
        oxygenSaturation: 98,
      });

      const observations = await exportClient.exportVitals(patientHash);

      expect(observations).toBeInstanceOf(Array);
      expect(observations.length).toBeGreaterThan(0);
      expect(observations[0].resourceType).toBe('Observation');
    });

    it('should export conditions as FHIR Condition', async () => {
      vi.mocked(mockClient.callZome).mockResolvedValue([
        {
          diagnosisId: 'dx-1',
          icd10Code: 'I10',
          description: 'Essential hypertension',
          status: 'Active',
          onsetDate: '2020-01-15',
        },
        {
          diagnosisId: 'dx-2',
          icd10Code: 'E11.9',
          description: 'Type 2 diabetes mellitus',
          status: 'Active',
          onsetDate: '2019-06-20',
        },
      ]);

      const conditions = await exportClient.exportConditions(new Uint8Array(39));

      expect(conditions).toHaveLength(2);
      expect(conditions[0].resourceType).toBe('Condition');
      expect(conditions[0].code?.coding?.[0]?.code).toBe('I10');
    });

    it('should export allergies as FHIR AllergyIntolerance', async () => {
      vi.mocked(mockClient.callZome).mockResolvedValue([
        {
          allergyId: 'allergy-1',
          allergen: 'Penicillin',
          reaction: 'Anaphylaxis',
          severity: 'LifeThreatening',
          recordedDate: '2015-03-10',
        },
      ]);

      const allergies = await exportClient.exportAllergies(new Uint8Array(39));

      expect(allergies).toHaveLength(1);
      expect(allergies[0].resourceType).toBe('AllergyIntolerance');
      expect(allergies[0].criticality).toBe('high');
    });

    it('should export medications as FHIR MedicationStatement', async () => {
      vi.mocked(mockClient.callZome).mockResolvedValue([
        {
          prescriptionId: 'rx-1',
          medicationName: 'Lisinopril',
          rxNormCode: '197361',
          dosage: '10mg',
          frequency: 'once daily',
          status: 'Active',
          startDate: '2023-01-01',
        },
      ]);

      const medications = await exportClient.exportMedications(new Uint8Array(39));

      expect(medications).toHaveLength(1);
      expect(medications[0].resourceType).toBe('MedicationStatement');
      expect(medications[0].medicationCodeableConcept?.coding?.[0]?.system).toBe(
        'http://www.nlm.nih.gov/research/umls/rxnorm'
      );
    });

    it('should export consents as FHIR Consent', async () => {
      vi.mocked(mockClient.callZome).mockResolvedValue([
        {
          consentId: 'consent-1',
          scope: 'Research',
          status: 'Active',
          grantedDate: '2024-01-01',
          expirationDate: '2025-01-01',
          dataCategories: ['Labs', 'Vitals'],
        },
      ]);

      const consents = await exportClient.exportConsents(new Uint8Array(39));

      expect(consents).toHaveLength(1);
      expect(consents[0].resourceType).toBe('Consent');
      expect(consents[0].status).toBe('active');
    });

    it('should export complete patient bundle', async () => {
      vi.mocked(mockClient.callZome)
        .mockResolvedValueOnce({ patientId: 'p1', firstName: 'Jane', lastName: 'Doe' })
        .mockResolvedValueOnce({ bloodPressureSystolic: 118 })
        .mockResolvedValueOnce([{ diagnosisId: 'dx-1' }])
        .mockResolvedValueOnce([{ allergyId: 'a-1' }])
        .mockResolvedValueOnce([{ prescriptionId: 'rx-1' }])
        .mockResolvedValueOnce([{ consentId: 'c-1' }]);

      const bundle = await exportClient.exportPatientBundle(new Uint8Array(39));

      expect(bundle.resourceType).toBe('Bundle');
      expect(bundle.type).toBe('collection');
      expect(bundle.entry).toBeDefined();
      expect(bundle.entry!.length).toBeGreaterThan(0);
    });
  });

  describe('FHIRImportClient', () => {
    let mockClient: MycelixClient;
    let importClient: FHIRImportClient;

    beforeEach(() => {
      mockClient = {
        callZome: vi.fn(),
      } as unknown as MycelixClient;
      importClient = new FHIRImportClient(mockClient);
    });

    it('should import FHIR Patient', async () => {
      const fhirPatient: FHIRPatient = {
        resourceType: 'Patient',
        id: 'ext-patient-1',
        name: [{ family: 'Johnson', given: ['Alice'] }],
        gender: 'female',
        birthDate: '1990-08-20',
        telecom: [
          { system: 'phone', value: '+1-555-987-6543' },
          { system: 'email', value: 'alice@email.com' },
        ],
        address: [
          {
            line: ['456 Oak Ave'],
            city: 'San Francisco',
            state: 'CA',
            postalCode: '94102',
          },
        ],
      };

      vi.mocked(mockClient.callZome).mockResolvedValue({
        patientHash: new Uint8Array(39),
        imported: true,
      });

      const result = await importClient.importPatient(fhirPatient);

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'health',
        zome_name: 'patient',
        fn_name: 'create_patient_from_fhir',
        payload: expect.objectContaining({
          firstName: 'Alice',
          lastName: 'Johnson',
          gender: 'Female',
        }),
      });
      expect(result.imported).toBe(true);
    });

    it('should import FHIR Observation as vitals', async () => {
      const observation: FHIRObservation = {
        resourceType: 'Observation',
        id: 'obs-1',
        status: 'final',
        code: {
          coding: [{ system: 'http://loinc.org', code: '8867-4', display: 'Heart rate' }],
        },
        valueQuantity: {
          value: 72,
          unit: 'beats/minute',
          system: 'http://unitsofmeasure.org',
          code: '/min',
        },
        effectiveDateTime: '2024-06-15T10:00:00Z',
        subject: { reference: 'Patient/patient-1' },
      };

      vi.mocked(mockClient.callZome).mockResolvedValue({
        vitalRecordHash: new Uint8Array(39),
        imported: true,
      });

      const result = await importClient.importObservation(observation, new Uint8Array(39));

      expect(result.imported).toBe(true);
    });

    it('should import FHIR Condition', async () => {
      const condition: FHIRCondition = {
        resourceType: 'Condition',
        id: 'cond-1',
        clinicalStatus: {
          coding: [{ code: 'active' }],
        },
        code: {
          coding: [createICD10Coding('E11.9', 'Type 2 diabetes')],
        },
        subject: { reference: 'Patient/patient-1' },
        onsetDateTime: '2020-03-15',
      };

      vi.mocked(mockClient.callZome).mockResolvedValue({
        diagnosisHash: new Uint8Array(39),
        imported: true,
      });

      const result = await importClient.importCondition(condition, new Uint8Array(39));

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'health',
        zome_name: 'records',
        fn_name: 'create_diagnosis_from_fhir',
        payload: expect.objectContaining({
          icd10Code: 'E11.9',
        }),
      });
      expect(result.imported).toBe(true);
    });

    it('should import FHIR AllergyIntolerance', async () => {
      const allergy: FHIRAllergyIntolerance = {
        resourceType: 'AllergyIntolerance',
        id: 'allergy-1',
        clinicalStatus: { coding: [{ code: 'active' }] },
        criticality: 'high',
        code: {
          coding: [createSNOMEDCoding('764146007', 'Penicillin')],
          text: 'Penicillin',
        },
        patient: { reference: 'Patient/patient-1' },
        reaction: [
          {
            manifestation: [{ coding: [{ display: 'Anaphylaxis' }] }],
            severity: 'severe',
          },
        ],
      };

      vi.mocked(mockClient.callZome).mockResolvedValue({
        allergyHash: new Uint8Array(39),
        imported: true,
      });

      const result = await importClient.importAllergyIntolerance(allergy, new Uint8Array(39));

      expect(result.imported).toBe(true);
    });

    it('should import FHIR Bundle', async () => {
      const bundle: FHIRBundle = {
        resourceType: 'Bundle',
        type: 'collection',
        entry: [
          {
            resource: {
              resourceType: 'Patient',
              id: 'p1',
              name: [{ family: 'Test' }],
            } as FHIRPatient,
          },
          {
            resource: {
              resourceType: 'Condition',
              id: 'c1',
              code: { coding: [createICD10Coding('I10')] },
              subject: { reference: 'Patient/p1' },
            } as FHIRCondition,
          },
        ],
      };

      vi.mocked(mockClient.callZome)
        .mockResolvedValueOnce({ patientHash: new Uint8Array(39), imported: true })
        .mockResolvedValueOnce({ diagnosisHash: new Uint8Array(39), imported: true });

      const result = await importClient.importBundle(bundle);

      expect(result.totalResources).toBe(2);
      expect(result.importedResources).toBe(2);
      expect(result.errors).toHaveLength(0);
    });
  });

  describe('FHIRBridge', () => {
    let mockClient: MycelixClient;
    let bridge: FHIRBridge;

    beforeEach(() => {
      mockClient = {
        callZome: vi.fn(),
      } as unknown as MycelixClient;
      bridge = new FHIRBridge(mockClient);
    });

    it('should provide access to export and import clients', () => {
      expect(bridge.export).toBeInstanceOf(FHIRExportClient);
      expect(bridge.import).toBeInstanceOf(FHIRImportClient);
    });

    it('should validate FHIR resource', () => {
      const patient: FHIRPatient = {
        resourceType: 'Patient',
        id: 'patient-1',
        name: [{ family: 'Test' }],
      };

      const validation = bridge.validateResource(patient);
      expect(validation.valid).toBe(true);
    });

    it('should detect invalid FHIR resource', () => {
      const invalid = {
        resourceType: 'InvalidType',
      } as unknown as FHIRResource;

      const validation = bridge.validateResource(invalid);
      expect(validation.valid).toBe(false);
    });
  });

  describe('Cross-System Interoperability Workflows', () => {
    let mockClient: MycelixClient;
    let bridge: FHIRBridge;

    beforeEach(() => {
      mockClient = {
        callZome: vi.fn(),
      } as unknown as MycelixClient;
      bridge = new FHIRBridge(mockClient);
    });

    it('should export patient for EHR transfer', async () => {
      // Simulate complete export for hospital transfer
      vi.mocked(mockClient.callZome)
        .mockResolvedValueOnce({
          patientId: 'p1',
          firstName: 'Robert',
          lastName: 'Williams',
          dateOfBirth: '1965-11-22',
          gender: 'Male',
        })
        .mockResolvedValueOnce({
          bloodPressureSystolic: 145,
          bloodPressureDiastolic: 92,
          heartRate: 78,
        })
        .mockResolvedValueOnce([
          { diagnosisId: 'dx1', icd10Code: 'I10', description: 'Hypertension' },
          { diagnosisId: 'dx2', icd10Code: 'E78.5', description: 'Hyperlipidemia' },
        ])
        .mockResolvedValueOnce([
          { allergyId: 'a1', allergen: 'Aspirin', severity: 'Moderate' },
        ])
        .mockResolvedValueOnce([
          { prescriptionId: 'rx1', medicationName: 'Amlodipine', dosage: '5mg' },
          { prescriptionId: 'rx2', medicationName: 'Atorvastatin', dosage: '20mg' },
        ])
        .mockResolvedValueOnce([]);

      const bundle = await bridge.export.exportPatientBundle(new Uint8Array(39));

      expect(bundle.resourceType).toBe('Bundle');
      expect(bundle.type).toBe('collection');
      // Should contain Patient + Observations + Conditions + Allergies + Medications
      expect(bundle.entry!.length).toBeGreaterThan(4);
    });

    it('should import patient from external EHR', async () => {
      const externalBundle: FHIRBundle = {
        resourceType: 'Bundle',
        type: 'collection',
        entry: [
          {
            resource: {
              resourceType: 'Patient',
              id: 'external-123',
              identifier: [
                { system: 'http://hospital.org/mrn', value: 'MRN12345' },
              ],
              name: [{ family: 'External', given: ['Patient'] }],
              gender: 'male',
              birthDate: '1970-05-10',
            } as FHIRPatient,
          },
          {
            resource: {
              resourceType: 'Condition',
              id: 'ext-cond-1',
              code: { coding: [createICD10Coding('J45.909', 'Asthma')] },
              clinicalStatus: { coding: [{ code: 'active' }] },
              subject: { reference: 'Patient/external-123' },
            } as FHIRCondition,
          },
          {
            resource: {
              resourceType: 'MedicationStatement',
              id: 'ext-med-1',
              status: 'active',
              medicationCodeableConcept: {
                coding: [createRxNormCoding('745679', 'Albuterol inhaler')],
              },
              subject: { reference: 'Patient/external-123' },
            } as FHIRMedicationStatement,
          },
        ],
      };

      vi.mocked(mockClient.callZome)
        .mockResolvedValueOnce({ patientHash: new Uint8Array(39), imported: true })
        .mockResolvedValueOnce({ diagnosisHash: new Uint8Array(39), imported: true })
        .mockResolvedValueOnce({ prescriptionHash: new Uint8Array(39), imported: true });

      const result = await bridge.import.importBundle(externalBundle);

      expect(result.totalResources).toBe(3);
      expect(result.importedResources).toBe(3);
    });

    it('should handle partial import failures gracefully', async () => {
      const bundle: FHIRBundle = {
        resourceType: 'Bundle',
        type: 'collection',
        entry: [
          {
            resource: {
              resourceType: 'Patient',
              id: 'p1',
              name: [{ family: 'Valid' }],
            } as FHIRPatient,
          },
          {
            resource: {
              resourceType: 'Condition',
              id: 'invalid-cond',
              // Missing required code field
            } as FHIRCondition,
          },
        ],
      };

      vi.mocked(mockClient.callZome)
        .mockResolvedValueOnce({ patientHash: new Uint8Array(39), imported: true })
        .mockRejectedValueOnce(new Error('Invalid condition: missing code'));

      const result = await bridge.import.importBundle(bundle);

      expect(result.totalResources).toBe(2);
      expect(result.importedResources).toBe(1);
      expect(result.errors).toHaveLength(1);
      expect(result.errors[0]).toContain('missing code');
    });
  });

  describe('Type Validation', () => {
    it('should validate FHIRCoding structure', () => {
      const coding: FHIRCoding = {
        system: 'http://loinc.org',
        code: '8867-4',
        display: 'Heart rate',
      };

      expect(coding.system).toBeDefined();
      expect(coding.code).toBeDefined();
    });

    it('should validate FHIRCodeableConcept structure', () => {
      const concept: FHIRCodeableConcept = {
        coding: [createSNOMEDCoding('38341003', 'Hypertension')],
        text: 'Hypertensive disorder',
      };

      expect(concept.coding).toBeInstanceOf(Array);
      expect(concept.text).toBeDefined();
    });

    it('should validate FHIRIdentifier structure', () => {
      const identifier: FHIRIdentifier = {
        system: 'http://hl7.org/fhir/sid/us-ssn',
        value: '123-45-6789',
        use: 'official',
      };

      expect(identifier.system).toBeDefined();
      expect(identifier.value).toBeDefined();
    });

    it('should validate FHIRHumanName structure', () => {
      const name: FHIRHumanName = {
        use: 'official',
        family: 'Smith',
        given: ['John', 'Michael'],
        prefix: ['Dr.'],
        suffix: ['Jr.'],
      };

      expect(name.family).toBeDefined();
      expect(name.given).toBeInstanceOf(Array);
    });

    it('should validate FHIRAddress structure', () => {
      const address: FHIRAddress = {
        use: 'home',
        type: 'physical',
        line: ['123 Main St', 'Apt 4B'],
        city: 'New York',
        state: 'NY',
        postalCode: '10001',
        country: 'USA',
      };

      expect(address.city).toBeDefined();
      expect(address.line).toBeInstanceOf(Array);
    });

    it('should validate FHIRContactPoint structure', () => {
      const contact: FHIRContactPoint = {
        system: 'phone',
        value: '+1-555-123-4567',
        use: 'mobile',
        rank: 1,
      };

      expect(contact.system).toBeDefined();
      expect(contact.value).toBeDefined();
    });

    it('should validate FHIRReference structure', () => {
      const reference: FHIRReference = {
        reference: 'Patient/patient-123',
        display: 'John Smith',
      };

      expect(reference.reference).toBeDefined();
    });

    it('should validate FHIRQuantity structure', () => {
      const quantity: FHIRQuantity = {
        value: 120,
        unit: 'mmHg',
        system: 'http://unitsofmeasure.org',
        code: 'mm[Hg]',
      };

      expect(quantity.value).toBeDefined();
      expect(quantity.unit).toBeDefined();
    });

    it('should validate FHIRPeriod structure', () => {
      const period: FHIRPeriod = {
        start: '2024-01-01T00:00:00Z',
        end: '2024-12-31T23:59:59Z',
      };

      expect(period.start).toBeDefined();
    });
  });

  describe('Edge Cases', () => {
    it('should handle missing optional FHIR fields', () => {
      const minimalPatient: FHIRPatient = {
        resourceType: 'Patient',
      };

      expect(minimalPatient.resourceType).toBe('Patient');
      expect(minimalPatient.name).toBeUndefined();
    });

    it('should handle empty coding arrays', () => {
      const concept: FHIRCodeableConcept = {
        coding: [],
        text: 'Unknown condition',
      };

      expect(concept.coding).toHaveLength(0);
      expect(concept.text).toBeDefined();
    });

    it('should handle unknown vital type gracefully', () => {
      // This tests error handling for unsupported vital types
      expect(() => createVitalsCoding('unknown-vital' as any)).not.toThrow();
    });

    it('should handle timestamp edge cases', () => {
      // Unix epoch
      const epoch = toFHIRDateTime(0);
      expect(epoch).toContain('1970');

      // Far future date
      const future = toFHIRDateTime(new Date('2100-01-01').getTime());
      expect(future).toContain('2100');
    });

    it('should handle bundle with no entries', () => {
      const emptyBundle: FHIRBundle = {
        resourceType: 'Bundle',
        type: 'collection',
        entry: [],
      };

      expect(emptyBundle.entry).toHaveLength(0);
    });
  });

  describe('Error Handling', () => {
    let mockClient: MycelixClient;
    let bridge: FHIRBridge;

    beforeEach(() => {
      mockClient = {
        callZome: vi.fn(),
      } as unknown as MycelixClient;
      bridge = new FHIRBridge(mockClient);
    });

    it('should handle export errors', async () => {
      vi.mocked(mockClient.callZome).mockRejectedValue(
        new Error('Patient not found')
      );

      await expect(
        bridge.export.exportPatient(new Uint8Array(39))
      ).rejects.toThrow('Patient not found');
    });

    it('should handle import validation errors', async () => {
      vi.mocked(mockClient.callZome).mockRejectedValue(
        new Error('Invalid FHIR resource: missing required field')
      );

      const invalidPatient: FHIRPatient = {
        resourceType: 'Patient',
        id: 'test',
      };

      await expect(
        bridge.import.importPatient(invalidPatient)
      ).rejects.toThrow('Invalid FHIR resource');
    });

    it('should handle network timeouts', async () => {
      vi.mocked(mockClient.callZome).mockRejectedValue(
        new Error('Request timeout')
      );

      await expect(
        bridge.export.exportConditions(new Uint8Array(39))
      ).rejects.toThrow('Request timeout');
    });

    it('should handle malformed FHIR datetime', () => {
      // fromFHIRDateTime should handle invalid dates gracefully
      const result = fromFHIRDateTime('not-a-date');
      expect(isNaN(result)).toBe(true);
    });
  });
});
