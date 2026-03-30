// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Health-Identity Cross-Domain Linking Tests
 *
 * Tests the bidirectional DID ↔ Patient identity resolution
 * that enables federated health data sharing across Mycelix hApps.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  HealthIdentityBridgeClient,
  getHealthIdentityBridgeClient,
  resetHealthIdentityBridgeClient,
  type LinkPatientIdentityInput,
  type GetPatientByDIDInput,
  type VerifyDIDPatientLinkInput,
  type PatientDIDInfo,
  type PatientIdentityLinkRecord,
} from '../../src/integrations/identity/index.js';
import type { MycelixClient } from '../../src/client/index.js';

/** Create a mock MycelixClient with callZome method */
function createMockClient(): MycelixClient {
  return {
    callZome: vi.fn(),
    connect: vi.fn().mockResolvedValue(undefined),
    disconnect: vi.fn().mockResolvedValue(undefined),
    isConnected: vi.fn().mockReturnValue(true),
  } as unknown as MycelixClient;
}

describe('Health-Identity Cross-Domain Linking', () => {
  let client: MycelixClient;
  let bridge: HealthIdentityBridgeClient;

  beforeEach(() => {
    resetHealthIdentityBridgeClient();
    client = createMockClient();
    bridge = getHealthIdentityBridgeClient(client);
  });

  describe('linkPatientToIdentity', () => {
    it('should link a patient to a DID with valid input', async () => {
      const input: LinkPatientIdentityInput = {
        patient_hash: 'uhCkkABC123',
        did: 'did:mycelix:xyz789',
        identity_provider: 'Mycelix',
        verification_method: 'credential',
        confidence_score: 0.95,
      };

      const mockResponse: PatientIdentityLinkRecord = {
        patient_hash: input.patient_hash,
        did: input.did,
        identity_provider: input.identity_provider,
        verified_at: Date.now(),
        verification_method: input.verification_method,
        confidence_score: input.confidence_score,
      };

      vi.mocked(client.callZome).mockResolvedValue(mockResponse);

      const result = await bridge.linkPatientToIdentity(input);

      expect(result.patient_hash).toBe(input.patient_hash);
      expect(result.did).toBe(input.did);
      expect(result.confidence_score).toBe(0.95);
      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'health',
        zome_name: 'patient',
        fn_name: 'link_patient_to_identity',
        payload: input,
      });
    });

    it('should support did:web method', async () => {
      const input: LinkPatientIdentityInput = {
        patient_hash: 'uhCkkDEF456',
        did: 'did:web:example.com',
        identity_provider: 'Web Identity',
        verification_method: 'domain_verification',
        confidence_score: 0.8,
      };

      const mockResponse: PatientIdentityLinkRecord = {
        ...input,
        verified_at: Date.now(),
      };

      vi.mocked(client.callZome).mockResolvedValue(mockResponse);

      const result = await bridge.linkPatientToIdentity(input);

      expect(result.did).toBe('did:web:example.com');
    });

    it('should support government ID verification', async () => {
      const input: LinkPatientIdentityInput = {
        patient_hash: 'uhCkkGHI789',
        did: 'did:mycelix:gov123',
        identity_provider: 'Government ID',
        verification_method: 'in-person',
        confidence_score: 0.99,
      };

      const mockResponse: PatientIdentityLinkRecord = {
        ...input,
        verified_at: Date.now(),
      };

      vi.mocked(client.callZome).mockResolvedValue(mockResponse);

      const result = await bridge.linkPatientToIdentity(input);

      expect(result.identity_provider).toBe('Government ID');
      expect(result.verification_method).toBe('in-person');
      expect(result.confidence_score).toBe(0.99);
    });
  });

  describe('getPatientByDID', () => {
    it('should look up patient by DID', async () => {
      const input: GetPatientByDIDInput = {
        did: 'did:mycelix:abc123',
        is_emergency: false,
      };

      const mockPatient = {
        patient_id: 'PAT-001',
        first_name: 'John',
        last_name: 'Doe',
        date_of_birth: '1980-01-15',
      };

      vi.mocked(client.callZome).mockResolvedValue(mockPatient);

      const result = await bridge.getPatientByDID(input);

      expect(result).toBeDefined();
      expect((result as { patient_id: string }).patient_id).toBe('PAT-001');
      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'health',
        zome_name: 'patient',
        fn_name: 'get_patient_by_did',
        payload: input,
      });
    });

    it('should return null for unknown DID', async () => {
      const input: GetPatientByDIDInput = {
        did: 'did:mycelix:unknown',
        is_emergency: false,
      };

      vi.mocked(client.callZome).mockResolvedValue(null);

      const result = await bridge.getPatientByDID(input);

      expect(result).toBeNull();
    });

    it('should support emergency access', async () => {
      const input: GetPatientByDIDInput = {
        did: 'did:mycelix:emergency123',
        is_emergency: true,
        emergency_reason: 'Patient unconscious, requires immediate care',
      };

      const mockPatient = {
        patient_id: 'PAT-002',
        allergies: [{ allergen: 'Penicillin', severity: 'Severe' }],
      };

      vi.mocked(client.callZome).mockResolvedValue(mockPatient);

      const result = await bridge.getPatientByDID(input);

      expect(result).toBeDefined();
      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            is_emergency: true,
            emergency_reason: 'Patient unconscious, requires immediate care',
          }),
        })
      );
    });
  });

  describe('getDIDForPatient', () => {
    it('should return DID info for linked patient', async () => {
      const patientHash = 'uhCkkXYZ789';

      const mockDIDInfo: PatientDIDInfo = {
        patient_hash: patientHash,
        did: 'did:mycelix:linked123',
        identity_provider: 'Mycelix',
        verified_at: Date.now(),
        confidence_score: 0.92,
      };

      vi.mocked(client.callZome).mockResolvedValue(mockDIDInfo);

      const result = await bridge.getDIDForPatient(patientHash);

      expect(result).toBeDefined();
      expect(result?.did).toBe('did:mycelix:linked123');
      expect(result?.confidence_score).toBe(0.92);
    });

    it('should return null for patient without DID link', async () => {
      const patientHash = 'uhCkkNoLink';

      vi.mocked(client.callZome).mockResolvedValue(null);

      const result = await bridge.getDIDForPatient(patientHash);

      expect(result).toBeNull();
    });
  });

  describe('verifyDIDPatientLink', () => {
    it('should verify valid DID-Patient link', async () => {
      const input: VerifyDIDPatientLinkInput = {
        did: 'did:mycelix:verified123',
        patient_hash: 'uhCkkVerified',
      };

      vi.mocked(client.callZome).mockResolvedValue(true);

      const result = await bridge.verifyDIDPatientLink(input);

      expect(result).toBe(true);
      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'health',
        zome_name: 'patient',
        fn_name: 'verify_did_patient_link',
        payload: input,
      });
    });

    it('should return false for non-existent link', async () => {
      const input: VerifyDIDPatientLinkInput = {
        did: 'did:mycelix:notlinked',
        patient_hash: 'uhCkkWrongPatient',
      };

      vi.mocked(client.callZome).mockResolvedValue(false);

      const result = await bridge.verifyDIDPatientLink(input);

      expect(result).toBe(false);
    });

    it('should return false for mismatched DID-Patient pair', async () => {
      const input: VerifyDIDPatientLinkInput = {
        did: 'did:mycelix:patientA',
        patient_hash: 'uhCkkPatientB',  // Wrong patient
      };

      vi.mocked(client.callZome).mockResolvedValue(false);

      const result = await bridge.verifyDIDPatientLink(input);

      expect(result).toBe(false);
    });
  });

  describe('Cross-Domain Identity Workflows', () => {
    it('should support full identity linking workflow', async () => {
      // Step 1: Create identity link
      const linkInput: LinkPatientIdentityInput = {
        patient_hash: 'uhCkkWorkflow',
        did: 'did:mycelix:workflow123',
        identity_provider: 'Mycelix',
        verification_method: 'credential',
        confidence_score: 0.9,
      };

      const mockLink: PatientIdentityLinkRecord = {
        ...linkInput,
        verified_at: Date.now(),
      };

      vi.mocked(client.callZome)
        .mockResolvedValueOnce(mockLink)
        .mockResolvedValueOnce(true)
        .mockResolvedValueOnce({
          patient_id: 'PAT-WORKFLOW',
          first_name: 'Workflow',
          last_name: 'Test',
        });

      // Create the link
      const link = await bridge.linkPatientToIdentity(linkInput);
      expect(link.did).toBe(linkInput.did);

      // Step 2: Verify the link
      const isLinked = await bridge.verifyDIDPatientLink({
        did: linkInput.did,
        patient_hash: linkInput.patient_hash,
      });
      expect(isLinked).toBe(true);

      // Step 3: Look up by DID
      const patient = await bridge.getPatientByDID({
        did: linkInput.did,
        is_emergency: false,
      });
      expect(patient).toBeDefined();
    });

    it('should enable cross-hApp health data sharing', async () => {
      // Scenario: Another hApp (e.g., Pharmacy) needs to verify patient identity
      // before dispensing medication

      const pharmacyDID = 'did:mycelix:pharmacy456';
      const patientDID = 'did:mycelix:patient789';

      // Step 1: Pharmacy verifies patient identity
      vi.mocked(client.callZome).mockResolvedValueOnce({
        patient_hash: 'uhCkkPharmacyPatient',
        allergies: [{ allergen: 'Codeine', severity: 'Moderate' }],
        medications: ['Lisinopril 10mg'],
      });

      const patient = await bridge.getPatientByDID({
        did: patientDID,
        is_emergency: false,
      });

      expect(patient).toBeDefined();
      // Pharmacy can now check for drug interactions with existing medications
    });

    it('should support insurance verification workflow', async () => {
      // Insurance hApp verifies patient identity before processing claims

      const insuranceLinkInput: LinkPatientIdentityInput = {
        patient_hash: 'uhCkkInsurancePatient',
        did: 'did:mycelix:insured123',
        identity_provider: 'Insurance Provider',
        verification_method: 'member_id',
        confidence_score: 0.98,
      };

      const mockLink: PatientIdentityLinkRecord = {
        ...insuranceLinkInput,
        verified_at: Date.now(),
      };

      vi.mocked(client.callZome).mockResolvedValue(mockLink);

      const link = await bridge.linkPatientToIdentity(insuranceLinkInput);

      expect(link.identity_provider).toBe('Insurance Provider');
      expect(link.verification_method).toBe('member_id');
      expect(link.confidence_score).toBe(0.98);
    });
  });

  describe('DID Format Validation', () => {
    it('should accept did:mycelix format', async () => {
      const input: LinkPatientIdentityInput = {
        patient_hash: 'uhCkkTest',
        did: 'did:mycelix:abc123def456',
        identity_provider: 'Mycelix',
        verification_method: 'credential',
        confidence_score: 0.9,
      };

      vi.mocked(client.callZome).mockResolvedValue({
        ...input,
        verified_at: Date.now(),
      });

      const result = await bridge.linkPatientToIdentity(input);
      expect(result.did.startsWith('did:mycelix:')).toBe(true);
    });

    it('should accept did:web format', async () => {
      const input: LinkPatientIdentityInput = {
        patient_hash: 'uhCkkTest',
        did: 'did:web:hospital.example.com',
        identity_provider: 'Web Identity',
        verification_method: 'domain',
        confidence_score: 0.85,
      };

      vi.mocked(client.callZome).mockResolvedValue({
        ...input,
        verified_at: Date.now(),
      });

      const result = await bridge.linkPatientToIdentity(input);
      expect(result.did.startsWith('did:web:')).toBe(true);
    });

    it('should accept did:key format', async () => {
      const input: LinkPatientIdentityInput = {
        patient_hash: 'uhCkkTest',
        did: 'did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK',
        identity_provider: 'Key-based',
        verification_method: 'public_key',
        confidence_score: 0.88,
      };

      vi.mocked(client.callZome).mockResolvedValue({
        ...input,
        verified_at: Date.now(),
      });

      const result = await bridge.linkPatientToIdentity(input);
      expect(result.did.startsWith('did:key:')).toBe(true);
    });
  });

  describe('Singleton Pattern', () => {
    it('should return same instance for same client', () => {
      const bridge1 = getHealthIdentityBridgeClient(client);
      const bridge2 = getHealthIdentityBridgeClient(client);

      expect(bridge1).toBe(bridge2);
    });

    it('should reset instance correctly', () => {
      const bridge1 = getHealthIdentityBridgeClient(client);
      resetHealthIdentityBridgeClient();
      const bridge2 = getHealthIdentityBridgeClient(client);

      expect(bridge1).not.toBe(bridge2);
    });
  });

  describe('Edge Cases', () => {
    it('should handle very long DIDs', async () => {
      const longDID = `did:mycelix:${'a'.repeat(1000)}`;

      const input: LinkPatientIdentityInput = {
        patient_hash: 'uhCkkLongDID',
        did: longDID,
        identity_provider: 'Test',
        verification_method: 'test',
        confidence_score: 0.5,
      };

      vi.mocked(client.callZome).mockResolvedValue({
        ...input,
        verified_at: Date.now(),
      });

      const result = await bridge.linkPatientToIdentity(input);
      expect(result.did.length).toBeGreaterThan(1000);
    });

    it('should handle minimum confidence score', async () => {
      const input: LinkPatientIdentityInput = {
        patient_hash: 'uhCkkMinConfidence',
        did: 'did:mycelix:min',
        identity_provider: 'Test',
        verification_method: 'test',
        confidence_score: 0.0,
      };

      vi.mocked(client.callZome).mockResolvedValue({
        ...input,
        verified_at: Date.now(),
      });

      const result = await bridge.linkPatientToIdentity(input);
      expect(result.confidence_score).toBe(0.0);
    });

    it('should handle maximum confidence score', async () => {
      const input: LinkPatientIdentityInput = {
        patient_hash: 'uhCkkMaxConfidence',
        did: 'did:mycelix:max',
        identity_provider: 'Test',
        verification_method: 'test',
        confidence_score: 1.0,
      };

      vi.mocked(client.callZome).mockResolvedValue({
        ...input,
        verified_at: Date.now(),
      });

      const result = await bridge.linkPatientToIdentity(input);
      expect(result.confidence_score).toBe(1.0);
    });

    it('should handle special characters in identity provider', async () => {
      const input: LinkPatientIdentityInput = {
        patient_hash: 'uhCkkSpecialChars',
        did: 'did:mycelix:special',
        identity_provider: "Hospital's ID System (v2.0)",
        verification_method: 'oauth2/oidc',
        confidence_score: 0.75,
      };

      vi.mocked(client.callZome).mockResolvedValue({
        ...input,
        verified_at: Date.now(),
      });

      const result = await bridge.linkPatientToIdentity(input);
      expect(result.identity_provider).toBe("Hospital's ID System (v2.0)");
    });
  });

  describe('Error Handling', () => {
    it('should propagate zome call errors', async () => {
      const input: LinkPatientIdentityInput = {
        patient_hash: 'uhCkkError',
        did: 'did:mycelix:error',
        identity_provider: 'Test',
        verification_method: 'test',
        confidence_score: 0.5,
      };

      vi.mocked(client.callZome).mockRejectedValue(new Error('Zome error: unauthorized'));

      await expect(bridge.linkPatientToIdentity(input)).rejects.toThrow('Zome error: unauthorized');
    });

    it('should handle network errors', async () => {
      vi.mocked(client.callZome).mockRejectedValue(new Error('Network timeout'));

      await expect(
        bridge.getPatientByDID({ did: 'did:mycelix:timeout', is_emergency: false })
      ).rejects.toThrow('Network timeout');
    });

    it('should handle consent errors', async () => {
      vi.mocked(client.callZome).mockRejectedValue(
        new Error('Authorization failed: no valid consent found')
      );

      await expect(
        bridge.getPatientByDID({ did: 'did:mycelix:noconsent', is_emergency: false })
      ).rejects.toThrow('Authorization failed');
    });
  });
});
