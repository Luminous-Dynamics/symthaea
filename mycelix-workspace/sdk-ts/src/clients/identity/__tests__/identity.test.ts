/**
 * Identity Client Tests
 *
 * Verifies zome call arguments and record mapping for
 * DidClient and TrustClient (the two most critical identity clients).
 *
 * Identity clients extend ZomeClient with custom record mappers that return
 * { hash, document/credential } wrappers instead of raw entries.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { DidClient } from '../did';
import { TrustClient } from '../trust';
import type { AppClient } from '@holochain/client';

// ============================================================================
// MOCK HELPERS
// ============================================================================

function createMockClient(): AppClient {
  return {
    callZome: vi.fn(),
  } as unknown as AppClient;
}

function mockRecord(entry: Record<string, unknown>) {
  return {
    entry: { Present: entry },
    signed_action: { hashed: { hash: new Uint8Array(32) } },
  };
}

// ============================================================================
// MOCK ENTRIES (snake_case, matching Rust serde output)
// ============================================================================

const DID_DOCUMENT_ENTRY = {
  id: 'did:mycelix:alice123',
  controller: 'did:mycelix:alice123',
  verificationMethod: [
    {
      id: 'did:mycelix:alice123#key-1',
      type: 'Ed25519VerificationKey2020',
      controller: 'did:mycelix:alice123',
      publicKeyMultibase: 'z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK',
    },
  ],
  authentication: ['did:mycelix:alice123#key-1'],
  keyAgreement: [],
  service: [],
  created: 1708200000,
  updated: 1708200000,
  version: 1,
};

const DEACTIVATION_ENTRY = {
  did: 'did:mycelix:alice123',
  reason: 'Account compromised',
  deactivated_at: 1708300000,
};

const TRUST_CREDENTIAL_ENTRY = {
  id: 'tc-001',
  subjectDid: 'did:mycelix:alice123',
  issuerDid: 'did:mycelix:validator456',
  kvectorCommitment: new Uint8Array(64),
  rangeProof: new Uint8Array(128),
  trustScoreRange: { lower: 0.7, upper: 0.85 },
  trustTier: 'High',
  issuedAt: 1708200000,
  expiresAt: 1739736000,
  revoked: false,
};

const TRUST_PRESENTATION_ENTRY = {
  id: 'tp-001',
  credentialId: 'tc-001',
  subjectDid: 'did:mycelix:alice123',
  disclosedTier: 'High',
  disclosedRange: null,
  presentationProof: new Uint8Array(64),
  verifierDid: 'did:mycelix:marketplace001',
  purpose: 'Marketplace access verification',
  presentedAt: 1708200000,
  nonce: 'abc123',
};

// ============================================================================
// DID CLIENT TESTS
// ============================================================================

describe('DidClient', () => {
  let mockClient: AppClient;
  let did: DidClient;

  beforeEach(() => {
    mockClient = createMockClient();
    did = new DidClient(mockClient);
  });

  it('createDid calls correct zome and returns DidRecord', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockRecord(DID_DOCUMENT_ENTRY)
    );

    const result = await did.createDid();

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'identity',
      zome_name: 'did_registry',
      fn_name: 'create_did',
      payload: null,
    });
    // recordToDidRecord extracts { hash, document }
    expect(result.hash).toBeInstanceOf(Uint8Array);
    expect(result.document.id).toBe('did:mycelix:alice123');
    expect(result.document.controller).toBe('did:mycelix:alice123');
  });

  it('resolveDid resolves DID to document', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockRecord(DID_DOCUMENT_ENTRY)
    );

    const result = await did.resolveDid('did:mycelix:alice123');

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'identity',
      zome_name: 'did_registry',
      fn_name: 'resolve_did',
      payload: 'did:mycelix:alice123',
    });
    expect(result).not.toBeNull();
    expect(result!.document.verificationMethod).toHaveLength(1);
  });

  it('isDidActive returns boolean', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(true);

    const result = await did.isDidActive('did:mycelix:alice123');

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'identity',
      zome_name: 'did_registry',
      fn_name: 'is_did_active',
      payload: 'did:mycelix:alice123',
    });
    expect(result).toBe(true);
  });

  it('deactivateDid returns deactivation entry', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockRecord(DEACTIVATION_ENTRY)
    );

    const result = await did.deactivateDid('Account compromised');

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'identity',
      zome_name: 'did_registry',
      fn_name: 'deactivate_did',
      payload: 'Account compromised',
    });
    expect(result.reason).toBe('Account compromised');
  });
});

// ============================================================================
// TRUST CLIENT TESTS
// ============================================================================

describe('TrustClient', () => {
  let mockClient: AppClient;
  let trust: TrustClient;

  beforeEach(() => {
    mockClient = createMockClient();
    trust = new TrustClient(mockClient);
  });

  it('issueTrustCredential calls correct zome and returns record', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockRecord(TRUST_CREDENTIAL_ENTRY)
    );

    const input = {
      subject_did: 'did:mycelix:alice123',
      issuer_did: 'did:mycelix:validator456',
      kvector_commitment: new Uint8Array(64),
      range_proof: new Uint8Array(128),
      trust_score_lower: 0.7,
      trust_score_upper: 0.85,
      expires_at: 1739736000,
    };
    const result = await trust.issueTrustCredential(input);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'identity',
      zome_name: 'trust_credential',
      fn_name: 'issue_trust_credential',
      payload: input,
    });
    // recordToCredentialRecord extracts { hash, credential }
    expect(result.hash).toBeInstanceOf(Uint8Array);
    expect(result.credential.trustTier).toBe('High');
    expect(result.credential.revoked).toBe(false);
  });

  it('getSubjectCredentials returns credential list', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
      mockRecord(TRUST_CREDENTIAL_ENTRY),
    ]);

    const result = await trust.getSubjectCredentials('did:mycelix:alice123');

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'identity',
      zome_name: 'trust_credential',
      fn_name: 'get_subject_credentials',
      payload: 'did:mycelix:alice123',
    });
    expect(result).toHaveLength(1);
    expect(result[0].credential.subjectDid).toBe('did:mycelix:alice123');
  });

  it('createPresentation returns presentation record', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockRecord(TRUST_PRESENTATION_ENTRY)
    );

    const input = {
      credential_id: 'tc-001',
      subject_did: 'did:mycelix:alice123',
      disclosed_tier: 'High' as const,
      disclose_range: false,
      trust_range: { lower: 0.7, upper: 0.85 },
      presentation_proof: new Uint8Array(64),
      verifier_did: 'did:mycelix:marketplace001',
      purpose: 'Marketplace access verification',
    };
    const result = await trust.createPresentation(input);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'identity',
      zome_name: 'trust_credential',
      fn_name: 'create_presentation',
      payload: input,
    });
    expect(result.hash).toBeInstanceOf(Uint8Array);
    expect(result.presentation.disclosedTier).toBe('High');
    expect(result.presentation.verifierDid).toBe('did:mycelix:marketplace001');
  });
});
