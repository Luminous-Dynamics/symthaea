// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Decentralized Identity Service
 *
 * Provides:
 * - DID (Decentralized Identifier) management
 * - Verifiable Credentials issuance and verification
 * - Blockchain timestamp anchoring
 * - Zero-knowledge proof generation
 * - Identity resolution and linking
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// ============================================================================
// Types
// ============================================================================

export interface DID {
  id: string; // e.g., "did:mycelix:abc123"
  method: 'mycelix' | 'key' | 'web' | 'ion';
  publicKey: string;
  created: Date;
  updated: Date;
  controller?: string;
  service?: DIDService[];
  verificationMethod?: VerificationMethod[];
}

export interface DIDService {
  id: string;
  type: 'MessagingService' | 'LinkedDomains' | 'TrustAttestation';
  serviceEndpoint: string;
}

export interface VerificationMethod {
  id: string;
  type: 'Ed25519VerificationKey2020' | 'EcdsaSecp256k1VerificationKey2019';
  controller: string;
  publicKeyMultibase: string;
}

export interface VerifiableCredential {
  '@context': string[];
  id: string;
  type: string[];
  issuer: string | { id: string; name?: string };
  issuanceDate: string;
  expirationDate?: string;
  credentialSubject: {
    id: string;
    [key: string]: unknown;
  };
  proof?: CredentialProof;
}

export interface CredentialProof {
  type: string;
  created: string;
  verificationMethod: string;
  proofPurpose: string;
  proofValue: string;
}

export interface VerifiablePresentation {
  '@context': string[];
  type: string[];
  holder: string;
  verifiableCredential: VerifiableCredential[];
  proof?: CredentialProof;
}

export interface TrustCredential extends VerifiableCredential {
  credentialSubject: {
    id: string;
    trustScore: number;
    trustType: 'personal' | 'professional' | 'organization';
    attestations: string[];
    validFrom: string;
    validUntil?: string;
  };
}

export interface EmailVerificationCredential extends VerifiableCredential {
  credentialSubject: {
    id: string;
    email: string;
    verificationMethod: 'dns' | 'oauth' | 'confirmation';
    verifiedAt: string;
  };
}

export interface BlockchainAnchor {
  id: string;
  type: 'email' | 'attestation' | 'credential';
  resourceId: string;
  hash: string;
  timestamp: Date;
  blockHeight?: number;
  transactionId?: string;
  chain: 'bitcoin' | 'ethereum' | 'polygon' | 'mock';
  status: 'pending' | 'confirmed' | 'failed';
  confirmations: number;
}

export interface ZKProof {
  id: string;
  type: 'trust_threshold' | 'email_domain' | 'credential_possession';
  claim: string;
  proof: string;
  publicInputs: string[];
  verificationKey: string;
  created: Date;
}

export interface IdentityLink {
  id: string;
  did: string;
  linkedIdentity: {
    type: 'email' | 'domain' | 'social' | 'pgp';
    value: string;
    verified: boolean;
    verifiedAt?: Date;
    proof?: string;
  };
}

// ============================================================================
// DID Document Builder
// ============================================================================

class DIDDocumentBuilder {
  private did: Partial<DID> = {};
  private services: DIDService[] = [];
  private verificationMethods: VerificationMethod[] = [];

  setMethod(method: DID['method']): this {
    this.did.method = method;
    return this;
  }

  setPublicKey(publicKey: string): this {
    this.did.publicKey = publicKey;
    return this;
  }

  addService(service: DIDService): this {
    this.services.push(service);
    return this;
  }

  addVerificationMethod(method: VerificationMethod): this {
    this.verificationMethods.push(method);
    return this;
  }

  build(): DID {
    const id = `did:${this.did.method}:${this.generateIdentifier()}`;

    return {
      id,
      method: this.did.method || 'mycelix',
      publicKey: this.did.publicKey || '',
      created: new Date(),
      updated: new Date(),
      service: this.services,
      verificationMethod: this.verificationMethods.map((vm) => ({
        ...vm,
        id: `${id}#${vm.id}`,
        controller: id,
      })),
    };
  }

  private generateIdentifier(): string {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    return Array.from(bytes)
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  }
}

// ============================================================================
// Credential Builder
// ============================================================================

class CredentialBuilder {
  private credential: Partial<VerifiableCredential> = {
    '@context': ['https://www.w3.org/2018/credentials/v1'],
    type: ['VerifiableCredential'],
  };

  setId(id: string): this {
    this.credential.id = id;
    return this;
  }

  setIssuer(issuer: string | { id: string; name?: string }): this {
    this.credential.issuer = issuer;
    return this;
  }

  addType(type: string): this {
    this.credential.type = [...(this.credential.type || []), type];
    return this;
  }

  addContext(context: string): this {
    this.credential['@context'] = [...(this.credential['@context'] || []), context];
    return this;
  }

  setSubject(subject: VerifiableCredential['credentialSubject']): this {
    this.credential.credentialSubject = subject;
    return this;
  }

  setExpiration(date: Date): this {
    this.credential.expirationDate = date.toISOString();
    return this;
  }

  build(): VerifiableCredential {
    return {
      '@context': this.credential['@context'] || [],
      id: this.credential.id || `urn:uuid:${crypto.randomUUID()}`,
      type: this.credential.type || ['VerifiableCredential'],
      issuer: this.credential.issuer || '',
      issuanceDate: new Date().toISOString(),
      expirationDate: this.credential.expirationDate,
      credentialSubject: this.credential.credentialSubject || { id: '' },
    };
  }
}

// ============================================================================
// Identity Service
// ============================================================================

class DecentralizedIdentityService {
  private keyPair: CryptoKeyPair | null = null;

  async initialize(): Promise<void> {
    // Generate or load key pair
    this.keyPair = await crypto.subtle.generateKey(
      {
        name: 'ECDSA',
        namedCurve: 'P-256',
      },
      true,
      ['sign', 'verify']
    );
  }

  async createDID(method: DID['method'] = 'mycelix'): Promise<DID> {
    if (!this.keyPair) await this.initialize();

    const publicKeyBuffer = await crypto.subtle.exportKey('raw', this.keyPair!.publicKey);
    const publicKeyHex = Array.from(new Uint8Array(publicKeyBuffer))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');

    return new DIDDocumentBuilder()
      .setMethod(method)
      .setPublicKey(publicKeyHex)
      .addVerificationMethod({
        id: 'key-1',
        type: 'EcdsaSecp256k1VerificationKey2019',
        controller: '',
        publicKeyMultibase: `z${publicKeyHex}`,
      })
      .addService({
        id: 'messaging',
        type: 'MessagingService',
        serviceEndpoint: 'https://mycelix.mail/messaging',
      })
      .build();
  }

  async resolveDID(didString: string): Promise<DID | null> {
    // Parse DID
    const parts = didString.split(':');
    if (parts.length < 3 || parts[0] !== 'did') {
      return null;
    }

    const method = parts[1] as DID['method'];
    const identifier = parts.slice(2).join(':');

    // In production, this would resolve from the appropriate network
    // For now, return mock data
    return {
      id: didString,
      method,
      publicKey: identifier,
      created: new Date(),
      updated: new Date(),
    };
  }

  async issueCredential<T extends VerifiableCredential>(
    credential: Omit<T, 'proof'>,
    issuerDID: string
  ): Promise<T> {
    if (!this.keyPair) await this.initialize();

    // Create proof
    const proofData = JSON.stringify({
      ...credential,
      issuer: issuerDID,
    });

    const encoder = new TextEncoder();
    const data = encoder.encode(proofData);
    const signature = await crypto.subtle.sign(
      { name: 'ECDSA', hash: 'SHA-256' },
      this.keyPair!.privateKey,
      data
    );

    const proofValue = btoa(String.fromCharCode(...new Uint8Array(signature)));

    const proof: CredentialProof = {
      type: 'EcdsaSecp256k1Signature2019',
      created: new Date().toISOString(),
      verificationMethod: `${issuerDID}#key-1`,
      proofPurpose: 'assertionMethod',
      proofValue,
    };

    return {
      ...credential,
      proof,
    } as T;
  }

  async verifyCredential(credential: VerifiableCredential): Promise<{
    valid: boolean;
    checks: { name: string; passed: boolean; message?: string }[];
  }> {
    const checks: { name: string; passed: boolean; message?: string }[] = [];

    // Check structure
    checks.push({
      name: 'structure',
      passed: !!credential['@context'] && !!credential.type && !!credential.credentialSubject,
    });

    // Check expiration
    if (credential.expirationDate) {
      const expired = new Date(credential.expirationDate) < new Date();
      checks.push({
        name: 'expiration',
        passed: !expired,
        message: expired ? 'Credential has expired' : undefined,
      });
    }

    // Check proof (in production, would verify cryptographic signature)
    if (credential.proof) {
      checks.push({
        name: 'proof',
        passed: true, // Mock verification
      });
    } else {
      checks.push({
        name: 'proof',
        passed: false,
        message: 'No proof attached',
      });
    }

    return {
      valid: checks.every((c) => c.passed),
      checks,
    };
  }

  createTrustCredential(
    subjectDID: string,
    issuerDID: string,
    trustScore: number,
    trustType: TrustCredential['credentialSubject']['trustType']
  ): Omit<TrustCredential, 'proof'> {
    return new CredentialBuilder()
      .addType('TrustCredential')
      .addContext('https://mycelix.mail/credentials/trust/v1')
      .setIssuer(issuerDID)
      .setSubject({
        id: subjectDID,
        trustScore,
        trustType,
        attestations: [],
        validFrom: new Date().toISOString(),
      })
      .build() as Omit<TrustCredential, 'proof'>;
  }

  createEmailVerificationCredential(
    subjectDID: string,
    issuerDID: string,
    email: string,
    method: EmailVerificationCredential['credentialSubject']['verificationMethod']
  ): Omit<EmailVerificationCredential, 'proof'> {
    return new CredentialBuilder()
      .addType('EmailVerificationCredential')
      .addContext('https://mycelix.mail/credentials/email/v1')
      .setIssuer(issuerDID)
      .setSubject({
        id: subjectDID,
        email,
        verificationMethod: method,
        verifiedAt: new Date().toISOString(),
      })
      .build() as Omit<EmailVerificationCredential, 'proof'>;
  }
}

// ============================================================================
// Blockchain Anchoring Service
// ============================================================================

class BlockchainAnchorService {
  private pendingAnchors: BlockchainAnchor[] = [];

  async anchor(
    type: BlockchainAnchor['type'],
    resourceId: string,
    data: unknown,
    chain: BlockchainAnchor['chain'] = 'mock'
  ): Promise<BlockchainAnchor> {
    // Hash the data
    const encoder = new TextEncoder();
    const dataBuffer = encoder.encode(JSON.stringify(data));
    const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer);
    const hash = Array.from(new Uint8Array(hashBuffer))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');

    const anchor: BlockchainAnchor = {
      id: `anchor_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type,
      resourceId,
      hash,
      timestamp: new Date(),
      chain,
      status: 'pending',
      confirmations: 0,
    };

    this.pendingAnchors.push(anchor);

    // In production, this would submit to the actual blockchain
    // Simulate confirmation after delay
    setTimeout(() => {
      anchor.status = 'confirmed';
      anchor.confirmations = 1;
      anchor.blockHeight = Math.floor(Math.random() * 1000000);
      anchor.transactionId = `0x${hash.slice(0, 64)}`;
    }, 2000);

    return anchor;
  }

  async verify(anchor: BlockchainAnchor): Promise<{
    valid: boolean;
    blockConfirmed: boolean;
    timestampValid: boolean;
  }> {
    // In production, would verify against actual blockchain
    return {
      valid: anchor.status === 'confirmed',
      blockConfirmed: anchor.confirmations > 0,
      timestampValid: true,
    };
  }

  getPendingAnchors(): BlockchainAnchor[] {
    return this.pendingAnchors.filter((a) => a.status === 'pending');
  }
}

// ============================================================================
// Zero-Knowledge Proof Service
// ============================================================================

class ZKProofService {
  async generateTrustThresholdProof(
    trustScore: number,
    threshold: number,
    credential: TrustCredential
  ): Promise<ZKProof> {
    // In production, would use actual ZK proof library (e.g., snarkjs, circom)
    // This is a mock implementation

    const claim = `Trust score >= ${threshold}`;
    const proof = btoa(
      JSON.stringify({
        trustScoreHash: await this.hash(trustScore.toString()),
        thresholdHash: await this.hash(threshold.toString()),
        result: trustScore >= threshold,
      })
    );

    return {
      id: `zkproof_${Date.now()}`,
      type: 'trust_threshold',
      claim,
      proof,
      publicInputs: [threshold.toString()],
      verificationKey: 'mock_verification_key',
      created: new Date(),
    };
  }

  async generateDomainProof(email: string, allowedDomains: string[]): Promise<ZKProof> {
    const domain = email.split('@')[1];
    const isAllowed = allowedDomains.includes(domain);

    const claim = `Email domain is in allowed list`;
    const proof = btoa(
      JSON.stringify({
        domainHash: await this.hash(domain),
        allowedDomainsHash: await this.hash(allowedDomains.join(',')),
        result: isAllowed,
      })
    );

    return {
      id: `zkproof_${Date.now()}`,
      type: 'email_domain',
      claim,
      proof,
      publicInputs: allowedDomains,
      verificationKey: 'mock_verification_key',
      created: new Date(),
    };
  }

  async verifyProof(proof: ZKProof): Promise<boolean> {
    // In production, would verify using ZK verification
    try {
      const decoded = JSON.parse(atob(proof.proof));
      return decoded.result === true;
    } catch {
      return false;
    }
  }

  private async hash(data: string): Promise<string> {
    const encoder = new TextEncoder();
    const dataBuffer = encoder.encode(data);
    const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer);
    return Array.from(new Uint8Array(hashBuffer))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  }
}

// ============================================================================
// Store
// ============================================================================

interface DecentralizedIdentityState {
  myDID: DID | null;
  credentials: VerifiableCredential[];
  issuedCredentials: VerifiableCredential[];
  identityLinks: IdentityLink[];
  anchors: BlockchainAnchor[];
  zkProofs: ZKProof[];

  setMyDID: (did: DID) => void;
  addCredential: (credential: VerifiableCredential) => void;
  removeCredential: (id: string) => void;
  addIssuedCredential: (credential: VerifiableCredential) => void;
  addIdentityLink: (link: IdentityLink) => void;
  addAnchor: (anchor: BlockchainAnchor) => void;
  updateAnchor: (id: string, updates: Partial<BlockchainAnchor>) => void;
  addZKProof: (proof: ZKProof) => void;
}

export const useDecentralizedIdentityStore = create<DecentralizedIdentityState>()(
  persist(
    (set) => ({
      myDID: null,
      credentials: [],
      issuedCredentials: [],
      identityLinks: [],
      anchors: [],
      zkProofs: [],

      setMyDID: (did) => set({ myDID: did }),

      addCredential: (credential) =>
        set((state) => ({
          credentials: [...state.credentials, credential],
        })),

      removeCredential: (id) =>
        set((state) => ({
          credentials: state.credentials.filter((c) => c.id !== id),
        })),

      addIssuedCredential: (credential) =>
        set((state) => ({
          issuedCredentials: [...state.issuedCredentials, credential],
        })),

      addIdentityLink: (link) =>
        set((state) => ({
          identityLinks: [...state.identityLinks, link],
        })),

      addAnchor: (anchor) =>
        set((state) => ({
          anchors: [...state.anchors, anchor],
        })),

      updateAnchor: (id, updates) =>
        set((state) => ({
          anchors: state.anchors.map((a) => (a.id === id ? { ...a, ...updates } : a)),
        })),

      addZKProof: (proof) =>
        set((state) => ({
          zkProofs: [...state.zkProofs, proof],
        })),
    }),
    {
      name: 'mycelix-decentralized-identity',
    }
  )
);

// ============================================================================
// Singleton Services
// ============================================================================

const identityService = new DecentralizedIdentityService();
const anchorService = new BlockchainAnchorService();
const zkProofService = new ZKProofService();

// ============================================================================
// React Hooks
// ============================================================================

import { useState, useCallback, useEffect } from 'react';

export function useDecentralizedIdentity() {
  const { myDID, setMyDID, credentials, addCredential } = useDecentralizedIdentityStore();
  const [isInitializing, setIsInitializing] = useState(false);

  const initialize = useCallback(async () => {
    if (myDID) return myDID;

    setIsInitializing(true);
    try {
      await identityService.initialize();
      const did = await identityService.createDID();
      setMyDID(did);
      return did;
    } finally {
      setIsInitializing(false);
    }
  }, [myDID, setMyDID]);

  const resolveDID = useCallback(async (didString: string) => {
    return identityService.resolveDID(didString);
  }, []);

  return {
    myDID,
    credentials,
    isInitializing,
    initialize,
    resolveDID,
    addCredential,
  };
}

export function useVerifiableCredentials() {
  const { myDID, credentials, addCredential, addIssuedCredential, issuedCredentials } =
    useDecentralizedIdentityStore();

  const issueTrustCredential = useCallback(
    async (
      subjectDID: string,
      trustScore: number,
      trustType: TrustCredential['credentialSubject']['trustType']
    ) => {
      if (!myDID) throw new Error('DID not initialized');

      const credential = identityService.createTrustCredential(
        subjectDID,
        myDID.id,
        trustScore,
        trustType
      );

      const signedCredential = await identityService.issueCredential<TrustCredential>(
        credential,
        myDID.id
      );

      addIssuedCredential(signedCredential);
      return signedCredential;
    },
    [myDID, addIssuedCredential]
  );

  const issueEmailVerification = useCallback(
    async (
      subjectDID: string,
      email: string,
      method: EmailVerificationCredential['credentialSubject']['verificationMethod']
    ) => {
      if (!myDID) throw new Error('DID not initialized');

      const credential = identityService.createEmailVerificationCredential(
        subjectDID,
        myDID.id,
        email,
        method
      );

      const signedCredential = await identityService.issueCredential<EmailVerificationCredential>(
        credential,
        myDID.id
      );

      addIssuedCredential(signedCredential);
      return signedCredential;
    },
    [myDID, addIssuedCredential]
  );

  const verifyCredential = useCallback(async (credential: VerifiableCredential) => {
    return identityService.verifyCredential(credential);
  }, []);

  return {
    credentials,
    issuedCredentials,
    issueTrustCredential,
    issueEmailVerification,
    verifyCredential,
    addCredential,
  };
}

export function useBlockchainAnchoring() {
  const { anchors, addAnchor, updateAnchor } = useDecentralizedIdentityStore();

  const anchorEmail = useCallback(
    async (emailId: string, emailHash: string) => {
      const anchor = await anchorService.anchor('email', emailId, { hash: emailHash });
      addAnchor(anchor);
      return anchor;
    },
    [addAnchor]
  );

  const anchorAttestation = useCallback(
    async (attestationId: string, attestationData: unknown) => {
      const anchor = await anchorService.anchor('attestation', attestationId, attestationData);
      addAnchor(anchor);
      return anchor;
    },
    [addAnchor]
  );

  const verifyAnchor = useCallback(async (anchor: BlockchainAnchor) => {
    return anchorService.verify(anchor);
  }, []);

  // Poll for confirmations
  useEffect(() => {
    const pending = anchors.filter((a) => a.status === 'pending');
    if (pending.length === 0) return;

    const interval = setInterval(() => {
      // In production, would check actual blockchain
      pending.forEach((anchor) => {
        if (Math.random() > 0.7) {
          updateAnchor(anchor.id, {
            status: 'confirmed',
            confirmations: 1,
            blockHeight: Math.floor(Math.random() * 1000000),
          });
        }
      });
    }, 5000);

    return () => clearInterval(interval);
  }, [anchors, updateAnchor]);

  return {
    anchors,
    pendingAnchors: anchors.filter((a) => a.status === 'pending'),
    confirmedAnchors: anchors.filter((a) => a.status === 'confirmed'),
    anchorEmail,
    anchorAttestation,
    verifyAnchor,
  };
}

export function useZeroKnowledgeProofs() {
  const { zkProofs, addZKProof } = useDecentralizedIdentityStore();

  const proveTrustThreshold = useCallback(
    async (trustScore: number, threshold: number, credential: TrustCredential) => {
      const proof = await zkProofService.generateTrustThresholdProof(
        trustScore,
        threshold,
        credential
      );
      addZKProof(proof);
      return proof;
    },
    [addZKProof]
  );

  const proveDomainMembership = useCallback(
    async (email: string, allowedDomains: string[]) => {
      const proof = await zkProofService.generateDomainProof(email, allowedDomains);
      addZKProof(proof);
      return proof;
    },
    [addZKProof]
  );

  const verifyProof = useCallback(async (proof: ZKProof) => {
    return zkProofService.verifyProof(proof);
  }, []);

  return {
    zkProofs,
    proveTrustThreshold,
    proveDomainMembership,
    verifyProof,
  };
}
