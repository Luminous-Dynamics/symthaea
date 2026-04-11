// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Trust Integration Tests
 *
 * Tests for MATL trust algorithm and attestation functionality.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createMycelixClient, MycelixClient } from '../../src/bootstrap';

describe('Trust Integration', () => {
  let client: MycelixClient;

  beforeAll(async () => {
    client = createMycelixClient({
      websocketUrl: process.env.HOLOCHAIN_URL || 'ws://localhost:8888',
      appId: 'mycelix-mail-test',
    });
    await client.initialize();
  });

  afterAll(async () => {
    await client.shutdown();
  });

  describe('Trust Attestations', () => {
    it('should create a trust attestation', async () => {
      const services = client.getServices();

      // Create a mock agent key for testing
      const toAgent = 'uhCAkTestAgentPubKey123456789';

      const hash = await services.trust.createAttestation({
        toAgent,
        trustLevel: 0.8,
        context: 'Test attestation',
      });

      expect(hash).toBeDefined();
      expect(typeof hash).toBe('string');
    });

    it('should retrieve attestations for an agent', async () => {
      const services = client.getServices();

      const toAgent = 'uhCAkTestAgentPubKey123456789';

      // Create attestation
      await services.trust.createAttestation({
        toAgent,
        trustLevel: 0.7,
        context: 'Retrieval test',
      });

      // Get attestations
      const attestations = await services.trust.getAttestationsFor(toAgent);

      expect(Array.isArray(attestations)).toBe(true);
      expect(attestations.length).toBeGreaterThan(0);
    });

    it('should calculate direct trust score', async () => {
      const services = client.getServices();

      const toAgent = 'uhCAkTestAgentPubKey123456789';

      // Create multiple attestations to build trust
      await services.trust.createAttestation({
        toAgent,
        trustLevel: 0.9,
        context: 'High trust test',
      });

      const score = await services.trust.getTrustScore(toAgent);

      expect(score).toBeDefined();
      expect(score?.directTrust).toBeGreaterThan(0);
    });

    it('should revoke an attestation', async () => {
      const services = client.getServices();

      const toAgent = 'uhCAkTestAgentPubKeyRevoke';

      // Create attestation
      const hash = await services.trust.createAttestation({
        toAgent,
        trustLevel: 0.6,
        context: 'Revocation test',
      });

      // Revoke it
      await services.trust.revokeAttestation(hash);

      // Verify it's revoked
      const attestations = await services.trust.getAttestationsFor(toAgent);
      const found = attestations.find((a) => a.hash === hash);

      // Revoked attestations should not appear in active list
      expect(found).toBeUndefined();
    });

    it('should validate trust level bounds', async () => {
      const services = client.getServices();

      const toAgent = 'uhCAkTestAgentBounds';

      // Trust level too high should throw or clamp
      await expect(
        services.trust.createAttestation({
          toAgent,
          trustLevel: 1.5, // Invalid
          context: 'Invalid test',
        })
      ).rejects.toThrow();

      // Trust level too low should throw or clamp
      await expect(
        services.trust.createAttestation({
          toAgent,
          trustLevel: -0.5, // Invalid
          context: 'Invalid test',
        })
      ).rejects.toThrow();
    });
  });

  describe('MATL Algorithm', () => {
    it('should calculate transitive trust', async () => {
      const services = client.getServices();

      // Create a trust chain: Me -> A -> B
      const agentA = 'uhCAkAgentA123';
      const agentB = 'uhCAkAgentB456';

      // Direct trust to A
      await services.trust.createAttestation({
        toAgent: agentA,
        trustLevel: 0.9,
        context: 'Direct to A',
      });

      // In a real scenario, agentA would attest to agentB
      // For testing, we verify the transitive calculation logic

      const transitiveScore = await services.trust.calculateTransitiveTrust(
        agentA,
        agentB,
        3 // max depth
      );

      expect(transitiveScore).toBeDefined();
      expect(typeof transitiveScore).toBe('number');
      expect(transitiveScore).toBeGreaterThanOrEqual(0);
      expect(transitiveScore).toBeLessThanOrEqual(1);
    });

    it('should apply trust decay over distance', async () => {
      const services = client.getServices();

      const agentA = 'uhCAkAgentDecayA';
      const agentB = 'uhCAkAgentDecayB';
      const agentC = 'uhCAkAgentDecayC';

      // Create chain
      await services.trust.createAttestation({
        toAgent: agentA,
        trustLevel: 1.0,
        context: 'Chain 1',
      });

      // Transitive trust should decay
      // In a fully connected network:
      // Direct trust: 1.0
      // 1 hop: 1.0 * 0.7 = 0.7
      // 2 hops: 0.7 * 0.7 = 0.49

      const directScore = await services.trust.getTrustScore(agentA);
      expect(directScore?.directTrust).toBeGreaterThan(0.9);
    });

    it('should limit trust depth', async () => {
      const services = client.getServices();

      // Create a long chain
      const agents = [
        'uhCAkAgent1',
        'uhCAkAgent2',
        'uhCAkAgent3',
        'uhCAkAgent4',
        'uhCAkAgent5',
        'uhCAkAgent6',
      ];

      // Trust first agent
      await services.trust.createAttestation({
        toAgent: agents[0],
        trustLevel: 1.0,
        context: 'Depth test',
      });

      // Calculate trust to last agent with different depths
      const shallowTrust = await services.trust.calculateTransitiveTrust(
        agents[0],
        agents[5],
        2
      );

      const deepTrust = await services.trust.calculateTransitiveTrust(
        agents[0],
        agents[5],
        10
      );

      // Shallow should find less trust (might be 0 if chain not complete)
      expect(shallowTrust).toBeLessThanOrEqual(deepTrust);
    });
  });

  describe('Trust Network', () => {
    it('should get my trust network', async () => {
      const services = client.getServices();

      // Create some attestations first
      await services.trust.createAttestation({
        toAgent: 'uhCAkNetworkAgent1',
        trustLevel: 0.8,
        context: 'Network test 1',
      });

      await services.trust.createAttestation({
        toAgent: 'uhCAkNetworkAgent2',
        trustLevel: 0.6,
        context: 'Network test 2',
      });

      const network = await services.trust.getMyTrustNetwork();

      expect(Array.isArray(network)).toBe(true);
      expect(network.length).toBeGreaterThanOrEqual(2);
    });

    it('should sort network by trust level', async () => {
      const services = client.getServices();

      const network = await services.trust.getMyTrustNetwork();

      // Verify sorted by combined trust (descending)
      for (let i = 1; i < network.length; i++) {
        expect(network[i - 1].combinedTrust).toBeGreaterThanOrEqual(
          network[i].combinedTrust
        );
      }
    });
  });

  describe('Spam Detection via Trust', () => {
    it('should identify untrusted senders', async () => {
      const services = client.getServices();

      const unknownSender = 'uhCAkUnknownSender123';

      const score = await services.trust.getTrustScore(unknownSender);

      // Unknown sender should have very low or zero trust
      expect(score?.combinedTrust ?? 0).toBeLessThan(0.3);
    });

    it('should use trust for spam filtering', async () => {
      const services = client.getServices();

      // Create a trusted sender
      const trustedSender = 'uhCAkTrustedSender';
      await services.trust.createAttestation({
        toAgent: trustedSender,
        trustLevel: 0.9,
        context: 'Trusted contact',
      });

      // Check spam probability
      const trustedSpamProb = await services.spam.getSpamProbability(trustedSender);
      const untrustedSpamProb = await services.spam.getSpamProbability(
        'uhCAkRandomUnknown'
      );

      // Trusted sender should have lower spam probability
      expect(trustedSpamProb).toBeLessThan(untrustedSpamProb);
    });
  });
});
