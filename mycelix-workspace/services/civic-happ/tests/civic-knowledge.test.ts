import { describe, it, expect } from 'vitest';
import { runScenario } from '@holochain/tryorama';
import type { ActionHash } from '@holochain/client';
import { setupPlayers, callKnowledge, sampleKnowledge } from './setup.js';

describe('Civic Knowledge Zome', () => {
  it('create_knowledge + get_knowledge round-trip', async () => {
    await runScenario(async (scenario) => {
      const [alice] = await setupPlayers(scenario, 1);

      const actionHash: ActionHash = await callKnowledge(
        alice,
        'create_knowledge',
        sampleKnowledge,
      );
      expect(actionHash).toBeDefined();
      expect(actionHash.length).toBeGreaterThan(0);

      const retrieved = await callKnowledge(alice, 'get_knowledge', actionHash);
      expect(retrieved).toBeDefined();
      expect((retrieved as any).title).toBe(sampleKnowledge.title);
      expect((retrieved as any).domain).toBe(sampleKnowledge.domain);
      expect((retrieved as any).content).toBe(sampleKnowledge.content);
    });
  });

  it('search_by_domain returns correct results', async () => {
    await runScenario(async (scenario) => {
      const [alice] = await setupPlayers(scenario, 1);

      // Create entries in two domains
      await callKnowledge(alice, 'create_knowledge', sampleKnowledge);
      await callKnowledge(alice, 'create_knowledge', {
        ...sampleKnowledge,
        domain: 'voting',
        title: 'Voter Registration',
        keywords: ['voting', 'register'],
      });

      const benefitsResults: any[] = await callKnowledge(
        alice,
        'search_by_domain',
        'benefits',
      );
      expect(benefitsResults.length).toBe(1);
      expect(benefitsResults[0].knowledge.domain).toBe('benefits');

      const votingResults: any[] = await callKnowledge(
        alice,
        'search_by_domain',
        'voting',
      );
      expect(votingResults.length).toBe(1);
      expect(votingResults[0].knowledge.domain).toBe('voting');
    });
  });

  it('search_by_location with geographic scope filtering', async () => {
    await runScenario(async (scenario) => {
      const [alice] = await setupPlayers(scenario, 1);

      await callKnowledge(alice, 'create_knowledge', {
        ...sampleKnowledge,
        geographic_scope: 'Texas',
        title: 'Texas Benefits',
      });
      await callKnowledge(alice, 'create_knowledge', {
        ...sampleKnowledge,
        geographic_scope: 'California',
        title: 'California Benefits',
      });

      const texasResults: any[] = await callKnowledge(
        alice,
        'search_by_location',
        'Texas',
      );
      expect(texasResults.length).toBe(1);
      expect(texasResults[0].knowledge.title).toBe('Texas Benefits');
    });
  });

  it('search_by_keyword matches title/content keywords', async () => {
    await runScenario(async (scenario) => {
      const [alice] = await setupPlayers(scenario, 1);

      await callKnowledge(alice, 'create_knowledge', sampleKnowledge);
      await callKnowledge(alice, 'create_knowledge', {
        ...sampleKnowledge,
        title: 'Medicaid Info',
        keywords: ['medicaid', 'health'],
      });

      const snapResults: any[] = await callKnowledge(
        alice,
        'search_by_keyword',
        'snap',
      );
      expect(snapResults.length).toBe(1);
      expect(snapResults[0].knowledge.title).toContain('SNAP');

      const medicaidResults: any[] = await callKnowledge(
        alice,
        'search_by_keyword',
        'medicaid',
      );
      expect(medicaidResults.length).toBe(1);
    });
  });

  it('validate_knowledge records authority validation', async () => {
    await runScenario(async (scenario) => {
      const [alice] = await setupPlayers(scenario, 1);

      const hash: ActionHash = await callKnowledge(
        alice,
        'create_knowledge',
        sampleKnowledge,
      );

      const validationHash: ActionHash = await callKnowledge(
        alice,
        'validate_knowledge',
        {
          knowledge_hash: hash,
          is_valid: true,
          authority_type: 'government_agency',
          notes: 'Verified with USDA data',
        },
      );
      expect(validationHash).toBeDefined();

      const validations: any[] = await callKnowledge(
        alice,
        'get_knowledge_validations',
        hash,
      );
      expect(validations.length).toBe(1);
      expect(validations[0].is_valid).toBe(true);
      expect(validations[0].authority_type).toBe('government_agency');
    });
  });

  it('search_knowledge combined multi-criteria search', async () => {
    await runScenario(async (scenario) => {
      const [alice] = await setupPlayers(scenario, 1);

      await callKnowledge(alice, 'create_knowledge', sampleKnowledge);
      await callKnowledge(alice, 'create_knowledge', {
        ...sampleKnowledge,
        domain: 'tax',
        geographic_scope: 'Texas',
        title: 'Texas Tax Info',
        keywords: ['tax', 'texas'],
      });

      const results: any[] = await callKnowledge(alice, 'search_knowledge', {
        domain: 'benefits',
        geo_scope: 'Texas',
        keywords: ['snap'],
        limit: 10,
      });
      expect(results.length).toBeGreaterThanOrEqual(1);
      expect(results[0].knowledge.domain).toBe('benefits');
    });
  });
});
