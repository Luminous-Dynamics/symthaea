// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * End-to-End Smoke Test
 *
 * Validates the entire chain the SMS gateway relies on:
 * seed knowledge → search → agent registration → reputation → trust scoring
 */

import { describe, it, expect } from 'vitest';
import { runScenario } from '@holochain/tryorama';
import type { ActionHash } from '@holochain/client';
import { setupPlayers, callKnowledge, callReputation, sampleAgent } from './setup.js';

// Subset of seed data for the smoke test
const seedEntries = [
  {
    domain: 'benefits',
    knowledge_type: 'eligibility_rule',
    title: 'SNAP (Food Stamps) Eligibility',
    content: 'SNAP eligibility is based on gross monthly income at or below 130% of the federal poverty level.',
    geographic_scope: 'United States',
    keywords: ['snap', 'food stamps', 'food assistance', 'ebt'],
    source: 'USDA Food and Nutrition Service',
    links: ['https://www.fns.usda.gov/snap/recipient/eligibility'],
    contact_phone: '1-800-221-5689',
  },
  {
    domain: 'benefits',
    knowledge_type: 'eligibility_rule',
    title: 'Medicaid Eligibility',
    content: 'Medicaid provides free or low-cost health coverage for adults with income up to 138% FPL.',
    geographic_scope: 'United States',
    keywords: ['medicaid', 'health insurance', 'healthcare'],
    source: 'CMS',
    links: ['https://www.medicaid.gov/medicaid/eligibility/index.html'],
    contact_phone: '1-877-267-2323',
  },
  {
    domain: 'benefits',
    knowledge_type: 'process',
    title: 'How to Apply for Benefits - Texas',
    content: 'In Texas, apply for SNAP, Medicaid, and TANF through YourTexasBenefits.com.',
    geographic_scope: 'Texas',
    keywords: ['apply', 'texas benefits', 'yourtexasbenefits'],
    source: 'Texas HHS',
    links: ['https://www.yourtexasbenefits.com'],
    contact_phone: '2-1-1',
  },
  {
    domain: 'benefits',
    knowledge_type: 'resource',
    title: 'WIC Program',
    content: 'WIC provides supplemental foods and nutrition education for low-income pregnant women and children up to 5.',
    geographic_scope: 'United States',
    keywords: ['wic', 'women infants children', 'nutrition'],
    source: 'USDA',
    links: ['https://www.fns.usda.gov/wic'],
  },
  {
    domain: 'benefits',
    knowledge_type: 'resource',
    title: 'SSI Disability Benefits',
    content: 'SSI provides monthly payments to adults and children with disabilities who have limited income.',
    geographic_scope: 'United States',
    keywords: ['ssi', 'disability', 'social security'],
    source: 'SSA',
    links: ['https://www.ssa.gov/ssi'],
    contact_phone: '1-800-772-1213',
  },
];

describe('End-to-End Smoke Test', () => {
  it('full path: seed → search → agent → reputation → trust', async () => {
    await runScenario(async (scenario) => {
      const [alice, bob] = await setupPlayers(scenario, 2);
      const alicePubkey = alice.cells[0].cell_id[1];

      // Step 1: Seed knowledge entries
      const hashes: ActionHash[] = [];
      for (const entry of seedEntries) {
        const hash: ActionHash = await callKnowledge(alice, 'create_knowledge', entry);
        expect(hash).toBeDefined();
        hashes.push(hash);
      }
      expect(hashes.length).toBe(5);

      // Step 2: Search by domain → verify results
      const domainResults: any[] = await callKnowledge(
        alice,
        'search_by_domain',
        'benefits',
      );
      expect(domainResults.length).toBe(5);
      expect(domainResults.every((r: any) => r.knowledge.domain === 'benefits')).toBe(true);

      // Step 3: Search by keyword → verify relevant results
      const snapResults: any[] = await callKnowledge(
        alice,
        'search_by_keyword',
        'snap',
      );
      expect(snapResults.length).toBeGreaterThanOrEqual(1);
      expect(snapResults[0].knowledge.title).toContain('SNAP');

      // Step 4: Register agent
      const agentHash: ActionHash = await callReputation(
        alice,
        'register_agent',
        sampleAgent,
      );
      expect(agentHash).toBeDefined();

      // Step 5: Record reputation events (helpful + accurate)
      await callReputation(bob, 'record_event', {
        agent_pubkey: alicePubkey,
        event_type: 'helpful',
        context: 'Provided accurate SNAP info',
        conversation_id: 'smoke-001',
      });
      await callReputation(bob, 'record_event', {
        agent_pubkey: alicePubkey,
        event_type: 'accurate',
        context: 'Medicaid eligibility was correct',
        conversation_id: 'smoke-002',
      });
      await callReputation(bob, 'record_event', {
        agent_pubkey: alicePubkey,
        event_type: 'helpful',
        conversation_id: 'smoke-003',
      });

      // Allow DHT propagation between agents
      await new Promise((r) => setTimeout(r, 3000));

      // Step 6: Compute trust score and verify MATL formula
      const score: any = await callReputation(
        alice,
        'compute_trust_score',
        alicePubkey,
      );

      expect(score).toBeDefined();
      expect(score.positive_count).toBeGreaterThanOrEqual(3);
      expect(score.negative_count).toBe(0);
      expect(score.composite).toBeGreaterThan(0);
      expect(score.composite).toBeLessThanOrEqual(1);

      // Verify MATL formula
      const expectedComposite =
        0.4 * score.quality + 0.3 * score.consistency + 0.3 * score.reputation;
      expect(score.composite).toBeCloseTo(expectedComposite, 1);

      // Step 7: Search by location
      const texasResults: any[] = await callKnowledge(
        alice,
        'search_by_location',
        'Texas',
      );
      expect(texasResults.length).toBeGreaterThanOrEqual(1);
      expect(texasResults[0].knowledge.geographic_scope).toBe('Texas');
    });
  });
});
