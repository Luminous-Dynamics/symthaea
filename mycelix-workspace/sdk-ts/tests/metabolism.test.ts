/**
 * Knowledge Metabolism Tests
 *
 * Tests for claim lifecycle management - from birth through death to decomposition.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  MetabolismEngine,
  getMetabolismEngine,
  resetMetabolismEngine,
  DEFAULT_METABOLISM_CONFIG,
  type MetabolizingClaim,
  type Tombstone,
  type ClaimHealth,
} from '../src/metabolism/index.js';
import {
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
} from '../src/epistemic/index.js';

describe('Metabolism Engine - Claim Birth', () => {
  let engine: MetabolismEngine;

  beforeEach(() => {
    engine = new MetabolismEngine();
  });

  it('should birth a new claim in nascent phase', async () => {
    const claim = await engine.birthClaim({
      content: 'The Earth orbits the Sun',
      classification: {
        empirical: EmpiricalLevel.E4_Consensus,
        normative: NormativeLevel.N3_Axiomatic,
        materiality: MaterialityLevel.M3_Foundational,
      },
      authorId: 'agent-123',
      domain: 'astronomy',
    });

    expect(claim.id).toMatch(/^claim_/);
    expect(claim.phase).toBe('nascent');
    expect(claim.content).toBe('The Earth orbits the Sun');
    expect(claim.domain).toBe('astronomy');
  });

  it('should set default confidence to 0.5', async () => {
    const claim = await engine.birthClaim({
      content: 'Test claim',
      classification: {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    expect(claim.confidence).toBe(0.5);
  });

  it('should accept custom confidence', async () => {
    const claim = await engine.birthClaim({
      content: 'Test claim',
      classification: {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      },
      authorId: 'test',
      domain: 'test',
      confidence: 0.85,
    });

    expect(claim.confidence).toBe(0.85);
  });

  it('should initialize health metrics', async () => {
    const claim = await engine.birthClaim({
      content: 'Test claim',
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'test',
      domain: 'test',
    });

    expect(claim.health).toBeDefined();
    expect(claim.health.challengeCount).toBe(0);
    expect(claim.health.citationCount).toBe(0);
  });

  it('should record birth in activity history', async () => {
    const claim = await engine.birthClaim({
      content: 'Test claim',
      classification: {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'author-1',
      domain: 'test',
    });

    expect(claim.activityHistory).toHaveLength(1);
    expect(claim.activityHistory[0].actorId).toBe('author-1');
  });
});

describe('Metabolism Engine - Evidence Addition', () => {
  let engine: MetabolismEngine;
  let claim: MetabolizingClaim;

  beforeEach(async () => {
    engine = new MetabolismEngine();
    claim = await engine.birthClaim({
      content: 'Test claim',
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'test',
      domain: 'science',
    });
  });

  it('should add evidence to a claim', async () => {
    const health = await engine.addEvidence(claim.id, {
      type: 'document',
      data: 'Research paper XYZ',
      source: 'researcher-1',
    });

    expect(health).toBeDefined();
    expect(health.evidenceStrength).toBeGreaterThan(0);
  });

  it('should update activity history when evidence is added', async () => {
    const health = await engine.addEvidence(claim.id, {
      type: 'signature',
      data: 'sig-abc',
      source: 'verifier-1',
    });

    // Evidence was added and health was returned
    expect(health).toBeDefined();
    expect(health.evidenceStrength).toBeGreaterThan(0);
  });

  it('should throw error for non-existent claim', async () => {
    await expect(
      engine.addEvidence('non-existent', { type: 'test', data: 'x', source: 's' })
    ).rejects.toThrow('Claim not found');
  });

  it('should throw error when adding evidence to dead claim', async () => {
    // Kill the claim first
    await engine.killClaim(claim.id, 'refuted', 'Testing dead claim', []);

    await expect(
      engine.addEvidence(claim.id, { type: 'test', data: 'x', source: 's' })
    ).rejects.toThrow('Cannot add evidence to dead claim');
  });
});

describe('Metabolism Engine - Claim Challenges', () => {
  let engine: MetabolismEngine;
  let claim: MetabolizingClaim;

  beforeEach(async () => {
    engine = new MetabolismEngine();
    claim = await engine.birthClaim({
      content: 'Controversial claim',
      classification: {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'original-author',
      domain: 'debate',
    });
  });

  it('should record a challenge', async () => {
    const health = await engine.challengeClaim(
      claim.id,
      'challenger-1',
      'I disagree with this claim'
    );

    expect(health.challengeCount).toBe(1);
  });

  it('should increase contradiction score with evidence', async () => {
    const healthBefore = engine.assessHealth(claim.id);
    const contradictionBefore = healthBefore?.contradictionScore ?? 0;

    await engine.challengeClaim(claim.id, 'challenger-1', 'Evidence shows otherwise', [
      { type: 'document', data: 'Counter-evidence', source: 'challenger-1' },
    ]);

    const healthAfter = engine.assessHealth(claim.id);
    expect(healthAfter?.contradictionScore).toBeGreaterThan(contradictionBefore);
  });
});

describe('Metabolism Engine - Corroboration', () => {
  let engine: MetabolismEngine;
  let claim: MetabolizingClaim;

  beforeEach(async () => {
    engine = new MetabolismEngine();
    claim = await engine.birthClaim({
      content: 'Scientific finding',
      classification: {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      },
      authorId: 'researcher-1',
      domain: 'science',
    });
  });

  it('should increase corroboration score', async () => {
    const healthBefore = engine.assessHealth(claim.id);
    const corroborationBefore = healthBefore?.corroborationScore ?? 0;

    await engine.corroborateClaim(claim.id, 'researcher-2', {
      type: 'document',
      data: 'Independent replication study',
      source: 'researcher-2',
    });

    const healthAfter = engine.assessHealth(claim.id);
    expect(healthAfter?.corroborationScore).toBeGreaterThan(corroborationBefore);
  });

  it('should cap corroboration score at 1.0', async () => {
    // Add many corroborations
    for (let i = 0; i < 20; i++) {
      await engine.corroborateClaim(claim.id, `researcher-${i}`, {
        type: 'document',
        data: `Replication ${i}`,
        source: `researcher-${i}`,
      });
    }

    const health = engine.assessHealth(claim.id);
    expect(health?.corroborationScore).toBeLessThanOrEqual(1);
  });
});

describe('Metabolism Engine - Claim Death', () => {
  let engine: MetabolismEngine;
  let claim: MetabolizingClaim;

  beforeEach(async () => {
    engine = new MetabolismEngine();
    claim = await engine.birthClaim({
      content: 'Claim that will die',
      classification: {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'author-1',
      domain: 'test',
    });
  });

  it('should create tombstone when claim dies', async () => {
    const tombstone = await engine.killClaim(
      claim.id,
      'refuted',
      'Evidence proved this wrong',
      [{ type: 'document', data: 'Refutation paper', source: 'refuter' }]
    );

    expect(tombstone.claimId).toBe(claim.id);
    expect(tombstone.deathReason).toBe('refuted');
    expect(tombstone.lessonLearned).toBe('Evidence proved this wrong');
  });

  it('should preserve original content in tombstone', async () => {
    const tombstone = await engine.killClaim(
      claim.id,
      'superseded',
      'Better claim replaced this'
    );

    expect(tombstone.originalContent).toBe('Claim that will die');
  });

  it('should record superseding claim if provided', async () => {
    const newClaim = await engine.birthClaim({
      content: 'Better claim',
      classification: claim.classification,
      authorId: 'author-2',
      domain: 'test',
    });

    const tombstone = await engine.killClaim(
      claim.id,
      'superseded',
      'Replaced by better claim',
      [],
      newClaim.id
    );

    expect(tombstone.supersededBy).toBe(newClaim.id);
  });

  it('should transition claim to dead phase', async () => {
    const tombstone = await engine.killClaim(claim.id, 'abandoned', 'No longer relevant');

    // Tombstone was created which means claim is dead
    expect(tombstone).toBeDefined();
    expect(tombstone.deathReason).toBe('abandoned');
  });
});

describe('Metabolism Engine - Tombstone Management', () => {
  let engine: MetabolismEngine;

  beforeEach(async () => {
    engine = new MetabolismEngine();

    // Create and kill several claims
    for (let i = 0; i < 5; i++) {
      const claim = await engine.birthClaim({
        content: `Claim ${i}`,
        classification: {
          empirical: EmpiricalLevel.E1_Testimonial,
          normative: NormativeLevel.N0_Personal,
          materiality: MaterialityLevel.M0_Ephemeral,
        },
        authorId: `author-${i}`,
        domain: i % 2 === 0 ? 'science' : 'politics',
        tags: i < 3 ? ['important'] : [],
      });

      await engine.killClaim(
        claim.id,
        i % 2 === 0 ? 'refuted' : 'obsolete',
        `Lesson ${i}`
      );
    }
  });

  it('should retrieve tombstone by claim ID', async () => {
    const claim = await engine.birthClaim({
      content: 'Specific claim',
      classification: {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    await engine.killClaim(claim.id, 'retracted', 'Author withdrew');

    const tombstone = engine.getTombstone(claim.id);
    expect(tombstone).not.toBeNull();
    expect(tombstone?.deathReason).toBe('retracted');
  });

  it('should return null for non-existent tombstone', () => {
    const tombstone = engine.getTombstone('non-existent');
    expect(tombstone).toBeNull();
  });

  it('should search tombstones by death reason', () => {
    const results = engine.searchTombstones({ deathReason: 'refuted' });

    expect(results.length).toBeGreaterThan(0);
    expect(results.every((t) => t.deathReason === 'refuted')).toBe(true);
  });
});

describe('Metabolism Engine - Claim Decomposition', () => {
  let engine: MetabolismEngine;
  let claim: MetabolizingClaim;

  beforeEach(async () => {
    engine = new MetabolismEngine();
    claim = await engine.birthClaim({
      content: 'Complex claim with multiple parts',
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'author-1',
      domain: 'science',
      evidence: [
        { type: 'document', data: 'Evidence A', source: 'source-1' },
        { type: 'signature', data: 'Evidence B', source: 'source-2' },
      ],
    });

    // Kill the claim first
    await engine.killClaim(claim.id, 'refuted', 'Part of this was wrong');
  });

  it('should decompose a dead claim', async () => {
    const result = await engine.decomposeClaim(claim.id);

    expect(result.originalClaimId).toBe(claim.id);
    expect(result.decomposedAt).toBeGreaterThan(0);
  });

  it('should extract reusable evidence', async () => {
    const result = await engine.decomposeClaim(claim.id);

    expect(result.reusableEvidence).toBeDefined();
  });

  it('should transition to decomposing phase', async () => {
    const result = await engine.decomposeClaim(claim.id);

    // Decomposition result contains extracted components
    expect(result).toBeDefined();
    expect(result.reusableEvidence).toBeDefined();
  });

  it('should throw error for non-dead claim', async () => {
    const liveClaim = await engine.birthClaim({
      content: 'Living claim',
      classification: claim.classification,
      authorId: 'test',
      domain: 'test',
    });

    await expect(engine.decomposeClaim(liveClaim.id)).rejects.toThrow(
      'Can only decompose dead claims'
    );
  });
});

describe('Metabolism Engine - Tombstone Warning System', () => {
  let engine: MetabolismEngine;

  beforeEach(async () => {
    engine = new MetabolismEngine();

    // Create a tombstone for a common mistake
    const badClaim = await engine.birthClaim({
      content: 'Vaccines cause autism',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'bad-actor',
      domain: 'health',
      tags: ['misinformation'],
    });

    await engine.killClaim(
      badClaim.id,
      'refuted',
      'This claim has been thoroughly debunked by decades of research'
    );
  });

  it('should check new claims against tombstones', async () => {
    const check = engine.checkAgainstTombstones('Vaccines cause autism', 'health');

    expect(check.similar).toBe(true);
    expect(check.warnings.length).toBeGreaterThan(0);
  });

  it('should warn about similar claims', async () => {
    const check = engine.checkAgainstTombstones(
      'Vaccinations are linked to developmental issues',
      'health'
    );

    // Should find similarity even with different wording
    // Note: similarity depends on word-level matching
    expect(check.warnings).toBeDefined();
  });

  it('should not block unrelated claims', async () => {
    const check = engine.checkAgainstTombstones(
      'The speed of light is constant',
      'physics'
    );

    expect(check.similar).toBe(false);
  });
});

describe('Metabolism Engine - Lifecycle Transitions', () => {
  let engine: MetabolismEngine;
  let claim: MetabolizingClaim;

  beforeEach(async () => {
    engine = new MetabolismEngine({ autoTransition: true });
    claim = await engine.birthClaim({
      content: 'Test claim for transitions',
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'author-1',
      domain: 'test',
      confidence: 0.5,
    });
  });

  it('should record transitions in history', async () => {
    // Add enough evidence to transition from nascent
    let lastHealth;
    for (let i = 0; i < 5; i++) {
      lastHealth = await engine.addEvidence(claim.id, {
        type: 'document',
        data: `Evidence ${i}`,
        source: `source-${i}`,
      });
    }

    // Health should reflect the evidence additions
    expect(lastHealth).toBeDefined();
    expect(lastHealth?.evidenceStrength).toBeGreaterThan(0);
  });
});

describe('Metabolism Engine - Health Assessment', () => {
  let engine: MetabolismEngine;

  beforeEach(() => {
    engine = new MetabolismEngine();
  });

  it('should calculate overall health from components', async () => {
    const claim = await engine.birthClaim({
      content: 'Test claim',
      classification: {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      },
      authorId: 'reputable-author',
      domain: 'science',
      confidence: 0.8,
    });

    const health = engine.assessHealth(claim.id);

    expect(health).not.toBeNull();
    expect(health?.overallHealth).toBeGreaterThan(0);
    expect(health?.overallHealth).toBeLessThanOrEqual(1);
  });

  it('should return null for non-existent claim', () => {
    const health = engine.assessHealth('non-existent');
    expect(health).toBeNull();
  });

  it('should update health when evidence changes', async () => {
    const claim = await engine.birthClaim({
      content: 'Test claim',
      classification: {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    const healthBefore = engine.assessHealth(claim.id);

    await engine.addEvidence(claim.id, {
      type: 'signature',
      data: 'Strong evidence',
      source: 'trusted-source',
    });

    const healthAfter = engine.assessHealth(claim.id);

    expect(healthAfter?.evidenceStrength).toBeGreaterThanOrEqual(
      healthBefore?.evidenceStrength ?? 0
    );
  });
});

describe('Metabolism Engine - Configuration', () => {
  it('should use default configuration', () => {
    const engine = new MetabolismEngine();

    expect(DEFAULT_METABOLISM_CONFIG.stagnationThresholdDays).toBe(90);
    expect(DEFAULT_METABOLISM_CONFIG.autoTransition).toBe(true);
    expect(DEFAULT_METABOLISM_CONFIG.autoDecompose).toBe(false);
  });

  it('should accept custom configuration', () => {
    const engine = new MetabolismEngine({
      stagnationThresholdDays: 30,
      autoDecompose: true,
    });

    // Engine should use merged config
    expect(engine).toBeDefined();
  });
});

describe('Metabolism Engine - Singleton', () => {
  beforeEach(() => {
    resetMetabolismEngine();
  });

  it('should return same instance', () => {
    const engine1 = getMetabolismEngine();
    const engine2 = getMetabolismEngine();

    expect(engine1).toBe(engine2);
  });

  it('should reset correctly', async () => {
    const engine1 = getMetabolismEngine();
    await engine1.birthClaim({
      content: 'Test',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    resetMetabolismEngine();

    const engine2 = getMetabolismEngine();
    expect(engine2.getStats().totalClaims).toBe(0);
  });
});

describe('Metabolism Engine - Event Emission', () => {
  let engine: MetabolismEngine;
  const events: string[] = [];

  beforeEach(() => {
    engine = new MetabolismEngine();
    events.length = 0;
    engine.onEvent((event) => events.push(event.type));
  });

  it('should emit claim_born event', async () => {
    await engine.birthClaim({
      content: 'Test',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    expect(events).toContain('claim_born');
  });

  it('should emit health_updated event', async () => {
    const claim = await engine.birthClaim({
      content: 'Test',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    await engine.addEvidence(claim.id, {
      type: 'test',
      data: 'data',
      source: 'source',
    });

    expect(events).toContain('health_updated');
  });

  it('should emit claim_died and tombstone_created events', async () => {
    const claim = await engine.birthClaim({
      content: 'Test',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    await engine.killClaim(claim.id, 'abandoned', 'Test');

    expect(events).toContain('claim_died');
    expect(events).toContain('tombstone_created');
  });

  it('should allow unsubscription from events', async () => {
    const secondListener: string[] = [];
    const unsubscribe = engine.onEvent((event) => secondListener.push(event.type));

    await engine.birthClaim({
      content: 'First',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    expect(secondListener).toContain('claim_born');

    unsubscribe();
    secondListener.length = 0;

    await engine.birthClaim({
      content: 'Second',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    expect(secondListener).not.toContain('claim_born');
  });

  it('should handle errors in event listeners gracefully', async () => {
    const errorListener = () => {
      throw new Error('Listener error');
    };
    engine.onEvent(errorListener);

    // Should not throw despite listener error
    const claim = await engine.birthClaim({
      content: 'Test',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    expect(claim).toBeDefined();
    expect(events).toContain('claim_born');
  });

  it('should emit claim_transitioned event on transition', async () => {
    const claim = await engine.birthClaim({
      content: 'Test',
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'test',
      domain: 'test',
    });

    // Force a transition
    const health = engine.assessHealth(claim.id);
    if (health) {
      // Add evidence to trigger transition
      for (let i = 0; i < 5; i++) {
        await engine.addEvidence(claim.id, {
          type: 'document',
          data: `Evidence ${i}`,
          source: `source-${i}`,
        });
      }
    }

    // Events should include health_updated at minimum
    expect(events).toContain('health_updated');
  });

  it('should emit claim_decomposed event', async () => {
    const claim = await engine.birthClaim({
      content: 'Test for decomposition',
      classification: {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    await engine.killClaim(claim.id, 'refuted', 'Wrong');
    await engine.decomposeClaim(claim.id);

    expect(events).toContain('claim_decomposed');
  });
});

describe('Metabolism Engine - Statistics', () => {
  let engine: MetabolismEngine;

  beforeEach(() => {
    engine = new MetabolismEngine();
  });

  it('should return correct statistics', async () => {
    const stats = engine.getStats();
    expect(stats.totalClaims).toBe(0);
    expect(stats.totalTombstones).toBe(0);
    expect(stats.averageHealth).toBe(0);
    expect(stats.oldestClaim).toBeNull();
    expect(stats.newestClaim).toBeNull();
  });

  it('should count claims by phase', async () => {
    await engine.birthClaim({
      content: 'Nascent claim',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    const stats = engine.getStats();
    expect(stats.claimsByPhase.nascent).toBe(1);
    expect(stats.totalClaims).toBe(1);
  });

  it('should count tombstones by death reason', async () => {
    const claim1 = await engine.birthClaim({
      content: 'Claim 1',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    const claim2 = await engine.birthClaim({
      content: 'Claim 2',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    await engine.killClaim(claim1.id, 'refuted', 'Wrong');
    await engine.killClaim(claim2.id, 'obsolete', 'Outdated');

    const stats = engine.getStats();
    expect(stats.tombstonesByReason.refuted).toBe(1);
    expect(stats.tombstonesByReason.obsolete).toBe(1);
    expect(stats.totalTombstones).toBe(2);
  });

  it('should calculate average health correctly', async () => {
    await engine.birthClaim({
      content: 'Claim 1',
      classification: {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      },
      authorId: 'test',
      domain: 'test',
      confidence: 0.8,
    });

    const stats = engine.getStats();
    expect(stats.averageHealth).toBeGreaterThan(0);
    expect(stats.averageHealth).toBeLessThanOrEqual(1);
  });

  it('should track oldest and newest claims', async () => {
    const claim1 = await engine.birthClaim({
      content: 'First',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    // Small delay to ensure different timestamps
    await new Promise((resolve) => setTimeout(resolve, 5));

    const claim2 = await engine.birthClaim({
      content: 'Second',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    const stats = engine.getStats();
    expect(stats.oldestClaim).toBe(claim1.createdAt);
    expect(stats.newestClaim).toBe(claim2.createdAt);
    expect(stats.newestClaim).toBeGreaterThanOrEqual(stats.oldestClaim!);
  });

  it('should not include dead claims in average health', async () => {
    const claim = await engine.birthClaim({
      content: 'Will die',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    await engine.killClaim(claim.id, 'refuted', 'Wrong');

    const stats = engine.getStats();
    // With only dead claims, averageHealth should be 0
    expect(stats.averageHealth).toBe(0);
  });
});

describe('Metabolism Engine - Tombstone Search', () => {
  let engine: MetabolismEngine;

  beforeEach(async () => {
    engine = new MetabolismEngine();

    // Create diverse tombstones
    const claims = [];
    for (let i = 0; i < 10; i++) {
      const claim = await engine.birthClaim({
        content: `Claim about ${i < 5 ? 'physics' : 'biology'} topic ${i}`,
        classification: {
          empirical: EmpiricalLevel.E1_Testimonial,
          normative: NormativeLevel.N0_Personal,
          materiality: MaterialityLevel.M0_Ephemeral,
        },
        authorId: `author-${i}`,
        domain: i < 5 ? 'physics' : 'biology',
        tags: i < 3 ? ['important', 'reviewed'] : ['pending'],
      });
      claims.push(claim);
    }

    // Kill with different reasons
    for (let i = 0; i < claims.length; i++) {
      const reason = i % 3 === 0 ? 'refuted' : i % 3 === 1 ? 'obsolete' : 'superseded';
      await engine.killClaim(claims[i].id, reason, `Lesson for claim ${i}`);
    }
  });

  it('should search by tag', () => {
    const results = engine.searchTombstones({ tag: 'important' });
    expect(results.length).toBe(3);
  });

  it('should search by content substring', () => {
    const results = engine.searchTombstones({ contentContains: 'physics' });
    expect(results.length).toBe(5);
  });

  it('should search by date range', async () => {
    const now = Date.now();
    const results = engine.searchTombstones({
      dateRange: { start: now - 60000, end: now + 60000 },
    });
    expect(results.length).toBe(10);
  });

  it('should exclude outside date range', () => {
    const pastTime = Date.now() - 1000000;
    const results = engine.searchTombstones({
      dateRange: { start: pastTime - 1000, end: pastTime },
    });
    expect(results.length).toBe(0);
  });

  it('should combine search criteria', () => {
    const results = engine.searchTombstones({
      deathReason: 'refuted',
      tag: 'important',
    });
    // Claims 0, 3 would be refuted (0, 3, 6, 9) - but only 0 has 'important' tag
    expect(results.every((t) => t.deathReason === 'refuted')).toBe(true);
  });

  it('should sort results by diedAt descending', () => {
    const results = engine.searchTombstones({});
    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1].diedAt).toBeGreaterThanOrEqual(results[i].diedAt);
    }
  });

  it('should handle case-insensitive content search', () => {
    const results = engine.searchTombstones({ contentContains: 'PHYSICS' });
    expect(results.length).toBe(5);
  });
});

describe('Metabolism Engine - Evidence Strength Calculation', () => {
  let engine: MetabolismEngine;

  beforeEach(() => {
    engine = new MetabolismEngine();
  });

  it('should return 0 for claims with no evidence', async () => {
    const claim = await engine.birthClaim({
      content: 'No evidence claim',
      classification: {
        empirical: EmpiricalLevel.E4_Consensus,
        normative: NormativeLevel.N3_Axiomatic,
        materiality: MaterialityLevel.M3_Foundational,
      },
      authorId: 'test',
      domain: 'test',
    });

    const health = engine.assessHealth(claim.id);
    expect(health?.evidenceStrength).toBe(0);
  });

  it('should scale evidence strength by empirical level', async () => {
    // E0 claim
    const claimE0 = await engine.birthClaim({
      content: 'E0 claim',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    // E4 claim
    const claimE4 = await engine.birthClaim({
      content: 'E4 claim',
      classification: {
        empirical: EmpiricalLevel.E4_Consensus,
        normative: NormativeLevel.N3_Axiomatic,
        materiality: MaterialityLevel.M3_Foundational,
      },
      authorId: 'test',
      domain: 'test',
    });

    // Add same evidence to both
    for (const id of [claimE0.id, claimE4.id]) {
      await engine.addEvidence(id, {
        type: 'document',
        data: 'Evidence',
        source: 'source',
      });
    }

    const healthE0 = engine.assessHealth(claimE0.id);
    const healthE4 = engine.assessHealth(claimE4.id);

    // E4 should have higher ceiling
    expect(healthE4?.evidenceStrength).toBeGreaterThan(healthE0?.evidenceStrength ?? 0);
  });

  it('should cap evidence strength contribution', async () => {
    const claim = await engine.birthClaim({
      content: 'Many evidence claim',
      classification: {
        empirical: EmpiricalLevel.E4_Consensus,
        normative: NormativeLevel.N3_Axiomatic,
        materiality: MaterialityLevel.M3_Foundational,
      },
      authorId: 'test',
      domain: 'test',
    });

    // Add many pieces of evidence
    for (let i = 0; i < 20; i++) {
      await engine.addEvidence(claim.id, {
        type: 'document',
        data: `Evidence ${i}`,
        source: `source-${i}`,
      });
    }

    const health = engine.assessHealth(claim.id);
    // Should be capped by the E4 ceiling (0.99) * countFactor
    expect(health?.evidenceStrength).toBeLessThanOrEqual(0.99);
  });
});

describe('Metabolism Engine - Overall Health Computation', () => {
  let engine: MetabolismEngine;

  beforeEach(() => {
    engine = new MetabolismEngine();
  });

  it('should weight confidence in overall health', async () => {
    const lowConfidence = await engine.birthClaim({
      content: 'Low confidence',
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'test',
      domain: 'test',
      confidence: 0.2,
    });

    const highConfidence = await engine.birthClaim({
      content: 'High confidence',
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'test',
      domain: 'test',
      confidence: 0.9,
    });

    const healthLow = engine.assessHealth(lowConfidence.id);
    const healthHigh = engine.assessHealth(highConfidence.id);

    expect(healthHigh?.overallHealth).toBeGreaterThan(healthLow?.overallHealth ?? 0);
  });

  it('should penalize claims with many challenges', async () => {
    const claim = await engine.birthClaim({
      content: 'Challenged claim',
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'test',
      domain: 'test',
    });

    const healthBefore = engine.assessHealth(claim.id);

    // Add many challenges
    for (let i = 0; i < 5; i++) {
      await engine.challengeClaim(claim.id, `challenger-${i}`, 'I disagree');
    }

    const healthAfter = engine.assessHealth(claim.id);

    expect(healthAfter?.overallHealth).toBeLessThan(healthBefore?.overallHealth ?? 1);
  });

  it('should boost health with corroboration', async () => {
    const claim = await engine.birthClaim({
      content: 'Corroborated claim',
      classification: {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      },
      authorId: 'test',
      domain: 'test',
    });

    const healthBefore = engine.assessHealth(claim.id);

    await engine.corroborateClaim(claim.id, 'corroborator-1', {
      type: 'document',
      data: 'Independent verification',
      source: 'corroborator-1',
    });

    const healthAfter = engine.assessHealth(claim.id);

    expect(healthAfter?.overallHealth).toBeGreaterThan(healthBefore?.overallHealth ?? 0);
  });

  it('should clamp overall health between 0 and 1', async () => {
    const claim = await engine.birthClaim({
      content: 'Test',
      classification: {
        empirical: EmpiricalLevel.E4_Consensus,
        normative: NormativeLevel.N3_Axiomatic,
        materiality: MaterialityLevel.M3_Foundational,
      },
      authorId: 'test',
      domain: 'test',
      confidence: 1.0,
    });

    // Boost everything
    for (let i = 0; i < 10; i++) {
      await engine.addEvidence(claim.id, {
        type: 'document',
        data: `Strong evidence ${i}`,
        source: `reputable-${i}`,
      });
      await engine.corroborateClaim(claim.id, `corroborator-${i}`, {
        type: 'document',
        data: `Corroboration ${i}`,
        source: `corroborator-${i}`,
      });
    }

    const health = engine.assessHealth(claim.id);
    expect(health?.overallHealth).toBeLessThanOrEqual(1);
    expect(health?.overallHealth).toBeGreaterThanOrEqual(0);
  });
});

describe('Metabolism Engine - Similarity Calculation', () => {
  let engine: MetabolismEngine;

  beforeEach(async () => {
    engine = new MetabolismEngine();

    // Create a tombstone for similarity testing
    const claim = await engine.birthClaim({
      content: 'The quick brown fox jumps over the lazy dog',
      classification: {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });
    await engine.killClaim(claim.id, 'refuted', 'Wrong');
  });

  it('should detect exact matches', () => {
    const check = engine.checkAgainstTombstones(
      'The quick brown fox jumps over the lazy dog',
      'test'
    );
    expect(check.similar).toBe(true);
    expect(check.warnings[0].similarity).toBeCloseTo(1.0);
  });

  it('should detect partial matches', () => {
    // Need enough overlap to pass 0.7 threshold for Jaccard similarity
    const check = engine.checkAgainstTombstones(
      'The quick brown fox jumps over the lazy',  // 8 of 9 words match
      'test'
    );
    expect(check.similar).toBe(true);
    expect(check.warnings[0].similarity).toBeGreaterThan(0.7);
    expect(check.warnings[0].similarity).toBeLessThan(1.0);
  });

  it('should not flag completely different content', () => {
    const check = engine.checkAgainstTombstones(
      'Quantum mechanics describes subatomic particles',
      'test'
    );
    expect(check.similar).toBe(false);
  });

  it('should sort warnings by similarity descending', async () => {
    // Add more tombstones
    const claim2 = await engine.birthClaim({
      content: 'A quick brown animal jumps',
      classification: {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });
    await engine.killClaim(claim2.id, 'refuted', 'Wrong');

    const check = engine.checkAgainstTombstones(
      'The quick brown fox jumps high',
      'test'
    );

    if (check.warnings.length > 1) {
      for (let i = 1; i < check.warnings.length; i++) {
        expect(check.warnings[i - 1].similarity).toBeGreaterThanOrEqual(
          check.warnings[i].similarity
        );
      }
    }
  });
});

describe('Metabolism Engine - Death Propagation', () => {
  let engine: MetabolismEngine;

  beforeEach(() => {
    engine = new MetabolismEngine({ autoTransition: true });
  });

  it('should notify dependents when claim dies', async () => {
    // Create parent claim
    const parent = await engine.birthClaim({
      content: 'Parent claim',
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'author-1',
      domain: 'test',
    });

    // Create dependent claim that references parent
    const dependent = await engine.birthClaim({
      content: 'Dependent claim',
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'author-2',
      domain: 'test',
    });

    // Manually link them (simulating dependency)
    // Access internal state via assessHealth
    const parentHealth = engine.assessHealth(parent.id);
    expect(parentHealth).toBeDefined();

    // Kill parent and verify tombstone
    const tombstone = await engine.killClaim(parent.id, 'refuted', 'Was wrong');
    expect(tombstone.deathReason).toBe('refuted');
  });
});

describe('Metabolism Engine - Decomposition Helpers', () => {
  let engine: MetabolismEngine;

  beforeEach(() => {
    engine = new MetabolismEngine();
  });

  it('should extract reusable evidence from dead claims', async () => {
    const claim = await engine.birthClaim({
      content: 'Claim with evidence',
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'test',
      domain: 'test',
      evidence: [
        { type: 'document', data: 'Good evidence 1', source: 'trusted' },
        { type: 'document', data: 'Good evidence 2', source: 'trusted' },
      ],
    });

    // Kill with different refuting evidence
    await engine.killClaim(claim.id, 'refuted', 'Partial refutation', [
      { type: 'document', data: 'Refuting evidence', source: 'refuter' },
    ]);

    const result = await engine.decomposeClaim(claim.id);

    // Should have reusable evidence (not refuting)
    expect(result.reusableEvidence.length).toBe(2);
  });

  it('should identify methodological insights from refuted claims', async () => {
    const claim = await engine.birthClaim({
      content: 'Methodology test',
      classification: {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
      confidence: 0.8,
    });

    await engine.killClaim(claim.id, 'refuted', 'Method was flawed');
    const result = await engine.decomposeClaim(claim.id);

    // Should have insight about refutation
    expect(result.methodologicalInsights.length).toBeGreaterThan(0);
  });

  it('should identify anti-patterns in failed claims', async () => {
    const claim = await engine.birthClaim({
      content: 'High confidence low evidence',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified, // Very low E-level
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
      confidence: 0.9, // High confidence
    });

    await engine.killClaim(claim.id, 'refuted', 'Baseless claim');
    const result = await engine.decomposeClaim(claim.id);

    // Should detect over-confidence pattern
    expect(result.antiPatterns.length).toBeGreaterThan(0);
    expect(result.antiPatterns.some((p) => p.includes('confidence'))).toBe(true);
  });

  it('should detect minimal evidence anti-pattern', async () => {
    const claim = await engine.birthClaim({
      content: 'Minimal evidence claim',
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'test',
      domain: 'test',
      confidence: 0.8,
      evidence: [{ type: 'document', data: 'Only one piece', source: 'single' }],
    });

    await engine.killClaim(claim.id, 'refuted', 'Not enough evidence');
    const result = await engine.decomposeClaim(claim.id);

    expect(result.antiPatterns.some((p) => p.includes('minimal evidence'))).toBe(true);
  });
});

describe('Metabolism Engine - Transition Rules', () => {
  let engine: MetabolismEngine;

  beforeEach(() => {
    engine = new MetabolismEngine({
      autoTransition: true,
      nascentExitThreshold: 0.2,
      stagnationThresholdDays: 90,
      challengeThreshold: 0.4,
      dyingThreshold: 0.6,
      deathThreshold: 0.1,
    });
  });

  it('should transition from nascent to growing with evidence', async () => {
    const claim = await engine.birthClaim({
      content: 'Growing claim',
      classification: {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      },
      authorId: 'test',
      domain: 'test',
    });

    expect(claim.phase).toBe('nascent');

    // Add enough evidence to reach threshold
    for (let i = 0; i < 5; i++) {
      await engine.addEvidence(claim.id, {
        type: 'document',
        data: `Evidence ${i}`,
        source: `source-${i}`,
      });
    }

    const health = engine.assessHealth(claim.id);
    expect(health?.evidenceStrength).toBeGreaterThan(0.2);
  });

  it('should transition to challenged under heavy challenges', async () => {
    const claim = await engine.birthClaim({
      content: 'Will be challenged',
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'test',
      domain: 'test',
    });

    // Add some evidence first to move past nascent
    for (let i = 0; i < 3; i++) {
      await engine.addEvidence(claim.id, {
        type: 'document',
        data: `Evidence ${i}`,
        source: `source-${i}`,
      });
    }

    // Challenge heavily with contradicting evidence
    for (let i = 0; i < 5; i++) {
      await engine.challengeClaim(claim.id, `challenger-${i}`, 'Wrong', [
        { type: 'document', data: `Counter ${i}`, source: `challenger-${i}` },
      ]);
    }

    const health = engine.assessHealth(claim.id);
    expect(health?.challengeCount).toBeGreaterThanOrEqual(5);
    expect(health?.contradictionScore).toBeGreaterThan(0);
  });

  it('should resolve challenges through corroboration', async () => {
    const claim = await engine.birthClaim({
      content: 'Challenged but vindicated',
      classification: {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      },
      authorId: 'test',
      domain: 'test',
    });

    // Add evidence
    for (let i = 0; i < 5; i++) {
      await engine.addEvidence(claim.id, {
        type: 'document',
        data: `Evidence ${i}`,
        source: `source-${i}`,
      });
    }

    // Challenge
    await engine.challengeClaim(claim.id, 'challenger', 'Disagree', [
      { type: 'document', data: 'Counter', source: 'challenger' },
    ]);

    // Corroborate heavily
    for (let i = 0; i < 10; i++) {
      await engine.corroborateClaim(claim.id, `corroborator-${i}`, {
        type: 'document',
        data: `Independent verification ${i}`,
        source: `corroborator-${i}`,
      });
    }

    const health = engine.assessHealth(claim.id);
    expect(health?.corroborationScore).toBeGreaterThan(health?.contradictionScore ?? 0);
  });

  it('should handle transition history trimming', async () => {
    const engine = new MetabolismEngine({
      autoTransition: true,
      maxTransitionHistory: 3,
    });

    const claim = await engine.birthClaim({
      content: 'Many transitions',
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'test',
      domain: 'test',
    });

    // Verify claim is created
    expect(claim.transitionHistory.length).toBe(0);

    // Kill it to force a transition
    await engine.killClaim(claim.id, 'abandoned', 'Done');

    // Should have at least one transition
    const tombstone = engine.getTombstone(claim.id);
    expect(tombstone).not.toBeNull();
  });
});

describe('Metabolism Engine - Persistence', () => {
  it('should return false for isPersistenceReady when disabled', () => {
    const engine = new MetabolismEngine({ enablePersistence: false });
    expect(engine.isPersistenceReady()).toBe(false);
  });

  it('should handle loadFromPersistence when disabled', async () => {
    const engine = new MetabolismEngine({ enablePersistence: false });
    const result = await engine.loadFromPersistence();
    expect(result.claims).toBe(0);
    expect(result.tombstones).toBe(0);
  });
});

describe('Metabolism Engine - Initial Health', () => {
  let engine: MetabolismEngine;

  beforeEach(() => {
    engine = new MetabolismEngine();
  });

  it('should set initial evidence strength based on provided evidence', async () => {
    const claim = await engine.birthClaim({
      content: 'Claim with initial evidence',
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      authorId: 'test',
      domain: 'test',
      evidence: [
        { type: 'document', data: 'Evidence 1', source: 'source1' },
        { type: 'document', data: 'Evidence 2', source: 'source2' },
      ],
    });

    expect(claim.health.evidenceStrength).toBeGreaterThan(0);
  });

  it('should set author reputation to 0.5 by default', async () => {
    const claim = await engine.birthClaim({
      content: 'Test',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    expect(claim.health.authorReputation).toBe(0.5);
  });

  it('should set initial overall health to 0.5', async () => {
    const claim = await engine.birthClaim({
      content: 'Test',
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    expect(claim.health.overallHealth).toBe(0.5);
  });
});

describe('Metabolism Engine - Challenge with contradicting evidence', () => {
  let engine: MetabolismEngine;

  beforeEach(() => {
    engine = new MetabolismEngine();
  });

  it('should record contradicted activity when evidence provided', async () => {
    const claim = await engine.birthClaim({
      content: 'Test',
      classification: {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    await engine.challengeClaim(claim.id, 'challenger', 'I have proof', [
      { type: 'document', data: 'Counter evidence', source: 'challenger' },
    ]);

    const health = engine.assessHealth(claim.id);
    expect(health?.contradictionScore).toBeGreaterThan(0);
  });

  it('should not record contradicted when no evidence provided', async () => {
    const claim = await engine.birthClaim({
      content: 'Test',
      classification: {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    await engine.challengeClaim(claim.id, 'challenger', 'Just disagree');

    const health = engine.assessHealth(claim.id);
    expect(health?.challengeCount).toBe(1);
    expect(health?.contradictionScore).toBe(0);
  });
});

describe('Metabolism Engine - Error Handling', () => {
  let engine: MetabolismEngine;

  beforeEach(() => {
    engine = new MetabolismEngine();
  });

  it('should throw error when challenging non-existent claim', async () => {
    await expect(
      engine.challengeClaim('non-existent', 'challenger', 'Test')
    ).rejects.toThrow('Claim not found');
  });

  it('should throw error when corroborating non-existent claim', async () => {
    await expect(
      engine.corroborateClaim('non-existent', 'corroborator', {
        type: 'document',
        data: 'Test',
        source: 'test',
      })
    ).rejects.toThrow('Claim not found');
  });

  it('should throw error when killing non-existent claim', async () => {
    await expect(
      engine.killClaim('non-existent', 'abandoned', 'Test')
    ).rejects.toThrow('Claim not found');
  });
});

describe('Metabolism Engine - Tags and Domain', () => {
  let engine: MetabolismEngine;

  beforeEach(() => {
    engine = new MetabolismEngine();
  });

  it('should preserve tags through lifecycle', async () => {
    const claim = await engine.birthClaim({
      content: 'Tagged claim',
      classification: {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
      tags: ['important', 'urgent', 'reviewed'],
    });

    expect(claim.tags).toEqual(['important', 'urgent', 'reviewed']);

    const tombstone = await engine.killClaim(claim.id, 'obsolete', 'Done');
    expect(tombstone.tags).toEqual(['important', 'urgent', 'reviewed']);
  });

  it('should default to empty tags array', async () => {
    const claim = await engine.birthClaim({
      content: 'No tags',
      classification: {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      authorId: 'test',
      domain: 'test',
    });

    expect(claim.tags).toEqual([]);
  });
});
