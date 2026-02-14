/**
 * Epistemic Module Tests
 */

import { describe, it, expect } from 'vitest';
import {
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  classificationCode,
  meetsMinimum,
  parseClassificationCode,
  parseClassificationCodeStrict,
  createClaim,
  addEvidence,
  meetsStandard,
  isExpired,
  ClaimBuilder,
  claim,
  Standards,
  EpistemicBatch,
  EpistemicClaimPool,
  type EpistemicClaim,
} from '../src/epistemic/index.js';

describe('Epistemic - Classification', () => {
  it('should generate correct classification code', () => {
    const code = classificationCode({
      empirical: EmpiricalLevel.E3_Cryptographic,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    });

    expect(code).toBe('E3-N2-M2');
  });

  it('should parse classification code', () => {
    const parsed = parseClassificationCode('E3-N2-M2');

    expect(parsed).not.toBeNull();
    expect(parsed!.empirical).toBe(EmpiricalLevel.E3_Cryptographic);
    expect(parsed!.normative).toBe(NormativeLevel.N2_Network);
    expect(parsed!.materiality).toBe(MaterialityLevel.M2_Persistent);
  });

  it('should return null for invalid classification code', () => {
    expect(parseClassificationCode('invalid')).toBeNull();
    expect(parseClassificationCode('E5-N2-M2')).toBeNull(); // E5 is out of range [0-4]
    expect(parseClassificationCode('E3-N4-M2')).toBeNull(); // N4 is out of range [0-3]
    expect(parseClassificationCode('E3-N2-M4')).toBeNull(); // M4 is out of range [0-3]
    expect(parseClassificationCode('')).toBeNull();
  });

  it('should check minimum requirements', () => {
    const classification = {
      empirical: EmpiricalLevel.E3_Cryptographic,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    };

    expect(meetsMinimum(
      classification,
      EmpiricalLevel.E2_PrivateVerify,
      NormativeLevel.N1_Communal
    )).toBe(true);

    expect(meetsMinimum(
      classification,
      EmpiricalLevel.E4_Consensus,
      NormativeLevel.N1_Communal
    )).toBe(false);
  });
});

describe('Epistemic - Claims', () => {
  it('should create a claim', () => {
    const c = createClaim(
      'User has verified email',
      EmpiricalLevel.E2_PrivateVerify,
      NormativeLevel.N1_Communal,
      MaterialityLevel.M1_Temporal,
      'email-service'
    );

    expect(c.content).toBe('User has verified email');
    expect(c.classification.empirical).toBe(EmpiricalLevel.E2_PrivateVerify);
    expect(c.issuer).toBe('email-service');
    expect(c.evidence).toHaveLength(0);
    expect(c.id).toMatch(/^claim_/);
  });

  it('should add evidence to a claim', () => {
    let c = createClaim(
      'Test claim',
      EmpiricalLevel.E1_Testimonial,
      NormativeLevel.N0_Personal,
      MaterialityLevel.M0_Ephemeral,
      'test'
    );

    c = addEvidence(c, {
      type: 'signature',
      data: 'abc123',
      source: 'user',
      timestamp: Date.now(),
    });

    expect(c.evidence).toHaveLength(1);
    expect(c.evidence[0].type).toBe('signature');
  });

  it('should check if claim meets standard', () => {
    const c = createClaim(
      'Test',
      EmpiricalLevel.E3_Cryptographic,
      NormativeLevel.N2_Network,
      MaterialityLevel.M2_Persistent,
      'test'
    );

    expect(meetsStandard(c, EmpiricalLevel.E3_Cryptographic)).toBe(true);
    expect(meetsStandard(c, EmpiricalLevel.E4_Consensus)).toBe(false);
  });

  it('should check expiration', () => {
    const unexpiredClaim = createClaim(
      'Test',
      EmpiricalLevel.E0_Unverified,
      NormativeLevel.N0_Personal,
      MaterialityLevel.M0_Ephemeral,
      'test'
    );

    const expiredClaim = {
      ...unexpiredClaim,
      expiresAt: Date.now() - 1000,
    };

    const futureClaim = {
      ...unexpiredClaim,
      expiresAt: Date.now() + 1000000,
    };

    expect(isExpired(unexpiredClaim)).toBe(false); // No expiration
    expect(isExpired(expiredClaim)).toBe(true);
    expect(isExpired(futureClaim)).toBe(false);
  });
});

describe('Epistemic - ClaimBuilder', () => {
  it('should build claims with fluent API', () => {
    const c = claim('Identity verified via cryptographic proof')
      .withEmpirical(EmpiricalLevel.E3_Cryptographic)
      .withNormative(NormativeLevel.N2_Network)
      .withMateriality(MaterialityLevel.M2_Persistent)
      .withIssuer('identity-service')
      .build();

    expect(c.content).toBe('Identity verified via cryptographic proof');
    expect(c.classification.empirical).toBe(EmpiricalLevel.E3_Cryptographic);
    expect(c.classification.normative).toBe(NormativeLevel.N2_Network);
    expect(c.classification.materiality).toBe(MaterialityLevel.M2_Persistent);
    expect(c.issuer).toBe('identity-service');
  });

  it('should support withClassification shorthand', () => {
    const c = claim('Test')
      .withClassification(
        EmpiricalLevel.E4_Consensus,
        NormativeLevel.N3_Universal,
        MaterialityLevel.M3_Immutable
      )
      .build();

    expect(classificationCode(c.classification)).toBe('E4-N3-M3');
  });

  it('should support evidence in builder', () => {
    const c = claim('Signed document')
      .withEmpirical(EmpiricalLevel.E3_Cryptographic)
      .withEvidence({
        type: 'ed25519_signature',
        data: 'sig_abc123',
        source: 'user_pubkey',
        timestamp: Date.now(),
      })
      .build();

    expect(c.evidence).toHaveLength(1);
    expect(c.evidence[0].type).toBe('ed25519_signature');
  });

  it('should support expiration', () => {
    const expiresAt = Date.now() + 86400000; // 24 hours
    const c = claim('Temporary access')
      .withExpiration(expiresAt)
      .build();

    expect(c.expiresAt).toBe(expiresAt);
  });

  it('should default to lowest levels', () => {
    const c = claim('Simple claim').build();

    expect(c.classification.empirical).toBe(EmpiricalLevel.E0_Unverified);
    expect(c.classification.normative).toBe(NormativeLevel.N0_Personal);
    expect(c.classification.materiality).toBe(MaterialityLevel.M0_Ephemeral);
  });
});

describe('Epistemic - Standards', () => {
  it('should define HighTrust standard', () => {
    expect(Standards.HighTrust.minE).toBe(EmpiricalLevel.E3_Cryptographic);
    expect(Standards.HighTrust.minN).toBe(NormativeLevel.N2_Network);
    expect(Standards.HighTrust.minM).toBe(MaterialityLevel.M2_Persistent);
  });

  it('should define MediumTrust standard', () => {
    expect(Standards.MediumTrust.minE).toBe(EmpiricalLevel.E2_PrivateVerify);
    expect(Standards.MediumTrust.minN).toBe(NormativeLevel.N1_Communal);
    expect(Standards.MediumTrust.minM).toBe(MaterialityLevel.M1_Temporal);
  });

  it('should define LowTrust standard', () => {
    expect(Standards.LowTrust.minE).toBe(EmpiricalLevel.E1_Testimonial);
    expect(Standards.LowTrust.minN).toBe(NormativeLevel.N0_Personal);
    expect(Standards.LowTrust.minM).toBe(MaterialityLevel.M0_Ephemeral);
  });
});

describe('Epistemic - Input Validation', () => {
  it('should throw on empty content in createClaim', () => {
    expect(() => createClaim(
      '',
      EmpiricalLevel.E1_Testimonial,
      NormativeLevel.N0_Personal,
      MaterialityLevel.M0_Ephemeral,
      'test'
    )).toThrow();
  });

  it('should throw on empty issuer in createClaim', () => {
    expect(() => createClaim(
      'Test content',
      EmpiricalLevel.E1_Testimonial,
      NormativeLevel.N0_Personal,
      MaterialityLevel.M0_Ephemeral,
      ''
    )).toThrow();
  });

  it('should throw on invalid empirical level', () => {
    expect(() => createClaim(
      'Test',
      99 as EmpiricalLevel,
      NormativeLevel.N0_Personal,
      MaterialityLevel.M0_Ephemeral,
      'issuer'
    )).toThrow(/Invalid empirical level/);
  });

  it('should throw on empty content in ClaimBuilder', () => {
    expect(() => claim('')).toThrow(/content must not be empty/);
  });

  it('should throw on past expiration in ClaimBuilder', () => {
    expect(() => claim('Test').withExpiration(Date.now() - 1000)).toThrow(/must be in the future/);
  });

  it('should throw on invalid evidence in addEvidence', () => {
    const c = createClaim(
      'Test',
      EmpiricalLevel.E1_Testimonial,
      NormativeLevel.N0_Personal,
      MaterialityLevel.M0_Ephemeral,
      'issuer'
    );

    expect(() => addEvidence(c, {
      type: '',
      data: 'data',
      source: 'source',
      timestamp: Date.now(),
    })).toThrow();
  });

  it('should throw on strict parse with invalid code', () => {
    expect(() => parseClassificationCodeStrict('invalid')).toThrow(/Invalid classification code/);
    expect(() => parseClassificationCodeStrict('E5-N2-M2')).toThrow();
  });

  it('should parse valid code with strict', () => {
    const result = parseClassificationCodeStrict('E3-N2-M2');
    expect(result.empirical).toBe(EmpiricalLevel.E3_Cryptographic);
  });
});

describe('EpistemicBatch', () => {
  it('should add claims with builder callback', () => {
    const batch = new EpistemicBatch();
    batch.addClaim('Claim 1', (b: any) => b
      .withEmpirical(EmpiricalLevel.E3_Cryptographic)
      .withNormative(NormativeLevel.N2_Network));
    batch.addClaim('Claim 2');

    expect(batch.size).toBe(2);
    const claims = batch.getClaims();
    expect(claims[0].content).toBe('Claim 1');
    expect(claims[0].classification.empirical).toBe(EmpiricalLevel.E3_Cryptographic);
    expect(claims[1].classification.empirical).toBe(EmpiricalLevel.E0_Unverified);
  });

  it('should support chaining when adding claims', () => {
    const batch = new EpistemicBatch()
      .addClaim('First')
      .addClaim('Second')
      .addClaim('Third');

    expect(batch.size).toBe(3);
  });

  it('should filter by standard', () => {
    const batch = new EpistemicBatch();
    batch.addClaim('Low trust', (b: any) => b
      .withEmpirical(EmpiricalLevel.E1_Testimonial)
      .withNormative(NormativeLevel.N0_Personal));
    batch.addClaim('High trust', (b: any) => b
      .withEmpirical(EmpiricalLevel.E3_Cryptographic)
      .withNormative(NormativeLevel.N2_Network));

    const highTrust = batch.filterByStandard(
      EmpiricalLevel.E2_PrivateVerify,
      NormativeLevel.N1_Communal
    );

    expect(highTrust.length).toBe(1);
    expect(highTrust[0].content).toBe('High trust');
  });

  it('should filter by preset standard', () => {
    const batch = new EpistemicBatch();
    batch.addClaim('Low', (b: any) => b
      .withEmpirical(EmpiricalLevel.E1_Testimonial)
      .withNormative(NormativeLevel.N0_Personal)
      .withMateriality(MaterialityLevel.M0_Ephemeral));
    batch.addClaim('High', (b: any) => b
      .withEmpirical(EmpiricalLevel.E3_Cryptographic)
      .withNormative(NormativeLevel.N2_Network)
      .withMateriality(MaterialityLevel.M2_Persistent));

    const highTrust = batch.filterByPreset(Standards.HighTrust);
    expect(highTrust.length).toBe(1);
    expect(highTrust[0].content).toBe('High');

    const lowTrust = batch.filterByPreset(Standards.LowTrust);
    expect(lowTrust.length).toBe(2); // Both meet low trust threshold
  });

  it('should filter out expired claims', () => {
    const batch = new EpistemicBatch();
    // Add a claim with future expiration
    batch.addClaim('Valid', (b: any) => b.withExpiration(Date.now() + 60000));
    // Add a claim without expiration (never expires)
    batch.addClaim('No expiry');
    // Add existing claim that's expired
    const expiredClaim = claim('Expired')
      .withExpiration(Date.now() + 1)
      .build();
    // Wait a tiny bit for it to expire
    const originalExpiresAt = expiredClaim.expiresAt;
    (expiredClaim as any).expiresAt = Date.now() - 1000; // Force expired
    batch.addExistingClaim(expiredClaim);

    const valid = batch.filterByExpiry();
    expect(valid.length).toBe(2);
    expect(valid.map((c: any) => c.content)).toContain('Valid');
    expect(valid.map((c: any) => c.content)).toContain('No expiry');
  });

  it('should generate summary statistics', () => {
    const batch = new EpistemicBatch();
    batch.addClaim('Claim 1', (b: any) => b
      .withEmpirical(EmpiricalLevel.E3_Cryptographic)
      .withNormative(NormativeLevel.N2_Network)
      .withMateriality(MaterialityLevel.M2_Persistent));
    batch.addClaim('Claim 2', (b: any) => b
      .withEmpirical(EmpiricalLevel.E3_Cryptographic)
      .withNormative(NormativeLevel.N1_Communal)
      .withMateriality(MaterialityLevel.M1_Temporal));
    batch.addClaim('Claim 3', (b: any) => b
      .withEmpirical(EmpiricalLevel.E1_Testimonial)
      .withNormative(NormativeLevel.N0_Personal)
      .withMateriality(MaterialityLevel.M0_Ephemeral));

    const summary = batch.summary();
    expect(summary.total).toBe(3);
    expect(summary.expired).toBe(0);
    expect(summary.valid).toBe(3);
    expect(summary.byEmpiricalLevel['E3_Cryptographic']).toBe(2);
    expect(summary.byEmpiricalLevel['E1_Testimonial']).toBe(1);
    expect(summary.byNormativeLevel['N2_Network']).toBe(1);
    expect(summary.byMaterialityLevel['M2_Persistent']).toBe(1);
  });

  it('should clear all claims', () => {
    const batch = new EpistemicBatch()
      .addClaim('One')
      .addClaim('Two');

    expect(batch.size).toBe(2);
    batch.clear();
    expect(batch.size).toBe(0);
    expect(batch.getClaims()).toEqual([]);
  });

  it('should filter claims with predicate', () => {
    const batch = new EpistemicBatch()
      .addClaim('Alice verified')
      .addClaim('Bob verified')
      .addClaim('Charlie unverified');

    const aliceBatch = batch.filter((c: any) => c.content.includes('Alice'));
    expect(aliceBatch.size).toBe(1);
    expect(aliceBatch.getClaims()[0].content).toBe('Alice verified');

    // Original batch unchanged
    expect(batch.size).toBe(3);
  });

  it('should map claims to values', () => {
    const batch = new EpistemicBatch()
      .addClaim('Claim A')
      .addClaim('Claim B');

    const contents = batch.map((c: any) => c.content);
    expect(contents).toEqual(['Claim A', 'Claim B']);

    const lengths = batch.map((c: any) => c.content.length);
    expect(lengths).toEqual([7, 7]);
  });

  it('should add existing claims', () => {
    const existingClaim = claim('Pre-built claim')
      .withEmpirical(EmpiricalLevel.E2_PrivateVerify)
      .build();

    const batch = new EpistemicBatch();
    batch.addExistingClaim(existingClaim);

    expect(batch.size).toBe(1);
    expect(batch.getClaims()[0]).toBe(existingClaim);
  });
});

describe('EpistemicClaimPool', () => {
  it('should add and retrieve claims', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 60000 });
    const c = claim('Test claim').build();

    pool.add(c);
    expect(pool.size).toBe(1);
    expect(pool.get(c.id)).toBe(c);
    expect(pool.has(c.id)).toBe(true);

    pool.stop();
  });

  it('should apply TTL multipliers based on empirical level', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 10000 });

    const unverified = claim('Unverified').build(); // E0
    const cryptographic = claim('Crypto')
      .withEmpirical(EmpiricalLevel.E3_Cryptographic)
      .build();

    pool.add(unverified);
    pool.add(cryptographic);

    // E0 gets 25% = 2500ms, E3 gets 200% = 20000ms
    const unverifiedTtl = pool.getRemainingTtl(unverified.id);
    const cryptoTtl = pool.getRemainingTtl(cryptographic.id);

    expect(unverifiedTtl).toBeLessThan(3000);
    expect(cryptoTtl).toBeGreaterThan(15000);

    pool.stop();
  });

  it('should respect explicit expiration over calculated TTL', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 1000 });
    const futureExpiry = Date.now() + 60000; // 1 minute

    const c = claim('Explicit expiry')
      .withExpiration(futureExpiry)
      .build();

    pool.add(c);
    const ttl = pool.getRemainingTtl(c.id);

    expect(ttl).toBeGreaterThan(50000);
    expect(ttl).toBeLessThanOrEqual(60000);

    pool.stop();
  });

  it('should identify expired claims', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 60000 });

    const active = claim('Active').build();
    const expired = claim('Expired')
      .withExpiration(Date.now() + 1)
      .build();

    pool.add(active);
    pool.add(expired);

    // Force expiration
    (expired as any).expiresAt = Date.now() - 1000;
    // Re-add to update internal tracking
    pool.remove(expired.id);
    pool.add(expired);

    expect(pool.getActive().length).toBe(1);
    expect(pool.getExpired().length).toBe(1);
    expect(pool.get(expired.id)).toBeUndefined();

    pool.stop();
  });

  it('should cleanup expired claims', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 60000 });

    const active = claim('Active').build();
    pool.add(active);

    // Create and add an already-expired claim
    const expiredClaim = claim('Expired').build();
    pool.add(expiredClaim);
    // Manually set expiry to past
    (pool as any).effectiveExpiry.set(expiredClaim.id, Date.now() - 1000);

    expect(pool.size).toBe(2);

    const removed = pool.cleanup();
    expect(removed).toBe(1);
    expect(pool.size).toBe(1);
    expect(pool.has(active.id)).toBe(true);

    pool.stop();
  });

  it('should extend claim TTL', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 10000 });
    const c = claim('Extendable').build();

    pool.add(c);
    const originalTtl = pool.getRemainingTtl(c.id);

    pool.extend(c.id, 5000);
    const newTtl = pool.getRemainingTtl(c.id);

    expect(newTtl).toBeGreaterThan(originalTtl);
    expect(newTtl - originalTtl).toBeCloseTo(5000, -2);

    pool.stop();
  });

  it('should get pool statistics', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 60000 });

    pool.add(claim('One').build());
    pool.add(claim('Two').build());

    const stats = pool.getStats();

    expect(stats.total).toBe(2);
    expect(stats.active).toBe(2);
    expect(stats.expired).toBe(0);
    expect(stats.cleanupCount).toBe(0);

    pool.stop();
  });

  it('should enforce max claims limit', () => {
    const pool = new EpistemicClaimPool({
      defaultTtlMs: 60000,
      maxClaims: 3,
    });

    pool.add(claim('One').build());
    pool.add(claim('Two').build());
    pool.add(claim('Three').build());
    pool.add(claim('Four').build()); // Should evict oldest

    expect(pool.size).toBe(3);

    pool.stop();
  });

  it('should add many claims at once', () => {
    const pool = new EpistemicClaimPool();
    const claims = [
      claim('A').build(),
      claim('B').build(),
      claim('C').build(),
    ];

    pool.addMany(claims);
    expect(pool.size).toBe(3);

    pool.stop();
  });

  it('should remove claims by ID', () => {
    const pool = new EpistemicClaimPool();
    const c = claim('Removable').build();

    pool.add(c);
    expect(pool.has(c.id)).toBe(true);

    const removed = pool.remove(c.id);
    expect(removed).toBe(true);
    expect(pool.has(c.id)).toBe(false);

    pool.stop();
  });

  it('should clear all claims', () => {
    const pool = new EpistemicClaimPool();
    pool.add(claim('One').build());
    pool.add(claim('Two').build());

    pool.clear();
    expect(pool.size).toBe(0);

    pool.stop();
  });

  it('should track activeSize separately from size', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 60000 });

    pool.add(claim('Active').build());
    const expiredClaim = claim('Expired').build();
    pool.add(expiredClaim);
    // Force expired
    (pool as any).effectiveExpiry.set(expiredClaim.id, Date.now() - 1000);

    expect(pool.size).toBe(2);
    expect(pool.activeSize).toBe(1);

    pool.stop();
  });

  it('should find claims expiring soon', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 60000 });

    const expiringSoon = claim('Soon').build();
    pool.add(expiringSoon);
    // Set to expire in 3 seconds
    (pool as any).effectiveExpiry.set(expiringSoon.id, Date.now() + 3000);

    const notSoon = claim('Not soon').build();
    pool.add(notSoon);
    // Set to expire in 30 seconds
    (pool as any).effectiveExpiry.set(notSoon.id, Date.now() + 30000);

    const expiring = pool.getExpiringSoon(5000); // Within 5 seconds
    expect(expiring.length).toBe(1);
    expect(expiring[0].content).toBe('Soon');

    pool.stop();
  });
});
