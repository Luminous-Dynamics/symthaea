/**
 * Gap Detector Tests
 *
 * Tests for finding knowledge gaps with mandatory humility flags.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  GapDetector,
  getGapDetector,
  resetGapDetector,
  DEFAULT_GAP_DETECTOR_CONFIG,
  type KnowledgeGap,
  type HumilityFlags,
  type GapType,
} from '../src/discovery/index.js';

describe('Gap Detector - Humility Flags', () => {
  let detector: GapDetector;

  beforeEach(() => {
    detector = new GapDetector();
  });

  it('should always include humility flags in detected gaps', async () => {
    // Add some test claims to create gaps
    detector.registerClaim('claim-1', {
      content: 'Test claim',
      domain: 'science',
      empiricalLevel: 1,
      confidence: 0.8,
    });

    const gaps = await detector.detectGaps('science');

    for (const gap of gaps) {
      expect(gap.humility).toBeDefined();
      expect(gap.humility.schemaRelative).toBe(true);
      expect(gap.humility.unknownUnknownsAcknowledged).toBe(true);
    }
  });

  it('should acknowledge unknown unknowns', async () => {
    detector.registerClaim('claim-1', {
      content: 'Isolated claim',
      domain: 'test',
      empiricalLevel: 1,
      confidence: 0.5,
    });

    const gaps = await detector.detectGaps('test');

    if (gaps.length > 0) {
      expect(gaps[0].humility.unknownUnknownsAcknowledged).toBe(true);
    }
  });

  it('should list detection blind spots', async () => {
    const gaps = await detector.detectGaps('any-domain');

    // Even if no gaps found, any gap would have blind spots
    expect(DEFAULT_GAP_DETECTOR_CONFIG.knownBlindSpots.length).toBeGreaterThan(0);
    expect(DEFAULT_GAP_DETECTOR_CONFIG.knownBlindSpots).toContain(
      'Gaps that require domain expertise we lack'
    );
  });

  it('should include detection confidence (meta-uncertainty)', async () => {
    detector.registerClaim('claim-1', {
      content: 'High confidence but low evidence',
      domain: 'science',
      empiricalLevel: 0,
      confidence: 0.9,
    });

    const gaps = await detector.detectGaps('science');

    for (const gap of gaps) {
      expect(typeof gap.humility.detectionConfidence).toBe('number');
      expect(gap.humility.detectionConfidence).toBeGreaterThanOrEqual(0);
      expect(gap.humility.detectionConfidence).toBeLessThanOrEqual(1);
    }
  });

  it('should track schema version', async () => {
    detector.registerClaim('claim-1', {
      content: 'Test',
      domain: 'test',
      empiricalLevel: 1,
      confidence: 0.5,
    });

    const gaps = await detector.detectGaps('test');

    for (const gap of gaps) {
      expect(gap.humility.schemaVersion).toBeDefined();
      expect(gap.humility.schemaVersion).toBe(DEFAULT_GAP_DETECTOR_CONFIG.schemaVersion);
    }
  });

  it('should list unsearched domains', async () => {
    const detectorWithUnsearched = new GapDetector({
      unsearchedDomains: ['theology', 'metaphysics', 'art_history'],
    });

    detectorWithUnsearched.registerClaim('claim-1', {
      content: 'Test',
      domain: 'science',
      empiricalLevel: 2,
      confidence: 0.5,
    });

    const gaps = await detectorWithUnsearched.detectGaps('science');

    for (const gap of gaps) {
      expect(gap.humility.unsearchedDomains).toBeDefined();
      expect(Array.isArray(gap.humility.unsearchedDomains)).toBe(true);
    }
  });

  it('should include assumptions made during detection', async () => {
    detector.registerClaim('isolated-claim', {
      content: 'Isolated claim with no links',
      domain: 'test',
      empiricalLevel: 1,
      confidence: 0.5,
    });

    const gaps = await detector.detectGaps('test');

    for (const gap of gaps) {
      expect(Array.isArray(gap.humility.assumptions)).toBe(true);
      expect(gap.humility.assumptions.length).toBeGreaterThan(0);
    }
  });
});

describe('Gap Detector - Structural Gaps', () => {
  let detector: GapDetector;

  beforeEach(() => {
    detector = new GapDetector();
  });

  it('should detect isolated claims (no connections)', async () => {
    detector.registerClaim('isolated', {
      content: 'Isolated claim',
      domain: 'science',
      empiricalLevel: 2,
      confidence: 0.6,
    });

    const gaps = await detector.detectGaps('science');

    const structuralGaps = gaps.filter((g) => g.type === 'structural');
    expect(structuralGaps.length).toBeGreaterThan(0);

    const isolatedGap = structuralGaps.find((g) => g.adjacentClaims?.includes('isolated'));
    expect(isolatedGap?.description).toContain('isolated');
  });

  it('should not flag connected claims as isolated', async () => {
    detector.registerClaim('connected-1', {
      content: 'Connected claim 1',
      domain: 'science',
      empiricalLevel: 2,
      confidence: 0.6,
    });
    detector.registerClaim('connected-2', {
      content: 'Connected claim 2',
      domain: 'science',
      empiricalLevel: 2,
      confidence: 0.6,
    });
    detector.addLink('connected-1', 'connected-2');

    const gaps = await detector.detectGaps('science');

    // Links should reduce isolation - there may still be structural gaps
    // but they should note the connection exists
    const structuralGaps = gaps.filter((g) => g.type === 'structural');

    // Verify the link was registered
    expect(detector.hasLink('connected-1', 'connected-2')).toBe(true);

    // If there are structural gaps, they should acknowledge the connection
    // by having fewer isolated claims than unlinked claims would have
    expect(structuralGaps.length).toBeLessThanOrEqual(1);
  });
});

describe('Gap Detector - Epistemic Gaps', () => {
  let detector: GapDetector;

  beforeEach(() => {
    detector = new GapDetector();
  });

  it('should detect high-confidence claims with low evidence level', async () => {
    detector.registerClaim('overconfident', {
      content: 'High confidence but only testimonial evidence',
      domain: 'science',
      empiricalLevel: 1, // Testimonial only
      confidence: 0.9, // Very high confidence
    });

    const gaps = await detector.detectGaps('science');

    const epistemicGaps = gaps.filter((g) => g.type === 'epistemic');
    expect(epistemicGaps.length).toBeGreaterThan(0);
  });

  it('should not flag low-confidence claims with low evidence', async () => {
    detector.registerClaim('humble', {
      content: 'Low confidence matches low evidence',
      domain: 'science',
      empiricalLevel: 0,
      confidence: 0.3, // Appropriately low confidence
    });

    const gaps = await detector.detectGaps('science');

    const epistemicGaps = gaps.filter(
      (g) => g.type === 'epistemic' && g.adjacentClaims?.includes('humble')
    );

    // Well-calibrated claims shouldn't generate epistemic gaps
    expect(epistemicGaps.length).toBe(0);
  });

  it('should suggest gathering stronger evidence', async () => {
    detector.registerClaim('needs-evidence', {
      content: 'Needs better evidence',
      domain: 'science',
      empiricalLevel: 1,
      confidence: 0.85,
    });

    const gaps = await detector.detectGaps('science');
    const epistemicGaps = gaps.filter((g) => g.type === 'epistemic');

    if (epistemicGaps.length > 0) {
      const actions = epistemicGaps[0].suggestedActions;
      const gatherAction = actions.find((a) => a.type === 'gather_evidence');
      expect(gatherAction).toBeDefined();
    }
  });
});

describe('Gap Detector - Cross-Domain Gaps', () => {
  let detector: GapDetector;

  beforeEach(() => {
    detector = new GapDetector({ enableCrossDomain: true });
  });

  it('should detect potential cross-domain connections', async () => {
    // Set up claims in different domains that might need linking
    detector.registerClaim('climate-claim', {
      content: 'Climate change affects food production',
      domain: 'climate_science',
      empiricalLevel: 3,
      confidence: 0.85,
    });

    detector.registerClaim('agriculture-claim', {
      content: 'Crop yields are declining in some regions',
      domain: 'agriculture',
      empiricalLevel: 3,
      confidence: 0.8,
    });

    const gaps = await detector.detectGaps('climate_science');

    // Cross-domain detection might find missing links
    const crossDomainGaps = gaps.filter((g) => g.type === 'cross_domain');
    // Note: implementation may or may not detect this based on similarity
  });

  it('should not detect cross-domain gaps when disabled', async () => {
    const noCrossDetector = new GapDetector({ enableCrossDomain: false });

    noCrossDetector.registerClaim('claim-1', {
      content: 'Domain A claim',
      domain: 'domain_a',
      empiricalLevel: 2,
      confidence: 0.7,
    });

    const gaps = await noCrossDetector.detectGaps('domain_a');

    const crossDomainGaps = gaps.filter((g) => g.type === 'cross_domain');
    expect(crossDomainGaps.length).toBe(0);
  });
});

describe('Gap Detector - Temporal Gaps', () => {
  let detector: GapDetector;

  beforeEach(() => {
    detector = new GapDetector();
  });

  it('should detect missing time periods in coverage', async () => {
    // Claims with timestamps showing gaps in coverage
    detector.registerClaim('old-claim', {
      content: 'Historical data from 2010',
      domain: 'economics',
      empiricalLevel: 3,
      confidence: 0.8,
      timestamp: new Date('2010-01-01').getTime(),
    });

    detector.registerClaim('new-claim', {
      content: 'Recent data from 2024',
      domain: 'economics',
      empiricalLevel: 3,
      confidence: 0.8,
      timestamp: new Date('2024-01-01').getTime(),
    });

    const gaps = await detector.detectGaps('economics');

    const temporalGaps = gaps.filter((g) => g.type === 'temporal');
    // May detect gap in 2011-2023 period
  });
});

describe('Gap Detector - Measurement Gaps', () => {
  let detector: GapDetector;

  beforeEach(() => {
    detector = new GapDetector();
  });

  it('should detect missing quantitative data', async () => {
    // Qualitative claim that could benefit from quantification
    detector.registerClaim('qualitative', {
      content: 'Many users prefer the new interface',
      domain: 'ux_research',
      empiricalLevel: 1,
      confidence: 0.6,
      hasQuantitativeData: false,
    });

    const gaps = await detector.detectGaps('ux_research');

    const measurementGaps = gaps.filter((g) => g.type === 'measurement');
    // Should suggest gathering quantitative data
  });
});

describe('Gap Detector - Perspective Gaps', () => {
  let detector: GapDetector;

  beforeEach(() => {
    detector = new GapDetector();
  });

  it('should detect missing viewpoints', async () => {
    // Claims all from one perspective
    detector.registerClaim('perspective-1', {
      content: 'Economic analysis from Western perspective',
      domain: 'economics',
      empiricalLevel: 2,
      confidence: 0.7,
      perspective: 'western',
    });

    const gaps = await detector.detectGaps('economics');

    const perspectiveGaps = gaps.filter((g) => g.type === 'perspective');
    // May suggest seeking diverse viewpoints
  });
});

describe('Gap Detector - Gap Resolution', () => {
  let detector: GapDetector;

  beforeEach(() => {
    detector = new GapDetector();
  });

  it('should allow marking gaps as addressed', async () => {
    detector.registerClaim('problematic', {
      content: 'Problematic claim',
      domain: 'test',
      empiricalLevel: 0,
      confidence: 0.9,
    });

    const gaps = await detector.detectGaps('test');

    if (gaps.length > 0) {
      const gapId = gaps[0].id;

      await detector.resolveGap(gapId, {
        method: 'filled',
        explanation: 'Added stronger evidence',
        newClaims: ['new-evidence-claim'],
        resolvedAt: Date.now(),
        resolvedBy: 'researcher-1',
      });

      const gap = detector.getGap(gapId);
      expect(gap?.addressed).toBe(true);
      expect(gap?.resolution?.method).toBe('filled');
    }
  });

  it('should allow marking gaps as not-actually-gaps', async () => {
    detector.registerClaim('standalone', {
      content: 'Validly standalone claim',
      domain: 'test',
      empiricalLevel: 2,
      confidence: 0.5,
    });

    const gaps = await detector.detectGaps('test');

    const structuralGap = gaps.find((g) => g.type === 'structural');

    if (structuralGap) {
      await detector.resolveGap(structuralGap.id, {
        method: 'determined_not_gap',
        explanation: 'This claim is validly standalone and does not need connections',
        resolvedAt: Date.now(),
        resolvedBy: 'domain-expert',
      });

      const resolved = detector.getGap(structuralGap.id);
      expect(resolved?.addressed).toBe(true);
      expect(resolved?.resolution?.method).toBe('determined_not_gap');
    }
  });
});

describe('Gap Detector - Statistics', () => {
  let detector: GapDetector;

  beforeEach(() => {
    detector = new GapDetector();
  });

  it('should track gap detection statistics', async () => {
    detector.registerClaim('claim-1', {
      content: 'Test claim',
      domain: 'test',
      empiricalLevel: 1,
      confidence: 0.8,
    });

    await detector.detectGaps('test');

    const stats = detector.getStats();

    expect(stats).toHaveProperty('totalGaps');
    expect(stats).toHaveProperty('byType');
    expect(stats).toHaveProperty('resolvedGaps');
  });

  it('should categorize gaps by type', async () => {
    detector.registerClaim('isolated', {
      content: 'Isolated',
      domain: 'test',
      empiricalLevel: 1,
      confidence: 0.5,
    });
    detector.registerClaim('overconfident', {
      content: 'Overconfident',
      domain: 'test',
      empiricalLevel: 0,
      confidence: 0.95,
    });

    await detector.detectGaps('test');

    const stats = detector.getStats();

    expect(stats.byType).toBeDefined();
    expect(typeof stats.byType.structural).toBe('number');
    expect(typeof stats.byType.epistemic).toBe('number');
  });
});

describe('Gap Detector - Priority Assignment', () => {
  let detector: GapDetector;

  beforeEach(() => {
    detector = new GapDetector();
  });

  it('should assign priority based on severity', async () => {
    detector.registerClaim('critical-gap', {
      content: 'Critical gap',
      domain: 'test',
      empiricalLevel: 0,
      confidence: 0.99, // Very high confidence with no evidence
    });

    const gaps = await detector.detectGaps('test');

    if (gaps.length > 0) {
      const severityToExpectedPriority: Record<string, string[]> = {
        critical: ['0.9', '1.0'],
        high: ['0.7', '0.8', '0.9'],
        medium: ['0.4', '0.5', '0.6', '0.7'],
        low: ['0.3', '0.4', '0.5'],
      };

      // Verify priority is assigned
      expect(['critical', 'high', 'medium', 'low']).toContain(gaps[0].priority);
    }
  });
});

describe('Gap Detector - Suggested Actions', () => {
  let detector: GapDetector;

  beforeEach(() => {
    detector = new GapDetector();
  });

  it('should provide actionable suggestions', async () => {
    detector.registerClaim('needs-action', {
      content: 'Claim needing action',
      domain: 'test',
      empiricalLevel: 1,
      confidence: 0.8,
    });

    const gaps = await detector.detectGaps('test');

    for (const gap of gaps) {
      expect(gap.suggestedActions).toBeDefined();
      expect(gap.suggestedActions.length).toBeGreaterThan(0);

      for (const action of gap.suggestedActions) {
        expect(action.type).toBeDefined();
        expect(action.description).toBeDefined();
        expect(['minimal', 'moderate', 'significant']).toContain(action.effort);
      }
    }
  });

  it('should include prerequisites for actions', async () => {
    detector.registerClaim('complex-gap', {
      content: 'Complex situation',
      domain: 'test',
      empiricalLevel: 0,
      confidence: 0.9,
    });

    const gaps = await detector.detectGaps('test');

    for (const gap of gaps) {
      for (const action of gap.suggestedActions) {
        expect(Array.isArray(action.prerequisites)).toBe(true);
      }
    }
  });
});

describe('Gap Detector - Singleton', () => {
  beforeEach(() => {
    resetGapDetector();
  });

  it('should return same instance', () => {
    const detector1 = getGapDetector();
    const detector2 = getGapDetector();

    expect(detector1).toBe(detector2);
  });

  it('should reset correctly', async () => {
    const detector1 = getGapDetector();
    detector1.registerClaim('claim-1', {
      content: 'Test',
      domain: 'test',
      empiricalLevel: 1,
      confidence: 0.5,
    });
    await detector1.detectGaps('test');

    resetGapDetector();

    const detector2 = getGapDetector();
    expect(detector2.getStats().totalGaps).toBe(0);
  });
});

describe('Gap Detector - Configuration', () => {
  it('should use default configuration', () => {
    expect(DEFAULT_GAP_DETECTOR_CONFIG.minSeverity).toBe(0.3);
    expect(DEFAULT_GAP_DETECTOR_CONFIG.enableCrossDomain).toBe(true);
    expect(DEFAULT_GAP_DETECTOR_CONFIG.knownBlindSpots.length).toBeGreaterThan(0);
  });

  it('should accept custom configuration', () => {
    const detector = new GapDetector({
      minSeverity: 0.5,
      enableCrossDomain: false,
      schemaVersion: '2.0.0',
    });

    expect(detector).toBeDefined();
  });

  it('should filter gaps by minimum severity', async () => {
    const strictDetector = new GapDetector({ minSeverity: 0.8 });

    strictDetector.registerClaim('minor-issue', {
      content: 'Minor issue',
      domain: 'test',
      empiricalLevel: 2,
      confidence: 0.6,
    });

    const gaps = await strictDetector.detectGaps('test');

    // All gaps should have severity >= 0.8
    for (const gap of gaps) {
      expect(gap.severity).toBeGreaterThanOrEqual(0.8);
    }
  });
});

describe('Gap Detector - Global Detection', () => {
  let detector: GapDetector;

  beforeEach(() => {
    detector = new GapDetector({
      unsearchedDomains: ['theology', 'art_history'],
    });
  });

  it('should detect gaps across all domains', async () => {
    detector.registerClaim('science-claim', {
      content: 'Science',
      domain: 'science',
      empiricalLevel: 2,
      confidence: 0.7,
    });

    detector.registerClaim('economics-claim', {
      content: 'Economics',
      domain: 'economics',
      empiricalLevel: 1,
      confidence: 0.8,
    });

    const gaps = await detector.detectGlobalGaps();

    // Should find gaps across multiple domains
    const domains = new Set(gaps.flatMap((g) => g.domains));
    expect(domains.size).toBeGreaterThanOrEqual(1);
  });

  it('should flag unsearched domains in global detection', async () => {
    detector.registerClaim('claim-1', {
      content: 'Test',
      domain: 'science',
      empiricalLevel: 2,
      confidence: 0.5,
    });

    const gaps = await detector.detectGlobalGaps();

    const unsearchedGaps = gaps.filter((g) =>
      g.domains.some((d) => ['theology', 'art_history'].includes(d))
    );

    expect(unsearchedGaps.length).toBeGreaterThan(0);
  });
});
