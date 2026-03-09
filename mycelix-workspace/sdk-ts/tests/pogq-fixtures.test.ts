/**
 * Cross-runtime PoGQ fixture test.
 *
 * Verifies that the TypeScript MATL implementation matches the
 * shared PoGQ composite score fixtures defined under Mycelix-Core.
 */

import fs from 'node:fs';
import path from 'node:path';
import { describe, it, expect } from 'vitest';
import * as matl from '../src/matl';

type PogqFixtureCase = {
  name: string;
  quality: number;
  consistency: number;
  entropy: number;
  reputation: number;
  expected_composite: number;
};

describe('PoGQ fixtures – TS matches shared cases', () => {
  it('should match expected composite scores from shared fixtures', () => {
    const manifestDir = path.resolve(__dirname, '..');
    const fixturePath = path.resolve(
      manifestDir,
      '../../Mycelix-Core/tests/shared-fixtures/pogq/simple_cases.json',
    );

    if (!fs.existsSync(fixturePath)) {
      // Fixtures not available in this checkout (e.g., standalone mycelix repo)
      return;
    }

    const raw = fs.readFileSync(fixturePath, 'utf8');
    const cases: PogqFixtureCase[] = JSON.parse(raw);

    for (const c of cases) {
      const pogq = matl.createPoGQ(c.quality, c.consistency, c.entropy);
      // Shared fixtures were computed without entropy penalty, so disable it
      const composite = matl.compositeScore(pogq, c.reputation, { entropyPenalty: 0 });

      expect(composite).toBeCloseTo(c.expected_composite, 1);
    }
  });
});

