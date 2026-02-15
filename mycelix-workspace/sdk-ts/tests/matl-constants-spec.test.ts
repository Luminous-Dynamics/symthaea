import { describe, it, expect } from 'vitest';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import {
  DEFAULT_BYZANTINE_THRESHOLD,
  DEFAULT_CONSISTENCY_WEIGHT,
  DEFAULT_QUALITY_WEIGHT,
  DEFAULT_REPUTATION_WEIGHT,
  MAX_BYZANTINE_TOLERANCE,
} from '../src/index.js';

function loadSpec(): Record<string, number> {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const specPath = path.resolve(__dirname, '../../shared/matl_constants.toml');

  const contents = fs.readFileSync(specPath, 'utf8');
  const result: Record<string, number> = {};

  for (const rawLine of contents.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;

    const [keyPart, valuePart] = line.split('=', 2);
    if (!keyPart || !valuePart) {
      throw new Error(`Invalid line in spec: ${rawLine}`);
    }

    const key = keyPart.trim();
    const value = Number.parseFloat(valuePart.trim());
    if (!Number.isFinite(value)) {
      throw new Error(`Invalid numeric value for ${key}: ${valuePart}`);
    }

    result[key] = value;
  }

  return result;
}

describe('MATL constants spec consistency', () => {
  it('matches shared/matl_constants.toml', () => {
    const spec = loadSpec();

    expect(DEFAULT_QUALITY_WEIGHT).toBeCloseTo(spec['default_quality_weight'], 10);
    expect(DEFAULT_CONSISTENCY_WEIGHT).toBeCloseTo(spec['default_consistency_weight'], 10);
    expect(DEFAULT_REPUTATION_WEIGHT).toBeCloseTo(spec['default_reputation_weight'], 10);
    expect(MAX_BYZANTINE_TOLERANCE).toBeCloseTo(spec['max_byzantine_tolerance'], 10);
    expect(DEFAULT_BYZANTINE_THRESHOLD).toBeCloseTo(spec['default_byzantine_threshold'], 10);
  });
});

