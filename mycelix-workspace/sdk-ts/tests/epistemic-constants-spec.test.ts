// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { describe, it, expect } from 'vitest';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import {
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  classificationCode,
} from '../src/epistemic/index.js';

function loadSpec(): Record<string, string> {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const specPath = path.resolve(__dirname, '../../shared/epistemic_spec.toml');

  const contents = fs.readFileSync(specPath, 'utf8');
  const result: Record<string, string> = {};

  for (const rawLine of contents.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const [keyPart, valuePart] = line.split('=', 2);
    if (!keyPart || !valuePart) continue;
    const key = keyPart.trim();
    const value = valuePart.trim().replace(/^"|"$/g, '');
    if (key && value) {
      result[key] = value;
    }
  }

  return result;
}

describe('Epistemic classification spec consistency', () => {
  it('matches shared/epistemic_spec.toml', () => {
    const spec = loadSpec();

    const basic = {
      empirical: EmpiricalLevel.E1_Testimonial,
      normative: NormativeLevel.N1_Communal,
      materiality: MaterialityLevel.M1_Temporal,
    };
    expect(classificationCode(basic)).toBe(spec['basic']);

    const financial = {
      empirical: EmpiricalLevel.E3_Cryptographic,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    };
    expect(classificationCode(financial)).toBe(spec['financial']);

    const foundational = {
      empirical: EmpiricalLevel.E4_Consensus,
      normative: NormativeLevel.N3_Universal,
      materiality: MaterialityLevel.M3_Immutable,
    };
    expect(classificationCode(foundational)).toBe(spec['foundational']);
  });
});

