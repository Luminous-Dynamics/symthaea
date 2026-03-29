// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { favorites } from '../stores/favorites';

describe('favorites store', () => {
  beforeEach(() => {
    favorites.clear();
  });

  afterEach(() => {
    favorites.clear();
  });

  it('toggles favorites on and off', async () => {
    let values: string[] = [];
    const unsubscribe = favorites.subscribe((v) => (values = v));

    favorites.toggle('abc');
    expect(values).toContain('abc');

    favorites.toggle('abc');
    expect(values).not.toContain('abc');

    unsubscribe();
  });

  it('adds unique ids and removes them', () => {
    let values: string[] = [];
    const unsubscribe = favorites.subscribe((v) => (values = v));

    favorites.add('x');
    favorites.add('x');
    expect(values).toEqual(['x']);

    favorites.remove('x');
    expect(values).toEqual([]);

    unsubscribe();
  });
});
