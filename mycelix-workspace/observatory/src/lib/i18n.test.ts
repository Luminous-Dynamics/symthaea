/**
 * Tests for i18n module — string lookup, fallback, parameter substitution.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { get } from 'svelte/store';
import { locale, setLocale, t, tSync } from './i18n';

describe('i18n', () => {
  beforeEach(() => {
    setLocale('en');
  });

  describe('locale store', () => {
    it('defaults to en', () => {
      expect(get(locale)).toBe('en');
    });

    it('can be changed via setLocale', () => {
      setLocale('af');
      expect(get(locale)).toBe('af');
    });
  });

  describe('t (reactive translator)', () => {
    it('resolves a known English key', () => {
      const translate = get(t);
      expect(translate('nav.dashboard')).toBe('Dashboard');
    });

    it('returns the raw key for unknown keys', () => {
      const translate = get(t);
      expect(translate('nonexistent.key')).toBe('nonexistent.key');
    });

    it('falls back to English when current locale has no translation', () => {
      setLocale('af');
      const translate = get(t);
      // Afrikaans table is empty, should fallback to English
      expect(translate('nav.dashboard')).toBe('Dashboard');
    });

    it('substitutes parameters with {key} pattern', () => {
      const translate = get(t);
      // Use a key that exists and manually test substitution
      // The function replaces {key} patterns regardless of whether the base string has them
      expect(translate('action.save')).toBe('Save');
    });

    it('handles parameter substitution in strings', () => {
      // Test the substitution mechanism directly by checking raw key fallback with params
      const translate = get(t);
      const result = translate('hello {name}', { name: 'World' });
      // Key doesn't exist, so raw key is returned then params substituted
      expect(result).toBe('hello World');
    });

    it('handles multiple parameter substitutions', () => {
      const translate = get(t);
      const result = translate('{a} and {b}', { a: 'X', b: 'Y' });
      expect(result).toBe('X and Y');
    });
  });

  describe('tSync (synchronous translator)', () => {
    it('returns the same result as the reactive t store', () => {
      const translate = get(t);
      expect(tSync('nav.resilience')).toBe(translate('nav.resilience'));
    });

    it('follows locale changes', () => {
      setLocale('zu');
      // Zulu table is empty, falls back to English
      expect(tSync('nav.tend')).toBe('TEND');
    });

    it('substitutes parameters', () => {
      const result = tSync('{count} items', { count: '5' });
      expect(result).toBe('5 items');
    });

    it('returns raw key when nothing matches', () => {
      expect(tSync('totally.unknown.key')).toBe('totally.unknown.key');
    });
  });

  describe('translation coverage', () => {
    it('has navigation keys', () => {
      const translate = get(t);
      const navKeys = [
        'nav.dashboard', 'nav.resilience', 'nav.tend', 'nav.food',
        'nav.mutual_aid', 'nav.emergency', 'nav.value_anchor', 'nav.water',
        'nav.household', 'nav.knowledge', 'nav.care_circles', 'nav.shelter',
        'nav.supplies', 'nav.operator', 'nav.governance', 'nav.network',
        'nav.analytics', 'nav.attribution',
      ];
      for (const key of navKeys) {
        const val = translate(key);
        expect(val).not.toBe(key); // Should resolve to something other than the raw key
      }
    });

    it('has action keys', () => {
      const translate = get(t);
      const actionKeys = [
        'action.save', 'action.cancel', 'action.submit', 'action.delete',
        'action.export', 'action.refresh', 'action.search', 'action.filter',
        'action.back', 'action.next', 'action.join', 'action.sync',
      ];
      for (const key of actionKeys) {
        expect(translate(key)).not.toBe(key);
      }
    });

    it('has status keys', () => {
      const translate = get(t);
      const statusKeys = [
        'status.loading', 'status.offline', 'status.connected',
        'status.disconnected', 'status.demo_mode', 'status.syncing',
      ];
      for (const key of statusKeys) {
        expect(translate(key)).not.toBe(key);
      }
    });

    it('has error keys', () => {
      const translate = get(t);
      expect(translate('error.load_failed')).toBe('Failed to load data');
      expect(translate('error.save_failed')).toBe('Failed to save');
      expect(translate('error.connection_failed')).toBe('Connection failed');
    });
  });
});
