import { describe, it, expect } from 'vitest';
import { escCsv, toCsv } from './data-export';

// ============================================================================
// escCsv
// ============================================================================

describe('escCsv', () => {
  it('passes through plain strings', () => {
    expect(escCsv('hello')).toBe('hello');
  });

  it('wraps strings containing commas in quotes', () => {
    expect(escCsv('hello, world')).toBe('"hello, world"');
  });

  it('wraps strings containing double quotes and escapes them', () => {
    expect(escCsv('say "hi"')).toBe('"say ""hi"""');
  });

  it('wraps strings containing newlines in quotes', () => {
    expect(escCsv('line1\nline2')).toBe('"line1\nline2"');
  });

  it('converts numbers to strings', () => {
    expect(escCsv(42)).toBe('42');
    expect(escCsv(3.14)).toBe('3.14');
  });

  it('converts null/undefined to empty string', () => {
    expect(escCsv(null)).toBe('');
    expect(escCsv(undefined)).toBe('');
  });

  it('converts booleans to strings', () => {
    expect(escCsv(true)).toBe('true');
    expect(escCsv(false)).toBe('false');
  });
});

// ============================================================================
// toCsv
// ============================================================================

describe('toCsv', () => {
  it('creates a header-only CSV for empty rows', () => {
    const result = toCsv(['Name', 'Age'], []);
    expect(result).toBe('Name,Age');
  });

  it('creates a valid CSV with headers and rows', () => {
    const result = toCsv(['Name', 'Age'], [['Alice', '30'], ['Bob', '25']]);
    const lines = result.split('\n');
    expect(lines).toHaveLength(3);
    expect(lines[0]).toBe('Name,Age');
    expect(lines[1]).toBe('Alice,30');
    expect(lines[2]).toBe('Bob,25');
  });

  it('escapes cell values that contain commas', () => {
    const result = toCsv(['Name', 'Desc'], [['Alice', 'tall, smart']]);
    expect(result).toContain('"tall, smart"');
  });

  it('handles a single column', () => {
    const result = toCsv(['ID'], [['1'], ['2'], ['3']]);
    expect(result).toBe('ID\n1\n2\n3');
  });

  it('handles empty cell values', () => {
    const result = toCsv(['A', 'B'], [['x', '']]);
    expect(result).toBe('A,B\nx,');
  });
});
