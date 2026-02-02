import { describe, it, expect } from 'vitest';
import {
  formatRate,
  formatCurrency,
  isValidJurisdiction,
  isValidTaxYear,
  getRegion,
} from '../index';

describe('Utility Functions', () => {
  describe('formatRate', () => {
    it('formats basis points to percentage', () => {
      expect(formatRate(1000)).toBe('10%');
      expect(formatRate(2200)).toBe('22%');
      expect(formatRate(3700)).toBe('37%');
    });
  });

  describe('formatCurrency', () => {
    it('formats USD', () => {
      const result = formatCurrency(85000, 'USD');
      expect(result).toContain('85,000');
    });

    it('formats EUR', () => {
      const result = formatCurrency(85000, 'EUR', 'de-DE');
      expect(result).toContain('85');
    });
  });

  describe('isValidJurisdiction', () => {
    it('validates known jurisdictions', () => {
      expect(isValidJurisdiction('US')).toBe(true);
      expect(isValidJurisdiction('UK')).toBe(true);
      expect(isValidJurisdiction('DE')).toBe(true);
      expect(isValidJurisdiction('JP')).toBe(true);
    });

    it('rejects unknown jurisdictions', () => {
      expect(isValidJurisdiction('XX')).toBe(false);
      expect(isValidJurisdiction('ZZ')).toBe(false);
      expect(isValidJurisdiction('')).toBe(false);
    });
  });

  describe('isValidTaxYear', () => {
    it('validates supported years', () => {
      expect(isValidTaxYear(2020)).toBe(true);
      expect(isValidTaxYear(2024)).toBe(true);
      expect(isValidTaxYear(2025)).toBe(true);
    });

    it('rejects unsupported years', () => {
      expect(isValidTaxYear(2019)).toBe(false);
      expect(isValidTaxYear(2026)).toBe(false);
    });
  });

  describe('getRegion', () => {
    it('returns Americas for American countries', () => {
      expect(getRegion('US')).toBe('Americas');
      expect(getRegion('CA')).toBe('Americas');
      expect(getRegion('BR')).toBe('Americas');
    });

    it('returns Europe for European countries', () => {
      expect(getRegion('UK')).toBe('Europe');
      expect(getRegion('DE')).toBe('Europe');
      expect(getRegion('FR')).toBe('Europe');
    });

    it('returns Asia-Pacific for APAC countries', () => {
      expect(getRegion('JP')).toBe('Asia-Pacific');
      expect(getRegion('AU')).toBe('Asia-Pacific');
      expect(getRegion('SG')).toBe('Asia-Pacific');
    });

    it('returns Middle East for ME countries', () => {
      expect(getRegion('AE')).toBe('Middle East');
      expect(getRegion('SA')).toBe('Middle East');
    });

    it('returns Africa for African countries', () => {
      expect(getRegion('ZA')).toBe('Africa');
      expect(getRegion('NG')).toBe('Africa');
    });
  });
});

describe('Type Definitions', () => {
  it('has all 58 jurisdiction codes', () => {
    const codes = [
      'US', 'CA', 'MX', 'BR', 'AR', 'CL', 'CO', 'PE', 'EC', 'UY',
      'UK', 'DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'AT', 'PT', 'IE',
      'PL', 'SE', 'DK', 'FI', 'NO', 'CH', 'CZ', 'GR', 'HU', 'RO',
      'RU', 'TR', 'UA',
      'JP', 'CN', 'IN', 'KR', 'ID', 'AU', 'NZ', 'SG', 'HK', 'TW',
      'MY', 'TH', 'VN', 'PH', 'PK',
      'SA', 'AE', 'IL', 'EG', 'QA',
      'ZA', 'NG', 'KE', 'MA', 'GH',
    ];
    expect(codes.length).toBe(58);
    codes.forEach(code => {
      expect(isValidJurisdiction(code)).toBe(true);
    });
  });
});
