// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Validation Edge Case Tests
 *
 * Comprehensive edge case testing for the validation module.
 * Tests boundary conditions, Unicode handling, and injection prevention.
 */

import { describe, it, expect } from 'vitest';
import {
  validateDID,
  validateNonEmpty,
  validateNonNegative,
  validatePositive,
  validateURL,
  validateNonEmptyArray,
  validatePublicKey,
  validateRange,
  combineResults,
  assertValid,
  validResult,
  invalidResult,
  // Service validators
  FinanceValidators,
  GovernanceValidators,
  IdentityValidators,
  MediaValidators,
  JusticeValidators,
  KnowledgeValidators,
  EnergyValidators,
  PropertyValidators,
} from '../../src/utils/validation.js';

// ============================================================================
// Core Validation Functions
// ============================================================================

describe('Core Validation Functions', () => {
  describe('validateDID', () => {
    it('accepts valid Mycelix DIDs', () => {
      const result = validateDID('did:mycelix:abc123def456xyz789012345678901', 'Test DID');
      expect(result.valid).toBe(true);
    });

    it('rejects empty DID', () => {
      const result = validateDID('', 'Test DID');
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Test DID cannot be empty');
    });

    it('rejects whitespace-only DID', () => {
      const result = validateDID('   ', 'Test DID');
      expect(result.valid).toBe(false);
    });

    it('rejects DID without proper prefix', () => {
      const result = validateDID('not-a-did-at-all', 'Test DID');
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('valid DID format'))).toBe(true);
    });

    it('handles Unicode in identifier part', () => {
      // DIDs can contain various characters after the prefix
      const result = validateDID('did:mycelix:éàü12345678901234567890', 'Test DID');
      // Should still validate based on format rules
      expect(result.valid).toBe(true);
    });

    it('accepts very long valid DIDs', () => {
      const longIdentifier = 'a'.repeat(100);
      const result = validateDID(`did:mycelix:${longIdentifier}`, 'Test DID');
      expect(result.valid).toBe(true);
    });
  });

  describe('validateNonEmpty', () => {
    it('accepts non-empty strings', () => {
      expect(validateNonEmpty('hello', 'Field').valid).toBe(true);
    });

    it('rejects empty strings', () => {
      const result = validateNonEmpty('', 'Field');
      expect(result.valid).toBe(false);
    });

    it('rejects whitespace-only strings', () => {
      const result = validateNonEmpty('   ', 'Field');
      expect(result.valid).toBe(false);
    });

    it('accepts strings with only whitespace padding', () => {
      const result = validateNonEmpty('  hello  ', 'Field');
      expect(result.valid).toBe(true);
    });

    it('handles Unicode whitespace', () => {
      // Non-breaking space and other Unicode whitespace
      const result = validateNonEmpty('\u00A0\u2003', 'Field');
      expect(result.valid).toBe(false);
    });

    it('accepts single character', () => {
      expect(validateNonEmpty('x', 'Field').valid).toBe(true);
    });

    it('handles null-like characters', () => {
      expect(validateNonEmpty('\x00', 'Field').valid).toBe(true); // Null char is still a char
    });
  });

  describe('validateNonNegative', () => {
    it('accepts zero', () => {
      expect(validateNonNegative(0, 'Amount').valid).toBe(true);
    });

    it('accepts positive numbers', () => {
      expect(validateNonNegative(100, 'Amount').valid).toBe(true);
      expect(validateNonNegative(0.001, 'Amount').valid).toBe(true);
    });

    it('rejects negative numbers', () => {
      const result = validateNonNegative(-1, 'Amount');
      expect(result.valid).toBe(false);
    });

    it('handles very small negative', () => {
      const result = validateNonNegative(-0.0000001, 'Amount');
      expect(result.valid).toBe(false);
    });

    it('handles very large positive', () => {
      expect(validateNonNegative(Number.MAX_SAFE_INTEGER, 'Amount').valid).toBe(true);
    });

    it('handles infinity', () => {
      expect(validateNonNegative(Infinity, 'Amount').valid).toBe(true);
    });

    it('handles negative infinity', () => {
      expect(validateNonNegative(-Infinity, 'Amount').valid).toBe(false);
    });
  });

  describe('validatePositive', () => {
    it('accepts positive numbers', () => {
      expect(validatePositive(1, 'Amount').valid).toBe(true);
      expect(validatePositive(0.001, 'Amount').valid).toBe(true);
    });

    it('rejects zero', () => {
      const result = validatePositive(0, 'Amount');
      expect(result.valid).toBe(false);
    });

    it('rejects negative', () => {
      expect(validatePositive(-1, 'Amount').valid).toBe(false);
    });

    it('handles very small positive', () => {
      expect(validatePositive(Number.MIN_VALUE, 'Amount').valid).toBe(true);
    });
  });

  describe('validateURL', () => {
    it('accepts valid HTTPS URLs', () => {
      expect(validateURL('https://example.com', 'URL').valid).toBe(true);
      expect(validateURL('https://sub.example.com/path', 'URL').valid).toBe(true);
      expect(validateURL('https://example.com/path?query=value', 'URL').valid).toBe(true);
    });

    it('accepts valid HTTP URLs', () => {
      expect(validateURL('http://localhost:3000', 'URL').valid).toBe(true);
    });

    it('rejects empty URLs', () => {
      expect(validateURL('', 'URL').valid).toBe(false);
    });

    it('rejects invalid URL format', () => {
      expect(validateURL('not-a-url', 'URL').valid).toBe(false);
      expect(validateURL('just words', 'URL').valid).toBe(false);
    });

    it('accepts technically valid URL schemes (URL spec compliance)', () => {
      // Note: These are valid URLs per URL spec, but may be security risks
      // Application-level filtering should handle dangerous schemes
      expect(validateURL('javascript:alert(1)', 'URL').valid).toBe(true); // Valid URL syntax
      expect(validateURL('data:text/html,test', 'URL').valid).toBe(true); // Valid URL syntax
      expect(validateURL('ftp://files.example.com', 'URL').valid).toBe(true);
    });

    it('handles Unicode in URLs', () => {
      // IDN domains
      expect(validateURL('https://例え.jp/path', 'URL').valid).toBe(true);
    });
  });

  describe('validateNonEmptyArray', () => {
    it('accepts arrays with elements', () => {
      expect(validateNonEmptyArray(['a'], 'Items').valid).toBe(true);
      expect(validateNonEmptyArray([1, 2, 3], 'Items').valid).toBe(true);
    });

    it('rejects empty arrays', () => {
      expect(validateNonEmptyArray([], 'Items').valid).toBe(false);
    });

    it('accepts arrays with null/undefined elements', () => {
      // Array is non-empty, even if elements are null
      expect(validateNonEmptyArray([null], 'Items').valid).toBe(true);
    });
  });

  describe('validateRange', () => {
    it('accepts values in valid range [0, 1]', () => {
      expect(validateRange(0, 'Score', { min: 0, max: 1 }).valid).toBe(true);
      expect(validateRange(0.5, 'Score', { min: 0, max: 1 }).valid).toBe(true);
      expect(validateRange(1, 'Score', { min: 0, max: 1 }).valid).toBe(true);
    });

    it('rejects values below min', () => {
      expect(validateRange(-0.1, 'Score', { min: 0, max: 1 }).valid).toBe(false);
    });

    it('rejects values above max', () => {
      expect(validateRange(1.1, 'Score', { min: 0, max: 1 }).valid).toBe(false);
    });

    it('handles boundary values precisely', () => {
      expect(validateRange(0.0000001, 'Score', { min: 0, max: 1 }).valid).toBe(true);
      expect(validateRange(0.9999999, 'Score', { min: 0, max: 1 }).valid).toBe(true);
    });

    it('supports exclusive bounds', () => {
      expect(validateRange(0, 'Score', { min: 0, max: 1, minInclusive: false }).valid).toBe(false);
      expect(validateRange(1, 'Score', { min: 0, max: 1, maxInclusive: false }).valid).toBe(false);
    });
  });

  describe('validatePublicKey', () => {
    it('accepts keys of minimum length 32', () => {
      const key32 = 'a'.repeat(32);
      expect(validatePublicKey(key32, 'Key').valid).toBe(true);
    });

    it('accepts keys longer than 32', () => {
      const key64 = 'a'.repeat(64);
      expect(validatePublicKey(key64, 'Key').valid).toBe(true);
    });

    it('rejects keys shorter than 32', () => {
      const key31 = 'a'.repeat(31);
      expect(validatePublicKey(key31, 'Key').valid).toBe(false);
    });

    it('rejects empty keys', () => {
      expect(validatePublicKey('', 'Key').valid).toBe(false);
    });
  });
});

// ============================================================================
// Combinator Functions
// ============================================================================

describe('Validation Combinators', () => {
  describe('combineResults', () => {
    it('combines multiple valid results', () => {
      const combined = combineResults(
        validResult(),
        validResult(),
        validResult()
      );
      expect(combined.valid).toBe(true);
      expect(combined.errors).toHaveLength(0);
    });

    it('combines with one invalid result', () => {
      const combined = combineResults(
        validResult(),
        invalidResult('Error 1'),
        validResult()
      );
      expect(combined.valid).toBe(false);
      expect(combined.errors).toContain('Error 1');
    });

    it('collects all errors from invalid results', () => {
      const combined = combineResults(
        invalidResult('Error 1'),
        invalidResult('Error 2'),
        invalidResult('Error 3')
      );
      expect(combined.valid).toBe(false);
      expect(combined.errors).toHaveLength(3);
      expect(combined.errors).toContain('Error 1');
      expect(combined.errors).toContain('Error 2');
      expect(combined.errors).toContain('Error 3');
    });

    it('handles empty arguments', () => {
      const combined = combineResults();
      expect(combined.valid).toBe(true);
    });
  });

  describe('assertValid', () => {
    it('does not throw for valid results', () => {
      expect(() => assertValid(validResult())).not.toThrow();
    });

    it('throws MycelixError for invalid results', () => {
      expect(() => assertValid(invalidResult('Test error'))).toThrow('Test error');
    });

    it('joins multiple errors in message', () => {
      const result = combineResults(
        invalidResult('Error 1'),
        invalidResult('Error 2')
      );
      expect(() => assertValid(result)).toThrow('Error 1; Error 2');
    });
  });
});

// ============================================================================
// Service Validators
// ============================================================================

describe('Service Validators', () => {
  describe('FinanceValidators', () => {
    // FinanceValidators.loan(amount, termMonths, interestRate)
    it('validates loan amount must be positive', () => {
      expect(() => FinanceValidators.loan(0, 12, 0.1)).toThrow();
      expect(() => FinanceValidators.loan(-100, 12, 0.1)).toThrow();
    });

    it('validates term must be positive', () => {
      expect(() => FinanceValidators.loan(1000, 0, 0.1)).toThrow();
      expect(() => FinanceValidators.loan(1000, -1, 0.1)).toThrow();
    });

    it('validates interest rate must be non-negative', () => {
      expect(() => FinanceValidators.loan(1000, 12, -0.1)).toThrow();
    });

    it('accepts valid loan parameters', () => {
      expect(() => FinanceValidators.loan(1000, 12, 0.1)).not.toThrow();
      expect(() => FinanceValidators.loan(1000, 12, 0)).not.toThrow(); // 0% interest is valid
    });

    it('validates transfer amount must be positive', () => {
      expect(() => FinanceValidators.transfer(0, 'MCX')).toThrow();
      expect(() => FinanceValidators.transfer(-100, 'MCX')).toThrow();
    });

    it('accepts valid transfer', () => {
      expect(() => FinanceValidators.transfer(100, 'MCX')).not.toThrow();
    });
  });

  describe('GovernanceValidators', () => {
    it('validates voting period must be valid number', () => {
      expect(() => GovernanceValidators.proposal(NaN, 0.5)).toThrow();
    });

    it('validates quorum must be valid number', () => {
      expect(() => GovernanceValidators.proposal(24, NaN)).toThrow();
    });

    it('validates voting period must be positive', () => {
      expect(() => GovernanceValidators.proposal(0, 0.5)).toThrow();
      expect(() => GovernanceValidators.proposal(-1, 0.5)).toThrow();
    });

    it('accepts valid proposals', () => {
      expect(() => GovernanceValidators.proposal(24, 0.5)).not.toThrow();
    });
  });

  describe('IdentityValidators', () => {
    it('validates public key minimum length', () => {
      expect(() => IdentityValidators.createIdentity('short')).toThrow();
      expect(() => IdentityValidators.createIdentity('a'.repeat(31))).toThrow();
    });

    it('accepts valid public keys', () => {
      expect(() => IdentityValidators.createIdentity('a'.repeat(32))).not.toThrow();
      expect(() => IdentityValidators.createIdentity('a'.repeat(64))).not.toThrow();
    });

    it('validates credential issuer DID', () => {
      expect(() => IdentityValidators.credential('', 'did:mycelix:subject-1234567890123456789012')).toThrow();
    });

    it('validates credential subject DID', () => {
      expect(() => IdentityValidators.credential('did:mycelix:issuer-12345678901234567890123', '')).toThrow();
    });
  });

  describe('MediaValidators', () => {
    it('validates non-empty title', () => {
      expect(() => MediaValidators.content('', 'Body content')).toThrow();
    });

    it('validates non-empty body', () => {
      expect(() => MediaValidators.content('Title', '')).toThrow();
    });

    it('accepts valid content', () => {
      expect(() => MediaValidators.content('Valid Title', 'Valid body content')).not.toThrow();
    });
  });

  describe('JusticeValidators', () => {
    it('validates complainant DID', () => {
      expect(() => JusticeValidators.case('', 'did:mycelix:valid-respondent-1234567890123')).toThrow();
    });

    it('validates respondent DID', () => {
      expect(() => JusticeValidators.case('did:mycelix:valid-complainant-12345678901', '')).toThrow();
    });

    it('validates evidence hash', () => {
      expect(() => JusticeValidators.evidence('')).toThrow();
    });

    it('validates mediator specializations array', () => {
      expect(() => JusticeValidators.mediator([])).toThrow();
    });

    it('accepts valid case', () => {
      expect(() => JusticeValidators.case(
        'did:mycelix:complainant-123456789012345678901',
        'did:mycelix:respondent-1234567890123456789012'
      )).not.toThrow();
    });
  });

  describe('KnowledgeValidators', () => {
    it('validates non-empty claim title', () => {
      expect(() => KnowledgeValidators.claim('', 'Content')).toThrow();
    });

    it('validates non-empty claim content', () => {
      expect(() => KnowledgeValidators.claim('Title', '')).toThrow();
    });

    it('validates evidence URL format', () => {
      expect(() => KnowledgeValidators.evidence('not-a-url')).toThrow();
    });

    it('validates endorser DID', () => {
      expect(() => KnowledgeValidators.endorsement('')).toThrow();
    });

    it('accepts valid knowledge claim', () => {
      expect(() => KnowledgeValidators.claim('Valid Title', 'Valid Content')).not.toThrow();
    });
  });

  describe('EnergyValidators', () => {
    it('validates reading values are non-negative', () => {
      expect(() => EnergyValidators.reading(-1, 0)).toThrow();
      expect(() => EnergyValidators.reading(0, -1)).toThrow();
    });

    it('validates trade quantity must be positive', () => {
      expect(() => EnergyValidators.trade(10, 0)).toThrow(); // Zero amount invalid
      expect(() => EnergyValidators.trade(10, -1)).toThrow(); // Negative amount invalid
    });

    it('allows zero price for free energy', () => {
      expect(() => EnergyValidators.trade(0, 100)).not.toThrow(); // Free energy is valid
    });

    it('rejects negative price', () => {
      expect(() => EnergyValidators.trade(-1, 100)).toThrow();
    });

    it('accepts valid readings', () => {
      expect(() => EnergyValidators.reading(100, 50)).not.toThrow();
      expect(() => EnergyValidators.reading(0, 0)).not.toThrow(); // Zero production/consumption is valid
    });
  });

  describe('PropertyValidators', () => {
    it('validates asset value is non-negative', () => {
      expect(() => PropertyValidators.asset(-1)).toThrow();
    });

    it('accepts zero value assets', () => {
      // Zero value is acceptable for registering valueless items
      expect(() => PropertyValidators.asset(0)).not.toThrow();
    });
  });
});

// ============================================================================
// Security Edge Cases
// ============================================================================

describe('Security Edge Cases', () => {
  describe('Injection Prevention', () => {
    it('handles SQL-like injection in strings', () => {
      // These should not cause errors but should be handled safely
      const result = validateNonEmpty("'; DROP TABLE users;--", 'Field');
      expect(result.valid).toBe(true); // String is non-empty
    });

    it('handles script injection in strings', () => {
      const result = validateNonEmpty('<script>alert("xss")</script>', 'Field');
      expect(result.valid).toBe(true); // String is non-empty
    });

    it('handles null byte injection', () => {
      const result = validateNonEmpty('valid\x00hidden', 'Field');
      expect(result.valid).toBe(true);
    });
  });

  describe('Boundary Testing', () => {
    it('handles Number.MAX_VALUE', () => {
      expect(validateNonNegative(Number.MAX_VALUE, 'Value').valid).toBe(true);
    });

    it('handles Number.MIN_SAFE_INTEGER', () => {
      expect(validateNonNegative(Number.MIN_SAFE_INTEGER, 'Value').valid).toBe(false);
    });

    it('handles NaN', () => {
      // NaN is not >= 0, so should be invalid
      expect(validateNonNegative(NaN, 'Value').valid).toBe(false);
    });
  });

  describe('Unicode and International', () => {
    it('handles RTL text', () => {
      const result = validateNonEmpty('مرحبا بالعالم', 'Field');
      expect(result.valid).toBe(true);
    });

    it('handles emoji', () => {
      const result = validateNonEmpty('🎉', 'Field');
      expect(result.valid).toBe(true);
    });

    it('handles combining characters', () => {
      // e + combining acute accent
      const result = validateNonEmpty('e\u0301', 'Field');
      expect(result.valid).toBe(true);
    });

    it('handles zero-width characters in DIDs', () => {
      // Zero-width joiner embedded - this might need special handling
      const result = validateDID('did:mycelix:abc\u200Bdef456789012345678901234567890', 'DID');
      // The validation should still check format properly
      expect(result.valid).toBe(true);
    });
  });
});
