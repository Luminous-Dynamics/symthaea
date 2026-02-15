/**
 * @mycelix/sdk Validation Module
 *
 * Centralized input validation for all Civilizational OS services.
 * Addresses security audit findings for consistent, robust validation.
 *
 * @packageDocumentation
 * @module utils/validation
 */

import { MycelixError, ErrorCode } from '../errors.js';

// ============================================================================
// Validation Error Types
// ============================================================================

/**
 * Validation result for composable validation
 */
export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

/**
 * Create a successful validation result
 */
export function validResult(): ValidationResult {
  return { valid: true, errors: [] };
}

/**
 * Create a failed validation result
 */
export function invalidResult(...errors: string[]): ValidationResult {
  return { valid: false, errors };
}

/**
 * Combine multiple validation results
 */
export function combineResults(...results: ValidationResult[]): ValidationResult {
  const errors = results.flatMap((r) => r.errors);
  return { valid: errors.length === 0, errors };
}

// ============================================================================
// String Validators
// ============================================================================

/**
 * Validate that a string is not empty or whitespace-only
 */
export function validateNonEmpty(value: string, fieldName: string): ValidationResult {
  if (!value || value.trim().length === 0) {
    return invalidResult(`${fieldName} cannot be empty or whitespace-only`);
  }
  return validResult();
}

/**
 * Validate string length constraints
 */
export function validateLength(
  value: string,
  fieldName: string,
  options: { min?: number; max?: number }
): ValidationResult {
  const { min, max } = options;
  const length = value?.length ?? 0;

  if (min !== undefined && length < min) {
    return invalidResult(`${fieldName} must be at least ${min} characters`);
  }
  if (max !== undefined && length > max) {
    return invalidResult(`${fieldName} must be at most ${max} characters`);
  }
  return validResult();
}

/**
 * Validate a DID format
 */
export function validateDID(did: string, fieldName: string = 'DID'): ValidationResult {
  if (!did || did.trim().length === 0) {
    return invalidResult(`${fieldName} cannot be empty`);
  }
  if (!did.startsWith('did:')) {
    return invalidResult(`${fieldName} must be a valid DID format (did:method:identifier)`);
  }
  return validResult();
}

/**
 * Validate URL format
 */
export function validateURL(url: string, fieldName: string = 'URL'): ValidationResult {
  if (!url || url.trim().length === 0) {
    return invalidResult(`${fieldName} cannot be empty`);
  }
  try {
    new URL(url);
    return validResult();
  } catch {
    return invalidResult(`${fieldName} must be a valid URL`);
  }
}

/**
 * Validate hash format (non-empty hex string)
 */
export function validateHash(hash: string, fieldName: string = 'Hash'): ValidationResult {
  if (!hash || hash.trim().length === 0) {
    return invalidResult(`${fieldName} cannot be empty`);
  }
  if (!/^[a-fA-F0-9]+$/.test(hash)) {
    return invalidResult(`${fieldName} must be a valid hexadecimal hash`);
  }
  return validResult();
}

// ============================================================================
// Numeric Validators
// ============================================================================

/**
 * Validate that a number is positive (> 0)
 */
export function validatePositive(value: number, fieldName: string): ValidationResult {
  if (typeof value !== 'number' || isNaN(value)) {
    return invalidResult(`${fieldName} must be a valid number`);
  }
  if (value <= 0) {
    return invalidResult(`${fieldName} must be positive (got ${value})`);
  }
  return validResult();
}

/**
 * Validate that a number is non-negative (>= 0)
 */
export function validateNonNegative(value: number, fieldName: string): ValidationResult {
  if (typeof value !== 'number' || isNaN(value)) {
    return invalidResult(`${fieldName} must be a valid number`);
  }
  if (value < 0) {
    return invalidResult(`${fieldName} cannot be negative (got ${value})`);
  }
  return validResult();
}

/**
 * Validate number is within a range
 */
export function validateRange(
  value: number,
  fieldName: string,
  options: { min?: number; max?: number; minInclusive?: boolean; maxInclusive?: boolean }
): ValidationResult {
  const { min, max, minInclusive = true, maxInclusive = true } = options;

  if (typeof value !== 'number' || isNaN(value)) {
    return invalidResult(`${fieldName} must be a valid number`);
  }

  if (min !== undefined) {
    if (minInclusive && value < min) {
      return invalidResult(`${fieldName} must be >= ${min} (got ${value})`);
    }
    if (!minInclusive && value <= min) {
      return invalidResult(`${fieldName} must be > ${min} (got ${value})`);
    }
  }

  if (max !== undefined) {
    if (maxInclusive && value > max) {
      return invalidResult(`${fieldName} must be <= ${max} (got ${value})`);
    }
    if (!maxInclusive && value >= max) {
      return invalidResult(`${fieldName} must be < ${max} (got ${value})`);
    }
  }

  return validResult();
}

/**
 * Validate percentage (0-1 or 0-100 depending on format)
 */
export function validatePercentage(
  value: number,
  fieldName: string,
  format: 'decimal' | 'percent' = 'decimal'
): ValidationResult {
  const max = format === 'decimal' ? 1 : 100;
  return validateRange(value, fieldName, { min: 0, max });
}

// ============================================================================
// Array/Collection Validators
// ============================================================================

/**
 * Validate array is not empty
 */
export function validateNonEmptyArray<T>(arr: T[], fieldName: string): ValidationResult {
  if (!Array.isArray(arr) || arr.length === 0) {
    return invalidResult(`${fieldName} must contain at least one item`);
  }
  return validResult();
}

/**
 * Validate array length constraints
 */
export function validateArrayLength<T>(
  arr: T[],
  fieldName: string,
  options: { min?: number; max?: number }
): ValidationResult {
  const { min, max } = options;
  const length = arr?.length ?? 0;

  if (min !== undefined && length < min) {
    return invalidResult(`${fieldName} must contain at least ${min} items`);
  }
  if (max !== undefined && length > max) {
    return invalidResult(`${fieldName} must contain at most ${max} items`);
  }
  return validResult();
}

// ============================================================================
// Composite Validators (Domain-Specific)
// ============================================================================

/**
 * Validate financial amount (positive number)
 */
export function validateAmount(amount: number, fieldName: string = 'Amount'): ValidationResult {
  return validatePositive(amount, fieldName);
}

/**
 * Validate interest rate (non-negative)
 */
export function validateInterestRate(rate: number): ValidationResult {
  return validateNonNegative(rate, 'Interest rate');
}

/**
 * Validate loan term (positive integer representing months)
 */
export function validateLoanTerm(termMonths: number): ValidationResult {
  const result = validatePositive(termMonths, 'Loan term');
  if (!result.valid) return result;

  if (!Number.isInteger(termMonths)) {
    return invalidResult('Loan term must be a whole number of months');
  }
  return validResult();
}

/**
 * Validate voting period (positive hours)
 */
export function validateVotingPeriod(hours: number): ValidationResult {
  const result = validatePositive(hours, 'Voting period');
  if (!result.valid) return result;

  // Minimum 1 hour, maximum 90 days
  return validateRange(hours, 'Voting period', { min: 1, max: 2160 });
}

/**
 * Validate quorum percentage (0-1)
 */
export function validateQuorum(quorum: number): ValidationResult {
  return validatePercentage(quorum, 'Quorum percentage', 'decimal');
}

/**
 * Validate public key for identity creation
 */
export function validatePublicKey(publicKey: string): ValidationResult {
  const nonEmpty = validateNonEmpty(publicKey, 'Public key');
  if (!nonEmpty.valid) return nonEmpty;

  // Minimum length for cryptographic keys
  return validateLength(publicKey, 'Public key', { min: 32 });
}

/**
 * Validate energy price (non-negative, can be zero for free energy)
 */
export function validateEnergyPrice(price: number): ValidationResult {
  return validateNonNegative(price, 'Energy price');
}

/**
 * Validate energy production/consumption values
 */
export function validateEnergyValue(
  value: number,
  fieldName: string = 'Energy value'
): ValidationResult {
  return validateNonNegative(value, fieldName);
}

// ============================================================================
// Validation Assertion (Throws on Failure)
// ============================================================================

/**
 * Assert validation passes, throw MycelixError if not
 */
export function assertValid(result: ValidationResult): void {
  if (!result.valid) {
    throw new MycelixError(result.errors.join('; '), ErrorCode.INVALID_ARGUMENT);
  }
}

/**
 * Run validation and throw if invalid
 */
export function validate(
  value: unknown,
  ...validators: ((v: unknown) => ValidationResult)[]
): void {
  for (const validator of validators) {
    const result = validator(value);
    assertValid(result);
  }
}

// ============================================================================
// Service-Specific Validators
// ============================================================================

/**
 * Finance service validators
 */
export const FinanceValidators = {
  transfer(amount: number, _currency: string): void {
    assertValid(validateAmount(amount, 'Transfer amount'));
  },

  loan(amount: number, termMonths: number, interestRate: number): void {
    assertValid(
      combineResults(
        validateAmount(amount, 'Loan amount'),
        validateLoanTerm(termMonths),
        validateInterestRate(interestRate)
      )
    );
  },
};

/**
 * Governance service validators
 */
export const GovernanceValidators = {
  proposal(votingPeriodHours: number, quorumPercentage: number): void {
    assertValid(
      combineResults(validateVotingPeriod(votingPeriodHours), validateQuorum(quorumPercentage))
    );
  },
};

/**
 * Identity service validators
 */
export const IdentityValidators = {
  createIdentity(publicKey: string): void {
    assertValid(validatePublicKey(publicKey));
  },

  credential(issuerId: string, subjectId: string): void {
    assertValid(
      combineResults(validateDID(issuerId, 'Issuer DID'), validateDID(subjectId, 'Subject DID'))
    );
  },
};

/**
 * Energy service validators
 */
export const EnergyValidators = {
  reading(production: number, consumption: number): void {
    assertValid(
      combineResults(
        validateEnergyValue(production, 'Production'),
        validateEnergyValue(consumption, 'Consumption')
      )
    );
  },

  trade(price: number, quantity: number): void {
    assertValid(
      combineResults(validateEnergyPrice(price), validatePositive(quantity, 'Trade quantity'))
    );
  },
};

/**
 * Justice service validators
 */
export const JusticeValidators = {
  case(complainantId: string, respondentId: string): void {
    assertValid(
      combineResults(
        validateDID(complainantId, 'Complainant DID'),
        validateDID(respondentId, 'Respondent DID')
      )
    );
  },

  evidence(contentHash: string): void {
    assertValid(validateNonEmpty(contentHash, 'Evidence content hash'));
  },

  mediator(specializations: string[]): void {
    assertValid(validateNonEmptyArray(specializations, 'Mediator specializations'));
  },
};

/**
 * Knowledge service validators
 */
export const KnowledgeValidators = {
  claim(title: string, content: string): void {
    assertValid(
      combineResults(
        validateNonEmpty(title, 'Claim title'),
        validateNonEmpty(content, 'Claim content')
      )
    );
  },

  evidence(sourceUrl: string): void {
    assertValid(validateURL(sourceUrl, 'Evidence source URL'));
  },

  endorsement(endorserId: string): void {
    assertValid(validateDID(endorserId, 'Endorser DID'));
  },
};

/**
 * Media service validators
 */
export const MediaValidators = {
  content(title: string, body: string): void {
    assertValid(
      combineResults(
        validateNonEmpty(title, 'Content title'),
        validateNonEmpty(body, 'Content body')
      )
    );
  },
};

/**
 * Property service validators
 */
export const PropertyValidators = {
  asset(value: number): void {
    // Allow zero value for valueless items being registered
    assertValid(validateNonNegative(value, 'Asset value'));
  },
};

// ============================================================================
// Holochain-Specific Validators (Security Hardening)
// ============================================================================

/**
 * Validate Holochain agent public key (base64 encoded, 39 bytes)
 */
export function validateAgentPubKey(
  key: string | Uint8Array,
  fieldName: string = 'Agent public key'
): ValidationResult {
  if (key instanceof Uint8Array) {
    if (key.length !== 39) {
      return invalidResult(`${fieldName} must be 39 bytes (got ${key.length})`);
    }
    return validResult();
  }

  if (typeof key !== 'string') {
    return invalidResult(`${fieldName} must be a string or Uint8Array`);
  }

  // Base64 encoded 39 bytes = 52 chars
  if (key.length < 50 || key.length > 60) {
    return invalidResult(`${fieldName} has invalid length for base64-encoded agent key`);
  }

  // Check valid base64
  if (!/^[A-Za-z0-9+/=_-]+$/.test(key)) {
    return invalidResult(`${fieldName} contains invalid base64 characters`);
  }

  return validResult();
}

/**
 * Validate Holochain action hash (base64 encoded, 39 bytes)
 */
export function validateActionHash(
  hash: string | Uint8Array,
  fieldName: string = 'Action hash'
): ValidationResult {
  return validateAgentPubKey(hash, fieldName); // Same format
}

/**
 * Validate Holochain entry hash (base64 encoded, 39 bytes)
 */
export function validateEntryHash(
  hash: string | Uint8Array,
  fieldName: string = 'Entry hash'
): ValidationResult {
  return validateAgentPubKey(hash, fieldName); // Same format
}

/**
 * Validate hApp ID format
 */
export function validateHappId(happId: string, fieldName: string = 'hApp ID'): ValidationResult {
  const nonEmpty = validateNonEmpty(happId, fieldName);
  if (!nonEmpty.valid) return nonEmpty;

  // hApp IDs should be alphanumeric with hyphens/underscores, 3-64 chars
  if (!/^[a-zA-Z][a-zA-Z0-9_-]{2,63}$/.test(happId)) {
    return invalidResult(
      `${fieldName} must start with a letter and contain only alphanumeric, hyphens, underscores (3-64 chars)`
    );
  }

  return validResult();
}

/**
 * Validate zome name format
 */
export function validateZomeName(zomeName: string, fieldName: string = 'Zome name'): ValidationResult {
  const nonEmpty = validateNonEmpty(zomeName, fieldName);
  if (!nonEmpty.valid) return nonEmpty;

  // Zome names follow Rust identifier rules
  if (!/^[a-z][a-z0-9_]{0,63}$/.test(zomeName)) {
    return invalidResult(
      `${fieldName} must be a valid Rust identifier (lowercase, start with letter, max 64 chars)`
    );
  }

  return validResult();
}

// ============================================================================
// MATL Trust Score Validators
// ============================================================================

/**
 * Validate trust score (0.0 - 1.0)
 */
export function validateTrustScore(
  score: number,
  fieldName: string = 'Trust score'
): ValidationResult {
  return validateRange(score, fieldName, { min: 0, max: 1 });
}

/**
 * Validate Byzantine threshold (0.0 - 0.5)
 */
export function validateByzantineThreshold(
  threshold: number,
  fieldName: string = 'Byzantine threshold'
): ValidationResult {
  return validateRange(threshold, fieldName, { min: 0, max: 0.5 });
}

/**
 * Validate MATL composite score parameters
 */
export function validateMATLWeights(weights: {
  quality: number;
  consistency: number;
  reputation: number;
}): ValidationResult {
  const results = [
    validatePercentage(weights.quality, 'Quality weight'),
    validatePercentage(weights.consistency, 'Consistency weight'),
    validatePercentage(weights.reputation, 'Reputation weight'),
  ];

  const combined = combineResults(...results);
  if (!combined.valid) return combined;

  // Weights should sum to 1.0 (with small tolerance for floating point)
  const sum = weights.quality + weights.consistency + weights.reputation;
  if (Math.abs(sum - 1.0) > 0.001) {
    return invalidResult(`MATL weights must sum to 1.0 (got ${sum.toFixed(4)})`);
  }

  return validResult();
}

// ============================================================================
// Timestamp Validators
// ============================================================================

/**
 * Validate timestamp is in the past
 */
export function validatePastTimestamp(
  timestamp: number,
  fieldName: string = 'Timestamp',
  toleranceMs: number = 60000 // 1 minute tolerance for clock drift
): ValidationResult {
  if (typeof timestamp !== 'number' || isNaN(timestamp)) {
    return invalidResult(`${fieldName} must be a valid number`);
  }

  const now = Date.now();
  if (timestamp > now + toleranceMs) {
    return invalidResult(`${fieldName} cannot be in the future`);
  }

  return validResult();
}

/**
 * Validate timestamp is in the future
 */
export function validateFutureTimestamp(
  timestamp: number,
  fieldName: string = 'Timestamp',
  toleranceMs: number = 60000
): ValidationResult {
  if (typeof timestamp !== 'number' || isNaN(timestamp)) {
    return invalidResult(`${fieldName} must be a valid number`);
  }

  const now = Date.now();
  if (timestamp < now - toleranceMs) {
    return invalidResult(`${fieldName} must be in the future`);
  }

  return validResult();
}

/**
 * Validate timestamp is within a reasonable range (not too old)
 */
export function validateReasonableTimestamp(
  timestamp: number,
  fieldName: string = 'Timestamp',
  maxAgeDays: number = 365 // 1 year default
): ValidationResult {
  if (typeof timestamp !== 'number' || isNaN(timestamp)) {
    return invalidResult(`${fieldName} must be a valid number`);
  }

  const now = Date.now();
  const maxAgeMs = maxAgeDays * 24 * 60 * 60 * 1000;

  if (timestamp < now - maxAgeMs) {
    return invalidResult(`${fieldName} is too old (max ${maxAgeDays} days)`);
  }

  if (timestamp > now + 60000) {
    return invalidResult(`${fieldName} cannot be in the future`);
  }

  return validResult();
}

// ============================================================================
// Security Hardening - Input Sanitization
// ============================================================================

/** Characters that should never appear in user inputs (null bytes, control chars) */
// eslint-disable-next-line no-control-regex -- Intentional: this regex matches control chars for security sanitization
const DANGEROUS_CHARS = /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g;

/** Potential XSS patterns */
const XSS_PATTERNS = [
  /<script[\s\S]*?>/gi,
  /javascript:/gi,
  /on\w+\s*=/gi,
  /data:text\/html/gi,
  /<iframe[\s\S]*?>/gi,
  /<object[\s\S]*?>/gi,
  /<embed[\s\S]*?>/gi,
];

/** SQL injection patterns (basic) */
const SQL_PATTERNS = [
  /'\s*or\s+'?\d+'?\s*=\s*'?\d+'?/gi,
  /;\s*drop\s+table/gi,
  /;\s*delete\s+from/gi,
  /union\s+select/gi,
  /--\s*$/gm,
];

/**
 * Check for dangerous characters (null bytes, control characters)
 */
export function validateNoDangerousChars(
  value: string,
  fieldName: string = 'Input'
): ValidationResult {
  if (DANGEROUS_CHARS.test(value)) {
    return invalidResult(`${fieldName} contains invalid control characters`);
  }
  return validResult();
}

/**
 * Check for potential XSS patterns
 */
export function validateNoXSS(value: string, fieldName: string = 'Input'): ValidationResult {
  for (const pattern of XSS_PATTERNS) {
    if (pattern.test(value)) {
      return invalidResult(`${fieldName} contains potentially dangerous content`);
    }
  }
  return validResult();
}

/**
 * Check for SQL injection patterns (defense in depth)
 */
export function validateNoSQLInjection(
  value: string,
  fieldName: string = 'Input'
): ValidationResult {
  for (const pattern of SQL_PATTERNS) {
    if (pattern.test(value)) {
      return invalidResult(`${fieldName} contains suspicious patterns`);
    }
  }
  return validResult();
}

/**
 * Comprehensive security validation for user inputs
 */
export function validateSecureInput(
  value: string,
  fieldName: string = 'Input',
  options: {
    maxLength?: number;
    allowHtml?: boolean;
    strict?: boolean;
  } = {}
): ValidationResult {
  const { maxLength = 10000, allowHtml = false, strict = true } = options;

  const results: ValidationResult[] = [
    validateNonEmpty(value, fieldName),
    validateLength(value, fieldName, { max: maxLength }),
    validateNoDangerousChars(value, fieldName),
  ];

  if (!allowHtml) {
    results.push(validateNoXSS(value, fieldName));
  }

  if (strict) {
    results.push(validateNoSQLInjection(value, fieldName));
  }

  return combineResults(...results);
}

/**
 * Sanitize string by removing dangerous characters
 */
export function sanitizeString(value: string): string {
  return value.replace(DANGEROUS_CHARS, '').trim();
}

/**
 * Escape HTML entities for safe display
 */
export function escapeHtml(value: string): string {
  const htmlEntities: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#x27;',
    '/': '&#x2F;',
  };
  return value.replace(/[&<>"'/]/g, (char) => htmlEntities[char] || char);
}

// ============================================================================
// Bridge Message Validators
// ============================================================================

/**
 * Validate bridge message structure
 */
export function validateBridgeMessage(message: {
  type: string;
  sourceHapp: string;
  targetHapp: string;
  payload?: unknown;
}): ValidationResult {
  return combineResults(
    validateNonEmpty(message.type, 'Message type'),
    validateHappId(message.sourceHapp, 'Source hApp'),
    validateHappId(message.targetHapp, 'Target hApp')
  );
}

// ============================================================================
// Extended Service Validators
// ============================================================================

/**
 * Bridge service validators
 */
export const BridgeValidators = {
  register(happId: string, agentKey: string | Uint8Array): void {
    assertValid(
      combineResults(validateHappId(happId, 'hApp ID'), validateAgentPubKey(agentKey, 'Agent key'))
    );
  },

  message(sourceHapp: string, targetHapp: string, messageType: string): void {
    assertValid(
      combineResults(
        validateHappId(sourceHapp, 'Source hApp'),
        validateHappId(targetHapp, 'Target hApp'),
        validateNonEmpty(messageType, 'Message type')
      )
    );
  },

  reputation(agentKey: string | Uint8Array, score: number): void {
    assertValid(
      combineResults(validateAgentPubKey(agentKey, 'Agent key'), validateTrustScore(score))
    );
  },
};

/**
 * MATL validators
 */
export const MATLValidators = {
  pogq(quality: number, consistency: number, entropy: number): void {
    assertValid(
      combineResults(
        validatePercentage(quality, 'Quality'),
        validatePercentage(consistency, 'Consistency'),
        validatePercentage(entropy, 'Entropy')
      )
    );
  },

  reputation(agentId: string, score: number): void {
    assertValid(
      combineResults(validateNonEmpty(agentId, 'Agent ID'), validateTrustScore(score))
    );
  },

  threshold(value: number): void {
    assertValid(validateByzantineThreshold(value));
  },
};

/**
 * Epistemic validators
 */
export const EpistemicValidators = {
  claim(
    subject: string,
    empirical: number,
    normative: number,
    materiality: number
  ): void {
    assertValid(
      combineResults(
        validateNonEmpty(subject, 'Claim subject'),
        validateRange(empirical, 'Empirical level', { min: 0, max: 4 }),
        validateRange(normative, 'Normative level', { min: 0, max: 3 }),
        validateRange(materiality, 'Materiality level', { min: 0, max: 3 })
      )
    );
  },

  evidence(claimHash: string | Uint8Array, sourceUrl: string): void {
    assertValid(
      combineResults(
        validateEntryHash(claimHash, 'Claim hash'),
        validateURL(sourceUrl, 'Source URL')
      )
    );
  },
};

/**
 * Federated Learning validators
 */
export const FLValidators = {
  gradient(values: number[], agentKey: string | Uint8Array): void {
    assertValid(
      combineResults(
        validateNonEmptyArray(values, 'Gradient values'),
        validateAgentPubKey(agentKey, 'Participant key')
      )
    );

    // Validate gradient values are finite numbers
    for (let i = 0; i < values.length; i++) {
      if (!Number.isFinite(values[i])) {
        assertValid(invalidResult(`Gradient value at index ${i} is not a finite number`));
      }
    }
  },

  round(roundId: number, minParticipants: number): void {
    assertValid(
      combineResults(
        validateNonNegative(roundId, 'Round ID'),
        validatePositive(minParticipants, 'Minimum participants')
      )
    );
  },
};
