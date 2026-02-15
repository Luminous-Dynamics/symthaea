/**
 * @mycelix/sdk SCEI Shared Validation Utilities
 *
 * Runtime validation for all SCEI modules to prevent edge cases
 * and provide clear error messages.
 *
 * @packageDocumentation
 * @module scei/validation
 */

// ============================================================================
// VALIDATION RESULT TYPES
// ============================================================================

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
}

export interface ValidationError {
  field: string;
  message: string;
  value: unknown;
  expected?: string;
}

export interface ValidationWarning {
  field: string;
  message: string;
  suggestion?: string;
}

// ============================================================================
// NUMERIC VALIDATION
// ============================================================================

/**
 * Validate a confidence value is in [0, 1] range
 */
export function validateConfidence(
  value: unknown,
  field: string = 'confidence'
): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  if (typeof value !== 'number') {
    errors.push({
      field,
      message: `${field} must be a number`,
      value,
      expected: 'number in range [0, 1]',
    });
    return { valid: false, errors, warnings };
  }

  if (!Number.isFinite(value)) {
    errors.push({
      field,
      message: `${field} must be a finite number (got ${value})`,
      value,
      expected: 'finite number in range [0, 1]',
    });
    return { valid: false, errors, warnings };
  }

  if (value < 0 || value > 1) {
    errors.push({
      field,
      message: `${field} must be between 0 and 1 (got ${value})`,
      value,
      expected: 'number in range [0, 1]',
    });
    return { valid: false, errors, warnings };
  }

  // Warnings for edge values
  if (value === 0) {
    warnings.push({
      field,
      message: `${field} is exactly 0 - this represents absolute certainty of falsehood`,
      suggestion: 'Consider using a small positive value like 0.01 for very low confidence',
    });
  }

  if (value === 1) {
    warnings.push({
      field,
      message: `${field} is exactly 1 - this represents absolute certainty`,
      suggestion: 'Consider using a value like 0.99 as absolute certainty is rare',
    });
  }

  return { valid: true, errors, warnings };
}

/**
 * Validate a degradation factor is in (0, 1) range (exclusive)
 */
export function validateDegradationFactor(
  value: unknown,
  field: string = 'degradationFactor'
): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  if (typeof value !== 'number') {
    errors.push({
      field,
      message: `${field} must be a number`,
      value,
      expected: 'number in range (0, 1)',
    });
    return { valid: false, errors, warnings };
  }

  if (!Number.isFinite(value)) {
    errors.push({
      field,
      message: `${field} must be a finite number`,
      value,
      expected: 'finite number in range (0, 1)',
    });
    return { valid: false, errors, warnings };
  }

  if (value <= 0 || value >= 1) {
    errors.push({
      field,
      message: `${field} must be strictly between 0 and 1 (got ${value})`,
      value,
      expected: 'number in range (0, 1) exclusive',
    });
    return { valid: false, errors, warnings };
  }

  // Warning for very high degradation (almost no decay)
  if (value > 0.99) {
    warnings.push({
      field,
      message: `${field} is very high (${value}) - chain will barely degrade`,
      suggestion: 'Typical values are 0.8-0.95',
    });
  }

  // Warning for very low degradation (rapid decay)
  if (value < 0.5) {
    warnings.push({
      field,
      message: `${field} is very low (${value}) - chain will degrade rapidly`,
      suggestion: 'Typical values are 0.8-0.95',
    });
  }

  return { valid: true, errors, warnings };
}

/**
 * Validate a positive integer
 */
export function validatePositiveInteger(
  value: unknown,
  field: string,
  options: { min?: number; max?: number } = {}
): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];
  const { min = 1, max = Number.MAX_SAFE_INTEGER } = options;

  if (typeof value !== 'number') {
    errors.push({
      field,
      message: `${field} must be a number`,
      value,
      expected: `integer >= ${min}`,
    });
    return { valid: false, errors, warnings };
  }

  if (!Number.isInteger(value)) {
    errors.push({
      field,
      message: `${field} must be an integer (got ${value})`,
      value,
      expected: `integer >= ${min}`,
    });
    return { valid: false, errors, warnings };
  }

  if (value < min) {
    errors.push({
      field,
      message: `${field} must be at least ${min} (got ${value})`,
      value,
      expected: `integer >= ${min}`,
    });
    return { valid: false, errors, warnings };
  }

  if (value > max) {
    errors.push({
      field,
      message: `${field} must be at most ${max} (got ${value})`,
      value,
      expected: `integer <= ${max}`,
    });
    return { valid: false, errors, warnings };
  }

  return { valid: true, errors, warnings };
}

/**
 * Validate timestamp is reasonable (not in far future, not too old)
 */
export function validateTimestamp(
  value: unknown,
  field: string = 'timestamp',
  options: { maxAgeMs?: number; allowFuture?: boolean } = {}
): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];
  const { maxAgeMs = 365 * 24 * 60 * 60 * 1000, allowFuture = false } = options; // 1 year default

  if (typeof value !== 'number') {
    errors.push({
      field,
      message: `${field} must be a number (Unix timestamp in ms)`,
      value,
      expected: 'Unix timestamp in milliseconds',
    });
    return { valid: false, errors, warnings };
  }

  const now = Date.now();

  if (!allowFuture && value > now + 60000) {
    // Allow 1 minute clock skew
    errors.push({
      field,
      message: `${field} is in the future`,
      value,
      expected: `timestamp <= ${now}`,
    });
    return { valid: false, errors, warnings };
  }

  if (now - value > maxAgeMs) {
    warnings.push({
      field,
      message: `${field} is very old (${Math.round((now - value) / 86400000)} days ago)`,
      suggestion: 'Consider whether this data is still relevant',
    });
  }

  return { valid: true, errors, warnings };
}

// ============================================================================
// STRING VALIDATION
// ============================================================================

/**
 * Validate a non-empty string
 */
export function validateNonEmptyString(
  value: unknown,
  field: string,
  options: { maxLength?: number; pattern?: RegExp } = {}
): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];
  const { maxLength = 10000, pattern } = options;

  if (typeof value !== 'string') {
    errors.push({
      field,
      message: `${field} must be a string`,
      value,
      expected: 'non-empty string',
    });
    return { valid: false, errors, warnings };
  }

  if (value.trim().length === 0) {
    errors.push({
      field,
      message: `${field} cannot be empty`,
      value,
      expected: 'non-empty string',
    });
    return { valid: false, errors, warnings };
  }

  if (value.length > maxLength) {
    errors.push({
      field,
      message: `${field} exceeds maximum length of ${maxLength}`,
      value: `${value.slice(0, 50)}...`,
      expected: `string with length <= ${maxLength}`,
    });
    return { valid: false, errors, warnings };
  }

  if (pattern && !pattern.test(value)) {
    errors.push({
      field,
      message: `${field} does not match required pattern`,
      value,
      expected: `string matching ${pattern}`,
    });
    return { valid: false, errors, warnings };
  }

  return { valid: true, errors, warnings };
}

/**
 * Validate an ID string (alphanumeric with underscores/hyphens)
 */
export function validateId(value: unknown, field: string = 'id'): ValidationResult {
  return validateNonEmptyString(value, field, {
    maxLength: 256,
    pattern: /^[a-zA-Z0-9_-]+$/,
  });
}

// ============================================================================
// ENUM VALIDATION
// ============================================================================

/**
 * Validate a value is one of the allowed enum values
 */
export function validateEnum<T extends string | number>(
  value: unknown,
  allowedValues: readonly T[],
  field: string
): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  if (!allowedValues.includes(value as T)) {
    errors.push({
      field,
      message: `${field} must be one of: ${allowedValues.join(', ')}`,
      value,
      expected: `one of [${allowedValues.join(', ')}]`,
    });
    return { valid: false, errors, warnings };
  }

  return { valid: true, errors, warnings };
}

// ============================================================================
// ARRAY VALIDATION
// ============================================================================

/**
 * Validate an array with optional element validation
 */
export function validateArray(
  value: unknown,
  field: string,
  options: {
    minLength?: number;
    maxLength?: number;
    elementValidator?: (element: unknown, index: number) => ValidationResult;
  } = {}
): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];
  const { minLength = 0, maxLength = 10000, elementValidator } = options;

  if (!Array.isArray(value)) {
    errors.push({
      field,
      message: `${field} must be an array`,
      value,
      expected: 'array',
    });
    return { valid: false, errors, warnings };
  }

  if (value.length < minLength) {
    errors.push({
      field,
      message: `${field} must have at least ${minLength} elements (has ${value.length})`,
      value: `array of length ${value.length}`,
      expected: `array with length >= ${minLength}`,
    });
    return { valid: false, errors, warnings };
  }

  if (value.length > maxLength) {
    errors.push({
      field,
      message: `${field} must have at most ${maxLength} elements (has ${value.length})`,
      value: `array of length ${value.length}`,
      expected: `array with length <= ${maxLength}`,
    });
    return { valid: false, errors, warnings };
  }

  if (elementValidator) {
    for (let i = 0; i < value.length; i++) {
      const result = elementValidator(value[i], i);
      errors.push(...result.errors.map((e) => ({ ...e, field: `${field}[${i}].${e.field}` })));
      warnings.push(...result.warnings.map((w) => ({ ...w, field: `${field}[${i}].${w.field}` })));
    }
  }

  return { valid: errors.length === 0, errors, warnings };
}

// ============================================================================
// COMPOSITE VALIDATION
// ============================================================================

/**
 * Combine multiple validation results
 */
export function combineValidations(...results: ValidationResult[]): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  for (const result of results) {
    errors.push(...result.errors);
    warnings.push(...result.warnings);
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validate and throw if invalid
 */
export function assertValid(result: ValidationResult, context: string = 'Validation'): void {
  if (!result.valid) {
    const errorMessages = result.errors.map((e) => `${e.field}: ${e.message}`).join('; ');
    throw new SCEIValidationError(`${context} failed: ${errorMessages}`, result.errors);
  }
}

// ============================================================================
// ERROR CLASS
// ============================================================================

/**
 * SCEI Validation Error
 */
export class SCEIValidationError extends Error {
  public readonly errors: ValidationError[];

  constructor(message: string, errors: ValidationError[]) {
    super(message);
    this.name = 'SCEIValidationError';
    this.errors = errors;
  }
}

// ============================================================================
// NUMERICAL STABILITY HELPERS
// ============================================================================

/**
 * Safe log that handles edge cases
 */
export function safeLog(value: number, epsilon: number = 1e-15): number {
  if (!Number.isFinite(value)) {
    return Number.NEGATIVE_INFINITY;
  }
  return Math.log(Math.max(epsilon, Math.min(1 - epsilon, value)));
}

/**
 * Clamp a value to [0, 1] with numerical stability
 */
export function clampConfidence(value: number): number {
  if (!Number.isFinite(value)) {
    return 0.5; // Default to uncertain if invalid
  }
  return Math.max(0, Math.min(1, value));
}

/**
 * Safe division that handles divide-by-zero
 */
export function safeDivide(numerator: number, denominator: number, fallback: number = 0): number {
  if (denominator === 0 || !Number.isFinite(denominator)) {
    return fallback;
  }
  const result = numerator / denominator;
  return Number.isFinite(result) ? result : fallback;
}

/**
 * Calculate weighted average with safety checks
 */
export function safeWeightedAverage(
  values: number[],
  weights: number[],
  fallback: number = 0
): number {
  if (values.length === 0 || values.length !== weights.length) {
    return fallback;
  }

  let weightedSum = 0;
  let totalWeight = 0;

  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    const w = weights[i];

    if (!Number.isFinite(v) || !Number.isFinite(w) || w < 0) {
      continue; // Skip invalid entries
    }

    weightedSum += v * w;
    totalWeight += w;
  }

  return safeDivide(weightedSum, totalWeight, fallback);
}
