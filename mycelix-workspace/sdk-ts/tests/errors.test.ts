/**
 * Tests for Mycelix error handling and validation utilities
 */

import { describe, it, expect } from 'vitest';
import {
  MycelixError,
  ValidationError,
  MATLError,
  EpistemicError,
  FederatedLearningError,
  BridgeError,
  ConnectionError,
  ErrorCode,
  validate,
  withErrorHandling,
  withRetry,
  withRecovery,
  withRecoveryOrThrow,
  exponentialBackoffStrategy,
  fallbackChainStrategy,
  circuitBreakerStrategy,
  gracePeriodStrategy,
  combineStrategies,
  assert,
  assertDefined,
} from '../src/errors';

describe('MycelixError', () => {
  it('should create error with code and context', () => {
    const error = new MycelixError(
      'Test error',
      ErrorCode.INVALID_ARGUMENT,
      { field: 'test' }
    );

    expect(error.message).toBe('Test error');
    expect(error.code).toBe(ErrorCode.INVALID_ARGUMENT);
    expect(error.context.field).toBe('test');
    expect(error.timestamp).toBeInstanceOf(Date);
  });

  it('should provide recovery suggestions', () => {
    const error = new MycelixError('Not found', ErrorCode.NOT_FOUND);

    expect(error.recoverySuggestions.length).toBeGreaterThan(0);
    expect(error.recoverySuggestions[0].action).toBe('create');
  });

  it('should serialize to JSON', () => {
    const error = new MycelixError('Test', ErrorCode.TIMEOUT);
    const json = error.toJSON();

    expect(json).toHaveProperty('name', 'MycelixError');
    expect(json).toHaveProperty('code', ErrorCode.TIMEOUT);
    expect(json).toHaveProperty('codeString', 'TIMEOUT');
    expect(json).toHaveProperty('timestamp');
  });

  it('should format as string with context and suggestions', () => {
    const error = new MycelixError(
      'Invalid input',
      ErrorCode.INVALID_ARGUMENT,
      { value: 42 }
    );

    const str = error.toString();
    expect(str).toContain('INVALID_ARGUMENT');
    expect(str).toContain('Invalid input');
    expect(str).toContain('Context');
    expect(str).toContain('Suggestions');
  });

  it('should preserve cause', () => {
    const cause = new Error('Original error');
    const error = new MycelixError('Wrapped', ErrorCode.UNKNOWN, {}, cause);

    expect(error.cause).toBe(cause);
  });
});

describe('ValidationError', () => {
  it('should capture field and constraint', () => {
    const error = new ValidationError('username', 'must not be empty', '');

    expect(error.field).toBe('username');
    expect(error.constraint).toBe('must not be empty');
    expect(error.code).toBe(ErrorCode.INVALID_ARGUMENT);
    expect(error.context.value).toBe('');
  });
});

describe('Specialized Errors', () => {
  it('should create MATLError', () => {
    const error = new MATLError(
      'Invalid score',
      ErrorCode.MATL_INVALID_SCORE,
      { score: 1.5 }
    );

    expect(error.name).toBe('MATLError');
    expect(error.code).toBe(ErrorCode.MATL_INVALID_SCORE);
  });

  it('should create EpistemicError', () => {
    const error = new EpistemicError(
      'Claim expired',
      ErrorCode.EPISTEMIC_CLAIM_EXPIRED
    );

    expect(error.name).toBe('EpistemicError');
  });

  it('should create FederatedLearningError', () => {
    const error = new FederatedLearningError(
      'Not enough participants',
      ErrorCode.FL_NOT_ENOUGH_PARTICIPANTS,
      { current: 2, required: 3 }
    );

    expect(error.name).toBe('FederatedLearningError');
    expect(error.context.current).toBe(2);
  });

  it('should create BridgeError', () => {
    const error = new BridgeError(
      'hApp not found',
      ErrorCode.BRIDGE_HAPP_NOT_FOUND
    );

    expect(error.name).toBe('BridgeError');
  });

  it('should create ConnectionError with retryable flag', () => {
    const error = new ConnectionError(
      'Connection failed',
      ErrorCode.CONNECTION_FAILED,
      'ws://localhost:8080',
      true
    );

    expect(error.name).toBe('ConnectionError');
    expect(error.endpoint).toBe('ws://localhost:8080');
    expect(error.retryable).toBe(true);
  });
});

describe('Validator', () => {
  describe('required', () => {
    it('should pass for defined values', () => {
      const result = validate()
        .required('field', 'value')
        .result();

      expect(result.valid).toBe(true);
    });

    it('should fail for null/undefined', () => {
      const result = validate()
        .required('field', null)
        .required('field2', undefined)
        .result();

      expect(result.valid).toBe(false);
      expect(result.errors.length).toBe(2);
    });
  });

  describe('inRange', () => {
    it('should pass for values in range', () => {
      const result = validate()
        .inRange('score', 0.5, 0, 1)
        .result();

      expect(result.valid).toBe(true);
    });

    it('should fail for values out of range', () => {
      const result = validate()
        .inRange('score', 1.5, 0, 1)
        .inRange('negative', -0.5, 0, 1)
        .result();

      expect(result.valid).toBe(false);
      expect(result.errors.length).toBe(2);
    });

    it('should fail for non-numbers', () => {
      const result = validate()
        .inRange('score', NaN, 0, 1)
        .result();

      expect(result.valid).toBe(false);
    });
  });

  describe('positive', () => {
    it('should pass for positive numbers', () => {
      const result = validate()
        .positive('count', 5)
        .result();

      expect(result.valid).toBe(true);
    });

    it('should fail for zero and negative', () => {
      const result = validate()
        .positive('zero', 0)
        .positive('neg', -1)
        .result();

      expect(result.valid).toBe(false);
      expect(result.errors.length).toBe(2);
    });
  });

  describe('nonNegative', () => {
    it('should pass for zero and positive', () => {
      const result = validate()
        .nonNegative('zero', 0)
        .nonNegative('pos', 5)
        .result();

      expect(result.valid).toBe(true);
    });

    it('should fail for negative', () => {
      const result = validate()
        .nonNegative('neg', -1)
        .result();

      expect(result.valid).toBe(false);
    });
  });

  describe('notEmpty', () => {
    it('should pass for non-empty strings', () => {
      const result = validate()
        .notEmpty('name', 'Alice')
        .result();

      expect(result.valid).toBe(true);
    });

    it('should fail for empty strings', () => {
      const result = validate()
        .notEmpty('name', '')
        .notEmpty('spaces', '   ')
        .result();

      expect(result.valid).toBe(false);
      expect(result.errors.length).toBe(2);
    });
  });

  describe('pattern', () => {
    it('should pass for matching patterns', () => {
      const result = validate()
        .pattern('email', 'test@example.com', /^.+@.+\..+$/)
        .result();

      expect(result.valid).toBe(true);
    });

    it('should fail for non-matching patterns', () => {
      const result = validate()
        .pattern('email', 'invalid', /^.+@.+\..+$/, 'must be a valid email')
        .result();

      expect(result.valid).toBe(false);
      expect(result.errors[0].constraint).toBe('must be a valid email');
    });
  });

  describe('minLength', () => {
    it('should pass for arrays with enough items', () => {
      const result = validate()
        .minLength('items', [1, 2, 3], 2)
        .result();

      expect(result.valid).toBe(true);
    });

    it('should fail for arrays with too few items', () => {
      const result = validate()
        .minLength('items', [1], 3)
        .result();

      expect(result.valid).toBe(false);
    });
  });

  describe('custom', () => {
    it('should pass for valid custom condition', () => {
      const result = validate()
        .custom('age', 25, (v) => typeof v === 'number' && v >= 18, 'must be adult')
        .result();

      expect(result.valid).toBe(true);
    });

    it('should fail for invalid custom condition', () => {
      const result = validate()
        .custom('age', 15, (v) => typeof v === 'number' && v >= 18, 'must be adult')
        .result();

      expect(result.valid).toBe(false);
    });
  });

  describe('throwIfInvalid', () => {
    it('should not throw for valid input', () => {
      expect(() => {
        validate()
          .required('field', 'value')
          .throwIfInvalid();
      }).not.toThrow();
    });

    it('should throw MycelixError for invalid input', () => {
      expect(() => {
        validate()
          .required('field', null)
          .throwIfInvalid();
      }).toThrow(MycelixError);
    });
  });

  describe('reset', () => {
    it('should clear previous errors', () => {
      const validator = validate()
        .required('field', null);

      expect(validator.result().valid).toBe(false);

      validator.reset();
      expect(validator.result().valid).toBe(true);
    });
  });
});

describe('withErrorHandling', () => {
  it('should wrap function and preserve behavior', () => {
    const fn = (x: number) => x * 2;
    const wrapped = withErrorHandling(fn, 'test function');

    expect(wrapped(5)).toBe(10);
  });

  it('should convert thrown errors to MycelixError', () => {
    const fn = () => {
      throw new Error('Original error');
    };
    const wrapped = withErrorHandling(fn, 'failing function');

    expect(() => wrapped()).toThrow(MycelixError);
  });

  it('should preserve MycelixError as-is', () => {
    const original = new MycelixError('Test', ErrorCode.INVALID_ARGUMENT);
    const fn = () => {
      throw original;
    };
    const wrapped = withErrorHandling(fn, 'context');

    try {
      wrapped();
    } catch (error) {
      expect(error).toBe(original);
    }
  });

  it('should handle async functions', async () => {
    const fn = async () => {
      throw new Error('Async error');
    };
    const wrapped = withErrorHandling(fn, 'async function');

    await expect(wrapped()).rejects.toThrow(MycelixError);
  });
});

describe('withRetry', () => {
  it('should return result on success', async () => {
    let attempts = 0;
    const fn = async () => {
      attempts++;
      return 'success';
    };

    const result = await withRetry(fn);
    expect(result).toBe('success');
    expect(attempts).toBe(1);
  });

  it('should retry on failure', async () => {
    let attempts = 0;
    const fn = async () => {
      attempts++;
      if (attempts < 3) {
        throw new Error('Temporary failure');
      }
      return 'success';
    };

    const result = await withRetry(fn, { maxRetries: 3, initialDelay: 10 });
    expect(result).toBe('success');
    expect(attempts).toBe(3);
  });

  it('should throw after max retries', async () => {
    let attempts = 0;
    const fn = async () => {
      attempts++;
      throw new Error('Permanent failure');
    };

    await expect(
      withRetry(fn, { maxRetries: 2, initialDelay: 10 })
    ).rejects.toThrow();
    expect(attempts).toBe(3); // Initial + 2 retries
  });

  it('should respect retryOn predicate', async () => {
    let attempts = 0;
    const fn = async () => {
      attempts++;
      throw new MycelixError('Not retryable', ErrorCode.INVALID_ARGUMENT);
    };

    await expect(
      withRetry(fn, {
        maxRetries: 3,
        initialDelay: 10,
        retryOn: (error) => error instanceof MycelixError && error.code === ErrorCode.TIMEOUT,
      })
    ).rejects.toThrow();
    expect(attempts).toBe(1); // No retries because predicate returned false
  });
});

describe('assert', () => {
  it('should pass for true condition', () => {
    expect(() => assert(true, 'field', 'must be true')).not.toThrow();
  });

  it('should throw ValidationError for false condition', () => {
    expect(() => assert(false, 'field', 'must be true')).toThrow(ValidationError);
  });
});

describe('assertDefined', () => {
  it('should pass for defined values', () => {
    expect(() => assertDefined('value', 'field')).not.toThrow();
    expect(() => assertDefined(0, 'field')).not.toThrow();
    expect(() => assertDefined(false, 'field')).not.toThrow();
  });

  it('should throw for null/undefined', () => {
    expect(() => assertDefined(null, 'field')).toThrow(ValidationError);
    expect(() => assertDefined(undefined, 'field')).toThrow(ValidationError);
  });
});

describe('Recovery Strategies', () => {
  describe('exponentialBackoffStrategy', () => {
    it('should retry on retryable errors', async () => {
      let attempts = 0;
      const strategy = exponentialBackoffStrategy<string>({
        maxRetries: 3,
        initialDelayMs: 10,
        retryableErrors: [ErrorCode.TIMEOUT],
      });

      const operation = async () => {
        attempts++;
        if (attempts < 3) {
          throw new MycelixError('Timeout', ErrorCode.TIMEOUT);
        }
        return 'success';
      };

      const result = await withRecovery(operation, [strategy], { maxDuration: 5000 });

      expect(result.success).toBe(true);
      expect(result.value).toBe('success');
      expect(result.strategyUsed).toBe('exponentialBackoff');
    });

    it('should not retry on non-retryable errors', async () => {
      let attempts = 0;
      const strategy = exponentialBackoffStrategy<string>({
        maxRetries: 3,
        retryableErrors: [ErrorCode.TIMEOUT],
      });

      const operation = async () => {
        attempts++;
        throw new MycelixError('Invalid', ErrorCode.INVALID_ARGUMENT);
      };

      const result = await withRecovery(operation, [strategy]);

      expect(result.success).toBe(false);
      expect(attempts).toBe(1);
    });
  });

  describe('fallbackChainStrategy', () => {
    it('should try fallbacks in order', async () => {
      const calls: string[] = [];

      const strategy = fallbackChainStrategy<string>([
        async () => {
          calls.push('fallback1');
          throw new Error('Fallback 1 failed');
        },
        async () => {
          calls.push('fallback2');
          return 'fallback2-result';
        },
      ]);

      const operation = async () => {
        calls.push('primary');
        throw new Error('Primary failed');
      };

      const result = await withRecovery(operation, [strategy]);

      expect(result.success).toBe(true);
      expect(result.value).toBe('fallback2-result');
      expect(calls).toContain('primary');
      expect(calls).toContain('fallback1');
      expect(calls).toContain('fallback2');
    });
  });

  describe('circuitBreakerStrategy', () => {
    it('should open circuit after threshold failures', async () => {
      const strategy = circuitBreakerStrategy<string>({
        failureThreshold: 3,
        resetTimeoutMs: 100,
      });

      const operation = async () => {
        throw new Error('Always fails');
      };

      // Trigger failures to open the circuit
      for (let i = 0; i < 3; i++) {
        await withRecovery(operation, [strategy], { maxDuration: 100 });
      }

      expect(strategy.getState()).toBe('open');

      // Should not handle while open
      expect(strategy.canHandle(new Error('test'))).toBe(false);

      strategy.reset();
      expect(strategy.getState()).toBe('closed');
    });

    it('should transition to half-open after timeout', async () => {
      const strategy = circuitBreakerStrategy<string>({
        failureThreshold: 2,
        resetTimeoutMs: 50,
      });

      // Open the circuit
      for (let i = 0; i < 2; i++) {
        try {
          await strategy.recover(new Error('fail'), async () => { throw new Error('fail'); }, 1);
        } catch {}
      }

      expect(strategy.getState()).toBe('open');

      // Wait for reset timeout
      await new Promise(resolve => setTimeout(resolve, 60));

      // Should be able to handle (half-open)
      expect(strategy.canHandle(new Error('test'))).toBe(true);

      strategy.reset();
    });
  });

  describe('gracePeriodStrategy', () => {
    it('should return default value during grace period', async () => {
      const strategy = gracePeriodStrategy<string>({
        gracePeriodMs: 1000,
        defaultValue: 'default',
        errorCodes: [ErrorCode.NOT_FOUND],
      });

      const operation = async () => {
        throw new MycelixError('Not found', ErrorCode.NOT_FOUND);
      };

      const result = await withRecovery(operation, [strategy]);

      expect(result.success).toBe(true);
      expect(result.value).toBe('default');
    });

    it('should not handle errors outside configured codes', async () => {
      const strategy = gracePeriodStrategy<string>({
        errorCodes: [ErrorCode.NOT_FOUND],
      });

      expect(strategy.canHandle(new MycelixError('Invalid', ErrorCode.INVALID_ARGUMENT))).toBe(false);
      expect(strategy.canHandle(new MycelixError('Not found', ErrorCode.NOT_FOUND))).toBe(true);
    });
  });

  describe('combineStrategies', () => {
    it('should try strategies in order', async () => {
      const strategy1 = exponentialBackoffStrategy<string>({
        maxRetries: 1,
        initialDelayMs: 10,
        retryableErrors: [ErrorCode.TIMEOUT],
      });

      const strategy2 = fallbackChainStrategy<string>([
        async () => 'fallback-result',
      ]);

      const combined = combineStrategies([strategy1, strategy2]);

      expect(combined.name).toContain('exponentialBackoff');
      expect(combined.name).toContain('fallbackChain');
    });
  });

  describe('withRecovery', () => {
    it('should return success on immediate success', async () => {
      const result = await withRecovery(
        async () => 'immediate-success',
        []
      );

      expect(result.success).toBe(true);
      expect(result.value).toBe('immediate-success');
      expect(result.attempts).toBe(1);
      expect(result.strategyUsed).toBe('none');
    });

    it('should respect maxDuration', async () => {
      const strategy = exponentialBackoffStrategy<string>({
        maxRetries: 100,
        initialDelayMs: 50,
      });

      const result = await withRecovery(
        async () => { throw new MycelixError('Timeout', ErrorCode.TIMEOUT); },
        [strategy],
        { maxDuration: 100 }
      );

      expect(result.success).toBe(false);
      // Allow generous tolerance for CI/test environment variability
      expect(result.duration).toBeLessThan(500);
    });
  });

  describe('withRecoveryOrThrow', () => {
    it('should return value on success', async () => {
      const value = await withRecoveryOrThrow(
        async () => 'success-value',
        []
      );

      expect(value).toBe('success-value');
    });

    it('should throw on failure', async () => {
      await expect(
        withRecoveryOrThrow(
          async () => { throw new Error('Always fails'); },
          [],
          { maxDuration: 100 }
        )
      ).rejects.toThrow();
    });
  });
});
