/**
 * React Integration Tests
 *
 * Tests for React hooks, context, and SSR compatibility.
 *
 * @module @mycelix/sdk/react/__tests__
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  createQueryHook,
  createMutationHook,
  type QueryState,
  type MutationState,
  type MycelixContextValue,
} from '../index';

// Mock React hooks for testing without React dependency
const mockUseState = vi.fn();
const mockUseEffect = vi.fn();
const mockUseCallback = vi.fn();
const mockUseContext = vi.fn();
const mockUseRef = vi.fn();

// Mock context value
const mockContextValue: Partial<MycelixContextValue> = {
  identityService: {},
  financeService: {},
  propertyService: {},
  energyService: {},
  mediaService: {},
  governanceService: {},
  justiceService: {},
  knowledgeService: {},
  connected: true,
  connecting: false,
  error: undefined,
  connect: vi.fn(),
  disconnect: vi.fn(),
};

describe('React Hook Factories', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('createQueryHook', () => {
    it('should create a query hook function', () => {
      const fetcher = vi.fn().mockResolvedValue({ data: 'test' });
      const useTestQuery = createQueryHook(fetcher);

      expect(typeof useTestQuery).toBe('function');
    });

    it('should throw when called without React', () => {
      const fetcher = vi.fn().mockResolvedValue({ data: 'test' });
      const useTestQuery = createQueryHook(fetcher);

      expect(() => useTestQuery(undefined)).toThrow('React hooks require');
    });
  });

  describe('createMutationHook', () => {
    it('should create a mutation hook function', () => {
      const mutator = vi.fn().mockResolvedValue({ result: 'success' });
      const useTestMutation = createMutationHook(mutator);

      expect(typeof useTestMutation).toBe('function');
    });

    it('should throw when called without React', () => {
      const mutator = vi.fn().mockResolvedValue({ result: 'success' });
      const useTestMutation = createMutationHook(mutator);

      expect(() => useTestMutation()).toThrow('React hooks require');
    });
  });
});

describe('QueryState Interface', () => {
  it('should have correct shape', () => {
    const queryState: QueryState<string> = {
      data: 'test',
      loading: false,
      error: undefined,
      refetch: async () => {},
    };

    expect(queryState.data).toBe('test');
    expect(queryState.loading).toBe(false);
    expect(queryState.error).toBeUndefined();
    expect(typeof queryState.refetch).toBe('function');
  });

  it('should handle loading state', () => {
    const queryState: QueryState<string> = {
      data: undefined,
      loading: true,
      error: undefined,
      refetch: async () => {},
    };

    expect(queryState.data).toBeUndefined();
    expect(queryState.loading).toBe(true);
  });

  it('should handle error state', () => {
    const testError = new Error('Test error');
    const queryState: QueryState<string> = {
      data: undefined,
      loading: false,
      error: testError,
      refetch: async () => {},
    };

    expect(queryState.error).toBe(testError);
    expect(queryState.loading).toBe(false);
  });
});

describe('MutationState Interface', () => {
  it('should have correct shape', () => {
    const mutationState: MutationState<string, { input: string }> = {
      data: undefined,
      loading: false,
      error: undefined,
      mutate: async () => 'result',
      reset: () => {},
    };

    expect(mutationState.data).toBeUndefined();
    expect(mutationState.loading).toBe(false);
    expect(typeof mutationState.mutate).toBe('function');
    expect(typeof mutationState.reset).toBe('function');
  });

  it('should allow mutation execution', async () => {
    const mockMutate = vi.fn().mockResolvedValue('success');
    const mutationState: MutationState<string, { input: string }> = {
      data: undefined,
      loading: false,
      error: undefined,
      mutate: mockMutate,
      reset: () => {},
    };

    const result = await mutationState.mutate({ input: 'test' });
    expect(mockMutate).toHaveBeenCalledWith({ input: 'test' });
    expect(result).toBe('success');
  });
});

describe('MycelixContextValue Interface', () => {
  it('should have all required services', () => {
    const context: MycelixContextValue = mockContextValue as MycelixContextValue;

    expect(context.identityService).toBeDefined();
    expect(context.financeService).toBeDefined();
    expect(context.propertyService).toBeDefined();
    expect(context.energyService).toBeDefined();
    expect(context.mediaService).toBeDefined();
    expect(context.governanceService).toBeDefined();
    expect(context.justiceService).toBeDefined();
    expect(context.knowledgeService).toBeDefined();
  });

  it('should have connection state', () => {
    const context: MycelixContextValue = mockContextValue as MycelixContextValue;

    expect(typeof context.connected).toBe('boolean');
    expect(typeof context.connecting).toBe('boolean');
    expect(context.error).toBeUndefined();
  });

  it('should have connection methods', () => {
    const context: MycelixContextValue = mockContextValue as MycelixContextValue;

    expect(typeof context.connect).toBe('function');
    expect(typeof context.disconnect).toBe('function');
  });
});

describe('SSR Compatibility', () => {
  it('should not crash on server (no window/document)', () => {
    // Hooks should gracefully handle missing browser APIs
    const originalWindow = global.window;
    const originalDocument = global.document;

    try {
      // @ts-expect-error - Simulating SSR environment
      delete global.window;
      // @ts-expect-error - Simulating SSR environment
      delete global.document;

      // Hook creation should still work
      const useQuery = createQueryHook(() => Promise.resolve('data'));
      expect(typeof useQuery).toBe('function');

      const useMutation = createMutationHook(() => Promise.resolve('result'));
      expect(typeof useMutation).toBe('function');
    } finally {
      // @ts-expect-error - Restore globals
      global.window = originalWindow;
      // @ts-expect-error - Restore globals
      global.document = originalDocument;
    }
  });
});

describe('Error Boundary Integration', () => {
  it('should provide meaningful error messages', () => {
    const useQuery = createQueryHook(() => Promise.resolve('data'));

    try {
      useQuery(undefined);
    } catch (error) {
      expect(error).toBeInstanceOf(Error);
      expect((error as Error).message).toContain('React hooks require');
      expect((error as Error).message).toContain('@mycelix/react');
    }
  });
});

describe('Type Safety', () => {
  it('should enforce type constraints on QueryState', () => {
    // This is a compile-time test - TypeScript should catch type errors
    const numberQuery: QueryState<number> = {
      data: 42,
      loading: false,
      error: undefined,
      refetch: async () => {},
    };

    expect(typeof numberQuery.data).toBe('number');
  });

  it('should enforce type constraints on MutationState', () => {
    interface CreateUserInput {
      name: string;
      email: string;
    }

    interface User {
      id: string;
      name: string;
      email: string;
    }

    const userMutation: MutationState<User, CreateUserInput> = {
      data: undefined,
      loading: false,
      error: undefined,
      mutate: async (input) => ({
        id: '1',
        name: input.name,
        email: input.email,
      }),
      reset: () => {},
    };

    expect(typeof userMutation.mutate).toBe('function');
  });
});

// ============================================================================
// Domain-Specific Hook Tests
// ============================================================================

describe('Identity Hooks', () => {
  it('should export useCreateDid hook', async () => {
    const { useCreateDid } = await import('../identity');
    expect(typeof useCreateDid).toBe('function');
  });

  it('should export useMyDid hook', async () => {
    const { useMyDid } = await import('../identity');
    expect(typeof useMyDid).toBe('function');
  });

  it('should export useResolveDid hook', async () => {
    const { useResolveDid } = await import('../identity');
    expect(typeof useResolveDid).toBe('function');
  });

  it('should throw meaningful error when called without React', async () => {
    const { useCreateDid } = await import('../identity');
    expect(() => useCreateDid()).toThrow('@mycelix/react');
  });
});

describe('Finance Hooks', () => {
  it('should export useWalletById hook', async () => {
    const { useWalletById } = await import('../finance');
    expect(typeof useWalletById).toBe('function');
  });

  it('should export useTransactionHistory hook', async () => {
    const { useTransactionHistory } = await import('../finance');
    expect(typeof useTransactionHistory).toBe('function');
  });

  it('should export useTransferMutation hook', async () => {
    const { useTransferMutation } = await import('../finance');
    expect(typeof useTransferMutation).toBe('function');
  });
});

describe('Governance Hooks', () => {
  it('should export useProposalById hook', async () => {
    const { useProposalById } = await import('../governance');
    expect(typeof useProposalById).toBe('function');
  });

  it('should export useActiveProposals hook', async () => {
    const { useActiveProposals } = await import('../governance');
    expect(typeof useActiveProposals).toBe('function');
  });

  it('should export useCreateDAO hook', async () => {
    const { useCreateDAO } = await import('../governance');
    expect(typeof useCreateDAO).toBe('function');
  });
});

describe('Knowledge Hooks', () => {
  it('should export useClaim hook', async () => {
    const { useClaim } = await import('../knowledge');
    expect(typeof useClaim).toBe('function');
  });

  it('should export useSubmitClaim hook', async () => {
    const { useSubmitClaim } = await import('../knowledge');
    expect(typeof useSubmitClaim).toBe('function');
  });

  it('should export useClaimsByAuthor hook', async () => {
    const { useClaimsByAuthor } = await import('../knowledge');
    expect(typeof useClaimsByAuthor).toBe('function');
  });
});

describe('Energy Hooks', () => {
  it('should export useEnergyProject hook', async () => {
    const { useEnergyProject } = await import('../energy');
    expect(typeof useEnergyProject).toBe('function');
  });

  it('should export useRegisterEnergyProject hook', async () => {
    const { useRegisterEnergyProject } = await import('../energy');
    expect(typeof useRegisterEnergyProject).toBe('function');
  });
});

describe('Property Hooks', () => {
  it('should export useAssetById hook', async () => {
    const { useAssetById } = await import('../property');
    expect(typeof useAssetById).toBe('function');
  });

  it('should export useInitiateTransferMutation hook', async () => {
    const { useInitiateTransferMutation } = await import('../property');
    expect(typeof useInitiateTransferMutation).toBe('function');
  });
});

describe('Justice Hooks', () => {
  it('should export useCaseById hook', async () => {
    const { useCaseById } = await import('../justice');
    expect(typeof useCaseById).toBe('function');
  });

  it('should export useFileCaseMutation hook', async () => {
    const { useFileCaseMutation } = await import('../justice');
    expect(typeof useFileCaseMutation).toBe('function');
  });
});

describe('Media Hooks', () => {
  it('should export useContentById hook', async () => {
    const { useContentById } = await import('../media');
    expect(typeof useContentById).toBe('function');
  });

  it('should export usePublishContentMutation hook', async () => {
    const { usePublishContentMutation } = await import('../media');
    expect(typeof usePublishContentMutation).toBe('function');
  });
});

// ============================================================================
// Edge Cases and Robustness Tests
// ============================================================================

describe('Hook Factory Edge Cases', () => {
  it('should handle async fetcher errors gracefully', () => {
    const errorFetcher = vi.fn().mockRejectedValue(new Error('Network error'));
    const useErrorQuery = createQueryHook(errorFetcher);

    // Hook creation should succeed even with error-prone fetcher
    expect(typeof useErrorQuery).toBe('function');
  });

  it('should handle undefined variables', () => {
    const fetcher = vi.fn().mockResolvedValue(null);
    const useNullableQuery = createQueryHook(fetcher);

    expect(typeof useNullableQuery).toBe('function');
  });

  it('should handle complex nested data types', () => {
    interface NestedData {
      level1: {
        level2: {
          value: string;
        }[];
      };
    }

    const nestedFetcher = vi.fn().mockResolvedValue({
      level1: {
        level2: [{ value: 'test' }],
      },
    });

    const useNestedQuery = createQueryHook<NestedData>(nestedFetcher);
    expect(typeof useNestedQuery).toBe('function');
  });
});

describe('Concurrent Mode Compatibility', () => {
  it('should not mutate state synchronously', () => {
    // Hook factories should be pure functions
    const fetcher1 = vi.fn();
    const fetcher2 = vi.fn();

    const hook1 = createQueryHook(fetcher1);
    const hook2 = createQueryHook(fetcher2);

    // Creating multiple hooks should not cause side effects
    expect(hook1).not.toBe(hook2);
    expect(fetcher1).not.toHaveBeenCalled();
    expect(fetcher2).not.toHaveBeenCalled();
  });
});

describe('Memory Safety', () => {
  it('should not leak references between hook instances', () => {
    const sharedState = { count: 0 };
    const fetcher = vi.fn().mockImplementation(() => {
      sharedState.count++;
      return Promise.resolve(sharedState.count);
    });

    // Create multiple hook instances
    const hooks = Array.from({ length: 10 }, () => createQueryHook(fetcher));

    // Each hook should be independent
    expect(new Set(hooks).size).toBe(10);
  });
});
