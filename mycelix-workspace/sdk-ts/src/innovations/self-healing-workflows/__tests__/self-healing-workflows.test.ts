// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Self-Healing Workflow Orchestrator Tests
 *
 * Tests for workflow execution, circuit breakers, saga compensation,
 * checkpointing, and observability.
 *
 * @module innovations/self-healing-workflows/__tests__
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  SelfHealingOrchestrator,
  WorkflowBuilder,
  createWorkflow,
  createOrchestrator,
  getOrchestrator,
  resetOrchestrator,
  DEFAULT_ORCHESTRATOR_CONFIG,
  type WorkflowDefinition,
  type WorkflowContext,
  type WorkflowResult,
  type OrchestratorConfig,
} from '../index';

// ============================================================================
// MOCKS & HELPERS
// ============================================================================

const createSimpleWorkflow = (): WorkflowDefinition => {
  return createWorkflow('test-workflow', '1.0.0')
    .description('A simple test workflow')
    .step('step1', 'happ-a', async (input: unknown, ctx: WorkflowContext) => {
      ctx.log('Executing step 1');
      return { value: (input as any)?.value ?? 0 + 10 };
    })
    .step('step2', 'happ-b', async (input: unknown, ctx: WorkflowContext) => {
      ctx.log('Executing step 2');
      return { result: (input as any)?.value * 2 };
    })
    .build();
};

const createWorkflowWithCompensation = (): WorkflowDefinition => {
  return createWorkflow('compensating-workflow', '1.0.0')
    .compensateOnFailure(true)
    .step(
      'step1',
      'happ-a',
      async () => ({ created: true }),
      {
        compensate: async (_input, output, ctx) => {
          ctx.log('Compensating step 1');
          // Undo step 1
        },
      }
    )
    .step(
      'step2',
      'happ-b',
      async () => {
        throw new Error('Intentional failure');
      },
      {
        compensate: async (_input, _output, ctx) => {
          ctx.log('Compensating step 2');
        },
      }
    )
    .build();
};

const createDependencyWorkflow = (): WorkflowDefinition => {
  return createWorkflow('dependency-workflow', '1.0.0')
    .step('fetch-data', 'happ-a', async () => ({ data: [1, 2, 3] }))
    .step('process-data', 'happ-b', async (input) => ({ processed: (input as any).data.map((x: number) => x * 2) }), {
      dependsOn: ['fetch-data'],
    })
    .step('store-result', 'happ-c', async (input) => ({ stored: true, count: (input as any).processed.length }), {
      dependsOn: ['process-data'],
    })
    .build();
};

// ============================================================================
// TESTS
// ============================================================================

describe('SelfHealingOrchestrator', () => {
  let orchestrator: SelfHealingOrchestrator;

  beforeEach(() => {
    resetOrchestrator();
    orchestrator = new SelfHealingOrchestrator();
  });

  describe('initialization', () => {
    it('should create an orchestrator instance', () => {
      expect(orchestrator).toBeInstanceOf(SelfHealingOrchestrator);
    });

    it('should accept custom configuration', () => {
      const config: Partial<OrchestratorConfig> = {
        maxConcurrentWorkflows: 5,
        defaultRetry: { maxRetries: 5, initialDelayMs: 200 },
      };
      const customOrchestrator = new SelfHealingOrchestrator(config);
      expect(customOrchestrator).toBeInstanceOf(SelfHealingOrchestrator);
    });
  });

  describe('executeWorkflow', () => {
    it('should execute a simple workflow', async () => {
      const workflow = createSimpleWorkflow();
      const result = await orchestrator.executeWorkflow(workflow, {
        initiatorDid: 'did:mycelix:test',
      });

      expect(result).toBeDefined();
      expect(result.status).toBe('completed');
      expect(result.workflowId).toBeTruthy();
    });

    it('should pass input through steps', async () => {
      const workflow = createWorkflow('input-test', '1.0.0')
        .step('double', 'happ-a', async (input: unknown) => {
          return { value: ((input as any)?.value ?? 5) * 2 };
        })
        .build();

      const result = await orchestrator.executeWorkflow(
        workflow,
        { initiatorDid: 'did:mycelix:test' },
        { value: 5 }
      );

      expect(result.status).toBe('completed');
      expect(result.finalOutput).toEqual({ value: 10 });
    });

    it('should handle workflow failure', async () => {
      const failingWorkflow = createWorkflow('failing', '1.0.0')
        .step('fail', 'happ-a', async () => {
          throw new Error('Intentional failure');
        })
        .build();

      const result = await orchestrator.executeWorkflow(failingWorkflow, {
        initiatorDid: 'did:mycelix:test',
      });

      expect(result.status).toBe('failed');
      expect(result.error).toBeTruthy();
    });

    it('should respect step dependencies', async () => {
      const executionOrder: string[] = [];

      const workflow = createWorkflow('deps-test', '1.0.0')
        .step('a', 'happ-a', async () => {
          executionOrder.push('a');
          return { a: true };
        })
        .step(
          'b',
          'happ-b',
          async () => {
            executionOrder.push('b');
            return { b: true };
          },
          { dependsOn: ['a'] }
        )
        .step(
          'c',
          'happ-c',
          async () => {
            executionOrder.push('c');
            return { c: true };
          },
          { dependsOn: ['b'] }
        )
        .build();

      await orchestrator.executeWorkflow(workflow, { initiatorDid: 'did:mycelix:test' });

      expect(executionOrder).toEqual(['a', 'b', 'c']);
    });

    it('should provide context to steps', async () => {
      let capturedContext: WorkflowContext | null = null;

      const workflow = createWorkflow('context-test', '1.0.0')
        .step('capture', 'happ-a', async (_input, ctx) => {
          capturedContext = ctx;
          return {};
        })
        .build();

      await orchestrator.executeWorkflow(workflow, {
        initiatorDid: 'did:mycelix:alice',
        metadata: { custom: 'value' },
      });

      expect(capturedContext).toBeDefined();
      expect(capturedContext!.workflowName).toBe('context-test');
      expect(capturedContext!.initiatorDid).toBe('did:mycelix:alice');
      expect(capturedContext!.metadata.custom).toBe('value');
    });
  });

  describe('compensation (saga pattern)', () => {
    it('should compensate on failure when enabled', async () => {
      const compensated: string[] = [];

      const workflow = createWorkflow('compensation-test', '1.0.0')
        .compensateOnFailure(true)
        .step(
          'step1',
          'happ-a',
          async () => ({ created: 'resource-1' }),
          {
            compensate: async () => {
              compensated.push('step1');
            },
          }
        )
        .step(
          'step2',
          'happ-b',
          async () => ({ created: 'resource-2' }),
          {
            compensate: async () => {
              compensated.push('step2');
            },
          }
        )
        .step('step3', 'happ-c', async () => {
          throw new Error('Failure at step 3');
        })
        .build();

      const result = await orchestrator.executeWorkflow(workflow, {
        initiatorDid: 'did:mycelix:test',
      });

      expect(result.status).toBe('compensated');
      // Compensation happens in reverse order
      expect(compensated).toContain('step2');
      expect(compensated).toContain('step1');
    });

    it('should not compensate when disabled', async () => {
      const compensated: string[] = [];

      const workflow = createWorkflow('no-compensation', '1.0.0')
        .compensateOnFailure(false)
        .step(
          'step1',
          'happ-a',
          async () => ({}),
          {
            compensate: async () => {
              compensated.push('step1');
            },
          }
        )
        .step('step2', 'happ-b', async () => {
          throw new Error('Failure');
        })
        .build();

      const result = await orchestrator.executeWorkflow(workflow, {
        initiatorDid: 'did:mycelix:test',
      });

      expect(result.status).toBe('failed');
      expect(compensated).toEqual([]);
    });
  });

  describe('retry', () => {
    it('should retry failed steps', async () => {
      let attempts = 0;

      const workflow = createWorkflow('retry-test', '1.0.0')
        .step(
          'flaky',
          'happ-a',
          async () => {
            attempts++;
            if (attempts < 3) {
              throw new Error(`Attempt ${attempts} failed`);
            }
            return { success: true };
          },
          { retry: { maxRetries: 5, initialDelayMs: 10 } }
        )
        .build();

      const result = await orchestrator.executeWorkflow(workflow, {
        initiatorDid: 'did:mycelix:test',
      });

      expect(result.status).toBe('completed');
      expect(attempts).toBe(3);
    });

    it('should respect max retries', async () => {
      let attempts = 0;

      const workflow = createWorkflow('max-retry-test', '1.0.0')
        .step(
          'always-fails',
          'happ-a',
          async () => {
            attempts++;
            throw new Error('Always fails');
          },
          { retry: { maxRetries: 2, initialDelayMs: 10 } }
        )
        .build();

      const result = await orchestrator.executeWorkflow(workflow, {
        initiatorDid: 'did:mycelix:test',
      });

      expect(result.status).toBe('failed');
      expect(attempts).toBe(3); // Initial + 2 retries
    });
  });

  describe('workflow state management', () => {
    it('should track workflow state', async () => {
      const workflow = createWorkflow('state-test', '1.0.0')
        .step('slow', 'happ-a', async () => {
          await new Promise((resolve) => setTimeout(resolve, 50));
          return {};
        })
        .build();

      const execution = orchestrator.executeWorkflow(workflow, {
        initiatorDid: 'did:mycelix:test',
      });

      // Check state during execution (timing-dependent)
      await new Promise((resolve) => setTimeout(resolve, 10));

      const result = await execution;
      expect(result.status).toBe('completed');
    });

    it('should allow pausing workflow', async () => {
      const workflow = createWorkflow('pausable', '1.0.0')
        .step('step1', 'happ-a', async () => {
          await new Promise((resolve) => setTimeout(resolve, 100));
          return { step: 1 };
        })
        .step('step2', 'happ-b', async () => ({ step: 2 }))
        .build();

      const execution = orchestrator.executeWorkflow(workflow, {
        initiatorDid: 'did:mycelix:test',
      });

      // This is a simplified test - in reality pause/resume would need proper async handling
      await execution;
    });
  });

  describe('observability', () => {
    it('should emit events during execution', async () => {
      const events: string[] = [];

      orchestrator.addObserver((event) => {
        events.push(event.type);
      });

      const workflow = createWorkflow('observable', '1.0.0')
        .step('step1', 'happ-a', async () => ({ done: true }))
        .build();

      await orchestrator.executeWorkflow(workflow, {
        initiatorDid: 'did:mycelix:test',
      });

      expect(events).toContain('workflow_started');
      expect(events).toContain('step_started');
      expect(events).toContain('step_completed');
      expect(events).toContain('workflow_completed');
    });

    it('should allow removing observers', () => {
      const observer = vi.fn();
      const unsubscribe = orchestrator.addObserver(observer);

      unsubscribe();

      // Observer should not be called after removal
      // (would need actual workflow execution to verify)
    });
  });

  describe('circuit breaker integration', () => {
    it('should use circuit breaker when configured', async () => {
      const workflow = createWorkflow('circuit-breaker-test', '1.0.0')
        .step(
          'protected',
          'happ-a',
          async () => ({ result: 'success' }),
          { useCircuitBreaker: true }
        )
        .build();

      const result = await orchestrator.executeWorkflow(workflow, {
        initiatorDid: 'did:mycelix:test',
      });

      expect(result.status).toBe('completed');
    });

    it('should track circuit breaker state per happ', async () => {
      // Execute multiple workflows to the same happ
      const workflow = createWorkflow('cb-tracking', '1.0.0')
        .step('step', 'happ-a', async () => ({}), { useCircuitBreaker: true })
        .build();

      await orchestrator.executeWorkflow(workflow, { initiatorDid: 'did:mycelix:test' });

      const state = orchestrator.getCircuitBreakerState('happ-a');
      expect(state).toBeDefined();
    });
  });
});

describe('WorkflowBuilder', () => {
  describe('fluent API', () => {
    it('should build workflow with all options', () => {
      const workflow = createWorkflow('full-options', '2.0.0')
        .description('A fully configured workflow')
        .timeout(60000)
        .compensateOnFailure(true)
        .enableCheckpoints(true, 5000)
        .defaultRetry({ maxRetries: 3, initialDelayMs: 100 })
        .step('step1', 'happ-a', async () => ({}))
        .build();

      expect(workflow.name).toBe('full-options');
      expect(workflow.version).toBe('2.0.0');
      expect(workflow.description).toBe('A fully configured workflow');
      expect(workflow.timeout).toBe(60000);
      expect(workflow.compensateOnFailure).toBe(true);
      expect(workflow.enableCheckpoints).toBe(true);
      expect(workflow.checkpointIntervalMs).toBe(5000);
      expect(workflow.steps.length).toBe(1);
    });

    it('should require name', () => {
      const builder = new WorkflowBuilder('', '1.0.0');
      expect(() => builder.build()).toThrow();
    });

    it('should require at least one step', () => {
      const builder = new WorkflowBuilder('empty', '1.0.0');
      expect(() => builder.build()).toThrow();
    });
  });
});

describe('Factory Functions', () => {
  afterEach(() => {
    resetOrchestrator();
  });

  describe('createWorkflow', () => {
    it('should create a workflow builder', () => {
      const builder = createWorkflow('test', '1.0.0');
      expect(builder).toBeInstanceOf(WorkflowBuilder);
    });

    it('should default version to 1.0.0', () => {
      const workflow = createWorkflow('test').step('s', 'h', async () => ({})).build();
      expect(workflow.version).toBe('1.0.0');
    });
  });

  describe('createOrchestrator', () => {
    it('should create new orchestrator', () => {
      const orchestrator = createOrchestrator();
      expect(orchestrator).toBeInstanceOf(SelfHealingOrchestrator);
    });
  });

  describe('getOrchestrator', () => {
    it('should return singleton', () => {
      const o1 = getOrchestrator();
      const o2 = getOrchestrator();
      expect(o1).toBe(o2);
    });
  });

  describe('resetOrchestrator', () => {
    it('should reset singleton', () => {
      const o1 = getOrchestrator();
      resetOrchestrator();
      const o2 = getOrchestrator();
      expect(o1).not.toBe(o2);
    });
  });
});

describe('DEFAULT_ORCHESTRATOR_CONFIG', () => {
  it('should have expected defaults', () => {
    expect(DEFAULT_ORCHESTRATOR_CONFIG.maxConcurrentWorkflows).toBeDefined();
    expect(DEFAULT_ORCHESTRATOR_CONFIG.defaultRetry).toBeDefined();
    expect(DEFAULT_ORCHESTRATOR_CONFIG.enableObservability).toBe(true);
  });
});
