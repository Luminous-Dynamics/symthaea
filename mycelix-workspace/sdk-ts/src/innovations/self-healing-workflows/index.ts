/**
 * @mycelix/sdk Self-Healing Workflow Orchestrator
 *
 * Enterprise-grade workflow orchestration with:
 * - Circuit breaker integration (fail fast, recover gracefully)
 * - Saga pattern compensation (rollback on partial failure)
 * - Observable state (real-time progress tracking)
 * - Automatic retry with exponential backoff
 * - Checkpoint and resume for long-running workflows
 *
 * Philosophy: Distributed systems fail. Good systems expect and handle it.
 * The question isn't "will it fail?" but "how gracefully will it recover?"
 *
 * @packageDocumentation
 * @module innovations/self-healing-workflows
 */

import {
  getCrossHappBridge,
  type CrossHappBridge,
  type HappId,
} from '../../bridge/cross-happ.js';
import {
  CircuitBreaker,
  type CircuitState,
  type CircuitBreakerConfig,
} from '../../common/circuit-breaker.js';
import {
  RetryPolicy,
  type RetryOptions,
} from '../../common/retry.js';

// ============================================================================
// TYPES
// ============================================================================

/**
 * Workflow status
 */
export type WorkflowStatus =
  | 'pending'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'compensating'
  | 'compensated';

/**
 * Step status
 */
export type StepStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'skipped'
  | 'compensating'
  | 'compensated';

/**
 * Workflow step definition
 */
export interface WorkflowStepDefinition<TInput = unknown, TOutput = unknown> {
  /** Step name */
  name: string;
  /** Step description */
  description?: string;
  /** Target hApp for this step */
  targetHapp: HappId;
  /** Execute function */
  execute: (input: TInput, context: WorkflowContext) => Promise<TOutput>;
  /** Compensation function (rollback) */
  compensate?: (input: TInput, output: TOutput | undefined, context: WorkflowContext) => Promise<void>;
  /** Retry configuration for this step */
  retry?: RetryOptions;
  /** Timeout in ms */
  timeout?: number;
  /** Whether this step is critical (failure stops workflow) */
  critical?: boolean;
  /** Dependencies (step names that must complete first) */
  dependsOn?: string[];
  /** Whether to use circuit breaker */
  useCircuitBreaker?: boolean;
  /** Custom circuit breaker config */
  circuitBreakerConfig?: Partial<CircuitBreakerConfig>;
}

/**
 * Workflow definition
 */
export interface WorkflowDefinition {
  /** Workflow name */
  name: string;
  /** Workflow description */
  description?: string;
  /** Version */
  version: string;
  /** Steps in the workflow */
  steps: WorkflowStepDefinition[];
  /** Global retry configuration */
  defaultRetry?: RetryOptions;
  /** Global timeout in ms */
  timeout?: number;
  /** Whether to compensate on failure */
  compensateOnFailure?: boolean;
  /** Maximum compensation retries */
  maxCompensationRetries?: number;
  /** Enable checkpoints for long-running workflows */
  enableCheckpoints?: boolean;
  /** Checkpoint interval in ms */
  checkpointIntervalMs?: number;
}

/**
 * Workflow context passed through execution
 */
export interface WorkflowContext {
  /** Workflow instance ID */
  workflowId: string;
  /** Workflow name */
  workflowName: string;
  /** Initiator DID */
  initiatorDid: string;
  /** Bridge for cross-hApp communication */
  bridge: CrossHappBridge;
  /** Step outputs indexed by step name */
  stepOutputs: Map<string, unknown>;
  /** Workflow metadata */
  metadata: Record<string, unknown>;
  /** Started at timestamp */
  startedAt: number;
  /** Current step */
  currentStep?: string;
  /** Logger function */
  log: (message: string, level?: 'info' | 'warn' | 'error') => void;
}

/**
 * Workflow step state
 */
export interface WorkflowStepState {
  /** Step name */
  name: string;
  /** Status */
  status: StepStatus;
  /** Input */
  input?: unknown;
  /** Output */
  output?: unknown;
  /** Error if failed */
  error?: string;
  /** Started at */
  startedAt?: number;
  /** Completed at */
  completedAt?: number;
  /** Retry count */
  retryCount: number;
  /** Circuit breaker state */
  circuitState?: CircuitState;
}

/**
 * Workflow instance state
 */
export interface WorkflowState {
  /** Workflow ID */
  id: string;
  /** Workflow name */
  name: string;
  /** Version */
  version: string;
  /** Status */
  status: WorkflowStatus;
  /** Initiator */
  initiatorDid: string;
  /** Step states */
  steps: Map<string, WorkflowStepState>;
  /** Started at */
  startedAt: number;
  /** Completed at */
  completedAt?: number;
  /** Final result */
  result?: unknown;
  /** Error if failed */
  error?: string;
  /** Metadata */
  metadata: Record<string, unknown>;
  /** Last checkpoint */
  lastCheckpoint?: WorkflowCheckpoint;
}

/**
 * Workflow checkpoint for resume
 */
export interface WorkflowCheckpoint {
  /** Checkpoint ID */
  id: string;
  /** Workflow ID */
  workflowId: string;
  /** Completed steps */
  completedSteps: string[];
  /** Step outputs */
  stepOutputs: Map<string, unknown>;
  /** Timestamp */
  timestamp: number;
}

/**
 * Workflow execution result
 */
export interface WorkflowResult<T = unknown> {
  /** Success status */
  success: boolean;
  /** Workflow ID */
  workflowId: string;
  /** Final result */
  result?: T;
  /** Final output (alias for result, for test compatibility) */
  finalOutput?: T;
  /** Status for test compatibility */
  status: WorkflowStatus;
  /** Error if failed */
  error?: string;
  /** Step results */
  stepResults: WorkflowStepState[];
  /** Total duration */
  durationMs: number;
  /** Whether compensation was needed */
  compensated: boolean;
  /** Compensation errors if any */
  compensationErrors?: string[];
  /** Started at timestamp */
  startedAt: number;
  /** Completed at timestamp */
  completedAt?: number;
}

/**
 * State change event
 */
export interface WorkflowStateEvent {
  type: 'workflow_started' | 'step_started' | 'step_completed' | 'step_failed' |
        'workflow_completed' | 'workflow_failed' | 'compensation_started' |
        'compensation_completed' | 'checkpoint_created';
  workflowId: string;
  stepName?: string;
  data?: unknown;
  timestamp: number;
}

/**
 * Workflow observer callback
 */
export type WorkflowObserver = (event: WorkflowStateEvent) => void;

/**
 * Orchestrator configuration
 */
export interface OrchestratorConfig {
  /** Default retry policy */
  defaultRetry: RetryOptions;
  /** Default circuit breaker config */
  defaultCircuitBreaker: Partial<CircuitBreakerConfig>;
  /** Enable global observability */
  enableObservability: boolean;
  /** Maximum concurrent workflows */
  maxConcurrentWorkflows: number;
  /** Persist workflow state */
  enablePersistence: boolean;
  /** Cleanup completed workflows after (ms) */
  cleanupAfterMs: number;
}

// ============================================================================
// DEFAULT CONFIGURATION
// ============================================================================

const DEFAULT_ORCHESTRATOR_CONFIG: OrchestratorConfig = {
  defaultRetry: {
    maxAttempts: 3,
    initialDelayMs: 1000,
    backoffMultiplier: 2,
    maxDelayMs: 30000,
    jitter: true,
  },
  defaultCircuitBreaker: {
    failureThreshold: 5,
    resetTimeoutMs: 60000,
    successThreshold: 2,
  },
  enableObservability: true,
  maxConcurrentWorkflows: 100,
  enablePersistence: true,
  cleanupAfterMs: 24 * 60 * 60 * 1000, // 24 hours
};

// ============================================================================
// SELF-HEALING WORKFLOW ORCHESTRATOR
// ============================================================================

/**
 * Self-Healing Workflow Orchestrator
 *
 * Provides resilient workflow execution with:
 * - Circuit breakers per hApp/step
 * - Saga compensation for rollback
 * - Observable state for monitoring
 * - Automatic retry with backoff
 * - Checkpoint/resume for long workflows
 *
 * @example
 * ```typescript
 * const orchestrator = new SelfHealingOrchestrator();
 *
 * // Define a cross-hApp workflow
 * const workflow: WorkflowDefinition = {
 *   name: 'onboard-citizen',
 *   version: '1.0.0',
 *   compensateOnFailure: true,
 *   steps: [
 *     {
 *       name: 'create-identity',
 *       targetHapp: 'identity',
 *       execute: async (input, ctx) => {
 *         // Create identity
 *         return { did: 'did:mycelix:new-citizen' };
 *       },
 *       compensate: async (input, output, ctx) => {
 *         // Rollback: delete identity
 *       },
 *       useCircuitBreaker: true,
 *     },
 *     {
 *       name: 'setup-wallet',
 *       targetHapp: 'finance',
 *       dependsOn: ['create-identity'],
 *       execute: async (input, ctx) => {
 *         const identity = ctx.stepOutputs.get('create-identity');
 *         // Create wallet
 *       },
 *     },
 *   ],
 * };
 *
 * // Execute with observability
 * orchestrator.observe((event) => {
 *   console.log(`[${event.type}] ${event.stepName || event.workflowId}`);
 * });
 *
 * const result = await orchestrator.execute(workflow, {
 *   initiatorDid: 'did:mycelix:admin',
 *   input: { name: 'Alice' },
 * });
 * ```
 */
class SelfHealingOrchestrator {
  private config: OrchestratorConfig;
  private bridge: CrossHappBridge;
  private workflows: Map<string, WorkflowState> = new Map();
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();
  private observers: Set<WorkflowObserver> = new Set();
  private checkpoints: Map<string, WorkflowCheckpoint[]> = new Map();
  private activeWorkflows: number = 0;

  constructor(config: Partial<OrchestratorConfig> = {}) {
    this.config = { ...DEFAULT_ORCHESTRATOR_CONFIG, ...config };
    this.bridge = getCrossHappBridge();
  }

  /**
   * Execute a workflow
   */
  async execute<TInput = unknown, TResult = unknown>(
    definition: WorkflowDefinition,
    options: {
      initiatorDid: string;
      input: TInput;
      metadata?: Record<string, unknown>;
      resumeFrom?: string; // Checkpoint ID to resume from
    }
  ): Promise<WorkflowResult<TResult>> {
    // Check concurrency limit
    if (this.activeWorkflows >= this.config.maxConcurrentWorkflows) {
      throw new Error(`Maximum concurrent workflows (${this.config.maxConcurrentWorkflows}) reached`);
    }

    const workflowId = this.generateWorkflowId();
    const startTime = Date.now();
    this.activeWorkflows++;

    // Initialize state
    const state = this.initWorkflowState(workflowId, definition, options);
    this.workflows.set(workflowId, state);

    // Create context
    const context = this.createContext(workflowId, definition, options);

    // Emit started event
    this.emit({
      type: 'workflow_started',
      workflowId,
      data: { name: definition.name, input: options.input },
      timestamp: Date.now(),
    });

    try {
      // Check for resume
      if (options.resumeFrom) {
        const checkpoint = this.findCheckpoint(workflowId, options.resumeFrom);
        if (checkpoint) {
          this.restoreFromCheckpoint(state, context, checkpoint);
        }
      }

      // Execute steps
      const result = await this.executeSteps<TInput, TResult>(
        definition,
        state,
        context,
        options.input
      );

      state.status = 'completed';
      state.result = result;
      state.completedAt = Date.now();

      this.emit({
        type: 'workflow_completed',
        workflowId,
        data: { result },
        timestamp: Date.now(),
      });

      return {
        success: true,
        workflowId,
        result,
        finalOutput: result,
        status: 'completed' as WorkflowStatus,
        stepResults: Array.from(state.steps.values()),
        durationMs: Date.now() - startTime,
        compensated: false,
        startedAt: startTime,
        completedAt: Date.now(),
      };
    } catch (error) {
      state.status = 'failed';
      state.error = String(error);

      this.emit({
        type: 'workflow_failed',
        workflowId,
        data: { error: String(error) },
        timestamp: Date.now(),
      });

      // Compensate if enabled
      let compensationErrors: string[] | undefined;
      if (definition.compensateOnFailure) {
        compensationErrors = await this.compensate(definition, state, context);
      }

      // Determine final status
      const finalStatus: WorkflowStatus = compensationErrors && compensationErrors.length === 0
        ? 'compensated'
        : (definition.compensateOnFailure ? 'compensated' : 'failed');

      return {
        success: false,
        workflowId,
        error: String(error),
        status: finalStatus,
        stepResults: Array.from(state.steps.values()),
        durationMs: Date.now() - startTime,
        compensated: definition.compensateOnFailure ?? false,
        compensationErrors,
        startedAt: startTime,
        completedAt: Date.now(),
      };
    } finally {
      this.activeWorkflows--;

      // Schedule cleanup
      if (this.config.cleanupAfterMs > 0) {
        setTimeout(() => {
          this.workflows.delete(workflowId);
          this.checkpoints.delete(workflowId);
        }, this.config.cleanupAfterMs);
      }
    }
  }

  /**
   * Get workflow state
   */
  getWorkflowState(workflowId: string): WorkflowState | undefined {
    return this.workflows.get(workflowId);
  }

  /**
   * Pause a running workflow
   */
  pauseWorkflow(workflowId: string): boolean {
    const state = this.workflows.get(workflowId);
    if (state && state.status === 'running') {
      state.status = 'paused';
      return true;
    }
    return false;
  }

  /**
   * Resume a paused workflow
   */
  resumeWorkflow(workflowId: string): boolean {
    const state = this.workflows.get(workflowId);
    if (state && state.status === 'paused') {
      state.status = 'running';
      return true;
    }
    return false;
  }

  /**
   * Cancel a workflow
   */
  async cancelWorkflow(workflowId: string): Promise<boolean> {
    const state = this.workflows.get(workflowId);
    if (state && (state.status === 'running' || state.status === 'paused')) {
      state.status = 'failed';
      state.error = 'Workflow cancelled';
      return true;
    }
    return false;
  }

  /**
   * Subscribe to workflow events
   */
  observe(observer: WorkflowObserver): () => void {
    this.observers.add(observer);
    return () => this.observers.delete(observer);
  }

  /**
   * Subscribe to workflow events (alias for observe)
   */
  addObserver(observer: WorkflowObserver): () => void {
    return this.observe(observer);
  }

  /**
   * Get circuit breaker state for a hApp
   */
  getCircuitState(happId: HappId): CircuitState | undefined {
    return this.circuitBreakers.get(happId)?.getState();
  }

  /**
   * Get circuit breaker state for a hApp (alias for getCircuitState)
   */
  getCircuitBreakerState(happId: string): CircuitState | undefined {
    return this.getCircuitState(happId as HappId);
  }

  /**
   * Execute a workflow (alias for execute with different signature)
   */
  async executeWorkflow<TInput = unknown, TResult = unknown>(
    definition: WorkflowDefinition,
    options: {
      initiatorDid: string;
      metadata?: Record<string, unknown>;
      resumeFrom?: string;
    },
    input?: TInput
  ): Promise<WorkflowResult<TResult>> {
    return this.execute<TInput, TResult>(definition, {
      ...options,
      input: input ?? ({} as TInput),
    });
  }

  /**
   * Reset a circuit breaker
   */
  resetCircuitBreaker(happId: HappId): void {
    this.circuitBreakers.get(happId)?.reset();
  }

  /**
   * Get active workflow count
   */
  getActiveWorkflowCount(): number {
    return this.activeWorkflows;
  }

  // ==========================================================================
  // PRIVATE METHODS
  // ==========================================================================

  /**
   * Execute workflow steps
   */
  private async executeSteps<TInput, TResult>(
    definition: WorkflowDefinition,
    state: WorkflowState,
    context: WorkflowContext,
    input: TInput
  ): Promise<TResult> {
    state.status = 'running';
    const completedSteps = new Set<string>();
    let lastOutput: unknown = input;

    // Build dependency graph
    const pendingSteps = new Map<string, WorkflowStepDefinition>();
    for (const step of definition.steps) {
      pendingSteps.set(step.name, step);
    }

    // Helper to get current status (can be changed by external cancel/pause)
    const getCurrentStatus = (): WorkflowStatus => state.status;

    // Execute steps respecting dependencies
    while (pendingSteps.size > 0) {
      // Check if paused or cancelled (status may change externally)
      const currentStatus = getCurrentStatus();
      if (currentStatus === 'paused') {
        await this.waitForResume(state);
      }
      if (currentStatus === 'failed') {
        throw new Error(state.error || 'Workflow cancelled');
      }

      // Find steps that can run (dependencies satisfied)
      const runnableSteps: WorkflowStepDefinition[] = [];
      for (const [_name, step] of pendingSteps) {
        const deps = step.dependsOn || [];
        if (deps.every((d) => completedSteps.has(d))) {
          runnableSteps.push(step);
        }
      }

      if (runnableSteps.length === 0 && pendingSteps.size > 0) {
        throw new Error('Circular dependency detected in workflow');
      }

      // Execute runnable steps (could parallelize independent steps)
      for (const step of runnableSteps) {
        context.currentStep = step.name;

        const stepState = state.steps.get(step.name)!;
        stepState.status = 'running';
        stepState.startedAt = Date.now();
        stepState.input = lastOutput;

        this.emit({
          type: 'step_started',
          workflowId: state.id,
          stepName: step.name,
          timestamp: Date.now(),
        });

        try {
          // Get or create circuit breaker
          let circuitBreaker: CircuitBreaker | undefined;
          if (step.useCircuitBreaker) {
            circuitBreaker = this.getOrCreateCircuitBreaker(
              step.targetHapp,
              step.circuitBreakerConfig
            );
          }

          // Execute with retry and circuit breaker
          const output = await this.executeStepWithResilience(
            step,
            lastOutput,
            context,
            circuitBreaker
          );

          stepState.output = output;
          stepState.status = 'completed';
          stepState.completedAt = Date.now();
          context.stepOutputs.set(step.name, output);
          lastOutput = output;

          this.emit({
            type: 'step_completed',
            workflowId: state.id,
            stepName: step.name,
            data: { output },
            timestamp: Date.now(),
          });

          completedSteps.add(step.name);
          pendingSteps.delete(step.name);

          // Create checkpoint if enabled
          if (definition.enableCheckpoints) {
            this.createCheckpoint(state, context);
          }
        } catch (error) {
          stepState.status = 'failed';
          stepState.error = String(error);
          stepState.completedAt = Date.now();

          this.emit({
            type: 'step_failed',
            workflowId: state.id,
            stepName: step.name,
            data: { error: String(error) },
            timestamp: Date.now(),
          });

          if (step.critical !== false) {
            throw new Error(`Step "${step.name}" failed: ${error}`);
          }

          // Non-critical step: mark as skipped and continue
          stepState.status = 'skipped';
          pendingSteps.delete(step.name);
        }
      }
    }

    return lastOutput as TResult;
  }

  /**
   * Execute a step with retry and circuit breaker
   */
  private async executeStepWithResilience(
    step: WorkflowStepDefinition,
    input: unknown,
    context: WorkflowContext,
    circuitBreaker?: CircuitBreaker
  ): Promise<unknown> {
    const retryOptions = step.retry || this.config.defaultRetry;
    // For workflow steps, retry all errors by default unless explicitly configured
    const retryPolicy = new RetryPolicy({
      ...retryOptions,
      isRetryable: retryOptions.isRetryable ?? (() => true),
    });
    const stepState = this.workflows.get(context.workflowId)?.steps.get(step.name);

    const executeFn = async () => {
      // Check if circuit breaker is open
      if (circuitBreaker?.isOpen()) {
        if (stepState) {
          stepState.circuitState = circuitBreaker.getState();
        }
        throw new Error(`Circuit breaker open for ${step.targetHapp}`);
      }

      // Execute with timeout (circuit breaker wraps the call if present)
      const operation = () => this.withTimeout(
        step.execute(input, context),
        step.timeout || 30000
      );

      try {
        // Use circuit breaker's execute method if available (handles success/failure)
        const result = circuitBreaker
          ? await circuitBreaker.execute(operation)
          : await operation();

        return result;
      } catch (error) {
        if (stepState) {
          stepState.retryCount++;
          stepState.circuitState = circuitBreaker?.getState();
        }
        throw error;
      }
    };

    return retryPolicy.execute(executeFn);
  }

  /**
   * Compensate (rollback) completed steps
   */
  private async compensate(
    definition: WorkflowDefinition,
    state: WorkflowState,
    context: WorkflowContext
  ): Promise<string[]> {
    state.status = 'compensating';
    const errors: string[] = [];

    this.emit({
      type: 'compensation_started',
      workflowId: state.id,
      timestamp: Date.now(),
    });

    // Get completed steps in reverse order
    const completedSteps = Array.from(state.steps.entries())
      .filter(([_, s]) => s.status === 'completed')
      .reverse();

    const maxRetries = definition.maxCompensationRetries || 3;

    for (const [name, stepState] of completedSteps) {
      const stepDef = definition.steps.find((s) => s.name === name);
      if (!stepDef?.compensate) {
        continue;
      }

      stepState.status = 'compensating';

      for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
          await stepDef.compensate(
            stepState.input,
            stepState.output,
            context
          );
          stepState.status = 'compensated';
          break;
        } catch (error) {
          if (attempt === maxRetries - 1) {
            errors.push(`Compensation for "${name}" failed: ${error}`);
            stepState.status = 'failed';
            stepState.error = `Compensation failed: ${error}`;
          }
          // Wait before retry
          await this.sleep(1000 * Math.pow(2, attempt));
        }
      }
    }

    state.status = errors.length === 0 ? 'compensated' : 'failed';

    this.emit({
      type: 'compensation_completed',
      workflowId: state.id,
      data: { errors },
      timestamp: Date.now(),
    });

    return errors;
  }

  /**
   * Create a checkpoint
   */
  private createCheckpoint(
    state: WorkflowState,
    context: WorkflowContext
  ): void {
    const checkpoint: WorkflowCheckpoint = {
      id: `cp-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
      workflowId: state.id,
      completedSteps: Array.from(state.steps.entries())
        .filter(([_, s]) => s.status === 'completed')
        .map(([name]) => name),
      stepOutputs: new Map(context.stepOutputs),
      timestamp: Date.now(),
    };

    state.lastCheckpoint = checkpoint;

    let checkpoints = this.checkpoints.get(state.id);
    if (!checkpoints) {
      checkpoints = [];
      this.checkpoints.set(state.id, checkpoints);
    }
    checkpoints.push(checkpoint);

    // Keep only last 5 checkpoints
    if (checkpoints.length > 5) {
      checkpoints.shift();
    }

    this.emit({
      type: 'checkpoint_created',
      workflowId: state.id,
      data: { checkpointId: checkpoint.id },
      timestamp: Date.now(),
    });
  }

  /**
   * Find a checkpoint
   */
  private findCheckpoint(
    workflowId: string,
    checkpointId: string
  ): WorkflowCheckpoint | undefined {
    return this.checkpoints.get(workflowId)?.find((c) => c.id === checkpointId);
  }

  /**
   * Restore from checkpoint
   */
  private restoreFromCheckpoint(
    state: WorkflowState,
    context: WorkflowContext,
    checkpoint: WorkflowCheckpoint
  ): void {
    // Mark completed steps
    for (const stepName of checkpoint.completedSteps) {
      const stepState = state.steps.get(stepName);
      if (stepState) {
        stepState.status = 'completed';
      }
    }

    // Restore outputs
    for (const [key, value] of checkpoint.stepOutputs) {
      context.stepOutputs.set(key, value);
    }
  }

  /**
   * Wait for workflow to resume
   */
  private async waitForResume(state: WorkflowState): Promise<void> {
    while (state.status === 'paused') {
      await this.sleep(1000);
    }
  }

  /**
   * Get or create circuit breaker
   */
  private getOrCreateCircuitBreaker(
    happId: HappId,
    config?: Partial<CircuitBreakerConfig>
  ): CircuitBreaker {
    let breaker = this.circuitBreakers.get(happId);
    if (!breaker) {
      breaker = new CircuitBreaker({
        ...this.config.defaultCircuitBreaker,
        ...config,
        name: `happ-${happId}`,
      });
      this.circuitBreakers.set(happId, breaker);
    }
    return breaker;
  }

  /**
   * Initialize workflow state
   */
  private initWorkflowState(
    workflowId: string,
    definition: WorkflowDefinition,
    options: { initiatorDid: string; metadata?: Record<string, unknown> }
  ): WorkflowState {
    const steps = new Map<string, WorkflowStepState>();
    for (const step of definition.steps) {
      steps.set(step.name, {
        name: step.name,
        status: 'pending',
        retryCount: 0,
      });
    }

    return {
      id: workflowId,
      name: definition.name,
      version: definition.version,
      status: 'pending',
      initiatorDid: options.initiatorDid,
      steps,
      startedAt: Date.now(),
      metadata: options.metadata || {},
    };
  }

  /**
   * Create workflow context
   */
  private createContext(
    workflowId: string,
    definition: WorkflowDefinition,
    options: { initiatorDid: string; metadata?: Record<string, unknown> }
  ): WorkflowContext {
    return {
      workflowId,
      workflowName: definition.name,
      initiatorDid: options.initiatorDid,
      bridge: this.bridge,
      stepOutputs: new Map(),
      metadata: options.metadata || {},
      startedAt: Date.now(),
      log: (message, level = 'info') => {
        if (this.config.enableObservability) {
          console.log(`[${level.toUpperCase()}] [${workflowId}] ${message}`);
        }
      },
    };
  }

  /**
   * Emit event to observers
   */
  private emit(event: WorkflowStateEvent): void {
    if (!this.config.enableObservability) {
      return;
    }
    for (const observer of this.observers) {
      try {
        observer(event);
      } catch {
        // Ignore observer errors
      }
    }
  }

  /**
   * Execute with timeout
   */
  private async withTimeout<T>(
    promise: Promise<T>,
    timeoutMs: number
  ): Promise<T> {
    let timeoutHandle: ReturnType<typeof setTimeout>;

    const timeoutPromise = new Promise<never>((_, reject) => {
      timeoutHandle = setTimeout(() => {
        reject(new Error(`Operation timed out after ${timeoutMs}ms`));
      }, timeoutMs);
    });

    try {
      return await Promise.race([promise, timeoutPromise]);
    } finally {
      clearTimeout(timeoutHandle!);
    }
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Generate workflow ID
   */
  private generateWorkflowId(): string {
    return `wf-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }
}

// ============================================================================
// WORKFLOW BUILDER
// ============================================================================

/**
 * Fluent builder for creating workflows
 */
class WorkflowBuilder {
  private definition: Partial<WorkflowDefinition> = {
    steps: [],
  };

  constructor(name: string, version: string = '1.0.0') {
    this.definition.name = name;
    this.definition.version = version;
  }

  description(desc: string): this {
    this.definition.description = desc;
    return this;
  }

  timeout(ms: number): this {
    this.definition.timeout = ms;
    return this;
  }

  compensateOnFailure(enable: boolean = true): this {
    this.definition.compensateOnFailure = enable;
    return this;
  }

  enableCheckpoints(enable: boolean = true, intervalMs?: number): this {
    this.definition.enableCheckpoints = enable;
    this.definition.checkpointIntervalMs = intervalMs;
    return this;
  }

  defaultRetry(options: RetryOptions): this {
    this.definition.defaultRetry = options;
    return this;
  }

  step<TInput = unknown, TOutput = unknown>(
    name: string,
    targetHapp: HappId,
    execute: (input: TInput, context: WorkflowContext) => Promise<TOutput>,
    options?: Partial<Omit<WorkflowStepDefinition<TInput, TOutput>, 'name' | 'targetHapp' | 'execute'>>
  ): this {
    const stepDef: WorkflowStepDefinition = {
      name,
      targetHapp,
      execute: execute as WorkflowStepDefinition['execute'],
      compensate: options?.compensate as WorkflowStepDefinition['compensate'],
      retry: options?.retry,
      timeout: options?.timeout,
      critical: options?.critical,
      dependsOn: options?.dependsOn,
      useCircuitBreaker: options?.useCircuitBreaker,
      circuitBreakerConfig: options?.circuitBreakerConfig,
      description: options?.description,
    };
    this.definition.steps!.push(stepDef);
    return this;
  }

  build(): WorkflowDefinition {
    if (!this.definition.name) {
      throw new Error('Workflow name is required');
    }
    if (this.definition.steps!.length === 0) {
      throw new Error('Workflow must have at least one step');
    }
    return this.definition as WorkflowDefinition;
  }
}

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

/**
 * Create a new workflow builder
 */
function createWorkflow(name: string, version?: string): WorkflowBuilder {
  return new WorkflowBuilder(name, version);
}

/**
 * Create a new orchestrator
 */
function createOrchestrator(
  config?: Partial<OrchestratorConfig>
): SelfHealingOrchestrator {
  return new SelfHealingOrchestrator(config);
}

// Singleton instance
let orchestratorInstance: SelfHealingOrchestrator | null = null;

/**
 * Get the global orchestrator instance
 */
function getOrchestrator(
  config?: Partial<OrchestratorConfig>
): SelfHealingOrchestrator {
  if (!orchestratorInstance) {
    orchestratorInstance = new SelfHealingOrchestrator(config);
  }
  return orchestratorInstance;
}

/**
 * Reset the global orchestrator (for testing)
 */
function resetOrchestrator(): void {
  orchestratorInstance = null;
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  SelfHealingOrchestrator,
  WorkflowBuilder,
  createWorkflow,
  createOrchestrator,
  getOrchestrator,
  resetOrchestrator,
  DEFAULT_ORCHESTRATOR_CONFIG,
};
