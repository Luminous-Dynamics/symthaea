/**
 * Hearth Care SDK client.
 * Wraps zome calls to the hearth-care coordinator.
 */

import type { AppClient, Record as HolochainRecord, ActionHash } from '@holochain/client';
import type {
  CreateCareScheduleInput,
  CompleteTaskInput,
  ProposeSwapInput,
  CreateMealPlanInput,
  DigestEpochInput,
  CareSummary,
  CareTaskCompletedSignal,
} from './types';
import { HearthError, classifyError } from './errors';
import { withGateRetry } from './consciousness-gate';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_care';

const CARE_SIGNAL_TYPES: ReadonlySet<string> = new Set([
  'CareTaskCompleted',
]);

export type CareSignalHandler = (signal: CareTaskCompletedSignal) => void;

export class CareClient {
  private signalHandlers: Map<string, Set<CareSignalHandler>> = new Map();
  private listening = false;
  private refreshFn?: () => Promise<void>;

  constructor(
    private readonly client: AppClient,
    private readonly roleName = ROLE_NAME,
    refreshFn?: () => Promise<void>,
  ) {
    this.refreshFn = refreshFn;
  }

  // ============================================================================
  // Private helper
  // ============================================================================

  private async callZome<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      return await this.client.callZome({
        role_name: this.roleName,
        zome_name: ZOME_NAME,
        fn_name: fnName,
        payload,
      });
    } catch (err) {
      throw new HearthError({
        code: classifyError(err),
        message: `${ZOME_NAME}.${fnName} failed: ${err}`,
        zome: ZOME_NAME,
        fnName,
        cause: err,
      });
    }
  }

  // ============================================================================
  // Zome Calls
  // ============================================================================

  /** Create a care schedule for the hearth. */
  async createCareSchedule(input: CreateCareScheduleInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('create_care_schedule', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Mark a care task as completed. */
  async completeTask(input: CompleteTaskInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('complete_task', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Propose a care duty swap with another member. */
  async proposeSwap(input: ProposeSwapInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('propose_swap', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Accept a proposed care swap. */
  async acceptSwap(swapHash: ActionHash): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('accept_swap', swapHash);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Decline a proposed care swap. */
  async declineSwap(swapHash: ActionHash): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('decline_swap', swapHash);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Create a meal plan for the hearth. */
  async createMealPlan(input: CreateMealPlanInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('create_meal_plan', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Get the caller's current care duties. */
  async getMyCareDuties(): Promise<HolochainRecord[]> {
    return this.callZome('get_my_care_duties', null);
  }

  /** Get the full care schedule for a hearth. */
  async getHearthSchedule(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_hearth_schedule', hearthHash);
  }

  /** Get all meal plans for a hearth. */
  async getHearthMealPlans(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_hearth_meal_plans', hearthHash);
  }

  /** Create a care digest for a time epoch. */
  async createCareDigest(input: DigestEpochInput): Promise<CareSummary[]> {
    const call = () => this.callZome<CareSummary[]>('create_care_digest', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  // ============================================================================
  // Signal Handling
  // ============================================================================

  /**
   * Subscribe to care signals. Returns an unsubscribe function.
   *
   * @param handler - Callback invoked for each matching signal
   * @param signalType - Optional filter: only receive signals of this type.
   *                     Pass '*' or omit to receive all care signals.
   *
   * @example
   * ```ts
   * const unsub = client.onSignal((signal) => {
   *   console.log('Task completed!', signal.care_type);
   * });
   * // Later:
   * unsub();
   * ```
   */
  onSignal(
    handler: CareSignalHandler,
    signalType: 'CareTaskCompleted' | '*' = '*',
  ): () => void {
    this.ensureListening();

    const key = signalType;
    if (!this.signalHandlers.has(key)) {
      this.signalHandlers.set(key, new Set());
    }
    this.signalHandlers.get(key)!.add(handler);

    return () => {
      const handlers = this.signalHandlers.get(key);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          this.signalHandlers.delete(key);
        }
      }
    };
  }

  private ensureListening(): void {
    if (this.listening) return;
    this.listening = true;

    this.client.on('signal', (signal) => {
      try {
        if (signal.type !== 'app') return;
        const parsed = signal.value.payload as Record<string, unknown>;
        if (!parsed || typeof parsed !== 'object') return;

        // Rust enums serialize as { "VariantName": { fields... } }
        const variantName = Object.keys(parsed)[0];
        if (!variantName || !CARE_SIGNAL_TYPES.has(variantName)) return;

        const fields = parsed[variantName] as Record<string, unknown>;
        const typedSignal = { type: variantName, ...fields } as CareTaskCompletedSignal;

        // Notify type-specific handlers
        const typeHandlers = this.signalHandlers.get(variantName);
        if (typeHandlers) {
          typeHandlers.forEach((h) => h(typedSignal));
        }

        // Notify wildcard handlers
        const wildcardHandlers = this.signalHandlers.get('*');
        if (wildcardHandlers) {
          wildcardHandlers.forEach((h) => h(typedSignal));
        }
      } catch {
        // Ignore non-care signals
      }
    });
  }
}
