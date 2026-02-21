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

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_care';

const CARE_SIGNAL_TYPES: ReadonlySet<string> = new Set([
  'CareTaskCompleted',
]);

export type CareSignalHandler = (signal: CareTaskCompletedSignal) => void;

export class CareClient {
  private signalHandlers: Map<string, Set<CareSignalHandler>> = new Map();
  private listening = false;

  constructor(private readonly client: AppClient, private readonly roleName = ROLE_NAME) {}

  // ============================================================================
  // Zome Calls
  // ============================================================================

  async createCareSchedule(input: CreateCareScheduleInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_care_schedule',
      payload: input,
    });
  }

  async completeTask(input: CompleteTaskInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'complete_task',
      payload: input,
    });
  }

  async proposeSwap(input: ProposeSwapInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'propose_swap',
      payload: input,
    });
  }

  async acceptSwap(swapHash: ActionHash): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'accept_swap',
      payload: swapHash,
    });
  }

  async declineSwap(swapHash: ActionHash): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'decline_swap',
      payload: swapHash,
    });
  }

  async createMealPlan(input: CreateMealPlanInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_meal_plan',
      payload: input,
    });
  }

  async getMyCareDuties(): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_my_care_duties',
      payload: null,
    });
  }

  async getHearthSchedule(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_hearth_schedule',
      payload: hearthHash,
    });
  }

  async getHearthMealPlans(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_hearth_meal_plans',
      payload: hearthHash,
    });
  }

  async createCareDigest(input: DigestEpochInput): Promise<CareSummary[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_care_digest',
      payload: input,
    });
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
        const parsed = signal.payload as Record<string, unknown>;
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
