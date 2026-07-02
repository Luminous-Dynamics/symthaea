// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Hearth Stories SDK client.
 * Wraps zome calls to the hearth-stories coordinator.
 */

import type { AppClient, Record as HolochainRecord, ActionHash } from '@holochain/client';
import type {
  CreateStoryInput,
  UpdateStoryInput,
  AddMediaInput,
  CreateCollectionInput,
  AddToCollectionInput,
  CreateTraditionInput,
  StorySignal,
  StorySignalType,
} from './types';
import { HearthError, classifyError } from './errors';
import { withGateRetry } from './consciousness-gate';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_stories';

const STORY_SIGNAL_TYPES: ReadonlySet<string> = new Set([
  'StoryCreated',
  'StoryUpdated',
  'TraditionObserved',
]);

export type StorySignalHandler = (signal: StorySignal) => void;

export class StoriesClient {
  private signalHandlers: Map<string, Set<StorySignalHandler>> = new Map();
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
  // Zome Calls
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

  /** Create a new family story. */
  async createStory(input: CreateStoryInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('create_story', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Update an existing family story. */
  async updateStory(input: UpdateStoryInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('update_story', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Add media (photo, video, audio) to a story. */
  async addMediaToStory(input: AddMediaInput): Promise<void> {
    const call = () => this.callZome<void>('add_media_to_story', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Create a new story collection. */
  async createCollection(input: CreateCollectionInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('create_collection', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Add a story to an existing collection. */
  async addToCollection(input: AddToCollectionInput): Promise<void> {
    const call = () => this.callZome<void>('add_to_collection', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Create a new family tradition. */
  async createTradition(input: CreateTraditionInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('create_tradition', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Record an observation of a family tradition. */
  async observeTradition(traditionHash: ActionHash): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('observe_tradition', traditionHash);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Get all stories for a hearth. */
  async getHearthStories(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_hearth_stories', hearthHash);
  }

  /** Get all traditions for a hearth. */
  async getHearthTraditions(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_hearth_traditions', hearthHash);
  }

  /** Get all story collections for a hearth. */
  async getHearthCollections(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_hearth_collections', hearthHash);
  }

  /** Search stories by tag. */
  async searchStoriesByTag(tag: string): Promise<HolochainRecord[]> {
    return this.callZome('search_stories_by_tag', tag);
  }

  // ============================================================================
  // Signal Handling
  // ============================================================================

  /**
   * Subscribe to story signals. Returns an unsubscribe function.
   *
   * @param handler - Callback invoked for each matching signal
   * @param signalType - Optional filter: only receive signals of this type.
   *                     Pass '*' or omit to receive all story signals.
   *
   * @example
   * ```ts
   * const unsub = client.onSignal((signal) => {
   *   if (signal.type === 'StoryCreated') console.log('New story!', signal.story_hash);
   * });
   * // Later:
   * unsub();
   * ```
   */
  onSignal(
    handler: StorySignalHandler,
    signalType: StorySignalType | '*' = '*',
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
        if (!variantName || !STORY_SIGNAL_TYPES.has(variantName)) return;

        const fields = parsed[variantName] as Record<string, unknown>;
        const typedSignal = { type: variantName, ...fields } as StorySignal;

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
        // Ignore non-story signals
      }
    });
  }
}
