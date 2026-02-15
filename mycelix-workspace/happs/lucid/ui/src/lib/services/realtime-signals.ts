/**
 * Real-time Signal Subscriptions
 *
 * Provides real-time updates for collective sensemaking events.
 * Features:
 * - WebSocket-like event subscriptions
 * - Automatic reconnection
 * - Event batching and debouncing
 * - Local simulation mode
 */

import { writable, derived, get, type Writable, type Readable } from 'svelte/store';
import type { BeliefShare, ValidationVote, ConsensusRecord, EmergentPattern } from './collective-sensemaking';
import { ValidationVoteType, ConsensusType } from './collective-sensemaking';

// ============================================================================
// TYPES
// ============================================================================

export type SignalType =
  | 'belief_shared'
  | 'vote_cast'
  | 'consensus_changed'
  | 'pattern_detected'
  | 'relationship_updated'
  | 'agent_online'
  | 'agent_offline';

export interface Signal<T = unknown> {
  type: SignalType;
  payload: T;
  timestamp: number;
  source: string;
}

export interface BeliefSharedSignal extends Signal<BeliefShare> {
  type: 'belief_shared';
}

export interface VoteCastSignal extends Signal<ValidationVote> {
  type: 'vote_cast';
}

export interface ConsensusChangedSignal extends Signal<ConsensusRecord> {
  type: 'consensus_changed';
}

export interface PatternDetectedSignal extends Signal<EmergentPattern> {
  type: 'pattern_detected';
}

export interface AgentStatusSignal extends Signal<{ agentId: string; status: 'online' | 'offline' }> {
  type: 'agent_online' | 'agent_offline';
}

export type AnySignal =
  | BeliefSharedSignal
  | VoteCastSignal
  | ConsensusChangedSignal
  | PatternDetectedSignal
  | AgentStatusSignal
  | Signal;

export interface SubscriptionOptions {
  filter?: (signal: AnySignal) => boolean;
  debounceMs?: number;
  batchSize?: number;
}

type Unsubscribe = () => void;

// ============================================================================
// SIGNAL STORES
// ============================================================================

// Connection state
export const signalConnectionStatus = writable<'disconnected' | 'connecting' | 'connected' | 'simulated'>('disconnected');

// Signal streams
export const beliefSignals = writable<BeliefSharedSignal[]>([]);
export const voteSignals = writable<VoteCastSignal[]>([]);
export const consensusSignals = writable<ConsensusChangedSignal[]>([]);
export const patternSignals = writable<PatternDetectedSignal[]>([]);
export const agentStatusSignals = writable<AgentStatusSignal[]>([]);

// All signals combined
export const allSignals: Readable<AnySignal[]> = derived(
  [beliefSignals, voteSignals, consensusSignals, patternSignals, agentStatusSignals],
  ([$beliefs, $votes, $consensus, $patterns, $agents]) => {
    return [...$beliefs, ...$votes, ...$consensus, ...$patterns, ...$agents].sort(
      (a, b) => b.timestamp - a.timestamp
    );
  }
);

// Unread signal count
export const unreadSignalCount = writable(0);

// ============================================================================
// SIGNAL SERVICE
// ============================================================================

class SignalService {
  private subscribers = new Map<SignalType | '*', Set<(signal: AnySignal) => void>>();
  private simulationInterval: number | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor() {
    this.initializeSubscribers();
  }

  private initializeSubscribers(): void {
    const types: (SignalType | '*')[] = [
      '*',
      'belief_shared',
      'vote_cast',
      'consensus_changed',
      'pattern_detected',
      'relationship_updated',
      'agent_online',
      'agent_offline',
    ];
    types.forEach((type) => this.subscribers.set(type, new Set()));
  }

  /**
   * Connect to real-time signal source
   */
  async connect(): Promise<void> {
    signalConnectionStatus.set('connecting');

    try {
      // In a real implementation, this would connect to Holochain's signal system
      // For now, we'll use simulation mode
      await this.startSimulation();
      signalConnectionStatus.set('simulated');
    } catch (error) {
      console.error('Failed to connect to signal service:', error);
      signalConnectionStatus.set('disconnected');
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from signal source
   */
  disconnect(): void {
    this.stopSimulation();
    signalConnectionStatus.set('disconnected');
    this.reconnectAttempts = 0;
  }

  /**
   * Subscribe to signals
   */
  subscribe(type: SignalType | '*', callback: (signal: AnySignal) => void, options?: SubscriptionOptions): Unsubscribe {
    const subscribers = this.subscribers.get(type);
    if (!subscribers) return () => {};

    let handler = callback;

    // Apply debouncing if requested
    if (options?.debounceMs) {
      let timeout: number | null = null;
      let pendingSignals: AnySignal[] = [];

      handler = (signal: AnySignal) => {
        pendingSignals.push(signal);

        if (timeout) clearTimeout(timeout);
        timeout = window.setTimeout(() => {
          const signals = pendingSignals;
          pendingSignals = [];
          signals.forEach(callback);
        }, options.debounceMs);
      };
    }

    // Apply filtering if requested
    if (options?.filter) {
      const originalHandler = handler;
      handler = (signal: AnySignal) => {
        if (options.filter!(signal)) {
          originalHandler(signal);
        }
      };
    }

    subscribers.add(handler);

    return () => {
      subscribers.delete(handler);
    };
  }

  /**
   * Emit a signal (for local/testing use)
   */
  emit(signal: AnySignal): void {
    // Add to appropriate store
    switch (signal.type) {
      case 'belief_shared':
        beliefSignals.update((s) => [...s.slice(-99), signal as BeliefSharedSignal]);
        break;
      case 'vote_cast':
        voteSignals.update((s) => [...s.slice(-99), signal as VoteCastSignal]);
        break;
      case 'consensus_changed':
        consensusSignals.update((s) => [...s.slice(-99), signal as ConsensusChangedSignal]);
        break;
      case 'pattern_detected':
        patternSignals.update((s) => [...s.slice(-99), signal as PatternDetectedSignal]);
        break;
      case 'agent_online':
      case 'agent_offline':
        agentStatusSignals.update((s) => [...s.slice(-99), signal as AgentStatusSignal]);
        break;
    }

    // Notify subscribers
    this.notifySubscribers(signal);
    unreadSignalCount.update((n) => n + 1);
  }

  private notifySubscribers(signal: AnySignal): void {
    // Notify type-specific subscribers
    const typeSubscribers = this.subscribers.get(signal.type);
    typeSubscribers?.forEach((callback) => callback(signal));

    // Notify wildcard subscribers
    const wildcardSubscribers = this.subscribers.get('*');
    wildcardSubscribers?.forEach((callback) => callback(signal));
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.warn('Max reconnection attempts reached');
      return;
    }

    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
    this.reconnectAttempts++;

    setTimeout(() => this.connect(), delay);
  }

  // ============================================================================
  // SIMULATION MODE
  // ============================================================================

  private async startSimulation(): Promise<void> {
    // Generate simulated signals periodically
    this.simulationInterval = window.setInterval(() => {
      this.generateSimulatedSignal();
    }, 5000 + Math.random() * 10000);

    // Generate initial signals
    for (let i = 0; i < 3; i++) {
      setTimeout(() => this.generateSimulatedSignal(), i * 1000);
    }
  }

  private stopSimulation(): void {
    if (this.simulationInterval !== null) {
      clearInterval(this.simulationInterval);
      this.simulationInterval = null;
    }
  }

  private generateSimulatedSignal(): void {
    const types: SignalType[] = ['belief_shared', 'vote_cast', 'consensus_changed', 'pattern_detected'];
    const type = types[Math.floor(Math.random() * types.length)];
    const timestamp = Date.now() * 1000;
    const source = `agent-${Math.random().toString(36).substr(2, 8)}`;

    let signal: AnySignal;

    switch (type) {
      case 'belief_shared':
        signal = {
          type: 'belief_shared',
          timestamp,
          source,
          payload: {
            content_hash: `hash-${Math.random().toString(36).substr(2, 12)}`,
            thought: {
              id: `thought-${Date.now()}`,
              content: this.getRandomBeliefContent(),
              thought_type: 'Claim' as any,
              confidence: 0.7 + Math.random() * 0.3,
              tags: ['simulated'],
              epistemic: { empirical: 'E2', normative: 'N1', materiality: 'M2', harmonic: 'H1' },
              domain: 'general',
              related_thoughts: [],
              source_hashes: [],
              parent_thought: null,
              created_at: timestamp,
              updated_at: timestamp,
              version: 1,
            },
            epistemic_code: 'E2N1M2H1',
            tags: ['simulated', 'test'],
            shared_at: timestamp,
            evidence_hashes: [],
            embedding: [],
            stance: null,
          } as unknown as BeliefShare,
        };
        break;

      case 'vote_cast':
        signal = {
          type: 'vote_cast',
          timestamp,
          source,
          payload: {
            belief_share_hash: new Uint8Array(39) as any,
            vote_type: [ValidationVoteType.Corroborate, ValidationVoteType.Plausible, ValidationVoteType.Abstain][
              Math.floor(Math.random() * 3)
            ],
            evidence: null,
            voter_weight: 0.8 + Math.random() * 0.4,
            voted_at: timestamp,
          } as ValidationVote,
        };
        break;

      case 'consensus_changed':
        signal = {
          type: 'consensus_changed',
          timestamp,
          source,
          payload: {
            belief_share_hash: new Uint8Array(39) as any,
            consensus_type: ConsensusType.ModerateConsensus,
            validator_count: 5 + Math.floor(Math.random() * 10),
            agreement_score: 0.6 + Math.random() * 0.3,
            summary: 'Consensus updated',
            reached_at: timestamp,
          } as ConsensusRecord,
        };
        break;

      case 'pattern_detected':
        signal = {
          type: 'pattern_detected',
          timestamp,
          source,
          payload: {
            pattern_id: `pattern-${Date.now()}`,
            pattern_type: 'Convergence' as any,
            description: 'New pattern detected in collective beliefs',
            belief_hashes: [],
            confidence: 0.7 + Math.random() * 0.3,
            detected_at: timestamp,
          } as EmergentPattern,
        };
        break;

      default:
        return;
    }

    this.emit(signal);
  }

  private getRandomBeliefContent(): string {
    const beliefs = [
      'Distributed systems improve resilience through redundancy',
      'Zero-knowledge proofs enable privacy-preserving verification',
      'Collective intelligence emerges from diverse perspectives',
      'Reputation systems can incentivize honest behavior',
      'Consensus mechanisms balance decentralization and efficiency',
    ];
    return beliefs[Math.floor(Math.random() * beliefs.length)];
  }

  /**
   * Mark all signals as read
   */
  markAllRead(): void {
    unreadSignalCount.set(0);
  }

  /**
   * Clear all signals
   */
  clearAll(): void {
    beliefSignals.set([]);
    voteSignals.set([]);
    consensusSignals.set([]);
    patternSignals.set([]);
    agentStatusSignals.set([]);
    unreadSignalCount.set(0);
  }
}

// ============================================================================
// SINGLETON EXPORT
// ============================================================================

export const signalService = new SignalService();

// Convenience functions
export const connectSignals = signalService.connect.bind(signalService);
export const disconnectSignals = signalService.disconnect.bind(signalService);
export const subscribeToSignals = signalService.subscribe.bind(signalService);
export const emitSignal = signalService.emit.bind(signalService);
export const markSignalsRead = signalService.markAllRead.bind(signalService);
export const clearSignals = signalService.clearAll.bind(signalService);
