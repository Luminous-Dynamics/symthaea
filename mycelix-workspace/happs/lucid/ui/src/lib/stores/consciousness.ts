/**
 * Consciousness Store
 *
 * Reactive store for consciousness state from LucidMind (Symthaea's ContinuousMind wrapper).
 * Provides:
 * - Real-time phi/meta-awareness tracking
 * - Consciousness evolution history
 * - Entrenchment detection
 * - Past self dialogue support
 */

import { writable, derived, get } from 'svelte/store';
import { invoke } from '@tauri-apps/api/core';
import { lucidClient } from './holochain';
import type {
  BeliefTrajectoryRecord,
  TrajectoryAnalysis,
  ConsciousnessEvolution,
  RecordSnapshotInput,
  RecordEvolutionInput,
  SnapshotTrigger as SnapshotTriggerType,
} from '@mycelix/lucid-client';

// ============================================================================
// TYPES
// ============================================================================

export interface ConsciousnessProfile {
  phi: number;
  meta_awareness: number;
  cognitive_load: number;
  emotional_valence: number;
  arousal: number;
}

export interface MindState {
  tick: number;
  is_active: boolean;
  is_conscious: boolean;
  consciousness_level: number;
  meta_awareness: number;
  cognitive_load: number;
  emotional_valence: number;
  arousal: number;
  memory_utilization: number;
  time_awake_ms: number;
  working_memory_size: number;
  session_memory_size: number;
  thoughts_analyzed: number;
  is_seeded: boolean;
}

export interface ConsciousnessSnapshot {
  timestamp: number;
  tick: number;
  phi: number;
  meta_awareness: number;
  cognitive_load: number;
  emotional_valence: number;
  arousal: number;
  thoughts_analyzed: number;
  working_memory_size: number;
}

export interface EvolutionSummary {
  session_duration_secs: number;
  total_thoughts_analyzed: number;
  current_phi: number;
  current_meta_awareness: number;
  current_cognitive_load: number;
  phi_trend: string;
  cognitive_health: string;
  memory_utilization: number;
  recommendations: string[];
}

export interface EntrenchmentWarning {
  type: 'phi_stagnation' | 'echo_chamber' | 'cognitive_rigidity' | 'confirmation_bias';
  severity: number; // 0-1
  message: string;
  suggestion: string;
  detectedAt: number;
}

export interface PastSelfMessage {
  timestamp: number;
  phi: number;
  message: string;
  emotionalContext: string;
}

// ============================================================================
// STORES
// ============================================================================

// Core consciousness state
export const consciousnessProfile = writable<ConsciousnessProfile | null>(null);
export const mindState = writable<MindState | null>(null);
export const isConnected = writable(false);
export const lastUpdated = writable<number>(0);

// History for tracking evolution
export const consciousnessHistory = writable<ConsciousnessSnapshot[]>([]);
const MAX_HISTORY_SIZE = 100;

// Entrenchment warnings
export const entrenchmentWarnings = writable<EntrenchmentWarning[]>([]);

// Past self dialogue
export const pastSelfMessages = writable<PastSelfMessage[]>([]);

// Holochain-persisted trajectory and evolution data
export const evolutionHistory = writable<ConsciousnessEvolution[]>([]);
export const trajectoryCache = writable<Map<string, TrajectoryAnalysis>>(new Map());

// Snapshot persistence counter (persist every Nth snapshot to reduce DHT writes)
let snapshotCounter = 0;
const PERSIST_EVERY_N_SNAPSHOTS = 5;

// Polling state
let pollInterval: number | null = null;
let isPolling = false;

// ============================================================================
// DERIVED STORES
// ============================================================================

// Current phi level with human-readable state
export const consciousnessLevel = derived(consciousnessProfile, ($profile) => {
  if (!$profile) return { value: 0, label: 'Disconnected', color: '#6b7280' };

  const phi = $profile.phi;
  if (phi >= 0.8) return { value: phi, label: 'Highly Integrated', color: '#22c55e' };
  if (phi >= 0.6) return { value: phi, label: 'Conscious', color: '#3b82f6' };
  if (phi >= 0.4) return { value: phi, label: 'Aware', color: '#f59e0b' };
  if (phi >= 0.2) return { value: phi, label: 'Minimal', color: '#9ca3af' };
  return { value: phi, label: 'Dormant', color: '#6b7280' };
});

// Phi trend based on history
export const phiTrend = derived(consciousnessHistory, ($history) => {
  if ($history.length < 5) return 'stable';

  const recent = $history.slice(-5);
  const older = $history.slice(-10, -5);

  if (older.length === 0) return 'stable';

  const recentAvg = recent.reduce((sum, s) => sum + s.phi, 0) / recent.length;
  const olderAvg = older.reduce((sum, s) => sum + s.phi, 0) / older.length;

  const diff = recentAvg - olderAvg;
  if (diff > 0.05) return 'increasing';
  if (diff < -0.05) return 'decreasing';
  return 'stable';
});

// Active warnings count
export const activeWarningsCount = derived(entrenchmentWarnings, ($warnings) => {
  const now = Date.now();
  const oneHourAgo = now - 60 * 60 * 1000;
  return $warnings.filter((w) => w.detectedAt > oneHourAgo).length;
});

// ============================================================================
// ACTIONS
// ============================================================================

/**
 * Fetch consciousness profile from LucidMind
 */
export async function fetchProfile(): Promise<ConsciousnessProfile | null> {
  try {
    const profile = await invoke<ConsciousnessProfile>('get_mind_consciousness_profile');
    consciousnessProfile.set(profile);
    isConnected.set(true);
    lastUpdated.set(Date.now());
    return profile;
  } catch {
    consciousnessProfile.set(null);
    return null;
  }
}

/**
 * Fetch mind state from LucidMind
 */
export async function fetchMindState(): Promise<MindState | null> {
  try {
    const state = await invoke<MindState>('get_mind_state');
    mindState.set(state);
    isConnected.set(true);
    lastUpdated.set(Date.now());
    return state;
  } catch {
    mindState.set(null);
    return null;
  }
}

/**
 * Take a consciousness snapshot and add to history.
 * Periodically persists to Holochain temporal-consciousness zome.
 */
export async function takeSnapshot(): Promise<ConsciousnessSnapshot | null> {
  try {
    const snapshot = await invoke<ConsciousnessSnapshot>('take_consciousness_snapshot');

    consciousnessHistory.update((history) => {
      const newHistory = [...history, snapshot];
      if (newHistory.length > MAX_HISTORY_SIZE) {
        newHistory.shift();
      }
      return newHistory;
    });

    // Check for entrenchment patterns
    checkEntrenchment(snapshot);

    // Persist to Holochain every Nth snapshot to reduce DHT writes
    snapshotCounter++;
    if (snapshotCounter >= PERSIST_EVERY_N_SNAPSHOTS) {
      snapshotCounter = 0;
      persistSnapshotToHolochain(snapshot).catch(() => {
        // Silently fail — Holochain persistence is best-effort
      });
    }

    return snapshot;
  } catch {
    return null;
  }
}

/**
 * Persist a consciousness snapshot to Holochain's temporal-consciousness zome.
 * Maps the Symthaea snapshot format to the Holochain RecordSnapshotInput.
 */
async function persistSnapshotToHolochain(snapshot: ConsciousnessSnapshot): Promise<void> {
  const client = get(lucidClient);
  if (!client) return;

  const input: RecordSnapshotInput = {
    thought_id: `session_${snapshot.tick}`,
    epistemic_code: `phi:${snapshot.phi.toFixed(2)}`,
    confidence: snapshot.meta_awareness,
    phi: snapshot.phi,
    coherence: 1.0 - snapshot.cognitive_load, // Inverse of load as coherence proxy
    trigger: 'Scheduled' as SnapshotTriggerType,
  };

  await client.temporalConsciousness.recordBeliefSnapshot(input);
}

/**
 * Fetch trajectory analysis for a thought from Holochain
 */
export async function fetchTrajectoryAnalysis(thoughtId: string): Promise<TrajectoryAnalysis | null> {
  const client = get(lucidClient);
  if (!client) return null;

  try {
    const analysis = await client.temporalConsciousness.analyzeBeliefTrajectory(thoughtId);

    // Cache it
    trajectoryCache.update((cache) => {
      const newCache = new Map(cache);
      newCache.set(thoughtId, analysis);
      return newCache;
    });

    return analysis;
  } catch {
    return null;
  }
}

/**
 * Fetch consciousness evolution history from Holochain
 */
export async function fetchEvolutionHistory(): Promise<ConsciousnessEvolution[]> {
  const client = get(lucidClient);
  if (!client) return [];

  try {
    const history = await client.temporalConsciousness.getMyEvolutionHistory();
    evolutionHistory.set(history);
    return history;
  } catch {
    return [];
  }
}

/**
 * Record a consciousness evolution summary to Holochain.
 * Called at the end of a session or periodically.
 */
export async function recordEvolutionSummary(): Promise<void> {
  const client = get(lucidClient);
  if (!client) return;

  const history = get(consciousnessHistory);
  if (history.length < 5) return;

  const firstSnapshot = history[0];
  const lastSnapshot = history[history.length - 1];

  // Calculate trends
  const phiValues = history.map((s) => s.phi);
  const avgPhi = phiValues.reduce((a, b) => a + b, 0) / phiValues.length;
  const phiTrend = phiValues.length >= 2
    ? (phiValues[phiValues.length - 1] - phiValues[0]) / phiValues.length
    : 0;

  const cogLoadValues = history.map((s) => s.cognitive_load);
  const avgCoherence = 1 - cogLoadValues.reduce((a, b) => a + b, 0) / cogLoadValues.length;
  const coherenceTrend = cogLoadValues.length >= 2
    ? (cogLoadValues[0] - cogLoadValues[cogLoadValues.length - 1]) / cogLoadValues.length
    : 0;

  // Gather insights from warnings
  const warnings = get(entrenchmentWarnings);
  const insights = warnings.map((w) => w.message);

  const input: RecordEvolutionInput = {
    period_start: firstSnapshot.timestamp * 1000, // to microseconds
    period_end: lastSnapshot.timestamp * 1000,
    avg_phi: avgPhi,
    phi_trend: Math.max(-1, Math.min(1, phiTrend)),
    avg_coherence: avgCoherence,
    coherence_trend: Math.max(-1, Math.min(1, coherenceTrend)),
    stable_belief_count: history.filter((s) => Math.abs(s.phi - avgPhi) < 0.05).length,
    growing_belief_count: history.filter((_, i, arr) =>
      i > 0 && arr[i].phi > arr[i - 1].phi + 0.02
    ).length,
    weakening_belief_count: history.filter((_, i, arr) =>
      i > 0 && arr[i].phi < arr[i - 1].phi - 0.02
    ).length,
    entrenched_belief_count: warnings.filter((w) => w.type === 'phi_stagnation').length,
    insights,
  };

  try {
    await client.temporalConsciousness.recordConsciousnessEvolution(input);
  } catch {
    // Best-effort persistence
  }
}

/**
 * Get evolution summary
 */
export async function getEvolutionSummary(): Promise<EvolutionSummary | null> {
  try {
    return await invoke<EvolutionSummary>('get_consciousness_evolution_summary');
  } catch {
    return null;
  }
}

/**
 * Start polling consciousness state
 */
export function startPolling(intervalMs: number = 5000): void {
  if (isPolling) return;
  isPolling = true;

  // Load evolution history from Holochain on start
  fetchEvolutionHistory().catch(() => {});

  const poll = async () => {
    await Promise.all([fetchProfile(), fetchMindState()]);

    // Take snapshot every 6th poll (30 seconds if polling at 5s)
    const history = get(consciousnessHistory);
    const lastSnapshot = history[history.length - 1];
    if (!lastSnapshot || Date.now() - lastSnapshot.timestamp > 30000) {
      await takeSnapshot();
    }
  };

  poll();
  pollInterval = setInterval(poll, intervalMs) as unknown as number;
}

/**
 * Stop polling. Records an evolution summary to Holochain before stopping.
 */
export function stopPolling(): void {
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
  isPolling = false;

  // Record evolution summary to Holochain on session end
  recordEvolutionSummary().catch(() => {});
}

// ============================================================================
// ENTRENCHMENT DETECTION
// ============================================================================

/**
 * Check for entrenchment patterns in consciousness evolution
 */
function checkEntrenchment(snapshot: ConsciousnessSnapshot): void {
  const history = get(consciousnessHistory);
  if (history.length < 10) return;

  const warnings: EntrenchmentWarning[] = [];
  const now = Date.now();

  // Check 1: Phi stagnation (phi hasn't changed significantly in last 10 snapshots)
  const recentSnapshots = history.slice(-10);
  const phiValues = recentSnapshots.map((s) => s.phi);
  const phiVariance = calculateVariance(phiValues);

  if (phiVariance < 0.001 && snapshot.thoughts_analyzed > 20) {
    warnings.push({
      type: 'phi_stagnation',
      severity: 0.6,
      message: 'Your thinking patterns have become stable - consider exploring new perspectives.',
      suggestion: 'Try challenging an assumption or exploring a topic from a different angle.',
      detectedAt: now,
    });
  }

  // Check 2: Cognitive rigidity (low meta-awareness with high phi)
  if (snapshot.meta_awareness < 0.3 && snapshot.phi > 0.6) {
    warnings.push({
      type: 'cognitive_rigidity',
      severity: 0.5,
      message: 'High integration with low self-reflection may indicate cognitive rigidity.',
      suggestion: 'Take a moment to question your reasoning or seek alternative viewpoints.',
      detectedAt: now,
    });
  }

  // Check 3: Echo chamber (increasing phi without external input)
  const oldestRecent = recentSnapshots[0];
  const phiChange = snapshot.phi - oldestRecent.phi;
  const thoughtsChange = snapshot.thoughts_analyzed - oldestRecent.thoughts_analyzed;

  if (phiChange > 0.1 && thoughtsChange < 3 && snapshot.working_memory_size > 20) {
    warnings.push({
      type: 'echo_chamber',
      severity: 0.4,
      message: 'Integration is increasing without new information - possible echo chamber effect.',
      suggestion: 'Consider incorporating external sources or opposing viewpoints.',
      detectedAt: now,
    });
  }

  // Add new warnings (avoiding duplicates within last hour)
  if (warnings.length > 0) {
    entrenchmentWarnings.update((existing) => {
      const oneHourAgo = now - 60 * 60 * 1000;
      const recentExisting = existing.filter((w) => w.detectedAt > oneHourAgo);

      const newWarnings = warnings.filter(
        (w) => !recentExisting.some((e) => e.type === w.type)
      );

      return [...recentExisting, ...newWarnings];
    });
  }
}

function calculateVariance(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const squaredDiffs = values.map((v) => Math.pow(v - mean, 2));
  return squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
}

// ============================================================================
// PAST SELF DIALOGUE
// ============================================================================

/**
 * Generate a message from your past self based on consciousness history
 */
export function generatePastSelfMessage(): PastSelfMessage | null {
  const history = get(consciousnessHistory);
  if (history.length < 5) return null;

  // Find a snapshot from earlier with notably different consciousness state
  const currentSnapshot = history[history.length - 1];
  const targetTimeAgo = Math.floor(history.length / 2); // Halfway through session

  const pastSnapshot = history[Math.max(0, history.length - 1 - targetTimeAgo)];
  if (!pastSnapshot) return null;

  const phiDiff = currentSnapshot.phi - pastSnapshot.phi;
  const metaDiff = currentSnapshot.meta_awareness - pastSnapshot.meta_awareness;

  // Generate message based on differences
  let message: string;
  let emotionalContext: string;

  if (phiDiff > 0.1) {
    message =
      'I notice you\'ve become more integrated in your thinking. What insight led to this growth?';
    emotionalContext = 'curious';
  } else if (phiDiff < -0.1) {
    message =
      'Your integration seems lower than before. Are you exploring new territory or feeling uncertain?';
    emotionalContext = 'concerned';
  } else if (metaDiff > 0.1) {
    message =
      'You\'re more self-aware now. What sparked this increased meta-awareness?';
    emotionalContext = 'proud';
  } else if (metaDiff < -0.1) {
    message =
      'You seem more immersed in your thoughts. Remember to step back occasionally.';
    emotionalContext = 'gentle';
  } else {
    message =
      'Your consciousness state has been relatively stable. Are you consolidating or in a holding pattern?';
    emotionalContext = 'neutral';
  }

  const pastMessage: PastSelfMessage = {
    timestamp: pastSnapshot.timestamp,
    phi: pastSnapshot.phi,
    message,
    emotionalContext,
  };

  pastSelfMessages.update((messages) => {
    const newMessages = [...messages, pastMessage];
    if (newMessages.length > 10) {
      newMessages.shift();
    }
    return newMessages;
  });

  return pastMessage;
}

/**
 * Clear all past self messages
 */
export function clearPastSelfMessages(): void {
  pastSelfMessages.set([]);
}

/**
 * Dismiss an entrenchment warning
 */
export function dismissWarning(type: EntrenchmentWarning['type']): void {
  entrenchmentWarnings.update((warnings) =>
    warnings.filter((w) => w.type !== type)
  );
}

// ============================================================================
// CLEANUP
// ============================================================================

/**
 * Reset all consciousness state
 */
export function resetConsciousnessState(): void {
  stopPolling();
  consciousnessProfile.set(null);
  mindState.set(null);
  consciousnessHistory.set([]);
  entrenchmentWarnings.set([]);
  pastSelfMessages.set([]);
  evolutionHistory.set([]);
  trajectoryCache.set(new Map());
  isConnected.set(false);
  snapshotCounter = 0;
}
