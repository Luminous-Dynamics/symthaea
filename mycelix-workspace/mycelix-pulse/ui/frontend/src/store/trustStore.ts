import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Email } from '@/types';

export type TrustTier = 'high' | 'medium' | 'low' | 'unknown';

export interface TrustSummary {
  score?: number;
  tier: TrustTier;
  reasons: string[];
  pathLength?: number;
  decayAt?: string;
  quarantined?: boolean;
  fetchedAt?: string;
}

export type TrustPolicy = 'strict' | 'balanced' | 'open';

interface Thresholds {
  high: number;
  low: number;
}

const MAX_SUMMARIES = 200;

interface TrustStore {
  enabled: boolean;
  quarantineEnabled: boolean;
  hideLowTrust: boolean;
  thresholds: Thresholds;
  policy: TrustPolicy;
  summaries: Record<string, TrustSummary>;
  overrides: Record<string, TrustSummary>;
  cacheTimes: Record<string, number>;
  ttlMs: number;
  notificationsEnabled: boolean;
  getOverrideCount: () => number;
  getAttestationCount: () => number;
  enableTrust: (enabled: boolean) => void;
  enableQuarantine: (enabled: boolean) => void;
  setHideLowTrust: (enabled: boolean) => void;
  enableNotifications: (enabled: boolean) => void;
  setThresholds: (thresholds: Partial<Thresholds>) => void;
  setTtlMs: (ttl: number) => void;
  setPolicy: (policy: TrustPolicy) => void;
  ingestSummary: (key: string, summary: TrustSummary) => void;
  hasSummary: (key: string) => boolean;
  getSummary: (key: string) => TrustSummary | undefined;
  getCacheTime: (key: string) => number | undefined;
  setOverride: (key: string, summary: TrustSummary) => void;
  clearOverride: (key: string) => void;
  hasOverride: (key: string) => boolean;
  clearSummaries: () => void;
  evaluateTrust: (email: Email) => TrustSummary;
  shouldQuarantine: (email: Email) => boolean;
}

const clamp = (value: number, min = 0, max = 100) => Math.min(Math.max(value, min), max);

const normalizeThresholds = (incoming: Thresholds): Thresholds => {
  const high = clamp(incoming.high);
  // Ensure low is at least 1 point below high to avoid a dead zone; clamp to [0, high - 1]
  const low = clamp(Math.min(incoming.low, Math.max(0, high - 1)));
  return { high, low };
};

const deriveScoreFromEmail = (email: Email): { score?: number; reasons: string[] } => {
  // Prefer server-provided score/reasons when available
  if (typeof email.trustScore === 'number' || (email.trustReasons && email.trustReasons.length > 0)) {
    return { score: email.trustScore, reasons: email.trustReasons || [] };
  }

  // Lightweight heuristic fallback so the UI can still render trust context
  let score = 55;
  const reasons: string[] = [];

  if (!email.from.name) {
    score -= 10;
    reasons.push('No display name');
  }

  if (email.attachments && email.attachments.length > 0) {
    score -= 5;
    reasons.push('Has attachments');
  }

  const subject = email.subject.toLowerCase();
  if (subject.includes('winner') || subject.includes('prize') || subject.includes('urgent')) {
    score -= 15;
    reasons.push('Suspicious subject language');
  }

  if (email.from.address && email.from.address.includes('+')) {
    score -= 5;
    reasons.push('Tagged sender address');
  }

  return { score: clamp(score), reasons };
};

const tierFromScore = (score: number | undefined, thresholds: Thresholds): TrustTier => {
  if (typeof score !== 'number') return 'unknown';
  if (score >= thresholds.high) return 'high';
  if (score > thresholds.low) return 'medium';
  return 'low';
};

export const useTrustStore = create<TrustStore>()(
  persist(
    (set, get) => ({
      enabled: true,
      quarantineEnabled: true,
      hideLowTrust: false,
      policy: 'balanced',
      thresholds: {
        high: 70,
        low: 30,
      },
      summaries: {},
      overrides: {},
      cacheTimes: {},
      ttlMs: 1000 * 60 * 60, // 1 hour default
      notificationsEnabled: true,

      enableTrust: (enabled) => set({ enabled }),

      enableQuarantine: (enabled) => set({ quarantineEnabled: enabled }),

      setHideLowTrust: (enabled) => set({ hideLowTrust: enabled }),

      enableNotifications: (enabled) => set({ notificationsEnabled: enabled }),

      setThresholds: (thresholds) =>
        set((state) => ({
          thresholds: {
            ...state.thresholds,
            ...normalizeThresholds({ ...state.thresholds, ...thresholds }),
          },
        })),

      setTtlMs: (ttl) => set({ ttlMs: Math.max(1000 * 60 * 5, ttl) }), // minimum 5 minutes

      setPolicy: (policy) => set({ policy }),

      ingestSummary: (key, summary) =>
        set((state) => {
          const existing = state.summaries[key];
          // Skip update if unchanged to avoid churn
          if (existing && JSON.stringify(existing) === JSON.stringify(summary)) {
            return state;
          }

          const nextSummaries = {
            ...state.summaries,
            [key]: summary,
          };
          const nextCacheTimes = {
            ...state.cacheTimes,
            [key]: Date.now(),
          };

          const keys = Object.keys(nextSummaries);
          if (keys.length > MAX_SUMMARIES) {
            const oldestKey = keys[0];
            delete nextSummaries[oldestKey];
            delete nextCacheTimes[oldestKey];
          }

          return { summaries: nextSummaries, cacheTimes: nextCacheTimes };
        }),

      hasSummary: (key) => {
        const state = get();
        const summary = state.summaries[key];
        if (!summary) return false;

        const fetchedAt = state.cacheTimes[key];
        if (!fetchedAt) return false;
        const isFresh = Date.now() - fetchedAt < state.ttlMs;

        if (!isFresh) {
          // Evict stale entry only
          set((s) => {
            const { [key]: _, ...restSummaries } = s.summaries;
            const { [key]: __, ...restTimes } = s.cacheTimes;
            return { summaries: restSummaries, cacheTimes: restTimes };
          });
          return false;
        }

        return true;
      },

      getSummary: (key) => get().summaries[key],

      getCacheTime: (key) => get().cacheTimes[key],

      setOverride: (key, summary) =>
        set((state) => ({
          overrides: {
            ...state.overrides,
            [key]: summary,
          },
        })),

      clearOverride: (key) =>
        set((state) => {
          const { [key]: _, ...rest } = state.overrides;
          return { overrides: rest };
        }),

      hasOverride: (key) => Boolean(get().overrides[key]),

      clearSummaries: () => set({ summaries: {}, cacheTimes: {} }),

      getOverrideCount: () => Object.keys(get().overrides).length,

      getAttestationCount: () =>
        Object.values(get().summaries).reduce((count, summary) => count + (summary.attestations?.length || 0), 0),

      evaluateTrust: (email) => {
        const { thresholds, summaries, overrides } = get();

        const senderKey = email.from?.address;
        const override = senderKey ? overrides[senderKey] : undefined;
        if (override) {
          return {
            ...override,
            quarantined: email.isQuarantined || override.quarantined,
          };
        }

        const providedSummary = senderKey ? summaries[senderKey] : undefined;
        if (providedSummary) {
          return {
            ...providedSummary,
            quarantined: email.isQuarantined || providedSummary.quarantined,
          };
        }

        const { score, reasons } = deriveScoreFromEmail(email);

        const tier = email.trustTier || tierFromScore(score, thresholds);
        const quarantined = email.isQuarantined || tier === 'low';

        return {
          score,
          tier,
          reasons,
          pathLength: email.trustPathLength,
          decayAt: email.trustDecayAt,
          quarantined,
        };
      },

      shouldQuarantine: (email) => {
        const { enabled, quarantineEnabled } = get();
        if (!enabled || !quarantineEnabled) return false;

        if (email.isQuarantined) return true;

        const summary = get().evaluateTrust(email);
        return summary.quarantined || summary.tier === 'low';
      },
    }),
    {
      name: 'trust-preferences',
      partialize: (state) => ({
        enabled: state.enabled,
        quarantineEnabled: state.quarantineEnabled,
        hideLowTrust: state.hideLowTrust,
        policy: state.policy,
        thresholds: state.thresholds,
        summaries: state.summaries,
        overrides: state.overrides,
        cacheTimes: state.cacheTimes,
        ttlMs: state.ttlMs,
        notificationsEnabled: state.notificationsEnabled,
      }),
    }
  )
);
