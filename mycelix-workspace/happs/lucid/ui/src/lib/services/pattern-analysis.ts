/**
 * Pattern Analysis Service
 *
 * Provides temporal pattern detection, trend analysis, and collective
 * sensemaking insights from the collective zome.
 *
 * Features:
 * - Frequency-over-time analysis
 * - Belief velocity detection (rapid adoption)
 * - Emerging consensus alerts
 * - Herding effect detection
 * - Belief lifecycle visualization data
 */

import { invoke } from '@tauri-apps/api/core';
import { patternCache, memoize } from '../utils/cache';
import { withRetry, withFallback, isRetryableError } from '../utils/resilience';

// ============================================================================
// TYPES
// ============================================================================

export interface TrendAnalysisInput {
  time_window_hours?: number;
  min_occurrences?: number;
  velocity_threshold?: number;
}

export interface TrendAnalysisResult {
  time_window_hours: number;
  beliefs_in_window: number;
  total_beliefs: number;
  tag_trends: TagTrend[];
  velocity_alerts: VelocityAlert[];
  emerging_consensus: EmergingConsensusAlert[];
  herding_warnings: HerdingWarning[];
  trend_direction: TrendDirection;
}

export interface TagTrend {
  tag: string;
  count_in_window: number;
  count_total: number;
  growth_rate: number;
  is_trending: boolean;
}

export interface VelocityAlert {
  tag: string;
  beliefs_per_hour: number;
  peak_hour_beliefs: number;
  description: string;
  severity: number;
}

export interface EmergingConsensusAlert {
  topic: string;
  belief_hashes: string[];
  agreement_level: number;
  participant_count: number;
  description: string;
}

export interface HerdingWarning {
  topic: string;
  warning_type: HerdingType;
  severity: number;
  description: string;
  suggestion: string;
}

export type HerdingType = 'RapidAdoption' | 'UniformityBias' | 'InfluenceCascade';

export type TrendDirection = 'Converging' | 'Diverging' | 'Stable' | 'Declining';

export interface BeliefLifecycle {
  tag: string;
  total_beliefs: number;
  lifecycle_stage: LifecycleStage;
  first_appearance: number | null;
  last_activity: number | null;
  growth_phases: GrowthPhase[];
  current_activity_level: number;
}

export type LifecycleStage = 'Nascent' | 'Rapid' | 'Mature' | 'Stable' | 'Dormant';

export interface GrowthPhase {
  start: number;
  end: number;
  phase_type: string;
  belief_count: number;
}

export interface PatternCluster {
  pattern_id: string;
  representative_content: string;
  representative_hash: string;
  member_hashes: string[];
  member_count: number;
  pattern_type: PatternType;
  coherence: number;
  tags: string[];
}

export type PatternType =
  | 'Convergence'
  | 'Divergence'
  | 'Trend'
  | 'Cluster'
  | 'ContradictionCluster';

// ============================================================================
// STATE
// ============================================================================

let lastTrendAnalysis: TrendAnalysisResult | null = null;
let lastAnalysisTime = 0;
const CACHE_DURATION_MS = 60_000; // 1 minute cache

// ============================================================================
// API FUNCTIONS
// ============================================================================

/**
 * Analyze belief trends over a time window
 */
export async function analyzeBeliefTrends(
  input: TrendAnalysisInput = {}
): Promise<TrendAnalysisResult | null> {
  const cacheKey = `trends-${input.time_window_hours ?? 168}-${input.min_occurrences ?? 2}`;

  return patternCache.get(cacheKey, async () => {
    return withFallback(
      async () => {
        return withRetry(
          async () => {
            const result = await invoke<TrendAnalysisResult>('analyze_belief_trends', { input });
            // Also update legacy cache for backwards compatibility
            lastTrendAnalysis = result;
            lastAnalysisTime = Date.now();
            return result;
          },
          {
            maxAttempts: 2,
            initialDelay: 500,
            shouldRetry: isRetryableError,
          }
        );
      },
      {
        fallback: lastTrendAnalysis, // Use stale data as fallback
        logErrors: true,
      }
    );
  }) as Promise<TrendAnalysisResult | null>;
}

/**
 * Get lifecycle analysis for a specific topic/tag
 */
export async function getBeliefLifecycle(tag: string): Promise<BeliefLifecycle | null> {
  try {
    return await invoke<BeliefLifecycle>('get_belief_lifecycle', { tag });
  } catch (error) {
    console.error('Failed to get belief lifecycle:', error);
    return null;
  }
}

/**
 * Detect semantic patterns across all beliefs
 */
export async function detectPatterns(
  similarityThreshold = 0.7,
  minClusterSize = 2
): Promise<PatternCluster[]> {
  try {
    const input = {
      similarity_threshold: similarityThreshold,
      min_cluster_size: minClusterSize,
    };
    return await invoke<PatternCluster[]>('detect_patterns', { input });
  } catch (error) {
    console.error('Failed to detect patterns:', error);
    return [];
  }
}

/**
 * Invalidate the trend analysis cache
 */
export function invalidateTrendCache(): void {
  lastTrendAnalysis = null;
  lastAnalysisTime = 0;
}

// ============================================================================
// DERIVED INSIGHTS
// ============================================================================

/**
 * Get high-priority alerts that need attention
 */
export function getHighPriorityAlerts(result: TrendAnalysisResult): {
  type: 'velocity' | 'herding' | 'consensus';
  severity: number;
  message: string;
  suggestion?: string;
}[] {
  const alerts: {
    type: 'velocity' | 'herding' | 'consensus';
    severity: number;
    message: string;
    suggestion?: string;
  }[] = [];

  // High velocity alerts
  for (const v of result.velocity_alerts.filter((a) => a.severity > 0.7)) {
    alerts.push({
      type: 'velocity',
      severity: v.severity,
      message: v.description,
    });
  }

  // Herding warnings
  for (const h of result.herding_warnings.filter((w) => w.severity > 0.5)) {
    alerts.push({
      type: 'herding',
      severity: h.severity,
      message: h.description,
      suggestion: h.suggestion,
    });
  }

  // Emerging consensus (high agreement)
  for (const c of result.emerging_consensus.filter((e) => e.agreement_level > 0.8)) {
    alerts.push({
      type: 'consensus',
      severity: 0.6,
      message: c.description,
    });
  }

  return alerts.sort((a, b) => b.severity - a.severity);
}

/**
 * Calculate collective health score based on trends
 */
export function calculateCollectiveHealth(result: TrendAnalysisResult): {
  score: number;
  label: string;
  factors: { name: string; value: number; impact: 'positive' | 'negative' | 'neutral' }[];
} {
  const factors: { name: string; value: number; impact: 'positive' | 'negative' | 'neutral' }[] = [];
  let score = 0.5; // Start at neutral

  // Activity level
  const activityRatio = result.beliefs_in_window / Math.max(result.total_beliefs, 1);
  if (activityRatio > 0.3) {
    score += 0.1;
    factors.push({ name: 'High Activity', value: activityRatio, impact: 'positive' });
  } else if (activityRatio < 0.1) {
    score -= 0.1;
    factors.push({ name: 'Low Activity', value: activityRatio, impact: 'negative' });
  }

  // Diversity (based on trend direction)
  if (result.trend_direction === 'Diverging') {
    score += 0.1;
    factors.push({ name: 'Diverse Perspectives', value: 1, impact: 'positive' });
  } else if (result.trend_direction === 'Converging') {
    factors.push({ name: 'Converging Views', value: 1, impact: 'neutral' });
  }

  // Herding risk
  const herdingRisk = result.herding_warnings.reduce((sum, w) => sum + w.severity, 0);
  if (herdingRisk > 1) {
    score -= 0.15;
    factors.push({ name: 'Herding Risk', value: herdingRisk, impact: 'negative' });
  }

  // Emerging consensus (positive when balanced)
  if (result.emerging_consensus.length > 0 && result.emerging_consensus.length < 5) {
    score += 0.1;
    factors.push({
      name: 'Healthy Consensus Building',
      value: result.emerging_consensus.length,
      impact: 'positive',
    });
  }

  // Clamp score
  score = Math.max(0, Math.min(1, score));

  // Determine label
  let label: string;
  if (score >= 0.7) {
    label = 'Excellent';
  } else if (score >= 0.5) {
    label = 'Good';
  } else if (score >= 0.3) {
    label = 'Needs Attention';
  } else {
    label = 'Concerning';
  }

  return { score, label, factors };
}

/**
 * Generate weekly digest summary
 */
export function generateWeeklyDigest(result: TrendAnalysisResult): {
  summary: string;
  highlights: string[];
  concerns: string[];
  recommendations: string[];
} {
  const highlights: string[] = [];
  const concerns: string[] = [];
  const recommendations: string[] = [];

  // Activity summary
  const activityPct = ((result.beliefs_in_window / Math.max(result.total_beliefs, 1)) * 100).toFixed(
    0
  );
  highlights.push(
    `${result.beliefs_in_window} new beliefs shared (${activityPct}% of total)`
  );

  // Trending topics
  const trendingTags = result.tag_trends.filter((t) => t.is_trending);
  if (trendingTags.length > 0) {
    highlights.push(
      `Trending topics: ${trendingTags.map((t) => t.tag).join(', ')}`
    );
  }

  // Emerging consensus
  if (result.emerging_consensus.length > 0) {
    highlights.push(
      `${result.emerging_consensus.length} emerging consensus point(s) detected`
    );
  }

  // Concerns
  for (const warning of result.herding_warnings) {
    concerns.push(warning.description);
  }

  for (const velocity of result.velocity_alerts.filter((v) => v.severity > 0.5)) {
    concerns.push(velocity.description);
  }

  // Recommendations based on trend direction
  switch (result.trend_direction) {
    case 'Converging':
      recommendations.push(
        'The collective is converging. Consider introducing alternative perspectives.'
      );
      break;
    case 'Diverging':
      recommendations.push(
        'Healthy diversity in perspectives. Look for common ground opportunities.'
      );
      break;
    case 'Declining':
      recommendations.push(
        'Activity is declining. Consider re-engaging with provocative questions.'
      );
      break;
    case 'Stable':
      recommendations.push('Collective is stable. Good time to deepen existing topics.');
      break;
  }

  // Add herding suggestions
  for (const warning of result.herding_warnings.slice(0, 2)) {
    recommendations.push(warning.suggestion);
  }

  const summary = `Over the past ${result.time_window_hours} hours, the collective shared ${result.beliefs_in_window} beliefs. The overall trend is ${result.trend_direction.toLowerCase()}.`;

  return { summary, highlights, concerns, recommendations };
}

// ============================================================================
// VISUALIZATION HELPERS
// ============================================================================

/**
 * Prepare data for trend chart visualization
 */
export function prepareTrendChartData(
  tagTrends: TagTrend[]
): { label: string; value: number; isHot: boolean }[] {
  return tagTrends.map((t) => ({
    label: t.tag,
    value: t.count_in_window,
    isHot: t.is_trending,
  }));
}

/**
 * Prepare lifecycle stage for visualization
 */
export function getLifecycleStageInfo(stage: LifecycleStage): {
  label: string;
  color: string;
  icon: string;
  description: string;
} {
  switch (stage) {
    case 'Nascent':
      return {
        label: 'Nascent',
        color: '#a3e635',
        icon: '🌱',
        description: 'Just emerging, still being explored',
      };
    case 'Rapid':
      return {
        label: 'Rapid Growth',
        color: '#f97316',
        icon: '🚀',
        description: 'High activity, rapid adoption',
      };
    case 'Mature':
      return {
        label: 'Mature',
        color: '#3b82f6',
        icon: '🌳',
        description: 'Well-established with steady activity',
      };
    case 'Stable':
      return {
        label: 'Stable',
        color: '#8b5cf6',
        icon: '⚓',
        description: 'Stable with occasional updates',
      };
    case 'Dormant':
      return {
        label: 'Dormant',
        color: '#6b7280',
        icon: '💤',
        description: 'No recent activity',
      };
  }
}

/**
 * Get trend direction styling
 */
export function getTrendDirectionInfo(direction: TrendDirection): {
  label: string;
  color: string;
  icon: string;
} {
  switch (direction) {
    case 'Converging':
      return { label: 'Converging', color: '#22c55e', icon: '🎯' };
    case 'Diverging':
      return { label: 'Diverging', color: '#f59e0b', icon: '🌿' };
    case 'Stable':
      return { label: 'Stable', color: '#3b82f6', icon: '⚖️' };
    case 'Declining':
      return { label: 'Declining', color: '#ef4444', icon: '📉' };
  }
}
