/**
 * Insight Surfacing Service
 *
 * Surfaces emergent insights from collective sensemaking:
 * - Weekly belief digests
 * - Blind spot detection (high uncertainty topics)
 * - Cross-domain connections
 * - Change summaries
 * - Significant shift notifications
 */

import { invoke } from '@tauri-apps/api/core';
import {
  analyzeBeliefTrends,
  generateWeeklyDigest as generateTrendDigest,
  type TrendAnalysisResult,
  type PatternCluster,
} from './pattern-analysis';

// ============================================================================
// TYPES
// ============================================================================

export interface WeeklyDigest {
  generatedAt: number;
  period: { start: number; end: number };
  summary: string;
  highlights: string[];
  concerns: string[];
  recommendations: string[];
  metrics: DigestMetrics;
  topTopics: TopicSummary[];
  blindSpots: BlindSpot[];
  crossDomainInsights: CrossDomainInsight[];
  significantShifts: SignificantShift[];
}

export interface DigestMetrics {
  totalBeliefs: number;
  newBeliefs: number;
  activeParticipants: number;
  consensusReached: number;
  contradictionsDetected: number;
  averageConfidence: number;
}

export interface TopicSummary {
  tag: string;
  beliefCount: number;
  trend: 'rising' | 'stable' | 'declining';
  averageConfidence: number;
  consensusLevel: number | null;
}

export interface BlindSpot {
  topic: string;
  reason: BlindSpotReason;
  severity: number;
  description: string;
  suggestion: string;
}

export type BlindSpotReason =
  | 'high_uncertainty'
  | 'low_coverage'
  | 'expert_disagreement'
  | 'stale_data';

export interface CrossDomainInsight {
  domains: string[];
  connectionType: 'similarity' | 'contradiction' | 'complementary';
  description: string;
  beliefHashes: string[];
  strength: number;
}

export interface SignificantShift {
  topic: string;
  shiftType: ShiftType;
  previousState: string;
  currentState: string;
  magnitude: number;
  detectedAt: number;
}

export type ShiftType =
  | 'consensus_formed'
  | 'consensus_broken'
  | 'confidence_change'
  | 'new_topic_emerged'
  | 'topic_abandoned';

export interface InsightNotification {
  id: string;
  type: 'shift' | 'blind_spot' | 'cross_domain' | 'trend';
  priority: 'high' | 'medium' | 'low';
  title: string;
  message: string;
  timestamp: number;
  dismissed: boolean;
  actionUrl?: string;
}

// ============================================================================
// STATE
// ============================================================================

let lastDigest: WeeklyDigest | null = null;
let lastDigestTime = 0;
const DIGEST_CACHE_DURATION = 6 * 60 * 60 * 1000; // 6 hours

let notifications: InsightNotification[] = [];
let notificationListeners: ((notifications: InsightNotification[]) => void)[] = [];

// ============================================================================
// WEEKLY DIGEST
// ============================================================================

/**
 * Generate a comprehensive weekly digest
 */
export async function generateWeeklyDigest(): Promise<WeeklyDigest | null> {
  const now = Date.now();

  // Return cached if fresh
  if (lastDigest && now - lastDigestTime < DIGEST_CACHE_DURATION) {
    return lastDigest;
  }

  try {
    // Get trend analysis for the week
    const trendResult = await analyzeBeliefTrends({
      time_window_hours: 168, // 7 days
      min_occurrences: 2,
      velocity_threshold: 5,
    });

    if (!trendResult) {
      return null;
    }

    // Get basic digest from trend analysis
    const basicDigest = generateTrendDigest(trendResult);

    // Detect blind spots
    const blindSpots = detectBlindSpots(trendResult);

    // Detect cross-domain insights
    const crossDomainInsights = await detectCrossDomainInsights();

    // Get significant shifts
    const significantShifts = detectSignificantShifts(trendResult);

    // Build topic summaries
    const topTopics = buildTopicSummaries(trendResult);

    // Calculate metrics
    const metrics: DigestMetrics = {
      totalBeliefs: trendResult.total_beliefs,
      newBeliefs: trendResult.beliefs_in_window,
      activeParticipants: 0, // Would need to track unique authors
      consensusReached: trendResult.emerging_consensus.length,
      contradictionsDetected: 0, // Would come from separate analysis
      averageConfidence: 0.7, // Would calculate from beliefs
    };

    const digest: WeeklyDigest = {
      generatedAt: now,
      period: {
        start: now - 7 * 24 * 60 * 60 * 1000,
        end: now,
      },
      summary: basicDigest.summary,
      highlights: basicDigest.highlights,
      concerns: basicDigest.concerns,
      recommendations: basicDigest.recommendations,
      metrics,
      topTopics,
      blindSpots,
      crossDomainInsights,
      significantShifts,
    };

    lastDigest = digest;
    lastDigestTime = now;

    // Generate notifications for significant findings
    generateNotifications(digest);

    return digest;
  } catch (error) {
    console.error('Failed to generate weekly digest:', error);
    return null;
  }
}

// ============================================================================
// BLIND SPOT DETECTION
// ============================================================================

function detectBlindSpots(trendResult: TrendAnalysisResult): BlindSpot[] {
  const blindSpots: BlindSpot[] = [];

  // Check for high uncertainty topics (many beliefs but no consensus)
  for (const consensus of trendResult.emerging_consensus) {
    if (consensus.agreement_level < 0.4) {
      blindSpots.push({
        topic: consensus.topic,
        reason: 'high_uncertainty',
        severity: 1 - consensus.agreement_level,
        description: `'${consensus.topic}' has high participation but low agreement (${Math.round(consensus.agreement_level * 100)}%)`,
        suggestion: 'Consider facilitating structured dialogue to identify core disagreements.',
      });
    }
  }

  // Check for herding (potential blind spot from groupthink)
  for (const warning of trendResult.herding_warnings) {
    if (warning.warning_type === 'UniformityBias') {
      blindSpots.push({
        topic: warning.topic,
        reason: 'expert_disagreement',
        severity: warning.severity,
        description: warning.description,
        suggestion: 'Seek out dissenting voices or alternative perspectives.',
      });
    }
  }

  // Check for topics with declining activity (might be abandoned important questions)
  for (const trend of trendResult.tag_trends) {
    if (trend.growth_rate < -30 && trend.count_total > 5) {
      blindSpots.push({
        topic: trend.tag,
        reason: 'stale_data',
        severity: 0.4,
        description: `'${trend.tag}' activity has declined significantly`,
        suggestion: 'Check if this topic needs revisiting with fresh perspectives.',
      });
    }
  }

  return blindSpots.sort((a, b) => b.severity - a.severity).slice(0, 5);
}

// ============================================================================
// CROSS-DOMAIN INSIGHTS
// ============================================================================

async function detectCrossDomainInsights(): Promise<CrossDomainInsight[]> {
  try {
    // Get patterns from the collective
    const patterns = await invoke<PatternCluster[]>('detect_patterns', {
      input: { similarity_threshold: 0.6, min_cluster_size: 2 },
    });

    const insights: CrossDomainInsight[] = [];

    for (const pattern of patterns) {
      if (pattern.tags.length >= 2) {
        // Multi-domain cluster found
        let connectionType: 'similarity' | 'contradiction' | 'complementary';
        let description: string;

        if (pattern.pattern_type === 'ContradictionCluster') {
          connectionType = 'contradiction';
          description = `Conflicting views between ${pattern.tags.join(' and ')}`;
        } else if (pattern.pattern_type === 'Convergence') {
          connectionType = 'similarity';
          description = `Shared insights across ${pattern.tags.join(' and ')}`;
        } else {
          connectionType = 'complementary';
          description = `Related ideas spanning ${pattern.tags.join(' and ')}`;
        }

        insights.push({
          domains: pattern.tags,
          connectionType,
          description,
          beliefHashes: pattern.member_hashes,
          strength: pattern.coherence,
        });
      }
    }

    return insights.sort((a, b) => b.strength - a.strength).slice(0, 5);
  } catch {
    return [];
  }
}

// ============================================================================
// SIGNIFICANT SHIFTS
// ============================================================================

function detectSignificantShifts(trendResult: TrendAnalysisResult): SignificantShift[] {
  const shifts: SignificantShift[] = [];
  const now = Date.now();

  // New consensus formed
  for (const consensus of trendResult.emerging_consensus) {
    if (consensus.agreement_level > 0.7 && consensus.participant_count >= 3) {
      shifts.push({
        topic: consensus.topic,
        shiftType: 'consensus_formed',
        previousState: 'Uncertain',
        currentState: `${Math.round(consensus.agreement_level * 100)}% consensus`,
        magnitude: consensus.agreement_level,
        detectedAt: now,
      });
    }
  }

  // Rapid topic emergence (velocity alerts)
  for (const velocity of trendResult.velocity_alerts) {
    if (velocity.severity > 0.6) {
      shifts.push({
        topic: velocity.tag,
        shiftType: 'new_topic_emerged',
        previousState: 'Low activity',
        currentState: `${velocity.peak_hour_beliefs} beliefs/hour peak`,
        magnitude: velocity.severity,
        detectedAt: now,
      });
    }
  }

  // Major trend changes
  for (const trend of trendResult.tag_trends) {
    if (Math.abs(trend.growth_rate) > 50) {
      const isGrowing = trend.growth_rate > 0;
      shifts.push({
        topic: trend.tag,
        shiftType: isGrowing ? 'confidence_change' : 'topic_abandoned',
        previousState: isGrowing ? 'Moderate activity' : 'Active',
        currentState: isGrowing
          ? `${Math.round(trend.growth_rate)}% growth`
          : `${Math.round(-trend.growth_rate)}% decline`,
        magnitude: Math.abs(trend.growth_rate) / 100,
        detectedAt: now,
      });
    }
  }

  return shifts.sort((a, b) => b.magnitude - a.magnitude).slice(0, 5);
}

// ============================================================================
// TOPIC SUMMARIES
// ============================================================================

function buildTopicSummaries(trendResult: TrendAnalysisResult): TopicSummary[] {
  return trendResult.tag_trends.map((trend) => {
    // Find consensus for this topic
    const consensus = trendResult.emerging_consensus.find((c) => c.topic === trend.tag);

    let trendDirection: 'rising' | 'stable' | 'declining';
    if (trend.growth_rate > 20) {
      trendDirection = 'rising';
    } else if (trend.growth_rate < -20) {
      trendDirection = 'declining';
    } else {
      trendDirection = 'stable';
    }

    return {
      tag: trend.tag,
      beliefCount: trend.count_in_window,
      trend: trendDirection,
      averageConfidence: 0.7, // Would calculate from actual beliefs
      consensusLevel: consensus?.agreement_level ?? null,
    };
  });
}

// ============================================================================
// NOTIFICATIONS
// ============================================================================

function generateNotifications(digest: WeeklyDigest): void {
  const newNotifications: InsightNotification[] = [];

  // High priority: significant shifts
  for (const shift of digest.significantShifts.filter((s) => s.magnitude > 0.7)) {
    newNotifications.push({
      id: `shift-${shift.topic}-${shift.detectedAt}`,
      type: 'shift',
      priority: 'high',
      title: `Major shift in "${shift.topic}"`,
      message: `${shift.previousState} → ${shift.currentState}`,
      timestamp: shift.detectedAt,
      dismissed: false,
    });
  }

  // Medium priority: blind spots
  for (const blindSpot of digest.blindSpots.filter((b) => b.severity > 0.5)) {
    newNotifications.push({
      id: `blindspot-${blindSpot.topic}`,
      type: 'blind_spot',
      priority: 'medium',
      title: `Blind spot detected: "${blindSpot.topic}"`,
      message: blindSpot.description,
      timestamp: Date.now(),
      dismissed: false,
    });
  }

  // Low priority: cross-domain discoveries
  for (const insight of digest.crossDomainInsights.filter((i) => i.strength > 0.7)) {
    newNotifications.push({
      id: `crossdomain-${insight.domains.join('-')}`,
      type: 'cross_domain',
      priority: 'low',
      title: `Cross-domain connection found`,
      message: insight.description,
      timestamp: Date.now(),
      dismissed: false,
    });
  }

  // Merge with existing (avoid duplicates)
  const existingIds = new Set(notifications.map((n) => n.id));
  const uniqueNew = newNotifications.filter((n) => !existingIds.has(n.id));

  if (uniqueNew.length > 0) {
    notifications = [...uniqueNew, ...notifications].slice(0, 20);
    notifyListeners();
  }
}

/**
 * Subscribe to notification updates
 */
export function subscribeToNotifications(
  callback: (notifications: InsightNotification[]) => void
): () => void {
  notificationListeners.push(callback);
  callback(notifications); // Immediate callback with current state

  return () => {
    notificationListeners = notificationListeners.filter((l) => l !== callback);
  };
}

/**
 * Dismiss a notification
 */
export function dismissNotification(id: string): void {
  notifications = notifications.map((n) =>
    n.id === id ? { ...n, dismissed: true } : n
  );
  notifyListeners();
}

/**
 * Get active (undismissed) notifications
 */
export function getActiveNotifications(): InsightNotification[] {
  return notifications.filter((n) => !n.dismissed);
}

/**
 * Clear all notifications
 */
export function clearAllNotifications(): void {
  notifications = [];
  notifyListeners();
}

function notifyListeners(): void {
  for (const listener of notificationListeners) {
    listener(notifications);
  }
}

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * Format digest period for display
 */
export function formatDigestPeriod(digest: WeeklyDigest): string {
  const start = new Date(digest.period.start);
  const end = new Date(digest.period.end);

  return `${start.toLocaleDateString()} - ${end.toLocaleDateString()}`;
}

/**
 * Get digest age description
 */
export function getDigestAge(digest: WeeklyDigest): string {
  const age = Date.now() - digest.generatedAt;
  const hours = Math.floor(age / (1000 * 60 * 60));

  if (hours < 1) return 'Just generated';
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

/**
 * Invalidate cached digest
 */
export function invalidateDigestCache(): void {
  lastDigest = null;
  lastDigestTime = 0;
}
