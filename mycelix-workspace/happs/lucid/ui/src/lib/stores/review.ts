/**
 * Spaced Repetition Review Store
 *
 * Implements SM-2-like algorithm for belief review scheduling.
 * Tracks confidence decay and schedules reviews for important thoughts.
 */

import { writable, derived, get } from 'svelte/store';
import type { Thought } from '@mycelix/lucid-client';
import { thoughts } from './thoughts';

// ============================================================================
// TYPES
// ============================================================================

export interface ReviewItem {
  thoughtId: string;
  lastReviewed: number; // timestamp
  nextReview: number; // timestamp
  interval: number; // days
  easeFactor: number; // 1.3 - 2.5
  repetitions: number;
  decayedConfidence: number;
}

export interface ReviewSession {
  items: ReviewItem[];
  currentIndex: number;
  completed: ReviewItem[];
  startTime: number;
}

export type ReviewRating = 'forgot' | 'hard' | 'good' | 'easy';

// ============================================================================
// STORES
// ============================================================================

// Review items keyed by thought ID
export const reviewItems = writable<Map<string, ReviewItem>>(new Map());

// Current review session
export const reviewSession = writable<ReviewSession | null>(null);

// Daily reflection prompts
export const dailyPrompts = writable<string[]>([
  'What belief have you reconsidered recently?',
  'What knowledge gap do you want to fill today?',
  'What thought needs more evidence?',
  'What assumption should you question?',
  'What connection between ideas have you noticed?',
]);

// ============================================================================
// DERIVED STORES
// ============================================================================

// Thoughts due for review
export const dueForReview = derived(
  [thoughts, reviewItems],
  ([$thoughts, $reviewItems]) => {
    const now = Date.now();
    return $thoughts.filter((t) => {
      const item = $reviewItems.get(t.id);
      if (!item) return false;
      return item.nextReview <= now;
    });
  }
);

// Thoughts that have never been reviewed
export const neverReviewed = derived(
  [thoughts, reviewItems],
  ([$thoughts, $reviewItems]) => {
    return $thoughts.filter((t) => !$reviewItems.has(t.id));
  }
);

// Review stats
export const reviewStats = derived(reviewItems, ($items) => {
  const items = Array.from($items.values());
  return {
    total: items.length,
    dueToday: items.filter((i) => i.nextReview <= Date.now()).length,
    avgConfidence: items.length > 0
      ? items.reduce((sum, i) => sum + i.decayedConfidence, 0) / items.length
      : 0,
    avgInterval: items.length > 0
      ? items.reduce((sum, i) => sum + i.interval, 0) / items.length
      : 0,
  };
});

// ============================================================================
// CONFIDENCE DECAY
// ============================================================================

/**
 * Calculate decayed confidence based on time since last review
 * Uses exponential decay: C(t) = C0 * e^(-λt)
 * Half-life of ~30 days by default
 */
export function calculateDecayedConfidence(
  originalConfidence: number,
  lastReviewed: number,
  halfLifeDays: number = 30
): number {
  const daysSinceReview = (Date.now() - lastReviewed) / (1000 * 60 * 60 * 24);
  const decayConstant = Math.LN2 / halfLifeDays;
  const decayedConfidence = originalConfidence * Math.exp(-decayConstant * daysSinceReview);
  return Math.max(0.1, Math.min(1, decayedConfidence));
}

/**
 * Update all confidence values based on time decay
 */
export function updateDecayedConfidences(): void {
  reviewItems.update((items) => {
    for (const [id, item] of items) {
      item.decayedConfidence = calculateDecayedConfidence(
        item.decayedConfidence,
        item.lastReviewed
      );
    }
    return items;
  });
}

// ============================================================================
// SM-2 ALGORITHM
// ============================================================================

/**
 * Calculate next review interval using SM-2 algorithm
 */
function calculateNextInterval(
  rating: ReviewRating,
  currentInterval: number,
  easeFactor: number,
  repetitions: number
): { interval: number; easeFactor: number; repetitions: number } {
  let newInterval: number;
  let newEaseFactor = easeFactor;
  let newRepetitions = repetitions;

  // Rating to quality (0-5 scale for SM-2)
  const qualityMap: Record<ReviewRating, number> = {
    forgot: 0,
    hard: 2,
    good: 4,
    easy: 5,
  };
  const quality = qualityMap[rating];

  if (quality < 3) {
    // Forgot - reset
    newRepetitions = 0;
    newInterval = 1;
  } else {
    // Remember - increase interval
    if (repetitions === 0) {
      newInterval = 1;
    } else if (repetitions === 1) {
      newInterval = 6;
    } else {
      newInterval = Math.round(currentInterval * easeFactor);
    }
    newRepetitions = repetitions + 1;
  }

  // Update ease factor
  newEaseFactor = easeFactor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02));
  newEaseFactor = Math.max(1.3, Math.min(2.5, newEaseFactor));

  return {
    interval: newInterval,
    easeFactor: newEaseFactor,
    repetitions: newRepetitions,
  };
}

// ============================================================================
// REVIEW FUNCTIONS
// ============================================================================

/**
 * Add a thought to the review system
 */
export function addToReview(thought: Thought): void {
  reviewItems.update((items) => {
    if (!items.has(thought.id)) {
      items.set(thought.id, {
        thoughtId: thought.id,
        lastReviewed: Date.now(),
        nextReview: Date.now() + 24 * 60 * 60 * 1000, // Tomorrow
        interval: 1,
        easeFactor: 2.5,
        repetitions: 0,
        decayedConfidence: thought.confidence,
      });
    }
    return items;
  });
}

/**
 * Remove a thought from review
 */
export function removeFromReview(thoughtId: string): void {
  reviewItems.update((items) => {
    items.delete(thoughtId);
    return items;
  });
}

/**
 * Start a review session with due items
 */
export function startReviewSession(maxItems: number = 20): void {
  const $thoughts = get(thoughts);
  const $reviewItems = get(reviewItems);
  const now = Date.now();

  // Get thoughts due for review
  const dueItems: ReviewItem[] = [];
  for (const [id, item] of $reviewItems) {
    if (item.nextReview <= now) {
      dueItems.push(item);
    }
  }

  // Sort by overdue time
  dueItems.sort((a, b) => a.nextReview - b.nextReview);

  // Limit to maxItems
  const sessionItems = dueItems.slice(0, maxItems);

  if (sessionItems.length > 0) {
    reviewSession.set({
      items: sessionItems,
      currentIndex: 0,
      completed: [],
      startTime: Date.now(),
    });
  }
}

/**
 * Record a review rating and advance to next item
 */
export function recordReview(rating: ReviewRating, newConfidence?: number): void {
  reviewSession.update((session) => {
    if (!session) return null;

    const currentItem = session.items[session.currentIndex];
    if (!currentItem) return session;

    // Update the review item
    reviewItems.update((items) => {
      const item = items.get(currentItem.thoughtId);
      if (item) {
        const { interval, easeFactor, repetitions } = calculateNextInterval(
          rating,
          item.interval,
          item.easeFactor,
          item.repetitions
        );

        item.interval = interval;
        item.easeFactor = easeFactor;
        item.repetitions = repetitions;
        item.lastReviewed = Date.now();
        item.nextReview = Date.now() + interval * 24 * 60 * 60 * 1000;

        if (newConfidence !== undefined) {
          item.decayedConfidence = newConfidence;
        }
      }
      return items;
    });

    // Move to next item
    session.completed.push(currentItem);
    session.currentIndex++;

    // End session if complete
    if (session.currentIndex >= session.items.length) {
      return null;
    }

    return session;
  });
}

/**
 * End the current review session
 */
export function endReviewSession(): void {
  reviewSession.set(null);
}

/**
 * Get random daily reflection prompt
 */
export function getDailyPrompt(): string {
  const prompts = get(dailyPrompts);
  const dayOfYear = Math.floor(
    (Date.now() - new Date(new Date().getFullYear(), 0, 0).getTime()) / (1000 * 60 * 60 * 24)
  );
  return prompts[dayOfYear % prompts.length];
}

// ============================================================================
// PERSISTENCE
// ============================================================================

const STORAGE_KEY = 'lucid-review-items';

/**
 * Save review items to localStorage
 */
export function saveReviewItems(): void {
  const items = get(reviewItems);
  const serialized = JSON.stringify(Array.from(items.entries()));
  localStorage.setItem(STORAGE_KEY, serialized);
}

/**
 * Load review items from localStorage
 */
export function loadReviewItems(): void {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored) {
    try {
      const entries = JSON.parse(stored);
      reviewItems.set(new Map(entries));
    } catch (e) {
      console.error('Failed to load review items:', e);
    }
  }
}

// Auto-save when items change
reviewItems.subscribe(() => {
  saveReviewItems();
});
