// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Animated Balance & UI Polish - Numbers That Feel Alive
 *
 * Provides spring-physics animations, smooth number transitions, and
 * visual feedback utilities for creating delightful wallet UX.
 *
 * @example
 * ```typescript
 * // Animated balance
 * const balance = new AnimatedValue(0);
 * balance.springTo(1234.56); // Smooth spring animation
 *
 * // Format with animation
 * const formatted = formatAnimatedCurrency(balance.value, 'MYC');
 *
 * // Visual feedback
 * const pulse = createSuccessPulse();
 * ```
 */

import { BehaviorSubject, type Subscription } from '../reactive/index.js';

// =============================================================================
// Environment Polyfills (for Node.js test environments)
// =============================================================================

// Polyfill requestAnimationFrame for Node.js
const _requestAnimationFrame =
  typeof requestAnimationFrame !== 'undefined'
    ? requestAnimationFrame
    : (callback: FrameRequestCallback): number => {
        return setTimeout(() => callback(performance.now()), 16) as unknown as number;
      };

const _cancelAnimationFrame =
  typeof cancelAnimationFrame !== 'undefined'
    ? cancelAnimationFrame
    : (id: number): void => {
        clearTimeout(id);
      };

// =============================================================================
// Types
// =============================================================================

/** Spring configuration for physics-based animations */
export interface SpringConfig {
  /** Stiffness (spring constant). Higher = faster. Default: 100 */
  stiffness: number;
  /** Damping ratio. Higher = less bouncy. Default: 10 */
  damping: number;
  /** Mass. Higher = more inertia. Default: 1 */
  mass: number;
  /** Velocity threshold to stop. Default: 0.01 */
  restVelocityThreshold: number;
  /** Displacement threshold to stop. Default: 0.01 */
  restDisplacementThreshold: number;
}

/** Animation state */
export interface AnimationState {
  /** Current value */
  value: number;
  /** Current velocity */
  velocity: number;
  /** Target value */
  target: number;
  /** Whether animation is active */
  isAnimating: boolean;
}

/** Timing function type */
export type EasingFunction = (t: number) => number;

/** Currency display options */
export interface CurrencyFormatOptions {
  /** Currency code (MYC, USD, etc.) */
  currency?: string;
  /** Symbol to display */
  symbol?: string;
  /** Decimal places */
  decimals?: number;
  /** Use thousands separators */
  useSeparators?: boolean;
  /** Locale for formatting */
  locale?: string;
  /** Show currency suffix */
  showCurrency?: boolean;
  /** Compact notation for large numbers */
  compact?: boolean;
}

/** Visual feedback pulse */
export interface Pulse {
  /** Start timestamp */
  startTime: number;
  /** Duration in ms */
  duration: number;
  /** Color (CSS) */
  color: string;
  /** Current progress (0-1) */
  progress: number;
}

// =============================================================================
// Spring Physics
// =============================================================================

const DEFAULT_SPRING_CONFIG: SpringConfig = {
  stiffness: 100,
  damping: 10,
  mass: 1,
  restVelocityThreshold: 0.01,
  restDisplacementThreshold: 0.01,
};

/**
 * Spring presets for common animations
 */
export const SPRING_PRESETS = {
  /** Default balanced spring */
  default: { ...DEFAULT_SPRING_CONFIG },

  /** Gentle, slow spring (for large value changes) */
  gentle: {
    stiffness: 50,
    damping: 14,
    mass: 1,
    restVelocityThreshold: 0.01,
    restDisplacementThreshold: 0.01,
  },

  /** Snappy, quick spring (for small changes) */
  snappy: {
    stiffness: 200,
    damping: 20,
    mass: 1,
    restVelocityThreshold: 0.01,
    restDisplacementThreshold: 0.01,
  },

  /** Bouncy spring (for celebratory animations) */
  bouncy: {
    stiffness: 180,
    damping: 8,
    mass: 1,
    restVelocityThreshold: 0.01,
    restDisplacementThreshold: 0.01,
  },

  /** Stiff spring (for immediate response) */
  stiff: {
    stiffness: 400,
    damping: 30,
    mass: 1,
    restVelocityThreshold: 0.01,
    restDisplacementThreshold: 0.01,
  },

  /** Molasses (for dramatic reveals) */
  molasses: {
    stiffness: 30,
    damping: 20,
    mass: 2,
    restVelocityThreshold: 0.01,
    restDisplacementThreshold: 0.01,
  },
} as const;

/**
 * Calculate spring physics step
 */
function springStep(
  current: number,
  velocity: number,
  target: number,
  config: SpringConfig,
  deltaTime: number
): { value: number; velocity: number } {
  // Spring force: F = -k * (x - target)
  const displacement = current - target;
  const springForce = -config.stiffness * displacement;

  // Damping force: F = -c * v
  const dampingForce = -config.damping * velocity;

  // Acceleration: a = F / m
  const acceleration = (springForce + dampingForce) / config.mass;

  // Update velocity and position (semi-implicit Euler)
  const newVelocity = velocity + acceleration * deltaTime;
  const newValue = current + newVelocity * deltaTime;

  return { value: newValue, velocity: newVelocity };
}

/**
 * Check if spring is at rest
 */
function isAtRest(
  value: number,
  velocity: number,
  target: number,
  config: SpringConfig
): boolean {
  return (
    Math.abs(velocity) < config.restVelocityThreshold &&
    Math.abs(value - target) < config.restDisplacementThreshold
  );
}

// =============================================================================
// Animated Value
// =============================================================================

/**
 * A value that animates smoothly using spring physics
 */
export class AnimatedValue {
  private _state$: BehaviorSubject<AnimationState>;
  private config: SpringConfig;
  private animationFrame?: number;
  private lastTime?: number;

  constructor(initialValue: number = 0, config?: Partial<SpringConfig>) {
    this.config = { ...DEFAULT_SPRING_CONFIG, ...config };
    this._state$ = new BehaviorSubject<AnimationState>({
      value: initialValue,
      velocity: 0,
      target: initialValue,
      isAnimating: false,
    });
  }

  // ===========================================================================
  // Reactive State
  // ===========================================================================

  /** Observable state */
  get state$(): BehaviorSubject<AnimationState> {
    return this._state$;
  }

  /** Current animation state */
  get state(): AnimationState {
    return this._state$.value;
  }

  /** Current value */
  get value(): number {
    return this._state$.value.value;
  }

  /** Target value */
  get target(): number {
    return this._state$.value.target;
  }

  /** Whether animating */
  get isAnimating(): boolean {
    return this._state$.value.isAnimating;
  }

  /** Subscribe to value changes */
  subscribe(observer: (state: AnimationState) => void): Subscription {
    return this._state$.subscribe(observer);
  }

  // ===========================================================================
  // Animation Methods
  // ===========================================================================

  /**
   * Animate to a new value using spring physics
   */
  springTo(target: number, config?: Partial<SpringConfig>): void {
    const springConfig = { ...this.config, ...config };

    // Update target
    this._state$.next({
      ...this._state$.value,
      target,
      isAnimating: true,
    });

    // Start animation loop if not already running
    if (!this.animationFrame) {
      this.lastTime = performance.now();
      this.tick(springConfig);
    }
  }

  /**
   * Set value immediately (no animation)
   */
  setValue(value: number): void {
    this.stop();
    this._state$.next({
      value,
      velocity: 0,
      target: value,
      isAnimating: false,
    });
  }

  /**
   * Add to current value with animation
   */
  add(delta: number, config?: Partial<SpringConfig>): void {
    this.springTo(this._state$.value.target + delta, config);
  }

  /**
   * Stop animation
   */
  stop(): void {
    if (this.animationFrame) {
      _cancelAnimationFrame(this.animationFrame);
      this.animationFrame = undefined;
    }
    this.lastTime = undefined;
    this._state$.next({
      ...this._state$.value,
      isAnimating: false,
    });
  }

  /**
   * Snap to target instantly
   */
  finish(): void {
    this.stop();
    this._state$.next({
      value: this._state$.value.target,
      velocity: 0,
      target: this._state$.value.target,
      isAnimating: false,
    });
  }

  // ===========================================================================
  // Internal Animation Loop
  // ===========================================================================

  private tick(config: SpringConfig): void {
    const now = performance.now();
    const deltaTime = Math.min((now - (this.lastTime ?? now)) / 1000, 0.1); // Cap at 100ms
    this.lastTime = now;

    const currentState = this._state$.value;
    const { value: newValue, velocity: newVelocity } = springStep(
      currentState.value,
      currentState.velocity,
      currentState.target,
      config,
      deltaTime
    );

    if (isAtRest(newValue, newVelocity, currentState.target, config)) {
      // Animation complete
      this._state$.next({
        value: currentState.target,
        velocity: 0,
        target: currentState.target,
        isAnimating: false,
      });
      this.animationFrame = undefined;
      this.lastTime = undefined;
    } else {
      // Continue animating
      this._state$.next({
        value: newValue,
        velocity: newVelocity,
        target: currentState.target,
        isAnimating: true,
      });

      this.animationFrame = _requestAnimationFrame(() => this.tick(config));
    }
  }

  // ===========================================================================
  // Cleanup
  // ===========================================================================

  destroy(): void {
    this.stop();
  }
}

// =============================================================================
// Animated Counter (for integers)
// =============================================================================

/**
 * Animated integer counter with optional formatting
 */
export class AnimatedCounter {
  private animated: AnimatedValue;
  private _formattedValue$: BehaviorSubject<string>;
  private subscription: Subscription;

  constructor(
    initialValue: number = 0,
    private formatter?: (value: number) => string,
    config?: Partial<SpringConfig>
  ) {
    this.animated = new AnimatedValue(initialValue, config);
    this._formattedValue$ = new BehaviorSubject(this.format(initialValue));

    this.subscription = this.animated.subscribe((state) => {
      this._formattedValue$.next(this.format(state.value));
    });
  }

  private format(value: number): string {
    const rounded = Math.round(value);
    return this.formatter ? this.formatter(rounded) : rounded.toLocaleString();
  }

  get value(): number {
    return Math.round(this.animated.value);
  }

  get formattedValue(): string {
    return this._formattedValue$.value;
  }

  get formattedValue$(): BehaviorSubject<string> {
    return this._formattedValue$;
  }

  countTo(target: number, config?: Partial<SpringConfig>): void {
    this.animated.springTo(target, config);
  }

  setValue(value: number): void {
    this.animated.setValue(value);
  }

  subscribe(observer: (formatted: string) => void): Subscription {
    return this._formattedValue$.subscribe(observer);
  }

  stop(): void {
    this.animated.stop();
  }

  finish(): void {
    this.animated.finish();
  }

  destroy(): void {
    this.subscription.unsubscribe();
    this.animated.destroy();
  }
}

// =============================================================================
// Currency Formatting
// =============================================================================

const DEFAULT_FORMAT_OPTIONS: Required<CurrencyFormatOptions> = {
  currency: 'MYC',
  symbol: '',
  decimals: 2,
  useSeparators: true,
  locale: 'en-US',
  showCurrency: true,
  compact: false,
};

/**
 * Format a number as currency
 */
export function formatCurrency(
  value: number,
  options?: CurrencyFormatOptions
): string {
  const opts = { ...DEFAULT_FORMAT_OPTIONS, ...options };

  // Handle compact notation for large numbers
  if (opts.compact && Math.abs(value) >= 1000) {
    return formatCompactCurrency(value, opts);
  }

  // Format the number
  let formatted: string;
  if (opts.useSeparators) {
    formatted = value.toLocaleString(opts.locale, {
      minimumFractionDigits: opts.decimals,
      maximumFractionDigits: opts.decimals,
    });
  } else {
    formatted = value.toFixed(opts.decimals);
  }

  // Add symbol and currency
  const symbol = opts.symbol || getCurrencySymbol(opts.currency);
  const parts: string[] = [];

  if (symbol) parts.push(symbol);
  parts.push(formatted);
  if (opts.showCurrency && opts.currency) parts.push(opts.currency);

  return parts.join(' ').trim();
}

/**
 * Format large numbers compactly (1.2K, 3.4M, etc.)
 */
function formatCompactCurrency(
  value: number,
  options: Required<CurrencyFormatOptions>
): string {
  const absValue = Math.abs(value);
  let suffix = '';
  let divisor = 1;

  if (absValue >= 1e12) {
    suffix = 'T';
    divisor = 1e12;
  } else if (absValue >= 1e9) {
    suffix = 'B';
    divisor = 1e9;
  } else if (absValue >= 1e6) {
    suffix = 'M';
    divisor = 1e6;
  } else if (absValue >= 1e3) {
    suffix = 'K';
    divisor = 1e3;
  }

  const compactValue = value / divisor;
  const formatted = compactValue.toLocaleString(options.locale, {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });

  const symbol = options.symbol || getCurrencySymbol(options.currency);
  const parts: string[] = [];

  if (symbol) parts.push(symbol);
  parts.push(formatted + suffix);
  if (options.showCurrency && options.currency) parts.push(options.currency);

  return parts.join(' ').trim();
}

/**
 * Get currency symbol
 */
export function getCurrencySymbol(currency: string): string {
  const symbols: Record<string, string> = {
    MYC: 'M',
    USD: '$',
    EUR: '\u20AC',
    GBP: '\u00A3',
    JPY: '\u00A5',
    BTC: '\u20BF',
    ETH: '\u039E',
    HOT: 'H',
  };
  return symbols[currency.toUpperCase()] ?? '';
}

/**
 * Format a change in value (with + or - prefix)
 */
export function formatChange(
  value: number,
  options?: CurrencyFormatOptions
): string {
  const formatted = formatCurrency(Math.abs(value), options);
  const prefix = value >= 0 ? '+' : '-';
  return `${prefix}${formatted}`;
}

/**
 * Format as percentage
 */
export function formatPercentage(
  value: number,
  decimals: number = 1
): string {
  const formatted = (value * 100).toFixed(decimals);
  return `${formatted}%`;
}

// =============================================================================
// Visual Feedback
// =============================================================================

/**
 * Create a success pulse animation
 */
export function createSuccessPulse(duration: number = 300): Pulse {
  return {
    startTime: Date.now(),
    duration,
    color: '#22c55e', // Green
    progress: 0,
  };
}

/**
 * Create an error pulse animation
 */
export function createErrorPulse(duration: number = 500): Pulse {
  return {
    startTime: Date.now(),
    duration,
    color: '#ef4444', // Red
    progress: 0,
  };
}

/**
 * Create a warning pulse animation
 */
export function createWarningPulse(duration: number = 400): Pulse {
  return {
    startTime: Date.now(),
    duration,
    color: '#f59e0b', // Amber
    progress: 0,
  };
}

/**
 * Update pulse progress
 */
export function updatePulse(pulse: Pulse): Pulse {
  const elapsed = Date.now() - pulse.startTime;
  const progress = Math.min(elapsed / pulse.duration, 1);
  return { ...pulse, progress };
}

/**
 * Check if pulse is complete
 */
export function isPulseComplete(pulse: Pulse): boolean {
  return Date.now() - pulse.startTime >= pulse.duration;
}

// =============================================================================
// Easing Functions
// =============================================================================

export const EASING = {
  /** Linear - no easing */
  linear: (t: number): number => t,

  /** Ease in (slow start) */
  easeIn: (t: number): number => t * t,

  /** Ease out (slow end) */
  easeOut: (t: number): number => t * (2 - t),

  /** Ease in-out (slow start and end) */
  easeInOut: (t: number): number => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),

  /** Cubic ease in */
  easeInCubic: (t: number): number => t * t * t,

  /** Cubic ease out */
  easeOutCubic: (t: number): number => {
    const t1 = t - 1;
    return t1 * t1 * t1 + 1;
  },

  /** Cubic ease in-out */
  easeInOutCubic: (t: number): number =>
    t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,

  /** Bounce effect */
  bounce: (t: number): number => {
    if (t < 1 / 2.75) {
      return 7.5625 * t * t;
    } else if (t < 2 / 2.75) {
      const t1 = t - 1.5 / 2.75;
      return 7.5625 * t1 * t1 + 0.75;
    } else if (t < 2.5 / 2.75) {
      const t1 = t - 2.25 / 2.75;
      return 7.5625 * t1 * t1 + 0.9375;
    } else {
      const t1 = t - 2.625 / 2.75;
      return 7.5625 * t1 * t1 + 0.984375;
    }
  },

  /** Elastic (overshoot then settle) */
  elastic: (t: number): number => {
    if (t === 0 || t === 1) return t;
    const p = 0.3;
    const s = p / 4;
    return Math.pow(2, -10 * t) * Math.sin(((t - s) * (2 * Math.PI)) / p) + 1;
  },
} as const;

// =============================================================================
// Tween Utility
// =============================================================================

/**
 * Create a tween animation
 */
export function tween(
  from: number,
  to: number,
  duration: number,
  easing: EasingFunction = EASING.easeOut
): {
  getValue: () => number;
  getProgress: () => number;
  isComplete: () => boolean;
  start: () => void;
} {
  let startTime: number | null = null;

  return {
    getValue: () => {
      if (startTime === null) return from;
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easedProgress = easing(progress);
      return from + (to - from) * easedProgress;
    },
    getProgress: () => {
      if (startTime === null) return 0;
      return Math.min((Date.now() - startTime) / duration, 1);
    },
    isComplete: () => {
      if (startTime === null) return false;
      return Date.now() - startTime >= duration;
    },
    start: () => {
      startTime = Date.now();
    },
  };
}

// =============================================================================
// Balance Animation Helper
// =============================================================================

/**
 * Creates an animated balance display
 */
export function createAnimatedBalance(
  initialBalance: number = 0,
  currency: string = 'MYC',
  springConfig?: Partial<SpringConfig>
): {
  value$: BehaviorSubject<number>;
  formatted$: BehaviorSubject<string>;
  setBalance: (balance: number) => void;
  add: (amount: number) => void;
  subtract: (amount: number) => void;
  destroy: () => void;
} {
  const animated = new AnimatedValue(initialBalance, springConfig ?? SPRING_PRESETS.gentle);
  const formatted$ = new BehaviorSubject<string>(formatCurrency(initialBalance, { currency }));

  const subscription = animated.subscribe((state) => {
    formatted$.next(formatCurrency(state.value, { currency }));
  });

  return {
    value$: animated.state$ as unknown as BehaviorSubject<number>,
    formatted$,
    setBalance: (balance: number) => animated.springTo(balance),
    add: (amount: number) => animated.add(amount),
    subtract: (amount: number) => animated.add(-amount),
    destroy: () => {
      subscription.unsubscribe();
      animated.destroy();
    },
  };
}

// =============================================================================
// Color Utilities
// =============================================================================

/**
 * Get color for a value change
 */
export function getChangeColor(change: number): string {
  if (change > 0) return '#22c55e'; // Green
  if (change < 0) return '#ef4444'; // Red
  return '#6b7280'; // Gray
}

/**
 * Interpolate between two colors
 */
export function interpolateColor(
  color1: string,
  color2: string,
  factor: number
): string {
  const c1 = hexToRgb(color1);
  const c2 = hexToRgb(color2);

  if (!c1 || !c2) return color1;

  const r = Math.round(c1.r + (c2.r - c1.r) * factor);
  const g = Math.round(c1.g + (c2.g - c1.g) * factor);
  const b = Math.round(c1.b + (c2.b - c1.b) * factor);

  return `rgb(${r}, ${g}, ${b})`;
}

function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : null;
}

// =============================================================================
// Haptic Feedback (Placeholder for native integration)
// =============================================================================

/** Haptic feedback types */
export type HapticType =
  | 'light' // Light tap
  | 'medium' // Medium tap
  | 'heavy' // Heavy tap
  | 'success' // Success pattern
  | 'warning' // Warning pattern
  | 'error'; // Error pattern

/**
 * Trigger haptic feedback (requires native bridge)
 */
export function triggerHaptic(type: HapticType): void {
  // Check for native haptic support
  if (typeof navigator !== 'undefined' && 'vibrate' in navigator) {
    const patterns: Record<HapticType, number[]> = {
      light: [10],
      medium: [20],
      heavy: [50],
      success: [10, 50, 10],
      warning: [30, 50, 30],
      error: [50, 100, 50],
    };
    navigator.vibrate(patterns[type] ?? [10]);
  }
}

// =============================================================================
// Factory
// =============================================================================

/**
 * Create an animated value with default settings
 */
export function createAnimatedValue(
  initial: number = 0,
  preset: keyof typeof SPRING_PRESETS = 'default'
): AnimatedValue {
  return new AnimatedValue(initial, SPRING_PRESETS[preset]);
}
