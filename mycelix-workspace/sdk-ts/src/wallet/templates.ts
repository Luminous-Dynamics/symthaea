// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Transaction Templates - One-Tap Payments for Recurring Use
 *
 * Save and reuse common payment configurations like "Weekly Rent",
 * "Morning Coffee", or "Monthly Subscription". Templates can be
 * scheduled for automatic execution.
 *
 * @example
 * ```typescript
 * const templates = new TemplateManager(storage);
 *
 * // Create a template
 * const coffeeTemplate = await templates.create({
 *   name: 'Morning Coffee',
 *   emoji: '\u2615',
 *   to: '@local-cafe',
 *   amount: 5.50,
 *   currency: 'MYC',
 * });
 *
 * // Execute with one tap
 * const tx = await templates.execute(coffeeTemplate.id, wallet);
 *
 * // Schedule recurring
 * await templates.schedule(coffeeTemplate.id, {
 *   type: 'daily',
 *   time: '08:00',
 * });
 * ```
 */

import { BehaviorSubject, type Subscription } from '../reactive/index.js';

// =============================================================================
// Types
// =============================================================================

/** Unique template identifier */
export type TemplateId = string & { readonly __brand: 'TemplateId' };

/** Transaction template definition */
export interface Template {
  /** Unique ID */
  id: TemplateId;
  /** Display name */
  name: string;
  /** Emoji/icon */
  emoji?: string;
  /** Recipient address or @nickname */
  to: string;
  /** Amount to send */
  amount: number;
  /** Currency */
  currency: string;
  /** Optional memo */
  memo?: string;
  /** Category for organization */
  category?: TemplateCategory;
  /** Custom tags */
  tags: string[];
  /** Number of times used */
  usageCount: number;
  /** Last used timestamp */
  lastUsedAt?: number;
  /** When created */
  createdAt: number;
  /** Schedule configuration */
  schedule?: ScheduleConfig;
  /** Whether template is active */
  isActive: boolean;
  /** Color for UI */
  color?: string;
}

/** Template categories */
export type TemplateCategory =
  | 'food' // Coffee, lunch, groceries
  | 'transport' // Gas, parking, transit
  | 'bills' // Rent, utilities, subscriptions
  | 'shopping' // Online purchases
  | 'entertainment' // Games, streaming
  | 'donations' // Charities, tips
  | 'savings' // Regular savings
  | 'business' // Business expenses
  | 'personal' // Personal transfers
  | 'other';

/** Schedule configuration */
export interface ScheduleConfig {
  /** Schedule type */
  type: ScheduleType;
  /** Time of day (HH:MM) for daily/weekly */
  time?: string;
  /** Day of week (0-6) for weekly */
  dayOfWeek?: number;
  /** Day of month (1-31) for monthly */
  dayOfMonth?: number;
  /** Start date */
  startDate?: number;
  /** End date (optional) */
  endDate?: number;
  /** Next scheduled execution */
  nextExecutionAt?: number;
  /** Whether currently enabled */
  enabled: boolean;
}

/** Schedule types */
export type ScheduleType =
  | 'once' // One-time scheduled
  | 'daily' // Every day
  | 'weekly' // Once a week
  | 'biweekly' // Every two weeks
  | 'monthly' // Once a month
  | 'custom'; // Custom interval

/** Input for creating a template */
export interface CreateTemplateInput {
  name: string;
  emoji?: string;
  to: string;
  amount: number;
  currency?: string;
  memo?: string;
  category?: TemplateCategory;
  tags?: string[];
  color?: string;
}

/** Input for updating a template */
export interface UpdateTemplateInput {
  name?: string;
  emoji?: string;
  to?: string;
  amount?: number;
  currency?: string;
  memo?: string;
  category?: TemplateCategory;
  tags?: string[];
  color?: string;
  isActive?: boolean;
}

/** Template execution result */
export interface TemplateExecutionResult {
  success: boolean;
  templateId: TemplateId;
  transactionId?: string;
  error?: string;
  executedAt: number;
}

/** Storage interface */
export interface TemplateStorage {
  getAll(): Promise<Template[]>;
  get(id: TemplateId): Promise<Template | null>;
  save(template: Template): Promise<void>;
  delete(id: TemplateId): Promise<void>;
  clear(): Promise<void>;
}

/** Wallet interface for execution */
export interface TemplateWallet {
  send(
    to: string,
    amount: number,
    currency: string,
    options?: { memo?: string }
  ): Promise<{ id: string }>;
}

/** Template state (reactive) */
export interface TemplateState {
  templates: Template[];
  recentTemplates: Template[];
  scheduledTemplates: Template[];
  loading: boolean;
  error: string | null;
}

// =============================================================================
// In-Memory Storage
// =============================================================================

class InMemoryTemplateStorage implements TemplateStorage {
  private templates: Map<TemplateId, Template> = new Map();

  async getAll(): Promise<Template[]> {
    return Array.from(this.templates.values());
  }

  async get(id: TemplateId): Promise<Template | null> {
    return this.templates.get(id) ?? null;
  }

  async save(template: Template): Promise<void> {
    this.templates.set(template.id, template);
  }

  async delete(id: TemplateId): Promise<void> {
    this.templates.delete(id);
  }

  async clear(): Promise<void> {
    this.templates.clear();
  }
}

// =============================================================================
// Template Manager
// =============================================================================

/**
 * Manages transaction templates and schedules
 */
export class TemplateManager {
  private storage: TemplateStorage;
  private _state$: BehaviorSubject<TemplateState>;
  private scheduleCheckInterval?: ReturnType<typeof setInterval>;
  private executionListeners: Array<(result: TemplateExecutionResult) => void> = [];

  constructor(storage?: TemplateStorage) {
    this.storage = storage ?? new InMemoryTemplateStorage();
    this._state$ = new BehaviorSubject<TemplateState>({
      templates: [],
      recentTemplates: [],
      scheduledTemplates: [],
      loading: true,
      error: null,
    });
  }

  // ===========================================================================
  // Reactive State
  // ===========================================================================

  /** Observable state */
  get state$(): BehaviorSubject<TemplateState> {
    return this._state$;
  }

  /** Current state */
  get state(): TemplateState {
    return this._state$.value;
  }

  /** All templates */
  get templates(): Template[] {
    return this._state$.value.templates;
  }

  /** Recently used templates */
  get recentTemplates(): Template[] {
    return this._state$.value.recentTemplates;
  }

  /** Scheduled templates */
  get scheduledTemplates(): Template[] {
    return this._state$.value.scheduledTemplates;
  }

  /** Subscribe to state changes */
  subscribe(observer: (state: TemplateState) => void): Subscription {
    return this._state$.subscribe(observer);
  }

  /** Subscribe to execution events */
  onExecution(listener: (result: TemplateExecutionResult) => void): () => void {
    this.executionListeners.push(listener);
    return () => {
      const index = this.executionListeners.indexOf(listener);
      if (index !== -1) this.executionListeners.splice(index, 1);
    };
  }

  // ===========================================================================
  // Lifecycle
  // ===========================================================================

  /**
   * Initialize and load templates
   */
  async initialize(): Promise<void> {
    this._state$.next({ ...this._state$.value, loading: true });

    try {
      const templates = await this.storage.getAll();
      this.updateState(templates);

      // Start schedule checker (every minute)
      this.scheduleCheckInterval = setInterval(() => {
        void this.checkSchedules();
      }, 60000);
    } catch (error) {
      this._state$.next({
        ...this._state$.value,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to load templates',
      });
    }
  }

  /**
   * Cleanup
   */
  destroy(): void {
    if (this.scheduleCheckInterval) {
      clearInterval(this.scheduleCheckInterval);
    }
    this.executionListeners = [];
  }

  // ===========================================================================
  // CRUD Operations
  // ===========================================================================

  /**
   * Create a new template
   */
  async create(input: CreateTemplateInput): Promise<Template> {
    const template: Template = {
      id: generateTemplateId(),
      name: input.name,
      emoji: input.emoji,
      to: input.to,
      amount: input.amount,
      currency: input.currency ?? 'MYC',
      memo: input.memo,
      category: input.category,
      tags: input.tags ?? [],
      usageCount: 0,
      createdAt: Date.now(),
      isActive: true,
      color: input.color,
    };

    await this.storage.save(template);
    this.updateState([...this._state$.value.templates, template]);

    return template;
  }

  /**
   * Update an existing template
   */
  async update(id: TemplateId, input: UpdateTemplateInput): Promise<Template> {
    const template = await this.storage.get(id);
    if (!template) {
      throw new Error(`Template ${id} not found`);
    }

    const updated: Template = {
      ...template,
      ...(input.name !== undefined && { name: input.name }),
      ...(input.emoji !== undefined && { emoji: input.emoji }),
      ...(input.to !== undefined && { to: input.to }),
      ...(input.amount !== undefined && { amount: input.amount }),
      ...(input.currency !== undefined && { currency: input.currency }),
      ...(input.memo !== undefined && { memo: input.memo }),
      ...(input.category !== undefined && { category: input.category }),
      ...(input.tags !== undefined && { tags: input.tags }),
      ...(input.color !== undefined && { color: input.color }),
      ...(input.isActive !== undefined && { isActive: input.isActive }),
    };

    await this.storage.save(updated);
    this.updateState(this._state$.value.templates.map((t) => (t.id === id ? updated : t)));

    return updated;
  }

  /**
   * Delete a template
   */
  async delete(id: TemplateId): Promise<void> {
    await this.storage.delete(id);
    this.updateState(this._state$.value.templates.filter((t) => t.id !== id));
  }

  /**
   * Get a template by ID
   */
  get(id: TemplateId): Template | null {
    return this._state$.value.templates.find((t) => t.id === id) ?? null;
  }

  /**
   * Get templates by category
   */
  getByCategory(category: TemplateCategory): Template[] {
    return this._state$.value.templates.filter((t) => t.category === category);
  }

  /**
   * Search templates
   */
  search(query: string): Template[] {
    const lowerQuery = query.toLowerCase();
    return this._state$.value.templates.filter((t) => {
      if (t.name.toLowerCase().includes(lowerQuery)) return true;
      if (t.to.toLowerCase().includes(lowerQuery)) return true;
      if (t.memo?.toLowerCase().includes(lowerQuery)) return true;
      if (t.tags.some((tag) => tag.toLowerCase().includes(lowerQuery))) return true;
      return false;
    });
  }

  // ===========================================================================
  // Execution
  // ===========================================================================

  /**
   * Execute a template (send the transaction)
   */
  async execute(id: TemplateId, wallet: TemplateWallet): Promise<TemplateExecutionResult> {
    const template = this.get(id);
    if (!template) {
      return {
        success: false,
        templateId: id,
        error: 'Template not found',
        executedAt: Date.now(),
      };
    }

    if (!template.isActive) {
      return {
        success: false,
        templateId: id,
        error: 'Template is inactive',
        executedAt: Date.now(),
      };
    }

    try {
      const result = await wallet.send(template.to, template.amount, template.currency, {
        memo: template.memo,
      });

      // Update usage stats
      const updated: Template = {
        ...template,
        usageCount: template.usageCount + 1,
        lastUsedAt: Date.now(),
      };
      await this.storage.save(updated);
      this.updateState(this._state$.value.templates.map((t) => (t.id === id ? updated : t)));

      const executionResult: TemplateExecutionResult = {
        success: true,
        templateId: id,
        transactionId: result.id,
        executedAt: Date.now(),
      };

      this.emitExecution(executionResult);
      return executionResult;
    } catch (error) {
      const executionResult: TemplateExecutionResult = {
        success: false,
        templateId: id,
        error: error instanceof Error ? error.message : 'Execution failed',
        executedAt: Date.now(),
      };

      this.emitExecution(executionResult);
      return executionResult;
    }
  }

  /**
   * Preview a template execution (dry run)
   */
  preview(id: TemplateId): { to: string; amount: number; currency: string; memo?: string } | null {
    const template = this.get(id);
    if (!template) return null;

    return {
      to: template.to,
      amount: template.amount,
      currency: template.currency,
      memo: template.memo,
    };
  }

  // ===========================================================================
  // Scheduling
  // ===========================================================================

  /**
   * Schedule a template for automatic execution
   */
  async schedule(id: TemplateId, config: Omit<ScheduleConfig, 'nextExecutionAt' | 'enabled'>): Promise<Template> {
    const template = this.get(id);
    if (!template) {
      throw new Error(`Template ${id} not found`);
    }

    const nextExecution = calculateNextExecution(config);
    const schedule: ScheduleConfig = {
      ...config,
      nextExecutionAt: nextExecution,
      enabled: true,
    };

    const updated: Template = {
      ...template,
      schedule,
    };

    await this.storage.save(updated);
    this.updateState(this._state$.value.templates.map((t) => (t.id === id ? updated : t)));

    return updated;
  }

  /**
   * Enable or disable a schedule
   */
  async setScheduleEnabled(id: TemplateId, enabled: boolean): Promise<Template> {
    const template = this.get(id);
    if (!template || !template.schedule) {
      throw new Error(`Template ${id} not found or has no schedule`);
    }

    const updated: Template = {
      ...template,
      schedule: {
        ...template.schedule,
        enabled,
        nextExecutionAt: enabled ? calculateNextExecution(template.schedule) : undefined,
      },
    };

    await this.storage.save(updated);
    this.updateState(this._state$.value.templates.map((t) => (t.id === id ? updated : t)));

    return updated;
  }

  /**
   * Remove schedule from template
   */
  async unschedule(id: TemplateId): Promise<Template> {
    const template = this.get(id);
    if (!template) {
      throw new Error(`Template ${id} not found`);
    }

    const updated: Template = {
      ...template,
      schedule: undefined,
    };

    await this.storage.save(updated);
    this.updateState(this._state$.value.templates.map((t) => (t.id === id ? updated : t)));

    return updated;
  }

  /**
   * Check schedules and execute due templates
   */
  private async checkSchedules(): Promise<void> {
    const now = Date.now();
    const dueTemplates = this._state$.value.templates.filter(
      (t) => t.schedule?.enabled && t.schedule.nextExecutionAt && t.schedule.nextExecutionAt <= now
    );

    // Note: Actual execution would need a wallet reference
    // This is typically handled by the app layer
    for (const template of dueTemplates) {
      if (template.schedule) {
        // Update next execution time
        const nextExecution = calculateNextExecution(template.schedule, now);
        const updated: Template = {
          ...template,
          schedule: {
            ...template.schedule,
            nextExecutionAt: nextExecution,
          },
        };
        await this.storage.save(updated);
        this.updateState(this._state$.value.templates.map((t) => (t.id === template.id ? updated : t)));
      }
    }
  }

  // ===========================================================================
  // Import/Export
  // ===========================================================================

  /**
   * Export templates for backup
   */
  export(): ExportedTemplates {
    return {
      version: 1,
      exportedAt: Date.now(),
      templates: this._state$.value.templates.map((t) => ({
        name: t.name,
        emoji: t.emoji,
        to: t.to,
        amount: t.amount,
        currency: t.currency,
        memo: t.memo,
        category: t.category,
        tags: t.tags,
        color: t.color,
      })),
    };
  }

  /**
   * Import templates from backup
   */
  async import(data: ExportedTemplates, merge: boolean = true): Promise<number> {
    let imported = 0;

    for (const item of data.templates) {
      // Check for duplicate by name and recipient
      const existing = this._state$.value.templates.find(
        (t) => t.name === item.name && t.to === item.to
      );

      if (existing && merge) {
        // Update existing
        await this.update(existing.id, {
          amount: item.amount,
          currency: item.currency,
          memo: item.memo,
          category: item.category,
          tags: item.tags,
          color: item.color,
        });
        imported++;
      } else if (!existing) {
        // Create new
        await this.create(item);
        imported++;
      }
    }

    return imported;
  }

  // ===========================================================================
  // Quick Templates
  // ===========================================================================

  /**
   * Create a quick template from a recent transaction
   */
  async createFromTransaction(
    name: string,
    transaction: { to: string; amount: number; currency: string; memo?: string }
  ): Promise<Template> {
    return this.create({
      name,
      to: transaction.to,
      amount: transaction.amount,
      currency: transaction.currency,
      memo: transaction.memo,
    });
  }

  /**
   * Suggest template name based on recipient
   */
  suggestName(to: string): string {
    // If it's a nickname, use it
    if (to.startsWith('@')) {
      const nickname = to.slice(1);
      return `Payment to ${nickname.charAt(0).toUpperCase() + nickname.slice(1)}`;
    }
    return 'New Template';
  }

  // ===========================================================================
  // Private Helpers
  // ===========================================================================

  private updateState(templates: Template[]): void {
    // Sort by usage count (most used first)
    const sorted = [...templates].sort((a, b) => {
      // Active templates first
      if (a.isActive && !b.isActive) return -1;
      if (!a.isActive && b.isActive) return 1;
      // Then by usage
      return b.usageCount - a.usageCount;
    });

    // Recent templates (used in last 30 days)
    const thirtyDaysAgo = Date.now() - 30 * 24 * 60 * 60 * 1000;
    const recentTemplates = sorted
      .filter((t) => t.lastUsedAt && t.lastUsedAt > thirtyDaysAgo)
      .slice(0, 10);

    // Scheduled templates
    const scheduledTemplates = sorted.filter((t) => t.schedule?.enabled);

    this._state$.next({
      templates: sorted,
      recentTemplates,
      scheduledTemplates,
      loading: false,
      error: null,
    });
  }

  private emitExecution(result: TemplateExecutionResult): void {
    for (const listener of this.executionListeners) {
      try {
        listener(result);
      } catch {
        // Ignore listener errors
      }
    }
  }
}

// =============================================================================
// Export Format
// =============================================================================

export interface ExportedTemplates {
  version: number;
  exportedAt: number;
  templates: Array<{
    name: string;
    emoji?: string;
    to: string;
    amount: number;
    currency?: string;
    memo?: string;
    category?: TemplateCategory;
    tags?: string[];
    color?: string;
  }>;
}

// =============================================================================
// Schedule Calculation
// =============================================================================

function calculateNextExecution(config: Omit<ScheduleConfig, 'nextExecutionAt' | 'enabled'>, fromTime?: number): number | undefined {
  const now = fromTime ?? Date.now();
  const today = new Date(now);

  // Parse time
  let hours = 9,
    minutes = 0;
  if (config.time) {
    const [h, m] = config.time.split(':').map(Number);
    hours = h ?? 9;
    minutes = m ?? 0;
  }

  let next: Date;

  switch (config.type) {
    case 'once':
      // Already scheduled for a specific time
      return config.startDate;

    case 'daily':
      next = new Date(today);
      next.setHours(hours, minutes, 0, 0);
      if (next.getTime() <= now) {
        next.setDate(next.getDate() + 1);
      }
      return next.getTime();

    case 'weekly': {
      next = new Date(today);
      next.setHours(hours, minutes, 0, 0);
      const targetDay = config.dayOfWeek ?? 1; // Monday default
      const currentDay = next.getDay();
      let daysUntil = targetDay - currentDay;
      if (daysUntil < 0 || (daysUntil === 0 && next.getTime() <= now)) {
        daysUntil += 7;
      }
      next.setDate(next.getDate() + daysUntil);
      return next.getTime();
    }

    case 'biweekly': {
      next = new Date(today);
      next.setHours(hours, minutes, 0, 0);
      const biweeklyDay = config.dayOfWeek ?? 1;
      const currentDayBi = next.getDay();
      let daysUntilBi = biweeklyDay - currentDayBi;
      if (daysUntilBi < 0 || (daysUntilBi === 0 && next.getTime() <= now)) {
        daysUntilBi += 14;
      }
      next.setDate(next.getDate() + daysUntilBi);
      return next.getTime();
    }

    case 'monthly': {
      next = new Date(today);
      next.setHours(hours, minutes, 0, 0);
      const targetDate = config.dayOfMonth ?? 1;
      next.setDate(targetDate);
      if (next.getTime() <= now) {
        next.setMonth(next.getMonth() + 1);
      }
      return next.getTime();
    }

    default:
      return undefined;
  }
}

// =============================================================================
// Utilities
// =============================================================================

function generateTemplateId(): TemplateId {
  return `tpl-${Date.now()}-${Math.random().toString(36).slice(2, 8)}` as TemplateId;
}

/**
 * Get emoji for a category
 */
export function getCategoryEmoji(category: TemplateCategory): string {
  const emojis: Record<TemplateCategory, string> = {
    food: '\uD83C\uDF55',
    transport: '\uD83D\uDE97',
    bills: '\uD83D\uDCDD',
    shopping: '\uD83D\uDED2',
    entertainment: '\uD83C\uDFAE',
    donations: '\uD83D\uDC96',
    savings: '\uD83D\uDCB0',
    business: '\uD83D\uDCBC',
    personal: '\uD83D\uDC64',
    other: '\uD83D\uDCE6',
  };
  return emojis[category] ?? '\uD83D\uDCE6';
}

/**
 * Get display name for a category
 */
export function getCategoryName(category: TemplateCategory): string {
  const names: Record<TemplateCategory, string> = {
    food: 'Food & Drinks',
    transport: 'Transport',
    bills: 'Bills & Utilities',
    shopping: 'Shopping',
    entertainment: 'Entertainment',
    donations: 'Donations & Tips',
    savings: 'Savings',
    business: 'Business',
    personal: 'Personal',
    other: 'Other',
  };
  return names[category] ?? 'Other';
}

/**
 * Get schedule description
 */
export function formatSchedule(schedule: ScheduleConfig): string {
  if (!schedule.enabled) return 'Paused';

  const time = schedule.time ?? '09:00';

  switch (schedule.type) {
    case 'once':
      return schedule.startDate
        ? `Once on ${new Date(schedule.startDate).toLocaleDateString()}`
        : 'Once';
    case 'daily':
      return `Daily at ${time}`;
    case 'weekly': {
      const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
      return `Every ${days[schedule.dayOfWeek ?? 1]} at ${time}`;
    }
    case 'biweekly':
      return `Every 2 weeks at ${time}`;
    case 'monthly':
      return `Monthly on day ${schedule.dayOfMonth ?? 1} at ${time}`;
    default:
      return 'Custom schedule';
  }
}

// =============================================================================
// Factory
// =============================================================================

/**
 * Create a template manager
 */
export function createTemplateManager(storage?: TemplateStorage): TemplateManager {
  return new TemplateManager(storage);
}
