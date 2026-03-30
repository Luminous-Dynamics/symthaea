// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Contacts & Favorites - Never Type an Address Twice
 *
 * Manages saved contacts, favorites, and recent recipients for
 * frictionless payments. Integrates with Identity resolver for
 * automatic profile enrichment.
 *
 * @example
 * ```typescript
 * const contacts = new ContactsManager(storage, identityResolver);
 *
 * // Add a contact
 * await contacts.add({
 *   address: '@alice',
 *   nickname: 'Alice from Work',
 *   favorite: true,
 * });
 *
 * // Quick access
 * const favorites = contacts.favorites;
 * const recent = contacts.recentRecipients;
 *
 * // Smart search
 * const results = await contacts.search('ali');
 * ```
 */

import { BehaviorSubject, type Subscription } from '../reactive/index.js';

import type { Identity, IdentityResolver } from './index.js';

// =============================================================================
// Types
// =============================================================================

/** Unique contact identifier */
export type ContactId = string & { readonly __brand: 'ContactId' };

/** Contact with enriched identity data */
export interface Contact {
  /** Unique ID */
  id: ContactId;
  /** The address (agent ID or @nickname) */
  address: string;
  /** Your custom name for this contact */
  nickname?: string;
  /** Custom emoji or avatar override */
  emoji?: string;
  /** Notes about this contact */
  notes?: string;
  /** Whether this is a favorite */
  favorite: boolean;
  /** Tags for organization */
  tags: string[];
  /** When added */
  createdAt: number;
  /** Last interaction timestamp */
  lastInteractionAt?: number;
  /** Number of transactions with this contact */
  transactionCount: number;
  /** Total amount sent to this contact */
  totalSent: number;
  /** Total amount received from this contact */
  totalReceived: number;
  /** Resolved identity (enriched from Identity hApp) */
  identity?: Identity;
}

/** Input for adding a contact */
export interface AddContactInput {
  address: string;
  nickname?: string;
  emoji?: string;
  notes?: string;
  favorite?: boolean;
  tags?: string[];
}

/** Input for updating a contact */
export interface UpdateContactInput {
  nickname?: string;
  emoji?: string;
  notes?: string;
  favorite?: boolean;
  tags?: string[];
}

/** Contact suggestion based on interaction patterns */
export interface ContactSuggestion {
  contact: Contact;
  reason: SuggestionReason;
  confidence: number; // 0-1
}

/** Why a contact is being suggested */
export type SuggestionReason =
  | 'frequent' // You interact with them often
  | 'recent' // You interacted recently
  | 'time_pattern' // You usually pay them at this time
  | 'amount_pattern' // Similar amount to previous
  | 'group_pattern'; // Often paid together with others

/** Storage interface for contacts */
export interface ContactsStorage {
  getAll(): Promise<Contact[]>;
  get(id: ContactId): Promise<Contact | null>;
  save(contact: Contact): Promise<void>;
  delete(id: ContactId): Promise<void>;
  clear(): Promise<void>;
}

/** Contacts state (reactive) */
export interface ContactsState {
  contacts: Contact[];
  favorites: Contact[];
  recentRecipients: Contact[];
  loading: boolean;
  error: string | null;
}

// =============================================================================
// In-Memory Storage (Default)
// =============================================================================

class InMemoryContactsStorage implements ContactsStorage {
  private contacts: Map<ContactId, Contact> = new Map();

  async getAll(): Promise<Contact[]> {
    return Array.from(this.contacts.values());
  }

  async get(id: ContactId): Promise<Contact | null> {
    return this.contacts.get(id) ?? null;
  }

  async save(contact: Contact): Promise<void> {
    this.contacts.set(contact.id, contact);
  }

  async delete(id: ContactId): Promise<void> {
    this.contacts.delete(id);
  }

  async clear(): Promise<void> {
    this.contacts.clear();
  }
}

// =============================================================================
// Contacts Manager
// =============================================================================

/**
 * Manages contacts, favorites, and recent recipients
 */
export class ContactsManager {
  private storage: ContactsStorage;
  private identityResolver?: IdentityResolver;
  private _state$: BehaviorSubject<ContactsState>;

  constructor(storage?: ContactsStorage, identityResolver?: IdentityResolver) {
    this.storage = storage ?? new InMemoryContactsStorage();
    this.identityResolver = identityResolver;
    this._state$ = new BehaviorSubject<ContactsState>({
      contacts: [],
      favorites: [],
      recentRecipients: [],
      loading: true,
      error: null,
    });
  }

  // ===========================================================================
  // Reactive State
  // ===========================================================================

  /** Observable state */
  get state$(): BehaviorSubject<ContactsState> {
    return this._state$;
  }

  /** Current state */
  get state(): ContactsState {
    return this._state$.value;
  }

  /** All contacts */
  get contacts(): Contact[] {
    return this._state$.value.contacts;
  }

  /** Favorite contacts */
  get favorites(): Contact[] {
    return this._state$.value.favorites;
  }

  /** Recent recipients (last 10) */
  get recentRecipients(): Contact[] {
    return this._state$.value.recentRecipients;
  }

  /** Subscribe to state changes */
  subscribe(observer: (state: ContactsState) => void): Subscription {
    return this._state$.subscribe(observer);
  }

  // ===========================================================================
  // Core Operations
  // ===========================================================================

  /**
   * Initialize and load contacts
   */
  async initialize(): Promise<void> {
    this._state$.next({ ...this._state$.value, loading: true });

    try {
      const contacts = await this.storage.getAll();

      // Enrich with identity data
      const enriched = await this.enrichContacts(contacts);

      this.updateState(enriched);
    } catch (error) {
      this._state$.next({
        ...this._state$.value,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to load contacts',
      });
    }
  }

  /**
   * Add a new contact
   */
  async add(input: AddContactInput): Promise<Contact> {
    // Check if already exists
    const existing = this._state$.value.contacts.find(
      (c) => c.address.toLowerCase() === input.address.toLowerCase()
    );
    if (existing) {
      throw new Error(`Contact with address ${input.address} already exists`);
    }

    // Resolve identity
    let identity: Identity | undefined;
    if (this.identityResolver) {
      try {
        identity = await this.identityResolver.resolve(input.address);
      } catch {
        // Identity resolution failed, that's ok
      }
    }

    const contact: Contact = {
      id: generateContactId(),
      address: input.address,
      nickname: input.nickname,
      emoji: input.emoji,
      notes: input.notes,
      favorite: input.favorite ?? false,
      tags: input.tags ?? [],
      createdAt: Date.now(),
      transactionCount: 0,
      totalSent: 0,
      totalReceived: 0,
      identity,
    };

    await this.storage.save(contact);
    this.updateState([...this._state$.value.contacts, contact]);

    return contact;
  }

  /**
   * Update an existing contact
   */
  async update(id: ContactId, input: UpdateContactInput): Promise<Contact> {
    const contact = await this.storage.get(id);
    if (!contact) {
      throw new Error(`Contact ${id} not found`);
    }

    const updated: Contact = {
      ...contact,
      ...(input.nickname !== undefined && { nickname: input.nickname }),
      ...(input.emoji !== undefined && { emoji: input.emoji }),
      ...(input.notes !== undefined && { notes: input.notes }),
      ...(input.favorite !== undefined && { favorite: input.favorite }),
      ...(input.tags !== undefined && { tags: input.tags }),
    };

    await this.storage.save(updated);
    this.updateState(
      this._state$.value.contacts.map((c) => (c.id === id ? updated : c))
    );

    return updated;
  }

  /**
   * Delete a contact
   */
  async delete(id: ContactId): Promise<void> {
    await this.storage.delete(id);
    this.updateState(this._state$.value.contacts.filter((c) => c.id !== id));
  }

  /**
   * Toggle favorite status
   */
  async toggleFavorite(id: ContactId): Promise<Contact> {
    const contact = this._state$.value.contacts.find((c) => c.id === id);
    if (!contact) {
      throw new Error(`Contact ${id} not found`);
    }
    return this.update(id, { favorite: !contact.favorite });
  }

  /**
   * Get contact by address
   */
  getByAddress(address: string): Contact | null {
    return (
      this._state$.value.contacts.find(
        (c) => c.address.toLowerCase() === address.toLowerCase()
      ) ?? null
    );
  }

  /**
   * Get contact by ID
   */
  getById(id: ContactId): Contact | null {
    return this._state$.value.contacts.find((c) => c.id === id) ?? null;
  }

  // ===========================================================================
  // Search & Suggestions
  // ===========================================================================

  /**
   * Search contacts by name, address, or tags
   */
  search(query: string): Contact[] {
    const lowerQuery = query.toLowerCase();
    return this._state$.value.contacts.filter((contact) => {
      // Search in nickname
      if (contact.nickname?.toLowerCase().includes(lowerQuery)) return true;
      // Search in address
      if (contact.address.toLowerCase().includes(lowerQuery)) return true;
      // Search in identity nickname
      if (contact.identity?.nickname?.toLowerCase().includes(lowerQuery)) return true;
      // Search in identity display name
      if (contact.identity?.displayName?.toLowerCase().includes(lowerQuery)) return true;
      // Search in tags
      if (contact.tags.some((t) => t.toLowerCase().includes(lowerQuery))) return true;
      // Search in notes
      if (contact.notes?.toLowerCase().includes(lowerQuery)) return true;
      return false;
    });
  }

  /**
   * Get smart suggestions based on context
   */
  getSuggestions(context?: {
    amount?: number;
    time?: Date;
    recentRecipients?: string[];
  }): ContactSuggestion[] {
    const suggestions: ContactSuggestion[] = [];
    const now = context?.time ?? new Date();
    const hour = now.getHours();

    for (const contact of this._state$.value.contacts) {
      let reason: SuggestionReason | null = null;
      let confidence = 0;

      // Frequent contacts
      if (contact.transactionCount >= 5) {
        reason = 'frequent';
        confidence = Math.min(contact.transactionCount / 20, 0.9);
      }

      // Recent contacts (last 7 days)
      if (contact.lastInteractionAt) {
        const daysSince = (Date.now() - contact.lastInteractionAt) / (1000 * 60 * 60 * 24);
        if (daysSince < 7) {
          if (!reason || confidence < 0.7) {
            reason = 'recent';
            confidence = Math.max(confidence, 0.7 - daysSince * 0.1);
          }
        }
      }

      // Time patterns (simplified - would need historical data)
      // Example: If you usually pay this contact in the morning
      if (contact.tags.includes('morning') && hour < 12) {
        if (!reason || confidence < 0.6) {
          reason = 'time_pattern';
          confidence = Math.max(confidence, 0.6);
        }
      }

      if (reason) {
        suggestions.push({ contact, reason, confidence });
      }
    }

    // Sort by confidence
    return suggestions.sort((a, b) => b.confidence - a.confidence).slice(0, 5);
  }

  /**
   * Get contacts by tag
   */
  getByTag(tag: string): Contact[] {
    return this._state$.value.contacts.filter((c) =>
      c.tags.some((t) => t.toLowerCase() === tag.toLowerCase())
    );
  }

  // ===========================================================================
  // Interaction Tracking
  // ===========================================================================

  /**
   * Record a transaction with a contact
   */
  async recordTransaction(
    address: string,
    amount: number,
    direction: 'sent' | 'received'
  ): Promise<void> {
    let contact = this.getByAddress(address);

    if (!contact) {
      // Auto-create contact for new recipients
      contact = await this.add({ address });
    }

    const updated: Contact = {
      ...contact,
      lastInteractionAt: Date.now(),
      transactionCount: contact.transactionCount + 1,
      totalSent: direction === 'sent' ? contact.totalSent + amount : contact.totalSent,
      totalReceived:
        direction === 'received' ? contact.totalReceived + amount : contact.totalReceived,
    };

    await this.storage.save(updated);
    this.updateState(
      this._state$.value.contacts.map((c) => (c.id === contact.id ? updated : c))
    );
  }

  // ===========================================================================
  // Import/Export
  // ===========================================================================

  /**
   * Export contacts for backup
   */
  export(): ExportedContacts {
    return {
      version: 1,
      exportedAt: Date.now(),
      contacts: this._state$.value.contacts.map((c) => ({
        address: c.address,
        nickname: c.nickname,
        emoji: c.emoji,
        notes: c.notes,
        favorite: c.favorite,
        tags: c.tags,
      })),
    };
  }

  /**
   * Import contacts from backup
   */
  async import(data: ExportedContacts, merge: boolean = true): Promise<number> {
    let imported = 0;

    for (const item of data.contacts) {
      const existing = this.getByAddress(item.address);

      if (existing && merge) {
        // Merge with existing
        await this.update(existing.id, {
          nickname: item.nickname ?? existing.nickname,
          emoji: item.emoji ?? existing.emoji,
          notes: item.notes ?? existing.notes,
          favorite: item.favorite || existing.favorite,
          tags: [...new Set([...existing.tags, ...item.tags])],
        });
        imported++;
      } else if (!existing) {
        // Add new
        await this.add(item);
        imported++;
      }
    }

    return imported;
  }

  // ===========================================================================
  // Private Helpers
  // ===========================================================================

  private async enrichContacts(contacts: Contact[]): Promise<Contact[]> {
    if (!this.identityResolver) return contacts;

    return Promise.all(
      contacts.map(async (contact) => {
        try {
          const identity = await this.identityResolver!.resolve(contact.address);
          return { ...contact, identity };
        } catch {
          return contact;
        }
      })
    );
  }

  private updateState(contacts: Contact[]): void {
    // Sort by last interaction
    const sorted = [...contacts].sort((a, b) => {
      // Favorites first
      if (a.favorite && !b.favorite) return -1;
      if (!a.favorite && b.favorite) return 1;
      // Then by last interaction
      return (b.lastInteractionAt ?? 0) - (a.lastInteractionAt ?? 0);
    });

    const favorites = sorted.filter((c) => c.favorite);
    const recentRecipients = sorted
      .filter((c) => c.lastInteractionAt)
      .slice(0, 10);

    this._state$.next({
      contacts: sorted,
      favorites,
      recentRecipients,
      loading: false,
      error: null,
    });
  }
}

// =============================================================================
// Export Format
// =============================================================================

export interface ExportedContacts {
  version: number;
  exportedAt: number;
  contacts: Array<{
    address: string;
    nickname?: string;
    emoji?: string;
    notes?: string;
    favorite: boolean;
    tags: string[];
  }>;
}

// =============================================================================
// Utilities
// =============================================================================

function generateContactId(): ContactId {
  return `contact-${Date.now()}-${Math.random().toString(36).slice(2, 8)}` as ContactId;
}

/**
 * Format contact display name
 */
export function formatContactName(contact: Contact): string {
  // User's custom nickname takes priority
  if (contact.nickname) return contact.nickname;
  // Then identity nickname
  if (contact.identity?.nickname) return contact.identity.nickname;
  // Then identity display name
  if (contact.identity?.displayName) return contact.identity.displayName;
  // Fallback to truncated address
  return truncateAddress(contact.address);
}

/**
 * Truncate address for display
 */
export function truncateAddress(address: string): string {
  if (address.startsWith('@')) return address;
  if (address.length <= 12) return address;
  return `${address.slice(0, 6)}...${address.slice(-4)}`;
}

/**
 * Get initials for avatar fallback
 */
export function getContactInitials(contact: Contact): string {
  const name = contact.nickname ?? contact.identity?.displayName ?? contact.address;
  if (name.startsWith('@')) return name.slice(1, 3).toUpperCase();
  return name.slice(0, 2).toUpperCase();
}

/**
 * Generate a color from address (for avatar background)
 */
export function getContactColor(contact: Contact): string {
  const colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
  ];

  let hash = 0;
  const str = contact.address;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }

  return colors[Math.abs(hash) % colors.length];
}

// =============================================================================
// Factory
// =============================================================================

/**
 * Create a contacts manager
 */
export function createContactsManager(
  storage?: ContactsStorage,
  identityResolver?: IdentityResolver
): ContactsManager {
  return new ContactsManager(storage, identityResolver);
}
