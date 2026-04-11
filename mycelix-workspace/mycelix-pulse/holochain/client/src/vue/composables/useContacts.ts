// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Vue Composable - useContacts
 *
 * Reactive contact management with trust integration.
 */

import { ref, computed, onMounted } from 'vue';
import type { Ref, ComputedRef } from 'vue';
import { getMycelixClient } from '../../bootstrap';

export interface Contact {
  id: string;
  email: string;
  name?: string;
  nickname?: string;
  organization?: string;
  phone?: string;
  avatarUrl?: string;
  trustLevel?: number;
  attestationCount?: number;
  tags: string[];
  favorite: boolean;
  blocked: boolean;
  lastContacted?: number;
  createdAt: number;
  updatedAt: number;
}

export interface ContactGroup {
  id: string;
  name: string;
  contacts: string[];
  color?: string;
}

export interface UseContactsReturn {
  // State
  contacts: Ref<Contact[]>;
  groups: Ref<ContactGroup[]>;
  loading: Ref<boolean>;
  error: Ref<string | null>;

  // Computed
  favoriteContacts: ComputedRef<Contact[]>;
  blockedContacts: ComputedRef<Contact[]>;
  recentContacts: ComputedRef<Contact[]>;

  // Actions
  refresh: () => Promise<void>;
  search: (query: string) => Contact[];
  getContact: (id: string) => Contact | undefined;
  getContactByEmail: (email: string) => Contact | undefined;
  createContact: (contact: Omit<Contact, 'id' | 'createdAt' | 'updatedAt'>) => Promise<string>;
  updateContact: (id: string, updates: Partial<Contact>) => Promise<void>;
  deleteContact: (id: string) => Promise<void>;
  toggleFavorite: (id: string) => Promise<void>;
  toggleBlock: (id: string) => Promise<void>;
  addTag: (id: string, tag: string) => Promise<void>;
  removeTag: (id: string, tag: string) => Promise<void>;

  // Group actions
  createGroup: (name: string, color?: string) => Promise<string>;
  deleteGroup: (id: string) => Promise<void>;
  addToGroup: (contactId: string, groupId: string) => Promise<void>;
  removeFromGroup: (contactId: string, groupId: string) => Promise<void>;

  // Import/Export
  importVCard: (vcard: string) => Promise<Contact[]>;
  exportVCard: (contactIds?: string[]) => Promise<string>;

  // Suggestions
  getSuggestions: (query: string, limit?: number) => Promise<Contact[]>;
}

export function useContacts(): UseContactsReturn {
  const contacts = ref<Contact[]>([]);
  const groups = ref<ContactGroup[]>([]);
  const loading = ref(false);
  const error = ref<string | null>(null);

  const favoriteContacts = computed(() =>
    contacts.value.filter((c) => c.favorite && !c.blocked)
  );

  const blockedContacts = computed(() =>
    contacts.value.filter((c) => c.blocked)
  );

  const recentContacts = computed(() =>
    [...contacts.value]
      .filter((c) => c.lastContacted && !c.blocked)
      .sort((a, b) => (b.lastContacted ?? 0) - (a.lastContacted ?? 0))
      .slice(0, 10)
  );

  async function refresh() {
    loading.value = true;
    error.value = null;

    try {
      const client = getMycelixClient();
      const services = client.getServices();

      const [contactList, groupList] = await Promise.all([
        services.contacts.getAllContacts(),
        services.contacts.getAllGroups(),
      ]);

      contacts.value = contactList;
      groups.value = groupList;
    } catch (e) {
      error.value = String(e);
    } finally {
      loading.value = false;
    }
  }

  function search(query: string): Contact[] {
    const q = query.toLowerCase();
    return contacts.value.filter(
      (c) =>
        c.name?.toLowerCase().includes(q) ||
        c.email.toLowerCase().includes(q) ||
        c.nickname?.toLowerCase().includes(q) ||
        c.organization?.toLowerCase().includes(q) ||
        c.tags.some((t) => t.toLowerCase().includes(q))
    );
  }

  function getContact(id: string): Contact | undefined {
    return contacts.value.find((c) => c.id === id);
  }

  function getContactByEmail(email: string): Contact | undefined {
    return contacts.value.find(
      (c) => c.email.toLowerCase() === email.toLowerCase()
    );
  }

  async function createContact(
    contact: Omit<Contact, 'id' | 'createdAt' | 'updatedAt'>
  ): Promise<string> {
    try {
      const client = getMycelixClient();
      const id = await client.getServices().contacts.createContact(contact);

      await refresh();
      return id;
    } catch (e) {
      error.value = String(e);
      throw e;
    }
  }

  async function updateContact(id: string, updates: Partial<Contact>): Promise<void> {
    try {
      const client = getMycelixClient();
      await client.getServices().contacts.updateContact(id, updates);

      const contact = contacts.value.find((c) => c.id === id);
      if (contact) {
        Object.assign(contact, updates);
      }
    } catch (e) {
      error.value = String(e);
      throw e;
    }
  }

  async function deleteContact(id: string): Promise<void> {
    try {
      const client = getMycelixClient();
      await client.getServices().contacts.deleteContact(id);

      contacts.value = contacts.value.filter((c) => c.id !== id);
    } catch (e) {
      error.value = String(e);
      throw e;
    }
  }

  async function toggleFavorite(id: string): Promise<void> {
    const contact = contacts.value.find((c) => c.id === id);
    if (contact) {
      await updateContact(id, { favorite: !contact.favorite });
    }
  }

  async function toggleBlock(id: string): Promise<void> {
    const contact = contacts.value.find((c) => c.id === id);
    if (contact) {
      await updateContact(id, { blocked: !contact.blocked });
    }
  }

  async function addTag(id: string, tag: string): Promise<void> {
    const contact = contacts.value.find((c) => c.id === id);
    if (contact && !contact.tags.includes(tag)) {
      await updateContact(id, { tags: [...contact.tags, tag] });
    }
  }

  async function removeTag(id: string, tag: string): Promise<void> {
    const contact = contacts.value.find((c) => c.id === id);
    if (contact) {
      await updateContact(id, { tags: contact.tags.filter((t) => t !== tag) });
    }
  }

  async function createGroup(name: string, color?: string): Promise<string> {
    try {
      const client = getMycelixClient();
      const id = await client.getServices().contacts.createGroup({ name, color });

      await refresh();
      return id;
    } catch (e) {
      error.value = String(e);
      throw e;
    }
  }

  async function deleteGroup(id: string): Promise<void> {
    try {
      const client = getMycelixClient();
      await client.getServices().contacts.deleteGroup(id);

      groups.value = groups.value.filter((g) => g.id !== id);
    } catch (e) {
      error.value = String(e);
      throw e;
    }
  }

  async function addToGroup(contactId: string, groupId: string): Promise<void> {
    try {
      const client = getMycelixClient();
      await client.getServices().contacts.addToGroup(contactId, groupId);

      const group = groups.value.find((g) => g.id === groupId);
      if (group && !group.contacts.includes(contactId)) {
        group.contacts = [...group.contacts, contactId];
      }
    } catch (e) {
      error.value = String(e);
      throw e;
    }
  }

  async function removeFromGroup(contactId: string, groupId: string): Promise<void> {
    try {
      const client = getMycelixClient();
      await client.getServices().contacts.removeFromGroup(contactId, groupId);

      const group = groups.value.find((g) => g.id === groupId);
      if (group) {
        group.contacts = group.contacts.filter((id) => id !== contactId);
      }
    } catch (e) {
      error.value = String(e);
      throw e;
    }
  }

  async function importVCard(vcard: string): Promise<Contact[]> {
    try {
      const client = getMycelixClient();
      const imported = await client.getServices().contacts.importVCard(vcard);

      await refresh();
      return imported;
    } catch (e) {
      error.value = String(e);
      throw e;
    }
  }

  async function exportVCard(contactIds?: string[]): Promise<string> {
    try {
      const client = getMycelixClient();
      return await client.getServices().contacts.exportVCard(contactIds);
    } catch (e) {
      error.value = String(e);
      throw e;
    }
  }

  async function getSuggestions(query: string, limit = 5): Promise<Contact[]> {
    try {
      const client = getMycelixClient();
      return await client.getServices().contacts.getSuggestions(query, limit);
    } catch (e) {
      error.value = String(e);
      return [];
    }
  }

  onMounted(() => {
    refresh();
  });

  return {
    contacts,
    groups,
    loading,
    error,
    favoriteContacts,
    blockedContacts,
    recentContacts,
    refresh,
    search,
    getContact,
    getContactByEmail,
    createContact,
    updateContact,
    deleteContact,
    toggleFavorite,
    toggleBlock,
    addTag,
    removeTag,
    createGroup,
    deleteGroup,
    addToGroup,
    removeFromGroup,
    importVCard,
    exportVCard,
    getSuggestions,
  };
}

export default useContacts;
