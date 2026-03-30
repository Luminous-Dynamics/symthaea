// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Contacts Page
 *
 * Contact management with trust score integration
 */

import React, { useState, useCallback, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useI18n } from '../lib/i18n';
import { useAccessibility, useKeyboardShortcut } from '../lib/a11y/AccessibilityProvider';
import { graphqlClient } from '../lib/api/graphql-client';
import { useTrustScore } from '../lib/api/holochain-client';

// Types
interface Contact {
  id: string;
  email: string;
  displayName?: string;
  avatarUrl?: string;
  holochainAgentId?: string;
  trustScore: number;
  interactionCount: number;
  lastInteractionAt?: string;
  isBlocked: boolean;
  isFavorite?: boolean;
  notes?: string;
  tags: string[];
}

// Trust Badge Component
function TrustBadge({ score, size = 'medium' }: { score: number; size?: 'small' | 'medium' | 'large' }) {
  const getColor = (s: number) => {
    if (s >= 0.8) return '#4CAF50';
    if (s >= 0.5) return '#FFC107';
    if (s >= 0.2) return '#FF9800';
    return '#F44336';
  };

  const getLabel = (s: number) => {
    if (s >= 0.8) return 'Trusted';
    if (s >= 0.5) return 'Known';
    if (s >= 0.2) return 'Caution';
    return 'Unknown';
  };

  const sizes = { small: 20, medium: 28, large: 36 };
  const badgeSize = sizes[size];

  return (
    <div
      role="img"
      aria-label={`Trust: ${getLabel(score)} (${Math.round(score * 100)}%)`}
      style={{
        width: badgeSize,
        height: badgeSize,
        borderRadius: '50%',
        backgroundColor: getColor(score),
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'white',
        fontSize: badgeSize * 0.5,
        fontWeight: 'bold',
      }}
    >
      {score >= 0.8 ? '✓' : score >= 0.5 ? '~' : score >= 0.2 ? '!' : '?'}
    </div>
  );
}

// Contact Card Component
function ContactCard({
  contact,
  onEdit,
  onDelete,
  onBlock,
  onFavorite,
}: {
  contact: Contact;
  onEdit: (id: string) => void;
  onDelete: (id: string) => void;
  onBlock: (id: string, blocked: boolean) => void;
  onFavorite: (id: string) => void;
}) {
  const { t, formatRelativeTime } = useI18n();

  // Get live trust score from Holochain if agent ID available
  const { data: liveScore } = useTrustScore(contact.holochainAgentId || contact.email);
  const trustScore = liveScore?.score ?? contact.trustScore;

  return (
    <article
      className="contact-card"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '16px',
        padding: '16px',
        borderRadius: '8px',
        backgroundColor: contact.isBlocked ? '#ffebee' : '#fff',
        border: '1px solid #e0e0e0',
        opacity: contact.isBlocked ? 0.7 : 1,
      }}
    >
      {/* Avatar */}
      <div
        style={{
          width: 48,
          height: 48,
          borderRadius: '50%',
          backgroundColor: '#e0e0e0',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '20px',
          fontWeight: 'bold',
          color: '#666',
          overflow: 'hidden',
        }}
      >
        {contact.avatarUrl ? (
          <img src={contact.avatarUrl} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        ) : (
          (contact.displayName || contact.email)[0].toUpperCase()
        )}
      </div>

      {/* Info */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 600 }}>
            {contact.displayName || contact.email}
          </h3>
          {contact.isFavorite && <span aria-label="Favorite">⭐</span>}
          {contact.isBlocked && (
            <span style={{ color: '#d32f2f', fontSize: '12px' }}>({t('contacts.blocked')})</span>
          )}
        </div>
        <p style={{ margin: '4px 0 0', fontSize: '14px', color: '#666' }}>{contact.email}</p>
        {contact.lastInteractionAt && (
          <p style={{ margin: '4px 0 0', fontSize: '12px', color: '#999' }}>
            Last contact: {formatRelativeTime(contact.lastInteractionAt)}
          </p>
        )}
        {contact.tags.length > 0 && (
          <div style={{ display: 'flex', gap: '4px', marginTop: '8px', flexWrap: 'wrap' }}>
            {contact.tags.map((tag) => (
              <span
                key={tag}
                style={{
                  backgroundColor: '#e3f2fd',
                  padding: '2px 8px',
                  borderRadius: '12px',
                  fontSize: '11px',
                }}
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Trust Badge */}
      <TrustBadge score={trustScore} size="medium" />

      {/* Actions */}
      <div style={{ display: 'flex', gap: '8px' }}>
        <button
          onClick={() => onFavorite(contact.id)}
          aria-label={contact.isFavorite ? t('contacts.unfavorite') : t('contacts.favorite')}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: '18px',
            padding: '4px',
          }}
        >
          {contact.isFavorite ? '⭐' : '☆'}
        </button>
        <button
          onClick={() => onEdit(contact.id)}
          aria-label={t('contacts.editContact')}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: '16px',
            padding: '4px',
          }}
        >
          ✏️
        </button>
        <button
          onClick={() => onBlock(contact.id, !contact.isBlocked)}
          aria-label={contact.isBlocked ? t('contacts.unblock') : t('contacts.block')}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: '16px',
            padding: '4px',
          }}
        >
          {contact.isBlocked ? '🔓' : '🚫'}
        </button>
        <button
          onClick={() => onDelete(contact.id)}
          aria-label={t('contacts.deleteContact')}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: '16px',
            padding: '4px',
            color: '#d32f2f',
          }}
        >
          🗑️
        </button>
      </div>
    </article>
  );
}

// Add Contact Modal
function AddContactModal({
  isOpen,
  onClose,
  onAdd,
}: {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (data: { email: string; displayName?: string; tags?: string[] }) => void;
}) {
  const { t } = useI18n();
  const { focusTrap } = useAccessibility();
  const [email, setEmail] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [tags, setTags] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onAdd({
      email,
      displayName: displayName || undefined,
      tags: tags ? tags.split(',').map((t) => t.trim()) : undefined,
    });
    setEmail('');
    setDisplayName('');
    setTags('');
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="add-contact-title"
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0,0,0,0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={onClose}
    >
      <div
        ref={(el) => el && focusTrap(el)}
        onClick={(e) => e.stopPropagation()}
        style={{
          backgroundColor: 'white',
          borderRadius: '8px',
          padding: '24px',
          width: '400px',
          maxWidth: '90vw',
        }}
      >
        <h2 id="add-contact-title" style={{ margin: '0 0 16px' }}>
          {t('contacts.addContact')}
        </h2>
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: '16px' }}>
            <label htmlFor="contact-email" style={{ display: 'block', marginBottom: '4px' }}>
              Email *
            </label>
            <input
              id="contact-email"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
            />
          </div>
          <div style={{ marginBottom: '16px' }}>
            <label htmlFor="contact-name" style={{ display: 'block', marginBottom: '4px' }}>
              Display Name
            </label>
            <input
              id="contact-name"
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
            />
          </div>
          <div style={{ marginBottom: '16px' }}>
            <label htmlFor="contact-tags" style={{ display: 'block', marginBottom: '4px' }}>
              Tags (comma-separated)
            </label>
            <input
              id="contact-tags"
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="work, family, friends"
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
            />
          </div>
          <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
            <button type="button" onClick={onClose} style={{ padding: '8px 16px' }}>
              {t('common.cancel')}
            </button>
            <button
              type="submit"
              style={{
                padding: '8px 16px',
                backgroundColor: '#4CAF50',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              {t('contacts.addContact')}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// Edit Contact Modal
function EditContactModal({
  isOpen,
  contact,
  onClose,
  onSave,
}: {
  isOpen: boolean;
  contact: Contact | null;
  onClose: () => void;
  onSave: (id: string, data: { displayName?: string; notes?: string; tags?: string[] }) => void;
}) {
  const { t } = useI18n();
  const { focusTrap } = useAccessibility();
  const [displayName, setDisplayName] = useState('');
  const [notes, setNotes] = useState('');
  const [tags, setTags] = useState('');

  // Reset form when contact changes
  React.useEffect(() => {
    if (contact) {
      setDisplayName(contact.displayName || '');
      setNotes(contact.notes || '');
      setTags(contact.tags.join(', '));
    }
  }, [contact]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!contact) return;
    onSave(contact.id, {
      displayName: displayName || undefined,
      notes: notes || undefined,
      tags: tags ? tags.split(',').map((t) => t.trim()) : undefined,
    });
    onClose();
  };

  if (!isOpen || !contact) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="edit-contact-title"
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0,0,0,0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={onClose}
    >
      <div
        ref={(el) => el && focusTrap(el)}
        onClick={(e) => e.stopPropagation()}
        style={{
          backgroundColor: 'white',
          borderRadius: '8px',
          padding: '24px',
          width: '400px',
          maxWidth: '90vw',
        }}
      >
        <h2 id="edit-contact-title" style={{ margin: '0 0 16px' }}>
          {t('contacts.editContact')}
        </h2>
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', marginBottom: '4px', color: '#666' }}>
              Email
            </label>
            <p style={{ margin: 0, padding: '8px', backgroundColor: '#f5f5f5', borderRadius: '4px' }}>
              {contact.email}
            </p>
          </div>
          <div style={{ marginBottom: '16px' }}>
            <label htmlFor="edit-contact-name" style={{ display: 'block', marginBottom: '4px' }}>
              Display Name
            </label>
            <input
              id="edit-contact-name"
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
            />
          </div>
          <div style={{ marginBottom: '16px' }}>
            <label htmlFor="edit-contact-notes" style={{ display: 'block', marginBottom: '4px' }}>
              Notes
            </label>
            <textarea
              id="edit-contact-notes"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={3}
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ccc', resize: 'vertical' }}
            />
          </div>
          <div style={{ marginBottom: '16px' }}>
            <label htmlFor="edit-contact-tags" style={{ display: 'block', marginBottom: '4px' }}>
              Tags (comma-separated)
            </label>
            <input
              id="edit-contact-tags"
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="work, family, friends"
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
            />
          </div>
          <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
            <button type="button" onClick={onClose} style={{ padding: '8px 16px' }}>
              {t('common.cancel')}
            </button>
            <button
              type="submit"
              style={{
                padding: '8px 16px',
                backgroundColor: '#2196F3',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              {t('common.save', { defaultValue: 'Save' })}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// Main Page Component
export default function ContactsPage() {
  const { t } = useI18n();
  const { announce } = useAccessibility();
  const queryClient = useQueryClient();

  const [searchQuery, setSearchQuery] = useState('');
  const [filterTag, setFilterTag] = useState<string | null>(null);
  const [showBlocked, setShowBlocked] = useState(false);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [editingContact, setEditingContact] = useState<Contact | null>(null);

  // Fetch contacts
  const { data: contacts = [], isLoading } = useQuery({
    queryKey: ['contacts'],
    queryFn: async () => {
      const response = await graphqlClient.query<{ contacts: Contact[] }>(
        `query GetContacts {
          contacts {
            id email displayName avatarUrl holochainAgentId
            trustScore interactionCount lastInteractionAt
            isBlocked notes tags
          }
        }`,
        {}
      );
      return response.contacts;
    },
  });

  // Mutations
  const addContactMutation = useMutation({
    mutationFn: async (data: { email: string; displayName?: string; tags?: string[] }) => {
      return graphqlClient.mutate(
        `mutation AddContact($input: CreateContactInput!) {
          createContact(input: $input) { id }
        }`,
        { input: data }
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['contacts'] });
      announce(t('contacts.addContact') + ' successful');
    },
  });

  const blockContactMutation = useMutation({
    mutationFn: async ({ id, blocked }: { id: string; blocked: boolean }) => {
      return graphqlClient.mutate(
        `mutation BlockContact($id: ID!, $blocked: Boolean!) {
          updateContact(id: $id, input: { isBlocked: $blocked }) { id }
        }`,
        { id, blocked }
      );
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['contacts'] }),
  });

  const deleteContactMutation = useMutation({
    mutationFn: async (id: string) => {
      return graphqlClient.mutate(
        `mutation DeleteContact($id: ID!) { deleteContact(id: $id) }`,
        { id }
      );
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['contacts'] }),
  });

  const updateContactMutation = useMutation({
    mutationFn: async ({ id, data }: { id: string; data: { displayName?: string; notes?: string; tags?: string[] } }) => {
      return graphqlClient.mutate(
        `mutation UpdateContact($id: ID!, $input: UpdateContactInput!) {
          updateContact(id: $id, input: $input) { id }
        }`,
        { id, input: data }
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['contacts'] });
      announce(t('contacts.editContact') + ' successful');
    },
  });

  const favoriteContactMutation = useMutation({
    mutationFn: async ({ id, isFavorite }: { id: string; isFavorite: boolean }) => {
      return graphqlClient.mutate(
        `mutation FavoriteContact($id: ID!, $isFavorite: Boolean!) {
          updateContact(id: $id, input: { isFavorite: $isFavorite }) { id }
        }`,
        { id, isFavorite }
      );
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['contacts'] }),
  });

  // Filtered contacts
  const filteredContacts = useMemo(() => {
    return contacts.filter((contact) => {
      if (!showBlocked && contact.isBlocked) return false;
      if (filterTag && !contact.tags.includes(filterTag)) return false;
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        return (
          contact.email.toLowerCase().includes(query) ||
          contact.displayName?.toLowerCase().includes(query)
        );
      }
      return true;
    });
  }, [contacts, searchQuery, filterTag, showBlocked]);

  // All unique tags
  const allTags = useMemo(() => {
    const tags = new Set<string>();
    contacts.forEach((c) => c.tags.forEach((t) => tags.add(t)));
    return Array.from(tags).sort();
  }, [contacts]);

  // Keyboard shortcuts
  useKeyboardShortcut('n', () => setIsAddModalOpen(true));

  const handleEdit = useCallback((id: string) => {
    const contact = contacts.find((c) => c.id === id);
    if (contact) {
      setEditingContact(contact);
      setIsEditModalOpen(true);
    }
  }, [contacts]);

  const handleDelete = useCallback(
    (id: string) => {
      if (window.confirm(t('contacts.confirmDelete', { defaultValue: 'Delete this contact?' }))) {
        deleteContactMutation.mutate(id);
      }
    },
    [deleteContactMutation, t]
  );

  const handleBlock = useCallback(
    (id: string, blocked: boolean) => {
      blockContactMutation.mutate({ id, blocked });
    },
    [blockContactMutation]
  );

  const handleFavorite = useCallback((id: string) => {
    const contact = contacts.find((c) => c.id === id);
    if (contact) {
      favoriteContactMutation.mutate({ id, isFavorite: !contact.isFavorite });
    }
  }, [contacts, favoriteContactMutation]);

  if (isLoading) {
    return (
      <div role="status" style={{ padding: '2rem', textAlign: 'center' }}>
        {t('common.loading')}
      </div>
    );
  }

  return (
    <div style={{ padding: '24px', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Header */}
      <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
        <h1 style={{ margin: 0 }}>{t('contacts.contacts')}</h1>
        <button
          onClick={() => setIsAddModalOpen(true)}
          style={{
            padding: '10px 20px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold',
          }}
        >
          + {t('contacts.addContact')}
        </button>
      </header>

      {/* Filters */}
      <div style={{ display: 'flex', gap: '16px', marginBottom: '24px', flexWrap: 'wrap' }}>
        <input
          type="search"
          placeholder={t('contacts.searchContacts')}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          data-search-input
          style={{
            flex: 1,
            minWidth: '200px',
            padding: '10px 16px',
            borderRadius: '4px',
            border: '1px solid #ccc',
          }}
        />

        <select
          value={filterTag || ''}
          onChange={(e) => setFilterTag(e.target.value || null)}
          style={{ padding: '10px 16px', borderRadius: '4px', border: '1px solid #ccc' }}
        >
          <option value="">All Tags</option>
          {allTags.map((tag) => (
            <option key={tag} value={tag}>
              {tag}
            </option>
          ))}
        </select>

        <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <input
            type="checkbox"
            checked={showBlocked}
            onChange={(e) => setShowBlocked(e.target.checked)}
          />
          Show blocked
        </label>
      </div>

      {/* Stats */}
      <div style={{ marginBottom: '24px', color: '#666' }}>
        {filteredContacts.length} of {contacts.length} contacts
      </div>

      {/* Contact List */}
      <div role="list" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {filteredContacts.length === 0 ? (
          <p style={{ textAlign: 'center', color: '#666', padding: '2rem' }}>
            {t('contacts.noContacts')}
          </p>
        ) : (
          filteredContacts.map((contact) => (
            <ContactCard
              key={contact.id}
              contact={contact}
              onEdit={handleEdit}
              onDelete={handleDelete}
              onBlock={handleBlock}
              onFavorite={handleFavorite}
            />
          ))
        )}
      </div>

      {/* Add Modal */}
      <AddContactModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onAdd={(data) => addContactMutation.mutate(data)}
      />

      {/* Edit Modal */}
      <EditContactModal
        isOpen={isEditModalOpen}
        contact={editingContact}
        onClose={() => {
          setIsEditModalOpen(false);
          setEditingContact(null);
        }}
        onSave={(id, data) => updateContactMutation.mutate({ id, data })}
      />
    </div>
  );
}
