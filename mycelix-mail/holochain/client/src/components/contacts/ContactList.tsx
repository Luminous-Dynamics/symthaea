// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * ContactList Component
 *
 * Searchable, grouped contact list with trust integration.
 */

import React, { useState, useCallback, useMemo } from 'react';
import { Avatar } from '../common/Avatar';
import { TrustBadge } from '../common/TrustBadge';

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
  tags?: string[];
  favorite?: boolean;
  lastContacted?: number;
  blocked?: boolean;
}

export interface ContactGroup {
  id: string;
  name: string;
  contacts: string[]; // Contact IDs
  color?: string;
}

export interface ContactListProps {
  contacts: Contact[];
  groups?: ContactGroup[];
  loading?: boolean;
  selectedId?: string;
  onSelect?: (contact: Contact) => void;
  onEdit?: (contact: Contact) => void;
  onDelete?: (contactId: string) => void;
  onToggleFavorite?: (contactId: string, favorite: boolean) => void;
  onToggleBlock?: (contactId: string, blocked: boolean) => void;
  onAddToGroup?: (contactId: string, groupId: string) => void;
  onCompose?: (contact: Contact) => void;
}

type SortOption = 'name' | 'trust' | 'recent' | 'organization';
type ViewMode = 'list' | 'grid';

export const ContactList: React.FC<ContactListProps> = ({
  contacts,
  groups = [],
  loading = false,
  selectedId,
  onSelect,
  onEdit,
  onDelete,
  onToggleFavorite,
  onToggleBlock,
  onAddToGroup,
  onCompose,
}) => {
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState<SortOption>('name');
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [showBlocked, setShowBlocked] = useState(false);
  const [selectedGroup, setSelectedGroup] = useState<string | null>(null);

  // Filter contacts
  const filteredContacts = useMemo(() => {
    let result = contacts;

    // Filter by search
    if (search) {
      const query = search.toLowerCase();
      result = result.filter(
        (c) =>
          c.name?.toLowerCase().includes(query) ||
          c.email.toLowerCase().includes(query) ||
          c.nickname?.toLowerCase().includes(query) ||
          c.organization?.toLowerCase().includes(query) ||
          c.tags?.some((t) => t.toLowerCase().includes(query))
      );
    }

    // Filter by group
    if (selectedGroup) {
      const group = groups.find((g) => g.id === selectedGroup);
      if (group) {
        result = result.filter((c) => group.contacts.includes(c.id));
      }
    }

    // Filter blocked
    if (!showBlocked) {
      result = result.filter((c) => !c.blocked);
    }

    // Sort
    result = [...result].sort((a, b) => {
      switch (sortBy) {
        case 'trust':
          return (b.trustLevel ?? 0) - (a.trustLevel ?? 0);
        case 'recent':
          return (b.lastContacted ?? 0) - (a.lastContacted ?? 0);
        case 'organization':
          return (a.organization ?? '').localeCompare(b.organization ?? '');
        case 'name':
        default:
          return (a.name ?? a.email).localeCompare(b.name ?? b.email);
      }
    });

    // Put favorites first
    result.sort((a, b) => {
      if (a.favorite && !b.favorite) return -1;
      if (!a.favorite && b.favorite) return 1;
      return 0;
    });

    return result;
  }, [contacts, search, sortBy, selectedGroup, showBlocked, groups]);

  // Group by first letter for list view
  const groupedContacts = useMemo(() => {
    const grouped: Record<string, Contact[]> = {};

    filteredContacts.forEach((contact) => {
      const key = contact.favorite
        ? 'Favorites'
        : (contact.name?.[0] ?? contact.email[0]).toUpperCase();

      if (!grouped[key]) {
        grouped[key] = [];
      }
      grouped[key].push(contact);
    });

    return grouped;
  }, [filteredContacts]);

  const sortedKeys = useMemo(() => {
    const keys = Object.keys(groupedContacts);
    return keys.sort((a, b) => {
      if (a === 'Favorites') return -1;
      if (b === 'Favorites') return 1;
      return a.localeCompare(b);
    });
  }, [groupedContacts]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <SpinnerIcon className="w-8 h-8 animate-spin text-gray-400" />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="px-4 py-3 border-b">
        {/* Search */}
        <div className="relative">
          <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search contacts..."
            className="w-full pl-9 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Filters */}
        <div className="flex items-center justify-between mt-3">
          <div className="flex items-center gap-2">
            {/* Sort dropdown */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as SortOption)}
              className="text-sm border rounded px-2 py-1"
            >
              <option value="name">Name</option>
              <option value="trust">Trust level</option>
              <option value="recent">Recent</option>
              <option value="organization">Organization</option>
            </select>

            {/* Group filter */}
            {groups.length > 0 && (
              <select
                value={selectedGroup ?? ''}
                onChange={(e) => setSelectedGroup(e.target.value || null)}
                className="text-sm border rounded px-2 py-1"
              >
                <option value="">All contacts</option>
                {groups.map((g) => (
                  <option key={g.id} value={g.id}>
                    {g.name}
                  </option>
                ))}
              </select>
            )}

            {/* Show blocked toggle */}
            <label className="flex items-center gap-1 text-sm text-gray-600 cursor-pointer">
              <input
                type="checkbox"
                checked={showBlocked}
                onChange={(e) => setShowBlocked(e.target.checked)}
                className="rounded"
              />
              Show blocked
            </label>
          </div>

          {/* View mode toggle */}
          <div className="flex items-center border rounded overflow-hidden">
            <button
              onClick={() => setViewMode('list')}
              className={`p-1.5 ${viewMode === 'list' ? 'bg-blue-500 text-white' : ''}`}
            >
              <ListIcon className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('grid')}
              className={`p-1.5 ${viewMode === 'grid' ? 'bg-blue-500 text-white' : ''}`}
            >
              <GridIcon className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Contact count */}
      <div className="px-4 py-2 text-sm text-gray-500 border-b">
        {filteredContacts.length} contact{filteredContacts.length !== 1 ? 's' : ''}
      </div>

      {/* Contact list */}
      <div className="flex-1 overflow-auto">
        {filteredContacts.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <UsersIcon className="w-16 h-16 mb-4" />
            <p className="text-lg">No contacts found</p>
          </div>
        ) : viewMode === 'grid' ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4 p-4">
            {filteredContacts.map((contact) => (
              <ContactCard
                key={contact.id}
                contact={contact}
                selected={contact.id === selectedId}
                onSelect={() => onSelect?.(contact)}
                onEdit={() => onEdit?.(contact)}
                onCompose={() => onCompose?.(contact)}
                onToggleFavorite={() =>
                  onToggleFavorite?.(contact.id, !contact.favorite)
                }
              />
            ))}
          </div>
        ) : (
          <div>
            {sortedKeys.map((key) => (
              <div key={key}>
                <div className="sticky top-0 px-4 py-1 bg-gray-100 text-sm font-medium text-gray-600">
                  {key}
                </div>
                {groupedContacts[key].map((contact) => (
                  <ContactRow
                    key={contact.id}
                    contact={contact}
                    selected={contact.id === selectedId}
                    onSelect={() => onSelect?.(contact)}
                    onEdit={() => onEdit?.(contact)}
                    onDelete={() => onDelete?.(contact.id)}
                    onCompose={() => onCompose?.(contact)}
                    onToggleFavorite={() =>
                      onToggleFavorite?.(contact.id, !contact.favorite)
                    }
                    onToggleBlock={() =>
                      onToggleBlock?.(contact.id, !contact.blocked)
                    }
                  />
                ))}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// Contact Row Component (List View)
interface ContactRowProps {
  contact: Contact;
  selected: boolean;
  onSelect: () => void;
  onEdit: () => void;
  onDelete: () => void;
  onCompose: () => void;
  onToggleFavorite: () => void;
  onToggleBlock: () => void;
}

const ContactRow: React.FC<ContactRowProps> = ({
  contact,
  selected,
  onSelect,
  onEdit,
  onDelete,
  onCompose,
  onToggleFavorite,
  onToggleBlock,
}) => {
  const [showMenu, setShowMenu] = useState(false);

  return (
    <div
      className={`flex items-center gap-3 px-4 py-3 cursor-pointer border-b border-gray-100 group ${
        selected ? 'bg-blue-50' : 'hover:bg-gray-50'
      } ${contact.blocked ? 'opacity-50' : ''}`}
      onClick={onSelect}
    >
      <Avatar
        name={contact.name}
        email={contact.email}
        src={contact.avatarUrl}
        size={44}
      />

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-900 truncate">
            {contact.name || contact.email}
          </span>
          {contact.favorite && (
            <StarIcon className="w-4 h-4 text-yellow-500" filled />
          )}
          {contact.blocked && (
            <span className="text-xs px-1.5 py-0.5 bg-red-100 text-red-700 rounded">
              Blocked
            </span>
          )}
        </div>
        <div className="text-sm text-gray-500 truncate">
          {contact.name && contact.email}
          {contact.organization && ` | ${contact.organization}`}
        </div>
      </div>

      {contact.trustLevel !== undefined && (
        <TrustBadge
          level={contact.trustLevel}
          attestationCount={contact.attestationCount}
          size="sm"
        />
      )}

      {/* Tags */}
      {contact.tags && contact.tags.length > 0 && (
        <div className="hidden sm:flex gap-1">
          {contact.tags.slice(0, 2).map((tag) => (
            <span
              key={tag}
              className="text-xs px-2 py-0.5 bg-gray-100 text-gray-600 rounded-full"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onCompose();
          }}
          className="p-1.5 hover:bg-gray-200 rounded"
          title="Send email"
        >
          <MailIcon className="w-4 h-4 text-gray-600" />
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onEdit();
          }}
          className="p-1.5 hover:bg-gray-200 rounded"
          title="Edit"
        >
          <EditIcon className="w-4 h-4 text-gray-600" />
        </button>
        <div className="relative">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowMenu(!showMenu);
            }}
            className="p-1.5 hover:bg-gray-200 rounded"
          >
            <MoreIcon className="w-4 h-4 text-gray-600" />
          </button>
          {showMenu && (
            <div
              className="absolute right-0 top-full mt-1 bg-white border rounded shadow-lg py-1 z-10 min-w-[120px]"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                onClick={() => {
                  onToggleFavorite();
                  setShowMenu(false);
                }}
                className="w-full text-left px-3 py-1.5 hover:bg-gray-100 text-sm"
              >
                {contact.favorite ? 'Remove favorite' : 'Add to favorites'}
              </button>
              <button
                onClick={() => {
                  onToggleBlock();
                  setShowMenu(false);
                }}
                className="w-full text-left px-3 py-1.5 hover:bg-gray-100 text-sm"
              >
                {contact.blocked ? 'Unblock' : 'Block'}
              </button>
              <hr className="my-1" />
              <button
                onClick={() => {
                  onDelete();
                  setShowMenu(false);
                }}
                className="w-full text-left px-3 py-1.5 hover:bg-gray-100 text-sm text-red-600"
              >
                Delete
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Contact Card Component (Grid View)
interface ContactCardProps {
  contact: Contact;
  selected: boolean;
  onSelect: () => void;
  onEdit: () => void;
  onCompose: () => void;
  onToggleFavorite: () => void;
}

const ContactCard: React.FC<ContactCardProps> = ({
  contact,
  selected,
  onSelect,
  onEdit,
  onCompose,
  onToggleFavorite,
}) => {
  return (
    <div
      className={`flex flex-col items-center p-4 border rounded-lg cursor-pointer ${
        selected ? 'border-blue-500 bg-blue-50' : 'hover:bg-gray-50'
      }`}
      onClick={onSelect}
    >
      <div className="relative">
        <Avatar
          name={contact.name}
          email={contact.email}
          src={contact.avatarUrl}
          size={64}
        />
        {contact.favorite && (
          <StarIcon className="absolute -top-1 -right-1 w-5 h-5 text-yellow-500" filled />
        )}
      </div>

      <div className="mt-2 text-center">
        <div className="font-medium text-gray-900 truncate max-w-full">
          {contact.name || contact.email.split('@')[0]}
        </div>
        <div className="text-sm text-gray-500 truncate max-w-full">
          {contact.email}
        </div>
      </div>

      {contact.trustLevel !== undefined && (
        <TrustBadge level={contact.trustLevel} size="sm" className="mt-2" />
      )}

      <div className="flex items-center gap-1 mt-3">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onCompose();
          }}
          className="p-1.5 hover:bg-gray-200 rounded"
          title="Send email"
        >
          <MailIcon className="w-4 h-4 text-gray-600" />
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onEdit();
          }}
          className="p-1.5 hover:bg-gray-200 rounded"
          title="Edit"
        >
          <EditIcon className="w-4 h-4 text-gray-600" />
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onToggleFavorite();
          }}
          className="p-1.5 hover:bg-gray-200 rounded"
          title={contact.favorite ? 'Remove favorite' : 'Add to favorites'}
        >
          <StarIcon
            className={`w-4 h-4 ${
              contact.favorite ? 'text-yellow-500' : 'text-gray-400'
            }`}
            filled={contact.favorite}
          />
        </button>
      </div>
    </div>
  );
};

// Icon components
const SearchIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

const ListIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="8" y1="6" x2="21" y2="6" />
    <line x1="8" y1="12" x2="21" y2="12" />
    <line x1="8" y1="18" x2="21" y2="18" />
    <line x1="3" y1="6" x2="3.01" y2="6" />
    <line x1="3" y1="12" x2="3.01" y2="12" />
    <line x1="3" y1="18" x2="3.01" y2="18" />
  </svg>
);

const GridIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="3" width="7" height="7" />
    <rect x="14" y="3" width="7" height="7" />
    <rect x="14" y="14" width="7" height="7" />
    <rect x="3" y="14" width="7" height="7" />
  </svg>
);

const UsersIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </svg>
);

const StarIcon: React.FC<{ className?: string; filled?: boolean }> = ({ className, filled }) => (
  <svg className={className} viewBox="0 0 24 24" fill={filled ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2">
    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
  </svg>
);

const MailIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="2" y="4" width="20" height="16" rx="2" />
    <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7" />
  </svg>
);

const EditIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
  </svg>
);

const MoreIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="1" />
    <circle cx="12" cy="5" r="1" />
    <circle cx="12" cy="19" r="1" />
  </svg>
);

const SpinnerIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 12a9 9 0 11-6.219-8.56" />
  </svg>
);

export default ContactList;
