// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Contacts Page
 *
 * Contact management with CRM features and relationship tracking
 */

import React, { useEffect, useState } from 'react';

interface Contact {
  id: string;
  email: string;
  name?: string;
  company?: string;
  title?: string;
  phone?: string;
  avatarUrl?: string;
  notes?: string;
  tags: string[];
  relationshipStrength: number;
  lastInteraction?: string;
  interactionCount: number;
}

interface ContactGroup {
  id: string;
  name: string;
  memberCount: number;
}

interface Interaction {
  id: string;
  interactionType: string;
  createdAt: string;
  emailId?: string;
  notes?: string;
}

export default function ContactsPage() {
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [groups, setGroups] = useState<ContactGroup[]>([]);
  const [tags, setTags] = useState<{ tag: string; count: number }[]>([]);
  const [selectedContact, setSelectedContact] = useState<Contact | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterTag, setFilterTag] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<'grid' | 'list'>('list');

  useEffect(() => {
    fetchContacts();
    fetchGroups();
    fetchTags();
  }, []);

  async function fetchContacts() {
    try {
      let url = '/api/contacts';
      if (searchQuery) url += `?q=${encodeURIComponent(searchQuery)}`;
      if (filterTag) url += `${searchQuery ? '&' : '?'}tag=${encodeURIComponent(filterTag)}`;

      const response = await fetch(url);
      if (response.ok) {
        setContacts(await response.json());
      }
    } catch (error) {
      console.error('Failed to fetch contacts:', error);
    } finally {
      setLoading(false);
    }
  }

  async function fetchGroups() {
    try {
      const response = await fetch('/api/contacts/groups');
      if (response.ok) {
        setGroups(await response.json());
      }
    } catch (error) {
      console.error('Failed to fetch groups:', error);
    }
  }

  async function fetchTags() {
    try {
      const response = await fetch('/api/contacts/tags');
      if (response.ok) {
        setTags(await response.json());
      }
    } catch (error) {
      console.error('Failed to fetch tags:', error);
    }
  }

  async function deleteContact(id: string) {
    if (!confirm('Delete this contact?')) return;
    try {
      await fetch(`/api/contacts/${id}`, { method: 'DELETE' });
      setContacts(contacts.filter((c) => c.id !== id));
      if (selectedContact?.id === id) setSelectedContact(null);
    } catch (error) {
      console.error('Failed to delete contact:', error);
    }
  }

  function getStrengthColor(strength: number): string {
    if (strength >= 70) return 'text-green-600';
    if (strength >= 40) return 'text-yellow-600';
    return 'text-gray-400';
  }

  function getStrengthLabel(strength: number): string {
    if (strength >= 70) return 'Strong';
    if (strength >= 40) return 'Moderate';
    if (strength > 0) return 'Weak';
    return 'New';
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <div className="w-64 border-r border-border p-4 space-y-6">
        <div>
          <input
            type="text"
            placeholder="Search contacts..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              fetchContacts();
            }}
            className="w-full px-3 py-2 border border-border rounded-lg"
          />
        </div>

        <div>
          <h3 className="font-medium mb-2">Groups</h3>
          <ul className="space-y-1">
            {groups.map((group) => (
              <li
                key={group.id}
                className="flex justify-between items-center px-2 py-1 hover:bg-muted/30 rounded cursor-pointer"
              >
                <span>{group.name}</span>
                <span className="text-xs text-muted">{group.memberCount}</span>
              </li>
            ))}
            <li className="px-2 py-1 text-sm text-primary cursor-pointer hover:underline">
              + Create Group
            </li>
          </ul>
        </div>

        <div>
          <h3 className="font-medium mb-2">Tags</h3>
          <div className="flex flex-wrap gap-2">
            {tags.map((tag) => (
              <button
                key={tag.tag}
                onClick={() => {
                  setFilterTag(filterTag === tag.tag ? null : tag.tag);
                  fetchContacts();
                }}
                className={`px-2 py-1 text-xs rounded-full ${
                  filterTag === tag.tag
                    ? 'bg-primary text-white'
                    : 'bg-muted/30 hover:bg-muted/50'
                }`}
              >
                {tag.tag} ({tag.count})
              </button>
            ))}
          </div>
        </div>

        <div>
          <h3 className="font-medium mb-2">Quick Filters</h3>
          <ul className="space-y-1 text-sm">
            <li className="px-2 py-1 hover:bg-muted/30 rounded cursor-pointer">
              Recently Contacted
            </li>
            <li className="px-2 py-1 hover:bg-muted/30 rounded cursor-pointer">
              Needs Follow-up
            </li>
            <li className="px-2 py-1 hover:bg-muted/30 rounded cursor-pointer">
              VIP Contacts
            </li>
          </ul>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Contact List */}
        <div className={`${selectedContact ? 'w-1/2' : 'flex-1'} border-r border-border`}>
          <div className="p-4 border-b border-border flex items-center justify-between">
            <h1 className="text-xl font-bold">Contacts ({contacts.length})</h1>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setView('list')}
                className={`p-2 rounded ${view === 'list' ? 'bg-muted/50' : ''}`}
              >
                List
              </button>
              <button
                onClick={() => setView('grid')}
                className={`p-2 rounded ${view === 'grid' ? 'bg-muted/50' : ''}`}
              >
                Grid
              </button>
              <button className="px-3 py-1.5 bg-primary text-white rounded-lg text-sm">
                Add Contact
              </button>
            </div>
          </div>

          {contacts.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-muted">No contacts found</p>
            </div>
          ) : view === 'list' ? (
            <div className="divide-y divide-border">
              {contacts.map((contact) => (
                <div
                  key={contact.id}
                  className={`p-4 hover:bg-muted/30 cursor-pointer ${
                    selectedContact?.id === contact.id ? 'bg-primary/10' : ''
                  }`}
                  onClick={() => setSelectedContact(contact)}
                >
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                      {contact.avatarUrl ? (
                        <img src={contact.avatarUrl} className="w-10 h-10 rounded-full" />
                      ) : (
                        (contact.name || contact.email)[0].toUpperCase()
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium truncate">{contact.name || contact.email}</p>
                      <p className="text-sm text-muted truncate">
                        {contact.company && `${contact.company} - `}
                        {contact.email}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className={`text-sm font-medium ${getStrengthColor(contact.relationshipStrength)}`}>
                        {getStrengthLabel(contact.relationshipStrength)}
                      </div>
                      <p className="text-xs text-muted">
                        {contact.interactionCount} interactions
                      </p>
                    </div>
                  </div>
                  {contact.tags.length > 0 && (
                    <div className="flex gap-1 mt-2 ml-14">
                      {contact.tags.slice(0, 3).map((tag) => (
                        <span key={tag} className="px-2 py-0.5 bg-muted/30 rounded text-xs">
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 p-4">
              {contacts.map((contact) => (
                <div
                  key={contact.id}
                  className="p-4 border border-border rounded-lg hover:border-primary cursor-pointer"
                  onClick={() => setSelectedContact(contact)}
                >
                  <div className="text-center">
                    <div className="w-16 h-16 mx-auto rounded-full bg-primary/20 flex items-center justify-center text-2xl">
                      {(contact.name || contact.email)[0].toUpperCase()}
                    </div>
                    <p className="font-medium mt-2">{contact.name || contact.email}</p>
                    <p className="text-sm text-muted truncate">{contact.company}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Contact Detail */}
        {selectedContact && (
          <ContactDetail
            contact={selectedContact}
            onClose={() => setSelectedContact(null)}
            onDelete={() => deleteContact(selectedContact.id)}
          />
        )}
      </div>
    </div>
  );
}

function ContactDetail({
  contact,
  onClose,
  onDelete,
}: {
  contact: Contact;
  onClose: () => void;
  onDelete: () => void;
}) {
  const [interactions, setInteractions] = useState<Interaction[]>([]);
  const [activeTab, setActiveTab] = useState<'details' | 'activity' | 'emails'>('details');

  useEffect(() => {
    fetchInteractions();
  }, [contact.id]);

  async function fetchInteractions() {
    try {
      const response = await fetch(`/api/contacts/${contact.id}/interactions`);
      if (response.ok) {
        setInteractions(await response.json());
      }
    } catch (error) {
      console.error('Failed to fetch interactions:', error);
    }
  }

  return (
    <div className="w-1/2 flex flex-col">
      <div className="p-4 border-b border-border flex items-center justify-between">
        <h2 className="font-semibold">Contact Details</h2>
        <button onClick={onClose} className="p-2 hover:bg-muted/30 rounded">
          X
        </button>
      </div>

      <div className="p-6 border-b border-border">
        <div className="flex items-start gap-4">
          <div className="w-20 h-20 rounded-full bg-primary/20 flex items-center justify-center text-3xl">
            {(contact.name || contact.email)[0].toUpperCase()}
          </div>
          <div className="flex-1">
            <h3 className="text-xl font-bold">{contact.name || contact.email}</h3>
            {contact.title && contact.company && (
              <p className="text-muted">{contact.title} at {contact.company}</p>
            )}
            <div className="flex gap-2 mt-2">
              <button className="px-3 py-1.5 bg-primary text-white rounded text-sm">
                Send Email
              </button>
              <button className="px-3 py-1.5 border border-border rounded text-sm">
                Edit
              </button>
              <button
                onClick={onDelete}
                className="px-3 py-1.5 text-red-600 border border-red-200 rounded text-sm"
              >
                Delete
              </button>
            </div>
          </div>
        </div>

        {/* Relationship Strength */}
        <div className="mt-4 p-3 bg-muted/20 rounded-lg">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium">Relationship Strength</span>
            <span className="text-sm">{Math.round(contact.relationshipStrength)}%</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full">
            <div
              className="h-full bg-primary rounded-full"
              style={{ width: `${contact.relationshipStrength}%` }}
            />
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-border">
        <div className="flex">
          {(['details', 'activity', 'emails'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 text-sm font-medium capitalize ${
                activeTab === tab
                  ? 'border-b-2 border-primary text-primary'
                  : 'text-muted'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'details' && (
          <div className="space-y-4">
            <div>
              <label className="text-sm text-muted">Email</label>
              <p>{contact.email}</p>
            </div>
            {contact.phone && (
              <div>
                <label className="text-sm text-muted">Phone</label>
                <p>{contact.phone}</p>
              </div>
            )}
            {contact.company && (
              <div>
                <label className="text-sm text-muted">Company</label>
                <p>{contact.company}</p>
              </div>
            )}
            {contact.tags.length > 0 && (
              <div>
                <label className="text-sm text-muted">Tags</label>
                <div className="flex gap-2 mt-1">
                  {contact.tags.map((tag) => (
                    <span key={tag} className="px-2 py-1 bg-muted/30 rounded text-sm">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {contact.notes && (
              <div>
                <label className="text-sm text-muted">Notes</label>
                <p className="whitespace-pre-wrap">{contact.notes}</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'activity' && (
          <div className="space-y-3">
            {interactions.length === 0 ? (
              <p className="text-muted text-center py-4">No activity yet</p>
            ) : (
              interactions.map((interaction) => (
                <div key={interaction.id} className="flex gap-3">
                  <div className="w-8 h-8 rounded-full bg-muted/30 flex items-center justify-center text-sm">
                    {interaction.interactionType === 'email_sent' && '📤'}
                    {interaction.interactionType === 'email_received' && '📥'}
                    {interaction.interactionType === 'meeting_held' && '📅'}
                    {interaction.interactionType === 'note' && '📝'}
                  </div>
                  <div>
                    <p className="text-sm">
                      {interaction.interactionType.replace('_', ' ')}
                    </p>
                    <p className="text-xs text-muted">
                      {new Date(interaction.createdAt).toLocaleString()}
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === 'emails' && (
          <div className="text-center py-4 text-muted">
            <p>Email history would appear here</p>
          </div>
        )}
      </div>
    </div>
  );
}
