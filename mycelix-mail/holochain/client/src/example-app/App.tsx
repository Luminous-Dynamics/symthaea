// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail - Example Application
 *
 * Full-featured email client demonstrating all capabilities.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { createMycelixClient, useMycelixClient } from '../bootstrap';
import { InboxList, ThreadView, ComposeEmail, ContactList } from '../components';
import type { Email, Contact, ThreadMessage, ComposedEmail, Recipient } from '../components';

// Initialize client
const client = createMycelixClient({
  websocketUrl: 'ws://localhost:8888',
  appId: 'mycelix-mail',
  cache: {
    maxSize: 1000,
    ttlMs: 5 * 60 * 1000,
    persistToIndexedDB: true,
  },
  notifications: {
    enabled: true,
    quietHoursStart: 22,
    quietHoursEnd: 8,
  },
  offline: {
    maxQueueSize: 10000,
    syncIntervalMs: 30000,
  },
  metrics: {
    enabled: true,
    flushIntervalMs: 60000,
  },
});

type View = 'inbox' | 'sent' | 'drafts' | 'trash' | 'contacts' | 'settings';

export const App: React.FC = () => {
  const [initialized, setInitialized] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentView, setCurrentView] = useState<View>('inbox');
  const [selectedEmail, setSelectedEmail] = useState<Email | null>(null);
  const [showCompose, setShowCompose] = useState(false);
  const [replyTo, setReplyTo] = useState<{ hash: string; email: Email } | null>(null);

  // Email state
  const [emails, setEmails] = useState<Email[]>([]);
  const [loadingEmails, setLoadingEmails] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);

  // Thread state
  const [threadMessages, setThreadMessages] = useState<ThreadMessage[]>([]);
  const [loadingThread, setLoadingThread] = useState(false);

  // Contact state
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [loadingContacts, setLoadingContacts] = useState(false);

  // Initialize client
  useEffect(() => {
    client
      .initialize()
      .then(() => {
        setInitialized(true);
        loadEmails();
      })
      .catch((e) => {
        setError(`Failed to initialize: ${e.message}`);
      });

    return () => {
      client.shutdown();
    };
  }, []);

  // Load emails
  const loadEmails = useCallback(async () => {
    setLoadingEmails(true);
    try {
      const services = client.getServices();
      const result = await services.messages.getMessages({
        folder: currentView === 'sent' ? 'sent' : currentView === 'drafts' ? 'drafts' : 'inbox',
        limit: 50,
      });

      setEmails(result.messages);
      setUnreadCount(result.messages.filter((m: Email) => !m.read).length);
    } catch (e) {
      console.error('Failed to load emails:', e);
    } finally {
      setLoadingEmails(false);
    }
  }, [currentView]);

  // Reload emails when view changes
  useEffect(() => {
    if (initialized) {
      loadEmails();
    }
  }, [currentView, initialized, loadEmails]);

  // Load thread when email selected
  const loadThread = useCallback(async (email: Email) => {
    setLoadingThread(true);
    try {
      const services = client.getServices();
      const thread = await services.thread.getThread(email.threadId || email.hash);
      setThreadMessages(thread.messages);
    } catch (e) {
      console.error('Failed to load thread:', e);
    } finally {
      setLoadingThread(false);
    }
  }, []);

  // Handle email click
  const handleEmailClick = useCallback((email: Email) => {
    setSelectedEmail(email);
    loadThread(email);
  }, [loadThread]);

  // Handle back from thread
  const handleBackFromThread = useCallback(() => {
    setSelectedEmail(null);
    setThreadMessages([]);
  }, []);

  // Handle bulk actions
  const handleBulkAction = useCallback(async (action: string, hashes: string[]) => {
    const services = client.getServices();

    try {
      switch (action) {
        case 'archive':
          await services.batch.archiveMany(hashes);
          break;
        case 'delete':
          await services.batch.trashMany(hashes);
          break;
        case 'markRead':
          await services.batch.markManyAsRead(hashes);
          break;
        case 'markUnread':
          await services.batch.markManyAsUnread(hashes);
          break;
        case 'star':
          await services.batch.starMany(hashes);
          break;
        case 'unstar':
          await services.batch.unstarMany(hashes);
          break;
        case 'spam':
          await services.batch.markManyAsSpam(hashes);
          break;
      }

      loadEmails();
    } catch (e) {
      console.error('Bulk action failed:', e);
    }
  }, [loadEmails]);

  // Handle reply
  const handleReply = useCallback((messageHash: string, replyAll: boolean) => {
    const message = threadMessages.find((m) => m.hash === messageHash);
    if (message && selectedEmail) {
      setReplyTo({ hash: messageHash, email: selectedEmail });
      setShowCompose(true);
    }
  }, [threadMessages, selectedEmail]);

  // Handle send email
  const handleSend = useCallback(async (email: ComposedEmail) => {
    const services = client.getServices();

    await services.messages.send(email);
    setShowCompose(false);
    setReplyTo(null);

    if (currentView === 'sent') {
      loadEmails();
    }
  }, [currentView, loadEmails]);

  // Handle schedule send
  const handleSchedule = useCallback(async (email: ComposedEmail, sendAt: Date) => {
    const services = client.getServices();

    await services.scheduler.schedule({
      email,
      sendAt: sendAt.getTime() * 1000,
    });

    setShowCompose(false);
    setReplyTo(null);
  }, []);

  // Contact suggestions
  const handleSuggestContacts = useCallback(async (query: string): Promise<Recipient[]> => {
    const services = client.getServices();
    return services.contacts.getSuggestions(query, 5);
  }, []);

  // Load contacts
  const loadContacts = useCallback(async () => {
    setLoadingContacts(true);
    try {
      const services = client.getServices();
      const result = await services.contacts.getAllContacts();
      setContacts(result);
    } catch (e) {
      console.error('Failed to load contacts:', e);
    } finally {
      setLoadingContacts(false);
    }
  }, []);

  // Handle contact compose
  const handleComposeToContact = useCallback((contact: Contact) => {
    setReplyTo(null);
    setShowCompose(true);
    // The compose component will receive the initial recipient
  }, []);

  if (!initialized) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-100">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Connecting to Holochain...</p>
          {error && <p className="text-red-500 mt-2">{error}</p>}
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r flex flex-col">
        {/* Logo */}
        <div className="p-4 border-b">
          <h1 className="text-xl font-bold text-gray-800">Mycelix Mail</h1>
          <p className="text-xs text-gray-500">Decentralized Email</p>
        </div>

        {/* Compose button */}
        <div className="p-4">
          <button
            onClick={() => {
              setReplyTo(null);
              setShowCompose(true);
            }}
            className="w-full py-2 px-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center justify-center gap-2"
          >
            <PlusIcon className="w-5 h-5" />
            Compose
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-2 py-4 space-y-1">
          <NavItem
            icon={<InboxIcon />}
            label="Inbox"
            active={currentView === 'inbox'}
            badge={unreadCount > 0 ? unreadCount : undefined}
            onClick={() => setCurrentView('inbox')}
          />
          <NavItem
            icon={<SendIcon />}
            label="Sent"
            active={currentView === 'sent'}
            onClick={() => setCurrentView('sent')}
          />
          <NavItem
            icon={<DraftIcon />}
            label="Drafts"
            active={currentView === 'drafts'}
            onClick={() => setCurrentView('drafts')}
          />
          <NavItem
            icon={<TrashIcon />}
            label="Trash"
            active={currentView === 'trash'}
            onClick={() => setCurrentView('trash')}
          />

          <div className="border-t my-4" />

          <NavItem
            icon={<ContactsIcon />}
            label="Contacts"
            active={currentView === 'contacts'}
            onClick={() => {
              setCurrentView('contacts');
              loadContacts();
            }}
          />
          <NavItem
            icon={<SettingsIcon />}
            label="Settings"
            active={currentView === 'settings'}
            onClick={() => setCurrentView('settings')}
          />
        </nav>

        {/* Status */}
        <div className="p-4 border-t">
          <HealthStatus />
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {selectedEmail && threadMessages.length > 0 ? (
          <ThreadView
            subject={selectedEmail.subject}
            messages={threadMessages}
            currentUserEmail="me@mycelix.local"
            loading={loadingThread}
            onReply={handleReply}
            onForward={(hash) => console.log('Forward:', hash)}
            onArchive={() => {
              handleBulkAction('archive', [selectedEmail.hash]);
              handleBackFromThread();
            }}
            onDelete={() => {
              handleBulkAction('delete', [selectedEmail.hash]);
              handleBackFromThread();
            }}
            onBack={handleBackFromThread}
          />
        ) : currentView === 'contacts' ? (
          <ContactList
            contacts={contacts}
            loading={loadingContacts}
            onCompose={handleComposeToContact}
          />
        ) : currentView === 'settings' ? (
          <SettingsView />
        ) : (
          <InboxList
            emails={emails}
            loading={loadingEmails}
            onEmailClick={handleEmailClick}
            onBulkAction={handleBulkAction}
            onRefresh={loadEmails}
          />
        )}
      </main>

      {/* Compose modal */}
      {showCompose && (
        <div className="fixed bottom-4 right-4 z-50 w-[600px]">
          <ComposeEmail
            to={replyTo ? [{ address: replyTo.email.from.address, name: replyTo.email.from.name }] : []}
            subject={replyTo ? `Re: ${replyTo.email.subject}` : ''}
            replyTo={replyTo?.hash}
            threadId={replyTo?.email.threadId}
            onSuggestContacts={handleSuggestContacts}
            onSend={handleSend}
            onSchedule={handleSchedule}
            onDiscard={() => {
              setShowCompose(false);
              setReplyTo(null);
            }}
            onMinimize={() => setShowCompose(false)}
          />
        </div>
      )}
    </div>
  );
};

// Navigation Item Component
interface NavItemProps {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  badge?: number;
  onClick: () => void;
}

const NavItem: React.FC<NavItemProps> = ({ icon, label, active, badge, onClick }) => (
  <button
    onClick={onClick}
    className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left ${
      active ? 'bg-blue-100 text-blue-700' : 'text-gray-700 hover:bg-gray-100'
    }`}
  >
    <span className="w-5 h-5">{icon}</span>
    <span className="flex-1">{label}</span>
    {badge !== undefined && (
      <span className="px-2 py-0.5 text-xs bg-blue-600 text-white rounded-full">
        {badge}
      </span>
    )}
  </button>
);

// Health Status Component
const HealthStatus: React.FC = () => {
  const [health, setHealth] = useState<Record<string, any>>({});

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const status = await client.getHealth();
        setHealth(status);
      } catch (e) {
        console.error('Health check failed:', e);
      }
    }, 10000);

    // Initial check
    client.getHealth().then(setHealth).catch(console.error);

    return () => clearInterval(interval);
  }, []);

  const isHealthy = Object.values(health).every(
    (check: any) => check?.status === 'healthy'
  );

  return (
    <div className="flex items-center gap-2 text-sm">
      <span
        className={`w-2 h-2 rounded-full ${
          isHealthy ? 'bg-green-500' : 'bg-yellow-500'
        }`}
      />
      <span className="text-gray-600">
        {isHealthy ? 'Connected' : 'Syncing...'}
      </span>
    </div>
  );
};

// Settings View Component
const SettingsView: React.FC = () => {
  const [metrics, setMetrics] = useState<Record<string, any>>({});

  useEffect(() => {
    setMetrics(client.getMetrics());
  }, []);

  return (
    <div className="p-6 max-w-2xl">
      <h2 className="text-2xl font-bold mb-6">Settings</h2>

      {/* Profile section */}
      <section className="mb-8">
        <h3 className="text-lg font-semibold mb-4">Profile</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Display Name
            </label>
            <input
              type="text"
              className="w-full px-3 py-2 border rounded-lg"
              placeholder="Your name"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Email Address
            </label>
            <input
              type="email"
              className="w-full px-3 py-2 border rounded-lg bg-gray-50"
              value="me@mycelix.local"
              readOnly
            />
          </div>
        </div>
      </section>

      {/* Security section */}
      <section className="mb-8">
        <h3 className="text-lg font-semibold mb-4">Security</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div>
              <div className="font-medium">End-to-End Encryption</div>
              <div className="text-sm text-gray-500">
                Encrypt all outgoing messages by default
              </div>
            </div>
            <input type="checkbox" defaultChecked className="w-5 h-5" />
          </div>
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div>
              <div className="font-medium">Key Rotation</div>
              <div className="text-sm text-gray-500">
                Automatically rotate encryption keys
              </div>
            </div>
            <button className="px-4 py-2 border rounded-lg hover:bg-gray-100">
              Rotate Now
            </button>
          </div>
        </div>
      </section>

      {/* Metrics section */}
      <section className="mb-8">
        <h3 className="text-lg font-semibold mb-4">Statistics</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold">
              {metrics['emails.sent']?.value ?? 0}
            </div>
            <div className="text-sm text-gray-500">Emails Sent</div>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold">
              {metrics['emails.received']?.value ?? 0}
            </div>
            <div className="text-sm text-gray-500">Emails Received</div>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold">
              {metrics['trust.attestations_given']?.value ?? 0}
            </div>
            <div className="text-sm text-gray-500">Trust Attestations</div>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold">
              {metrics['sync.operations']?.value ?? 0}
            </div>
            <div className="text-sm text-gray-500">Sync Operations</div>
          </div>
        </div>
      </section>
    </div>
  );
};

// Icon components
const PlusIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="12" y1="5" x2="12" y2="19" />
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);

const InboxIcon: React.FC = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="22 12 16 12 14 15 10 15 8 12 2 12" />
    <path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" />
  </svg>
);

const SendIcon: React.FC = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);

const DraftIcon: React.FC = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
  </svg>
);

const TrashIcon: React.FC = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
  </svg>
);

const ContactsIcon: React.FC = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </svg>
);

const SettingsIcon: React.FC = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);

export default App;
