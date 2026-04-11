// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * MobileEmailList Component Stories
 */

import React, { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react';
import { MobileEmailList, Email } from '../mobile/MobileEmailList';

const meta: Meta<typeof MobileEmailList> = {
  title: 'Mobile/MobileEmailList',
  component: MobileEmailList,
  parameters: {
    layout: 'fullscreen',
    viewport: {
      defaultViewport: 'mobile1',
    },
    docs: {
      description: {
        component: 'Mobile-optimized email list with swipe actions, pull-to-refresh, and long-press selection.',
      },
    },
  },
  tags: ['autodocs'],
};

export default meta;
type Story = StoryObj<typeof MobileEmailList>;

// Generate sample emails
const generateEmails = (count: number): Email[] => {
  const senders = [
    { name: 'Alice Johnson', email: 'alice@example.com' },
    { name: 'Bob Smith', email: 'bob@company.org' },
    { name: 'Carol Williams', email: 'carol@business.net' },
    { name: 'Dave Brown', email: 'dave@startup.io' },
    { name: 'Eve Davis', email: 'eve@agency.co' },
  ];

  const subjects = [
    'Project Update - Q4 Planning',
    'Meeting Tomorrow at 2 PM',
    'Invoice #1234 Attached',
    'Quick question about the proposal',
    'Welcome to the team!',
    'Reminder: Submit timesheet',
    'New feature release notes',
    'Your account verification',
    'Weekly newsletter',
    'Action required: Review document',
  ];

  const previews = [
    'Hi team, I wanted to share some updates on our progress this quarter...',
    'Just a reminder that we have a meeting scheduled for tomorrow...',
    'Please find the attached invoice for services rendered...',
    'I was wondering if you could clarify a few points in the proposal...',
    'We are excited to have you join our team! Here is everything you need...',
    'This is a friendly reminder to submit your timesheet by end of day...',
    'We are thrilled to announce the release of version 2.0 with new features...',
    'Please verify your account by clicking the link below...',
    'Here is your weekly digest of the most important updates...',
    'A document requires your review and approval. Please take action...',
  ];

  const labels = ['work', 'personal', 'important', 'follow-up', 'later'];

  return Array.from({ length: count }, (_, i) => {
    const sender = senders[i % senders.length];
    const daysAgo = Math.floor(Math.random() * 14);
    const hoursAgo = Math.floor(Math.random() * 24);

    return {
      hash: `email-${i}`,
      from: sender,
      subject: subjects[i % subjects.length],
      preview: previews[i % previews.length],
      timestamp: Date.now() - (daysAgo * 24 * 60 * 60 * 1000) - (hoursAgo * 60 * 60 * 1000),
      isRead: Math.random() > 0.4,
      isStarred: Math.random() > 0.8,
      hasAttachments: Math.random() > 0.7,
      trustLevel: 0.3 + Math.random() * 0.6,
      labels: Math.random() > 0.6 ? [labels[Math.floor(Math.random() * labels.length)]] : [],
    };
  });
};

const defaultEmails = generateEmails(20);

// Wrapper with state management
const MobileEmailListWrapper: React.FC<{
  initialEmails?: Email[];
  hasMore?: boolean;
}> = ({ initialEmails = defaultEmails, hasMore = true }) => {
  const [emails, setEmails] = useState(initialEmails);
  const [isLoading, setIsLoading] = useState(false);

  const handleRefresh = async () => {
    setIsLoading(true);
    await new Promise((resolve) => setTimeout(resolve, 1500));
    setEmails(generateEmails(20));
    setIsLoading(false);
  };

  const handleArchive = async (hash: string) => {
    await new Promise((resolve) => setTimeout(resolve, 300));
    setEmails((prev) => prev.filter((e) => e.hash !== hash));
  };

  const handleDelete = async (hash: string) => {
    await new Promise((resolve) => setTimeout(resolve, 300));
    setEmails((prev) => prev.filter((e) => e.hash !== hash));
  };

  const handleToggleRead = async (hash: string) => {
    setEmails((prev) =>
      prev.map((e) => (e.hash === hash ? { ...e, isRead: !e.isRead } : e))
    );
  };

  const handleToggleStar = async (hash: string) => {
    setEmails((prev) =>
      prev.map((e) => (e.hash === hash ? { ...e, isStarred: !e.isStarred } : e))
    );
  };

  const handleLoadMore = async () => {
    setIsLoading(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setEmails((prev) => [...prev, ...generateEmails(10)]);
    setIsLoading(false);
  };

  return (
    <MobileEmailList
      emails={emails}
      onRefresh={handleRefresh}
      onEmailClick={(email) => console.log('Open email:', email.subject)}
      onArchive={handleArchive}
      onDelete={handleDelete}
      onToggleRead={handleToggleRead}
      onToggleStar={handleToggleStar}
      onSelectionChange={(ids) => console.log('Selected:', ids.size)}
      isLoading={isLoading}
      hasMore={hasMore}
      onLoadMore={handleLoadMore}
    />
  );
};

export const Default: Story = {
  render: () => <MobileEmailListWrapper />,
};

export const EmptyInbox: Story = {
  render: () => <MobileEmailListWrapper initialEmails={[]} hasMore={false} />,
  parameters: {
    docs: {
      description: {
        story: 'Empty inbox state with pull-to-refresh hint.',
      },
    },
  },
};

export const AllUnread: Story = {
  render: () => (
    <MobileEmailListWrapper
      initialEmails={generateEmails(10).map((e) => ({ ...e, isRead: false }))}
    />
  ),
  parameters: {
    docs: {
      description: {
        story: 'All emails shown as unread with bold styling.',
      },
    },
  },
};

export const AllRead: Story = {
  render: () => (
    <MobileEmailListWrapper
      initialEmails={generateEmails(10).map((e) => ({ ...e, isRead: true }))}
    />
  ),
};

export const WithLowTrust: Story = {
  render: () => (
    <MobileEmailListWrapper
      initialEmails={generateEmails(10).map((e, i) => ({
        ...e,
        trustLevel: i < 3 ? 0.15 : e.trustLevel,
        from: i < 3 ? { name: 'Unknown Sender', email: 'unknown@suspicious.net' } : e.from,
      }))}
    />
  ),
  parameters: {
    docs: {
      description: {
        story: 'Shows how low-trust senders are displayed with warning colors.',
      },
    },
  },
};

export const ManyLabels: Story = {
  render: () => (
    <MobileEmailListWrapper
      initialEmails={generateEmails(10).map((e) => ({
        ...e,
        labels: ['work', 'important', 'follow-up'].slice(0, Math.floor(Math.random() * 3) + 1),
      }))}
    />
  ),
};

export const LongSubjects: Story = {
  render: () => (
    <MobileEmailListWrapper
      initialEmails={generateEmails(5).map((e) => ({
        ...e,
        subject: 'This is an extremely long email subject that should be truncated properly on mobile devices to maintain the layout',
      }))}
    />
  ),
  parameters: {
    docs: {
      description: {
        story: 'Tests text truncation with very long subject lines.',
      },
    },
  },
};

// Interactive demo with instructions
export const SwipeDemo: Story = {
  render: () => (
    <div>
      <div
        style={{
          padding: 16,
          backgroundColor: '#f0f9ff',
          borderBottom: '1px solid #bae6fd',
        }}
      >
        <h3 style={{ margin: 0, fontSize: 16, color: '#0369a1' }}>Swipe Gestures Demo</h3>
        <ul style={{ margin: '8px 0 0', paddingLeft: 20, fontSize: 14, color: '#0c4a6e' }}>
          <li>Swipe left → Delete</li>
          <li>Swipe right → Archive</li>
          <li>Long press → Select multiple</li>
          <li>Pull down → Refresh</li>
        </ul>
      </div>
      <MobileEmailListWrapper />
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Interactive demo showing all available swipe gestures.',
      },
    },
  },
};

// Tablet viewport
export const TabletView: Story = {
  render: () => <MobileEmailListWrapper />,
  parameters: {
    viewport: {
      defaultViewport: 'tablet',
    },
    docs: {
      description: {
        story: 'Email list on tablet-sized viewport.',
      },
    },
  },
};
