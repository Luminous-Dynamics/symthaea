// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * EmailListItem Component Stories
 */

import type { Meta, StoryObj } from '@storybook/react';
import { EmailListItem } from '../components/inbox/EmailListItem';

const meta: Meta<typeof EmailListItem> = {
  title: 'Inbox/EmailListItem',
  component: EmailListItem,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: 'Individual email list item showing sender, subject, preview, and trust level.',
      },
    },
  },
  tags: ['autodocs'],
  decorators: [
    (Story) => (
      <div style={{ maxWidth: 600, backgroundColor: '#f8fafc', padding: 16 }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof EmailListItem>;

const baseEmail = {
  hash: 'uhCkkExample123',
  from: {
    name: 'Alice Johnson',
    email: 'alice@example.com',
    agentPubKey: 'uhCAkAgent123',
  },
  to: ['bob@example.com'],
  subject: 'Project Update - Q4 Planning',
  preview: 'Hi team, I wanted to share some updates on our Q4 planning. The new feature roadmap has been finalized...',
  body: 'Full email content here',
  timestamp: Date.now() - 1000 * 60 * 30, // 30 minutes ago
  isRead: false,
  isStarred: false,
  hasAttachments: false,
  trustLevel: 0.85,
  labels: [],
};

export const Default: Story = {
  args: {
    email: baseEmail,
    onClick: () => console.log('Email clicked'),
  },
};

export const Unread: Story = {
  args: {
    email: {
      ...baseEmail,
      isRead: false,
    },
    onClick: () => console.log('Email clicked'),
  },
  parameters: {
    docs: {
      description: {
        story: 'Unread emails have a bold subject and slightly different background.',
      },
    },
  },
};

export const Read: Story = {
  args: {
    email: {
      ...baseEmail,
      isRead: true,
    },
    onClick: () => console.log('Email clicked'),
  },
};

export const Starred: Story = {
  args: {
    email: {
      ...baseEmail,
      isStarred: true,
    },
    onClick: () => console.log('Email clicked'),
    onToggleStar: () => console.log('Star toggled'),
  },
};

export const WithAttachments: Story = {
  args: {
    email: {
      ...baseEmail,
      hasAttachments: true,
      attachments: [
        { name: 'document.pdf', size: 1024 * 500 },
        { name: 'image.png', size: 1024 * 200 },
      ],
    },
    onClick: () => console.log('Email clicked'),
  },
};

export const WithLabels: Story = {
  args: {
    email: {
      ...baseEmail,
      labels: ['work', 'important', 'follow-up'],
    },
    onClick: () => console.log('Email clicked'),
  },
};

export const LowTrust: Story = {
  args: {
    email: {
      ...baseEmail,
      trustLevel: 0.25,
      from: {
        name: 'Unknown Sender',
        email: 'unknown@suspicious.com',
        agentPubKey: 'uhCAkUnknown',
      },
    },
    onClick: () => console.log('Email clicked'),
  },
  parameters: {
    docs: {
      description: {
        story: 'Low trust senders show a warning indicator.',
      },
    },
  },
};

export const Selected: Story = {
  args: {
    email: baseEmail,
    isSelected: true,
    onClick: () => console.log('Email clicked'),
  },
};

export const CompactView: Story = {
  args: {
    email: baseEmail,
    compact: true,
    onClick: () => console.log('Email clicked'),
  },
};

export const LongSubject: Story = {
  args: {
    email: {
      ...baseEmail,
      subject: 'This is a very long email subject that should be truncated with an ellipsis when it exceeds the available width',
    },
    onClick: () => console.log('Email clicked'),
  },
};

export const OldEmail: Story = {
  args: {
    email: {
      ...baseEmail,
      timestamp: Date.now() - 1000 * 60 * 60 * 24 * 7, // 7 days ago
    },
    onClick: () => console.log('Email clicked'),
  },
};

export const EmailList: Story = {
  render: () => {
    const emails = [
      { ...baseEmail, hash: '1', isRead: false, isStarred: true },
      { ...baseEmail, hash: '2', isRead: true, subject: 'Meeting Tomorrow', from: { ...baseEmail.from, name: 'Bob Smith' } },
      { ...baseEmail, hash: '3', isRead: true, trustLevel: 0.3, hasAttachments: true },
      { ...baseEmail, hash: '4', isRead: false, labels: ['urgent'] },
    ];

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        {emails.map((email) => (
          <EmailListItem
            key={email.hash}
            email={email}
            onClick={() => console.log('Clicked', email.hash)}
          />
        ))}
      </div>
    );
  },
  parameters: {
    docs: {
      description: {
        story: 'Multiple email items stacked to show how they appear in a list.',
      },
    },
  },
};
