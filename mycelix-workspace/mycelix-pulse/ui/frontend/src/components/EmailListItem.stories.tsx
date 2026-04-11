// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Meta, StoryObj } from '@storybook/react';
import { action } from '@storybook/addon-actions';
import type { Email } from '../lib/hooks/useEmails';

/**
 * EmailListItem displays a single email entry in the mail list.
 * Since the actual component has complex dependencies (i18n, accessibility, Holochain),
 * we create a standalone story component that mirrors its functionality.
 */

// Story-specific EmailListItem component (mirrors actual component behavior)
function EmailListItemStory({
  email,
  isSelected = false,
  onSelect,
  onStar,
  onMarkRead,
  onArchive,
  onDelete,
}: {
  email: Email;
  isSelected?: boolean;
  onSelect?: (id: string) => void;
  onStar?: (id: string, isStarred: boolean) => void;
  onMarkRead?: (id: string, isRead: boolean) => void;
  onArchive?: (id: string) => void;
  onDelete?: (id: string) => void;
}) {
  const trustScore = email.trustScore ?? 0.5;

  const getColor = (s: number) => {
    if (s >= 0.8) return '#4CAF50';
    if (s >= 0.5) return '#FFC107';
    if (s >= 0.2) return '#FF9800';
    return '#F44336';
  };

  const formatRelativeTime = (dateStr?: string) => {
    if (!dateStr) return 'Unknown';
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  const color = getColor(trustScore);

  return (
    <article
      role="listitem"
      tabIndex={0}
      onClick={() => onSelect?.(email.id)}
      aria-selected={isSelected}
      aria-label={`${email.isRead ? '' : 'Unread '}Email from ${email.from.name || email.from.email}: ${email.subject}`}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        padding: '12px 16px',
        borderBottom: '1px solid #e0e0e0',
        backgroundColor: isSelected ? '#e3f2fd' : email.isRead ? '#ffffff' : '#f5f5f5',
        cursor: 'pointer',
        transition: 'background-color 0.15s ease',
        fontWeight: email.isRead ? 'normal' : 'bold',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      }}
    >
      {/* Star button */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onStar?.(email.id, !email.isStarred);
        }}
        aria-label={email.isStarred ? 'Unstar' : 'Star'}
        aria-pressed={email.isStarred}
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          padding: '4px',
          fontSize: '18px',
          color: email.isStarred ? '#FFC107' : '#9e9e9e',
        }}
      >
        {email.isStarred ? '\u2605' : '\u2606'}
      </button>

      {/* Trust badge */}
      <div
        style={{
          width: 16,
          height: 16,
          borderRadius: '50%',
          backgroundColor: color,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white',
          fontSize: 10,
          fontWeight: 'bold',
          flexShrink: 0,
        }}
        title={`Trust: ${Math.round(trustScore * 100)}%`}
      >
        {trustScore >= 0.8 ? '\u2713' : trustScore >= 0.5 ? '~' : trustScore >= 0.2 ? '!' : '?'}
      </div>

      {/* Sender */}
      <div
        style={{
          width: '180px',
          flexShrink: 0,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {email.from.name || email.from.email}
      </div>

      {/* Subject and preview */}
      <div
        style={{
          flex: 1,
          overflow: 'hidden',
          display: 'flex',
          gap: '8px',
        }}
      >
        <span
          style={{
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {email.subject || '(No subject)'}
        </span>
        {email.bodyText && (
          <span
            style={{
              color: '#666',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              fontWeight: 'normal',
            }}
          >
            - {email.bodyText.slice(0, 100)}
          </span>
        )}
      </div>

      {/* Attachments indicator */}
      {email.attachments && email.attachments.length > 0 && (
        <span
          aria-label={`${email.attachments.length} attachments`}
          style={{ color: '#666' }}
        >
          \uD83D\uDCCE {email.attachments.length}
        </span>
      )}

      {/* Labels */}
      {email.labels && email.labels.length > 0 && (
        <div style={{ display: 'flex', gap: '4px' }}>
          {email.labels.slice(0, 2).map((label) => (
            <span
              key={label}
              style={{
                backgroundColor: '#e0e0e0',
                padding: '2px 6px',
                borderRadius: '4px',
                fontSize: '11px',
              }}
            >
              {label}
            </span>
          ))}
        </div>
      )}

      {/* Date */}
      <time
        dateTime={email.receivedAt || email.sentAt}
        style={{
          color: '#666',
          fontSize: '13px',
          whiteSpace: 'nowrap',
        }}
      >
        {formatRelativeTime(email.receivedAt || email.sentAt)}
      </time>
    </article>
  );
}

const meta: Meta<typeof EmailListItemStory> = {
  title: 'Email/EmailListItem',
  component: EmailListItemStory,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: `
The EmailListItem component displays a single email entry with trust integration.

## Features
- **Trust badge**: Visual indicator of sender's trust level
- **Star toggle**: Quick favorite/unfavorite action
- **Read state**: Bold text for unread emails
- **Selection**: Highlighted background when selected
- **Labels**: Shows up to 2 labels inline
- **Attachments**: Paper clip icon with count
- **Relative time**: Human-readable timestamps
- **Keyboard support**: Full keyboard navigation

## Keyboard Shortcuts (when focused)
- Enter/Space: Select email
- S: Toggle star
- R: Toggle read state
- E: Archive
- Delete/Backspace: Delete
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    isSelected: {
      control: 'boolean',
      description: 'Whether this email is currently selected',
    },
    onSelect: { action: 'selected' },
    onStar: { action: 'starred' },
    onMarkRead: { action: 'marked read' },
    onArchive: { action: 'archived' },
    onDelete: { action: 'deleted' },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

// Helper to create sample emails
const createEmail = (overrides: Partial<Email> = {}): Email => ({
  id: `email-${Math.random().toString(36).substr(2, 9)}`,
  subject: 'Meeting tomorrow',
  bodyText: 'Hi there, I wanted to confirm our meeting scheduled for tomorrow at 2pm...',
  from: { email: 'sender@example.com', name: 'John Doe' },
  to: [{ email: 'you@example.com', name: 'You' }],
  cc: [],
  isRead: false,
  isStarred: false,
  isArchived: false,
  trustScore: 0.75,
  labels: [],
  receivedAt: new Date().toISOString(),
  attachments: [],
  ...overrides,
});

/**
 * Default unread email with moderate trust.
 */
export const Default: Story = {
  args: {
    email: createEmail(),
    isSelected: false,
    onSelect: action('onSelect'),
    onStar: action('onStar'),
    onMarkRead: action('onMarkRead'),
    onArchive: action('onArchive'),
    onDelete: action('onDelete'),
  },
};

/**
 * Unread email - Shows bold text styling.
 */
export const Unread: Story = {
  args: {
    email: createEmail({
      isRead: false,
      subject: 'Important: Action Required',
      from: { email: 'manager@company.com', name: 'Sarah Manager' },
    }),
  },
};

/**
 * Read email - Normal text weight.
 */
export const Read: Story = {
  args: {
    email: createEmail({
      isRead: true,
      subject: 'Weekly newsletter',
      bodyText: 'This week in tech: Top stories and updates from around the industry...',
    }),
  },
};

/**
 * Selected email - Highlighted background.
 */
export const Selected: Story = {
  args: {
    email: createEmail(),
    isSelected: true,
  },
};

/**
 * Starred email - Yellow star indicator.
 */
export const Starred: Story = {
  args: {
    email: createEmail({
      isStarred: true,
      subject: 'Project proposal - please review',
      from: { email: 'partner@startup.io', name: 'Alex Partner' },
    }),
  },
};

/**
 * High trust sender - Green trust badge.
 */
export const HighTrust: Story = {
  args: {
    email: createEmail({
      trustScore: 0.92,
      subject: 'Verified contract signed',
      from: { email: 'legal@trusted-corp.com', name: 'Legal Department' },
    }),
  },
};

/**
 * Medium trust sender - Yellow trust badge.
 */
export const MediumTrust: Story = {
  args: {
    email: createEmail({
      trustScore: 0.55,
      subject: 'Introduction from Bob',
      from: { email: 'new.contact@external.com', name: 'New Contact' },
    }),
  },
};

/**
 * Low trust sender - Orange trust badge with warning.
 */
export const LowTrust: Story = {
  args: {
    email: createEmail({
      trustScore: 0.25,
      subject: 'Fw: Fw: Fw: You won!!!',
      bodyText: 'Click here to claim your prize...',
      from: { email: 'winner@suspicious-site.xyz', name: undefined },
    }),
  },
};

/**
 * Unknown trust sender - Red trust badge.
 */
export const UnknownTrust: Story = {
  args: {
    email: createEmail({
      trustScore: 0.1,
      subject: 'Account verification needed',
      from: { email: 'noreply@unknowndomain.net' },
    }),
  },
};

/**
 * With attachments - Shows paper clip icon.
 */
export const WithAttachments: Story = {
  args: {
    email: createEmail({
      subject: 'Q4 Report attached',
      attachments: [
        { id: '1', filename: 'Q4_Report.pdf', contentType: 'application/pdf', size: 2456000 },
        { id: '2', filename: 'Charts.xlsx', contentType: 'application/vnd.ms-excel', size: 156000 },
      ],
    }),
  },
};

/**
 * With labels - Shows label chips.
 */
export const WithLabels: Story = {
  args: {
    email: createEmail({
      subject: 'Sprint planning notes',
      labels: ['Work', 'Important', 'Project-X'],
    }),
  },
};

/**
 * Long content - Tests text truncation.
 */
export const LongContent: Story = {
  args: {
    email: createEmail({
      subject: 'This is an extremely long subject line that should definitely be truncated when displayed in the email list item component',
      bodyText: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.',
      from: { email: 'verbose.sender@long-company-name-incorporated.com', name: 'Very Long Sender Name That Should Also Truncate' },
    }),
  },
};

/**
 * No subject - Shows placeholder text.
 */
export const NoSubject: Story = {
  args: {
    email: createEmail({
      subject: '',
      bodyText: 'Quick message with no subject line...',
    }),
  },
};

/**
 * Email from earlier today.
 */
export const ReceivedToday: Story = {
  args: {
    email: createEmail({
      receivedAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(), // 2 hours ago
      subject: 'Lunch meeting reminder',
    }),
  },
};

/**
 * Email from yesterday.
 */
export const ReceivedYesterday: Story = {
  args: {
    email: createEmail({
      receivedAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(), // 1 day ago
      subject: "Yesterday's notes",
    }),
  },
};

/**
 * Email list - Multiple items together.
 */
export const EmailList: Story = {
  render: () => (
    <div style={{ maxWidth: '800px', border: '1px solid #e0e0e0', borderRadius: '8px', overflow: 'hidden' }}>
      <EmailListItemStory
        email={createEmail({
          isRead: false,
          isStarred: true,
          trustScore: 0.95,
          subject: 'Urgent: Server maintenance tonight',
          from: { email: 'ops@company.com', name: 'IT Operations' },
          labels: ['Important'],
          receivedAt: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
        })}
        onSelect={action('onSelect')}
        onStar={action('onStar')}
      />
      <EmailListItemStory
        email={createEmail({
          isRead: false,
          trustScore: 0.78,
          subject: 'Re: Project timeline update',
          from: { email: 'pm@company.com', name: 'Project Manager' },
          receivedAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
        })}
        isSelected={true}
        onSelect={action('onSelect')}
        onStar={action('onStar')}
      />
      <EmailListItemStory
        email={createEmail({
          isRead: true,
          trustScore: 0.55,
          subject: 'Weekly team sync - notes',
          bodyText: 'Here are the notes from today\'s meeting. Please review and add any items I missed...',
          from: { email: 'colleague@company.com', name: 'Team Lead' },
          attachments: [{ id: '1', filename: 'notes.pdf', contentType: 'application/pdf', size: 125000 }],
          receivedAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        })}
        onSelect={action('onSelect')}
        onStar={action('onStar')}
      />
      <EmailListItemStory
        email={createEmail({
          isRead: true,
          trustScore: 0.22,
          subject: 'External: Partnership proposal',
          from: { email: 'sales@unknown-vendor.com', name: 'Sales Team' },
          labels: ['External'],
          receivedAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
        })}
        onSelect={action('onSelect')}
        onStar={action('onStar')}
      />
    </div>
  ),
};

/**
 * Trust level comparison in list context.
 */
export const TrustLevelComparison: Story = {
  render: () => (
    <div style={{ maxWidth: '800px', border: '1px solid #e0e0e0', borderRadius: '8px', overflow: 'hidden' }}>
      <div style={{ padding: '8px 16px', backgroundColor: '#f5f5f5', fontWeight: 'bold', fontSize: '12px', color: '#666' }}>
        TRUST LEVEL COMPARISON
      </div>
      <EmailListItemStory
        email={createEmail({
          trustScore: 0.95,
          subject: 'From trusted colleague',
          from: { email: 'trusted@company.com', name: 'Trusted Colleague' },
        })}
        onSelect={action('onSelect')}
        onStar={action('onStar')}
      />
      <EmailListItemStory
        email={createEmail({
          trustScore: 0.65,
          subject: 'From known contact',
          from: { email: 'known@partner.com', name: 'Known Contact' },
        })}
        onSelect={action('onSelect')}
        onStar={action('onStar')}
      />
      <EmailListItemStory
        email={createEmail({
          trustScore: 0.35,
          subject: 'From new contact (caution)',
          from: { email: 'new@external.com', name: 'New Contact' },
        })}
        onSelect={action('onSelect')}
        onStar={action('onStar')}
      />
      <EmailListItemStory
        email={createEmail({
          trustScore: 0.1,
          subject: 'From unknown sender',
          from: { email: 'unknown@suspicious.xyz' },
        })}
        onSelect={action('onSelect')}
        onStar={action('onStar')}
      />
    </div>
  ),
};
