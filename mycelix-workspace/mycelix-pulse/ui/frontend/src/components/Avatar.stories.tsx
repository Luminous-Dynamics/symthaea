// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Meta, StoryObj } from '@storybook/react';
import { action } from '@storybook/addon-actions';
import Avatar, { AvatarGroup } from './Avatar';

/**
 * The Avatar component displays a user's profile picture or their initials
 * as a fallback. It supports multiple sizes, tooltips, and integrates with
 * Gravatar for automatic profile picture loading.
 */
const meta: Meta<typeof Avatar> = {
  title: 'Components/Avatar',
  component: Avatar,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
The Avatar component displays user avatars with automatic Gravatar integration.

## Features
- **Gravatar support**: Automatically loads profile pictures from Gravatar
- **Fallback initials**: Shows initials when no Gravatar is available
- **Consistent colors**: Email-based color generation for consistent initials
- **Multiple sizes**: sm (32px), md (40px), lg (48px), xl (64px)
- **Interactive tooltip**: Shows name and email on hover
- **Click handler**: Optional click action for profiles

## Avatar Group
The AvatarGroup component displays multiple avatars with overlap,
perfect for showing participants in a thread or team members.
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    email: {
      control: 'text',
      description: 'Email address (used for Gravatar lookup and initials)',
    },
    name: {
      control: 'text',
      description: 'Display name (used for initials if provided)',
    },
    size: {
      control: 'select',
      options: ['sm', 'md', 'lg', 'xl'],
      description: 'Avatar size',
    },
    showTooltip: {
      control: 'boolean',
      description: 'Show name/email tooltip on hover',
    },
    onClick: { action: 'clicked' },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

/**
 * Default avatar with name - Shows initials derived from full name.
 */
export const Default: Story = {
  args: {
    email: 'jane.doe@example.com',
    name: 'Jane Doe',
    size: 'md',
    showTooltip: true,
  },
};

/**
 * Avatar without name - Uses email username for initials.
 */
export const EmailOnly: Story = {
  args: {
    email: 'johndoe@example.com',
    size: 'md',
    showTooltip: true,
  },
};

/**
 * Small size (32px) - For compact list items.
 */
export const Small: Story = {
  args: {
    email: 'alice@example.com',
    name: 'Alice Smith',
    size: 'sm',
  },
};

/**
 * Medium size (40px) - Default size for most use cases.
 */
export const Medium: Story = {
  args: {
    email: 'bob@example.com',
    name: 'Bob Johnson',
    size: 'md',
  },
};

/**
 * Large size (48px) - For email detail headers.
 */
export const Large: Story = {
  args: {
    email: 'carol@example.com',
    name: 'Carol Williams',
    size: 'lg',
  },
};

/**
 * Extra large (64px) - For profile pages and contact cards.
 */
export const ExtraLarge: Story = {
  args: {
    email: 'david@example.com',
    name: 'David Brown',
    size: 'xl',
  },
};

/**
 * All sizes comparison.
 */
export const AllSizes: Story = {
  render: () => (
    <div className="flex items-end gap-4">
      <div className="text-center">
        <Avatar email="user@example.com" name="User Name" size="sm" />
        <div className="text-xs text-gray-500 mt-1">sm</div>
      </div>
      <div className="text-center">
        <Avatar email="user@example.com" name="User Name" size="md" />
        <div className="text-xs text-gray-500 mt-1">md</div>
      </div>
      <div className="text-center">
        <Avatar email="user@example.com" name="User Name" size="lg" />
        <div className="text-xs text-gray-500 mt-1">lg</div>
      </div>
      <div className="text-center">
        <Avatar email="user@example.com" name="User Name" size="xl" />
        <div className="text-xs text-gray-500 mt-1">xl</div>
      </div>
    </div>
  ),
};

/**
 * Clickable avatar - Shows cursor and hover state.
 */
export const Clickable: Story = {
  args: {
    email: 'emma@example.com',
    name: 'Emma Wilson',
    size: 'lg',
    onClick: action('avatar clicked'),
  },
};

/**
 * Without tooltip.
 */
export const NoTooltip: Story = {
  args: {
    email: 'frank@example.com',
    name: 'Frank Miller',
    size: 'md',
    showTooltip: false,
  },
};

/**
 * Color consistency - Same email always produces same color.
 */
export const ColorConsistency: Story = {
  render: () => (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <Avatar email="alice@example.com" name="Alice" size="md" />
        <Avatar email="alice@example.com" name="Alice" size="lg" />
        <Avatar email="alice@example.com" name="Alice" size="xl" />
        <span className="text-sm text-gray-600">Same email = same color</span>
      </div>
      <div className="flex items-center gap-4">
        <Avatar email="bob@example.com" name="Bob" size="md" />
        <Avatar email="bob@example.com" name="Bob" size="lg" />
        <Avatar email="bob@example.com" name="Bob" size="xl" />
        <span className="text-sm text-gray-600">Different email = different color</span>
      </div>
    </div>
  ),
};

/**
 * Various email addresses showing color variety.
 */
export const ColorVariety: Story = {
  render: () => (
    <div className="flex flex-wrap gap-3">
      {[
        'alice@example.com',
        'bob@company.io',
        'carol@startup.co',
        'david@enterprise.org',
        'emma@freelance.net',
        'frank@agency.biz',
        'grace@nonprofit.org',
        'henry@tech.io',
        'iris@design.co',
        'jack@finance.com',
        'kate@legal.firm',
        'leo@marketing.co',
      ].map((email) => (
        <Avatar key={email} email={email} size="lg" showTooltip />
      ))}
    </div>
  ),
};

/**
 * Single name - Uses first two characters.
 */
export const SingleName: Story = {
  args: {
    email: 'admin@example.com',
    name: 'Admin',
    size: 'lg',
  },
};

/**
 * Multi-word name - Uses first and last initial.
 */
export const MultiWordName: Story = {
  args: {
    email: 'john.paul.jones@example.com',
    name: 'John Paul Jones',
    size: 'lg',
  },
};

// AvatarGroup Stories

/**
 * Avatar group - Shows multiple participants with overlap.
 */
export const Group: Story = {
  render: () => (
    <AvatarGroup
      emails={[
        { email: 'alice@example.com', name: 'Alice Smith' },
        { email: 'bob@example.com', name: 'Bob Johnson' },
        { email: 'carol@example.com', name: 'Carol Williams' },
      ]}
      size="md"
    />
  ),
};

/**
 * Large avatar group - Truncates to show "+N" for additional members.
 */
export const LargeGroup: Story = {
  render: () => (
    <AvatarGroup
      emails={[
        { email: 'user1@example.com', name: 'User One' },
        { email: 'user2@example.com', name: 'User Two' },
        { email: 'user3@example.com', name: 'User Three' },
        { email: 'user4@example.com', name: 'User Four' },
        { email: 'user5@example.com', name: 'User Five' },
        { email: 'user6@example.com', name: 'User Six' },
      ]}
      max={3}
      size="md"
    />
  ),
};

/**
 * Group size variants.
 */
export const GroupSizes: Story = {
  render: () => {
    const emails = [
      { email: 'a@example.com', name: 'Alice' },
      { email: 'b@example.com', name: 'Bob' },
      { email: 'c@example.com', name: 'Carol' },
      { email: 'd@example.com', name: 'David' },
    ];
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-4">
          <AvatarGroup emails={emails} max={3} size="sm" />
          <span className="text-sm text-gray-600">Small</span>
        </div>
        <div className="flex items-center gap-4">
          <AvatarGroup emails={emails} max={3} size="md" />
          <span className="text-sm text-gray-600">Medium</span>
        </div>
        <div className="flex items-center gap-4">
          <AvatarGroup emails={emails} max={3} size="lg" />
          <span className="text-sm text-gray-600">Large</span>
        </div>
      </div>
    );
  },
};

/**
 * In email thread context - Showing participants.
 */
export const InThreadContext: Story = {
  render: () => (
    <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 max-w-md">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium text-gray-900 dark:text-gray-100">Project Discussion</h3>
        <span className="text-xs text-gray-500">12 messages</span>
      </div>
      <div className="flex items-center gap-3">
        <AvatarGroup
          emails={[
            { email: 'alice@company.com', name: 'Alice Chen' },
            { email: 'bob@company.com', name: 'Bob Smith' },
            { email: 'carol@company.com', name: 'Carol Davis' },
            { email: 'david@company.com', name: 'David Lee' },
            { email: 'emma@company.com', name: 'Emma Wang' },
          ]}
          max={4}
          size="sm"
        />
        <span className="text-sm text-gray-600 dark:text-gray-400">
          5 participants
        </span>
      </div>
    </div>
  ),
};

/**
 * In contact card context.
 */
export const InContactCard: Story = {
  render: () => (
    <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 max-w-xs">
      <div className="flex flex-col items-center">
        <Avatar
          email="jane.doe@example.com"
          name="Jane Doe"
          size="xl"
          onClick={action('profile clicked')}
        />
        <h3 className="mt-3 font-semibold text-gray-900 dark:text-gray-100">
          Jane Doe
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          jane.doe@example.com
        </p>
        <div className="mt-3 flex items-center gap-2">
          <span className="px-2 py-1 text-xs bg-emerald-100 text-emerald-700 rounded-full">
            Trusted
          </span>
          <span className="text-xs text-gray-400">92% trust score</span>
        </div>
      </div>
    </div>
  ),
};

/**
 * In email header context.
 */
export const InEmailHeader: Story = {
  render: () => (
    <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 max-w-lg">
      <div className="flex items-start gap-3">
        <Avatar
          email="sender@company.com"
          name="Sarah Miller"
          size="lg"
          onClick={action('avatar clicked')}
        />
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-900 dark:text-gray-100">Sarah Miller</span>
            <span className="px-1.5 py-0.5 text-xs bg-emerald-100 text-emerald-700 rounded-full">
              Trusted
            </span>
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">
            sender@company.com
          </div>
          <div className="text-sm text-gray-400 mt-1">
            To: you@example.com
          </div>
        </div>
        <span className="text-sm text-gray-400">Today, 2:34 PM</span>
      </div>
    </div>
  ),
};

/**
 * Dark mode.
 */
export const DarkMode: Story = {
  parameters: {
    backgrounds: { default: 'dark' },
  },
  render: () => (
    <div className="dark p-6 bg-gray-900 rounded-lg">
      <div className="space-y-4">
        <div className="flex items-center gap-4">
          <Avatar email="alice@example.com" name="Alice Smith" size="lg" />
          <div>
            <div className="text-white font-medium">Alice Smith</div>
            <div className="text-gray-400 text-sm">alice@example.com</div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <Avatar email="bob@example.com" name="Bob Johnson" size="lg" />
          <div>
            <div className="text-white font-medium">Bob Johnson</div>
            <div className="text-gray-400 text-sm">bob@example.com</div>
          </div>
        </div>
        <div className="mt-4">
          <div className="text-gray-400 text-sm mb-2">Thread participants:</div>
          <AvatarGroup
            emails={[
              { email: 'user1@example.com', name: 'User One' },
              { email: 'user2@example.com', name: 'User Two' },
              { email: 'user3@example.com', name: 'User Three' },
              { email: 'user4@example.com', name: 'User Four' },
            ]}
            max={3}
            size="md"
          />
        </div>
      </div>
    </div>
  ),
};
