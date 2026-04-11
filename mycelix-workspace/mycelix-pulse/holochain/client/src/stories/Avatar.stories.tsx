// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Avatar Component Stories
 */

import type { Meta, StoryObj } from '@storybook/react';
import { Avatar } from '../components/common/Avatar';

const meta: Meta<typeof Avatar> = {
  title: 'Common/Avatar',
  component: Avatar,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: 'User avatar component with image, initials fallback, and trust indicator.',
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    size: {
      control: { type: 'select' },
      options: ['xs', 'sm', 'md', 'lg', 'xl'],
    },
    name: {
      control: 'text',
      description: 'User name for generating initials fallback',
    },
    email: {
      control: 'text',
      description: 'Email address (used if name not provided)',
    },
    imageUrl: {
      control: 'text',
      description: 'URL of the avatar image',
    },
    showTrustIndicator: {
      control: 'boolean',
      description: 'Whether to show the trust level indicator ring',
    },
    trustLevel: {
      control: { type: 'range', min: 0, max: 1, step: 0.1 },
    },
    isOnline: {
      control: 'boolean',
      description: 'Show online status indicator',
    },
  },
};

export default meta;
type Story = StoryObj<typeof Avatar>;

export const Default: Story = {
  args: {
    name: 'John Doe',
    size: 'md',
  },
};

export const WithImage: Story = {
  args: {
    name: 'John Doe',
    imageUrl: 'https://i.pravatar.cc/150?u=john',
    size: 'md',
  },
};

export const WithTrustIndicator: Story = {
  args: {
    name: 'Alice Smith',
    trustLevel: 0.85,
    showTrustIndicator: true,
    size: 'lg',
  },
  parameters: {
    docs: {
      description: {
        story: 'Avatar with a colored ring indicating trust level.',
      },
    },
  },
};

export const OnlineStatus: Story = {
  args: {
    name: 'Bob Wilson',
    isOnline: true,
    size: 'lg',
  },
};

export const EmailFallback: Story = {
  args: {
    email: 'user@example.com',
    size: 'md',
  },
  parameters: {
    docs: {
      description: {
        story: 'When only email is provided, uses first letter of email.',
      },
    },
  },
};

export const AllSizes: Story = {
  render: () => (
    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
      <Avatar name="User" size="xs" />
      <Avatar name="User" size="sm" />
      <Avatar name="User" size="md" />
      <Avatar name="User" size="lg" />
      <Avatar name="User" size="xl" />
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Avatar available in 5 sizes: xs (24px), sm (32px), md (40px), lg (48px), xl (64px).',
      },
    },
  },
};

export const TrustLevelVariants: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: '16px' }}>
      <Avatar name="Untrusted" trustLevel={0.1} showTrustIndicator size="lg" />
      <Avatar name="Low Trust" trustLevel={0.3} showTrustIndicator size="lg" />
      <Avatar name="Medium" trustLevel={0.5} showTrustIndicator size="lg" />
      <Avatar name="Trusted" trustLevel={0.7} showTrustIndicator size="lg" />
      <Avatar name="High Trust" trustLevel={0.9} showTrustIndicator size="lg" />
    </div>
  ),
};

export const AvatarGroup: Story = {
  render: () => (
    <div style={{ display: 'flex' }}>
      {['Alice', 'Bob', 'Carol', 'Dave', 'Eve'].map((name, i) => (
        <div key={name} style={{ marginLeft: i === 0 ? 0 : -12 }}>
          <Avatar
            name={name}
            imageUrl={`https://i.pravatar.cc/100?u=${name.toLowerCase()}`}
            size="md"
          />
        </div>
      ))}
      <div
        style={{
          marginLeft: -12,
          width: 40,
          height: 40,
          borderRadius: '50%',
          backgroundColor: '#e2e8f0',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 12,
          fontWeight: 500,
          color: '#64748b',
        }}
      >
        +5
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Avatars can be stacked to show groups of users.',
      },
    },
  },
};
