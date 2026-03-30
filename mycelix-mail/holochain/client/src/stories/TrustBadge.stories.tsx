// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * TrustBadge Component Stories
 */

import type { Meta, StoryObj } from '@storybook/react';
import { TrustBadge } from '../components/common/TrustBadge';

const meta: Meta<typeof TrustBadge> = {
  title: 'Common/TrustBadge',
  component: TrustBadge,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: 'Displays a visual indicator of trust level using MATL algorithm scores.',
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    trustLevel: {
      control: { type: 'range', min: 0, max: 1, step: 0.1 },
      description: 'Trust level from 0 (untrusted) to 1 (fully trusted)',
    },
    size: {
      control: { type: 'select' },
      options: ['sm', 'md', 'lg'],
      description: 'Size of the badge',
    },
    showLabel: {
      control: 'boolean',
      description: 'Whether to show the trust level label',
    },
    showPercentage: {
      control: 'boolean',
      description: 'Whether to show the percentage value',
    },
    interactive: {
      control: 'boolean',
      description: 'Whether the badge is interactive (clickable)',
    },
  },
};

export default meta;
type Story = StoryObj<typeof TrustBadge>;

export const Default: Story = {
  args: {
    trustLevel: 0.75,
    size: 'md',
    showLabel: true,
    showPercentage: false,
  },
};

export const HighTrust: Story = {
  args: {
    trustLevel: 0.95,
    size: 'md',
    showLabel: true,
    showPercentage: true,
  },
  parameters: {
    docs: {
      description: {
        story: 'A high trust level (80-100%) displays in blue with "Highly Trusted" label.',
      },
    },
  },
};

export const MediumTrust: Story = {
  args: {
    trustLevel: 0.55,
    size: 'md',
    showLabel: true,
    showPercentage: true,
  },
};

export const LowTrust: Story = {
  args: {
    trustLevel: 0.25,
    size: 'md',
    showLabel: true,
    showPercentage: true,
  },
};

export const Untrusted: Story = {
  args: {
    trustLevel: 0.1,
    size: 'md',
    showLabel: true,
    showPercentage: true,
  },
};

export const Unknown: Story = {
  args: {
    trustLevel: 0,
    size: 'md',
    showLabel: true,
    showPercentage: false,
  },
};

export const SmallSize: Story = {
  args: {
    trustLevel: 0.8,
    size: 'sm',
    showLabel: false,
    showPercentage: false,
  },
};

export const LargeSize: Story = {
  args: {
    trustLevel: 0.8,
    size: 'lg',
    showLabel: true,
    showPercentage: true,
  },
};

export const Interactive: Story = {
  args: {
    trustLevel: 0.75,
    size: 'md',
    showLabel: true,
    interactive: true,
    onClick: () => alert('Trust badge clicked!'),
  },
};

export const AllTrustLevels: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
      {[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1].map((level) => (
        <TrustBadge
          key={level}
          trustLevel={level}
          showLabel={true}
          showPercentage={true}
        />
      ))}
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Shows all trust level variations from unknown to fully trusted.',
      },
    },
  },
};
