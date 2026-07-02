// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Meta, StoryObj } from '@storybook/react';
import { CredibilityBadge } from '../../client/src/visualizations';

const meta: Meta<typeof CredibilityBadge> = {
  title: 'Components/CredibilityBadge',
  component: CredibilityBadge,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
# CredibilityBadge

A circular progress indicator displaying credibility scores.
Color-coded from red (low) through amber to green (high).

Uses a ring visualization that fills based on the score,
with optional numeric label and descriptive text.
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    score: {
      control: { type: 'range', min: 0, max: 1, step: 0.01 },
      description: 'Credibility score from 0 to 1',
    },
    size: {
      control: 'select',
      options: ['sm', 'md', 'lg'],
    },
    showLabel: {
      control: 'boolean',
    },
    showDescription: {
      control: 'boolean',
    },
    animated: {
      control: 'boolean',
    },
  },
};

export default meta;
type Story = StoryObj<typeof CredibilityBadge>;

export const Default: Story = {
  args: {
    score: 0.85,
    size: 'md',
    showLabel: true,
    showDescription: false,
    animated: true,
  },
};

export const AllSizes: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: '24px', alignItems: 'flex-end' }}>
      <CredibilityBadge score={0.85} size="sm" showLabel />
      <CredibilityBadge score={0.85} size="md" showLabel />
      <CredibilityBadge score={0.85} size="lg" showLabel />
    </div>
  ),
};

export const ScoreRange: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
      {[0.95, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1].map(score => (
        <CredibilityBadge key={score} score={score} size="md" showLabel showDescription />
      ))}
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Credibility badges across the full score range showing color transitions',
      },
    },
  },
};

export const WithDescription: Story = {
  args: {
    score: 0.78,
    size: 'lg',
    showLabel: true,
    showDescription: true,
  },
};

export const Excellent: Story = {
  args: {
    score: 0.95,
    size: 'lg',
    showLabel: true,
    showDescription: true,
  },
};

export const Poor: Story = {
  args: {
    score: 0.15,
    size: 'lg',
    showLabel: true,
    showDescription: true,
  },
};

export const NoAnimation: Story = {
  args: {
    score: 0.75,
    size: 'md',
    showLabel: true,
    animated: false,
  },
};
