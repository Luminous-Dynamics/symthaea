// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Meta, StoryObj } from '@storybook/react';
import { VerdictIndicator, Verdict } from '../../client/src/visualizations';

const meta: Meta<typeof VerdictIndicator> = {
  title: 'Components/VerdictIndicator',
  component: VerdictIndicator,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
# VerdictIndicator

Displays fact-check verdicts with appropriate styling.

**Verdict Types:**
- ✓ **True** - Fully verified
- ◐ **Mostly True** - Minor inaccuracies
- ◑ **Mixed** - Contains both true and false elements
- ◔ **Mostly False** - Significant inaccuracies
- ✗ **False** - Demonstrably incorrect
- ? **Unverifiable** - Cannot be verified
- … **Insufficient Evidence** - Not enough data
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    verdict: {
      control: 'select',
      options: ['True', 'MostlyTrue', 'Mixed', 'MostlyFalse', 'False', 'Unverifiable', 'InsufficientEvidence'],
    },
    confidence: {
      control: { type: 'range', min: 0, max: 1, step: 0.01 },
    },
    size: {
      control: 'select',
      options: ['sm', 'md', 'lg'],
    },
    compact: {
      control: 'boolean',
    },
    showConfidence: {
      control: 'boolean',
    },
  },
};

export default meta;
type Story = StoryObj<typeof VerdictIndicator>;

export const True: Story = {
  args: {
    verdict: 'True',
    confidence: 0.95,
    size: 'md',
    showConfidence: true,
  },
};

export const MostlyTrue: Story = {
  args: {
    verdict: 'MostlyTrue',
    confidence: 0.82,
    size: 'md',
    showConfidence: true,
  },
};

export const Mixed: Story = {
  args: {
    verdict: 'Mixed',
    confidence: 0.7,
    size: 'md',
    showConfidence: true,
  },
};

export const MostlyFalse: Story = {
  args: {
    verdict: 'MostlyFalse',
    confidence: 0.78,
    size: 'md',
    showConfidence: true,
  },
};

export const False: Story = {
  args: {
    verdict: 'False',
    confidence: 0.92,
    size: 'md',
    showConfidence: true,
  },
};

export const Unverifiable: Story = {
  args: {
    verdict: 'Unverifiable',
    confidence: 0.6,
    size: 'md',
    showConfidence: true,
  },
};

export const InsufficientEvidence: Story = {
  args: {
    verdict: 'InsufficientEvidence',
    confidence: 0.45,
    size: 'md',
    showConfidence: true,
  },
};

export const AllVerdicts: Story = {
  render: () => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {(['True', 'MostlyTrue', 'Mixed', 'MostlyFalse', 'False', 'Unverifiable', 'InsufficientEvidence'] as Verdict[]).map(verdict => (
        <VerdictIndicator key={verdict} verdict={verdict} confidence={0.85} size="md" showConfidence />
      ))}
    </div>
  ),
};

export const CompactMode: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: '8px' }}>
      {(['True', 'MostlyTrue', 'Mixed', 'MostlyFalse', 'False'] as Verdict[]).map(verdict => (
        <VerdictIndicator key={verdict} verdict={verdict} compact size="md" />
      ))}
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Compact icon-only mode for inline use',
      },
    },
  },
};

export const Sizes: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: '24px', alignItems: 'flex-start' }}>
      <VerdictIndicator verdict="True" confidence={0.9} size="sm" showConfidence />
      <VerdictIndicator verdict="True" confidence={0.9} size="md" showConfidence />
      <VerdictIndicator verdict="True" confidence={0.9} size="lg" showConfidence />
    </div>
  ),
};
