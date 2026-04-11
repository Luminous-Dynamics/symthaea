// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Meta, StoryObj } from '@storybook/react';
import { TrustBadge } from './TrustBadge';
import type { TrustSummary, TrustTier } from '@/store/trustStore';

/**
 * The TrustBadge component displays a visual indicator of a contact's trust level
 * in the Mycelix decentralized trust network.
 *
 * Trust levels are determined by:
 * - Direct attestations from contacts you trust
 * - Transitive trust through your network
 * - Credential verification (DID, organizational membership)
 * - Historical interaction patterns
 */
const meta: Meta<typeof TrustBadge> = {
  title: 'Trust/TrustBadge',
  component: TrustBadge,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
The TrustBadge component displays a visual indicator of a contact's trust score
in the Mycelix trust network.

## Trust Tiers
- **High**: Score >= 70% - Green indicator, highly trusted contact
- **Medium**: Score 30-69% - Amber indicator, moderately trusted
- **Low**: Score < 30% - Rose indicator, low trust, proceed with caution
- **Unknown**: No score available - Gray indicator

## Features
- Compact mode for inline display
- Full mode with label and score
- Dark mode support
- Hover tooltip with trust reasons
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    summary: {
      description: 'Trust summary object containing score, tier, and reasons',
    },
    compact: {
      control: 'boolean',
      description: 'Show compact version without label text',
    },
    className: {
      control: 'text',
      description: 'Additional CSS classes',
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

// Helper to create trust summaries
const createSummary = (
  tier: TrustTier,
  score: number | undefined,
  reasons: string[]
): TrustSummary => ({
  tier,
  score,
  reasons,
});

/**
 * High trust contact - Someone you've directly attested or who has strong
 * network connections to people you trust.
 */
export const HighTrust: Story = {
  args: {
    summary: createSummary('high', 92, [
      'Direct attestation from you',
      'Verified organizational credential',
      'Long history of trusted interactions',
    ]),
    compact: false,
  },
};

/**
 * Medium trust - A contact known through your network but without
 * direct attestation or with moderate trust signals.
 */
export const MediumTrust: Story = {
  args: {
    summary: createSummary('medium', 55, [
      'Known through 2 trusted contacts',
      'New to your network',
    ]),
    compact: false,
  },
};

/**
 * Low trust - A contact with concerning signals or very weak
 * network connections.
 */
export const LowTrust: Story = {
  args: {
    summary: createSummary('low', 18, [
      'No direct trust path',
      'Suspicious email patterns detected',
    ]),
    compact: false,
  },
};

/**
 * Unknown trust - No trust information available for this contact.
 * This is common for first-time correspondents.
 */
export const UnknownTrust: Story = {
  args: {
    summary: createSummary('unknown', undefined, []),
    compact: false,
  },
};

/**
 * Compact mode - Smaller badge for use in email list items
 * and other space-constrained contexts.
 */
export const CompactMode: Story = {
  args: {
    summary: createSummary('high', 88, ['Trusted contact']),
    compact: true,
  },
};

/**
 * Compact variants showing all trust levels in compact mode.
 */
export const CompactVariants: Story = {
  render: () => (
    <div className="flex items-center gap-4">
      <TrustBadge summary={createSummary('high', 95, ['Direct trust'])} compact />
      <TrustBadge summary={createSummary('medium', 55, ['Known'])} compact />
      <TrustBadge summary={createSummary('low', 12, ['Caution'])} compact />
      <TrustBadge summary={createSummary('unknown', undefined, [])} compact />
    </div>
  ),
};

/**
 * Full variants showing all trust levels with labels.
 */
export const AllVariants: Story = {
  render: () => (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-4">
        <span className="w-24 text-sm text-gray-600">High:</span>
        <TrustBadge summary={createSummary('high', 95, ['Direct attestation', 'Verified credentials'])} />
      </div>
      <div className="flex items-center gap-4">
        <span className="w-24 text-sm text-gray-600">Medium:</span>
        <TrustBadge summary={createSummary('medium', 55, ['Known through network'])} />
      </div>
      <div className="flex items-center gap-4">
        <span className="w-24 text-sm text-gray-600">Low:</span>
        <TrustBadge summary={createSummary('low', 18, ['No trust path', 'New sender'])} />
      </div>
      <div className="flex items-center gap-4">
        <span className="w-24 text-sm text-gray-600">Unknown:</span>
        <TrustBadge summary={createSummary('unknown', undefined, [])} />
      </div>
    </div>
  ),
};

/**
 * Score range demonstration showing badges at various score levels.
 */
export const ScoreRange: Story = {
  render: () => (
    <div className="flex flex-col gap-3">
      {[100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0].map((score) => {
        const tier: TrustTier = score >= 70 ? 'high' : score >= 30 ? 'medium' : 'low';
        return (
          <div key={score} className="flex items-center gap-4">
            <span className="w-16 text-sm text-gray-600 text-right">{score}%</span>
            <TrustBadge summary={createSummary(tier, score, [`Score: ${score}`])} />
          </div>
        );
      })}
    </div>
  ),
};

/**
 * In context - How the badge looks within an email header.
 */
export const InEmailHeader: Story = {
  render: () => (
    <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 max-w-lg">
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center text-white font-semibold">
          JD
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-900 dark:text-gray-100">Jane Doe</span>
            <TrustBadge
              summary={createSummary('high', 87, [
                'Colleague at Acme Corp',
                'Verified work email',
              ])}
              compact
            />
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">jane.doe@acme.com</div>
        </div>
        <span className="text-sm text-gray-400">2:34 PM</span>
      </div>
      <div className="mt-3">
        <h3 className="font-medium text-gray-900 dark:text-gray-100">Q4 Budget Review</h3>
        <p className="text-sm text-gray-600 dark:text-gray-300 mt-1 line-clamp-2">
          Hi team, I've attached the Q4 budget review for your consideration.
          Please review and provide feedback by Friday...
        </p>
      </div>
    </div>
  ),
};

/**
 * In quarantine warning - Badge displayed in a quarantine alert.
 */
export const InQuarantineWarning: Story = {
  render: () => (
    <div className="p-4 bg-rose-50 dark:bg-rose-900/20 rounded-lg border border-rose-200 dark:border-rose-800 max-w-lg">
      <div className="flex items-start gap-3">
        <div className="text-rose-500 text-xl">!</div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-medium text-rose-800 dark:text-rose-200">
              Quarantined Message
            </span>
            <TrustBadge
              summary={createSummary('low', 8, [
                'Unknown sender',
                'Suspicious link detected',
                'First contact',
              ])}
            />
          </div>
          <p className="text-sm text-rose-700 dark:text-rose-300">
            This email was quarantined due to low trust signals. Review carefully before interacting.
          </p>
        </div>
      </div>
    </div>
  ),
};

/**
 * Dark mode demonstration.
 */
export const DarkMode: Story = {
  parameters: {
    backgrounds: { default: 'dark' },
  },
  render: () => (
    <div className="dark p-6 bg-gray-900 rounded-lg">
      <div className="flex flex-col gap-4">
        <TrustBadge summary={createSummary('high', 92, ['Trusted'])} />
        <TrustBadge summary={createSummary('medium', 55, ['Known'])} />
        <TrustBadge summary={createSummary('low', 15, ['Caution'])} />
        <TrustBadge summary={createSummary('unknown', undefined, [])} />
      </div>
    </div>
  ),
};
