// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Meta, StoryObj } from '@storybook/react';
import TrustGraphVisualizer, { TrustPathInline } from './TrustGraphVisualizer';

/**
 * The TrustGraphVisualizer component displays an interactive visualization of the
 * trust path between you and a sender in the Mycelix decentralized trust network.
 *
 * It shows:
 * - The chain of trust connections from you to the sender
 * - Trust scores for each hop in the path
 * - Relationship types (direct trust, vouches, credentials, etc.)
 * - Visual indicators for weak links in the trust chain
 */
const meta: Meta<typeof TrustGraphVisualizer> = {
  title: 'Trust/TrustGraphVisualizer',
  component: TrustGraphVisualizer,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: `
The TrustGraphVisualizer displays the trust path between you and an email sender.

## Features
- **Interactive nodes**: Each person in the trust chain is shown as a node
- **Relationship edges**: Colored lines show the type of trust relationship
- **Trust scores**: Percentage indicators on each hop
- **Path explanation**: Human-readable description of the trust chain
- **Weak link warnings**: Highlights concerning connections

## Relationship Types
- **Direct Trust** (Green): You've personally vouched for this person
- **Credential Issuer** (Indigo): An organization issued credentials
- **Organization Member** (Purple): Shared organizational membership
- **Vouch** (Amber): Someone vouched for this person
- **Introduction** (Pink): Formal introduction through the network
- **Transitive Trust** (Gray): Indirect connection
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    compact: {
      control: 'boolean',
      description: 'Show compact version with smaller nodes and less padding',
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

// Sample trust paths
const directTrustPath = {
  hops: [
    {
      from: 'did:mycelix:you123',
      to: 'did:mycelix:alice456',
      from_name: 'You',
      to_name: 'Alice Chen',
      relationship: 'direct_trust',
      trust_score: 0.95,
      reason: 'Personal friend and colleague',
    },
  ],
  final_trust: 0.95,
  path_length: 1,
  strongest_link: undefined,
  weakest_link: undefined,
};

const twoHopPath = {
  hops: [
    {
      from: 'did:mycelix:you123',
      to: 'did:mycelix:bob789',
      from_name: 'You',
      to_name: 'Bob Smith',
      relationship: 'direct_trust',
      trust_score: 0.88,
      reason: 'Work colleague',
    },
    {
      from: 'did:mycelix:bob789',
      to: 'did:mycelix:carol012',
      from_name: 'Bob Smith',
      to_name: 'Carol Johnson',
      relationship: 'vouch',
      trust_score: 0.72,
      reason: 'Bob vouched for Carol',
    },
  ],
  final_trust: 0.63,
  path_length: 2,
  strongest_link: {
    from: 'did:mycelix:you123',
    to: 'did:mycelix:bob789',
    from_name: 'You',
    to_name: 'Bob Smith',
    relationship: 'direct_trust',
    trust_score: 0.88,
    reason: 'Work colleague',
  },
  weakest_link: {
    from: 'did:mycelix:bob789',
    to: 'did:mycelix:carol012',
    from_name: 'Bob Smith',
    to_name: 'Carol Johnson',
    relationship: 'vouch',
    trust_score: 0.72,
    reason: 'Bob vouched for Carol',
  },
};

const threeHopPath = {
  hops: [
    {
      from: 'did:mycelix:you123',
      to: 'did:mycelix:david345',
      from_name: 'You',
      to_name: 'David Lee',
      relationship: 'organization_member',
      trust_score: 0.82,
      reason: 'Same organization',
    },
    {
      from: 'did:mycelix:david345',
      to: 'did:mycelix:emma678',
      from_name: 'David Lee',
      to_name: 'Emma Wilson',
      relationship: 'credential_issuer',
      trust_score: 0.90,
      reason: 'Issued verified credential',
    },
    {
      from: 'did:mycelix:emma678',
      to: 'did:mycelix:frank901',
      from_name: 'Emma Wilson',
      to_name: 'Frank Brown',
      relationship: 'introduction',
      trust_score: 0.65,
      reason: 'Formal introduction',
    },
  ],
  final_trust: 0.48,
  path_length: 3,
  strongest_link: {
    from: 'did:mycelix:david345',
    to: 'did:mycelix:emma678',
    from_name: 'David Lee',
    to_name: 'Emma Wilson',
    relationship: 'credential_issuer',
    trust_score: 0.90,
    reason: 'Issued verified credential',
  },
  weakest_link: {
    from: 'did:mycelix:emma678',
    to: 'did:mycelix:frank901',
    from_name: 'Emma Wilson',
    to_name: 'Frank Brown',
    relationship: 'introduction',
    trust_score: 0.65,
    reason: 'Formal introduction',
  },
};

const weakLinkPath = {
  hops: [
    {
      from: 'did:mycelix:you123',
      to: 'did:mycelix:grace234',
      from_name: 'You',
      to_name: 'Grace Taylor',
      relationship: 'direct_trust',
      trust_score: 0.78,
      reason: 'Former colleague',
    },
    {
      from: 'did:mycelix:grace234',
      to: 'did:mycelix:henry567',
      from_name: 'Grace Taylor',
      to_name: 'Henry Davis',
      relationship: 'transitive_trust',
      trust_score: 0.35,
      reason: 'Weak network connection',
    },
  ],
  final_trust: 0.27,
  path_length: 2,
  strongest_link: {
    from: 'did:mycelix:you123',
    to: 'did:mycelix:grace234',
    from_name: 'You',
    to_name: 'Grace Taylor',
    relationship: 'direct_trust',
    trust_score: 0.78,
    reason: 'Former colleague',
  },
  weakest_link: {
    from: 'did:mycelix:grace234',
    to: 'did:mycelix:henry567',
    from_name: 'Grace Taylor',
    to_name: 'Henry Davis',
    relationship: 'transitive_trust',
    trust_score: 0.35,
    reason: 'Weak network connection',
  },
};

const noPath = {
  hops: [],
  final_trust: 0,
  path_length: 0,
};

/**
 * Direct trust path - The simplest case where you have directly
 * attested trust for the sender.
 */
export const DirectTrust: Story = {
  args: {
    path: directTrustPath,
    senderDid: 'did:mycelix:alice456',
    senderName: 'Alice Chen',
    recipientDid: 'did:mycelix:you123',
    recipientName: 'You',
    compact: false,
  },
};

/**
 * Two-hop path - Trust flows through one intermediary contact.
 */
export const TwoHopPath: Story = {
  args: {
    path: twoHopPath,
    senderDid: 'did:mycelix:carol012',
    senderName: 'Carol Johnson',
    recipientDid: 'did:mycelix:you123',
    recipientName: 'You',
    compact: false,
  },
};

/**
 * Three-hop path - A longer trust chain with multiple relationship types.
 */
export const ThreeHopPath: Story = {
  args: {
    path: threeHopPath,
    senderDid: 'did:mycelix:frank901',
    senderName: 'Frank Brown',
    recipientDid: 'did:mycelix:you123',
    recipientName: 'You',
    compact: false,
  },
};

/**
 * Weak link warning - Shows the warning when a trust path contains
 * a connection with low trust score.
 */
export const WeakLinkWarning: Story = {
  args: {
    path: weakLinkPath,
    senderDid: 'did:mycelix:henry567',
    senderName: 'Henry Davis',
    recipientDid: 'did:mycelix:you123',
    recipientName: 'You',
    compact: false,
  },
};

/**
 * No trust path - When there's no connection to the sender.
 */
export const NoPath: Story = {
  args: {
    path: noPath,
    senderDid: 'did:mycelix:unknown999',
    senderName: 'Unknown Sender',
    recipientDid: 'did:mycelix:you123',
    recipientName: 'You',
    compact: false,
  },
};

/**
 * Compact mode - Smaller visualization for email preview panels.
 */
export const CompactMode: Story = {
  args: {
    path: twoHopPath,
    senderDid: 'did:mycelix:carol012',
    senderName: 'Carol Johnson',
    recipientDid: 'did:mycelix:you123',
    recipientName: 'You',
    compact: true,
  },
};

/**
 * Inline version - A compact single-line version for email lists.
 */
export const InlineVersion: Story = {
  render: () => (
    <div className="space-y-4">
      <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded">
        <div className="text-sm font-medium mb-2">Direct trust:</div>
        <TrustPathInline path={directTrustPath} />
      </div>
      <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded">
        <div className="text-sm font-medium mb-2">Two hops:</div>
        <TrustPathInline path={twoHopPath} />
      </div>
      <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded">
        <div className="text-sm font-medium mb-2">Three hops:</div>
        <TrustPathInline path={threeHopPath} />
      </div>
      <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded">
        <div className="text-sm font-medium mb-2">No path:</div>
        <TrustPathInline path={noPath} />
      </div>
    </div>
  ),
};

/**
 * Relationship type showcase - Demonstrates all relationship type colors.
 */
export const RelationshipTypes: Story = {
  render: () => {
    const relationships = [
      { type: 'direct_trust', label: 'Direct Trust', color: '#10B981' },
      { type: 'credential_issuer', label: 'Credential Issuer', color: '#6366F1' },
      { type: 'organization_member', label: 'Organization Member', color: '#8B5CF6' },
      { type: 'vouch', label: 'Vouch', color: '#F59E0B' },
      { type: 'introduction', label: 'Introduction', color: '#EC4899' },
      { type: 'transitive_trust', label: 'Transitive Trust', color: '#6B7280' },
    ];

    return (
      <div className="p-4 bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
        <h3 className="text-sm font-semibold mb-4 text-gray-900 dark:text-gray-100">
          Trust Relationship Types
        </h3>
        <div className="grid grid-cols-2 gap-3">
          {relationships.map(({ type, label, color }) => (
            <div key={type} className="flex items-center gap-2">
              <div
                className="w-4 h-4 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">{label}</span>
            </div>
          ))}
        </div>
      </div>
    );
  },
};

/**
 * In email context - How the graph appears in an email detail view.
 */
export const InEmailContext: Story = {
  parameters: {
    layout: 'padded',
  },
  render: () => (
    <div className="max-w-2xl">
      <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700">
        {/* Email header */}
        <div className="flex items-start gap-3 pb-4 border-b border-gray-200 dark:border-gray-700">
          <div className="w-12 h-12 rounded-full bg-purple-500 flex items-center justify-center text-white font-semibold">
            CJ
          </div>
          <div className="flex-1">
            <div className="font-medium text-gray-900 dark:text-gray-100">Carol Johnson</div>
            <div className="text-sm text-gray-500">carol.johnson@example.com</div>
          </div>
          <div className="text-sm text-gray-400">Today at 3:45 PM</div>
        </div>

        {/* Trust graph section */}
        <div className="py-4">
          <TrustGraphVisualizer
            path={twoHopPath}
            senderDid="did:mycelix:carol012"
            senderName="Carol Johnson"
            recipientDid="did:mycelix:you123"
            recipientName="You"
          />
        </div>

        {/* Email content */}
        <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Project Collaboration Proposal
          </h2>
          <p className="text-gray-600 dark:text-gray-300">
            Hi there! Bob mentioned that you might be interested in collaborating on
            the new trust network visualization project. I'd love to discuss some ideas...
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
    <div className="dark">
      <TrustGraphVisualizer
        path={threeHopPath}
        senderDid="did:mycelix:frank901"
        senderName="Frank Brown"
        recipientDid="did:mycelix:you123"
        recipientName="You"
      />
    </div>
  ),
};

/**
 * All path lengths comparison.
 */
export const PathLengthComparison: Story = {
  render: () => (
    <div className="space-y-6">
      <div>
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          1 Hop (Direct Trust)
        </h4>
        <TrustGraphVisualizer
          path={directTrustPath}
          senderDid="did:mycelix:alice456"
          senderName="Alice Chen"
          recipientDid="did:mycelix:you123"
          recipientName="You"
          compact
        />
      </div>
      <div>
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          2 Hops
        </h4>
        <TrustGraphVisualizer
          path={twoHopPath}
          senderDid="did:mycelix:carol012"
          senderName="Carol Johnson"
          recipientDid="did:mycelix:you123"
          recipientName="You"
          compact
        />
      </div>
      <div>
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          3 Hops
        </h4>
        <TrustGraphVisualizer
          path={threeHopPath}
          senderDid="did:mycelix:frank901"
          senderName="Frank Brown"
          recipientDid="did:mycelix:you123"
          recipientName="You"
          compact
        />
      </div>
    </div>
  ),
};
