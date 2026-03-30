// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Meta, StoryObj } from '@storybook/react';
import { ClaimCard, ClaimData, CredibilityData, RelationshipData } from '../../client/src/visualizations';

const meta: Meta<typeof ClaimCard> = {
  title: 'Components/ClaimCard',
  component: ClaimCard,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: `
# ClaimCard

Comprehensive card component for displaying claim information.
Shows epistemic classification, credibility, relationships, and actions.

**Variants:**
- **default**: Standard display with all information
- **compact**: Condensed view for lists
- **detailed**: Expanded view with all factors
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: 'select',
      options: ['default', 'compact', 'detailed'],
    },
    showEpistemicBar: {
      control: 'boolean',
    },
    showCredibilityFactors: {
      control: 'boolean',
    },
    showRelationships: {
      control: 'boolean',
    },
    showTags: {
      control: 'boolean',
    },
    selected: {
      control: 'boolean',
    },
  },
};

export default meta;
type Story = StoryObj<typeof ClaimCard>;

const sampleClaim: ClaimData = {
  id: 'claim-123',
  content: 'Global temperatures have increased by approximately 1.1°C since the pre-industrial era, with most of the warming occurring in the past 50 years.',
  classification: {
    empirical: 0.85,
    normative: 0.1,
    mythic: 0.05,
  },
  author: 'did:example:scientist123',
  createdAt: Date.now() - 86400000 * 30,
  tags: ['climate', 'science', 'temperature', 'global warming', 'empirical'],
  sourceCount: 12,
};

const sampleCredibility: CredibilityData = {
  score: 0.89,
  factors: {
    sourceDiversity: 0.92,
    authorReputation: 0.87,
    temporalConsistency: 0.85,
    crossValidation: 0.91,
  },
  verdict: 'True',
  confidence: 0.92,
};

const sampleRelationships: RelationshipData = {
  supports: 8,
  contradicts: 1,
  refines: 3,
  dependsOn: 2,
};

export const Default: Story = {
  args: {
    claim: sampleClaim,
    credibility: sampleCredibility,
    relationships: sampleRelationships,
    variant: 'default',
    showEpistemicBar: true,
    showCredibilityFactors: false,
    showRelationships: true,
    showTags: true,
    onViewDetails: () => console.log('View details clicked'),
    onVerify: () => console.log('Verify clicked'),
  },
};

export const Compact: Story = {
  args: {
    ...Default.args,
    variant: 'compact',
  },
  parameters: {
    docs: {
      description: {
        story: 'Condensed view suitable for list displays',
      },
    },
  },
};

export const Detailed: Story = {
  args: {
    ...Default.args,
    variant: 'detailed',
    showCredibilityFactors: true,
  },
  parameters: {
    docs: {
      description: {
        story: 'Full detail view with credibility factor breakdown',
      },
    },
  },
};

export const Selected: Story = {
  args: {
    ...Default.args,
    selected: true,
  },
};

export const HighCredibility: Story = {
  args: {
    claim: {
      ...sampleClaim,
      content: 'Water molecules consist of two hydrogen atoms and one oxygen atom (H₂O).',
    },
    credibility: {
      score: 0.99,
      verdict: 'True',
      confidence: 0.99,
    },
    variant: 'default',
    showEpistemicBar: true,
  },
};

export const LowCredibility: Story = {
  args: {
    claim: {
      ...sampleClaim,
      content: '5G towers cause COVID-19 transmission and are part of a global conspiracy.',
      classification: { empirical: 0.1, normative: 0.2, mythic: 0.7 },
      tags: ['conspiracy', 'misinformation', 'debunked'],
    },
    credibility: {
      score: 0.05,
      verdict: 'False',
      confidence: 0.95,
    },
    relationships: {
      supports: 0,
      contradicts: 45,
      refines: 0,
      dependsOn: 0,
    },
    variant: 'default',
    showEpistemicBar: true,
    showRelationships: true,
  },
};

export const NormativeClaim: Story = {
  args: {
    claim: {
      id: 'claim-456',
      content: 'Universal basic income would reduce poverty and increase economic mobility while preserving human dignity.',
      classification: { empirical: 0.25, normative: 0.7, mythic: 0.05 },
      author: 'did:example:economist789',
      createdAt: Date.now() - 86400000 * 7,
      tags: ['economics', 'policy', 'ethics', 'UBI'],
      sourceCount: 5,
    },
    credibility: {
      score: 0.65,
      verdict: 'Mixed',
      confidence: 0.7,
    },
    variant: 'default',
    showEpistemicBar: true,
  },
};

export const MythicClaim: Story = {
  args: {
    claim: {
      id: 'claim-789',
      content: 'The universe is fundamentally conscious and our individual awareness is a localized expression of cosmic consciousness.',
      classification: { empirical: 0.1, normative: 0.1, mythic: 0.8 },
      author: 'did:example:philosopher321',
      createdAt: Date.now() - 86400000 * 60,
      tags: ['philosophy', 'consciousness', 'metaphysics', 'spirituality'],
      sourceCount: 2,
    },
    credibility: {
      score: 0.35,
      verdict: 'Unverifiable',
      confidence: 0.6,
    },
    variant: 'default',
    showEpistemicBar: true,
  },
};

export const CardList: Story = {
  render: () => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', maxWidth: '600px' }}>
      <ClaimCard
        claim={sampleClaim}
        credibility={sampleCredibility}
        variant="compact"
        showEpistemicBar
      />
      <ClaimCard
        claim={{
          ...sampleClaim,
          id: '2',
          content: 'Sea levels are rising at an accelerating rate due to thermal expansion and ice melt.',
        }}
        credibility={{ score: 0.87, verdict: 'True', confidence: 0.9 }}
        variant="compact"
        showEpistemicBar
      />
      <ClaimCard
        claim={{
          ...sampleClaim,
          id: '3',
          content: 'Renewable energy is now cheaper than fossil fuels in most markets.',
        }}
        credibility={{ score: 0.82, verdict: 'MostlyTrue', confidence: 0.85 }}
        variant="compact"
        showEpistemicBar
      />
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Multiple compact cards in a list layout',
      },
    },
  },
};
