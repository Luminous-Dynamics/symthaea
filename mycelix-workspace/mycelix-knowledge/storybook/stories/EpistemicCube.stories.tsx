// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Meta, StoryObj } from '@storybook/react';
import { EpistemicCube, CubePoint } from '../../client/src/visualizations';

const meta: Meta<typeof EpistemicCube> = {
  title: 'Visualizations/EpistemicCube',
  component: EpistemicCube,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: `
# EpistemicCube

3D visualization of the Empirical-Normative-Mythic classification space.

Claims are plotted as points in a 3D cube where:
- **X-axis (Red)**: Empirical - verifiable through observation
- **Y-axis (Green)**: Normative - value-based, ethical
- **Z-axis (Blue)**: Mythic - meaning-making, narrative

**Interactions:**
- Drag to rotate the cube
- Hover over points to see details
- Click points to select
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    autoRotate: {
      control: 'boolean',
      description: 'Automatically rotate the cube',
    },
    showLabels: {
      control: 'boolean',
      description: 'Show labels on all points',
    },
    showGrid: {
      control: 'boolean',
      description: 'Display grid lines on the cube',
    },
    quadrantFilter: {
      control: 'select',
      options: [null, 'empirical', 'normative', 'mythic', 'balanced'],
      description: 'Filter to show only claims in specific quadrant',
    },
  },
};

export default meta;
type Story = StoryObj<typeof EpistemicCube>;

// Sample claims across the epistemic spectrum
const sampleClaims: CubePoint[] = [
  // Highly empirical
  {
    id: '1',
    label: 'Water boils at 100°C at sea level',
    position: { empirical: 0.95, normative: 0.02, mythic: 0.03 },
    credibility: 0.99,
  },
  {
    id: '2',
    label: 'The Earth orbits the Sun',
    position: { empirical: 0.98, normative: 0.01, mythic: 0.01 },
    credibility: 0.99,
  },
  {
    id: '3',
    label: 'Global temperature has risen 1.1°C since 1880',
    position: { empirical: 0.92, normative: 0.05, mythic: 0.03 },
    credibility: 0.95,
  },
  // Highly normative
  {
    id: '4',
    label: 'Universal healthcare is a human right',
    position: { empirical: 0.1, normative: 0.85, mythic: 0.05 },
    credibility: 0.7,
  },
  {
    id: '5',
    label: 'We should reduce carbon emissions',
    position: { empirical: 0.3, normative: 0.65, mythic: 0.05 },
    credibility: 0.8,
  },
  {
    id: '6',
    label: 'Capital punishment is morally wrong',
    position: { empirical: 0.05, normative: 0.9, mythic: 0.05 },
    credibility: 0.6,
  },
  // Highly mythic
  {
    id: '7',
    label: 'Life has inherent meaning and purpose',
    position: { empirical: 0.05, normative: 0.15, mythic: 0.8 },
    credibility: 0.5,
  },
  {
    id: '8',
    label: 'The universe is fundamentally conscious',
    position: { empirical: 0.1, normative: 0.05, mythic: 0.85 },
    credibility: 0.3,
  },
  {
    id: '9',
    label: 'Ancient wisdom holds timeless truths',
    position: { empirical: 0.15, normative: 0.1, mythic: 0.75 },
    credibility: 0.45,
  },
  // Balanced/mixed
  {
    id: '10',
    label: 'Democracy is the best form of government',
    position: { empirical: 0.35, normative: 0.45, mythic: 0.2 },
    credibility: 0.65,
  },
  {
    id: '11',
    label: 'Education improves society',
    position: { empirical: 0.4, normative: 0.4, mythic: 0.2 },
    credibility: 0.8,
  },
  {
    id: '12',
    label: 'Art enriches human experience',
    position: { empirical: 0.2, normative: 0.3, mythic: 0.5 },
    credibility: 0.75,
  },
];

export const Default: Story = {
  args: {
    claims: sampleClaims,
    width: 700,
    height: 500,
    showAxisLabels: true,
    showLabels: false,
    showGrid: true,
    autoRotate: false,
  },
};

export const AutoRotating: Story = {
  args: {
    ...Default.args,
    autoRotate: true,
    rotationSpeed: 0.005,
  },
  parameters: {
    docs: {
      description: {
        story: 'The cube automatically rotates to show all angles',
      },
    },
  },
};

export const WithLabels: Story = {
  args: {
    ...Default.args,
    showLabels: true,
  },
};

export const EmpiricalFilter: Story = {
  args: {
    ...Default.args,
    quadrantFilter: 'empirical',
  },
  parameters: {
    docs: {
      description: {
        story: 'Showing only claims with dominant empirical classification',
      },
    },
  },
};

export const NormativeFilter: Story = {
  args: {
    ...Default.args,
    quadrantFilter: 'normative',
  },
  parameters: {
    docs: {
      description: {
        story: 'Showing only claims with dominant normative classification',
      },
    },
  },
};

export const MythicFilter: Story = {
  args: {
    ...Default.args,
    quadrantFilter: 'mythic',
  },
  parameters: {
    docs: {
      description: {
        story: 'Showing only claims with dominant mythic classification',
      },
    },
  },
};

export const BalancedFilter: Story = {
  args: {
    ...Default.args,
    quadrantFilter: 'balanced',
  },
  parameters: {
    docs: {
      description: {
        story: 'Showing only claims with balanced E-N-M distribution',
      },
    },
  },
};

export const CustomColors: Story = {
  args: {
    ...Default.args,
    axisColors: {
      x: '#ff6b6b',  // Coral red
      y: '#4ecdc4',  // Teal
      z: '#ffe66d',  // Yellow
    },
    backgroundColor: '#2d3436',
  },
  parameters: {
    backgrounds: { default: 'dark' },
    docs: {
      description: {
        story: 'Custom color scheme for different branding',
      },
    },
  },
};

// Scientific claims dataset
const scientificClaims: CubePoint[] = [
  { id: 's1', label: 'E=mc²', position: { empirical: 0.95, normative: 0.02, mythic: 0.03 }, credibility: 0.99 },
  { id: 's2', label: 'DNA is the hereditary material', position: { empirical: 0.98, normative: 0.01, mythic: 0.01 }, credibility: 0.99 },
  { id: 's3', label: 'Evolution explains biodiversity', position: { empirical: 0.9, normative: 0.02, mythic: 0.08 }, credibility: 0.98 },
  { id: 's4', label: 'The universe began with the Big Bang', position: { empirical: 0.85, normative: 0.02, mythic: 0.13 }, credibility: 0.95 },
  { id: 's5', label: 'Quantum mechanics is probabilistic', position: { empirical: 0.88, normative: 0.02, mythic: 0.1 }, credibility: 0.97 },
  { id: 's6', label: 'Climate change is human-caused', position: { empirical: 0.85, normative: 0.1, mythic: 0.05 }, credibility: 0.93 },
];

export const ScientificClaims: Story = {
  args: {
    claims: scientificClaims,
    width: 700,
    height: 500,
    showAxisLabels: true,
    autoRotate: true,
    rotationSpeed: 0.003,
  },
  parameters: {
    docs: {
      description: {
        story: 'Scientific claims cluster in the high-empirical region',
      },
    },
  },
};

// Philosophical claims
const philosophicalClaims: CubePoint[] = [
  { id: 'p1', label: 'Free will is an illusion', position: { empirical: 0.4, normative: 0.2, mythic: 0.4 }, credibility: 0.5 },
  { id: 'p2', label: 'Consciousness is fundamental', position: { empirical: 0.3, normative: 0.1, mythic: 0.6 }, credibility: 0.4 },
  { id: 'p3', label: 'Morality is objective', position: { empirical: 0.15, normative: 0.7, mythic: 0.15 }, credibility: 0.5 },
  { id: 'p4', label: 'Beauty is subjective', position: { empirical: 0.2, normative: 0.5, mythic: 0.3 }, credibility: 0.6 },
  { id: 'p5', label: 'Time is an illusion', position: { empirical: 0.35, normative: 0.05, mythic: 0.6 }, credibility: 0.35 },
  { id: 'p6', label: 'Knowledge requires justification', position: { empirical: 0.5, normative: 0.3, mythic: 0.2 }, credibility: 0.7 },
];

export const PhilosophicalClaims: Story = {
  args: {
    claims: philosophicalClaims,
    width: 700,
    height: 500,
    showAxisLabels: true,
    showLabels: true,
  },
  parameters: {
    docs: {
      description: {
        story: 'Philosophical claims spread across normative and mythic dimensions',
      },
    },
  },
};
