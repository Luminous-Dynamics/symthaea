// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Meta, StoryObj } from '@storybook/react';
import { BeliefGraph, GraphNode, GraphEdge } from '../../client/src/visualizations';

const meta: Meta<typeof BeliefGraph> = {
  title: 'Visualizations/BeliefGraph',
  component: BeliefGraph,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: `
# BeliefGraph

Interactive force-directed graph visualization for knowledge networks.
Displays claims and their relationships with support for:

- **Zoom & Pan**: Mouse wheel and drag to navigate
- **Node Interaction**: Click and drag nodes
- **Color Schemes**: Credibility, epistemic position, or type-based coloring
- **Real-time Physics**: Force simulation for automatic layout
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    colorScheme: {
      control: 'select',
      options: ['credibility', 'epistemic', 'type', 'custom'],
      description: 'How nodes are colored',
    },
    showCredibility: {
      control: 'boolean',
      description: 'Show credibility rings around nodes',
    },
    showEdgeLabels: {
      control: 'boolean',
      description: 'Display labels on relationship edges',
    },
    enablePhysics: {
      control: 'boolean',
      description: 'Enable force-directed physics simulation',
    },
    width: {
      control: { type: 'range', min: 400, max: 1200, step: 50 },
    },
    height: {
      control: { type: 'range', min: 300, max: 800, step: 50 },
    },
  },
};

export default meta;
type Story = StoryObj<typeof BeliefGraph>;

// Sample data
const sampleNodes: GraphNode[] = [
  {
    id: '1',
    label: 'Climate change is accelerating',
    type: 'claim',
    credibility: 0.89,
    epistemicPosition: { empirical: 0.9, normative: 0.05, mythic: 0.05 },
  },
  {
    id: '2',
    label: 'Global temperatures have risen 1.1°C',
    type: 'claim',
    credibility: 0.95,
    epistemicPosition: { empirical: 0.95, normative: 0.02, mythic: 0.03 },
  },
  {
    id: '3',
    label: 'Arctic ice is melting rapidly',
    type: 'claim',
    credibility: 0.92,
    epistemicPosition: { empirical: 0.88, normative: 0.05, mythic: 0.07 },
  },
  {
    id: '4',
    label: 'We must act now to prevent disaster',
    type: 'claim',
    credibility: 0.65,
    epistemicPosition: { empirical: 0.2, normative: 0.7, mythic: 0.1 },
  },
  {
    id: '5',
    label: 'Dr. Jane Smith',
    type: 'author',
    credibility: 0.88,
  },
  {
    id: '6',
    label: 'Nature Journal',
    type: 'source',
    credibility: 0.94,
  },
];

const sampleEdges: GraphEdge[] = [
  { source: '2', target: '1', type: 'supports', weight: 0.9 },
  { source: '3', target: '1', type: 'supports', weight: 0.85 },
  { source: '1', target: '4', type: 'supports', weight: 0.6 },
  { source: '5', target: '1', type: 'authored', weight: 1.0 },
  { source: '6', target: '2', type: 'cites', weight: 0.95 },
];

export const Default: Story = {
  args: {
    nodes: sampleNodes,
    edges: sampleEdges,
    width: 800,
    height: 600,
    showCredibility: true,
    showEdgeLabels: false,
    colorScheme: 'credibility',
    enablePhysics: true,
  },
};

export const EpistemicColoring: Story = {
  args: {
    ...Default.args,
    colorScheme: 'epistemic',
  },
  parameters: {
    docs: {
      description: {
        story: 'Nodes colored by their E-N-M epistemic position (red=empirical, green=normative, blue=mythic)',
      },
    },
  },
};

export const TypeColoring: Story = {
  args: {
    ...Default.args,
    colorScheme: 'type',
  },
  parameters: {
    docs: {
      description: {
        story: 'Nodes colored by type: claims (blue), authors (purple), sources (amber), markets (green)',
      },
    },
  },
};

export const WithEdgeLabels: Story = {
  args: {
    ...Default.args,
    showEdgeLabels: true,
  },
};

export const StaticLayout: Story = {
  args: {
    ...Default.args,
    enablePhysics: false,
  },
  parameters: {
    docs: {
      description: {
        story: 'Physics simulation disabled - nodes stay in place unless dragged',
      },
    },
  },
};

export const Loading: Story = {
  args: {
    nodes: [],
    edges: [],
    loading: true,
    width: 800,
    height: 400,
  },
};

export const Error: Story = {
  args: {
    nodes: [],
    edges: [],
    error: new Error('Failed to load graph data'),
    width: 800,
    height: 400,
  },
};

// Contradiction example
const contradictionNodes: GraphNode[] = [
  {
    id: 'c1',
    label: 'Vaccines are safe and effective',
    type: 'claim',
    credibility: 0.92,
    epistemicPosition: { empirical: 0.85, normative: 0.1, mythic: 0.05 },
  },
  {
    id: 'c2',
    label: 'Vaccines cause autism',
    type: 'claim',
    credibility: 0.08,
    epistemicPosition: { empirical: 0.3, normative: 0.2, mythic: 0.5 },
  },
  {
    id: 'c3',
    label: 'CDC study on vaccine safety',
    type: 'source',
    credibility: 0.95,
  },
  {
    id: 'c4',
    label: 'Retracted Wakefield paper',
    type: 'source',
    credibility: 0.05,
  },
];

const contradictionEdges: GraphEdge[] = [
  { source: 'c3', target: 'c1', type: 'supports', weight: 0.95 },
  { source: 'c4', target: 'c2', type: 'supports', weight: 0.3 },
  { source: 'c1', target: 'c2', type: 'contradicts', weight: 0.95 },
];

export const ContradictingClaims: Story = {
  args: {
    nodes: contradictionNodes,
    edges: contradictionEdges,
    width: 800,
    height: 500,
    showCredibility: true,
    colorScheme: 'credibility',
  },
  parameters: {
    docs: {
      description: {
        story: 'Visualizing contradictory claims with supporting sources. Red edges indicate contradiction.',
      },
    },
  },
};
