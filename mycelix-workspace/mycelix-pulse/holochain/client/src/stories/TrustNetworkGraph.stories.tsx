// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * TrustNetworkGraph Component Stories
 */

import type { Meta, StoryObj } from '@storybook/react';
import { TrustNetworkGraph, TrustNode, TrustEdge } from '../visualization/TrustNetworkGraph';

const meta: Meta<typeof TrustNetworkGraph> = {
  title: 'Visualization/TrustNetworkGraph',
  component: TrustNetworkGraph,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'D3.js-powered interactive graph visualization of the trust network with MATL algorithm insights.',
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    width: {
      control: { type: 'range', min: 400, max: 1200, step: 50 },
    },
    height: {
      control: { type: 'range', min: 300, max: 800, step: 50 },
    },
    colorScheme: {
      control: { type: 'select' },
      options: ['trust', 'group', 'activity'],
    },
    showLabels: {
      control: 'boolean',
    },
    showLegend: {
      control: 'boolean',
    },
  },
};

export default meta;
type Story = StoryObj<typeof TrustNetworkGraph>;

// Sample data generators
const generateNodes = (count: number, includeMe: boolean = true): TrustNode[] => {
  const names = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'];
  const groups = ['family', 'work', 'friends', 'community'];

  const nodes: TrustNode[] = [];

  if (includeMe) {
    nodes.push({
      id: 'me',
      name: 'You',
      email: 'me@example.com',
      trustLevel: 1,
      attestationCount: 0,
      isMe: true,
    });
  }

  for (let i = 0; i < count; i++) {
    nodes.push({
      id: `node-${i}`,
      name: names[i % names.length],
      email: `${names[i % names.length].toLowerCase()}@example.com`,
      trustLevel: Math.random() * 0.8 + 0.1, // 0.1 to 0.9
      attestationCount: Math.floor(Math.random() * 15),
      group: groups[Math.floor(Math.random() * groups.length)],
    });
  }

  return nodes;
};

const generateEdges = (nodes: TrustNode[]): TrustEdge[] => {
  const edges: TrustEdge[] = [];
  const contexts = ['colleague', 'friend', 'verified', 'introduced by mutual'];

  nodes.forEach((node, i) => {
    if (node.isMe) return;

    // Connect to me
    edges.push({
      source: 'me',
      target: node.id,
      trustLevel: node.trustLevel,
      createdAt: Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000,
      context: contexts[Math.floor(Math.random() * contexts.length)],
    });

    // Connect to some other nodes
    const connectCount = Math.floor(Math.random() * 3);
    for (let j = 0; j < connectCount; j++) {
      const targetIndex = Math.floor(Math.random() * nodes.length);
      if (targetIndex !== i && !nodes[targetIndex].isMe) {
        edges.push({
          source: node.id,
          target: nodes[targetIndex].id,
          trustLevel: Math.random() * 0.6 + 0.2,
          createdAt: Date.now() - Math.random() * 60 * 24 * 60 * 60 * 1000,
        });
      }
    }
  });

  return edges;
};

const smallNodes = generateNodes(5);
const smallEdges = generateEdges(smallNodes);

const mediumNodes = generateNodes(15);
const mediumEdges = generateEdges(mediumNodes);

const largeNodes = generateNodes(30);
const largeEdges = generateEdges(largeNodes);

export const Default: Story = {
  args: {
    nodes: mediumNodes,
    edges: mediumEdges,
    width: 800,
    height: 600,
    showLabels: true,
    showLegend: true,
    colorScheme: 'trust',
  },
};

export const SmallNetwork: Story = {
  args: {
    nodes: smallNodes,
    edges: smallEdges,
    width: 600,
    height: 400,
    showLabels: true,
    showLegend: true,
    colorScheme: 'trust',
  },
  parameters: {
    docs: {
      description: {
        story: 'A small trust network with just a few connections.',
      },
    },
  },
};

export const LargeNetwork: Story = {
  args: {
    nodes: largeNodes,
    edges: largeEdges,
    width: 1000,
    height: 700,
    showLabels: true,
    showLegend: true,
    colorScheme: 'trust',
  },
  parameters: {
    docs: {
      description: {
        story: 'A larger network demonstrating performance with many nodes.',
      },
    },
  },
};

export const GroupColorScheme: Story = {
  args: {
    nodes: mediumNodes,
    edges: mediumEdges,
    width: 800,
    height: 600,
    showLabels: true,
    showLegend: true,
    colorScheme: 'group',
  },
  parameters: {
    docs: {
      description: {
        story: 'Nodes colored by their group (family, work, friends, community).',
      },
    },
  },
};

export const ActivityColorScheme: Story = {
  args: {
    nodes: mediumNodes,
    edges: mediumEdges,
    width: 800,
    height: 600,
    showLabels: true,
    showLegend: true,
    colorScheme: 'activity',
  },
  parameters: {
    docs: {
      description: {
        story: 'Nodes colored by activity level (attestation count).',
      },
    },
  },
};

export const WithoutLabels: Story = {
  args: {
    nodes: largeNodes,
    edges: largeEdges,
    width: 800,
    height: 600,
    showLabels: false,
    showLegend: true,
    colorScheme: 'trust',
  },
  parameters: {
    docs: {
      description: {
        story: 'Labels hidden for cleaner visualization of large networks.',
      },
    },
  },
};

export const SelectedNode: Story = {
  args: {
    nodes: mediumNodes,
    edges: mediumEdges,
    width: 800,
    height: 600,
    showLabels: true,
    showLegend: true,
    colorScheme: 'trust',
    selectedNodeId: 'node-3',
  },
  parameters: {
    docs: {
      description: {
        story: 'A node selected for detailed view with highlight ring.',
      },
    },
  },
};

export const Interactive: Story = {
  args: {
    nodes: mediumNodes,
    edges: mediumEdges,
    width: 800,
    height: 600,
    showLabels: true,
    showLegend: true,
    colorScheme: 'trust',
    onNodeClick: (node) => alert(`Clicked: ${node.name} (Trust: ${Math.round(node.trustLevel * 100)}%)`),
    onEdgeClick: (edge) => alert(`Connection: ${edge.source} → ${edge.target}`),
  },
  parameters: {
    docs: {
      description: {
        story: 'Interactive graph with click handlers for nodes and edges.',
      },
    },
  },
};

export const EmptyNetwork: Story = {
  args: {
    nodes: [
      {
        id: 'me',
        name: 'You',
        email: 'me@example.com',
        trustLevel: 1,
        attestationCount: 0,
        isMe: true,
      },
    ],
    edges: [],
    width: 600,
    height: 400,
    showLabels: true,
    showLegend: true,
    colorScheme: 'trust',
  },
  parameters: {
    docs: {
      description: {
        story: 'Empty trust network with only the user node.',
      },
    },
  },
};

export const HighlyConnected: Story = {
  render: () => {
    const hub = {
      id: 'hub',
      name: 'Community Leader',
      email: 'leader@example.com',
      trustLevel: 0.95,
      attestationCount: 50,
      group: 'community',
    };

    const nodes: TrustNode[] = [
      { id: 'me', name: 'You', trustLevel: 1, attestationCount: 0, isMe: true },
      hub,
      ...Array.from({ length: 12 }, (_, i) => ({
        id: `member-${i}`,
        name: `Member ${i + 1}`,
        trustLevel: 0.5 + Math.random() * 0.4,
        attestationCount: Math.floor(Math.random() * 10),
        group: 'community',
      })),
    ];

    const edges: TrustEdge[] = [
      { source: 'me', target: 'hub', trustLevel: 0.9, createdAt: Date.now() },
      ...nodes.slice(2).map((node) => ({
        source: 'hub',
        target: node.id,
        trustLevel: 0.7 + Math.random() * 0.2,
        createdAt: Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000,
      })),
    ];

    return (
      <TrustNetworkGraph
        nodes={nodes}
        edges={edges}
        width={800}
        height={600}
        showLabels={true}
        showLegend={true}
        colorScheme="trust"
      />
    );
  },
  parameters: {
    docs: {
      description: {
        story: 'A hub-and-spoke pattern showing a highly connected community leader.',
      },
    },
  },
};
