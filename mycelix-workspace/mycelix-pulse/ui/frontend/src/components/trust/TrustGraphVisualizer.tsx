// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useMemo, useState } from 'react';

// Types matching backend
interface TrustHop {
  from: string;
  to: string;
  from_name?: string;
  to_name?: string;
  relationship: string;
  trust_score: number;
  reason?: string;
}

interface TrustPath {
  hops: TrustHop[];
  final_trust: number;
  path_length: number;
  strongest_link?: TrustHop;
  weakest_link?: TrustHop;
}

interface TrustGraphVisualizerProps {
  path: TrustPath;
  senderDid: string;
  senderName?: string;
  recipientDid: string;
  recipientName?: string;
  compact?: boolean;
  className?: string;
}

// Relationship type colors
const relationshipColors: Record<string, string> = {
  direct_trust: '#10B981',      // Emerald
  credential_issuer: '#6366F1', // Indigo
  organization_member: '#8B5CF6', // Purple
  vouch: '#F59E0B',             // Amber
  introduction: '#EC4899',       // Pink
  transitive_trust: '#6B7280',   // Gray
};

// Get display name for relationship
const getRelationshipLabel = (rel: string): string => {
  const labels: Record<string, string> = {
    direct_trust: 'trusts directly',
    credential_issuer: 'issued credential to',
    organization_member: 'organization member with',
    vouch: 'vouches for',
    introduction: 'introduced',
    transitive_trust: 'connected to',
  };
  return labels[rel] || rel.replace(/_/g, ' ');
};

// Trust level badge
function TrustLevelBadge({ score }: { score: number }) {
  const level = score >= 0.7 ? 'high' : score >= 0.4 ? 'medium' : 'low';
  const colors = {
    high: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-200',
    medium: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-200',
    low: 'bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-200',
  };

  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colors[level]}`}>
      {(score * 100).toFixed(0)}% trust
    </span>
  );
}

// Node in the graph
function GraphNode({
  x,
  y,
  did,
  name,
  isYou,
  isSender,
}: {
  x: number;
  y: number;
  did: string;
  name?: string;
  isYou?: boolean;
  isSender?: boolean;
}) {
  const displayName = name || did.slice(-8);
  const bgColor = isYou
    ? '#3B82F6'
    : isSender
    ? '#10B981'
    : '#6B7280';

  return (
    <g transform={`translate(${x}, ${y})`}>
      {/* Node circle */}
      <circle
        r={24}
        fill={bgColor}
        className="drop-shadow-md"
      />
      {/* Initials or icon */}
      <text
        textAnchor="middle"
        dominantBaseline="central"
        fill="white"
        fontSize="12"
        fontWeight="600"
      >
        {isYou ? 'You' : displayName.slice(0, 2).toUpperCase()}
      </text>
      {/* Name label below */}
      <text
        y={36}
        textAnchor="middle"
        fill="currentColor"
        fontSize="11"
        className="text-gray-700 dark:text-gray-300"
      >
        {isYou ? 'You' : displayName}
      </text>
    </g>
  );
}

// Edge between nodes
function GraphEdge({
  x1,
  y1,
  x2,
  y2,
  relationship,
  trustScore,
  animated,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  relationship: string;
  trustScore: number;
  animated?: boolean;
}) {
  const color = relationshipColors[relationship] || '#6B7280';
  const strokeWidth = Math.max(2, trustScore * 6);

  // Calculate arrow position
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const nodeRadius = 24;
  const arrowX = x2 - Math.cos(angle) * nodeRadius;
  const arrowY = y2 - Math.sin(angle) * nodeRadius;

  return (
    <g>
      {/* Line */}
      <line
        x1={x1}
        y1={y1}
        x2={arrowX}
        y2={arrowY}
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        opacity={0.7}
        className={animated ? 'animate-pulse' : ''}
      />
      {/* Arrow head */}
      <polygon
        points={`0,-4 8,0 0,4`}
        fill={color}
        transform={`translate(${arrowX}, ${arrowY}) rotate(${(angle * 180) / Math.PI})`}
      />
      {/* Trust score label */}
      <text
        x={(x1 + x2) / 2}
        y={(y1 + y2) / 2 - 8}
        textAnchor="middle"
        fontSize="10"
        fill={color}
        fontWeight="500"
      >
        {(trustScore * 100).toFixed(0)}%
      </text>
    </g>
  );
}

export default function TrustGraphVisualizer({
  path,
  senderDid,
  senderName,
  recipientDid,
  recipientName,
  compact = false,
  className = '',
}: TrustGraphVisualizerProps) {
  const [hoveredHop, setHoveredHop] = useState<number | null>(null);

  // Calculate node positions
  const { nodes, edges, width, height } = useMemo(() => {
    const nodeCount = path.hops.length + 1;
    const spacing = compact ? 80 : 120;
    const w = nodeCount * spacing + 100;
    const h = compact ? 100 : 140;
    const startX = 50;
    const y = h / 2;

    const nodeList: Array<{
      did: string;
      name?: string;
      x: number;
      y: number;
      isYou: boolean;
      isSender: boolean;
    }> = [];

    const edgeList: Array<{
      from: number;
      to: number;
      relationship: string;
      trustScore: number;
    }> = [];

    // Add recipient (you) as first node
    nodeList.push({
      did: recipientDid,
      name: recipientName,
      x: startX,
      y,
      isYou: true,
      isSender: false,
    });

    // Add intermediate nodes
    path.hops.forEach((hop, i) => {
      const x = startX + (i + 1) * spacing;
      nodeList.push({
        did: hop.to,
        name: hop.to_name,
        x,
        y,
        isYou: false,
        isSender: i === path.hops.length - 1,
      });

      edgeList.push({
        from: i,
        to: i + 1,
        relationship: hop.relationship,
        trustScore: hop.trust_score,
      });
    });

    return { nodes: nodeList, edges: edgeList, width: w, height: h };
  }, [path, recipientDid, recipientName, compact]);

  if (path.hops.length === 0) {
    return (
      <div className={`p-4 bg-gray-50 dark:bg-gray-800 rounded-lg ${className}`}>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          No trust path found to this sender.
        </p>
      </div>
    );
  }

  return (
    <div className={`bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Trust Path
          </h3>
          <TrustLevelBadge score={path.final_trust} />
        </div>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          {path.path_length} hop{path.path_length !== 1 ? 's' : ''} from you to{' '}
          {senderName || senderDid.slice(-8)}
        </p>
      </div>

      {/* Graph visualization */}
      <div className="p-4 overflow-x-auto">
        <svg
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          className="mx-auto"
        >
          {/* Edges first (behind nodes) */}
          {edges.map((edge, i) => (
            <GraphEdge
              key={i}
              x1={nodes[edge.from].x}
              y1={nodes[edge.from].y}
              x2={nodes[edge.to].x}
              y2={nodes[edge.to].y}
              relationship={edge.relationship}
              trustScore={edge.trustScore}
              animated={hoveredHop === i}
            />
          ))}

          {/* Nodes */}
          {nodes.map((node, i) => (
            <GraphNode
              key={node.did}
              x={node.x}
              y={node.y}
              did={node.did}
              name={node.name}
              isYou={node.isYou}
              isSender={node.isSender}
            />
          ))}
        </svg>
      </div>

      {/* Path explanation */}
      {!compact && (
        <div className="px-4 pb-4">
          <h4 className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
            Trust Path Explanation
          </h4>
          <ul className="space-y-1">
            {path.hops.map((hop, i) => (
              <li
                key={i}
                className="flex items-center text-xs text-gray-600 dark:text-gray-400"
                onMouseEnter={() => setHoveredHop(i)}
                onMouseLeave={() => setHoveredHop(null)}
              >
                <span
                  className="w-2 h-2 rounded-full mr-2"
                  style={{ backgroundColor: relationshipColors[hop.relationship] || '#6B7280' }}
                />
                <span className="font-medium">{hop.from_name || hop.from.slice(-8)}</span>
                <span className="mx-1">{getRelationshipLabel(hop.relationship)}</span>
                <span className="font-medium">{hop.to_name || hop.to.slice(-8)}</span>
                {hop.reason && (
                  <span className="ml-1 text-gray-400">({hop.reason})</span>
                )}
              </li>
            ))}
          </ul>

          {/* Weakest link warning */}
          {path.weakest_link && path.weakest_link.trust_score < 0.5 && (
            <div className="mt-3 p-2 bg-amber-50 dark:bg-amber-900/20 rounded border border-amber-200 dark:border-amber-800">
              <p className="text-xs text-amber-800 dark:text-amber-200">
                <span className="font-medium">Weak link:</span>{' '}
                {path.weakest_link.from_name || path.weakest_link.from.slice(-8)} →{' '}
                {path.weakest_link.to_name || path.weakest_link.to.slice(-8)} has only{' '}
                {(path.weakest_link.trust_score * 100).toFixed(0)}% trust
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Compact inline version
export function TrustPathInline({ path, className = '' }: { path: TrustPath; className?: string }) {
  if (path.hops.length === 0) {
    return (
      <span className={`text-xs text-gray-500 ${className}`}>
        No trust path
      </span>
    );
  }

  return (
    <span className={`inline-flex items-center text-xs ${className}`}>
      <span className="text-blue-600 dark:text-blue-400 font-medium">You</span>
      {path.hops.map((hop, i) => (
        <span key={i} className="flex items-center">
          <svg className="w-4 h-4 mx-1 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          <span
            className="px-1.5 py-0.5 rounded text-gray-700 dark:text-gray-300"
            style={{ backgroundColor: `${relationshipColors[hop.relationship]}20` }}
          >
            {hop.to_name || hop.to.slice(-8)}
          </span>
        </span>
      ))}
      <span className="ml-2 text-gray-500">
        ({(path.final_trust * 100).toFixed(0)}% trust)
      </span>
    </span>
  );
}
