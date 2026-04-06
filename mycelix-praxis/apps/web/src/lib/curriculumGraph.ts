// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Curriculum graph data loader and transformer for Cytoscape.js

export interface CurriculumNode {
  id: string;
  title: string;
  description: string;
  node_type: string;
  difficulty: string;
  domain: string;
  subdomain: string;
  tags: string[];
  estimated_hours: number;
  grade_levels: string[];
  bloom_level: string;
  subject_area: string;
}

export interface CurriculumEdge {
  from: string;
  to: string;
  edge_type: string;
  strength_permille: number;
  rationale: string;
}

export interface CurriculumDocument {
  metadata: {
    title: string;
    total_standards: number;
  };
  nodes: CurriculumNode[];
  edges: CurriculumEdge[];
}

// ISCED-F Broad Field classification (mirrors taxonomy.rs)
export function classifyField(subjectArea: string): string {
  const s = subjectArea.toLowerCase();
  if (s.includes('computer') || s.includes('software') || s.includes('cs2013') ||
      s.includes('algorithm') || s.includes('programming') || s.includes('cyber') ||
      s.includes('information')) return 'ICT';
  if (s.includes('math') || s.includes('physics') || s.includes('chemistry') ||
      s.includes('biology') || s.includes('science') || s.includes('statistics')) return 'NatSci';
  if (s.includes('english') || s.includes('literature') || s.includes('philosophy') ||
      s.includes('history') || s.includes('art') || s.includes('language') ||
      s.includes('linguistics') || s.includes('music')) return 'ArtsHum';
  if (s.includes('politic') || s.includes('sociology') || s.includes('econom') ||
      s.includes('psychology') || s.includes('social')) return 'SocSci';
  if (s.includes('engineer')) return 'Engineering';
  if (s.includes('education') || s.includes('teaching')) return 'Education';
  if (s.includes('health') || s.includes('medic') || s.includes('nurs')) return 'Health';
  if (s.includes('business') || s.includes('finance') || s.includes('law')) return 'Business';
  if (s.includes('physical education') || s.includes('sport')) return 'Services';
  if (s.includes('symthaea') || s.includes('mycelix') || s.includes('consciousness') ||
      s.includes('decentralized')) return 'MetaLearning';
  return 'NatSci';
}

// Color palette for ISCED-F fields
export const FIELD_COLORS: Record<string, string> = {
  'NatSci': '#2196F3',       // Blue
  'ArtsHum': '#9C27B0',      // Purple
  'SocSci': '#FF9800',       // Orange
  'ICT': '#4CAF50',          // Green
  'Engineering': '#795548',   // Brown
  'Education': '#E91E63',     // Pink
  'Health': '#F44336',        // Red
  'Business': '#607D8B',      // Blue Grey
  'Services': '#CDDC39',      // Lime
  'MetaLearning': '#00BCD4',  // Cyan
};

// Edge style based on type
export function edgeStyle(edgeType: string): { lineStyle: string; color: string } {
  switch (edgeType) {
    case 'Requires': return { lineStyle: 'solid', color: '#666' };
    case 'LeadsTo': return { lineStyle: 'dashed', color: '#999' };
    case 'Recommends': return { lineStyle: 'dotted', color: '#bbb' };
    default: return { lineStyle: 'solid', color: '#ccc' };
  }
}

// Node size based on estimated hours
export function nodeSize(hours: number): number {
  return Math.max(10, Math.min(50, 10 + Math.sqrt(hours) * 2));
}

// Simple BFS pathfinding (client-side)
export function findPath(
  nodes: CurriculumNode[],
  edges: CurriculumEdge[],
  startId: string,
  endId: string
): string[] {
  const adj: Map<string, string[]> = new Map();
  for (const edge of edges) {
    if (!adj.has(edge.from)) adj.set(edge.from, []);
    adj.get(edge.from)!.push(edge.to);
  }

  const visited = new Set<string>();
  const queue: [string, string[]][] = [[startId, [startId]]];

  while (queue.length > 0) {
    const [current, path] = queue.shift()!;
    if (current === endId) return path;
    if (visited.has(current)) continue;
    visited.add(current);

    const neighbors = adj.get(current) || [];
    for (const next of neighbors) {
      if (!visited.has(next)) {
        queue.push([next, [...path, next]]);
      }
    }
  }

  return []; // no path found
}

// Load the unified curriculum graph
export async function loadCurriculumGraph(): Promise<CurriculumDocument> {
  const response = await fetch('/unified_k_to_phd.json');
  return response.json();
}

// Convert to Cytoscape elements
export function toCytoscapeElements(doc: CurriculumDocument) {
  const nodeElements = doc.nodes.map(node => ({
    data: {
      id: node.id,
      label: node.title.length > 40 ? node.title.slice(0, 37) + '...' : node.title,
      fullTitle: node.title,
      description: node.description,
      nodeType: node.node_type,
      difficulty: node.difficulty,
      domain: node.domain,
      subdomain: node.subdomain,
      tags: node.tags,
      estimatedHours: node.estimated_hours,
      gradeLevel: node.grade_levels[0] || 'Unknown',
      bloomLevel: node.bloom_level,
      subjectArea: node.subject_area,
      field: classifyField(node.subject_area),
      color: FIELD_COLORS[classifyField(node.subject_area)] || '#999',
      size: nodeSize(node.estimated_hours),
    },
  }));

  const nodeIds = new Set(doc.nodes.map(n => n.id));
  const edgeElements = doc.edges
    .filter(e => nodeIds.has(e.from) && nodeIds.has(e.to))
    .map((edge, i) => {
      const style = edgeStyle(edge.edge_type);
      return {
        data: {
          id: `e${i}`,
          source: edge.from,
          target: edge.to,
          edgeType: edge.edge_type,
          strength: edge.strength_permille,
          rationale: edge.rationale,
          lineStyle: style.lineStyle,
          color: style.color,
        },
      };
    });

  return [...nodeElements, ...edgeElements];
}
