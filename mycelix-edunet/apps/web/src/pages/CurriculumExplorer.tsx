// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Interactive curriculum graph explorer

import { useEffect, useRef, useState, useCallback } from 'react';
import {
  loadCurriculumGraph,
  toCytoscapeElements,
  findPath,
  FIELD_COLORS,
  CurriculumNode,
  CurriculumDocument,
} from '../lib/curriculumGraph';

// Cytoscape will be loaded dynamically
let cytoscape: any = null;

interface NodeDetail {
  id: string;
  title: string;
  description: string;
  nodeType: string;
  difficulty: string;
  bloomLevel: string;
  gradeLevel: string;
  subjectArea: string;
  estimatedHours: number;
  tags: string[];
}

export default function CurriculumExplorer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<any>(null);
  const [doc, setDoc] = useState<CurriculumDocument | null>(null);
  const [selectedNode, setSelectedNode] = useState<NodeDetail | null>(null);
  const [filter, setFilter] = useState({ subject: '', level: '' });
  const [pathStart, setPathStart] = useState('');
  const [pathEnd, setPathEnd] = useState('');
  const [pathResult, setPathResult] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load graph data
  useEffect(() => {
    loadCurriculumGraph()
      .then(data => {
        setDoc(data);
        setLoading(false);
      })
      .catch(err => {
        setError(`Failed to load curriculum graph: ${err.message}`);
        setLoading(false);
      });
  }, []);

  // Initialize Cytoscape
  useEffect(() => {
    if (!doc || !containerRef.current) return;

    // Dynamic import of cytoscape
    import('cytoscape').then(mod => {
      cytoscape = mod.default;

      const elements = toCytoscapeElements(doc);
      const cy = cytoscape({
        container: containerRef.current,
        elements,
        style: [
          {
            selector: 'node',
            style: {
              'background-color': 'data(color)',
              'label': 'data(label)',
              'width': 'data(size)',
              'height': 'data(size)',
              'font-size': 8,
              'text-wrap': 'ellipsis',
              'text-max-width': '80px',
              'color': '#333',
              'text-outline-color': '#fff',
              'text-outline-width': 1,
            },
          },
          {
            selector: 'edge',
            style: {
              'width': 1,
              'line-color': 'data(color)',
              'target-arrow-color': 'data(color)',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier',
              'opacity': 0.4,
            },
          },
          {
            selector: 'node.highlighted',
            style: {
              'border-width': 3,
              'border-color': '#FFD700',
              'background-color': '#FFD700',
              'z-index': 999,
            },
          },
          {
            selector: 'edge.highlighted',
            style: {
              'line-color': '#FFD700',
              'target-arrow-color': '#FFD700',
              'width': 3,
              'opacity': 1,
              'z-index': 999,
            },
          },
          {
            selector: 'node.selected',
            style: {
              'border-width': 4,
              'border-color': '#FF4444',
            },
          },
        ],
        layout: {
          name: 'cose',
          animate: false,
          nodeRepulsion: () => 8000,
          idealEdgeLength: () => 80,
          gravity: 0.3,
          numIter: 300,
        },
        wheelSensitivity: 0.3,
      });

      // Click handler
      cy.on('tap', 'node', (evt: any) => {
        const data = evt.target.data();
        setSelectedNode({
          id: data.id,
          title: data.fullTitle,
          description: data.description,
          nodeType: data.nodeType,
          difficulty: data.difficulty,
          bloomLevel: data.bloomLevel,
          gradeLevel: data.gradeLevel,
          subjectArea: data.subjectArea,
          estimatedHours: data.estimatedHours,
          tags: data.tags || [],
        });

        // Remove previous selection
        cy.nodes().removeClass('selected');
        evt.target.addClass('selected');
      });

      cy.on('tap', (evt: any) => {
        if (evt.target === cy) {
          setSelectedNode(null);
          cy.nodes().removeClass('selected');
        }
      });

      cyRef.current = cy;
    });

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, [doc]);

  // Filter handler
  const applyFilter = useCallback(() => {
    if (!cyRef.current || !doc) return;
    const cy = cyRef.current;

    cy.nodes().show();
    cy.edges().show();

    if (filter.subject) {
      cy.nodes().forEach((node: any) => {
        const subj = node.data('subjectArea')?.toLowerCase() || '';
        if (!subj.includes(filter.subject.toLowerCase())) {
          node.hide();
        }
      });
    }

    if (filter.level) {
      cy.nodes().forEach((node: any) => {
        const grade = node.data('gradeLevel') || '';
        if (!grade.includes(filter.level)) {
          node.hide();
        }
      });
    }
  }, [filter, doc]);

  useEffect(() => { applyFilter(); }, [applyFilter]);

  // Pathfinding
  const runPathfind = useCallback(() => {
    if (!doc || !cyRef.current || !pathStart || !pathEnd) return;
    const cy = cyRef.current;

    // Clear previous highlights
    cy.elements().removeClass('highlighted');

    const path = findPath(doc.nodes, doc.edges, pathStart, pathEnd);
    setPathResult(path);

    if (path.length > 0) {
      // Highlight path nodes
      for (const nodeId of path) {
        cy.getElementById(nodeId).addClass('highlighted');
      }
      // Highlight path edges
      for (let i = 0; i < path.length - 1; i++) {
        const edges = cy.edges(`[source="${path[i]}"][target="${path[i+1]}"]`);
        edges.addClass('highlighted');
      }
      // Fit view to path
      const pathNodes = cy.nodes('.highlighted');
      if (pathNodes.length > 0) {
        cy.fit(pathNodes, 50);
      }
    }
  }, [doc, pathStart, pathEnd]);

  if (loading) {
    return <div style={{ padding: 20, textAlign: 'center' }}>Loading curriculum graph (1,718 nodes)...</div>;
  }

  if (error) {
    return <div style={{ padding: 20, color: 'red' }}>{error}</div>;
  }

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'system-ui' }}>
      {/* Sidebar */}
      <div style={{ width: 320, padding: 16, overflowY: 'auto', borderRight: '1px solid #ddd', background: '#f9f9f9' }}>
        <h2 style={{ margin: '0 0 16px' }}>Curriculum Explorer</h2>
        <p style={{ fontSize: 12, color: '#666' }}>
          {doc?.metadata.total_standards || 0} nodes | Click to inspect | Scroll to zoom
        </p>

        {/* Filters */}
        <div style={{ marginBottom: 16 }}>
          <h3 style={{ fontSize: 14, marginBottom: 8 }}>Filters</h3>
          <input
            placeholder="Filter by subject..."
            value={filter.subject}
            onChange={e => setFilter(f => ({ ...f, subject: e.target.value }))}
            style={{ width: '100%', padding: 6, marginBottom: 8, border: '1px solid #ccc', borderRadius: 4 }}
          />
          <select
            value={filter.level}
            onChange={e => setFilter(f => ({ ...f, level: e.target.value }))}
            style={{ width: '100%', padding: 6, border: '1px solid #ccc', borderRadius: 4 }}
          >
            <option value="">All levels</option>
            <option value="Kindergarten">Kindergarten</option>
            {[1,2,3,4,5,6,7,8,9,10,11,12].map(g =>
              <option key={g} value={`Grade${g}`}>Grade {g}</option>
            )}
            <option value="Undergraduate">Undergraduate</option>
            <option value="Graduate">Graduate</option>
            <option value="Doctoral">Doctoral</option>
          </select>
        </div>

        {/* Pathfinding */}
        <div style={{ marginBottom: 16 }}>
          <h3 style={{ fontSize: 14, marginBottom: 8 }}>Find Path</h3>
          <input
            placeholder="Start node ID..."
            value={pathStart}
            onChange={e => setPathStart(e.target.value)}
            style={{ width: '100%', padding: 6, marginBottom: 4, border: '1px solid #ccc', borderRadius: 4, fontSize: 11 }}
          />
          <input
            placeholder="End node ID..."
            value={pathEnd}
            onChange={e => setPathEnd(e.target.value)}
            style={{ width: '100%', padding: 6, marginBottom: 8, border: '1px solid #ccc', borderRadius: 4, fontSize: 11 }}
          />
          <button onClick={runPathfind} style={{ width: '100%', padding: 8, background: '#4CAF50', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}>
            Find Path
          </button>
          {pathResult.length > 0 && (
            <div style={{ marginTop: 8, fontSize: 11, color: '#666' }}>
              Path: {pathResult.length} steps
              <div style={{ maxHeight: 100, overflowY: 'auto', marginTop: 4 }}>
                {pathResult.map((id, i) => <div key={i}>{i+1}. {id}</div>)}
              </div>
            </div>
          )}
          {pathResult.length === 0 && pathStart && pathEnd && (
            <div style={{ marginTop: 8, fontSize: 11, color: '#c00' }}>No path found</div>
          )}
        </div>

        {/* Legend */}
        <div style={{ marginBottom: 16 }}>
          <h3 style={{ fontSize: 14, marginBottom: 8 }}>ISCED-F Fields</h3>
          {Object.entries(FIELD_COLORS).map(([field, color]) => (
            <div key={field} style={{ display: 'flex', alignItems: 'center', marginBottom: 4, fontSize: 11 }}>
              <div style={{ width: 12, height: 12, background: color, borderRadius: '50%', marginRight: 8 }} />
              {field}
            </div>
          ))}
        </div>

        {/* Node Detail */}
        {selectedNode && (
          <div style={{ background: 'white', padding: 12, borderRadius: 8, border: '1px solid #ddd' }}>
            <h3 style={{ fontSize: 14, marginTop: 0 }}>{selectedNode.title}</h3>
            <div style={{ fontSize: 11, color: '#666', marginBottom: 8 }}>
              {selectedNode.id}
            </div>
            <p style={{ fontSize: 12, lineHeight: 1.4 }}>
              {selectedNode.description.slice(0, 200)}
              {selectedNode.description.length > 200 ? '...' : ''}
            </p>
            <div style={{ fontSize: 11, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
              <div><strong>Type:</strong> {selectedNode.nodeType}</div>
              <div><strong>Difficulty:</strong> {selectedNode.difficulty}</div>
              <div><strong>Bloom:</strong> {selectedNode.bloomLevel}</div>
              <div><strong>Hours:</strong> {selectedNode.estimatedHours}</div>
              <div><strong>Level:</strong> {selectedNode.gradeLevel}</div>
              <div><strong>Subject:</strong> {selectedNode.subjectArea}</div>
            </div>
            {selectedNode.tags.length > 0 && (
              <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {selectedNode.tags.slice(0, 8).map(tag => (
                  <span key={tag} style={{ fontSize: 10, padding: '2px 6px', background: '#e0e0e0', borderRadius: 10 }}>
                    {tag}
                  </span>
                ))}
              </div>
            )}
            <div style={{ marginTop: 8 }}>
              <button
                onClick={() => { setPathStart(selectedNode.id); }}
                style={{ fontSize: 10, padding: '4px 8px', marginRight: 4, cursor: 'pointer' }}
              >
                Set as Start
              </button>
              <button
                onClick={() => { setPathEnd(selectedNode.id); }}
                style={{ fontSize: 10, padding: '4px 8px', cursor: 'pointer' }}
              >
                Set as End
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Graph Container */}
      <div ref={containerRef} style={{ flex: 1 }} />
    </div>
  );
}
