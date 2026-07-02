// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * EpistemicCube - 3D Visualization of E-N-M Classification Space
 *
 * Renders claims as points in a 3D cube where:
 * - X-axis: Empirical (verifiable through observation)
 * - Y-axis: Normative (value-based, ethical)
 * - Z-axis: Mythic (meaning-making, narrative)
 *
 * @example
 * ```tsx
 * import { EpistemicCube } from '@mycelix/knowledge-sdk/visualizations';
 *
 * <EpistemicCube
 *   claims={claims}
 *   onClaimClick={(claim) => console.log('Selected:', claim)}
 *   showLabels={true}
 *   autoRotate={true}
 * />
 * ```
 */

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';

// ============================================================================
// Types
// ============================================================================

export interface CubePoint {
  id: string;
  label: string;
  position: {
    empirical: number;   // 0-1, X-axis
    normative: number;   // 0-1, Y-axis
    mythic: number;      // 0-1, Z-axis
  };
  credibility?: number;
  size?: number;
  color?: string;
  metadata?: Record<string, unknown>;
}

export interface EpistemicCubeProps {
  /** Claims to visualize as points in the cube */
  claims: CubePoint[];
  /** Width of the canvas */
  width?: number | string;
  /** Height of the canvas */
  height?: number | string;
  /** Show axis labels */
  showAxisLabels?: boolean;
  /** Show claim labels */
  showLabels?: boolean;
  /** Show grid lines */
  showGrid?: boolean;
  /** Auto-rotate the cube */
  autoRotate?: boolean;
  /** Rotation speed (degrees per frame) */
  rotationSpeed?: number;
  /** Initial rotation angles */
  initialRotation?: { x: number; y: number };
  /** Background color */
  backgroundColor?: string;
  /** Axis colors */
  axisColors?: { x: string; y: string; z: string };
  /** Callback when a claim is clicked */
  onClaimClick?: (claim: CubePoint) => void;
  /** Callback when a claim is hovered */
  onClaimHover?: (claim: CubePoint | null) => void;
  /** CSS class name */
  className?: string;
  /** Inline styles */
  style?: React.CSSProperties;
  /** Loading state */
  loading?: boolean;
  /** Error state */
  error?: Error | null;
  /** Highlight specific points */
  highlightedIds?: string[];
  /** Filter by quadrant */
  quadrantFilter?: 'empirical' | 'normative' | 'mythic' | 'balanced' | null;
}

interface Rotation {
  x: number;
  y: number;
}

interface ProjectedPoint extends CubePoint {
  screenX: number;
  screenY: number;
  depth: number;
}

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_AXIS_COLORS = {
  x: '#ef4444', // Red for Empirical
  y: '#22c55e', // Green for Normative
  z: '#3b82f6', // Blue for Mythic
};

const AXIS_LABELS = {
  x: { label: 'Empirical', description: 'Verifiable through observation' },
  y: { label: 'Normative', description: 'Value-based, ethical' },
  z: { label: 'Mythic', description: 'Meaning-making, narrative' },
};

const QUADRANT_COLORS = {
  empirical: 'rgba(239, 68, 68, 0.3)',   // Red tint
  normative: 'rgba(34, 197, 94, 0.3)',   // Green tint
  mythic: 'rgba(59, 130, 246, 0.3)',     // Blue tint
  balanced: 'rgba(168, 85, 247, 0.3)',   // Purple tint
};

// ============================================================================
// 3D Math Utilities
// ============================================================================

function rotateX(point: [number, number, number], angle: number): [number, number, number] {
  const [x, y, z] = point;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [x, y * cos - z * sin, y * sin + z * cos];
}

function rotateY(point: [number, number, number], angle: number): [number, number, number] {
  const [x, y, z] = point;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [x * cos + z * sin, y, -x * sin + z * cos];
}

function project(
  point: [number, number, number],
  rotation: Rotation,
  scale: number,
  center: { x: number; y: number }
): { x: number; y: number; depth: number } {
  // Apply rotations
  let rotated = rotateX(point, rotation.x);
  rotated = rotateY(rotated, rotation.y);

  // Simple perspective projection
  const perspective = 600;
  const z = rotated[2] + 2; // Move back to prevent negative z
  const projectionScale = perspective / (perspective + z * scale);

  return {
    x: center.x + rotated[0] * scale * projectionScale,
    y: center.y - rotated[1] * scale * projectionScale,
    depth: z,
  };
}

// ============================================================================
// Component
// ============================================================================

export function EpistemicCube({
  claims,
  width = '100%',
  height = 500,
  showAxisLabels = true,
  showLabels = false,
  showGrid = true,
  autoRotate = false,
  rotationSpeed = 0.003,
  initialRotation = { x: -0.4, y: 0.6 },
  backgroundColor = '#f8fafc',
  axisColors = DEFAULT_AXIS_COLORS,
  onClaimClick,
  onClaimHover,
  className,
  style,
  loading = false,
  error = null,
  highlightedIds = [],
  quadrantFilter = null,
}: EpistemicCubeProps): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number>();

  const [rotation, setRotation] = useState<Rotation>(initialRotation);
  const [isDragging, setIsDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [hoveredPoint, setHoveredPoint] = useState<CubePoint | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 });

  // Filter claims based on quadrant
  const filteredClaims = useMemo(() => {
    if (!quadrantFilter) return claims;

    return claims.filter(claim => {
      const { empirical, normative, mythic } = claim.position;
      const maxDim = Math.max(empirical, normative, mythic);
      const isBalanced = Math.abs(empirical - normative) < 0.2 &&
                         Math.abs(normative - mythic) < 0.2 &&
                         Math.abs(empirical - mythic) < 0.2;

      switch (quadrantFilter) {
        case 'empirical':
          return maxDim === empirical && !isBalanced;
        case 'normative':
          return maxDim === normative && !isBalanced;
        case 'mythic':
          return maxDim === mythic && !isBalanced;
        case 'balanced':
          return isBalanced;
        default:
          return true;
      }
    });
  }, [claims, quadrantFilter]);

  // Get dimensions
  useEffect(() => {
    const updateDimensions = () => {
      const container = containerRef.current;
      if (!container) return;

      const w = typeof width === 'number' ? width : container.clientWidth;
      const h = typeof height === 'number' ? height : 500;

      setDimensions({ width: w, height: h });

      const canvas = canvasRef.current;
      if (canvas) {
        canvas.width = w;
        canvas.height = h;
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [width, height]);

  // Project points to screen coordinates
  const projectPoints = useCallback((): ProjectedPoint[] => {
    const scale = Math.min(dimensions.width, dimensions.height) * 0.35;
    const center = { x: dimensions.width / 2, y: dimensions.height / 2 };

    return filteredClaims.map(claim => {
      // Convert 0-1 coordinates to -1 to 1 for cube
      const point: [number, number, number] = [
        claim.position.empirical * 2 - 1,
        claim.position.normative * 2 - 1,
        claim.position.mythic * 2 - 1,
      ];

      const projected = project(point, rotation, scale, center);

      return {
        ...claim,
        screenX: projected.x,
        screenY: projected.y,
        depth: projected.depth,
      };
    }).sort((a, b) => b.depth - a.depth); // Sort by depth for proper layering
  }, [filteredClaims, rotation, dimensions]);

  // Render
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const { width: w, height: h } = dimensions;
    const scale = Math.min(w, h) * 0.35;
    const center = { x: w / 2, y: h / 2 };

    // Clear
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, w, h);

    // Draw cube edges
    drawCubeEdges(ctx, rotation, scale, center);

    // Draw grid
    if (showGrid) {
      drawGrid(ctx, rotation, scale, center);
    }

    // Draw axis labels
    if (showAxisLabels) {
      drawAxisLabels(ctx, rotation, scale, center, axisColors);
    }

    // Draw points
    const projectedPoints = projectPoints();
    drawPoints(ctx, projectedPoints, highlightedIds, hoveredPoint, showLabels);
  }, [
    dimensions, backgroundColor, showGrid, showAxisLabels,
    axisColors, rotation, projectPoints, highlightedIds, hoveredPoint, showLabels
  ]);

  // Draw cube edges
  const drawCubeEdges = (
    ctx: CanvasRenderingContext2D,
    rot: Rotation,
    scale: number,
    center: { x: number; y: number }
  ) => {
    const vertices: [number, number, number][] = [
      [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ];

    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 0], // Back face
      [4, 5], [5, 6], [6, 7], [7, 4], // Front face
      [0, 4], [1, 5], [2, 6], [3, 7], // Connecting edges
    ];

    const projected = vertices.map(v => project(v, rot, scale, center));

    // Draw edges
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;

    for (const [i, j] of edges) {
      const p1 = projected[i];
      const p2 = projected[j];

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    // Draw colored axes from origin
    const origin = project([0, 0, 0], rot, scale, center);
    const axisEnd = 1.2;

    // X-axis (Empirical)
    const xEnd = project([axisEnd, 0, 0], rot, scale, center);
    ctx.strokeStyle = axisColors.x;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(xEnd.x, xEnd.y);
    ctx.stroke();

    // Y-axis (Normative)
    const yEnd = project([0, axisEnd, 0], rot, scale, center);
    ctx.strokeStyle = axisColors.y;
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(yEnd.x, yEnd.y);
    ctx.stroke();

    // Z-axis (Mythic)
    const zEnd = project([0, 0, axisEnd], rot, scale, center);
    ctx.strokeStyle = axisColors.z;
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(zEnd.x, zEnd.y);
    ctx.stroke();
  };

  // Draw grid
  const drawGrid = (
    ctx: CanvasRenderingContext2D,
    rot: Rotation,
    scale: number,
    center: { x: number; y: number }
  ) => {
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
    ctx.lineWidth = 0.5;

    const steps = 4;
    const step = 2 / steps;

    // XY plane (z = -1)
    for (let i = 0; i <= steps; i++) {
      const v = -1 + i * step;

      // Horizontal lines
      const h1 = project([-1, v, -1], rot, scale, center);
      const h2 = project([1, v, -1], rot, scale, center);
      ctx.beginPath();
      ctx.moveTo(h1.x, h1.y);
      ctx.lineTo(h2.x, h2.y);
      ctx.stroke();

      // Vertical lines
      const v1 = project([v, -1, -1], rot, scale, center);
      const v2 = project([v, 1, -1], rot, scale, center);
      ctx.beginPath();
      ctx.moveTo(v1.x, v1.y);
      ctx.lineTo(v2.x, v2.y);
      ctx.stroke();
    }
  };

  // Draw axis labels
  const drawAxisLabels = (
    ctx: CanvasRenderingContext2D,
    rot: Rotation,
    scale: number,
    center: { x: number; y: number },
    colors: typeof DEFAULT_AXIS_COLORS
  ) => {
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // X-axis label
    const xLabel = project([1.4, 0, 0], rot, scale, center);
    ctx.fillStyle = colors.x;
    ctx.fillText('E', xLabel.x, xLabel.y);

    // Y-axis label
    const yLabel = project([0, 1.4, 0], rot, scale, center);
    ctx.fillStyle = colors.y;
    ctx.fillText('N', yLabel.x, yLabel.y);

    // Z-axis label
    const zLabel = project([0, 0, 1.4], rot, scale, center);
    ctx.fillStyle = colors.z;
    ctx.fillText('M', zLabel.x, zLabel.y);
  };

  // Draw points
  const drawPoints = (
    ctx: CanvasRenderingContext2D,
    points: ProjectedPoint[],
    highlighted: string[],
    hovered: CubePoint | null,
    labels: boolean
  ) => {
    for (const point of points) {
      const isHighlighted = highlighted.includes(point.id);
      const isHovered = hovered?.id === point.id;

      // Calculate size based on depth and credibility
      const baseSize = point.size ?? 8;
      const depthScale = 0.8 + (point.depth / 4) * 0.4;
      const size = baseSize * depthScale * (isHighlighted || isHovered ? 1.5 : 1);

      // Calculate color
      let color = point.color;
      if (!color) {
        // Default color based on position (blend of E-N-M)
        const r = Math.round(255 * point.position.empirical);
        const g = Math.round(255 * point.position.normative);
        const b = Math.round(255 * point.position.mythic);
        color = `rgb(${r}, ${g}, ${b})`;
      }

      // Draw point
      ctx.beginPath();
      ctx.arc(point.screenX, point.screenY, size, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = isHighlighted || isHovered ? 1 : 0.7;
      ctx.fill();

      // Draw border for highlighted/hovered
      if (isHighlighted || isHovered) {
        ctx.strokeStyle = '#1f2937';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Draw credibility ring
      if (point.credibility !== undefined) {
        ctx.beginPath();
        ctx.arc(point.screenX, point.screenY, size + 3, 0, Math.PI * 2 * point.credibility);
        ctx.strokeStyle = point.credibility >= 0.7 ? '#22c55e' :
                          point.credibility >= 0.4 ? '#f59e0b' : '#ef4444';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      ctx.globalAlpha = 1;

      // Draw label
      if (labels || isHovered) {
        ctx.font = '11px sans-serif';
        ctx.fillStyle = '#374151';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(
          point.label.length > 25 ? point.label.slice(0, 22) + '...' : point.label,
          point.screenX,
          point.screenY + size + 5
        );
      }
    }
  };

  // Animation loop
  useEffect(() => {
    const animate = () => {
      if (autoRotate && !isDragging) {
        setRotation(prev => ({
          x: prev.x,
          y: prev.y + rotationSpeed,
        }));
      }

      render();
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [autoRotate, isDragging, rotationSpeed, render]);

  // Mouse handlers
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    setLastMouse({ x: e.clientX, y: e.clientY });
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Check for hover
    const projectedPoints = projectPoints();
    let found: CubePoint | null = null;

    for (const point of projectedPoints) {
      const dist = Math.sqrt(
        (mouseX - point.screenX) ** 2 + (mouseY - point.screenY) ** 2
      );
      const size = (point.size ?? 8) * 1.5;

      if (dist <= size) {
        found = point;
        break;
      }
    }

    if (found !== hoveredPoint) {
      setHoveredPoint(found);
      onClaimHover?.(found);
    }

    // Handle dragging
    if (isDragging) {
      const dx = e.clientX - lastMouse.x;
      const dy = e.clientY - lastMouse.y;

      setRotation(prev => ({
        x: prev.x + dy * 0.01,
        y: prev.y + dx * 0.01,
      }));

      setLastMouse({ x: e.clientX, y: e.clientY });
    }
  }, [isDragging, lastMouse, projectPoints, hoveredPoint, onClaimHover]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleClick = useCallback((e: React.MouseEvent) => {
    if (hoveredPoint) {
      onClaimClick?.(hoveredPoint);
    }
  }, [hoveredPoint, onClaimClick]);

  // Loading/error states
  if (loading) {
    return (
      <div
        className={className}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width,
          height,
          backgroundColor,
          borderRadius: '8px',
          ...style,
        }}
      >
        <div style={{ textAlign: 'center', color: '#6b7280' }}>
          Loading cube...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div
        className={className}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width,
          height,
          backgroundColor: '#fef2f2',
          borderRadius: '8px',
          color: '#dc2626',
          ...style,
        }}
      >
        {error.message}
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={className}
      style={{
        width,
        height,
        position: 'relative',
        ...style,
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          cursor: isDragging ? 'grabbing' : hoveredPoint ? 'pointer' : 'grab',
          borderRadius: '8px',
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={handleClick}
      />

      {/* Legend */}
      <div
        style={{
          position: 'absolute',
          top: '16px',
          left: '16px',
          backgroundColor: 'rgba(255, 255, 255, 0.9)',
          padding: '12px',
          borderRadius: '8px',
          fontSize: '12px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        }}
      >
        <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>Epistemic Axes</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
          <div style={{ width: '12px', height: '12px', backgroundColor: axisColors.x, borderRadius: '2px' }} />
          <span><strong>E</strong>mpirical - Observable facts</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
          <div style={{ width: '12px', height: '12px', backgroundColor: axisColors.y, borderRadius: '2px' }} />
          <span><strong>N</strong>ormative - Values & ethics</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '12px', height: '12px', backgroundColor: axisColors.z, borderRadius: '2px' }} />
          <span><strong>M</strong>ythic - Meaning & narrative</span>
        </div>
      </div>

      {/* Stats */}
      <div
        style={{
          position: 'absolute',
          top: '16px',
          right: '16px',
          backgroundColor: 'rgba(255, 255, 255, 0.9)',
          padding: '12px',
          borderRadius: '8px',
          fontSize: '12px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        }}
      >
        <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>Statistics</div>
        <div>Total: {filteredClaims.length} claims</div>
        {quadrantFilter && <div>Filter: {quadrantFilter}</div>}
      </div>

      {/* Tooltip */}
      {hoveredPoint && (
        <div
          style={{
            position: 'absolute',
            left: (projectPoints().find(p => p.id === hoveredPoint.id)?.screenX ?? 0) + 20,
            top: (projectPoints().find(p => p.id === hoveredPoint.id)?.screenY ?? 0) - 60,
            backgroundColor: '#1f2937',
            color: '#ffffff',
            padding: '12px',
            borderRadius: '8px',
            fontSize: '12px',
            maxWidth: '250px',
            zIndex: 10,
            pointerEvents: 'none',
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>
            {hoveredPoint.label}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px', marginBottom: '8px' }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ color: axisColors.x, fontWeight: 'bold' }}>E</div>
              <div>{(hoveredPoint.position.empirical * 100).toFixed(0)}%</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ color: axisColors.y, fontWeight: 'bold' }}>N</div>
              <div>{(hoveredPoint.position.normative * 100).toFixed(0)}%</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ color: axisColors.z, fontWeight: 'bold' }}>M</div>
              <div>{(hoveredPoint.position.mythic * 100).toFixed(0)}%</div>
            </div>
          </div>
          {hoveredPoint.credibility !== undefined && (
            <div style={{ borderTop: '1px solid #374151', paddingTop: '8px' }}>
              Credibility: {(hoveredPoint.credibility * 100).toFixed(1)}%
            </div>
          )}
        </div>
      )}

      {/* Controls */}
      <div
        style={{
          position: 'absolute',
          bottom: '16px',
          left: '16px',
          display: 'flex',
          gap: '8px',
        }}
      >
        <button
          onClick={() => setRotation(initialRotation)}
          style={{
            padding: '8px 16px',
            borderRadius: '6px',
            border: '1px solid #e5e7eb',
            backgroundColor: '#ffffff',
            cursor: 'pointer',
            fontSize: '12px',
          }}
        >
          Reset View
        </button>
      </div>
    </div>
  );
}

export default EpistemicCube;
