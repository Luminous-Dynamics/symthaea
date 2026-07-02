// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Timeline Component
 *
 * Multi-track arrangement view with:
 * - Drag and drop clips
 * - Track headers with controls
 * - Time ruler with markers
 * - Playhead and loop region
 * - Zoom and scroll
 * - Snap to grid
 */

import React, {
  useRef,
  useState,
  useEffect,
  useCallback,
  memo,
  forwardRef,
  useImperativeHandle,
} from 'react';

// ==================== Types ====================

export interface TimelineClip {
  id: string;
  trackId: string;
  name: string;
  start: number;        // beats
  duration: number;     // beats
  offset: number;       // clip content offset
  color: string;
  type: 'audio' | 'midi' | 'automation';
  waveformData?: Float32Array;
  midiNotes?: { pitch: number; start: number; duration: number }[];
  selected?: boolean;
  muted?: boolean;
}

export interface TimelineTrack {
  id: string;
  name: string;
  color: string;
  height: number;
  mute: boolean;
  solo: boolean;
  armed: boolean;
  volume: number;
  pan: number;
  collapsed?: boolean;
}

export interface TimelineMarker {
  id: string;
  time: number;         // beats
  name: string;
  color: string;
  type: 'marker' | 'loop-start' | 'loop-end';
}

export interface LoopRegion {
  start: number;
  end: number;
  enabled: boolean;
}

export interface TimelineProps {
  /** Array of tracks */
  tracks: TimelineTrack[];
  /** Array of clips */
  clips: TimelineClip[];
  /** Array of markers */
  markers?: TimelineMarker[];
  /** Total length in beats */
  length: number;
  /** Tempo in BPM */
  tempo?: number;
  /** Time signature [numerator, denominator] */
  timeSignature?: [number, number];
  /** Current playhead position in beats */
  playheadPosition?: number;
  /** Loop region */
  loop?: LoopRegion;
  /** Callback when clips change */
  onClipsChange: (clips: TimelineClip[]) => void;
  /** Callback when tracks change */
  onTracksChange: (tracks: TimelineTrack[]) => void;
  /** Callback when playhead is moved */
  onSeek?: (position: number) => void;
  /** Callback when loop region changes */
  onLoopChange?: (loop: LoopRegion) => void;
  /** Callback when marker is added/changed */
  onMarkerChange?: (markers: TimelineMarker[]) => void;
  /** Grid subdivision */
  quantize?: number;
  /** Zoom level */
  zoom?: number;
  /** Scroll offset X in beats */
  scrollX?: number;
  /** Scroll offset Y in pixels */
  scrollY?: number;
  /** Pixels per beat at zoom 1 */
  basePixelsPerBeat?: number;
  /** Track header width */
  headerWidth?: number;
  /** Ruler height */
  rulerHeight?: number;
  /** Show minimap */
  showMinimap?: boolean;
  /** Height in pixels */
  height?: number;
  /** Class name */
  className?: string;
}

export interface TimelineRef {
  scrollToPosition: (beat: number) => void;
  zoomToFit: () => void;
  selectAll: () => void;
  deleteSelected: () => void;
}

// ==================== Helper Functions ====================

function formatTime(beats: number, tempo: number, timeSignature: [number, number]): string {
  const beatsPerMeasure = timeSignature[0];
  const measure = Math.floor(beats / beatsPerMeasure) + 1;
  const beat = Math.floor(beats % beatsPerMeasure) + 1;
  const tick = Math.floor((beats % 1) * 960);
  return `${measure}.${beat}.${tick.toString().padStart(3, '0')}`;
}

function beatsToSeconds(beats: number, tempo: number): number {
  return (beats / tempo) * 60;
}

// ==================== Sub-Components ====================

interface TrackHeaderProps {
  track: TimelineTrack;
  onUpdate: (updates: Partial<TimelineTrack>) => void;
  selected: boolean;
  onSelect: () => void;
}

const TrackHeader = memo(function TrackHeader({
  track,
  onUpdate,
  selected,
  onSelect,
}: TrackHeaderProps) {
  return (
    <div
      onClick={onSelect}
      style={{
        height: track.collapsed ? 24 : track.height,
        background: selected
          ? 'var(--color-surface-active, #333)'
          : 'var(--color-surface, #1F1F1F)',
        borderBottom: '1px solid var(--color-surface-border, #333)',
        display: 'flex',
        flexDirection: 'column',
        padding: 4,
        gap: 4,
        cursor: 'pointer',
        transition: 'height 150ms ease',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        {/* Color indicator */}
        <div
          style={{
            width: 4,
            height: 16,
            background: track.color,
            borderRadius: 2,
          }}
        />

        {/* Track name */}
        <span
          style={{
            flex: 1,
            fontSize: 11,
            fontWeight: 500,
            color: 'var(--color-text, #FAFAFA)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {track.name}
        </span>

        {/* Collapse button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onUpdate({ collapsed: !track.collapsed });
          }}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--color-text-subtle, #6B6B6B)',
            cursor: 'pointer',
            fontSize: 10,
            padding: 2,
          }}
        >
          {track.collapsed ? '▶' : '▼'}
        </button>
      </div>

      {!track.collapsed && (
        <div style={{ display: 'flex', gap: 2 }}>
          {/* Mute */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onUpdate({ mute: !track.mute });
            }}
            style={{
              width: 20,
              height: 16,
              border: 'none',
              borderRadius: 3,
              background: track.mute ? '#EF4444' : 'var(--color-background-sunken, #080808)',
              color: track.mute ? '#000' : 'var(--color-text-subtle, #6B6B6B)',
              fontSize: 9,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            M
          </button>

          {/* Solo */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onUpdate({ solo: !track.solo });
            }}
            style={{
              width: 20,
              height: 16,
              border: 'none',
              borderRadius: 3,
              background: track.solo ? '#FBBF24' : 'var(--color-background-sunken, #080808)',
              color: track.solo ? '#000' : 'var(--color-text-subtle, #6B6B6B)',
              fontSize: 9,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            S
          </button>

          {/* Record arm */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onUpdate({ armed: !track.armed });
            }}
            style={{
              width: 20,
              height: 16,
              border: 'none',
              borderRadius: 3,
              background: track.armed ? '#EF4444' : 'var(--color-background-sunken, #080808)',
              color: track.armed ? '#000' : 'var(--color-text-subtle, #6B6B6B)',
              fontSize: 9,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            R
          </button>
        </div>
      )}
    </div>
  );
});

// ==================== Main Component ====================

export const Timeline = memo(forwardRef<TimelineRef, TimelineProps>(
  function Timeline(
    {
      tracks,
      clips,
      markers = [],
      length,
      tempo = 120,
      timeSignature = [4, 4],
      playheadPosition = 0,
      loop,
      onClipsChange,
      onTracksChange,
      onSeek,
      onLoopChange,
      onMarkerChange,
      quantize = 4,
      zoom = 1,
      scrollX = 0,
      scrollY = 0,
      basePixelsPerBeat = 20,
      headerWidth = 150,
      rulerHeight = 28,
      showMinimap = true,
      height = 400,
      className,
    },
    ref
  ) {
    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const rulerCanvasRef = useRef<HTMLCanvasElement>(null);

    const [canvasWidth, setCanvasWidth] = useState(800);
    const [isDragging, setIsDragging] = useState(false);
    const [dragMode, setDragMode] = useState<'move' | 'resize-left' | 'resize-right' | 'select' | null>(null);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
    const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null);

    const pixelsPerBeat = basePixelsPerBeat * zoom;
    const totalTracksHeight = tracks.reduce((sum, t) => sum + (t.collapsed ? 24 : t.height), 0);
    const minimapHeight = showMinimap ? 40 : 0;
    const contentHeight = height - rulerHeight - minimapHeight;

    // Handle resize
    useEffect(() => {
      const container = containerRef.current;
      if (!container) return;

      const observer = new ResizeObserver((entries) => {
        setCanvasWidth(entries[0].contentRect.width - headerWidth);
      });

      observer.observe(container);
      return () => observer.disconnect();
    }, [headerWidth]);

    // Coordinate conversions
    const beatToX = useCallback((beat: number): number => {
      return (beat - scrollX) * pixelsPerBeat;
    }, [scrollX, pixelsPerBeat]);

    const xToBeat = useCallback((x: number): number => {
      return scrollX + x / pixelsPerBeat;
    }, [scrollX, pixelsPerBeat]);

    const getTrackAtY = useCallback((y: number): TimelineTrack | null => {
      let currentY = 0;
      for (const track of tracks) {
        const trackHeight = track.collapsed ? 24 : track.height;
        if (y >= currentY && y < currentY + trackHeight) {
          return track;
        }
        currentY += trackHeight;
      }
      return null;
    }, [tracks]);

    const getTrackY = useCallback((trackId: string): number => {
      let y = 0;
      for (const track of tracks) {
        if (track.id === trackId) return y;
        y += track.collapsed ? 24 : track.height;
      }
      return 0;
    }, [tracks]);

    // Snap to grid
    const snapToGrid = useCallback((beat: number): number => {
      const gridSize = 1 / quantize;
      return Math.round(beat / gridSize) * gridSize;
    }, [quantize]);

    // Draw ruler
    const drawRuler = useCallback(() => {
      const canvas = rulerCanvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (!canvas || !ctx) return;

      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvasWidth * dpr;
      canvas.height = rulerHeight * dpr;
      canvas.style.width = `${canvasWidth}px`;
      canvas.style.height = `${rulerHeight}px`;
      ctx.scale(dpr, dpr);

      // Background
      ctx.fillStyle = 'var(--color-surface, #1F1F1F)';
      ctx.fillRect(0, 0, canvasWidth, rulerHeight);

      // Loop region
      if (loop?.enabled) {
        const loopStartX = beatToX(loop.start);
        const loopEndX = beatToX(loop.end);
        ctx.fillStyle = 'rgba(139, 92, 246, 0.2)';
        ctx.fillRect(loopStartX, 0, loopEndX - loopStartX, rulerHeight);
      }

      // Time markings
      const beatsPerMeasure = timeSignature[0];
      const startBeat = Math.floor(scrollX / beatsPerMeasure) * beatsPerMeasure;
      const endBeat = scrollX + canvasWidth / pixelsPerBeat;

      ctx.font = '10px monospace';
      ctx.textAlign = 'left';

      for (let beat = startBeat; beat <= endBeat; beat++) {
        const x = beatToX(beat);
        if (x < 0) continue;

        const isMeasure = beat % beatsPerMeasure === 0;
        const isHalfMeasure = beat % (beatsPerMeasure / 2) === 0;

        if (isMeasure) {
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x, rulerHeight - 12);
          ctx.lineTo(x, rulerHeight);
          ctx.stroke();

          ctx.fillStyle = 'var(--color-text, #FAFAFA)';
          const measure = Math.floor(beat / beatsPerMeasure) + 1;
          ctx.fillText(`${measure}`, x + 4, 12);
        } else if (isHalfMeasure && pixelsPerBeat > 10) {
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x, rulerHeight - 6);
          ctx.lineTo(x, rulerHeight);
          ctx.stroke();
        }
      }

      // Markers
      markers.forEach(marker => {
        const x = beatToX(marker.time);
        if (x < 0 || x > canvasWidth) return;

        ctx.fillStyle = marker.color;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x + 6, 0);
        ctx.lineTo(x, 10);
        ctx.closePath();
        ctx.fill();

        if (marker.name) {
          ctx.fillStyle = marker.color;
          ctx.font = '9px sans-serif';
          ctx.fillText(marker.name, x + 8, 10);
        }
      });

      // Playhead
      const playheadX = beatToX(playheadPosition);
      if (playheadX >= 0 && playheadX <= canvasWidth) {
        ctx.fillStyle = 'var(--color-error, #EF4444)';
        ctx.beginPath();
        ctx.moveTo(playheadX - 6, 0);
        ctx.lineTo(playheadX + 6, 0);
        ctx.lineTo(playheadX, 10);
        ctx.closePath();
        ctx.fill();
      }

      // Border
      ctx.strokeStyle = 'var(--color-surface-border, #333)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, rulerHeight - 0.5);
      ctx.lineTo(canvasWidth, rulerHeight - 0.5);
      ctx.stroke();
    }, [canvasWidth, rulerHeight, scrollX, pixelsPerBeat, beatToX, timeSignature, loop, markers, playheadPosition]);

    // Draw tracks and clips
    const drawCanvas = useCallback(() => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (!canvas || !ctx) return;

      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvasWidth * dpr;
      canvas.height = totalTracksHeight * dpr;
      canvas.style.width = `${canvasWidth}px`;
      canvas.style.height = `${totalTracksHeight}px`;
      ctx.scale(dpr, dpr);

      // Background
      ctx.fillStyle = 'var(--color-background-sunken, #080808)';
      ctx.fillRect(0, 0, canvasWidth, totalTracksHeight);

      // Draw track backgrounds and grid
      let trackY = 0;
      tracks.forEach(track => {
        const trackHeight = track.collapsed ? 24 : track.height;

        // Track background (alternating)
        ctx.fillStyle = tracks.indexOf(track) % 2 === 0
          ? 'rgba(255, 255, 255, 0.02)'
          : 'transparent';
        ctx.fillRect(0, trackY, canvasWidth, trackHeight);

        // Track border
        ctx.strokeStyle = 'var(--color-surface-border, #333)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, trackY + trackHeight - 0.5);
        ctx.lineTo(canvasWidth, trackY + trackHeight - 0.5);
        ctx.stroke();

        trackY += trackHeight;
      });

      // Draw grid
      const beatsPerMeasure = timeSignature[0];
      const startBeat = Math.floor(scrollX);
      const endBeat = Math.ceil(scrollX + canvasWidth / pixelsPerBeat);

      for (let beat = startBeat; beat <= endBeat; beat++) {
        const x = beatToX(beat);
        if (x < 0) continue;

        const isMeasure = beat % beatsPerMeasure === 0;

        ctx.strokeStyle = isMeasure
          ? 'rgba(255, 255, 255, 0.15)'
          : 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, totalTracksHeight);
        ctx.stroke();
      }

      // Loop region
      if (loop?.enabled) {
        const loopStartX = beatToX(loop.start);
        const loopEndX = beatToX(loop.end);
        ctx.fillStyle = 'rgba(139, 92, 246, 0.1)';
        ctx.fillRect(loopStartX, 0, loopEndX - loopStartX, totalTracksHeight);
      }

      // Draw clips
      clips.forEach(clip => {
        const track = tracks.find(t => t.id === clip.trackId);
        if (!track) return;

        const x = beatToX(clip.start);
        const width = clip.duration * pixelsPerBeat;
        const y = getTrackY(clip.trackId);
        const clipHeight = (track.collapsed ? 24 : track.height) - 4;

        if (x + width < 0 || x > canvasWidth) return;

        // Clip background
        ctx.fillStyle = clip.muted ? 'rgba(100, 100, 100, 0.5)' : clip.color;
        ctx.globalAlpha = clip.selected ? 1 : 0.8;
        ctx.beginPath();
        ctx.roundRect(x + 1, y + 2, width - 2, clipHeight, 4);
        ctx.fill();
        ctx.globalAlpha = 1;

        // Clip border
        ctx.strokeStyle = clip.selected ? '#fff' : 'rgba(0, 0, 0, 0.3)';
        ctx.lineWidth = clip.selected ? 2 : 1;
        ctx.stroke();

        // Waveform preview
        if (clip.waveformData && width > 20 && clipHeight > 20) {
          ctx.save();
          ctx.beginPath();
          ctx.roundRect(x + 1, y + 2, width - 2, clipHeight, 4);
          ctx.clip();

          ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
          ctx.lineWidth = 1;
          ctx.beginPath();

          const centerY = y + 2 + clipHeight / 2;
          const amplitude = clipHeight / 2 - 4;
          const samplesPerPixel = Math.ceil(clip.waveformData.length / width);

          for (let px = 0; px < width; px++) {
            const sampleIndex = Math.floor(px * samplesPerPixel);
            const sample = clip.waveformData[sampleIndex] || 0;
            const sampleY = centerY - sample * amplitude;

            if (px === 0) {
              ctx.moveTo(x + px, sampleY);
            } else {
              ctx.lineTo(x + px, sampleY);
            }
          }
          ctx.stroke();
          ctx.restore();
        }

        // MIDI preview
        if (clip.midiNotes && width > 20 && clipHeight > 20) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
          clip.midiNotes.forEach(note => {
            const noteX = x + (note.start / clip.duration) * width;
            const noteWidth = Math.max(2, (note.duration / clip.duration) * width);
            const noteY = y + 4 + ((127 - note.pitch) / 127) * (clipHeight - 8);
            ctx.fillRect(noteX, noteY, noteWidth, 2);
          });
        }

        // Clip name
        if (width > 40) {
          ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
          ctx.font = '10px sans-serif';
          ctx.fillText(clip.name, x + 6, y + 14, width - 12);
        }

        // Resize handles
        if (clip.selected) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
          ctx.fillRect(x + 1, y + 2, 4, clipHeight);
          ctx.fillRect(x + width - 5, y + 2, 4, clipHeight);
        }
      });

      // Playhead
      const playheadX = beatToX(playheadPosition);
      if (playheadX >= 0 && playheadX <= canvasWidth) {
        ctx.strokeStyle = 'var(--color-error, #EF4444)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(playheadX, 0);
        ctx.lineTo(playheadX, totalTracksHeight);
        ctx.stroke();
      }
    }, [
      canvasWidth, totalTracksHeight, tracks, clips, scrollX, pixelsPerBeat,
      beatToX, getTrackY, timeSignature, loop, playheadPosition
    ]);

    // Animation loop
    useEffect(() => {
      drawRuler();
      drawCanvas();
    }, [drawRuler, drawCanvas]);

    // Mouse handlers
    const handleCanvasMouseDown = useCallback((e: React.MouseEvent) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top + scrollY;
      const beat = xToBeat(x);

      // Find clicked clip
      const clickedClip = clips.find(clip => {
        const clipX = beatToX(clip.start);
        const clipWidth = clip.duration * pixelsPerBeat;
        const clipY = getTrackY(clip.trackId);
        const track = tracks.find(t => t.id === clip.trackId);
        const clipHeight = track ? (track.collapsed ? 24 : track.height) : 60;

        return x >= clipX && x <= clipX + clipWidth && y >= clipY && y <= clipY + clipHeight;
      });

      if (clickedClip) {
        const clipX = beatToX(clickedClip.start);
        const clipWidth = clickedClip.duration * pixelsPerBeat;

        // Determine drag mode
        let mode: 'move' | 'resize-left' | 'resize-right' = 'move';
        if (x < clipX + 8) mode = 'resize-left';
        else if (x > clipX + clipWidth - 8) mode = 'resize-right';

        // Select clip
        if (!e.shiftKey) {
          onClipsChange(clips.map(c => ({
            ...c,
            selected: c.id === clickedClip.id,
          })));
        } else {
          onClipsChange(clips.map(c => ({
            ...c,
            selected: c.id === clickedClip.id ? !c.selected : c.selected,
          })));
        }

        setDragMode(mode);
        setDragStart({ x: beat, y });
        setIsDragging(true);
      } else {
        // Click on empty space - seek or start selection
        if (e.shiftKey) {
          setDragMode('select');
          setDragStart({ x: beat, y });
          setIsDragging(true);
        } else {
          onSeek?.(snapToGrid(beat));
          onClipsChange(clips.map(c => ({ ...c, selected: false })));
        }
      }
    }, [clips, onClipsChange, xToBeat, beatToX, getTrackY, tracks, pixelsPerBeat, scrollY, onSeek, snapToGrid]);

    const handleCanvasMouseMove = useCallback((e: React.MouseEvent) => {
      if (!isDragging) return;

      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top + scrollY;
      const beat = xToBeat(x);

      const selectedClips = clips.filter(c => c.selected);
      if (selectedClips.length === 0) return;

      if (dragMode === 'move') {
        const deltaBeat = snapToGrid(beat) - snapToGrid(dragStart.x);
        const targetTrack = getTrackAtY(y);

        onClipsChange(clips.map(clip => {
          if (!clip.selected) return clip;
          return {
            ...clip,
            start: Math.max(0, clip.start + deltaBeat),
            trackId: targetTrack?.id || clip.trackId,
          };
        }));

        setDragStart({ x: snapToGrid(beat), y });
      } else if (dragMode === 'resize-right') {
        const snappedBeat = snapToGrid(beat);

        onClipsChange(clips.map(clip => {
          if (!clip.selected) return clip;
          const newDuration = Math.max(1 / quantize, snappedBeat - clip.start);
          return { ...clip, duration: newDuration };
        }));
      } else if (dragMode === 'resize-left') {
        const snappedBeat = snapToGrid(beat);

        onClipsChange(clips.map(clip => {
          if (!clip.selected) return clip;
          const endBeat = clip.start + clip.duration;
          const newStart = Math.min(snappedBeat, endBeat - 1 / quantize);
          const newDuration = endBeat - newStart;
          return { ...clip, start: Math.max(0, newStart), duration: newDuration };
        }));
      }
    }, [isDragging, dragMode, dragStart, clips, onClipsChange, xToBeat, scrollY, snapToGrid, getTrackAtY, quantize]);

    const handleMouseUp = useCallback(() => {
      setIsDragging(false);
      setDragMode(null);
    }, []);

    // Ruler click for seeking
    const handleRulerClick = useCallback((e: React.MouseEvent) => {
      const rect = rulerCanvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const x = e.clientX - rect.left;
      const beat = xToBeat(x);
      onSeek?.(snapToGrid(beat));
    }, [xToBeat, onSeek, snapToGrid]);

    // Track updates
    const handleTrackUpdate = useCallback((trackId: string, updates: Partial<TimelineTrack>) => {
      onTracksChange(tracks.map(t => t.id === trackId ? { ...t, ...updates } : t));
    }, [tracks, onTracksChange]);

    // Keyboard shortcuts
    useEffect(() => {
      const handleKeyDown = (e: KeyboardEvent) => {
        if (e.key === 'Delete' || e.key === 'Backspace') {
          onClipsChange(clips.filter(c => !c.selected));
        } else if (e.key === 'a' && (e.ctrlKey || e.metaKey)) {
          e.preventDefault();
          onClipsChange(clips.map(c => ({ ...c, selected: true })));
        } else if (e.key === 'Escape') {
          onClipsChange(clips.map(c => ({ ...c, selected: false })));
        } else if (e.key === 'd' && (e.ctrlKey || e.metaKey)) {
          e.preventDefault();
          const selectedClips = clips.filter(c => c.selected);
          const newClips = selectedClips.map(c => ({
            ...c,
            id: `${c.id}-copy-${Date.now()}`,
            start: c.start + c.duration,
            selected: true,
          }));
          onClipsChange([
            ...clips.map(c => ({ ...c, selected: false })),
            ...newClips,
          ]);
        }
      };

      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }, [clips, onClipsChange]);

    // Expose methods via ref
    useImperativeHandle(ref, () => ({
      scrollToPosition: (beat: number) => {
        console.log('Scroll to:', beat);
      },
      zoomToFit: () => {
        console.log('Zoom to fit');
      },
      selectAll: () => {
        onClipsChange(clips.map(c => ({ ...c, selected: true })));
      },
      deleteSelected: () => {
        onClipsChange(clips.filter(c => !c.selected));
      },
    }), [clips, onClipsChange]);

    return (
      <div
        ref={containerRef}
        className={className}
        style={{
          display: 'flex',
          flexDirection: 'column',
          height,
          background: 'var(--color-background, #0F0F0F)',
          borderRadius: 8,
          overflow: 'hidden',
        }}
      >
        {/* Ruler row */}
        <div style={{ display: 'flex', height: rulerHeight }}>
          <div
            style={{
              width: headerWidth,
              background: 'var(--color-surface, #1F1F1F)',
              borderRight: '1px solid var(--color-surface-border, #333)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 10,
              color: 'var(--color-text-muted, #A1A1A1)',
            }}
          >
            {formatTime(playheadPosition, tempo, timeSignature)}
          </div>
          <canvas
            ref={rulerCanvasRef}
            onClick={handleRulerClick}
            style={{ display: 'block', cursor: 'pointer' }}
          />
        </div>

        {/* Main content */}
        <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          {/* Track headers */}
          <div
            style={{
              width: headerWidth,
              overflowY: 'auto',
              background: 'var(--color-surface, #1F1F1F)',
              borderRight: '1px solid var(--color-surface-border, #333)',
            }}
          >
            {tracks.map(track => (
              <TrackHeader
                key={track.id}
                track={track}
                onUpdate={(updates) => handleTrackUpdate(track.id, updates)}
                selected={track.id === selectedTrackId}
                onSelect={() => setSelectedTrackId(track.id)}
              />
            ))}
          </div>

          {/* Clip canvas */}
          <div style={{ flex: 1, overflow: 'auto' }}>
            <canvas
              ref={canvasRef}
              onMouseDown={handleCanvasMouseDown}
              onMouseMove={handleCanvasMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              style={{
                display: 'block',
                cursor: isDragging
                  ? dragMode?.includes('resize') ? 'ew-resize' : 'move'
                  : 'default',
              }}
            />
          </div>
        </div>

        {/* Minimap */}
        {showMinimap && (
          <div
            style={{
              height: minimapHeight,
              background: 'var(--color-surface, #1F1F1F)',
              borderTop: '1px solid var(--color-surface-border, #333)',
              display: 'flex',
              alignItems: 'center',
              padding: '0 8px',
            }}
          >
            <div
              style={{
                flex: 1,
                height: 24,
                background: 'var(--color-background-sunken, #080808)',
                borderRadius: 4,
                position: 'relative',
                overflow: 'hidden',
              }}
            >
              {/* Minimap clips */}
              {clips.map(clip => {
                const x = (clip.start / length) * 100;
                const width = (clip.duration / length) * 100;
                return (
                  <div
                    key={clip.id}
                    style={{
                      position: 'absolute',
                      left: `${x}%`,
                      width: `${Math.max(0.5, width)}%`,
                      height: 4,
                      top: `${(tracks.findIndex(t => t.id === clip.trackId) / tracks.length) * 100}%`,
                      background: clip.color,
                      borderRadius: 1,
                    }}
                  />
                );
              })}

              {/* Viewport indicator */}
              <div
                style={{
                  position: 'absolute',
                  left: `${(scrollX / length) * 100}%`,
                  width: `${(canvasWidth / pixelsPerBeat / length) * 100}%`,
                  top: 0,
                  bottom: 0,
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  borderRadius: 2,
                }}
              />

              {/* Playhead */}
              <div
                style={{
                  position: 'absolute',
                  left: `${(playheadPosition / length) * 100}%`,
                  top: 0,
                  bottom: 0,
                  width: 2,
                  background: 'var(--color-error, #EF4444)',
                }}
              />
            </div>
          </div>
        )}
      </div>
    );
  }
));

export default Timeline;
