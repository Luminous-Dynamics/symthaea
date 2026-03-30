// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Piano Roll Component
 *
 * MIDI note editor with:
 * - Keyboard display with octave labels
 * - Note drawing, resizing, moving
 * - Velocity editing
 * - Quantization grid
 * - Selection and multi-edit
 * - Zoom and scroll
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

export interface MIDINote {
  id: string;
  pitch: number;        // 0-127
  start: number;        // beats
  duration: number;     // beats
  velocity: number;     // 0-127
  selected?: boolean;
}

export interface PianoRollProps {
  /** Array of MIDI notes */
  notes: MIDINote[];
  /** Total length in beats */
  length: number;
  /** Callback when notes change */
  onNotesChange: (notes: MIDINote[]) => void;
  /** Current playhead position in beats */
  playheadPosition?: number;
  /** Grid subdivision (4 = 16th notes, 2 = 8th notes, etc.) */
  quantize?: number;
  /** Lowest visible note */
  minNote?: number;
  /** Highest visible note */
  maxNote?: number;
  /** Zoom level */
  zoom?: number;
  /** Scroll offset in beats */
  scrollX?: number;
  /** Scroll offset in notes */
  scrollY?: number;
  /** Note color */
  noteColor?: string;
  /** Selected note color */
  selectedColor?: string;
  /** Grid color */
  gridColor?: string;
  /** Show velocity bars */
  showVelocity?: boolean;
  /** Height of velocity lane */
  velocityHeight?: number;
  /** Keyboard width */
  keyboardWidth?: number;
  /** Row height per note */
  noteHeight?: number;
  /** Pixels per beat */
  pixelsPerBeat?: number;
  /** Height in pixels */
  height?: number;
  /** Note pressed callback (for preview) */
  onNotePreview?: (pitch: number) => void;
  /** Class name */
  className?: string;
}

export interface PianoRollRef {
  scrollToNote: (pitch: number) => void;
  scrollToBeat: (beat: number) => void;
  selectAll: () => void;
  clearSelection: () => void;
  deleteSelected: () => void;
}

// ==================== Constants ====================

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const BLACK_KEYS = [1, 3, 6, 8, 10]; // Indices of black keys in octave

function getNoteLabel(pitch: number): string {
  const octave = Math.floor(pitch / 12) - 1;
  const note = NOTE_NAMES[pitch % 12];
  return `${note}${octave}`;
}

function isBlackKey(pitch: number): boolean {
  return BLACK_KEYS.includes(pitch % 12);
}

// ==================== Component ====================

export const PianoRoll = memo(forwardRef<PianoRollRef, PianoRollProps>(
  function PianoRoll(
    {
      notes,
      length,
      onNotesChange,
      playheadPosition = 0,
      quantize = 4,
      minNote = 24,
      maxNote = 96,
      zoom = 1,
      scrollX = 0,
      scrollY = 0,
      noteColor = 'var(--color-primary, #8B5CF6)',
      selectedColor = 'var(--color-accent, #F59E0B)',
      gridColor = 'var(--color-surface-border, #333)',
      showVelocity = true,
      velocityHeight = 60,
      keyboardWidth = 60,
      noteHeight = 16,
      pixelsPerBeat = 40,
      height = 400,
      onNotePreview,
      className,
    },
    ref
  ) {
    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const velocityCanvasRef = useRef<HTMLCanvasElement>(null);

    const [canvasWidth, setCanvasWidth] = useState(800);
    const [isDragging, setIsDragging] = useState(false);
    const [dragMode, setDragMode] = useState<'move' | 'resize' | 'draw' | 'velocity' | null>(null);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
    const [hoveredNote, setHoveredNote] = useState<string | null>(null);

    const totalNotes = maxNote - minNote + 1;
    const gridHeight = totalNotes * noteHeight;
    const effectivePixelsPerBeat = pixelsPerBeat * zoom;
    const totalWidth = length * effectivePixelsPerBeat;

    // Handle resize
    useEffect(() => {
      const container = containerRef.current;
      if (!container) return;

      const observer = new ResizeObserver((entries) => {
        setCanvasWidth(entries[0].contentRect.width - keyboardWidth);
      });

      observer.observe(container);
      return () => observer.disconnect();
    }, [keyboardWidth]);

    // Convert coordinates
    const beatToX = useCallback((beat: number): number => {
      return (beat - scrollX) * effectivePixelsPerBeat;
    }, [scrollX, effectivePixelsPerBeat]);

    const xToBeat = useCallback((x: number): number => {
      return scrollX + x / effectivePixelsPerBeat;
    }, [scrollX, effectivePixelsPerBeat]);

    const pitchToY = useCallback((pitch: number): number => {
      return (maxNote - pitch - scrollY) * noteHeight;
    }, [maxNote, scrollY, noteHeight]);

    const yToPitch = useCallback((y: number): number => {
      return maxNote - Math.floor(y / noteHeight) - scrollY;
    }, [maxNote, scrollY, noteHeight]);

    // Quantize to grid
    const snapToGrid = useCallback((beat: number): number => {
      const gridSize = 1 / quantize;
      return Math.round(beat / gridSize) * gridSize;
    }, [quantize]);

    // Generate unique ID
    const generateId = (): string => `note-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    // Draw piano roll grid
    const drawGrid = useCallback(() => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (!canvas || !ctx) return;

      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvasWidth * dpr;
      canvas.height = gridHeight * dpr;
      canvas.style.width = `${canvasWidth}px`;
      canvas.style.height = `${gridHeight}px`;
      ctx.scale(dpr, dpr);

      // Background
      ctx.fillStyle = 'var(--color-background-sunken, #080808)';
      ctx.fillRect(0, 0, canvasWidth, gridHeight);

      // Draw note rows
      for (let pitch = minNote; pitch <= maxNote; pitch++) {
        const y = pitchToY(pitch);
        const isBlack = isBlackKey(pitch);

        // Row background
        ctx.fillStyle = isBlack
          ? 'rgba(0, 0, 0, 0.3)'
          : 'rgba(255, 255, 255, 0.02)';
        ctx.fillRect(0, y, canvasWidth, noteHeight);

        // Row border
        ctx.strokeStyle = gridColor;
        ctx.lineWidth = pitch % 12 === 0 ? 1 : 0.5; // Thicker line for C notes
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvasWidth, y);
        ctx.stroke();
      }

      // Draw beat grid
      const startBeat = Math.floor(scrollX);
      const endBeat = Math.ceil(scrollX + canvasWidth / effectivePixelsPerBeat);

      for (let beat = startBeat; beat <= endBeat; beat++) {
        const x = beatToX(beat);
        if (x < 0 || x > canvasWidth) continue;

        const isMeasure = beat % 4 === 0;
        ctx.strokeStyle = isMeasure ? 'rgba(255, 255, 255, 0.3)' : gridColor;
        ctx.lineWidth = isMeasure ? 1 : 0.5;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, gridHeight);
        ctx.stroke();

        // Subdivision lines
        if (quantize > 1) {
          for (let sub = 1; sub < quantize; sub++) {
            const subX = beatToX(beat + sub / quantize);
            if (subX < 0 || subX > canvasWidth) continue;

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(subX, 0);
            ctx.lineTo(subX, gridHeight);
            ctx.stroke();
          }
        }
      }

      // Draw notes
      notes.forEach(note => {
        const x = beatToX(note.start);
        const y = pitchToY(note.pitch);
        const width = note.duration * effectivePixelsPerBeat;

        if (x + width < 0 || x > canvasWidth) return;
        if (y < -noteHeight || y > gridHeight) return;

        // Note body
        const isHovered = note.id === hoveredNote;
        const color = note.selected ? selectedColor : noteColor;

        ctx.fillStyle = color;
        ctx.globalAlpha = isHovered ? 1 : 0.85;
        ctx.beginPath();
        ctx.roundRect(x + 1, y + 1, width - 2, noteHeight - 2, 3);
        ctx.fill();
        ctx.globalAlpha = 1;

        // Velocity indicator (brightness)
        const velocityAlpha = 0.3 + (note.velocity / 127) * 0.7;
        ctx.fillStyle = `rgba(255, 255, 255, ${velocityAlpha * 0.3})`;
        ctx.fill();

        // Note border
        ctx.strokeStyle = isHovered ? '#fff' : 'rgba(0, 0, 0, 0.5)';
        ctx.lineWidth = isHovered ? 2 : 1;
        ctx.stroke();

        // Resize handle
        if (isHovered || note.selected) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
          ctx.fillRect(x + width - 6, y + 1, 5, noteHeight - 2);
        }

        // Note name for wider notes
        if (width > 30) {
          ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
          ctx.font = '10px sans-serif';
          ctx.fillText(getNoteLabel(note.pitch), x + 4, y + noteHeight - 4);
        }
      });

      // Draw playhead
      if (playheadPosition >= scrollX) {
        const playheadX = beatToX(playheadPosition);
        if (playheadX >= 0 && playheadX <= canvasWidth) {
          ctx.strokeStyle = 'var(--color-error, #EF4444)';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(playheadX, 0);
          ctx.lineTo(playheadX, gridHeight);
          ctx.stroke();
        }
      }
    }, [
      canvasWidth, gridHeight, minNote, maxNote, notes, scrollX, zoom,
      quantize, noteColor, selectedColor, gridColor, hoveredNote,
      playheadPosition, beatToX, pitchToY, effectivePixelsPerBeat, noteHeight
    ]);

    // Draw velocity lane
    const drawVelocity = useCallback(() => {
      if (!showVelocity) return;

      const canvas = velocityCanvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (!canvas || !ctx) return;

      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvasWidth * dpr;
      canvas.height = velocityHeight * dpr;
      canvas.style.width = `${canvasWidth}px`;
      canvas.style.height = `${velocityHeight}px`;
      ctx.scale(dpr, dpr);

      // Background
      ctx.fillStyle = 'var(--color-surface, #1F1F1F)';
      ctx.fillRect(0, 0, canvasWidth, velocityHeight);

      // Grid lines for velocity levels
      [32, 64, 96, 127].forEach(vel => {
        const y = velocityHeight - (vel / 127) * velocityHeight;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvasWidth, y);
        ctx.stroke();
      });

      // Draw velocity bars
      notes.forEach(note => {
        const x = beatToX(note.start);
        const width = Math.max(4, note.duration * effectivePixelsPerBeat - 2);
        const barHeight = (note.velocity / 127) * (velocityHeight - 4);

        if (x + width < 0 || x > canvasWidth) return;

        const color = note.selected ? selectedColor : noteColor;
        ctx.fillStyle = color;
        ctx.fillRect(x + 1, velocityHeight - barHeight - 2, width, barHeight);
      });
    }, [showVelocity, canvasWidth, velocityHeight, notes, beatToX, effectivePixelsPerBeat, noteColor, selectedColor]);

    // Animation loop
    useEffect(() => {
      drawGrid();
      drawVelocity();
    }, [drawGrid, drawVelocity]);

    // Mouse handlers
    const handleMouseDown = useCallback((e: React.MouseEvent) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const beat = xToBeat(x);
      const pitch = yToPitch(y);

      // Check if clicking on existing note
      const clickedNote = notes.find(note => {
        const noteX = beatToX(note.start);
        const noteWidth = note.duration * effectivePixelsPerBeat;
        const noteY = pitchToY(note.pitch);
        return x >= noteX && x <= noteX + noteWidth && y >= noteY && y <= noteY + noteHeight;
      });

      if (clickedNote) {
        const noteX = beatToX(clickedNote.start);
        const noteWidth = clickedNote.duration * effectivePixelsPerBeat;
        const isResizeHandle = x > noteX + noteWidth - 10;

        // Select note
        if (!e.shiftKey) {
          onNotesChange(notes.map(n => ({
            ...n,
            selected: n.id === clickedNote.id,
          })));
        } else {
          onNotesChange(notes.map(n => ({
            ...n,
            selected: n.id === clickedNote.id ? !n.selected : n.selected,
          })));
        }

        setDragMode(isResizeHandle ? 'resize' : 'move');
        setDragStart({ x: beat, y: pitch });
        setIsDragging(true);
        onNotePreview?.(clickedNote.pitch);
      } else {
        // Create new note
        const snappedBeat = snapToGrid(beat);
        const newNote: MIDINote = {
          id: generateId(),
          pitch: Math.max(minNote, Math.min(maxNote, pitch)),
          start: snappedBeat,
          duration: 1 / quantize,
          velocity: 100,
          selected: true,
        };

        onNotesChange([
          ...notes.map(n => ({ ...n, selected: false })),
          newNote,
        ]);

        setDragMode('draw');
        setDragStart({ x: snappedBeat, y: pitch });
        setIsDragging(true);
        onNotePreview?.(pitch);
      }
    }, [notes, onNotesChange, xToBeat, yToPitch, beatToX, pitchToY, effectivePixelsPerBeat, noteHeight, snapToGrid, quantize, minNote, maxNote, onNotePreview]);

    const handleMouseMove = useCallback((e: React.MouseEvent) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const beat = xToBeat(x);
      const pitch = yToPitch(y);

      // Hover detection
      const hovered = notes.find(note => {
        const noteX = beatToX(note.start);
        const noteWidth = note.duration * effectivePixelsPerBeat;
        const noteY = pitchToY(note.pitch);
        return x >= noteX && x <= noteX + noteWidth && y >= noteY && y <= noteY + noteHeight;
      });
      setHoveredNote(hovered?.id || null);

      if (!isDragging) return;

      const selectedNotes = notes.filter(n => n.selected);
      if (selectedNotes.length === 0) return;

      if (dragMode === 'move') {
        const deltaBeat = snapToGrid(beat) - snapToGrid(dragStart.x);
        const deltaPitch = pitch - dragStart.y;

        onNotesChange(notes.map(note => {
          if (!note.selected) return note;
          return {
            ...note,
            start: Math.max(0, note.start + deltaBeat),
            pitch: Math.max(minNote, Math.min(maxNote, note.pitch + deltaPitch)),
          };
        }));

        setDragStart({ x: snapToGrid(beat), y: pitch });
      } else if (dragMode === 'resize') {
        const snappedBeat = snapToGrid(beat);

        onNotesChange(notes.map(note => {
          if (!note.selected) return note;
          const newDuration = Math.max(1 / quantize, snappedBeat - note.start);
          return { ...note, duration: newDuration };
        }));
      } else if (dragMode === 'draw') {
        const snappedBeat = snapToGrid(beat);

        onNotesChange(notes.map(note => {
          if (!note.selected) return note;
          const duration = Math.max(1 / quantize, snappedBeat - note.start);
          return { ...note, duration };
        }));
      }
    }, [notes, onNotesChange, isDragging, dragMode, dragStart, xToBeat, yToPitch, beatToX, pitchToY, effectivePixelsPerBeat, noteHeight, snapToGrid, quantize, minNote, maxNote]);

    const handleMouseUp = useCallback(() => {
      setIsDragging(false);
      setDragMode(null);
    }, []);

    // Keyboard handlers
    useEffect(() => {
      const handleKeyDown = (e: KeyboardEvent) => {
        if (e.key === 'Delete' || e.key === 'Backspace') {
          onNotesChange(notes.filter(n => !n.selected));
        } else if (e.key === 'a' && (e.ctrlKey || e.metaKey)) {
          e.preventDefault();
          onNotesChange(notes.map(n => ({ ...n, selected: true })));
        } else if (e.key === 'Escape') {
          onNotesChange(notes.map(n => ({ ...n, selected: false })));
        }
      };

      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }, [notes, onNotesChange]);

    // Expose methods via ref
    useImperativeHandle(ref, () => ({
      scrollToNote: (pitch: number) => {
        console.log('Scroll to note:', pitch);
      },
      scrollToBeat: (beat: number) => {
        console.log('Scroll to beat:', beat);
      },
      selectAll: () => {
        onNotesChange(notes.map(n => ({ ...n, selected: true })));
      },
      clearSelection: () => {
        onNotesChange(notes.map(n => ({ ...n, selected: false })));
      },
      deleteSelected: () => {
        onNotesChange(notes.filter(n => !n.selected));
      },
    }), [notes, onNotesChange]);

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
        <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          {/* Piano keyboard */}
          <div
            style={{
              width: keyboardWidth,
              background: 'var(--color-surface, #1F1F1F)',
              borderRight: '1px solid var(--color-surface-border, #333)',
              overflowY: 'auto',
            }}
          >
            {Array.from({ length: totalNotes }).map((_, i) => {
              const pitch = maxNote - i;
              const isBlack = isBlackKey(pitch);
              const isC = pitch % 12 === 0;

              return (
                <div
                  key={pitch}
                  onClick={() => onNotePreview?.(pitch)}
                  style={{
                    height: noteHeight,
                    background: isBlack ? '#1a1a1a' : '#2a2a2a',
                    borderBottom: '1px solid var(--color-surface-border, #333)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'flex-end',
                    paddingRight: 4,
                    fontSize: 9,
                    color: isC ? 'var(--color-text, #FAFAFA)' : 'var(--color-text-subtle, #6B6B6B)',
                    fontWeight: isC ? 600 : 400,
                    cursor: 'pointer',
                  }}
                >
                  {getNoteLabel(pitch)}
                </div>
              );
            })}
          </div>

          {/* Note grid */}
          <div style={{ flex: 1, overflow: 'auto' }}>
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              style={{
                display: 'block',
                cursor: isDragging
                  ? dragMode === 'resize' ? 'ew-resize' : 'move'
                  : hoveredNote ? 'pointer' : 'crosshair',
              }}
            />
          </div>
        </div>

        {/* Velocity lane */}
        {showVelocity && (
          <div style={{ display: 'flex', borderTop: '1px solid var(--color-surface-border, #333)' }}>
            <div
              style={{
                width: keyboardWidth,
                background: 'var(--color-surface, #1F1F1F)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 10,
                color: 'var(--color-text-muted, #A1A1A1)',
              }}
            >
              VEL
            </div>
            <canvas
              ref={velocityCanvasRef}
              style={{ display: 'block', flex: 1 }}
            />
          </div>
        )}
      </div>
    );
  }
));

export default PianoRoll;
