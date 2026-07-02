// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useRef } from 'react';
import {
  Volume2,
  VolumeX,
  Lock,
  Unlock,
  Trash2,
  MoreHorizontal,
  Headphones,
} from 'lucide-react';

interface Stem {
  id: string;
  name: string;
  type: string;
  color: string;
  audioUrl: string;
  waveformData?: number[];
  volume: number;
  pan: number;
  muted: boolean;
  solo: boolean;
  startTime: number;
  duration: number;
  lockedBy?: string;
}

interface Participant {
  id: string;
  name: string;
  color: string;
}

interface StemTrackProps {
  stem: Stem;
  zoom: number;
  totalDuration: number;
  participants: Participant[];
  onUpdate: (updates: Partial<Stem>) => void;
  onLock: () => void;
  onUnlock: () => void;
  onDelete: () => void;
}

const STEM_COLORS: Record<string, string> = {
  VOCALS: '#EC4899',
  DRUMS: '#F59E0B',
  BASS: '#8B5CF6',
  GUITAR: '#10B981',
  KEYS: '#3B82F6',
  SYNTH: '#06B6D4',
  STRINGS: '#EF4444',
  OTHER: '#6B7280',
};

export function StemTrack({
  stem,
  zoom,
  totalDuration,
  participants,
  onUpdate,
  onLock,
  onUnlock,
  onDelete,
}: StemTrackProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [showMenu, setShowMenu] = useState(false);
  const trackRef = useRef<HTMLDivElement>(null);

  const color = stem.color || STEM_COLORS[stem.type] || STEM_COLORS.OTHER;
  const lockedByUser = stem.lockedBy ? participants.find(p => p.id === stem.lockedBy) : null;
  const isLocked = !!stem.lockedBy;

  // Handle volume change
  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onUpdate({ volume: parseFloat(e.target.value) });
  };

  // Handle pan change
  const handlePanChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onUpdate({ pan: parseFloat(e.target.value) });
  };

  // Toggle mute
  const toggleMute = () => {
    onUpdate({ muted: !stem.muted });
  };

  // Toggle solo
  const toggleSolo = () => {
    onUpdate({ solo: !stem.solo });
  };

  // Handle drag to reposition
  const handleDragStart = (e: React.MouseEvent) => {
    if (isLocked) return;
    setIsDragging(true);
    onLock();
  };

  const handleDragEnd = () => {
    setIsDragging(false);
    onUnlock();
  };

  // Generate waveform visualization
  const renderWaveform = () => {
    const waveform = stem.waveformData || generateMockWaveform(100);
    const width = stem.duration * 50 * zoom;

    return (
      <svg
        className="absolute inset-0 w-full h-full"
        viewBox={`0 0 ${waveform.length} 100`}
        preserveAspectRatio="none"
      >
        {waveform.map((value, i) => (
          <rect
            key={i}
            x={i}
            y={50 - value / 2}
            width="0.8"
            height={value}
            fill={stem.muted ? '#4B5563' : color}
            opacity={stem.muted ? 0.3 : 0.8}
          />
        ))}
      </svg>
    );
  };

  return (
    <div className="flex h-20 border-b border-white/10 group">
      {/* Track Controls */}
      <div className="w-[200px] flex-shrink-0 bg-gray-900 border-r border-white/10 p-2 flex flex-col justify-between">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: color }}
            />
            <span className="text-sm font-medium truncate">{stem.name}</span>
          </div>

          <div className="flex items-center gap-1">
            {isLocked && lockedByUser && (
              <div
                className="w-5 h-5 rounded-full flex items-center justify-center text-xs"
                style={{ backgroundColor: lockedByUser.color }}
                title={`Locked by ${lockedByUser.name}`}
              >
                <Lock className="w-3 h-3" />
              </div>
            )}

            <button
              onClick={() => setShowMenu(!showMenu)}
              className="p-1 rounded hover:bg-white/10 opacity-0 group-hover:opacity-100"
            >
              <MoreHorizontal className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Mute */}
          <button
            onClick={toggleMute}
            className={`p-1 rounded text-xs ${stem.muted ? 'bg-red-500/20 text-red-400' : 'hover:bg-white/10'}`}
            title="Mute"
          >
            {stem.muted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
          </button>

          {/* Solo */}
          <button
            onClick={toggleSolo}
            className={`p-1 rounded text-xs ${stem.solo ? 'bg-yellow-500/20 text-yellow-400' : 'hover:bg-white/10'}`}
            title="Solo"
          >
            <Headphones className="w-4 h-4" />
          </button>

          {/* Volume */}
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={stem.volume}
            onChange={handleVolumeChange}
            className="flex-1 h-1 bg-white/20 rounded-full appearance-none cursor-pointer"
            title={`Volume: ${Math.round(stem.volume * 100)}%`}
          />

          {/* Pan */}
          <input
            type="range"
            min="-1"
            max="1"
            step="0.1"
            value={stem.pan}
            onChange={handlePanChange}
            className="w-12 h-1 bg-white/20 rounded-full appearance-none cursor-pointer"
            title={`Pan: ${stem.pan > 0 ? 'R' : stem.pan < 0 ? 'L' : 'C'} ${Math.abs(Math.round(stem.pan * 100))}%`}
          />
        </div>

        {/* Context Menu */}
        {showMenu && (
          <div className="absolute left-[180px] top-0 z-50 bg-gray-800 rounded-lg shadow-xl border border-white/10 py-1 min-w-[150px]">
            <button
              onClick={() => { onLock(); setShowMenu(false); }}
              className="w-full px-3 py-1.5 text-left text-sm hover:bg-white/10 flex items-center gap-2"
            >
              <Lock className="w-4 h-4" />
              Lock for editing
            </button>
            <button
              onClick={() => { onDelete(); setShowMenu(false); }}
              className="w-full px-3 py-1.5 text-left text-sm hover:bg-white/10 flex items-center gap-2 text-red-400"
            >
              <Trash2 className="w-4 h-4" />
              Delete stem
            </button>
          </div>
        )}
      </div>

      {/* Waveform Area */}
      <div
        ref={trackRef}
        className="flex-1 relative overflow-hidden"
        style={{ width: `${totalDuration * 50 * zoom}px` }}
      >
        {/* Stem block */}
        <div
          className={`absolute top-1 bottom-1 rounded-lg overflow-hidden cursor-move transition-opacity ${
            isDragging ? 'opacity-70' : ''
          } ${isLocked ? 'cursor-not-allowed' : ''}`}
          style={{
            left: `${stem.startTime * 50 * zoom}px`,
            width: `${stem.duration * 50 * zoom}px`,
            backgroundColor: `${color}20`,
            border: `1px solid ${color}40`,
          }}
          onMouseDown={handleDragStart}
          onMouseUp={handleDragEnd}
          onMouseLeave={handleDragEnd}
        >
          {renderWaveform()}

          {/* Stem label */}
          <div className="absolute top-1 left-2 text-xs font-medium opacity-70">
            {stem.name}
          </div>
        </div>
      </div>
    </div>
  );
}

// Generate mock waveform data
function generateMockWaveform(points: number): number[] {
  return Array.from({ length: points }, () => 20 + Math.random() * 60);
}
