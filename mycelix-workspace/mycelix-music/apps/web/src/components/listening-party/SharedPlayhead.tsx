// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

/**
 * SharedPlayhead Component
 *
 * Displays a progress bar with indicators showing where each participant
 * is in the track. Useful for visualizing sync status in listening parties.
 */

import React from 'react';
import type { Participant } from '@/hooks/useListeningParty';

export interface SharedPlayheadProps {
  /** Current playback position (0-1) */
  progress: number;
  /** Total duration in seconds */
  duration: number;
  /** List of participants with their positions */
  participants: Participant[];
  /** Callback when user seeks */
  onSeek?: (position: number) => void;
  /** Whether the current user is the host */
  isHost?: boolean;
  /** Sync status */
  syncStatus?: 'synced' | 'syncing' | 'drifting';
  /** Additional CSS class */
  className?: string;
}

export function SharedPlayhead({
  progress,
  duration,
  participants,
  onSeek,
  isHost = false,
  syncStatus = 'synced',
  className = '',
}: SharedPlayheadProps) {
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!onSeek || !isHost) return;

    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const position = x / rect.width;
    onSeek(Math.max(0, Math.min(1, position)));
  };

  const currentTime = progress * duration;

  return (
    <div className={`shared-playhead ${className}`}>
      {/* Time display */}
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{formatTime(currentTime)}</span>
        <span className="flex items-center gap-2">
          {syncStatus === 'syncing' && (
            <span className="text-yellow-400 animate-pulse">Syncing...</span>
          )}
          {syncStatus === 'drifting' && (
            <span className="text-red-400">Out of sync</span>
          )}
          <span>{formatTime(duration)}</span>
        </span>
      </div>

      {/* Progress bar container */}
      <div
        className={`relative h-3 bg-gray-800 rounded-full overflow-hidden ${
          isHost ? 'cursor-pointer' : 'cursor-default'
        }`}
        onClick={handleClick}
      >
        {/* Main progress */}
        <div
          className="absolute h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-100"
          style={{ width: `${progress * 100}%` }}
        />

        {/* Participant indicators */}
        {participants.map((participant) => {
          const participantProgress = participant.position / duration;
          const isLocal = participant.id.startsWith('local');

          return (
            <div
              key={participant.id}
              className="absolute top-0 bottom-0 transition-all duration-200"
              style={{ left: `${participantProgress * 100}%` }}
            >
              {/* Position indicator */}
              <div
                className={`w-1 h-full ${isLocal ? 'bg-white' : ''}`}
                style={{ backgroundColor: isLocal ? undefined : participant.color }}
              />

              {/* Participant avatar/initial */}
              <div
                className={`absolute -top-6 left-1/2 -translate-x-1/2 w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
                  isLocal ? 'ring-2 ring-white' : ''
                }`}
                style={{ backgroundColor: participant.color }}
                title={participant.displayName}
              >
                {participant.displayName.charAt(0).toUpperCase()}
              </div>
            </div>
          );
        })}

        {/* Hover effect */}
        {isHost && (
          <div className="absolute inset-0 bg-white/5 opacity-0 hover:opacity-100 transition-opacity" />
        )}
      </div>

      {/* Participant legend */}
      <div className="flex flex-wrap gap-3 mt-2">
        {participants.map((participant) => (
          <div key={participant.id} className="flex items-center gap-1.5 text-xs">
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: participant.color }}
            />
            <span className={`${participant.role === 'host' ? 'font-medium' : 'text-gray-400'}`}>
              {participant.displayName}
              {participant.role === 'host' && (
                <span className="ml-1 text-yellow-400">&#9733;</span>
              )}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default SharedPlayhead;
