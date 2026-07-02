// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

/**
 * ParticipantsList Component
 *
 * Displays a list or grid of listening party participants with
 * their avatars, status, and activity indicators.
 */

import React from 'react';
import type { Participant } from '@/hooks/useListeningParty';

export interface ParticipantsListProps {
  /** List of participants */
  participants: Participant[];
  /** Current user's ID */
  currentUserId?: string;
  /** Whether the current user is the host */
  isHost?: boolean;
  /** Callback when host kicks a participant */
  onKick?: (participantId: string) => void;
  /** Display mode */
  layout?: 'list' | 'grid' | 'compact';
  /** Maximum visible participants before "+N more" */
  maxVisible?: number;
  /** Additional CSS class */
  className?: string;
}

export function ParticipantsList({
  participants,
  currentUserId,
  isHost = false,
  onKick,
  layout = 'grid',
  maxVisible = 8,
  className = '',
}: ParticipantsListProps) {
  const visibleParticipants = participants.slice(0, maxVisible);
  const hiddenCount = Math.max(0, participants.length - maxVisible);

  if (layout === 'compact') {
    return (
      <div className={`participants-compact flex items-center ${className}`}>
        {/* Stacked avatars */}
        <div className="flex -space-x-2">
          {visibleParticipants.slice(0, 5).map((participant, index) => (
            <div
              key={participant.id}
              className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ring-2 ring-gray-900"
              style={{
                backgroundColor: participant.color,
                zIndex: 5 - index,
              }}
              title={participant.displayName}
            >
              {participant.avatarUrl ? (
                <img
                  src={participant.avatarUrl}
                  alt={participant.displayName}
                  className="w-full h-full rounded-full object-cover"
                />
              ) : (
                participant.displayName.charAt(0).toUpperCase()
              )}
            </div>
          ))}

          {hiddenCount > 0 && (
            <div
              className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium bg-gray-700 ring-2 ring-gray-900"
              style={{ zIndex: 0 }}
            >
              +{hiddenCount}
            </div>
          )}
        </div>

        <span className="ml-3 text-sm text-gray-400">
          {participants.length} listening
        </span>
      </div>
    );
  }

  if (layout === 'list') {
    return (
      <div className={`participants-list space-y-2 ${className}`}>
        {visibleParticipants.map((participant) => (
          <ParticipantRow
            key={participant.id}
            participant={participant}
            isCurrentUser={participant.id === currentUserId}
            canKick={isHost && participant.id !== currentUserId}
            onKick={onKick}
          />
        ))}

        {hiddenCount > 0 && (
          <div className="text-sm text-gray-500 pl-11">
            +{hiddenCount} more participants
          </div>
        )}
      </div>
    );
  }

  // Grid layout (default)
  return (
    <div className={`participants-grid ${className}`}>
      <div className="grid grid-cols-4 gap-3">
        {visibleParticipants.map((participant) => (
          <ParticipantCard
            key={participant.id}
            participant={participant}
            isCurrentUser={participant.id === currentUserId}
            canKick={isHost && participant.id !== currentUserId}
            onKick={onKick}
          />
        ))}

        {hiddenCount > 0 && (
          <div className="aspect-square rounded-xl bg-gray-800/50 flex items-center justify-center">
            <span className="text-gray-400 text-sm">+{hiddenCount}</span>
          </div>
        )}
      </div>
    </div>
  );
}

// Individual participant card (grid view)
interface ParticipantCardProps {
  participant: Participant;
  isCurrentUser: boolean;
  canKick: boolean;
  onKick?: (id: string) => void;
}

function ParticipantCard({ participant, isCurrentUser, canKick, onKick }: ParticipantCardProps) {
  return (
    <div
      className={`participant-card group relative aspect-square rounded-xl p-3 flex flex-col items-center justify-center ${
        isCurrentUser ? 'ring-2 ring-indigo-500' : ''
      }`}
      style={{ backgroundColor: `${participant.color}20` }}
    >
      {/* Avatar */}
      <div
        className="w-12 h-12 rounded-full flex items-center justify-center text-lg font-bold mb-2"
        style={{ backgroundColor: participant.color }}
      >
        {participant.avatarUrl ? (
          <img
            src={participant.avatarUrl}
            alt={participant.displayName}
            className="w-full h-full rounded-full object-cover"
          />
        ) : (
          participant.displayName.charAt(0).toUpperCase()
        )}
      </div>

      {/* Name */}
      <span className="text-xs text-center text-gray-300 truncate w-full">
        {participant.displayName}
      </span>

      {/* Role badge */}
      {participant.role === 'host' && (
        <span className="absolute top-2 right-2 text-yellow-400 text-xs">
          &#9733;
        </span>
      )}

      {/* Activity indicator */}
      <div
        className={`absolute bottom-2 right-2 w-2 h-2 rounded-full ${
          participant.isActive ? 'bg-green-400' : 'bg-gray-500'
        }`}
      />

      {/* Kick button (host only) */}
      {canKick && onKick && (
        <button
          onClick={() => onKick(participant.id)}
          className="absolute top-2 left-2 w-5 h-5 rounded-full bg-red-500/80 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
          title="Remove from party"
        >
          <span className="text-xs">&times;</span>
        </button>
      )}
    </div>
  );
}

// Individual participant row (list view)
function ParticipantRow({ participant, isCurrentUser, canKick, onKick }: ParticipantCardProps) {
  return (
    <div
      className={`participant-row group flex items-center gap-3 p-2 rounded-lg hover:bg-gray-800/50 ${
        isCurrentUser ? 'bg-indigo-500/10' : ''
      }`}
    >
      {/* Avatar */}
      <div
        className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold shrink-0"
        style={{ backgroundColor: participant.color }}
      >
        {participant.avatarUrl ? (
          <img
            src={participant.avatarUrl}
            alt={participant.displayName}
            className="w-full h-full rounded-full object-cover"
          />
        ) : (
          participant.displayName.charAt(0).toUpperCase()
        )}
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-200 truncate">
            {participant.displayName}
          </span>
          {participant.role === 'host' && (
            <span className="text-yellow-400 text-xs">Host</span>
          )}
          {isCurrentUser && (
            <span className="text-gray-500 text-xs">(You)</span>
          )}
        </div>
      </div>

      {/* Activity indicator */}
      <div
        className={`w-2 h-2 rounded-full ${
          participant.isActive ? 'bg-green-400' : 'bg-gray-500'
        }`}
      />

      {/* Kick button */}
      {canKick && onKick && (
        <button
          onClick={() => onKick(participant.id)}
          className="w-6 h-6 rounded-full bg-red-500/80 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
          title="Remove from party"
        >
          <span className="text-xs">&times;</span>
        </button>
      )}
    </div>
  );
}

export default ParticipantsList;
