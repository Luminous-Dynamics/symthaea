// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

/**
 * ListeningParty Component
 *
 * Main container for listening party features.
 * Combines shared playhead, participants, and reactions.
 */

import React, { useState } from 'react';
import { useListeningParty } from '@/hooks/useListeningParty';
import { usePlayerStore } from '@/store/playerStore';
import { SharedPlayhead } from './SharedPlayhead';
import { ParticipantsList } from './ParticipantsList';
import { ReactionDisplay, QuickReactionBar } from './ReactionDisplay';

export interface ListeningPartyProps {
  /** Party ID to join (if joining existing) */
  partyId?: string;
  /** Callback when party is closed */
  onClose?: () => void;
  /** Additional CSS class */
  className?: string;
}

export function ListeningParty({ partyId: initialPartyId, onClose, className = '' }: ListeningPartyProps) {
  const {
    partyId,
    isHost,
    isConnected,
    participants,
    settings,
    reactions,
    syncStatus,
    error,
    createParty,
    joinParty,
    leaveParty,
    pauseForAll,
    playForAll,
    skipTrack,
    kickParticipant,
    sendReaction,
    getShareLink,
  } = useListeningParty();

  const { currentSong, isPlaying, position, duration } = usePlayerStore();
  const [showInvite, setShowInvite] = useState(false);
  const [joinName, setJoinName] = useState('');

  // If no party, show create/join options
  if (!partyId) {
    return (
      <div className={`listening-party-setup p-6 ${className}`}>
        <h2 className="text-xl font-bold mb-4">Start a Listening Party</h2>
        <p className="text-gray-400 mb-6">
          Listen together with friends in real-time sync.
        </p>

        <div className="space-y-4">
          <button
            onClick={() => createParty()}
            className="w-full py-3 px-4 bg-indigo-600 hover:bg-indigo-500 rounded-xl font-medium transition-colors"
          >
            Create New Party
          </button>

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-700" />
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-gray-900 text-gray-500">or join existing</span>
            </div>
          </div>

          <div className="flex gap-2">
            <input
              type="text"
              placeholder="Your name"
              value={joinName}
              onChange={(e) => setJoinName(e.target.value)}
              className="flex-1 px-4 py-3 bg-gray-800 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            <input
              type="text"
              placeholder="Party code"
              className="flex-1 px-4 py-3 bg-gray-800 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          <button
            onClick={() => initialPartyId && joinParty(initialPartyId, joinName || 'Guest')}
            disabled={!joinName}
            className="w-full py-3 px-4 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl font-medium transition-colors"
          >
            Join Party
          </button>
        </div>
      </div>
    );
  }

  // Active party view
  return (
    <div className={`listening-party relative flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="party-header p-4 border-b border-gray-800">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold flex items-center gap-2">
              {settings?.title || 'Listening Party'}
              {isHost && <span className="text-yellow-400 text-sm">&#9733; Host</span>}
            </h2>
            <p className="text-sm text-gray-400">
              {participants.length} listening together
            </p>
          </div>

          <div className="flex items-center gap-2">
            {/* Invite button */}
            <button
              onClick={() => setShowInvite(!showInvite)}
              className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm transition-colors"
            >
              Invite
            </button>

            {/* Leave/End button */}
            <button
              onClick={() => {
                leaveParty();
                onClose?.();
              }}
              className="px-4 py-2 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded-lg text-sm transition-colors"
            >
              {isHost ? 'End Party' : 'Leave'}
            </button>
          </div>
        </div>

        {/* Invite panel */}
        {showInvite && (
          <div className="mt-4 p-4 bg-gray-800/50 rounded-xl">
            <p className="text-sm text-gray-400 mb-2">Share this link to invite friends:</p>
            <div className="flex gap-2">
              <input
                type="text"
                readOnly
                value={getShareLink() || ''}
                className="flex-1 px-3 py-2 bg-gray-900 rounded-lg text-sm text-gray-300"
              />
              <button
                onClick={() => {
                  const link = getShareLink();
                  if (link) navigator.clipboard.writeText(link);
                }}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm transition-colors"
              >
                Copy
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Current track info */}
      {currentSong && (
        <div className="track-info p-4 flex items-center gap-4 border-b border-gray-800">
          <img
            src={currentSong.coverArt}
            alt={currentSong.title}
            className="w-16 h-16 rounded-lg object-cover"
          />
          <div className="flex-1 min-w-0">
            <h3 className="font-medium truncate">{currentSong.title}</h3>
            <p className="text-sm text-gray-400 truncate">{currentSong.artist}</p>
          </div>
        </div>
      )}

      {/* Shared playhead */}
      <div className="playhead-section p-4">
        <SharedPlayhead
          progress={duration > 0 ? position / duration : 0}
          duration={duration}
          participants={participants}
          isHost={isHost}
          syncStatus={syncStatus}
          onSeek={isHost ? (pos) => usePlayerStore.getState().seek(pos * duration) : undefined}
        />

        {/* Host controls */}
        {isHost && (
          <div className="host-controls flex items-center justify-center gap-3 mt-4">
            <button
              onClick={isPlaying ? pauseForAll : playForAll}
              className="w-12 h-12 rounded-full bg-indigo-600 hover:bg-indigo-500 flex items-center justify-center transition-colors"
            >
              {isPlaying ? (
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z"
                    clipRule="evenodd"
                  />
                </svg>
              ) : (
                <svg className="w-5 h-5 ml-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
                    clipRule="evenodd"
                  />
                </svg>
              )}
            </button>
            <button
              onClick={skipTrack}
              className="w-10 h-10 rounded-full bg-gray-700 hover:bg-gray-600 flex items-center justify-center transition-colors"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M4.555 5.168A1 1 0 003 6v8a1 1 0 001.555.832L10 11.202V14a1 1 0 001.555.832l6-4a1 1 0 000-1.664l-6-4A1 1 0 0010 6v2.798l-5.445-3.63z" />
              </svg>
            </button>
          </div>
        )}
      </div>

      {/* Participants */}
      <div className="participants-section p-4 flex-1 overflow-auto">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Participants</h3>
        <ParticipantsList
          participants={participants}
          isHost={isHost}
          onKick={isHost ? kickParticipant : undefined}
          layout="grid"
        />
      </div>

      {/* Quick reactions */}
      <div className="reactions-bar p-4 border-t border-gray-800">
        <QuickReactionBar
          onReact={sendReaction}
          enabled={settings?.allowReactions}
        />
      </div>

      {/* Floating reactions overlay */}
      <ReactionDisplay
        reactions={reactions}
        onReact={sendReaction}
        enabled={settings?.allowReactions}
        className="absolute inset-0 pointer-events-none"
      />

      {/* Error display */}
      {error && (
        <div className="absolute bottom-20 left-4 right-4 p-3 bg-red-600/20 border border-red-600/50 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}
    </div>
  );
}

export default ListeningParty;
