// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { usePlayerStore, QueueItem } from '@/store/playerStore';
import { formatDuration } from '@/lib/utils';
import { useState } from 'react';
import { X, GripVertical, Music } from 'lucide-react';

export function QueuePanel() {
  const {
    queue,
    queueIndex,
    currentSong,
    history,
    queueVisible,
    playFromQueue,
    removeFromQueue,
    clearQueue,
    reorderQueue,
    toggleQueue,
  } = usePlayerStore();

  const [draggedItem, setDraggedItem] = useState<string | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);

  if (!queueVisible) return null;

  const upNext = queue.slice(queueIndex + 1);

  const handleDragStart = (e: React.DragEvent, queueId: string) => {
    setDraggedItem(queueId);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    setDragOverIndex(index);
  };

  const handleDrop = (e: React.DragEvent, toIndex: number) => {
    e.preventDefault();
    if (!draggedItem) return;

    const fromIndex = queue.findIndex((t) => t.queueId === draggedItem);
    if (fromIndex !== -1 && fromIndex !== toIndex) {
      reorderQueue(fromIndex, toIndex);
    }

    setDraggedItem(null);
    setDragOverIndex(null);
  };

  return (
    <div className="fixed right-0 top-0 bottom-20 w-96 bg-gray-900/95 backdrop-blur-md border-l border-white/10 z-40 overflow-hidden flex flex-col shadow-2xl">
      {/* Header */}
      <div className="p-4 border-b border-white/10 flex items-center justify-between">
        <h3 className="font-semibold text-lg">Queue</h3>
        <div className="flex items-center gap-2">
          {queue.length > 0 && (
            <button
              onClick={clearQueue}
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              Clear
            </button>
          )}
          <button
            onClick={toggleQueue}
            className="p-1 text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Now Playing */}
        {currentSong && (
          <div className="p-4 border-b border-white/10">
            <h4 className="text-sm text-gray-400 mb-3">Now Playing</h4>
            <QueueTrackItem
              track={currentSong as QueueItem}
              isPlaying
              onPlay={() => {}}
              onRemove={() => {}}
            />
          </div>
        )}

        {/* Up Next */}
        {upNext.length > 0 && (
          <div className="p-4 border-b border-white/10">
            <h4 className="text-sm text-gray-400 mb-3">
              Up Next ({upNext.length})
            </h4>
            <div className="space-y-1">
              {upNext.map((track, i) => (
                <div
                  key={track.queueId}
                  draggable
                  onDragStart={(e) => handleDragStart(e, track.queueId)}
                  onDragOver={(e) => handleDragOver(e, queueIndex + 1 + i)}
                  onDrop={(e) => handleDrop(e, queueIndex + 1 + i)}
                  className={`${
                    dragOverIndex === queueIndex + 1 + i
                      ? 'border-t-2 border-purple-500'
                      : ''
                  }`}
                >
                  <QueueTrackItem
                    track={track}
                    onPlay={() => playFromQueue(queueIndex + 1 + i)}
                    onRemove={() => removeFromQueue(track.queueId)}
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* History */}
        {history.length > 0 && (
          <div className="p-4">
            <h4 className="text-sm text-gray-400 mb-3">Recently Played</h4>
            <div className="space-y-1 opacity-60">
              {history.slice(-10).reverse().map((track, index) => (
                <QueueTrackItem
                  key={`history-${track.id}-${index}`}
                  track={track as QueueItem}
                  onPlay={() => {}}
                  onRemove={() => {}}
                  isHistory
                />
              ))}
            </div>
          </div>
        )}

        {/* Empty State */}
        {upNext.length === 0 && history.length === 0 && !currentSong && (
          <div className="p-8 text-center text-gray-500">
            <Music className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p className="font-medium">Your queue is empty</p>
            <p className="text-sm mt-2">Add songs to start playing</p>
          </div>
        )}
      </div>
    </div>
  );
}

function QueueTrackItem({
  track,
  isPlaying = false,
  isHistory = false,
  onPlay,
  onRemove,
}: {
  track: QueueItem;
  isPlaying?: boolean;
  isHistory?: boolean;
  onPlay: () => void;
  onRemove: () => void;
}) {
  return (
    <div
      className={`flex items-center gap-3 p-2 rounded-lg group cursor-pointer transition-colors ${
        isPlaying ? 'bg-purple-500/20' : 'hover:bg-white/5'
      }`}
      onClick={onPlay}
    >
      {/* Cover Art */}
      <div className="relative flex-shrink-0">
        <img
          src={track.coverArt || '/placeholder-album.png'}
          alt={track.title}
          className="w-10 h-10 rounded object-cover"
        />
        {isPlaying && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded">
            <div className="flex gap-0.5 items-end h-4">
              <div className="w-1 bg-purple-400 animate-[pulse_0.8s_ease-in-out_infinite]" style={{ height: '60%' }} />
              <div className="w-1 bg-purple-400 animate-[pulse_0.8s_ease-in-out_infinite_0.2s]" style={{ height: '100%' }} />
              <div className="w-1 bg-purple-400 animate-[pulse_0.8s_ease-in-out_infinite_0.4s]" style={{ height: '40%' }} />
            </div>
          </div>
        )}
      </div>

      {/* Track Info */}
      <div className="flex-1 min-w-0">
        <p className={`text-sm truncate ${isPlaying ? 'text-purple-400 font-medium' : ''}`}>
          {track.title}
        </p>
        <p className="text-xs text-gray-400 truncate">{track.artist}</p>
      </div>

      {/* Duration */}
      <span className="text-xs text-gray-500 tabular-nums">
        {formatDuration(track.duration)}
      </span>

      {/* Actions */}
      {!isHistory && !isPlaying && (
        <>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onRemove();
            }}
            className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-400 transition-all"
            title="Remove from queue"
          >
            <X className="w-4 h-4" />
          </button>
          <div
            className="opacity-0 group-hover:opacity-100 cursor-grab text-gray-500"
            title="Drag to reorder"
          >
            <GripVertical className="w-4 h-4" />
          </div>
        </>
      )}
    </div>
  );
}
