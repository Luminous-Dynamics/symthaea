// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  useRemixStudio,
  RemixTrack,
  Clip,
  Collaborator,
  Comment,
} from '@/hooks/useRemixStudio';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Plus,
  Trash2,
  Volume2,
  VolumeX,
  Lock,
  Unlock,
  Users,
  MessageSquare,
  GitBranch,
  Save,
  Download,
  ZoomIn,
  ZoomOut,
  Copy,
  Scissors,
  Undo,
  Redo,
  Music,
  Mic,
  Drum,
  Waves,
} from 'lucide-react';

interface RemixStudioProps {
  projectId?: string;
  className?: string;
}

const TRACK_HEIGHT = 80;
const HEADER_WIDTH = 200;
const BEATS_PER_PIXEL = 0.1;

export function RemixStudio({ projectId, className = '' }: RemixStudioProps) {
  const {
    project,
    collaborators,
    currentUser,
    comments,
    isConnected,
    playbackPosition,
    isPlaying,
    selectedTracks,
    selectedClips,
    error,
    createProject,
    addTrack,
    removeTrack,
    updateTrack,
    lockTrack,
    unlockTrack,
    addClip,
    removeClip,
    moveClip,
    updateCursor,
    createVersion,
    addComment,
    setPlaybackPosition,
    togglePlayback,
    selectTracks,
    exportProject,
  } = useRemixStudio(projectId);

  const [zoom, setZoom] = useState(1);
  const [showComments, setShowComments] = useState(false);
  const [showVersions, setShowVersions] = useState(false);
  const [showCollaborators, setShowCollaborators] = useState(false);
  const [newComment, setNewComment] = useState('');
  const [draggedClip, setDraggedClip] = useState<{ trackId: string; clipId: string; offset: number } | null>(null);

  const timelineRef = useRef<HTMLDivElement>(null);
  const playheadRef = useRef<HTMLDivElement>(null);

  // Initialize with demo project if none provided
  useEffect(() => {
    if (!project && !projectId) {
      createProject('New Remix', 120);
    }
  }, [project, projectId, createProject]);

  // Animate playhead
  useEffect(() => {
    if (!isPlaying || !project) return;

    const startTime = Date.now();
    const startPosition = playbackPosition;
    const beatsPerSecond = project.tempo / 60;

    const animate = () => {
      const elapsed = (Date.now() - startTime) / 1000;
      const newPosition = startPosition + elapsed * beatsPerSecond;

      if (newPosition >= project.duration) {
        setPlaybackPosition(0);
        togglePlayback();
        return;
      }

      setPlaybackPosition(newPosition);
      if (isPlaying) {
        requestAnimationFrame(animate);
      }
    };

    const frameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameId);
  }, [isPlaying, project, playbackPosition, setPlaybackPosition, togglePlayback]);

  const handleTimelineClick = (e: React.MouseEvent) => {
    if (!timelineRef.current) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left - HEADER_WIDTH;
    const beat = x * BEATS_PER_PIXEL / zoom;
    setPlaybackPosition(Math.max(0, beat));
  };

  const handleAddTrack = (type: RemixTrack['type']) => {
    const names: Record<string, string> = {
      audio: 'Audio Track',
      stem: 'Stem Track',
      midi: 'MIDI Track',
      generated: 'AI Track',
    };
    addTrack({ type, name: names[type] });
  };

  const handleTrackVolumeChange = (trackId: string, volume: number) => {
    updateTrack(trackId, { volume });
  };

  const handleTrackMute = (trackId: string) => {
    const track = project?.tracks.find(t => t.id === trackId);
    if (track) {
      updateTrack(trackId, { muted: !track.muted });
    }
  };

  const handleTrackSolo = (trackId: string) => {
    const track = project?.tracks.find(t => t.id === trackId);
    if (track) {
      updateTrack(trackId, { solo: !track.solo });
    }
  };

  const handleClipDragStart = (e: React.DragEvent, trackId: string, clipId: string) => {
    const clip = project?.tracks.find(t => t.id === trackId)?.clips.find(c => c.id === clipId);
    if (!clip) return;

    const rect = (e.target as HTMLElement).getBoundingClientRect();
    const offset = (e.clientX - rect.left) * BEATS_PER_PIXEL / zoom;
    setDraggedClip({ trackId, clipId, offset });
  };

  const handleClipDrop = (e: React.DragEvent, targetTrackId: string) => {
    if (!draggedClip || !timelineRef.current) return;

    const rect = timelineRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left - HEADER_WIDTH;
    const newStart = Math.max(0, (x * BEATS_PER_PIXEL / zoom) - draggedClip.offset);

    moveClip(targetTrackId, draggedClip.clipId, newStart);
    setDraggedClip(null);
  };

  const handleSaveVersion = () => {
    const name = prompt('Version name:');
    if (name) {
      createVersion(name);
    }
  };

  const handleAddComment = () => {
    if (newComment.trim()) {
      addComment(newComment, playbackPosition);
      setNewComment('');
    }
  };

  const beatsToPixels = (beats: number) => beats / BEATS_PER_PIXEL * zoom;
  const pixelsToBeats = (pixels: number) => pixels * BEATS_PER_PIXEL / zoom;

  if (!project) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-900 text-gray-400">
        Loading project...
      </div>
    );
  }

  return (
    <div className={`flex flex-col h-full bg-gray-950 ${className}`}>
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-900 border-b border-gray-800">
        <div className="flex items-center gap-4">
          {/* Transport Controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPlaybackPosition(0)}
              className="p-2 text-gray-400 hover:text-white transition-colors"
            >
              <SkipBack className="w-4 h-4" />
            </button>
            <button
              onClick={togglePlayback}
              className="p-3 bg-purple-500 text-white rounded-full hover:bg-purple-600 transition-colors"
            >
              {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            </button>
            <button
              onClick={() => setPlaybackPosition(project.duration)}
              className="p-2 text-gray-400 hover:text-white transition-colors"
            >
              <SkipForward className="w-4 h-4" />
            </button>
          </div>

          {/* Time Display */}
          <div className="text-white font-mono">
            {Math.floor(playbackPosition / 4) + 1}:{Math.floor(playbackPosition % 4) + 1}
            <span className="text-gray-500 ml-2">{project.tempo} BPM</span>
          </div>

          {/* Zoom */}
          <div className="flex items-center gap-1 border-l border-gray-800 pl-4">
            <button
              onClick={() => setZoom(z => Math.max(0.5, z - 0.25))}
              className="p-1.5 text-gray-400 hover:text-white transition-colors"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <span className="text-sm text-gray-400 w-12 text-center">{Math.round(zoom * 100)}%</span>
            <button
              onClick={() => setZoom(z => Math.min(4, z + 0.25))}
              className="p-1.5 text-gray-400 hover:text-white transition-colors"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Collaborators */}
          <button
            onClick={() => setShowCollaborators(!showCollaborators)}
            className={`relative p-2 rounded-lg transition-colors ${
              showCollaborators ? 'bg-purple-500 text-white' : 'text-gray-400 hover:text-white'
            }`}
          >
            <Users className="w-5 h-5" />
            {collaborators.length > 1 && (
              <span className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 text-white text-xs rounded-full flex items-center justify-center">
                {collaborators.length}
              </span>
            )}
          </button>

          {/* Comments */}
          <button
            onClick={() => setShowComments(!showComments)}
            className={`relative p-2 rounded-lg transition-colors ${
              showComments ? 'bg-purple-500 text-white' : 'text-gray-400 hover:text-white'
            }`}
          >
            <MessageSquare className="w-5 h-5" />
            {comments.length > 0 && (
              <span className="absolute -top-1 -right-1 w-4 h-4 bg-purple-500 text-white text-xs rounded-full flex items-center justify-center">
                {comments.length}
              </span>
            )}
          </button>

          {/* Versions */}
          <button
            onClick={() => setShowVersions(!showVersions)}
            className={`p-2 rounded-lg transition-colors ${
              showVersions ? 'bg-purple-500 text-white' : 'text-gray-400 hover:text-white'
            }`}
          >
            <GitBranch className="w-5 h-5" />
          </button>

          {/* Save Version */}
          <button
            onClick={handleSaveVersion}
            className="p-2 text-gray-400 hover:text-white transition-colors"
          >
            <Save className="w-5 h-5" />
          </button>

          {/* Export */}
          <button
            onClick={exportProject}
            className="flex items-center gap-2 px-3 py-1.5 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors"
          >
            <Download className="w-4 h-4" />
            Export
          </button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Main Timeline */}
        <div className="flex-1 overflow-auto" ref={timelineRef}>
          {/* Timeline Header */}
          <div className="sticky top-0 z-10 flex bg-gray-900 border-b border-gray-800">
            <div className="w-[200px] shrink-0 border-r border-gray-800" />
            <div
              className="relative h-8 bg-gray-900"
              style={{ width: beatsToPixels(project.duration) }}
              onClick={handleTimelineClick}
            >
              {/* Beat markers */}
              {Array.from({ length: Math.ceil(project.duration / 4) }, (_, i) => (
                <div
                  key={i}
                  className="absolute top-0 h-full border-l border-gray-700"
                  style={{ left: beatsToPixels(i * 4) }}
                >
                  <span className="text-xs text-gray-500 ml-1">{i + 1}</span>
                </div>
              ))}

              {/* Playhead */}
              <div
                ref={playheadRef}
                className="absolute top-0 w-0.5 h-full bg-purple-500 z-20"
                style={{ left: beatsToPixels(playbackPosition) }}
              >
                <div className="absolute -top-1 -left-1.5 w-3 h-3 bg-purple-500 rotate-45" />
              </div>

              {/* Collaborator cursors */}
              {collaborators
                .filter(c => c.id !== currentUser?.id && c.cursor)
                .map(collaborator => (
                  <div
                    key={collaborator.id}
                    className="absolute top-0 w-0.5 h-full opacity-50"
                    style={{
                      left: beatsToPixels(collaborator.cursor!.time),
                      backgroundColor: collaborator.color,
                    }}
                  >
                    <span
                      className="absolute -top-5 -left-3 text-xs px-1 rounded"
                      style={{ backgroundColor: collaborator.color }}
                    >
                      {collaborator.name}
                    </span>
                  </div>
                ))}
            </div>
          </div>

          {/* Tracks */}
          <div className="relative">
            {project.tracks.map((track, trackIndex) => (
              <TrackRow
                key={track.id}
                track={track}
                trackIndex={trackIndex}
                zoom={zoom}
                projectDuration={project.duration}
                currentUser={currentUser}
                collaborators={collaborators}
                isSelected={selectedTracks.includes(track.id)}
                selectedClips={selectedClips}
                onVolumeChange={(v) => handleTrackVolumeChange(track.id, v)}
                onMute={() => handleTrackMute(track.id)}
                onSolo={() => handleTrackSolo(track.id)}
                onLock={() => lockTrack(track.id)}
                onUnlock={() => unlockTrack(track.id)}
                onDelete={() => removeTrack(track.id)}
                onClipDragStart={handleClipDragStart}
                onClipDrop={handleClipDrop}
                beatsToPixels={beatsToPixels}
              />
            ))}

            {/* Add Track Button */}
            <div className="flex items-center h-12 border-b border-gray-800">
              <div className="w-[200px] shrink-0 px-4 border-r border-gray-800">
                <div className="flex gap-2">
                  <button
                    onClick={() => handleAddTrack('audio')}
                    className="flex items-center gap-1 px-2 py-1 text-sm text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
                  >
                    <Plus className="w-3 h-3" /> Audio
                  </button>
                  <button
                    onClick={() => handleAddTrack('stem')}
                    className="flex items-center gap-1 px-2 py-1 text-sm text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
                  >
                    <Plus className="w-3 h-3" /> Stem
                  </button>
                </div>
              </div>
            </div>

            {/* Playhead line extending through tracks */}
            <div
              className="absolute top-0 w-0.5 bg-purple-500 pointer-events-none z-10"
              style={{
                left: HEADER_WIDTH + beatsToPixels(playbackPosition),
                height: `${project.tracks.length * TRACK_HEIGHT + 48}px`,
              }}
            />
          </div>
        </div>

        {/* Side Panels */}
        {showComments && (
          <div className="w-80 bg-gray-900 border-l border-gray-800 flex flex-col">
            <div className="p-4 border-b border-gray-800">
              <h3 className="font-semibold text-white">Comments</h3>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {comments.map(comment => (
                <div key={comment.id} className="bg-gray-800 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-sm font-medium text-white">{comment.userName}</span>
                    {comment.timePosition !== undefined && (
                      <span className="text-xs text-purple-400">
                        @ {Math.floor(comment.timePosition / 4) + 1}:{Math.floor(comment.timePosition % 4) + 1}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-300">{comment.content}</p>
                </div>
              ))}
            </div>
            <div className="p-4 border-t border-gray-800">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={newComment}
                  onChange={(e) => setNewComment(e.target.value)}
                  placeholder="Add a comment..."
                  className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-purple-500"
                  onKeyPress={(e) => e.key === 'Enter' && handleAddComment()}
                />
                <button
                  onClick={handleAddComment}
                  className="px-3 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600"
                >
                  Send
                </button>
              </div>
            </div>
          </div>
        )}

        {showCollaborators && (
          <div className="w-64 bg-gray-900 border-l border-gray-800">
            <div className="p-4 border-b border-gray-800">
              <h3 className="font-semibold text-white">Collaborators</h3>
            </div>
            <div className="p-4 space-y-2">
              {collaborators.map(collaborator => (
                <div
                  key={collaborator.id}
                  className="flex items-center gap-3 p-2 rounded-lg hover:bg-gray-800"
                >
                  <div
                    className="w-8 h-8 rounded-full flex items-center justify-center text-white font-medium"
                    style={{ backgroundColor: collaborator.color }}
                  >
                    {collaborator.name[0].toUpperCase()}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm text-white">
                      {collaborator.name}
                      {collaborator.isOwner && (
                        <span className="text-xs text-purple-400 ml-1">(Owner)</span>
                      )}
                    </p>
                    <p className="text-xs text-gray-500">
                      {collaborator.id === currentUser?.id ? 'You' : 'Active'}
                    </p>
                  </div>
                  <div className="w-2 h-2 bg-green-500 rounded-full" />
                </div>
              ))}
            </div>
          </div>
        )}

        {showVersions && (
          <div className="w-72 bg-gray-900 border-l border-gray-800">
            <div className="p-4 border-b border-gray-800">
              <h3 className="font-semibold text-white">Version History</h3>
            </div>
            <div className="p-4 space-y-2">
              <div className="p-3 bg-purple-500/20 border border-purple-500/50 rounded-lg">
                <p className="text-sm font-medium text-white">{project.version.name}</p>
                <p className="text-xs text-gray-400">Current version</p>
              </div>
              {project.history.map(version => (
                <div
                  key={version.id}
                  className="p-3 bg-gray-800 rounded-lg hover:bg-gray-700 cursor-pointer"
                >
                  <p className="text-sm font-medium text-white">{version.name}</p>
                  <p className="text-xs text-gray-500">
                    {new Date(version.createdAt).toLocaleString()}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="absolute bottom-4 right-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400">
          {error}
        </div>
      )}
    </div>
  );
}

// Track Row Component
interface TrackRowProps {
  track: RemixTrack;
  trackIndex: number;
  zoom: number;
  projectDuration: number;
  currentUser: Collaborator | null;
  collaborators: Collaborator[];
  isSelected: boolean;
  selectedClips: string[];
  onVolumeChange: (volume: number) => void;
  onMute: () => void;
  onSolo: () => void;
  onLock: () => void;
  onUnlock: () => void;
  onDelete: () => void;
  onClipDragStart: (e: React.DragEvent, trackId: string, clipId: string) => void;
  onClipDrop: (e: React.DragEvent, trackId: string) => void;
  beatsToPixels: (beats: number) => number;
}

function TrackRow({
  track,
  trackIndex,
  zoom,
  projectDuration,
  currentUser,
  collaborators,
  isSelected,
  selectedClips,
  onVolumeChange,
  onMute,
  onSolo,
  onLock,
  onUnlock,
  onDelete,
  onClipDragStart,
  onClipDrop,
  beatsToPixels,
}: TrackRowProps) {
  const isLocked = track.lockedBy !== undefined;
  const isLockedByMe = track.lockedBy === currentUser?.id;
  const lockedByUser = collaborators.find(c => c.id === track.lockedBy);

  const getTrackIcon = () => {
    switch (track.type) {
      case 'audio': return <Waves className="w-4 h-4" />;
      case 'stem': return <Music className="w-4 h-4" />;
      case 'midi': return <Mic className="w-4 h-4" />;
      case 'generated': return <Drum className="w-4 h-4" />;
    }
  };

  return (
    <div
      className={`flex border-b border-gray-800 ${isSelected ? 'bg-purple-500/5' : ''}`}
      style={{ height: TRACK_HEIGHT }}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => onClipDrop(e, track.id)}
    >
      {/* Track Header */}
      <div className="w-[200px] shrink-0 flex flex-col p-2 border-r border-gray-800 bg-gray-900">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded"
              style={{ backgroundColor: track.color }}
            />
            <span className="text-sm font-medium text-white truncate">{track.name}</span>
            {getTrackIcon()}
          </div>
          <div className="flex items-center gap-1">
            {isLocked && (
              <button
                onClick={isLockedByMe ? onUnlock : undefined}
                className="p-1 text-yellow-500"
                title={lockedByUser ? `Locked by ${lockedByUser.name}` : 'Locked'}
              >
                <Lock className="w-3 h-3" />
              </button>
            )}
            <button
              onClick={onDelete}
              className="p-1 text-gray-500 hover:text-red-400 opacity-0 group-hover:opacity-100"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onMute}
            className={`px-1.5 py-0.5 text-xs rounded ${
              track.muted ? 'bg-red-500 text-white' : 'bg-gray-700 text-gray-400'
            }`}
          >
            M
          </button>
          <button
            onClick={onSolo}
            className={`px-1.5 py-0.5 text-xs rounded ${
              track.solo ? 'bg-yellow-500 text-black' : 'bg-gray-700 text-gray-400'
            }`}
          >
            S
          </button>
          <input
            type="range"
            min="0"
            max="100"
            value={track.volume * 100}
            onChange={(e) => onVolumeChange(Number(e.target.value) / 100)}
            className="flex-1 h-1 accent-purple-500"
          />
          <span className="text-xs text-gray-500 w-8">
            {Math.round(track.volume * 100)}%
          </span>
        </div>
      </div>

      {/* Track Content */}
      <div
        className="relative flex-1"
        style={{ width: beatsToPixels(projectDuration) }}
      >
        {/* Clips */}
        {track.clips.map(clip => (
          <div
            key={clip.id}
            draggable
            onDragStart={(e) => onClipDragStart(e, track.id, clip.id)}
            className={`absolute top-2 h-[calc(100%-16px)] rounded cursor-move overflow-hidden ${
              selectedClips.includes(clip.id)
                ? 'ring-2 ring-purple-500'
                : 'hover:ring-1 hover:ring-white/30'
            }`}
            style={{
              left: beatsToPixels(clip.start),
              width: beatsToPixels(clip.duration),
              backgroundColor: track.color + '40',
              borderLeft: `3px solid ${track.color}`,
            }}
          >
            <div className="p-1">
              <span className="text-xs text-white/80 truncate">{clip.name}</span>
            </div>
            {/* Waveform placeholder */}
            <div className="absolute inset-0 flex items-center justify-center opacity-30">
              <Waves className="w-8 h-8 text-white" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default RemixStudio;
