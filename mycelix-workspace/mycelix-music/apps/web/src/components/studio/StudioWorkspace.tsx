// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useCollaboration } from '@/hooks/useCollaboration';
import {
  Play,
  Pause,
  Square,
  SkipBack,
  Volume2,
  VolumeX,
  Lock,
  Unlock,
  Plus,
  Trash2,
  Users,
  MessageCircle,
  GitBranch,
  Save,
  Download,
  Settings,
  Layers,
} from 'lucide-react';
import { StemTrack } from './StemTrack';
import { CollaboratorCursors } from './CollaboratorCursors';
import { StudioChat } from './StudioChat';
import { VersionHistory } from './VersionHistory';
import { ContributionSplits } from './ContributionSplits';

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
  cursor?: { stemId: string; position: number };
}

interface StudioWorkspaceProps {
  projectId: string;
  projectName: string;
  initialStems?: Stem[];
  bpm: number;
  isOwner: boolean;
}

export function StudioWorkspace({
  projectId,
  projectName,
  initialStems = [],
  bpm: initialBpm,
  isOwner,
}: StudioWorkspaceProps) {
  const [stems, setStems] = useState<Stem[]>(initialStems);
  const [playhead, setPlayhead] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [bpm, setBpm] = useState(initialBpm);
  const [zoom, setZoom] = useState(1);
  const [showChat, setShowChat] = useState(false);
  const [showVersions, setShowVersions] = useState(false);
  const [showSplits, setShowSplits] = useState(false);

  const timelineRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number>();

  // Real-time collaboration
  const {
    participants,
    isConnected,
    sendMessage,
    lastEvent,
  } = useCollaboration(projectId);

  // Handle collaboration events
  useEffect(() => {
    if (!lastEvent) return;

    switch (lastEvent.type) {
      case 'stem:added':
        setStems(prev => [...prev, lastEvent.payload.stem]);
        break;

      case 'stem:updated':
        setStems(prev => prev.map(s =>
          s.id === lastEvent.payload.stemId
            ? { ...s, ...lastEvent.payload.updates }
            : s
        ));
        break;

      case 'stem:deleted':
        setStems(prev => prev.filter(s => s.id !== lastEvent.payload.stemId));
        break;

      case 'stem:locked':
        setStems(prev => prev.map(s =>
          s.id === lastEvent.payload.stemId
            ? { ...s, lockedBy: lastEvent.payload.lockedBy }
            : s
        ));
        break;

      case 'stem:unlocked':
        setStems(prev => prev.map(s =>
          s.id === lastEvent.payload.stemId
            ? { ...s, lockedBy: undefined }
            : s
        ));
        break;

      case 'playhead:moved':
        setPlayhead(lastEvent.payload.position);
        break;

      case 'transport:play':
        setIsPlaying(true);
        break;

      case 'transport:pause':
      case 'transport:stop':
        setIsPlaying(false);
        if (lastEvent.type === 'transport:stop') {
          setPlayhead(0);
        }
        break;
    }
  }, [lastEvent]);

  // Playback animation
  useEffect(() => {
    if (isPlaying) {
      const startTime = Date.now() - (playhead * 1000);

      const animate = () => {
        const elapsed = (Date.now() - startTime) / 1000;
        setPlayhead(elapsed);
        animationRef.current = requestAnimationFrame(animate);
      };

      animationRef.current = requestAnimationFrame(animate);
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying]);

  // Transport controls
  const handlePlay = () => {
    setIsPlaying(true);
    sendMessage({ type: 'transport:play', payload: {} });
  };

  const handlePause = () => {
    setIsPlaying(false);
    sendMessage({ type: 'transport:pause', payload: {} });
  };

  const handleStop = () => {
    setIsPlaying(false);
    setPlayhead(0);
    sendMessage({ type: 'transport:stop', payload: {} });
  };

  // Stem operations
  const handleStemUpdate = (stemId: string, updates: Partial<Stem>) => {
    setStems(prev => prev.map(s =>
      s.id === stemId ? { ...s, ...updates } : s
    ));
    sendMessage({ type: 'stem:update', payload: { stemId, updates } });
  };

  const handleStemLock = (stemId: string) => {
    sendMessage({ type: 'stem:lock', payload: { stemId } });
  };

  const handleStemUnlock = (stemId: string) => {
    sendMessage({ type: 'stem:unlock', payload: { stemId } });
  };

  const handleAddStem = () => {
    // Open file picker or stem creation modal
  };

  const handleDeleteStem = (stemId: string) => {
    sendMessage({ type: 'stem:delete', payload: { stemId } });
  };

  // Calculate total duration
  const totalDuration = Math.max(
    ...stems.map(s => s.startTime + s.duration),
    60 // Minimum 60 seconds
  );

  // Format time
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  };

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-white">
      {/* Header */}
      <div className="h-14 border-b border-white/10 flex items-center justify-between px-4">
        <div className="flex items-center gap-4">
          <h1 className="font-semibold text-lg">{projectName}</h1>
          <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 text-xs rounded-full">
            {bpm} BPM
          </span>
          {isConnected ? (
            <span className="flex items-center gap-1 text-xs text-green-400">
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              Live
            </span>
          ) : (
            <span className="text-xs text-yellow-400">Connecting...</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Participants */}
          <div className="flex items-center -space-x-2 mr-4">
            {participants.slice(0, 4).map((p) => (
              <div
                key={p.id}
                className="w-8 h-8 rounded-full border-2 border-gray-950 flex items-center justify-center text-xs font-medium"
                style={{ backgroundColor: p.color }}
                title={p.name}
              >
                {p.name[0].toUpperCase()}
              </div>
            ))}
            {participants.length > 4 && (
              <div className="w-8 h-8 rounded-full bg-gray-700 border-2 border-gray-950 flex items-center justify-center text-xs">
                +{participants.length - 4}
              </div>
            )}
          </div>

          <button
            onClick={() => setShowChat(!showChat)}
            className={`p-2 rounded-lg transition-colors ${showChat ? 'bg-purple-500' : 'hover:bg-white/10'}`}
          >
            <MessageCircle className="w-5 h-5" />
          </button>

          <button
            onClick={() => setShowVersions(!showVersions)}
            className={`p-2 rounded-lg transition-colors ${showVersions ? 'bg-purple-500' : 'hover:bg-white/10'}`}
          >
            <GitBranch className="w-5 h-5" />
          </button>

          <button
            onClick={() => setShowSplits(!showSplits)}
            className={`p-2 rounded-lg transition-colors ${showSplits ? 'bg-purple-500' : 'hover:bg-white/10'}`}
          >
            <Users className="w-5 h-5" />
          </button>

          <button className="p-2 rounded-lg hover:bg-white/10">
            <Save className="w-5 h-5" />
          </button>

          <button className="p-2 rounded-lg hover:bg-white/10">
            <Download className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Timeline Area */}
        <div className="flex-1 flex flex-col">
          {/* Transport Controls */}
          <div className="h-12 border-b border-white/10 flex items-center gap-4 px-4">
            <div className="flex items-center gap-1">
              <button
                onClick={handleStop}
                className="p-2 rounded hover:bg-white/10"
              >
                <Square className="w-4 h-4" />
              </button>
              <button
                onClick={isPlaying ? handlePause : handlePlay}
                className="p-2 rounded bg-purple-500 hover:bg-purple-600"
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </button>
              <button className="p-2 rounded hover:bg-white/10">
                <SkipBack className="w-4 h-4" />
              </button>
            </div>

            <div className="font-mono text-lg tabular-nums">
              {formatTime(playhead)}
            </div>

            <div className="flex-1" />

            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Zoom:</span>
              <input
                type="range"
                min="0.5"
                max="4"
                step="0.1"
                value={zoom}
                onChange={(e) => setZoom(parseFloat(e.target.value))}
                className="w-24"
              />
            </div>

            <button
              onClick={handleAddStem}
              className="flex items-center gap-2 px-3 py-1.5 bg-purple-500 rounded-lg hover:bg-purple-600"
            >
              <Plus className="w-4 h-4" />
              Add Stem
            </button>
          </div>

          {/* Timeline */}
          <div className="flex-1 overflow-auto" ref={timelineRef}>
            {/* Time ruler */}
            <div className="h-8 border-b border-white/10 sticky top-0 bg-gray-950 z-10">
              <div
                className="h-full relative"
                style={{ width: `${totalDuration * 50 * zoom}px` }}
              >
                {Array.from({ length: Math.ceil(totalDuration) + 1 }, (_, i) => (
                  <div
                    key={i}
                    className="absolute top-0 h-full border-l border-white/20"
                    style={{ left: `${i * 50 * zoom}px` }}
                  >
                    <span className="text-xs text-muted-foreground ml-1">
                      {Math.floor(i / 60)}:{(i % 60).toString().padStart(2, '0')}
                    </span>
                  </div>
                ))}

                {/* Playhead */}
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-purple-500 z-20"
                  style={{ left: `${playhead * 50 * zoom}px` }}
                >
                  <div className="w-3 h-3 bg-purple-500 rounded-full -ml-[5px] -mt-1" />
                </div>
              </div>
            </div>

            {/* Stem tracks */}
            <div className="relative min-h-full">
              {stems.map((stem) => (
                <StemTrack
                  key={stem.id}
                  stem={stem}
                  zoom={zoom}
                  totalDuration={totalDuration}
                  participants={participants}
                  onUpdate={(updates) => handleStemUpdate(stem.id, updates)}
                  onLock={() => handleStemLock(stem.id)}
                  onUnlock={() => handleStemUnlock(stem.id)}
                  onDelete={() => handleDeleteStem(stem.id)}
                />
              ))}

              {/* Collaborator cursors */}
              <CollaboratorCursors
                participants={participants}
                stems={stems}
                zoom={zoom}
              />

              {/* Playhead line through tracks */}
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-purple-500/50 pointer-events-none z-10"
                style={{ left: `${200 + playhead * 50 * zoom}px` }}
              />
            </div>
          </div>
        </div>

        {/* Side Panels */}
        {showChat && (
          <StudioChat
            projectId={projectId}
            participants={participants}
            onClose={() => setShowChat(false)}
          />
        )}

        {showVersions && (
          <VersionHistory
            projectId={projectId}
            onClose={() => setShowVersions(false)}
          />
        )}

        {showSplits && (
          <ContributionSplits
            projectId={projectId}
            stems={stems}
            participants={participants}
            onClose={() => setShowSplits(false)}
          />
        )}
      </div>
    </div>
  );
}
