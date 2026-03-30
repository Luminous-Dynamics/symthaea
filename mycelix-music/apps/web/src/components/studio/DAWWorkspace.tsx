// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  Play,
  Pause,
  Square,
  Circle,
  SkipBack,
  FastForward,
  Rewind,
  Volume2,
  Headphones,
  Mic,
  Music,
  Layers,
  Sliders,
  Wand2,
  Undo,
  Redo,
  Save,
  Download,
  Upload,
  Settings,
  Grid,
  Magnet,
  Scissors,
  Copy,
  Trash2,
  Plus,
  Minus,
  ZoomIn,
  ZoomOut,
  Maximize2,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/design-system/Button';

// ==================== Types ====================

interface Track {
  id: string;
  name: string;
  type: 'audio' | 'midi' | 'bus' | 'master';
  color: string;
  volume: number;
  pan: number;
  mute: boolean;
  solo: boolean;
  armed: boolean;
  clips: Clip[];
  effects: Effect[];
}

interface Clip {
  id: string;
  trackId: string;
  startTime: number;
  duration: number;
  offset: number;
  name: string;
  color?: string;
  waveformData?: number[];
  midiData?: MidiNote[];
}

interface MidiNote {
  pitch: number;
  startTime: number;
  duration: number;
  velocity: number;
}

interface Effect {
  id: string;
  name: string;
  type: string;
  enabled: boolean;
  parameters: Record<string, number>;
}

interface TimelineState {
  playheadPosition: number;
  loopStart: number;
  loopEnd: number;
  loopEnabled: boolean;
  tempo: number;
  timeSignature: [number, number];
  zoom: number;
  scrollX: number;
}

interface DAWState {
  isPlaying: boolean;
  isRecording: boolean;
  metronomeEnabled: boolean;
  snapEnabled: boolean;
  snapValue: string;
  selectedTrackId: string | null;
  selectedClipIds: string[];
}

// ==================== Track Header Component ====================

const TrackHeader: React.FC<{
  track: Track;
  isSelected: boolean;
  onSelect: () => void;
  onVolumeChange: (volume: number) => void;
  onPanChange: (pan: number) => void;
  onMuteToggle: () => void;
  onSoloToggle: () => void;
  onArmToggle: () => void;
}> = ({
  track,
  isSelected,
  onSelect,
  onVolumeChange,
  onPanChange,
  onMuteToggle,
  onSoloToggle,
  onArmToggle,
}) => {
  return (
    <div
      className={cn(
        'h-24 border-b border-gray-200 dark:border-gray-700 p-2 flex flex-col justify-between cursor-pointer transition-colors',
        isSelected ? 'bg-primary-50 dark:bg-primary-900/20' : 'hover:bg-gray-50 dark:hover:bg-gray-800/50'
      )}
      onClick={onSelect}
    >
      <div className="flex items-center gap-2">
        <div
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: track.color }}
        />
        <span className="text-sm font-medium truncate flex-1">{track.name}</span>
        <div className="flex gap-1">
          <button
            onClick={(e) => { e.stopPropagation(); onMuteToggle(); }}
            className={cn(
              'w-6 h-6 text-xs font-bold rounded transition-colors',
              track.mute ? 'bg-yellow-500 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-300 dark:hover:bg-gray-600'
            )}
          >
            M
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onSoloToggle(); }}
            className={cn(
              'w-6 h-6 text-xs font-bold rounded transition-colors',
              track.solo ? 'bg-green-500 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-300 dark:hover:bg-gray-600'
            )}
          >
            S
          </button>
          {track.type === 'audio' && (
            <button
              onClick={(e) => { e.stopPropagation(); onArmToggle(); }}
              className={cn(
                'w-6 h-6 text-xs font-bold rounded transition-colors',
                track.armed ? 'bg-red-500 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-300 dark:hover:bg-gray-600'
              )}
            >
              R
            </button>
          )}
        </div>
      </div>

      <div className="flex items-center gap-2">
        <input
          type="range"
          min="-60"
          max="6"
          step="0.1"
          value={20 * Math.log10(track.volume || 0.001)}
          onChange={(e) => onVolumeChange(Math.pow(10, parseFloat(e.target.value) / 20))}
          onClick={(e) => e.stopPropagation()}
          className="flex-1 h-1 accent-primary-500"
        />
        <span className="text-xs text-gray-500 w-8 text-right">
          {(20 * Math.log10(track.volume || 0.001)).toFixed(1)}
        </span>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-400">Pan</span>
        <input
          type="range"
          min="-1"
          max="1"
          step="0.01"
          value={track.pan}
          onChange={(e) => onPanChange(parseFloat(e.target.value))}
          onClick={(e) => e.stopPropagation()}
          className="flex-1 h-1 accent-primary-500"
        />
        <span className="text-xs text-gray-500 w-6 text-right">
          {track.pan === 0 ? 'C' : track.pan > 0 ? `R${Math.round(track.pan * 100)}` : `L${Math.round(Math.abs(track.pan) * 100)}`}
        </span>
      </div>
    </div>
  );
};

// ==================== Timeline Ruler ====================

const TimelineRuler: React.FC<{
  zoom: number;
  scrollX: number;
  tempo: number;
  timeSignature: [number, number];
  width: number;
  playheadPosition: number;
}> = ({ zoom, scrollX, tempo, timeSignature, width, playheadPosition }) => {
  const pixelsPerBeat = 50 * zoom;
  const pixelsPerBar = pixelsPerBeat * timeSignature[0];
  const startBar = Math.floor(scrollX / pixelsPerBar);
  const endBar = Math.ceil((scrollX + width) / pixelsPerBar);

  const markers = [];
  for (let bar = startBar; bar <= endBar; bar++) {
    const x = bar * pixelsPerBar - scrollX;
    markers.push(
      <div
        key={`bar-${bar}`}
        className="absolute top-0 h-full flex flex-col"
        style={{ left: x }}
      >
        <div className="h-4 border-l border-gray-400 dark:border-gray-600">
          <span className="text-xs text-gray-500 ml-1">{bar + 1}</span>
        </div>
        {/* Beat markers */}
        {Array.from({ length: timeSignature[0] - 1 }).map((_, i) => (
          <div
            key={`beat-${i}`}
            className="absolute h-2 border-l border-gray-300 dark:border-gray-700"
            style={{ left: (i + 1) * pixelsPerBeat, top: 16 }}
          />
        ))}
      </div>
    );
  }

  return (
    <div className="relative h-6 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 overflow-hidden">
      {markers}
      {/* Playhead indicator */}
      <div
        className="absolute top-0 h-full w-0.5 bg-red-500 z-10"
        style={{ left: playheadPosition * pixelsPerBeat - scrollX }}
      />
    </div>
  );
};

// ==================== Clip Component ====================

const ClipComponent: React.FC<{
  clip: Clip;
  track: Track;
  zoom: number;
  scrollX: number;
  isSelected: boolean;
  onSelect: () => void;
}> = ({ clip, track, zoom, scrollX, isSelected, onSelect }) => {
  const pixelsPerBeat = 50 * zoom;
  const left = clip.startTime * pixelsPerBeat - scrollX;
  const width = clip.duration * pixelsPerBeat;

  return (
    <div
      className={cn(
        'absolute top-1 bottom-1 rounded overflow-hidden cursor-pointer transition-all',
        isSelected ? 'ring-2 ring-primary-500' : 'hover:ring-1 hover:ring-primary-300'
      )}
      style={{
        left,
        width,
        backgroundColor: clip.color || track.color,
      }}
      onClick={(e) => {
        e.stopPropagation();
        onSelect();
      }}
    >
      <div className="absolute inset-0 bg-black/10" />
      <div className="relative px-1 py-0.5">
        <span className="text-xs text-white font-medium truncate block">
          {clip.name}
        </span>
      </div>
      {/* Waveform visualization */}
      {clip.waveformData && (
        <div className="absolute bottom-0 left-0 right-0 h-12 flex items-end">
          {clip.waveformData.map((amplitude, i) => (
            <div
              key={i}
              className="flex-1 bg-white/30"
              style={{ height: `${amplitude * 100}%` }}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// ==================== Track Lane Component ====================

const TrackLane: React.FC<{
  track: Track;
  zoom: number;
  scrollX: number;
  selectedClipIds: string[];
  onClipSelect: (clipId: string) => void;
}> = ({ track, zoom, scrollX, selectedClipIds, onClipSelect }) => {
  return (
    <div className="h-24 border-b border-gray-200 dark:border-gray-700 relative bg-gray-50 dark:bg-gray-900">
      {track.clips.map((clip) => (
        <ClipComponent
          key={clip.id}
          clip={clip}
          track={track}
          zoom={zoom}
          scrollX={scrollX}
          isSelected={selectedClipIds.includes(clip.id)}
          onSelect={() => onClipSelect(clip.id)}
        />
      ))}
    </div>
  );
};

// ==================== Transport Controls ====================

const TransportControls: React.FC<{
  isPlaying: boolean;
  isRecording: boolean;
  tempo: number;
  timeSignature: [number, number];
  playheadPosition: number;
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  onRecord: () => void;
  onRewind: () => void;
  onFastForward: () => void;
  onTempoChange: (tempo: number) => void;
  metronomeEnabled: boolean;
  onMetronomeToggle: () => void;
}> = ({
  isPlaying,
  isRecording,
  tempo,
  timeSignature,
  playheadPosition,
  onPlay,
  onPause,
  onStop,
  onRecord,
  onRewind,
  onFastForward,
  onTempoChange,
  metronomeEnabled,
  onMetronomeToggle,
}) => {
  const formatTime = (beats: number) => {
    const bar = Math.floor(beats / timeSignature[0]) + 1;
    const beat = Math.floor(beats % timeSignature[0]) + 1;
    const tick = Math.floor((beats % 1) * 960);
    return `${bar}.${beat}.${tick.toString().padStart(3, '0')}`;
  };

  return (
    <div className="h-14 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center px-4 gap-4">
      {/* Time display */}
      <div className="bg-gray-900 text-green-400 font-mono text-lg px-4 py-1 rounded">
        {formatTime(playheadPosition)}
      </div>

      {/* Transport buttons */}
      <div className="flex items-center gap-1">
        <button
          onClick={onRewind}
          className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors"
        >
          <Rewind className="w-5 h-5" />
        </button>
        <button
          onClick={onStop}
          className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors"
        >
          <Square className="w-5 h-5" />
        </button>
        <button
          onClick={isPlaying ? onPause : onPlay}
          className="p-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
        >
          {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
        </button>
        <button
          onClick={onRecord}
          className={cn(
            'p-2 rounded transition-colors',
            isRecording ? 'bg-red-500 text-white' : 'hover:bg-gray-200 dark:hover:bg-gray-700'
          )}
        >
          <Circle className={cn('w-5 h-5', isRecording && 'fill-current animate-pulse')} />
        </button>
        <button
          onClick={onFastForward}
          className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors"
        >
          <FastForward className="w-5 h-5" />
        </button>
      </div>

      {/* Tempo */}
      <div className="flex items-center gap-2 px-4 border-l border-gray-300 dark:border-gray-600">
        <span className="text-sm text-gray-500">BPM</span>
        <input
          type="number"
          value={tempo}
          onChange={(e) => onTempoChange(parseInt(e.target.value) || 120)}
          className="w-16 px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded text-center"
        />
      </div>

      {/* Time signature */}
      <div className="flex items-center gap-1 text-sm text-gray-500">
        <span>{timeSignature[0]}</span>
        <span>/</span>
        <span>{timeSignature[1]}</span>
      </div>

      {/* Metronome */}
      <button
        onClick={onMetronomeToggle}
        className={cn(
          'p-2 rounded transition-colors',
          metronomeEnabled ? 'bg-primary-100 dark:bg-primary-900 text-primary-600' : 'hover:bg-gray-200 dark:hover:bg-gray-700'
        )}
        title="Metronome"
      >
        <Music className="w-5 h-5" />
      </button>
    </div>
  );
};

// ==================== Toolbar ====================

const Toolbar: React.FC<{
  snapEnabled: boolean;
  snapValue: string;
  zoom: number;
  onSnapToggle: () => void;
  onSnapValueChange: (value: string) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onSave: () => void;
}> = ({
  snapEnabled,
  snapValue,
  zoom,
  onSnapToggle,
  onSnapValueChange,
  onZoomIn,
  onZoomOut,
  onUndo,
  onRedo,
  onSave,
}) => {
  return (
    <div className="h-10 bg-gray-50 dark:bg-gray-850 border-b border-gray-200 dark:border-gray-700 flex items-center px-4 gap-2">
      {/* Edit tools */}
      <div className="flex items-center gap-1 pr-4 border-r border-gray-300 dark:border-gray-600">
        <button
          onClick={onUndo}
          className="p-1.5 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors"
          title="Undo"
        >
          <Undo className="w-4 h-4" />
        </button>
        <button
          onClick={onRedo}
          className="p-1.5 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors"
          title="Redo"
        >
          <Redo className="w-4 h-4" />
        </button>
      </div>

      {/* Clip tools */}
      <div className="flex items-center gap-1 pr-4 border-r border-gray-300 dark:border-gray-600">
        <button className="p-1.5 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors" title="Cut">
          <Scissors className="w-4 h-4" />
        </button>
        <button className="p-1.5 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors" title="Copy">
          <Copy className="w-4 h-4" />
        </button>
        <button className="p-1.5 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors" title="Delete">
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Snap */}
      <div className="flex items-center gap-2 pr-4 border-r border-gray-300 dark:border-gray-600">
        <button
          onClick={onSnapToggle}
          className={cn(
            'p-1.5 rounded transition-colors',
            snapEnabled ? 'bg-primary-100 dark:bg-primary-900 text-primary-600' : 'hover:bg-gray-200 dark:hover:bg-gray-700'
          )}
          title="Snap to grid"
        >
          <Magnet className="w-4 h-4" />
        </button>
        <select
          value={snapValue}
          onChange={(e) => onSnapValueChange(e.target.value)}
          className="px-2 py-1 text-xs bg-gray-200 dark:bg-gray-700 rounded"
        >
          <option value="1">1 Bar</option>
          <option value="1/2">1/2</option>
          <option value="1/4">1/4</option>
          <option value="1/8">1/8</option>
          <option value="1/16">1/16</option>
        </select>
      </div>

      {/* Zoom */}
      <div className="flex items-center gap-1 pr-4 border-r border-gray-300 dark:border-gray-600">
        <button
          onClick={onZoomOut}
          className="p-1.5 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors"
          title="Zoom out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <span className="text-xs text-gray-500 w-12 text-center">{Math.round(zoom * 100)}%</span>
        <button
          onClick={onZoomIn}
          className="p-1.5 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors"
          title="Zoom in"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
      </div>

      <div className="flex-1" />

      {/* Save */}
      <Button size="sm" onClick={onSave} leftIcon={<Save className="w-4 h-4" />}>
        Save
      </Button>
    </div>
  );
};

// ==================== Main DAW Workspace ====================

export const DAWWorkspace: React.FC<{
  className?: string;
}> = ({ className }) => {
  const [tracks, setTracks] = useState<Track[]>([
    {
      id: '1',
      name: 'Drums',
      type: 'audio',
      color: '#ef4444',
      volume: 0.8,
      pan: 0,
      mute: false,
      solo: false,
      armed: false,
      clips: [
        { id: 'c1', trackId: '1', startTime: 0, duration: 8, offset: 0, name: 'Drums Loop', waveformData: Array(100).fill(0).map(() => Math.random() * 0.8 + 0.2) },
      ],
      effects: [],
    },
    {
      id: '2',
      name: 'Bass',
      type: 'audio',
      color: '#f59e0b',
      volume: 0.7,
      pan: 0,
      mute: false,
      solo: false,
      armed: false,
      clips: [
        { id: 'c2', trackId: '2', startTime: 4, duration: 4, offset: 0, name: 'Bass Line', waveformData: Array(50).fill(0).map(() => Math.random() * 0.6 + 0.1) },
      ],
      effects: [],
    },
    {
      id: '3',
      name: 'Synth',
      type: 'midi',
      color: '#10b981',
      volume: 0.6,
      pan: -0.2,
      mute: false,
      solo: false,
      armed: false,
      clips: [
        { id: 'c3', trackId: '3', startTime: 0, duration: 16, offset: 0, name: 'Synth Pad' },
      ],
      effects: [],
    },
    {
      id: '4',
      name: 'Vocals',
      type: 'audio',
      color: '#6366f1',
      volume: 0.9,
      pan: 0,
      mute: false,
      solo: false,
      armed: true,
      clips: [],
      effects: [],
    },
  ]);

  const [timeline, setTimeline] = useState<TimelineState>({
    playheadPosition: 0,
    loopStart: 0,
    loopEnd: 16,
    loopEnabled: false,
    tempo: 120,
    timeSignature: [4, 4],
    zoom: 1,
    scrollX: 0,
  });

  const [dawState, setDawState] = useState<DAWState>({
    isPlaying: false,
    isRecording: false,
    metronomeEnabled: true,
    snapEnabled: true,
    snapValue: '1/4',
    selectedTrackId: null,
    selectedClipIds: [],
  });

  const timelineRef = useRef<HTMLDivElement>(null);

  // Playback animation
  useEffect(() => {
    if (!dawState.isPlaying) return;

    const interval = setInterval(() => {
      setTimeline((prev) => ({
        ...prev,
        playheadPosition: prev.playheadPosition + (prev.tempo / 60) / 60,
      }));
    }, 1000 / 60);

    return () => clearInterval(interval);
  }, [dawState.isPlaying, timeline.tempo]);

  const handleTrackVolumeChange = (trackId: string, volume: number) => {
    setTracks((prev) =>
      prev.map((t) => (t.id === trackId ? { ...t, volume } : t))
    );
  };

  const handleTrackPanChange = (trackId: string, pan: number) => {
    setTracks((prev) =>
      prev.map((t) => (t.id === trackId ? { ...t, pan } : t))
    );
  };

  const handleTrackMuteToggle = (trackId: string) => {
    setTracks((prev) =>
      prev.map((t) => (t.id === trackId ? { ...t, mute: !t.mute } : t))
    );
  };

  const handleTrackSoloToggle = (trackId: string) => {
    setTracks((prev) =>
      prev.map((t) => (t.id === trackId ? { ...t, solo: !t.solo } : t))
    );
  };

  const handleTrackArmToggle = (trackId: string) => {
    setTracks((prev) =>
      prev.map((t) => (t.id === trackId ? { ...t, armed: !t.armed } : t))
    );
  };

  return (
    <div className={cn('h-screen flex flex-col bg-white dark:bg-gray-900', className)}>
      {/* Toolbar */}
      <Toolbar
        snapEnabled={dawState.snapEnabled}
        snapValue={dawState.snapValue}
        zoom={timeline.zoom}
        onSnapToggle={() => setDawState((s) => ({ ...s, snapEnabled: !s.snapEnabled }))}
        onSnapValueChange={(value) => setDawState((s) => ({ ...s, snapValue: value }))}
        onZoomIn={() => setTimeline((t) => ({ ...t, zoom: Math.min(4, t.zoom * 1.25) }))}
        onZoomOut={() => setTimeline((t) => ({ ...t, zoom: Math.max(0.25, t.zoom / 1.25) }))}
        onUndo={() => {}}
        onRedo={() => {}}
        onSave={() => {}}
      />

      {/* Transport */}
      <TransportControls
        isPlaying={dawState.isPlaying}
        isRecording={dawState.isRecording}
        tempo={timeline.tempo}
        timeSignature={timeline.timeSignature}
        playheadPosition={timeline.playheadPosition}
        onPlay={() => setDawState((s) => ({ ...s, isPlaying: true }))}
        onPause={() => setDawState((s) => ({ ...s, isPlaying: false }))}
        onStop={() => {
          setDawState((s) => ({ ...s, isPlaying: false, isRecording: false }));
          setTimeline((t) => ({ ...t, playheadPosition: 0 }));
        }}
        onRecord={() => setDawState((s) => ({ ...s, isRecording: !s.isRecording, isPlaying: true }))}
        onRewind={() => setTimeline((t) => ({ ...t, playheadPosition: Math.max(0, t.playheadPosition - 4) }))}
        onFastForward={() => setTimeline((t) => ({ ...t, playheadPosition: t.playheadPosition + 4 }))}
        onTempoChange={(tempo) => setTimeline((t) => ({ ...t, tempo }))}
        metronomeEnabled={dawState.metronomeEnabled}
        onMetronomeToggle={() => setDawState((s) => ({ ...s, metronomeEnabled: !s.metronomeEnabled }))}
      />

      {/* Main area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Track headers */}
        <div className="w-48 border-r border-gray-200 dark:border-gray-700 overflow-y-auto">
          <div className="h-6" /> {/* Ruler spacer */}
          {tracks.map((track) => (
            <TrackHeader
              key={track.id}
              track={track}
              isSelected={dawState.selectedTrackId === track.id}
              onSelect={() => setDawState((s) => ({ ...s, selectedTrackId: track.id }))}
              onVolumeChange={(v) => handleTrackVolumeChange(track.id, v)}
              onPanChange={(p) => handleTrackPanChange(track.id, p)}
              onMuteToggle={() => handleTrackMuteToggle(track.id)}
              onSoloToggle={() => handleTrackSoloToggle(track.id)}
              onArmToggle={() => handleTrackArmToggle(track.id)}
            />
          ))}
          {/* Add track button */}
          <button className="w-full h-12 border-b border-gray-200 dark:border-gray-700 flex items-center justify-center gap-2 text-gray-500 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors">
            <Plus className="w-4 h-4" />
            <span className="text-sm">Add Track</span>
          </button>
        </div>

        {/* Timeline */}
        <div ref={timelineRef} className="flex-1 overflow-auto">
          <TimelineRuler
            zoom={timeline.zoom}
            scrollX={timeline.scrollX}
            tempo={timeline.tempo}
            timeSignature={timeline.timeSignature}
            width={timelineRef.current?.clientWidth || 1000}
            playheadPosition={timeline.playheadPosition}
          />
          {tracks.map((track) => (
            <TrackLane
              key={track.id}
              track={track}
              zoom={timeline.zoom}
              scrollX={timeline.scrollX}
              selectedClipIds={dawState.selectedClipIds}
              onClipSelect={(clipId) =>
                setDawState((s) => ({
                  ...s,
                  selectedClipIds: s.selectedClipIds.includes(clipId)
                    ? s.selectedClipIds.filter((id) => id !== clipId)
                    : [...s.selectedClipIds, clipId],
                }))
              }
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default DAWWorkspace;
