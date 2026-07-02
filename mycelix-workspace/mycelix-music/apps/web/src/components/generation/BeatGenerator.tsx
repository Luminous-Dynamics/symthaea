// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState, useRef, useEffect } from 'react';
import {
  useAIGeneration,
  MusicStyle,
  GenerationMode,
  BeatPattern,
  DrumTrack,
} from '@/hooks/useAIGeneration';
import {
  Drum,
  Play,
  Pause,
  RefreshCw,
  Download,
  Wand2,
  Volume2,
  Sliders,
  Music,
  Zap,
  Layers,
  Save,
} from 'lucide-react';

interface BeatGeneratorProps {
  onGenerated?: (pattern: BeatPattern, audioBuffer: AudioBuffer) => void;
  className?: string;
}

const STYLE_OPTIONS: { value: MusicStyle; label: string; tempo: number }[] = [
  { value: 'electronic', label: 'Electronic', tempo: 128 },
  { value: 'house', label: 'House', tempo: 124 },
  { value: 'techno', label: 'Techno', tempo: 138 },
  { value: 'trance', label: 'Trance', tempo: 140 },
  { value: 'dnb', label: 'Drum & Bass', tempo: 174 },
  { value: 'hiphop', label: 'Hip Hop', tempo: 90 },
  { value: 'trap', label: 'Trap', tempo: 140 },
  { value: 'lofi', label: 'Lo-Fi', tempo: 75 },
  { value: 'rock', label: 'Rock', tempo: 120 },
  { value: 'jazz', label: 'Jazz', tempo: 110 },
];

export function BeatGenerator({ onGenerated, className = '' }: BeatGeneratorProps) {
  const {
    isGenerating,
    progress,
    currentTask,
    history,
    error,
    generate,
    play,
  } = useAIGeneration();

  const [style, setStyle] = useState<MusicStyle>('electronic');
  const [tempo, setTempo] = useState(128);
  const [bars, setBars] = useState(4);
  const [complexity, setComplexity] = useState(0.5);
  const [energy, setEnergy] = useState(0.7);
  const [mode, setMode] = useState<GenerationMode>('beat');

  const [currentPattern, setCurrentPattern] = useState<BeatPattern | null>(null);
  const [currentBuffer, setCurrentBuffer] = useState<AudioBuffer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const sourceRef = useRef<AudioBufferSourceNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  // Initialize audio context
  useEffect(() => {
    audioContextRef.current = new AudioContext();
    return () => {
      audioContextRef.current?.close();
    };
  }, []);

  // Update tempo when style changes
  useEffect(() => {
    const styleOption = STYLE_OPTIONS.find(s => s.value === style);
    if (styleOption) {
      setTempo(styleOption.tempo);
    }
  }, [style]);

  const handleGenerate = async () => {
    try {
      const result = await generate({
        mode,
        style,
        tempo,
        bars,
        complexity,
        energy,
      });

      if (result.beatPattern) {
        setCurrentPattern(result.beatPattern);
      }
      if (result.audioBuffer) {
        setCurrentBuffer(result.audioBuffer);
        onGenerated?.(result.beatPattern!, result.audioBuffer);
      }
    } catch (err) {
      console.error('Generation failed:', err);
    }
  };

  const handlePlay = () => {
    if (!currentBuffer) return;

    if (isPlaying) {
      sourceRef.current?.stop();
      setIsPlaying(false);
      return;
    }

    const ctx = audioContextRef.current;
    if (!ctx) return;

    const source = ctx.createBufferSource();
    source.buffer = currentBuffer;
    source.connect(ctx.destination);
    source.loop = true;
    source.start();

    source.onended = () => setIsPlaying(false);
    sourceRef.current = source;
    setIsPlaying(true);
  };

  const handleDownload = () => {
    if (!currentBuffer) return;

    // Convert to WAV
    const numChannels = currentBuffer.numberOfChannels;
    const sampleRate = currentBuffer.sampleRate;
    const length = currentBuffer.length;

    const buffer = new ArrayBuffer(44 + length * numChannels * 2);
    const view = new DataView(buffer);

    // WAV header
    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * numChannels * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * 2, true);
    view.setUint16(32, numChannels * 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, length * numChannels * 2, true);

    // Audio data
    let offset = 44;
    for (let i = 0; i < length; i++) {
      for (let ch = 0; ch < numChannels; ch++) {
        const sample = currentBuffer.getChannelData(ch)[i];
        const clamped = Math.max(-1, Math.min(1, sample));
        view.setInt16(offset, clamped * 0x7FFF, true);
        offset += 2;
      }
    }

    const blob = new Blob([buffer], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${style}-beat-${tempo}bpm.wav`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className={`bg-gray-900 rounded-xl p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
            <Wand2 className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">AI Beat Generator</h2>
            <p className="text-sm text-gray-400">Create unique rhythms with AI</p>
          </div>
        </div>
      </div>

      {/* Mode Selection */}
      <div className="flex gap-2 mb-6">
        {[
          { value: 'beat', label: 'Drums Only', icon: Drum },
          { value: 'melody', label: 'Melody', icon: Music },
          { value: 'full', label: 'Full Track', icon: Layers },
        ].map(({ value, label, icon: Icon }) => (
          <button
            key={value}
            onClick={() => setMode(value as GenerationMode)}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-medium transition-all ${
              mode === value
                ? 'bg-purple-500 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* Style Grid */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-400 mb-3">Style</label>
        <div className="grid grid-cols-5 gap-2">
          {STYLE_OPTIONS.map(({ value, label }) => (
            <button
              key={value}
              onClick={() => setStyle(value)}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                style === value
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Parameters */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {/* Tempo */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Tempo: {tempo} BPM
          </label>
          <input
            type="range"
            min="60"
            max="200"
            value={tempo}
            onChange={(e) => setTempo(Number(e.target.value))}
            className="w-full accent-purple-500"
          />
        </div>

        {/* Bars */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Length: {bars} bars
          </label>
          <input
            type="range"
            min="1"
            max="16"
            value={bars}
            onChange={(e) => setBars(Number(e.target.value))}
            className="w-full accent-purple-500"
          />
        </div>

        {/* Complexity */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Complexity: {Math.round(complexity * 100)}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={complexity * 100}
            onChange={(e) => setComplexity(Number(e.target.value) / 100)}
            className="w-full accent-purple-500"
          />
        </div>

        {/* Energy */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Energy: {Math.round(energy * 100)}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={energy * 100}
            onChange={(e) => setEnergy(Number(e.target.value) / 100)}
            className="w-full accent-purple-500"
          />
        </div>
      </div>

      {/* Pattern Display */}
      {currentPattern && (
        <div className="mb-6 bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Pattern</h3>
          <div className="space-y-2">
            {currentPattern.tracks.map((track, trackIndex) => (
              <div key={trackIndex} className="flex items-center gap-2">
                <span className="w-16 text-xs text-gray-500 truncate">{track.name}</span>
                <div className="flex gap-0.5 flex-1">
                  {track.pattern.slice(0, 16).map((hit, stepIndex) => (
                    <div
                      key={stepIndex}
                      className={`flex-1 h-6 rounded transition-colors ${
                        hit
                          ? `bg-purple-500 opacity-${Math.round(track.velocity[stepIndex] * 100)}`
                          : 'bg-gray-700'
                      }`}
                      style={{
                        opacity: hit ? 0.3 + track.velocity[stepIndex] * 0.7 : 0.3,
                      }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Progress */}
      {isGenerating && (
        <div className="mb-6">
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-gray-400">{currentTask}</span>
            <span className="text-purple-400">{progress}%</span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mb-6 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={handleGenerate}
          disabled={isGenerating}
          className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-semibold rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50"
        >
          {isGenerating ? (
            <RefreshCw className="w-5 h-5 animate-spin" />
          ) : (
            <Zap className="w-5 h-5" />
          )}
          {isGenerating ? 'Generating...' : 'Generate'}
        </button>

        {currentBuffer && (
          <>
            <button
              onClick={handlePlay}
              className={`px-4 py-3 rounded-lg transition-colors ${
                isPlaying
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white'
              }`}
            >
              {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            </button>
            <button
              onClick={handleDownload}
              className="px-4 py-3 bg-gray-800 text-gray-400 rounded-lg hover:bg-gray-700 hover:text-white transition-colors"
            >
              <Download className="w-5 h-5" />
            </button>
          </>
        )}
      </div>

      {/* History */}
      {history.length > 0 && (
        <div className="mt-6 pt-6 border-t border-gray-800">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Recent Generations</h3>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {history.slice(0, 5).map((item) => (
              <div
                key={item.id}
                className="flex items-center justify-between p-3 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors cursor-pointer"
                onClick={() => {
                  if (item.beatPattern) setCurrentPattern(item.beatPattern);
                  if (item.audioBuffer) setCurrentBuffer(item.audioBuffer);
                }}
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-gray-700 rounded flex items-center justify-center">
                    {item.type === 'beat' ? (
                      <Drum className="w-4 h-4 text-purple-400" />
                    ) : item.type === 'melody' ? (
                      <Music className="w-4 h-4 text-pink-400" />
                    ) : (
                      <Layers className="w-4 h-4 text-blue-400" />
                    )}
                  </div>
                  <div>
                    <p className="text-sm text-white font-medium">
                      {item.metadata.style.charAt(0).toUpperCase() + item.metadata.style.slice(1)} {item.type}
                    </p>
                    <p className="text-xs text-gray-500">
                      {item.metadata.tempo} BPM • {item.metadata.duration.toFixed(1)}s
                    </p>
                  </div>
                </div>
                <span className="text-xs text-gray-500">
                  {new Date(item.metadata.createdAt).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default BeatGenerator;
