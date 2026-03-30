// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useAIGeneration, MusicStyle, StyleTransferParams } from '@/hooks/useAIGeneration';
import {
  Shuffle,
  Play,
  Pause,
  Upload,
  Download,
  ArrowRight,
  Sliders,
  Waveform,
  RefreshCw,
} from 'lucide-react';

interface StyleTransferProps {
  className?: string;
}

const TARGET_STYLES: { value: MusicStyle; label: string; description: string }[] = [
  { value: 'electronic', label: 'Electronic', description: 'Modern EDM sound' },
  { value: 'lofi', label: 'Lo-Fi', description: 'Vintage, relaxed vibe' },
  { value: 'techno', label: 'Techno', description: 'Hard, industrial' },
  { value: 'ambient', label: 'Ambient', description: 'Atmospheric, ethereal' },
  { value: 'jazz', label: 'Jazz', description: 'Warm, organic tones' },
  { value: 'hiphop', label: 'Hip Hop', description: 'Punchy, rhythmic' },
];

export function StyleTransfer({ className = '' }: StyleTransferProps) {
  const { isGenerating, progress, currentTask, error, styleTransfer } = useAIGeneration();

  const [sourceBuffer, setSourceBuffer] = useState<AudioBuffer | null>(null);
  const [sourceFileName, setSourceFileName] = useState<string>('');
  const [resultBuffer, setResultBuffer] = useState<AudioBuffer | null>(null);

  const [targetStyle, setTargetStyle] = useState<MusicStyle>('lofi');
  const [intensity, setIntensity] = useState(0.5);
  const [preserveRhythm, setPreserveRhythm] = useState(true);
  const [preserveMelody, setPreserveMelody] = useState(true);
  const [preserveHarmony, setPreserveHarmony] = useState(false);

  const [playingSource, setPlayingSource] = useState(false);
  const [playingResult, setPlayingResult] = useState(false);

  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const resultNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    audioContextRef.current = new AudioContext();
    return () => {
      audioContextRef.current?.close();
    };
  }, []);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const ctx = audioContextRef.current;
    if (!ctx) return;

    setSourceFileName(file.name);
    setResultBuffer(null);

    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      setSourceBuffer(audioBuffer);
    } catch (err) {
      console.error('Failed to decode audio:', err);
    }
  }, []);

  const handleTransfer = async () => {
    if (!sourceBuffer) return;

    const params: StyleTransferParams = {
      sourceTrackId: 'source',
      targetStyle,
      intensity,
      preserveRhythm,
      preserveMelody,
      preserveHarmony,
    };

    try {
      const result = await styleTransfer(sourceBuffer, params);
      setResultBuffer(result);
    } catch (err) {
      console.error('Style transfer failed:', err);
    }
  };

  const playBuffer = (
    buffer: AudioBuffer,
    setPlaying: React.Dispatch<React.SetStateAction<boolean>>,
    nodeRef: React.MutableRefObject<AudioBufferSourceNode | null>
  ) => {
    const ctx = audioContextRef.current;
    if (!ctx) return;

    // Stop if already playing
    if (nodeRef.current) {
      nodeRef.current.stop();
      nodeRef.current = null;
      setPlaying(false);
      return;
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.onended = () => {
      setPlaying(false);
      nodeRef.current = null;
    };
    source.start();
    nodeRef.current = source;
    setPlaying(true);
  };

  const downloadResult = () => {
    if (!resultBuffer) return;

    const numChannels = resultBuffer.numberOfChannels;
    const sampleRate = resultBuffer.sampleRate;
    const length = resultBuffer.length;

    const buffer = new ArrayBuffer(44 + length * numChannels * 2);
    const view = new DataView(buffer);

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

    let offset = 44;
    for (let i = 0; i < length; i++) {
      for (let ch = 0; ch < numChannels; ch++) {
        const sample = resultBuffer.getChannelData(ch)[i];
        const clamped = Math.max(-1, Math.min(1, sample));
        view.setInt16(offset, clamped * 0x7FFF, true);
        offset += 2;
      }
    }

    const blob = new Blob([buffer], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${sourceFileName.replace(/\.[^/.]+$/, '')}-${targetStyle}.wav`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className={`bg-gray-900 rounded-xl p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-500 rounded-lg flex items-center justify-center">
          <Shuffle className="w-5 h-5 text-white" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">AI Style Transfer</h2>
          <p className="text-sm text-gray-400">Transform your music into different styles</p>
        </div>
      </div>

      {/* Source Upload */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-400 mb-3">Source Audio</label>
        <div
          className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
            sourceBuffer
              ? 'border-purple-500/50 bg-purple-500/5'
              : 'border-gray-700 hover:border-gray-600'
          }`}
        >
          {sourceBuffer ? (
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center">
                  <Waveform className="w-6 h-6 text-purple-400" />
                </div>
                <div className="text-left">
                  <p className="text-white font-medium">{sourceFileName}</p>
                  <p className="text-sm text-gray-500">
                    {sourceBuffer.duration.toFixed(1)}s • {sourceBuffer.sampleRate}Hz
                  </p>
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => playBuffer(sourceBuffer, setPlayingSource, sourceNodeRef)}
                  className={`p-2 rounded-lg transition-colors ${
                    playingSource
                      ? 'bg-purple-500 text-white'
                      : 'bg-gray-800 text-gray-400 hover:text-white'
                  }`}
                >
                  {playingSource ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                </button>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="p-2 bg-gray-800 text-gray-400 rounded-lg hover:text-white transition-colors"
                >
                  <Upload className="w-5 h-5" />
                </button>
              </div>
            </div>
          ) : (
            <div
              className="cursor-pointer"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="w-8 h-8 text-gray-600 mx-auto mb-2" />
              <p className="text-gray-400">Drop audio file or click to upload</p>
              <p className="text-sm text-gray-600 mt-1">MP3, WAV, OGG supported</p>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>
      </div>

      {/* Transfer Arrow */}
      <div className="flex items-center justify-center my-4">
        <ArrowRight className="w-6 h-6 text-gray-600" />
      </div>

      {/* Target Style Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-400 mb-3">Target Style</label>
        <div className="grid grid-cols-3 gap-3">
          {TARGET_STYLES.map(({ value, label, description }) => (
            <button
              key={value}
              onClick={() => setTargetStyle(value)}
              className={`p-3 rounded-lg text-left transition-all ${
                targetStyle === value
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white'
              }`}
            >
              <p className="font-medium">{label}</p>
              <p className={`text-xs mt-1 ${targetStyle === value ? 'text-purple-200' : 'text-gray-500'}`}>
                {description}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Parameters */}
      <div className="mb-6 p-4 bg-gray-800/50 rounded-lg">
        <div className="flex items-center gap-2 mb-4">
          <Sliders className="w-4 h-4 text-gray-400" />
          <span className="text-sm font-medium text-gray-400">Transfer Settings</span>
        </div>

        {/* Intensity */}
        <div className="mb-4">
          <label className="flex justify-between text-sm text-gray-400 mb-2">
            <span>Intensity</span>
            <span className="text-purple-400">{Math.round(intensity * 100)}%</span>
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={intensity * 100}
            onChange={(e) => setIntensity(Number(e.target.value) / 100)}
            className="w-full accent-purple-500"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-1">
            <span>Subtle</span>
            <span>Extreme</span>
          </div>
        </div>

        {/* Preservation toggles */}
        <div className="space-y-2">
          <label className="flex items-center justify-between p-2 rounded hover:bg-gray-700/50">
            <span className="text-sm text-gray-300">Preserve Rhythm</span>
            <input
              type="checkbox"
              checked={preserveRhythm}
              onChange={(e) => setPreserveRhythm(e.target.checked)}
              className="w-4 h-4 accent-purple-500"
            />
          </label>
          <label className="flex items-center justify-between p-2 rounded hover:bg-gray-700/50">
            <span className="text-sm text-gray-300">Preserve Melody</span>
            <input
              type="checkbox"
              checked={preserveMelody}
              onChange={(e) => setPreserveMelody(e.target.checked)}
              className="w-4 h-4 accent-purple-500"
            />
          </label>
          <label className="flex items-center justify-between p-2 rounded hover:bg-gray-700/50">
            <span className="text-sm text-gray-300">Preserve Harmony</span>
            <input
              type="checkbox"
              checked={preserveHarmony}
              onChange={(e) => setPreserveHarmony(e.target.checked)}
              className="w-4 h-4 accent-purple-500"
            />
          </label>
        </div>
      </div>

      {/* Progress */}
      {isGenerating && (
        <div className="mb-6">
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-gray-400">{currentTask}</span>
            <span className="text-purple-400">{progress}%</span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
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

      {/* Result */}
      {resultBuffer && (
        <div className="mb-6 p-4 bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-purple-500/30 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
                <Waveform className="w-5 h-5 text-purple-400" />
              </div>
              <div>
                <p className="text-white font-medium">Transformed Audio</p>
                <p className="text-sm text-gray-400">
                  {targetStyle.charAt(0).toUpperCase() + targetStyle.slice(1)} style • {resultBuffer.duration.toFixed(1)}s
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => playBuffer(resultBuffer, setPlayingResult, resultNodeRef)}
                className={`p-2 rounded-lg transition-colors ${
                  playingResult
                    ? 'bg-purple-500 text-white'
                    : 'bg-gray-800 text-gray-400 hover:text-white'
                }`}
              >
                {playingResult ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              </button>
              <button
                onClick={downloadResult}
                className="p-2 bg-gray-800 text-gray-400 rounded-lg hover:text-white transition-colors"
              >
                <Download className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Transfer Button */}
      <button
        onClick={handleTransfer}
        disabled={!sourceBuffer || isGenerating}
        className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white font-semibold rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50"
      >
        {isGenerating ? (
          <>
            <RefreshCw className="w-5 h-5 animate-spin" />
            Transforming...
          </>
        ) : (
          <>
            <Shuffle className="w-5 h-5" />
            Transform Style
          </>
        )}
      </button>
    </div>
  );
}

export default StyleTransfer;
