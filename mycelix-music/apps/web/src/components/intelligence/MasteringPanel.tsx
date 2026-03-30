// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState, useRef, useCallback } from 'react';
import {
  useAIMastering,
  MasteringPreset,
  TargetPlatform,
  AudioAnalysisResult,
} from '@/hooks/useAIMastering';
import {
  Sliders,
  Upload,
  Download,
  Play,
  Pause,
  RefreshCw,
  Settings,
  BarChart2,
  AlertTriangle,
  CheckCircle,
  Volume2,
  Gauge,
} from 'lucide-react';

interface MasteringPanelProps {
  className?: string;
}

const PLATFORM_ICONS: Record<TargetPlatform, string> = {
  'spotify': '🎵',
  'apple-music': '🍎',
  'youtube': '📺',
  'soundcloud': '☁️',
  'cd': '💿',
  'vinyl': '🎸',
  'broadcast': '📻',
};

export function MasteringPanel({ className = '' }: MasteringPanelProps) {
  const {
    isProcessing,
    progress,
    stage,
    chain,
    inputAnalysis,
    outputAnalysis,
    preset,
    error,
    presets,
    platforms,
    platformTargets,
    process: processAudio,
    applyPreset,
    setTargetPlatform,
    updateChain,
  } = useAIMastering();

  const [inputBuffer, setInputBuffer] = useState<AudioBuffer | null>(null);
  const [outputBuffer, setOutputBuffer] = useState<AudioBuffer | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [playingInput, setPlayingInput] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);

  // Initialize audio context
  const getAudioContext = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
    }
    return audioContextRef.current;
  };

  // Handle file upload
  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setOutputBuffer(null);

    try {
      const ctx = getAudioContext();
      const arrayBuffer = await file.arrayBuffer();
      const buffer = await ctx.decodeAudioData(arrayBuffer);
      setInputBuffer(buffer);
    } catch (err) {
      console.error('Failed to load audio:', err);
    }
  };

  // Process audio
  const handleProcess = async () => {
    if (!inputBuffer) return;

    try {
      const result = await processAudio(inputBuffer);
      setOutputBuffer(result);
    } catch (err) {
      console.error('Processing failed:', err);
    }
  };

  // Play audio
  const handlePlay = (useInput: boolean) => {
    const ctx = getAudioContext();
    const buffer = useInput ? inputBuffer : outputBuffer;

    if (!buffer) return;

    // Stop current playback
    if (sourceNodeRef.current) {
      sourceNodeRef.current.stop();
      sourceNodeRef.current = null;
      setIsPlaying(false);
      return;
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.onended = () => {
      setIsPlaying(false);
      sourceNodeRef.current = null;
    };
    source.start();
    sourceNodeRef.current = source;
    setIsPlaying(true);
    setPlayingInput(useInput);
  };

  // Download processed audio
  const handleDownload = () => {
    if (!outputBuffer) return;

    const numChannels = outputBuffer.numberOfChannels;
    const sampleRate = outputBuffer.sampleRate;
    const length = outputBuffer.length;

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
        const sample = outputBuffer.getChannelData(ch)[i];
        const clamped = Math.max(-1, Math.min(1, sample));
        view.setInt16(offset, clamped * 0x7FFF, true);
        offset += 2;
      }
    }

    const blob = new Blob([buffer], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${fileName.replace(/\.[^/.]+$/, '')}-mastered.wav`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className={`bg-gray-900 rounded-xl overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-amber-500 to-orange-500 rounded-lg flex items-center justify-center">
            <Sliders className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">AI Mastering</h2>
            <p className="text-sm text-gray-400">Professional audio mastering</p>
          </div>
        </div>

        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className={`p-2 rounded-lg transition-colors ${
            showAdvanced ? 'bg-purple-500 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'
          }`}
        >
          <Settings className="w-5 h-5" />
        </button>
      </div>

      <div className="p-6">
        {/* File Upload */}
        <div className="mb-6">
          <div
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
              inputBuffer
                ? 'border-amber-500/50 bg-amber-500/5'
                : 'border-gray-700 hover:border-gray-600'
            }`}
          >
            {inputBuffer ? (
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Volume2 className="w-8 h-8 text-amber-400" />
                  <div className="text-left">
                    <p className="text-white font-medium">{fileName}</p>
                    <p className="text-sm text-gray-500">
                      {inputBuffer.duration.toFixed(1)}s • {inputBuffer.sampleRate}Hz
                    </p>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handlePlay(true);
                    }}
                    className={`p-2 rounded-lg ${
                      isPlaying && playingInput
                        ? 'bg-amber-500 text-white'
                        : 'bg-gray-800 text-gray-400 hover:text-white'
                    }`}
                  >
                    {isPlaying && playingInput ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            ) : (
              <>
                <Upload className="w-8 h-8 text-gray-600 mx-auto mb-2" />
                <p className="text-gray-400">Drop audio file or click to upload</p>
              </>
            )}
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>

        {/* Preset Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-400 mb-3">Mastering Preset</label>
          <div className="grid grid-cols-5 gap-2">
            {presets.map(p => (
              <button
                key={p}
                onClick={() => applyPreset(p)}
                className={`px-3 py-2 text-sm rounded-lg capitalize transition-colors ${
                  preset === p
                    ? 'bg-amber-500 text-white'
                    : 'bg-gray-800 text-gray-400 hover:text-white'
                }`}
              >
                {p}
              </button>
            ))}
          </div>
        </div>

        {/* Target Platform */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-400 mb-3">Target Platform</label>
          <div className="grid grid-cols-4 gap-2">
            {platforms.map(p => (
              <button
                key={p}
                onClick={() => setTargetPlatform(p)}
                className={`flex flex-col items-center gap-1 p-3 rounded-lg transition-colors ${
                  chain.loudness.platform === p
                    ? 'bg-amber-500 text-white'
                    : 'bg-gray-800 text-gray-400 hover:text-white'
                }`}
              >
                <span className="text-xl">{PLATFORM_ICONS[p]}</span>
                <span className="text-xs capitalize">{p.replace('-', ' ')}</span>
                <span className="text-xs opacity-60">{platformTargets[p].lufs} LUFS</span>
              </button>
            ))}
          </div>
        </div>

        {/* Advanced Settings */}
        {showAdvanced && (
          <div className="mb-6 p-4 bg-gray-800/50 rounded-lg space-y-4">
            <h3 className="text-sm font-medium text-white mb-3">Advanced Settings</h3>

            {/* EQ */}
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">EQ</span>
              <input
                type="checkbox"
                checked={chain.eq.enabled}
                onChange={(e) => updateChain({ eq: { ...chain.eq, enabled: e.target.checked } })}
                className="w-4 h-4 accent-amber-500"
              />
            </div>

            {/* Compressor */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Compressor</span>
                <input
                  type="checkbox"
                  checked={chain.compressor.enabled}
                  onChange={(e) => updateChain({ compressor: { ...chain.compressor, enabled: e.target.checked } })}
                  className="w-4 h-4 accent-amber-500"
                />
              </div>
              {chain.compressor.enabled && (
                <div className="grid grid-cols-2 gap-3 mt-2">
                  <div>
                    <label className="text-xs text-gray-500">Threshold</label>
                    <input
                      type="range"
                      min="-40"
                      max="0"
                      value={chain.compressor.threshold}
                      onChange={(e) => updateChain({
                        compressor: { ...chain.compressor, threshold: Number(e.target.value) }
                      })}
                      className="w-full accent-amber-500"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500">Ratio</label>
                    <input
                      type="range"
                      min="1"
                      max="20"
                      value={chain.compressor.ratio}
                      onChange={(e) => updateChain({
                        compressor: { ...chain.compressor, ratio: Number(e.target.value) }
                      })}
                      className="w-full accent-amber-500"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Saturation */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Saturation</span>
                <input
                  type="checkbox"
                  checked={chain.saturation.enabled}
                  onChange={(e) => updateChain({ saturation: { ...chain.saturation, enabled: e.target.checked } })}
                  className="w-4 h-4 accent-amber-500"
                />
              </div>
              {chain.saturation.enabled && (
                <div className="flex gap-2 mt-2">
                  {(['tape', 'tube', 'transistor'] as const).map(type => (
                    <button
                      key={type}
                      onClick={() => updateChain({ saturation: { ...chain.saturation, type } })}
                      className={`px-3 py-1 text-xs rounded capitalize ${
                        chain.saturation.type === type
                          ? 'bg-amber-500 text-white'
                          : 'bg-gray-700 text-gray-400'
                      }`}
                    >
                      {type}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Limiter */}
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Limiter (Ceiling: {chain.limiter.ceiling} dB)</span>
              <input
                type="checkbox"
                checked={chain.limiter.enabled}
                onChange={(e) => updateChain({ limiter: { ...chain.limiter, enabled: e.target.checked } })}
                className="w-4 h-4 accent-amber-500"
              />
            </div>
          </div>
        )}

        {/* Analysis Display */}
        {inputAnalysis && (
          <div className="mb-6 grid grid-cols-2 gap-4">
            <AnalysisCard title="Input" analysis={inputAnalysis} />
            {outputAnalysis && <AnalysisCard title="Output" analysis={outputAnalysis} />}
          </div>
        )}

        {/* Issues */}
        {inputAnalysis?.issues && inputAnalysis.issues.length > 0 && (
          <div className="mb-6 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
            <h3 className="flex items-center gap-2 text-sm font-medium text-yellow-400 mb-2">
              <AlertTriangle className="w-4 h-4" />
              Detected Issues
            </h3>
            <ul className="space-y-1">
              {inputAnalysis.issues.map((issue, i) => (
                <li key={i} className="text-sm text-gray-300">
                  • {issue.description}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Progress */}
        {isProcessing && (
          <div className="mb-6">
            <div className="flex items-center justify-between text-sm mb-2">
              <span className="text-gray-400">{stage}</span>
              <span className="text-amber-400">{progress}%</span>
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-amber-500 to-orange-500 transition-all duration-300"
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

        {/* Output */}
        {outputBuffer && (
          <div className="mb-6 p-4 bg-gradient-to-r from-amber-500/10 to-orange-500/10 border border-amber-500/30 rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <CheckCircle className="w-8 h-8 text-amber-400" />
                <div>
                  <p className="text-white font-medium">Mastered Audio</p>
                  <p className="text-sm text-gray-400">
                    {outputAnalysis?.lufs.toFixed(1)} LUFS • Peak: {outputAnalysis?.peakLevel.toFixed(1)} dB
                  </p>
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => handlePlay(false)}
                  className={`p-2 rounded-lg ${
                    isPlaying && !playingInput
                      ? 'bg-amber-500 text-white'
                      : 'bg-gray-800 text-gray-400 hover:text-white'
                  }`}
                >
                  {isPlaying && !playingInput ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>
                <button
                  onClick={handleDownload}
                  className="p-2 bg-gray-800 text-gray-400 rounded-lg hover:text-white"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Process Button */}
        <button
          onClick={handleProcess}
          disabled={!inputBuffer || isProcessing}
          className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-amber-500 to-orange-500 text-white font-semibold rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50"
        >
          {isProcessing ? (
            <>
              <RefreshCw className="w-5 h-5 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Sliders className="w-5 h-5" />
              Master Audio
            </>
          )}
        </button>
      </div>
    </div>
  );
}

// Analysis Card Component
function AnalysisCard({ title, analysis }: { title: string; analysis: AudioAnalysisResult }) {
  return (
    <div className="p-4 bg-gray-800 rounded-lg">
      <h4 className="text-sm font-medium text-gray-400 mb-3">{title}</h4>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-500">LUFS</span>
          <span className="text-white">{analysis.lufs.toFixed(1)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Peak</span>
          <span className="text-white">{analysis.peakLevel.toFixed(1)} dB</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Dynamic Range</span>
          <span className="text-white">{analysis.dynamicRange.toFixed(1)} dB</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Stereo Width</span>
          <span className="text-white">{Math.round(analysis.stereoWidth * 100)}%</span>
        </div>
        <div className="mt-2">
          <span className="text-gray-500 text-xs">Spectral Balance</span>
          <div className="flex gap-1 mt-1">
            <div
              className="h-2 bg-amber-500 rounded"
              style={{ width: `${analysis.spectralBalance.low * 100}%` }}
            />
            <div
              className="h-2 bg-orange-500 rounded"
              style={{ width: `${analysis.spectralBalance.mid * 100}%` }}
            />
            <div
              className="h-2 bg-red-500 rounded"
              style={{ width: `${analysis.spectralBalance.high * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default MasteringPanel;
