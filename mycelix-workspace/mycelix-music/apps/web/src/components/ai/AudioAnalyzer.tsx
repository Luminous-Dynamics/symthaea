// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Audio Analyzer Component
 *
 * Visual interface for AI audio analysis:
 * - Drag & drop audio files
 * - Real-time analysis display
 * - Genre/mood/key visualization
 * - Beat grid overlay
 * - Instrument breakdown
 */

import React, { useState, useCallback, useRef } from 'react';
import {
  useAudioAnalysis,
  useGenreClassification,
  useMoodDetection,
  useBeatDetection,
  useKeyDetection,
  useInstrumentRecognition,
  type AudioAnalysis,
} from '../../lib/ai';

// ==================== Types ====================

interface AudioAnalyzerProps {
  className?: string;
  onAnalysisComplete?: (analysis: AudioAnalysis) => void;
}

// ==================== Sub-components ====================

const ConfidenceBar: React.FC<{ value: number; label: string; color?: string }> = ({
  value,
  label,
  color = '#8B5CF6',
}) => (
  <div className="mb-2">
    <div className="flex justify-between text-sm mb-1">
      <span>{label}</span>
      <span>{Math.round(value * 100)}%</span>
    </div>
    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
      <div
        className="h-full transition-all duration-300"
        style={{
          width: `${value * 100}%`,
          backgroundColor: color,
        }}
      />
    </div>
  </div>
);

const GenreDisplay: React.FC<{ data: AudioAnalysis['genre'] | null }> = ({ data }) => {
  if (!data) return null;

  const sortedGenres = Object.entries(data.allGenres)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-3 flex items-center">
        <span className="mr-2">Genre</span>
        <span className="text-2xl">{data.genre}</span>
      </h3>
      <div className="space-y-2">
        {sortedGenres.map(([genre, confidence]) => (
          <ConfidenceBar
            key={genre}
            label={genre}
            value={confidence}
            color={genre === data.genre ? '#8B5CF6' : '#4B5563'}
          />
        ))}
      </div>
    </div>
  );
};

const MoodDisplay: React.FC<{ data: AudioAnalysis['mood'] | null }> = ({ data }) => {
  if (!data) return null;

  const moodEmojis: Record<string, string> = {
    happy: '😊',
    sad: '😢',
    energetic: '⚡',
    calm: '😌',
    angry: '😠',
    romantic: '💕',
    nostalgic: '🌅',
    mysterious: '🔮',
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-3 flex items-center">
        <span className="mr-2">Mood</span>
        <span className="text-2xl">{moodEmojis[data.primaryMood] || '🎵'} {data.primaryMood}</span>
      </h3>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <div className="text-sm text-gray-400 mb-1">Valence</div>
          <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-yellow-500"
              style={{ width: `${(data.valence + 1) * 50}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Negative</span>
            <span>Positive</span>
          </div>
        </div>

        <div>
          <div className="text-sm text-gray-400 mb-1">Energy</div>
          <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-green-500 to-red-500"
              style={{ width: `${data.energy * 100}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Calm</span>
            <span>Energetic</span>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {Object.entries(data.moods)
          .filter(([, v]) => v > 0.1)
          .sort((a, b) => b[1] - a[1])
          .map(([mood, confidence]) => (
            <span
              key={mood}
              className="px-2 py-1 rounded-full text-sm"
              style={{
                backgroundColor: `rgba(139, 92, 246, ${confidence})`,
              }}
            >
              {moodEmojis[mood]} {mood}
            </span>
          ))}
      </div>
    </div>
  );
};

const BeatsDisplay: React.FC<{ data: AudioAnalysis['beats'] | null }> = ({ data }) => {
  if (!data) return null;

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-3">Rhythm</h3>

      <div className="grid grid-cols-2 gap-4">
        <div className="text-center">
          <div className="text-4xl font-bold text-purple-400">{data.tempo}</div>
          <div className="text-sm text-gray-400">BPM</div>
        </div>

        <div className="text-center">
          <div className="text-4xl font-bold text-purple-400">{data.timeSignature}</div>
          <div className="text-sm text-gray-400">Time Signature</div>
        </div>
      </div>

      <div className="mt-4">
        <div className="text-sm text-gray-400 mb-2">Beat Grid ({data.beatTimes.length} beats)</div>
        <div className="h-8 bg-gray-700 rounded flex items-center px-2 overflow-hidden">
          {data.beatTimes.slice(0, 32).map((time, i) => (
            <div
              key={i}
              className={`w-1 mx-0.5 rounded-full ${
                data.downbeatTimes.includes(time)
                  ? 'h-6 bg-purple-400'
                  : 'h-4 bg-gray-500'
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

const KeyDisplay: React.FC<{ data: AudioAnalysis['key'] | null }> = ({ data }) => {
  if (!data) return null;

  const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const keyIndex = notes.indexOf(data.key);

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-3">Key</h3>

      <div className="text-center mb-4">
        <div className="text-4xl font-bold text-purple-400">
          {data.key} {data.mode}
        </div>
        <div className="text-sm text-gray-400">{Math.round(data.confidence * 100)}% confidence</div>
      </div>

      {/* Circle of fifths visualization */}
      <div className="relative w-32 h-32 mx-auto">
        {notes.map((note, i) => {
          const angle = (i * 30 - 90) * (Math.PI / 180);
          const x = 50 + 40 * Math.cos(angle);
          const y = 50 + 40 * Math.sin(angle);
          const isKey = note === data.key;

          return (
            <div
              key={note}
              className={`absolute w-6 h-6 -ml-3 -mt-3 rounded-full flex items-center justify-center text-xs ${
                isKey ? 'bg-purple-500 text-white font-bold' : 'bg-gray-700 text-gray-400'
              }`}
              style={{ left: `${x}%`, top: `${y}%` }}
            >
              {note}
            </div>
          );
        })}
      </div>
    </div>
  );
};

const InstrumentsDisplay: React.FC<{ data: AudioAnalysis['instruments'] | null }> = ({ data }) => {
  if (!data) return null;

  const instrumentIcons: Record<string, string> = {
    acoustic_guitar: '🎸',
    electric_guitar: '🎸',
    bass: '🎸',
    drums: '🥁',
    piano: '🎹',
    keyboard: '🎹',
    synth: '🎛️',
    strings: '🎻',
    violin: '🎻',
    saxophone: '🎷',
    trumpet: '🎺',
    flute: '🪈',
    voice: '🎤',
    organ: '🎹',
    percussion: '🥁',
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-3">Instruments</h3>

      <div className="space-y-2">
        {data.instruments.map(({ name, confidence }) => (
          <div key={name} className="flex items-center">
            <span className="text-2xl mr-3">{instrumentIcons[name] || '🎵'}</span>
            <div className="flex-1">
              <div className="flex justify-between text-sm mb-1">
                <span className="capitalize">{name.replace(/_/g, ' ')}</span>
                <span>{Math.round(confidence * 100)}%</span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-purple-500"
                  style={{ width: `${confidence * 100}%` }}
                />
              </div>
            </div>
          </div>
        ))}

        {data.instruments.length === 0 && (
          <div className="text-gray-500 text-center py-4">
            No instruments detected above threshold
          </div>
        )}
      </div>
    </div>
  );
};

// ==================== Main Component ====================

export const AudioAnalyzer: React.FC<AudioAnalyzerProps> = ({
  className = '',
  onAnalysisComplete,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const {
    data: analysis,
    loading,
    error,
    analyzeFromFile,
  } = useAudioAnalysis();

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) {
      setFileName(file.name);
      const result = await analyzeFromFile(file);
      onAnalysisComplete?.(result);
    }
  }, [analyzeFromFile, onAnalysisComplete]);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFileName(file.name);
      const result = await analyzeFromFile(file);
      onAnalysisComplete?.(result);
    }
  }, [analyzeFromFile, onAnalysisComplete]);

  const handleClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  return (
    <div className={`${className}`}>
      {/* Drop Zone */}
      <div
        className={`
          border-2 border-dashed rounded-lg p-8 mb-6 text-center cursor-pointer
          transition-colors duration-200
          ${isDragging ? 'border-purple-500 bg-purple-500/10' : 'border-gray-600 hover:border-gray-500'}
          ${loading ? 'opacity-50 pointer-events-none' : ''}
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          className="hidden"
          onChange={handleFileSelect}
        />

        {loading ? (
          <div className="space-y-2">
            <div className="animate-spin w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full mx-auto" />
            <p className="text-gray-400">Analyzing {fileName}...</p>
          </div>
        ) : (
          <>
            <div className="text-4xl mb-2">🎵</div>
            <p className="text-gray-300 mb-1">
              Drop an audio file here or click to select
            </p>
            <p className="text-sm text-gray-500">
              Supports MP3, WAV, FLAC, and other audio formats
            </p>
          </>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-500/20 border border-red-500 rounded-lg p-4 mb-6">
          <p className="text-red-400">{error.message}</p>
        </div>
      )}

      {/* Analysis Results */}
      {analysis && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold">Analysis Results</h2>
            <span className="text-sm text-gray-500">
              Processed in {Math.round(analysis.processingTime)}ms
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <GenreDisplay data={analysis.genre} />
            <MoodDisplay data={analysis.mood} />
            <BeatsDisplay data={analysis.beats} />
            <KeyDisplay data={analysis.key} />
            <InstrumentsDisplay data={analysis.instruments} />

            {/* Summary Card */}
            <div className="bg-gradient-to-br from-purple-600 to-pink-600 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3">Summary</h3>
              <div className="space-y-2 text-sm">
                <p>
                  <span className="opacity-70">Duration:</span>{' '}
                  {Math.round(analysis.duration)}s
                </p>
                <p>
                  <span className="opacity-70">Genre:</span>{' '}
                  {analysis.genre.genre} ({Math.round(analysis.genre.confidence * 100)}%)
                </p>
                <p>
                  <span className="opacity-70">Mood:</span>{' '}
                  {analysis.mood.primaryMood}
                </p>
                <p>
                  <span className="opacity-70">Tempo:</span>{' '}
                  {analysis.beats.tempo} BPM
                </p>
                <p>
                  <span className="opacity-70">Key:</span>{' '}
                  {analysis.key.key} {analysis.key.mode}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AudioAnalyzer;
