// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState, useRef } from 'react';
import {
  useMusicTheory,
  NoteName,
  Chord,
  ScaleInfo,
} from '@/hooks/useMusicTheory';
import {
  Music,
  Upload,
  Lightbulb,
  BookOpen,
  Piano,
  RefreshCw,
  ChevronRight,
  Info,
} from 'lucide-react';

interface TheoryAssistantProps {
  className?: string;
}

const NOTE_COLORS: Record<NoteName, string> = {
  'C': 'bg-red-500',
  'C#': 'bg-red-400',
  'D': 'bg-orange-500',
  'D#': 'bg-orange-400',
  'E': 'bg-yellow-500',
  'F': 'bg-green-500',
  'F#': 'bg-green-400',
  'G': 'bg-blue-500',
  'G#': 'bg-blue-400',
  'A': 'bg-indigo-500',
  'A#': 'bg-purple-500',
  'B': 'bg-pink-500',
};

export function TheoryAssistant({ className = '' }: TheoryAssistantProps) {
  const {
    isAnalyzing,
    progress,
    analysis,
    selectedScale,
    availableScales,
    error,
    scalePatterns,
    analyze,
    selectScale,
    suggestNextChord,
  } = useMusicTheory();

  const [activeTab, setActiveTab] = useState<'analysis' | 'scales' | 'chords' | 'suggestions'>('analysis');
  const [selectedChord, setSelectedChord] = useState<Chord | null>(null);
  const [chordSuggestions, setChordSuggestions] = useState<Chord[]>([]);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext();
      }

      const arrayBuffer = await file.arrayBuffer();
      const buffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
      await analyze(buffer);
    } catch (err) {
      console.error('Failed to analyze:', err);
    }
  };

  const handleChordClick = (chord: Chord) => {
    setSelectedChord(chord);
    const suggestions = suggestNextChord(chord);
    setChordSuggestions(suggestions);
  };

  const formatChord = (chord: Chord): string => {
    let name = chord.root;
    if (chord.quality === 'minor') name += 'm';
    else if (chord.quality === 'diminished') name += 'dim';
    else if (chord.quality === 'augmented') name += 'aug';
    else if (chord.quality === 'major7') name += 'maj7';
    else if (chord.quality === 'minor7') name += 'm7';
    else if (chord.quality === 'dominant7') name += '7';
    else if (chord.quality === 'sus2') name += 'sus2';
    else if (chord.quality === 'sus4') name += 'sus4';
    return name;
  };

  return (
    <div className={`bg-gray-900 rounded-xl overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center">
            <BookOpen className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">Music Theory Assistant</h2>
            <p className="text-sm text-gray-400">
              {analysis ? `${analysis.key.root} ${analysis.key.mode}` : 'Analyze audio to get started'}
            </p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-800">
        {[
          { id: 'analysis', label: 'Analysis', icon: Music },
          { id: 'scales', label: 'Scales', icon: Piano },
          { id: 'chords', label: 'Chords', icon: BookOpen },
          { id: 'suggestions', label: 'Ideas', icon: Lightbulb },
        ].map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id as any)}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === id
                ? 'text-purple-400 border-b-2 border-purple-400'
                : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      <div className="p-6">
        {/* Upload */}
        {!analysis && (
          <div className="mb-6">
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-gray-700 hover:border-gray-600 rounded-lg p-8 text-center cursor-pointer"
            >
              <Upload className="w-10 h-10 text-gray-600 mx-auto mb-3" />
              <p className="text-gray-400">Upload audio to analyze</p>
              <p className="text-sm text-gray-600 mt-1">MP3, WAV, or OGG</p>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>
        )}

        {/* Progress */}
        {isAnalyzing && (
          <div className="mb-6">
            <div className="flex items-center justify-between text-sm mb-2">
              <span className="text-gray-400">Analyzing...</span>
              <span className="text-purple-400">{progress}%</span>
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all"
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

        {/* Analysis Tab */}
        {activeTab === 'analysis' && analysis && (
          <div className="space-y-6">
            {/* Key Display */}
            <div className="text-center py-6 bg-gray-800 rounded-lg">
              <p className="text-gray-400 mb-2">Detected Key</p>
              <p className="text-4xl font-bold text-white">
                {analysis.key.root} {analysis.key.mode}
              </p>
              <p className="text-sm text-gray-500 mt-2">
                Confidence: {Math.round(analysis.key.confidence * 100)}%
              </p>
              {analysis.key.alternates.length > 0 && (
                <div className="flex justify-center gap-2 mt-3">
                  <span className="text-xs text-gray-500">Also possible:</span>
                  {analysis.key.alternates.slice(0, 2).map((alt, i) => (
                    <span key={i} className="text-xs text-gray-400">
                      {alt.root} {alt.mode} ({Math.round(alt.confidence * 100)}%)
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Tempo & Time Signature */}
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-gray-800 rounded-lg text-center">
                <p className="text-gray-400 text-sm">Tempo</p>
                <p className="text-2xl font-bold text-white">{analysis.tempo} BPM</p>
              </div>
              <div className="p-4 bg-gray-800 rounded-lg text-center">
                <p className="text-gray-400 text-sm">Time Signature</p>
                <p className="text-2xl font-bold text-white">
                  {analysis.timeSignature[0]}/{analysis.timeSignature[1]}
                </p>
              </div>
            </div>

            {/* Detected Chords Timeline */}
            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-3">Chord Progression</h3>
              <div className="flex flex-wrap gap-2">
                {analysis.chords.map((chord, i) => (
                  <button
                    key={i}
                    onClick={() => handleChordClick(chord)}
                    className={`px-3 py-2 rounded-lg font-medium transition-colors ${
                      selectedChord === chord
                        ? 'bg-purple-500 text-white'
                        : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                    }`}
                  >
                    {formatChord(chord)}
                  </button>
                ))}
              </div>
            </div>

            {/* Progressions */}
            {analysis.progressions.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-400 mb-3">Identified Progressions</h3>
                {analysis.progressions.map((prog, i) => (
                  <div key={i} className="p-4 bg-gray-800 rounded-lg mb-2">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-white font-medium">
                        {prog.romanNumerals.join(' - ')}
                      </span>
                      {prog.commonNames && (
                        <span className="text-xs text-purple-400 bg-purple-500/20 px-2 py-1 rounded">
                          {prog.commonNames[0]}
                        </span>
                      )}
                    </div>
                    <div className="flex gap-2">
                      {prog.chords.map((chord, j) => (
                        <span key={j} className="text-sm text-gray-400">
                          {formatChord(chord)}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Scales Tab */}
        {activeTab === 'scales' && analysis && (
          <div className="space-y-6">
            {/* Selected Scale */}
            {selectedScale && (
              <div className="p-6 bg-gradient-to-r from-indigo-500/10 to-purple-500/10 border border-purple-500/30 rounded-lg">
                <h3 className="text-xl font-bold text-white mb-2">{selectedScale.name}</h3>
                <p className="text-gray-400 mb-4">{selectedScale.mood}</p>

                {/* Piano keys visualization */}
                <div className="flex gap-1 mb-4">
                  {['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'].map((note) => {
                    const isInScale = selectedScale.notes.includes(note as NoteName);
                    const isBlack = note.includes('#');

                    return (
                      <div
                        key={note}
                        className={`relative flex items-end justify-center pb-2 rounded-b ${
                          isBlack
                            ? `w-6 h-20 -mx-2 z-10 ${isInScale ? 'bg-purple-600' : 'bg-gray-700'}`
                            : `w-8 h-28 ${isInScale ? 'bg-white' : 'bg-gray-600'}`
                        }`}
                      >
                        <span className={`text-xs font-medium ${
                          isBlack
                            ? 'text-white'
                            : isInScale ? 'text-gray-800' : 'text-gray-400'
                        }`}>
                          {note}
                        </span>
                      </div>
                    );
                  })}
                </div>

                {/* Scale notes */}
                <div className="flex gap-2 mb-4">
                  {selectedScale.notes.map((note, i) => (
                    <div
                      key={i}
                      className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-medium ${NOTE_COLORS[note]}`}
                    >
                      {note}
                    </div>
                  ))}
                </div>

                {/* Related chords */}
                <div>
                  <p className="text-sm text-gray-400 mb-2">Related Chords:</p>
                  <div className="flex gap-2">
                    {selectedScale.relatedChords.map((chord, i) => (
                      <span key={i} className="px-3 py-1 bg-gray-800 text-gray-300 rounded-lg text-sm">
                        {chord}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Genres */}
                <div className="mt-4">
                  <p className="text-sm text-gray-400 mb-2">Common in:</p>
                  <div className="flex gap-2">
                    {selectedScale.genres.map((genre, i) => (
                      <span key={i} className="px-2 py-1 bg-purple-500/20 text-purple-400 rounded text-xs">
                        {genre}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Scale Selection */}
            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-3">Available Scales</h3>
              <div className="grid grid-cols-2 gap-2">
                {scalePatterns.map(scale => (
                  <button
                    key={scale}
                    onClick={() => selectScale(scale)}
                    className={`p-3 rounded-lg text-left transition-colors ${
                      selectedScale?.name.includes(scale)
                        ? 'bg-purple-500 text-white'
                        : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                    }`}
                  >
                    <p className="font-medium">{scale}</p>
                    <p className="text-xs opacity-60">
                      {analysis.key.root} {scale}
                    </p>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Chords Tab */}
        {activeTab === 'chords' && analysis && (
          <div className="space-y-6">
            {/* Chord Building */}
            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-3">Chords in {analysis.key.root} {analysis.key.mode}</h3>
              <div className="grid grid-cols-7 gap-2">
                {['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°'].map((numeral, i) => (
                  <div key={i} className="text-center">
                    <div className="w-full aspect-square bg-gray-800 rounded-lg flex items-center justify-center mb-1">
                      <span className="text-lg font-bold text-white">{numeral}</span>
                    </div>
                    <span className="text-xs text-gray-500">
                      {selectedScale?.notes[i] || '-'}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Selected Chord Details */}
            {selectedChord && (
              <div className="p-4 bg-gray-800 rounded-lg">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold text-white">{formatChord(selectedChord)}</h3>
                  <span className="text-sm text-gray-400">
                    Confidence: {Math.round(selectedChord.confidence * 100)}%
                  </span>
                </div>

                <p className="text-sm text-gray-400 mb-4">
                  Position: {selectedChord.time.toFixed(1)}s • Duration: {selectedChord.duration.toFixed(1)}s
                </p>

                {/* Next chord suggestions */}
                {chordSuggestions.length > 0 && (
                  <div>
                    <p className="text-sm text-gray-400 mb-2">Suggested next chords:</p>
                    <div className="flex gap-2">
                      {chordSuggestions.map((suggestion, i) => (
                        <button
                          key={i}
                          className="flex items-center gap-1 px-3 py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600"
                        >
                          <ChevronRight className="w-3 h-3 text-purple-400" />
                          {formatChord(suggestion)}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Suggestions Tab */}
        {activeTab === 'suggestions' && analysis && (
          <div className="space-y-4">
            {analysis.suggestions.map((suggestion, i) => (
              <div key={i} className="p-4 bg-gray-800 rounded-lg">
                <div className="flex items-start gap-3">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                    suggestion.type === 'chord' ? 'bg-purple-500/20 text-purple-400' :
                    suggestion.type === 'scale' ? 'bg-blue-500/20 text-blue-400' :
                    suggestion.type === 'progression' ? 'bg-green-500/20 text-green-400' :
                    'bg-amber-500/20 text-amber-400'
                  }`}>
                    <Lightbulb className="w-4 h-4" />
                  </div>
                  <div>
                    <p className="text-white mb-1">{suggestion.description}</p>
                    {suggestion.example && (
                      <p className="text-sm text-purple-400 mb-2">{suggestion.example}</p>
                    )}
                    {suggestion.theory && (
                      <div className="flex items-start gap-2 mt-2 p-2 bg-gray-700/50 rounded text-xs text-gray-400">
                        <Info className="w-3 h-3 mt-0.5 shrink-0" />
                        {suggestion.theory}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {/* Quick Tips */}
            <div className="mt-6">
              <h3 className="text-sm font-medium text-gray-400 mb-3">Quick Tips</h3>
              <div className="space-y-2 text-sm text-gray-400">
                <p>• Use the ii-V-I progression for jazz-influenced sections</p>
                <p>• Try borrowing chords from the parallel minor for emotional depth</p>
                <p>• The tritone substitution (bII7 for V7) adds jazz sophistication</p>
                <p>• Modal interchange can add unexpected color to your progressions</p>
              </div>
            </div>
          </div>
        )}

        {/* No Analysis State */}
        {!analysis && !isAnalyzing && (
          <div className="text-center py-8 text-gray-500">
            <BookOpen className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>Upload audio to analyze its music theory</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default TheoryAssistant;
