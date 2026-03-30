// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { usePlayerStore, equalizerPresets } from '@/store/playerStore';
import { X, Sliders } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';

interface EqualizerProps {
  isOpen: boolean;
  onClose: () => void;
}

export function Equalizer({ isOpen, onClose }: EqualizerProps) {
  const {
    audioRef,
    equalizerEnabled,
    equalizerPreset,
    equalizerBands,
    setEqualizerEnabled,
    setEqualizerPreset,
    setEqualizerBand,
  } = usePlayerStore();

  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const filtersRef = useRef<BiquadFilterNode[]>([]);

  // Initialize Web Audio API
  useEffect(() => {
    if (!audioRef || !equalizerEnabled) return;

    // Create audio context if needed
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }

    const ctx = audioContextRef.current;

    // Create source from audio element (only once)
    if (!sourceRef.current) {
      sourceRef.current = ctx.createMediaElementSource(audioRef);
    }

    // Create filters for each band
    const frequencies = equalizerBands.map(b => b.frequency);
    filtersRef.current = frequencies.map((freq, i) => {
      const filter = ctx.createBiquadFilter();

      if (i === 0) {
        filter.type = 'lowshelf';
      } else if (i === frequencies.length - 1) {
        filter.type = 'highshelf';
      } else {
        filter.type = 'peaking';
      }

      filter.frequency.value = freq;
      filter.Q.value = 1;
      filter.gain.value = equalizerBands[i].gain;

      return filter;
    });

    // Connect filters in series
    sourceRef.current.disconnect();

    let lastNode: AudioNode = sourceRef.current;
    filtersRef.current.forEach(filter => {
      lastNode.connect(filter);
      lastNode = filter;
    });

    lastNode.connect(ctx.destination);

    return () => {
      // Cleanup: reconnect source directly to destination
      if (sourceRef.current && audioContextRef.current) {
        filtersRef.current.forEach(f => f.disconnect());
        sourceRef.current.disconnect();
        sourceRef.current.connect(audioContextRef.current.destination);
      }
    };
  }, [audioRef, equalizerEnabled]);

  // Update filter gains when bands change
  useEffect(() => {
    filtersRef.current.forEach((filter, i) => {
      if (equalizerBands[i]) {
        filter.gain.value = equalizerBands[i].gain;
      }
    });
  }, [equalizerBands]);

  if (!isOpen) return null;

  const presetNames = Object.keys(equalizerPresets);

  const formatFrequency = (freq: number) => {
    if (freq >= 1000) {
      return `${freq / 1000}kHz`;
    }
    return `${freq}Hz`;
  };

  return (
    <div className="fixed right-0 top-0 bottom-20 w-96 bg-gray-900/95 backdrop-blur-md border-l border-white/10 z-40 overflow-hidden flex flex-col shadow-2xl">
      {/* Header */}
      <div className="p-4 border-b border-white/10 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Sliders className="w-5 h-5 text-purple-400" />
          <h3 className="font-semibold text-lg">Equalizer</h3>
        </div>
        <button
          onClick={onClose}
          className="p-1 text-gray-400 hover:text-white transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {/* Enable Toggle */}
        <div className="flex items-center justify-between mb-6">
          <span className="text-sm font-medium">Enable Equalizer</span>
          <button
            onClick={() => setEqualizerEnabled(!equalizerEnabled)}
            className={`relative w-12 h-6 rounded-full transition-colors ${
              equalizerEnabled ? 'bg-purple-500' : 'bg-gray-600'
            }`}
          >
            <div
              className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
                equalizerEnabled ? 'left-7' : 'left-1'
              }`}
            />
          </button>
        </div>

        {/* Presets */}
        <div className="mb-6">
          <label className="text-sm text-gray-400 mb-2 block">Preset</label>
          <div className="flex flex-wrap gap-2">
            {presetNames.map(preset => (
              <button
                key={preset}
                onClick={() => setEqualizerPreset(preset)}
                disabled={!equalizerEnabled}
                className={`px-3 py-1.5 rounded-full text-sm capitalize transition-colors ${
                  equalizerPreset === preset
                    ? 'bg-purple-500 text-white'
                    : 'bg-white/10 text-gray-300 hover:bg-white/20'
                } ${!equalizerEnabled ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {preset}
              </button>
            ))}
          </div>
        </div>

        {/* Frequency Bands */}
        <div className="space-y-4">
          <label className="text-sm text-gray-400 block">Frequency Bands</label>
          <div className="flex gap-2 justify-between">
            {equalizerBands.map((band, index) => (
              <div
                key={band.frequency}
                className="flex flex-col items-center gap-2"
              >
                {/* Vertical Slider */}
                <div className="relative h-32 w-6 flex items-center justify-center">
                  <input
                    type="range"
                    min="-12"
                    max="12"
                    step="1"
                    value={band.gain}
                    onChange={(e) => setEqualizerBand(band.frequency, parseFloat(e.target.value))}
                    disabled={!equalizerEnabled}
                    className="absolute h-32 w-6 appearance-none bg-transparent cursor-pointer"
                    style={{
                      writingMode: 'vertical-lr',
                      direction: 'rtl',
                    }}
                  />
                  {/* Track visualization */}
                  <div className="absolute h-full w-1 bg-white/20 rounded-full pointer-events-none">
                    <div
                      className={`absolute bottom-1/2 left-0 w-full rounded-full transition-all ${
                        equalizerEnabled ? 'bg-purple-500' : 'bg-gray-500'
                      }`}
                      style={{
                        height: `${Math.abs(band.gain) / 24 * 100}%`,
                        transform: band.gain >= 0 ? 'translateY(0)' : 'translateY(100%)',
                        transformOrigin: band.gain >= 0 ? 'bottom' : 'top',
                      }}
                    />
                  </div>
                </div>

                {/* Gain value */}
                <span className={`text-xs tabular-nums ${
                  equalizerEnabled ? 'text-gray-300' : 'text-gray-500'
                }`}>
                  {band.gain > 0 ? '+' : ''}{band.gain}dB
                </span>

                {/* Frequency label */}
                <span className={`text-xs ${
                  equalizerEnabled ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  {formatFrequency(band.frequency)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Reset Button */}
        <button
          onClick={() => setEqualizerPreset('flat')}
          disabled={!equalizerEnabled}
          className={`mt-6 w-full py-2 rounded-lg text-sm transition-colors ${
            equalizerEnabled
              ? 'bg-white/10 hover:bg-white/20 text-white'
              : 'bg-white/5 text-gray-500 cursor-not-allowed'
          }`}
        >
          Reset to Flat
        </button>
      </div>
    </div>
  );
}
