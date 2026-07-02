// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  useSpatialAudio,
  Position3D,
  RoomPreset,
  SpatialSource,
} from '@/hooks/useSpatialAudio';
import {
  Volume2,
  VolumeX,
  Play,
  Pause,
  Settings,
  Move,
  Headphones,
  Box,
  Music,
  MapPin,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Grid,
} from 'lucide-react';

interface SpatialVenueProps {
  className?: string;
}

// Demo sound sources
const DEMO_SOURCES = [
  { id: 'drums', name: 'Drums', position: { x: -2, y: 0, z: 0 } },
  { id: 'bass', name: 'Bass', position: { x: 0, y: 0, z: -2 } },
  { id: 'vocals', name: 'Vocals', position: { x: 0, y: 0, z: 2 } },
  { id: 'synth', name: 'Synth', position: { x: 2, y: 0, z: 0 } },
];

export function SpatialVenue({ className = '' }: SpatialVenueProps) {
  const {
    isInitialized,
    listener,
    sources,
    roomAcoustics,
    masterVolume,
    isEnabled,
    roomPresets,
    initialize,
    addSource,
    removeSource,
    updateSourcePosition,
    playSource,
    stopSource,
    updateListenerPosition,
    updateListenerOrientation,
    setRoomAcoustics,
    setMasterVolume,
    toggleEnabled,
    createVenue,
  } = useSpatialAudio();

  const [showSettings, setShowSettings] = useState(false);
  const [selectedSource, setSelectedSource] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [showGrid, setShowGrid] = useState(true);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Initialize on mount
  useEffect(() => {
    initialize();
    createVenue('Virtual Concert Hall', 'concert-hall', { width: 30, height: 10, depth: 40 });
  }, [initialize, createVenue]);

  // Add demo sources
  useEffect(() => {
    if (isInitialized && sources.length === 0) {
      DEMO_SOURCES.forEach(demo => {
        addSource(demo.id, demo.name, demo.position);
      });
    }
  }, [isInitialized, sources.length, addSource]);

  // Draw venue map
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const width = canvas.width;
      const height = canvas.height;
      const centerX = width / 2;
      const centerY = height / 2;
      const scale = 40 * zoom;

      // Clear
      ctx.fillStyle = '#0a0a0f';
      ctx.fillRect(0, 0, width, height);

      // Draw grid
      if (showGrid) {
        ctx.strokeStyle = '#1a1a2e';
        ctx.lineWidth = 1;

        for (let x = -10; x <= 10; x++) {
          const screenX = centerX + x * scale;
          ctx.beginPath();
          ctx.moveTo(screenX, 0);
          ctx.lineTo(screenX, height);
          ctx.stroke();
        }

        for (let y = -10; y <= 10; y++) {
          const screenY = centerY + y * scale;
          ctx.beginPath();
          ctx.moveTo(0, screenY);
          ctx.lineTo(width, screenY);
          ctx.stroke();
        }
      }

      // Draw venue boundary
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(
        centerX - 5 * scale,
        centerY - 5 * scale,
        10 * scale,
        10 * scale
      );
      ctx.setLineDash([]);

      // Draw sources
      sources.forEach(source => {
        const x = centerX + source.position.x * scale;
        const y = centerY - source.position.z * scale;
        const isSelected = selectedSource === source.id;

        // Outer glow for playing sources
        if (source.isPlaying) {
          const gradient = ctx.createRadialGradient(x, y, 0, x, y, 30);
          gradient.addColorStop(0, 'rgba(139, 92, 246, 0.3)');
          gradient.addColorStop(1, 'rgba(139, 92, 246, 0)');
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(x, y, 30, 0, Math.PI * 2);
          ctx.fill();
        }

        // Source circle
        ctx.beginPath();
        ctx.arc(x, y, isSelected ? 15 : 12, 0, Math.PI * 2);
        ctx.fillStyle = source.isPlaying ? '#8B5CF6' : '#4c4c6d';
        ctx.fill();

        // Border
        if (isSelected) {
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        // Icon
        ctx.fillStyle = '#fff';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(source.name[0], x, y);

        // Label
        ctx.font = '11px sans-serif';
        ctx.fillStyle = '#888';
        ctx.fillText(source.name, x, y + 25);

        // Distance/angle indicators
        ctx.font = '10px sans-serif';
        ctx.fillStyle = '#555';
        ctx.fillText(
          `${source.distance.toFixed(1)}m`,
          x,
          y + 38
        );
      });

      // Draw listener
      const listenerX = centerX + listener.position.x * scale;
      const listenerY = centerY - listener.position.z * scale;

      // Listener direction indicator
      const dirAngle = (listener.orientation.yaw - 90) * (Math.PI / 180);
      ctx.beginPath();
      ctx.moveTo(listenerX, listenerY);
      ctx.lineTo(
        listenerX + Math.cos(dirAngle) * 30,
        listenerY + Math.sin(dirAngle) * 30
      );
      ctx.strokeStyle = '#10B981';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Listener hearing cone
      ctx.beginPath();
      ctx.moveTo(listenerX, listenerY);
      ctx.arc(
        listenerX,
        listenerY,
        60,
        dirAngle - Math.PI / 4,
        dirAngle + Math.PI / 4
      );
      ctx.closePath();
      ctx.fillStyle = 'rgba(16, 185, 129, 0.1)';
      ctx.fill();

      // Listener circle
      ctx.beginPath();
      ctx.arc(listenerX, listenerY, 10, 0, Math.PI * 2);
      ctx.fillStyle = '#10B981';
      ctx.fill();

      // Headphone icon
      ctx.fillStyle = '#fff';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('👤', listenerX, listenerY);

      requestAnimationFrame(draw);
    };

    draw();
  }, [sources, listener, selectedSource, zoom, showGrid]);

  // Handle canvas click
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const scale = 40 * zoom;

    // Check if clicked on a source
    for (const source of sources) {
      const sourceX = centerX + source.position.x * scale;
      const sourceY = centerY - source.position.z * scale;
      const distance = Math.sqrt(Math.pow(x - sourceX, 2) + Math.pow(y - sourceY, 2));

      if (distance < 15) {
        setSelectedSource(source.id);
        return;
      }
    }

    // Otherwise, move listener
    const worldX = (x - centerX) / scale;
    const worldZ = -(y - centerY) / scale;
    updateListenerPosition({ x: worldX, y: 0, z: worldZ });
  };

  // Handle source drag
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging || !selectedSource) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const scale = 40 * zoom;

    const worldX = (x - centerX) / scale;
    const worldZ = -(y - centerY) / scale;

    updateSourcePosition(selectedSource, { x: worldX, y: 0, z: worldZ });
  };

  // Handle listener rotation
  const handleRotation = (delta: number) => {
    updateListenerOrientation({
      ...listener.orientation,
      yaw: (listener.orientation.yaw + delta + 360) % 360,
    });
  };

  return (
    <div className={`bg-gray-900 rounded-xl overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-green-500 to-teal-500 rounded-lg flex items-center justify-center">
            <Headphones className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">3D Spatial Venue</h2>
            <p className="text-sm text-gray-400">
              {roomAcoustics.preset.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={toggleEnabled}
            className={`p-2 rounded-lg transition-colors ${
              isEnabled ? 'bg-green-500 text-white' : 'bg-gray-800 text-gray-400'
            }`}
          >
            {isEnabled ? <Volume2 className="w-5 h-5" /> : <VolumeX className="w-5 h-5" />}
          </button>

          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`p-2 rounded-lg transition-colors ${
              showSettings ? 'bg-purple-500 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'
            }`}
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </div>

      <div className="flex">
        {/* Canvas Area */}
        <div className="flex-1 relative" ref={containerRef}>
          <canvas
            ref={canvasRef}
            width={600}
            height={400}
            className="w-full h-96 cursor-crosshair"
            onClick={handleCanvasClick}
            onMouseDown={() => selectedSource && setIsDragging(true)}
            onMouseUp={() => setIsDragging(false)}
            onMouseLeave={() => setIsDragging(false)}
            onMouseMove={handleCanvasMouseMove}
          />

          {/* Overlay Controls */}
          <div className="absolute top-4 left-4 flex gap-2">
            <button
              onClick={() => setZoom(z => Math.min(2, z + 0.25))}
              className="p-2 bg-gray-800/80 text-gray-400 rounded-lg hover:text-white"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={() => setZoom(z => Math.max(0.5, z - 0.25))}
              className="p-2 bg-gray-800/80 text-gray-400 rounded-lg hover:text-white"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <button
              onClick={() => setShowGrid(!showGrid)}
              className={`p-2 rounded-lg ${
                showGrid ? 'bg-purple-500 text-white' : 'bg-gray-800/80 text-gray-400'
              }`}
            >
              <Grid className="w-4 h-4" />
            </button>
          </div>

          {/* Rotation Controls */}
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-4 bg-gray-800/80 rounded-lg px-4 py-2">
            <button
              onClick={() => handleRotation(-15)}
              className="p-1 text-gray-400 hover:text-white"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
            <span className="text-sm text-gray-400">
              {Math.round(listener.orientation.yaw)}°
            </span>
            <button
              onClick={() => handleRotation(15)}
              className="p-1 text-gray-400 hover:text-white"
              style={{ transform: 'scaleX(-1)' }}
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          {/* Legend */}
          <div className="absolute top-4 right-4 bg-gray-800/80 rounded-lg p-3 text-xs">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span className="text-gray-400">Listener</span>
            </div>
            <div className="flex items-center gap-2 mb-1">
              <div className="w-3 h-3 rounded-full bg-purple-500" />
              <span className="text-gray-400">Playing Source</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gray-500" />
              <span className="text-gray-400">Stopped Source</span>
            </div>
          </div>
        </div>

        {/* Side Panel */}
        {showSettings && (
          <div className="w-72 border-l border-gray-800 p-4">
            {/* Room Acoustics */}
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-400 mb-3">Room Acoustics</h3>
              <div className="grid grid-cols-2 gap-2">
                {roomPresets.map(preset => (
                  <button
                    key={preset}
                    onClick={() => setRoomAcoustics(preset)}
                    className={`px-3 py-2 text-sm rounded-lg transition-colors ${
                      roomAcoustics.preset === preset
                        ? 'bg-purple-500 text-white'
                        : 'bg-gray-800 text-gray-400 hover:text-white'
                    }`}
                  >
                    {preset.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </button>
                ))}
              </div>
            </div>

            {/* Master Volume */}
            <div className="mb-6">
              <label className="flex justify-between text-sm text-gray-400 mb-2">
                <span>Master Volume</span>
                <span>{Math.round(masterVolume * 100)}%</span>
              </label>
              <input
                type="range"
                min="0"
                max="100"
                value={masterVolume * 100}
                onChange={(e) => setMasterVolume(Number(e.target.value) / 100)}
                className="w-full accent-purple-500"
              />
            </div>

            {/* Sources */}
            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-3">Sound Sources</h3>
              <div className="space-y-2">
                {sources.map(source => (
                  <div
                    key={source.id}
                    className={`flex items-center justify-between p-3 rounded-lg transition-colors ${
                      selectedSource === source.id
                        ? 'bg-purple-500/20 border border-purple-500'
                        : 'bg-gray-800 hover:bg-gray-700'
                    }`}
                    onClick={() => setSelectedSource(source.id)}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                        source.isPlaying ? 'bg-purple-500' : 'bg-gray-700'
                      }`}>
                        <Music className="w-4 h-4 text-white" />
                      </div>
                      <div>
                        <p className="text-sm text-white">{source.name}</p>
                        <p className="text-xs text-gray-500">
                          {source.distance.toFixed(1)}m away
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        source.isPlaying ? stopSource(source.id) : playSource(source.id);
                      }}
                      className="p-2 bg-gray-700 rounded-lg hover:bg-gray-600"
                    >
                      {source.isPlaying ? (
                        <Pause className="w-4 h-4 text-white" />
                      ) : (
                        <Play className="w-4 h-4 text-white" />
                      )}
                    </button>
                  </div>
                ))}
              </div>
            </div>

            {/* Selected Source Details */}
            {selectedSource && (
              <div className="mt-4 p-3 bg-gray-800 rounded-lg">
                <h4 className="text-sm font-medium text-white mb-2">Position</h4>
                <div className="grid grid-cols-3 gap-2 text-sm">
                  <div className="text-center">
                    <span className="text-gray-500">X</span>
                    <p className="text-white">
                      {sources.find(s => s.id === selectedSource)?.position.x.toFixed(2)}
                    </p>
                  </div>
                  <div className="text-center">
                    <span className="text-gray-500">Y</span>
                    <p className="text-white">
                      {sources.find(s => s.id === selectedSource)?.position.y.toFixed(2)}
                    </p>
                  </div>
                  <div className="text-center">
                    <span className="text-gray-500">Z</span>
                    <p className="text-white">
                      {sources.find(s => s.id === selectedSource)?.position.z.toFixed(2)}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default SpatialVenue;
