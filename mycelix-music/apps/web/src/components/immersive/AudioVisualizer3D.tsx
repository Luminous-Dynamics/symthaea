// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  useImmersiveVisuals,
  VisualizationMode,
  ColorScheme,
} from '@/hooks/useImmersiveVisuals';
import {
  Play,
  Pause,
  Settings,
  Maximize2,
  Minimize2,
  Palette,
  Sliders,
  Zap,
  Waves,
  Box,
  BarChart3,
} from 'lucide-react';

interface AudioVisualizer3DProps {
  audioSource?: AudioNode;
  className?: string;
}

const MODE_ICONS: Record<VisualizationMode, React.ReactNode> = {
  particles: <Zap className="w-4 h-4" />,
  waveform: <Waves className="w-4 h-4" />,
  spectrum: <BarChart3 className="w-4 h-4" />,
  galaxy: <Box className="w-4 h-4" />,
  tunnel: <Box className="w-4 h-4" />,
  terrain: <Box className="w-4 h-4" />,
  abstract: <Box className="w-4 h-4" />,
};

export function AudioVisualizer3D({ audioSource, className = '' }: AudioVisualizer3DProps) {
  const {
    isInitialized,
    isRunning,
    settings,
    fps,
    audioAnalysis,
    colorSchemes,
    visualizationModes,
    initialize,
    start,
    stop,
    updateSettings,
  } = useImmersiveVisuals(audioSource);

  const [showSettings, setShowSettings] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Initialize visualizer
  useEffect(() => {
    if (canvasRef.current && !isInitialized) {
      initialize(canvasRef.current);
    }
  }, [initialize, isInitialized]);

  // Handle fullscreen
  const toggleFullscreen = useCallback(() => {
    if (!containerRef.current) return;

    if (!isFullscreen) {
      containerRef.current.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  }, [isFullscreen]);

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  // Resize canvas to container
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const resizeObserver = new ResizeObserver(() => {
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
    });

    resizeObserver.observe(container);
    return () => resizeObserver.disconnect();
  }, []);

  return (
    <div
      ref={containerRef}
      className={`relative bg-black rounded-xl overflow-hidden ${className}`}
    >
      {/* Canvas */}
      <canvas
        ref={canvasRef}
        className="w-full h-full"
      />

      {/* Overlay UI */}
      <div className="absolute inset-0 pointer-events-none">
        {/* Top Bar */}
        <div className="absolute top-0 left-0 right-0 p-4 flex items-center justify-between pointer-events-auto">
          <div className="flex items-center gap-3">
            <button
              onClick={isRunning ? stop : start}
              className="p-3 bg-purple-500 text-white rounded-full hover:bg-purple-600 transition-colors"
            >
              {isRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            </button>

            {/* FPS Counter */}
            {isRunning && (
              <div className="px-3 py-1.5 bg-black/50 rounded-lg text-sm text-gray-300">
                {fps} FPS
              </div>
            )}

            {/* Audio Levels */}
            {audioAnalysis && (
              <div className="flex gap-2 px-3 py-1.5 bg-black/50 rounded-lg">
                <div className="w-2 rounded-full bg-gray-700 h-6 overflow-hidden">
                  <div
                    className="w-full bg-purple-500 transition-all"
                    style={{ height: `${audioAnalysis.bassLevel * 100}%`, marginTop: 'auto' }}
                  />
                </div>
                <div className="w-2 rounded-full bg-gray-700 h-6 overflow-hidden">
                  <div
                    className="w-full bg-pink-500 transition-all"
                    style={{ height: `${audioAnalysis.midLevel * 100}%`, marginTop: 'auto' }}
                  />
                </div>
                <div className="w-2 rounded-full bg-gray-700 h-6 overflow-hidden">
                  <div
                    className="w-full bg-blue-500 transition-all"
                    style={{ height: `${audioAnalysis.trebleLevel * 100}%`, marginTop: 'auto' }}
                  />
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`p-2 rounded-lg transition-colors ${
                showSettings ? 'bg-purple-500 text-white' : 'bg-black/50 text-gray-300 hover:text-white'
              }`}
            >
              <Settings className="w-5 h-5" />
            </button>

            <button
              onClick={toggleFullscreen}
              className="p-2 bg-black/50 text-gray-300 rounded-lg hover:text-white transition-colors"
            >
              {isFullscreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="absolute top-16 right-4 w-80 bg-gray-900/95 backdrop-blur border border-gray-800 rounded-xl p-4 pointer-events-auto">
            <h3 className="text-white font-semibold mb-4">Visualizer Settings</h3>

            {/* Visualization Mode */}
            <div className="mb-4">
              <label className="block text-sm text-gray-400 mb-2">Mode</label>
              <div className="grid grid-cols-4 gap-2">
                {visualizationModes.slice(0, 4).map(mode => (
                  <button
                    key={mode}
                    onClick={() => updateSettings({ mode })}
                    className={`flex flex-col items-center gap-1 p-2 rounded-lg transition-colors ${
                      settings.mode === mode
                        ? 'bg-purple-500 text-white'
                        : 'bg-gray-800 text-gray-400 hover:text-white'
                    }`}
                  >
                    {MODE_ICONS[mode]}
                    <span className="text-xs capitalize">{mode}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Color Scheme */}
            <div className="mb-4">
              <label className="block text-sm text-gray-400 mb-2">Color Scheme</label>
              <div className="grid grid-cols-4 gap-2">
                {colorSchemes.map(scheme => (
                  <button
                    key={scheme}
                    onClick={() => updateSettings({ colorScheme: scheme })}
                    className={`p-2 rounded-lg text-xs capitalize transition-colors ${
                      settings.colorScheme === scheme
                        ? 'bg-purple-500 text-white'
                        : 'bg-gray-800 text-gray-400 hover:text-white'
                    }`}
                  >
                    {scheme}
                  </button>
                ))}
              </div>
            </div>

            {/* Sliders */}
            <div className="space-y-3">
              <div>
                <label className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Intensity</span>
                  <span>{Math.round(settings.intensity * 100)}%</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={settings.intensity * 100}
                  onChange={(e) => updateSettings({ intensity: Number(e.target.value) / 100 })}
                  className="w-full accent-purple-500"
                />
              </div>

              <div>
                <label className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Speed</span>
                  <span>{settings.speed.toFixed(1)}x</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="200"
                  value={settings.speed * 100}
                  onChange={(e) => updateSettings({ speed: Number(e.target.value) / 100 })}
                  className="w-full accent-purple-500"
                />
              </div>

              <div>
                <label className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Bloom</span>
                  <span>{Math.round(settings.bloom * 100)}%</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={settings.bloom * 100}
                  onChange={(e) => updateSettings({ bloom: Number(e.target.value) / 100 })}
                  className="w-full accent-purple-500"
                />
              </div>

              <div>
                <label className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Motion Blur</span>
                  <span>{Math.round(settings.motionBlur * 100)}%</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={settings.motionBlur * 100}
                  onChange={(e) => updateSettings({ motionBlur: Number(e.target.value) / 100 })}
                  className="w-full accent-purple-500"
                />
              </div>

              <div>
                <label className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Bass Response</span>
                  <span>{settings.bassResponse.toFixed(1)}x</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="200"
                  value={settings.bassResponse * 100}
                  onChange={(e) => updateSettings({ bassResponse: Number(e.target.value) / 100 })}
                  className="w-full accent-purple-500"
                />
              </div>

              <div>
                <label className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Particle Count</span>
                  <span>{settings.particleCount}</span>
                </label>
                <input
                  type="range"
                  min="100"
                  max="5000"
                  step="100"
                  value={settings.particleCount}
                  onChange={(e) => updateSettings({ particleCount: Number(e.target.value) })}
                  className="w-full accent-purple-500"
                />
              </div>

              {/* Toggles */}
              <label className="flex items-center justify-between p-2 rounded-lg hover:bg-gray-800">
                <span className="text-sm text-gray-300">Camera Rotation</span>
                <input
                  type="checkbox"
                  checked={settings.cameraRotation}
                  onChange={(e) => updateSettings({ cameraRotation: e.target.checked })}
                  className="w-4 h-4 accent-purple-500"
                />
              </label>
            </div>
          </div>
        )}

        {/* Beat Indicator */}
        {audioAnalysis?.beat && (
          <div className="absolute inset-0 border-4 border-purple-500 pointer-events-none animate-pulse" />
        )}

        {/* Mode Label */}
        <div className="absolute bottom-4 left-4">
          <div className="px-3 py-1.5 bg-black/50 rounded-lg text-sm text-gray-300 capitalize">
            {settings.mode} • {settings.colorScheme}
          </div>
        </div>
      </div>
    </div>
  );
}

export default AudioVisualizer3D;
