// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Advanced Visuals Hook
 *
 * GPU-accelerated visualizations:
 * - WebGPU shader-based visualizers
 * - VR/AR integration
 * - AI music video generation
 * - Reactive album art
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// Types
export interface ShaderConfig {
  id: string;
  name: string;
  category: 'geometric' | 'fluid' | 'particle' | 'fractal' | 'waveform' | 'custom';
  vertexShader: string;
  fragmentShader: string;
  uniforms: ShaderUniform[];
  audioReactive: AudioReactiveConfig;
}

export interface ShaderUniform {
  name: string;
  type: 'float' | 'vec2' | 'vec3' | 'vec4' | 'mat4' | 'sampler2D';
  value: number | number[] | Float32Array;
  audioBinding?: 'bass' | 'mid' | 'high' | 'volume' | 'beat' | 'spectrum';
  range?: { min: number; max: number };
}

export interface AudioReactiveConfig {
  bassMultiplier: number;
  midMultiplier: number;
  highMultiplier: number;
  beatSensitivity: number;
  smoothing: number;
}

export interface VRConfig {
  enabled: boolean;
  mode: 'immersive' | 'inline';
  environment: 'void' | 'space' | 'nature' | 'abstract' | 'custom';
  controller: 'gaze' | 'pointer' | 'hand';
  position: { x: number; y: number; z: number };
  roomScale: boolean;
}

export interface MusicVideoConfig {
  style: 'abstract' | 'realistic' | 'anime' | 'psychedelic' | 'minimal' | 'cinematic';
  aspectRatio: '16:9' | '9:16' | '1:1' | '4:3';
  fps: 24 | 30 | 60;
  duration: 'full' | 'clip';
  colorPalette: string[];
  motionIntensity: number;
  sceneChanges: 'beat-sync' | 'measure-sync' | 'phrase-sync';
  elements: VideoElement[];
}

export interface VideoElement {
  type: 'shape' | 'particle' | 'text' | 'image' | 'ai-generated';
  audioBinding?: string;
  position: { x: number; y: number; z?: number };
  scale: number;
  rotation: number;
  color?: string;
  content?: string;
}

export interface AlbumArtConfig {
  style: 'generated' | 'reactive' | 'static';
  baseImage?: string;
  overlays: ArtOverlay[];
  colorScheme: 'from-audio' | 'custom';
  customColors?: string[];
  animationSpeed: number;
}

export interface ArtOverlay {
  type: 'waveform' | 'spectrum' | 'particles' | 'glow' | 'text';
  opacity: number;
  blendMode: 'normal' | 'multiply' | 'screen' | 'overlay' | 'add';
  position: 'center' | 'bottom' | 'top' | 'fill';
  audioReactive: boolean;
}

export interface AdvancedVisualsState {
  isInitialized: boolean;
  isRendering: boolean;
  gpuAvailable: boolean;
  vrAvailable: boolean;
  activeShader: ShaderConfig | null;
  vrSession: any | null;
  generatingVideo: boolean;
  videoProgress: number;
  fps: number;
  error: string | null;
}

// Built-in shaders
const BUILT_IN_SHADERS: Omit<ShaderConfig, 'vertexShader' | 'fragmentShader'>[] = [
  {
    id: 'audio-waves',
    name: 'Audio Waves',
    category: 'waveform',
    uniforms: [
      { name: 'u_color1', type: 'vec3', value: [0.5, 0.0, 1.0] },
      { name: 'u_color2', type: 'vec3', value: [0.0, 0.5, 1.0] },
      { name: 'u_thickness', type: 'float', value: 0.02, range: { min: 0.001, max: 0.1 } },
    ],
    audioReactive: {
      bassMultiplier: 1.5,
      midMultiplier: 1.0,
      highMultiplier: 0.8,
      beatSensitivity: 0.7,
      smoothing: 0.8,
    },
  },
  {
    id: 'particle-storm',
    name: 'Particle Storm',
    category: 'particle',
    uniforms: [
      { name: 'u_particleCount', type: 'float', value: 10000 },
      { name: 'u_particleSize', type: 'float', value: 2.0, audioBinding: 'volume' },
      { name: 'u_speed', type: 'float', value: 1.0, audioBinding: 'beat' },
    ],
    audioReactive: {
      bassMultiplier: 2.0,
      midMultiplier: 1.0,
      highMultiplier: 1.5,
      beatSensitivity: 0.9,
      smoothing: 0.6,
    },
  },
  {
    id: 'fluid-sim',
    name: 'Fluid Simulation',
    category: 'fluid',
    uniforms: [
      { name: 'u_viscosity', type: 'float', value: 0.5, audioBinding: 'bass' },
      { name: 'u_diffusion', type: 'float', value: 0.3 },
      { name: 'u_color', type: 'vec3', value: [1.0, 0.2, 0.5] },
    ],
    audioReactive: {
      bassMultiplier: 1.8,
      midMultiplier: 1.2,
      highMultiplier: 0.5,
      beatSensitivity: 0.8,
      smoothing: 0.7,
    },
  },
  {
    id: 'fractal-zoom',
    name: 'Fractal Zoom',
    category: 'fractal',
    uniforms: [
      { name: 'u_zoom', type: 'float', value: 1.0, audioBinding: 'mid' },
      { name: 'u_iterations', type: 'float', value: 100 },
      { name: 'u_colorShift', type: 'float', value: 0.0, audioBinding: 'high' },
    ],
    audioReactive: {
      bassMultiplier: 1.0,
      midMultiplier: 1.5,
      highMultiplier: 1.2,
      beatSensitivity: 0.5,
      smoothing: 0.9,
    },
  },
];

export function useAdvancedVisuals() {
  const [state, setState] = useState<AdvancedVisualsState>({
    isInitialized: false,
    isRendering: false,
    gpuAvailable: false,
    vrAvailable: false,
    activeShader: null,
    vrSession: null,
    generatingVideo: false,
    videoProgress: 0,
    fps: 0,
    error: null,
  });

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const gpuDeviceRef = useRef<GPUDevice | null>(null);
  const gpuContextRef = useRef<GPUCanvasContext | null>(null);
  const animationFrameRef = useRef<number>(0);
  const audioAnalyserRef = useRef<AnalyserNode | null>(null);
  const lastFrameTimeRef = useRef<number>(0);

  /**
   * Initialize WebGPU
   */
  const initialize = useCallback(async (
    canvas: HTMLCanvasElement
  ): Promise<boolean> => {
    canvasRef.current = canvas;

    try {
      // Check WebGPU support
      if (!navigator.gpu) {
        // Fall back to WebGL
        console.log('WebGPU not available, using WebGL fallback');
        setState(prev => ({ ...prev, isInitialized: true, gpuAvailable: false }));
        return true;
      }

      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error('No GPU adapter found');
      }

      const device = await adapter.requestDevice();
      gpuDeviceRef.current = device;

      const context = canvas.getContext('webgpu');
      if (!context) {
        throw new Error('Failed to get WebGPU context');
      }

      gpuContextRef.current = context;

      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({
        device,
        format,
        alphaMode: 'premultiplied',
      });

      // Check VR support
      const vrSupported = 'xr' in navigator && await (navigator as any).xr?.isSessionSupported('immersive-vr');

      setState(prev => ({
        ...prev,
        isInitialized: true,
        gpuAvailable: true,
        vrAvailable: vrSupported,
      }));

      return true;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isInitialized: true,
        gpuAvailable: false,
        error: error instanceof Error ? error.message : 'Initialization failed',
      }));
      return false;
    }
  }, []);

  /**
   * Set audio source for visualization
   */
  const setAudioSource = useCallback((analyser: AnalyserNode) => {
    audioAnalyserRef.current = analyser;
  }, []);

  /**
   * Start rendering shader
   */
  const startShader = useCallback(async (shaderId: string): Promise<boolean> => {
    const shaderTemplate = BUILT_IN_SHADERS.find(s => s.id === shaderId);
    if (!shaderTemplate) {
      setState(prev => ({ ...prev, error: 'Shader not found' }));
      return false;
    }

    // Load full shader code
    const shader: ShaderConfig = {
      ...shaderTemplate,
      vertexShader: await loadShaderCode(shaderId, 'vertex'),
      fragmentShader: await loadShaderCode(shaderId, 'fragment'),
    };

    setState(prev => ({
      ...prev,
      activeShader: shader,
      isRendering: true,
    }));

    // Start render loop
    startRenderLoop(shader);

    return true;
  }, []);

  /**
   * Start custom shader
   */
  const startCustomShader = useCallback((shader: ShaderConfig): boolean => {
    setState(prev => ({
      ...prev,
      activeShader: shader,
      isRendering: true,
    }));

    startRenderLoop(shader);
    return true;
  }, []);

  /**
   * Render loop
   */
  const startRenderLoop = useCallback((shader: ShaderConfig) => {
    const render = (timestamp: number) => {
      // Calculate FPS
      const deltaTime = timestamp - lastFrameTimeRef.current;
      lastFrameTimeRef.current = timestamp;
      const fps = Math.round(1000 / deltaTime);

      // Get audio data
      let audioData: Float32Array | null = null;
      if (audioAnalyserRef.current) {
        audioData = new Float32Array(audioAnalyserRef.current.frequencyBinCount);
        audioAnalyserRef.current.getFloatFrequencyData(audioData);
      }

      // Render frame
      if (state.gpuAvailable && gpuDeviceRef.current && gpuContextRef.current) {
        renderFrameWebGPU(shader, audioData, timestamp);
      } else if (canvasRef.current) {
        renderFrameWebGL(shader, audioData, timestamp);
      }

      setState(prev => ({ ...prev, fps }));
      animationFrameRef.current = requestAnimationFrame(render);
    };

    animationFrameRef.current = requestAnimationFrame(render);
  }, [state.gpuAvailable]);

  /**
   * Stop rendering
   */
  const stopRendering = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    setState(prev => ({
      ...prev,
      isRendering: false,
      activeShader: null,
    }));
  }, []);

  /**
   * Enter VR mode
   */
  const enterVR = useCallback(async (config: VRConfig): Promise<boolean> => {
    if (!state.vrAvailable) {
      setState(prev => ({ ...prev, error: 'VR not available' }));
      return false;
    }

    try {
      const xr = (navigator as any).xr;
      const session = await xr.requestSession('immersive-vr', {
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['hand-tracking'],
      });

      setState(prev => ({ ...prev, vrSession: session }));

      session.addEventListener('end', () => {
        setState(prev => ({ ...prev, vrSession: null }));
      });

      // Set up VR rendering
      await setupVRRendering(session, config);

      return true;
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to enter VR',
      }));
      return false;
    }
  }, [state.vrAvailable]);

  /**
   * Exit VR mode
   */
  const exitVR = useCallback(async () => {
    if (state.vrSession) {
      await state.vrSession.end();
      setState(prev => ({ ...prev, vrSession: null }));
    }
  }, [state.vrSession]);

  /**
   * Generate AI music video
   */
  const generateMusicVideo = useCallback(async (
    audioBuffer: AudioBuffer,
    config: MusicVideoConfig
  ): Promise<Blob | null> => {
    setState(prev => ({
      ...prev,
      generatingVideo: true,
      videoProgress: 0,
    }));

    try {
      // Analyze audio for beat/structure detection
      const audioAnalysis = await analyzeAudioForVideo(audioBuffer);

      // Generate frames
      const frames: ImageData[] = [];
      const totalFrames = Math.ceil(audioBuffer.duration * config.fps);

      for (let i = 0; i < totalFrames; i++) {
        const time = i / config.fps;
        const frame = await generateVideoFrame(config, audioAnalysis, time);
        frames.push(frame);

        setState(prev => ({
          ...prev,
          videoProgress: ((i + 1) / totalFrames) * 100,
        }));
      }

      // Encode video
      const videoBlob = await encodeVideo(frames, config.fps);

      setState(prev => ({
        ...prev,
        generatingVideo: false,
        videoProgress: 100,
      }));

      return videoBlob;
    } catch (error) {
      setState(prev => ({
        ...prev,
        generatingVideo: false,
        error: error instanceof Error ? error.message : 'Video generation failed',
      }));
      return null;
    }
  }, []);

  /**
   * Generate reactive album art
   */
  const generateAlbumArt = useCallback(async (
    audioBuffer: AudioBuffer,
    config: AlbumArtConfig
  ): Promise<Blob | null> => {
    try {
      // Analyze audio for color extraction
      const colors = config.colorScheme === 'from-audio'
        ? await extractColorsFromAudio(audioBuffer)
        : config.customColors || ['#8B5CF6', '#EC4899', '#3B82F6'];

      // Generate base image with AI
      const baseImage = config.baseImage || await generateAIImage(config.style, colors);

      // Apply overlays
      const canvas = document.createElement('canvas');
      canvas.width = 3000;
      canvas.height = 3000;
      const ctx = canvas.getContext('2d')!;

      // Draw base
      const img = await loadImage(baseImage);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Apply overlays
      for (const overlay of config.overlays) {
        await applyOverlay(ctx, overlay, audioBuffer, canvas.width, canvas.height);
      }

      // Export as blob
      return new Promise((resolve) => {
        canvas.toBlob(resolve, 'image/png', 1.0);
      });
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Art generation failed',
      }));
      return null;
    }
  }, []);

  /**
   * Update shader uniform
   */
  const updateUniform = useCallback((name: string, value: number | number[]) => {
    setState(prev => {
      if (!prev.activeShader) return prev;

      return {
        ...prev,
        activeShader: {
          ...prev.activeShader,
          uniforms: prev.activeShader.uniforms.map(u =>
            u.name === name ? { ...u, value } : u
          ),
        },
      };
    });
  }, []);

  /**
   * Cleanup
   */
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return {
    ...state,
    builtInShaders: BUILT_IN_SHADERS,
    initialize,
    setAudioSource,
    startShader,
    startCustomShader,
    stopRendering,
    enterVR,
    exitVR,
    generateMusicVideo,
    generateAlbumArt,
    updateUniform,
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

async function loadShaderCode(shaderId: string, type: 'vertex' | 'fragment'): Promise<string> {
  // Would load from shader files
  if (type === 'vertex') {
    return `
      @vertex
      fn main(@builtin(vertex_index) VertexIndex : u32) -> @builtin(position) vec4<f32> {
        var pos = array<vec2<f32>, 6>(
          vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
          vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
        );
        return vec4<f32>(pos[VertexIndex], 0.0, 1.0);
      }
    `;
  }

  return `
    @fragment
    fn main() -> @location(0) vec4<f32> {
      return vec4(1.0, 0.0, 1.0, 1.0);
    }
  `;
}

function renderFrameWebGPU(
  shader: ShaderConfig,
  audioData: Float32Array | null,
  timestamp: number
): void {
  // WebGPU rendering implementation
}

function renderFrameWebGL(
  shader: ShaderConfig,
  audioData: Float32Array | null,
  timestamp: number
): void {
  // WebGL fallback implementation
}

async function setupVRRendering(session: any, config: VRConfig): Promise<void> {
  // VR setup implementation
}

async function analyzeAudioForVideo(buffer: AudioBuffer): Promise<any> {
  // Audio analysis for video sync
  return {};
}

async function generateVideoFrame(
  config: MusicVideoConfig,
  analysis: any,
  time: number
): Promise<ImageData> {
  // Generate single video frame
  const canvas = new OffscreenCanvas(1920, 1080);
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, 1920, 1080);
  return ctx.getImageData(0, 0, 1920, 1080);
}

async function encodeVideo(frames: ImageData[], fps: number): Promise<Blob> {
  // Video encoding
  return new Blob([], { type: 'video/mp4' });
}

async function extractColorsFromAudio(buffer: AudioBuffer): Promise<string[]> {
  // Extract colors based on audio characteristics
  return ['#8B5CF6', '#EC4899', '#3B82F6', '#10B981'];
}

async function generateAIImage(style: string, colors: string[]): Promise<string> {
  // AI image generation
  return '';
}

async function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

async function applyOverlay(
  ctx: CanvasRenderingContext2D,
  overlay: ArtOverlay,
  audio: AudioBuffer,
  width: number,
  height: number
): Promise<void> {
  // Apply visual overlay
}

export default useAdvancedVisuals;
