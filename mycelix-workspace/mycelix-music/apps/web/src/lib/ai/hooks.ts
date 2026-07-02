// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * AI Model React Hooks
 *
 * React integration for AI models:
 * - useAudioAnalysis - Analyze audio files
 * - useGenreClassification - Real-time genre detection
 * - useBeatDetection - Tempo and beat tracking
 * - useStemSeparation - Audio source separation
 * - useRecommendations - AI-powered recommendations
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  analyzeAudio,
  classifyGenre,
  detectMood,
  recognizeInstruments,
  detectBeats,
  detectKey,
  separateStems,
  computeAudioEmbedding,
  computeSimilarity,
  AudioAnalysis,
  GenreResult,
  MoodResult,
  InstrumentResult,
  BeatResult,
  KeyResult,
  StemResult,
  SimilarityResult,
} from './audio-models';
import { getModelRuntime } from './model-runtime';

// ==================== Types ====================

export interface UseAnalysisOptions {
  sampleRate?: number;
  autoAnalyze?: boolean;
}

export interface AnalysisState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
}

// ==================== useModelRuntime ====================

export function useModelRuntime() {
  const [initialized, setInitialized] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const runtime = getModelRuntime();
    runtime.initialize()
      .then(() => setInitialized(true))
      .catch(setError);

    return () => {
      // Cleanup on unmount
    };
  }, []);

  const getStats = useCallback(() => {
    return getModelRuntime().getStats();
  }, []);

  const clearCache = useCallback(() => {
    getModelRuntime().clearCache();
  }, []);

  return { initialized, error, getStats, clearCache };
}

// ==================== useAudioAnalysis ====================

export function useAudioAnalysis(options: UseAnalysisOptions = {}) {
  const { sampleRate = 44100 } = options;
  const [state, setState] = useState<AnalysisState<AudioAnalysis>>({
    data: null,
    loading: false,
    error: null,
  });

  const analyze = useCallback(async (audio: Float32Array) => {
    setState(s => ({ ...s, loading: true, error: null }));

    try {
      const result = await analyzeAudio(audio, sampleRate);
      setState({ data: result, loading: false, error: null });
      return result;
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Analysis failed');
      setState({ data: null, loading: false, error: err });
      throw err;
    }
  }, [sampleRate]);

  const analyzeFromFile = useCallback(async (file: File) => {
    const arrayBuffer = await file.arrayBuffer();
    const audioContext = new AudioContext({ sampleRate });
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const channelData = audioBuffer.getChannelData(0);
    await audioContext.close();

    return analyze(channelData);
  }, [analyze, sampleRate]);

  const analyzeFromUrl = useCallback(async (url: string) => {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const audioContext = new AudioContext({ sampleRate });
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const channelData = audioBuffer.getChannelData(0);
    await audioContext.close();

    return analyze(channelData);
  }, [analyze, sampleRate]);

  return {
    ...state,
    analyze,
    analyzeFromFile,
    analyzeFromUrl,
  };
}

// ==================== useGenreClassification ====================

export function useGenreClassification(options: UseAnalysisOptions = {}) {
  const { sampleRate = 44100 } = options;
  const [state, setState] = useState<AnalysisState<GenreResult>>({
    data: null,
    loading: false,
    error: null,
  });

  const classify = useCallback(async (audio: Float32Array) => {
    setState(s => ({ ...s, loading: true, error: null }));

    try {
      const result = await classifyGenre(audio, sampleRate);
      setState({ data: result, loading: false, error: null });
      return result;
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Classification failed');
      setState({ data: null, loading: false, error: err });
      throw err;
    }
  }, [sampleRate]);

  return { ...state, classify };
}

// ==================== useMoodDetection ====================

export function useMoodDetection(options: UseAnalysisOptions = {}) {
  const { sampleRate = 44100 } = options;
  const [state, setState] = useState<AnalysisState<MoodResult>>({
    data: null,
    loading: false,
    error: null,
  });

  const detect = useCallback(async (audio: Float32Array) => {
    setState(s => ({ ...s, loading: true, error: null }));

    try {
      const result = await detectMood(audio, sampleRate);
      setState({ data: result, loading: false, error: null });
      return result;
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Detection failed');
      setState({ data: null, loading: false, error: err });
      throw err;
    }
  }, [sampleRate]);

  return { ...state, detect };
}

// ==================== useInstrumentRecognition ====================

export function useInstrumentRecognition(options: UseAnalysisOptions = {}) {
  const { sampleRate = 44100 } = options;
  const [state, setState] = useState<AnalysisState<InstrumentResult>>({
    data: null,
    loading: false,
    error: null,
  });

  const recognize = useCallback(async (audio: Float32Array, threshold = 0.3) => {
    setState(s => ({ ...s, loading: true, error: null }));

    try {
      const result = await recognizeInstruments(audio, sampleRate, threshold);
      setState({ data: result, loading: false, error: null });
      return result;
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Recognition failed');
      setState({ data: null, loading: false, error: err });
      throw err;
    }
  }, [sampleRate]);

  return { ...state, recognize };
}

// ==================== useBeatDetection ====================

export function useBeatDetection(options: UseAnalysisOptions = {}) {
  const { sampleRate = 44100 } = options;
  const [state, setState] = useState<AnalysisState<BeatResult>>({
    data: null,
    loading: false,
    error: null,
  });

  const detect = useCallback(async (audio: Float32Array) => {
    setState(s => ({ ...s, loading: true, error: null }));

    try {
      const result = await detectBeats(audio, sampleRate);
      setState({ data: result, loading: false, error: null });
      return result;
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Detection failed');
      setState({ data: null, loading: false, error: err });
      throw err;
    }
  }, [sampleRate]);

  return { ...state, detect };
}

// ==================== useKeyDetection ====================

export function useKeyDetection(options: UseAnalysisOptions = {}) {
  const { sampleRate = 44100 } = options;
  const [state, setState] = useState<AnalysisState<KeyResult>>({
    data: null,
    loading: false,
    error: null,
  });

  const detect = useCallback(async (audio: Float32Array) => {
    setState(s => ({ ...s, loading: true, error: null }));

    try {
      const result = await detectKey(audio, sampleRate);
      setState({ data: result, loading: false, error: null });
      return result;
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Detection failed');
      setState({ data: null, loading: false, error: err });
      throw err;
    }
  }, [sampleRate]);

  return { ...state, detect };
}

// ==================== useStemSeparation ====================

export function useStemSeparation(options: UseAnalysisOptions = {}) {
  const { sampleRate = 44100 } = options;
  const [state, setState] = useState<AnalysisState<StemResult>>({
    data: null,
    loading: false,
    error: null,
  });
  const [progress, setProgress] = useState(0);

  const separate = useCallback(async (audio: Float32Array) => {
    setState(s => ({ ...s, loading: true, error: null }));
    setProgress(0);

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress(p => Math.min(p + 10, 90));
      }, 500);

      const result = await separateStems(audio, sampleRate);

      clearInterval(progressInterval);
      setProgress(100);

      setState({ data: result, loading: false, error: null });
      return result;
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Separation failed');
      setState({ data: null, loading: false, error: err });
      setProgress(0);
      throw err;
    }
  }, [sampleRate]);

  return { ...state, progress, separate };
}

// ==================== useAudioSimilarity ====================

export function useAudioSimilarity(options: UseAnalysisOptions = {}) {
  const { sampleRate = 44100 } = options;
  const [state, setState] = useState<AnalysisState<SimilarityResult>>({
    data: null,
    loading: false,
    error: null,
  });

  const compare = useCallback(async (audio1: Float32Array, audio2: Float32Array) => {
    setState(s => ({ ...s, loading: true, error: null }));

    try {
      const result = await computeSimilarity(audio1, audio2, sampleRate);
      setState({ data: result, loading: false, error: null });
      return result;
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Comparison failed');
      setState({ data: null, loading: false, error: err });
      throw err;
    }
  }, [sampleRate]);

  return { ...state, compare };
}

// ==================== useAudioEmbedding ====================

export function useAudioEmbedding(options: UseAnalysisOptions = {}) {
  const { sampleRate = 44100 } = options;
  const [state, setState] = useState<AnalysisState<Float32Array>>({
    data: null,
    loading: false,
    error: null,
  });

  const embed = useCallback(async (audio: Float32Array) => {
    setState(s => ({ ...s, loading: true, error: null }));

    try {
      const result = await computeAudioEmbedding(audio, sampleRate);
      setState({ data: result, loading: false, error: null });
      return result;
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Embedding failed');
      setState({ data: null, loading: false, error: err });
      throw err;
    }
  }, [sampleRate]);

  return { ...state, embed };
}

// ==================== useRealtimeAnalysis ====================

export interface RealtimeAnalysisOptions {
  sampleRate?: number;
  bufferSize?: number;
  analyzeInterval?: number;
  onGenre?: (result: GenreResult) => void;
  onMood?: (result: MoodResult) => void;
  onBeat?: (beat: number) => void;
}

export function useRealtimeAnalysis(options: RealtimeAnalysisOptions = {}) {
  const {
    sampleRate = 44100,
    bufferSize = 4096,
    analyzeInterval = 1000,
    onGenre,
    onMood,
    onBeat,
  } = options;

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentGenre, setCurrentGenre] = useState<GenreResult | null>(null);
  const [currentMood, setCurrentMood] = useState<MoodResult | null>(null);
  const [currentTempo, setCurrentTempo] = useState<number>(0);

  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const bufferRef = useRef<Float32Array>(new Float32Array(bufferSize * 10));
  const bufferIndexRef = useRef(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const startAnalysis = useCallback(async (stream: MediaStream) => {
    audioContextRef.current = new AudioContext({ sampleRate });
    analyserRef.current = audioContextRef.current.createAnalyser();
    analyserRef.current.fftSize = bufferSize;

    const source = audioContextRef.current.createMediaStreamSource(stream);
    source.connect(analyserRef.current);

    const dataArray = new Float32Array(bufferSize);

    // Collect audio data
    const collectData = () => {
      if (!analyserRef.current || !isAnalyzing) return;

      analyserRef.current.getFloatTimeDomainData(dataArray);

      // Add to rolling buffer
      const remaining = bufferRef.current.length - bufferIndexRef.current;
      if (remaining >= bufferSize) {
        bufferRef.current.set(dataArray, bufferIndexRef.current);
        bufferIndexRef.current += bufferSize;
      } else {
        // Wrap around
        bufferRef.current.set(dataArray.slice(0, remaining), bufferIndexRef.current);
        bufferRef.current.set(dataArray.slice(remaining), 0);
        bufferIndexRef.current = bufferSize - remaining;
      }

      requestAnimationFrame(collectData);
    };

    // Periodic analysis
    intervalRef.current = setInterval(async () => {
      if (!isAnalyzing) return;

      const audio = new Float32Array(bufferRef.current);

      // Genre classification
      try {
        const genreResult = await classifyGenre(audio, sampleRate);
        setCurrentGenre(genreResult);
        onGenre?.(genreResult);
      } catch {
        // Ignore errors during real-time analysis
      }

      // Mood detection
      try {
        const moodResult = await detectMood(audio, sampleRate);
        setCurrentMood(moodResult);
        onMood?.(moodResult);
      } catch {
        // Ignore errors
      }

      // Beat detection
      try {
        const beatResult = await detectBeats(audio, sampleRate);
        setCurrentTempo(beatResult.tempo);

        // Emit beat events
        beatResult.beatTimes.forEach(time => {
          onBeat?.(time);
        });
      } catch {
        // Ignore errors
      }
    }, analyzeInterval);

    setIsAnalyzing(true);
    collectData();
  }, [sampleRate, bufferSize, analyzeInterval, isAnalyzing, onGenre, onMood, onBeat]);

  const stopAnalysis = useCallback(() => {
    setIsAnalyzing(false);

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    analyserRef.current = null;
  }, []);

  useEffect(() => {
    return () => {
      stopAnalysis();
    };
  }, [stopAnalysis]);

  return {
    isAnalyzing,
    currentGenre,
    currentMood,
    currentTempo,
    startAnalysis,
    stopAnalysis,
  };
}

// ==================== useSmartRecommendations ====================

export interface RecommendationOptions {
  basedOn: 'listening_history' | 'current_track' | 'mood' | 'time_of_day';
  limit?: number;
}

export interface Recommendation {
  trackId: string;
  score: number;
  reason: string;
}

export function useSmartRecommendations(options: RecommendationOptions) {
  const { basedOn, limit = 20 } = options;
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);

  const generateRecommendations = useCallback(async (
    currentTrackEmbedding?: Float32Array,
    listeningHistory?: string[],
    mood?: MoodResult
  ) => {
    setLoading(true);

    try {
      // This would call your recommendation API with embeddings
      const recs: Recommendation[] = [];

      // Placeholder - real implementation would:
      // 1. Get embeddings for candidate tracks
      // 2. Compute similarity scores
      // 3. Apply collaborative filtering
      // 4. Factor in time of day, mood, etc.

      if (basedOn === 'mood' && mood) {
        // Recommend tracks matching current mood
        recs.push({
          trackId: 'track-1',
          score: 0.95,
          reason: `Matches your ${mood.primaryMood} mood`,
        });
      }

      if (basedOn === 'current_track' && currentTrackEmbedding) {
        // Recommend similar tracks
        recs.push({
          trackId: 'track-2',
          score: 0.88,
          reason: 'Similar sound profile',
        });
      }

      setRecommendations(recs.slice(0, limit));
    } finally {
      setLoading(false);
    }
  }, [basedOn, limit]);

  return { recommendations, loading, generateRecommendations };
}

export default {
  useModelRuntime,
  useAudioAnalysis,
  useGenreClassification,
  useMoodDetection,
  useInstrumentRecognition,
  useBeatDetection,
  useKeyDetection,
  useStemSeparation,
  useAudioSimilarity,
  useAudioEmbedding,
  useRealtimeAnalysis,
  useSmartRecommendations,
};
