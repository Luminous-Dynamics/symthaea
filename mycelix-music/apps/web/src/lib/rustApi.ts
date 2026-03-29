// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Rust Core API Client
 *
 * TypeScript client for the Mycelix Rust backend (mycelix-api).
 * Handles audio analysis, transcoding, sessions, and similarity search.
 */

const RUST_API_BASE = process.env.NEXT_PUBLIC_RUST_API_URL || 'http://localhost:3000';

// === Types ===

export interface TrackInfo {
  id: string;
  title?: string;
  artist?: string;
  duration: number;
  sample_rate: number;
  channels: number;
  analyzed: boolean;
}

export interface AnalysisResult {
  track_id: string;
  genre: string;
  genre_confidence: number;
  mood: string;
  bpm: number;
  key: string;
  beats: number[];
  segments: string[];
}

export interface SimilarTrack {
  id: string;
  title?: string;
  artist?: string;
  similarity: number;
}

export interface SearchResult {
  id: string;
  title?: string;
  artist?: string;
  score: number;
}

export interface SearchQuery {
  text?: string;
  genre?: string;
  mood?: string;
  bpm_min?: number;
  bpm_max?: number;
  key?: string;
  limit?: number;
}

export interface SessionInfo {
  id: string;
  status: string;
}

export interface SessionDetail {
  id: string;
  current_track?: string;
  position: number;
  playback: 'Playing' | 'Paused' | 'Stopped';
  volume: number;
  peers: string[];
}

export interface FormatInfo {
  name: string;
  description: string;
  extensions: string[];
}

export interface FormatsResponse {
  encoders: FormatInfo[];
  decoders: FormatInfo[];
}

export type QualityPreset = 'low' | 'medium' | 'high' | 'lossless';

// === API Response Wrapper ===

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HTTP ${response.status}: ${text}`);
  }

  const json: ApiResponse<T> = await response.json();

  if (!json.success || !json.data) {
    throw new Error(json.error || 'Unknown error');
  }

  return json.data;
}

// === API Client Class ===

class RustApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = RUST_API_BASE) {
    this.baseUrl = baseUrl;
  }

  // --- Health Check ---

  async health(): Promise<{ status: string; version: string }> {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }

  // --- Track Operations ---

  /**
   * Upload a new track for processing
   */
  async uploadTrack(file: File, metadata?: { title?: string; artist?: string }): Promise<TrackInfo> {
    const formData = new FormData();
    formData.append('file', file);

    if (metadata?.title) {
      formData.append('title', metadata.title);
    }
    if (metadata?.artist) {
      formData.append('artist', metadata.artist);
    }

    const response = await fetch(`${this.baseUrl}/api/tracks`, {
      method: 'POST',
      body: formData,
    });

    return handleResponse<TrackInfo>(response);
  }

  /**
   * Get track information by ID
   */
  async getTrack(trackId: string): Promise<TrackInfo> {
    const response = await fetch(`${this.baseUrl}/api/tracks/${trackId}`);
    return handleResponse<TrackInfo>(response);
  }

  /**
   * Delete a track
   */
  async deleteTrack(trackId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/tracks/${trackId}`, {
      method: 'DELETE',
    });
    await handleResponse<void>(response);
  }

  /**
   * Analyze a track (genre, mood, BPM, key, beats)
   */
  async analyzeTrack(trackId: string): Promise<AnalysisResult> {
    const response = await fetch(`${this.baseUrl}/api/tracks/${trackId}/analyze`, {
      method: 'POST',
    });
    return handleResponse<AnalysisResult>(response);
  }

  /**
   * Get stream URL for a track
   */
  getStreamUrl(trackId: string, codec: string = 'opus', quality: QualityPreset = 'high'): string {
    return `${this.baseUrl}/api/tracks/${trackId}/stream?codec=${codec}&quality=${quality}`;
  }

  // --- Analysis ---

  /**
   * Upload and analyze audio in one request
   */
  async analyzeAudio(file: File): Promise<AnalysisResult> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/api/analyze/audio`, {
      method: 'POST',
      body: formData,
    });

    return handleResponse<AnalysisResult>(response);
  }

  /**
   * Batch analyze multiple tracks
   */
  async batchAnalyze(trackIds: string[]): Promise<AnalysisResult[]> {
    const response = await fetch(`${this.baseUrl}/api/analyze/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ track_ids: trackIds }),
    });

    return handleResponse<AnalysisResult[]>(response);
  }

  // --- Search ---

  /**
   * Find tracks similar to a given track
   */
  async findSimilar(trackId: string, limit: number = 10): Promise<SimilarTrack[]> {
    const response = await fetch(
      `${this.baseUrl}/api/search/similar/${trackId}?limit=${limit}`
    );
    return handleResponse<SimilarTrack[]>(response);
  }

  /**
   * Search tracks by query parameters
   */
  async search(query: SearchQuery): Promise<SearchResult[]> {
    const response = await fetch(`${this.baseUrl}/api/search/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(query),
    });

    return handleResponse<SearchResult[]>(response);
  }

  // --- Sessions ---

  /**
   * Create a new playback session
   */
  async createSession(): Promise<SessionInfo> {
    const response = await fetch(`${this.baseUrl}/api/sessions`, {
      method: 'POST',
    });
    return handleResponse<SessionInfo>(response);
  }

  /**
   * Get session details
   */
  async getSession(sessionId: string): Promise<SessionDetail> {
    const response = await fetch(`${this.baseUrl}/api/sessions/${sessionId}`);
    return handleResponse<SessionDetail>(response);
  }

  /**
   * Close a session
   */
  async closeSession(sessionId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/sessions/${sessionId}`, {
      method: 'DELETE',
    });
    await handleResponse<void>(response);
  }

  /**
   * Play in session
   */
  async sessionPlay(sessionId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/sessions/${sessionId}/play`, {
      method: 'POST',
    });
    await handleResponse<void>(response);
  }

  /**
   * Pause in session
   */
  async sessionPause(sessionId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/sessions/${sessionId}/pause`, {
      method: 'POST',
    });
    await handleResponse<void>(response);
  }

  /**
   * Seek in session
   */
  async sessionSeek(sessionId: string, position: number): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/sessions/${sessionId}/seek`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ position }),
    });
    await handleResponse<void>(response);
  }

  // --- Codec Operations ---

  /**
   * Transcode audio to a different format
   */
  async transcode(
    file: File,
    toCodec: string,
    quality: QualityPreset = 'high',
    fromCodec: string = 'auto'
  ): Promise<Blob> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('from', fromCodec);
    formData.append('to', toCodec);
    formData.append('quality', quality);

    const response = await fetch(`${this.baseUrl}/api/codec/transcode`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Transcode failed: ${text}`);
    }

    return response.blob();
  }

  /**
   * Get supported audio formats
   */
  async getFormats(): Promise<FormatsResponse> {
    const response = await fetch(`${this.baseUrl}/api/codec/formats`);
    return handleResponse<FormatsResponse>(response);
  }
}

// Export singleton instance
export const rustApi = new RustApiClient();

// Export class for custom instances
export { RustApiClient };

export default rustApi;
