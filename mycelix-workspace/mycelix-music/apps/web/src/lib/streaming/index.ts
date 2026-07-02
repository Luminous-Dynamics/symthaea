// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Edge Streaming
 *
 * High-performance audio streaming:
 * - HLS/DASH adaptive bitrate
 * - WebCodecs for decoding
 * - CDN optimization
 * - Buffer management
 * - Quality adaptation
 */

// ==================== Types ====================

export interface StreamQuality {
  id: string;
  bitrate: number;
  codec: string;
  sampleRate: number;
  label: string;
}

export interface StreamManifest {
  id: string;
  title: string;
  duration: number;
  qualities: StreamQuality[];
  segments: StreamSegment[];
  encryption?: {
    method: 'AES-128' | 'SAMPLE-AES';
    keyUrl: string;
  };
}

export interface StreamSegment {
  index: number;
  url: string;
  duration: number;
  byteRange?: { start: number; end: number };
}

export interface BufferState {
  buffered: number;
  currentTime: number;
  duration: number;
  isBuffering: boolean;
  quality: StreamQuality;
}

export interface StreamStats {
  downloadSpeed: number;
  bufferHealth: number;
  droppedFrames: number;
  latency: number;
  qualitySwitches: number;
}

// ==================== Adaptive Bitrate Manager ====================

export class ABRManager {
  private qualities: StreamQuality[] = [];
  private currentQualityIndex = 0;
  private downloadHistory: number[] = [];
  private bufferTarget = 30; // seconds

  constructor(qualities: StreamQuality[]) {
    this.qualities = qualities.sort((a, b) => a.bitrate - b.bitrate);
  }

  recordDownload(bytesDownloaded: number, timeMs: number): void {
    const speedBps = (bytesDownloaded * 8 * 1000) / timeMs;
    this.downloadHistory.push(speedBps);

    // Keep last 10 measurements
    if (this.downloadHistory.length > 10) {
      this.downloadHistory.shift();
    }
  }

  getRecommendedQuality(bufferLevel: number): StreamQuality {
    const avgSpeed = this.getAverageSpeed();

    // Find highest quality that fits within bandwidth
    let recommendedIndex = 0;
    for (let i = this.qualities.length - 1; i >= 0; i--) {
      if (this.qualities[i].bitrate < avgSpeed * 0.8) {
        recommendedIndex = i;
        break;
      }
    }

    // Adjust based on buffer level
    if (bufferLevel < 5) {
      recommendedIndex = Math.max(0, recommendedIndex - 2);
    } else if (bufferLevel < 10) {
      recommendedIndex = Math.max(0, recommendedIndex - 1);
    } else if (bufferLevel > this.bufferTarget) {
      recommendedIndex = Math.min(this.qualities.length - 1, recommendedIndex + 1);
    }

    // Prevent rapid quality changes
    if (Math.abs(recommendedIndex - this.currentQualityIndex) > 1) {
      recommendedIndex = this.currentQualityIndex + Math.sign(recommendedIndex - this.currentQualityIndex);
    }

    this.currentQualityIndex = recommendedIndex;
    return this.qualities[recommendedIndex];
  }

  private getAverageSpeed(): number {
    if (this.downloadHistory.length === 0) return Infinity;
    return this.downloadHistory.reduce((a, b) => a + b, 0) / this.downloadHistory.length;
  }

  getCurrentQuality(): StreamQuality {
    return this.qualities[this.currentQualityIndex];
  }

  setQuality(qualityId: string): void {
    const index = this.qualities.findIndex(q => q.id === qualityId);
    if (index !== -1) {
      this.currentQualityIndex = index;
    }
  }
}

// ==================== Buffer Manager ====================

export class BufferManager {
  private sourceBuffer: SourceBuffer | null = null;
  private mediaSource: MediaSource;
  private queue: ArrayBuffer[] = [];
  private isUpdating = false;
  private onBufferUpdate?: (state: BufferState) => void;

  constructor(mimeType: string) {
    this.mediaSource = new MediaSource();

    this.mediaSource.addEventListener('sourceopen', () => {
      this.sourceBuffer = this.mediaSource.addSourceBuffer(mimeType);
      this.sourceBuffer.addEventListener('updateend', () => {
        this.isUpdating = false;
        this.processQueue();
      });
    });
  }

  getMediaSourceUrl(): string {
    return URL.createObjectURL(this.mediaSource);
  }

  appendBuffer(data: ArrayBuffer): void {
    this.queue.push(data);
    this.processQueue();
  }

  private processQueue(): void {
    if (this.isUpdating || !this.sourceBuffer || this.queue.length === 0) {
      return;
    }

    try {
      this.isUpdating = true;
      const data = this.queue.shift()!;
      this.sourceBuffer.appendBuffer(data);
    } catch (error) {
      console.error('Buffer append error:', error);
      this.isUpdating = false;
    }
  }

  getBufferedRanges(): Array<{ start: number; end: number }> {
    if (!this.sourceBuffer) return [];

    const ranges: Array<{ start: number; end: number }> = [];
    for (let i = 0; i < this.sourceBuffer.buffered.length; i++) {
      ranges.push({
        start: this.sourceBuffer.buffered.start(i),
        end: this.sourceBuffer.buffered.end(i),
      });
    }
    return ranges;
  }

  removeBuffer(start: number, end: number): void {
    if (this.sourceBuffer && !this.isUpdating) {
      try {
        this.sourceBuffer.remove(start, end);
      } catch (error) {
        console.error('Buffer remove error:', error);
      }
    }
  }

  endOfStream(): void {
    if (this.mediaSource.readyState === 'open') {
      this.mediaSource.endOfStream();
    }
  }

  dispose(): void {
    if (this.sourceBuffer) {
      try {
        this.mediaSource.removeSourceBuffer(this.sourceBuffer);
      } catch {
        // Ignore
      }
    }
  }
}

// ==================== Segment Loader ====================

export class SegmentLoader {
  private abortController: AbortController | null = null;
  private cache: Map<string, ArrayBuffer> = new Map();
  public onProgress?: (loaded: number, total: number) => void;

  async loadSegment(segment: StreamSegment, keyUrl?: string): Promise<ArrayBuffer> {
    // Check cache
    const cached = this.cache.get(segment.url);
    if (cached) return cached;

    this.abortController = new AbortController();

    const headers: HeadersInit = {};
    if (segment.byteRange) {
      headers['Range'] = `bytes=${segment.byteRange.start}-${segment.byteRange.end}`;
    }

    const startTime = performance.now();

    const response = await fetch(segment.url, {
      headers,
      signal: this.abortController.signal,
    });

    if (!response.ok) {
      throw new Error(`Failed to load segment: ${response.status}`);
    }

    const data = await response.arrayBuffer();

    const downloadTime = performance.now() - startTime;
    this.onDownloadComplete?.(data.byteLength, downloadTime);

    // Decrypt if needed
    let decrypted = data;
    if (keyUrl) {
      decrypted = await this.decryptSegment(data, keyUrl, segment.index);
    }

    // Cache segment
    this.cache.set(segment.url, decrypted);

    // Limit cache size
    if (this.cache.size > 20) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    return decrypted;
  }

  private async decryptSegment(
    data: ArrayBuffer,
    keyUrl: string,
    segmentIndex: number
  ): Promise<ArrayBuffer> {
    // Fetch key
    const keyResponse = await fetch(keyUrl);
    const keyData = await keyResponse.arrayBuffer();

    // Create IV from segment index
    const iv = new Uint8Array(16);
    new DataView(iv.buffer).setUint32(12, segmentIndex, false);

    // Import key
    const key = await crypto.subtle.importKey(
      'raw',
      keyData,
      { name: 'AES-CBC' },
      false,
      ['decrypt']
    );

    // Decrypt
    const decrypted = await crypto.subtle.decrypt(
      { name: 'AES-CBC', iv },
      key,
      data
    );

    return decrypted;
  }

  onDownloadComplete?: (bytes: number, timeMs: number) => void;

  abort(): void {
    this.abortController?.abort();
  }

  clearCache(): void {
    this.cache.clear();
  }
}

// ==================== HLS Parser ====================

export class HLSParser {
  async parseManifest(url: string): Promise<StreamManifest> {
    const response = await fetch(url);
    const text = await response.text();

    const lines = text.split('\n').map(l => l.trim()).filter(l => l);

    if (!lines[0].startsWith('#EXTM3U')) {
      throw new Error('Invalid HLS manifest');
    }

    // Check if master or media playlist
    if (lines.some(l => l.includes('#EXT-X-STREAM-INF'))) {
      return this.parseMasterPlaylist(url, lines);
    }
    return this.parseMediaPlaylist(url, lines);
  }

  private async parseMasterPlaylist(baseUrl: string, lines: string[]): Promise<StreamManifest> {
    const qualities: StreamQuality[] = [];
    const base = new URL(baseUrl);

    for (let i = 0; i < lines.length; i++) {
      if (lines[i].startsWith('#EXT-X-STREAM-INF:')) {
        const attrs = this.parseAttributes(lines[i]);
        const url = lines[i + 1];

        qualities.push({
          id: `quality-${qualities.length}`,
          bitrate: parseInt(attrs.BANDWIDTH) || 0,
          codec: attrs.CODECS || 'mp4a.40.2',
          sampleRate: 48000,
          label: this.getBitrateLabel(parseInt(attrs.BANDWIDTH) || 0),
        });
      }
    }

    return {
      id: baseUrl,
      title: 'Stream',
      duration: 0,
      qualities,
      segments: [],
    };
  }

  private parseMediaPlaylist(baseUrl: string, lines: string[]): StreamManifest {
    const segments: StreamSegment[] = [];
    const base = new URL(baseUrl);
    let duration = 0;
    let segmentDuration = 0;
    let encryption: StreamManifest['encryption'];

    for (const line of lines) {
      if (line.startsWith('#EXTINF:')) {
        segmentDuration = parseFloat(line.split(':')[1].split(',')[0]);
        duration += segmentDuration;
      } else if (line.startsWith('#EXT-X-KEY:')) {
        const attrs = this.parseAttributes(line);
        if (attrs.METHOD !== 'NONE') {
          encryption = {
            method: attrs.METHOD as 'AES-128' | 'SAMPLE-AES',
            keyUrl: new URL(attrs.URI.replace(/"/g, ''), base).href,
          };
        }
      } else if (!line.startsWith('#')) {
        segments.push({
          index: segments.length,
          url: new URL(line, base).href,
          duration: segmentDuration,
        });
      }
    }

    return {
      id: baseUrl,
      title: 'Stream',
      duration,
      qualities: [{
        id: 'default',
        bitrate: 256000,
        codec: 'mp4a.40.2',
        sampleRate: 48000,
        label: 'Default',
      }],
      segments,
      encryption,
    };
  }

  private parseAttributes(line: string): Record<string, string> {
    const attrs: Record<string, string> = {};
    const match = line.match(/:(.*)/);
    if (!match) return attrs;

    const pairs = match[1].match(/([A-Z-]+)=("[^"]*"|[^,]*)/g) || [];
    for (const pair of pairs) {
      const [key, value] = pair.split('=');
      attrs[key] = value?.replace(/"/g, '') || '';
    }
    return attrs;
  }

  private getBitrateLabel(bitrate: number): string {
    if (bitrate >= 320000) return 'Ultra';
    if (bitrate >= 256000) return 'High';
    if (bitrate >= 128000) return 'Normal';
    return 'Low';
  }
}

// ==================== Stream Player ====================

export class StreamPlayer {
  private audio: HTMLAudioElement;
  private bufferManager: BufferManager | null = null;
  private segmentLoader: SegmentLoader;
  private abrManager: ABRManager | null = null;
  private manifest: StreamManifest | null = null;
  private currentSegmentIndex = 0;
  private isPlaying = false;
  private stats: StreamStats = {
    downloadSpeed: 0,
    bufferHealth: 0,
    droppedFrames: 0,
    latency: 0,
    qualitySwitches: 0,
  };

  public onStateChange?: (state: BufferState) => void;
  public onError?: (error: Error) => void;
  public onStats?: (stats: StreamStats) => void;

  constructor() {
    this.audio = new Audio();
    this.segmentLoader = new SegmentLoader();

    this.segmentLoader.onDownloadComplete = (bytes, timeMs) => {
      this.abrManager?.recordDownload(bytes, timeMs);
      this.stats.downloadSpeed = (bytes * 8 * 1000) / timeMs;
    };

    this.audio.addEventListener('waiting', () => this.loadNextSegments());
    this.audio.addEventListener('timeupdate', () => this.updateState());
  }

  async load(manifestUrl: string): Promise<void> {
    const parser = new HLSParser();
    this.manifest = await parser.parseManifest(manifestUrl);
    this.abrManager = new ABRManager(this.manifest.qualities);

    // Initialize buffer
    const quality = this.abrManager.getCurrentQuality();
    this.bufferManager = new BufferManager(`audio/mp4; codecs="${quality.codec}"`);
    this.audio.src = this.bufferManager.getMediaSourceUrl();

    // Start loading segments
    await this.loadNextSegments();
  }

  private async loadNextSegments(): Promise<void> {
    if (!this.manifest || !this.bufferManager) return;

    const buffered = this.getBufferedTime();
    const targetBuffer = 30; // seconds

    while (
      this.currentSegmentIndex < this.manifest.segments.length &&
      buffered < targetBuffer
    ) {
      const segment = this.manifest.segments[this.currentSegmentIndex];

      try {
        const data = await this.segmentLoader.loadSegment(
          segment,
          this.manifest.encryption?.keyUrl
        );
        this.bufferManager.appendBuffer(data);
        this.currentSegmentIndex++;
      } catch (error) {
        this.onError?.(error instanceof Error ? error : new Error('Load failed'));
        break;
      }
    }

    if (this.currentSegmentIndex >= this.manifest.segments.length) {
      this.bufferManager.endOfStream();
    }
  }

  private getBufferedTime(): number {
    const ranges = this.bufferManager?.getBufferedRanges() || [];
    if (ranges.length === 0) return 0;

    const currentTime = this.audio.currentTime;
    for (const range of ranges) {
      if (currentTime >= range.start && currentTime <= range.end) {
        return range.end - currentTime;
      }
    }
    return 0;
  }

  private updateState(): void {
    this.onStateChange?.({
      buffered: this.getBufferedTime(),
      currentTime: this.audio.currentTime,
      duration: this.manifest?.duration || 0,
      isBuffering: this.audio.readyState < 3,
      quality: this.abrManager?.getCurrentQuality() || this.manifest?.qualities[0]!,
    });
  }

  play(): void {
    this.audio.play();
    this.isPlaying = true;
  }

  pause(): void {
    this.audio.pause();
    this.isPlaying = false;
  }

  seek(time: number): void {
    this.audio.currentTime = time;

    // Find segment for time
    if (this.manifest) {
      let accumulated = 0;
      for (let i = 0; i < this.manifest.segments.length; i++) {
        accumulated += this.manifest.segments[i].duration;
        if (accumulated >= time) {
          this.currentSegmentIndex = i;
          break;
        }
      }
    }

    this.loadNextSegments();
  }

  setVolume(volume: number): void {
    this.audio.volume = Math.max(0, Math.min(1, volume));
  }

  setQuality(qualityId: string): void {
    this.abrManager?.setQuality(qualityId);
    this.stats.qualitySwitches++;
  }

  getStats(): StreamStats {
    return { ...this.stats };
  }

  dispose(): void {
    this.segmentLoader.abort();
    this.segmentLoader.clearCache();
    this.bufferManager?.dispose();
    this.audio.src = '';
  }
}

export default {
  StreamPlayer,
  ABRManager,
  BufferManager,
  SegmentLoader,
  HLSParser,
};
