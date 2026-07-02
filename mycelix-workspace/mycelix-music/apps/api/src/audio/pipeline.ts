// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Audio Processing Pipeline
 *
 * Handles transcoding, waveform generation, audio analysis,
 * fingerprinting, and streaming optimization.
 */

import { spawn, ChildProcess } from 'child_process';
import { createReadStream, createWriteStream, promises as fs } from 'fs';
import { pipeline as streamPipeline } from 'stream/promises';
import * as path from 'path';
import * as crypto from 'crypto';
import { Pool } from 'pg';
import { S3Client, PutObjectCommand, GetObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { PriorityQueue } from '../queue/priority-queue';
import { getLogger } from '../logging';
import { getMetrics } from '../metrics';
import { EventEmitter } from '../events';

const logger = getLogger();

// ==================== Types ====================

export interface AudioFile {
  id: string;
  originalPath: string;
  originalFormat: string;
  duration: number;
  sampleRate: number;
  channels: number;
  bitDepth: number;
  fileSize: number;
}

export interface TranscodeOptions {
  format: 'mp3' | 'aac' | 'opus' | 'flac';
  quality: 'low' | 'medium' | 'high' | 'lossless';
  normalize?: boolean;
  loudnessTarget?: number; // LUFS
}

export interface TranscodedFile {
  format: string;
  quality: string;
  path: string;
  url: string;
  bitrate: number;
  fileSize: number;
}

export interface WaveformData {
  peaks: number[];
  duration: number;
  sampleRate: number;
  channels: number;
  resolution: number;
}

export interface AudioAnalysis {
  bpm: number;
  key: string;
  scale: 'major' | 'minor';
  energy: number;
  danceability: number;
  speechiness: number;
  instrumentalness: number;
  valence: number;
  loudness: number;
  acousticness: number;
}

export interface AudioFingerprint {
  hash: string;
  duration: number;
  algorithm: string;
}

export interface ProcessingJob {
  songId: string;
  inputPath: string;
  tasks: ProcessingTask[];
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  error?: string;
  startedAt?: Date;
  completedAt?: Date;
}

export type ProcessingTask =
  | { type: 'transcode'; options: TranscodeOptions }
  | { type: 'waveform'; resolution?: number }
  | { type: 'analyze' }
  | { type: 'fingerprint' }
  | { type: 'normalize'; targetLufs?: number };

// ==================== Quality Presets ====================

const QUALITY_PRESETS = {
  mp3: {
    low: { bitrate: 128, codec: 'libmp3lame' },
    medium: { bitrate: 192, codec: 'libmp3lame' },
    high: { bitrate: 320, codec: 'libmp3lame' },
    lossless: { bitrate: 320, codec: 'libmp3lame' },
  },
  aac: {
    low: { bitrate: 96, codec: 'aac' },
    medium: { bitrate: 128, codec: 'aac' },
    high: { bitrate: 256, codec: 'aac' },
    lossless: { bitrate: 256, codec: 'aac' },
  },
  opus: {
    low: { bitrate: 64, codec: 'libopus' },
    medium: { bitrate: 96, codec: 'libopus' },
    high: { bitrate: 160, codec: 'libopus' },
    lossless: { bitrate: 160, codec: 'libopus' },
  },
  flac: {
    low: { compression: 5 },
    medium: { compression: 5 },
    high: { compression: 8 },
    lossless: { compression: 8 },
  },
};

// ==================== Audio Pipeline ====================

export class AudioPipeline {
  private s3: S3Client;
  private queue: PriorityQueue;
  private activeJobs: Map<string, ProcessingJob> = new Map();
  private tempDir: string;
  private outputBucket: string;

  constructor(
    private pool: Pool,
    private events: EventEmitter,
    config: {
      s3Client: S3Client;
      queue: PriorityQueue;
      tempDir?: string;
      outputBucket: string;
    }
  ) {
    this.s3 = config.s3Client;
    this.queue = config.queue;
    this.tempDir = config.tempDir || '/tmp/audio-processing';
    this.outputBucket = config.outputBucket;

    // Register job handler
    this.queue.registerHandler('audio-processing', this.processJob.bind(this));

    // Metrics
    const metrics = getMetrics();
    metrics.createCounter('audio_processing_jobs_total', 'Audio processing jobs', ['type', 'status']);
    metrics.createHistogram('audio_processing_duration_seconds', 'Audio processing duration', ['type']);
    metrics.createGauge('audio_processing_queue_size', 'Audio processing queue size', []);
  }

  // ==================== Job Submission ====================

  /**
   * Submit a song for processing
   */
  async submitForProcessing(
    songId: string,
    inputPath: string,
    tasks: ProcessingTask[] = [
      { type: 'transcode', options: { format: 'mp3', quality: 'high' } },
      { type: 'transcode', options: { format: 'mp3', quality: 'medium' } },
      { type: 'transcode', options: { format: 'mp3', quality: 'low' } },
      { type: 'transcode', options: { format: 'opus', quality: 'high' } },
      { type: 'waveform', resolution: 1000 },
      { type: 'analyze' },
      { type: 'fingerprint' },
    ]
  ): Promise<string> {
    const jobId = `audio_${songId}_${Date.now()}`;

    const job: ProcessingJob = {
      songId,
      inputPath,
      tasks,
      status: 'pending',
      progress: 0,
    };

    this.activeJobs.set(jobId, job);

    // Queue the job
    await this.queue.enqueue(
      jobId,
      'audio-processing',
      { jobId, songId, inputPath, tasks },
      2 // Medium-high priority
    );

    logger.info('Audio processing job submitted', { jobId, songId, taskCount: tasks.length });

    return jobId;
  }

  /**
   * Get job status
   */
  getJobStatus(jobId: string): ProcessingJob | undefined {
    return this.activeJobs.get(jobId);
  }

  // ==================== Job Processing ====================

  /**
   * Process a job (called by queue handler)
   */
  private async processJob(data: {
    jobId: string;
    songId: string;
    inputPath: string;
    tasks: ProcessingTask[];
  }): Promise<void> {
    const { jobId, songId, inputPath, tasks } = data;
    const job = this.activeJobs.get(jobId);

    if (!job) {
      logger.warn('Job not found', { jobId });
      return;
    }

    job.status = 'processing';
    job.startedAt = new Date();

    const workDir = path.join(this.tempDir, jobId);
    await fs.mkdir(workDir, { recursive: true });

    const metrics = getMetrics();

    try {
      // Download input file if remote
      let localInputPath = inputPath;
      if (inputPath.startsWith('s3://') || inputPath.startsWith('http')) {
        localInputPath = path.join(workDir, 'input.audio');
        await this.downloadFile(inputPath, localInputPath);
      }

      // Probe input file
      const audioInfo = await this.probeAudio(localInputPath);

      const results: Record<string, unknown> = {
        original: audioInfo,
        transcoded: [] as TranscodedFile[],
      };

      // Process each task
      for (let i = 0; i < tasks.length; i++) {
        const task = tasks[i];
        const taskStart = Date.now();

        try {
          switch (task.type) {
            case 'transcode':
              const transcoded = await this.transcode(
                localInputPath,
                workDir,
                songId,
                task.options
              );
              (results.transcoded as TranscodedFile[]).push(transcoded);
              break;

            case 'waveform':
              results.waveform = await this.generateWaveform(
                localInputPath,
                task.resolution || 1000
              );
              break;

            case 'analyze':
              results.analysis = await this.analyzeAudio(localInputPath);
              break;

            case 'fingerprint':
              results.fingerprint = await this.generateFingerprint(localInputPath);
              break;

            case 'normalize':
              await this.normalizeAudio(
                localInputPath,
                path.join(workDir, 'normalized.audio'),
                task.targetLufs || -14
              );
              break;
          }

          const taskDuration = (Date.now() - taskStart) / 1000;
          metrics.recordHistogram('audio_processing_duration_seconds', taskDuration, { type: task.type });

        } catch (taskError) {
          logger.error(`Task failed: ${task.type}`, taskError as Error, { jobId, songId });
          metrics.incCounter('audio_processing_jobs_total', { type: task.type, status: 'failed' });
        }

        job.progress = ((i + 1) / tasks.length) * 100;
        this.events.emit('audio:progress', { jobId, progress: job.progress });
      }

      // Update database with results
      await this.saveResults(songId, results);

      job.status = 'completed';
      job.completedAt = new Date();

      metrics.incCounter('audio_processing_jobs_total', { type: 'all', status: 'completed' });

      this.events.emit('audio:completed', { jobId, songId, results });

      logger.info('Audio processing completed', { jobId, songId });

    } catch (error) {
      job.status = 'failed';
      job.error = (error as Error).message;

      metrics.incCounter('audio_processing_jobs_total', { type: 'all', status: 'failed' });

      this.events.emit('audio:failed', { jobId, songId, error: job.error });

      logger.error('Audio processing failed', error as Error, { jobId, songId });

      throw error;
    } finally {
      // Cleanup temp files
      try {
        await fs.rm(workDir, { recursive: true, force: true });
      } catch (e) {
        // Ignore cleanup errors
      }
    }
  }

  // ==================== Audio Operations ====================

  /**
   * Probe audio file for metadata
   */
  async probeAudio(inputPath: string): Promise<AudioFile> {
    const output = await this.runFFprobe([
      '-v', 'quiet',
      '-print_format', 'json',
      '-show_format',
      '-show_streams',
      inputPath,
    ]);

    const data = JSON.parse(output);
    const audioStream = data.streams.find((s: any) => s.codec_type === 'audio');

    return {
      id: crypto.randomUUID(),
      originalPath: inputPath,
      originalFormat: data.format.format_name,
      duration: parseFloat(data.format.duration),
      sampleRate: parseInt(audioStream.sample_rate),
      channels: audioStream.channels,
      bitDepth: audioStream.bits_per_sample || 16,
      fileSize: parseInt(data.format.size),
    };
  }

  /**
   * Transcode audio to target format
   */
  async transcode(
    inputPath: string,
    workDir: string,
    songId: string,
    options: TranscodeOptions
  ): Promise<TranscodedFile> {
    const preset = QUALITY_PRESETS[options.format][options.quality];
    const ext = options.format === 'opus' ? 'webm' : options.format;
    const outputFileName = `${songId}_${options.quality}.${ext}`;
    const localOutput = path.join(workDir, outputFileName);

    const ffmpegArgs: string[] = [
      '-i', inputPath,
      '-y',
      '-vn', // No video
    ];

    if (options.format === 'flac') {
      ffmpegArgs.push(
        '-c:a', 'flac',
        '-compression_level', String((preset as any).compression)
      );
    } else {
      ffmpegArgs.push(
        '-c:a', (preset as any).codec,
        '-b:a', `${(preset as any).bitrate}k`
      );
    }

    // Audio normalization
    if (options.normalize) {
      const targetLufs = options.loudnessTarget || -14;
      ffmpegArgs.push(
        '-af', `loudnorm=I=${targetLufs}:TP=-1.5:LRA=11`
      );
    }

    ffmpegArgs.push(localOutput);

    await this.runFFmpeg(ffmpegArgs);

    // Upload to S3
    const s3Key = `audio/${songId}/${outputFileName}`;
    await this.uploadToS3(localOutput, s3Key);

    // Get file stats
    const stats = await fs.stat(localOutput);

    const url = await getSignedUrl(
      this.s3,
      new GetObjectCommand({
        Bucket: this.outputBucket,
        Key: s3Key,
      }),
      { expiresIn: 86400 * 7 } // 7 days
    );

    return {
      format: options.format,
      quality: options.quality,
      path: s3Key,
      url,
      bitrate: (preset as any).bitrate || 0,
      fileSize: stats.size,
    };
  }

  /**
   * Generate waveform data
   */
  async generateWaveform(
    inputPath: string,
    resolution: number = 1000
  ): Promise<WaveformData> {
    // Use ffmpeg to extract audio as raw PCM
    const rawPath = `${inputPath}.raw`;

    await this.runFFmpeg([
      '-i', inputPath,
      '-ac', '1',           // Mono
      '-ar', '8000',        // 8kHz sample rate
      '-f', 's16le',        // Signed 16-bit little-endian
      '-y',
      rawPath,
    ]);

    // Read raw audio data
    const buffer = await fs.readFile(rawPath);
    await fs.unlink(rawPath);

    // Calculate peaks
    const samples = new Int16Array(buffer.buffer);
    const samplesPerPeak = Math.floor(samples.length / resolution);
    const peaks: number[] = [];

    for (let i = 0; i < resolution; i++) {
      const start = i * samplesPerPeak;
      const end = Math.min(start + samplesPerPeak, samples.length);

      let max = 0;
      for (let j = start; j < end; j++) {
        const value = Math.abs(samples[j]);
        if (value > max) max = value;
      }

      // Normalize to 0-1
      peaks.push(max / 32768);
    }

    return {
      peaks,
      duration: samples.length / 8000, // 8kHz sample rate
      sampleRate: 8000,
      channels: 1,
      resolution,
    };
  }

  /**
   * Analyze audio features (BPM, key, etc.)
   */
  async analyzeAudio(inputPath: string): Promise<AudioAnalysis> {
    // Extract audio features using essentia or aubio (simulated here)
    // In production, you'd use a library like essentia.js or a Python subprocess

    // For now, use ffmpeg's loudness measurement
    const output = await this.runFFmpeg([
      '-i', inputPath,
      '-af', 'ebur128=framelog=verbose',
      '-f', 'null',
      '-',
    ], true);

    // Parse loudness from output
    const loudnessMatch = output.match(/I:\s*(-?\d+\.?\d*)/);
    const loudness = loudnessMatch ? parseFloat(loudnessMatch[1]) : -14;

    // BPM detection would require a dedicated library
    // Using placeholder values - in production use librosa, essentia, or similar
    return {
      bpm: 120, // Would be detected by aubio/essentia
      key: 'C',
      scale: 'major',
      energy: 0.7,
      danceability: 0.65,
      speechiness: 0.1,
      instrumentalness: 0.8,
      valence: 0.6,
      loudness,
      acousticness: 0.3,
    };
  }

  /**
   * Generate audio fingerprint
   */
  async generateFingerprint(inputPath: string): Promise<AudioFingerprint> {
    // Use chromaprint/fpcalc for fingerprinting
    // Fallback to file hash if not available

    try {
      const output = await this.runCommand('fpcalc', [
        '-json',
        inputPath,
      ]);

      const data = JSON.parse(output);

      return {
        hash: data.fingerprint,
        duration: data.duration,
        algorithm: 'chromaprint',
      };
    } catch {
      // Fallback: generate hash from audio content
      const hash = await this.hashFile(inputPath);

      return {
        hash,
        duration: 0,
        algorithm: 'sha256',
      };
    }
  }

  /**
   * Normalize audio loudness
   */
  async normalizeAudio(
    inputPath: string,
    outputPath: string,
    targetLufs: number = -14
  ): Promise<void> {
    // Two-pass loudness normalization
    // Pass 1: Measure loudness
    const measureOutput = await this.runFFmpeg([
      '-i', inputPath,
      '-af', `loudnorm=I=${targetLufs}:TP=-1.5:LRA=11:print_format=json`,
      '-f', 'null',
      '-',
    ], true);

    // Parse measured values
    const jsonMatch = measureOutput.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('Failed to measure loudness');
    }

    const measured = JSON.parse(jsonMatch[0]);

    // Pass 2: Apply normalization
    await this.runFFmpeg([
      '-i', inputPath,
      '-af', `loudnorm=I=${targetLufs}:TP=-1.5:LRA=11:measured_I=${measured.input_i}:measured_LRA=${measured.input_lra}:measured_TP=${measured.input_tp}:measured_thresh=${measured.input_thresh}:offset=${measured.target_offset}:linear=true`,
      '-y',
      outputPath,
    ]);
  }

  // ==================== Helpers ====================

  private async runFFmpeg(args: string[], captureStderr = false): Promise<string> {
    return this.runCommand('ffmpeg', args, captureStderr);
  }

  private async runFFprobe(args: string[]): Promise<string> {
    return this.runCommand('ffprobe', args);
  }

  private runCommand(
    command: string,
    args: string[],
    captureStderr = false
  ): Promise<string> {
    return new Promise((resolve, reject) => {
      const proc = spawn(command, args);

      let stdout = '';
      let stderr = '';

      proc.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      proc.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      proc.on('close', (code) => {
        if (code === 0) {
          resolve(captureStderr ? stderr : stdout);
        } else {
          reject(new Error(`${command} exited with code ${code}: ${stderr}`));
        }
      });

      proc.on('error', reject);
    });
  }

  private async downloadFile(url: string, localPath: string): Promise<void> {
    if (url.startsWith('s3://')) {
      const match = url.match(/s3:\/\/([^/]+)\/(.+)/);
      if (!match) throw new Error('Invalid S3 URL');

      const command = new GetObjectCommand({
        Bucket: match[1],
        Key: match[2],
      });

      const response = await this.s3.send(command);
      const writeStream = createWriteStream(localPath);
      await streamPipeline(response.Body as NodeJS.ReadableStream, writeStream);
    } else {
      // HTTP download
      const response = await fetch(url);
      const writeStream = createWriteStream(localPath);
      await streamPipeline(response.body as unknown as NodeJS.ReadableStream, writeStream);
    }
  }

  private async uploadToS3(localPath: string, key: string): Promise<void> {
    const fileStream = createReadStream(localPath);
    const stats = await fs.stat(localPath);

    await this.s3.send(
      new PutObjectCommand({
        Bucket: this.outputBucket,
        Key: key,
        Body: fileStream,
        ContentLength: stats.size,
      })
    );
  }

  private async hashFile(filePath: string): Promise<string> {
    const hash = crypto.createHash('sha256');
    const stream = createReadStream(filePath);

    return new Promise((resolve, reject) => {
      stream.on('data', (data) => hash.update(data));
      stream.on('end', () => resolve(hash.digest('hex')));
      stream.on('error', reject);
    });
  }

  private async saveResults(
    songId: string,
    results: Record<string, unknown>
  ): Promise<void> {
    const transcoded = results.transcoded as TranscodedFile[];
    const waveform = results.waveform as WaveformData | undefined;
    const analysis = results.analysis as AudioAnalysis | undefined;
    const fingerprint = results.fingerprint as AudioFingerprint | undefined;

    // Update song with audio URLs
    if (transcoded.length > 0) {
      const audioUrls: Record<string, string> = {};
      for (const t of transcoded) {
        audioUrls[`${t.format}_${t.quality}`] = t.path;
      }

      await this.pool.query(
        `UPDATE songs SET audio_urls = $1 WHERE id = $2`,
        [JSON.stringify(audioUrls), songId]
      );
    }

    // Save waveform
    if (waveform) {
      await this.pool.query(
        `UPDATE songs SET waveform_data = $1 WHERE id = $2`,
        [JSON.stringify(waveform), songId]
      );
    }

    // Save analysis
    if (analysis) {
      await this.pool.query(
        `UPDATE songs SET
          bpm = $1,
          audio_key = $2,
          energy = $3,
          danceability = $4,
          valence = $5,
          loudness = $6,
          acousticness = $7,
          instrumentalness = $8
        WHERE id = $9`,
        [
          analysis.bpm,
          `${analysis.key} ${analysis.scale}`,
          analysis.energy,
          analysis.danceability,
          analysis.valence,
          analysis.loudness,
          analysis.acousticness,
          analysis.instrumentalness,
          songId,
        ]
      );
    }

    // Save fingerprint
    if (fingerprint) {
      await this.pool.query(
        `UPDATE songs SET audio_fingerprint = $1 WHERE id = $2`,
        [fingerprint.hash, songId]
      );
    }
  }
}

export default AudioPipeline;
