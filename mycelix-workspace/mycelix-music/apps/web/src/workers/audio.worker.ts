// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Audio Processing Web Worker
 *
 * Handles heavy audio processing off the main thread:
 * - FFT analysis
 * - Stem separation
 * - Effect processing
 * - Waveform generation
 * - Audio encoding/decoding
 */

// Message types
interface WorkerMessage {
  id: string;
  type: string;
  data: any;
}

interface WorkerResponse {
  id: string;
  type: string;
  success: boolean;
  data?: any;
  error?: string;
}

// FFT utilities
class FFT {
  private size: number;
  private cosTable: Float32Array;
  private sinTable: Float32Array;
  private reverseTable: Uint32Array;

  constructor(size: number) {
    this.size = size;
    this.cosTable = new Float32Array(size);
    this.sinTable = new Float32Array(size);
    this.reverseTable = new Uint32Array(size);

    // Precompute tables
    for (let i = 0; i < size; i++) {
      const angle = (-2 * Math.PI * i) / size;
      this.cosTable[i] = Math.cos(angle);
      this.sinTable[i] = Math.sin(angle);
    }

    // Bit reversal
    let limit = 1;
    let bit = size >> 1;
    while (limit < size) {
      for (let i = 0; i < limit; i++) {
        this.reverseTable[i + limit] = this.reverseTable[i] + bit;
      }
      limit <<= 1;
      bit >>= 1;
    }
  }

  forward(real: Float32Array, imag: Float32Array): void {
    const n = this.size;

    // Bit reversal
    for (let i = 0; i < n; i++) {
      const j = this.reverseTable[i];
      if (i < j) {
        const tempR = real[i];
        const tempI = imag[i];
        real[i] = real[j];
        imag[i] = imag[j];
        real[j] = tempR;
        imag[j] = tempI;
      }
    }

    // Cooley-Tukey FFT
    for (let size = 2; size <= n; size *= 2) {
      const halfSize = size / 2;
      const tableStep = n / size;

      for (let i = 0; i < n; i += size) {
        let k = 0;
        for (let j = i; j < i + halfSize; j++) {
          const l = j + halfSize;
          const tr = real[l] * this.cosTable[k] - imag[l] * this.sinTable[k];
          const ti = real[l] * this.sinTable[k] + imag[l] * this.cosTable[k];
          real[l] = real[j] - tr;
          imag[l] = imag[j] - ti;
          real[j] += tr;
          imag[j] += ti;
          k += tableStep;
        }
      }
    }
  }

  inverse(real: Float32Array, imag: Float32Array): void {
    // Conjugate
    for (let i = 0; i < this.size; i++) {
      imag[i] = -imag[i];
    }

    this.forward(real, imag);

    // Conjugate and scale
    const scale = 1 / this.size;
    for (let i = 0; i < this.size; i++) {
      real[i] *= scale;
      imag[i] = -imag[i] * scale;
    }
  }
}

// Cached FFT instances
const fftCache = new Map<number, FFT>();

function getFFT(size: number): FFT {
  if (!fftCache.has(size)) {
    fftCache.set(size, new FFT(size));
  }
  return fftCache.get(size)!;
}

// ============================================================================
// Audio Processing Functions
// ============================================================================

/**
 * Compute spectrum from audio data
 */
function computeSpectrum(audioData: Float32Array, fftSize: number = 2048): Float32Array {
  const fft = getFFT(fftSize);
  const real = new Float32Array(fftSize);
  const imag = new Float32Array(fftSize);

  // Apply Hann window and copy data
  for (let i = 0; i < fftSize; i++) {
    const window = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
    real[i] = (audioData[i] || 0) * window;
    imag[i] = 0;
  }

  fft.forward(real, imag);

  // Compute magnitude spectrum
  const spectrum = new Float32Array(fftSize / 2);
  for (let i = 0; i < fftSize / 2; i++) {
    const magnitude = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
    spectrum[i] = 20 * Math.log10(magnitude + 1e-10);
  }

  return spectrum;
}

/**
 * Generate waveform data (downsampled peaks)
 */
function generateWaveform(audioData: Float32Array, targetLength: number = 1000): Float32Array {
  const waveform = new Float32Array(targetLength * 2); // min and max for each point
  const samplesPerPoint = Math.floor(audioData.length / targetLength);

  for (let i = 0; i < targetLength; i++) {
    let min = 1;
    let max = -1;

    for (let j = 0; j < samplesPerPoint; j++) {
      const sample = audioData[i * samplesPerPoint + j] || 0;
      if (sample < min) min = sample;
      if (sample > max) max = sample;
    }

    waveform[i * 2] = min;
    waveform[i * 2 + 1] = max;
  }

  return waveform;
}

/**
 * Calculate RMS level
 */
function calculateRMS(audioData: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < audioData.length; i++) {
    sum += audioData[i] * audioData[i];
  }
  return Math.sqrt(sum / audioData.length);
}

/**
 * Calculate peak level
 */
function calculatePeak(audioData: Float32Array): number {
  let peak = 0;
  for (let i = 0; i < audioData.length; i++) {
    const abs = Math.abs(audioData[i]);
    if (abs > peak) peak = abs;
  }
  return peak;
}

/**
 * Apply gain to audio
 */
function applyGain(audioData: Float32Array, gain: number): Float32Array {
  const result = new Float32Array(audioData.length);
  for (let i = 0; i < audioData.length; i++) {
    result[i] = audioData[i] * gain;
  }
  return result;
}

/**
 * Mix multiple audio buffers
 */
function mixAudio(buffers: Float32Array[], gains: number[]): Float32Array {
  const length = Math.max(...buffers.map((b) => b.length));
  const result = new Float32Array(length);

  for (let i = 0; i < length; i++) {
    let sample = 0;
    for (let j = 0; j < buffers.length; j++) {
      sample += (buffers[j][i] || 0) * (gains[j] || 1);
    }
    result[i] = sample;
  }

  return result;
}

/**
 * Apply simple low-pass filter
 */
function lowPassFilter(audioData: Float32Array, cutoff: number, sampleRate: number): Float32Array {
  const result = new Float32Array(audioData.length);
  const rc = 1 / (2 * Math.PI * cutoff);
  const dt = 1 / sampleRate;
  const alpha = dt / (rc + dt);

  result[0] = audioData[0];
  for (let i = 1; i < audioData.length; i++) {
    result[i] = result[i - 1] + alpha * (audioData[i] - result[i - 1]);
  }

  return result;
}

/**
 * Apply simple high-pass filter
 */
function highPassFilter(audioData: Float32Array, cutoff: number, sampleRate: number): Float32Array {
  const result = new Float32Array(audioData.length);
  const rc = 1 / (2 * Math.PI * cutoff);
  const dt = 1 / sampleRate;
  const alpha = rc / (rc + dt);

  result[0] = audioData[0];
  for (let i = 1; i < audioData.length; i++) {
    result[i] = alpha * (result[i - 1] + audioData[i] - audioData[i - 1]);
  }

  return result;
}

/**
 * Simple compression
 */
function compress(
  audioData: Float32Array,
  threshold: number,
  ratio: number,
  attack: number,
  release: number,
  sampleRate: number
): Float32Array {
  const result = new Float32Array(audioData.length);
  const attackCoeff = Math.exp(-1 / (attack * sampleRate / 1000));
  const releaseCoeff = Math.exp(-1 / (release * sampleRate / 1000));

  let envelope = 0;

  for (let i = 0; i < audioData.length; i++) {
    const input = Math.abs(audioData[i]);

    // Envelope follower
    if (input > envelope) {
      envelope = attackCoeff * envelope + (1 - attackCoeff) * input;
    } else {
      envelope = releaseCoeff * envelope + (1 - releaseCoeff) * input;
    }

    // Compute gain reduction
    const envelopeDb = 20 * Math.log10(envelope + 1e-10);
    let gain = 1;

    if (envelopeDb > threshold) {
      const overDb = envelopeDb - threshold;
      const reducedDb = threshold + overDb / ratio;
      gain = Math.pow(10, (reducedDb - envelopeDb) / 20);
    }

    result[i] = audioData[i] * gain;
  }

  return result;
}

/**
 * Simple limiting
 */
function limit(audioData: Float32Array, ceiling: number): Float32Array {
  const result = new Float32Array(audioData.length);
  const ceilingLinear = Math.pow(10, ceiling / 20);

  for (let i = 0; i < audioData.length; i++) {
    let sample = audioData[i];
    if (sample > ceilingLinear) sample = ceilingLinear;
    if (sample < -ceilingLinear) sample = -ceilingLinear;
    result[i] = sample;
  }

  return result;
}

/**
 * Normalize audio to peak
 */
function normalize(audioData: Float32Array, targetPeak: number = 1): Float32Array {
  const peak = calculatePeak(audioData);
  if (peak === 0) return audioData;

  const gain = targetPeak / peak;
  return applyGain(audioData, gain);
}

/**
 * Resample audio (simple linear interpolation)
 */
function resample(audioData: Float32Array, sourceSampleRate: number, targetSampleRate: number): Float32Array {
  const ratio = sourceSampleRate / targetSampleRate;
  const newLength = Math.floor(audioData.length / ratio);
  const result = new Float32Array(newLength);

  for (let i = 0; i < newLength; i++) {
    const sourceIndex = i * ratio;
    const index0 = Math.floor(sourceIndex);
    const index1 = Math.min(index0 + 1, audioData.length - 1);
    const frac = sourceIndex - index0;
    result[i] = audioData[index0] * (1 - frac) + audioData[index1] * frac;
  }

  return result;
}

/**
 * Detect beats in audio
 */
function detectBeats(audioData: Float32Array, sampleRate: number): number[] {
  const windowSize = Math.floor(sampleRate * 0.02); // 20ms windows
  const hopSize = Math.floor(windowSize / 2);
  const beats: number[] = [];

  let prevEnergy = 0;
  const energyHistory: number[] = [];
  const historyLength = 43; // ~1 second at typical hop size

  for (let i = 0; i < audioData.length - windowSize; i += hopSize) {
    // Calculate energy in window
    let energy = 0;
    for (let j = 0; j < windowSize; j++) {
      energy += audioData[i + j] * audioData[i + j];
    }
    energy /= windowSize;

    // Maintain history
    energyHistory.push(energy);
    if (energyHistory.length > historyLength) {
      energyHistory.shift();
    }

    // Calculate local average
    const avgEnergy = energyHistory.reduce((a, b) => a + b, 0) / energyHistory.length;

    // Detect beat (onset)
    if (energy > avgEnergy * 1.5 && energy > prevEnergy * 1.3) {
      beats.push(i / sampleRate);
    }

    prevEnergy = energy;
  }

  return beats;
}

/**
 * Estimate BPM from beat times
 */
function estimateBPM(beats: number[]): number {
  if (beats.length < 2) return 120;

  // Calculate inter-beat intervals
  const intervals: number[] = [];
  for (let i = 1; i < beats.length; i++) {
    intervals.push(beats[i] - beats[i - 1]);
  }

  // Find most common interval (histogram)
  const histogram = new Map<number, number>();
  for (const interval of intervals) {
    const rounded = Math.round(interval * 100) / 100;
    histogram.set(rounded, (histogram.get(rounded) || 0) + 1);
  }

  // Find peak
  let maxCount = 0;
  let peakInterval = 0.5; // default to 120 BPM

  for (const [interval, count] of histogram) {
    if (count > maxCount && interval > 0.2 && interval < 2) {
      maxCount = count;
      peakInterval = interval;
    }
  }

  return Math.round(60 / peakInterval);
}

// ============================================================================
// Message Handler
// ============================================================================

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const { id, type, data } = event.data;

  const respond = (success: boolean, result?: any, error?: string) => {
    const response: WorkerResponse = { id, type, success };
    if (result !== undefined) response.data = result;
    if (error) response.error = error;
    self.postMessage(response);
  };

  try {
    switch (type) {
      case 'computeSpectrum': {
        const spectrum = computeSpectrum(data.audioData, data.fftSize);
        respond(true, { spectrum });
        break;
      }

      case 'generateWaveform': {
        const waveform = generateWaveform(data.audioData, data.targetLength);
        respond(true, { waveform });
        break;
      }

      case 'analyzeAudio': {
        const rms = calculateRMS(data.audioData);
        const peak = calculatePeak(data.audioData);
        const spectrum = computeSpectrum(data.audioData);
        const beats = detectBeats(data.audioData, data.sampleRate);
        const bpm = estimateBPM(beats);

        respond(true, {
          rms,
          peak,
          rmsDb: 20 * Math.log10(rms + 1e-10),
          peakDb: 20 * Math.log10(peak + 1e-10),
          spectrum,
          beats,
          bpm,
        });
        break;
      }

      case 'applyGain': {
        const result = applyGain(data.audioData, data.gain);
        respond(true, { audioData: result });
        break;
      }

      case 'mixAudio': {
        const result = mixAudio(data.buffers, data.gains);
        respond(true, { audioData: result });
        break;
      }

      case 'lowPassFilter': {
        const result = lowPassFilter(data.audioData, data.cutoff, data.sampleRate);
        respond(true, { audioData: result });
        break;
      }

      case 'highPassFilter': {
        const result = highPassFilter(data.audioData, data.cutoff, data.sampleRate);
        respond(true, { audioData: result });
        break;
      }

      case 'compress': {
        const result = compress(
          data.audioData,
          data.threshold,
          data.ratio,
          data.attack,
          data.release,
          data.sampleRate
        );
        respond(true, { audioData: result });
        break;
      }

      case 'limit': {
        const result = limit(data.audioData, data.ceiling);
        respond(true, { audioData: result });
        break;
      }

      case 'normalize': {
        const result = normalize(data.audioData, data.targetPeak);
        respond(true, { audioData: result });
        break;
      }

      case 'resample': {
        const result = resample(data.audioData, data.sourceSampleRate, data.targetSampleRate);
        respond(true, { audioData: result });
        break;
      }

      case 'detectBeats': {
        const beats = detectBeats(data.audioData, data.sampleRate);
        const bpm = estimateBPM(beats);
        respond(true, { beats, bpm });
        break;
      }

      default:
        respond(false, undefined, `Unknown message type: ${type}`);
    }
  } catch (error) {
    respond(false, undefined, error instanceof Error ? error.message : 'Unknown error');
  }
};

// Signal that worker is ready
self.postMessage({ type: 'ready' });
