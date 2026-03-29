// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Audio Worklet Processor
 *
 * Low-latency audio processing:
 * - Custom DSP algorithms
 * - Plugin-like effect processing
 * - Real-time parameter automation
 * - SIMD optimization ready
 */

// ==================== Base Processor ====================

interface ProcessorMessage {
  type: string;
  [key: string]: unknown;
}

abstract class BaseProcessor extends AudioWorkletProcessor {
  protected params: Record<string, number> = {};
  protected sampleRate: number;
  protected bypassed = false;

  constructor() {
    super();
    this.sampleRate = sampleRate;

    this.port.onmessage = (event: MessageEvent<ProcessorMessage>) => {
      this.handleMessage(event.data);
    };
  }

  protected handleMessage(message: ProcessorMessage): void {
    switch (message.type) {
      case 'setParam':
        this.params[message.name as string] = message.value as number;
        break;
      case 'bypass':
        this.bypassed = message.value as boolean;
        break;
    }
  }

  abstract processAudio(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>
  ): void;

  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>
  ): boolean {
    if (this.bypassed) {
      // Pass through
      for (let i = 0; i < inputs.length; i++) {
        for (let c = 0; c < inputs[i].length; c++) {
          outputs[i][c].set(inputs[i][c]);
        }
      }
    } else {
      this.processAudio(inputs, outputs, parameters);
    }
    return true;
  }
}

// ==================== Gain Processor ====================

class GainProcessor extends BaseProcessor {
  static get parameterDescriptors() {
    return [
      {
        name: 'gain',
        defaultValue: 1,
        minValue: 0,
        maxValue: 2,
        automationRate: 'a-rate',
      },
    ];
  }

  processAudio(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>
  ): void {
    const input = inputs[0];
    const output = outputs[0];
    const gain = parameters.gain;

    for (let channel = 0; channel < input.length; channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];

      for (let i = 0; i < inputChannel.length; i++) {
        const g = gain.length > 1 ? gain[i] : gain[0];
        outputChannel[i] = inputChannel[i] * g;
      }
    }
  }
}

// ==================== Compressor Processor ====================

class CompressorProcessor extends BaseProcessor {
  private envelope = 0;

  static get parameterDescriptors() {
    return [
      { name: 'threshold', defaultValue: -24, minValue: -60, maxValue: 0 },
      { name: 'ratio', defaultValue: 4, minValue: 1, maxValue: 20 },
      { name: 'attack', defaultValue: 0.003, minValue: 0.0001, maxValue: 1 },
      { name: 'release', defaultValue: 0.25, minValue: 0.01, maxValue: 3 },
      { name: 'knee', defaultValue: 6, minValue: 0, maxValue: 40 },
      { name: 'makeupGain', defaultValue: 0, minValue: 0, maxValue: 24 },
    ];
  }

  processAudio(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>
  ): void {
    const input = inputs[0];
    const output = outputs[0];

    const threshold = parameters.threshold[0];
    const ratio = parameters.ratio[0];
    const attack = parameters.attack[0];
    const release = parameters.release[0];
    const knee = parameters.knee[0];
    const makeupGain = parameters.makeupGain[0];

    const attackCoeff = Math.exp(-1 / (this.sampleRate * attack));
    const releaseCoeff = Math.exp(-1 / (this.sampleRate * release));

    for (let i = 0; i < input[0].length; i++) {
      // Get input level (peak of all channels)
      let inputLevel = 0;
      for (let c = 0; c < input.length; c++) {
        inputLevel = Math.max(inputLevel, Math.abs(input[c][i]));
      }

      // Convert to dB
      const inputDb = 20 * Math.log10(inputLevel + 1e-10);

      // Compute gain reduction
      let gainDb = 0;
      if (inputDb > threshold + knee / 2) {
        gainDb = threshold + (inputDb - threshold) / ratio - inputDb;
      } else if (inputDb > threshold - knee / 2) {
        const x = inputDb - threshold + knee / 2;
        gainDb = (1 / ratio - 1) * x * x / (2 * knee);
      }

      // Envelope follower
      const targetEnv = -gainDb;
      if (targetEnv > this.envelope) {
        this.envelope = attackCoeff * this.envelope + (1 - attackCoeff) * targetEnv;
      } else {
        this.envelope = releaseCoeff * this.envelope + (1 - releaseCoeff) * targetEnv;
      }

      // Apply gain
      const gain = Math.pow(10, (-this.envelope + makeupGain) / 20);

      for (let c = 0; c < input.length; c++) {
        output[c][i] = input[c][i] * gain;
      }
    }

    // Send metering data
    this.port.postMessage({
      type: 'meter',
      gainReduction: this.envelope,
    });
  }
}

// ==================== Limiter Processor ====================

class LimiterProcessor extends BaseProcessor {
  private lookAheadBuffer: Float32Array[] = [];
  private bufferIndex = 0;
  private envelope = 0;

  static get parameterDescriptors() {
    return [
      { name: 'ceiling', defaultValue: -0.1, minValue: -20, maxValue: 0 },
      { name: 'release', defaultValue: 0.1, minValue: 0.01, maxValue: 1 },
    ];
  }

  constructor() {
    super();
    const lookAheadSamples = Math.ceil(this.sampleRate * 0.005); // 5ms
    this.lookAheadBuffer = [
      new Float32Array(lookAheadSamples),
      new Float32Array(lookAheadSamples),
    ];
  }

  processAudio(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>
  ): void {
    const input = inputs[0];
    const output = outputs[0];

    const ceiling = Math.pow(10, parameters.ceiling[0] / 20);
    const release = parameters.release[0];
    const releaseCoeff = Math.exp(-1 / (this.sampleRate * release));

    const bufferLength = this.lookAheadBuffer[0].length;

    for (let i = 0; i < input[0].length; i++) {
      // Find peak in look-ahead buffer
      let peak = 0;
      for (let c = 0; c < input.length; c++) {
        this.lookAheadBuffer[c][this.bufferIndex] = input[c][i];
        for (let j = 0; j < bufferLength; j++) {
          peak = Math.max(peak, Math.abs(this.lookAheadBuffer[c][j]));
        }
      }

      // Compute gain
      const targetGain = peak > ceiling ? ceiling / peak : 1;

      // Envelope
      if (targetGain < this.envelope) {
        this.envelope = targetGain; // Instant attack
      } else {
        this.envelope = releaseCoeff * this.envelope + (1 - releaseCoeff) * targetGain;
      }

      // Output delayed signal with gain
      const readIndex = (this.bufferIndex + 1) % bufferLength;
      for (let c = 0; c < output.length; c++) {
        output[c][i] = this.lookAheadBuffer[c][readIndex] * this.envelope;
      }

      this.bufferIndex = (this.bufferIndex + 1) % bufferLength;
    }
  }
}

// ==================== EQ Processor ====================

class EQProcessor extends BaseProcessor {
  private filters: { a: number[]; b: number[]; z: number[][] }[] = [];

  static get parameterDescriptors() {
    return [
      { name: 'lowGain', defaultValue: 0, minValue: -12, maxValue: 12 },
      { name: 'lowFreq', defaultValue: 100, minValue: 20, maxValue: 500 },
      { name: 'midGain', defaultValue: 0, minValue: -12, maxValue: 12 },
      { name: 'midFreq', defaultValue: 1000, minValue: 200, maxValue: 5000 },
      { name: 'midQ', defaultValue: 1, minValue: 0.1, maxValue: 10 },
      { name: 'highGain', defaultValue: 0, minValue: -12, maxValue: 12 },
      { name: 'highFreq', defaultValue: 8000, minValue: 2000, maxValue: 20000 },
    ];
  }

  constructor() {
    super();
    // Initialize 3-band EQ filters
    for (let i = 0; i < 3; i++) {
      this.filters.push({
        a: [1, 0, 0],
        b: [1, 0, 0],
        z: [[0, 0], [0, 0]], // State for stereo
      });
    }
  }

  private updateFilter(
    index: number,
    type: 'lowshelf' | 'peaking' | 'highshelf',
    freq: number,
    gain: number,
    q: number
  ): void {
    const w0 = (2 * Math.PI * freq) / this.sampleRate;
    const A = Math.pow(10, gain / 40);
    const cosw0 = Math.cos(w0);
    const sinw0 = Math.sin(w0);
    const alpha = sinw0 / (2 * q);

    let b0: number, b1: number, b2: number, a0: number, a1: number, a2: number;

    switch (type) {
      case 'lowshelf':
        b0 = A * ((A + 1) - (A - 1) * cosw0 + 2 * Math.sqrt(A) * alpha);
        b1 = 2 * A * ((A - 1) - (A + 1) * cosw0);
        b2 = A * ((A + 1) - (A - 1) * cosw0 - 2 * Math.sqrt(A) * alpha);
        a0 = (A + 1) + (A - 1) * cosw0 + 2 * Math.sqrt(A) * alpha;
        a1 = -2 * ((A - 1) + (A + 1) * cosw0);
        a2 = (A + 1) + (A - 1) * cosw0 - 2 * Math.sqrt(A) * alpha;
        break;

      case 'peaking':
        b0 = 1 + alpha * A;
        b1 = -2 * cosw0;
        b2 = 1 - alpha * A;
        a0 = 1 + alpha / A;
        a1 = -2 * cosw0;
        a2 = 1 - alpha / A;
        break;

      case 'highshelf':
        b0 = A * ((A + 1) + (A - 1) * cosw0 + 2 * Math.sqrt(A) * alpha);
        b1 = -2 * A * ((A - 1) + (A + 1) * cosw0);
        b2 = A * ((A + 1) + (A - 1) * cosw0 - 2 * Math.sqrt(A) * alpha);
        a0 = (A + 1) - (A - 1) * cosw0 + 2 * Math.sqrt(A) * alpha;
        a1 = 2 * ((A - 1) - (A + 1) * cosw0);
        a2 = (A + 1) - (A - 1) * cosw0 - 2 * Math.sqrt(A) * alpha;
        break;
    }

    this.filters[index].b = [b0 / a0, b1 / a0, b2 / a0];
    this.filters[index].a = [1, a1 / a0, a2 / a0];
  }

  processAudio(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>
  ): void {
    // Update filter coefficients
    this.updateFilter(0, 'lowshelf', parameters.lowFreq[0], parameters.lowGain[0], 0.707);
    this.updateFilter(1, 'peaking', parameters.midFreq[0], parameters.midGain[0], parameters.midQ[0]);
    this.updateFilter(2, 'highshelf', parameters.highFreq[0], parameters.highGain[0], 0.707);

    const input = inputs[0];
    const output = outputs[0];

    for (let c = 0; c < input.length; c++) {
      for (let i = 0; i < input[c].length; i++) {
        let sample = input[c][i];

        // Apply each filter stage
        for (let f = 0; f < this.filters.length; f++) {
          const filter = this.filters[f];
          const z = filter.z[c];

          const filtered =
            filter.b[0] * sample +
            filter.b[1] * z[0] +
            filter.b[2] * z[1] -
            filter.a[1] * z[0] -
            filter.a[2] * z[1];

          z[1] = z[0];
          z[0] = filtered;
          sample = filtered;
        }

        output[c][i] = sample;
      }
    }
  }
}

// ==================== Distortion Processor ====================

class DistortionProcessor extends BaseProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'drive', defaultValue: 1, minValue: 1, maxValue: 100 },
      { name: 'mix', defaultValue: 1, minValue: 0, maxValue: 1 },
      { name: 'type', defaultValue: 0, minValue: 0, maxValue: 3 },
    ];
  }

  processAudio(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>
  ): void {
    const input = inputs[0];
    const output = outputs[0];

    const drive = parameters.drive[0];
    const mix = parameters.mix[0];
    const type = Math.floor(parameters.type[0]);

    for (let c = 0; c < input.length; c++) {
      for (let i = 0; i < input[c].length; i++) {
        const dry = input[c][i];
        let wet = dry * drive;

        switch (type) {
          case 0: // Soft clip (tanh)
            wet = Math.tanh(wet);
            break;
          case 1: // Hard clip
            wet = Math.max(-1, Math.min(1, wet));
            break;
          case 2: // Tube-like
            wet = (3 + drive) * wet * 20 / (Math.PI + drive * Math.abs(wet * 20));
            wet = wet / 20;
            break;
          case 3: // Fuzz
            wet = Math.sign(wet) * (1 - Math.exp(-Math.abs(wet * 3)));
            break;
        }

        output[c][i] = dry * (1 - mix) + wet * mix;
      }
    }
  }
}

// ==================== Delay Processor ====================

class DelayProcessor extends BaseProcessor {
  private buffer: Float32Array[] = [];
  private writeIndex = 0;
  private maxDelaySamples: number;

  static get parameterDescriptors() {
    return [
      { name: 'time', defaultValue: 0.5, minValue: 0, maxValue: 5 },
      { name: 'feedback', defaultValue: 0.3, minValue: 0, maxValue: 0.95 },
      { name: 'mix', defaultValue: 0.5, minValue: 0, maxValue: 1 },
    ];
  }

  constructor() {
    super();
    this.maxDelaySamples = Math.ceil(this.sampleRate * 5);
    this.buffer = [
      new Float32Array(this.maxDelaySamples),
      new Float32Array(this.maxDelaySamples),
    ];
  }

  processAudio(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>
  ): void {
    const input = inputs[0];
    const output = outputs[0];

    const time = parameters.time[0];
    const feedback = parameters.feedback[0];
    const mix = parameters.mix[0];

    const delaySamples = Math.floor(time * this.sampleRate);

    for (let i = 0; i < input[0].length; i++) {
      const readIndex = (this.writeIndex - delaySamples + this.maxDelaySamples) % this.maxDelaySamples;

      for (let c = 0; c < input.length; c++) {
        const delayed = this.buffer[c][readIndex];
        const inputSample = input[c][i];

        // Write to buffer with feedback
        this.buffer[c][this.writeIndex] = inputSample + delayed * feedback;

        // Mix dry and wet
        output[c][i] = inputSample * (1 - mix) + delayed * mix;
      }

      this.writeIndex = (this.writeIndex + 1) % this.maxDelaySamples;
    }
  }
}

// ==================== Register Processors ====================

registerProcessor('gain-processor', GainProcessor);
registerProcessor('compressor-processor', CompressorProcessor);
registerProcessor('limiter-processor', LimiterProcessor);
registerProcessor('eq-processor', EQProcessor);
registerProcessor('distortion-processor', DistortionProcessor);
registerProcessor('delay-processor', DelayProcessor);
