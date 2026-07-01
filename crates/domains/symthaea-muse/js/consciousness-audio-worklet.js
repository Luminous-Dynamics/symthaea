// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Consciousness Audio Worklet: real-time consciousness → music in the browser.
 *
 * Architecture:
 *   Main Thread: SporeEngine (WASM) → tick_and_render() → SharedArrayBuffer
 *   AudioWorklet: reads SharedArrayBuffer → outputs to speakers
 *
 * Usage:
 *   const ctx = new AudioContext({ sampleRate: 44100 });
 *   await ctx.audioWorklet.addModule('consciousness-audio-worklet.js');
 *   const node = new AudioWorkletNode(ctx, 'consciousness-processor', {
 *     processorOptions: { bufferSize: 4096 }
 *   });
 *   node.connect(ctx.destination);
 *
 *   // From main thread, push PCM data:
 *   node.port.postMessage({ type: 'audio', samples: float32Array });
 */

class ConsciousnessAudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();

    // Ring buffer: holds pre-rendered PCM from the main thread
    const bufferSize = options.processorOptions?.bufferSize || 8192;
    this.ringBuffer = new Float32Array(bufferSize * 2); // stereo interleaved
    this.writePos = 0;
    this.readPos = 0;
    this.bufferMask = bufferSize * 2 - 1; // assumes power-of-2
    this.bufferSize = bufferSize * 2;
    this.underrunCount = 0;

    // Receive audio data from main thread
    this.port.onmessage = (event) => {
      if (event.data.type === 'audio') {
        this.pushSamples(event.data.samples);
      } else if (event.data.type === 'reset') {
        this.writePos = 0;
        this.readPos = 0;
        this.underrunCount = 0;
      }
    };
  }

  pushSamples(samples) {
    // Write interleaved stereo samples into ring buffer
    for (let i = 0; i < samples.length; i++) {
      this.ringBuffer[this.writePos % this.bufferSize] = samples[i];
      this.writePos++;
    }
  }

  available() {
    return (this.writePos - this.readPos + this.bufferSize) % this.bufferSize;
  }

  process(inputs, outputs, parameters) {
    const output = outputs[0];
    if (!output || output.length < 2) return true;

    const left = output[0];
    const right = output[1];
    const frames = left.length; // typically 128

    const needed = frames * 2; // stereo interleaved
    const avail = this.available();

    if (avail >= needed) {
      // Normal: read from ring buffer
      for (let i = 0; i < frames; i++) {
        left[i] = this.ringBuffer[this.readPos % this.bufferSize];
        this.readPos++;
        right[i] = this.ringBuffer[this.readPos % this.bufferSize];
        this.readPos++;
      }
    } else {
      // Underrun: output silence
      left.fill(0);
      right.fill(0);
      this.underrunCount++;

      // Notify main thread of underrun (every 10th to avoid spam)
      if (this.underrunCount % 10 === 1) {
        this.port.postMessage({
          type: 'underrun',
          count: this.underrunCount,
          available: avail,
          needed: needed,
        });
      }
    }

    return true; // keep processor alive
  }
}

registerProcessor('consciousness-processor', ConsciousnessAudioProcessor);
