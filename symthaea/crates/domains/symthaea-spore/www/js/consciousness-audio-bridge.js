// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Consciousness Audio Bridge: connects SporeEngine (WASM) to AudioWorklet.
 *
 * Main thread orchestrator:
 * 1. Loads SporeEngine WASM module
 * 2. Creates AudioContext + ConsciousnessAudioProcessor worklet
 * 3. Runs consciousness loop at 31Hz, rendering audio chunks
 * 4. Pushes PCM to worklet via MessagePort
 *
 * Usage:
 *   import { ConsciousnessAudioBridge } from './consciousness-audio-bridge.js';
 *   const bridge = new ConsciousnessAudioBridge();
 *   await bridge.init();
 *   bridge.start();
 *   // Later:
 *   bridge.setConsciousnessInput("Hello world");
 *   bridge.stop();
 */

export class ConsciousnessAudioBridge {
  constructor(options = {}) {
    this.sampleRate = options.sampleRate || 44100;
    this.chunkMs = options.chunkMs || 32;
    this.running = false;
    this.ctx = null;
    this.workletNode = null;
    this.sporeEngine = null;
    this.loopHandle = null;

    // Metrics
    this.cycleCount = 0;
    this.underrunCount = 0;
    this.lastPhiValue = 0;
    this.lastHarmonies = new Float32Array(8);
  }

  /**
   * Initialize the audio bridge.
   * @param {Object} wasmModule - The SporeEngine WASM module (from wasm_bindgen)
   * @param {string} workletUrl - URL to consciousness-audio-worklet.js
   */
  async init(wasmModule, workletUrl = 'consciousness-audio-worklet.js') {
    // Create audio context
    this.ctx = new AudioContext({ sampleRate: this.sampleRate });

    // Load AudioWorklet processor
    await this.ctx.audioWorklet.addModule(workletUrl);

    // Create worklet node
    this.workletNode = new AudioWorkletNode(this.ctx, 'consciousness-processor', {
      processorOptions: { bufferSize: 8192 },
      numberOfOutputs: 1,
      outputChannelCount: [2],
    });

    // Listen for underrun reports
    this.workletNode.port.onmessage = (event) => {
      if (event.data.type === 'underrun') {
        this.underrunCount = event.data.count;
        console.warn(`Audio underrun #${this.underrunCount} (available: ${event.data.available})`);
      }
    };

    // Connect to speakers
    this.workletNode.connect(this.ctx.destination);

    // Initialize SporeEngine (from WASM)
    if (wasmModule && wasmModule.SporeEngine) {
      this.sporeEngine = new wasmModule.SporeEngine();
    }

    console.log('ConsciousnessAudioBridge initialized', {
      sampleRate: this.sampleRate,
      chunkMs: this.chunkMs,
    });
  }

  /**
   * Start the consciousness → music loop.
   * Runs at ~31Hz (matching the cognitive loop).
   */
  start() {
    if (this.running) return;
    this.running = true;

    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }

    const intervalMs = this.chunkMs;
    this.loopHandle = setInterval(() => this._tick(), intervalMs);
    console.log('Consciousness audio loop started');
  }

  /**
   * Stop the consciousness → music loop.
   */
  stop() {
    this.running = false;
    if (this.loopHandle) {
      clearInterval(this.loopHandle);
      this.loopHandle = null;
    }
    // Send reset to worklet
    if (this.workletNode) {
      this.workletNode.port.postMessage({ type: 'reset' });
    }
    console.log('Consciousness audio loop stopped');
  }

  /**
   * Set text input for the consciousness engine.
   */
  setConsciousnessInput(text) {
    if (this.sporeEngine) {
      this.sporeEngine.cycle(text);
    }
  }

  /**
   * One tick of the consciousness → music loop.
   * @private
   */
  _tick() {
    if (!this.running || !this.sporeEngine) return;

    try {
      // 1. Run consciousness cycle
      this.sporeEngine.cycle('');

      // 2. Extract consciousness metrics
      this.lastPhiValue = Number(this.sporeEngine.consciousness_level());

      // 3. Render audio chunk
      // In production: sporeEngine.render_audio_chunk() returns Float32Array
      // For now, generate a simple consciousness-modulated tone
      const chunkSamples = Math.floor(this.sampleRate * this.chunkMs / 1000);
      const stereoSamples = new Float32Array(chunkSamples * 2);

      // Placeholder: consciousness-modulated sine (replace with SporeAudioBridge.tick_and_render())
      const freq = 220 + this.lastPhiValue * 440; // 220-660Hz based on Phi
      const dt = 1.0 / this.sampleRate;
      for (let i = 0; i < chunkSamples; i++) {
        const t = (this.cycleCount * chunkSamples + i) * dt;
        const sample = Math.sin(2 * Math.PI * freq * t) * 0.3 * this.lastPhiValue;
        stereoSamples[i * 2] = sample;     // left
        stereoSamples[i * 2 + 1] = sample; // right
      }

      // 4. Push to AudioWorklet
      this.workletNode.port.postMessage({
        type: 'audio',
        samples: stereoSamples,
      });

      this.cycleCount++;
    } catch (e) {
      console.error('Consciousness audio tick error:', e);
    }
  }

  /**
   * Get current metrics for UI display.
   */
  getMetrics() {
    return {
      cycleCount: this.cycleCount,
      underrunCount: this.underrunCount,
      phi: this.lastPhiValue,
      running: this.running,
      sampleRate: this.sampleRate,
    };
  }

  /**
   * Clean up resources.
   */
  destroy() {
    this.stop();
    if (this.workletNode) {
      this.workletNode.disconnect();
    }
    if (this.ctx) {
      this.ctx.close();
    }
  }
}
