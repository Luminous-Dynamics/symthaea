// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================
// consciousness-music.js — Live consciousness → music in the browser
//
// Integrates with the Spore portal: reads harmony_scores and
// consciousness_level from each cycle, synthesizes audio via
// AudioWorklet. Toggle with the music button in the UI.
// ==================================================================

(function() {
  'use strict';

  var audioCtx = null;
  var workletNode = null;
  var running = false;
  var cycleCount = 0;
  var phase = 0;

  // Eight Harmonies frequencies (C Lydian: C D E F# G A B C)
  var HARMONY_FREQS = [261.63, 293.66, 329.63, 369.99, 392.00, 440.00, 493.88, 523.25];

  /**
   * Initialize audio context and worklet.
   * Must be called from a user gesture (click handler).
   */
  async function initAudio() {
    if (audioCtx) return;

    audioCtx = new AudioContext({ sampleRate: 44100 });

    try {
      await audioCtx.audioWorklet.addModule('js/consciousness-audio-worklet.js');
      workletNode = new AudioWorkletNode(audioCtx, 'consciousness-processor', {
        processorOptions: { bufferSize: 8192 },
        numberOfOutputs: 1,
        outputChannelCount: [2],
      });
      workletNode.connect(audioCtx.destination);

      workletNode.port.onmessage = function(ev) {
        if (ev.data.type === 'underrun') {
          console.warn('Audio underrun #' + ev.data.count);
        }
      };

      console.log('Consciousness music initialized');
    } catch (e) {
      console.error('AudioWorklet failed, falling back to oscillator mode:', e);
      workletNode = null;
    }
  }

  /**
   * Generate and push one chunk of consciousness-driven audio.
   * Called from the cycle result handler.
   */
  function renderChunk(consciousnessLevel, harmonies, neuromod) {
    if (!running || !audioCtx) return;

    var sr = 44100;
    var chunkSamples = 1411; // ~32ms at 44.1kHz
    var samples = new Float32Array(chunkSamples * 2); // stereo interleaved

    var phi = consciousnessLevel || 0;
    var da = (neuromod && neuromod[0]) || 0.5;
    var ne = (neuromod && neuromod[1]) || 0.3;
    var sht = (neuromod && neuromod[2]) || 0.5;

    // Synthesis: weighted sum of harmony tones
    var dt = 1.0 / sr;
    for (var i = 0; i < chunkSamples; i++) {
      var sampleL = 0, sampleR = 0;

      for (var h = 0; h < 8; h++) {
        var activation = (harmonies && harmonies[h]) || 0;
        if (activation < 0.1) continue;

        var freq = HARMONY_FREQS[h];
        var amp = activation * 0.08 * phi;

        // FM modulation driven by dopamine
        var fmDepth = da * 2.0;
        var fmPhase = phase * freq * 2.0;
        var fm = Math.sin(fmPhase * Math.PI * 2) * fmDepth;

        var osc = Math.sin((phase * freq + fm) * Math.PI * 2) * amp;

        // Stereo: odd harmonies slightly left, even slightly right
        var pan = (h % 2 === 0) ? 0.6 : 0.4;
        sampleL += osc * (1.0 - pan);
        sampleR += osc * pan;
      }

      // Soft clip
      sampleL = Math.max(-0.95, Math.min(0.95, sampleL));
      sampleR = Math.max(-0.95, Math.min(0.95, sampleR));

      samples[i * 2] = sampleL;
      samples[i * 2 + 1] = sampleR;
      phase += dt;
      if (phase > 100) phase -= 100; // prevent float precision loss
    }

    // Push to AudioWorklet
    if (workletNode) {
      workletNode.port.postMessage({ type: 'audio', samples: samples });
    }

    cycleCount++;
  }

  /**
   * Toggle consciousness music on/off.
   */
  async function toggle() {
    if (running) {
      running = false;
      if (workletNode) {
        workletNode.port.postMessage({ type: 'reset' });
      }
      updateButton(false);
      console.log('Consciousness music stopped');
    } else {
      await initAudio();
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }
      running = true;
      updateButton(true);
      console.log('Consciousness music started');
    }
  }

  function updateButton(isPlaying) {
    var btn = document.getElementById('music-toggle');
    if (btn) {
      btn.textContent = isPlaying ? '🔊 Music ON' : '🔇 Music OFF';
      btn.style.background = isPlaying ? 'var(--accent)' : 'var(--panel)';
    }
  }

  // Expose to app.js
  window.consciousnessMusic = {
    toggle: toggle,
    renderChunk: renderChunk,
    isRunning: function() { return running; },
  };

})();
