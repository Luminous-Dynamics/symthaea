// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Example: connecting MuseEngine to WebAudio.
//
// For production use, pair with the AudioWorklet in
// symthaea-muse/js/consciousness-audio-bridge.js instead of
// ScriptProcessorNode.

import init, { MuseEngine } from '../pkg/symthaea_muse_wasm.js';

const SAMPLE_RATE = 44100;
const CHUNK_SIZE = 1024;

async function startSynthesis() {
  await init();

  const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
  const engine = new MuseEngine(SAMPLE_RATE, CHUNK_SIZE);

  // ScriptProcessor for simplicity — use AudioWorklet in production.
  const processor = ctx.createScriptProcessor(CHUNK_SIZE, 0, 2);

  processor.onaudioprocess = (e) => {
    const chunk = engine.render_chunk();
    const left = e.outputBuffer.getChannelData(0);
    const right = e.outputBuffer.getChannelData(1);
    for (let i = 0; i < left.length; i++) {
      left[i] = chunk[i * 2] || 0;
      right[i] = chunk[i * 2 + 1] || 0;
    }
  };

  processor.connect(ctx.destination);

  // Wire up sliders (expects elements with ids: arousal, valence, phi)
  const syncParams = () => {
    const arousal = parseFloat(document.getElementById('arousal')?.value ?? '0.4');
    const valence = parseFloat(document.getElementById('valence')?.value ?? '0.0');
    const phi = parseFloat(document.getElementById('phi')?.value ?? '0.5');
    engine.set_params(arousal, valence, phi);
  };

  for (const id of ['arousal', 'valence', 'phi']) {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', syncParams);
  }

  // Initial state
  syncParams();

  console.log('Consciousness synthesis started', engine.get_metrics());
  return { ctx, engine, processor };
}

// Auto-start on user gesture (browsers require user interaction for AudioContext)
document.addEventListener('click', () => startSynthesis(), { once: true });
