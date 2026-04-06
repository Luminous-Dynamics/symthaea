// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Praxis Spore Worker — consciousness engine running off-main-thread
// Provides a small action surface for the WASM sandbox labs.

import init, { SporeEngine } from './spore/symthaea_spore.js';

let engine = null;
let initialized = false;
let loopTimer = null;

async function initialize(config) {
    if (initialized) return true;
    try {
        await init('./spore/symthaea_spore_bg.wasm');
        engine = new SporeEngine(config || null);
        initialized = true;
        console.log('[Spore] Engine initialized');
        return true;
    } catch (e) {
        console.warn('[Spore] Init failed:', e);
        return false;
    }
}

function safeCall(fn, fallback) {
    try {
        return fn();
    } catch (e) {
        console.warn('[Spore] Call failed:', e);
        return fallback;
    }
}

function metrics() {
    if (!engine) return null;
    return safeCall(() => ({
        consciousness_level: engine.consciousness_level(),
        honest_confidence: engine.honest_confidence(),
        harmony_alignment: engine.harmony_alignment(),
        free_energy: engine.free_energy(),
        cycle_count: engine.cycle_count(),
        dominant_harmony: engine.dominant_harmony(),
        workspace_ignited: engine.workspace_ignition(),
        safety_level: engine.safety_level(),
    }), null);
}

self.onmessage = async function(e) {
    const { id, action, params } = e.data || {};

    if (action === 'init') {
        const ok = await initialize(params && params.config);
        self.postMessage({ id, type: 'response', action, result: { ok } });
        return;
    }

    if (!initialized) {
        self.postMessage({ id, type: 'error', action, error: 'Engine not initialized' });
        return;
    }

    let result = null;

    switch (action) {
        case 'cycle':
            result = safeCall(() => engine.cycle(params && params.input ? params.input : ''), null);
            break;
        case 'generate':
            result = safeCall(() => {
                const input = params && params.input ? params.input : '';
                const maxTokens = params && params.maxTokens ? params.maxTokens : 60;
                if (input) return engine.generate_text_with_input(input, maxTokens);
                return engine.generate_text(maxTokens);
            }, null);
            break;
        case 'reasoning':
            result = safeCall(() => engine.reasoning_cycle(params && params.input ? params.input : ''), null);
            break;
        case 'threat':
            result = safeCall(() => engine.threat_assessment(params && params.input ? params.input : ''), null);
            break;
        case 'harmony_scores':
            result = safeCall(() => engine.harmony_scores(), null);
            break;
        case 'startLoop': {
            const interval = (params && params.interval) || 200;
            if (loopTimer) clearInterval(loopTimer);
            loopTimer = setInterval(() => {
                try {
                    engine.cycle('');
                    const m = metrics();
                    if (!m) return;
                    // Parse neuromodulator JSON for individual values
                    let neuromod = { dopamine: 0.5, serotonin: 0.5, norepinephrine: 0.5 };
                    try { neuromod = JSON.parse(engine.neuromod_state()); } catch(_) {}
                    // Send as 'cycle' type — matches SporeEngineBridge.onmessage
                    self.postMessage({
                        type: 'cycle',
                        result: {
                            consciousness_level: m.consciousness_level,
                            phi: m.consciousness_level,
                            coherence: m.harmony_alignment,
                            free_energy: m.free_energy,
                            cycle_count: Number(m.cycle_count),
                            dominant_harmony: m.dominant_harmony,
                            workspace_ignited: m.workspace_ignited,
                            safety_level: m.safety_level,
                            neuromod_dopamine: neuromod.dopamine || 0.5,
                            neuromod_serotonin: neuromod.serotonin || 0.5,
                            neuromod_norepinephrine: neuromod.norepinephrine || 0.5,
                        }
                    });
                } catch(err) {
                    self.postMessage({ type: 'error', error: err.message });
                }
            }, interval);
            result = { ok: true, interval };
            break;
        }
        case 'stopLoop':
            if (loopTimer) { clearInterval(loopTimer); loopTimer = null; }
            result = { ok: true };
            break;
        case 'metrics':
            result = metrics();
            break;
        default:
            self.postMessage({ id, type: 'error', action, error: 'Unknown action: ' + action });
            return;
    }

    self.postMessage({ id, type: 'response', action, result });
};
