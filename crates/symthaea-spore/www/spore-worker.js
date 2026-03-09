// Web Worker for Spore consciousness engine.
// Keeps experiments and heavy cycles off the main thread.

import init, { SporeEngine } from './pkg/symthaea_spore.js';

let engine = null;
let running = false;
let cycleInterval = null;

// Initialize WASM and engine
async function initialize(config) {
  await init();
  engine = new SporeEngine(config || null);
  return { ok: true, instances: SporeEngine.active_instance_count() };
}

// Run a single cycle
function cycle(input) {
  if (!engine) throw new Error('Engine not initialized');
  return engine.cycle(input || 'auto');
}

// Start auto-cycling at given interval (ms)
function startLoop(intervalMs) {
  if (cycleInterval) clearInterval(cycleInterval);
  running = true;
  cycleInterval = setInterval(() => {
    if (!running || !engine) return;
    try {
      const result = engine.cycle('auto');
      self.postMessage({ type: 'cycle', result });
    } catch (e) {
      self.postMessage({ type: 'error', error: e.message });
    }
  }, intervalMs || 50);
}

function stopLoop() {
  running = false;
  if (cycleInterval) { clearInterval(cycleInterval); cycleInterval = null; }
}

// Run experiments on a fresh engine to avoid contaminating the main one
function runExperiment(name, params) {
  const exp = new SporeEngine(null);
  let result;
  switch (name) {
    case 'anesthesia':
      result = exp.anesthesia_experiment(
        params.warmup || 20, params.suppression || 20, params.recovery || 20
      );
      break;
    case 'pci':
      result = exp.measure_pci(params.magnitude || 0.3, params.cycles || 30);
      break;
    case 'split_brain':
      result = exp.split_brain_experiment(params.cycles || 20);
      break;
    case 'collapse':
      result = exp.collapse_threshold_experiment(params.steps || 10, params.cyclesPerStep || 10);
      break;
    default:
      exp.free();
      throw new Error(`Unknown experiment: ${name}`);
  }
  exp.free();
  return result;
}

// Run all experiments and return unified report
function runBattery() {
  const results = {};
  const experiments = ['anesthesia', 'pci', 'split_brain', 'collapse'];

  for (const name of experiments) {
    self.postMessage({ type: 'battery_progress', experiment: name, status: 'running' });
    try {
      results[name] = runExperiment(name, {});
      self.postMessage({ type: 'battery_progress', experiment: name, status: 'done' });
    } catch (e) {
      results[name] = { error: e.message };
      self.postMessage({ type: 'battery_progress', experiment: name, status: 'error', error: e.message });
    }
  }

  return results;
}

// Multi-substrate comparison: run N cycles on each substrate, return time series
function multiSubstrateComparison(substrates, cycles) {
  const results = {};
  for (const substrate of substrates) {
    const eng = new SporeEngine(null);
    eng.set_substrate(substrate);
    const series = [];
    for (let i = 0; i < cycles; i++) {
      const r = eng.cycle('multi-substrate comparison');
      series.push({
        cycle: i,
        consciousness: r.consciousness_level,
        prediction_error: r.prediction_error,
        harmony: r.harmony_alignment,
      });
    }
    results[substrate] = series;
    eng.free();
    self.postMessage({ type: 'multi_progress', substrate, status: 'done' });
  }
  return results;
}

// Message handler
self.onmessage = async function(e) {
  const { id, action, ...params } = e.data;
  try {
    let result;
    switch (action) {
      case 'init':
        result = await initialize(params.config);
        break;
      case 'cycle':
        result = cycle(params.input);
        break;
      case 'startLoop':
        startLoop(params.interval);
        result = { ok: true };
        break;
      case 'stopLoop':
        stopLoop();
        result = { ok: true };
        break;
      case 'setSubstrate':
        if (engine) engine.set_substrate(params.substrate);
        result = { ok: true };
        break;
      case 'inject':
        if (engine) engine.inject_neuromodulator(params.name, params.amount);
        result = { ok: true };
        break;
      case 'experiment':
        stopLoop();
        result = runExperiment(params.name, params);
        break;
      case 'battery':
        stopLoop();
        result = runBattery();
        break;
      case 'multiSubstrate':
        stopLoop();
        result = multiSubstrateComparison(params.substrates, params.cycles || 50);
        break;
      case 'report':
        if (engine) result = engine.consciousness_report();
        break;
      default:
        throw new Error(`Unknown action: ${action}`);
    }
    self.postMessage({ id, type: 'response', result });
  } catch (err) {
    self.postMessage({ id, type: 'error', error: err.message });
  }
};
