// Web Worker for Spore consciousness engine.
// Keeps experiments and heavy cycles off the main thread.

import init, { SporeEngine, QuickeningOrchestrator, probe_hardware } from './pkg/symthaea_spore.js';

let engine = null;
let quickening = null;
let running = false;
let cycleInterval = null;
let currentThought = 'awareness';
let cycleCount = 0;

// Initialize WASM and engine
async function initialize(config) {
  await init();
  engine = new SporeEngine(config || null);
  return { ok: true, instances: SporeEngine.active_instance_count() };
}

// Run a single cycle
function cycle(input) {
  if (!engine) throw new Error('Engine not initialized');
  return engine.cycle(input || currentThought);
}

// Start auto-cycling at given interval (ms)
function startLoop(intervalMs) {
  if (cycleInterval) clearInterval(cycleInterval);
  running = true;
  cycleCount = 0;
  cycleInterval = setInterval(() => {
    if (!running || !engine) return;
    try {
      const result = engine.cycle(currentThought);
      cycleCount++;

      // Every 10th cycle, also send the output HV for visualization
      let hv = null;
      if (cycleCount % 10 === 0) {
        hv = Array.from(engine.get_output_hv());
      }

      self.postMessage({ type: 'cycle', result, hv });
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

// Collect hardware info from browser APIs and pass to WASM for Nix config generation
async function probeHardware() {
  const hw = {
    cpuCores: navigator.hardwareConcurrency || 1,
    deviceMemoryGb: navigator.deviceMemory || 0,
    gpuVendor: '',
    gpuArchitecture: '',
    gpuDescription: '',
    hasWebgpu: !!navigator.gpu,
    hasWebusb: !!navigator.usb,
    storageQuotaBytes: 0,
    platform: '',
    isMobile: false,
    browserName: '',
  };

  // Platform and browser detection
  if (navigator.userAgentData) {
    hw.platform = navigator.userAgentData.platform || '';
    hw.isMobile = navigator.userAgentData.mobile || false;
    const brands = navigator.userAgentData.brands || [];
    const brand = brands.find(b => !b.brand.includes('Not')) || brands[0];
    if (brand) hw.browserName = brand.brand;
  } else {
    hw.platform = navigator.platform || '';
    hw.isMobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
    // Basic UA parsing fallback
    if (/Firefox/i.test(navigator.userAgent)) hw.browserName = 'Firefox';
    else if (/Chrome/i.test(navigator.userAgent)) hw.browserName = 'Chrome';
    else if (/Safari/i.test(navigator.userAgent)) hw.browserName = 'Safari';
    else hw.browserName = 'Unknown';
  }

  // Storage quota
  if (navigator.storage && navigator.storage.estimate) {
    try {
      const est = await navigator.storage.estimate();
      hw.storageQuotaBytes = est.quota || 0;
    } catch (_) { /* not available */ }
  }

  // WebGPU adapter info
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        // adapterInfo available in recent browsers
        const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : null);
        if (info) {
          hw.gpuVendor = info.vendor || '';
          hw.gpuArchitecture = info.architecture || '';
          hw.gpuDescription = info.description || info.device || '';
        }
      }
    } catch (_) { /* WebGPU available but adapter request failed */ }
  }

  // Pass collected data to WASM for processing
  return probe_hardware(hw);
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
      case 'setThought':
        currentThought = params.text || 'awareness';
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
      case 'getOutputHV':
        if (engine) result = Array.from(engine.get_output_hv());
        break;
      case 'encodePair':
        if (engine) {
          result = {
            a: Array.from(engine.encode_text(params.textA)),
            b: Array.from(engine.encode_text(params.textB)),
          };
        }
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
      // Sovereign Quickening: Hardware probe
      case 'probeHardware':
        result = await probeHardware();
        break;
      // Phase 1: Language generation (Broca)
      case 'generateText':
        if (engine) result = engine.generate_text(params.maxTokens || 32);
        break;
      // Phase 2: Dream engine
      case 'dreamCycle':
        if (engine) result = engine.dream_cycle();
        break;
      case 'dreamSession':
        if (engine) result = engine.dream_session(params.cycles || 5);
        break;
      case 'dreamStats':
        if (engine) result = engine.dream_stats();
        break;
      case 'dreamWisdomCount':
        if (engine) result = engine.dream_wisdom_count();
        break;
      // Phase 3: Memory
      case 'memoryStats':
        if (engine) result = engine.memory_stats();
        break;
      // Phase 4: Topology
      case 'topologyAnalysis':
        if (engine) result = engine.topology_analysis();
        break;
      case 'topologyReport':
        if (engine) result = engine.topology_report();
        break;
      // Phase 5: FEP
      case 'freeEnergy':
        if (engine) result = engine.free_energy();
        break;
      case 'fepCycle':
        if (engine) result = engine.fep_cycle();
        break;
      // Phase 6: Sovereign Quickening
      case 'quickeningState':
        if (!quickening) quickening = new QuickeningOrchestrator();
        result = quickening.quickening_state();
        break;
      case 'quickeningAdvance':
        if (!quickening) quickening = new QuickeningOrchestrator();
        result = quickening.quickening_advance(params.phaseData || params);
        break;
      case 'quickeningNarrate':
        if (!quickening) quickening = new QuickeningOrchestrator();
        result = quickening.quickening_narrate(params.phase, params.context || null);
        break;
      default:
        throw new Error(`Unknown action: ${action}`);
    }
    self.postMessage({ id, type: 'response', result });
  } catch (err) {
    self.postMessage({ id, type: 'error', error: err.message });
  }
};
