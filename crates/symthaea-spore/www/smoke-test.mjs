#!/usr/bin/env node
// Smoke test: verify Spore WASM loads and exports expected symbols
// Usage: node smoke-test.mjs

import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

let pass = 0, fail = 0;

function assert(condition, msg) {
  if (condition) { pass++; console.log(`  PASS: ${msg}`); }
  else { fail++; console.log(`  FAIL: ${msg}`); }
}

console.log('=== Spore WASM Smoke Test (Node.js) ===\n');

try {
  const wasmPath = join(__dirname, 'pkg', 'symthaea_spore_bg.wasm');
  const wasmBytes = await readFile(wasmPath);
  assert(wasmBytes.length > 100000, `WASM binary loaded (${(wasmBytes.length / 1024).toFixed(0)}KB)`);
  assert(wasmBytes.length < 512000, `WASM under 500KB budget (${(wasmBytes.length / 1024).toFixed(0)}KB)`);

  // Compile WASM module
  const wasmModule = await WebAssembly.compile(wasmBytes);
  assert(wasmModule instanceof WebAssembly.Module, 'WASM compiled successfully');

  // Check imports
  const imports = WebAssembly.Module.imports(wasmModule);
  const importModules = [...new Set(imports.map(i => i.module))];
  console.log(`  INFO: ${imports.length} imports from: ${importModules.join(', ')}`);

  // Check exports
  const exports = WebAssembly.Module.exports(wasmModule);
  const exportNames = exports.map(e => e.name);
  console.log(`  INFO: ${exports.length} total exports`);

  // Core engine exports
  const required = [
    'sporeengine_cycle',
    'sporeengine_cycle_hv',
    'sporeengine_set_substrate',
    'sporeengine_inject_neuromodulator',
    'sporeengine_consciousness_level',
    'sporeengine_honest_confidence',
    'sporeengine_harmony_alignment',
    'sporeengine_substrate_feasibility',
    'sporeengine_consciousness_report',
    'sporeengine_neuromod_state',
    'sporeengine_cycle_count',
    'sporeengine_active_instance_count',
    // Validation experiments
    'sporeengine_anesthesia_experiment',
    'sporeengine_measure_pci',
    'sporeengine_split_brain_experiment',
    'sporeengine_collapse_threshold_experiment',
    // Phase 1-5: Broca, Dream, FEP, Topology, Memory
    'sporeengine_generate_text',
    'sporeengine_dream_cycle',
    'sporeengine_dream_session',
    'sporeengine_dream_wisdom_count',
    'sporeengine_dream_stats',
    'sporeengine_free_energy',
    'sporeengine_fep_cycle',
    'sporeengine_topology_analysis',
    'sporeengine_topology_report',
    'sporeengine_memory_stats',
    // Infrastructure
    'memory',
  ];

  for (const name of required) {
    assert(exportNames.includes(name), `Export: ${name}`);
  }

  const missing = required.filter(e => !exportNames.includes(e));
  assert(missing.length === 0, `All ${required.length} required exports present`);

  // Check JS bindings file exists and has the right class
  const jsPath = join(__dirname, 'pkg', 'symthaea_spore.js');
  const jsContent = await readFile(jsPath, 'utf-8');
  assert(jsContent.includes('class SporeEngine'), 'JS: SporeEngine class');
  assert(jsContent.includes('anesthesia_experiment'), 'JS: anesthesia_experiment');
  assert(jsContent.includes('measure_pci'), 'JS: measure_pci');
  assert(jsContent.includes('split_brain_experiment'), 'JS: split_brain_experiment');
  assert(jsContent.includes('collapse_threshold_experiment'), 'JS: collapse_threshold_experiment');
  assert(jsContent.includes('generate_text'), 'JS: generate_text (Broca)');
  assert(jsContent.includes('dream_cycle'), 'JS: dream_cycle');
  assert(jsContent.includes('dream_session'), 'JS: dream_session');
  assert(jsContent.includes('fep_cycle'), 'JS: fep_cycle');
  assert(jsContent.includes('topology_analysis'), 'JS: topology_analysis');
  assert(jsContent.includes('memory_stats'), 'JS: memory_stats');
  assert(jsContent.includes('symthaea_spore_bg.wasm'), 'JS: references correct WASM filename');

  // Check TypeScript defs
  const dtsPath = join(__dirname, 'pkg', 'symthaea_spore.d.ts');
  const dtsContent = await readFile(dtsPath, 'utf-8');
  assert(dtsContent.includes('export class SporeEngine'), 'TS: SporeEngine typed');

} catch (e) {
  fail++;
  console.log(`  FAIL: ${e.message}`);
  if (e.code === 'ENOENT') {
    console.log('  HINT: Run ./build-wasm.sh first to produce the WASM binary');
  }
}

console.log(`\n=== Results: ${pass}/${pass + fail} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
