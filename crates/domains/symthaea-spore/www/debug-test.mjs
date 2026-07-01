#!/usr/bin/env node
import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const wasmBytes = await readFile(join(__dirname, 'pkg', 'symthaea_spore_bg.wasm'));
const mod = await import(join(__dirname, 'pkg', 'symthaea_spore.js'));
const { SporeEngine, initSync } = mod;
const wasmModule = await WebAssembly.compile(wasmBytes);
initSync({ module: wasmModule });

// Test 1: cycle_hv with non-zero values (fresh engine)
console.log('Test 1: cycle_hv with non-zero values...');
try {
  const e = new SporeEngine(null);
  const hv = new Float32Array(16384);
  for (let i = 0; i < hv.length; i++) hv[i] = Math.sin(i * 0.01) * 0.5 + 0.5;
  const r = e.cycle_hv(hv);
  console.log('  OK:', JSON.stringify(r).slice(0, 300));
  e.free();
} catch (e) { console.log('  FAIL:', e.message); }

// Test 2: cycle with text (fresh engine)
console.log('\nTest 2: cycle("hello")...');
try {
  const e = new SporeEngine(null);
  const r = e.cycle('hello');
  console.log('  OK:', JSON.stringify(r).slice(0, 300));
  e.free();
} catch (e) { console.log('  FAIL:', e.message); }

// Test 3: cycle_hv with ones (fresh engine)
console.log('\nTest 3: cycle_hv with ones...');
try {
  const e = new SporeEngine(null);
  const hv = new Float32Array(16384).fill(1.0);
  const r = e.cycle_hv(hv);
  console.log('  OK:', JSON.stringify(r).slice(0, 300));
  e.free();
} catch (e) { console.log('  FAIL:', e.message); }

// Test 4: cycle_hv with small values (fresh engine)
console.log('\nTest 4: cycle_hv with small values...');
try {
  const e = new SporeEngine(null);
  const hv = new Float32Array(16384).fill(0.001);
  const r = e.cycle_hv(hv);
  console.log('  OK:', JSON.stringify(r).slice(0, 300));
  e.free();
} catch (e) { console.log('  FAIL:', e.message); }
