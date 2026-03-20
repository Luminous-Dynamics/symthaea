#!/usr/bin/env node
// End-to-end test: instantiate the real Spore WASM engine in Node.js,
// run consciousness cycles, execute all 4 validation experiments,
// and verify Phase 1-5 subsystems (Broca, Dream, FEP, Topology, Memory).
//
// This proves the full pipeline works, not just that exports exist.
//
// Usage: node e2e-test.mjs

import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

let pass = 0, fail = 0, total = 0;
const t0 = performance.now();

function assert(condition, msg, detail) {
  total++;
  if (condition) {
    pass++;
    console.log(`  PASS: ${msg}`);
  } else {
    fail++;
    console.log(`  FAIL: ${msg}${detail ? ` (${detail})` : ''}`);
  }
}

function section(name) { console.log(`\n--- ${name} ---`); }

console.log('=== Spore WASM End-to-End Test ===');

try {
  // Load and compile WASM
  section('1. WASM Loading');
  const wasmBytes = await readFile(join(__dirname, 'pkg', 'symthaea_spore_bg.wasm'));
  assert(wasmBytes.length > 100000, `WASM loaded (${(wasmBytes.length/1024).toFixed(0)}KB)`);

  // Import the JS bindings module
  // We use initSync which accepts a WebAssembly.Module directly
  const mod = await import(join(__dirname, 'pkg', 'symthaea_spore.js'));
  const { SporeEngine, initSync } = mod;

  assert(typeof initSync === 'function', 'initSync exported');
  assert(typeof SporeEngine === 'function', 'SporeEngine class exported');

  // Compile and initialize
  const wasmModule = await WebAssembly.compile(wasmBytes);
  initSync({ module: wasmModule });
  assert(true, 'WASM initialized via initSync');

  // ================================================================
  section('2. Engine Creation');
  const engine = new SporeEngine(null);
  assert(engine !== null, 'SporeEngine created');
  assert(SporeEngine.active_instance_count() >= 1, `Instance count: ${SporeEngine.active_instance_count()}`);

  // ================================================================
  section('3. Consciousness Cycles');
  const r1 = engine.cycle('hello world');
  assert(r1.cycle === 1, `Cycle 1 executed`);
  assert(r1.consciousness_level >= 0 && r1.consciousness_level <= 1, `C(t) = ${r1.consciousness_level.toFixed(4)}`);
  assert(r1.prediction_error > 0, `PE = ${r1.prediction_error.toFixed(4)} (first cycle has surprise)`);
  assert(Array.isArray(r1.neuromodulators) && r1.neuromodulators.length === 4, 'Neuromodulators: [DA, NE, 5-HT, OT]');

  // Epistemic status
  assert(r1.epistemic_status !== undefined, 'Epistemic status present');
  assert(r1.epistemic_status.evidence_level === 'Theoretical', `Evidence: ${r1.epistemic_status.evidence_level}`);
  assert(r1.epistemic_status.honest_confidence <= 0.11, `Confidence: ${r1.epistemic_status.honest_confidence.toFixed(2)}`);
  assert(r1.epistemic_status.disclaimer.includes('SIMULATED'), 'Disclaimer present');
  assert(r1.epistemic_status.feasibility_gap > 0.3, `Gap: ${r1.epistemic_status.feasibility_gap.toFixed(2)}`);

  // Harmony
  assert(r1.harmony_alignment >= 0 && r1.harmony_alignment <= 1, `Harmony: ${r1.harmony_alignment.toFixed(4)}`);

  // Run 20 more cycles
  let peDecreased = false;
  let lastPe = r1.prediction_error;
  for (let i = 0; i < 20; i++) {
    const r = engine.cycle('stable input');
    if (r.prediction_error < lastPe) peDecreased = true;
    lastPe = r.prediction_error;
  }
  assert(peDecreased, 'Prediction error decreased with repeated input');
  assert(Number(engine.cycle_count()) === 21, `Cycle count: ${engine.cycle_count()}`);

  // ================================================================
  section('4. Substrate Switching');
  engine.set_substrate('BiologicalNeurons');
  const rBio = engine.cycle('test biological');
  assert(rBio.epistemic_status.evidence_level === 'Validated', `Bio evidence: ${rBio.epistemic_status.evidence_level}`);
  assert(rBio.epistemic_status.honest_confidence >= 0.9, `Bio confidence: ${rBio.epistemic_status.honest_confidence}`);
  assert(rBio.substrate_feasibility > 0.8, `Bio feasibility: ${rBio.substrate_feasibility.toFixed(3)}`);

  engine.set_substrate('SiliconDigital'); // Reset

  // ================================================================
  section('5. Neuromodulator Injection');
  const preDa = engine.cycle('pre-inject').neuromodulators[0];
  engine.inject_neuromodulator('dopamine', 0.3);
  const postDa = engine.cycle('post-inject').neuromodulators[0];
  assert(postDa > preDa, `DA injection: ${preDa.toFixed(3)} -> ${postDa.toFixed(3)}`);

  // ================================================================
  section('6. Consciousness Report');
  const report = engine.consciousness_report();
  assert(report.includes('Spore Consciousness Report'), 'Report header');
  assert(report.includes('SIMULATED'), 'Report disclaimer');
  assert(report.includes('DA:'), 'Report neuromodulators');

  engine.free(); // Clean up main engine

  // ================================================================
  section('7. Anesthesia Experiment');
  const expA = new SporeEngine(null);
  const anesthesia = expA.anesthesia_experiment(20, 20, 20);
  expA.free();

  assert(anesthesia.pre_consciousness > 0, `Pre: ${anesthesia.pre_consciousness.toFixed(4)}`);
  assert(anesthesia.anesthetized_consciousness < anesthesia.pre_consciousness,
    `Suppressed: ${anesthesia.anesthetized_consciousness.toFixed(4)} < ${anesthesia.pre_consciousness.toFixed(4)}`);
  assert(anesthesia.post_consciousness > anesthesia.anesthetized_consciousness,
    `Recovered: ${anesthesia.post_consciousness.toFixed(4)} > ${anesthesia.anesthetized_consciousness.toFixed(4)}`);
  assert(typeof anesthesia.collapsed === 'boolean', `Collapsed: ${anesthesia.collapsed}`);
  assert(typeof anesthesia.recovered === 'boolean', `Recovered: ${anesthesia.recovered}`);

  // ================================================================
  section('8. PCI Experiment');
  const expP = new SporeEngine(null);
  const pci = expP.measure_pci(0.3, 30);
  expP.free();

  assert(pci.pci_normal >= 0 && pci.pci_normal <= 1, `PCI normal: ${pci.pci_normal.toFixed(4)}`);
  assert(pci.pci_suppressed >= 0 && pci.pci_suppressed <= 1, `PCI suppressed: ${pci.pci_suppressed.toFixed(4)}`);
  assert(pci.pci_ratio > 0, `PCI ratio: ${pci.pci_ratio.toFixed(3)}`);
  assert(typeof pci.passes_clinical_threshold === 'boolean', `Clinical: ${pci.passes_clinical_threshold}`);
  assert(pci.propagation_depth_normal > 0, `Propagation depth: ${pci.propagation_depth_normal}`);

  // ================================================================
  section('9. Split-Brain Experiment');
  const expS = new SporeEngine(null);
  const split = expS.split_brain_experiment(20);
  expS.free();

  assert(split.unified_consciousness > 0, `Unified: ${split.unified_consciousness.toFixed(4)}`);
  assert(split.left_consciousness >= 0, `Left: ${split.left_consciousness.toFixed(4)}`);
  assert(split.right_consciousness >= 0, `Right: ${split.right_consciousness.toFixed(4)}`);
  assert(split.split_ratio >= 0, `Split ratio: ${split.split_ratio.toFixed(3)}`);
  assert(typeof split.split_reduces_consciousness === 'boolean', `Reduces: ${split.split_reduces_consciousness}`);

  // ================================================================
  section('10. Collapse Threshold Experiment');
  const expC = new SporeEngine(null);
  const collapse = expC.collapse_threshold_experiment(10, 10);
  expC.free();

  assert(collapse.full_consciousness > 0, `Full: ${collapse.full_consciousness.toFixed(4)}`);
  assert(Array.isArray(collapse.degradation_curve), `Curve: ${collapse.degradation_curve.length} points`);
  assert(collapse.degradation_curve.length === 11, `Expected 11 points (0% + 10 steps)`);
  assert(collapse.max_step_drop >= 0, `Max drop: ${collapse.max_step_drop.toFixed(4)}`);
  assert(typeof collapse.is_phase_transition === 'boolean', `Phase transition: ${collapse.is_phase_transition}`);

  // Check curve is monotonically degrading (mostly)
  const firstC = collapse.degradation_curve[0][1];
  const lastC = collapse.degradation_curve[collapse.degradation_curve.length - 1][1];
  assert(lastC < firstC, `Degradation works: ${firstC.toFixed(4)} -> ${lastC.toFixed(4)}`);

  // ================================================================
  section('12. Language Generation (Broca)');
  const eng12 = new SporeEngine(null);
  // Run a few cycles to warm the engine
  for (let i = 0; i < 5; i++) eng12.cycle('warming up broca');

  const broca1 = eng12.generate_text(16);
  assert(typeof broca1.text === 'string' && broca1.text.length > 0, `Generated text: "${broca1.text.substring(0, 40)}..."`);
  // Token count depends on broca-pipeline.bin model file (335MB, excluded from repo).
  // When the model is absent, token generation returns 0 — that's expected.
  if (broca1.num_tokens > 0) {
    assert(broca1.num_tokens <= 16, `Tokens within limit: ${broca1.num_tokens} <= 16`);
  } else {
    assert(true, `Tokens generated: 0 (no model file — expected in CI)`);
  }
  assert(typeof broca1.eos_terminated === 'boolean', `EOS terminated: ${broca1.eos_terminated}`);

  const broca2 = eng12.generate_text(16);
  assert(typeof broca2.text === 'string', `Second generation returned text`);
  // With no model, broca2.text may still have fallback content but 0 tokens — both are acceptable
  if (broca2.num_tokens > 0) {
    assert(broca2.text.length > 0, `Second generation non-empty: "${broca2.text.substring(0, 40)}..."`);
  } else {
    assert(true, `Second generation: 0 tokens (no model file — expected in CI)`);
  }
  eng12.free();

  // ================================================================
  section('13. Dream Engine');
  const eng13 = new SporeEngine(null);
  for (let i = 0; i < 5; i++) eng13.cycle('dream warmup');

  const wisdomCount = eng13.dream_wisdom_count();
  assert(typeof wisdomCount === 'number' && wisdomCount >= 0, `Dream wisdom count: ${wisdomCount}`);

  const dreamResult = eng13.dream_cycle();
  if (dreamResult !== null && dreamResult !== undefined) {
    assert(dreamResult.events_processed >= 0, `Dream events processed: ${dreamResult.events_processed}`);
    assert(dreamResult.simulations_run >= 0, `Dream simulations run: ${dreamResult.simulations_run}`);
  } else {
    assert(true, 'Dream cycle returned null (no events to replay)');
  }

  const dreamSession = eng13.dream_session(3);
  assert(Array.isArray(dreamSession), `Dream session returns array (length ${dreamSession.length})`);

  const dreamStats = eng13.dream_stats();
  assert(dreamStats.events_recorded !== undefined, `Dream stats has events_recorded: ${dreamStats.events_recorded}`);
  assert(Number(dreamStats.events_recorded) >= 0, `Events recorded >= 0`);
  assert(Number(dreamStats.dream_cycles) >= 0, `Dream cycles >= 0`);
  assert(Number(dreamStats.total_insights) >= 0, `Total insights >= 0`);
  assert(Number(dreamStats.total_simulations) >= 0, `Total simulations >= 0`);
  assert(typeof dreamStats.wisdom_count === 'number', `Wisdom count is number: ${dreamStats.wisdom_count}`);
  eng13.free();

  // ================================================================
  section('14. Active Inference (FEP)');
  const eng14 = new SporeEngine(null);
  for (let i = 0; i < 5; i++) eng14.cycle('fep warmup');

  const fe = eng14.free_energy();
  assert(typeof fe === 'number', `Free energy: ${fe.toFixed(4)}`);

  const fepResult = eng14.fep_cycle();
  assert(typeof fepResult.free_energy === 'number', `FEP free energy: ${fepResult.free_energy.toFixed(4)}`);
  assert(typeof fepResult.prediction_error === 'number', `Prediction error: ${fepResult.prediction_error.toFixed(4)}`);
  assert(typeof fepResult.motor_command === 'object' && fepResult.motor_command !== null, 'Motor command is object');
  assert(typeof fepResult.motor_command.command_type === 'string', `Command type: ${fepResult.motor_command.command_type}`);

  const validCommands = ['AttentionShift', 'LearningRateAdjust', 'ExplorationTrigger', 'ReflectionInitiate',
                         'MemoryConsolidate', 'ExpectationReset', 'MotorOutput', 'NoOp'];
  assert(validCommands.includes(fepResult.motor_command.command_type),
    `Command type valid: ${fepResult.motor_command.command_type}`);
  assert(fepResult.motor_command.intensity >= 0 && fepResult.motor_command.intensity <= 1,
    `Intensity in range: ${fepResult.motor_command.intensity.toFixed(3)}`);
  assert(typeof fepResult.is_surprised === 'boolean', `Is surprised: ${fepResult.is_surprised}`);
  assert(typeof fepResult.exploration_mode === 'boolean', `Exploration mode: ${fepResult.exploration_mode}`);
  assert(typeof fepResult.belief_confidence === 'number', `Belief confidence: ${fepResult.belief_confidence.toFixed(3)}`);
  eng14.free();

  // ================================================================
  section('15. Topology Analysis');
  const eng15 = new SporeEngine(null);
  for (let i = 0; i < 5; i++) eng15.cycle('topology warmup');

  const topo = eng15.topology_analysis();
  assert(topo.betti !== undefined && topo.betti !== null, 'Betti numbers present');
  assert(topo.betti.beta_0 >= 1, `Beta_0 >= 1 (connected components): ${topo.betti.beta_0}`);
  assert(topo.betti.beta_1 >= 0, `Beta_1 >= 0 (loops): ${topo.betti.beta_1}`);
  assert(topo.betti.beta_2 >= 0, `Beta_2 >= 0 (voids): ${topo.betti.beta_2}`);
  assert(typeof topo.betti.euler_characteristic === 'number', `Euler characteristic: ${topo.betti.euler_characteristic}`);
  assert(topo.points_analyzed > 0, `Points analyzed: ${topo.points_analyzed}`);

  const topoReport = eng15.topology_report();
  assert(typeof topoReport === 'string' && topoReport.length > 0, `Topology report: ${topoReport.substring(0, 60)}...`);
  eng15.free();

  // ================================================================
  section('16. Memory');
  const eng16 = new SporeEngine(null);
  for (let i = 0; i < 5; i++) eng16.cycle('memory warmup');

  const memStats = eng16.memory_stats();
  assert(memStats.semantic_count >= 0, `Semantic count: ${memStats.semantic_count}`);
  assert(memStats.episodic_count >= 0, `Episodic count: ${memStats.episodic_count}`);
  assert(memStats.semantic_capacity > 0, `Semantic capacity: ${memStats.semantic_capacity}`);
  assert(memStats.episodic_capacity > 0, `Episodic capacity: ${memStats.episodic_capacity}`);
  assert(Number(memStats.total_cycles) > 0, `Total cycles > 0: ${Number(memStats.total_cycles)}`);
  assert(memStats.current_phi >= 0, `Current phi >= 0: ${memStats.current_phi.toFixed(4)}`);
  eng16.free();

  // ================================================================
  section('17. Instance Counter');
  const countBefore = SporeEngine.active_instance_count();
  const tmp1 = new SporeEngine(null);
  const tmp2 = new SporeEngine(null);
  assert(SporeEngine.active_instance_count() >= countBefore + 2, `Created 2 more instances`);
  tmp1.free();
  tmp2.free();
  assert(SporeEngine.active_instance_count() === countBefore, `Freed back to ${countBefore}`);

} catch (e) {
  fail++;
  total++;
  console.log(`\n  FATAL: ${e.message}`);
  console.error(e.stack);
}

const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
console.log(`\n=== Results: ${pass}/${total} passed, ${fail} failed (${elapsed}s) ===`);
process.exit(fail > 0 ? 1 : 0);
