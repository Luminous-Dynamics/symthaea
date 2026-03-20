/* tslint:disable */
/* eslint-disable */

/**
 * Sovereign Birth ceremony state machine.
 *
 * Tracks progress through the Inoculation phases, accumulating narration
 * and awakening Harmony tones as each subsystem installs.
 */
export class InoculationOrchestrator {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get the Harmony tones array as JSON.
     */
    harmony_tones(): any;
    /**
     * Advance to the next phase. Takes a JSON object with:
     * - `phase` (string): phase name (e.g. "TrustVerification", "StorePopulation")
     * - `subsystem` (string, optional): for StorePopulation, the subsystem name
     * - `elapsed` (number): elapsed seconds since ceremony start
     * - `context` (object, optional): template variables for narration
     *
     * Returns a `PhaseAdvanceResult` with narration, haptic, tone, and state.
     */
    inoculation_advance(phase_data: any): any;
    /**
     * Get narration for a specific phase without advancing state.
     *
     * `phase` is the phase name string, `context` is a JSON object of template variables.
     */
    inoculation_narrate(phase: string, context: any): string;
    /**
     * Get the current InoculationState as a JSON object.
     *
     * Returns: `{ current_phase, phases_completed, harmonies_awakened,
     *             consciousness_level, elapsed_seconds, narration_history }`
     */
    inoculation_state(): any;
    /**
     * Whether the ceremony is complete.
     */
    is_complete(): boolean;
    /**
     * Create a new Inoculation orchestrator at the start of the ceremony.
     */
    constructor();
    /**
     * Current progress fraction (0.0 to 1.0).
     */
    progress(): number;
}

/**
 * WASM-exported Spore consciousness engine.
 */
export class SporeEngine {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Number of active SporeEngine instances globally.
     */
    static active_instance_count(): number;
    /**
     * Run an anesthesia analogue experiment. Returns AnesthesiaResult as JS object.
     *
     * Suppresses neuromodulators, observes consciousness collapse, then restores
     * and observes recovery. Models clinical anesthesia (propofol/sevoflurane).
     */
    anesthesia_experiment(warmup_cycles: number, suppression_cycles: number, recovery_cycles: number): any;
    /**
     * Find the consciousness collapse threshold. Returns CollapseThresholdResult as JS object.
     *
     * Systematically degrades the network and finds the point where consciousness
     * collapses. IIT predicts a phase transition, not gradual decline.
     */
    collapse_threshold_experiment(steps: number, cycles_per_step: number): any;
    /**
     * Current consciousness level.
     */
    consciousness_level(): number;
    /**
     * Human-readable consciousness report with epistemic disclaimers.
     */
    consciousness_report(): string;
    /**
     * Run a consciousness cycle with text input. Returns CycleResult as JS object.
     */
    cycle(input: string): any;
    /**
     * Current cycle count.
     */
    cycle_count(): bigint;
    /**
     * Run a consciousness cycle with raw hypervector input.
     */
    cycle_hv(hv: Float32Array): any;
    /**
     * Run a dream cycle — simulate counterfactual alternatives.
     */
    dream_cycle(): any;
    /**
     * Run a dream session (multiple dream cycles).
     */
    dream_session(cycles: number): any;
    /**
     * Dream engine statistics.
     */
    dream_stats(): any;
    /**
     * Number of wisdom entries accumulated from dreaming.
     */
    dream_wisdom_count(): number;
    /**
     * Encode text to an HDC hypervector without running a full cycle.
     * Returns bipolar encoding as f32 values. Used for thought comparison.
     */
    encode_text(text: string): Float32Array;
    /**
     * Run an explicit FEP cycle. Returns FepCycleResult.
     */
    fep_cycle(): any;
    /**
     * Current free energy value.
     */
    free_energy(): number;
    /**
     * Generate text from current consciousness state.
     * Returns GenerationResult as JS object with `text`, `num_tokens`, `eos_terminated`.
     */
    generate_text(max_tokens: number): any;
    /**
     * Generate text from current consciousness state, aware of user input.
     * The user's input is encoded as intent signals so generation relates to what was said.
     * Returns GenerationResult as JS object with `text`, `num_tokens`, `eos_terminated`.
     */
    generate_text_with_input(input: string, max_tokens: number): any;
    /**
     * Get the current network output hypervector (16,384 f32 values).
     * Used for live waveform visualization in the browser demo.
     */
    get_output_hv(): Float32Array;
    /**
     * Current harmony alignment score (0.0-1.0).
     */
    harmony_alignment(): number;
    /**
     * Honest confidence in the consciousness measurement (0.0-0.95).
     */
    honest_confidence(): number;
    /**
     * Inject neuromodulator impulse.
     */
    inject_neuromodulator(name: string, amount: number): void;
    /**
     * Compute Perturbational Complexity Index (PCI). Returns PciResult as JS object.
     *
     * Based on Casali et al. (2013): perturb the network and measure spatiotemporal
     * complexity of the response via Lempel-Ziv compression.
     */
    measure_pci(perturbation_magnitude: number, observation_cycles: number): any;
    /**
     * Memory subsystem statistics.
     */
    memory_stats(): any;
    /**
     * Neuromodulator state as JSON string.
     */
    neuromod_state(): string;
    /**
     * Create a new SporeEngine from a JSON configuration.
     * Pass `null` or `undefined` for defaults.
     */
    constructor(config: any);
    /**
     * Select the most resonant glyph for the current consciousness state and user input.
     * Returns JS object with `glyph_id` (string) and `echo_phrase` (string).
     */
    select_glyph(input: string): any;
    /**
     * Switch substrate type.
     */
    set_substrate(substrate: string): void;
    /**
     * Run a split-brain experiment. Returns SplitBrainResult as JS object.
     *
     * Partitions the network into hemispheres and measures whether splitting
     * reduces consciousness (IIT prediction).
     */
    split_brain_experiment(measurement_cycles: number): any;
    /**
     * Substrate feasibility score.
     */
    substrate_feasibility(): number;
    /**
     * Analyze consciousness topology. Returns TopologyAnalysis.
     */
    topology_analysis(): any;
    /**
     * Human-readable topology report.
     */
    topology_report(): string;
}

/**
 * Probe hardware capabilities and generate NixOS configuration.
 *
 * Takes browser-collected hardware data as a JS object (camelCase fields matching
 * `HardwareProfile`) and returns a `ProbeResult` containing the parsed profile
 * plus NixOS hardware configuration recommendations.
 *
 * # JS usage
 * ```js
 * const result = probe_hardware({
 *   cpuCores: navigator.hardwareConcurrency,
 *   deviceMemoryGb: navigator.deviceMemory || 0,
 *   hasWebgpu: !!navigator.gpu,
 *   // ... etc
 * });
 * console.log(result.nixConfig.nixHardwareConfig);
 * ```
 */
export function probe_hardware(js_data: any): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_inoculationorchestrator_free: (a: number, b: number) => void;
    readonly __wbg_sporeengine_free: (a: number, b: number) => void;
    readonly inoculationorchestrator_harmony_tones: (a: number, b: number) => void;
    readonly inoculationorchestrator_inoculation_advance: (a: number, b: number, c: number) => void;
    readonly inoculationorchestrator_inoculation_narrate: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly inoculationorchestrator_inoculation_state: (a: number, b: number) => void;
    readonly inoculationorchestrator_is_complete: (a: number) => number;
    readonly inoculationorchestrator_new: () => number;
    readonly inoculationorchestrator_progress: (a: number) => number;
    readonly probe_hardware: (a: number, b: number) => void;
    readonly sporeengine_active_instance_count: () => number;
    readonly sporeengine_anesthesia_experiment: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly sporeengine_collapse_threshold_experiment: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_consciousness_level: (a: number) => number;
    readonly sporeengine_consciousness_report: (a: number, b: number) => void;
    readonly sporeengine_cycle: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_cycle_count: (a: number) => bigint;
    readonly sporeengine_cycle_hv: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_dream_cycle: (a: number, b: number) => void;
    readonly sporeengine_dream_session: (a: number, b: number, c: number) => void;
    readonly sporeengine_dream_stats: (a: number, b: number) => void;
    readonly sporeengine_dream_wisdom_count: (a: number) => number;
    readonly sporeengine_encode_text: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_fep_cycle: (a: number, b: number) => void;
    readonly sporeengine_free_energy: (a: number) => number;
    readonly sporeengine_generate_text: (a: number, b: number, c: number) => void;
    readonly sporeengine_generate_text_with_input: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly sporeengine_get_output_hv: (a: number, b: number) => void;
    readonly sporeengine_harmony_alignment: (a: number) => number;
    readonly sporeengine_honest_confidence: (a: number) => number;
    readonly sporeengine_inject_neuromodulator: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_measure_pci: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_memory_stats: (a: number, b: number) => void;
    readonly sporeengine_neuromod_state: (a: number, b: number) => void;
    readonly sporeengine_new: (a: number, b: number) => void;
    readonly sporeengine_select_glyph: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_set_substrate: (a: number, b: number, c: number) => void;
    readonly sporeengine_split_brain_experiment: (a: number, b: number, c: number) => void;
    readonly sporeengine_substrate_feasibility: (a: number) => number;
    readonly sporeengine_topology_analysis: (a: number, b: number) => void;
    readonly sporeengine_topology_report: (a: number, b: number) => void;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number) => void;
    readonly __wbindgen_export4: (a: number, b: number, c: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
