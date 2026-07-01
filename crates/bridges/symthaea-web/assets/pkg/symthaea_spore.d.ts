/* tslint:disable */
/* eslint-disable */

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
     * Check if the full Broca pipeline is loaded and ready.
     *
     * Returns false if the `broca-pipeline` feature is not enabled or no
     * checkpoint has been loaded.
     */
    broca_pipeline_ready(): boolean;
    /**
     * Get the current causal graph as a JS object.
     * Returns { edges: [...], variables: [...] }.
     */
    causal_graph(): any;
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
     * Conversation context statistics: turn count, topic coherence, recent consciousness avg.
     *
     * Returns `{ turn_count, topic_coherence, recent_consciousness_avg }`.
     */
    conversation_stats(): any;
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
     * Name of the currently dominant harmony.
     */
    dominant_harmony(): string;
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
     * Generate text aware of user input.
     * Injects intent signals so generated language relates to user message.
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
     * Eight Harmonies scores as JSON array [RC, PSF, IW, IP, UI, SR, EP, SS].
     */
    harmony_scores(): string;
    /**
     * Honest confidence in the consciousness measurement (0.0-0.95).
     */
    honest_confidence(): number;
    /**
     * Inject neuromodulator impulse.
     */
    inject_neuromodulator(name: string, amount: number): void;
    /**
     * Query knowledge facts about a subject. Returns QueryResult as JS object.
     */
    knowledge_query(subject: string): any;
    /**
     * Knowledge engine statistics: fact count, contradiction count, etc.
     */
    knowledge_stats(): any;
    /**
     * Load trained Broca checkpoint weights from a binary buffer.
     *
     * The checkpoint file (broca-spore-v1.bin) contains trained token embeddings,
     * position embeddings, and gate weights. After loading, the autoregressive
     * generation path produces coherent language at consciousness levels >= 0.15.
     */
    load_broca_checkpoint(data: Uint8Array): void;
    /**
     * Load a full Broca pipeline checkpoint (symthaea-broca format).
     *
     * Requires `broca-pipeline` feature. After loading, `generate_text()` uses
     * the production HdcLtcUnifiedNetwork + epistemic gating + semantic veto
     * pipeline instead of BrocaLite.
     *
     * Pass the distilled checkpoint (broca-spore-distilled.bin, ~130MB) for
     * browser deployment, not the full v6-joint checkpoint (~976MB).
     */
    load_broca_pipeline_checkpoint(data: Uint8Array): void;
    /**
     * Compute Perturbational Complexity Index (PCI). Returns PciResult as JS object.
     *
     * Based on Casali et al. (2013): perturb the network and measure spatiotemporal
     * complexity of the response via Lempel-Ziv compression.
     */
    measure_pci(perturbation_magnitude: number, observation_cycles: number): any;
    /**
     * Trigger memory consolidation (dream replay). Returns ConsolidationResult.
     */
    memory_consolidate(): any;
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
     * Run a 7-step reasoning cycle on the given input.
     * Returns the full ReasoningCycle trace as a JS object.
     */
    reasoning_cycle(input: string): any;
    /**
     * Current safety level as a string ("Green", "Yellow", "Orange", "Red").
     */
    safety_level(): string;
    /**
     * Switch substrate type.
     */
    set_substrate(substrate: string): void;
    /**
     * Social mind state: partners, empathy level, coherence.
     */
    social_mind_state(): any;
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
     * Assess an input for threats. Returns ThreatAssessment as JS object.
     */
    threat_assessment(input: string): any;
    /**
     * Analyze consciousness topology. Returns TopologyAnalysis.
     */
    topology_analysis(): any;
    /**
     * Human-readable topology report.
     */
    topology_report(): string;
    /**
     * Whether the workspace has reached ignition (conscious experience).
     */
    workspace_ignition(): boolean;
    /**
     * Current workspace state: broadcast, contents, history.
     */
    workspace_state(): any;
}

/**
 * Generate disko disk configuration from hardware probe.
 */
export function generate_disko_config(hardware_json: string): string;

/**
 * Generate a complete NixOS flake.nix from hardware probe results.
 *
 * Takes the hardware profile JSON from the browser probe and returns
 * a full flake.nix string suitable for `nixos-install`.
 */
export function generate_flake(hardware_json: string, path: string, hostname: string): string;

/**
 * Generate hardware-configuration.nix from probe results.
 */
export function generate_hardware_nix(hardware_json: string): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_sporeengine_free: (a: number, b: number) => void;
    readonly generate_disko_config: (a: number, b: number, c: number) => void;
    readonly generate_flake: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly generate_hardware_nix: (a: number, b: number, c: number) => void;
    readonly sporeengine_active_instance_count: () => number;
    readonly sporeengine_anesthesia_experiment: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly sporeengine_broca_pipeline_ready: (a: number) => number;
    readonly sporeengine_causal_graph: (a: number, b: number) => void;
    readonly sporeengine_collapse_threshold_experiment: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_consciousness_level: (a: number) => number;
    readonly sporeengine_consciousness_report: (a: number, b: number) => void;
    readonly sporeengine_conversation_stats: (a: number, b: number) => void;
    readonly sporeengine_cycle: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_cycle_count: (a: number) => bigint;
    readonly sporeengine_cycle_hv: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_dominant_harmony: (a: number, b: number) => void;
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
    readonly sporeengine_harmony_scores: (a: number, b: number) => void;
    readonly sporeengine_honest_confidence: (a: number) => number;
    readonly sporeengine_inject_neuromodulator: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_knowledge_query: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_knowledge_stats: (a: number, b: number) => void;
    readonly sporeengine_load_broca_checkpoint: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_load_broca_pipeline_checkpoint: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_measure_pci: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_memory_consolidate: (a: number, b: number) => void;
    readonly sporeengine_memory_stats: (a: number, b: number) => void;
    readonly sporeengine_neuromod_state: (a: number, b: number) => void;
    readonly sporeengine_new: (a: number, b: number) => void;
    readonly sporeengine_reasoning_cycle: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_safety_level: (a: number, b: number) => void;
    readonly sporeengine_set_substrate: (a: number, b: number, c: number) => void;
    readonly sporeengine_social_mind_state: (a: number, b: number) => void;
    readonly sporeengine_split_brain_experiment: (a: number, b: number, c: number) => void;
    readonly sporeengine_substrate_feasibility: (a: number) => number;
    readonly sporeengine_threat_assessment: (a: number, b: number, c: number, d: number) => void;
    readonly sporeengine_topology_analysis: (a: number, b: number) => void;
    readonly sporeengine_topology_report: (a: number, b: number) => void;
    readonly sporeengine_workspace_ignition: (a: number) => number;
    readonly sporeengine_workspace_state: (a: number, b: number) => void;
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
