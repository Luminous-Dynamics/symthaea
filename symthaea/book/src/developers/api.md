# Core API Reference

## HDC Module (`symthaea-core::hdc`)

```rust
// Binary hypervectors (16,384 dimensions)
let hv = BinaryHV::random(seed);
let bound = hv1.bind(&hv2);         // XOR binding
let bundled = hv1.bundle(&hv2);     // Majority vote
let sim = hv1.similarity(&hv2);     // Hamming similarity

// Continuous hypervectors
let cv = ContinuousHV::random(seed);
let bound = cv1.bind(&cv2);         // Element-wise multiply
let sim = cv1.cosine_similarity(&cv2);
```

## Consciousness Module

```rust
use symthaea::CognitiveLoopService;

let config = CognitiveLoopConfig::default();
let mut cls = CognitiveLoopService::new(config);

// Run one cognitive cycle
let metadata = cls.cycle(input_hv);

// Access consciousness state
let phi = metadata.phi;
let consciousness_level = metadata.consciousness_level;
let workspace_ignited = metadata.workspace_ignited;
```

## LTC Module (`symthaea-core::hdc::hdc_ltc_unified`)

```rust
use symthaea_core::hdc::UnifiedHdcLtcNeuron;

let mut neuron = UnifiedHdcLtcNeuron::new(config);

// O(1) temporal jump to arbitrary time horizon
let new_state = neuron.evolve(input_hv, delta_t);
```

## Broca Language Pipeline

```rust
// With feature "ssm_language"
let result = cls.generate_text(max_tokens);
// Returns: generated text, epistemic support coverage, quality score
```

## SporeEngine (WASM Kernel)

```rust
use symthaea_spore::SporeEngine;

let mut engine = SporeEngine::new(SporeConfig::default());
let result = engine.cycle("input text");
// Result: consciousness_level, prediction_error, harmony_alignment, neuromods
```
