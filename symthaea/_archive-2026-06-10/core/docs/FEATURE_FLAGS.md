# Symthaea Feature Flag Reference

98 feature flags organized into 12 categories. Default: none enabled (`default = []`).

## Quick Reference: Dependencies & Conflicts

### Dependency Chains (enabling A automatically enables B)

```
reasoning_engine → magi_loop
reasoning_engine → symthaea-causal-reasoning/counterfactual
multi_agent → full_consciousness
full_language → neural-bridge → candle-core, candle-nn, candle-transformers, tokenizers, hf-hub
school_learning → code_generation → tree-sitter-rust, tree-sitter-python
genesis → genomics + cell-foundry + ectogenesis + nurture + population
genesis-missions → 19 mission features → physics, materials, etc.
liquid-mamba → ssm_language → symthaea-broca
foveation → vision-manifold
neural-vocoder → vocal-tract
ssm-power → mesh + school_learning
ssm-power-hal → ssm-power + hal → humanoid
hal → humanoid
perception → embeddings + vision
audio → voice-tts + rodio
```

### External Requirements (not pure Rust)

| Feature | System Dependency |
|---------|------------------|
| `voice-tts` | ONNX Runtime (ort) |
| `voice-stt` | ONNX Runtime (ort) |
| `neural-vocoder-gpu` | CUDA or CoreML |
| `live-voice` | ALSA/PulseAudio (cpal) |
| `vision-manifold-camera` | V4L2 (Linux video capture) |
| `foveation-perception` | SigLIP/OCR/Moondream models on disk |
| `neural-bridge` | ~2GB model download (BGE-M3 via HF Hub) |
| `neural-bridge-cuda` | CUDA toolkit |
| `embeddings-gpu` | WGPU-compatible GPU |
| `multirotor-mujoco` | MuJoCo physics engine |
| `humanoid-mujoco` | MuJoCo physics engine |
| `ssm-power-hal` | Linux I2C (embedded HAL) |
| `swarm` | Network access (Iroh P2P) |

### CI-Safe Feature Set (47 flags)

These are tested together in CI clippy/test-all-features:
```
service, shell, demo, api_module, voice-stt, vocal-tract,
embeddings, vision, perception, webcam, vision-manifold,
foveation, integrity, semantic-encoder, neural-bridge,
mesh, lz4_compression, mesh-encryption, mesh-key-exchange,
swarm, mycelix, notifications, nix-mind, identity, physics,
flight, humanoid, ssm-power, ssm_language, liquid-mamba,
lancedb-backend, multi_agent, full_consciousness,
full_perception, full_language, magi_loop, reasoning_engine,
code_generation, wasm-sandbox, school_learning, benchmarks,
integration_module, observability_module, support,
web_research_module, genesis, physics-bridge
```

### NOT in CI (require hardware or GPU)

```
gui (eframe/egui — needs display server)
audio (rodio — needs audio device)
voice-tts (ort — ONNX runtime)
neural-vocoder, neural-vocoder-gpu (ONNX + optional CUDA)
live-voice (cpal — needs audio device)
embeddings-gpu (WGPU)
vision-manifold-camera (V4L2)
foveation-perception (model files)
neural-bridge-cuda (CUDA)
multirotor-mujoco, multirotor-mujoco-renderer (MuJoCo)
multirotor-swarm (MuJoCo + rayon parallel)
humanoid-mujoco, humanoid-viewer (MuJoCo)
hal, ssm-power-hal (I2C hardware)
```

## Feature Matrix by Category

### Binary Build Gates (6)

| Feature | Enables | Deps |
|---------|---------|------|
| `service` | CLI service binary | clap |
| `shell` | TUI shell binary | crossterm, ratatui |
| `gui` | egui desktop GUI | eframe, egui |
| `demo` | Demo CLI binaries | clap, ctrlc |
| `full` | All binaries | service+shell+gui+demo |
| `api_module` | HTTP API (Axum) | axum, tower-http |

### Voice & Audio (7)

| Feature | Enables | Deps |
|---------|---------|------|
| `voice-tts` | Kokoro TTS (ONNX) | ort, hound, hf-hub, espeak-rs |
| `voice-stt` | HDC+LTC+CfC STT | symthaea-stt, hound |
| `audio` | Full audio + playback | voice-tts + rodio |
| `vocal-tract` | LTC articulatory synthesis | hound, symthaea-vocal-tract |
| `neural-vocoder` | BigVGAN ONNX vocoder | ort + vocal-tract |
| `neural-vocoder-gpu` | GPU vocoder | neural-vocoder |
| `live-voice` | Real-time speaker | cpal, ringbuf, vocal-tract |

### Perception & Embeddings (10)

| Feature | Enables | Deps |
|---------|---------|------|
| `embeddings` | Qwen3/BGE text embeddings | tokenizers, hf-hub, burn |
| `embeddings-gpu` | GPU embeddings | embeddings + WGPU |
| `vision` | SigLIP image embeddings | ort, hf-hub |
| `perception` | Full multimodal | embeddings + vision |
| `webcam` | Live webcam capture | (cfg gate only) |
| `vision-manifold` | Patch-based HDC video | symthaea-vision-manifold |
| `vision-manifold-camera` | + V4L2 camera | vision-manifold |
| `foveation` | Active vision dispatch | symthaea-foveation + vision-manifold |
| `foveation-perception` | + real models | foveation |
| `semantic-encoder` | Background Qwen3 channel | symthaea-embeddings |
| `neural-bridge` | BGE-M3 via Candle | candle-*, tokenizers, hf-hub |
| `neural-bridge-cuda` | + CUDA acceleration | neural-bridge |
| `integrity` | Tamper detection | (cfg gate only) |

### Distributed / Network (7)

| Feature | Enables | Deps |
|---------|---------|------|
| `mesh` | LoRa mesh radio | (cfg gate only) |
| `lz4_compression` | Mesh packet compression | mesh + lz4_flex |
| `mesh-encryption` | ChaCha20 mesh encryption | mesh + chacha20poly1305, zeroize |
| `mesh-key-exchange` | X25519 key exchange | mesh-encryption + x25519-dalek |
| `swarm` | P2P tensor streaming | iroh |
| `mycelix` | Mycelix FL bridge | mycelix-fl-core, sha3 |
| `mycelix_sdk` | Full Mycelix SDK | mycelix + mycelix-sdk |

### Desktop & System (14)

| Feature | Enables | Deps |
|---------|---------|------|
| `notifications` | D-Bus desktop notify | zbus |
| `nix-mind` | Conscious NixOS | symthaea-nix |
| `identity` | Ed25519 signing | ed25519-dalek, sha2 |
| `physics` | Tokamak plasma encoding | symthaea-physics |
| `multirotor` | Multirotor FEP control | symthaea-multirotor |
| `flight` | Backward-compatible alias for `multirotor` | symthaea-multirotor |
| `multirotor-mujoco` | + MuJoCo physics | multirotor |
| `multirotor-mujoco-renderer` | + video capture | multirotor-mujoco |
| `multirotor-swarm` | + parallel swarm | multirotor |
| `flight-mujoco` | Backward-compatible alias | multirotor-mujoco |
| `flight-mujoco-renderer` | Backward-compatible alias | multirotor-mujoco-renderer |
| `flight-swarm` | Backward-compatible alias | multirotor-swarm |
| `humanoid` | Bipedal DMC benchmark | symthaea-humanoid |
| `humanoid-mujoco` | + MuJoCo physics | humanoid |
| `humanoid-viewer` | + MuJoCo viewer | humanoid-mujoco |
| `hal` | Hardware abstraction | symthaea-hal + humanoid |
| `ssm_language` | Broca language center | symthaea-broca |
| `liquid-mamba` | + Mamba SSM fusion | ssm_language |

### Consciousness & Reasoning (7)

| Feature | Enables | Deps |
|---------|---------|------|
| `full_consciousness` | Extended consciousness | (cfg gate only) |
| `full_perception` | Extended perception | (cfg gate only) |
| `full_language` | Advanced language | neural-bridge |
| `magi_loop` | World-grounded prediction | (cfg gate only) |
| `reasoning_engine` | 7-step reasoning cycle | magi_loop + causal-reasoning |
| `multi_agent` | Byzantine trust scoring | full_consciousness |
| `consciousness_full` | Bundle: reasoning + identity | reasoning_engine + identity |

### Code Understanding (3)

| Feature | Enables | Deps |
|---------|---------|------|
| `code_generation` | Tree-sitter parsers | tree-sitter-rust, tree-sitter-python |
| `wasm-sandbox` | WASM code execution | wasmtime |
| `school_learning` | Curriculum learning | code_generation |

### Database (1)

| Feature | Enables | Deps |
|---------|---------|------|
| `lancedb-backend` | LanceDB vector store | lancedb, arrow-*, futures |

### Genesis Pipeline (6)

| Feature | Enables | Deps |
|---------|---------|------|
| `genomics` | DNA assembly/repair | symthaea-genomics |
| `cell-foundry` | iPSC/IVG/SCNT | symthaea-cell-foundry |
| `ectogenesis` | Artificial womb | symthaea-ectogenesis |
| `nurture` | Bowlby attachment | symthaea-nurture |
| `population` | Population genetics | symthaea-population |
| `genesis` | Full pipeline | all 5 above |

### Genesis Missions (20)

| Feature | Deps |
|---------|------|
| `fusion-twin` | physics |
| `safety-agents` | (none) |
| `lab-controller` | cell-foundry |
| `materials` | symthaea-materials |
| `nuclear-forensics` | symthaea-nuclear-forensics |
| `water-prediction` | cell-foundry |
| `physics-bridge` | physics + symthaea-physics-bridge |
| `physics-unification` | physics |
| `grid-scaling` | physics |
| `fission-reactor` | physics |
| `accelerator` | physics |
| `threat-assessment` | physics |
| `datacenter` | physics |
| `experiment-planner` | cell-foundry |
| `strategic-materials` | materials |
| `critical-minerals` | materials |
| `advanced-manufacturing` | symthaea-fabrication-kernel |
| `building-systems` | symthaea-fabrication-kernel |
| `design-production` | symthaea-fabrication-kernel |
| `proliferation-safeguards` | nuclear-forensics |
| `genesis-missions` | all 19 above |

### Module Gates (7)

| Feature | Enables |
|---------|---------|
| `benchmarks` | Causal validation benchmarks |
| `integration_module` | Integration module |
| `observability_module` | Prometheus metrics |
| `support` | IT support intelligence |
| `web_research_module` | Epistemic verification |
| `unstable-examples` | Quarantined examples |
| `parallel` | Rayon parallelism (WASM gate) |

### Convenience Bundles (4)

| Bundle | Includes |
|--------|----------|
| `full` | service + shell + gui + demo |
| `consciousness_full` | reasoning_engine + identity |
| `all_benchmarks` | benchmarks + physics |
