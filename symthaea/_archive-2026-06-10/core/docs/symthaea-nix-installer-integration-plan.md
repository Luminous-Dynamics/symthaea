# Symthaea-as-Architect: NixOS Installer Integration Plan

## Vision

Symthaea doesn't fill in templates — she **reasons about your system**. She understands your hardware (via HDC encoding), your intent (via active inference), the consequences of each choice (via 210+ causal patterns), and can explain every decision she makes (via Broca). The ceremony isn't decoration — it's her actually thinking about your system.

## Architecture

```
Portal UI (user choices + natural language)
  ↓ WebSocket
SSH Relay
  ↓
Symthaea Cognitive Loop (nix-mind feature)
  ├─ UserInputEncoder → intent HV (16,384D)
  ├─ HardwareProbe → hardware HV
  ├─ ActiveInference → select config strategy
  ├─ CausalGraph → predict side effects (210+ patterns)
  ├─ IntentResolver → option selection via HDC similarity
  ├─ NixEmitter → generate Nix code
  ├─ NixParser → validate with tree-sitter
  ├─ OptionValidator → check against NixOS options schema
  ├─ SafetyChecker → forbidden patterns + required options
  ├─ PhiGate → consciousness-gated execution
  └─ ConfigWriter → atomic write + git commit
  ↓
Upload to target → nix flake check → nixos-install
```

## Four-Layer Generation

### Layer 1: Deterministic Skeleton (never AI-generated)
- Boot loader (from UEFI/BIOS detection)
- Filesystem layout (from disko profile)
- Networking (NetworkManager, firewall always on)
- Locale (from user selection)
- Users (from user input)
- Security baseline (sudo, auto-upgrade, firewall)

### Layer 2: HDC Intent Resolution (Symthaea's unique capability)
- User intent encoded as 16,384D hypervector
- Hardware profile encoded as hypervector
- Similarity search against pre-encoded NixOS option clusters
- Concept clusters: "gaming", "creative", "development", "server", etc.
- Each cluster maps to 10-50 NixOS options with default values
- Confidence scoring: only apply options above 0.6 similarity
- Conflict resolution via vector orthogonality

### Layer 3: Causal Reasoning (210+ patterns)
- Before applying each option, check causal graph for side effects
- Example: enabling NVIDIA → requires modesetting → affects Wayland
- Example: enabling PipeWire → must disable PulseAudio
- Predict: "If I enable X, what else needs to change?"
- Build dependency chain automatically

### Layer 4: Polish & Explanation
- Generate comments explaining each section
- Handle edge cases and custom requests
- Produce human-readable reasoning trace
- Every ConfigDecision has: option, value, reasoning, confidence, alternatives

## Validation Pipeline

```
Generated Nix code
  → tree-sitter-nix syntax check (catches ~60% of errors)
  → Option path validation against 15K-option schema
  → Type checking (bool/string/list/enum match)
  → Cross-module conflict detection
  → SafetyChecker (forbidden patterns)
  → nix eval --sandbox (full Nix evaluation)
  → nix flake check (flake structure)
```

If any step fails → re-generate with error context → validate again → max 3 attempts.

## Data Requirements

### NixOS Options Schema
- Download: `curl -L https://channels.nixos.org/nixos-unstable/options.json.br | brotli -d`
- ~15,000 options, ~15-25MB JSON
- Index for fast lookup by path prefix
- Embed top 500 in HDC codebook

### HDC Codebook Construction
- Pre-compute atomic vectors for ~200 concepts (hardware, use cases, DEs, services)
- Pre-compute option cluster vectors for ~50 concept clusters
- Each cluster: 10-50 NixOS options with encoded default values
- Total codebook: ~500 option vectors × 16,384D = ~32MB

### Causal Graph (already exists)
- 210+ patterns in symthaea-nix/src/mind/causal_graph.rs (67,375 lines)
- Covers: boot, networking, GPU, desktop, services, security
- Bidirectional inference + Hebbian learning

## Implementation Phases

### Phase A: Wire Existing Infrastructure (1-2 sessions)
1. Build `symthaea-nix` with the nix-mind feature
2. Create a thin API: `fn generate_config(hardware: HardwareProfile, intent: UserIntent) -> NixConfig`
3. Wire into ssh-relay: after hardware probe + user choices → call Symthaea
4. Return generated flake.nix + configuration.nix + disko.nix
5. Upload to target, validate with `nix flake check`, then `nixos-install`

### Phase B: HDC Intent Resolution (1-2 sessions)
1. Download NixOS options schema, build lookup index
2. Construct HDC codebook from options (encode top 500 options)
3. Build concept clusters (gaming, creative, dev, server, etc.)
4. Implement IntentResolver: user choices → HV → similarity search → options
5. Wire into config generation pipeline

### Phase C: Explanation Engine (1 session)
1. ConfigDecision struct with reasoning trace
2. Portal UI: expandable explanations per config section
3. "Why did you choose this?" query system
4. Alternative suggestions with risk assessment

### Phase D: Broca for Nix (2-3 sessions)
1. Collect training data: NixOS configs from GitHub + synthetic from options schema
2. Train Broca on Nix code generation (consciousness-gated, epistemically honest)
3. Or: fine-tune qwen2.5-coder:7b on NixOS configs (QLoRA, RTX 2070)
4. Constrained generation: model fills details within validated skeleton
5. Multi-pass validation before accepting output

## What This Changes About the Installer

**Before**: Static shell scripts with hardcoded configuration.nix
**After**: Symthaea reasons about your hardware and intent, generates a flake, explains her choices, and learns from your feedback

The user experience:
1. Connect to target machine
2. Symthaea probes hardware, scans apps, detects existing OS
3. User picks use case ("I'm a developer who does Rust and Python")
4. Symthaea generates a complete flake with:
   - Correct GPU drivers for detected hardware
   - DE that works best with detected GPU (e.g., GNOME for NVIDIA+Wayland)
   - Development tools mapped from scanned apps
   - Security baseline
   - btrfs with snapshots
5. Each choice is explained: "I chose GNOME because your NVIDIA RTX 3060 has the best Wayland support with GDM"
6. User can override any choice, and Symthaea explains the consequences
7. Validated with `nix flake check` before install begins
8. Ceremony plays while Symthaea watches the install and narrates

## Key Files to Wire

| Existing Code | Purpose | Wire To |
|--------------|---------|---------|
| `symthaea-nix/src/encoding/config_encoder.rs` | Config → HV | Intent matching |
| `symthaea-nix/src/encoding/user_input_encoder.rs` | User text → HV | Portal input |
| `symthaea-nix/src/mind/causal_graph.rs` | 210+ patterns | Side effect prediction |
| `symthaea-nix/src/mind/active_inference.rs` | FEP action selection | Config strategy |
| `symthaea-nix/src/action/config_writer.rs` | Atomic config writes | Generated output |
| `symthaea-nix/src/action/flake_ops.rs` | Flake operations | Validation |
| `symthaea-nix/src/parser/nix_parser.rs` | Tree-sitter validation | Output checking |
| `src/language/emitters/nix.rs` | Nix code emission | Config generation |
| `src/language/code_intent.rs` | Intent classification | User request routing |
| `src/consciousness/reasoning_engine/` | 7-step reasoning | Decision making |
| `crates/symthaea-broca/` | Language generation | Explanations + code |
| `src/hdc/code_encoder.rs` | AST → HV | Config structure encoding |
