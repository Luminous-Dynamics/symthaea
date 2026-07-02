# Symtropy Foundry Neural-Link (LLM Integration)

> **Code status (2026-07-02 review, updated 2026-07-02):** `tools/symtropy_assets/neural_link.py`'s `generate_world_blueprint()` now calls a real local Ollama model (default `gemma4:e2b`, override via `--model` or `SYMTROPY_NEURAL_LINK_MODEL`) with `format: "json"` forced output, validated against `symtropy-foundry`'s `Blueprint`/`BiomeBlueprint`/`SamplingRule` structs (`crates/bridges/symtropy-foundry/src/orchestrator.rs`), with one retry on invalid JSON and a clearly-logged deterministic fallback if Ollama is unreachable. It also queries the asset registry's `behaviors` table for already-registered roles and hints the model to reuse them, then semantically snaps any invented `sampling_rule` role onto an existing registry role when they're close enough in meaning — embeds both via `embeddinggemma:300m` and cosine-matches (threshold 0.65, calibrated against real single-word similarity scores: near-synonyms ~0.73, thematically-related-but-distinct terms ~0.57, unrelated ~0.37-0.47), so generated worlds are more likely to populate with real assets instead of silently sampling zero for a near-miss role name. Falls back gracefully (blueprint unchanged, logged) if the embedding endpoint is unreachable. Verdict: REAL, not a stub — only the "Manifest Synthesizer" (new asset metadata generation, item 2 below) is still unimplemented; blueprint generation (items 1 and 3) is live.
>
> Fixed alongside this: `paths.py`'s `get_asset_root()` was resolving the registry path relative to `os.getcwd()` instead of walking up to the actual project root (the unused-but-tested `find_project_root()` already existed for exactly this) — running any CLI command from `tools/symtropy_assets/` instead of the symtropy root silently pointed at a different, usually-empty registry. This broke role lookups for the semantic-snap feature during testing; now fixed so registry resolution is CWD-independent.

## Overview
The Neural-Link connects high-level creative prompts to the Foundry Orchestrator. It acts as a generative layer that turns natural language intent into structured `world_blueprint.yaml` files and `manifest.yaml` definitions for new assets.

## Core Services
1.  **Semantic Planner**: Analyzes creative prompts to extract biome rules, density, and required behavior roles.
2.  **Manifest Synthesizer**: Generates new asset metadata and assigns roles based on the planned behaviors.
3.  **Orchestrator Injection**: Automatically writes the generated blueprints to the `blueprint_path` and notifies the `Foundry-Sync` listener to re-seed.

## Blueprint Generator (`tools/symtropy_assets/neural_link.py`)
This tool interfaces with your LLM provider to synthesize world state.
```python
def generate_world_blueprint(prompt: str, output_path: str):
    """
    1. Send prompt to LLM.
    2. Prompt requests YAML output matching Orchestrator schema.
    3. Save YAML to output_path.
    """
    pass
```
