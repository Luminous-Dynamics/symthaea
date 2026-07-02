# Symtropy Foundry Neural-Link (LLM Integration)

> **Code status (2026-07-02 review):** No corresponding implementation found in `symtropy/crates` or `symtropy/src`. Design/vision document only.

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
