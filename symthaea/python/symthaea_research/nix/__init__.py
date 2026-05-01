"""Nix and NixOS analysis helpers for research scripts."""

from .config_analysis import (
    KNOWN_CAUSAL_PATTERNS,
    CausalEdge,
    CausalGraph,
    ConfigOption,
    analyze_config,
    detect_causal_relationships,
    detect_conflicts,
    find_root_causes,
    generate_recommendations,
    parse_nix_file,
    predict_side_effects,
)

__all__ = [
    "KNOWN_CAUSAL_PATTERNS",
    "CausalEdge",
    "CausalGraph",
    "ConfigOption",
    "analyze_config",
    "detect_causal_relationships",
    "detect_conflicts",
    "find_root_causes",
    "generate_recommendations",
    "parse_nix_file",
    "predict_side_effects",
]
