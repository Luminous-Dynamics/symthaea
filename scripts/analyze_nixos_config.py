#!/usr/bin/env python3
"""
Real NixOS Configuration Causal Analyzer

Parses actual NixOS configuration and applies causal analysis to detect:
- Dependency chains and root causes
- Configuration conflicts
- Side effects of changes
- Recommended fixes

This is a Python prototype of what the Rust NixOSCausalAnalyzer does.
"""

import re
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

@dataclass
class ConfigOption:
    name: str
    value: str
    file: str
    line: int
    enabled: bool = True

@dataclass
class CausalEdge:
    from_option: str
    to_option: str
    relationship: str  # "enables", "requires", "blocks", "affects"
    strength: float = 1.0

@dataclass
class CausalGraph:
    options: Dict[str, ConfigOption] = field(default_factory=dict)
    edges: List[CausalEdge] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)

# Known causal relationships in NixOS
KNOWN_CAUSAL_PATTERNS = [
    # (cause, effect, relationship)
    ("hardware.opengl.enable", "services.xserver", "enables"),
    ("services.xserver.enable", "services.displayManager", "requires"),
    ("networking.firewall.enable", "services", "blocks"),
    ("boot.kernelPackages", "hardware.nvidia.package", "determines"),
    ("nixpkgs.config.allowUnfree", "packages", "enables"),
    ("hardware.nvidia.modesetting.enable", "services.xserver.videoDrivers", "affects"),
    ("hardware.nvidia.open", "hardware.nvidia.package", "affects"),
    ("services.pipewire.enable", "sound", "affects"),
    ("hardware.pulseaudio.enable", "services.pipewire.enable", "conflicts"),
    ("networking.networkmanager.enable", "networking.wireless.enable", "conflicts"),
    ("boot.loader.systemd-boot.enable", "boot.loader.grub.enable", "conflicts"),
    ("nix.settings.experimental-features", "nix-command", "enables"),
    ("home-manager", "users.users", "extends"),
]

def parse_nix_file(filepath: str) -> CausalGraph:
    """Parse a NixOS configuration file"""
    graph = CausalGraph()

    try:
        with open(filepath) as f:
            content = f.read()
    except FileNotFoundError:
        return graph
    except PermissionError:
        return graph
    except Exception:
        return graph

    lines = content.split('\n')

    # Track imports
    import_pattern = re.compile(r'\./([a-zA-Z0-9_-]+\.nix)')
    for match in import_pattern.finditer(content):
        graph.imports.append(match.group(1))

    # Parse options
    for i, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith('#'):
            continue

        # Look for option assignments
        option_pattern = re.compile(r'([a-zA-Z][a-zA-Z0-9_.]+)\s*=\s*(.+);')
        match = option_pattern.search(line)
        if match:
            name = match.group(1)
            value = match.group(2).strip()

            # Determine if enabled
            enabled = True
            if value.lower() in ['false', 'null', '[]', '{}']:
                enabled = False
            elif 'enable' in name.lower() and 'true' in value.lower():
                enabled = True
            elif 'enable' in name.lower() and 'false' in value.lower():
                enabled = False

            graph.options[name] = ConfigOption(
                name=name,
                value=value,
                file=filepath,
                line=i,
                enabled=enabled
            )

    return graph

def detect_causal_relationships(graph: CausalGraph) -> List[CausalEdge]:
    """Detect causal relationships based on known patterns and option analysis"""
    edges = []

    # Apply known patterns
    for cause, effect, rel in KNOWN_CAUSAL_PATTERNS:
        for opt_name in graph.options:
            if cause in opt_name:
                for target_name in graph.options:
                    if effect in target_name and opt_name != target_name:
                        edges.append(CausalEdge(
                            from_option=opt_name,
                            to_option=target_name,
                            relationship=rel,
                            strength=0.8
                        ))

    # Detect implicit relationships based on naming patterns
    option_groups = defaultdict(list)
    for name in graph.options:
        parts = name.split('.')
        if len(parts) >= 2:
            prefix = '.'.join(parts[:2])
            option_groups[prefix].append(name)

    # Options in same group likely have causal relationships
    for prefix, opts in option_groups.items():
        if len(opts) > 1:
            # First option in group often enables others
            for i, opt in enumerate(opts[1:], 1):
                if 'enable' in opts[0].lower():
                    edges.append(CausalEdge(
                        from_option=opts[0],
                        to_option=opt,
                        relationship="affects",
                        strength=0.5
                    ))

    graph.edges = edges
    return edges

def find_root_causes(graph: CausalGraph, symptom: str) -> List[Tuple[str, float, str]]:
    """Find potential root causes of a configuration issue"""
    causes = []

    # Find options that causally affect the symptom
    for edge in graph.edges:
        if symptom in edge.to_option:
            if edge.from_option in graph.options:
                opt = graph.options[edge.from_option]
                explanation = f"{edge.from_option} {edge.relationship} {edge.to_option}"
                causes.append((edge.from_option, edge.strength, explanation))

    # Check for missing dependencies
    for pattern_cause, pattern_effect, rel in KNOWN_CAUSAL_PATTERNS:
        if pattern_effect in symptom:
            matching_cause = None
            for opt_name in graph.options:
                if pattern_cause in opt_name:
                    matching_cause = opt_name
                    break

            if matching_cause is None and rel in ['requires', 'enables']:
                causes.append((
                    pattern_cause,
                    0.9,
                    f"Missing {pattern_cause} which {rel} {symptom}"
                ))

    # Sort by strength
    causes.sort(key=lambda x: x[1], reverse=True)
    return causes[:5]

def predict_side_effects(graph: CausalGraph, option: str) -> List[Tuple[str, str, float]]:
    """Predict side effects of changing an option"""
    effects = []

    for edge in graph.edges:
        if option in edge.from_option:
            effects.append((edge.to_option, edge.relationship, edge.strength))

    # Also check known patterns
    for pattern_cause, pattern_effect, rel in KNOWN_CAUSAL_PATTERNS:
        if pattern_cause in option:
            for opt_name in graph.options:
                if pattern_effect in opt_name:
                    effects.append((opt_name, rel, 0.7))

    # Remove duplicates
    seen = set()
    unique_effects = []
    for e in effects:
        if e[0] not in seen:
            seen.add(e[0])
            unique_effects.append(e)

    return sorted(unique_effects, key=lambda x: x[2], reverse=True)[:10]

def detect_conflicts(graph: CausalGraph) -> List[Tuple[str, str, str]]:
    """Detect conflicting options"""
    conflicts = []

    for edge in graph.edges:
        if edge.relationship == 'conflicts':
            from_opt = graph.options.get(edge.from_option)
            to_opt = graph.options.get(edge.to_option)

            if from_opt and to_opt and from_opt.enabled and to_opt.enabled:
                conflicts.append((
                    edge.from_option,
                    edge.to_option,
                    "Both options are enabled but they conflict"
                ))

    # Check known conflict patterns
    conflict_patterns = [
        ("hardware.pulseaudio.enable", "services.pipewire.enable"),
        ("boot.loader.systemd-boot.enable", "boot.loader.grub.enable"),
    ]

    for opt1, opt2 in conflict_patterns:
        o1 = graph.options.get(opt1)
        o2 = graph.options.get(opt2)
        if o1 and o2 and o1.enabled and o2.enabled:
            conflicts.append((opt1, opt2, "Conflicting options both enabled"))

    return conflicts

def generate_recommendations(graph: CausalGraph) -> List[str]:
    """Generate configuration recommendations"""
    recs = []

    # Check for common issues
    conflicts = detect_conflicts(graph)
    for c1, c2, msg in conflicts:
        recs.append(f"CONFLICT: {c1} and {c2} - {msg}")

    # Check for missing enables
    if any('nvidia' in opt.lower() for opt in graph.options):
        if 'hardware.opengl.enable' not in graph.options:
            recs.append("SUGGESTION: Add hardware.opengl.enable = true for NVIDIA support")

    if any('xserver' in opt.lower() for opt in graph.options):
        if 'services.xserver.enable' not in graph.options:
            recs.append("SUGGESTION: Ensure services.xserver.enable = true is set")

    # Check flakes
    if 'nix.settings.experimental-features' in graph.options:
        opt = graph.options['nix.settings.experimental-features']
        if 'flakes' not in opt.value:
            recs.append("SUGGESTION: Add 'flakes' to experimental-features for flake support")

    return recs

def analyze_config(config_path: str):
    """Full causal analysis of NixOS configuration"""
    print("=" * 76)
    print("     NIXOS CONFIGURATION CAUSAL ANALYSIS")
    print("=" * 76)
    print()

    # Parse main config
    graph = parse_nix_file(config_path)

    print(f"Configuration: {config_path}")
    print(f"Options found: {len(graph.options)}")
    print(f"Imports: {len(graph.imports)}")
    print()

    # Parse imported files
    config_dir = Path(config_path).parent
    for imp in graph.imports:
        imp_path = config_dir / imp
        if imp_path.exists():
            imp_graph = parse_nix_file(str(imp_path))
            graph.options.update(imp_graph.options)
            print(f"  + {imp}: {len(imp_graph.options)} options")

    print()
    print(f"Total options after imports: {len(graph.options)}")
    print()

    # Detect causal relationships
    print("-" * 76)
    print(" CAUSAL RELATIONSHIPS DETECTED")
    print("-" * 76)

    edges = detect_causal_relationships(graph)
    print(f"Found {len(edges)} causal relationships")

    # Group by relationship type
    by_type = defaultdict(list)
    for e in edges:
        by_type[e.relationship].append(e)

    for rel_type, rel_edges in by_type.items():
        print(f"\n  {rel_type.upper()} ({len(rel_edges)}):")
        for e in rel_edges[:5]:
            print(f"    {e.from_option} -> {e.to_option}")
    print()

    # Detect conflicts
    print("-" * 76)
    print(" CONFLICT DETECTION")
    print("-" * 76)

    conflicts = detect_conflicts(graph)
    if conflicts:
        for c1, c2, msg in conflicts:
            print(f"  WARNING: {c1} <-> {c2}")
            print(f"           {msg}")
    else:
        print("  No conflicts detected")
    print()

    # Root cause analysis for common issues
    print("-" * 76)
    print(" ROOT CAUSE ANALYSIS")
    print("-" * 76)

    common_symptoms = ["xserver", "nvidia", "networking"]
    for symptom in common_symptoms:
        causes = find_root_causes(graph, symptom)
        if causes:
            print(f"\n  Potential causes for {symptom} issues:")
            for cause, strength, explanation in causes[:3]:
                print(f"    - {cause} ({strength:.0%})")
                print(f"      {explanation}")
    print()

    # Side effect prediction
    print("-" * 76)
    print(" SIDE EFFECT PREDICTION")
    print("-" * 76)

    key_options = ["hardware.nvidia", "services.xserver", "nix.settings"]
    for key in key_options:
        for opt in graph.options:
            if key in opt:
                effects = predict_side_effects(graph, opt)
                if effects:
                    print(f"\n  Changing {opt} affects:")
                    for affected, rel, strength in effects[:3]:
                        print(f"    - {affected} ({rel}, {strength:.0%})")
                break
    print()

    # Recommendations
    print("-" * 76)
    print(" RECOMMENDATIONS")
    print("-" * 76)

    recs = generate_recommendations(graph)
    if recs:
        for rec in recs:
            print(f"  {rec}")
    else:
        print("  No issues found - configuration looks good!")
    print()

    # Summary
    print("=" * 76)
    print(" SUMMARY")
    print("=" * 76)
    print()
    print(f"  Total Options:     {len(graph.options)}")
    print(f"  Causal Relations:  {len(edges)}")
    print(f"  Conflicts:         {len(conflicts)}")
    print(f"  Recommendations:   {len(recs)}")
    print()

if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "/etc/nixos/configuration.nix"
    analyze_config(config_path)
