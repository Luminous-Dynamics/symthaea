"""Shared NixOS configuration parsing and causal analysis helpers."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from symthaea_research.cli import print_banner, print_section

IMPORT_PATTERN = re.compile(r"\./([a-zA-Z0-9_-]+\.nix)")
OPTION_PATTERN = re.compile(r"([a-zA-Z][a-zA-Z0-9_.]+)\s*=\s*(.+);")


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
    relationship: str
    strength: float = 1.0


@dataclass
class CausalGraph:
    options: Dict[str, ConfigOption] = field(default_factory=dict)
    edges: List[CausalEdge] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)


KNOWN_CAUSAL_PATTERNS = [
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


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def parse_nix_file(filepath: str) -> CausalGraph:
    graph = CausalGraph()
    content = _read_text(Path(filepath))
    if not content:
        return graph

    graph.imports.extend(match.group(1) for match in IMPORT_PATTERN.finditer(content))

    for line_number, line in enumerate(content.splitlines(), start=1):
        if line.strip().startswith("#"):
            continue

        match = OPTION_PATTERN.search(line)
        if not match:
            continue

        name = match.group(1)
        value = match.group(2).strip()
        enabled = True
        lower_value = value.lower()
        if lower_value in {"false", "null", "[]", "{}"}:
            enabled = False
        elif "enable" in name.lower():
            enabled = "true" in lower_value

        graph.options[name] = ConfigOption(
            name=name,
            value=value,
            file=filepath,
            line=line_number,
            enabled=enabled,
        )

    return graph


def detect_causal_relationships(graph: CausalGraph) -> List[CausalEdge]:
    edges: List[CausalEdge] = []

    for cause, effect, relationship in KNOWN_CAUSAL_PATTERNS:
        for option_name in graph.options:
            if cause not in option_name:
                continue
            for target_name in graph.options:
                if effect in target_name and option_name != target_name:
                    edges.append(
                        CausalEdge(
                            from_option=option_name,
                            to_option=target_name,
                            relationship=relationship,
                            strength=0.8,
                        )
                    )

    option_groups: Dict[str, List[str]] = defaultdict(list)
    for name in graph.options:
        parts = name.split(".")
        if len(parts) >= 2:
            option_groups[".".join(parts[:2])].append(name)

    for grouped_options in option_groups.values():
        if len(grouped_options) <= 1 or "enable" not in grouped_options[0].lower():
            continue
        for option_name in grouped_options[1:]:
            edges.append(
                CausalEdge(
                    from_option=grouped_options[0],
                    to_option=option_name,
                    relationship="affects",
                    strength=0.5,
                )
            )

    graph.edges = edges
    return edges


def find_root_causes(graph: CausalGraph, symptom: str) -> List[Tuple[str, float, str]]:
    causes: List[Tuple[str, float, str]] = []

    for edge in graph.edges:
        if symptom in edge.to_option and edge.from_option in graph.options:
            explanation = f"{edge.from_option} {edge.relationship} {edge.to_option}"
            causes.append((edge.from_option, edge.strength, explanation))

    for pattern_cause, pattern_effect, relationship in KNOWN_CAUSAL_PATTERNS:
        if pattern_effect not in symptom:
            continue
        matching_cause = next(
            (option_name for option_name in graph.options if pattern_cause in option_name),
            None,
        )
        if matching_cause is None and relationship in {"requires", "enables"}:
            causes.append(
                (
                    pattern_cause,
                    0.9,
                    f"Missing {pattern_cause} which {relationship} {symptom}",
                )
            )

    causes.sort(key=lambda item: item[1], reverse=True)
    return causes[:5]


def predict_side_effects(graph: CausalGraph, option: str) -> List[Tuple[str, str, float]]:
    effects: List[Tuple[str, str, float]] = []

    for edge in graph.edges:
        if option in edge.from_option:
            effects.append((edge.to_option, edge.relationship, edge.strength))

    for pattern_cause, pattern_effect, relationship in KNOWN_CAUSAL_PATTERNS:
        if pattern_cause in option:
            for option_name in graph.options:
                if pattern_effect in option_name:
                    effects.append((option_name, relationship, 0.7))

    unique_effects: List[Tuple[str, str, float]] = []
    seen = set()
    for effect in effects:
        if effect[0] in seen:
            continue
        seen.add(effect[0])
        unique_effects.append(effect)

    return sorted(unique_effects, key=lambda item: item[2], reverse=True)[:10]


def detect_conflicts(graph: CausalGraph) -> List[Tuple[str, str, str]]:
    conflicts: List[Tuple[str, str, str]] = []

    for edge in graph.edges:
        if edge.relationship != "conflicts":
            continue
        from_option = graph.options.get(edge.from_option)
        to_option = graph.options.get(edge.to_option)
        if from_option and to_option and from_option.enabled and to_option.enabled:
            conflicts.append(
                (
                    edge.from_option,
                    edge.to_option,
                    "Both options are enabled but they conflict",
                )
            )

    for option_a, option_b in [
        ("hardware.pulseaudio.enable", "services.pipewire.enable"),
        ("boot.loader.systemd-boot.enable", "boot.loader.grub.enable"),
    ]:
        left = graph.options.get(option_a)
        right = graph.options.get(option_b)
        if left and right and left.enabled and right.enabled:
            conflicts.append((option_a, option_b, "Conflicting options both enabled"))

    return conflicts


def generate_recommendations(graph: CausalGraph) -> List[str]:
    recommendations: List[str] = []

    for option_a, option_b, message in detect_conflicts(graph):
        recommendations.append(f"CONFLICT: {option_a} and {option_b} - {message}")

    if any("nvidia" in option.lower() for option in graph.options):
        if "hardware.opengl.enable" not in graph.options:
            recommendations.append(
                "SUGGESTION: Add hardware.opengl.enable = true for NVIDIA support"
            )

    if any("xserver" in option.lower() for option in graph.options):
        if "services.xserver.enable" not in graph.options:
            recommendations.append("SUGGESTION: Ensure services.xserver.enable = true is set")

    experimental_features = graph.options.get("nix.settings.experimental-features")
    if experimental_features and "flakes" not in experimental_features.value:
        recommendations.append(
            "SUGGESTION: Add 'flakes' to experimental-features for flake support"
        )

    return recommendations


def _print_relationships(edges: Iterable[CausalEdge]) -> None:
    grouped: Dict[str, List[CausalEdge]] = defaultdict(list)
    for edge in edges:
        grouped[edge.relationship].append(edge)

    for relationship, relationship_edges in grouped.items():
        print(f"\n  {relationship.upper()} ({len(relationship_edges)}):")
        for edge in relationship_edges[:5]:
            print(f"    {edge.from_option} -> {edge.to_option}")
    print()


def analyze_config(config_path: str) -> None:
    print_banner("NIXOS CONFIGURATION CAUSAL ANALYSIS")

    graph = parse_nix_file(config_path)

    print(f"Configuration: {config_path}")
    print(f"Options found: {len(graph.options)}")
    print(f"Imports: {len(graph.imports)}")
    print()

    config_dir = Path(config_path).parent
    for imported in graph.imports:
        imported_path = config_dir / imported
        if imported_path.exists():
            imported_graph = parse_nix_file(str(imported_path))
            graph.options.update(imported_graph.options)
            print(f"  + {imported}: {len(imported_graph.options)} options")

    print()
    print(f"Total options after imports: {len(graph.options)}")
    print()

    print_section("CAUSAL RELATIONSHIPS DETECTED")
    edges = detect_causal_relationships(graph)
    print(f"Found {len(edges)} causal relationships")
    _print_relationships(edges)

    print_section("CONFLICT DETECTION")
    conflicts = detect_conflicts(graph)
    if conflicts:
        for option_a, option_b, message in conflicts:
            print(f"  WARNING: {option_a} <-> {option_b}")
            print(f"           {message}")
    else:
        print("  No conflicts detected")
    print()

    print_section("ROOT CAUSE ANALYSIS")
    for symptom in ["xserver", "nvidia", "networking"]:
        causes = find_root_causes(graph, symptom)
        if not causes:
            continue
        print(f"\n  Potential causes for {symptom} issues:")
        for cause, strength, explanation in causes[:3]:
            print(f"    - {cause} ({strength:.0%})")
            print(f"      {explanation}")
    print()

    print_section("SIDE EFFECT PREDICTION")
    for key in ["hardware.nvidia", "services.xserver", "nix.settings"]:
        for option_name in graph.options:
            if key not in option_name:
                continue
            effects = predict_side_effects(graph, option_name)
            if effects:
                print(f"\n  Changing {option_name} affects:")
                for affected, relationship, strength in effects[:3]:
                    print(f"    - {affected} ({relationship}, {strength:.0%})")
            break
    print()

    print_section("RECOMMENDATIONS")
    recommendations = generate_recommendations(graph)
    if recommendations:
        for recommendation in recommendations:
            print(f"  {recommendation}")
    else:
        print("  No issues found - configuration looks good!")
    print()

    print_banner("SUMMARY")
    print(f"  Total Options:     {len(graph.options)}")
    print(f"  Causal Relations:  {len(edges)}")
    print(f"  Conflicts:         {len(conflicts)}")
    print(f"  Recommendations:   {len(recommendations)}")
    print()
