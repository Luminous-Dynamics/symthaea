#!/usr/bin/env python3
"""Audit source-declared musical controls for every executable Muse style.

This inventory inspects the actual Rust declarations for grammar family,
composition preset, meter, tempo range, intent-axis support, and realization
instrument pools. It is source evidence, not a claim that every behavior has
been acoustically validated or culturally authenticated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from muse_style_coverage import CoverageError, STYLE_RELATIVE, git_blob_sha, parse_style_contract

ROOT = Path(__file__).resolve().parents[1]
REALIZATION_RELATIVE = Path("crates/domains/symthaea-muse/src/theory_realize.rs")
GRAMMAR_RELATIVE = Path("crates/domains/symthaea-music-theory/src/grammar.rs")
FAMILY_METHOD = "pub fn grammar_family"
AXES_METHOD = "pub fn supported_intent_axes"
METER_METHOD = "pub fn meter"
TEMPO_METHOD = "pub fn tempo_range"
SPEC_METHOD = "pub fn spec"
REALIZATION_METHOD = "pub(crate) fn instruments_for"


def _skip_line_comment(source: str, index: int) -> int:
    newline = source.find("\n", index + 2)
    return len(source) if newline < 0 else newline + 1


def _matching_brace(source: str, opening: int) -> int:
    """Return the matching closing brace, ignoring quoted text and comments."""
    if opening >= len(source) or source[opening] != "{":
        raise CoverageError(f"Expected '{{' at source offset {opening}")
    depth = 1
    index = opening + 1
    while index < len(source) and depth:
        if source.startswith("//", index):
            index = _skip_line_comment(source, index)
        elif source.startswith("/*", index):
            from muse_style_coverage import _skip_block_comment

            index = _skip_block_comment(source, index)
        elif source[index] == '"':
            from muse_style_coverage import _skip_string

            index = _skip_string(source, index)
        elif source[index] == "{":
            depth += 1
            index += 1
        elif source[index] == "}":
            depth -= 1
            index += 1
        else:
            index += 1
    if depth:
        raise CoverageError(f"Unterminated Rust block starting at offset {opening}")
    return index - 1


def _block_after(source: str, marker: str) -> str:
    start = source.find(marker)
    if start < 0:
        raise CoverageError(f"Could not find source marker {marker!r}")
    opening = source.find("{", start + len(marker))
    if opening < 0:
        raise CoverageError(f"Could not find block body after {marker!r}")
    return source[opening + 1 : _matching_brace(source, opening)]


def _strip_comments(source: str) -> str:
    """Remove Rust comments while preserving code and quoted literals."""
    result: list[str] = []
    index = 0
    while index < len(source):
        if source.startswith("//", index):
            index = _skip_line_comment(source, index)
        elif source.startswith("/*", index):
            from muse_style_coverage import _skip_block_comment

            index = _skip_block_comment(source, index)
        elif source[index] == '"':
            from muse_style_coverage import _skip_string

            end = _skip_string(source, index)
            result.append(source[index:end])
            index = end
        else:
            result.append(source[index])
            index += 1
    return "".join(result)


def _split_top_level(source: str, delimiter: str = ",") -> list[str]:
    """Split source where delimiter occurs outside nested Rust delimiters."""
    parts: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    index = 0
    while index < len(source):
        if source.startswith("//", index):
            index = _skip_line_comment(source, index)
            continue
        if source.startswith("/*", index):
            from muse_style_coverage import _skip_block_comment

            index = _skip_block_comment(source, index)
            continue
        if source[index] == '"':
            from muse_style_coverage import _skip_string

            index = _skip_string(source, index)
            continue
        char = source[index]
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack[-1] != pairs[char]:
                raise CoverageError("Unbalanced delimiter while splitting Rust source")
            stack.pop()
        elif char == delimiter and not stack:
            parts.append(source[start:index].strip())
            start = index + 1
        index += 1
    if stack:
        raise CoverageError("Unbalanced delimiter while splitting Rust source")
    tail = source[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _top_level_arrow(arm: str) -> tuple[str, str] | None:
    """Split a match arm at its top-level fat arrow."""
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    index = 0
    while index < len(arm):
        if arm.startswith("//", index):
            index = _skip_line_comment(arm, index)
            continue
        if arm.startswith("/*", index):
            from muse_style_coverage import _skip_block_comment

            index = _skip_block_comment(arm, index)
            continue
        if arm[index] == '"':
            from muse_style_coverage import _skip_string

            index = _skip_string(arm, index)
            continue
        char = arm[index]
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if stack and stack[-1] == pairs[char]:
                stack.pop()
        elif arm.startswith("=>", index) and not stack:
            return arm[:index].strip(), arm[index + 2 :].strip()
        index += 1
    return None


def _match_arms(source: str, pattern: str) -> list[tuple[str, str]]:
    match = re.search(pattern, source)
    if match is None:
        raise CoverageError(f"Could not find match expression {pattern!r}")
    opening = source.find("{", match.end() - 1)
    body = source[opening + 1 : _matching_brace(source, opening)]
    arms: list[tuple[str, str]] = []
    for item in _split_top_level(body):
        if not item:
            continue
        split = _top_level_arrow(item)
        if split is None:
            raise CoverageError(f"Unrecognized Rust match arm: {item[:100]!r}")
        arms.append(split)
    if not arms:
        raise CoverageError(f"Match expression {pattern!r} has no arms")
    return arms


def _variant_names(pattern: str) -> list[str]:
    return re.findall(r"\bStyle::([A-Z][A-Za-z0-9_]*)\b", pattern)


def _parse_mapping(
    arms: list[tuple[str, str]],
    variants: list[str],
    context: str,
    allow_default: bool = False,
) -> tuple[dict[str, str], str | None]:
    known = set(variants)
    mapped: dict[str, str] = {}
    default: str | None = None
    for pattern, expression in arms:
        clean_pattern = _strip_comments(pattern).strip()
        clean_expression = _strip_comments(expression).strip()
        names = _variant_names(clean_pattern)
        if not names:
            if allow_default and re.fullmatch(r"_", clean_pattern):
                if default is not None:
                    raise CoverageError(f"{context} has multiple wildcard arms")
                default = clean_expression
                continue
            raise CoverageError(f"{context} has unsupported arm pattern: {pattern!r}")
        unknown = sorted(set(names) - known)
        if unknown:
            raise CoverageError(f"{context} references unknown styles: {unknown}")
        for name in names:
            if name in mapped:
                raise CoverageError(f"{context} maps {name} more than once")
            mapped[name] = clean_expression
    if not allow_default and set(mapped) != known:
        raise CoverageError(
            f"{context} does not cover Style::ALL; missing={sorted(known - set(mapped))}"
        )
    return mapped, default


def _method_match_arms(source: str, marker: str, method_name: str) -> list[tuple[str, str]]:
    method = _block_after(source, marker)
    return _match_arms(method, r"match\s+self\s*\{" if method_name != REALIZATION_METHOD else r"match\s+style\s*\{")


def _string_array_constants(source: str) -> dict[str, list[str]]:
    source = _strip_comments(source)
    constants: dict[str, list[str]] = {}
    pattern = re.compile(
        r"const\s+([A-Z][A-Z0-9_]*)\s*:\s*&\[&str\]\s*=\s*&\[(.*?)\]\s*;",
        re.DOTALL,
    )
    for match in pattern.finditer(source):
        constants[match.group(1)] = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', match.group(2))
    return constants


def _extract_style_spec_variants(style_source: str, variants: list[str]) -> set[str]:
    method = _block_after(style_source, SPEC_METHOD)
    arms = _match_arms(method, r"match\s+self\s*\{")
    mapped, _ = _parse_mapping(arms, variants, "CompositionSpec presets")
    return set(mapped)


def _parse_grammar_dispatch(
    grammar_source: str, used_families: set[str]
) -> dict[str, dict[str, str]]:
    """Distinguish named family dispatch from the compatibility fallback."""
    method = _block_after(grammar_source, "pub fn compose_with_grammar_plan")
    arms = _match_arms(method, r"match\s+profile\.family\s*\{")
    dispatch: dict[str, dict[str, str]] = {}
    fallback_seen = False
    fallback_marker = "crate::composer::compose_with_spec_and_form_and_grammar"
    dedicated_engines = {
        "crate::fugue::realize_fugue": "fugue",
        "crate::groove_cycle::realize_groove_cycle": "groove_cycle",
        "crate::process_grammar::realize_additive_process": "process_grammar",
        "crate::modal_arc::realize_modal_arc": "modal_arc",
        "crate::call_response::realize_call_response": "call_response",
        "crate::jazz_chorus::realize_jazz_chorus": "jazz_chorus",
    }

    for raw_pattern, raw_expression in arms:
        pattern = _strip_comments(raw_pattern).strip()
        expression = _strip_comments(raw_expression)
        families = re.findall(r"\bGrammarFamily::([A-Z][A-Za-z0-9_]*)\b", pattern)
        if not families:
            if re.fullmatch(r"[a-z][A-Za-z0-9_]*", pattern):
                if fallback_seen:
                    raise CoverageError("Grammar dispatcher has multiple fallback arms")
                fallback_seen = True
                if fallback_marker not in expression:
                    raise CoverageError(
                        "Grammar compatibility fallback no longer calls the shared pipeline"
                    )
                continue
            raise CoverageError(f"Unrecognized grammar dispatcher arm: {pattern!r}")

        for family in families:
            if family in dispatch:
                raise CoverageError(f"Grammar family {family} has duplicate dispatch arms")
            engine = next(
                (
                    (name, marker)
                    for marker, name in dedicated_engines.items()
                    if marker in expression
                ),
                None,
            )
            if engine is not None:
                dispatch[family] = {
                    "mode": "dedicated_engine",
                    "entrypoint": engine[0] + ":" + engine[1],
                }
            elif family == "PeriodSentence" and fallback_marker in expression:
                dispatch[family] = {
                    "mode": "period_sentence_pipeline",
                    "entrypoint": fallback_marker,
                }
            else:
                dispatch[family] = {
                    "mode": "named_shared_pipeline",
                    "entrypoint": fallback_marker
                    if fallback_marker in expression
                    else "explicit_family_arm",
                }

    if not fallback_seen:
        raise CoverageError("Grammar dispatcher has no compatibility fallback arm")
    missing = sorted(used_families - set(dispatch))
    if missing:
        raise CoverageError(
            "Grammar families have no explicit dispatch or compatibility fallback: "
            + ", ".join(missing)
        )
    return dispatch


def _runtime_report(root: Path) -> dict[str, Any]:
    style_path = root / STYLE_RELATIVE
    realization_path = root / REALIZATION_RELATIVE
    grammar_path = root / GRAMMAR_RELATIVE
    required = [style_path, realization_path, grammar_path]
    if any(not path.is_file() for path in required):
        raise CoverageError(
            "Run from a checkout containing "
            + ", ".join(str(path.relative_to(root)) for path in required)
        )
    style_bytes = style_path.read_bytes()
    realization_bytes = realization_path.read_bytes()
    grammar_bytes = grammar_path.read_bytes()
    try:
        style_source = style_bytes.decode("utf-8")
        realization_source = realization_bytes.decode("utf-8")
        grammar_source = grammar_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CoverageError(f"Rust runtime source is not UTF-8: {exc}") from exc

    variants, all_names, declared_count = parse_style_contract(style_source)
    family_arms = _method_match_arms(style_source, FAMILY_METHOD, FAMILY_METHOD)
    family_expr, _ = _parse_mapping(family_arms, variants, "grammar_family")
    grammar_families = {style: expr.rstrip(",") for style, expr in family_expr.items()}

    axes_method = _block_after(style_source, AXES_METHOD)
    constants = _string_array_constants(axes_method)
    axis_arms = _match_arms(axes_method, r"match\s+self\s*\{")
    axis_refs, axis_default = _parse_mapping(
        axis_arms, variants, "supported_intent_axes", allow_default=True
    )
    axes_by_style: dict[str, list[str]] = {}
    for style in variants:
        reference = axis_refs.get(style, axis_default)
        if reference is None:
            raise CoverageError(f"supported_intent_axes does not cover {style}")
        if reference not in constants:
            raise CoverageError(
                f"supported_intent_axes for {style} references unknown constant {reference!r}"
            )
        axes_by_style[style] = constants[reference]

    meter_arms = _method_match_arms(style_source, METER_METHOD, METER_METHOD)
    meter_exprs, meter_default = _parse_mapping(
        meter_arms, variants, "meter", allow_default=True
    )
    if meter_default is None or not re.fullmatch(r"\d+", meter_default):
        raise CoverageError("meter must declare an explicit numeric wildcard default")
    meters = {
        style: int(meter_exprs.get(style, meter_default).rstrip(","))
        for style in variants
    }
    if any(meter < 1 or meter > 64 for meter in meters.values()):
        raise CoverageError("meter values must be between 1 and 64 beats")

    tempo_arms = _method_match_arms(style_source, TEMPO_METHOD, TEMPO_METHOD)
    tempo_exprs, _ = _parse_mapping(tempo_arms, variants, "tempo_range")
    tempos: dict[str, list[float]] = {}
    for style, expression in tempo_exprs.items():
        match = re.fullmatch(
            r"\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)",
            expression.rstrip(","),
        )
        if match is None:
            raise CoverageError(f"Unrecognized tempo range for {style}: {expression!r}")
        low, high = float(match.group(1)), float(match.group(2))
        if low <= 0 or high < low:
            raise CoverageError(f"Invalid tempo range for {style}: {(low, high)}")
        tempos[style] = [low, high]

    spec_styles = _extract_style_spec_variants(style_source, variants)
    family_dispatch = _parse_grammar_dispatch(grammar_source, set(grammar_families.values()))
    realization_arms = _method_match_arms(
        realization_source, REALIZATION_METHOD, REALIZATION_METHOD
    )
    pool_exprs, _ = _parse_mapping(realization_arms, variants, "instruments_for")
    ensembles: dict[str, dict[str, Any]] = {}
    for style, expression in pool_exprs.items():
        code = _strip_comments(expression)
        instruments = sorted(set(re.findall(r"\bInstrument::([A-Z][A-Za-z0-9_]*)\b", code)))
        choices = len(re.findall(
            r"\(\s*Instrument::[A-Z][A-Za-z0-9_]*\s*,\s*"
            r"Instrument::[A-Z][A-Za-z0-9_]*\s*,\s*"
            r"Instrument::[A-Z][A-Za-z0-9_]*(?:\s*,)?\s*\)",
            code,
        ))
        if not instruments or choices == 0:
            raise CoverageError(
                f"instruments_for arm for {style} has no recognizable 3-instrument pool"
            )
        ensembles[style] = {
            "instrument_names": instruments,
            "ensemble_choices": choices,
        }

    profiles = []
    for style in variants:
        family = grammar_families[style].split("::")[-1]
        profile = {
            "style": style,
            "grammar_family": family,
            "grammar_dispatch": family_dispatch[family],
            "composition_spec_declared": style in spec_styles,
            "meter_beats": meters[style],
            "tempo_range_bpm": tempos[style],
            "supported_intent_axes": axes_by_style[style],
            "intent_axis_count": len(axes_by_style[style]),
            "instrument_pool": ensembles[style],
        }
        profiles.append(profile)

    axis_names = sorted({axis for axes in axes_by_style.values() for axis in axes})
    axis_counts = {
        axis: sum(axis in axes_by_style[style] for style in variants)
        for axis in axis_names
    }
    family_counts: dict[str, int] = {}
    for family in grammar_families.values():
        family_counts[family.split("::")[-1]] = family_counts.get(
            family.split("::")[-1], 0
        ) + 1

    return {
        "schema": "muse-runtime-style-capability-source-inventory-v1",
        "kind": "source_declared_capability_inventory_not_render_validation",
        "source": {
            "style_path": STYLE_RELATIVE.as_posix(),
            "style_blob_sha": git_blob_sha(style_bytes),
            "realization_path": REALIZATION_RELATIVE.as_posix(),
            "realization_blob_sha": git_blob_sha(realization_bytes),
            "grammar_dispatch_path": GRAMMAR_RELATIVE.as_posix(),
            "grammar_dispatch_blob_sha": git_blob_sha(grammar_bytes),
            "style_all_variant_count": declared_count,
        },
        "summary": {
            "executable_styles": len(variants),
            "styles_with_grammar_family": len(grammar_families),
            "styles_with_composition_spec": len(spec_styles),
            "styles_with_intent_axis_declarations": len(axes_by_style),
            "styles_with_instrument_pool": len(ensembles),
            "styles_with_dedicated_grammar_engine": sorted(
                style
                for style in variants
                if family_dispatch[grammar_families[style].split("::")[-1]]["mode"]
                == "dedicated_engine"
            ),
            "styles_using_compatibility_fallback": sorted(
                style
                for style in variants
                if family_dispatch[grammar_families[style].split("::")[-1]]["mode"]
                == "compatibility_fallback"
            ),
            "grammar_dispatch_modes": dict(
                sorted(
                    {
                        mode: sum(
                            family_dispatch[grammar_families[style].split("::")[-1]]["mode"] == mode
                            for style in variants
                        )
                        for mode in {
                            family_dispatch[grammar_families[style].split("::")[-1]]["mode"]
                            for style in variants
                        }
                    }.items()
                )
            ),
            "grammar_family_counts": dict(sorted(family_counts.items())),
            "intent_axis_style_counts": dict(sorted(axis_counts.items())),
            "styles_with_fewer_than_four_intent_axes": sorted(
                style for style in variants if len(axes_by_style[style]) < 4
            ),
            "routed_does_not_prove_distinct_or_rendered": True,
        },
        "profiles": profiles,
        "interpretation": (
            "These fields are extracted from executable source declarations, "
            "including the existing census-informed intent-axis declaration. "
            "This inventory does not independently rerun causal-effect probes, "
            "prove sample/render availability, or validate perceptual distinctness "
            "or cultural authenticity."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root (default: parent of the scripts directory)",
    )
    args = parser.parse_args(argv)
    try:
        report = _runtime_report(args.root.resolve())
    except (CoverageError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    summary = report["summary"]
    print(
        "Source inventory: {executable_styles} executable styles; "
        "{styles_with_grammar_family} grammar mappings; "
        "{styles_with_composition_spec} CompositionSpec presets; "
        "{styles_with_instrument_pool} instrument pools".format(**summary),
        file=sys.stderr,
    )
    print(
        "Interpretation: source-declared controls only; no claim of causal "
        "effect, renderer availability, perceptual distinctness, or authenticity.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
