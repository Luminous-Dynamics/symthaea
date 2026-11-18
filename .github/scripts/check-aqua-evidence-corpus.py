#!/usr/bin/env python3
"""Validate the versioned Luminous Aqua evidence corpus.

CI-only documentation/data validator. It does not calculate product performance,
qualify instruments, assess hygiene, or authorize standards/compliance claims.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "docs/research/aqua-evidence-corpus-v0.1.schema.json"
CORPUS_PATH = ROOT / "docs/research/aqua-evidence-corpus-v0.1.json"
RENDER_PATH = ROOT / "docs/research/AQUA_EVIDENCE_CORPUS_V0_1.md"

ALLOWED_SOURCE_TYPES = {
    "manufacturer_declaration": {"manufacturer_datasheet", "manufacturer_product_page"},
    "independent_laboratory_measurement": {"independent_market_study"},
    "standard_or_programme_criterion": {
        "official_standard_or_programme",
        "industry_association_notice",
    },
    "installation_design_reference": {"official_standard_or_programme"},
    "peer_reviewed_study_result": {"peer_reviewed_article"},
    "independent_market_sample": {"independent_market_study"},
}

EXPECTED_VALUES = {
    "flow_rate": ("scalar", "L/min"),
    "supply_pressure_range": ("range", "kPa"),
    "maximum_flow_rate": ("scalar", "L/min"),
    "flow_regulator_setting": ("scalar", "L/min"),
    "task_selective_boost_feature": ("text", None),
    "manufacturer_specification_scope_ambiguity": ("text", None),
    "minimum_operating_pressure": ("scalar", "kPa"),
    "maximum_operating_pressure": ("range", "kPa"),
    "minimum_function_flow": ("scalar", "L/min"),
    "operating_pressure_range": ("range", "kPa"),
    "pressure_compensation_feature": ("text", None),
    "thermostat_configuration": ("text", None),
    "temperature_safety_feature": ("text", None),
    "task_selective_boost_feature": ("text", None),
    "manufacturer_specification_scope_ambiguity": ("text", None),
    "maximum_rated_flow": ("scalar", "L/min"),
    "minimum_flow_fraction": ("scalar", "% of rated maximum"),
    "minimum_spray_force": ("scalar", "N"),
    "spray_coverage_criteria": ("text", None),
    "design_flow_and_pressure": ("text", None),
    "maximum_shower_discharge": ("scalar", "L/min"),
    "sample_size": ("scalar", "products"),
    "mean_flow_rate": ("scalar", "L/min"),
    "observed_flow_rate_range": ("range", "L/min"),
    "water_reduction_at_similar_mean_jet_velocity": ("scalar", "%"),
    "water_chemistry_observation": ("text", None),
    "potential_energy_reduction": ("scalar", "%"),
    "influential_design_parameters": ("text", None),
    "field_study_duration": ("scalar", "months"),
    "overall_heat_exchanger_efficiency": ("scalar", "%"),
    "fouling_nominal_efficiency_reduction": ("scalar", "%"),
    "overall_dhwr_system_efficiency": ("scalar", "%"),
    "global_dhw_cop": ("scalar", "COP"),
    "heat_pump_cop_reduction": ("scalar", "%"),
    "preferred_dhwr_coupling_topology": ("text", None),
    "model_validation_cv_rmse": ("scalar", "%"),
    "functional_test_scope": ("text", None),
    "shower_outlet_operating_pressure": ("text", None),
    "tap_shower_system_operating_pressure": ("text", None),
    "temperature_operating_limits": ("text", None),
    "uwl_dynamic_pressure_point": ("text", None),
    "test_instrument_calibration_requirement": ("text", None),
    "boost_function_definition": ("text", None),
    "boost_function_criteria": ("text", None),
    "design_for_disassembly_scope": ("text", None),
    "repairability_assessment_scope": ("text", None),
}


def fail(message: str) -> None:
    print(f"AQUA-EVIDENCE-CORPUS: FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot load {path.relative_to(ROOT)}: {exc}")
    if not isinstance(value, dict):
        fail(f"{path.relative_to(ROOT)} must contain a JSON object")
    return value


def pointer(path_parts: object) -> str:
    return "/" + "/".join(str(part) for part in path_parts)


def semantic_errors(corpus: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    entry_ids: set[str] = set()
    claim_ids: set[str] = set()

    for entry_index, entry in enumerate(corpus["entries"]):
        entry_id = entry["entry_id"]
        if entry_id in entry_ids:
            errors.append(f"/entries/{entry_index}/entry_id duplicates {entry_id}")
        entry_ids.add(entry_id)

        subject = entry["subject"]
        if subject["type"] == "product":
            for field in ("manufacturer", "sku", "market"):
                if not subject[field]:
                    errors.append(
                        f"/entries/{entry_index}/subject/{field}: "
                        "product identity must be exact"
                    )

        for claim_index, claim in enumerate(entry["claims"]):
            location = f"/entries/{entry_index}/claims/{claim_index}"
            claim_id = claim["claim_id"]
            if claim_id in claim_ids:
                errors.append(f"{location}/claim_id duplicates {claim_id}")
            claim_ids.add(claim_id)

            source_type = claim["source"]["source_type"]
            evidence_class = claim["evidence_class"]
            if source_type not in ALLOWED_SOURCE_TYPES[evidence_class]:
                errors.append(
                    f"{location}: evidence class {evidence_class} "
                    f"cannot be supported by source type {source_type}"
                )

            expected = EXPECTED_VALUES.get(claim["parameter"])
            if expected:
                expected_kind, expected_unit = expected
                canonical = claim["canonical_value"]
                if canonical["kind"] != expected_kind:
                    errors.append(
                        f"{location}: parameter {claim['parameter']} "
                        f"requires {expected_kind}, got {canonical['kind']}"
                    )
                if expected_unit and canonical.get("unit") != expected_unit:
                    errors.append(
                        f"{location}: parameter {claim['parameter']} "
                        f"requires unit {expected_unit}, got {canonical.get('unit')}"
                    )

            canonical = claim["canonical_value"]
            if canonical["kind"] == "range":
                minimum = canonical.get("minimum")
                maximum = canonical.get("maximum")
                if minimum is not None and minimum <= 0:
                    errors.append(f"{location}: range minimum must be positive")
                if maximum is not None and maximum <= 0:
                    errors.append(f"{location}: range maximum must be positive")
                if minimum is not None and maximum is not None and maximum < minimum:
                    errors.append(
                        f"{location}: range maximum {maximum} is below minimum {minimum}"
                    )

            if claim["evidence_class"] == "manufacturer_declaration":
                if claim["result_class"] not in {"declared", "derived"}:
                    errors.append(
                        f"{location}: manufacturer declaration cannot be relabelled "
                        f"as {claim['result_class']}"
                    )
            if claim["evidence_class"] == "independent_market_sample" and (
                claim["result_class"] != "measured"
            ):
                errors.append(
                    f"{location}: independent market sample must retain measured status"
                )
            if claim["evidence_class"] == "standard_or_programme_criterion" and (
                claim["result_class"] not in {"declared", "derived"}
            ):
                errors.append(
                    f"{location}: standard/programme criterion cannot be relabelled "
                    f"as {claim['result_class']}"
                )

            if claim["result_class"] == "derived" and "derivation" not in claim:
                errors.append(f"{location}: derived result is missing its derivation record")

            if claim["result_class"] == "proposed" and (
                claim["comparison"]["status"] != "context_only"
            ):
                errors.append(
                    f"{location}: proposed value cannot be marked as a matched-test reference"
                )

            conditions = claim["conditions"]
            if claim["parameter"] == "flow_rate":
                if not conditions["mode"]:
                    errors.append(f"{location}: flow-rate claim must name its exact mode")
                pressure_known = (
                    conditions["pressure_kpa"] is not None
                    or conditions.get("pressure_range_kpa") is not None
                    or bool(conditions.get("pressure_points_kpa", []))
                )
                pressure_explicitly_unknown = any(
                    "pressure" in item.lower()
                    for item in conditions["unknown_conditions"]
                )
                if not pressure_known and not pressure_explicitly_unknown:
                    errors.append(
                        f"{location}: flow-rate claim must state its pressure condition "
                        "or explicitly list pressure as unknown"
                    )

            if evidence_class == "manufacturer_declaration" and subject["type"] != "product":
                errors.append(
                    f"{location}: manufacturer declaration must attach to an exact product"
                )
            if evidence_class == "peer_reviewed_study_result" and subject["type"] != "research_study":
                errors.append(
                    f"{location}: peer-reviewed result must attach to a research-study entry"
                )

    if not entry_ids or not claim_ids:
        errors.append("corpus contains no entries or claims")
    return errors


def schema_errors(validator: Draft202012Validator, corpus: dict[str, Any]) -> list[str]:
    errors = sorted(
        validator.iter_errors(corpus),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    return [
        f"{pointer(error.absolute_path)}: {error.message}"
        for error in errors
    ]


def require_negative_control(
    name: str,
    mutated: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    rejected = bool(schema_errors(validator, mutated) or semantic_errors(mutated))
    if not rejected:
        fail(f"negative control was incorrectly accepted: {name}")


def first_claim(corpus: dict[str, Any], entry_id: str, claim_id: str) -> dict[str, Any]:
    for entry in corpus["entries"]:
        if entry["entry_id"] == entry_id:
            for claim in entry["claims"]:
                if claim["claim_id"] == claim_id:
                    return claim
    fail(f"negative-control fixture target not found: {entry_id}/{claim_id}")


def run_negative_controls(
    corpus: dict[str, Any],
    validator: Draft202012Validator,
) -> int:
    controls = 0

    missing_sku = copy.deepcopy(corpus)
    missing_sku["entries"][0]["subject"].pop("sku")
    require_negative_control("missing exact product SKU", missing_sku, validator)
    controls += 1

    wrong_unit = copy.deepcopy(corpus)
    claim = first_claim(
        wrong_unit,
        "brondell-nebia-corre-n400ro",
        "brondell.nebia.super-saver.flow",
    )
    claim["canonical_value"]["unit"] = "gpm"
    require_negative_control("canonical unit mismatch", wrong_unit, validator)
    controls += 1

    wrong_evidence_class = copy.deepcopy(corpus)
    claim = first_claim(
        wrong_evidence_class,
        "brondell-nebia-corre-n400ro",
        "brondell.nebia.super-saver.flow",
    )
    claim["evidence_class"] = "independent_laboratory_measurement"
    require_negative_control(
        "manufacturer declaration relabelled as independent measurement",
        wrong_evidence_class,
        validator,
    )
    controls += 1

    pressure_laundering = copy.deepcopy(corpus)
    claim = first_claim(
        pressure_laundering,
        "brondell-nebia-corre-n400ro",
        "brondell.nebia.super-saver.flow",
    )
    claim["conditions"]["unknown_conditions"] = []
    require_negative_control(
        "unspecified flow pressure silently omitted",
        pressure_laundering,
        validator,
    )
    controls += 1

    mode_confusion = copy.deepcopy(corpus)
    claim = first_claim(
        mode_confusion,
        "hansgrohe-raindance-select-e-28729000",
        "hansgrohe.28729000.rain.flow",
    )
    claim["conditions"]["mode"] = None
    require_negative_control("spray mode omitted", mode_confusion, validator)
    controls += 1

    missing_derivation = copy.deepcopy(corpus)
    claim = first_claim(
        missing_derivation,
        "shower-pattern-thermal-transfer-study-2024",
        "thermal-pattern.energy-reduction",
    )
    claim.pop("derivation")
    require_negative_control("derived result without derivation", missing_derivation, validator)
    controls += 1

    proposed_as_result = copy.deepcopy(corpus)
    claim = first_claim(
        proposed_as_result,
        "brondell-nebia-corre-n400ro",
        "brondell.nebia.super-saver.flow",
    )
    claim["result_class"] = "proposed"
    require_negative_control(
        "proposed target presented as matched-test reference",
        proposed_as_result,
        validator,
    )
    controls += 1

    inverted_range = copy.deepcopy(corpus)
    claim = first_claim(
        inverted_range,
        "brondell-nebia-corre-n400ro",
        "brondell.nebia.inlet-pressure.range",
    )
    claim["canonical_value"]["minimum"] = 600.0
    claim["canonical_value"]["maximum"] = 500.0
    require_negative_control(
        "inverted manufacturer pressure range",
        inverted_range,
        validator,
    )
    controls += 1

    return controls


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def format_canonical_value(value: dict[str, Any]) -> str:
    kind = value["kind"]
    if kind == "scalar":
        return f"{value['value']:g} {value['unit']}"
    if kind == "range":
        unit = value["unit"]
        minimum = value.get("minimum")
        maximum = value.get("maximum")
        if minimum is not None and maximum is not None:
            return f"{minimum:g}–{maximum:g} {unit}"
        if minimum is not None:
            return f"minimum ≥ {minimum:g} {unit}"
        return f"maximum ≤ {maximum:g} {unit}"
    return value["text"]


def format_conditions(conditions: dict[str, Any]) -> str:
    parts: list[str] = []
    if conditions.get("mode"):
        parts.append(f"mode={conditions['mode']}")
    if conditions.get("pressure_kpa") is not None:
        parts.append(f"pressure={conditions['pressure_kpa']:g} kPa")
    pressure_range = conditions.get("pressure_range_kpa")
    if pressure_range:
        parts.append(f"pressure range={format_canonical_value(pressure_range)}")
    pressure_points = conditions.get("pressure_points_kpa", [])
    if pressure_points:
        parts.append(
            "pressure points="
            + ", ".join(f"{point:g}" for point in pressure_points)
            + " kPa"
        )
    if conditions.get("pressure_tolerance_kpa") is not None:
        parts.append(f"pressure tolerance=±{conditions['pressure_tolerance_kpa']:g} kPa")
    if conditions.get("temperature_c") is not None:
        parts.append(f"water temperature={conditions['temperature_c']:g} °C")
    if conditions.get("duration_min") is not None:
        parts.append(f"duration={conditions['duration_min']:g} min")
    if conditions.get("task"):
        parts.append(f"task={conditions['task']}")
    if conditions.get("method"):
        parts.append(f"method={conditions['method']}")
    if conditions.get("sample_count") is not None:
        parts.append(f"sample n={conditions['sample_count']}")
    if conditions.get("pressure_datum"):
        parts.append(f"datum={conditions['pressure_datum']}")
    if conditions.get("unknown_conditions"):
        parts.append(
            "unknown: " + ", ".join(conditions["unknown_conditions"])
        )
    if conditions.get("notes"):
        parts.append(conditions["notes"])
    return markdown_escape("; ".join(parts) if parts else "Not specified")


def render_markdown(corpus: dict[str, Any]) -> str:
    claim_total = sum(len(entry["claims"]) for entry in corpus["entries"])
    lines = [
        "# Luminous Aqua — Evidence Corpus v0.1",
        "",
        f"**Snapshot:** {corpus['snapshot_date']}  ",
        f"**Entries:** {len(corpus['entries'])}  ",
        f"**Claims:** {claim_total}",
        "",
        "> Source-classified evidence register. Manufacturer declarations, standards/programme criteria, local installation references, independent market samples and peer-reviewed results are kept distinct. Values marked unknown have not been inferred. This corpus is not a product certification or evidence of Aqua performance.",
        "",
    ]

    for entry in corpus["entries"]:
        subject = entry["subject"]
        lines.extend(
            [
                f"## {markdown_escape(subject['name'])}",
                "",
                f"- **Entry ID:** `{entry['entry_id']}`",
                f"- **Subject type:** {subject['type']}",
                f"- **Manufacturer / publisher:** {markdown_escape(subject.get('manufacturer') or 'Not applicable / not stated')}",
                f"- **SKU:** {markdown_escape(subject.get('sku') or 'Not applicable / not stated')}",
                f"- **Market:** {markdown_escape(subject.get('market') or 'Not stated')}",
                f"- **Revision:** {markdown_escape(subject.get('revision') or 'Not stated')}",
                "",
                "| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for claim in entry["claims"]:
            source = claim["source"]
            source_label = markdown_escape(source["publisher"])
            source_link = f"[{source_label}]({source['url']})"
            source_date = source.get("published_on") or "date not stated"
            source_view = f"{source_link}; {source['source_type']}; {source_date}"
            comparison = (
                f"{claim['comparison']['status']}: "
                f"{claim['comparison']['reason']}"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{claim['claim_id']}`",
                        markdown_escape(claim["parameter"]),
                        markdown_escape(format_canonical_value(claim["canonical_value"])),
                        markdown_escape(claim["evidence_class"]),
                        markdown_escape(claim["result_class"]),
                        format_conditions(claim["conditions"]),
                        markdown_escape(comparison),
                        source_view,
                    ]
                )
                + " |"
            )
        lines.append("")
        for claim in entry["claims"]:
            limitations = claim["limitations"]
            if not limitations:
                continue
            lines.append(f"**Limitations — `{claim['claim_id']}`:**")
            for limitation in limitations:
                lines.append(f"- {markdown_escape(limitation)}")
            lines.append("")
    lines.extend(
        [
            "---",
            "",
            "Generated from `docs/research/aqua-evidence-corpus-v0.1.json` by `.github/scripts/check-aqua-evidence-corpus.py`. Do not edit this rendering independently; change the JSON source and regenerate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-render",
        action="store_true",
        help="fail unless the committed Markdown rendering matches the JSON corpus",
    )
    parser.add_argument(
        "--write-render",
        action="store_true",
        help="write the Markdown rendering from the JSON corpus",
    )
    args = parser.parse_args()
    if args.check_render and args.write_render:
        fail("use only one of --check-render and --write-render")

    schema = load_json(SCHEMA_PATH)
    corpus = load_json(CORPUS_PATH)

    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    errors = schema_errors(validator, corpus) + semantic_errors(corpus)
    if errors:
        for error in errors[:50]:
            fail(error)
        fail(f"{len(errors)} corpus errors; first 50 shown")

    negative_controls = run_negative_controls(corpus, validator)
    claim_count = sum(len(entry["claims"]) for entry in corpus["entries"])
    render = render_markdown(corpus)

    if args.write_render:
        RENDER_PATH.write_text(render, encoding="utf-8")
    if args.check_render:
        try:
            committed_render = RENDER_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            fail(f"cannot read generated rendering {RENDER_PATH.relative_to(ROOT)}: {exc}")
        if committed_render != render:
            fail(
                f"{RENDER_PATH.relative_to(ROOT)} is stale; regenerate from the JSON corpus"
            )

    print(
        "AQUA-EVIDENCE-CORPUS: PASS "
        f"schema={corpus['schema_version']} "
        f"snapshot={corpus['snapshot_date']} "
        f"entries={len(corpus['entries'])} "
        f"claims={claim_count} "
        f"negative_controls={negative_controls} "
        f"render={'checked' if args.check_render else 'written' if args.write_render else 'not_checked'}"
    )


if __name__ == "__main__":
    main()
