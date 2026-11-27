#!/usr/bin/env python3
"""Validate REPAIR-Bench synthetic fixture identities and declared transitions.

This checks fixture consistency only. It is not an agent run, scientific
adjudication, graph-update implementation, or benchmark scorer.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "docs/research/repair-bench/REPAIR_BENCH_V0_SYNTHETIC_EVENT_FIXTURES.json"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rows(raw: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(raw)))


def validate() -> None:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    require(document["benchmark"] == "REPAIR-Bench", "unexpected benchmark name")
    require(document["fixture_class"] == "synthetic_contract_only", "fixture must remain marked synthetic")

    baseline = document["baseline"]
    artifacts = {a["artifact_id"]: a for a in baseline["artifacts"]}
    require(set(artifacts) == {"artifact:data.csv", "artifact:protocol.json"}, "baseline artifact set mismatch")
    for artifact in artifacts.values():
        raw = artifact["raw_bytes_utf8"].encode("utf-8")
        require(len(raw) == artifact["raw_byte_length"], f"{artifact['artifact_id']}: byte length mismatch")
        require(sha256(raw) == artifact["raw_sha256"], f"{artifact['artifact_id']}: SHA-256 mismatch")

    nodes: dict[str, str] = {}
    for collection, kind in (
        ("observations", "observation"),
        ("analysis_nodes", "analysis"),
        ("claims", "claim"),
        ("decisions", "decision"),
    ):
        for item in baseline[collection]:
            require(item["id"] not in nodes, f"duplicate graph node {item['id']}")
            nodes[item["id"]] = kind
    require(
        (sum(v == "observation" for v in nodes.values()),
         sum(v == "analysis" for v in nodes.values()),
         sum(v == "claim" for v in nodes.values()),
         sum(v == "decision" for v in nodes.values())) == (7, 2, 4, 2),
        "baseline graph cardinalities differ from the declared first-slice fixture",
    )

    data_bytes = artifacts["artifact:data.csv"]["raw_bytes_utf8"].encode("utf-8")
    for observation in baseline["observations"]:
        locator = observation["locator"]
        if locator["kind"] == "CSV_CELL":
            start, end = locator["byte_start"], locator["byte_end"]
            require(0 <= start < end <= len(data_bytes), f"{observation['id']}: CSV span out of range")
            actual = data_bytes[start:end].decode("utf-8")
            require(actual == observation["raw_lexeme"], f"{observation['id']}: CSV span does not resolve")
        elif locator["kind"] == "JSON_POINTER":
            source_artifact = artifacts[observation["artifact_id"]]
            json_document = json.loads(source_artifact["raw_bytes_utf8"])
            pointer = locator["pointer"]
            target = json_document
            if pointer:
                require(pointer.startswith("/"), f"{observation['id']}: invalid JSON Pointer")
                for encoded_part in pointer[1:].split("/"):
                    part = encoded_part.replace("~1", "/").replace("~0", "~")
                    if isinstance(target, list):
                        require(part.isdigit() and int(part) < len(target),
                                f"{observation['id']}: JSON Pointer list index invalid")
                        target = target[int(part)]
                    else:
                        require(isinstance(target, dict) and part in target,
                                f"{observation['id']}: JSON Pointer target missing")
                        target = target[part]
            if isinstance(target, str):
                require(target == observation["raw_lexeme"],
                        f"{observation['id']}: JSON Pointer string value mismatch")
            elif isinstance(target, (dict, list)):
                require(json.dumps(target, separators=(",", ":"), ensure_ascii=False) == observation["raw_lexeme"],
                        f"{observation['id']}: JSON Pointer structured raw lexeme mismatch")
            else:
                require(str(target) == observation["raw_lexeme"],
                        f"{observation['id']}: JSON Pointer scalar raw lexeme mismatch")
        else:
            raise ValueError(f"{observation['id']}: unknown locator kind {locator['kind']}")

    for source, target in baseline["edges"]:
        require(source in nodes or source in artifacts, f"edge source is unknown: {source}")
        require(target in nodes or target in artifacts, f"edge target is unknown: {target}")

    # Verify the intentionally flawed baseline arithmetic and method comparator.
    obs = {o["id"]: o for o in baseline["observations"]}
    mean = (obs["obs_temp_s01"]["parsed_value"] + obs["obs_temp_s02"]["parsed_value"]) / 2
    analysis = next(a for a in baseline["analysis_nodes"] if a["id"] == "analysis_mean_temperature")
    require(mean == analysis["output"]["value"] == 20.0, "baseline mean fixture is inconsistent")
    protocol = json.loads(artifacts["artifact:protocol.json"]["raw_bytes_utf8"])
    require(protocol["threshold_C"] == 20.0 and protocol["comparison_rule"] == "at_or_above",
            "baseline protocol target changed")
    require(baseline["method"]["comparison_implementation"] == "strict_greater_than",
            "fixture must preserve the declared comparator defect")
    require(not (mean > protocol["threshold_C"]), "baseline strict-comparator control should be false")

    events = document["events"]
    ids = [event["event_id"] for event in events]
    require(len(ids) == 12 and len(set(ids)) == len(ids), "expected 12 uniquely identified events")
    for event in events:
        mutation = event["mutation"]
        if "new_artifact_bytes_utf8" in mutation:
            raw = mutation["new_artifact_bytes_utf8"].encode("utf-8")
            require(mutation.get("new_artifact_sha256") == sha256(raw),
                    f"{event['event_id']}: mutated artifact digest mismatch")
        elif "new_artifact_sha256" in mutation:
            raise ValueError(f"{event['event_id']}: digest supplied without artifact bytes")

        if event["event_id"] == "E09":
            require(mutation["mirror_sha256"] == artifacts["artifact:data.csv"]["raw_sha256"],
                    "E09 mirror must share the original artifact digest")
        expected = event["expected"]
        groups = [
            set(expected.get("reassessed_changed", [])),
            set(expected.get("reassessed_unchanged", [])),
            set(expected.get("blocked", [])),
            set(expected.get("unresolved", [])),
            set(expected.get("not_dependent", [])),
        ]
        for group in groups:
            require(group <= set(nodes), f"{event['event_id']}: expected node is absent from baseline graph")
        for i, group in enumerate(groups):
            for other in groups[i + 1:]:
                require(group.isdisjoint(other), f"{event['event_id']}: expected outcome groups overlap")
        combined = set().union(*groups)
        require(combined == set(nodes),
                f"{event['event_id']}: expected outcomes do not partition the complete baseline graph")
        ignored = expected.get("ignored_untrusted", [])
        require(all(isinstance(value, str) for value in ignored),
                f"{event['event_id']}: ignored-untrusted records must be string IDs")

    # Row reorder changes raw identity but preserves the canonical table records.
    baseline_rows = rows(artifacts["artifact:data.csv"]["raw_bytes_utf8"])
    reorder_event = next(event for event in events if event["event_id"] == "E10")
    reordered_rows = rows(reorder_event["mutation"]["new_artifact_bytes_utf8"])
    require(sorted(tuple(sorted(r.items())) for r in baseline_rows) ==
            sorted(tuple(sorted(r.items())) for r in reordered_rows),
            "E10 rows are not semantically equivalent after normalization")

    # Independently verify the deterministic numerical event expectations.
    e01 = next(event for event in events if event["event_id"] == "E01")
    corrected_rows = rows(e01["mutation"]["new_artifact_bytes_utf8"])
    corrected_mean = (int(float(corrected_rows[0]["temperature_C"])) +
                      int(float(corrected_rows[1]["temperature_C"]))) / 2
    require(corrected_mean == 20.5 and corrected_mean > protocol["threshold_C"],
            "E01 does not cross the strict baseline threshold as intended")

    e04 = next(event for event in events if event["event_id"] == "E04")
    threshold = json.loads(e04["mutation"]["new_artifact_bytes_utf8"])["threshold_C"]
    require(mean > threshold, "E04 threshold correction should change the threshold claim")

    e05 = next(event for event in events if event["event_id"] == "E05")
    require(e05["mutation"]["new_decimal_policy"] == "preserve_decimal_lexeme",
            "E05 parser correction policy mismatch")
    exact_mean = (21.5 + 19.0) / 2
    require(exact_mean > protocol["threshold_C"], "E05 should cross the baseline threshold")

    e06 = next(event for event in events if event["event_id"] == "E06")
    require(e06["mutation"]["new_comparison_implementation"] == "greater_than_or_equal",
            "E06 comparator correction mismatch")
    require(mean >= protocol["threshold_C"], "E06 should make the baseline threshold claim true")

    e07 = next(event for event in events if event["event_id"] == "E07")
    scope = json.loads(e07["mutation"]["new_artifact_bytes_utf8"])["included_samples"]
    scoped_mean = sum(float(r["temperature_C"]) for r in baseline_rows if r["sample_id"] in scope) / len(scope)
    require(scoped_mean == 19.0 and scoped_mean < protocol["threshold_C"],
            "E07 scope correction should change the mean but leave the threshold result false")

    print("REPAIR-Bench fixture: 2 raw artifact hashes, 7 source locators, 15 graph nodes and 12 events passed consistency checks")
    print("This validates fixture consistency only; expected transitions still require independent review.")


def main() -> int:
    try:
        validate()
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"REPAIR-Bench fixture validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
