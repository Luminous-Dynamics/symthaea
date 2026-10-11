#!/usr/bin/env python3
"""Validate the frozen JUNG-001C development corpus and normalized result schema.

This checks data/contract mechanics only. It does not judge whether a Jungian
interpretation is psychologically true or empirically valid.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "docs/research/data/jungian_symbolic_probe_v1.json"
CORPUS_SCHEMA = "symthaea-jungian-symbolic-probe-v1"
RESULT_SCHEMA = "symthaea-jungian-normalized-results-v1"
EXPECTED_CASE_COUNT = 14
HEX_256 = re.compile(r"^[0-9a-f]{64}$")

CASE_KEYS = {"id", "task", "input_text", "evidence_spans", "required_behaviors", "forbidden_behaviors", "oracle_note"}
SPAN_KEYS = {"id", "text"}
RESULT_TOP_KEYS = {"schema", "corpus_id", "corpus_version", "source_revision", "implementation_id", "config_sha256", "results"}
RESULT_CASE_KEYS = {"case_id", "observations", "interpretations", "open_questions", "abstention"}
OBSERVATION_KEYS = {"evidence_id", "exact_span", "source_scope"}
INTERPRETATION_KEYS = {"hypothesis_id", "statement", "scope", "supporting_evidence_ids", "contradicting_evidence_ids", "epistemic_status"}
ABSTENTION_KEYS = {"abstain", "reason"}


class ValidationError(ValueError):
    pass


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValidationError(f"duplicate JSON object key: {key}")
        out[key] = value
    return out


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def nonempty_string(value: Any, where: str) -> bool:
    return isinstance(value, str) and bool(value.strip())


def exact_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{where}: expected object")
    actual = set(value)
    require(actual == expected, f"{where}: keys differ (missing={sorted(expected - actual)}, extra={sorted(actual - expected)})")
    return value


def unique_strings(value: Any, where: str, nonempty: bool = True) -> list[str]:
    require(isinstance(value, list), f"{where}: expected array")
    require(all(isinstance(item, str) and (not nonempty or item.strip()) for item in value), f"{where}: expected string entries")
    require(len(set(value)) == len(value), f"{where}: duplicate entries")
    return value


def validate_corpus(data: Any) -> list[str]:
    errors: list[str] = []
    try:
        require(isinstance(data, dict), "corpus: expected object")
        require(data.get("schema") == CORPUS_SCHEMA, "corpus: unexpected schema")
        require(data.get("corpus_id") == "JUNG-001C-DEV-001", "corpus: unexpected corpus_id")
        require(data.get("version") == 1, "corpus: version mismatch; version and validator must be reviewed together")
        require(data.get("status") == "DevelopmentProbe", "corpus: status must remain DevelopmentProbe")
        require(data.get("split") == "ExposedDevelopment", "corpus: split must remain ExposedDevelopment")
        for field in ("source_policy", "purpose"):
            require(nonempty_string(data.get(field), f"corpus.{field}"), f"corpus: missing {field}")
        require(isinstance(data.get("claim_ceiling"), list) and data["claim_ceiling"], "corpus: claim_ceiling must be a nonempty array")

        behavior = set(unique_strings(data.get("behavior_vocabulary"), "behavior_vocabulary"))
        forbidden = set(unique_strings(data.get("forbidden_vocabulary"), "forbidden_vocabulary"))
        require(not (behavior & forbidden), "behavior vocabularies overlap")
        require(isinstance(data.get("output_contract"), dict), "output_contract: expected object")
        cases = data.get("cases")
        require(isinstance(cases, list), "cases: expected array")
        require(len(cases) == EXPECTED_CASE_COUNT, f"cases: expected {EXPECTED_CASE_COUNT}, got {len(cases)}")
        case_ids: set[str] = set()
        evidence_ids: set[str] = set()

        for index, case in enumerate(cases):
            where = f"cases[{index}]"
            exact_keys(case, CASE_KEYS, where)
            cid = case["id"]
            require(nonempty_string(cid, f"{where}.id"), f"{where}: empty id")
            require(cid not in case_ids, f"{where}: duplicate case id {cid}")
            case_ids.add(cid)
            require(nonempty_string(case["task"], f"{where}.task"), f"{where}: empty task")
            require(nonempty_string(case["input_text"], f"{where}.input_text"), f"{where}: empty input_text")
            require(nonempty_string(case["oracle_note"], f"{where}.oracle_note"), f"{where}: empty oracle_note")

            spans = case["evidence_spans"]
            require(isinstance(spans, list) and spans, f"{where}: evidence_spans must be nonempty")
            local_spans: set[str] = set()
            for sindex, span in enumerate(spans):
                sw = f"{where}.evidence_spans[{sindex}]"
                exact_keys(span, SPAN_KEYS, sw)
                sid = span["id"]
                require(nonempty_string(sid, f"{sw}.id"), f"{sw}: empty id")
                require(sid not in evidence_ids, f"{sw}: duplicate global evidence id {sid}")
                require(sid not in local_spans, f"{sw}: duplicate local evidence id {sid}")
                evidence_ids.add(sid)
                local_spans.add(sid)
                require(nonempty_string(span["text"], f"{sw}.text"), f"{sw}: empty evidence text")
                require(span["text"] in case["input_text"], f"{sw}: evidence text is not an exact substring of input_text")

            required = set(unique_strings(case["required_behaviors"], f"{where}.required_behaviors"))
            banned = set(unique_strings(case["forbidden_behaviors"], f"{where}.forbidden_behaviors"))
            require(bool(required), f"{where}: at least one required behavior is necessary")
            require(not (required & banned), f"{where}: a behavior is both required and forbidden")
            require(required <= behavior, f"{where}: unknown required behavior codes {sorted(required - behavior)}")
            require(banned <= forbidden, f"{where}: unknown forbidden behavior codes {sorted(banned - forbidden)}")
        require(case_ids == {f"JSP-{i:03d}" for i in range(1, EXPECTED_CASE_COUNT + 1)}, "case IDs must be the frozen JSP-001 through JSP-014 set")
    except (ValidationError, TypeError, KeyError) as exc:
        errors.append(str(exc))
    return errors


def validate_results(data: Any, corpus: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        top = exact_keys(data, RESULT_TOP_KEYS, "results")
        require(top.get("schema") == RESULT_SCHEMA, "results: unexpected schema")
        require(top.get("corpus_id") == corpus["corpus_id"], "results: corpus_id mismatch")
        require(top.get("corpus_version") == corpus["version"], "results: corpus_version mismatch")
        require(nonempty_string(top.get("source_revision"), "source_revision"), "results: missing source_revision")
        require(nonempty_string(top.get("implementation_id"), "implementation_id"), "results: missing implementation_id")
        require(isinstance(top.get("config_sha256"), str) and HEX_256.fullmatch(top["config_sha256"]) is not None, "results: config_sha256 must be 64 lowercase hex characters")
        require(isinstance(top.get("results"), list), "results.results: expected array")

        expected_cases = {case["id"]: case for case in corpus["cases"]}
        returned_ids: set[str] = set()
        for index, result in enumerate(top["results"]):
            where = f"results.results[{index}]"
            exact_keys(result, RESULT_CASE_KEYS, where)
            cid = result["case_id"]
            require(isinstance(cid, str) and cid in expected_cases, f"{where}: unknown case_id {cid!r}")
            require(cid not in returned_ids, f"{where}: duplicate case_id {cid}")
            returned_ids.add(cid)
            spans = {item["id"]: item["text"] for item in expected_cases[cid]["evidence_spans"]}

            observations = result["observations"]
            require(isinstance(observations, list), f"{where}.observations: expected array")
            observed_ids: set[str] = set()
            for oi, obs in enumerate(observations):
                ow = f"{where}.observations[{oi}]"
                exact_keys(obs, OBSERVATION_KEYS, ow)
                eid = obs["evidence_id"]
                require(isinstance(eid, str) and eid in spans, f"{ow}: unknown evidence_id {eid!r}")
                require(eid not in observed_ids, f"{ow}: duplicate evidence reference {eid}")
                observed_ids.add(eid)
                require(obs["exact_span"] == spans[eid], f"{ow}: evidence span does not match frozen source text")
                require(nonempty_string(obs["source_scope"], f"{ow}.source_scope"), f"{ow}: missing source_scope")

            interpretations = result["interpretations"]
            require(isinstance(interpretations, list), f"{where}.interpretations: expected array")
            hypothesis_ids: set[str] = set()
            for hi, hypothesis in enumerate(interpretations):
                hw = f"{where}.interpretations[{hi}]"
                exact_keys(hypothesis, INTERPRETATION_KEYS, hw)
                hid = hypothesis["hypothesis_id"]
                require(nonempty_string(hid, f"{hw}.hypothesis_id"), f"{hw}: empty hypothesis_id")
                require(hid not in hypothesis_ids, f"{hw}: duplicate hypothesis_id {hid}")
                hypothesis_ids.add(hid)
                require(nonempty_string(hypothesis["statement"], f"{hw}.statement"), f"{hw}: empty statement")
                require(nonempty_string(hypothesis["scope"], f"{hw}.scope"), f"{hw}: empty scope")
                require(hypothesis["epistemic_status"] in {"InterpretationHypothesis", "ScholarlyInterpretation", "OpenQuestion"}, f"{hw}: invalid epistemic_status")
                supports = unique_strings(hypothesis["supporting_evidence_ids"], f"{hw}.supporting_evidence_ids")
                contradicts = unique_strings(hypothesis["contradicting_evidence_ids"], f"{hw}.contradicting_evidence_ids")
                require(set(supports + contradicts) <= set(spans), f"{hw}: reference to evidence outside the frozen case")
                require(not (set(supports) & set(contradicts)), f"{hw}: evidence cannot both support and contradict the same hypothesis")
                if hypothesis["epistemic_status"] != "OpenQuestion":
                    require(bool(supports), f"{hw}: non-question interpretation needs explicit supporting evidence")

            questions = unique_strings(result["open_questions"], f"{where}.open_questions", nonempty=True)
            abstention = exact_keys(result["abstention"], ABSTENTION_KEYS, f"{where}.abstention")
            require(isinstance(abstention["abstain"], bool), f"{where}.abstention.abstain must be boolean")
            require(nonempty_string(abstention["reason"], f"{where}.abstention.reason"), f"{where}: abstention reason must be present even if abstain=false")
        missing = set(expected_cases) - returned_ids
        require(not missing, f"results: missing cases {sorted(missing)}")
    except (ValidationError, TypeError, KeyError) as exc:
        errors.append(str(exc))
    return errors


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_object)


def self_test(data: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if validate_corpus(data):
        failures.append("canonical corpus unexpectedly fails")
    mutations: list[tuple[str, Any, str]] = []
    mutation = copy.deepcopy(data)
    mutation["cases"][0]["evidence_spans"][0]["text"] = "not an exact source span"
    mutations.append(("reject altered evidence", mutation, "exact substring"))
    mutation = copy.deepcopy(data)
    mutation["cases"][1]["id"] = mutation["cases"][0]["id"]
    mutations.append(("reject duplicate case id", mutation, "duplicate case id"))
    mutation = copy.deepcopy(data)
    mutation["cases"][2]["required_behaviors"].append("made_up_behavior")
    mutations.append(("reject unknown behavior", mutation, "unknown required behavior"))
    mutation = copy.deepcopy(data)
    mutation["cases"][3]["evidence_spans"][0]["id"] = mutation["cases"][0]["evidence_spans"][0]["id"]
    mutations.append(("reject duplicate evidence id", mutation, "duplicate global evidence id"))
    mutation = copy.deepcopy(data)
    mutation["cases"].pop()
    mutations.append(("reject incomplete corpus", mutation, "expected 14"))
    for name, mutated, expected in mutations:
        errors = validate_corpus(mutated)
        if not errors or expected not in errors[0]:
            failures.append(f"{name}: expected error containing {expected!r}; got {errors!r}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="path to the frozen corpus JSON")
    parser.add_argument("--results", type=Path, help="optional normalized result bundle to validate")
    parser.add_argument("--self-test", action="store_true", help="run deterministic hostile mutations against the corpus validator")
    args = parser.parse_args()

    try:
        data = load_json(args.corpus)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(f"FAIL: cannot load strict JSON: {exc}", file=sys.stderr)
        return 2

    corpus_errors = validate_corpus(data)
    if corpus_errors:
        for error in corpus_errors:
            print(f"FAIL corpus: {error}", file=sys.stderr)
        return 1

    digest = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    print(f"PASS corpus structure: {len(data['cases'])} cases")
    print(f"corpus_sha256={digest}")
    if args.results:
        try:
            results_data = load_json(args.results)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            print(f"FAIL: cannot load strict results JSON: {exc}", file=sys.stderr)
            return 2
        result_errors = validate_results(results_data, data)
        if result_errors:
            for error in result_errors:
                print(f"FAIL results: {error}", file=sys.stderr)
            return 1
        print(f"PASS normalized result structure: {len(results_data['results'])} cases")

    if args.self_test:
        failures = self_test(data)
        if failures:
            for failure in failures:
                print(f"FAIL self-test: {failure}", file=sys.stderr)
            return 1
        print("PASS self-test: 5 validator mutation checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
