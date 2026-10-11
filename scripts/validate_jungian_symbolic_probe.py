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
EXPECTED_CORPUS_GIT_BLOB_SHA1 = "7503a660f0062571fe1d7094a00b80631680500a"
HEX_256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT_HEX = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
CORPUS_KEYS = {
    "schema", "corpus_id", "version", "status", "split", "source_policy", "purpose",
    "claim_ceiling", "output_contract", "behavior_vocabulary", "forbidden_vocabulary", "cases",
}
OUTPUT_CONTRACT_KEYS = {
    "required_top_level_fields", "observation_fields", "interpretation_fields",
    "epistemic_status_values", "abstention_fields", "constraints",
}

CASE_KEYS = {"id", "task", "input_text", "evidence_spans", "required_behaviors", "forbidden_behaviors", "oracle_note"}
SPAN_KEYS = {"id", "text"}
RESULT_TOP_KEYS = {"schema", "corpus_id", "corpus_version", "corpus_sha256", "source_revision", "implementation_id", "config_sha256", "results"}
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
        exact_keys(data, CORPUS_KEYS, "corpus")
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
        contract = exact_keys(data.get("output_contract"), OUTPUT_CONTRACT_KEYS, "output_contract")
        require(contract["required_top_level_fields"] == ["case_id", "observations", "interpretations", "open_questions", "abstention"], "output_contract: top-level fields changed")
        require(contract["observation_fields"] == ["evidence_id", "exact_span", "source_scope"], "output_contract: observation fields changed")
        require(contract["interpretation_fields"] == ["hypothesis_id", "statement", "scope", "supporting_evidence_ids", "contradicting_evidence_ids", "epistemic_status"], "output_contract: interpretation fields changed")
        require(contract["epistemic_status_values"] == ["InterpretationHypothesis", "ScholarlyInterpretation", "OpenQuestion"], "output_contract: epistemic status vocabulary changed")
        require(contract["abstention_fields"] == ["abstain", "reason"], "output_contract: abstention fields changed")
        unique_strings(contract["constraints"], "output_contract.constraints")
        require(bool(contract["constraints"]), "output_contract.constraints: must be nonempty")
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


def validate_results(data: Any, corpus: dict[str, Any], corpus_bytes: bytes) -> list[str]:
    errors: list[str] = []
    try:
        top = exact_keys(data, RESULT_TOP_KEYS, "results")
        require(top.get("schema") == RESULT_SCHEMA, "results: unexpected schema")
        require(top.get("corpus_id") == corpus["corpus_id"], "results: corpus_id mismatch")
        require(top.get("corpus_version") == corpus["version"], "results: corpus_version mismatch")
        require(top.get("corpus_sha256") == hashlib.sha256(corpus_bytes).hexdigest(), "results: corpus SHA-256 mismatch")
        require(isinstance(top.get("source_revision"), str) and COMMIT_HEX.fullmatch(top["source_revision"]) is not None, "results: source_revision must be a full lowercase Git object ID")
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
                require(set(supports + contradicts) <= observed_ids, f"{hw}: cited evidence must also appear in observations")
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


def git_blob_sha1(data: bytes) -> str:
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def load_json_bytes(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"), object_pairs_hook=strict_object)


def load_json(path: Path) -> Any:
    return load_json_bytes(path.read_bytes())


def valid_results_stub(corpus: dict[str, Any], corpus_bytes: bytes) -> dict[str, Any]:
    """Build a neutral schema-only fixture; it makes no interpretive claims."""
    return {
        "schema": RESULT_SCHEMA,
        "corpus_id": corpus["corpus_id"],
        "corpus_version": corpus["version"],
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "source_revision": "0" * 40,
        "implementation_id": "validator-self-test-only",
        "config_sha256": "0" * 64,
        "results": [
            {
                "case_id": case["id"],
                "observations": [
                    {"evidence_id": span["id"], "exact_span": span["text"], "source_scope": "InputText"}
                    for span in case["evidence_spans"]
                ],
                "interpretations": [],
                "open_questions": [],
                "abstention": {"abstain": False, "reason": "Validator self-test fixture; no interpretation asserted."},
            }
            for case in corpus["cases"]
        ],
    }


def self_test(data: dict[str, Any], corpus_bytes: bytes) -> list[str]:
    failures: list[str] = []
    if validate_corpus(data):
        failures.append("canonical corpus unexpectedly fails")

    corpus_mutations: list[tuple[str, Any, str]] = []
    mutation = copy.deepcopy(data)
    mutation["cases"][0]["evidence_spans"][0]["text"] = "not an exact source span"
    corpus_mutations.append(("reject altered evidence", mutation, "exact substring"))
    mutation = copy.deepcopy(data)
    mutation["cases"][1]["id"] = mutation["cases"][0]["id"]
    corpus_mutations.append(("reject duplicate case id", mutation, "duplicate case id"))
    mutation = copy.deepcopy(data)
    mutation["cases"][2]["required_behaviors"].append("made_up_behavior")
    corpus_mutations.append(("reject unknown behavior", mutation, "unknown required behavior"))
    mutation = copy.deepcopy(data)
    mutation["cases"][3]["evidence_spans"][0]["id"] = mutation["cases"][0]["evidence_spans"][0]["id"]
    corpus_mutations.append(("reject duplicate evidence id", mutation, "duplicate global evidence id"))
    mutation = copy.deepcopy(data)
    mutation["cases"].pop()
    corpus_mutations.append(("reject incomplete corpus", mutation, "expected 14"))
    for name, mutated, expected in corpus_mutations:
        errors = validate_corpus(mutated)
        if not errors or expected not in errors[0]:
            failures.append(f"{name}: expected error containing {expected!r}; got {errors!r}")

    try:
        json.loads('{"duplicate": 1, "duplicate": 2}', object_pairs_hook=strict_object)
        failures.append("duplicate JSON object keys were accepted")
    except ValidationError:
        pass

    good_results = valid_results_stub(data, corpus_bytes)
    if validate_results(good_results, data, corpus_bytes):
        failures.append("neutral normalized-results positive control was rejected")
    results_mutations: list[tuple[str, Any, str]] = []
    mutation = copy.deepcopy(good_results)
    mutation["results"].pop()
    results_mutations.append(("reject missing result case", mutation, "missing cases"))
    mutation = copy.deepcopy(good_results)
    mutation["results"][0]["observations"][0]["exact_span"] = "tampered evidence"
    results_mutations.append(("reject altered result evidence", mutation, "does not match frozen source text"))
    mutation = copy.deepcopy(good_results)
    mutation["results"][0]["interpretations"].append({
        "hypothesis_id": "H-UNBOUND",
        "statement": "Synthetic test-only unsupported claim.",
        "scope": "synthetic-validator-test",
        "supporting_evidence_ids": ["NOT-IN-CORPUS"],
        "contradicting_evidence_ids": [],
        "epistemic_status": "InterpretationHypothesis",
    })
    results_mutations.append(("reject unbound hypothesis evidence", mutation, "reference to evidence outside the frozen case"))
    mutation = copy.deepcopy(good_results)
    mutation["corpus_sha256"] = "f" * 64
    results_mutations.append(("reject mismatched corpus digest", mutation, "corpus SHA-256 mismatch"))
    mutation = copy.deepcopy(good_results)
    mutation["source_revision"] = "not-a-git-object-id"
    results_mutations.append(("reject weak source identity", mutation, "full lowercase Git object ID"))
    mutation = copy.deepcopy(good_results)
    mutation["results"].append(copy.deepcopy(mutation["results"][0]))
    results_mutations.append(("reject duplicate result case", mutation, "duplicate case_id"))
    for name, mutated, expected in results_mutations:
        errors = validate_results(mutated, data, corpus_bytes)
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
        corpus_bytes = args.corpus.read_bytes()
        data = load_json_bytes(corpus_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        print(f"FAIL: cannot load strict JSON: {exc}", file=sys.stderr)
        return 2

    actual_git_blob = git_blob_sha1(corpus_bytes)
    if actual_git_blob != EXPECTED_CORPUS_GIT_BLOB_SHA1:
        print(
            "FAIL corpus identity: Git blob SHA-1 mismatch "
            f"(got={actual_git_blob}, expected={EXPECTED_CORPUS_GIT_BLOB_SHA1}); "
            "bump the corpus version and review the changed cases before updating this pin.",
            file=sys.stderr,
        )
        return 1

    corpus_errors = validate_corpus(data)
    if corpus_errors:
        for error in corpus_errors:
            print(f"FAIL corpus: {error}", file=sys.stderr)
        return 1

    digest = hashlib.sha256(corpus_bytes).hexdigest()
    print(f"PASS corpus structure and pinned Git blob: {len(data['cases'])} cases")
    print(f"corpus_sha256={digest}")
    print(f"corpus_git_blob_sha1={actual_git_blob}")
    if args.results:
        try:
            results_data = load_json(args.results)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
            print(f"FAIL: cannot load strict results JSON: {exc}", file=sys.stderr)
            return 2
        result_errors = validate_results(results_data, data, corpus_bytes)
        if result_errors:
            for error in result_errors:
                print(f"FAIL results: {error}", file=sys.stderr)
            return 1
        print(f"PASS normalized result structure: {len(results_data['results'])} cases")

    if args.self_test:
        failures = self_test(data, corpus_bytes)
        if failures:
            for failure in failures:
                print(f"FAIL self-test: {failure}", file=sys.stderr)
            return 1
        print("PASS self-test: 12 hostile mutations plus normalized-results positive control; pinned corpus Git blob")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
