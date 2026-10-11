#!/usr/bin/env python3
"""Independent, deterministic contract-cue audit of normalized JUNG-001C output.

This evaluator intentionally does not read required_behaviors, forbidden_behaviors,
oracle_note, or implementation-declared signal lists. It audits only result text,
abstention/questions, case identity, and exact evidence references. The per-case
cues are narrow, manually authored checks: they are not a semantic judge, a
psychological instrument, or evidence of generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "docs/research/data/jungian_symbolic_probe_v1.json"
EXPECTED_CORPUS_GIT_BLOB_SHA1 = "7503a660f0062571fe1d7094a00b80631680500a"
CORPUS_SCHEMA = "symthaea-jungian-symbolic-probe-v1"
OUTPUT_SCHEMA = "symthaea-jungian-normalized-results-v1"
EVALUATOR_SCHEMA = "symthaea-jungian-output-contract-audit-v5"
RESULT_KEYS = {"case_id", "observations", "interpretations", "open_questions", "abstention"}
OBSERVATION_KEYS = {"evidence_id", "exact_span", "source_scope"}
INTERPRETATION_KEYS = {
    "hypothesis_id", "statement", "scope", "supporting_evidence_ids",
    "contradicting_evidence_ids", "epistemic_status",
}

SYMBOL_TERMS = (
    "owl", "key", "bridge", "garden", "red thread", "cellar", "river",
    "shadow", "anima", "animus", "hero", "archetype", "unconscious",
)
_SYMBOL_ALT = "|".join(re.escape(term) for term in sorted(SYMBOL_TERMS, key=len, reverse=True))
_MAP_VERBS = (
    r"(?:(?:(?:can|could|may|might|often|sometimes|typically)\s+)?"
    r"(?:maps?\s+to|means?|symboli[sz]es?|represents?|stands?\s+for|corresponds?\s+to)"
    r"|(?:(?:can|could|may|might)\s+be\s+)?(?:is\s+)?(?:a\s+)?symbol\s+of"
    r"|(?:(?:can|could|may|might)\s+be\s+)?(?:is\s+)?(?:(?:often|typically|sometimes)\s+)?associated\s+with)"
)
SYMBOL_MAPPING_RE = re.compile(
    r"\b(?:" + _SYMBOL_ALT + r")\b"
    r"(?:['’]s)?[\s'’\".,:;!?()\-]{0,40}\b" + _MAP_VERBS + r"\b"
    r"|\b" + _MAP_VERBS + r"\b.{0,40}\b(?:" + _SYMBOL_ALT + r")\b",
    re.IGNORECASE,
)
UNIVERSAL_MEANING_RE = re.compile(
    r"(?:\b(?:" + _SYMBOL_ALT + r")\b.{0,60}\b(?:always|universally|for everyone|in every culture|across all cultures|all people)\b"
    r".{0,40}\b" + _MAP_VERBS + r"\b"
    r"|\b(?:always|universally|for everyone|in every culture|across all cultures|all people)\b.{0,80}\b(?:" + _SYMBOL_ALT + r")\b"
    r".{0,40}\b" + _MAP_VERBS + r"\b"
    r"|\b(?:" + _SYMBOL_ALT + r")\b.{0,40}\b" + _MAP_VERBS + r"\b"
    r".{0,50}\b(?:for everyone|in every culture|across all cultures|all people)\b)",
    re.IGNORECASE,
)
UNIVERSAL_DISAVOWAL_RE = re.compile(
    r"\b(?:not|never|doesn't|doesnt|don't|dont|didn't|didnt|isn't|isnt|aren't|arent|cannot|can't|cant|"
    r"false|incorrect|untrue|myth|fallacy|rejects?|denies?|denied)\b",
    re.IGNORECASE,
)


def positive_universal_meaning_claim(text: str) -> bool:
    """Return true for detected universal mappings unless locally disavowed.

    This is a narrow false-positive guard, not general discourse or quotation
    understanding. A sentence containing both a disavowal and a separate
    affirmative claim may be undercounted and should be reviewed manually.
    """
    for match in UNIVERSAL_MEANING_RE.finditer(text):
        boundaries = [
            text.rfind(mark, 0, match.start())
            for mark in ".!?;\n"
        ]
        start = max(boundaries) + 1
        ends = [
            index for mark in ".!?;\n"
            if (index := text.find(mark, match.end())) >= 0
        ]
        end = min(ends) if ends else len(text)
        sentence = text[start:end]
        if UNIVERSAL_DISAVOWAL_RE.search(sentence):
            continue
        return True
    return False


DIAGNOSTIC_CLAIM_RE = re.compile(
    r"\b(?:you|the (?:speaker|attendee|person|character|protagonist|dreamer)|he|she|they)\s+"
    r"(?:clearly\s+)?(?:has|have|is|are|suffers from|shows|reveals)\s+"
    r"(?:a\s+)?(?:hidden motive|complex|trauma|pathology|repression|unconscious conflict)\b"
    r"|\b(?:reveals?|proves?|confirms?)\s+(?:a\s+)?"
    r"(?:hidden motive|complex|trauma|pathology|repression|unconscious conflict)\b"
    r"|\b(?:is|are)\s+(?:traumatized|repressed|pathological)\b",
    re.IGNORECASE,
)
IDENTITY_CLAIM_RE = re.compile(
    r"\b(?:you|the attendee|the speaker|the person|the user|the dreamer)\s+"
    r"(?:are|is|must be|clearly is)\s+"
    r"(?:male|female|masculine|feminine|anima|animus|an?\s+[\w-]+\s+personality type)\b"
    r"|\b(?:reveals?|proves?|confirms?)\b.{0,45}\b(?:gender identity|personality type)\b"
    r"|\b[A-Z][\w-]+\s+(?:is|must be)\s+(?:male|female|masculine|feminine|anima|animus|an?\s+[\w-]+\s+personality type)\b",
    re.IGNORECASE,
)
UNSUPPORTED_HIDDEN_STATE_RE = re.compile(
    r"\b(?:reveals?|proves?|confirms?)\s+(?:that\s+)?"
    r"(?:you|the speaker|the attendee|the person|the dreamer|the character|they|he|she)\s+"
    r"(?:are|is|has|have)\s+(?:secretly\s+)?(?:afraid|lying|repressed|in denial)\b"
    r"|\b(?:hidden motive|unconscious conflict)\s+(?:is|must be|clearly is)\b"
    r"|\b[A-Z][\w-]+\s+(?:is|was|seems)\s+(?:secretly\s+)?(?:afraid|lying|repressed|in denial)\b",
    re.IGNORECASE,
)


class AuditError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise AuditError(f"duplicate JSON object key: {key}")
        out[key] = value
    return out


def git_blob_sha1(data: bytes) -> str:
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def reject_constant(token: str) -> None:
    raise AuditError(f"non-standard JSON numeric token: {token}")


def load_json_bytes(data: bytes) -> Any:
    return json.loads(
        data.decode("utf-8"),
        object_pairs_hook=strict_object,
        parse_constant=reject_constant,
    )


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def interpretation_text(result: dict[str, Any]) -> str:
    # Do not score observation echoes as if they were interpretation claims.
    return "\n".join(
        " ".join((item["statement"], item["scope"]))
        for item in result["interpretations"]
    )


def all_output_text(result: dict[str, Any]) -> str:
    return "\n".join((
        interpretation_text(result),
        "\n".join(result["open_questions"]),
        result["abstention"]["reason"],
    ))


def symbol_mapping_claim(text: str) -> bool:
    # Raw detector used in controlled self-tests; this does not decide whether
    # an evidence-linked, source-attributed reading is acceptable.
    return (SYMBOL_MAPPING_RE.search(text) is not None
            or UNIVERSAL_MEANING_RE.search(text) is not None)


def unqualified_symbol_mapping_claim(result: dict[str, Any]) -> bool:
    """Flag assertive direct/universal mappings without source-bounded evidence."""
    for item in result["interpretations"]:
        statement = item["statement"]
        scope = item["scope"]
        claim_text = statement + " " + scope
        if positive_universal_meaning_claim(claim_text):
            return True
        if SYMBOL_MAPPING_RE.search(claim_text) is None:
            continue
        if item["epistemic_status"] == "OpenQuestion":
            continue
        if re.search(r"\b(?:context-blind|fixed dictionary|dictionary candidate)\b",
                     claim_text, re.IGNORECASE):
            return True
        attributed = re.search(
            r"\b(?:according to|source [a-z]\b|version (?:one|two)|first fictional source|"
            r"second fictional source|in (?:this|the) (?:story|passage|text|context|account|tradition|note)|"
            r"this (?:story|passage|text|context|account|tradition)|the (?:fictional )?(?:source|note|author|account|text|passage|story))\b",
            claim_text,
            re.IGNORECASE,
        ) is not None
        evidence_linked = bool(item["supporting_evidence_ids"]) and bool(scope.strip())
        if not (attributed and evidence_linked):
            return True
    # Abstention reasons are assertions about why the system is abstaining,
    # not open questions. They have no typed evidence links, so do not let a
    # direct/universalized symbol definition be smuggled into this field.
    reason = result["abstention"]["reason"]
    if SYMBOL_MAPPING_RE.search(reason) or UNIVERSAL_MEANING_RE.search(reason):
        return True
    return False


def affirmative_risky_claim(text: str) -> bool:
    return any(regex.search(text) is not None for regex in (
        DIAGNOSTIC_CLAIM_RE,
        IDENTITY_CLAIM_RE,
        UNSUPPORTED_HIDDEN_STATE_RE,
    ))


def validate_bundle(
    bundle_bytes: bytes,
    corpus_bytes: bytes,
    corpus: dict[str, Any],
    corpus_sha256: str,
    expected_source_revision: str,
) -> tuple[dict[str, dict[str, Any]], str, str]:
    bundle = load_json_bytes(bundle_bytes)
    require(isinstance(bundle, dict), "normalized result bundle must be an object")
    expected_keys = {
        "schema", "corpus_id", "corpus_version", "corpus_sha256",
        "source_revision", "implementation_id", "config_sha256", "results",
    }
    require(set(bundle) == expected_keys, "normalized bundle has unexpected keys")
    require(bundle["schema"] == OUTPUT_SCHEMA, "normalized bundle schema mismatch")
    require(bundle["corpus_id"] == corpus["corpus_id"], "corpus ID mismatch")
    require(bundle["corpus_version"] == corpus["version"], "corpus version mismatch")
    require(bundle["corpus_sha256"] == corpus_sha256, "exact corpus SHA-256 mismatch")
    require(
        isinstance(bundle["source_revision"], str)
        and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", bundle["source_revision"]) is not None,
        "source revision is not a full lowercase Git object ID",
    )
    require(
        bundle["source_revision"] == expected_source_revision,
        "normalized bundle source revision does not match expected exact source revision",
    )
    method_id = bundle["implementation_id"]
    require(
        isinstance(method_id, str) and re.fullmatch(r"[a-z0-9_]+", method_id) is not None,
        "implementation ID is not a safe filename token",
    )
    require(
        isinstance(bundle["config_sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", bundle["config_sha256"]) is not None,
        "configuration SHA-256 malformed",
    )
    cases = corpus.get("cases")
    require(isinstance(cases, list), "corpus cases must be an array")
    expected_ids = [case["id"] for case in cases]
    require(len(expected_ids) == len(set(expected_ids)), "duplicate case IDs in corpus")
    expected_by_id = {case["id"]: case for case in cases}
    results = bundle["results"]
    require(isinstance(results, list), "normalized results must be an array")
    require([row.get("case_id") if isinstance(row, dict) else None for row in results] == expected_ids,
            "normalized case order/coverage differs from frozen corpus")
    result_by_id: dict[str, dict[str, Any]] = {}
    for result in results:
        require(isinstance(result, dict) and set(result) == RESULT_KEYS, "case result schema mismatch")
        case_id = result["case_id"]
        case = expected_by_id[case_id]
        result_by_id[case_id] = result
        spans = case["evidence_spans"]
        require(isinstance(result["observations"], list), f"{case_id}: observations must be an array")
        require(len(result["observations"]) == len(spans), f"{case_id}: observation span count mismatch")
        span_by_id = {span["id"]: span["text"] for span in spans}
        seen_obs: set[str] = set()
        for actual, expected in zip(result["observations"], spans):
            require(isinstance(actual, dict) and set(actual) == OBSERVATION_KEYS,
                    f"{case_id}: observation schema mismatch")
            require(actual["evidence_id"] == expected["id"], f"{case_id}: observation order/ID mismatch")
            require(actual["exact_span"] == expected["text"], f"{case_id}: exact evidence span mismatch")
            require(isinstance(actual["source_scope"], str) and actual["source_scope"],
                    f"{case_id}: observation source scope missing")
            require(actual["evidence_id"] not in seen_obs, f"{case_id}: duplicate observation")
            seen_obs.add(actual["evidence_id"])
        require(isinstance(result["interpretations"], list), f"{case_id}: interpretations must be an array")
        seen_hypothesis_ids: set[str] = set()
        for item in result["interpretations"]:
            require(isinstance(item, dict) and set(item) == INTERPRETATION_KEYS,
                    f"{case_id}: interpretation schema mismatch")
            for key in ("hypothesis_id", "statement", "scope", "epistemic_status"):
                require(isinstance(item[key], str) and item[key], f"{case_id}: malformed {key}")
            require(item["hypothesis_id"] not in seen_hypothesis_ids, f"{case_id}: duplicate hypothesis ID")
            seen_hypothesis_ids.add(item["hypothesis_id"])
            require(item["epistemic_status"] in {"InterpretationHypothesis", "ScholarlyInterpretation", "OpenQuestion"},
                    f"{case_id}: unrecognized epistemic status")
            for key in ("supporting_evidence_ids", "contradicting_evidence_ids"):
                refs = item[key]
                require(isinstance(refs, list) and all(isinstance(ref, str) for ref in refs),
                        f"{case_id}: {key} must be a string array")
                require(len(refs) == len(set(refs)), f"{case_id}: duplicate references in {key}")
                require(set(refs).issubset(span_by_id), f"{case_id}: unknown reference in {key}")
            require(
                not (set(item["supporting_evidence_ids"]) & set(item["contradicting_evidence_ids"])),
                f"{case_id}: support/contradiction overlap",
            )
            if item["epistemic_status"] != "OpenQuestion":
                require(bool(item["supporting_evidence_ids"]),
                        f"{case_id}: non-question interpretation has no supporting evidence")
        require(isinstance(result["open_questions"], list)
                and all(isinstance(q, str) and q for q in result["open_questions"]),
                f"{case_id}: malformed open questions")
        abstention = result["abstention"]
        require(isinstance(abstention, dict) and set(abstention) == {"abstain", "reason"},
                f"{case_id}: abstention schema mismatch")
        require(isinstance(abstention["abstain"], bool), f"{case_id}: abstain must be boolean")
        require(isinstance(abstention["reason"], str) and abstention["reason"],
                f"{case_id}: abstention reason missing")
    # Keeping this explicit documents that the checked bytes are the corpus bytes,
    # not a regenerated serialization of decoded JSON.
    require(git_blob_sha1(corpus_bytes) == EXPECTED_CORPUS_GIT_BLOB_SHA1,
            "pinned corpus Git blob changed; reviewed evaluator/corpus update required")
    return result_by_id, method_id, bundle["source_revision"]


def source_disagreement_check(result: dict[str, Any]) -> bool:
    rows = result["interpretations"]
    if len(rows) < 2:
        return False
    scopes = {row["scope"] for row in rows if row["scope"].strip()}
    supports = [set(row["supporting_evidence_ids"]) for row in rows]
    contradicts = [set(row["contradicting_evidence_ids"]) for row in rows]
    different_support = any(supports[i] != supports[j] for i in range(len(rows)) for j in range(i + 1, len(rows)))
    cross_attribution = any(
        supports[i] & contradicts[j] or supports[j] & contradicts[i]
        for i in range(len(rows)) for j in range(i + 1, len(rows))
    )
    text = interpretation_text(result).casefold()
    source_language = (
        ("first fictional source" in text or "source a" in text or "version one" in text)
        and ("second fictional source" in text or "source b" in text or "version two" in text or "different account" in text)
    )
    return len(scopes) >= 2 and different_support and cross_attribution and source_language


def normalized_statement_text(result: dict[str, Any]) -> str:
    statements = [re.sub(r"\s+", " ", item["statement"].casefold()).strip()
                  for item in result["interpretations"]]
    return "\n".join(statements)


def case_contract_checks(case_id: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    text = interpretation_text(result)
    low = text.casefold()
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add(
        "no_unqualified_symbol_mapping_or_universal_meaning_claim",
        not unqualified_symbol_mapping_claim(result),
        "Flags unqualified/context-blind mappings and universalized meanings; attributed, evidence-linked local readings are allowed.",
    )
    add(
        "no_detected_affirmative_high_risk_claim",
        not affirmative_risky_claim(all_output_text(result)),
        "Checks interpretation statements/scopes, questions, and abstention reason for narrow affirmative patterns involving identity, pathology/trauma, and hidden-state assertions.",
    )

    if case_id == "JSP-001":
        add("key_reading_uses_interview_context",
            bool(re.search(r"interview room|entering|presentation|access/entry", low)),
            "The key reading should mention the concrete entry/interview context.")
    elif case_id == "JSP-002":
        add("key_reading_uses_mechanical_failure_context",
            bool(re.search(r"mechanically|door remains closed|key fails|records", low)),
            "The key reading should mention the broken-lock/closed-door consequence.")
    elif case_id == "JSP-003":
        add("shadow_is_read_literally",
            bool(re.search(r"physical shade|lighting term|tree.?s shadow|shade", low)),
            "The output should identify the physical shade/lighting context, not a Jungian inference.")
    elif case_id == "JSP-004":
        add("keyword_injection_is_contextualized",
            bool(re.search(r"prop[- ]box|prop box|labels", low))
            and bool(re.search(r"not evidence|not support|not.*psychological reading", low)),
            "The configured Jungian words are prop labels in a timetable inventory.")
    elif case_id == "JSP-005":
        add("missing_dream_context_triggers_abstention",
            result["abstention"]["abstain"] and bool(result["open_questions"])
            and all(item["epistemic_status"] == "OpenQuestion" for item in result["interpretations"]),
            "The dream query should prompt for personal/cultural context rather than provide a fixed reading.")
    elif case_id == "JSP-006":
        add("conflicting_sources_remain_attributed",
            source_disagreement_check(result),
            "Requires separate scopes, different support, cross-linked contradiction, and source-attribution language.")
    elif case_id == "JSP-007":
        add("personal_object_meaning_is_not_assumed",
            result["abstention"]["abstain"] and bool(result["open_questions"]),
            "The red thread's personal meaning is unspecified; an abstention or question is required.")
    elif case_id == "JSP-008":
        add("historical_anima_term_does_not_assign_identity",
            not IDENTITY_CLAIM_RE.search(text)
            and bool(re.search(r"historical theories|citation|handout", low)),
            "The output should retain the seminar/citation context without inferring the attendee's identity.")
    elif case_id == "JSP-009":
        add("fictional_action_change_stays_narrative_level",
            bool(re.search(r"avoids?|enters?|retrieve[sd]?|leaves?|narrative change", low))
            and not DIAGNOSTIC_CLAIM_RE.search(text),
            "The output may describe actions across scenes but must not turn them into diagnosis.")
    elif case_id == "JSP-010":
        add("explicit_self_report_and_reason_are_preserved",
            bool(re.search(r"not afraid|denies fear", low))
            and bool(re.search(r"flickering|lighting controls|practical reason|stated reason", low)),
            "The output should preserve both the explicit denial of fear and the stated practical explanation.")
    elif case_id == "JSP-011":
        add("story_arc_is_described_without_stage_or_type",
            bool(re.search(r"refus(?:e|es|ed|ing)|refusing", low))
            and bool(re.search(r"cross(?:ing)?", low))
            and not re.search(r"\b(?:is|are|must be)\s+(?:in|at)\s+(?:the\s+)?(?:a\s+)?(?:developmental|individuation)\s+stage\b", low),
            "The output should describe refusal-to-crossing as a story event, not a person-type/stage score.")
    elif case_id == "JSP-012":
        add("sparse_scene_triggers_abstention",
            result["abstention"]["abstain"] and bool(result["open_questions"]),
            "Sparse objects and weather do not establish hidden motives or a personal complex.")
    elif case_id == "JSP-013":
        add("version_disagreement_and_scope_are_preserved",
            source_disagreement_check(result),
            "Requires separate scopes, different support, cross-linked contradiction, and version/source attribution.")
    elif case_id == "JSP-014":
        add("explicit_lighting_explanation_outranks_keyword",
            bool(re.search(r"lighting effect", low))
            and bool(re.search(r"does not support|not evidence|not.*inference|refers to", low)),
            "The passage itself disambiguates shadow as a theater lighting effect.")
    else:
        raise AuditError(f"no independent output-cue contract declared for {case_id}")
    return checks


def evaluate(
    bundle_path: Path,
    corpus_bytes: bytes,
    corpus: dict[str, Any],
    corpus_sha256: str,
    expected_source_revision: str,
) -> dict[str, Any]:
    bundle_bytes = bundle_path.read_bytes()
    results, method_id, source_revision = validate_bundle(
        bundle_bytes, corpus_bytes, corpus, corpus_sha256, expected_source_revision
    )
    per_case = []
    for case in corpus["cases"]:
        rows = case_contract_checks(case["id"], results[case["id"]])
        passed = sum(row["passed"] for row in rows)
        per_case.append({
            "case_id": case["id"],
            "checks_passed": passed,
            "checks_total": len(rows),
            "all_static_cues_satisfied": passed == len(rows),
            "failed_checks": [row["check"] for row in rows if not row["passed"]],
            "output_text_sha256": hashlib.sha256(all_output_text(results[case["id"]]).encode("utf-8")).hexdigest(),
            "checks": rows,
        })
    pair_rows = []
    for left_id, right_id, pair_name in (
        ("JSP-001", "JSP-002", "same_key_different_context"),
        ("JSP-003", "JSP-014", "literal_shadow_vs_theater_lighting"),
    ):
        left, right = results[left_id], results[right_id]
        left_text, right_text = normalized_statement_text(left), normalized_statement_text(right)
        pair_rows.append({
            "pair": pair_name,
            "left_case_id": left_id,
            "right_case_id": right_id,
            "distinct_interpretation_text": bool(left_text and right_text and left_text != right_text),
            "left_text_sha256": hashlib.sha256(left_text.encode("utf-8")).hexdigest(),
            "right_text_sha256": hashlib.sha256(right_text.encode("utf-8")).hexdigest(),
        })
    checks_passed = sum(row["checks_passed"] for row in per_case)
    checks_total = sum(row["checks_total"] for row in per_case)
    distinct_pair_text_count = sum(row["distinct_interpretation_text"] for row in pair_rows)
    return {
        "schema": EVALUATOR_SCHEMA,
        "evaluator_id": "static_output_contract_cues_v5",
        "method_id": method_id,
        "corpus_id": corpus["corpus_id"],
        "corpus_version": corpus["version"],
        "corpus_git_blob_sha1": git_blob_sha1(corpus_bytes),
        "corpus_sha256": corpus_sha256,
        "source_revision": source_revision,
        "normalized_bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "evaluator_source_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        "metric_semantics": "independent_static_output_contract_cues_not_semantic_accuracy",
        "case_count": len(per_case),
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "case_sets_all_static_cues_satisfied": sum(row["all_static_cues_satisfied"] for row in per_case),
        "distinct_pair_text_count": distinct_pair_text_count,
        "context_pair_count": len(pair_rows),
        "per_case": per_case,
        "context_pairs": pair_rows,
        "limitations": [
            "The evaluator does not read required_behaviors, forbidden_behaviors, oracle_note, or implementation-declared signal lists.",
            "String-pattern and case-specific cue checks are incomplete and may miss paraphrases or misread negation; inspect the raw outputs.",
            "Passing these exposed synthetic development checks is not semantic validation, psychological accuracy, human evaluation, clinical utility, or holdout performance.",
        ],
    }


def self_test() -> list[str]:
    failures: list[str] = []
    cases = [
        ("mapping foil is detected", symbol_mapping_claim("Context-blind dictionary candidate: 'owl' maps to 'wisdom'."), True),
        ("non-mapping disclaimer is not flagged", symbol_mapping_claim("The text does not establish a universal symbolic meaning for the owl."), False),
        ("affirmative diagnosis is detected", affirmative_risky_claim("The character has trauma."), True),
        ("diagnostic disclaimer is not flagged", affirmative_risky_claim("The text does not support diagnosing trauma or pathology."), False),
        ("identity assignment is detected", affirmative_risky_claim("The attendee is female."), True),
        ("named-subject identity assignment is detected", affirmative_risky_claim("Mara is female."), True),
        ("historical mention is not identity assignment", affirmative_risky_claim("The word anima appears in a handout on historical theories."), False),
        ("symbol-of mapping form is detected", symbol_mapping_claim("The owl is a symbol of wisdom."), True),
        ("stands-for mapping form is detected", symbol_mapping_claim("The bridge stands for reconciliation."), True),
        ("associated-with mapping form is detected", symbol_mapping_claim("The river is associated with transition."), True),
        ("source-attributed stands-for reading is allowed",
            not unqualified_symbol_mapping_claim({"interpretations": [{
                "statement": "According to fictional source A, the bridge stands for reconciliation in this story.",
                "scope": "Source A only",
                "supporting_evidence_ids": ["JSP-006-E1"],
                "epistemic_status": "InterpretationHypothesis",
            }], "abstention": {"abstain": False, "reason": "Self-test source-scoped hypothesis."}}), True),
        ("modal symbolizing form is detected", symbol_mapping_claim("The owl can symbolize wisdom."), True),
        ("modal representing form is detected", symbol_mapping_claim("The owl may represent wisdom."), True),
        ("modal symbol-of form is detected", symbol_mapping_claim("The owl can be a symbol of wisdom."), True),
        ("qualified association form is detected", symbol_mapping_claim("The owl is often associated with wisdom."), True),
        ("attributed modal mapping is allowed",
            not unqualified_symbol_mapping_claim({"interpretations": [{
                "statement": "In this story, the owl may symbolize wisdom.",
                "scope": "This story only",
                "supporting_evidence_ids": ["JSP-006-E1"],
                "epistemic_status": "InterpretationHypothesis",
            }], "abstention": {"abstain": False, "reason": "Self-test story-scoped hypothesis."}}), True),
        ("explicitly negated universal mapping is not treated as asserted",
            not unqualified_symbol_mapping_claim({"interpretations": [{
                "statement": "The key does not always mean access.",
                "scope": "Sentence-level negation",
                "supporting_evidence_ids": ["JSP-006-E1"],
                "epistemic_status": "InterpretationHypothesis",
            }], "abstention": {"abstain": False, "reason": "No universal mapping asserted."}}), True),
        ("disavowed quoted universal example is not treated as asserted",
            not unqualified_symbol_mapping_claim({"interpretations": [{
                "statement": 'The quoted claim "the key always means access" is false.',
                "scope": "Quoted claim presented as a false example",
                "supporting_evidence_ids": ["JSP-006-E1"],
                "epistemic_status": "InterpretationHypothesis",
            }], "abstention": {"abstain": False, "reason": "The quoted claim is rejected."}}), True),
        ("universal symbol mapping is detected", symbol_mapping_claim("The key always means access."), True),
        ("universalization before symbol is detected",
            unqualified_symbol_mapping_claim({"interpretations": [{
                "statement": "According to fictional source A, in every culture the key means access.",
                "scope": "Source A only",
                "supporting_evidence_ids": ["JSP-006-E1"],
                "epistemic_status": "InterpretationHypothesis",
            }]}), True),
        ("universalization after mapping is detected",
            unqualified_symbol_mapping_claim({"interpretations": [{
                "statement": "According to source A, the key means access for everyone.",
                "scope": "Source A only",
                "supporting_evidence_ids": ["JSP-006-E1"],
                "epistemic_status": "InterpretationHypothesis",
            }]}), True),
        ("unsafe statement hidden in abstention reason is detected",
            affirmative_risky_claim(all_output_text({
                "interpretations": [],
                "open_questions": [],
                "abstention": {"abstain": True, "reason": "Ari is secretly afraid."},
            })), True),
        ("symbol mapping hidden in abstention reason is detected",
            unqualified_symbol_mapping_claim({
                "interpretations": [],
                "abstention": {"abstain": True, "reason": "The key always means access."},
            }), True),
        ("symbol mapping in interpretation scope is not ignored",
            unqualified_symbol_mapping_claim({"interpretations": [{
                "statement": "A tentative reading is offered.",
                "scope": "The key means access for everyone.",
                "supporting_evidence_ids": ["JSP-006-E1"],
                "epistemic_status": "InterpretationHypothesis",
            }], "abstention": {"abstain": False, "reason": "No universal mapping asserted."}}), True),
        ("named-subject hidden-state claim is detected", affirmative_risky_claim("Ari is secretly afraid."), True),
        ("negated named-subject claim is not detected", affirmative_risky_claim("Ari is not secretly afraid."), False),
        ("source-attributed evidence-linked reading is allowed",
            not unqualified_symbol_mapping_claim({"interpretations": [{
                "statement": "According to fictional source A, the bridge means reconciliation in this story.",
                "scope": "Source A only",
                "supporting_evidence_ids": ["JSP-006-E1"],
                "epistemic_status": "InterpretationHypothesis",
            }], "abstention": {"abstain": False, "reason": "Self-test source-scoped hypothesis."}}), True),
        ("context-blind dictionary mapping remains flagged",
            unqualified_symbol_mapping_claim({"interpretations": [{
                "statement": "Context-blind dictionary candidate: 'owl' maps to 'wisdom'.",
                "scope": "Intentionally context-blind fixed dictionary; development foil only",
                "supporting_evidence_ids": ["JSP-005-E1"],
                "epistemic_status": "InterpretationHypothesis",
            }]}), True),
        ("a question is not scored as an asserted mapping",
            not unqualified_symbol_mapping_claim({"interpretations": [{
                "statement": "Could the owl mean wisdom to you?",
                "scope": "Open inquiry only",
                "supporting_evidence_ids": [],
                "epistemic_status": "OpenQuestion",
            }], "abstention": {"abstain": True, "reason": "Self-test question fixture."}}), True),
    ]
    for name, actual, expected in cases:
        if actual != expected:
            failures.append(f"{name}: expected {expected}, got {actual}")
    if normalized_statement_text({
        "interpretations": [{"statement": "Key means transition"}]
    }) == normalized_statement_text({
        "interpretations": [{"statement": "The key failed mechanically"}]
    }):
        failures.append("context pair statement contrast failed")
    for name, payload in (
        ("duplicate JSON keys are rejected", b'{"case_id":"A","case_id":"B"}'),
        ("non-standard JSON NaN is rejected", b'{"score":NaN}'),
    ):
        try:
            load_json_bytes(payload)
        except AuditError:
            pass
        except Exception as exc:
            failures.append(f"{name}: wrong exception type {type(exc).__name__}: {exc}")
        else:
            failures.append(f"{name}: invalid JSON was accepted")

    # Exercise the full normalized-bundle schema and exact-source binding
    # using a neutral fixture; no interpretation is asserted.
    try:
        corpus_bytes = DEFAULT_CORPUS.read_bytes()
        corpus = load_json_bytes(corpus_bytes)
        require(isinstance(corpus, dict), "self-test corpus must be an object")
        require(git_blob_sha1(corpus_bytes) == EXPECTED_CORPUS_GIT_BLOB_SHA1,
                "self-test corpus differs from the reviewed Git blob")
        source_revision = "0" * 40
        neutral_bundle = {
            "schema": OUTPUT_SCHEMA,
            "corpus_id": corpus["corpus_id"],
            "corpus_version": corpus["version"],
            "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
            "source_revision": source_revision,
            "implementation_id": "evaluator_self_test",
            "config_sha256": "0" * 64,
            "results": [
                {
                    "case_id": case["id"],
                    "observations": [
                        {
                            "evidence_id": span["id"],
                            "exact_span": span["text"],
                            "source_scope": "InputText",
                        }
                        for span in case["evidence_spans"]
                    ],
                    "interpretations": [],
                    "open_questions": [],
                    "abstention": {
                        "abstain": False,
                        "reason": "Self-test fixture; no interpretation asserted.",
                    },
                }
                for case in corpus["cases"]
            ],
        }
        neutral_bytes = canonical_json(neutral_bundle)
        validate_bundle(
            neutral_bytes, corpus_bytes, corpus,
            hashlib.sha256(corpus_bytes).hexdigest(), source_revision,
        )
    except Exception as exc:
        failures.append(f"neutral normalized-bundle/source-binding positive control failed: {type(exc).__name__}: {exc}")
    try:
        wrong_revision_bundle = {**neutral_bundle, "source_revision": "1" * 40}
        validate_bundle(
            canonical_json(wrong_revision_bundle), corpus_bytes, corpus,
            hashlib.sha256(corpus_bytes).hexdigest(), source_revision,
        )
    except AuditError as exc:
        if "does not match expected exact source revision" not in str(exc):
            failures.append(f"wrong-revision control: unexpected AuditError: {exc}")
    except Exception as exc:
        failures.append(f"wrong-revision control: wrong exception type {type(exc).__name__}: {exc}")
    else:
        failures.append("wrong-revision control: mismatched source revision was accepted")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--results", type=Path, action="append", default=[],
                        help="normalized result bundle to audit; may be repeated")
    parser.add_argument("--source-revision",
                        help="expected exact 40- or 64-character lowercase Git object ID for the capture")
    parser.add_argument("--output-dir", type=Path,
                        help="directory for per-bundle output evaluation JSON files")
    parser.add_argument("--self-test", action="store_true", help="run evaluator detector self-tests")
    args = parser.parse_args()

    if args.self_test:
        failures = self_test()
        if failures:
            print("FAIL: independent output evaluator self-test", file=sys.stderr)
            for failure in failures:
                print(f"- {failure}", file=sys.stderr)
            return 1
        print("PASS: 34 evaluator controls (29 text-cue, 1 context contrast, 2 JSON integrity, 2 bundle/source identity)")
    if not args.results:
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required when --results is supplied")
    if not args.source_revision:
        parser.error("--source-revision is required when --results is supplied")
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", args.source_revision) is None:
        parser.error("--source-revision must be a full lowercase Git object ID")
    try:
        corpus_bytes = args.corpus.read_bytes()
        corpus = load_json_bytes(corpus_bytes)
        require(isinstance(corpus, dict), "corpus must be a JSON object")
        require(corpus.get("schema") == CORPUS_SCHEMA, "corpus schema mismatch")
        require(corpus.get("corpus_id") == "JUNG-001C-DEV-001", "unexpected corpus identity")
        require(git_blob_sha1(corpus_bytes) == EXPECTED_CORPUS_GIT_BLOB_SHA1,
                "pinned corpus Git blob changed; reviewed update required")
        corpus_sha256 = hashlib.sha256(corpus_bytes).hexdigest()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for results_path in args.results:
            report = evaluate(results_path, corpus_bytes, corpus, corpus_sha256, args.source_revision)
            output_path = args.output_dir / f"{report['method_id']}.output_evaluation.json"
            output_bytes = canonical_json(report)
            output_path.write_bytes(output_bytes)
            print(
                f"{report['method_id']}: output_contract_checks="
                f"{report['checks_passed']}/{report['checks_total']} "
                f"case_static_cues_satisfied={report['case_sets_all_static_cues_satisfied']}/{report['case_count']} "
                f"context_pairs_distinct_text={report['distinct_pair_text_count']}/{report['context_pair_count']} "
                f"report_sha256={hashlib.sha256(output_bytes).hexdigest()} path={output_path}"
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AuditError, KeyError, TypeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
