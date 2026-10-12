#!/usr/bin/env python3
"""Run transparent lexical-foil and context-rule baselines on JUNG-001C.

The lexical foil is intentionally context-blind. The context baseline is a
small deterministic rule set, not an HDC implementation or psychological
model. Their metrics indicate agreement with an authored development rubric,
not psychological truth or clinical validity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "docs/research/data/jungian_symbolic_probe_v1.json"
CORPUS_SCHEMA = "symthaea-jungian-symbolic-probe-v1"
EXPECTED_CORPUS_GIT_BLOB_SHA1 = "7503a660f0062571fe1d7094a00b80631680500a"
OUTPUT_SCHEMA = "symthaea-jungian-normalized-results-v1"
REPORT_SCHEMA = "symthaea-jungian-baseline-comparison-v1"
RESULT_KEYS = {"case_id", "observations", "interpretations", "open_questions", "abstention"}
OBSERVATION_KEYS = {"evidence_id", "exact_span", "source_scope"}
INTERPRETATION_KEYS = {"hypothesis_id", "statement", "scope", "supporting_evidence_ids", "contradicting_evidence_ids", "epistemic_status"}

# This dictionary intentionally assigns a single decontextualized association
# to keywords. It exists as a failure-prone foil, never as a production model.
LEXICAL_DICTIONARY = {
    "archetype": "archetypal pattern",
    "shadow": "unacknowledged aspect of the psyche",
    "anima": "Jungian anima motif",
    "animus": "Jungian animus motif",
    "unconscious": "unconscious-process motif",
    "hero": "hero-archetype motif",
    "owl": "wisdom",
    "key": "access or transition",
    "bridge": "transition or reconciliation",
    "garden": "growth or refuge",
    "red thread": "connection or fate",
    "cellar": "unconscious depth",
    "river": "transition",
}
TOKEN_RE = re.compile(r"(?<!\w)(" + "|".join(re.escape(x) for x in sorted(LEXICAL_DICTIONARY, key=len, reverse=True)) + r")(?!\w)", re.IGNORECASE)
JUNG_TERMS = {"archetype", "shadow", "anima", "animus", "unconscious", "hero"}


class BaselineError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise BaselineError(f"duplicate JSON object key: {key}")
        out[key] = value
    return out


def git_blob_sha1(data: bytes) -> str:
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def load_json_bytes(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"), object_pairs_hook=strict_object)


def load_json(path: Path) -> Any:
    return load_json_bytes(path.read_bytes())


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def unique(xs: list[str]) -> list[str]:
    return list(dict.fromkeys(xs))


def observation_rows(case: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "evidence_id": span["id"],
            "exact_span": span["text"],
            "source_scope": "Synthetic case input; exact frozen span",
        }
        for span in case["evidence_spans"]
    ]


def find_span_ids(case: dict[str, Any], needle: str) -> list[str]:
    low_needle = needle.casefold()
    return [
        span["id"] for span in case["evidence_spans"]
        if low_needle in span["text"].casefold()
    ]


def find_relevant_span_ids(case: dict[str, Any], terms: list[str]) -> list[str]:
    selected = []
    for span in case["evidence_spans"]:
        low = span["text"].casefold()
        if any(term.casefold() in low for term in terms):
            selected.append(span["id"])
    return selected


def make_hypothesis(
    hypothesis_id: str,
    statement: str,
    scope: str,
    support: list[str],
    contradict: list[str] | None = None,
    status: str = "InterpretationHypothesis",
) -> dict[str, Any]:
    return {
        "hypothesis_id": hypothesis_id,
        "statement": statement,
        "scope": scope,
        "supporting_evidence_ids": unique(support),
        "contradicting_evidence_ids": unique(contradict or []),
        "epistemic_status": status,
    }


def make_result(
    case: dict[str, Any],
    interpretations: list[dict[str, Any]],
    questions: list[str],
    abstain: bool,
    reason: str,
) -> dict[str, Any]:
    result = {
        "case_id": case["id"],
        "observations": observation_rows(case),
        "interpretations": interpretations,
        "open_questions": unique(questions),
        "abstention": {"abstain": abstain, "reason": reason},
    }
    require(set(result) == RESULT_KEYS, f"{case['id']}: internal result schema drift")
    for item in result["interpretations"]:
        require(set(item) == INTERPRETATION_KEYS, f"{case['id']}: internal interpretation schema drift")
    for item in result["observations"]:
        require(set(item) == OBSERVATION_KEYS, f"{case['id']}: internal observation schema drift")
    return result


def lexical_foil(case: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Emit fixed dictionary candidates regardless of local context."""
    text = case["input_text"]
    interpretations: list[dict[str, Any]] = []
    questions: list[str] = []
    signals = ["cite_input_evidence"]
    matches = list(TOKEN_RE.finditer(text))
    if matches:
        signals.append("treat_keyword_overlap_as_evidence")
        signals.append("claim_universal_symbol_meaning")
    terms_seen: set[str] = set()
    for index, match in enumerate(matches, start=1):
        term = match.group(0).casefold()
        if term in terms_seen:
            continue
        terms_seen.add(term)
        mapped = LEXICAL_DICTIONARY[term]
        direct_spans = find_span_ids(case, term)
        if direct_spans:
            support = direct_spans
            status = "InterpretationHypothesis"
            statement = f"Context-blind dictionary candidate: '{term}' maps to '{mapped}'."
        else:
            # Do not invent an evidence span if the corpus's selected spans
            # do not contain the token. Preserve the lexical hit as a question.
            support = []
            status = "OpenQuestion"
            statement = f"The input contains the token '{term}', but no frozen evidence span isolates it; what role does it play here?"
            questions.append(f"Where and how is '{term}' used in this passage?")
        interpretations.append(
            make_hypothesis(
                f"KW-{index:02d}",
                statement,
                "Intentionally context-blind fixed dictionary; development foil only",
                support,
                status=status,
            )
        )
    if any(term in terms_seen for term in {"anima", "animus"}):
        signals.append("infer_gender_identity")
    if not matches:
        questions.append("The lexical dictionary found no configured symbol token; context remains unclassified.")
    abstain = not bool(matches)
    reason = (
        "This foil intentionally maps configured words to fixed associations without context; outputs must not be used in user-facing inference."
        if matches
        else "No configured lexical trigger was found; no interpretation is emitted."
    )
    return make_result(case, interpretations, questions, abstain, reason), unique(signals)


def context_rule_baseline(case: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Conservative hand-authored rules using only the input and exact spans."""
    text = case["input_text"]
    low = text.casefold()
    interpretations: list[dict[str, Any]] = []
    questions: list[str] = []
    signals = ["cite_input_evidence", "distinguish_observation_from_hypothesis"]
    abstain = False
    reason = "A narrow deterministic rule matched local narrative evidence; this is not an empirical psychological inference."

    def add(statement: str, terms: list[str], hid: str = "CTX-01",
            contradict_terms: list[str] | None = None, scope: str = "This synthetic text only") -> None:
        support = find_relevant_span_ids(case, terms)
        contradiction = find_relevant_span_ids(case, contradict_terms or [])
        interpretations.append(make_hypothesis(hid, statement, scope, support, contradiction))

    # Explicitly literal disambiguation outranks the keyword foil.
    if "tree's shadow" in low or "lighting effect" in low:
        add(
            "The local text uses 'shadow' as a physical shade or lighting term; the passage does not support a Jungian inference from that word alone.",
            ["shadow", "tree", "lighting effect"],
        )
        signals.extend(["treat_literal_usage_literally", "reject_keyword_only_support", "return_context_bound_reading"])
        if "lighting effect" in low:
            add(
                "The text explicitly explains that the phrase refers to a lighting effect in a theater review.",
                ["lighting effect", "theater review"],
                hid="CTX-02",
            )
    elif "prop-box names" in low or "prop box names" in low:
        add(
            "The Jungian terms are explicitly described as prop-box labels, so their occurrence is not evidence for a psychological reading.",
            ["prop-box names", "simple timetable announcement"],
        )
        signals.extend(["reject_keyword_only_support", "treat_literal_usage_literally", "return_context_bound_reading"])
    # Conflicting source notes must remain separate, with source-specific support.
    elif (("source a" in low and "source b" in low) or ("version one" in low and "version two" in low)):
        spans = case["evidence_spans"]
        if len(spans) >= 2:
            left_id, right_id = spans[0]["id"], spans[1]["id"]
            interpretations.append(make_hypothesis(
                "CTX-A",
                "The first fictional source associates the symbol with the meaning stated in its own note; retain that attribution rather than generalizing it.",
                "First fictional source only",
                [left_id], [right_id],
            ))
            interpretations.append(make_hypothesis(
                "CTX-B",
                "The second fictional source gives a different account; preserve this as disagreement rather than averaging the meanings.",
                "Second fictional source only",
                [right_id], [left_id],
            ))
        questions.append("What are the provenance and context of each source? The notes do not establish a universal meaning.")
        signals.extend(["preserve_source_disagreement", "return_competing_readings", "keep_cultural_scope_explicit", "return_context_bound_reading"])
        reason = "The available notes conflict; both are retained with source-local scope."
    elif "dreamed about an owl" in low or ("dream" in low and "what does it mean" in low):
        questions.extend([
            "What associations do you personally have with the owl, if any?",
            "Is there a particular cultural, religious, literary, or personal context you want considered?",
        ])
        interpretations.append(make_hypothesis(
            "CTX-Q1",
            "A personal-association reading could begin with the dreamer's own associations, but this text does not identify them.",
            "Possible inquiry frame, not a dream interpretation",
            [],
            status="OpenQuestion",
        ))
        interpretations.append(make_hypothesis(
            "CTX-Q2",
            "A tradition-specific reading would require the relevant source or cultural context; no single reading can be selected from the prompt alone.",
            "Possible inquiry frame, not a universal symbol dictionary",
            [],
            status="OpenQuestion",
        ))
        signals.extend(["ask_clarifying_question_or_abstain", "return_competing_readings", "abstain_from_personal_inference"])
        abstain = True
        reason = "Personal and cultural context is missing; the baseline asks rather than assigning a dream meaning."
    elif "grandmother gave me a red thread" in low:
        questions.extend([
            "What did the thread mean to you or your grandmother, if anything?",
            "Would you like a personal, family-history, literary, or specific cultural reading?",
        ])
        interpretations.append(make_hypothesis(
            "CTX-Q1",
            "The passage establishes a gift and a time of moving away, but not the thread's personal meaning.",
            "Textual observation; personal meaning undetermined",
            find_relevant_span_ids(case, ["grandmother", "moved away", "red thread"]),
        ))
        signals.extend(["ask_clarifying_question_or_abstain", "abstain_from_personal_inference", "distinguish_observation_from_hypothesis"])
        abstain = True
        reason = "The personal association is not specified; no universal meaning is imposed."
    elif "i am not afraid" in low and "because the recording keeps flickering" in low:
        add(
            "Ari explicitly denies fear and gives a practical reason for checking the room; the text supplies no evidence to override that self-report.",
            ["I am not afraid", "recording keeps flickering"],
        )
        signals.extend(["preserve_self_report", "abstain_from_personal_inference", "return_context_bound_reading"])
    elif "anima" in low and "handout" in low:
        add(
            "The word appears in a handout comparing historical theories; the attendee's stated concern is citation quality, not a personal identity disclosure.",
            ["anima", "historical theories", "clearer citations"],
        )
        signals.extend(["abstain_from_personal_inference", "keep_cultural_scope_explicit", "return_context_bound_reading"])
    elif "turned the key" in low and "interview room" in low:
        add(
            "Turning the key is directly linked to entering the interview room; the text supports a concrete access/entry event but does not determine a universal symbolic meaning.",
            ["turned the key", "entered the interview room", "practiced every evening"],
        )
        signals.extend(["return_context_bound_reading", "distinguish_observation_from_hypothesis"])
    elif "key bent in the lock" in low and "door stayed shut" in low:
        add(
            "The key fails mechanically and the door remains closed; that local causal role is sufficient without a fixed symbolic meaning.",
            ["key bent in the lock", "door stayed shut", "could not retrieve the records"],
        )
        signals.extend(["return_context_bound_reading", "treat_literal_usage_literally"])
    elif "avoids the cellar in three scenes" in low:
        add(
            "The fictional character avoids the cellar earlier and later enters to retrieve a notebook; this supports describing a narrative change, not diagnosing a complex or trauma.",
            ["avoids the cellar", "enters it to retrieve a lost notebook", "leaves when the task is finished"],
        )
        signals.extend(["describe_narrative_change_only", "abstain_from_personal_inference", "return_context_bound_reading"])
    elif "at first" in low and "later" in low and "cross the river" in low:
        add(
            "The protagonist changes from refusing to crossing after checking conditions and making a safety marker; report that narrative sequence without assigning a developmental stage.",
            ["At first", "Later", "checking the current"],
        )
        signals.extend(["describe_narrative_change_only", "return_context_bound_reading", "abstain_from_personal_inference"])
    elif "chair stands beside a window" in low and "room is otherwise empty" in low:
        questions.append("What context, if any, should guide a reading of this scene?")
        signals.extend(["ask_clarifying_question_or_abstain", "abstain_from_personal_inference", "reject_keyword_only_support"])
        abstain = True
        reason = "The sparse scene does not support an interpretation of hidden motives, pathology, or a personal complex."
    else:
        # Generic, fail-closed fallback; state what the passage literally says.
        signals.extend(["ask_clarifying_question_or_abstain", "abstain_from_personal_inference"])
        questions.append("What context or interpretive tradition, if any, should guide the reading?")
        abstain = True
        reason = "No narrow context rule applied; the baseline abstains rather than inventing a symbolic meaning."

    return make_result(case, interpretations, questions, abstain, reason), unique(signals)


def validate_normalized_bundle(bundle: Any, corpus: dict[str, Any], corpus_digest: str) -> None:
    require(isinstance(bundle, dict), "normalized result bundle must be an object")
    required = {"schema", "corpus_id", "corpus_version", "corpus_sha256", "source_revision", "implementation_id", "config_sha256", "results"}
    require(set(bundle) == required, "normalized result bundle has unexpected keys")
    require(bundle["schema"] == OUTPUT_SCHEMA, "normalized result schema mismatch")
    require(bundle["corpus_id"] == corpus["corpus_id"] and bundle["corpus_version"] == corpus["version"], "normalized result corpus identity mismatch")
    require(bundle["corpus_sha256"] == corpus_digest, "normalized result corpus SHA-256 mismatch")
    require(isinstance(bundle["source_revision"], str) and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", bundle["source_revision"]), "source_revision must be a full lowercase Git object ID")
    require(bool(bundle["implementation_id"]), "implementation_id missing")
    require(re.fullmatch(r"[0-9a-f]{64}", bundle["config_sha256"]) is not None, "config_sha256 must be 64 lowercase hex characters")
    expected_cases = {case["id"]: case for case in corpus["cases"]}
    require(isinstance(bundle["results"], list) and len(bundle["results"]) == len(expected_cases), "normalized results case count mismatch")
    seen: set[str] = set()
    for result in bundle["results"]:
        require(set(result) == RESULT_KEYS, "normalized case has unexpected keys")
        cid = result["case_id"]
        require(cid in expected_cases and cid not in seen, f"unknown/duplicate normalized case ID: {cid}")
        seen.add(cid)
        span_map = {span["id"]: span["text"] for span in expected_cases[cid]["evidence_spans"]}
        seen_evidence: set[str] = set()
        require(isinstance(result["observations"], list), "observations must be an array")
        for obs in result["observations"]:
            require(set(obs) == OBSERVATION_KEYS, "observation schema mismatch")
            eid = obs["evidence_id"]
            require(eid in span_map and eid not in seen_evidence, f"unknown/duplicate evidence id in {cid}: {eid}")
            seen_evidence.add(eid)
            require(obs["exact_span"] == span_map[eid], f"tampered evidence span in {cid}: {eid}")
            require(bool(obs["source_scope"]), f"missing source scope for {cid}:{eid}")
        require(isinstance(result["interpretations"], list), "interpretations must be an array")
        hypothesis_ids: set[str] = set()
        for hypothesis in result["interpretations"]:
            require(set(hypothesis) == INTERPRETATION_KEYS, "interpretation schema mismatch")
            hid = hypothesis["hypothesis_id"]
            require(bool(hid) and hid not in hypothesis_ids, f"missing/duplicate hypothesis ID in {cid}")
            hypothesis_ids.add(hid)
            require(hypothesis["epistemic_status"] in {"InterpretationHypothesis", "ScholarlyInterpretation", "OpenQuestion"}, "unknown epistemic status")
            supports = hypothesis["supporting_evidence_ids"]
            contradicts = hypothesis["contradicting_evidence_ids"]
            require(isinstance(supports, list) and isinstance(contradicts, list), "evidence references must be arrays")
            require(len(supports) == len(set(supports)) and len(contradicts) == len(set(contradicts)), "duplicate hypothesis evidence reference")
            require(set(supports + contradicts) <= set(span_map), f"unknown evidence reference in {cid}:{hid}")
            require(set(supports + contradicts) <= seen_evidence, f"cited evidence not emitted as observation in {cid}:{hid}")
            require(not (set(supports) & set(contradicts)), f"evidence both supports and contradicts {cid}:{hid}")
            require(bool(hypothesis["statement"]) and bool(hypothesis["scope"]), "empty hypothesis statement or scope")
            if hypothesis["epistemic_status"] != "OpenQuestion":
                require(bool(supports), f"non-question hypothesis has no support in {cid}:{hid}")
        require(isinstance(result["open_questions"], list), "open_questions must be an array")
        require(len(result["open_questions"]) == len(set(result["open_questions"])), "duplicate open question")
        require(isinstance(result["abstention"], dict) and set(result["abstention"]) == {"abstain", "reason"}, "abstention schema mismatch")
        require(isinstance(result["abstention"]["abstain"], bool) and bool(result["abstention"]["reason"]), "bad abstention status/reason")


def write_result_bundle(corpus: dict[str, Any], corpus_digest: str, source_revision: str, method_id: str,
                        outputs: list[dict[str, Any]], output_path: Path, config: dict[str, Any]) -> str:
    bundle = {
        "schema": OUTPUT_SCHEMA,
        "corpus_id": corpus["corpus_id"],
        "corpus_version": corpus["version"],
        "corpus_sha256": corpus_digest,
        "source_revision": source_revision,
        "implementation_id": method_id,
        "config_sha256": hashlib.sha256(canonical_json(config)).hexdigest(),
        "results": outputs,
    }
    validate_normalized_bundle(bundle, corpus, corpus_digest)
    raw = canonical_json(bundle)
    output_path.write_bytes(raw)
    # Bind the generated bytes, not an assumed CI execution status.
    return hashlib.sha256(raw).hexdigest()


def summarize_method(cases: list[dict[str, Any]], declared_codes_by_case: dict[str, list[str]]) -> dict[str, Any]:
    """Measure hand-authored implementation claims against the rubric; not output behavior."""
    required_total = required_match_total = forbidden_total = forbidden_claim_total = 0
    case_missing_required = case_forbidden_claims = 0
    per_case = []
    for case in cases:
        required = set(case["required_behaviors"])
        forbidden = set(case["forbidden_behaviors"])
        declared = set(declared_codes_by_case[case["id"]])
        req_matches = required & declared
        forbidden_claims = forbidden & declared
        required_total += len(required)
        required_match_total += len(req_matches)
        forbidden_total += len(forbidden)
        forbidden_claim_total += len(forbidden_claims)
        case_missing_required += int(bool(required - declared))
        case_forbidden_claims += int(bool(forbidden_claims))
        per_case.append({
            "case_id": case["id"],
            "rubric_required_codes": sorted(required),
            "implementation_declared_codes": sorted(declared),
            "declared_required_code_matches": sorted(req_matches),
            "declared_required_codes_not_claimed": sorted(required - declared),
            "rubric_forbidden_codes": sorted(forbidden),
            "declared_forbidden_code_claims": sorted(forbidden_claims),
            "all_required_codes_declared_without_forbidden_claim": not bool(required - declared) and not bool(forbidden_claims),
        })
    return {
        "metric_semantics": "declared_rubric_coverage_only",
        "declared_required_code_total": required_total,
        "declared_required_code_matches": required_match_total,
        "declared_required_code_nonmatches": required_total - required_match_total,
        "declared_required_code_claim_coverage": required_match_total / required_total if required_total else None,
        "rubric_forbidden_code_total": forbidden_total,
        "declared_forbidden_code_claims": forbidden_claim_total,
        "cases_not_claiming_all_required_codes": case_missing_required,
        "cases_declaring_at_least_one_forbidden_code": case_forbidden_claims,
        "cases_declaring_all_required_codes_without_forbidden_claims": sum(
            row["all_required_codes_declared_without_forbidden_claim"] for row in per_case
        ),
        "per_case": per_case,
    }

def run_method(cases: list[dict[str, Any]], method) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    results: list[dict[str, Any]] = []
    signals: dict[str, list[str]] = {}
    for case in cases:
        result, emitted = method(case)
        require(result["case_id"] == case["id"], "method changed case identity")
        results.append(result)
        signals[case["id"]] = emitted
    return results, signals



def self_test(cases: list[dict[str, Any]], methods: list[tuple[str, Any, dict[str, Any]]]) -> list[str]:
    """Metamorphic and adversarial checks over the frozen exposed examples."""
    failures: list[str] = []

    def find_case(needle: str) -> dict[str, Any]:
        matches = [case for case in cases if needle.casefold() in case["input_text"].casefold()]
        if len(matches) != 1:
            raise BaselineError(f"self-test fixture selector {needle!r} matched {len(matches)} cases")
        return matches[0]

    lexical_by_id = {case["id"]: lexical_foil(case) for case in cases}
    contextual_by_id = {case["id"]: context_rule_baseline(case) for case in cases}

    # The context-blind foil must give the same fixed shadow mapping to literal
    # shade and theater-lighting uses; context should not influence this foil.
    shadow_cases = [find_case("tree's shadow"), find_case("lighting effect")]
    for case in shadow_cases:
        matches = [
            item["statement"] for item in lexical_by_id[case["id"]][0]["interpretations"]
            if "'shadow' maps to" in item["statement"]
        ]
        if matches != ["Context-blind dictionary candidate: 'shadow' maps to 'unacknowledged aspect of the psyche'."]:
            failures.append(f"lexical foil did not emit fixed shadow mapping for {case['id']}")

    # The context-aware rules must distinguish literal uses and retain source conflict.
    literal = find_case("tree's shadow")
    if not {"treat_literal_usage_literally", "reject_keyword_only_support"} <= set(contextual_by_id[literal["id"]][1]):
        failures.append("context rules missed literal-shadow negative control")
    injected = find_case("prop-box names")
    if "reject_keyword_only_support" not in contextual_by_id[injected["id"]][1]:
        failures.append("context rules missed keyword-injection control")
    dream = find_case("dreamed about an owl")
    if not contextual_by_id[dream["id"]][0]["abstention"]["abstain"] or "ask_clarifying_question_or_abstain" not in contextual_by_id[dream["id"]][1]:
        failures.append("context rules failed to abstain on underspecified dream")
    disagreement = find_case("Source A")
    if not {"preserve_source_disagreement", "return_competing_readings"} <= set(contextual_by_id[disagreement["id"]][1]):
        failures.append("context rules collapsed conflicting source notes")
    self_report = find_case("I am not afraid")
    if "preserve_self_report" not in contextual_by_id[self_report["id"]][1]:
        failures.append("context rules did not preserve explicit self-report")
    narrative = find_case("avoids the cellar in three scenes")
    if "describe_narrative_change_only" not in contextual_by_id[narrative["id"]][1]:
        failures.append("context rules missed narrative-change-only control")

    # Every case hit by the fixed symbol dictionary must count that
    # context-blind mapping as the two failure modes it is designed to expose.
    for case in cases:
        emitted = set(lexical_by_id[case["id"]][1])
        if "treat_keyword_overlap_as_evidence" in emitted:
            for forbidden_code in ("treat_keyword_overlap_as_evidence", "claim_universal_symbol_meaning"):
                if forbidden_code not in emitted or forbidden_code not in case["forbidden_behaviors"]:
                    failures.append(f"lexical-overreach negative control is not instrumented for {case['id']}:{forbidden_code}")

    # Required/forbidden annotations and oracle prose must not affect predictions.
    for case in cases:
        mutated = {
            **case,
            "required_behaviors": ["MUTATION_REQUIRED_LABEL_LEAK"],
            "forbidden_behaviors": ["MUTATION_FORBIDDEN_LABEL_LEAK"],
            "oracle_note": "MUTATION_ORACLE_LEAK",
        }
        for method_id, method, _ in methods:
            before_result, before_signals = method(case)
            after_result, after_signals = method(mutated)
            if canonical_json(before_result) != canonical_json(after_result) or before_signals != after_signals:
                failures.append(f"{method_id} changed output when labels/oracle prose changed for {case['id']}")
                break

    # Replays and normalized bundle checks are exercised in the runner; ensure
    # no behavior labels or rubric annotations were embedded in normalized rows.
    banned_fields = {"required_behaviors", "forbidden_behaviors", "oracle_note", "rubric_case_pass"}
    for method_id, _, _ in methods:
        for case_id, (result, _) in {
            cid: lexical_by_id[cid] if method_id == "keyword_dictionary_foil_v0" else contextual_by_id[cid]
            for cid in lexical_by_id
        }.items():
            if any(banned_fields & set(row) for row in [result] + result["observations"] + result["interpretations"]):
                failures.append(f"{method_id} leaked rubric fields into output for {case_id}")
                break
    return failures


def validate_glyph_projection(projection: Any, corpus: dict[str, Any], corpus_bytes: bytes, source_revision: str) -> dict[str, dict[str, Any]]:
    """Validate the representation-only receipt from the actual Rust Glyph/HDC API."""
    top_keys = {
        "schema", "corpus_id", "corpus_version", "corpus_status", "corpus_split",
        "corpus_blake3", "corpus_sha256", "source_revision", "glyph_source_blob", "glyph_registry_source_blob", "encoder_source_blob",
        "dimension", "registry_size", "encoder_profile", "model_scope", "basis_profile", "registry_profile",
        "retrieval", "claim_ceiling", "pairwise_input_similarity_profile", "pairwise_input_similarity", "cases",
    }
    require(isinstance(projection, dict) and set(projection) == top_keys, "Glyph projection top-level schema mismatch")
    require(projection["schema"] == "symthaea-jungian-glyph-projection-v1", "Glyph projection schema mismatch")
    require(projection["corpus_id"] == corpus["corpus_id"] and projection["corpus_version"] == corpus["version"], "Glyph projection corpus identity mismatch")
    require(projection["corpus_status"] == "DevelopmentProbe" and projection["corpus_split"] == "ExposedDevelopment", "Glyph projection split/status mismatch")
    require(projection["source_revision"] == source_revision, "Glyph projection source revision mismatch")
    require(re.fullmatch(r"[0-9a-f]{64}", projection["corpus_blake3"] or "") is not None, "Glyph projection corpus BLAKE3 identity is malformed")
    require(projection["corpus_sha256"] == hashlib.sha256(corpus_bytes).hexdigest(), "Glyph projection corpus SHA-256 does not match exact corpus bytes")
    for field in ("glyph_source_blob", "glyph_registry_source_blob", "encoder_source_blob"):
        require(isinstance(projection[field], str) and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", projection[field]) is not None, f"Glyph projection {field} malformed")
    require(projection["dimension"] == 16384, "Glyph projection must use 16,384 dimensions for this profile")
    require(projection["registry_size"] == 70, "Glyph projection registry size mismatch")
    require(projection["model_scope"] == "Existing general-purpose Glyph Codex substrate; not a dedicated Jungian model or validated archetype detector.", "Glyph projection model scope changed")
    require(projection["basis_profile"] == "GlyphBasis::new over the 11 committed MODALITY_KEYWORDS; vectors use TextHdcEncoder with 0.5 trigram and 0.2 sentiment weights.", "Glyph projection basis profile changed")
    require(projection["registry_profile"] == "GlyphRegistry::new encodes only name + core_function + echo_phrase; relational_archetype, somatic_practice, sonic_signature, and field_modality are not part of the embedding input text.", "Glyph projection registry encoding profile changed")
    require(
        projection["pairwise_input_similarity_profile"] == "Cosine between whole-text TextHdcEncoder outputs; reflects this committed lexical/morphological/positional encoding only, not semantic truth.",
        "Glyph projection pairwise similarity profile mismatch",
    )
    require(projection["encoder_profile"] == {
        "type": "TextHdcEncoder",
        "ngram_size": 3,
        "trigram_weight": 0.5,
        "sentiment_weight": 0.2,
        "seed_profile": "the exact committed TextHdcEncoder implementation",
    }, "Glyph projection encoder profile mismatch")
    expected_ids = {case["id"] for case in corpus["cases"]}
    input_sha256 = {case["id"]: hashlib.sha256(case["input_text"].encode("utf-8")).hexdigest() for case in corpus["cases"]}
    rows = projection["cases"]
    require(isinstance(rows, list) and len(rows) == len(expected_ids), "Glyph projection case count mismatch")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"case_id", "input_blake3", "input_sha256", "modality_scores", "top_modalities", "glyph_scores", "top_glyphs"}, "Glyph projection case schema mismatch")
        case_id = row["case_id"]
        require(case_id in expected_ids and case_id not in by_id, f"unknown/duplicate Glyph projection case: {case_id}")
        require(re.fullmatch(r"[0-9a-f]{64}", row["input_blake3"] or "") is not None, f"malformed input BLAKE3 digest for {case_id}")
        require(row["input_sha256"] == input_sha256[case_id], f"input SHA-256 does not bind exact corpus text for {case_id}")
        scores = row["modality_scores"]
        require(isinstance(scores, list) and len(scores) == 11, f"{case_id}: expected 11 modality scores")
        names: set[str] = set()
        for i, score in enumerate(scores):
            require(isinstance(score, dict) and set(score) == {"index", "modality", "cosine_similarity"}, f"{case_id}: modality score shape mismatch")
            require(score["index"] == i and isinstance(score["modality"], str), f"{case_id}: noncanonical modality order")
            require(score["modality"] not in names, f"{case_id}: duplicate modality name")
            names.add(score["modality"])
            value = score["cosine_similarity"]
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and -1 <= value <= 1, f"{case_id}: invalid modality similarity")
        top_modalities = row["top_modalities"]
        require(isinstance(top_modalities, list) and len(top_modalities) == 5, f"{case_id}: top_modalities must contain top five")
        expected_top = sorted(scores, key=lambda item: (-item["cosine_similarity"], item["index"]))[:5]
        for rank, (item, expected) in enumerate(zip(top_modalities, expected_top, strict=True), start=1):
            require(isinstance(item, dict) and set(item) == {"rank", "index", "modality", "cosine_similarity"}, f"{case_id}: top modality schema mismatch")
            require(item["rank"] == rank, f"{case_id}: invalid top modality rank")
            require(item["index"] == expected["index"] and item["modality"] == expected["modality"], f"{case_id}: top modality ranking does not derive from full modality scores")
            require(item["cosine_similarity"] == expected["cosine_similarity"], f"{case_id}: top modality score mismatches full score row")

        full_glyph_scores = row["glyph_scores"]
        require(isinstance(full_glyph_scores, list) and len(full_glyph_scores) == 70, f"{case_id}: glyph_scores must include all 70 registry entries")
        glyph_ids_all: set[str] = set()
        for rank, item in enumerate(full_glyph_scores, start=1):
            require(isinstance(item, dict) and set(item) == {"rank", "glyph_id", "name", "class", "field_modality", "cosine_similarity"}, f"{case_id}: full glyph-score schema mismatch")
            require(item["rank"] == rank, f"{case_id}: full glyph ranks are not contiguous")
            gid = item["glyph_id"]
            require(isinstance(gid, str) and gid and gid not in glyph_ids_all, f"{case_id}: empty/duplicate glyph id in full ranking")
            glyph_ids_all.add(gid)
            for name_field in ("name", "class", "field_modality"):
                require(isinstance(item[name_field], str) and item[name_field], f"{case_id}: missing full glyph {name_field}")
            metric = item["cosine_similarity"]
            require(isinstance(metric, (int, float)) and not isinstance(metric, bool) and math.isfinite(metric) and -1 <= metric <= 1, f"{case_id}: invalid full glyph score")
            if rank > 1:
                previous = full_glyph_scores[rank - 2]
                previous_metric = previous["cosine_similarity"]
                require(previous_metric > metric or (previous_metric == metric and previous["glyph_id"] <= gid), f"{case_id}: full glyph ranking not sorted by score/stable ID")

        top_glyphs = row["top_glyphs"]
        require(isinstance(top_glyphs, list) and len(top_glyphs) == 5, f"{case_id}: top_glyphs must contain top five")
        for rank, item in enumerate(top_glyphs, start=1):
            require(isinstance(item, dict) and set(item) == {"rank", "glyph_id", "name", "class", "field_modality", "cosine_similarity", "core_function", "echo_phrase"}, f"{case_id}: top glyph schema mismatch")
            require(item["rank"] == rank, f"{case_id}: invalid top glyph rank")
            metric = item.get("cosine_similarity")
            require(isinstance(metric, (int, float)) and not isinstance(metric, bool) and math.isfinite(metric) and -1 <= metric <= 1, f"{case_id}: invalid top glyph similarity")
            if rank > 1:
                previous = top_glyphs[rank - 2]
                previous_metric = previous["cosine_similarity"]
                require(previous_metric > metric or (previous_metric == metric and previous["glyph_id"] <= item["glyph_id"]), f"{case_id}: top glyphs not sorted by descending similarity/stable ID")
        glyph_ids = [glyph.get("glyph_id") for glyph in row["top_glyphs"]]
        require(all(isinstance(gid, str) and gid for gid in glyph_ids) and len(set(glyph_ids)) == 5, f"{case_id}: duplicate/missing glyph ID")
        for top_item, full_item in zip(top_glyphs, full_glyph_scores[:5], strict=True):
            for field in ("rank", "glyph_id", "name", "class", "field_modality", "cosine_similarity"):
                require(top_item[field] == full_item[field], f"{case_id}: top-five glyph ranking conflicts with complete 70-entry ranking")
        by_id[case_id] = row
    require(set(by_id) == expected_ids, "Glyph projection case set incomplete")

    pair_rows = projection["pairwise_input_similarity"]
    expected_case_order = [case["id"] for case in corpus["cases"]]
    expected_pairs = [
        (expected_case_order[left], expected_case_order[right])
        for left in range(len(expected_case_order))
        for right in range(left + 1, len(expected_case_order))
    ]
    require(isinstance(pair_rows, list) and len(pair_rows) == len(expected_pairs), "Glyph projection pairwise matrix must contain every unordered case pair")
    seen_pairs: list[tuple[str, str]] = []
    for row, expected_pair in zip(pair_rows, expected_pairs, strict=True):
        require(isinstance(row, dict) and set(row) == {"left_case_id", "right_case_id", "cosine_similarity"}, "Glyph projection pairwise row schema mismatch")
        pair = (row["left_case_id"], row["right_case_id"])
        require(pair == expected_pair, "Glyph projection pairwise rows are not in canonical corpus order")
        similarity = row["cosine_similarity"]
        require(isinstance(similarity, (int, float)) and not isinstance(similarity, bool) and math.isfinite(similarity) and -1 <= similarity <= 1, f"invalid pairwise input similarity: {pair}")
        seen_pairs.append(pair)
    require(len(seen_pairs) == len(set(seen_pairs)), "duplicate pairwise similarity row")
    return by_id




def context_pair_diagnostics(
    corpus: dict[str, Any],
    projection_by_id: dict[str, dict[str, Any]],
    projection: dict[str, Any],
) -> list[dict[str, Any]]:
    """Report exploratory context-pair geometry; deliberately has no pass threshold."""
    input_similarity = {
        (row["left_case_id"], row["right_case_id"]): row["cosine_similarity"]
        for row in projection["pairwise_input_similarity"]
    }
    pairs = [
        ("JSP-001", "JSP-002", "key object in two different access contexts"),
        ("JSP-003", "JSP-014", "the word shadow in tree-shade versus theatre-lighting contexts"),
    ]
    rows = []
    for left_id, right_id, rationale in pairs:
        left = projection_by_id[left_id]
        right = projection_by_id[right_id]
        left_vector = [item["cosine_similarity"] for item in left["modality_scores"]]
        right_vector = [item["cosine_similarity"] for item in right["modality_scores"]]
        dot = sum(a * b for a, b in zip(left_vector, right_vector, strict=True))
        norm_left = math.sqrt(sum(a * a for a in left_vector))
        norm_right = math.sqrt(sum(b * b for b in right_vector))
        projected_cosine = dot / (norm_left * norm_right) if norm_left and norm_right else None
        projected_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left_vector, right_vector, strict=True)))
        left_glyphs = [item["glyph_id"] for item in left["top_glyphs"]]
        right_glyphs = [item["glyph_id"] for item in right["top_glyphs"]]
        union = set(left_glyphs) | set(right_glyphs)
        intersection = set(left_glyphs) & set(right_glyphs)
        rows.append({
            "left_case_id": left_id,
            "right_case_id": right_id,
            "pair_rationale": rationale,
            "whole_text_hdc_cosine": input_similarity[(left_id, right_id)],
            "modality_projection_cosine": projected_cosine,
            "modality_projection_l2_distance": projected_distance,
            "top1_glyph_equal": left_glyphs[0] == right_glyphs[0],
            "top5_glyph_overlap": sorted(intersection),
            "top5_glyph_jaccard": len(intersection) / len(union) if union else 1.0,
            "claim_ceiling": "Exploratory representation geometry only; no expected direction, pass threshold, semantic correctness, or psychological interpretation claim.",
        })
    return rows


def self_test_glyph_projection_validator(
    corpus: dict[str, Any],
    corpus_bytes: bytes,
    source_revision: str,
) -> list[str]:
    """Positive and adversarial tests for the actual-HDC receipt validator."""
    case_rows = []
    for case in corpus["cases"]:
        full_scores = [
            {"index": index, "modality": f"Modality{index:02d}", "cosine_similarity": 1.0 - index / 10.0}
            for index in range(11)
        ]
        top_modalities = [
            {
                "rank": rank,
                "index": index,
                "modality": f"Modality{index:02d}",
                "cosine_similarity": 1.0 - index / 10.0,
            }
            for rank, index in enumerate(range(5), start=1)
        ]
        top_glyphs = [
            {
                "rank": rank,
                "glyph_id": f"G{index:02d}",
                "name": f"Glyph {index:02d}",
                "class": "SyntheticTestOnly",
                "field_modality": f"Modality{index:02d}",
                "cosine_similarity": 1.0 - index / 10.0,
                "core_function": "validator self-test fixture",
                "echo_phrase": "not a psychological interpretation",
            }
            for rank, index in enumerate(range(5), start=1)
        ]
        case_rows.append({
            "case_id": case["id"],
            "input_blake3": "0" * 64,
            "input_sha256": hashlib.sha256(case["input_text"].encode("utf-8")).hexdigest(),
            "modality_scores": full_scores,
            "top_modalities": top_modalities,
            "glyph_scores": [
                {
                    "rank": index + 1,
                    "glyph_id": f"G{index:02d}",
                    "name": f"Glyph {index:02d}",
                    "class": "SyntheticTestOnly",
                    "field_modality": f"Modality{index:02d}",
                    "cosine_similarity": 1.0 - index / 100.0,
                }
                for index in range(70)
            ],
            "top_glyphs": [
                {
                    **{
                        "rank": index + 1,
                        "glyph_id": f"G{index:02d}",
                        "name": f"Glyph {index:02d}",
                        "class": "SyntheticTestOnly",
                        "field_modality": f"Modality{index:02d}",
                        "cosine_similarity": 1.0 - index / 100.0,
                    },
                    "core_function": "validator self-test fixture",
                    "echo_phrase": "not a psychological interpretation",
                }
                for index in range(5)
            ],
        })
    pair_rows = [
        {"left_case_id": corpus["cases"][left]["id"], "right_case_id": corpus["cases"][right]["id"], "cosine_similarity": 0.0}
        for left in range(len(corpus["cases"]))
        for right in range(left + 1, len(corpus["cases"]))
    ]
    fixture = {
        "schema": "symthaea-jungian-glyph-projection-v1",
        "corpus_id": corpus["corpus_id"],
        "corpus_version": corpus["version"],
        "corpus_status": "DevelopmentProbe",
        "corpus_split": "ExposedDevelopment",
        "corpus_blake3": "0" * 64,
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "source_revision": source_revision,
        "glyph_source_blob": "1" * 40,
        "glyph_registry_source_blob": "2" * 40,
        "encoder_source_blob": "3" * 40,
        "dimension": 16384,
        "registry_size": 70,
        "model_scope": "Existing general-purpose Glyph Codex substrate; not a dedicated Jungian model or validated archetype detector.",
        "basis_profile": "GlyphBasis::new over the 11 committed MODALITY_KEYWORDS; vectors use TextHdcEncoder with 0.5 trigram and 0.2 sentiment weights.",
        "registry_profile": "GlyphRegistry::new encodes only name + core_function + echo_phrase; relational_archetype, somatic_practice, sonic_signature, and field_modality are not part of the embedding input text.",
        "encoder_profile": {
            "type": "TextHdcEncoder",
            "ngram_size": 3,
            "trigram_weight": 0.5,
            "sentiment_weight": 0.2,
            "seed_profile": "the exact committed TextHdcEncoder implementation",
        },
        "basis_profile": "synthetic validator fixture",
        "registry_profile": "synthetic validator fixture",
        "retrieval": "cosine similarity only; nearest-neighbor association is not an interpretation or truth score",
        "claim_ceiling": "validator fixture only",
        "pairwise_input_similarity_profile": "Cosine between whole-text TextHdcEncoder outputs; reflects this committed lexical/morphological/positional encoding only, not semantic truth.",
        "pairwise_input_similarity": pair_rows,
        "cases": case_rows,
    }

    failures: list[str] = []
    try:
        validate_glyph_projection(fixture, corpus, corpus_bytes, source_revision)
    except Exception as exc:
        failures.append(f"positive projection-receipt fixture rejected: {exc}")

    mutations: list[tuple[str, dict[str, Any], str]] = []
    mutation = json.loads(json.dumps(fixture))
    mutation["corpus_sha256"] = "f" * 64
    mutations.append(("corpus digest mismatch rejected", mutation, "corpus SHA-256"))
    mutation = json.loads(json.dumps(fixture))
    mutation["cases"][0]["input_sha256"] = "f" * 64
    mutations.append(("input digest mismatch rejected", mutation, "input SHA-256"))
    mutation = json.loads(json.dumps(fixture))
    mutation["cases"][0]["top_modalities"][0], mutation["cases"][0]["top_modalities"][1] = mutation["cases"][0]["top_modalities"][1], mutation["cases"][0]["top_modalities"][0]
    mutations.append(("modality rank corruption rejected", mutation, "invalid top modality rank"))
    mutation = json.loads(json.dumps(fixture))
    mutation["pairwise_input_similarity"][0], mutation["pairwise_input_similarity"][1] = mutation["pairwise_input_similarity"][1], mutation["pairwise_input_similarity"][0]
    mutations.append(("pair ordering corruption rejected", mutation, "canonical corpus order"))
    mutation = json.loads(json.dumps(fixture))
    mutation["encoder_source_blob"] = "not-a-git-blob"
    mutations.append(("encoder source identity corruption rejected", mutation, "encoder_source_blob malformed"))
    mutation = json.loads(json.dumps(fixture))
    mutation["cases"][0]["glyph_scores"][0], mutation["cases"][0]["glyph_scores"][1] = mutation["cases"][0]["glyph_scores"][1], mutation["cases"][0]["glyph_scores"][0]
    mutations.append(("full glyph ranking corruption rejected", mutation, "full glyph ranks are not contiguous"))
    mutation = json.loads(json.dumps(fixture))
    mutation["cases"][0]["top_glyphs"][0]["name"] = "Contradictory top-five name"
    mutations.append(("top-five/full-rank divergence rejected", mutation, "top-five glyph ranking conflicts with complete 70-entry ranking"))

    for name, mutated, expected_error in mutations:
        try:
            validate_glyph_projection(mutated, corpus, corpus_bytes, source_revision)
            failures.append(name + ": mutation unexpectedly accepted")
        except BaselineError as exc:
            if expected_error not in str(exc):
                failures.append(f"{name}: expected {expected_error!r}, received {str(exc)!r}")
        except Exception as exc:
            failures.append(f"{name}: wrong exception type {type(exc).__name__}: {exc}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for immutable-by-digest output captures")
    parser.add_argument("--source-revision", required=True, help="exact 40- or 64-character lowercase Git revision under test")
    parser.add_argument("--glyph-projection", type=Path, help="optional exact-source output from jungian_glyph_projection_probe")
    args = parser.parse_args()

    try:
        corpus_bytes = args.corpus.read_bytes()
        corpus = load_json_bytes(corpus_bytes)
        corpus_git_blob_sha1 = git_blob_sha1(corpus_bytes)
        require(
            corpus_git_blob_sha1 == EXPECTED_CORPUS_GIT_BLOB_SHA1,
            f"corpus Git blob identity changed: got {corpus_git_blob_sha1}, expected {EXPECTED_CORPUS_GIT_BLOB_SHA1}; bump/review the corpus version before re-baselining",
        )
        require(corpus.get("schema") == CORPUS_SCHEMA, "unexpected corpus schema")
        require(corpus.get("status") == "DevelopmentProbe" and corpus.get("split") == "ExposedDevelopment", "refusing to treat an exposed set as holdout data")
        require(re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", args.source_revision) is not None, "--source-revision must be a full lowercase Git object ID")
        cases = corpus["cases"]
        require(len(cases) == 14 and len({c["id"] for c in cases}) == len(cases), "unexpected or duplicate corpus case set")
        corpus_digest = hashlib.sha256(corpus_bytes).hexdigest()
        runner_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        projection_test_failures = self_test_glyph_projection_validator(corpus, corpus_bytes, args.source_revision)
        if projection_test_failures:
            raise BaselineError("Glyph projection validator self-test failures: " + "; ".join(projection_test_failures))
        print("PASS Glyph projection validator self-test: valid receipt plus seven hostile mutations")
        args.output_dir.mkdir(parents=True, exist_ok=True)

        glyph_projection_by_id = None
        glyph_projection_sha256 = None
        glyph_projection_meta = None
        if args.glyph_projection:
            projection_bytes = args.glyph_projection.read_bytes()
            projection = load_json(args.glyph_projection)
            glyph_projection_by_id = validate_glyph_projection(projection, corpus, corpus_bytes, args.source_revision)
            glyph_projection_sha256 = hashlib.sha256(projection_bytes).hexdigest()
            glyph_projection_meta = {
                "file": args.glyph_projection.name,
                "sha256": glyph_projection_sha256,
                "source_revision": projection["source_revision"],
                "glyph_source_blob": projection["glyph_source_blob"],
                "glyph_registry_source_blob": projection["glyph_registry_source_blob"],
                "encoder_source_blob": projection["encoder_source_blob"],
                "corpus_blake3": projection["corpus_blake3"],
                "corpus_sha256": projection["corpus_sha256"],
                "dimension": projection["dimension"],
                "registry_size": projection["registry_size"],
            }

        methods = [
            ("keyword_dictionary_foil_v0", lexical_foil, {
                "class": "Intentionally context-blind failure-prone foil",
                "purpose": "Expose cases where fixed keyword-to-meaning mappings overgeneralize.",
                "production_use": False,
            }),
            ("context_guardrails_rules_v0", context_rule_baseline, {
                "class": "Hand-authored deterministic context rules",
                "purpose": "Probe explicit context, attribution, and abstention paths independently of the HDC projection.",
                "production_use": False,
            }),
        ]
        capture = {
            "schema": REPORT_SCHEMA,
            "corpus_id": corpus["corpus_id"],
            "corpus_version": corpus["version"],
            "corpus_status": corpus["status"],
            "corpus_split": corpus["split"],
            "corpus_sha256": corpus_digest,
            "corpus_git_blob_sha1": corpus_git_blob_sha1,
            "runner_sha256": runner_digest,
            "source_revision": args.source_revision,
            "runtime_profile": {
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "python_version_full": sys.version.replace("\n", " "),
                "platform": platform.platform(),
                "architecture": platform.machine(),
                "rustc_version": os.environ.get("SYMTHAEA_RUSTC_VERSION", "not-provided"),
                "cargo_version": os.environ.get("SYMTHAEA_CARGO_VERSION", "not-provided"),
                "rust_toolchain_manifest_sha256": (
                    hashlib.sha256((ROOT / "rust-toolchain.toml").read_bytes()).hexdigest()
                    if (ROOT / "rust-toolchain.toml").is_file() else None
                ),
            },
            "claim_ceiling": "Rule output and rubric agreement only; HDC output is representation-level retrieval only. No psychological truth, archetype validity, clinical efficacy, or consciousness claim.",
            "metric_warning": "Coverage scores the behavior codes manually declared by each hand-authored baseline implementation; it does not infer behavior from output text and is not independent output evaluation, psychological accuracy, model generalization, or holdout performance.",
            "metric_semantics": "declared_rubric_coverage_only",
            "glyph_projection_artifact": glyph_projection_meta,
            "context_pair_diagnostics": (
                context_pair_diagnostics(corpus, glyph_projection_by_id, projection)
                if glyph_projection_by_id is not None else None
            ),
            "methods": [],
            "cases": [],
        }
        outputs_by_method: dict[str, list[dict[str, Any]]] = {}
        signals_by_method: dict[str, dict[str, list[str]]] = {}
        config_hashes: dict[str, str] = {}

        test_failures = self_test(cases, methods)
        if test_failures:
            raise BaselineError("baseline self-test failures: " + "; ".join(test_failures))
        print("PASS baseline self-test: literal/context controls, abstention, disagreement, self-report, narrative arc, and label-leakage invariance")

        for method_id, method, metadata in methods:
            outputs, signals = run_method(cases, method)
            # Re-run immediately to catch hidden nondeterminism in the baseline implementation.
            outputs_again, signals_again = run_method(cases, method)
            require(canonical_json(outputs) == canonical_json(outputs_again), f"{method_id} normalized outputs are nondeterministic")
            require(canonical_json(signals) == canonical_json(signals_again), f"{method_id} behavior signals are nondeterministic")
            normalized_path = args.output_dir / f"{method_id}.normalized.json"
            config = {"method_id": method_id, **metadata, "corpus_sha256": corpus_digest}
            result_digest = write_result_bundle(
                corpus, corpus_digest, args.source_revision, method_id, outputs, normalized_path, config
            )
            outputs_by_method[method_id] = outputs
            signals_by_method[method_id] = signals
            config_hashes[method_id] = hashlib.sha256(canonical_json(config)).hexdigest()
            summary = summarize_method(cases, signals)
            capture["methods"].append({
                "method_id": method_id,
                **metadata,
                "state": "executed",
                "normalized_output_file": normalized_path.name,
                "normalized_output_sha256": result_digest,
                "config_sha256": config_hashes[method_id],
                "execution_replay": "byte-identical within this process",
                "summary": {k: v for k, v in summary.items() if k != "per_case"},
                "metric_semantics": summary["metric_semantics"],
            })

        if glyph_projection_by_id is not None:
            capture["methods"].append({
                "method_id": "glyph_hdc_projection_v0",
                "class": "Actual Symthaea GlyphBasis / TextHdcEncoder / GlyphRegistry",
                "state": "projection_only_not_rubric_scored",
                "reason": "Produces modality scores and nearest existing glyph vectors only. It does not emit contextual interpretation hypotheses or behavior codes, so it is not scored against the interpretation rubric.",
                "summary": None,
                "artifact_sha256": glyph_projection_sha256,
            })
        else:
            capture["methods"].append({
                "method_id": "glyph_hdc_projection_v0",
                "class": "Actual Symthaea GlyphBasis / TextHdcEncoder / GlyphRegistry",
                "state": "not_run",
                "reason": "No Rust projection receipt supplied; no HDC result is fabricated from a Python stand-in.",
                "summary": None,
            })

        for case in cases:
            capture["cases"].append({
                "case_id": case["id"],
                "annotations": {
                    "required_behaviors": case["required_behaviors"],
                    "forbidden_behaviors": case["forbidden_behaviors"],
                },
                "methods": [
                    {
                        "method_id": method_id,
                        "implementation_declared_codes": signals_by_method[method_id][case["id"]],
                        "declared_required_code_matches": sorted(set(case["required_behaviors"]) & set(signals_by_method[method_id][case["id"]])),
                        "declared_required_codes_not_claimed": sorted(set(case["required_behaviors"]) - set(signals_by_method[method_id][case["id"]])),
                        "declared_forbidden_code_claims": sorted(set(case["forbidden_behaviors"]) & set(signals_by_method[method_id][case["id"]])),
                    }
                    for method_id, _, _ in methods
                ],
                "glyph_projection": (
                    {
                        "state": "projection_only_not_rubric_scored",
                        "input_blake3": glyph_projection_by_id[case["id"]]["input_blake3"],
                        "top_modalities": glyph_projection_by_id[case["id"]]["top_modalities"],
                        "top_glyphs": glyph_projection_by_id[case["id"]]["top_glyphs"],
                    }
                    if glyph_projection_by_id is not None else {"state": "not_run"}
                ),
            })

        report_path = args.output_dir / "baseline_comparison.json"
        report_path.write_bytes(canonical_json(capture))
        report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()

        print(f"corpus_sha256={corpus_digest}")
        print(f"corpus_git_blob_sha1={corpus_git_blob_sha1}")
        print(f"runner_sha256={runner_digest}")
        print(f"source_revision={args.source_revision}")
        print(f"comparison_report={report_path}")
        print(f"comparison_report_sha256={report_digest}")
        for item in capture["methods"]:
            if item["state"] == "projection_only_not_rubric_scored":
                print(f"{item['method_id']}: PROJECTION_ONLY; artifact_sha256={item['artifact_sha256']}; not interpreted or rubric-scored")
                continue
            if item["state"] != "executed":
                print(f"{item['method_id']}: NOT_RUN ({item['reason']})")
                continue
            summary = item["summary"]
            print(
                f"{item['method_id']}: declared_code_coverage={summary['declared_required_code_claim_coverage']:.4f}; "
                f"declared_matches={summary['declared_required_code_matches']}/{summary['declared_required_code_total']}; "
                f"declared_forbidden_claims={summary['declared_forbidden_code_claims']}; "
                f"cases_claiming_complete_coverage={summary['cases_declaring_all_required_codes_without_forbidden_claims']}/14; "
                f"metric=declared_rubric_coverage_only; replay=byte-identical"
            )
        return 0
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, BaselineError, KeyError, TypeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
