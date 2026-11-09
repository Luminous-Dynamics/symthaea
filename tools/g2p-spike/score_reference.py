#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Score G2P candidate output against independently adjudicated word references.

The input token columns are Rust Debug lists emitted by g2p-spike. This script
uses a deliberately coarse Unicode-codepoint edit measure over normalized IPA
text. It is NOT canonical phoneme error rate: combining diacritics, affricates,
diphthongs and provider-specific segmentations require a reviewed language
inventory before a true phone-level metric can be computed. Use tools/g2p-eval
with canonical phone tokens for PER and phone confusion reporting.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import string
from pathlib import Path
from typing import Any

STRESS_MARKERS = {"ˈ", "ˌ"}
PUNCTUATION = set(string.punctuation) | {
    "—", "–", "“", "”", "‘", "’", "…", "«", "»", "¿", "¡"
}


def is_punctuation_token(token: str) -> bool:
    return bool(token) and all(
        character in string.punctuation
        or character in {"—", "–", "“", "”", "‘", "’", "…", "«", "»", "¿", "¡"}
        for character in token
    )


def expected_punctuation_events(raw_tokens: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    word_index = 0
    character_index = 0
    pending_stress = False
    for raw_token_index, token in enumerate(raw_tokens):
        if token == " ":
            if pending_stress:
                raise ValueError("raw IPA stream has a stress marker before a word separator")
            word_index += 1
            character_index = 0
        elif token in STRESS_MARKERS:
            if pending_stress:
                raise ValueError("raw IPA stream has consecutive stress markers")
            pending_stress = True
        elif is_punctuation_token(token):
            if pending_stress:
                raise ValueError("raw IPA stream has a stress marker before punctuation")
            events.append(
                {
                    "raw_token_index": raw_token_index,
                    "word_index": word_index,
                    "after_ipa_character_index": character_index,
                    "token": token,
                }
            )
        else:
            pending_stress = False
            character_index += len(token)
    if pending_stress:
        raise ValueError("raw IPA stream ends with a dangling stress marker")
    return events


def parse_debug_list(raw: str, *, expected_nested: bool = False) -> list[Any]:
    if raw.startswith("ERROR"):
        return []
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid Rust Debug list: {raw!r}") from exc
    if not isinstance(value, list):
        raise ValueError(f"expected a list, got {type(value).__name__}")
    if expected_nested:
        if not all(isinstance(word, list) and all(isinstance(token, str) for token in word) for word in value):
            raise ValueError("expected a list of per-word string lists")
    elif not all(isinstance(token, str) for token in value):
        raise ValueError("expected a list of string tokens")
    return value


def split_piper_words(raw: str | list[str]) -> list[list[str]]:
    """Split either a Rust Debug list string or an already-parsed token list."""
    if isinstance(raw, str):
        tokens = parse_debug_list(raw)
    elif isinstance(raw, list) and all(isinstance(token, str) for token in raw):
        tokens = raw
    else:
        raise ValueError("Piper tokens must be a Debug-list string or list[str]")
    words: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token.isspace():
            # Preserve an empty lexical slot when a backend emits no phones
            # for a word between explicit separators. Dropping that slot would
            # shift every later reference word and corrupt the comparison.
            words.append(current)
            current = []
        elif is_punctuation_token(token) or token in STRESS_MARKERS:
            continue
        else:
            current.append(token)
    if current or not words:
        words.append(current)
    return words


def normalize_ipa(tokens: list[str]) -> list[str]:
    """Flatten token boundaries into Unicode code points; do not merge phones."""
    codepoints: list[str] = []
    for token in tokens:
        if token.isspace() or is_punctuation_token(token) or token in STRESS_MARKERS:
            continue
        codepoints.extend(list(token))
    return codepoints


def split_normalized_piper_words(
    raw_json: str,
    raw_tokens: list[str] | None = None,
) -> list[list[str]]:
    """Read the loss-preserving Rust word view; punctuation remains separate."""
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid normalized Piper hypothesis JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("words"), list):
        raise ValueError("normalized Piper hypothesis must contain a words array")

    provider_tokens = payload.get("raw_tokens")
    provider_prosody = payload.get("raw_prosody")
    if not isinstance(provider_tokens, list) or not all(
        isinstance(token, str) for token in provider_tokens
    ):
        raise ValueError("normalized Piper hypothesis must preserve raw_tokens")
    if not isinstance(provider_prosody, list) or len(provider_prosody) != len(provider_tokens):
        raise ValueError("normalized Piper raw token/prosody arrays are misaligned")
    if raw_tokens is not None and provider_tokens != raw_tokens:
        raise ValueError("normalized Piper raw token stream differs from the TSV source stream")
    expected_punctuation = expected_punctuation_events(provider_tokens)
    if payload.get("punctuation") != expected_punctuation:
        raise ValueError("normalized Piper punctuation offsets do not reconstruct the raw token stream")
    expected_word_slots = split_piper_words(provider_tokens)

    words: list[list[str]] = []
    for expected_index, word in enumerate(payload["words"]):
        if not isinstance(word, dict) or word.get("word_index") != expected_index:
            raise ValueError("normalized Piper word indices must be contiguous and zero-based")
        raw_tokens = word.get("ipa_character_tokens")
        raw_ipa = word.get("raw_ipa")
        if (
            not isinstance(raw_tokens, list)
            or not all(isinstance(token, str) for token in raw_tokens)
            or not isinstance(raw_ipa, str)
        ):
            raise ValueError(f"normalized Piper word {expected_index} has malformed IPA fields")
        if "".join(raw_tokens) != raw_ipa:
            raise ValueError(
                f"normalized Piper word {expected_index} raw_ipa disagrees with its character tokens"
            )
        if len(word.get("character_prosody", [])) != len(raw_tokens):
            raise ValueError(
                f"normalized Piper word {expected_index} character/prosody arrays are misaligned"
            )
        words.append(raw_tokens)
    if len(words) != len(expected_word_slots):
        raise ValueError(
            "normalized Piper word count differs from the raw token stream's explicit separator count"
        )
    if words != expected_word_slots:
        raise ValueError(
            "normalized Piper word tokens do not reconstruct the raw stream after removing only stress/punctuation"
        )
    return words


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_phone in enumerate(reference, start=1):
        current = [i]
        for j, hyp_phone in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_phone != hyp_phone),
                )
            )
        previous = current
    return previous[-1]


def reference_variants(raw: str) -> list[list[str]]:
    variants = [variant.strip() for variant in raw.split("||") if variant.strip()]
    if not variants:
        return []
    return [normalize_ipa(variant.split()) for variant in variants]


def score_system(
    references: list[dict[str, str]],
    candidates: dict[str, dict[str, list[list[str]]]],
    system_name: str,
) -> dict[str, Any]:
    phone_errors = 0
    reference_codepoints = 0
    exact_words = 0
    evaluated_words = 0
    missing_words = 0
    heteronym_exact = 0
    heteronym_total = 0
    oov_total = 0
    oov_nonempty_guesses = 0

    for reference in references:
        case_id = reference["case_id"]
        word_index = int(reference["word_index"])
        status = reference["reference_status"]
        case = candidates.get(case_id, {})
        words = case.get(system_name, [])
        observed = normalize_ipa(words[word_index]) if word_index < len(words) else []

        if status == "out_of_vocabulary":
            oov_total += 1
            if observed:
                oov_nonempty_guesses += 1
            continue
        if status != "reviewed":
            continue

        variants = reference_variants(reference["accepted_ipa_variants"])
        if not variants:
            raise ValueError(
                f"{case_id} word {word_index}: reviewed reference has no accepted IPA variant"
            )
        distances = [edit_distance(variant, observed) for variant in variants]
        best_index = min(range(len(distances)), key=distances.__getitem__)
        best_distance = distances[best_index]
        best_reference = variants[best_index]

        phone_errors += best_distance
        reference_codepoints += len(best_reference)
        evaluated_words += 1
        exact = best_distance == 0
        exact_words += int(exact)
        if case_id.startswith("heteronym_"):
            heteronym_total += 1
            heteronym_exact += int(exact)

        if word_index >= len(words) or not observed:
            missing_words += 1

    if evaluated_words == 0 and oov_total == 0:
        raise ValueError("no adjudicated reference rows found; pending rows are not scored")

    return {
        "system": system_name,
        "evaluated_words": evaluated_words,
        "reference_codepoints": reference_codepoints,
        "phone_edits": phone_errors,
        "ipa_codepoint_edit_rate": phone_errors / reference_codepoints if reference_codepoints else None,
        "exact_normalized_ipa_sequence_accuracy": exact_words / evaluated_words if evaluated_words else None,
        "missing_candidate_words": missing_words,
        "heteronym_words": heteronym_total,
        "heteronym_exact_sequence_accuracy": heteronym_exact / heteronym_total if heteronym_total else None,
        "oov_words": oov_total,
        "oov_nonempty_guess_rate": oov_nonempty_guesses / oov_total if oov_total else None,
        "oov_nonempty_guess_count": oov_nonempty_guesses,
    }


def evaluate(candidate_tsv: Path, reference_csv: Path) -> dict[str, Any]:
    candidate_rows: dict[str, dict[str, list[list[str]]]] = {}
    with candidate_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "case_id",
            "piper_ipa_tokens",
            "normalized_word_hypotheses_json",
            "muse_word_ipa_tokens",
        }
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"candidate TSV must contain columns: {sorted(required)}")
        for row in reader:
            case_id = row["case_id"]
            if case_id in candidate_rows:
                raise ValueError(f"duplicate candidate case_id {case_id!r}")
            raw_piper_tokens = parse_debug_list(row["piper_ipa_tokens"])
            piper_words = split_normalized_piper_words(
                row["normalized_word_hypotheses_json"],
                raw_tokens=raw_piper_tokens,
            )
            muse_words = parse_debug_list(row["muse_word_ipa_tokens"], expected_nested=True)
            candidate_rows[case_id] = {
                "piper": piper_words,
                "muse": muse_words,
            }

    with reference_csv.open("r", encoding="utf-8", newline="") as handle:
        reference_rows = list(csv.DictReader(handle))
    required = {
        "case_id",
        "word_index",
        "accepted_ipa_variants",
        "reference_status",
        "reviewer_evidence",
    }
    if not required.issubset(reference_rows[0].keys() if reference_rows else set()):
        raise ValueError(f"reference CSV must contain columns: {sorted(required)}")

    adjudicated = []
    reference_slots: dict[str, set[int]] = {}
    all_reference_slots: dict[str, set[int]] = {}
    for row in reference_rows:
        status = row["reference_status"].strip()
        case_id = row["case_id"].strip()
        try:
            word_index = int(row["word_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{case_id}: word_index must be a non-negative integer") from exc
        if not case_id or word_index < 0:
            raise ValueError("reference case IDs must be non-empty and word_index non-negative")
        all_reference_slots.setdefault(case_id, set())
        if word_index in all_reference_slots[case_id]:
            raise ValueError(f"{case_id}: duplicate word_index {word_index}")
        all_reference_slots[case_id].add(word_index)
        if status == "pending" or not status:
            continue
        if status == "reviewed" and not row["reviewer_evidence"].strip():
            raise ValueError(
                f"{case_id} word {word_index}: reviewed row requires reviewer evidence"
            )
        if status == "out_of_vocabulary":
            if not row["reviewer_evidence"].strip():
                raise ValueError(
                    f"{case_id} word {word_index}: OOV row requires reviewer evidence"
                )
            if row["accepted_ipa_variants"].strip():
                raise ValueError(
                    f"{case_id} word {word_index}: OOV row must not contain accepted IPA variants"
                )
        if status == "reviewed" and not row["accepted_ipa_variants"].strip():
            raise ValueError(
                f"{case_id} word {word_index}: reviewed row requires accepted IPA variants"
            )
        if status not in {"reviewed", "out_of_vocabulary"}:
            raise ValueError(f"unknown reference_status {status!r}")
        reference_slots.setdefault(case_id, set())
        if word_index in reference_slots[case_id]:
            raise ValueError(f"{case_id}: duplicate adjudicated word_index {word_index}")
        reference_slots[case_id].add(word_index)
        adjudicated.append(row)

    if not adjudicated:
        raise ValueError("no adjudicated reference rows found; pending rows are not scored")
    for case_id, indices in all_reference_slots.items():
        expected = set(range(len(indices)))
        if indices != expected:
            raise ValueError(
                f"{case_id}: reference word slots (including pending rows) must form a contiguous zero-based range"
            )
    if set(candidate_rows) != set(all_reference_slots):
        missing = sorted(set(all_reference_slots) - set(candidate_rows))
        extra = sorted(set(candidate_rows) - set(all_reference_slots))
        raise ValueError(f"candidate/reference case mismatch; missing={missing}, extra={extra}")
    for case_id, indices in all_reference_slots.items():
        expected_word_count = len(indices)
        for system_name in ("piper", "muse"):
            actual_word_count = len(candidate_rows[case_id][system_name])
            if actual_word_count != expected_word_count:
                raise ValueError(
                    f"{case_id}: {system_name} emitted {actual_word_count} word slots, "
                    f"but the adjudicated reference has {expected_word_count}; "
                    "explicit separators must preserve empty OOV/missing-word slots"
                )

    return {
        "schema": "g2p-reference-report/v1",
        "candidate_file": candidate_tsv.name,
        "reference_file": reference_csv.name,
        "adjudicated_rows": len(adjudicated),
        "systems": [
            score_system(adjudicated, candidate_rows, "piper"),
            score_system(adjudicated, candidate_rows, "muse"),
        ],
        "method": {
            "unit": "Unicode code point after IPA token-boundary flattening; not canonical phoneme tokens",
            "stress_markers": "excluded from codepoint edit rate; raw marker offsets and parallel A1/A2/A3 entries remain in normalized candidate JSON for separate review",
            "punctuation": "removed from IPA words and retained as separate raw-token evidence with word/character offsets; empty word slots are preserved",
            "piper_word_source": "normalized_word_hypotheses_json, not reparsed Rust Debug tokens",
            "codepoint_unit": "all IPA Unicode code points, including length marks, are compared individually; no diacritic/affricate/diphthong segmentation is inferred",
            "multiple_reference_variants": "minimum edit distance among explicitly accepted variants",
            "pending_rows": "excluded from metrics but retained as positional placeholders so later lexical words cannot shift",
            "oov": "non-empty returned IPA codepoint sequence on a reviewed OOV reference is reported as a non-empty-guess rate; this does not imply the G2P engine exposes an OOV classifier",
            "limitation": "this helper is a rough cross-segmentation IPA-text comparison only; do not label its output PER or treat it as a canonical phone metric",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="Piper/Muse diagnostic TSV")
    parser.add_argument("--references", type=Path, required=True, help="Completed human-review CSV")
    parser.add_argument("--output", type=Path, default=Path("g2p-reference-report.json"))
    args = parser.parse_args()

    report = evaluate(args.candidates, args.references)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
