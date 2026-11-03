#!/usr/bin/env python3
"""Fail-closed auditor for the MC Competition DIMACS-like input subset.

This is a semantic reference parser, not a competition solver. It preserves
ambiguity when the selected format profile does not settle projection defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from fractions import Fraction
from typing import Optional
import json
import re


class ParseFailure(ValueError):
    pass


def parse_rational(text: str) -> Fraction:
    if not isinstance(text, str) or not text:
        raise ParseFailure("empty rational")
    if "/" in text:
        if not re.fullmatch(r"[+-]?\d+/[+-]?\d+", text):
            raise ParseFailure(f"invalid fraction syntax: {text}")
        numerator_text, denominator_text = text.split("/", 1)
        denominator = int(denominator_text)
        if denominator <= 0:
            raise ParseFailure("fraction denominator must be positive")
        return Fraction(int(numerator_text), denominator)
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        raise ParseFailure(f"invalid decimal syntax: {text}")
    if "." in text and len(text.lstrip("+-").split(".", 1)[1]) > 9:
        raise ParseFailure("decimal precision exceeds 9 digits")
    return Fraction(text)


def parse_complex(text: str) -> tuple[Fraction, Fraction]:
    """Parse exact a+bi using decimal or rational components."""
    if not isinstance(text, str) or not text:
        raise ParseFailure("empty complex coefficient")
    if not text.endswith("i"):
        return parse_rational(text), Fraction(0)
    body = text[:-1]
    split_at = None
    for index in range(1, len(body)):
        if body[index] in "+-":
            split_at = index
    if split_at is None:
        real_text = "0"
        imag_text = body or "1"
    else:
        real_text = body[:split_at]
        imag_text = body[split_at:]
    if imag_text == "+":
        imag_text = "1"
    elif imag_text == "-":
        imag_text = "-1"
    return parse_rational(real_text), parse_rational(imag_text)


@dataclass
class AuditResult:
    status: str
    mode: Optional[str]
    num_variables: Optional[int]
    num_clauses_declared: Optional[int]
    num_clauses_parsed: int
    projection: list[int]
    weights: dict[str, list[Optional[str]]]
    warnings: list[str]
    errors: list[str]
    unresolved: list[str]
    format_profile: str

    def to_dict(self):
        return asdict(self)


def audit_dimacs(text: str, profile: str = "strict") -> AuditResult:
    """Audit one DIMACS-like MC instance.

    Profiles:
      * strict: no guessed projection default; PMC with no show declaration
        remains ambiguous.
      * mccomp_v1_2: follows the public 2025 format note: no show variables
        under PMC means a decision query, not #PMC.
      * ganak_2026_compat: follows Ganak's current documented no-show default
        (all variables projected), but only when explicitly requested.
    """
    if profile not in {"strict", "mccomp_v1_2", "ganak_2026_compat"}:
        raise ValueError("unknown format profile")

    warnings: list[str] = []
    errors: list[str] = []
    unresolved: list[str] = []
    mode: Optional[str] = None
    n: Optional[int] = None
    m: Optional_int] = None
    header_seen = False
    type_seen = False
    problem_directive_seen = False
    clause_started = False
    clause_buffer: list[int] = []
    clauses: list[tuple[int, ...]] = []
    projection: set[int] = set()
    show_seen = False
    raw_weights: dict[int, Fraction | tuple[Fraction, Fraction]] = {}
    weight_kind: Optional[str] = None

    def fail(message: str):
        errors.append(message)

    lines = text.splitlines()
    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("c"):
            tokens = stripped.split()
            if len(tokens) >= 2 and tokens[0] == "c" and tokens[1] == "t":
                if len(tokens) != 3:
                    fail(f"line {line_no}: malformed c t header")
                    continue
                if type_seen:
                    fail(f"line {line_no}: duplicate c t header")
                    continue
                if problem_directive_seen or clause_started or clauses:
                    fail(f"line {line_no}: c t header occurs after problem-specific data")
                type_seen = True
                candidate = tokens[2].lower()
                aliases = {"wpmc": "pwmc"}
                candidate = aliases.get(candidate, candidate)
                if candidate not in {"mc", "wmc", "pmc", "pwmc", "amc-complex"}:
                    unresolved.append(f"unsupported type header: {tokens[2]}")
                    mode = candidate
                else:
                    mode = candidate
                continue

            if len(tokens) >= 2 and tokens[0] == "c" and tokens[1] == "p":
                problem_directive_seen = True
                if len(tokens) < 4:
                    fail(f"line {line_no}: malformed c p directive")
                    continue
                directive = tokens[2].lower()
                if directive == "show":
                    if tokens[-1] != "0":
                        fail(f"line {line_no}: show directive must terminate with 0")
                        continue
                    values = tokens[3:-1]
                    if not values:
                        fail(f"line {line_no}: empty show directive")
                        continue
                    show_seen = True
                    for token in values:
                        try:
                            variable = int(token)
                        except ValueError:
                            fail(f"line {line_no}: non-integer projection variable {token}")
                            continue
                        if variable <= 0:
                            fail(f"line {line_no}: projection variable must be positive")
                        elif variable in projection:
                            warnings.append(f"line {line_no}: duplicate projection variable {variable}")
                        else:
                            projection.add(variable)
                    continue
                if directive == "weight":
                    if len(tokens) != 6 or tokens[-1] != "0":
                        fail(f"line {line_no}: weight syntax is c p weight literal value 0")
                        continue
                    try:
                        literal = int(tokens[3])
                    except ValueError:
                        fail(f"line {line_no}: invalid weight literal {tokens[3]}")
                        continue
                    if literal == 0:
                        fail(f"line {line_no}: weight literal cannot be zero")
                        continue
                    value_text = tokens[4]
                    complex_mode = ("i" in value_text) or mode == "amc-complex"
                    try:
                        value = parse_complex(value_text) if complex_mode else parse_rational(value_text)
                    except ParseFailure as exc:
                        fail(f"line {line_no}: {exc}")
                        continue
                    kind = "complex" if complex_mode else "rational"
                    if weight_kind is None:
                        weight_kind = kind
                    elif weight_kind != kind:
                        fail(f"line {line_no}: mixed rational and complex coefficient syntax")
                        continue
                    if literal in raw_weights:
                        if raw_weights[literal] != value:
                            fail(f"line {line_no}: conflicting duplicate weight for literal {literal}")
                        else:
                            warnings.append(f"line {line_no}: duplicate identical weight for literal {literal}")
                    else:
                        raw_weights[literal] = value
                    continue
                unresolved.append(f"unsupported problem directive on line {line_no}: {directive}")
                continue

            # Other comment forms (c c, c r, ordinary c) are ignored.
            continue

        tokens = stripped.split()
        if tokens[0] == "p":
            if header_seen:
                fail(f"line {line_no}: duplicate p header")
                continue
            if clauses or clause_buffer:
                fail(f"line {line_no}: p header after clause data")
                continue
            if len(tokens) != 4 or tokens[1] != "cnf":
                fail(f"line {line_no}: expected p cnf n m")
                continue
            try:
                n, m = int(tokens[2]), int(tokens[3])
            except ValueError:
                fail(f"line {line_no}: non-integer variable/clause count")
                continue
            if n < 0 or m < 0:
                fail(f"line {line_no}: negative variable/clause count")
            header_seen = True
            continue

        if not header_seen:
            fail(f"line {line_no}: DIMACS clause encountered before p cnf header")
            continue

        clause_started = True
        for token in tokens:
            try:
                literal = int(token)
            except ValueError:
                fail(f"line {line_no}: invalid clause token {token}")
                continue
            if literal == 0:
                clause = tuple(clause_buffer)
                clauses.append(clause)
                clause_buffer = []
                if len(clauses) > (m if m is not None else 0):
                    fail(f"line {line_no}: more clauses than declared")
            else:
                if n is not None and abs(literal) > n:
                    fail(f"line {line_no}: literal {literal} is outside 1..{n}")
                clause_buffer.append(literal)

    if not header_seen:
        fail("missing p cnf header")
    if clause_buffer:
        fail("unterminated final clause (missing 0)")
    if m is not None and len(clauses) != m:
        fail(f"declared {m} clauses but parsed {len(clauses)}")

    # Validate metadata against variable bounds after reading all lines.
    if n is not None:
        for variable in projection:
            if variable > n:
                fail(f"projection variable {variable} exceeds declared variable count {n}")
        for literal in raw_weights:
            if abs(literal) > n:
                fail(f"weight literal {literal} exceeds declared variable count {n}")

    # Resolve absent type header only in the unambiguous legacy-MC case.
    if mode is None:
        if raw_weights or show_seen or unresolved:
            unresolved.append("no c t mode header despite weighted/projected directives")
        else:
            mode = "mc"
            warnings.append("no c t header: interpreted as legacy unweighted MC")

    normalized_weights: dict[str, list[Optional[str]]] = {}
    if n is not None:
        for variable in range(1, n + 1):
            positive = raw_weights.get(variable)
            negative = raw_weights.get(-variable)
            normalized_weights[str(variable)] = [
                str(positive) if isinstance(positive, Fraction) else (
                    f"{positive[0]}+{positive[1]}i" if positive and positive[1] >= 0
                    else f"{positive[0]}{positive[1]}i" if positive else None
                ),
                str(negative) if isinstance(negative, Fraction) else (
                    f"{negative[0]}+{negative[1]}i" if negative and negative[1] >= 0
                    else f"{negative[0]}{negative[1]}i" if negative else None
                ),
            ]

    if mode == "mc":
        if raw_weights:
            fail("MC mode contradicts weight declarations")
        if show_seen:
            fail("MC mode contradicts projection declarations")

    elif mode == "wmc":
        if show_seen:
            fail("WMC mode contradicts projection declarations")

    elif mode == "pmc":
        if raw_weights:
            fail("PMC mode contradicts weight declarations")
        if not show_seen:
            if profile == "mccomp_v1_2":
                warnings.append("PMC has no show variables: v1.2 profile interprets this as SAT decision")
            elif profile == "ganak_2026_compat":
                if n is not None:
                    projection = set(range(1, n + 1))
                    warnings.append("Ganak compatibility profile treats missing show as all variables projected")
            else:
                unresolved.append("PMC projection set is absent; v1.2 and Ganak defaults differ")
        elif n is not None and projection == set(range(1, n + 1)):
            warnings.append("all variables projected: PMC is semantically equivalent to MC")

    elif mode == "pwmc":
        if not show_seen:
            unresolved.append("PWMC requires an explicit projection set in this auditor")
        elif raw_weights and any(abs(literal) not in projection for literal in raw_weights):
            warnings.append("weights declared for non-projection variables; format profile requires a warning")

    elif mode == "amc-complex":
        if show_seen:
            fail("AMC-complex mode does not accept projection directives in this profile")
        if raw_weights and weight_kind != "complex":
            fail("AMC-complex mode requires explicit complex coefficient syntax")
    elif mode is not None and mode not in {"mc", "wmc", "pmc", "pwmc", "amc-complex"}:
        unresolved.append(f"unsupported mode {mode}")

    if weight_kind == "complex" and mode != "amc-complex":
        fail("complex weights require amc-complex mode")

    # Resolve effective defaults and validate complementary literal weight rules.
    for variable in range(1, (n or 0) + 1):
        positive = raw_weights.get(variable)
        negative = raw_weights.get(-variable)
        if positive is None and negative is None:
            if mode in {"wmc", "pwmc"}:
                normalized_weights[str(variable)] = ["1", "1"]
            elif mode == "amc-complex":
                normalized_weights[str(variable)] = ["1+0i", "1+0i"]
            continue

        if mode == "amc-complex":
            if positive is None or negative is None:
                fail(f"AMC variable {variable} has only one literal weight")
        else:
            def scalar(value):
                return value if isinstance(value, Fraction) else None

            pair = normalized_weights.setdefault(str(variable), [None, None])
            for literal, value, opposite, own_index, opposite_index in (
                (variable, positive, negative, 0, 1),
                (-variable, negative, positive, 1, 0),
            ):
                if value is None:
                    continue
                v = scalar(value)
                if v is None:
                    fail("non-complex mode has complex coefficient")
                    continue
                if v <= 0 and opposite is None:
                    fail(f"non-positive weight for literal {literal} requires opposite literal weight")
                elif 0 < v < 1 and opposite is None:
                    inferred = Fraction(1) - v
                    pair[opposite_index] = str(inferred)
                    warnings.append(
                        f"literal {literal} missing opposite weight; inferred exact complement {inferred}"
                    )
                elif v >= 1 and opposite is None:
                    unresolved.append(
                        f"literal {literal} weight >= 1 has no documented inferred complement"
                    )

        if positive == 0 or negative == 0:
            warnings.append(f"variable {variable} has a zero literal weight")

    if errors:
        status = "invalid"
    elif unresolved:
        status = "ambiguous" if not any(msg.startswith("unsupported") for msg in unresolved) else "unsupported"
    else:
        status = "valid"

    return AuditResult(
        status=status,
        mode=mode,
        num_variables=n,
        num_clauses_declared=m,
        num_clauses_parsed=len(clauses),
        projection=sorted(projection),
        weights=normalized_weights,
        warnings=warnings,
        errors=errors,
        unresolved=unresolved,
        format_profile=profile,
    )


def fixture_tests():
    tests = []

    def add(name, text, expected, profile="strict"):
        got = audit_dimacs(text, profile).status
        tests.append((name, expected, got))
        assert got == expected, (name, expected, got, audit_dimacs(text, profile).to_dict())

    add("plain mc", "p cnf 2 1\n1 -2 0\n", "valid")
    add("explicit mc", "c t mc\np cnf 2 1\n1 -2 0\n", "valid")
    add("legacy header without mode with weights", "p cnf 1 1\nc p weight 1 1/2 0\n1 0\n", "ambiguous")
    add("wmc exact fraction", "c t wmc\np cnf 1 1\nc p weight 1 1/2 0\nc p weight -1 1/2 0\n1 0\n", "valid")
    add("wmc missing positive complement", "c t wmc\np cnf 1 1\nc p weight 1 1/2 0\n1 0\n", "valid")
    add("wmc negative missing opposite", "c t wmc\np cnf 1 1\nc p weight 1 -1/2 0\n1 0\n", "invalid")
    add("wmc complex forbidden", "c t wmc\np cnf 1 0\nc p weight 1 1+2i 0\n", "invalid")
    add("complex AMC", "c t amc-complex\np cnf 1 1\nc p weight 1 1/2+1/3i 0\nc p weight -1 1/4-1/5i 0\n1 0\n", "valid")
    add("AMC only one literal", "c t amc-complex\np cnf 1 0\nc p weight 1 1+2i 0\n", "invalid")
    add("PMC explicit subset", "c t pmc\np cnf 2 1\nc p show 1 0\n1 2 0\n", "valid")
    add("PMC no show strict", "c t pmc\np cnf 2 1\n1 2 0\n", "ambiguous")
    add("PMC no show v1.2 decision", "c t pmc\np cnf 2 1\n1 2 0\n", "valid", "mccomp_v1_2")
    add("PMC no show Ganak compatibility", "c t pmc\np cnf 2 1\n1 2 0\n", "valid", "ganak_2026_compat")
    add("PMC all variables projected", "c t pmc\np cnf 2 0\nc p show 1 2 0\n", "valid")
    add("PMC weights contradiction", "c t pmc\np cnf 1 0\nc p show 1 0\nc p weight 1 1/2 0\n", "invalid")
    add("WMC show contradiction", "c t wmc\np cnf 1 0\nc p show 1 0\n", "invalid")
    add("PWMC explicit set", "c t pwmc\np cnf 2 1\nc p show 1 0\nc p weight 1 1/2 0\nc p weight -1 1/2 0\n1 2 0\n", "valid")
    add("PWMC missing show", "c t pwmc\np cnf 2 1\n1 2 0\n", "ambiguous")
    add("PWMC nonprojection weights", "c t pwmc\np cnf 2 1\nc p show 1 0\nc p weight 2 1/2 0\n1 2 0\n", "valid")
    add("duplicate p header", "p cnf 1 0\np cnf 1 0\n", "invalid")
    add("clause count mismatch", "p cnf 2 2\n1 0\n", "invalid")
    add("literal out of range", "p cnf 1 1\n2 0\n", "invalid")
    add("unterminated clause", "p cnf 1 1\n1\n", "invalid")
    add("duplicate conflicting weight", "c t wmc\np cnf 1 0\nc p weight 1 1/2 0\nc p weight 1 1/3 0\n", "invalid")
    add("weight decimal exact", "c t wmc\np cnf 1 0\nc p weight 1 0.125 0\nc p weight -1 0.875 0\n", "valid")
    add("weight too precise", "c t wmc\np cnf 1 0\nc p weight 1 0.1234567890 0\nc p weight -1 1/2 0\n", "invalid")
    add("type header after directive", "p cnf 1 0\nc p show 1 0\nc t pmc\n", "invalid")
    add("unknown mode", "c t qmc\np cnf 1 0\n", "unsupported")
    add("duplicate projection declaration", "c t pmc\np cnf 2 0\nc p show 1 0\nc p show 1 2 0\n", "valid")
    add("show out of range", "c t pmc\np cnf 1 0\nc p show 2 0\n", "invalid")
    add("complex decimal/fraction", "c t amc-complex\np cnf 1 0\nc p weight 1 -0.25+3/10i 0\nc p weight -1 0.5-0.1i 0\n", "valid")

    add("AMC real-only coefficients promote to Q(i)", "c t amc-complex\np cnf 1 0\nc p weight 1 1/2 0\nc p weight -1 1/3 0\n", "valid")
    add("unknown problem directive", "c t mc\np cnf 1 0\nc p mystery 1 0\n", "unsupported")
    add("fraction negative denominator", "c t wmc\np cnf 1 0\nc p weight 1 1/-2 0\nc p weight -1 1/2 0\n", "invalid")

    return {
        "suite": "ETH-023 MC input semantic contract auditor",
        "tests": len(tests),
        "passed": sum(expected == got for _, expected, got in tests),
        "failed": [{"name": name, "expected": expected, "got": got}
                   for name, expected, got in tests if expected != got],
        "all_passed": all(expected == got for _, expected, got in tests),
    }


def effective_weight_checks():
    inferred = audit_dimacs(
        "c t wmc\np cnf 1 0\nc p weight 1 1/4 0\n", "strict"
    )
    defaulted = audit_dimacs("c t wmc\np cnf 2 0\n", "strict")
    amc_defaulted = audit_dimacs("c t amc-complex\np cnf 1 0\n", "strict")
    assert inferred.status == "valid" and inferred.weights["1"] == ["1/4", "3/4"]
    assert defaulted.weights == {"1": ["1", "1"], "2": ["1", "1"]}
    assert amc_defaulted.weights == {"1": ["1+0i", "1+0i"]}
    return {
        "complement_inference": inferred.weights,
        "wmc_defaults": defaulted.weights,
        "amc_defaults": amc_defaulted.weights,
    }


if __name__ == "__main__":
    result = fixture_tests()
    result["effective_weight_checks"] = effective_weight_checks()
    print(json.dumps(result, indent=2, sort_keys=True))
