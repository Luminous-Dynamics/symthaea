#!/usr/bin/env python3
"""HAK-019a retained provider-response envelope and Link projection replay.

Audit/evidence tooling only. This retains exact body bytes plus selected HTTP
header field values and deterministically projects an RFC-8288-compatible
`rel=next` relation from the retained Link header representation. It does not
authenticate the provider, retain raw HTTP wire bytes, prove collection
exhaustion, establish temporal snapshot consistency, or grant runtime authority.
"""
from __future__ import annotations

import base64
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import hak_canonical_json as canonical

ENVELOPE_SCHEMA = "hak.provider-response-envelope.v1"
ENVELOPE_DOMAIN = ENVELOPE_SCHEMA
PROJECTION_SCHEMA = "hak.link-pagination-projection-receipt.v1"
PROJECTION_DOMAIN = PROJECTION_SCHEMA
POLICY_SCHEMA = "hak.header-projection-policy.v1"
PROVIDER = "github-rest"
PROFILE = {
    "profile_id": "hak.canonical-json.v1",
    "artifact_ref": (
        "git:Luminous-Dynamics/symthaea@cf440ce4f813bb30a6b1948e5caac96feda10610:"
        "docs/architecture/hak/canonical-json-v1.profile.json"
    ),
    "raw_sha256": "sha256:f76152db3cce567bb4d24ca46b5ed94d95db698daa8882a6fcdb38828d40e830",
}
PARSER = {"id": "hak.rfc8288-link-pagination.strict", "version": 1}
HEADER_REPRESENTATION = "RetainedFieldValuesV1"
WIRE_REPRESENTATION = "NotRetained"
PROVIDER_AUTHENTICATION = "NotEstablished"
PROJECTION_VERIFICATION = "VerifiedAgainstRetainedEnvelope"
EXHAUSTION_VERIFICATION = "NotEstablished"
_SOURCE_SNAPSHOT = Path(__file__).read_bytes()

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_REF_RE = re.compile(r"^git:[^@]+@[0-9a-f]{40}:.+$")
TOKEN_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
REGISTERED_REL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.-]*$")


class ProviderResponseEnvelopeError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProviderResponseEnvelopeError(message)


def _text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{field} must be non-empty")
    return value


def _ows_strip(value: str) -> str:
    """HTTP optional whitespace is SP / HTAB only."""
    return value.strip(" \t")


def _safe_int(value: Any, field: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{field} must be integer")
    _require(value >= minimum, f"{field} below minimum")
    if maximum is not None:
        _require(value <= maximum, f"{field} above maximum")
    _require(value <= canonical.MAX_SAFE_INTEGER, f"{field} exceeds canonical safe integer range")
    return value


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _parse_time(value: str, field: str) -> datetime:
    _text(value, field)
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise ProviderResponseEnvelopeError(f"{field} must be ISO-8601") from exc
    _require(parsed.tzinfo is not None, f"{field} must include timezone")
    return parsed


def _profile() -> dict[str, str]:
    return dict(PROFILE)


def _canonical_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _decode_canonical_base64(value: Any, field: str) -> bytes:
    _require(isinstance(value, str), f"{field} must be base64 string")
    try:
        decoded = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ProviderResponseEnvelopeError(f"{field} invalid base64") from exc
    _require(_canonical_base64(decoded) == value, f"{field} must use canonical padded base64")
    return decoded


def _validate_url(value: Any, field: str) -> str:
    value = _text(value, field)
    parsed = urlparse(value)
    _require(parsed.scheme == "https" and bool(parsed.netloc), f"{field} must be absolute https URL")
    _require(parsed.username is None and parsed.password is None, f"{field} must not contain userinfo")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ProviderResponseEnvelopeError(f"{field} has invalid port") from exc
    return value


def _validate_continuation_target(base_url: str, resolved: str) -> None:
    """HAK-019a GitHub list profile allows pagination query drift only."""
    base = urlparse(_validate_url(base_url, "continuation base URL"))
    target = urlparse(_validate_url(resolved, "resolved Link target"))
    _require(target.fragment == "", "continuation target must not contain fragment")
    _require(target.scheme.lower() == base.scheme.lower(), "continuation target scheme changed")
    _require(target.hostname == base.hostname, "continuation target host changed")
    _require(target.port == base.port, "continuation target port changed")
    _require(target.path == base.path, "continuation target API path changed")


def _validate_headers(headers: Any) -> list[dict[str, str]]:
    _require(isinstance(headers, list), "response.headers must be array")
    result: list[dict[str, str]] = []
    for idx, raw in enumerate(headers):
        _require(isinstance(raw, dict), f"response.headers[{idx}] must be object")
        _require(set(raw) == {"name", "value"}, f"response.headers[{idx}] fields invalid")
        name = _text(raw.get("name"), f"response.headers[{idx}].name")
        value = raw.get("value")
        _require(TOKEN_RE.fullmatch(name) is not None, f"response.headers[{idx}].name invalid HTTP token")
        _require(isinstance(value, str), f"response.headers[{idx}].value must be string")
        _require("\r" not in value and "\n" not in value, f"response.headers[{idx}].value must not contain CR/LF")
        result.append({"name": name, "value": value})
    return result


def compute_headers_digest(headers: list[dict[str, str]]) -> str:
    return canonical.hak_sha256("hak.retained-http-header-fields.v1", headers)


def compute_envelope_digest(envelope: dict[str, Any]) -> str:
    payload = {k: v for k, v in envelope.items() if k != "envelope_digest"}
    return canonical.hak_sha256(ENVELOPE_DOMAIN, payload)


def make_envelope(
    *, resource_kind: str, request_ref: str, request_url: str, response_ref: str,
    status: int, headers: list[dict[str, str]], body_bytes: bytes,
    request_not_before: str, response_not_after: str,
) -> dict[str, Any]:
    _text(resource_kind, "resource_kind")
    _text(request_ref, "request_ref")
    _validate_url(request_url, "request_url")
    _text(response_ref, "response_ref")
    _safe_int(status, "status", minimum=100, maximum=599)
    _require(isinstance(body_bytes, bytes), "body_bytes must be bytes")
    normalized_headers = _validate_headers(headers)
    start = _parse_time(request_not_before, "request_not_before")
    end = _parse_time(response_not_after, "response_not_after")
    _require(start <= end, "observation bounds inverted")

    envelope: dict[str, Any] = {
        "schema_version": ENVELOPE_SCHEMA,
        "canonicalization_profile": _profile(),
        "digest_domain": ENVELOPE_DOMAIN,
        "provider": PROVIDER,
        "resource_kind": resource_kind,
        "request": {"request_ref": request_ref, "method": "GET", "url": request_url},
        "response": {
            "response_ref": response_ref,
            "status": status,
            "headers": normalized_headers,
            "headers_digest": compute_headers_digest(normalized_headers),
            "body_base64": _canonical_base64(body_bytes),
            "body_length": len(body_bytes),
            "body_raw_sha256": _sha256_bytes(body_bytes),
        },
        "observation_bounds": {
            "request_not_before": request_not_before,
            "response_not_after": response_not_after,
        },
        "retention": {
            "body_bytes_retained": True,
            "header_representation": HEADER_REPRESENTATION,
            "raw_http_wire_representation": WIRE_REPRESENTATION,
        },
        "provider_authentication": PROVIDER_AUTHENTICATION,
    }
    envelope["envelope_digest"] = compute_envelope_digest(envelope)
    validate_envelope(envelope)
    return envelope


def validate_envelope(envelope: dict[str, Any]) -> None:
    _require(isinstance(envelope, dict), "envelope must be object")
    _require(set(envelope) == {
        "schema_version", "canonicalization_profile", "digest_domain", "provider", "resource_kind",
        "request", "response", "observation_bounds", "retention", "provider_authentication", "envelope_digest",
    }, "envelope root fields invalid")
    _require(envelope.get("schema_version") == ENVELOPE_SCHEMA, "envelope schema_version mismatch")
    _require(envelope.get("canonicalization_profile") == _profile(), "canonicalization profile mismatch")
    _require(envelope.get("digest_domain") == ENVELOPE_DOMAIN, "envelope digest domain mismatch")
    _require(envelope.get("provider") == PROVIDER, "provider mismatch")
    _text(envelope.get("resource_kind"), "resource_kind")

    request = envelope.get("request")
    _require(isinstance(request, dict) and set(request) == {"request_ref", "method", "url"}, "request fields invalid")
    _text(request.get("request_ref"), "request.request_ref")
    _require(request.get("method") == "GET", "request.method must be GET")
    _validate_url(request.get("url"), "request.url")

    response = envelope.get("response")
    _require(isinstance(response, dict) and set(response) == {
        "response_ref", "status", "headers", "headers_digest", "body_base64", "body_length", "body_raw_sha256"
    }, "response fields invalid")
    _text(response.get("response_ref"), "response.response_ref")
    _safe_int(response.get("status"), "response.status", minimum=100, maximum=599)
    headers = _validate_headers(response.get("headers"))
    _require(response.get("headers_digest") == compute_headers_digest(headers), "headers_digest mismatch")
    body = _decode_canonical_base64(response.get("body_base64"), "response.body_base64")
    _require(response.get("body_length") == len(body), "body_length mismatch")
    _require(response.get("body_raw_sha256") == _sha256_bytes(body), "body_raw_sha256 mismatch")

    bounds = envelope.get("observation_bounds")
    _require(isinstance(bounds, dict) and set(bounds) == {"request_not_before", "response_not_after"},
             "observation_bounds fields invalid")
    start = _parse_time(bounds.get("request_not_before"), "observation_bounds.request_not_before")
    end = _parse_time(bounds.get("response_not_after"), "observation_bounds.response_not_after")
    _require(start <= end, "observation bounds inverted")

    _require(envelope.get("retention") == {
        "body_bytes_retained": True,
        "header_representation": HEADER_REPRESENTATION,
        "raw_http_wire_representation": WIRE_REPRESENTATION,
    }, "retention semantics drifted")
    _require(envelope.get("provider_authentication") == PROVIDER_AUTHENTICATION,
             "envelope does not establish provider authentication")
    digest = envelope.get("envelope_digest")
    _require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None, "envelope_digest invalid")
    _require(digest == compute_envelope_digest(envelope), "envelope_digest mismatch")


def retained_body_bytes(envelope: dict[str, Any]) -> bytes:
    validate_envelope(envelope)
    return _decode_canonical_base64(envelope["response"]["body_base64"], "response.body_base64")


def compute_projection_policy_digest(policy: dict[str, Any]) -> str:
    payload = {k: v for k, v in policy.items() if k != "policy_digest"}
    return canonical.hak_sha256(POLICY_SCHEMA, payload)


def validate_projection_policy(policy: dict[str, Any]) -> None:
    _require(isinstance(policy, dict), "projection policy must be object")
    expected_fields = {
        "schema_version", "policy_id", "provider", "request_method", "accepted_response_statuses",
        "header_name", "relation", "parser_profile", "header_field_semantics",
        "duplicate_parameter_policy", "duplicate_relation_policy", "duplicate_next_policy",
        "relative_target_resolution", "continuation_target_policy", "missing_link_semantics",
        "wire_representation_requirement", "provider_authentication", "policy_digest",
    }
    _require(set(policy) == expected_fields, "projection policy fields invalid")
    _require(policy.get("schema_version") == POLICY_SCHEMA, "projection policy schema mismatch")
    _require(policy.get("policy_id") == "hak019a-github-rest-link-pagination-v1", "projection policy id mismatch")
    _require(policy.get("provider") == PROVIDER, "projection policy provider mismatch")
    _require(policy.get("request_method") == "GET", "projection policy request_method mismatch")
    _require(policy.get("accepted_response_statuses") == [200], "projection policy status contract mismatch")
    _require(policy.get("header_name") == "link", "projection policy header_name mismatch")
    _require(policy.get("relation") == "next", "projection policy relation mismatch")
    _require(policy.get("parser_profile") == PARSER, "projection parser profile mismatch")
    _require(policy.get("header_field_semantics") == "ParseEachRetainedFieldValueAndUnionLinks",
             "header field semantics mismatch")
    for field in ("duplicate_parameter_policy", "duplicate_relation_policy", "duplicate_next_policy"):
        _require(policy.get(field) == "Reject", f"projection policy {field} mismatch")
    _require(policy.get("relative_target_resolution") == "ResolveAgainstExactRequestUrl",
             "relative-target policy mismatch")
    _require(policy.get("continuation_target_policy") == "SameSchemeAuthorityAndPathNoFragment",
             "continuation-target policy mismatch")
    _require(policy.get("missing_link_semantics") == "NoNextRelationObserved",
             "missing-Link semantics mismatch")
    _require(policy.get("wire_representation_requirement") == "RetainedFieldValuesSufficientForProjection",
             "wire representation policy mismatch")
    _require(policy.get("provider_authentication") == PROVIDER_AUTHENTICATION,
             "projection policy cannot establish provider authentication")
    digest = policy.get("policy_digest")
    _require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None, "policy_digest invalid")
    _require(digest == compute_projection_policy_digest(policy), "projection policy digest mismatch")


def _split_outside(value: str, delimiter: str, *, track_angle: bool) -> list[str]:
    parts: list[str] = []
    start = 0
    quoted = False
    escaped = False
    angle = False
    for idx, ch in enumerate(value):
        if quoted:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
            continue
        if ch == '"':
            quoted = True
        elif track_angle and ch == "<":
            _require(not angle, "nested '<' in Link header")
            angle = True
        elif track_angle and ch == ">":
            _require(angle, "unmatched '>' in Link header")
            angle = False
        elif ch == delimiter and not angle:
            parts.append(_ows_strip(value[start:idx]))
            start = idx + 1
    _require(not quoted and not escaped and not angle, "unterminated quoted-string or URI reference in Link header")
    parts.append(_ows_strip(value[start:]))
    _require(all(parts), "empty Link header component")
    return parts


def _unquote(value: str) -> str:
    if not value.startswith('"'):
        _require(TOKEN_RE.fullmatch(value) is not None, "Link parameter token value invalid")
        return value
    _require(len(value) >= 2 and value.endswith('"'), "unterminated quoted Link parameter")
    out: list[str] = []
    escaped = False
    for ch in value[1:-1]:
        if escaped:
            out.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            raise ProviderResponseEnvelopeError("unescaped quote in Link parameter")
        else:
            out.append(ch)
    _require(not escaped, "trailing escape in Link parameter")
    return "".join(out)


def _normalize_relation(value: str) -> str:
    _require(value, "empty relation type")
    if REGISTERED_REL_RE.fullmatch(value):
        return value.lower()
    parsed = urlparse(value)
    _require(bool(parsed.scheme), "extension relation type must be absolute URI")
    return value


def _parse_link_value(raw: str, base_url: str) -> dict[str, Any]:
    raw = _ows_strip(raw)
    _require(raw.startswith("<"), "Link value must begin with '<'")
    end = raw.find(">")
    _require(end > 1, "Link target URI reference missing")
    target_ref = raw[1:end]
    remainder = _ows_strip(raw[end + 1:])
    params: dict[str, str | None] = {}
    if remainder:
        _require(remainder.startswith(";"), "Link parameters must follow ';'")
        for piece in _split_outside(remainder[1:], ";", track_angle=False):
            if "=" in piece:
                name, value = piece.split("=", 1)
                name = _ows_strip(name).lower()
                value = _ows_strip(value)
                _require(TOKEN_RE.fullmatch(name) is not None, "Link parameter name invalid")
                _require(name not in params, f"duplicate Link parameter: {name}")
                params[name] = _unquote(value)
            else:
                name = _ows_strip(piece).lower()
                _require(TOKEN_RE.fullmatch(name) is not None, "Link parameter name invalid")
                _require(name not in params, f"duplicate Link parameter: {name}")
                params[name] = None

    relations: list[str] = []
    if "rel" in params:
        rel_value = params["rel"]
        _require(isinstance(rel_value, str) and rel_value, "rel parameter requires value")
        _require(rel_value == rel_value.strip(" "), "rel value must not have leading/trailing SP")
        _require(all(ch == " " or not ch.isspace() for ch in rel_value),
                 "relation types must be separated by SP only")
        relation_items = [item for item in rel_value.split(" ") if item]
        _require(relation_items, "rel parameter requires at least one relation type")
        for item in relation_items:
            normalized = _normalize_relation(item)
            _require(normalized not in relations, f"duplicate relation type: {normalized}")
            relations.append(normalized)

    resolved = urljoin(base_url, target_ref)
    _validate_continuation_target(base_url, resolved)
    return {"target_ref": target_ref, "target": resolved, "relations": relations}


def parse_retained_link_headers(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    validate_envelope(envelope)
    values = [item["value"] for item in envelope["response"]["headers"] if item["name"].lower() == "link"]
    links: list[dict[str, Any]] = []
    base_url = envelope["request"]["url"]
    for value in values:
        for raw_link in _split_outside(value, ",", track_angle=True):
            links.append(_parse_link_value(raw_link, base_url))
    return links


def current_parser_bytes() -> bytes:
    return _SOURCE_SNAPSHOT


def compute_projection_receipt_digest(receipt: dict[str, Any]) -> str:
    payload = {k: v for k, v in receipt.items() if k != "receipt_digest"}
    return canonical.hak_sha256(PROJECTION_DOMAIN, payload)


def _project_without_recursive_validation(
    envelope: dict[str, Any], policy: dict[str, Any], *, policy_artifact_ref: str,
    parser_ref: str, parser_bytes: bytes,
) -> dict[str, Any]:
    validate_envelope(envelope)
    validate_projection_policy(policy)
    _require(envelope["provider"] == policy["provider"], "envelope/provider policy mismatch")
    _require(envelope["request"]["method"] == policy["request_method"], "request method outside policy")
    _require(envelope["response"]["status"] in policy["accepted_response_statuses"],
             "response status outside projection policy")
    _require(isinstance(policy_artifact_ref, str) and GIT_REF_RE.fullmatch(policy_artifact_ref) is not None,
             "policy_artifact_ref must be exact git ref")
    _require(isinstance(parser_ref, str) and GIT_REF_RE.fullmatch(parser_ref) is not None,
             "parser_ref must be exact git ref")
    _require(isinstance(parser_bytes, bytes) and parser_bytes == current_parser_bytes(),
             "parser_bytes must equal import-time HAK-019a source snapshot")

    values = [
        item["value"] for item in envelope["response"]["headers"]
        if item["name"].lower() == policy["header_name"]
    ]
    links = parse_retained_link_headers(envelope)
    next_links = [link for link in links if policy["relation"] in link["relations"]]
    _require(len(next_links) <= 1, "ambiguous multiple rel=next Link targets")
    next_relation = (
        {"state": "Present", "target_ref": next_links[0]["target_ref"], "target": next_links[0]["target"]}
        if next_links else {"state": "Absent", "target_ref": None, "target": None}
    )

    out: dict[str, Any] = {
        "schema_version": PROJECTION_SCHEMA,
        "canonicalization_profile": _profile(),
        "digest_domain": PROJECTION_DOMAIN,
        "envelope": {
            "envelope_digest": envelope["envelope_digest"],
            "response_ref": envelope["response"]["response_ref"],
            "headers_digest": envelope["response"]["headers_digest"],
        },
        "policy": {
            "policy_id": policy["policy_id"],
            "policy_digest": policy["policy_digest"],
            "artifact_ref": policy_artifact_ref,
        },
        "parser": {**PARSER, "artifact_ref": parser_ref, "content_digest": _sha256_bytes(parser_bytes)},
        "projection_scope": "RetainedLinkHeaderFieldValuesOnly",
        "projection_verification": PROJECTION_VERIFICATION,
        "provider_authentication": PROVIDER_AUTHENTICATION,
        "http_wire_verification": "NotEstablished",
        "link_header_field_count": len(values),
        "parsed_links": links,
        "next_relation": next_relation,
        "exhaustion_verification": EXHAUSTION_VERIFICATION,
    }
    out["receipt_digest"] = compute_projection_receipt_digest(out)
    return out


def project_next_relation(
    envelope: dict[str, Any], policy: dict[str, Any], *, policy_artifact_ref: str,
    parser_ref: str, parser_bytes: bytes,
) -> dict[str, Any]:
    receipt = _project_without_recursive_validation(
        envelope, policy,
        policy_artifact_ref=policy_artifact_ref,
        parser_ref=parser_ref,
        parser_bytes=parser_bytes,
    )
    validate_projection_receipt(
        receipt, envelope, policy,
        policy_artifact_ref=policy_artifact_ref,
        parser_ref=parser_ref,
        parser_bytes=parser_bytes,
    )
    return receipt


def validate_projection_receipt(
    receipt: dict[str, Any], envelope: dict[str, Any], policy: dict[str, Any], *,
    policy_artifact_ref: str, parser_ref: str, parser_bytes: bytes,
) -> None:
    _require(isinstance(receipt, dict), "projection receipt must be object")
    digest = receipt.get("receipt_digest")
    _require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None, "receipt_digest invalid")
    _require(digest == compute_projection_receipt_digest(receipt), "receipt_digest mismatch")
    expected = _project_without_recursive_validation(
        envelope, policy,
        policy_artifact_ref=policy_artifact_ref,
        parser_ref=parser_ref,
        parser_bytes=parser_bytes,
    )
    _require(receipt == expected,
             "projection receipt is not deterministic replay of exact envelope/policy/parser inputs")
