from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

P = "ll009ag.site01-campaign-policy.v1"
E = "ll009ag.environment-capsule.v1"
M = "ll009ag.evidence-manifest.v2"
A = "ll009ag.semantic-assertion.v1"
PRE = "ll009ag.campaign-preparation-receipt.v1"
FRZ = "ll009ag.campaign-freeze-receipt.v2"
FIN = "ll009ag.campaign-finalization-receipt.v2"

MODES = {"none", "ag_compact_receipt_sha256", "ll009_indent2_receipt_sha256"}
AVAILABILITY = {"required", "optional_diagnostic", "not_yet_available"}
PARENT_DIGESTS = {"file_sha256", "receipt_sha256"}


class CE(RuntimeError):
    pass


def cb(v: Any) -> bytes:
    return (json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def hb(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hf(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def lj(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise CE(f"cannot read JSON {p}: {e}") from e


def wj(p: Path, v: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    data = cb(v)
    with tempfile.NamedTemporaryFile(dir=p.parent, prefix="." + p.name + ".", delete=False) as f:
        t = Path(f.name)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(t, p)


def receipt(v: dict[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in v:
        raise CE("payload already has receipt_sha256")
    o = dict(v)
    o["receipt_sha256"] = hb(cb(v))
    return o


def vr(v: Any, schema: str | None = None, mode: str = "ag_compact_receipt_sha256") -> str:
    if not isinstance(v, dict):
        raise CE("receipt must be object")
    if schema and v.get("schema_version") != schema:
        raise CE(f"expected {schema}, got {v.get('schema_version')!r}")
    got = v.get("receipt_sha256")
    u = dict(v)
    u.pop("receipt_sha256", None)
    if mode == "ag_compact_receipt_sha256":
        canonical = cb(u)
    elif mode == "ll009_indent2_receipt_sha256":
        canonical = (json.dumps(u, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode()
    else:
        raise CE(f"unsupported receipt self-hash mode: {mode}")
    want = hb(canonical)
    if got != want:
        raise CE(f"receipt self-hash mismatch: expected {want}, got {got}")
    return want


def req(v: Any, label: str) -> str:
    if not isinstance(v, str) or not v:
        raise CE(f"{label} must be non-empty string")
    return v


def sha(v: Any, label: str) -> str:
    v = req(v, label)
    if len(v) != 64:
        raise CE(f"{label} must be SHA-256")
    try:
        int(v, 16)
    except ValueError as e:
        raise CE(f"{label} must be hex") from e
    return v.lower()


def rel(v: Any, label: str) -> str:
    v = req(v, label)
    if "\\" in v or v.startswith("/") or any(x in ("", ".", "..") for x in v.split("/")):
        raise CE(f"unsafe {label}: {v!r}")
    return v


def file(root: Path, r: str, label: str) -> Path:
    r = rel(r, label)
    root = root.resolve()
    p = root
    for x in r.split("/"):
        p = p / x
        if p.is_symlink():
            raise CE(f"{label} traverses symlink: {p}")
    try:
        q = p.resolve(strict=True)
    except FileNotFoundError as e:
        raise CE(f"{label} missing: {p}") from e
    try:
        q.relative_to(root)
    except ValueError as e:
        raise CE(f"{label} escapes root") from e
    if not stat.S_ISREG(q.stat().st_mode):
        raise CE(f"{label} not regular file")
    return q


def closed(root: Path) -> set[str]:
    root = root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise CE("evidence root must be ordinary directory")
    out: set[str] = set()
    for dp, dn, fn in os.walk(root, followlinks=False):
        d = Path(dp)
        for n in dn:
            p = d / n
            if p.is_symlink() or not stat.S_ISDIR(p.stat().st_mode):
                raise CE(f"symlink/special directory: {p}")
        for n in fn:
            p = d / n
            if p.is_symlink() or not stat.S_ISREG(p.stat().st_mode):
                raise CE(f"symlink/special evidence: {p}")
            out.add(p.relative_to(root).as_posix())
    return out


def vp(v: Any) -> dict[str, Any]:
    if not isinstance(v, dict) or v.get("schema_version") != P:
        raise CE(f"policy schema must be {P}")
    req(v.get("study_id"), "study_id")
    prof = v.get("required_environment_profiles")
    if prof != ["acquisition", "gis", "analysis"]:
        raise CE("environment profiles must be exactly acquisition,gis,analysis")
    ec = v.get("environment_contracts")
    if not isinstance(ec, dict) or set(ec) != set(prof):
        raise CE("environment_contracts must exactly cover required profiles")
    for k in prof:
        c = ec[k]
        if not isinstance(c, dict):
            raise CE(f"invalid environment contract {k}")
        for field in ("required_packages", "required_libraries", "required_environment_variables"):
            z = c.get(field)
            if not isinstance(z, list) or len(z) != len(set(z)):
                raise CE(f"{k} {field} must be unique list")
            [req(x, f"{k} {field}") for x in z]
    pf = v.get("protected_files")
    if not isinstance(pf, list) or len(pf) != len(set(pf)):
        raise CE("protected_files must be unique list")
    [rel(x, "protected file") for x in pf]
    stages = v.get("stage_plan")
    auth = v.get("network_authorized_stage_ids")
    if not isinstance(stages, list) or not isinstance(auth, list):
        raise CE("stage plan/network authority must be lists")
    ids = []
    for x in stages:
        if not isinstance(x, dict):
            raise CE("stage must be object")
        i = req(x.get("id"), "stage id")
        ids.append(i)
        if x.get("environment_profile") not in prof or not isinstance(x.get("may_access_network"), bool):
            raise CE(f"invalid stage {i}")
        if x["may_access_network"] != (i in auth):
            raise CE(f"network authority mismatch for {i}")
    if len(ids) != len(set(ids)):
        raise CE("duplicate stage id")
    order = v.get("classification_order")
    ceil = v.get("classification_ceiling")
    rules = v.get("semantic_rules")
    if (
        not isinstance(order, list)
        or not order
        or len(order) != len(set(order))
        or ceil not in order
        or not isinstance(rules, dict)
        or set(rules) != set(order)
    ):
        raise CE("invalid classification policy")
    for c in order:
        rr = rules[c]
        if not isinstance(rr, dict) or not isinstance(rr.get("enabled"), bool) or not isinstance(rr.get("requires_all_artifact_ids"), list):
            raise CE(f"invalid semantic rule {c}")
        z = [req(x, f"{c} required artifact") for x in rr["requires_all_artifact_ids"]]
        if len(z) != len(set(z)):
            raise CE(f"duplicate required artifact in {c}")
    for c in order[order.index(ceil) + 1 :]:
        if rules[c]["enabled"]:
            raise CE(f"class above ceiling enabled: {c}")
    return v


def envmap(vals: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for x in vals:
        if "=" not in x:
            raise CE("--env must be PROFILE=relative/path.json")
        k, p = x.split("=", 1)
        k = req(k, "profile")
        p = rel(p, "environment path")
        if k in out:
            raise CE(f"duplicate environment profile {k}")
        out[k] = p
    return out


def _env_capsule(v: Any, profile: str, contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(v, dict) or v.get("schema_version") != E or v.get("profile") != profile:
        raise CE(f"invalid {profile} environment capsule identity")
    r = v.get("runtime")
    if not isinstance(r, dict) or set(r) != {"python", "platform", "packages", "libraries", "environment"}:
        raise CE(f"invalid {profile} runtime surface")
    py = r["python"]
    plat = r["platform"]
    packages = r["packages"]
    libraries = r["libraries"]
    env = r["environment"]
    if not all(isinstance(x, dict) for x in (py, plat, packages, libraries, env)):
        raise CE(f"invalid {profile} runtime sections")
    req(py.get("implementation"), f"{profile} python implementation")
    req(py.get("version"), f"{profile} python version")
    sha(py.get("executable_sha256"), f"{profile} python executable sha256")
    req(plat.get("system"), f"{profile} platform system")
    req(plat.get("machine"), f"{profile} platform machine")
    for name, value in packages.items():
        req(name, f"{profile} package name")
        req(value, f"{profile} package version")
    for name, value in libraries.items():
        req(name, f"{profile} library name")
        req(value, f"{profile} library version")
    for name, value in env.items():
        req(name, f"{profile} environment variable name")
        if value is not None and not isinstance(value, str):
            raise CE(f"{profile} environment variable {name} must be string or null")
    missing_packages = set(contract["required_packages"]) - set(packages)
    missing_libraries = set(contract["required_libraries"]) - set(libraries)
    missing_env = set(contract["required_environment_variables"]) - set(env)
    if missing_packages or missing_libraries or missing_env:
        raise CE(
            f"{profile} environment capsule incomplete packages={sorted(missing_packages)} "
            f"libraries={sorted(missing_libraries)} env={sorted(missing_env)}"
        )
    return v


def envbind(policy: dict[str, Any], root: Path, m: dict[str, str]) -> list[dict[str, Any]]:
    rr = policy["required_environment_profiles"]
    if set(m) != set(rr):
        raise CE(f"environment bindings must exactly be {rr}")
    out = []
    for k in rr:
        p = file(root, m[k], f"{k} environment")
        v = _env_capsule(lj(p), k, policy["environment_contracts"][k])
        out.append(
            {
                "profile": k,
                "path": m[k],
                "sha256": hf(p),
                "byte_count": p.stat().st_size,
                "canonical_payload_sha256": hb(cb(v)),
            }
        )
    return out


def pbind(policy: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    return [
        {"path": r, "sha256": hf(p := file(root, r, f"protected {r}")), "byte_count": p.stat().st_size}
        for r in policy["protected_files"]
    ]


def json_pointer(v: Any, pointer: str, label: str) -> Any:
    """Resolve a strict RFC 6901 JSON pointer without implicit coercion."""
    if pointer == "":
        return v
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise CE(f"{label} must be RFC6901 JSON pointer")
    cur = v
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        i = 0
        while i < len(raw):
            if raw[i] == "~":
                if i + 1 >= len(raw) or raw[i + 1] not in "01":
                    raise CE(f"malformed JSON pointer escape in {label}: {pointer!r}")
                i += 2
            else:
                i += 1
        if isinstance(cur, dict):
            if token not in cur:
                raise CE(f"{label} missing JSON pointer component {token!r}")
            cur = cur[token]
        elif isinstance(cur, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise CE(f"{label} invalid array index {token!r}")
            idx = int(token)
            if idx >= len(cur):
                raise CE(f"{label} array index out of range {idx}")
            cur = cur[idx]
        else:
            raise CE(f"{label} traverses non-container at {token!r}")
    return cur
