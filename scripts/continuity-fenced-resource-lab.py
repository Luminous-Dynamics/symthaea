#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, sqlite3, sys
from typing import Any

TOKEN_DOMAIN = b"symthaea.continuity.fenced-resource-lab-token.v1\0"

class LabError(RuntimeError):
    pass

def deny(code: str):
    raise LabError(code)

def hex32(v: Any, field: str) -> str:
    if not isinstance(v, str) or len(v) != 64 or v.lower() != v:
        deny(f"{field}:invalid")
    try:
        raw = bytes.fromhex(v)
    except ValueError:
        deny(f"{field}:invalid")
    if raw == b"\x00" * 32:
        deny(f"{field}:zero")
    return v

def u64(v: Any, field: str) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or v <= 0 or v > (1 << 64) - 1:
        deny(f"{field}:invalid")
    return v

def token_id(token: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(TOKEN_DOMAIN)
    for field in ("resource_id", "backend_id", "enforcement_profile_id"):
        h.update(bytes.fromhex(token[field]))
    h.update(token["generation"].to_bytes(8, "little"))
    h.update(b"\x01" if token["disposition"] == "permit" else b"\x02")
    reason = (token.get("deny_reason") or "").encode("utf-8")
    h.update(len(reason).to_bytes(2, "little"))
    h.update(reason)
    h.update(bytes.fromhex(token["challenge"]))
    return h.hexdigest()

def validate_token(token: Any) -> dict[str, Any]:
    if not isinstance(token, dict):
        deny("token:not_object")
    keys = {"resource_id", "backend_id", "enforcement_profile_id", "generation", "disposition", "deny_reason", "challenge", "token_id"}
    if set(token) != keys:
        deny("token:field_set")
    for field in ("resource_id", "backend_id", "enforcement_profile_id", "challenge"):
        hex32(token[field], field)
    u64(token["generation"], "generation")
    if token["disposition"] not in ("permit", "deny"):
        deny("token:disposition")
    if token["disposition"] == "permit" and token["deny_reason"] is not None:
        deny("token:permit_reason")
    if token["disposition"] == "deny" and token["deny_reason"] not in ("emergency_stop", "owner_revoked", "policy_deny"):
        deny("token:deny_reason")
    if token_id(token) != token["token_id"]:
        deny("token:id")
    return token

def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=5, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn

def init_db(path: str, resource_id: str, backend_id: str, profile_id: str) -> None:
    for value, field in ((resource_id, "resource_id"), (backend_id, "backend_id"), (profile_id, "enforcement_profile_id")):
        hex32(value, field)
    conn = connect(path)
    try:
        conn.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS resource_state(
          singleton INTEGER PRIMARY KEY CHECK(singleton=1),
          resource_id TEXT NOT NULL,
          backend_id TEXT NOT NULL,
          enforcement_profile_id TEXT NOT NULL,
          current_generation INTEGER NOT NULL,
          disposition TEXT NOT NULL,
          deny_reason TEXT,
          emergency_stop INTEGER NOT NULL,
          current_token_id TEXT,
          value INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS consumed_tokens(
          token_id TEXT PRIMARY KEY,
          generation INTEGER NOT NULL,
          operation_digest TEXT NOT NULL
        );
        """)
        row = conn.execute("SELECT resource_id,backend_id,enforcement_profile_id FROM resource_state WHERE singleton=1").fetchone()
        if row:
            if row != (resource_id, backend_id, profile_id):
                deny("init:identity_mismatch")
            return
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO resource_state VALUES(1,?,?,?,0,'deny','uninitialized',0,NULL,0)", (resource_id, backend_id, profile_id))
        conn.commit()
    finally:
        conn.close()

def issue(path: str, disposition: str, reason: str | None, challenge: str) -> dict[str, Any]:
    hex32(challenge, "challenge")
    if disposition == "permit" and reason is not None:
        deny("issue:permit_reason")
    if disposition == "deny" and reason not in ("emergency_stop", "owner_revoked", "policy_deny"):
        deny("issue:deny_reason")
    conn = connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT resource_id,backend_id,enforcement_profile_id,current_generation,emergency_stop FROM resource_state WHERE singleton=1").fetchone()
        if not row:
            deny("issue:not_initialized")
        resource_id, backend_id, profile_id, generation, stopped = row
        if stopped and generation > 0 and disposition == "permit":
            deny("issue:emergency_stop_sticky")
        new_generation = generation + 1
        new_stopped = 1 if stopped or reason == "emergency_stop" else 0
        token = {
            "resource_id": resource_id,
            "backend_id": backend_id,
            "enforcement_profile_id": profile_id,
            "generation": new_generation,
            "disposition": disposition,
            "deny_reason": reason,
            "challenge": challenge,
            "token_id": "",
        }
        token["token_id"] = token_id(token)
        conn.execute("UPDATE resource_state SET current_generation=?,disposition=?,deny_reason=?,emergency_stop=?,current_token_id=? WHERE singleton=1", (new_generation, disposition, reason, new_stopped, token["token_id"]))
        conn.commit()
        return token
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()

def actuate(path: str, token: dict[str, Any], delta: int, operation_digest: str, crash_before_commit: bool = False, crash_after_commit: bool = False) -> dict[str, Any]:
    token = validate_token(token)
    hex32(operation_digest, "operation_digest")
    if crash_before_commit and crash_after_commit:
        deny("actuate:conflicting_failpoints")
    if not isinstance(delta, int) or isinstance(delta, bool) or delta == 0:
        deny("actuate:delta")
    conn = connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT resource_id,backend_id,enforcement_profile_id,current_generation,disposition,emergency_stop,current_token_id,value FROM resource_state WHERE singleton=1").fetchone()
        resource_id, backend_id, profile_id, current_generation, disposition, stopped, current_token_id, value = row
        if token["resource_id"] != resource_id:
            deny("actuate:resource_mismatch")
        if token["backend_id"] != backend_id:
            deny("actuate:backend_mismatch")
        if token["enforcement_profile_id"] != profile_id:
            deny("actuate:profile_mismatch")
        if token["generation"] != current_generation:
            deny("actuate:stale_generation")
        if token["token_id"] != current_token_id:
            deny("actuate:token_mismatch")
        if stopped:
            deny("actuate:emergency_stop")
        if disposition != "permit" or token["disposition"] != "permit":
            deny("actuate:deny_disposition")
        if conn.execute("SELECT 1 FROM consumed_tokens WHERE token_id=?", (token["token_id"],)).fetchone():
            deny("actuate:replay")
        new_value = value + delta
        conn.execute("UPDATE resource_state SET value=? WHERE singleton=1", (new_value,))
        conn.execute("INSERT INTO consumed_tokens VALUES(?,?,?)", (token["token_id"], token["generation"], operation_digest))
        if crash_before_commit:
            os._exit(91)
        conn.commit()
        if crash_after_commit:
            os._exit(92)
        return {"status": "applied", "generation": current_generation, "value": new_value, "token_id": token["token_id"]}
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()

def snapshot(path: str) -> dict[str, Any]:
    conn = connect(path)
    try:
        row = conn.execute("SELECT resource_id,backend_id,enforcement_profile_id,current_generation,disposition,deny_reason,emergency_stop,current_token_id,value FROM resource_state WHERE singleton=1").fetchone()
        consumed = conn.execute("SELECT token_id,generation,operation_digest FROM consumed_tokens ORDER BY token_id").fetchall()
        return {
            "resource_id": row[0], "backend_id": row[1], "enforcement_profile_id": row[2],
            "current_generation": row[3], "disposition": row[4], "deny_reason": row[5],
            "emergency_stop": bool(row[6]), "current_token_id": row[7], "value": row[8],
            "consumed_tokens": [{"token_id": x[0], "generation": x[1], "operation_digest": x[2]} for x in consumed],
        }
    finally:
        conn.close()

def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    for arg in ("--db", "--resource-id", "--backend-id", "--profile-id"):
        p.add_argument(arg, required=True)
    p = sub.add_parser("issue")
    p.add_argument("--db", required=True); p.add_argument("--disposition", choices=["permit", "deny"], required=True)
    p.add_argument("--reason"); p.add_argument("--challenge", required=True)
    p = sub.add_parser("actuate")
    p.add_argument("--db", required=True); p.add_argument("--token", required=True); p.add_argument("--delta", type=int, required=True)
    p.add_argument("--operation-digest", required=True); p.add_argument("--crash-before-commit", action="store_true"); p.add_argument("--crash-after-commit", action="store_true")
    p = sub.add_parser("snapshot"); p.add_argument("--db", required=True)
    args = parser.parse_args()
    try:
        if args.cmd == "init":
            init_db(args.db, args.resource_id, args.backend_id, args.profile_id); result = {"status": "initialized"}
        elif args.cmd == "issue":
            result = issue(args.db, args.disposition, args.reason, args.challenge)
        elif args.cmd == "actuate":
            with open(args.token, "r", encoding="utf-8") as fh:
                token = json.load(fh)
            result = actuate(args.db, token, args.delta, args.operation_digest, args.crash_before_commit, args.crash_after_commit)
        else:
            result = snapshot(args.db)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except LabError as exc:
        print(f"DENY:{exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
