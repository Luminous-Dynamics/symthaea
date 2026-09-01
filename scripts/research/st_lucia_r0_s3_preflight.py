#!/usr/bin/env python3
"""HEAD-only preflight for the frozen St. Lucia R0 Sentinel asset plan.

Consumes the reviewed 29-object acquisition plan, performs only authenticated
`aws s3api head-object` calls, and emits a content-addressed metadata receipt.
No object body is requested by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

Json = dict[str, Any]
SCHEMA = "symthaea-st-lucia-r0-s3-preflight/v1"
TOOL_VERSION = "1.0.0"
EXPECTED_PLAN_FILE_SHA256 = "fa5dffc399fd0c120cdc59b479b0952862561d7db0ea10528623d8809133dff2"
EXPECTED_PLAN_INTERNAL_SHA256 = "53a12535acc9f02bf62d78c54c3a6b0631d6ba69e22eeb187e2ad20ecb330c46"
EXPECTED_PLANNER_HEAD = "90b557dfa69f0b9d228b8bc02a5907b5b8e58346"
EXPECTED_ENDPOINT = "https://eodata.dataspace.copernicus.eu/"
EXPECTED_BUCKET = "eodata"
EXPECTED_ASSET_COUNT = 29

class PreflightError(RuntimeError):
    pass

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())

def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def load_json(path: Path) -> Json:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreflightError("plan must be a JSON object")
    return value

def validate_plan(plan: Json) -> list[Json]:
    if plan.get("schema") != "symthaea-st-lucia-r0-asset-plan/v1":
        raise PreflightError("unexpected asset-plan schema")
    if plan.get("tool_version") != "1.2.0":
        raise PreflightError("unexpected asset-plan tool version")
    if plan.get("plan_sha256") != EXPECTED_PLAN_INTERNAL_SHA256:
        raise PreflightError("asset-plan internal digest mismatch")
    if plan.get("approved_s3_endpoint") != EXPECTED_ENDPOINT:
        raise PreflightError("asset-plan endpoint mismatch")
    if plan.get("approved_s3_bucket") != EXPECTED_BUCKET:
        raise PreflightError("asset-plan bucket mismatch")
    assets = plan.get("assets")
    if not isinstance(assets, list) or len(assets) != EXPECTED_ASSET_COUNT:
        raise PreflightError("asset-plan must contain exactly 29 assets")
    seen: set[tuple[str, str, str]] = set()
    for row in assets:
        if not isinstance(row, dict):
            raise PreflightError("invalid asset row")
        if row.get("access_method") != "s3":
            raise PreflightError("all frozen assets must use s3")
        if row.get("s3_endpoint") != EXPECTED_ENDPOINT or row.get("s3_bucket") != EXPECTED_BUCKET:
            raise PreflightError("asset endpoint/bucket mismatch")
        key = row.get("s3_key")
        href = row.get("stac_href")
        if not isinstance(key, str) or not key or not isinstance(href, str):
            raise PreflightError("invalid frozen S3 locator")
        if href != f"s3://{EXPECTED_BUCKET}/{key}":
            raise PreflightError("S3 URI/key decomposition mismatch")
        ident = (str(row.get("item_id")), str(row.get("asset_key")), key)
        if ident in seen:
            raise PreflightError("duplicate frozen asset row")
        seen.add(ident)
    return assets

def aws_head_command(asset: Json) -> list[str]:
    return [
        "aws", "s3api", "head-object",
        "--endpoint-url", EXPECTED_ENDPOINT,
        "--region", "default",
        "--bucket", EXPECTED_BUCKET,
        "--key", asset["s3_key"],
        "--output", "json",
        "--no-cli-pager",
    ]

def normalize_head(value: Json) -> Json:
    # Deliberately omit arbitrary user metadata and request IDs.
    allowed = (
        "ContentLength", "ETag", "LastModified", "ContentType", "VersionId",
        "ChecksumSHA256", "ChecksumCRC32", "ChecksumCRC32C", "ChecksumSHA1",
        "AcceptRanges", "CacheControl", "ContentDisposition", "ContentEncoding",
        "ContentLanguage", "Expires", "StorageClass",
    )
    out: Json = {}
    for key in allowed:
        if key in value and value[key] is not None:
            out[key] = value[key]
    if not isinstance(out.get("ContentLength"), int) or out["ContentLength"] < 0:
        raise PreflightError("HEAD response lacks valid ContentLength")
    return out

def run_head(asset: Json, env: dict[str, str]) -> Json:
    proc = subprocess.run(aws_head_command(asset), capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return {
            "status": "failed",
            "returncode": proc.returncode,
            "stderr_sha256": sha256_bytes(proc.stderr.encode("utf-8", errors="replace")),
        }
    try:
        value = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise PreflightError(f"aws head-object returned invalid JSON for {asset['asset_key']}") from exc
    if not isinstance(value, dict):
        raise PreflightError("aws head-object response must be an object")
    return {"status": "available", "head": normalize_head(value)}

def build_receipt(plan_path: Path) -> Json:
    actual_file_sha = sha256_file(plan_path)
    if actual_file_sha != EXPECTED_PLAN_FILE_SHA256:
        raise PreflightError(f"asset-plan file digest mismatch: {actual_file_sha}")
    plan = load_json(plan_path)
    assets = validate_plan(plan)

    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        raise PreflightError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be provided via environment")

    env = dict(os.environ)
    env["AWS_DEFAULT_REGION"] = "default"
    env["AWS_REGION"] = "default"
    env["AWS_PAGER"] = ""
    env["AWS_EC2_METADATA_DISABLED"] = "true"

    version = subprocess.run(["aws", "--version"], capture_output=True, text=True, env=env)
    if version.returncode != 0:
        raise PreflightError("aws CLI is unavailable")
    aws_version = (version.stdout or version.stderr).strip()

    results: list[Json] = []
    for asset in assets:
        result = run_head(asset, env)
        results.append({
            "collection": asset.get("collection"),
            "item_id": asset.get("item_id"),
            "asset_key": asset.get("asset_key"),
            "purpose": asset.get("purpose"),
            "stac_href": asset.get("stac_href"),
            "s3_endpoint": asset.get("s3_endpoint"),
            "s3_bucket": asset.get("s3_bucket"),
            "s3_key": asset.get("s3_key"),
            "result": result,
        })

    available = sum(row["result"]["status"] == "available" for row in results)
    failed = len(results) - available
    receipt: Json = {
        "schema": SCHEMA,
        "tool_version": TOOL_VERSION,
        "stage": "r0-authenticated-head-only-preflight",
        "request_method": "S3 HeadObject only",
        "object_body_reads_permitted": False,
        "credentials_recorded": False,
        "credential_source": "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY environment; values intentionally omitted",
        "planner_head": EXPECTED_PLANNER_HEAD,
        "asset_plan_file_sha256": EXPECTED_PLAN_FILE_SHA256,
        "asset_plan_internal_sha256": EXPECTED_PLAN_INTERNAL_SHA256,
        "endpoint": EXPECTED_ENDPOINT,
        "bucket": EXPECTED_BUCKET,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python": sys.version,
        "platform": platform.platform(),
        "aws_cli": aws_version,
        "asset_count": len(results),
        "available_count": available,
        "failed_count": failed,
        "status": "complete" if failed == 0 else "complete-with-failures",
        "etag_claim_boundary": "ETag is retained as server metadata and is not assumed to be SHA-256 or a content hash",
        "assets": results,
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    return receipt

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = build_receipt(args.plan)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"S3 preflight failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": receipt["status"],
        "available": receipt["available_count"],
        "failed": receipt["failed_count"],
        "receipt_sha256": receipt["receipt_sha256"],
        "out": str(args.out),
    }, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
