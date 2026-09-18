#!/usr/bin/env python3
"""Byte-reversible draft-admission transformer for Symthaea CI.

Authority: RunnerPlaneCandidateOnly. This tool does not qualify or merge CI.
"""

from __future__ import annotations
import argparse, hashlib, json, pathlib, re, tempfile
from dataclasses import dataclass
from typing import Any

AUTHORITY = "RunnerPlaneCandidateOnly"
SCHEMA = "symthaea.ci.draft-admission-transformer-contract.v1"
BLOCK_MARKERS = {"|", "|-", "|+", ">", ">-", ">+"}

@dataclass(frozen=True)
class Job:
    key: str
    start: int
    end: int
    if_line: int | None
    if_value: str | None
    block_end: int | None

@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    replacement: tuple[str, ...]
    original: tuple[str, ...]
    job: str
    operation: str

def req(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()

def indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))

def load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    req(isinstance(value, dict), f"{path}: top-level JSON must be object")
    return value

def jobs(lines: list[str]) -> list[Job]:
    root = next((i for i, x in enumerate(lines) if x in {"jobs:\n", "jobs:"}), None)
    req(root is not None, "top-level jobs mapping not found")
    rx = re.compile(r"^  ([A-Za-z0-9_.-]+):\s*(?:#.*)?\n?$")
    starts: list[tuple[str, int]] = []
    for i in range(root + 1, len(lines)):
        x = lines[i]
        if x and not x.startswith((" ", "\t", "#", "\n")):
            break
        m = rx.match(x)
        if m:
            starts.append((m.group(1), i))
    req(starts and len({k for k, _ in starts}) == len(starts), "invalid/duplicate job keys")

    out: list[Job] = []
    for n, (key, start) in enumerate(starts):
        end = starts[n + 1][1] if n + 1 < len(starts) else len(lines)
        entries = [(i, lines[i].split("if:", 1)[1].strip())
                   for i in range(start + 1, end) if lines[i].startswith("    if:")]
        req(len(entries) <= 1, f"{key}: duplicate job-level if")
        if not entries:
            out.append(Job(key, start, end, None, None, None))
            continue
        if_line, value = entries[0]
        req(value != "", f"{key}: empty job-level if")
        block_end = None
        if value in BLOCK_MARKERS:
            j = if_line + 1
            while j < end:
                x = lines[j]
                req(x.strip() != "", f"{key}: blank line around block if is ambiguous")
                if indent(x) >= 6:
                    j += 1
                    continue
                break
            req(j > if_line + 1, f"{key}: empty block payload")
            block_end = j
        out.append(Job(key, start, end, if_line, value, block_end))
    return out

def edits(lines: list[str], js: list[Job], c: dict[str, Any]) -> list[Edit]:
    admission = c["admission_predicate"]
    block_jobs = [j.key for j in js if j.if_value in BLOCK_MARKERS]
    req(block_jobs == c["expected_block_if_jobs"], f"block-job census mismatch: {block_jobs}")
    out: list[Edit] = []
    for j in js:
        if j.if_line is None:
            out.append(Edit(j.start + 1, j.start + 1,
                            (f"    if: {admission}\n",), (), j.key, "insert"))
            continue

        assert j.if_value is not None
        if j.if_value not in BLOCK_MARKERS:
            req(" #" not in j.if_value and "${{" not in j.if_value,
                f"{j.key}: unsupported scalar if syntax")
            out.append(Edit(
                j.if_line, j.if_line + 1,
                (f"    if: ({admission}) && ({j.if_value})\n",),
                (lines[j.if_line],), j.key, "scalar",
            ))
            continue

        req(j.key in c["expected_block_if_jobs"], f"{j.key}: unregistered block if")
        req(j.if_value == c["expected_block_marker"],
            f"{j.key}: unexpected block marker {j.if_value!r}")
        assert j.block_end is not None
        payload = tuple(lines[j.if_line + 1:j.block_end])
        req(all(indent(x) >= 6 for x in payload), f"{j.key}: nonstandard block payload")
        out.append(Edit(
            j.if_line, j.block_end,
            (lines[j.if_line],
             f"      ({admission}) &&\n",
             "      (\n",
             *payload,
             "      )\n"),
            tuple(lines[j.if_line:j.block_end]),
            j.key, "block",
        ))
    return out

def apply(source: list[str], es: list[Edit]) -> tuple[list[str], list[tuple[int,int,Edit]]]:
    out = list(source)
    applied: list[tuple[int,int,Edit]] = []
    delta = 0
    for e in sorted(es, key=lambda x: x.start):
        a, b = e.start + delta, e.end + delta
        req(tuple(out[a:b]) == e.original, f"{e.job}: source drift")
        out[a:b] = e.replacement
        applied.append((a, a + len(e.replacement), e))
        delta += len(e.replacement) - (e.end - e.start)
    return out, applied

def reverse(transformed: list[str], applied: list[tuple[int,int,Edit]]) -> list[str]:
    out = list(transformed)
    for a, b, e in reversed(applied):
        req(tuple(out[a:b]) == e.replacement, f"{e.job}: transformed drift")
        out[a:b] = e.original
    return out

def truth_table() -> int:
    count = 0
    for event, draft in [
        ("pull_request", True),
        ("pull_request", False),
        ("push", None),
        ("schedule", None),
        ("workflow_dispatch", None),
    ]:
        admission = event != "pull_request" or draft is False
        for original in (False, True):
            got = admission and original
            expected = False if (event == "pull_request" and draft is True) else original
            req(got == expected, f"semantic regression: {event=} {draft=} {original=}")
            count += 1
    return count

def transform(c: dict[str, Any], source_path: pathlib.Path) -> tuple[bytes, dict[str, Any]]:
    req(c["schema"] == SCHEMA and c["authority"] == AUTHORITY, "contract identity mismatch")
    raw = source_path.read_bytes()
    req(git_blob(raw) == c["source_ci_blob"], "source ci.yml Git blob mismatch")
    text = raw.decode("utf-8")
    req("\r" not in text, "CRLF unsupported")
    src = text.splitlines(keepends=True)
    js = jobs(src)
    es = edits(src, js, c)
    dst, applied = apply(src, es)
    req(reverse(dst, applied) == src, "byte-exact reverse proof failed")

    result = "".join(dst).encode()
    dst_lines = result.decode().splitlines(keepends=True)
    dst_jobs = jobs(dst_lines)
    req([j.key for j in dst_jobs] == [j.key for j in js], "job order changed")
    admission = c["admission_predicate"]
    for j in dst_jobs:
        req(j.if_line is not None, f"{j.key}: transformed job lacks if")
        if j.if_value in BLOCK_MARKERS:
            assert j.block_end is not None
            expr = "".join(dst_lines[j.if_line + 1:j.block_end])
        else:
            expr = j.if_value or ""
        req(expr.count(admission) == 1, f"{j.key}: admission count != 1")

    tt = truth_table()
    receipt = {
        "schema": "symthaea.ci.draft-admission-transform-receipt.v1",
        "authority": AUTHORITY,
        "source_ci_blob": c["source_ci_blob"],
        "source_ci_sha256": sha256(raw),
        "transformed_ci_blob": git_blob(result),
        "transformed_ci_sha256": sha256(result),
        "job_count": len(js),
        "jobs_inserted": sum(e.operation == "insert" for e in es),
        "jobs_scalar_conjoined": sum(e.operation == "scalar" for e in es),
        "jobs_block_wrapped": sum(e.operation == "block" for e in es),
        "block_jobs": c["expected_block_if_jobs"],
        "truth_table_cases": tt,
        "reverse_byte_exact": True,
        "source_job_order_preserved": True,
        "all_transformed_jobs_admission_gated_exactly_once": True,
        "claims": {
            "ci_qualified": False,
            "queue_repaired": False,
            "runner_capacity_increased": False,
            "scientific_authority": False,
        },
    }
    return result, receipt

def self_test() -> dict[str, Any]:
    admission = "github.event_name != 'pull_request' || github.event.pull_request.draft == false"
    source = (
        "name: T\njobs:\n"
        "  plain:\n    runs-on: ubuntu-latest\n    steps: []\n"
        "  scalar:\n    if: github.event_name == 'push'\n    runs-on: ubuntu-latest\n"
        "  block:\n    if: |\n"
        "      github.event_name == 'schedule' ||\n"
        "      github.event_name == 'pull_request'\n"
        "    runs-on: ubuntu-latest\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d, "ci.yml")
        p.write_text(source)
        c = {
            "schema": SCHEMA,
            "authority": AUTHORITY,
            "source_ci_blob": git_blob(source.encode()),
            "admission_predicate": admission,
            "expected_block_if_jobs": ["block"],
            "expected_block_marker": "|",
        }
        transformed, r = transform(c, p)
        out = transformed.decode()
        req(out.count(admission) == 3, "admission census self-test")
        req(r["jobs_inserted"] == r["jobs_scalar_conjoined"] == r["jobs_block_wrapped"] == 1,
            "operation census self-test")
        req(r["truth_table_cases"] == 10, "truth-table census self-test")
    return {
        "schema": "symthaea.ci.draft-admission-transformer-self-test.v2",
        "authority": AUTHORITY,
        "structural_cases": 3,
        "truth_table_cases": 10,
        "reverse_byte_exact": True,
        "pass": True,
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", type=pathlib.Path)
    ap.add_argument("--source", type=pathlib.Path)
    ap.add_argument("--output", type=pathlib.Path)
    ap.add_argument("--receipt", type=pathlib.Path)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    req(all((a.contract, a.source, a.output, a.receipt)),
        "--contract, --source, --output and --receipt are required")
    result, receipt = transform(load(a.contract), a.source)
    a.output.write_bytes(result)
    a.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
