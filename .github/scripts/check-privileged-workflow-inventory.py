#!/usr/bin/env python3
"""Fail-closed detector for repository GitHub Actions privileged trust boundaries."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any

PRIVILEGED_EVENTS = {"pull_request_target", "workflow_run"}
PINNED_USE_RE = re.compile(r"^([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@([0-9a-fA-F]{40})$")
KEY_RE = re.compile(r"^([A-Za-z0-9_.-]+):(?:\s*(.*))?$")

class InventoryError(RuntimeError):
    pass

def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))

def _strip_comment(value: str) -> str:
    return value.split("#", 1)[0].rstrip()

def _block(lines: list[str], start: int) -> tuple[list[tuple[int, str]], int]:
    base = _indent(lines[start])
    out = []
    i = start + 1
    while i < len(lines):
        raw = lines[i]
        if raw.strip() and _indent(raw) <= base:
            break
        if raw.strip() and not raw.lstrip().startswith("#"):
            out.append((_indent(raw), raw.strip()))
        i += 1
    return out, i

def _reject_yaml_meta(value: str, context: str) -> None:
    if re.search(r"(^|\\s)[&*][A-Za-z0-9_.-]+", value) or re.search(r"(^|\\s)!!?[A-Za-z0-9_.-]+", value):
        raise InventoryError(f"{context}: YAML anchors, aliases, or tags are unsupported")
    if re.search(r"(^|\\s)<<\\s*:", value):
        raise InventoryError(f"{context}: YAML merge-key syntax is unsupported")

def _parse_inline_list(value: str) -> list[str]:
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        raise InventoryError(f"expected YAML flow list, got: {value}")
    body = value[1:-1].strip()
    if not body:
        return []
    return [p.strip().strip("'\"") for p in body.split(",")]

def parse_workflow(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    top_keys: dict[str, int] = {}
    for i, raw in enumerate(lines):
        if raw.strip() and _indent(raw) == 0 and not raw.lstrip().startswith("#"):
            m = KEY_RE.match(raw.strip())
            if not m:
                raise InventoryError(f"{path}: unsupported top-level syntax: {raw}")
            key, _ = m.groups()
            if key in top_keys:
                raise InventoryError(f"{path}: duplicate top-level key: {key}")
            top_keys[key] = i
    if "on" not in top_keys:
        raise InventoryError(f"{path}: missing top-level 'on' mapping")
    on_line = KEY_RE.match(lines[top_keys["on"]].strip())
    on_value = _strip_comment((on_line.group(2) if on_line else "") or "").strip()
    if on_value and on_value not in {"{}", "{ }"}:
        raise InventoryError(f"{path}: inline or ambiguous top-level 'on' syntax is unsupported")

    on_items, _ = _block(lines, top_keys["on"])
    events: dict[str, dict[str, Any]] = {}
    event_lines: dict[str, int] = {}
    for indent, text in on_items:
        if indent != 2:
            continue
        m = KEY_RE.match(text)
        if not m:
            raise InventoryError(f"{path}: unsupported event syntax: {text}")
        event, value = m.groups()
        if event in events:
            raise InventoryError(f"{path}: duplicate event: {event}")
        events[event] = {"types": [], "workflows": []}
        event_lines[event] = next(i for i, raw in enumerate(lines) if _indent(raw) == 2 and raw.strip() == text)
        value = _strip_comment(value or "").strip()
        if value.startswith("["):
            events[event]["types"] = _parse_inline_list(value)
        elif value and value not in {"{}", "{ }"}:
            raise InventoryError(f"{path}: inline or ambiguous trigger syntax is unsupported: {text}")

    for event, start in event_lines.items():
        nested, _ = _block(lines, start)
        for indent, text in nested:
            if indent != 4:
                continue
            m = KEY_RE.match(text)
            if not m:
                raise InventoryError(f"{path}: unsupported trigger syntax: {text}")
            key, value = m.groups()
            value = _strip_comment(value or "")
            if key in {"types", "workflows"}:
                if value:
                    vals = _parse_inline_list(value)
                else:
                    vals = [nt[2:].strip().strip("'\"") for ni, nt in nested if ni == 6 and nt.startswith("- ")]
                events[event][key] = vals
            elif event in PRIVILEGED_EVENTS or key not in {"branches", "branches-ignore", "paths", "paths-ignore"}:
                raise InventoryError(f"{path}: unsupported trigger field for {event}: {key}")

    # A privileged workflow must have an explicit top-level permissions mapping
    # in this restricted v1 grammar. Omitting it makes the effective token surface
    # dependent on repository defaults, which this detector cannot safely infer.
    if any(e in events for e in PRIVILEGED_EVENTS) and "permissions" not in top_keys:
        raise InventoryError(f"{path}: privileged workflow must declare top-level permissions explicitly")

    privileged = [e for e in events if e in PRIVILEGED_EVENTS]
    if not privileged:
        return {"path": path.as_posix(), "privileged": False}
    if len(privileged) != 1:
        raise InventoryError(
            f"{path}: v1 requires one privileged trigger per workflow; found {sorted(privileged)}"
        )
    for event in privileged:
        if not events[event]["types"]:
            raise InventoryError(f"{path}: {event} must declare activity types explicitly")
        if event == "workflow_run" and not events[event]["workflows"]:
            raise InventoryError(f"{path}: workflow_run must declare upstream workflow names")
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith(("&", "*", "<<:")):
            raise InventoryError(f"{path}: YAML anchors/aliases/merge keys are unsupported")

    permissions: dict[str, Any] = {"top_level": {}, "jobs": {}}
    if "permissions" in top_keys:
        permission_line = KEY_RE.match(lines[top_keys["permissions"]].strip())
        permission_value = _strip_comment((permission_line.group(2) if permission_line else "") or "").strip()
        if permission_value and permission_value not in {"{}", "{ }"}:
            raise InventoryError(f"{path}: inline top-level permissions syntax is unsupported")
        items, _ = _block(lines, top_keys["permissions"])
        for indent, text in items:
            if indent == 2:
                m = KEY_RE.match(text)
                if not m:
                    raise InventoryError(f"{path}: unsupported permissions syntax: {text}")
                key, value = m.groups()
                permission = _strip_comment(value or "")
                _reject_yaml_meta(permission, f"{path}: top-level permission {key}")
                permissions["top_level"][key] = permission

    if "jobs" in top_keys:
        jobs, _ = _block(lines, top_keys["jobs"])
        job_names = [m.group(1) for indent, text in jobs if indent == 2 and (m := KEY_RE.match(text))]
        for job in job_names:
            start = next(i for i, raw in enumerate(lines) if _indent(raw) == 2 and raw.strip() == job + ":")
            jb, _ = _block(lines, start)
            for indent, text in jb:
                if indent == 4 and text.startswith("permissions:"):
                    permission_value = _strip_comment(text.split(":", 1)[1]).strip()
                    if permission_value and permission_value not in {"{}", "{ }"}:
                        raise InventoryError(f"{path}: inline job permissions syntax is unsupported for {job}")
                    pstart = next(i for i, raw in enumerate(lines) if i > start and _indent(raw) == 4 and raw.strip() == text)
                    pb, _ = _block(lines, pstart)
                    values = {}
                    for pi, pt in pb:
                        if pi == 6:
                            m = KEY_RE.match(pt)
                            if not m:
                                raise InventoryError(f"{path}: unsupported job permissions syntax: {pt}")
                            key, value = m.groups()
                            permission = _strip_comment(value or "")
                            _reject_yaml_meta(permission, f"{path}: job permission {job}.{key}")
                            values[key] = permission
                    permissions["jobs"][job] = values

    uses = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped.startswith("uses:"):
            continue
        value = stripped[len("uses:"):].strip()
        match = PINNED_USE_RE.match(value)
        if match:
            uses.append({"uses": match.group(1), "sha": match.group(2).lower()})
        elif value.startswith("./") or value.startswith("docker://"):
            continue
        else:
            raise InventoryError(f"{path}: privileged workflow has unpinned/unsupported uses: {value}")

    event = next(e for e in privileged if e in {"pull_request_target", "workflow_run"})
    contract = {
        "path": path.as_posix(),
        "trigger": {"event": event, "types": events[event]["types"]},
        "permissions": permissions,
        "third_party_actions": sorted(uses, key=lambda x: (x["uses"], x["sha"])),
    }
    download_artifact = False
    cache_modes: list[str] = []
    if event == "workflow_run":
        contract["trigger"]["workflows"] = events[event]["workflows"]
        source = "\n".join(lines)
        required_provenance_fragments = (
            "run.repository?.full_name",
            "run.name",
            "run.head_branch",
            "run.head_sha",
            "run.conclusion",
            "run.id",
            "run.run_attempt",
        )
        missing_provenance_fragments = [
            fragment for fragment in required_provenance_fragments
            if fragment not in source
        ]
        if missing_provenance_fragments:
            raise InventoryError(
                f"{path}: workflow_run must explicitly bind runtime provenance fields: "
                f"{missing_provenance_fragments}"
            )
        if "run.repository?.full_name !== expectedRepository" not in source:
            raise InventoryError(f"{path}: workflow_run repository identity guard is missing")
        if "run.name !== expectedWorkflowName" not in source:
            raise InventoryError(f"{path}: workflow_run workflow identity guard is missing")
        if "run.head_branch !== 'main'" not in source:
            raise InventoryError(f"{path}: workflow_run main-branch guard is missing")
        if "run.head_sha !== subject" not in source:
            raise InventoryError(f"{path}: workflow_run subject SHA guard is missing")
        if "!run.id || !run.run_attempt" not in source:
            raise InventoryError(f"{path}: workflow_run run identity guard is missing")
        download_artifact = any(
            "actions/download-artifact@" in raw.strip()
            for raw in lines
        )
        if download_artifact:
            download_lines = [raw.strip() for raw in lines if "actions/download-artifact@" in raw.strip()]
            if not any("with:" in raw for raw in lines):
                raise InventoryError(f"{path}: artifact-consuming workflow_run must declare structured download configuration")
            artifact_provenance_fragments = (
                "run.id",
                "run.run_attempt",
                "artifact",
            )
            missing_artifact_provenance = [
                fragment for fragment in artifact_provenance_fragments
                if fragment not in source
            ]
            if missing_artifact_provenance:
                raise InventoryError(
                    f"{path}: artifact-consuming workflow_run must bind artifact access to runtime provenance: "
                    f"{missing_artifact_provenance}"
                )
            # A workflow_run consumer executes in a new run. Artifact retrieval
            # must explicitly select the triggering upstream run.
            for line_index, raw in enumerate(lines):
                if "actions/download-artifact@" not in raw.strip():
                    continue
                step_indent = _indent(raw)
                step_end = line_index + 1
                while step_end < len(lines) and (
                    not lines[step_end].strip() or _indent(lines[step_end]) > step_indent
                ):
                    step_end += 1
                step_lines = lines[line_index:step_end]
                with_indices = [i for i, value in enumerate(step_lines) if value.strip() == "with:"]
                if len(with_indices) != 1:
                    raise InventoryError(f"{path}: each download-artifact step must have exactly one structured with block")
                with_start = with_indices[0]
                with_end = with_start + 1
                while with_end < len(step_lines) and (
                    not step_lines[with_end].strip() or _indent(step_lines[with_end]) > _indent(step_lines[with_start])
                ):
                    with_end += 1
                with_values = {}
                for value in step_lines[with_start + 1:with_end]:
                    if not value.strip():
                        continue
                    match = KEY_RE.match(value.strip())
                    if not match:
                        raise InventoryError(f"{path}: unsupported download-artifact input syntax: {value.strip()}")
                    key, value_text = match.groups()
                    value_text = _strip_comment(value_text or "").strip().strip("'\"")
                    if key in with_values:
                        raise InventoryError(f"{path}: duplicate download-artifact input: {key}")
                    with_values[key] = value_text
                artifact_name = with_values.get("name")
                run_id = with_values.get("run-id")
                if not artifact_name:
                    raise InventoryError(f"{path}: workflow_run artifact access requires an exact 'name' input")
                if not run_id or run_id != "${{ github.event.workflow_run.id }}":
                    raise InventoryError(f"{path}: workflow_run artifact access must use the triggering workflow_run id")
                if "pattern" in with_values:
                    raise InventoryError(f"{path}: broad artifact pattern access is unsupported in privileged v1")
                if "artifact-ids" in with_values:
                    raise InventoryError(f"{path}: artifact-id access is unsupported in privileged v1; bind to exact name + run-id")
    for raw in lines:
        stripped = raw.strip()
        match = re.match(r"cache-mode:\s*(read|write|write-only|none)\s*$", stripped)
        if not match:
            continue
        indentation = _indent(raw)
        if indentation not in {0, 4}:
            raise InventoryError(
                f"{path}: cache-mode is only supported at workflow or job scope; "
                f"unsupported indentation {indentation}"
            )
        cache_modes.append(match.group(1))
    explicit_cache_write = any(mode in {"write", "write-only"} for mode in cache_modes)
    local_reusable_calls = []
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("uses:"):
            value = stripped[len("uses:"):].strip()
            if value.startswith("./.github/workflows/"):
                local_reusable_calls.append(value)
    if local_reusable_calls and not cache_modes:
        raise InventoryError(
            f"{path}: privileged workflow calling a local reusable workflow must explicitly cap cache-mode"
        )
    contract["cache_influence"] = (
        "explicit_write_override"
        if explicit_cache_write
        else ("explicit_read_or_none" if cache_modes else "default_low_trust_read")
    )
    contract["cache_modes"] = sorted(set(cache_modes))
    contract["local_reusable_workflow_calls"] = sorted(set(local_reusable_calls))
    contract["cross_workflow_dataflow_observed"] = {
        "artifact_download_action_present": download_artifact,
        "explicit_cache_write_override_present": explicit_cache_write,
    }
    return {"path": path.as_posix(), "privileged": True, "contract": contract}

def load_inventory(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"invalid inventory: {exc}") from exc

def validate_inventory(workflows_dir: Path, inventory_path: Path) -> None:
    inventory = load_inventory(inventory_path)
    entries = inventory.get("workflows")
    if not isinstance(entries, list):
        raise InventoryError("inventory.workflows must be a list")
    declared = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise InventoryError("every inventory entry must contain a workflow path")
        path = entry["path"]
        if path in declared:
            raise InventoryError(f"duplicate inventory workflow: {path}")
        declared[path] = entry
    observed = {}
    for path in sorted(workflows_dir.glob("*.yml")) + sorted(workflows_dir.glob("*.yaml")):
        result = parse_workflow(path)
        if result["privileged"]:
            observed[result["path"]] = result["contract"]
    if set(observed) != set(declared):
        missing = sorted(set(observed) - set(declared))
        stale = sorted(set(declared) - set(observed))
        raise InventoryError(f"privileged inventory coverage mismatch: undocumented={missing}; stale={stale}")
    mismatches = []
    for path, observed_contract in observed.items():
        entry = declared[path]
        for field in (
            "trigger",
            "permissions",
            "third_party_actions",
            "cache_influence",
            "cache_modes",
            "local_reusable_workflow_calls",
        ):
            if entry.get(field) != observed_contract.get(field):
                mismatches.append(f"{path}: {field} differs")
        if observed_contract["trigger"]["event"] == "workflow_run":
            dataflow = entry.get("cross_workflow_dataflow")
            if not isinstance(dataflow, dict):
                mismatches.append(f"{path}: cross_workflow_dataflow is required for workflow_run")
            else:
                required = {
                    "upstream_workflow_names",
                    "artifacts_consumed",
                    "artifact_names",
                    "artifact_extraction",
                    "artifact_execution",
                    "upstream_identity_checks",
                    "provenance_binding",
                    "cache_influence",
                    "privileged_side_effects",
                }
                missing = sorted(required - set(dataflow))
                if missing:
                    mismatches.append(f"{path}: cross_workflow_dataflow missing {missing}")
                if dataflow.get("upstream_workflow_names") != observed_contract["trigger"]["workflows"]:
                    mismatches.append(f"{path}: cross_workflow_dataflow upstream_workflow_names differs")
                provenance = dataflow.get("provenance_binding")
                if not isinstance(provenance, dict):
                    mismatches.append(f"{path}: provenance_binding is required")
                else:
                    required_provenance = {
                        "repository",
                        "workflow_name",
                        "head_branch",
                        "head_sha",
                        "conclusion",
                        "run_identity",
                        "pr_association",
                    }
                    missing_provenance = sorted(required_provenance - set(provenance))
                    if missing_provenance:
                        mismatches.append(f"{path}: provenance_binding missing {missing_provenance}")
                if observed_contract["cross_workflow_dataflow_observed"]["artifact_download_action_present"] and not dataflow.get("artifacts_consumed"):
                    mismatches.append(f"{path}: artifact download is present but artifacts_consumed is false")
                if dataflow.get("artifacts_consumed"):
                    artifact_names = dataflow.get("artifact_names")
                    if not isinstance(artifact_names, list) or not artifact_names or not all(isinstance(name, str) and name.strip() for name in artifact_names):
                        mismatches.append(f"{path}: artifact-consuming workflow_run requires a non-empty exact artifact_names allowlist")
                    if dataflow.get("artifact_extraction") in {None, "", "none"}:
                        mismatches.append(f"{path}: artifact-consuming workflow_run requires explicit artifact_extraction handling")
                    if dataflow.get("artifact_execution") is not False:
                        mismatches.append(f"{path}: artifact_execution must be false for the current fail-closed v1 contract")
                if not observed_contract["cross_workflow_dataflow_observed"]["artifact_download_action_present"] and dataflow.get("artifacts_consumed"):
                    mismatches.append(f"{path}: artifacts_consumed is true but no download-artifact action is present")
                if dataflow.get("cache_influence") != observed_contract["cache_influence"]:
                    mismatches.append(f"{path}: cross_workflow_dataflow cache_influence differs from observed cache surface")
    if mismatches:
        raise InventoryError("privileged inventory mismatch: " + "; ".join(mismatches))

def self_test() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        wf = root / "example.yml"
        inv = root / "inventory.json"
        wf.write_text("""name: Example
on:
  workflow_run:
    workflows:
      - Trusted upstream
    types:
      - completed
permissions:
  contents: read
jobs:
  receipt:
    permissions:
      actions: read
      contents: read
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@0123456789abcdef0123456789abcdef01234567
        env:
          EXPECTED_REPOSITORY: Luminous-Dynamics/symthaea
          EXPECTED_WORKFLOW_NAME: Trusted upstream
          EXPECTED_SUBJECT: 0123456789abcdef0123456789abcdef01234567
        with:
          script: |
            const run = context.payload.workflow_run;
            const expectedRepository = process.env.EXPECTED_REPOSITORY;
            const expectedWorkflowName = process.env.EXPECTED_WORKFLOW_NAME;
            const subject = process.env.EXPECTED_SUBJECT;
            if (run.repository?.full_name !== expectedRepository) throw new Error('repository identity mismatch');
            if (run.name !== expectedWorkflowName) throw new Error('workflow identity mismatch');
            if (run.head_branch !== 'main') throw new Error('branch mismatch');
            if (run.head_sha !== subject) throw new Error('subject mismatch');
            if (!run.id || !run.run_attempt) throw new Error('run identity incomplete');
            if (run.conclusion !== 'success') throw new Error('conclusion mismatch');
""", encoding="utf-8")
        observed = parse_workflow(wf)["contract"]
        assert observed["trigger"] == {"event": "workflow_run", "types": ["completed"], "workflows": ["Trusted upstream"]}
        assert observed["cache_influence"] == "default_low_trust_read"
        assert observed["cache_modes"] == []
        assert observed["local_reusable_workflow_calls"] == []
        original = wf.read_text(encoding="utf-8")
        missing_permissions = original.replace("permissions:\n  contents: read\n", "")
        wf.write_text(missing_permissions, encoding="utf-8")
        try:
            parse_workflow(wf)
        except InventoryError:
            pass
        else:
            raise AssertionError("privileged workflow without explicit permissions must fail closed")
        wf.write_text(original, encoding="utf-8")
        for malformed in (
            original.replace("    types:\n      - completed\n", ""),
            original.replace("      - Trusted upstream\n", ""),
            original.replace("  workflow_run:", "  pull_request_target:\n    types: [opened]\n  workflow_run:"),
            original.replace("    permissions:\n      actions: read\n      contents: read", "    permissions: read-all"),
            original.replace("      - completed", "      - *activity_type"),
            original.replace("      - Trusted upstream", "      - &upstream Trusted upstream"),
        ):
            wf.write_text(malformed, encoding="utf-8")
            try:
                parse_workflow(wf)
            except InventoryError:
                continue
            raise AssertionError("ambiguous privileged trigger/permission must fail closed")
        wf.write_text(original, encoding="utf-8")
        assert observed["permissions"]["top_level"] == {"contents": "read"}
        assert observed["permissions"]["jobs"]["receipt"] == {"actions": "read", "contents": "read"}
        inv_entry = {
            "path": "example.yml",
            **observed,
            "cache_influence": "default_low_trust_read",
            "cache_modes": [],
            "local_reusable_workflow_calls": [],
            "cross_workflow_dataflow": {
                "upstream_workflow_names": ["Trusted upstream"],
                "artifacts_consumed": False,
                "artifact_names": [],
                "artifact_extraction": "none",
                "artifact_execution": False,
                "upstream_identity_checks": ["configured_workflow_name", "head_branch", "head_sha", "conclusion"],
                "provenance_binding": {
                    "repository": "event_repository",
                    "workflow_name": "configured_workflow_name",
                    "head_branch": "main",
                    "head_sha": "event_workflow_run.head_sha",
                    "conclusion": "event_workflow_run.conclusion",
                    "run_identity": "event_workflow_run.id+run_attempt",
                    "pr_association": "not_required_for_current_main-only consumer",
                },
                "cache_influence": "none_observed",
                "privileged_side_effects": ["none"],
            },
        }
        inv.write_text(json.dumps({"workflows": [inv_entry]}), encoding="utf-8")
        validate_inventory(root, inv)
        data = json.loads(inv.read_text(encoding="utf-8"))
        data["workflows"][0]["trigger"]["types"] = ["requested"]
        inv.write_text(json.dumps(data), encoding="utf-8")
        try:
            validate_inventory(root, inv)
        except InventoryError:
            pass
        else:
            raise AssertionError("trigger mismatch must fail closed")
        data = json.loads(inv.read_text(encoding="utf-8"))
        data["workflows"][0]["trigger"]["types"] = ["completed"]
        data["workflows"][0]["cross_workflow_dataflow"]["artifacts_consumed"] = True
        inv.write_text(json.dumps(data), encoding="utf-8")
        try:
            validate_inventory(root, inv)
        except InventoryError:
            pass
        else:
            raise AssertionError("artifact trust contract must reject false artifact-consumption declaration")
        data["workflows"][0]["cross_workflow_dataflow"]["artifacts_consumed"] = False
        data["workflows"][0]["cross_workflow_dataflow"].pop("upstream_workflow_names")
        inv.write_text(json.dumps(data), encoding="utf-8")
        try:
            validate_inventory(root, inv)
        except InventoryError:
            pass
        else:
            raise AssertionError("artifact trust contract must require upstream workflow identity")
        wf.write_text(wf.read_text(encoding="utf-8").replace("workflow_run:", "workflow_dispatch:"), encoding="utf-8")
        try:
            validate_inventory(root, inv)
        except InventoryError:
            pass
        else:
            raise AssertionError("stale inventory must fail closed")

        cache_write = original.replace(
            "permissions:\n  contents: read\n",
            "permissions:\n  contents: read\ncache-mode: write\n",
        )
        wf.write_text(cache_write, encoding="utf-8")
        observed_cache_write = parse_workflow(wf)["contract"]
        assert observed_cache_write["cache_influence"] == "explicit_write_override"
        assert observed_cache_write["cache_modes"] == ["write"]

        reusable_without_cap = original.replace(
            "    runs-on: ubuntu-latest\n",
            "    uses: ./.github/workflows/reusable.yml\n",
        )
        wf.write_text(reusable_without_cap, encoding="utf-8")
        try:
            parse_workflow(wf)
        except InventoryError:
            pass
        else:
            raise AssertionError("privileged reusable workflow caller without explicit cache cap must fail closed")

        reusable_with_cap = reusable_without_cap.replace(
            "permissions:\n  contents: read\n",
            "permissions:\n  contents: read\ncache-mode: read\n",
        )
        wf.write_text(reusable_with_cap, encoding="utf-8")
        observed_reusable = parse_workflow(wf)["contract"]
        assert observed_reusable["cache_influence"] == "explicit_read_or_none"
        assert observed_reusable["cache_modes"] == ["read"]
        assert observed_reusable["local_reusable_workflow_calls"] == ["./.github/workflows/reusable.yml"]

        step_cache = original.replace(
            "        with:\n",
            "        cache-mode: write\n        with:\n",
        )
        wf.write_text(step_cache, encoding="utf-8")
        try:
            parse_workflow(wf)
        except InventoryError:
            pass
        else:
            raise AssertionError("step-scoped cache-mode must fail closed")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--workflows-dir", default=".github/workflows")
    parser.add_argument("--inventory", default=".github/governance-privileged-workflow-inventory-v1.json")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("privileged workflow inventory self-test: PASS")
        return 0
    validate_inventory(Path(args.workflows_dir), Path(args.inventory))
    print("privileged workflow inventory validation: PASS")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InventoryError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
