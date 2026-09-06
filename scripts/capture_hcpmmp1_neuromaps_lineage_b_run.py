#!/usr/bin/env python3
"""Capture a closed-world Lineage-B run manifest from operator-selected local bytes."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
from typing import Iterable
from hcpmmp_neuromaps_common import (
    ContractError, REQUIRED_INPUT_ROLES, RUN_SCHEMA, canonical_json_bytes,
    digest_bytes, digest_file, load_method, load_run, nonempty,
)
CAPTURE_PROFILE="symthaea-hcpmmp1-lineage-b-run-capture-v1"
def parse_inputs(items:Iterable[str])->dict[str,Path]:
    parsed={}
    for item in items:
        if "=" not in item: raise ContractError("input: expected ROLE=PATH")
        role,raw=item.split("=",1)
        if role not in REQUIRED_INPUT_ROLES: raise ContractError(f"input: unknown role {role!r}")
        if role in parsed: raise ContractError(f"input: duplicate role {role!r}")
        if not raw: raise ContractError(f"input {role}: empty path")
        parsed[role]=Path(raw)
    missing=sorted(REQUIRED_INPUT_ROLES-set(parsed))
    if missing: raise ContractError(f"input: missing roles: {', '.join(missing)}")
    return parsed
def _resolve_file(path:Path,ctx:str)->Path:
    try: p=path.expanduser().resolve(strict=True)
    except OSError as e: raise ContractError(f"{ctx}: file not found") from e
    if not p.is_file(): raise ContractError(f"{ctx}: expected regular file")
    return p
def _workbench_identity(wb:Path)->tuple[str,str]:
    before=digest_file(wb)
    try: r=subprocess.run([str(wb),"-version"],check=True,capture_output=True)
    except (OSError,subprocess.CalledProcessError) as e: raise ContractError("workbench: failed exact -version probe") from e
    after=digest_file(wb)
    if before!=after: raise ContractError("workbench: bytes changed during -version probe")
    return before,digest_bytes(r.stdout+r.stderr)
def capture_manifest(method_manifest:Path,workbench:Path,input_items:Iterable[str],execution_id:str,authorization_reference:str)->dict:
    method_manifest=_resolve_file(method_manifest,"method manifest")
    method=load_method(method_manifest)
    nonempty(execution_id,"execution_id"); nonempty(authorization_reference,"authorization_reference")
    wb=_resolve_file(workbench,"workbench")
    paths=parse_inputs(input_items)
    inputs={}
    for role in sorted(REQUIRED_INPUT_ROLES):
        p=_resolve_file(paths[role],f"input {role}")
        inputs[role]={"path":str(p),"sha256":digest_file(p)}
    wb_sha,version_sha=_workbench_identity(wb)
    for role in sorted(REQUIRED_INPUT_ROLES):
        if digest_file(Path(inputs[role]["path"]))!=inputs[role]["sha256"]:
            raise ContractError(f"input {role}: bytes changed during capture")
    doc={
      "schema":RUN_SCHEMA,
      "method_manifest_digest":digest_file(method_manifest),
      "execution_id":execution_id,
      "authorization_reference":authorization_reference,
      "workbench":{"path":str(wb),"sha256":wb_sha,"version_output_sha256":version_sha},
      "inputs":inputs,
    }
    with tempfile.TemporaryDirectory(prefix="symthaea-hcpmmp-run-capture-") as td:
        candidate=Path(td)/"run.json"
        candidate.write_bytes(canonical_json_bytes(doc)+b"\n")
        load_run(candidate,method,method_manifest)
    return doc
def write_new(path:Path,doc:dict)->Path:
    path=path.expanduser()
    parent=path.parent.resolve(strict=True)
    target=parent/path.name
    payload=canonical_json_bytes(doc)+b"\n"
    fd,tmp_name=tempfile.mkstemp(prefix=f".{target.name}.",dir=parent)
    tmp=Path(tmp_name)
    try:
        os.chmod(tmp,0o600)
        with os.fdopen(fd,"wb") as h:
            h.write(payload);h.flush();os.fsync(h.fileno())
        try: os.link(tmp,target)
        except FileExistsError as e: raise ContractError("output: refusing to overwrite existing manifest") from e
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass
    return target
def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description="Capture a Lineage-B run manifest without downloading or transforming data.")
    p.add_argument("--method-manifest",required=True,type=Path);p.add_argument("--workbench",required=True,type=Path)
    p.add_argument("--execution-id",required=True);p.add_argument("--authorization-reference",required=True)
    p.add_argument("--input",action="append",required=True,metavar="ROLE=PATH");p.add_argument("--output",required=True,type=Path)
    a=p.parse_args(argv)
    try:
        doc=capture_manifest(a.method_manifest,a.workbench,a.input,a.execution_id,a.authorization_reference)
        target=write_new(a.output,doc)
    except (ContractError,OSError,json.JSONDecodeError) as e:
        print(f"ERROR: {e}",file=sys.stderr);return 2
    receipt={"profile":CAPTURE_PROFILE,"run_manifest_file_sha256":digest_file(target)}
    print(json.dumps(receipt,sort_keys=True));return 0
if __name__=="__main__": raise SystemExit(main())
