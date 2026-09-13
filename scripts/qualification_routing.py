#!/usr/bin/env python3
"""Content-addressed qualification routing decisions."""
from __future__ import annotations
import hashlib, json
from typing import Any
import integration_train_manifest as train
import qualification_profile as profile_mod
import qualification_subject as subject_mod

SCHEMA="symthaea.qualification-routing-decision.v1"
DOMAIN=b"symthaea.qualification-routing-decision.v1\0"
FOCUSED="FocusedProfilesRequired"
FULL="FullCiRequired"
CHANGED_PATHS_FORMAT="git-diff-name-status-z-no-renames/v1"


def _sha256(value:Any,where:str)->str:
    if not isinstance(value,str) or len(value)!=71 or not value.startswith("sha256:") or any(c not in "0123456789abcdef" for c in value[7:]):
        raise train.TrainManifestError(f"{where}: expected sha256:<64 lowercase hex>")
    return value


def _payload(value:dict[str,Any])->bytes:
    return json.dumps({k:v for k,v in value.items() if k!="decision_id"},sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()


def _id(value:dict[str,Any])->str:
    return "sha256:"+hashlib.sha256(DOMAIN+_payload(value)).hexdigest()


def normalize_decision(value:Any,*,verify_declared_id:bool=True,require_id:bool=False)->dict[str,Any]:
    if not isinstance(value,dict):
        raise train.TrainManifestError("routing decision: expected object")
    subject_mod._require_exact_keys(value,{"schema","subject_id","object_format","source_commit","base_commit","merge_base","changed_paths_format","changed_paths_sha256","router_recipe_id","disposition","required_profile_ids","reason"},{"decision_id"},where="routing decision")
    if value["schema"]!=SCHEMA:
        raise train.TrainManifestError(f"routing decision.schema: expected {SCHEMA!r}")
    if require_id and "decision_id" not in value:
        raise train.TrainManifestError("routing decision.decision_id: required but absent")
    object_format=subject_mod._require_object_format(value["object_format"],where="routing decision.object_format")
    changed_paths_format=subject_mod._require_string(value["changed_paths_format"],where="routing decision.changed_paths_format")
    if changed_paths_format!=CHANGED_PATHS_FORMAT:
        raise train.TrainManifestError(f"routing decision.changed_paths_format: expected {CHANGED_PATHS_FORMAT!r}")
    disposition=subject_mod._require_string(value["disposition"],where="routing decision.disposition")
    if disposition not in {FOCUSED,FULL}:
        raise train.TrainManifestError("routing decision.disposition: unsupported value")
    raw=value["required_profile_ids"]
    if not isinstance(raw,list):
        raise train.TrainManifestError("routing decision.required_profile_ids: expected array")
    profiles=[profile_mod._require_profile_id(v,where=f"routing decision.required_profile_ids[{i}]") for i,v in enumerate(raw)]
    if profiles!=sorted(set(profiles)):
        raise train.TrainManifestError("routing decision.required_profile_ids: must be sorted and unique")
    if disposition==FOCUSED and not profiles:
        raise train.TrainManifestError("focused routing requires at least one profile")
    if disposition==FULL and profiles:
        raise train.TrainManifestError("full-CI routing must not imply focused-profile coverage")
    out={
        "schema":SCHEMA,
        "subject_id":_sha256(value["subject_id"],"routing decision.subject_id"),
        "object_format":object_format,
        "source_commit":subject_mod._require_object_id(value["source_commit"],where="routing decision.source_commit",object_format=object_format),
        "base_commit":subject_mod._require_object_id(value["base_commit"],where="routing decision.base_commit",object_format=object_format),
        "merge_base":subject_mod._require_object_id(value["merge_base"],where="routing decision.merge_base",object_format=object_format),
        "changed_paths_format":changed_paths_format,
        "changed_paths_sha256":_sha256(value["changed_paths_sha256"],"routing decision.changed_paths_sha256"),
        "router_recipe_id":profile_mod._require_recipe_id(value["router_recipe_id"],where="routing decision.router_recipe_id"),
        "disposition":disposition,"required_profile_ids":profiles,
        "reason":subject_mod._require_string(value["reason"],where="routing decision.reason"),
    }
    out["decision_id"]=_id(out)
    if verify_declared_id and "decision_id" in value and value["decision_id"]!=out["decision_id"]:
        raise train.TrainManifestError("routing decision.decision_id: content mismatch")
    return out


def validate_against_subject(decision:Any,subject:Any)->None:
    route=normalize_decision(decision,require_id=True)
    source=subject_mod.normalize_subject(subject,require_id=True)
    if route["subject_id"]!=source["subject_id"] or route["source_commit"]!=source["source_commit"] or route["object_format"]!=source["object_format"]:
        raise train.TrainManifestError("routing decision does not bind the exact qualification subject")


def compute_decision_id(value:Any)->str:
    return normalize_decision(value,verify_declared_id=False)["decision_id"]
