#!/usr/bin/env python3
"""Bind an exact focused routing decision to an exact set of V3 admissions."""
from __future__ import annotations
import hashlib, json
from typing import Any
import integration_train_manifest as train
import qualification_admission_v3 as admission_mod
import qualification_routing as routing_mod

SCHEMA="symthaea.qualification-admission-coverage.v1"
DOMAIN=b"symthaea.qualification-admission-coverage.v1\0"
DISPOSITION="FocusedAdmissionCovered"


def _payload(value:dict[str,Any])->bytes:
    return json.dumps({k:v for k,v in value.items() if k!="coverage_id"},sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()


def _id(value:dict[str,Any])->str:
    return "sha256:"+hashlib.sha256(DOMAIN+_payload(value)).hexdigest()


def build_coverage(decision:Any,admissions:list[Any])->dict[str,Any]:
    route=routing_mod.normalize_decision(decision,require_id=True)
    if route["disposition"]!=routing_mod.FOCUSED:
        raise train.TrainManifestError("full-CI routing cannot be satisfied by focused admissions")
    if not admissions:
        raise train.TrainManifestError("focused routing requires admission evidence")

    by_profile:dict[str,dict[str,Any]]={}
    for index,raw in enumerate(admissions):
        item=admission_mod.normalize_request(raw,require_ids=True)
        subject=item["subject"]
        if subject["subject_id"]!=route["subject_id"] or subject["source_commit"]!=route["source_commit"]:
            raise train.TrainManifestError(f"admission[{index}] does not bind routing subject")
        profile_id=item["qualification_profile"]["profile_id"]
        if profile_id in by_profile:
            raise train.TrainManifestError(f"duplicate admission for routed profile {profile_id}")
        by_profile[profile_id]=item

    admitted=sorted(by_profile)
    required=route["required_profile_ids"]
    if admitted!=required:
        missing=sorted(set(required)-set(admitted))
        extra=sorted(set(admitted)-set(required))
        raise train.TrainManifestError(
            f"focused admission profile set mismatch: missing={missing}, extra={extra}"
        )

    ordered=[by_profile[profile_id] for profile_id in required]
    coverage:dict[str,Any]={
        "schema":SCHEMA,
        "disposition":DISPOSITION,
        "routing_decision_id":route["decision_id"],
        "subject_id":route["subject_id"],
        "source_commit":route["source_commit"],
        "required_profile_ids":required,
        "admission_ids":[item["admission_id"] for item in ordered],
        "admission_subject_ids":[item["admission_subject_id"] for item in ordered],
    }
    coverage["coverage_id"]=_id(coverage)
    return coverage


def verify_coverage(coverage:Any,decision:Any,admissions:list[Any])->None:
    if not isinstance(coverage,dict):
        raise train.TrainManifestError("admission coverage: expected object")
    expected=build_coverage(decision,admissions)
    if coverage!=expected:
        raise train.TrainManifestError("admission coverage does not equal canonical exact-set composition")
