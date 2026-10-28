#!/usr/bin/env python3
"""Adversarial tests for pinned VLEB admission-cohort aggregation."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = Path(__file__).with_name("summarize_admission_cohort.py")
SPEC = importlib.util.spec_from_file_location("vleb_summarize_admission_cohort",MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import cohort summarizer")
summarizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summarizer
SPEC.loader.exec_module(summarizer)
SCHEMA_PATH=Path(__file__).with_name("admission-cohort.schema.json")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value,indent=2,sort_keys=True)+"\n").encode("utf-8")


class CohortFixture:
    PROTOCOL="a"*64
    AUDITOR="b"*64

    def __init__(self) -> None:
        self.tmp=tempfile.TemporaryDirectory(prefix="vleb-admission-cohort-")
        self.root=Path(self.tmp.name)
        self.artifacts=self.root/"artifacts"
        self.artifacts.mkdir()
        self.schema=SCHEMA_PATH
        self.reports=[]
        task_shapes=[
            {
                "task_id":"task-001","eligible":True,"verdict":"pass",
                "receipt_sha256":"1"*64,"task_contract_sha256":"2"*64,
                "protocol_sha256":self.PROTOCOL,"candidate_patch_sha256":"3"*64,"auditor_sha256":self.AUDITOR,
                "requirement_traceability":{
                    "declared_source_item_count":4,"traced_source_item_count":4,
                    "required_test_count":2,"traced_required_test_count":2,"complete":True
                },
                "per_test_baseline_oracle_alignment":{
                    "concrete_execution_count":10,"expected_status_count":10,"observed_status_count":10,
                    "matching_status_count":10,"alignment_pct":100.0,"complete":True
                }
            },
            {
                "task_id":"task-002","eligible":False,"verdict":"fail",
                "receipt_sha256":"4"*64,"task_contract_sha256":"5"*64,
                "protocol_sha256":self.PROTOCOL,"candidate_patch_sha256":"6"*64,"auditor_sha256":self.AUDITOR,
                "requirement_traceability":{
                    "declared_source_item_count":2,"traced_source_item_count":1,
                    "required_test_count":5,"traced_required_test_count":4,"complete":False
                },
                "per_test_baseline_oracle_alignment":{
                    "concrete_execution_count":2,"expected_status_count":2,"observed_status_count":2,
                    "matching_status_count":1,"alignment_pct":50.0,"complete":False
                }
            }
        ]
        rows=[]
        for index,report in enumerate(task_shapes):
            path=f"audit/task-{index+1:03d}.json"
            out=self.artifacts/path
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_bytes(jbytes(report))
            row={
                "task_id":report["task_id"],"audit_report_path":path,"audit_report_sha256":sha(out.read_bytes()),
                "receipt_sha256":report["receipt_sha256"],"task_contract_sha256":report["task_contract_sha256"],
                "protocol_sha256":self.PROTOCOL,"candidate_patch_sha256":report["candidate_patch_sha256"],
                "auditor_sha256":self.AUDITOR
            }
            rows.append(row)
            self.reports.append((out,report,row))
        self.manifest={
            "schema_version":"vleb-admission-cohort-v1","cohort_id":"october-cohort-001",
            "run_kind":"confirmatory","protocol_sha256":self.PROTOCOL,
            "task_ids":[row["task_id"] for row in rows],"task_results":rows
        }
        self.manifest_path=self.root/"cohort.json"
        self.write_manifest()

    def write_manifest(self) -> None:
        self.manifest_path.write_bytes(jbytes(self.manifest))

    def summarize(self,**overrides: str) -> dict[str,Any]:
        values={
            "expected_cohort_sha256":sha(self.manifest_path.read_bytes()),
            "expected_schema_sha256":sha(self.schema.read_bytes()),
            "expected_protocol_sha256":self.PROTOCOL,
            "expected_auditor_sha256":self.AUDITOR,
            "expected_summarizer_sha256":sha(MODULE_PATH.read_bytes()),
        }
        values.update(overrides)
        return summarizer.summarize_cohort(self.manifest_path,self.schema,self.artifacts,**values)

    def close(self)->None:
        self.tmp.cleanup()


class AdmissionCohortTests(unittest.TestCase):
    def setUp(self)->None:
        self.fixture=CohortFixture()

    def tearDown(self)->None:
        self.fixture.close()

    def test_pools_counts_instead_of_averaging_per_task_percentages(self)->None:
        result=self.fixture.summarize()
        self.assertTrue(result["cohort_integrity_valid"],result["issues"])
        self.assertEqual(result["expected_task_count"],2)
        self.assertEqual(result["verified_report_count"],2)
        self.assertEqual(result["audit_eligible_task_count"],1)
        self.assertEqual(result["requirement_source_item_count"],6)
        self.assertEqual(result["requirement_source_traced_count"],5)
        self.assertAlmostEqual(result["requirement_source_coverage_pct"],100*5/6)
        self.assertEqual(result["required_test_id_count"],7)
        self.assertEqual(result["traced_required_test_id_count"],6)
        self.assertAlmostEqual(result["required_test_trace_coverage_pct"],100*6/7)
        self.assertEqual(result["baseline_concrete_execution_count"],12)
        self.assertEqual(result["baseline_matching_status_count"],11)
        self.assertAlmostEqual(result["per_test_baseline_oracle_alignment_pct"],100*11/12)
        self.assertEqual(result["full_admission_pass_task_count"],1)
        self.assertEqual(result["full_admission_pass_task_pct"],50.0)
        self.assertIs(result["score_eligible"],False)
        self.assertIs(result["analysis_only"],True)

    def test_eligible_report_cannot_claim_incomplete_admission_summaries(self)->None:
        path,report,row=self.fixture.reports[0]
        report["requirement_traceability"]["complete"]=False
        path.write_bytes(jbytes(report))
        row["audit_report_sha256"]=sha(path.read_bytes())
        self.fixture.write_manifest()
        result=self.fixture.summarize()
        self.assertFalse(result["cohort_integrity_valid"])
        self.assertIn("audit_eligibility_inconsistent",{item["code"] for item in result["issues"]})

    def test_candidate_failures_are_measured_not_cohort_integrity_errors(self)->None:
        result=self.fixture.summarize()
        self.assertTrue(result["cohort_integrity_valid"],result["issues"])
        self.assertFalse(result["audit_eligible_task_pct"]==100.0)
        self.assertEqual(result["audit_eligible_task_pct"],50.0)

    def test_cohort_manifest_must_match_external_digest(self)->None:
        result=self.fixture.summarize(expected_cohort_sha256="f"*64)
        self.assertFalse(result["cohort_integrity_valid"])
        self.assertIn("cohort_pin_mismatch",{row["code"] for row in result["issues"]})

    def test_audit_report_digest_tampering_is_rejected(self)->None:
        path,report,row=self.fixture.reports[0]
        report["eligible"]=False
        path.write_bytes(jbytes(report))
        result=self.fixture.summarize()
        self.assertFalse(result["cohort_integrity_valid"])
        self.assertIn("audit_report_digest_mismatch",{row["code"] for row in result["issues"]})

    def test_audit_report_receipt_pin_mismatch_is_rejected(self)->None:
        path,report,row=self.fixture.reports[0]
        report["receipt_sha256"]="9"*64
        path.write_bytes(jbytes(report))
        self.fixture.manifest["task_results"][0]["audit_report_sha256"]=sha(path.read_bytes())
        self.fixture.write_manifest()
        result=self.fixture.summarize()
        self.assertFalse(result["cohort_integrity_valid"])
        self.assertIn("receipt_sha256_mismatch",{row["code"] for row in result["issues"]})

    def test_missing_report_does_not_shrink_task_denominator(self)->None:
        self.fixture.manifest["task_results"].pop()
        self.fixture.write_manifest()
        result=self.fixture.summarize()
        self.assertFalse(result["cohort_integrity_valid"])
        self.assertEqual(result["expected_task_count"],2)
        self.assertEqual(result["verified_report_count"],1)
        self.assertEqual(result["audit_eligible_task_pct"],50.0)
        self.assertEqual(result["pooled_metrics_population"],"verified_reports_only")
        self.assertFalse(result["pooled_metrics_complete"])

    def test_task_inventory_order_mismatch_is_rejected(self)->None:
        self.fixture.manifest["task_results"].reverse()
        self.fixture.write_manifest()
        result=self.fixture.summarize()
        self.assertFalse(result["cohort_integrity_valid"])
        self.assertIn("task_inventory_mismatch",{row["code"] for row in result["issues"]})

    def test_duplicate_result_task_is_rejected(self)->None:
        self.fixture.manifest["task_results"].append(copy.deepcopy(self.fixture.manifest["task_results"][0]))
        self.fixture.write_manifest()
        result=self.fixture.summarize()
        self.assertFalse(result["cohort_integrity_valid"])
        self.assertIn("duplicate_task_result",{row["code"] for row in result["issues"]})

    def test_unknown_task_result_cannot_replace_a_locked_task(self)->None:
        self.fixture.manifest["task_results"][1]["task_id"]="task-foreign"
        self.fixture.manifest["task_results"][1]["audit_report_sha256"]=sha(self.fixture.reports[1][0].read_bytes())
        self.fixture.write_manifest()
        result=self.fixture.summarize()
        self.assertFalse(result["cohort_integrity_valid"])
        self.assertEqual(result["verified_report_count"],1)
        self.assertEqual(result["expected_task_count"],2)
        self.assertIn("unknown_task_result",{row["code"] for row in result["issues"]})
        self.assertEqual(result["pooled_metrics_population"],"verified_reports_only")

    def test_wrong_trusted_auditor_pin_is_rejected(self)->None:
        result=self.fixture.summarize(expected_auditor_sha256="c"*64)
        self.assertFalse(result["cohort_integrity_valid"])
        self.assertIn("cohort_row_auditor_mismatch",{row["code"] for row in result["issues"]})

    def test_duplicate_json_object_keys_are_rejected(self)->None:
        self.fixture.manifest_path.write_text('{"schema_version":"vleb-admission-cohort-v1","schema_version":"other"}')
        result=self.fixture.summarize()
        self.assertFalse(result["cohort_integrity_valid"])
        self.assertIn("strict_json_invalid",{row["code"] for row in result["issues"]})

    def test_nonstandard_json_constants_are_rejected(self)->None:
        self.fixture.manifest_path.write_text('{"schema_version":"vleb-admission-cohort-v1","bad":NaN}')
        result=self.fixture.summarize()
        self.assertFalse(result["cohort_integrity_valid"])
        self.assertIn("strict_json_invalid",{row["code"] for row in result["issues"]})

    def test_symlink_report_paths_are_rejected(self)->None:
        real_path,report,row=self.fixture.reports[0]
        link=self.fixture.artifacts/"audit"/"alias.json"
        link.symlink_to(real_path)
        self.fixture.manifest["task_results"][0]["audit_report_path"]="audit/alias.json"
        self.fixture.manifest["task_results"][0]["audit_report_sha256"]=sha(real_path.read_bytes())
        self.fixture.write_manifest()
        result=self.fixture.summarize()
        self.assertFalse(result["cohort_integrity_valid"])
        self.assertIn("audit_report_unavailable",{row["code"] for row in result["issues"]})

    def test_schema_itself_is_valid(self)->None:
        schema=json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(self.fixture.manifest)),[])


if __name__=="__main__":
    unittest.main(verbosity=2)
