#!/usr/bin/env python3
"""Independent checker for the monetary operational-workflow research contract."""
from __future__ import annotations
import json,sys
from pathlib import Path
REQUIRED_STATES={"requested","compliance_pending","authorized","source_reserved","settlement_ready","atomically_settled","redemption_pending","externally_finalized","rejected","timed_out","manual_intervention","reconciling","unresolved"}
REQUIRED_METRICS={"end_to_end_completion_time","technical_settlement_time","operational_waiting_time","manual_intervention_rate","fallback_activation_rate","reconciliation_backlog","dependency_count"}
REQUIRED_BINDING=["scenario_id","policy_digest","settlement_receipt_digest","seed","trace_digest"]
def fail(msg): raise ValueError(msg)
def check_schema(schema):
    if schema.get("schema_version")!="monetary-operational-workflow-v1": fail("schema version")
    if set(schema.get("states",[]))!=REQUIRED_STATES: fail("workflow state set")
    if set(schema.get("metrics",[]))!=REQUIRED_METRICS: fail("metric set")
    if schema.get("observation_binding",{}).get("required_fields")!=REQUIRED_BINDING: fail("observation binding")
    factors=schema.get("experiment_factors")
    required={"compliance_latency","approval_latency","manual_breakpoint_probability","external_system_availability","issuance_redemption_latency","quote_confirmation_latency","fallback_activation","operator_capacity","operating_window","asynchronous_delivery"}
    if not isinstance(factors,list) or {f["id"] for f in factors}!=required: fail("experiment factor set")
def check_fixtures(fixtures):
    if fixtures.get("schema_version")!="monetary-operational-workflow-negative-v1": fail("fixture schema version")
    cases=fixtures.get("cases"); expected={f"OP-X{i:02d}" for i in range(1,12)}
    if not isinstance(cases,list) or {c.get("id") for c in cases}!=expected: fail("fixture ID set")
    if any(not isinstance(c.get("expected"),str) or not c["expected"] for c in cases): fail("missing expected")
def main():
    if len(sys.argv)!=3: print("usage: verify_monetary_operational_workflow.py SCHEMA.json NEGATIVE.json",file=sys.stderr); return 2
    try: check_schema(json.loads(Path(sys.argv[1]).read_text())); check_fixtures(json.loads(Path(sys.argv[2]).read_text()))
    except (OSError,json.JSONDecodeError,ValueError) as exc: print(f"verification failed: {exc}",file=sys.stderr); return 1
    print("independent operational-workflow check: state/metric/factor/binding contract valid; 11 negative fixtures"); return 0
if __name__=="__main__": raise SystemExit(main())
