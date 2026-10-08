#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
REQUIRED_STATES={"requested","compliance_pending","authorized","source_reserved","settlement_ready","atomically_settled","redemption_pending","externally_finalized","rejected","timed_out","manual_intervention","reconciling","unresolved"}
REQUIRED_METRICS={"end_to_end_completion_time","technical_settlement_time","operational_waiting_time","manual_intervention_rate","fallback_activation_rate","reconciliation_backlog","dependency_count"}
REQUIRED_BINDING=["scenario_id","policy_digest","settlement_receipt_digest","seed","trace_digest"]
def fail(m): raise ValueError(m)
def main():
    if len(sys.argv)!=3:
        print("usage: verify_monetary_operational_workflow.py SCHEMA.json NEGATIVE.json",file=sys.stderr); return 2
    try:
        schema=json.loads(Path(sys.argv[1]).read_text()); fixtures=json.loads(Path(sys.argv[2]).read_text())
        if schema.get("schema_version")!="monetary-operational-workflow-v1": fail("schema version")
        if set(schema.get("states",[]))!=REQUIRED_STATES: fail("workflow state set")
        if set(schema.get("metrics",[]))!=REQUIRED_METRICS: fail("metric set")
        if schema.get("observation_binding",{}).get("required_fields")!=REQUIRED_BINDING: fail("observation binding")
        if schema.get("exogenous_random_namespace")!="world:{shock}:{seed}:obligation:{index}": fail("CRN namespace")
        if fixtures.get("schema_version")!="monetary-operational-workflow-negative-v1": fail("fixture schema version")
        cases=fixtures.get("cases"); expected={f"OP-X{i:02d}" for i in range(1,13)}
        if not isinstance(cases,list) or {c.get("id") for c in cases}!=expected: fail("fixture ID set")
        if any(not isinstance(c.get("expected"),str) or not c["expected"] for c in cases): fail("missing expected")
        print("independent operational-workflow check: state/metric/factor/binding/CRN contract valid; 12 negative fixtures"); return 0
    except (OSError,json.JSONDecodeError,ValueError) as exc: print(f"verification failed: {exc}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
