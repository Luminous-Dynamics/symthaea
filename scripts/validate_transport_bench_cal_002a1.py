#!/usr/bin/env python3
import copy, hashlib, json, subprocess, sys
from pathlib import Path

SOURCE_HEAD = "92990d2313743be501adfda2aa9200846c903650"
SOURCE_PARENT = "fbd7a754ea8389ca93f7680d93ed8b48553e6376"
DATA_SHA256 = "4c9ca2276101407960836ac81e19f9d78f7d263bdfa33c71c7d6eadbef887d68"
SCHEMA = "transport-bench-cal-002-ledger-v1"
CLAIM_CEILING = "DecisionBoundBenchmarkCalibrationMetadataOnly"
STATES = ["ObservedOnly","Calibrating","ConditionallyDecisionRelevant","TransferFailed","OutOfProfile","Expired","Retired"]
REQUIRED = {"external_registry_generation","benchmark_entry_id","benchmark_version","task","split","evaluator","symthaea_subject_generation","odd_profile","decision_proposition","transfer_hypothesis","independent_transfer_target","common_mode_roots","baseline_identity","ablation_identities","development_exposure","confirmatory_exposure","rights_use_disposition","resource_context","currentness_conditions","claim_ceiling"}
PROSPECTIVE = {"hypothesis_id","predicted_observable","expected_direction","uncertainty_interval","benchmark_subject","decision_proposition","independent_target","common_mode_roots","development_exposure","confirmatory_exposure","baseline","ablations"}
OUTCOME = {"observed_result","transfer_class","direction_correct","effect_estimate","interval_coverage","failure_mode","applicability_boundary","decision_consequence"}
MUTATIONS = {"benchmark_pass_to_decision_relevance","post_result_hypothesis_change","post_result_confirmatory_target_selection","silent_benchmark_version_change","evaluator_substitution","common_mode_root_removal","physical_transfer_failure_omission","retroactive_out_of_profile_laundering","expired_to_current","rights_change_ignored","model_generation_change_without_requalification","odd_expansion_without_transfer","score_to_authority","single_sample_to_mode_wide_claim","evidence_state_to_actuator_authority","historical_record_overwrite","baseline_removal_after_result","redundancy_as_independence","decision_consequence_without_new_hypothesis","uncertainty_collapse"}

def canon(x): return json.dumps(x,separators=(",",":"),ensure_ascii=False).encode()

def validate(d, check_digest=True):
    top={"schema","issue","status","registry_dependencies","evidence_use_states","required_bindings","prospective_fields","outcome_fields","records","mutation_requirements","claim_ceiling"}
    if set(d)!=top: raise ValueError("closed-world top-level schema")
    if d["schema"]!=SCHEMA or d["issue"]!=6277 or d["status"]!="SourceContractOnly": raise ValueError("identity/status")
    if d["evidence_use_states"]!=STATES: raise ValueError("state vocabulary")
    if set(d["required_bindings"])!=REQUIRED: raise ValueError("required vocabulary")
    if set(d["prospective_fields"])!=PROSPECTIVE or set(d["outcome_fields"])!=OUTCOME: raise ValueError("field vocabulary")
    if d["records"]!=[]: raise ValueError("pre-execution ledger must be empty")
    if set(d["mutation_requirements"])!=MUTATIONS: raise ValueError("mutation vocabulary")
    if d["claim_ceiling"]!=CLAIM_CEILING: raise ValueError("claim ceiling")
    if d["registry_dependencies"] != [{"issue":6255,"role":"external benchmark architecture"},{"issue":6264,"role":"benchmark-to-physical transfer calibration"},{"issue":6270,"role":"minimal complementary portfolio"}]: raise ValueError("dependency binding")
    if check_digest and hashlib.sha256(canon(d)).hexdigest()!=DATA_SHA256: raise ValueError("digest")
    return True

def reject(base, name, fn):
    x=copy.deepcopy(base); fn(x)
    try: validate(x,check_digest=False)
    except (AssertionError,KeyError,TypeError,ValueError): return
    raise AssertionError("accepted mutation: "+name)

def mutations(base):
    cases=[
      ("schema",lambda x:x.__setitem__("schema","x")),
      ("issue",lambda x:x.__setitem__("issue",1)),
      ("status",lambda x:x.__setitem__("status","Qualified")),
      ("state added",lambda x:x["evidence_use_states"].append("Approved")),
      ("state reordered",lambda x:x["evidence_use_states"].reverse()),
      ("required removed",lambda x:x["required_bindings"].pop()),
      ("required authority added",lambda x:x["required_bindings"].append("actuator_authority")),
      ("prospective outcome merge",lambda x:x["prospective_fields"].append("observed_result")),
      ("outcome prospective merge",lambda x:x["outcome_fields"].append("hypothesis_id")),
      ("record before execution",lambda x:x["records"].append({"state":"ConditionallyDecisionRelevant"})),
      ("mutation removed",lambda x:x["mutation_requirements"].pop()),
      ("mutation invented",lambda x:x["mutation_requirements"].append("benchmark_rank")),
      ("claim escalation",lambda x:x.__setitem__("claim_ceiling","OperationAuthority")),
      ("approved state",lambda x:x["evidence_use_states"].__setitem__(2,"Approved")),
      ("decision authorized state",lambda x:x["evidence_use_states"].__setitem__(1,"DecisionAuthorized")),
      ("dependency removed",lambda x:x["registry_dependencies"].pop()),
      ("dependency role drift",lambda x:x["registry_dependencies"][0].__setitem__("role","safety authority")),
      ("uncertainty score",lambda x:x["outcome_fields"].append("benchmark_trust_score")),
      ("regulatory field",lambda x:x["required_bindings"].append("regulatory_approval")),
      ("history overwrite",lambda x:x["required_bindings"].append("replace_prior_generation")),
      ("rights bypass",lambda x:x["required_bindings"].append("rights_assumed")),
      ("common-mode bypass",lambda x:x["required_bindings"].append("independence_score")),
    ]
    for n,f in cases: reject(base,n,f)
    return len(cases)

def git(*args): return subprocess.check_output(["git",*args],text=True).strip()

def main():
    data=json.loads(Path("docs/engineering/data/transport_bench_cal_002_ledger_v1.json").read_bytes())
    validate(data)
    doc=Path("docs/engineering/TRANSPORT_BENCH_CAL_002.md").read_text(encoding="utf-8")
    if "NaN" in doc: raise ValueError("NaN serialization defect")
    theorem="benchmark result != decision permission != permanent evidence validity != transfer validity outside its calibrated profile"
    if theorem not in doc: raise ValueError("core theorem mismatch")
    if subprocess.call(["git","merge-base","--is-ancestor",SOURCE_HEAD,"HEAD"]) != 0: raise ValueError("source head is not an ancestor of qualifier head")
    if git("rev-parse","HEAD")==SOURCE_HEAD: raise ValueError("qualifier has no independent changes")
    if git("rev-parse",SOURCE_HEAD+"^")!=SOURCE_PARENT: raise ValueError("source parent mismatch")
    paths=sorted(git("diff","--name-only",SOURCE_PARENT,QUALIFIER_HEAD).splitlines())
    expected=sorted(["docs/engineering/TRANSPORT_BENCH_CAL_002.md","docs/engineering/data/transport_bench_cal_002_ledger_v1.json",".github/workflows/transport-bench-cal-002a1-qualifier.yml","scripts/validate_transport_bench_cal_002a1.py"])
    if paths!=expected: raise ValueError("source scope drift: "+repr(paths))
    n=mutations(data)
    print(f"PASS: {SCHEMA}; 2 source files; {n} hostile mutations rejected; digest={DATA_SHA256}")

if __name__=="__main__": main()
