#!/usr/bin/env python3
"""Independent fixture-level checks for SAFRB evidence-acquisition semantics.

This is not the production Symthaea policy or a VoI/EVSI implementation.
"""
import copy, json, sys
from pathlib import Path

SINGLE="0a1c507d3bff9b3062e24ca8718e102244473d839c2a31c5ead31812ee249be2"
TEMPORAL="edfd1c2a5a3eb52a0026a0e63abb38ab2ca8109c4de3d96fbff1215ad78f0308"

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def verify(doc):
    require(doc["fixture_id"]=="SYMTHAEA-SAFRB-PUMP-VOI-001","fixture identity")
    require(doc["maturity"]=="synthetic_test_fixture","fixture maturity")
    scenarios={s["case_id"]:s["scenario_sha256"] for s in doc["source_scenarios"]}
    require(scenarios.get("SAFRB-SHARED-PUMP-2ROUTE-001")==SINGLE,"single-window digest mismatch")
    require(scenarios.get("SAFRB-TWO-WINDOW-SHARED-PUMP-001")==TEMPORAL,"two-window digest mismatch")
    decision=doc["decision"]
    require(decision["current_disposition"]=="indeterminate_not_guaranteed","initial disposition promotion")
    require(decision["explicit_non_goals"]==["recompute-network-flow","authorize-inspection","reserve-pump","dispatch-fuel","modify-inventory"],"authority boundary changed")
    require(decision["lower_bound_horizon_flow"]=={"quantity":9,"unit":"t"},"time-expanded lower-bound horizon total mismatch")
    require(decision["upper_bound_horizon_flow"]=={"quantity":12,"unit":"t"},"time-expanded upper-bound horizon total mismatch")
    windows={w["window"]:w for w in decision["time_window_obligations"]}
    require(set(windows)=={"[1,2)","[2,3)"},"service windows missing/merged")
    require(windows["[1,2)"]=={"window":"[1,2)","required":6,"lower_bound_delivered":6,"upper_bound_delivered":6,"disposition":"guaranteed_within_synthetic_fixture"},"window 1 service semantics changed")
    require(windows["[2,3)"]=={"window":"[2,3)","required":6,"lower_bound_delivered":3,"upper_bound_delivered":6,"disposition":"indeterminate_not_guaranteed"},"window 2 service semantics changed")
    models=doc["admissible_model_family"]
    require(len(models)>=2 and len({m["model_id"] for m in models})==len(models),"model family identity")
    require(all(m["prior_weight"] is None for m in models),"unjustified model weights introduced")
    actions={a["action_id"]:a for a in doc["candidate_investigations"]}
    require(len(actions)==len(doc["candidate_investigations"]),"duplicate investigation identity")
    independent=actions["independent-pump-throughput-test"]
    require(independent["scenario_case_id"]=="SAFRB-SHARED-PUMP-2ROUTE-001","single-window test scoped to wrong case")
    require(independent["measurement_origin"]=="independent_calibrated_test_rig","independent pump test lost its source independence")
    require(independent["can_change_physical_deliverability_decision"] is True,"decision-relevant test demoted")
    dashboard=actions["second-scada-dashboard-read"]
    require(dashboard["scenario_case_id"]=="SAFRB-SHARED-PUMP-2ROUTE-001","dashboard case scope")
    require(dashboard["measurement_origin"]=="same_sensor_lineage_as_existing_report","same-source dashboard independence promotion")
    require(dashboard["can_change_physical_deliverability_decision"] is False,"redundant dashboard given decision effect")
    gauge=actions["independent-tank-gauge"]
    require(gauge["can_change_current_pump_bottleneck_decision"] is False,"non-binding tank gauge treated as resolving pump bottleneck")
    title=actions["title-custody-contract-review"]
    require(title["can_change_access_eligibility"] is True and title["authority_effect"]=="no_dispatch_authority","access review promoted to dispatch authority")
    w2=actions["operation-window-two-independent-throughput-test"]
    require(w2["scenario_case_id"]=="SAFRB-TWO-WINDOW-SHARED-PUMP-001","window-two test scope mismatch")
    require(w2["targets"]==["pump-throughput-lower-bound-window-2"],"window-two test targets wrong proof obligation")
    require(w2["possible_outcomes"]==["qualified_lower_bound_gte_6_t","qualified_lower_bound_lt_6_t","inconclusive"],"window-two test outcome contract changed")
    require(w2["elapsed_time_class"]=="before-window-2-decision-deadline","test can finish after decision deadline")
    require(all(a["permission"] for a in actions.values()),"investigation permission missing")
    require(all(a["authority_effect"] in {"none","no_dispatch_authority"} for a in actions.values()),"investigation gained operational authority")
    gates=set(doc["hard_gates"])
    for item in ["no-unweighted-model-averaging","unknown-measurement-dependence-is-not-independence",
                 "investigation-result-does-not-auto-admit-evidence",
                 "do-not-aggregate-horizon-flow-to-hide-a-time-window-service-floor-breach",
                 "do-not-allow-later-window-observations-to-backdate-earlier-service"]:
        require(item in gates,f"required policy gate missing: {item}")
    return True

def main():
    path=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name("evidence-acquisition-known-answer.json")
    doc=json.loads(path.read_text(encoding="utf-8"))
    verify(doc)
    attacks=[]
    a=copy.deepcopy(doc); a["admissible_model_family"][0]["prior_weight"]=0.5; attacks.append(a)
    a=copy.deepcopy(doc); a["candidate_investigations"][1]["measurement_origin"]="independent_sensor"; attacks.append(a)
    a=copy.deepcopy(doc); a["candidate_investigations"][2]["can_change_current_pump_bottleneck_decision"]=True; attacks.append(a)
    a=copy.deepcopy(doc); a["candidate_investigations"][4]["elapsed_time_class"]="after-window-2-decision-deadline"; attacks.append(a)
    a=copy.deepcopy(doc); a["decision"]["time_window_obligations"][1]["lower_bound_delivered"]=6; a["decision"]["time_window_obligations"][1]["disposition"]="guaranteed_within_synthetic_fixture"; attacks.append(a)
    a=copy.deepcopy(doc); a["candidate_investigations"][4]["authority_effect"]="dispatch"; attacks.append(a)
    a=copy.deepcopy(doc); a["hard_gates"].remove("do-not-aggregate-horizon-flow-to-hide-a-time-window-service-floor-breach"); attacks.append(a)
    rejected=0
    for attack in attacks:
        try: verify(attack)
        except (AssertionError,KeyError,TypeError): rejected+=1
    require(rejected==len(attacks),"a decision/authority/model mutation escaped the fixture verifier")
    print("PASS: frozen source digests; separate window floors; unweighted models; investigation authority/source semantics; 7 hostile mutations rejected")

if __name__=="__main__":
    main()
