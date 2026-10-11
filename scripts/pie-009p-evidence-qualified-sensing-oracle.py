#!/usr/bin/env python3
"""PIE-009P independent evidence-qualified sensing oracle."""
from dataclasses import dataclass, replace
from enum import Enum
from typing import FrozenSet, Tuple
import hashlib, json

class Reading(str, Enum):
    TRUE="TRUE"; FALSE="FALSE"; UNKNOWN="UNKNOWN"
class FusionStatus(str, Enum):
    VERIFIED_TRUE="VERIFIED_TRUE"; VERIFIED_FALSE="VERIFIED_FALSE"; INDETERMINATE="INDETERMINATE"; CONFLICT="CONFLICT"

@dataclass(frozen=True)
class SensorSpec:
    sensor_id:str; proposition:str; max_age_steps:int; allowed_provenance:FrozenSet[str]
    failure_domain:str; power_domain:str; data_domain:str

@dataclass(frozen=True)
class Observation:
    sensor_id:str; reading:Reading; measured_step:int; calibration_valid_through:int
    self_test_passed:bool; provenance_id:str; topology_version:int

@dataclass(frozen=True)
class FusionPolicy:
    proposition:str; min_votes:int; min_failure_domains:int; min_power_domains:int; min_data_domains:int

@dataclass(frozen=True)
class FusionResult:
    proposition:str; status:FusionStatus; qualified_sensor_ids:Tuple[str,...]
    supporting_sensor_ids:Tuple[str,...]; reasons:Tuple[str,...]

def validate_registry(specs):
    out={}
    for s in specs:
        if not s.sensor_id or s.sensor_id in out: raise ValueError("sensor ids must be unique and nonempty")
        if not s.proposition or s.max_age_steps < 0: raise ValueError("invalid sensor")
        if not s.allowed_provenance: raise ValueError("provenance required")
        if not s.failure_domain or not s.power_domain or not s.data_domain: raise ValueError("domains required")
        out[s.sensor_id]=s
    return out

def validate_observations(registry, observations):
    out={}
    for o in observations:
        if o.sensor_id not in registry or o.sensor_id in out: raise ValueError("invalid or duplicate observation")
        if min(o.measured_step,o.calibration_valid_through,o.topology_version) < 0: raise ValueError("negative metadata")
        out[o.sensor_id]=o
    return out

def qualify(spec, obs, now_step, topology_version):
    reasons=[]
    age=now_step-obs.measured_step
    if age < 0: reasons.append("FUTURE_MEASUREMENT")
    elif age > spec.max_age_steps: reasons.append("STALE_MEASUREMENT")
    if obs.calibration_valid_through < now_step: reasons.append("CALIBRATION_EXPIRED")
    if not obs.self_test_passed: reasons.append("SELF_TEST_FAILED")
    if obs.provenance_id not in spec.allowed_provenance: reasons.append("PROVENANCE_NOT_ALLOWED")
    if obs.topology_version != topology_version: reasons.append("TOPOLOGY_VERSION_MISMATCH")
    return (not reasons, tuple(reasons))

def evidence_digest(observations):
    rows=[{"sensor_id":o.sensor_id,"reading":o.reading.value,"measured_step":o.measured_step,
           "calibration_valid_through":o.calibration_valid_through,"self_test_passed":o.self_test_passed,
           "provenance_id":o.provenance_id,"topology_version":o.topology_version}
          for o in sorted(observations,key=lambda x:x.sensor_id)]
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def fuse(specs, observations, policy, now_step, topology_version):
    if min(policy.min_votes,policy.min_failure_domains,policy.min_power_domains,policy.min_data_domains) < 1:
        raise ValueError("fusion minima must be positive")
    reg=validate_registry(specs); obs=validate_observations(reg,observations)
    relevant=[s for s in reg.values() if s.proposition==policy.proposition]
    if not relevant: raise ValueError("unknown proposition")
    qualified=[]; true_support=[]; false_support=[]; reasons=[]
    for s in sorted(relevant,key=lambda x:x.sensor_id):
        o=obs.get(s.sensor_id)
        if o is None:
            reasons.append(f"{s.sensor_id}:MISSING"); continue
        ok,rs=qualify(s,o,now_step,topology_version)
        if not ok:
            reasons += [f"{s.sensor_id}:{r}" for r in rs]; continue
        qualified.append(s.sensor_id)
        if o.reading==Reading.TRUE: true_support.append(s)
        elif o.reading==Reading.FALSE: false_support.append(s)
    if true_support and false_support:
        support=true_support+false_support
        return FusionResult(policy.proposition,FusionStatus.CONFLICT,tuple(qualified),
                            tuple(sorted(s.sensor_id for s in support)),
                            tuple(["QUALIFIED_SENSOR_CONTRADICTION"]+reasons))
    support=true_support or false_support
    if not support:
        return FusionResult(policy.proposition,FusionStatus.INDETERMINATE,tuple(qualified),(),
                            tuple(["NO_DIRECTIONAL_SUPPORT"]+reasons))
    deficiencies=[]
    if len(support) < policy.min_votes: deficiencies.append("INSUFFICIENT_VOTES")
    if len({s.failure_domain for s in support}) < policy.min_failure_domains: deficiencies.append("INSUFFICIENT_FAILURE_DOMAIN_DIVERSITY")
    if len({s.power_domain for s in support}) < policy.min_power_domains: deficiencies.append("INSUFFICIENT_POWER_DOMAIN_DIVERSITY")
    if len({s.data_domain for s in support}) < policy.min_data_domains: deficiencies.append("INSUFFICIENT_DATA_DOMAIN_DIVERSITY")
    if deficiencies:
        return FusionResult(policy.proposition,FusionStatus.INDETERMINATE,tuple(qualified),
                            tuple(sorted(s.sensor_id for s in support)),tuple(deficiencies+reasons))
    status=FusionStatus.VERIFIED_TRUE if true_support else FusionStatus.VERIFIED_FALSE
    return FusionResult(policy.proposition,status,tuple(qualified),tuple(sorted(s.sensor_id for s in support)),tuple(reasons))

def black_start_decision(gen,tie):
    if FusionStatus.CONFLICT in (gen.status,tie.status): return "BLOCK_CONFLICT"
    if FusionStatus.VERIFIED_FALSE in (gen.status,tie.status): return "BLOCK"
    if gen.status==tie.status==FusionStatus.VERIFIED_TRUE: return "PROCEED"
    return "INDETERMINATE"

def fixture():
    specs=[
        SensorSpec("gen_current","gen_ok",3,frozenset({"cal-lab-a"}),"gen_elec","pwr_a","data_a"),
        SensorSpec("gen_vibration","gen_ok",3,frozenset({"cal-lab-b"}),"gen_mech","pwr_b","data_b"),
        SensorSpec("gen_shadow","gen_ok",3,frozenset({"cal-lab-c"}),"gen_elec","pwr_a","data_a"),
        SensorSpec("tie_aux","tie_closed",2,frozenset({"cal-switch"}),"tie_switch","pwr_a","data_a"),
        SensorSpec("tie_optical","tie_closed",2,frozenset({"cal-optical"}),"tie_camera","pwr_b","data_b"),
    ]
    obs=[
        Observation("gen_current",Reading.TRUE,99,120,True,"cal-lab-a",7),
        Observation("gen_vibration",Reading.TRUE,99,120,True,"cal-lab-b",7),
        Observation("gen_shadow",Reading.TRUE,99,120,True,"cal-lab-c",7),
        Observation("tie_aux",Reading.TRUE,100,120,True,"cal-switch",7),
        Observation("tie_optical",Reading.TRUE,100,120,True,"cal-optical",7),
    ]
    return specs,obs

def self_test():
    specs,obs=fixture(); P=lambda p:FusionPolicy(p,2,2,2,2)
    g=fuse(specs,obs,P("gen_ok"),100,7); t=fuse(specs,obs,P("tie_closed"),100,7)
    assert g.status==FusionStatus.VERIFIED_TRUE and t.status==FusionStatus.VERIFIED_TRUE
    assert black_start_decision(g,t)=="PROCEED"

    stale=[replace(o,measured_step=90) if o.sensor_id=="gen_vibration" else o for o in obs]
    gs=fuse(specs,stale,P("gen_ok"),100,7)
    assert gs.status==FusionStatus.INDETERMINATE and "gen_vibration:STALE_MEASUREMENT" in gs.reasons

    bad=[replace(o,self_test_passed=False) if o.sensor_id=="tie_optical" else o for o in obs]
    assert fuse(specs,bad,P("tie_closed"),100,7).status==FusionStatus.INDETERMINATE

    expired=[replace(o,calibration_valid_through=99) if o.sensor_id=="gen_vibration" else o for o in obs]
    assert "gen_vibration:CALIBRATION_EXPIRED" in fuse(specs,expired,P("gen_ok"),100,7).reasons

    bad_prov=[replace(o,provenance_id="unknown") if o.sensor_id=="tie_optical" else o for o in obs]
    assert "tie_optical:PROVENANCE_NOT_ALLOWED" in fuse(specs,bad_prov,P("tie_closed"),100,7).reasons

    old_topo=[replace(o,topology_version=6) if o.sensor_id=="tie_optical" else o for o in obs]
    assert "tie_optical:TOPOLOGY_VERSION_MISMATCH" in fuse(specs,old_topo,P("tie_closed"),100,7).reasons

    conflict=[replace(o,reading=Reading.FALSE) if o.sensor_id=="gen_vibration" else o for o in obs]
    gc=fuse(specs,conflict,P("gen_ok"),100,7)
    assert gc.status==FusionStatus.CONFLICT and black_start_decision(gc,t)=="BLOCK_CONFLICT"

    unknown=[replace(o,reading=Reading.UNKNOWN) if o.sensor_id=="gen_vibration" else o for o in obs]
    assert fuse(specs,unknown,P("gen_ok"),100,7).status==FusionStatus.INDETERMINATE

    reduced=[o for o in obs if o.sensor_id!="gen_vibration"]
    gd=fuse(specs,reduced,P("gen_ok"),100,7)
    assert "INSUFFICIENT_FAILURE_DOMAIN_DIVERSITY" in gd.reasons
    assert "INSUFFICIENT_POWER_DOMAIN_DIVERSITY" in gd.reasons

    boundary=[replace(o,measured_step=97) if o.sensor_id=="gen_vibration" else o for o in obs]
    assert fuse(specs,boundary,P("gen_ok"),100,7).status==FusionStatus.VERIFIED_TRUE
    assert fuse(specs,boundary,P("gen_ok"),101,7).status==FusionStatus.INDETERMINATE

    assert evidence_digest(obs)==evidence_digest(list(reversed(obs)))
    changed=[replace(o,self_test_passed=False) if o.sensor_id=="gen_current" else o for o in obs]
    assert evidence_digest(changed)!=evidence_digest(obs)

    try:
        validate_registry(specs+[specs[0]]); raise AssertionError("duplicate registry accepted")
    except ValueError: pass
    try:
        validate_observations(validate_registry(specs),obs+[obs[0]]); raise AssertionError("duplicate observation accepted")
    except ValueError: pass
    print("ok")

if __name__=="__main__": self_test()
