#!/usr/bin/env python3
"""Deterministic operational-workflow oracle around a monetary settlement result."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json

class WorkflowState(str, Enum):
    REQUESTED="requested"; COMPLIANCE_PENDING="compliance_pending"; AUTHORIZED="authorized"
    SOURCE_RESERVED="source_reserved"; SETTLEMENT_READY="settlement_ready"; ATOMICALLY_SETTLED="atomically_settled"
    REDEMPTION_PENDING="redemption_pending"; EXTERNALLY_FINALIZED="externally_finalized"
    REJECTED="rejected"; TIMED_OUT="timed_out"; MANUAL_INTERVENTION="manual_intervention"
    RECONCILING="reconciling"; UNRESOLVED="unresolved"

class FailureReason(str, Enum):
    NONE="none"; COMPLIANCE_BLOCK="compliance_block"; APPROVAL_TIMEOUT="approval_timeout"
    OPERATOR_CAPACITY="operator_capacity"; QUOTE_STALE="quote_stale"; SOURCE_RESERVATION_FAILURE="source_reservation_failure"
    SETTLEMENT_REJECTED="settlement_rejected"; TARGET_ISSUER_UNAVAILABLE="target_issuer_unavailable"
    EXTERNAL_SYSTEM_UNAVAILABLE="external_system_unavailable"; BRIDGE_UNAVAILABLE="bridge_unavailable"
    ADAPTER_LIQUIDITY_LIMIT="adapter_liquidity_limit"; ATOMIC_TARGET_UNAVAILABLE="atomic_target_unavailable"
    SOURCE_INSUFFICIENT="source_insufficient"; NO_INTEROPERABILITY_ADAPTER="no_interoperability_adapter"
    DESTINATION_UNAVAILABLE_AFTER_SOURCE_COMMIT="destination_unavailable_after_source_commit"
    OPERATING_WINDOW_CLOSED="operating_window_closed"; MANUAL_FALLBACK_EXPIRED="manual_fallback_expired"
    DUPLICATE_APPROVAL="duplicate_manual_approval"

@dataclass(frozen=True)
class OperationalPolicy:
    compliance_latency:int=2; approval_latency:int=2; reservation_latency:int=1; quote_confirmation_latency:int=1
    issuance_redemption_latency:int=2; external_finalization_latency:int=2; manual_breakpoint_probability_ppm:int=0
    external_system_available:bool=True; target_issuer_available:bool=True; quote_current:bool=True
    fallback_enabled:bool=True; operator_capacity:int=1; operating_window_end:int=100
    asynchronous_delivery_delay:int=0; approval_timeout:int=20; fallback_timeout:int=20
    def validate(self):
        ints=("compliance_latency","approval_latency","reservation_latency","quote_confirmation_latency","issuance_redemption_latency","external_finalization_latency","asynchronous_delivery_delay","approval_timeout","fallback_timeout","operating_window_end")
        if any(getattr(self,k)<0 for k in ints): raise ValueError("negative policy duration")
        if not 0<=self.manual_breakpoint_probability_ppm<=1_000_000: raise ValueError("probability out of range")
        if self.operator_capacity<0: raise ValueError("negative operator capacity")

@dataclass(frozen=True)
class SettlementOutcome:
    status:str; technical_settlement_duration:int|None; requires_redemption:bool=False
    failure_reason:str|None=None; settlement_receipt_digest:str|None=None

@dataclass(frozen=True)
class WorkflowRequest:
    request_id:str; submitted_at:int; amount:int; source_profile_id:str; target_profile_id:str; settlement_adapter_id:str

@dataclass
class Transition:
    tick:int; state:WorkflowState; cause:str

@dataclass
class WorkflowResult:
    request_id:str; policy_digest:str; settlement_receipt_digest:str|None; seed:int
    final_state:WorkflowState; started_at:int; finished_at:int; technical_settlement_duration:int|None
    transitions:list[Transition]; manual_interventions:int; fallback_activations:int
    reconciliation_backlog:int; unresolved:bool; operator_capacity_consumed:int; dependency_count:int
    failure_reason:FailureReason; trace_digest:str
    @property
    def end_to_end_completion_time(self): return self.finished_at-self.started_at if self.final_state is WorkflowState.EXTERNALLY_FINALIZED else None
    @property
    def technical_settlement_time(self): return self.technical_settlement_duration
    @property
    def operational_waiting_time(self):
        total,tech=self.end_to_end_completion_time,self.technical_settlement_time
        return None if total is None or tech is None else max(0,total-tech)
    @property
    def manual_intervention_rate(self): return float(self.manual_interventions>0)
    @property
    def fallback_activation_rate(self): return float(self.fallback_activations>0)
    def as_dict(self):
        return {"request_id":self.request_id,"policy_digest":self.policy_digest,"settlement_receipt_digest":self.settlement_receipt_digest,"seed":self.seed,"final_state":self.final_state.value,"started_at":self.started_at,"finished_at":self.finished_at,"end_to_end_completion_time":self.end_to_end_completion_time,"technical_settlement_time":self.technical_settlement_time,"operational_waiting_time":self.operational_waiting_time,"manual_interventions":self.manual_interventions,"manual_intervention_rate":self.manual_intervention_rate,"fallback_activations":self.fallback_activations,"fallback_activation_rate":self.fallback_activation_rate,"reconciliation_backlog":self.reconciliation_backlog,"unresolved":self.unresolved,"operator_capacity_consumed":self.operator_capacity_consumed,"dependency_count":self.dependency_count,"failure_reason":self.failure_reason.value,"transitions":[{"tick":x.tick,"state":x.state.value,"cause":x.cause} for x in self.transitions],"trace_digest":self.trace_digest}

def _breakpoint_hit(random_namespace,seed,ppm):
    if ppm<=0:return False
    return int.from_bytes(sha256(f"{random_namespace}:{seed}".encode()).digest()[:8],"big")%1_000_000 < ppm

class OperationalWorkflow:
    def __init__(self,policy):
        policy.validate(); self.policy=policy
        self.policy_digest=sha256(json.dumps(policy.__dict__,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    def run(self,request,settlement,*,seed,exogenous_random_namespace=None,compliance_ok=True,duplicate_manual_approval=False):
        if settlement.status not in {"settled","queued","stranded","rejected"}: raise ValueError("unknown settlement status")
        if settlement.technical_settlement_duration is not None and settlement.technical_settlement_duration<0: raise ValueError("negative settlement duration")
        p=self.policy; t=request.submitted_at; tr=[]; manual=fallback=backlog=capacity=0; unresolved=False; reason=FailureReason.NONE
        tech=settlement.technical_settlement_duration; receipt=settlement.settlement_receipt_digest
        def emit(state,cause): tr.append(Transition(t,state,cause))
        def finish(final,when,tech_value):
            payload={"request":request.__dict__,"transitions":[x.__dict__|{"state":x.state.value} for x in tr],"finished_at":when,"technical_settlement_duration":tech_value,"receipt":receipt,"policy_digest":self.policy_digest,"seed":seed,"manual":manual,"fallback":fallback,"backlog":backlog,"unresolved":unresolved,"capacity":capacity,"reason":reason.value}
            digest=sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            return WorkflowResult(request.request_id,self.policy_digest,receipt,seed,tr[-1].state,request.submitted_at,when,tech_value,tr,manual,fallback,backlog,unresolved,capacity,3,reason,digest)
        emit(WorkflowState.REQUESTED,"request_submitted")
        t+=p.compliance_latency+p.asynchronous_delivery_delay; emit(WorkflowState.COMPLIANCE_PENDING,"compliance_check_started")
        if not compliance_ok:
            reason=FailureReason.COMPLIANCE_BLOCK; emit(WorkflowState.REJECTED,reason.value); return finish(WorkflowState.REJECTED,t,None)
        t+=p.approval_latency; emit(WorkflowState.AUTHORIZED,"approval_completed")
        if p.approval_latency>p.approval_timeout or t>p.operating_window_end:
            reason=FailureReason.APPROVAL_TIMEOUT; emit(WorkflowState.TIMED_OUT,reason.value); return finish(WorkflowState.TIMED_OUT,t,None)
        if duplicate_manual_approval:
            reason=FailureReason.DUPLICATE_APPROVAL; emit(WorkflowState.REJECTED,reason.value); return finish(WorkflowState.REJECTED,t,None)
        rng=exogenous_random_namespace or request.request_id
        if _breakpoint_hit(rng,seed,p.manual_breakpoint_probability_ppm):
            if p.operator_capacity<1:
                if p.fallback_enabled: fallback+=1; backlog+=1; unresolved=True; reason=FailureReason.OPERATOR_CAPACITY; emit(WorkflowState.RECONCILING,"manual_breakpoint_without_capacity"); emit(WorkflowState.UNRESOLVED,reason.value); return finish(WorkflowState.UNRESOLVED,t,None)
                reason=FailureReason.OPERATOR_CAPACITY; emit(WorkflowState.TIMED_OUT,reason.value); return finish(WorkflowState.TIMED_OUT,t,None)
            manual+=1; capacity+=1; emit(WorkflowState.MANUAL_INTERVENTION,"manual_breakpoint"); t+=1
        t+=p.reservation_latency; emit(WorkflowState.SOURCE_RESERVED,"source_reservation")
        if t>p.operating_window_end:
            reason=FailureReason.OPERATING_WINDOW_CLOSED; emit(WorkflowState.TIMED_OUT,reason.value); return finish(WorkflowState.TIMED_OUT,t,None)
        t+=p.quote_confirmation_latency; emit(WorkflowState.SETTLEMENT_READY,"quote_and_route_confirmed")
        if not p.quote_current:
            reason=FailureReason.QUOTE_STALE
            if p.fallback_enabled: fallback+=1; backlog+=1; unresolved=True; emit(WorkflowState.RECONCILING,"stale_quote_requires_requote"); emit(WorkflowState.UNRESOLVED,reason.value); return finish(WorkflowState.UNRESOLVED,t,None)
            emit(WorkflowState.REJECTED,reason.value); return finish(WorkflowState.REJECTED,t,None)
        if settlement.status=="rejected":
            try: reason=FailureReason(settlement.failure_reason or FailureReason.SETTLEMENT_REJECTED.value)
            except ValueError: reason=FailureReason.SETTLEMENT_REJECTED
            if p.fallback_enabled: fallback+=1; backlog+=1; unresolved=True; emit(WorkflowState.RECONCILING,reason.value); emit(WorkflowState.UNRESOLVED,reason.value); return finish(WorkflowState.UNRESOLVED,t,None)
            emit(WorkflowState.REJECTED,reason.value); return finish(WorkflowState.REJECTED,t,None)
        if settlement.status=="stranded":
            reason=FailureReason.TARGET_ISSUER_UNAVAILABLE
            if p.fallback_enabled and p.target_issuer_available:
                fallback+=1; backlog+=1; t+=p.issuance_redemption_latency; emit(WorkflowState.RECONCILING,"stranded_settlement"); emit(WorkflowState.ATOMICALLY_SETTLED,"recovery_redeemed")
            else:
                fallback+=int(p.fallback_enabled); backlog+=1; unresolved=True; emit(WorkflowState.RECONCILING,"stranded_settlement"); emit(WorkflowState.UNRESOLVED,reason.value); return finish(WorkflowState.UNRESOLVED,t,None)
        elif settlement.status=="queued":
            t+=2; emit(WorkflowState.SETTLEMENT_READY,"net_settlement_queue_wait")
            if t>p.operating_window_end:
                reason=FailureReason.OPERATING_WINDOW_CLOSED
                if p.fallback_enabled: fallback+=1; backlog+=1; unresolved=True; emit(WorkflowState.RECONCILING,reason.value); emit(WorkflowState.UNRESOLVED,reason.value); return finish(WorkflowState.UNRESOLVED,t,None)
                emit(WorkflowState.TIMED_OUT,reason.value); return finish(WorkflowState.TIMED_OUT,t,None)
        else:
            t+=tech or 0; emit(WorkflowState.ATOMICALLY_SETTLED,"settlement_completed")
        if settlement.requires_redemption:
            emit(WorkflowState.REDEMPTION_PENDING,"redemption_workflow"); t+=p.issuance_redemption_latency
            if not p.target_issuer_available:
                reason=FailureReason.TARGET_ISSUER_UNAVAILABLE; unresolved=True; emit(WorkflowState.UNRESOLVED,reason.value); return finish(WorkflowState.UNRESOLVED,t,tech)
        t+=p.external_finalization_latency
        if not p.external_system_available:
            reason=FailureReason.EXTERNAL_SYSTEM_UNAVAILABLE
            if p.fallback_enabled: fallback+=1; backlog+=1; unresolved=True; emit(WorkflowState.RECONCILING,reason.value); emit(WorkflowState.UNRESOLVED,FailureReason.MANUAL_FALLBACK_EXPIRED.value); return finish(WorkflowState.UNRESOLVED,t,tech)
            emit(WorkflowState.TIMED_OUT,reason.value); return finish(WorkflowState.TIMED_OUT,t,tech)
        emit(WorkflowState.EXTERNALLY_FINALIZED,"workflow_complete")
        return finish(WorkflowState.EXTERNALLY_FINALIZED,t,tech)

def demo():
    p=OperationalPolicy(); receipt=sha256(b"demo-settlement-receipt").hexdigest()
    r=OperationalWorkflow(p).run(WorkflowRequest("demo-1",0,10,"tokenized-deposit-v1","reserve-backed-stablecoin-v1","atomic-1"),SettlementOutcome("settled",2,False,settlement_receipt_digest=receipt),seed=7)
    assert r.final_state is WorkflowState.EXTERNALLY_FINALIZED and r.technical_settlement_time==2 and r.operational_waiting_time==8
    return r.as_dict()

if __name__=="__main__": print(json.dumps(demo(),indent=2,sort_keys=True))
