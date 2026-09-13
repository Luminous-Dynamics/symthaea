#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent interval-safe governance-time reference over OperationalClockBasisV1."""
import argparse, hashlib, json

DOMAIN=b"symthaea.trust.clock-governance-evaluation-envelope.v1\0"
SCHEMA="symthaea.trust.clock-governance-evaluation-envelope.v1"

EXPECTED={
 "operational_basis_id":"f9d30baa62bb317c52393f0fa181c8c0ceabe77bbe06d16ca8dc79e10a51443b",
 "clock_window_evidence_digest":"5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229",
 "trust_snapshot_digest":"609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633",
 "envelope_id":"afbbee8cf9de3d39600c3d3a51901e3563467b13be664a681e4b5cd84c90082d",
}

class Denied(ValueError): pass
def cj(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def dh(v): return hashlib.sha256(DOMAIN+cj(v)).hexdigest()

def envelope():
    pre={
      "schema":SCHEMA,
      "operational_basis_id":EXPECTED["operational_basis_id"],
      "clock_window_evidence_digest":EXPECTED["clock_window_evidence_digest"],
      "trust_snapshot_digest":EXPECTED["trust_snapshot_digest"],
      "epoch":42,
      "lower_unix_ms":1_499_920,
      "upper_unix_ms":1_500_100,
      "consensus_unix_ms":1_500_010,
    }
    return {**pre,"id":dh(pre)}

def validate(e):
    pre={k:e[k] for k in (
      "schema","operational_basis_id","clock_window_evidence_digest","trust_snapshot_digest",
      "epoch","lower_unix_ms","upper_unix_ms","consensus_unix_ms")}
    if e.get("schema")!=SCHEMA or e.get("id")!=dh(pre): raise Denied("envelope_identity")
    if not (e["epoch"]>0 and e["lower_unix_ms"]<=e["consensus_unix_ms"]<=e["upper_unix_ms"]):
        raise Denied("envelope_shape")
    for k in ("operational_basis_id","clock_window_evidence_digest","trust_snapshot_digest"):
        if len(e[k])!=64 or int(e[k],16)==0: raise Denied("envelope_digest")
    return e

def seconds_to_ms(value):
    if value < 0 or value > (2**64-1)//1000: raise Denied("time_scale_overflow")
    return value*1000

def valid_across_window(not_before_s, not_after_s, e):
    validate(e)
    lo=seconds_to_ms(not_before_s); hi=seconds_to_ms(not_after_s)
    if lo>e["lower_unix_ms"] or hi<=e["upper_unix_ms"]:
        raise Denied("not_valid_across_window")
    return True

def activation_safe(activates_at_s, maximum_delay_s, e):
    validate(e)
    activation=seconds_to_ms(activates_at_s)
    delay=seconds_to_ms(maximum_delay_s)
    latest=e["lower_unix_ms"]+delay
    if latest>2**64-1: raise Denied("activation_overflow")
    # For every possible true now in [lower, upper]:
    # activation must not be in the past, and must be within max delay.
    if activation < e["upper_unix_ms"]: raise Denied("activation_may_be_past")
    if activation > latest: raise Denied("activation_may_be_too_late")
    return True

def approval_safe(issued_at_s, expires_at_s, e):
    if issued_at_s>=expires_at_s: raise Denied("approval_window")
    return valid_across_window(issued_at_s,expires_at_s,e)

def snapshot_safe(issued_at_s, expires_at_s, e):
    if issued_at_s>=expires_at_s: raise Denied("snapshot_window")
    return valid_across_window(issued_at_s,expires_at_s,e)

def key_safe(not_before_s, not_after_s, active, usage_allowed, e):
    if not active: raise Denied("key_not_active")
    if not usage_allowed: raise Denied("key_usage")
    return valid_across_window(not_before_s,not_after_s,e)

def deny(fn,why):
    try: fn()
    except Denied as exc:
        if why not in str(exc): raise AssertionError((why,str(exc)))
    else: raise AssertionError(f"expected denial: {why}")

def self_test():
    e=envelope(); validate(e)
    assert e["id"]==EXPECTED["envelope_id"]

    # Entire-window temporal eligibility.
    approval_safe(1499,1501,e)
    snapshot_safe(1000,2000,e)
    key_safe(900,3000,True,True,e)

    # Safe activation must be future-safe at upper bound and delay-safe at lower bound.
    activation_safe(1501,10,e)

    deny(lambda: approval_safe(1499,1500,e),"not_valid_across_window")
    deny(lambda: snapshot_safe(1000,1500,e),"not_valid_across_window")
    deny(lambda: key_safe(1500,3000,True,True,e),"not_valid_across_window")
    deny(lambda: key_safe(900,3000,False,True,e),"key_not_active")
    deny(lambda: key_safe(900,3000,True,False,e),"key_usage")
    deny(lambda: activation_safe(1500,10,e),"activation_may_be_past")
    deny(lambda: activation_safe(1510,10,e),"activation_may_be_too_late")

    # Consensus-only validation would accept these two unsafe cases:
    assert 1_500_000 <= e["consensus_unix_ms"] < 1_501_000
    assert e["consensus_unix_ms"] < 1_510_000
    return {
      "envelope_id":e["id"],
      "lower_unix_ms":e["lower_unix_ms"],
      "upper_unix_ms":e["upper_unix_ms"],
      "consensus_unix_ms":e["consensus_unix_ms"],
    }

def main():
    p=argparse.ArgumentParser(); p.add_argument("--self-test",action="store_true")
    ns=p.parse_args()
    if ns.self_test: print(json.dumps(self_test(),sort_keys=True))
    else: print(json.dumps(EXPECTED,sort_keys=True,indent=2))

if __name__=="__main__": main()
