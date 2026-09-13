#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent clock-continuity-policy identity/authority-gap oracle."""
import argparse, hashlib, json

POLICY_D=b"symthaea.trust.clock-continuity-policy.v1\0"
CONTINUITY_D=b"symthaea.fabrication.clock-continuity-digest.v1\0"
PREVIOUS="5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229"
SUCCESSOR="d82fd355f03b961e2ccc52c886e5974084a2f245fea82ea7514432a25c25ab1b"
EXPECTED_POLICY="904ea7a18dc83f5964666cedfd4a7468b0848b3d952baa9b25ce6c6ba9252b06"
EXPECTED_WEAK="6bf922e4e7806fc9928a001928f4bfbf5ad3cdc370a8395e21c6a8c8706c70a6"
EXPECTED_CONTINUITY="ed00444d5c2a47380ac8e26aead198ab632d0aef092165a3d6c2a353d7f01d15"

class Denied(ValueError): pass
def cj(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def lj(v): return json.dumps(v,separators=(",",":"),ensure_ascii=False).encode()
def dh(d,v): return hashlib.sha256(d+cj(v)).hexdigest()

def policy(max_epoch=1,max_gap=10000,max_jump=60000,min_shared=1,shared_alg=True):
    if max_epoch<=0 or max_jump<=0 or min_shared<=0: raise Denied("invalid_policy")
    pre={"schema":"symthaea.trust.clock-continuity-policy.v1",
         "maximum_epoch_step":max_epoch,"maximum_forward_gap_ms":max_gap,
         "maximum_consensus_jump_ms":max_jump,"minimum_shared_sources":min_shared,
         "require_shared_algorithm":bool(shared_alg)}
    return {**pre,"id":dh(POLICY_D,pre)}

def verify(p,forward_gap=320,jump=500,shared_sources=("source-a","source-b"),
           shared_algorithms=("Ed25519","MlDsa65"),previous_epoch=42,successor_epoch=43):
    if successor_epoch<=previous_epoch: raise Denied("epoch_rollback")
    if successor_epoch-previous_epoch>p["maximum_epoch_step"]: raise Denied("epoch_step")
    if forward_gap>p["maximum_forward_gap_ms"]: raise Denied("forward_gap")
    if jump>p["maximum_consensus_jump_ms"]: raise Denied("consensus_jump")
    if len(shared_sources)<p["minimum_shared_sources"]: raise Denied("shared_sources")
    if p["require_shared_algorithm"] and not shared_algorithms: raise Denied("shared_algorithm")
    pre={"schema_version":"symthaea.fabrication.clock-continuity.v1",
         "previous_evidence_digest":list(bytes.fromhex(PREVIOUS)),
         "successor_evidence_digest":list(bytes.fromhex(SUCCESSOR)),
         "previous_epoch":previous_epoch,"successor_epoch":successor_epoch,
         "forward_gap_ms":forward_gap,"consensus_jump_ms":jump,
         "shared_sources":list(shared_sources),"shared_algorithms":list(shared_algorithms),
         "continuity_digest":[0]*32}
    return hashlib.sha256(CONTINUITY_D+lj(pre)).hexdigest()

def deny(fn,why):
    try: fn()
    except Denied as e:
        if why not in str(e): raise AssertionError((why,str(e)))
        return
    raise AssertionError(why)

def run():
    strict=policy(); weak=policy(max_gap=20000)
    assert strict["id"]==EXPECTED_POLICY,(strict["id"],EXPECTED_POLICY)
    assert weak["id"]==EXPECTED_WEAK,(weak["id"],EXPECTED_WEAK)
    assert strict["id"]!=weak["id"]

    # Existing V1 continuity identity does not commit policy identity.
    strict_digest=verify(strict); weak_digest=verify(weak)
    assert strict_digest==EXPECTED_CONTINUITY
    assert weak_digest==EXPECTED_CONTINUITY

    # A transition outside strict policy but inside weak policy distinguishes authority.
    deny(lambda: verify(strict,forward_gap=15000),"forward_gap")
    assert verify(weak,forward_gap=15000)!=EXPECTED_CONTINUITY

    changed_jump=policy(max_jump=120000)
    changed_epoch=policy(max_epoch=2)
    changed_alg=policy(shared_alg=False)
    assert len({strict["id"],weak["id"],changed_jump["id"],changed_epoch["id"],changed_alg["id"]})==5

    return {"continuity_policy_id":strict["id"],"weaker_policy_id":weak["id"],
            "legacy_continuity_id":strict_digest}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--vectors",action="store_true")
    args=ap.parse_args(); v=run()
    if args.self_test:
        for k in sorted(v): print(f"ok {k}={v[k]}")
    else:
        print(json.dumps(v if args.vectors else {"decision":"SelfTest","vectors":v},sort_keys=True,separators=(",",":")))
if __name__=="__main__": main()
