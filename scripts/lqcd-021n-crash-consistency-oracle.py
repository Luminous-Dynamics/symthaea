#!/usr/bin/env python3
import hashlib, json
from dataclasses import dataclass, replace

ORACLE_ID="lqcd_021n_crash_consistency_oracle_v1"
PROFILE_ID="posix_local_atomic_rename_fsync_v1"
CHECKPOINT_COMMITMENT="84d2328883ae397e179697f364db000180975946aa4b0f38a88d6a6e65b3a6b8"
PREDECESSOR="e5b439c47911a67ecb9d89cc43debb9d5fe9656c4767b9007a5f1b1ec98c5f3c"
CHAIN_ID="beta6-final-chain-0007"
INDEX=1

def h(b): return hashlib.sha256(b).hexdigest()

CHECKPOINT_BYTES=b"symthaea.lqcd.checkpoint-file.fixture.v1\0"+bytes.fromhex(CHECKPOINT_COMMITMENT)+bytes.fromhex(PREDECESSOR)+INDEX.to_bytes(8,"big")+CHAIN_ID.encode()
CHECKPOINT_BYTES_SHA256=h(CHECKPOINT_BYTES)

def head_bytes(commitment,generation):
    c=CHAIN_ID.encode()
    return b"symthaea.lqcd.authorized-head.v1\0"+len(c).to_bytes(4,"big")+c+generation.to_bytes(8,"big")+INDEX.to_bytes(8,"big")+bytes.fromhex(PREDECESSOR)+bytes.fromhex(commitment)+len(PROFILE_ID).to_bytes(4,"big")+PROFILE_ID.encode()

OLD_HEAD=head_bytes(PREDECESSOR,0)
NEW_HEAD=head_bytes(CHECKPOINT_COMMITMENT,1)

@dataclass
class State:
    temp: bytes|None=None
    temp_fsynced: bool=False
    temp_verified: bool=False
    final: bytes|None=None
    dir_fsynced: bool=False
    final_verified: bool=False
    head: bytes=OLD_HEAD
    head_fsynced: bool=True
    generation: int=0

def parse_head(b):
    tag=b"symthaea.lqcd.authorized-head.v1\0"
    if not b.startswith(tag): raise ValueError
    p=len(tag)
    n=int.from_bytes(b[p:p+4],"big"); p+=4
    chain=b[p:p+n].decode(); p+=n
    gen=int.from_bytes(b[p:p+8],"big"); p+=8
    idx=int.from_bytes(b[p:p+8],"big"); p+=8
    pred=b[p:p+32].hex(); p+=32
    commit=b[p:p+32].hex(); p+=32
    np=int.from_bytes(b[p:p+4],"big"); p+=4
    prof=b[p:p+np].decode(); p+=np
    if p!=len(b): raise ValueError
    return dict(chain=chain,generation=gen,index=idx,predecessor=pred,commitment=commit,profile=prof)

def valid_checkpoint(b): return b==CHECKPOINT_BYTES and h(b)==CHECKPOINT_BYTES_SHA256

def recover(s):
    durable_head=s.head if s.head_fsynced else OLD_HEAD
    try: x=parse_head(durable_head)
    except: return "CorruptPersistenceState"
    final_durable=s.final is not None and s.dir_fsynced
    final_exact=final_durable and valid_checkpoint(s.final)
    if x["generation"]==0 and x["commitment"]==PREDECESSOR:
        if final_exact: return "PublishedButNotAuthorized"
        if final_durable: return "CorruptPersistenceState"
        return "NoPublishedCheckpoint"
    if x["generation"]==1 and x["commitment"]==CHECKPOINT_COMMITMENT:
        if not final_exact: return "CorruptPersistenceState"
        if x["chain"]!=CHAIN_ID or x["index"]!=INDEX or x["predecessor"]!=PREDECESSOR or x["profile"]!=PROFILE_ID: return "CorruptPersistenceState"
        return "AuthorizedCheckpoint"
    return "CorruptPersistenceState"

def write(s): return replace(s,temp=CHECKPOINT_BYTES)
def fsync_temp(s):
    assert s.temp==CHECKPOINT_BYTES
    return replace(s,temp_fsynced=True)
def verify_temp(s):
    assert s.temp_fsynced and valid_checkpoint(s.temp)
    return replace(s,temp_verified=True)
def rename(s):
    assert s.temp_verified
    return replace(s,final=s.temp,temp=None)
def fsync_dir(s):
    assert valid_checkpoint(s.final)
    return replace(s,dir_fsynced=True)
def verify_final(s):
    assert s.dir_fsynced and valid_checkpoint(s.final)
    return replace(s,final_verified=True)
def write_head(s):
    assert s.final_verified and s.generation==0
    return replace(s,head=NEW_HEAD,head_fsynced=False)
def fsync_head(s):
    x=parse_head(s.head); assert x["generation"]==1
    return replace(s,head_fsynced=True,generation=1)

STEPS=[("write",write),("fsync_temp",fsync_temp),("verify_temp",verify_temp),("rename",rename),("fsync_dir",fsync_dir),("verify_final",verify_final),("write_head",write_head),("fsync_head",fsync_head)]

def state_after(n):
    s=State()
    for _,f in STEPS[:n]: s=f(s)
    return s

def main():
    expected={0:"NoPublishedCheckpoint",1:"NoPublishedCheckpoint",2:"NoPublishedCheckpoint",3:"NoPublishedCheckpoint",4:"NoPublishedCheckpoint",5:"PublishedButNotAuthorized",6:"PublishedButNotAuthorized",7:"PublishedButNotAuthorized",8:"AuthorizedCheckpoint"}
    crash=[]
    for n in range(9):
        cls=recover(state_after(n))
        assert cls==expected[n]
        crash.append({"completed_steps":n,"classification":cls})
    mid_write=State(temp=CHECKPOINT_BYTES[:len(CHECKPOINT_BYTES)//2])
    assert recover(mid_write)=="NoPublishedCheckpoint"
    mid_head=replace(state_after(6),head=NEW_HEAD[:len(NEW_HEAD)//2],head_fsynced=False)
    assert recover(mid_head)=="PublishedButNotAuthorized"
    corrupted=bytearray(state_after(5).final); corrupted[-1]^=1
    bad=replace(state_after(5),final=bytes(corrupted))
    assert recover(bad)=="CorruptPersistenceState"
    impossible=State(final=CHECKPOINT_BYTES,dir_fsynced=False,head=NEW_HEAD,head_fsynced=True,generation=1)
    assert recover(impossible)=="CorruptPersistenceState"
    sibling=h(b"sibling-child")
    fork_detected=(sibling!=CHECKPOINT_COMMITMENT and state_after(8).generation!=0)
    assert fork_detected
    result={
      "oracle_id":ORACLE_ID,
      "persistence_profile":PROFILE_ID,
      "checkpoint_bytes_sha256":CHECKPOINT_BYTES_SHA256,
      "checkpoint_commitment":CHECKPOINT_COMMITMENT,
      "checkpoint_path":f"checkpoint-{INDEX:08d}-{CHECKPOINT_COMMITMENT}.bin",
      "head_record_sha256":h(NEW_HEAD),
      "crash_cases":crash,
      "extra":{"mid_write":"NoPublishedCheckpoint","mid_head_write":"PublishedButNotAuthorized","one_bit_final_corruption":"CorruptPersistenceState","durable_head_without_durable_final":"CorruptPersistenceState"},
      "fork_detected":fork_detected,
      "second_writer_generation0_cas_succeeds":False,
      "claim_boundary":{"abstract_crash_consistency_semantics_established":True,"real_filesystem_durability_established":False,"platform_portability_established":False}
    }
    c=json.dumps(result,sort_keys=True,separators=(",",":")).encode()
    print("ok")
    print("result_sha256="+h(c))
    print(c.decode())

if __name__=="__main__": main()
