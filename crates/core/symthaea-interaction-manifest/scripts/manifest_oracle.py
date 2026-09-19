#!/usr/bin/env python3
"""Independent SHA-256 oracle for INTX-009 connector-manifest v1."""
from __future__ import annotations
import hashlib
import struct

S = 1
CON = b"symthaea.interaction.connector.v1\0"
OP = b"symthaea.interaction.operation.v1\0"
RF = b"symthaea.interaction.manifest.resource-family.v1\0"
PRO = b"symthaea.interaction.manifest.protocol.v1\0"
OBS = b"symthaea.interaction.manifest.observation.v1\0"
EFF = b"symthaea.interaction.manifest.effect.v1\0"
AUTH = b"symthaea.interaction.manifest.auth.v1\0"
REC = b"symthaea.interaction.manifest.reconciliation.v1\0"
MAN = b"symthaea.interaction.connector-manifest.v1\0"
EXPECTED_CONNECTOR = "8848e17021ae7a97d780efacf7f74f45b7dd3e3c40b6f3a3125c219b7a453f1f"
EXPECTED_MANIFEST = "71999264383969516149738867ad87fdb16947e8fdb0f0c6043b2136f69d4a0e"

def u8(v): return bytes([v])
def u16(v): return struct.pack(">H", v)
def u32(v): return struct.pack(">I", v)
def text(v):
    b = v.encode("ascii")
    return u32(len(b)) + b
def ot(v): return b"\x00" if v is None else b"\x01" + text(v)
def od(v): return b"\x00" if v is None else b"\x01" + v
def h(domain, payload): return hashlib.sha256(domain + payload).digest()

def connector(): return h(CON, u16(S)+text("mycelix/holochain")+text("native")+text("0.7")+od(bytes([0x11])*32))
def operation(name, profile): return h(OP, u16(S)+text("mycelix/holochain")+text(name)+ot(profile))
def family(): return h(RF, u16(S)+text("mycelix/holochain")+text("zome-function"))
def protocol(): return h(PRO, u16(S)+text("holochain")+text("0.7.0"))
def observation(): return h(OBS, u16(S)+family()+operation("signal","holochain-signal")+u16(2)+u16(2)+u16(3)+u16(3)+u16(1)+u16(0)+u8(1)+u8(1))
def effect(): return h(EFF, u16(S)+family()+operation("call","zome")+u16(2)+u16(3)+u16(2)+u16(2)+u16(1)+u8(1))
def auth(): return h(AUTH, u16(S)+text("holochain-capability")+u16(0)+u16(0)+ot("cell-zome-fn"))
def reconciliation(): return h(REC, u16(S)+u8(0)+u8(1)+u8(0)+u8(1)+u8(1)+u8(0)+u16(1))
def manifest():
    obs = sorted([observation()]); eff = sorted([effect()]); au = sorted([auth()])
    payload = u16(S)+connector()+text("native-observe-effect-v1")+protocol()+u32(len(obs))+b"".join(obs)+u32(len(eff))+b"".join(eff)+u32(len(au))+b"".join(au)+reconciliation()
    return h(MAN, payload)

def main():
    c = connector().hex(); m = manifest().hex()
    if c != EXPECTED_CONNECTOR: raise SystemExit(f"connector mismatch: {c}")
    if m != EXPECTED_MANIFEST: raise SystemExit(f"manifest mismatch: {m}")
    print(f"PASS connector {c}")
    print(f"PASS manifest  {m}")
if __name__ == "__main__": main()
