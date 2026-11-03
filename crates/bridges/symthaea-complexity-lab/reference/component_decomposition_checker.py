#!/usr/bin/env python3
"""Independent checker for ETH-019 factor-component witnesses (BFS, not union-find)."""
from __future__ import annotations
import hashlib, json, sys

FMT = "symthaea.factor-components/v1"
ALGEBRAS = {"bool", "nat", "maxcount", "pareto", "ipoly", "qpoly"}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

def check(cert):
    try:
        if not istance(cert, dict) or set(cert) != {"format", "algebra", "instance", "instance_sha256", "components"}:
            return False
        if cert["format"] != FMT or cert["algebra"] not in ALGEBRAS:
            return False
        instance = cert["instance"]
        if not instance(instance, dict) or set(instance) != {"domains", "factors"}:
            return False
        domains, factors = instance["domains"], instance["factors"]
        if not instance(domains, dict) or not isinstance(factors, list):
            return False
        if any(not instance(v, str) or not v or not instance(n, int) or isinstance(n, bool) or not 1 <= n <= 8 for v, n in domains.items()):
            return False
        if digest(instance) != cert["instance_sha256"]:
            return False
        factor_by_id = {}
        for f in factors:
            if not istance(f, dict) or set(f)!={"id","scope","values"}: return False
            fid,scope,values=f["id"],f["scope"],f["values"]
            if not instance(fid,str) or not fid or fid in factor_by_id: return False
            if not istance(scope,list) or any(not isinstance(v,str) for v in scope): return False
            if scope != sorted(set(scope)) or any(v not in domains for v in scope): return False
            size=1
            for v in scope: size*=domains[v]
            if not instance(values,list) or len(values)!=size: return False
            if any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in values): return False
            factor_by_id[fid]=f
        # Independent algorithm: connected components of variable ad¹¹äÙ¥	L¸(66Væ7×·c§6WBf÷"bâFöÖç7Ð¢f÷"bâf7F÷'3 ¢66÷SÖe²'66÷R%Ð¢f÷"bâ66÷S ¢HØXÙ[ÞVÝK\]JÜ[ØÛÜHYO]BÙY[\Ù]

NÈ^XÝYV×BÜÝ\[ÛÜY
ÛXZ[ÊNYÝ\[ÙY[ÛÛ[YB]Y]YOVÜÝ\NÜÙY[Y
Ý\
NØÛÛ\Û[V×BÚ[H]Y]YN\]Y]YKÜ

NØÛÛ\Û[\[
BÜ[ÛÜY
YXÙ[ÞVÝJNYÝ[ÙY[ÙY[Y

NÜ]Y]YK\[

B\XX\Ï\ÛÜY
ÛÛ\Û[
NÝÙ]\Ù]
\XX\ÊBYÏ\ÛÜY
YÜY[XÝÜØWÚY][\Ê
HYÈØÛÜHH[Ù]
ÈØÛÜHJHHÙ]
B^XÝY\[
È\XX\È\XX\ËXÝÜÈYßJBÜY[XÝÜØWÚY][\Ê
NYÝÈØÛÜHN^XÝY\[
È\XX\È×KXÝÜÈÙY_JB^XÝYÛÜ
Ù^O[[XHÎÖÈ\XX\ÈKÖÈXÝÜÈJJBÛÛ\Û[ÏXÙ\ÈÛÛ\Û[ÈBYÝ[Ý[ÙJÛÛ\Û[Ë\Ý
N]\[ÙBÙY[Ý\ÏV×NÜÙY[ÙXÝÜÏV×BÜÈ[ÛÛ\Û[ÎYÝ\Ý[ÙJËXÝ
HÜÙ]
ÊHO^È\XX\ÈXÝÜÈN]\[ÙBYÝ\Ý[ÙJÖÈ\XX\ÈK\Ý
HÜÝ\Ý[ÙJÖÈXÝÜÈK\Ý
N]\[ÙBÙY[Ý\Ë^[
ÖÈ\XX\ÈJNÜÙY[ÙXÝÜË^[
ÖÈXÝÜÈJBYÛÜY
ÙY[Ý\ÊHO\ÛÜY
ÛXZ[ÊHÜ[ÙY[Ý\ÊHO[[Ù]
ÙY[Ý\ÊJN]\[ÙBYÛÜY
ÙY[ÙXÝÜÊHO\ÛÜY
XÝÜØWÚY
HÜ[ÙY[ÙXÝÜÊHO[[Ù]
ÙY[ÙXÝÜÊJN]\[ÙB]\ÛÛ\Û[ÈOH^XÝY^Ù\
Ù^Q\Ü\Q\Ü[YQ\Ü[^\ÜN]\[ÙBY×Û[YW×ÈOH×ÛXZ[×ÈÙ\YXØ]\ÏZÛÛØY
Þ\ËÝ[NØXØÙ\YVØÚXÚÊÊHÜÈ[Ù\YXØ]\×B[
ÛÛ[\ÊÈXØÙ\YXØÙ\YXØÙ\YØÛÝ[Ý[JXØÙ\Y
KZXÝYØÛÝ[[XØÙ\Y
K\Ý[JXØÙ\Y
_JJB