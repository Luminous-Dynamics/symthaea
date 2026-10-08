#!/usr/bin/env python3
"""RH-006 exact quadratic image over a convex 2-D polygon.

Research-only. The analytic image uses vertices, edge stationary points, and
an admissible interior stationary point. Random convex-combination sampling is
only an independent containment falsification.
"""
from __future__ import annotations
import hashlib
import json
import math
import random
from dataclasses import dataclass
TOL=1e-12
@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
@dataclass(frozen=True)
class Quadratic1D:
    a: float
    b: float
    c: float
    def value(self, t: float) -> float:
        return self.a*t*t+self.b*t+self.c
    def range_unit(self) -> Interval:
        points=[0.0,1.0]
        if abs(self.a)>TOL:
            vertex=-self.b/(2.0*self.a)
            if -TOL<=vertex<=1.0+TOL: points.append(min(1.0,max(0.0,vertex)))
        values=[self.value(t) for t in points]
        return Interval(min(values),max(values))
@dataclass(frozen=True)
class Quadratic2D:
    a_xx: float; b_yy: float; c_xy: float; d_x: float; e_y: float; f0: float
    def value(self,x:float,y:float)->float:
        return self.a_xx*x*x+self.b_yy*y*y+self.c_xy*x*y+self.d_x*x+self.e_y*y+self.f0
    def stationary(self):
        det=4.0*self.a_xx*self.b_yy-self.c_xy*self.c_xy
        if abs(det)<=TOL: return None
        return ((self.c_xy*self.e_y-2.0*self.b_yy*self.d_x)/det,(self.c_xy*self.d_x-2.0*self.a_xx*self.e_y)/det)
@dataclass(frozen=True)
class ConvexPolygon:
    vertices: tuple[tuple[float,float],...]
    def contains(self,point):
        sign=0.0
        for i,a in enumerate(self.vertices):
            b=self.vertices[(i+1)%len(self.vertices)]
            cross=(b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0])
            if abs(cross)<=TOL: continue
            if sign==0.0: sign=math.copysign(1.0,cross)
            elif math.copysign(1.0,cross)!=sign: return False
        return True
    def range_of(self,q):
        values=[q.value(*v) for v in self.vertices]
        for i,(x0,y0) in enumerate(self.vertices):
            x1,y1=self.vertices[(i+1)%len(self.vertices)]
            dx,dy=x1-x0,y1-y0
            edge=Quadratic1D(q.a_xx*dx*dx+q.b_yy*dy*dy+q.c_xy*dx*dy,
                2.0*q.a_xx*x0*dx+2.0*q.b_yy*y0*dy+q.c_xy*(x0*dy+y0*dx)+q.d_x*dx+q.e_y*dy,
                q.value(x0,y0))
            er=edge.range_unit(); values.extend([er.lower,er.upper])
        stationary=q.stationary()
        if stationary is not None and self.contains(stationary): values.append(q.value(*stationary))
        return Interval(min(values),max(values))
def canonical():
    triangle=ConvexPolygon(((0.0,0.0),(1.0,0.0),(0.0,1.0)))
    d=Quadratic2D(-1.0,-1.0,0.0,0.0,0.0,0.25)
    image=triangle.range_of(d)
    return {"triangle_vertices":triangle.vertices,"surface":"D(x,y)=0.25-x^2-y^2","image":[image.lower,image.upper],"expected":[-0.75,0.25],"boundary_D_zero_reachable":True}
def fuzz(seed=20261008,trials=250,samples_per_trial=4096):
    rng=random.Random(seed)
    poly=ConvexPolygon(((-1.0,-0.4),(0.2,-1.0),(1.1,-0.1),(0.7,1.0),(-0.4,1.1),(-1.1,0.2)))
    max_gap_low=0.0; max_gap_high=0.0; violations=0
    for _ in range(trials):
        q=Quadratic2D(*(rng.uniform(-2.0,2.0) for _ in range(6)))
        exact=poly.range_of(q); sample_lo=math.inf; sample_hi=-math.inf
        for _ in range(samples_per_trial):
            weights=[rng.expovariate(1.0) for _ in poly.vertices]; total=sum(weights)
            x=sum(w*v[0] for w,v in zip(weights,poly.vertices))/total
            y=sum(w*v[1] for w,v in zip(weights,poly.vertices))/total
            value=q.value(x,y); sample_lo=min(sample_lo,value); sample_hi=max(sample_hi,value)
        if sample_lo<exact.lower-1e-9 or sample_hi>exact.upper+1e-9: violations+=1
        max_gap_low=max(max_gap_low,exact.lower-sample_lo); max_gap_high=max(max_gap_high,sample_hi-exact.upper)
    return {"seed":seed,"trials":trials,"random_convex_combinations_per_trial":samples_per_trial,"containment_violations":violations,"max_exact_minus_sample_lower":max_gap_low,"max_sample_minus_exact_upper":max_gap_high}
def main():
    result={"schema":"rh006-multivariate-polygon-propagation/v1","status":"research-diagnostic-only","canonical":canonical(),"falsification":fuzz(),"claim_boundary":{"polygon_is_sharp_identified_set":False,"formal_inference":False,"grid_or_sampling_defines_image":False}}
    assert result["canonical"]["image"]==[-0.75,0.25]
    assert result["falsification"]["containment_violations"]==0
    payload=json.dumps(result,sort_keys=True,separators=(",",":")); result["payload_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__": main()
