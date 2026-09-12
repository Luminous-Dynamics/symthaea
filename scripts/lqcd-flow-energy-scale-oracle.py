#!/usr/bin/env python3
"""Independent flowed-energy-density and scale-extraction algebra oracle.

Standard-library only. This does not generate an ensemble or claim a physical
scale. It qualifies the SU(3) field-strength energy normalization, gauge
conjugation invariance, and generic t0/w0-like interpolation on synthetic
ensemble-mean curves.
"""
import math

R3 = math.sqrt(3.0)
I = 1.0j
LM = [
 [[0,1,0],[1,0,0],[0,0,0]],
 [[0,-I,0],[I,0,0],[0,0,0]],
 [[1,0,0],[0,-1,0],[0,0,0]],
 [[0,0,1],[0,0,0],[1,0,0]],
 [[0,0,-I],[0,0,0],[I,0,0]],
 [[0,0,0],[0,0,1],[0,1,0]],
 [[0,0,0],[0,0,-I],[0,I,0]],
 [[1/R3,0,0],[0,1/R3,0],[0,0,-2/R3]],
]
LM = [[[complex(v) for v in row] for row in m] for m in LM]
T = [[[v/2 for v in row] for row in m] for m in LM]
COEFF = [
 [0.12,-0.04,0.07,0.00,0.03,-0.02,0.01,0.05],
 [-0.03,0.08,0.00,-0.06,0.02,0.01,0.04,-0.02],
 [0.05,0.00,-0.09,0.03,-0.01,0.02,0.00,0.04],
 [0.00,-0.02,0.06,0.05,0.01,-0.03,0.02,0.00],
 [0.04,0.03,-0.01,0.00,-0.05,0.07,0.02,0.01],
 [-0.02,0.01,0.03,-0.04,0.06,0.00,-0.05,0.02],
]

def mm(a,b):
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def tr(a):
    return sum(a[i][i] for i in range(3))

def dg(a):
    return [[a[j][i].conjugate() for j in range(3)] for i in range(3)]

def field_strength(c):
    z=[[0j]*3 for _ in range(3)]
    for x,g in zip(c,T):
        z=[[z[i][j]+x*g[i][j] for j in range(3)] for i in range(3)]
    return z

def energy(fields):
    # F = F^a T^a, Tr(T^a T^b)=delta_ab/2.
    # Thus sum_{mu<nu} Tr(F_munu^2) = (1/4) F^a_munu F^a_munu.
    return sum(tr(mm(f,f)).real for f in fields)

def crossing(xs,ys,target):
    if len(xs)!=len(ys) or len(xs)<2 or any(xs[i+1]<=xs[i] for i in range(len(xs)-1)):
        raise ValueError("invalid curve")
    hits=[]
    for i in range(len(xs)-1):
        a,b=ys[i]-target,ys[i+1]-target
        if a==0.0: hits.append((i,0.0))
        elif a*b<0.0 or b==0.0: hits.append((i,(target-ys[i])/(ys[i+1]-ys[i])))
    if len(hits)!=1:
        raise ValueError("crossing must be unique")
    i,f=hits[0]
    return xs[i]+f*(xs[i+1]-xs[i])

def t0_like(times,means,target):
    return crossing(times,[t*t*e for t,e in zip(times,means)],target)

def w0_like(times,means,target):
    if len(times)<4 or len(times)!=len(means):
        raise ValueError("need four matched samples")
    f=[t*t*e for t,e in zip(times,means)]
    tx=[]; response=[]
    for i in range(1,len(times)-1):
        derivative=(f[i+1]-f[i-1])/(times[i+1]-times[i-1])
        tx.append(times[i]); response.append(times[i]*derivative)
    t=crossing(tx,response,target)
    if t<=0: raise ValueError("nonpositive w0^2")
    return math.sqrt(t)

def main():
    fields=[field_strength(c) for c in COEFF]
    e=energy(fields)
    analytic=0.5*sum(x*x for row in COEFF for x in row)
    assert abs(e-analytic)<2e-16

    th,ph=0.23,-0.17
    g=[[0j]*3 for _ in range(3)]
    for i,p in enumerate([th,ph,-th-ph]):
        g[i][i]=complex(math.cos(p),math.sin(p))
    eg=energy([mm(mm(g,f),dg(g)) for f in fields])
    assert abs(eg-e)<2e-16
    assert energy([[[0j]*3 for _ in range(3)] for _ in range(6)])==0.0

    # Synthetic ENSEMBLE-MEAN curves only; not physical data.
    times=[0.10,0.20,0.30,0.40,0.50,0.60]
    f=[0.10,0.18,0.26,0.34,0.42,0.50]
    means=[x/(t*t) for t,x in zip(times,f)]
    t0=t0_like(times,means,0.30)
    assert abs(t0-0.35)<1e-15

    wf=[0.02+0.8*t for t in times]
    wmeans=[x/(t*t) for t,x in zip(times,wf)]
    w0=w0_like(times,wmeans,0.32)
    assert abs(w0-math.sqrt(0.4))<2e-15

    try:
        crossing([0.1,0.2,0.3],[0.2,0.4,0.2],0.3)
    except ValueError:
        pass
    else:
        raise AssertionError("multiple crossing must fail")

    print("ok")
    print(f"energy_density={e:.17g}")
    print(f"analytic_energy_density={analytic:.17g}")
    print(f"gauge_transformed_energy_density={eg:.17g}")
    print(f"t0_like_synthetic_crossing={t0:.17g}")
    print(f"w0_like_synthetic_crossing={w0:.17g}")

if __name__=="__main__":
    main()
