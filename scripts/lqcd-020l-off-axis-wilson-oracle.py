#!/usr/bin/env python3
"""Independent shortest-path-symmetrized off-axis Wilson-loop oracle.

Spatial transport uses an arithmetic average of all unique shortest Manhattan
path transporters through an APE-smeared spatial operator field. Temporal
transport uses the original unsmeared ensemble field only.
"""
import hashlib, json, math
from collections import Counter

DIMS=(3,3,3,2)
ALPHA=0.7
CONVENTION_ID='shortest_path_symmetrized_ape_spatial_unsmeared_temporal_wilson_v1'

def eye(): return [[1+0j if i==j else 0j for j in range(3)] for i in range(3)]
def zero(): return [[0j]*3 for _ in range(3)]
def add(a,b): return [[a[i][j]+b[i][j] for j in range(3)] for i in range(3)]
def scale(c,a): return [[c*a[i][j] for j in range(3)] for i in range(3)]
def mul(a,b): return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def dagger(a): return [[a[j][i].conjugate() for j in range(3)] for i in range(3)]
def trace(a): return a[0][0]+a[1][1]+a[2][2]
def det(a):
    return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])
def inv(a):
    d=det(a)
    if abs(d)<1e-15: raise ValueError('singular matrix')
    return [[(a[(j+1)%3][(i+1)%3]*a[(j+2)%3][(i+2)%3]-a[(j+1)%3][(i+2)%3]*a[(j+2)%3][(i+1)%3])/d for j in range(3)] for i in range(3)]
def frob(a): return math.sqrt(sum(abs(a[i][j])**2 for i in range(3) for j in range(3)))
def maxerr(a,b): return max(abs(a[i][j]-b[i][j]) for i in range(3) for j in range(3))

def polar_project_su3(m):
    n=frob(m)
    if not n>0 or not math.isfinite(n): raise ValueError('invalid projection input')
    x=scale(math.sqrt(3.0)/n,m)
    for _ in range(40):
        y=scale(0.5,add(x,dagger(inv(x))))
        if maxerr(x,y)<2e-15:
            x=y
            break
        x=y
    phase=math.atan2(det(x).imag,det(x).real)/3.0
    return scale(complex(math.cos(-phase),math.sin(-phase)),x)

def sites(d=DIMS):
    for x in range(d[0]):
      for y in range(d[1]):
       for z in range(d[2]):
        for t in range(d[3]): yield (x,y,z,t)
def idx(s,d=DIMS): x,y,z,t=s; return (((x*d[1]+y)*d[2]+z)*d[3]+t)
def shift(s,mu,n,d=DIMS):
    o=list(s); o[mu]=(o[mu]+n)%d[mu]; return tuple(o)
def identity_field(d=DIMS): return [eye() for _ in range(math.prod(d)*4)]
def get(f,s,mu,d=DIMS): return f[idx(s,d)*4+mu]
def put(f,s,mu,v,d=DIMS): f[idx(s,d)*4+mu]=v

def embedded(pair,axis,angle):
    nn=math.sqrt(sum(x*x for x in axis)); nx,ny,nz=[x/nn for x in axis]
    c=math.cos(angle); ss=math.sin(angle); a0,a1,a2,a3=(c,ss*nx,ss*ny,ss*nz)
    out=eye(); i,j=pair
    out[i][i]=a0+1j*a3; out[i][j]=a2+1j*a1
    out[j][i]=-a2+1j*a1; out[j][j]=a0-1j*a3
    return out

def fixture():
    f=identity_field()
    ops=[
        ((0,0,0,0),0,(0,1),(1,2,3),0.31),
        ((1,0,1,0),1,(0,2),(2,-1,1),-0.27),
        ((2,1,0,1),2,(1,2),(1,1,-2),0.22),
        ((1,2,2,0),0,(0,1),(-2,1,1),0.19),
    ]
    for site,mu,pair,axis,angle in ops:
        put(f,site,mu,mul(embedded(pair,axis,angle),get(f,site,mu)))
    return f

def spatial_staple_sum(f,s,mu,d=DIMS):
    if mu>=3: raise ValueError('spatial only')
    out=zero(); s_plus_mu=shift(s,mu,1,d)
    for nu in range(3):
        if nu==mu: continue
        s_plus_nu=shift(s,nu,1,d)
        forward=mul(mul(get(f,s,nu,d),get(f,s_plus_nu,mu,d)),dagger(get(f,s_plus_mu,nu,d)))
        s_minus_nu=shift(s,nu,-1,d); s_minus_nu_plus_mu=shift(s_minus_nu,mu,1,d)
        backward=mul(mul(dagger(get(f,s_minus_nu,nu,d)),get(f,s_minus_nu,mu,d)),get(f,s_minus_nu_plus_mu,nu,d))
        out=add(out,forward); out=add(out,backward)
    return out

def ape_step(f,alpha=ALPHA,d=DIMS):
    out=[[[v for v in row] for row in m] for m in f]
    for s in sites(d):
        for mu in range(3):
            put(out,s,mu,polar_project_su3(add(scale(alpha,get(f,s,mu,d)),spatial_staple_sum(f,s,mu,d))),d)
    return out

def gauge_transform(f,d=DIMS):
    omega={}
    for s in sites(d):
        k=idx(s,d)
        omega[s]=mul(embedded((0,1),(1,2,3),0.017*(k+1)),embedded((0,2),(2,-1,1),-0.011*(k+1)))
    out=identity_field(d)
    for s in sites(d):
        for mu in range(4):
            put(out,s,mu,mul(mul(omega[s],get(f,s,mu,d)),dagger(omega[shift(s,mu,1,d)])),d)
    return out

def spatial_step(f,site,axis,direction,d=DIMS):
    if axis>=3 or direction not in (-1,1): raise ValueError('invalid spatial step')
    if direction>0:
        return get(f,site,axis,d),shift(site,axis,1,d)
    previous=shift(site,axis,-1,d)
    return dagger(get(f,previous,axis,d)),previous

def step_multiset(displacement):
    if displacement==(0,0,0): raise ValueError('zero displacement')
    out=[]
    for axis,amount in enumerate(displacement):
        direction=1 if amount>0 else -1
        out.extend([(axis,direction)]*abs(amount))
    return out

def unique_permutations(items):
    counts=Counter(items); keys=sorted(counts); n=len(items); path=[]
    def rec():
        if len(path)==n:
            yield tuple(path); return
        for key in keys:
            if counts[key]:
                counts[key]-=1; path.append(key)
                yield from rec()
                path.pop(); counts[key]+=1
    yield from rec()

def path_transporter(f,start,path,d=DIMS):
    value=eye(); site=start
    for axis,direction in path:
        edge,site=spatial_step(f,site,axis,direction,d)
        value=mul(value,edge)
    return value,site

def symmetrized_spatial_transporter(f,start,displacement,d=DIMS):
    steps=step_multiset(displacement)
    paths=list(unique_permutations(steps))
    total=zero(); endpoint=None
    for path in paths:
        value,end=path_transporter(f,start,path,d)
        if endpoint is None: endpoint=end
        if end!=endpoint: raise AssertionError('path endpoints disagree')
        total=add(total,value)
    return scale(1.0/len(paths),total),endpoint,len(paths)

def temporal_transporter(original,start,t,d=DIMS):
    if t<=0 or t>=d[3]: raise ValueError('temporal winding')
    value=eye(); site=start
    for _ in range(t):
        value=mul(value,get(original,site,3,d))
        site=shift(site,3,1,d)
    return value,site

def off_axis_loop(original,operator,start,displacement,t,d=DIMS):
    if any(abs(displacement[a])>=d[a] for a in range(3)): raise ValueError('spatial extent')
    bottom,end,count=symmetrized_spatial_transporter(operator,start,displacement,d)
    temporal_end,top_end=temporal_transporter(original,end,t,d)
    top_start=shift(start,3,t,d)
    top,top_end_check,count2=symmetrized_spatial_transporter(operator,top_start,displacement,d)
    if top_end_check!=top_end or count2!=count: raise AssertionError('top geometry')
    temporal_start,top_start_check=temporal_transporter(original,start,t,d)
    if top_start_check!=top_start: raise AssertionError('temporal geometry')
    loop=mul(mul(mul(bottom,temporal_end),dagger(top)),dagger(temporal_start))
    return trace(loop).real/3.0,count

def average_loop(original,operator,displacement,t,d=DIMS):
    vals=[]; path_count=None
    for s in sites(d):
        v,c=off_axis_loop(original,operator,s,displacement,t,d)
        vals.append(v)
        if path_count is None: path_count=c
        if c!=path_count: raise AssertionError('path count')
    return sum(vals)/len(vals),path_count

def main():
    vectors=((1,0,0),(2,0,0),(1,1,0),(1,1,1),(2,1,0))
    original=fixture(); operator=ape_step(original)
    identity=identity_field(); identity_operator=ape_step(identity)
    values={}; counts={}; identity_values={}
    for v in vectors:
        key='r'+''.join(str(x) for x in v)
        values[key],counts[key]=average_loop(original,operator,v,1)
        identity_values[key],_=average_loop(identity,identity_operator,v,1)

    axis1,_=off_axis_loop(original,operator,(0,0,0,0),(1,0,0),1)
    bottom,end,c=symmetrized_spatial_transporter(operator,(0,0,0,0),(1,0,0))
    if c!=1: raise AssertionError('axis path count')
    tend,topend=temporal_transporter(original,end,1)
    topstart=shift((0,0,0,0),3,1)
    top,_,_=symmetrized_spatial_transporter(operator,topstart,(1,0,0))
    tstart,_=temporal_transporter(original,(0,0,0,0),1)
    direct_local=trace(mul(mul(mul(bottom,tend),dagger(top)),dagger(tstart))).real/3.0
    axis_collapse_error=abs(axis1-direct_local)

    transformed=gauge_transform(original); transformed_operator=ape_step(transformed)
    transformed_values={}
    for v in vectors:
        key='r'+''.join(str(x) for x in v)
        transformed_values[key],_=average_loop(transformed,transformed_operator,v,1)
    gauge_invariance=max(abs(values[k]-transformed_values[k]) for k in values)

    tampered=[[[z for z in row] for row in m] for m in operator]
    put(tampered,(0,0,0,0),3,embedded((0,1),(1,2,3),0.4))
    before,_=off_axis_loop(original,operator,(0,0,0,0),(1,1,0),1)
    after,_=off_axis_loop(original,tampered,(0,0,0,0),(1,1,0),1)
    temporal_operator_tamper_error=abs(before-after)

    expected_counts={'r100':1,'r200':1,'r110':2,'r111':6,'r210':3}
    if counts!=expected_counts: raise AssertionError((counts,expected_counts))
    if any(abs(v-1.0)>2e-15 for v in identity_values.values()): raise AssertionError(identity_values)
    if gauge_invariance>3e-15 or temporal_operator_tamper_error!=0.0 or axis_collapse_error!=0.0:
        raise AssertionError((gauge_invariance,temporal_operator_tamper_error,axis_collapse_error))

    result={
        'convention_id':CONVENTION_ID,
        'dims':DIMS,
        'alpha':ALPHA,
        'loop_means':values,
        'path_counts':counts,
        'identity_values':identity_values,
        'gauge_invariance_max_error':gauge_invariance,
        'temporal_operator_tamper_error':temporal_operator_tamper_error,
        'axis_unique_path_collapse_error':axis_collapse_error,
    }
    text=json.dumps(result,sort_keys=True,separators=(',',':'))
    digest=hashlib.sha256(text.encode()).hexdigest()
    expected='20458685d99bcd4d67d373dae12cded11665aa2784a1a8dd9965787628030906'
    if digest!=expected: raise AssertionError((digest,expected))
    print('ok'); print('result_sha256='+digest); print(text)

if __name__=='__main__': main()
