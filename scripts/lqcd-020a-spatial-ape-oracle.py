#!/usr/bin/env python3
"""Independent spatial APE-smearing + SU(3) polar-projection oracle.

Standard-library only; imports no Symthaea/Rust code. Convention:
M_k(x) = alpha U_k(x) + sum of the four spatial staples transverse to k,
for k=0,1,2 only. Temporal links are copied unchanged. M is projected to
SU(3) by the unitary polar factor followed by determinant-phase removal.
"""
import hashlib, json, math
DIMS=(3,3,3,2)
ALPHA=0.7
CONVENTION_ID='spatial_ape_ehk_polar_v1'

def eye(): return [[1+0j if i==j else 0j for j in range(3)] for i in range(3)]
def zero(): return [[0j]*3 for _ in range(3)]
def add(a,b): return [[a[i][j]+b[i][j] for j in range(3)] for i in range(3)]
def scale(c,a): return [[c*a[i][j] for j in range(3)] for i in range(3)]
def mul(a,b): return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def dagger(a): return [[a[j][i].conjugate() for j in range(3)] for i in range(3)]
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
            x=y; break
        x=y
    phase=math.atan2(det(x).imag,det(x).real)/3.0
    return scale(complex(math.cos(-phase),math.sin(-phase)),x)

def sites(d=DIMS):
    for x in range(d[0]):
      for y in range(d[1]):
       for z in range(d[2]):
        for t in range(d[3]): yield (x,y,z,t)
def idx(s,d=DIMS): x,y,z,t=s; return (((x*d[1]+y)*d[2]+z)*d[3]+t)
def shift(s,mu,n,d=DIMS): o=list(s);o[mu]=(o[mu]+n)%d[mu];return tuple(o)
def identity_field(d=DIMS): return [eye() for _ in range(math.prod(d)*4)]
def get(f,s,mu,d=DIMS): return f[idx(s,d)*4+mu]
def put(f,s,mu,v,d=DIMS): f[idx(s,d)*4+mu]=v

def embedded(pair,axis,angle):
    nn=math.sqrt(sum(x*x for x in axis)); nx,ny,nz=[x/nn for x in axis]
    c=math.cos(angle); ss=math.sin(angle); a0,a1,a2,a3=(c,ss*nx,ss*ny,ss*nz)
    out=eye(); i,j=pair
    out[i][i]=a0+1j*a3; out[i][j]=a2+1j*a1; out[j][i]=-a2+1j*a1; out[j][j]=a0-1j*a3
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
    if mu>=3: raise ValueError('APE updates spatial links only')
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
    if not math.isfinite(alpha) or alpha<=0: raise ValueError('invalid alpha')
    out=[[[v for v in row] for row in m] for m in f]
    for s in sites(d):
        for mu in range(3):
            candidate=add(scale(alpha,get(f,s,mu,d)),spatial_staple_sum(f,s,mu,d))
            put(out,s,mu,polar_project_su3(candidate),d)
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

def matrix_record(m):
    return [[[z.real,z.imag] for z in row] for row in m]

def main():
    # Projection exactness on a scaled SU(3) matrix.
    u=embedded((0,2),(2,-1,1),0.37)
    assert maxerr(polar_project_su3(scale(2.75,u)),u)<8e-16

    identity=identity_field(); identity_smeared=ape_step(identity)
    identity_error=max(maxerr(m,eye()) for m in identity_smeared)

    f=fixture(); smeared=ape_step(f)
    transformed_smeared=ape_step(gauge_transform(f))
    smeared_transformed=gauge_transform(smeared)
    gauge_covariance_error=max(maxerr(a,b) for a,b in zip(transformed_smeared,smeared_transformed))
    max_unitarity_error=max(maxerr(mul(dagger(m),m),eye()) for m in smeared)
    max_determinant_error=max(abs(det(m)-1) for m in smeared)
    temporal_link_error=max(maxerr(get(smeared,s,3),get(f,s,3)) for s in sites())

    probes=[
        ((0,0,0,0),0),
        ((1,0,1,0),1),
        ((2,1,0,1),2),
    ]
    matrices=[matrix_record(get(smeared,s,mu)) for s,mu in probes]
    result={
        'convention_id':CONVENTION_ID,
        'dims':DIMS,
        'alpha':ALPHA,
        'identity_fixed_point_error':identity_error,
        'gauge_covariance_max_error':gauge_covariance_error,
        'max_unitarity_error':max_unitarity_error,
        'max_determinant_error':max_determinant_error,
        'temporal_link_max_error':temporal_link_error,
        'probe_matrices':matrices,
    }
    expected_matrices=[
        [[[0.998954692724953,0.03665056708308],[0.02443371138872,0.01221685569436],[0.0,0.0]],[[-0.02443371138872,0.01221685569436],[0.998954692724953,-0.03665056708308],[0.0,0.0]],[[0.0,0.0],[0.0,0.0],[1.0,0.0]]],
        [[[0.999203292381324,-0.016293048079488],[0.0,0.0],[0.016293048079488,-0.032586096158976]],[[0.0,0.0],[1.0,0.0],[0.0,0.0]],[[-0.016293048079488,-0.032586096158976],[0.0,0.0],[0.999203292381324,0.016293048079488]]],
        [[[1.0,0.0],[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.999468412221736,-0.026619453675369],[0.013309726837685,0.013309726837685]],[[0.0,0.0],[-0.013309726837685,0.013309726837685],[0.999468412221736,0.026619453675369]]],
    ]
    for got,want in zip(matrices,expected_matrices):
        for i in range(3):
            for j in range(3):
                if abs(got[i][j][0]-want[i][j][0])>2e-14 or abs(got[i][j][1]-want[i][j][1])>2e-14:
                    raise AssertionError(('probe matrix',got,want))
    if identity_error>1e-15 or gauge_covariance_error>1e-14 or max_unitarity_error>2e-14 or max_determinant_error>2e-14 or temporal_link_error!=0:
        raise AssertionError(result)
    text=json.dumps(result,sort_keys=True,separators=(',',':')); digest=hashlib.sha256(text.encode()).hexdigest()
    print('ok'); print('result_sha256='+digest); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
