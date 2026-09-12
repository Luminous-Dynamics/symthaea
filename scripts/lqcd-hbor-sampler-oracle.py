#!/usr/bin/env python3
"""Independent Metropolis vs heat-bath+overrelaxation tiny-SU(3) pilot.

Standard-library only; imports no Symthaea/Rust code. This is sampler
qualification evidence, not a physical lattice-QCD result.
"""
import math, statistics, struct

MASK32 = 0xFFFFFFFF
PAIRS = ((0, 1), (0, 2), (1, 2))
DIMS = (2, 2, 2, 2)
BETA = 5.7
DOMAIN_TRANSITION = 1
DOMAIN_INITIALIZATION = 4
SEED = bytes([0x5A]) * 32
ENSEMBLE_SLOT = 29

def rotl32(x, n): return ((x << n) & MASK32) | (x >> (32 - n))
def quarter_round(x, a, b, c, d):
    x[a]=(x[a]+x[b])&MASK32; x[d]^=x[a]; x[d]=rotl32(x[d],16)
    x[c]=(x[c]+x[d])&MASK32; x[b]^=x[c]; x[b]=rotl32(x[b],12)
    x[a]=(x[a]+x[b])&MASK32; x[d]^=x[a]; x[d]=rotl32(x[d],8)
    x[c]=(x[c]+x[d])&MASK32; x[b]^=x[c]; x[b]=rotl32(x[b],7)

def chacha8_block(seed, counter, stream):
    state=[0x61707865,0x3320646E,0x79622D32,0x6B206574]
    state += list(struct.unpack("<8I", seed))
    state += [counter&MASK32,(counter>>32)&MASK32,stream&MASK32,(stream>>32)&MASK32]
    work=state.copy()
    for _ in range(4):
        quarter_round(work,0,4,8,12); quarter_round(work,1,5,9,13)
        quarter_round(work,2,6,10,14); quarter_round(work,3,7,11,15)
        quarter_round(work,0,5,10,15); quarter_round(work,1,6,11,12)
        quarter_round(work,2,7,8,13); quarter_round(work,3,4,9,14)
    return struct.pack("<16I", *[((work[i]+state[i])&MASK32) for i in range(16)])

def pack_stream_id(domain, slot, replica, rank=0):
    return (domain<<56)|(slot<<32)|(replica<<16)|rank

class ChaCha8Stream:
    def __init__(self, seed, stream):
        self.seed=seed; self.stream=stream; self.counter=0
        self.buffer=b""; self.cursor=0; self.draws=0
    def _bytes(self, count):
        out=bytearray()
        while len(out)<count:
            if self.cursor>=len(self.buffer):
                self.buffer=chacha8_block(self.seed,self.counter,self.stream)
                self.counter+=1; self.cursor=0
            take=min(count-len(out),len(self.buffer)-self.cursor)
            out.extend(self.buffer[self.cursor:self.cursor+take]); self.cursor+=take
        return bytes(out)
    def open01(self):
        self.draws += 1
        value=struct.unpack("<Q",self._bytes(8))[0]
        k=(value>>12)+1
        return k/((1<<52)+1)

def identity(): return [[1+0j if i==j else 0j for j in range(3)] for i in range(3)]
def mul(a,b): return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def dagger(a): return [[a[j][i].conjugate() for j in range(3)] for i in range(3)]
def trace(a): return a[0][0]+a[1][1]+a[2][2]
def sites(dims=DIMS):
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                for t in range(dims[3]):
                    yield (x,y,z,t)
def site_index(site,dims=DIMS):
    x,y,z,t=site
    return (((x*dims[1]+y)*dims[2]+z)*dims[3]+t)
def shift(site,mu,step,dims=DIMS):
    out=list(site); out[mu]=(out[mu]+step)%dims[mu]; return tuple(out)
def identity_field(dims=DIMS): return [identity() for _ in range(math.prod(dims)*4)]
def link(field,site,mu,dims=DIMS): return field[site_index(site,dims)*4+mu]
def set_link(field,site,mu,value,dims=DIMS): field[site_index(site,dims)*4+mu]=value

def plaquette(field,site,mu,nu,dims=DIMS):
    x_mu=shift(site,mu,1,dims); x_nu=shift(site,nu,1,dims)
    return mul(mul(mul(link(field,site,mu,dims),link(field,x_mu,nu,dims)),
                   dagger(link(field,x_nu,mu,dims))),
               dagger(link(field,site,nu,dims)))

def average_plaquette(field,dims=DIMS):
    values=[trace(plaquette(field,s,mu,nu,dims)).real/3
            for s in sites(dims) for mu in range(4) for nu in range(mu+1,4)]
    return sum(values)/len(values)

def affected_plaquettes(site,mu,dims=DIMS):
    result=set()
    for nu in range(4):
        if nu==mu: continue
        a,b=sorted((mu,nu))
        result.add((site,a,b)); result.add((shift(site,nu,-1,dims),a,b))
    return sorted(result)

def touching_trace(field,site,mu,dims=DIMS):
    return sum(trace(plaquette(field,base,a,b,dims)).real
               for base,a,b in affected_plaquettes(site,mu,dims))

def affected_action(field,site,mu,beta,dims=DIMS):
    return sum(beta*(1-trace(plaquette(field,base,a,b,dims)).real/3)
               for base,a,b in affected_plaquettes(site,mu,dims))

def embedded_quaternion(pair,q):
    a0,a1,a2,a3=q; out=identity(); i,j=pair
    out[i][i]=a0+1j*a3; out[i][j]=a2+1j*a1
    out[j][i]=-a2+1j*a1; out[j][j]=a0-1j*a3
    return out

def quaternion_from_embedded(pair,matrix):
    i,j=pair
    return ((matrix[i][i].real+matrix[j][j].real)/2,
            (matrix[i][j].imag+matrix[j][i].imag)/2,
            (matrix[i][j].real-matrix[j][i].real)/2,
            (matrix[i][i].imag-matrix[j][j].imag)/2)

def draw_rotation(source,pair,max_angle):
    uz,uphi,uangle=source.open01(),source.open01(),source.open01()
    z=2*uz-1; phi=2*math.pi*uphi; radial=math.sqrt(max(0,1-z*z))
    axis=(radial*math.cos(phi),radial*math.sin(phi),z)
    angle=max_angle*(2*uangle-1)
    c,s=math.cos(angle),math.sin(angle); nx,ny,nz=axis
    return embedded_quaternion(pair,(c,s*nx,s*ny,s*nz))

def metropolis_sweep(field,beta,max_angle,source,dims=DIMS):
    accepted=attempted=0
    for site in sites(dims):
        for mu in range(4):
            for pair in PAIRS:
                before=affected_action(field,site,mu,beta,dims)
                old=[row[:] for row in link(field,site,mu,dims)]
                set_link(field,site,mu,mul(draw_rotation(source,pair,max_angle),old),dims)
                delta=affected_action(field,site,mu,beta,dims)-before
                probability=1.0 if delta<=0 else math.exp(-delta)
                attempted+=1
                if source.open01()<probability: accepted+=1
                else: set_link(field,site,mu,old,dims)
    return accepted,attempted

def evaluated_trace(field,site,mu,pair,q,dims=DIMS):
    candidate=[[row[:] for row in matrix] for matrix in field]
    set_link(candidate,site,mu,mul(embedded_quaternion(pair,q),link(candidate,site,mu,dims)),dims)
    return touching_trace(candidate,site,mu,dims)

def finite_probe_force(field,site,mu,pair,dims=DIMS):
    plus=evaluated_trace(field,site,mu,pair,(1,0,0,0),dims)
    minus=evaluated_trace(field,site,mu,pair,(-1,0,0,0),dims)
    constant=.5*(plus+minus)
    q=(.5*(plus-minus),
       evaluated_trace(field,site,mu,pair,(0,1,0,0),dims)-constant,
       evaluated_trace(field,site,mu,pair,(0,0,1,0),dims)-constant,
       evaluated_trace(field,site,mu,pair,(0,0,0,1),dims)-constant)
    return constant,q

def kennedy_pendleton_scalar(source,alpha,max_attempts=256):
    if alpha==0:
        for attempt in range(1,max_attempts+1):
            a0=2*source.open01()-1
            if source.open01()<math.sqrt(max(0,1-a0*a0)): return a0,attempt
        raise RuntimeError("Haar rejection limit")
    for attempt in range(1,max_attempts+1):
        r0,r1,r2,r3=(source.open01() for _ in range(4))
        x1=-math.log(r1)/alpha; x2=-math.log(r2)/alpha
        c=math.cos(2*math.pi*r3)
        d=x2+x1*c*c; threshold=1-.5*d
        if threshold>0 and r0*r0<threshold:
            a0=1-d
            if -1<=a0<=1: return a0,attempt
    raise RuntimeError("Kennedy-Pendleton rejection limit")

def su2_heatbath(source,alpha):
    a0,attempts=kennedy_pendleton_scalar(source,alpha)
    z=2*source.open01()-1; phi=2*math.pi*source.open01()
    radial=math.sqrt(max(0,1-z*z)); direction=(radial*math.cos(phi),radial*math.sin(phi),z)
    radius=math.sqrt(max(0,1-a0*a0))
    return (a0,radius*direction[0],radius*direction[1],radius*direction[2]),attempts

def heatbath_subgroup_step(field,site,mu,pair,beta,source,dims=DIMS):
    _,q=finite_probe_force(field,site,mu,pair,dims)
    rho=math.sqrt(sum(value*value for value in q))
    alpha=beta*rho/3
    canonical,attempts=su2_heatbath(source,alpha)
    if rho>1e-14:
        qhat=tuple(value/rho for value in q)
        rotation=mul(embedded_quaternion(pair,qhat),embedded_quaternion(pair,canonical))
    else:
        rotation=embedded_quaternion(pair,canonical)
    oriented=quaternion_from_embedded(pair,rotation)
    if rho>1e-14:
        projected=sum(q[i]*oriented[i] for i in range(4))/rho
        if abs(projected-canonical[0])>1e-9: raise AssertionError("force orientation")
    set_link(field,site,mu,mul(rotation,link(field,site,mu,dims)),dims)
    return attempts

def heatbath_sweep(field,beta,source,dims=DIMS):
    attempts=0
    for site in sites(dims):
        for mu in range(4):
            for pair in PAIRS:
                attempts += heatbath_subgroup_step(field,site,mu,pair,beta,source,dims)
    return attempts

def overrelax_subgroup_step(field,site,mu,pair,dims=DIMS):
    before=touching_trace(field,site,mu,dims)
    _,q=finite_probe_force(field,site,mu,pair,dims)
    norm2=sum(value*value for value in q)
    if norm2<=1e-28: return
    reflection=[2*q[0]*value/norm2 for value in q]; reflection[0]-=1
    old=link(field,site,mu,dims)
    set_link(field,site,mu,mul(embedded_quaternion(pair,reflection),old),dims)
    after=touching_trace(field,site,mu,dims)
    if abs(after-before)>1e-8: raise AssertionError("microcanonical drift")

def overrelax_sweep(field,dims=DIMS):
    for site in sites(dims):
        for mu in range(4):
            for pair in PAIRS:
                overrelax_subgroup_step(field,site,mu,pair,dims)

def hbor_cycle(field,beta,source,overrelaxation_sweeps=1,dims=DIMS):
    attempts=heatbath_sweep(field,beta,source,dims)
    for _ in range(overrelaxation_sweeps):
        overrelax_sweep(field,dims)
    return attempts

def disordered_start(replica,rounds=2,dims=DIMS):
    field=identity_field(dims)
    source=ChaCha8Stream(SEED,pack_stream_id(DOMAIN_INITIALIZATION,ENSEMBLE_SLOT,replica))
    for _ in range(rounds):
        for site in sites(dims):
            for mu in range(4):
                for pair in PAIRS:
                    set_link(field,site,mu,mul(draw_rotation(source,pair,math.pi),link(field,site,mu,dims)),dims)
    return field

def split_rhat(chains):
    draw_count=len(chains[0])
    if draw_count%2 or any(len(chain)!=draw_count for chain in chains): raise ValueError
    n=draw_count//2; split=[]
    for chain in chains: split.extend((chain[:n],chain[n:]))
    means=[statistics.mean(chain) for chain in split]
    variances=[statistics.variance(chain) for chain in split]
    within=statistics.mean(variances); mean_of_means=statistics.mean(means)
    between=n*sum((value-mean_of_means)**2 for value in means)/(len(split)-1)
    estimate=((n-1)/n)*within+between/n
    return math.sqrt(estimate/within)

def run_chain(kind,start,replica,burn_in=20,stride=2,measurements=10):
    field=identity_field() if start=="cold" else disordered_start(replica)
    source=ChaCha8Stream(SEED,pack_stream_id(DOMAIN_TRANSITION,ENSEMBLE_SLOT,replica))
    keep={burn_in+stride*i for i in range(1,measurements+1)}
    final=burn_in+stride*measurements
    plaquettes=[]; rejection_work=0
    for cycle in range(1,final+1):
        if kind=="metropolis":
            metropolis_sweep(field,BETA,.5,source)
        elif kind=="hbor":
            rejection_work += hbor_cycle(field,BETA,source,1)
        else:
            raise ValueError(kind)
        if cycle in keep: plaquettes.append(average_plaquette(field))
    return plaquettes,source.draws,rejection_work

def pilot(kind):
    cold=run_chain(kind,"cold",0)
    disordered=run_chain(kind,"disordered",1)
    return {
        "cold_mean":statistics.mean(cold[0]),
        "disordered_mean":statistics.mean(disordered[0]),
        "split_rhat":split_rhat([cold[0],disordered[0]]),
        "cold_uniform_draws":cold[1],
        "disordered_uniform_draws":disordered[1],
        "cold_scalar_attempts":cold[2],
        "disordered_scalar_attempts":disordered[2],
    }

def self_test():
    zero=bytes.fromhex(
        "3e00ef2f895f40d67f5bb8e81f09a5a1"
        "2c840ec3ce9a7f3b181be188ef711a1e"
        "984ce172b9216f419f445367456d5619"
        "314a42a3da86b001387bfdb80e0cfe42")
    if chacha8_block(bytes(32),0,0)!=zero: raise AssertionError("ChaCha8 vector")
    results={kind:pilot(kind) for kind in ("metropolis","hbor")}
    expected={
        "metropolis":(0.5871001711606786,0.40110614655623167,8.803643775228498,30720,30720),
        "hbor":(0.5924613796816569,0.6002560138617226,0.9757806716162685,46992,46924),
    }
    for kind,(cm,dm,rh,cd,dd) in expected.items():
        got=results[kind]
        for value,target in ((got["cold_mean"],cm),(got["disordered_mean"],dm),(got["split_rhat"],rh)):
            if abs(value-target)>1e-12: raise AssertionError((kind,value,target))
        if got["cold_uniform_draws"]!=cd or got["disordered_uniform_draws"]!=dd:
            raise AssertionError((kind,"draw count"))
    print("ok")
    for kind,result in results.items():
        print(kind)
        for key,value in result.items():
            print(f"  {key}={value!r}")

if __name__=="__main__":
    self_test()
