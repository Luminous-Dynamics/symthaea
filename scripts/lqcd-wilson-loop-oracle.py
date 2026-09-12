#!/usr/bin/env python3
"""Independent rectangular Wilson-loop oracle for LQCD-017A.

Standard-library only; imports no Symthaea/Rust code.
"""
import math

DIMS=(3,3,2,2)

def identity():
    return [[1+0j if i==j else 0j for j in range(3)] for i in range(3)]
def mul(a,b):
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def dagger(a):
    return [[a[j][i].conjugate() for j in range(3)] for i in range(3)]
def trace(a):
    return a[0][0]+a[1][1]+a[2][2]
def diagonal(theta,phi):
    phases=(theta,phi,-theta-phi)
    out=[[0j]*3 for _ in range(3)]
    for i,p in enumerate(phases):
        out[i][i]=complex(math.cos(p),math.sin(p))
    return out
def sites(dims=DIMS):
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                for t in range(dims[3]):
                    yield (x,y,z,t)
def index(site,dims=DIMS):
    x,y,z,t=site
    return (((x*dims[1]+y)*dims[2]+z)*dims[3]+t)
def shift(site,mu,step,dims=DIMS):
    out=list(site); out[mu]=(out[mu]+step)%dims[mu]; return tuple(out)
def identity_field(dims=DIMS):
    return [identity() for _ in range(math.prod(dims)*4)]
def link(field,site,mu,dims=DIMS):
    return field[index(site,dims)*4+mu]
def set_link(field,site,mu,value,dims=DIMS):
    field[index(site,dims)*4+mu]=value

def rectangle(field,start,mu,nu,r,t,dims=DIMS):
    product=identity(); site=start
    for _ in range(r):
        product=mul(product,link(field,site,mu,dims)); site=shift(site,mu,1,dims)
    for _ in range(t):
        product=mul(product,link(field,site,nu,dims)); site=shift(site,nu,1,dims)
    for _ in range(r):
        site=shift(site,mu,-1,dims)
        product=mul(product,dagger(link(field,site,mu,dims)))
    for _ in range(t):
        site=shift(site,nu,-1,dims)
        product=mul(product,dagger(link(field,site,nu,dims)))
    if site != start:
        raise AssertionError("path did not close")
    return trace(product)/3

def average_rectangle(field,mu,nu,r,t,dims=DIMS):
    values=[rectangle(field,site,mu,nu,r,t,dims) for site in sites(dims)]
    return sum(values)/len(values)

def static_potential(w_t,w_t1,a_t=1.0):
    if w_t<=0 or w_t1<=0 or a_t<=0:
        raise ValueError
    return math.log(w_t/w_t1)/a_t

def creutz(w_rt,w_rm_tm,w_r_tm,w_rm_t):
    if min(w_rt,w_rm_tm,w_r_tm,w_rm_t)<=0:
        raise ValueError
    return -math.log((w_rt*w_rm_tm)/(w_r_tm*w_rm_t))

def self_test():
    field=identity_field()
    for r,t in ((1,1),(2,1),(1,2),(2,2)):
        value=average_rectangle(field,0,1,r,t)
        if abs(value-1)>1e-14:
            raise AssertionError(("identity",r,t,value))

    set_link(field,(0,0,0,0),0,diagonal(0.3,-0.1))
    loop=rectangle(field,(0,0,0,0),0,1,1,1)
    expected_loop=complex(0.9768024107482912,-0.0009941802601832657)
    if abs(loop-expected_loop)>1e-14:
        raise AssertionError(("loop",loop))
    expected={
        (1,1):0.9987112450415717,
        (2,1):0.9974224900831435,
        (1,2):0.9987112450415717,
        (2,2):0.9974224900831435,
    }
    for shape,target in expected.items():
        got=average_rectangle(field,0,1,*shape).real
        if abs(got-target)>1e-14:
            raise AssertionError((shape,got,target))

    sigma=0.23
    def area(r,t): return math.exp(-sigma*r*t)
    chi=creutz(area(2,2),area(1,1),area(2,1),area(1,2))
    if abs(chi-sigma)>1e-14:
        raise AssertionError(("creutz",chi))
    mass=0.41
    v=static_potential(math.exp(-mass*3),math.exp(-mass*4))
    if abs(v-mass)>1e-14:
        raise AssertionError(("potential",v))

    print("ok")
    print(f"localized_loop_re={loop.real:.17g}")
    print(f"localized_loop_im={loop.imag:.17g}")
    for shape,target in expected.items():
        print(f"average_{shape[0]}x{shape[1]}={target:.17g}")
    print(f"area_law_creutz={chi:.17g}")
    print(f"exponential_static_potential={v:.17g}")

if __name__=="__main__":
    self_test()
