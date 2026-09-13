#!/usr/bin/env python3
"""Independent mixed-link Wilson-loop oracle composed on exact LQCD-020A bytes.

Spatial legs use APE-smeared links. Temporal legs use the original unsmeared
ensemble links. The imported APE oracle's SHA-256 is checked before import.
"""
import hashlib, importlib.util, json, math
from pathlib import Path

APE_PATH=Path(__file__).with_name('lqcd-020a-spatial-ape-oracle.py')
APE_SHA256='074614e6c2bdba1949c9c6a3bf40458b00f6502c7dd3029ee201ae8eacb1d10c'
if hashlib.sha256(APE_PATH.read_bytes()).hexdigest()!=APE_SHA256:
    raise AssertionError('APE dependency digest mismatch')
spec=importlib.util.spec_from_file_location('lqcd020a',APE_PATH)
ape=importlib.util.module_from_spec(spec); spec.loader.exec_module(ape)

CONVENTION_ID='mixed_spatial_ape_temporal_unsmeared_wilson_v1'

def trace(m): return m[0][0]+m[1][1]+m[2][2]


def mixed_oriented_link(original,smeared,site,signed_direction,dims=ape.DIMS):
    if signed_direction==0 or abs(signed_direction)>4:
        raise ValueError('invalid signed direction')
    if signed_direction>0:
        mu=signed_direction-1
        source=smeared if mu<3 else original
        return ape.get(source,site,mu,dims),ape.shift(site,mu,1,dims)
    mu=-signed_direction-1
    previous=ape.shift(site,mu,-1,dims)
    source=smeared if mu<3 else original
    return ape.dagger(ape.get(source,previous,mu,dims)),previous


def mixed_transporter(original,smeared,start,directions,dims=ape.DIMS):
    value=ape.eye(); site=start
    for direction in directions:
        edge,site=mixed_oriented_link(original,smeared,site,direction,dims)
        value=ape.mul(value,edge)
    return value,site


def mixed_rectangle(original,smeared,start,spatial_mu,r,t,dims=ape.DIMS):
    if spatial_mu>=3: raise ValueError('spatial direction required')
    if r<=0 or t<=0: raise ValueError('positive extents required')
    if r>=dims[spatial_mu] or t>=dims[3]: raise ValueError('winding rectangle')
    directions=[spatial_mu+1]*r+[4]*t+[-(spatial_mu+1)]*r+[-4]*t
    loop,end=mixed_transporter(original,smeared,start,directions,dims)
    if end!=start: raise AssertionError('rectangle did not close')
    return trace(loop).real/3.0


def average_rectangle(original,smeared,spatial_mu,r,t,dims=ape.DIMS):
    vals=[mixed_rectangle(original,smeared,s,spatial_mu,r,t,dims) for s in ape.sites(dims)]
    return sum(vals)/len(vals)


def main():
    original=ape.fixture(); smeared=ape.ape_step(original)
    identity=ape.identity_field(); identity_smeared=ape.ape_step(identity)
    identity_values={(r,t):average_rectangle(identity,identity_smeared,0,r,t) for r in (1,2) for t in (1,) }
    # Temporal extent is two in the 020A fixture, so T=2 would wind and must fail.
    try:
        average_rectangle(original,smeared,0,1,2)
        raise AssertionError('winding T=2 rectangle did not fail')
    except ValueError:
        pass

    values={}
    for mu in range(3):
        for r in (1,2):
            values[f'mu{mu}_r{r}_t1']=average_rectangle(original,smeared,mu,r,1)

    transformed=ape.gauge_transform(original)
    transformed_smeared=ape.ape_step(transformed)
    transformed_values={}
    for mu in range(3):
        for r in (1,2):
            transformed_values[f'mu{mu}_r{r}_t1']=average_rectangle(transformed,transformed_smeared,mu,r,1)
    gauge_invariance=max(abs(values[k]-transformed_values[k]) for k in values)

    # Verify temporal edge comes from the original field, not a hypothetical
    # modified temporal field, by deliberately changing a temporal link only in
    # a clone of the operator field and showing it cannot affect this API.
    operator_tampered=[[[z for z in row] for row in m] for m in smeared]
    ape.put(operator_tampered,(0,0,0,0),3,ape.embedded((0,1),(1,2,3),0.4))
    before=mixed_rectangle(original,smeared,(0,0,0,0),0,1,1)
    after=mixed_rectangle(original,operator_tampered,(0,0,0,0),0,1,1)
    temporal_source_is_original_error=abs(before-after)

    result={
        'convention_id':CONVENTION_ID,
        'ape_dependency_sha256':APE_SHA256,
        'dims':ape.DIMS,
        'alpha':ape.ALPHA,
        'identity_values':{f'r{r}_t{t}':v for (r,t),v in identity_values.items()},
        'mixed_loop_means':values,
        'gauge_invariance_max_error':gauge_invariance,
        'temporal_source_is_original_error':temporal_source_is_original_error,
    }
    if any(abs(v-1.0)>1e-15 for v in identity_values.values()): raise AssertionError(result)
    if gauge_invariance>2e-15 or temporal_source_is_original_error!=0.0: raise AssertionError(result)
    text=json.dumps(result,sort_keys=True,separators=(',',':'))
    digest=hashlib.sha256(text.encode()).hexdigest()
    print('ok'); print('result_sha256='+digest); print(json.dumps(result,sort_keys=True))

if __name__=='__main__': main()
