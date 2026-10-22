#!/usr/bin/env python3
"""Independent Sommer-type scale extraction oracle.

Qualifies r_c = sqrt((c-e)/sigma) and covariance propagation for free-e and
fixed-e static-potential fits. Standard-library only.
"""
import hashlib, json, math

ORACLE_ID = "sommer_scale_from_correlated_potential_fit_v1"

def sommer_scale(c, sigma, e):
    if not all(math.isfinite(x) for x in (c, sigma, e)):
        raise ValueError("non-finite")
    if sigma <= 0.0 or c <= e:
        raise ValueError("invalid scale domain")
    return math.sqrt((c-e)/sigma)

def free_e_variance(c, sigma, e, var_sigma, var_e, cov_sigma_e):
    r = sommer_scale(c, sigma, e)
    ds = -r/(2.0*sigma)
    de = -1.0/(2.0*sigma*r)
    variance = ds*ds*var_sigma + de*de*var_e + 2.0*ds*de*cov_sigma_e
    if variance < 0.0 or not math.isfinite(variance):
        raise ValueError("invalid propagated variance")
    return variance

def fixed_e_variance(c, sigma, e, var_sigma):
    r = sommer_scale(c, sigma, e)
    ds = -r/(2.0*sigma)
    variance = ds*ds*var_sigma
    if variance < 0.0 or not math.isfinite(variance):
        raise ValueError("invalid propagated variance")
    return variance

def numeric_gradient(c, sigma, e):
    hs = 1.0e-6
    he = 1.0e-6
    ds = (sommer_scale(c,sigma+hs,e)-sommer_scale(c,sigma-hs,e))/(2*hs)
    de = (sommer_scale(c,sigma,e+he)-sommer_scale(c,sigma,e-he))/(2*he)
    return ds,de

def main():
    sigma = 0.18
    e = 0.25
    var_sigma = 2.5e-5
    var_e = 4.0e-5
    cov_sigma_e = -1.2e-5
    fixed_var_sigma = 1.6e-5
    cs = {"r0":1.65,"r4":4.0,"r6":6.0}
    free = {}
    fixed = {}
    for name,c in cs.items():
        r = sommer_scale(c,sigma,e)
        var = free_e_variance(c,sigma,e,var_sigma,var_e,cov_sigma_e)
        free[name] = {"value":r,"sigma":math.sqrt(var)}
        e_fixed = math.pi/12.0
        r_fixed = sommer_scale(c,sigma,e_fixed)
        fixed[name] = {
            "value":r_fixed,
            "sigma":math.sqrt(fixed_e_variance(c,sigma,e_fixed,fixed_var_sigma)),
        }

    ds_num,de_num = numeric_gradient(cs["r0"],sigma,e)
    r0 = sommer_scale(cs["r0"],sigma,e)
    ds_exact = -r0/(2*sigma)
    de_exact = -1/(2*sigma*r0)
    assert abs(ds_num-ds_exact) < 1e-8
    assert abs(de_num-de_exact) < 1e-8

    var_with = free_e_variance(cs["r0"],sigma,e,var_sigma,var_e,cov_sigma_e)
    var_without = free_e_variance(cs["r0"],sigma,e,var_sigma,var_e,0.0)
    assert abs(math.sqrt(var_with)-math.sqrt(var_without)) > 1e-4

    for bad in ((1.65,0.0,0.25),(0.2,0.18,0.25)):
        try:
            sommer_scale(*bad)
            raise AssertionError("invalid domain accepted")
        except ValueError:
            pass

    result = {
        "oracle_id":ORACLE_ID,
        "free_e_fixture":{
            "sigma":sigma,"e":e,
            "var_sigma":var_sigma,"var_e":var_e,"cov_sigma_e":cov_sigma_e,
            "scales":free,
        },
        "fixed_e_fixture":{
            "sigma":sigma,"e":math.pi/12.0,
            "var_sigma":fixed_var_sigma,
            "scales":fixed,
        },
        "r0_gradient":{
            "analytic_dsigma":ds_exact,
            "numeric_dsigma":ds_num,
            "analytic_de":de_exact,
            "numeric_de":de_num,
        },
        "free_r0_sigma_without_covariance":math.sqrt(var_without),
    }
    text=json.dumps(result,sort_keys=True,separators=(",",":"))
    digest=hashlib.sha256(text.encode()).hexdigest()
    print("ok")
    print("result_sha256="+digest)
    print(text)

if __name__=="__main__":
    main()
