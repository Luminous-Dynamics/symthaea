#!/usr/bin/env python3
import hashlib,json,math

ID="joint_jackknife_sommer_scale_v1"
CENTRAL={"sigma":0.18,"e":0.25}
TARGETS=(("r0",1.65),("r4",4.0),("r6",6.0))

def scale(parameters,c):
    sigma=parameters["sigma"]; e=parameters["e"]
    if not math.isfinite(sigma) or sigma<=0: raise ValueError("invalid sigma")
    if not math.isfinite(e) or not math.isfinite(c) or c<=e: raise ValueError("invalid scale target")
    return math.sqrt((c-e)/sigma)

def main():
    replicates=[]
    for k in range(12):
        theta=2*math.pi*(k+1)/12
        replicates.append({
            "sigma":0.18+0.0025*math.sin(theta)+0.001*math.cos(2*theta),
            "e":0.25+0.008*math.cos(theta)-0.003*math.sin(2*theta),
        })
    central_scales=[scale(CENTRAL,c) for _,c in TARGETS]
    replicate_scales=[[scale(p,c) for _,c in TARGETS] for p in replicates]
    n=len(replicate_scales)
    mean=[sum(row[j] for row in replicate_scales)/n for j in range(len(TARGETS))]
    factor=(n-1)/n
    covariance=[
        [
            factor*sum((row[i]-mean[i])*(row[j]-mean[j]) for row in replicate_scales)
            for j in range(len(TARGETS))
        ]
        for i in range(len(TARGETS))
    ]
    standard_errors=[math.sqrt(covariance[i][i]) for i in range(len(TARGETS))]
    result={
        "oracle_id":ID,
        "central_parameters":CENTRAL,
        "targets":[{"id":name,"c":c} for name,c in TARGETS],
        "central_scales":central_scales,
        "replicate_parameters":replicates,
        "replicate_scales":replicate_scales,
        "jackknife_mean":mean,
        "jackknife_covariance":covariance,
        "standard_errors":standard_errors,
    }
    text=json.dumps(result,sort_keys=True,separators=(",",":"))
    digest=hashlib.sha256(text.encode()).hexdigest()
    expected="f6c7470dd54ad15393791339605a0fa434496d335731337697a023320f2acd9f"
    if digest!=expected: raise AssertionError((digest,expected))
    if max(abs(covariance[i][j]-covariance[j][i]) for i in range(3) for j in range(3))>1e-15:
        raise AssertionError("covariance symmetry")
    try:
        scale({"sigma":0.18,"e":1.65},1.65)
        raise AssertionError("c<=e did not fail")
    except ValueError:
        pass
    print("ok")
    print("result_sha256="+digest)
    print(text)

if __name__=="__main__":
    main()
