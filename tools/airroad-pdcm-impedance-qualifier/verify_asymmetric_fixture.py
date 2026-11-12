#!/usr/bin/env python3
"""Independent exact-rational verifier for the asymmetric two-converter fixture."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "docs/research/fixtures/airroad-pdcm-asymmetric-two-converter-v0.1.json"


def q(item):
    return Fraction(int(item["n"]), int(item["d"]))


def cadd(a, b):
    return a[0] + b[0], a[1] + b[1]


def cmul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def cdiv(a, b):
    den = b[0] * b[0] + b[1] * b[1]
    return ((a[0] * b[0] + a[1] * b[1]) / den,
            (a[1] * b[0] - a[0] * b[1]) / den)


def complex_pair(obj):
    return q(obj["re"]), q(obj["im"])


def determinant_2(m):
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def determinant_3(m):
    a,b,c=m[0]; d,e,f=m[1]; g,h,i=m[2]
    return a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)


def main():
    data=json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema_version"]=="0.1.0"
    assert data["evidence_class"]=="synthetic_exact_rational_reference"
    p=data["model"]["parameters"]
    eq=data["equilibrium"]

    vs,rs=q(p["Vs"]),q(p["Rs"])
    c0=q(p["C0"])
    r1,r2=q(p["R1"]),q(p["R2"])
    c1,c2=q(p["C1"]),q(p["C2"])
    p1,p2=q(p["P1"]),q(p["P2"])
    v0,v1,v2=q(eq["V0"]),q(eq["V1"]),q(eq["V2"])

    isrc=(vs-v0)/rs
    i1=(v0-v1)/r1
    i2=(v0-v2)/r2
    assert isrc==q(eq["source_current"])
    assert i1==q(eq["branch_current_1"])
    assert i2==q(eq["branch_current_2"])
    assert i1==p1/v1 and i2==p2/v2
    assert isrc==i1+i2

    psrc=vs*isrc
    pload=p1+p2
    source_loss=isrc*isrc*rs
    f1loss=i1*i1*r1
    f2loss=i2*i2*r2
    assert psrc==q(eq["source_power"])
    assert pload==q(eq["load_power"])
    assert source_loss==q(eq["source_resistor_loss"])
    assert f1loss==q(eq["feeder_1_loss"])
    assert f2loss==q(eq["feeder_2_loss"])
    assert psrc==pload+source_loss+f1loss+f2loss

    gs,g1,g2=1/rs,1/r1,1/r2
    h1,h2=p1/(v1*v1),p2/(v2*v2)
    isolated=[(-g1+h1)/c1,(-g2+h2)/c2]
    assert isolated==[q(x) for x in data["isolated_branch_poles"]]
    assert all(x<0 for x in isolated)

    jac=[
        [-(gs+g1+g2)/c0,g1/c0,g2/c0],
        [g1/c1,(-g1+h1)/c1,Fraction(0)],
        [g2/c2,Fraction(0),(-g2+h2)/c2]
    ]
    observed=[[q(x) for x in row] for row in data["coupled_linearization"]["jacobian"]]
    assert jac==observed

    trace=sum(jac[i][i] for i in range(3))
    minors=(
        determinant_2([[jac[0][0],jac[0][1]],[jac[1][0],jac[1][1]]])
        + determinant_2([[jac[0][0],jac[0][2]],[jac[2][0],jac[2][2]]])
        + determinant_2([[jac[1][1],jac[1][2]],[jac[2][1],jac[2][2]]])
    )
    det=determinant_3(jac)
    coeff=[Fraction(1),-trace,minors,-det]
    expected=[q(x) for x in data["coupled_linearization"]["characteristic_polynomial_coefficients_descending"]]
    assert coeff==expected
    assert coeff==[Fraction(1),Fraction(16,5),Fraction(29,40),Fraction(-1,80)]
    assert coeff[-1]<0

    fd=data["frequency_domain"]
    expected_poly=[q(x) for x in fd["closed_loop_characteristic_polynomial_descending"]]
    assert expected_poly==[2*coeff[0],2*coeff[1],2*coeff[2],2*coeff[3]]

    # Recompute exact SISO port values at the frozen frequencies.
    for point in fd["expected_points"]:
        omega=q(point["omega_rad_s"])
        jw=(Fraction(0),omega)
        y1=cdiv((jw[1]-Fraction(1,2),jw[0]),(jw[1]+Fraction(1,2),jw[0]))
        # Y2 = (s - 1/20)/(2s + 2/5)
        y2=cdiv((jw[1]-Fraction(1,20),jw[0]),(2*jw[1]+Fraction(2,5),2*jw[0]))
        zsrc=cdiv((Fraction(1),Fraction(0)),(Fraction(1),omega))
        ret=cmul(zsrc,cadd(y1,y2))
        assert y1==complex_pair(point["branch_1"])
        assert y2==complex_pair(point["branch_2"])
        assert zsrc==complex_pair(point["source_impedance"])
        assert ret==complex_pair(point["return_ratio"])

    print("PASS exact asymmetric equilibrium, KCL and power closure")
    print("PASS both isolated branches have stable local poles")
    print("PASS full coupled Jacobian and positive-real-pole characteristic")
    print("PASS exact SISO return-ratio polynomial")
    print("PASS complex port responses at DC and 1 rad/s")


if __name__=="__main__":
    main()
