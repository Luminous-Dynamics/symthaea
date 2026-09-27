#!/usr/bin/env python3
import copy, hashlib, json, pathlib, subprocess

SOURCE_PARENT="fbd7a754ea8389ca93f7680d93ed8b48553e6376"
SOURCE_HEAD="c0f46c7fe0687fd516c1c9d3e217afee25784b05"
DOC_BLOB="14ec1fda1199b7bcdd840cc1b961ed2651943996"
CORPUS_BLOB="c9f7344acf21f9f84ad2f87a1cb82febd5f5d024"
CORPUS_SHA256="f9558e789ae9095830b50b0066cf5dca6eea0fd23c2ae1b2e3f51984bb46fab6"
SCHEMA="civ-eng-bearing-001a-reference-v1"
CORPUS=pathlib.Path("docs/engineering/data/civ_eng_bearing_001a_reference_v1.json")
DOC=pathlib.Path("docs/engineering/CIV_ENG_BEARING_001A.md")
AXES=["civ","bottleneck","material_research","material_state","process","precision","interface","function","experiment","uncertainty","return","generation","effect","authority"]
ROUTES={"Material":"MAT_OPPORTUNITY_THEN_MAT_ENG","PrecisionMetrology":"ENG_TOL_ENG_MEAS","InterfaceSurface":"MAT_INTERFACE_COMPONENT","Process":"MFG_PROC_COMPONENT","LifecycleEvidence":"MFG_LIFE_IND_COMP","MultipleOrUnresolved":"EVIDENCE_ACTION_BEFORE_INTERVENTION","NoMaterialBottleneck":"NON_MATERIAL_OR_BOUNDED_IMPORT"}

def die(m): raise AssertionError(m)
def git(*a): return subprocess.check_output(["git",*a], text=True).strip()
def git_blob(path): return git("hash-object",str(path))

def validate_raw(x):
    if x["profile"] not in {"B-L","B-P"}: die("unknown profile")
    if x["civ_leverage"] not in {"High","Low","Unresolved"}: die("bad civ leverage")
    if x["bottleneck"] not in ROUTES: die("unknown blocker")
    if x["route"] != ROUTES[x["bottleneck"]]: die("wrong route")
    if x["material_attribution"] not in {"Absent","Negative","Positive"}: die("bad attribution")
    if x["material_evidence"] not in {"None","Candidate","Computed","Qualified"}: die("bad material evidence")
    if x["experiment_level"] not in {"None","OneArticleManyReadings","OneBatchManyArticles","AdaptiveExploration","IndependentBatches"}: die("bad experiment")
    if x["uncertainty_state"] not in {"Insufficient","Nominal","CrossesHard","MeasurementDominates","Robust","NA"}: die("bad uncertainty")
    for k in ["predicted_gain","realized_gain"]:
        if type(x[k]) is not int or x[k] < 0: die("bad gain")
    return True

def derive(x):
    validate_raw(x)
    civ={"High":"H","Low":"L","Unresolved":"U"}[x["civ_leverage"]]
    bn={"Material":"M","PrecisionMetrology":"P","InterfaceSurface":"I","Process":"R","LifecycleEvidence":"L","MultipleOrUnresolved":"X","NoMaterialBottleneck":"N"}[x["bottleneck"]]
    if x["bottleneck"]=="Material":
        if (not x["material_leverage_asserted"]) or x["material_attribution"]=="Absent": mr="B"
        elif x["material_attribution"]=="Negative": mr="D"
        else: mr="E"
    elif x["material_leverage_asserted"] and x["material_attribution"]=="Negative": mr="D"
    else: mr="N"
    if x["material_evidence"]=="Candidate": ms="C"
    elif x["material_evidence"]=="Qualified" and x["material_applicable"]: ms="Q"
    elif x["material_evidence"]=="Qualified": ms="A"
    elif x["material_evidence"]=="None": ms="N" if x["bottleneck"]!="Material" else "I"
    else: ms="I"
    if x["bottleneck"] in {"Process","Material","MultipleOrUnresolved"} or x["process_current"]:
        pr="V" if not x["venue_ok"] else ("Q" if x["process_current"] else "I")
    else: pr="N"
    if x["fit_evidence"] and x["precision_current"] and x["calibration_current"]: prec="Q"
    elif x["fit_evidence"] and not x["precision_current"]: prec="F"
    elif not x["calibration_current"]: prec="S"
    elif x["bottleneck"]=="PrecisionMetrology" or x["fit_evidence"]: prec="I"
    else: prec="N"
    if (not x["lubrication_dependency_ok"]) and (x["interface_current"] or x["bottleneck"]=="InterfaceSurface"): inter="I"
    elif x["interface_current"]:
        inter="L" if (x["lifetime_required"] and not x["interface_lifecycle"]) else "Q"
    elif x["bottleneck"]=="InterfaceSurface" or x["fit_evidence"]: inter="I"
    else: inter="N"
    if x["functional_observation"] and x["duty_match"]:
        if x["profile"]=="B-L": fn="L"
        else: fn="Q" if ((not x["lifetime_required"]) or x["lifetime_evidence"]) else "F"
    elif x["functional_observation"]: fn="F"
    else: fn="I" if x["fit_evidence"] else "N"
    ex={"IndependentBatches":"Q","OneArticleManyReadings":"A","OneBatchManyArticles":"B","AdaptiveExploration":"E","None":"N"}[x["experiment_level"]]
    uq={"Robust":"R","Nominal":"N","CrossesHard":"C","MeasurementDominates":"M","Insufficient":"I","NA":"X"}[x["uncertainty_state"]]
    if not x["returned_generation_match"]: ret="S"
    else:
        basic=(x["fit_evidence"] and x["precision_current"] and x["calibration_current"] and inter in {"Q","L"} and x["functional_observation"] and x["duty_match"] and x["process_current"] and x["venue_ok"] and x["lubrication_dependency_ok"])
        if x["profile"]=="B-P":
            eligible=basic and inter=="Q" and fn=="Q" and uq=="R" and ex=="Q" and (x["bottleneck"]!="Material" or ms=="Q")
        else:
            eligible=basic and fn in {"L","Q"} and uq in {"R","N"} and ex in {"Q","A","B"}
        ret="E" if eligible else "B"
    if x["g2"] and x["renewal_tooling"] and x["renewal_metrology"] and x["renewal_material"]: gen="2"
    elif x["g1"] and not x["g2"]: gen="1"
    elif x["g1"] and x["g2"] and not (x["renewal_tooling"] and x["renewal_metrology"] and x["renewal_material"]): gen="D"
    elif not x["g1"] and not x["g2"]: gen="N"
    else: gen="U"
    if x["migration"] and x["realized_gain"]>0: eff="B"
    elif x["realized_gain"]==0 and x["predicted_gain"]==0: eff="U" if ret!="E" else "N"
    elif x["realized_gain"]==0: eff="N"
    elif x["predicted_gain"]>x["realized_gain"]: eff="O"
    elif x["predicted_gain"]<x["realized_gain"]: eff="I"
    else: eff="G"
    return [civ,bn,mr,ms,pr,prec,inter,fn,ex,uq,ret,gen,eff,"0"]

def apply(base,o):
    x=copy.deepcopy(base)
    unknown=set(o)-set(base)
    if unknown: die("unknown override keys")
    x.update(o)
    return x

def check_mutations(base):
    def d(**o): return derive(apply(base,o))
    checks=[]
    checks += [
      d(bottleneck="Material",route=ROUTES["Material"],material_leverage_asserted=True,material_attribution="Absent")[2]=="B",
      d(bottleneck="Material",route=ROUTES["Material"],material_leverage_asserted=True,material_attribution="Negative")[2]=="D",
      d(bottleneck="Material",route=ROUTES["Material"],material_leverage_asserted=True,material_attribution="Positive")[2]=="E",
      d(material_evidence="Candidate")[3]=="C",
      d(bottleneck="Material",route=ROUTES["Material"],material_leverage_asserted=True,material_attribution="Positive",material_evidence="Qualified",material_applicable=False)[3]=="A",
      d(bottleneck="Process",route=ROUTES["Process"],process_current=False)[4]=="I",
      d(bottleneck="Process",route=ROUTES["Process"],process_current=True,venue_ok=False)[4]=="V",
      d(fit_evidence=True,precision_current=False)[5]=="F",
      d(fit_evidence=True,precision_current=True,calibration_current=False)[5]=="S",
      d(fit_evidence=True,precision_current=True,interface_current=False)[6]=="I",
      d(fit_evidence=True,precision_current=True,interface_current=True,interface_lifecycle=True,lubrication_dependency_ok=False)[6]=="I",
      d(fit_evidence=True,precision_current=True,interface_current=True,interface_lifecycle=False)[6]=="L",
      d(fit_evidence=True,precision_current=True,interface_current=True,interface_lifecycle=True,process_current=True,functional_observation=True,duty_match=False)[7]=="F",
      d(profile="B-L",generation="BL-X",fit_evidence=True,precision_current=True,interface_current=True,interface_lifecycle=True,process_current=True,functional_observation=True,duty_match=True,lifetime_required=False,experiment_level="OneArticleManyReadings",uncertainty_state="Nominal")[10]=="E",
      d(profile="B-P",fit_evidence=True,precision_current=True,interface_current=True,interface_lifecycle=True,process_current=True,functional_observation=True,duty_match=True,lifetime_evidence=False,experiment_level="IndependentBatches",uncertainty_state="Robust")[10]=="B",
      d(experiment_level="OneArticleManyReadings")[8]=="A",
      d(experiment_level="OneBatchManyArticles")[8]=="B",
      d(experiment_level="AdaptiveExploration")[8]=="E",
      d(uncertainty_state="Nominal")[9]=="N",
      d(uncertainty_state="CrossesHard")[9]=="C",
      d(uncertainty_state="MeasurementDominates")[9]=="M",
      d(uncertainty_state="Robust")[9]=="R",
      d(returned_generation_match=False)[10]=="S",
      d(g1=True,g2=False)[11]=="1",
      d(g1=True,g2=True,renewal_tooling=False,renewal_metrology=True,renewal_material=True)[11]=="D",
      d(g1=True,g2=True,renewal_tooling=True,renewal_metrology=False,renewal_material=True)[11]=="D",
      d(g1=True,g2=True,renewal_tooling=True,renewal_metrology=True,renewal_material=False)[11]=="D",
      d(g1=True,g2=True,renewal_tooling=True,renewal_metrology=True,renewal_material=True)[11]=="2",
      d(predicted_gain=10,realized_gain=3)[12]=="O",
      d(predicted_gain=2,realized_gain=5)[12]=="I",
      d(predicted_gain=4,realized_gain=2,migration=True)[12]=="B",
      d(authority_input=True)[13]=="0",
    ]
    if not all(checks): die("mutation suite failed")
    for o in [{"profile":"BAD"},{"bottleneck":"BAD"},{"material_evidence":"BAD"},{"experiment_level":"BAD"},{"uncertainty_state":"BAD"},{"bottleneck":"Material","route":"ENG_TOL_ENG_MEAS"},{"predicted_gain":-1}]:
        try: d(**o)
        except AssertionError: pass
        else: die(f"mutation should fail closed: {o}")

def main():
    raw=CORPUS.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CORPUS_SHA256: die("corpus digest drift")
    if git_blob(DOC)!=DOC_BLOB: die("doc blob drift")
    if git_blob(CORPUS)!=CORPUS_BLOB: die("corpus blob drift")
    data=json.loads(raw)
    if data["schema"]!=SCHEMA: die("schema drift")
    if data["authority_ceiling"]!="SyntheticAnalysisOnly_NoPhysicalExecutionAuthority": die("authority drift")
    if data["axes"]!=AXES: die("axis drift")
    if data["route_map"]!=ROUTES: die("route map drift")
    if [c["id"] for c in data["cases"]] != [f"C{i:02d}" for i in range(1,41)]: die("case order drift")
    if len(set(c["id"] for c in data["cases"]))!=40: die("duplicate cases")
    base=data["base"]
    expected_keys=set(base)
    for c in data["cases"]:
        if set(c["o"])-expected_keys: die("case override key drift")
        if len(c["e"])!=len(AXES): die("expected vector width")
        x=apply(base,c["o"])
        got=derive(x)
        if got!=c["e"]: die(f"{c['id']} mismatch: got {got}, expected {c['e']}")
        for axis,code in zip(AXES,c["e"]):
            if code not in data["codes"][axis]: die(f"{c['id']} unknown code {axis}:{code}")
    check_mutations(base)
    print("CIV-ENG-BEARING-001A1: all 40 reference cases and mutation guards PASS")

if __name__=="__main__": main()
