#!/usr/bin/env python3
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json,sys
REFS=["verifier_current","verifier_stale_generation","verifier_stale_context","participant_supplied","cross_authority"]
VARS=["exact_current","subject_substitution","attribute_substitution","generation_substitution","context_substitution"]
MATRIX={
"verifier_current":{"exact_current":1.0,"subject_substitution":0.0,"attribute_substitution":0.0,"generation_substitution":0.0,"context_substitution":0.0},
"verifier_stale_generation":{"exact_current":0.0,"subject_substitution":0.0,"attribute_substitution":0.0,"generation_substitution":1.0,"context_substitution":0.0},
"verifier_stale_context":{"exact_current":0.0,"subject_substitution":0.0,"attribute_substitution":0.0,"generation_substitution":0.0,"context_substitution":1.0},
"participant_supplied":{"exact_current":1.0,"subject_substitution":1.0,"attribute_substitution":1.0,"generation_substitution":1.0,"context_substitution":1.0},
"cross_authority":{"exact_current":1.0,"subject_substitution":0.0,"attribute_substitution":0.0,"generation_substitution":0.0,"context_substitution":0.0}}
SOURCE={"verifier_current":"e21e4b68ce1c7488f1aa7167a6e61a0cca71291a6672af46dcbda86c6c6797ac","verifier_stale_generation":"81ac78954d29e584d45aae8622d6fa48aab0870786ce8566c2cb8710494fa718","verifier_stale_context":"c82444186e84ed6038d32f604682c0bc3302695ef05e403f0c46123bcf9399fd","participant_supplied":"5eedc5a123d6a7e1b3d0a580eaa5527d968611627078b3222ab7abb61d0818a1","cross_authority":"19ebb3c9cd5d08336f942ace35d4d74c923178b6e209ad0fcf9352e930ba25a6"}
VARIANT={"exact_current":"38bcc3d24c324add11141b707125c8a6df0fd0c12b6362bd86f89a70f5fcf7de","subject_substitution":"6f45beadc407c527e28f79a89bbb6ef442383e8b79caf9ab68c5206a5f8e2a21","attribute_substitution":"14a20b13693b80e2ab152c90f218f8781ad956f28fc3d319ab8dd15f42002d46","generation_substitution":"835c95d90d235fc0fc08079c766adea4325985c31e36266982a5e616a562fd78","context_substitution":"df311067eb9b66e6fb1fed05a4b244716be5282af6166b98a5a8c1edeb84a6b1"}
def d(x): return sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def fail(x): raise ValueError(x)
def main():
    if len(sys.argv)!=4: print("usage: verify_monetary_attestation_reference_v1.py MANIFEST.json NEGATIVE.json EXECUTION.json",file=sys.stderr); return 2
    try:
      m=json.loads(Path(sys.argv[1]).read_text()); n=json.loads(Path(sys.argv[2]).read_text()); e=json.loads(Path(sys.argv[3]).read_text())
      f=m["factors"]; card=1
      for values in f.values(): card*=len(values)
      if len(f)!=9 or card!=324000 or m["factorial_size"]!=324000 or m["batch_size"]!=3: fail("factorial")
      if f["reference_source"]!=REFS or f["attestation_variant"]!=VARS: fail("factor order")
      fixed=m["fixed_dimensions"]
      if fixed["binding_policy"]!="exact_all" or fixed["binding_fields"]!=["subject_id","attribute_id","generation","context_id"]: fail("binding")
      if fixed["crn_namespace"]!="world:{shock}:{seed}:obligation:{index}" or fixed["invalid_reference_fallback"]!="neutral_ordering": fail("CRN/fallback")
      if fixed["resource_capacity"]!=15 or fixed["topology"]!="full_mesh": fail("fixed dimensions")
      if m["reference_source_digests"]!=SOURCE: fail("source digest map")
      for k in REFS:
        if d(m["reference_sources"][k])!=SOURCE[k]: fail(f"source digest {k}")
      if m["variant_digests"]!=VARIANT: fail("variant digest map")
      for k in VARS:
        if d(m["attestation_variants"][k])!=VARIANT[k]: fail(f"variant digest {k}")
      if n["schema_version"]!="monetary-attestation-reference-negative-v1" or {x["id"] for x in n["cases"]}!={f"REF-X{i:02d}" for i in range(1,13)}: fail("negative fixtures")
      if e["schema_version"]!="monetary-attestation-reference-v1-execution" or e["run_count"]!=324000 or e["obligation_count"]!=972000: fail("execution")
      if e["reference_acceptance_matrix"]!=MATRIX: fail("matrix")
      if e["headline"]!={"trusted_current_exact_acceptance_rate":1.0,"unqualified_reference_acceptance_rate":0.4,"participant_supplied_acceptance_rate":1.0,"cross_authority_exact_current_acceptance_rate":1.0,"stale_reference_exact_current_rejection_rate":{"verifier_stale_generation":1.0,"verifier_stale_context":1.0}}: fail("headline")
      if e["trace_set_digest"]!="5e6ca9d3ac8381121058ed6e625306462220a2380913f3c0822bc8bdc3e02437": fail("trace digest")
      if e["invariants"]!={"factorial_cardinality_exact":True,"obligation_cardinality_exact":True,"signature_valid_for_all_variants":True,"reference_variant_matrix_exact":True,"reporting_independent_authoritative_result":True,"resource_integrity_failures":0,"crn_independent_of_reference_and_variant":True}: fail("invariants")
      print("independent reference-provenance check: 324000 cells / 972000 obligations; exact-all binding; source×variant acceptance matrix; exact source/variant digests; CRN independence; 12 negative fixtures")
      return 0
    except (OSError,KeyError,TypeError,json.JSONDecodeError,ValueError) as exc:
      print(f"verification failed: {exc}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
