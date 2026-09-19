#!/usr/bin/env python3
"""MATH-RET-RUNTIME-001E guarded F-arm fusion qualification."""
from __future__ import annotations
import importlib.util,json,shutil,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SEAM=ROOT/'tools'/'math-retrieval-runtime-seam'
SCRIPTS=ROOT/'.github'/'scripts'
WORK=HERE/'target'/'guarded-fusion-fixture';GRAPH=WORK/'graph';RUNTIME=WORK/'runtime-positive';NEG=WORK/'runtime-negative'
AUTH='MeasurementOnly'

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path)
    if s is None or s.loader is None: raise RuntimeError(f'cannot load {path}')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def write_config(path,values):
    path.write_text(''.join(f'{k}={v}\n' for k,v in values.items()),encoding='utf-8')

def main()->int:
    if WORK.exists(): shutil.rmtree(WORK)
    GRAPH.mkdir(parents=True)
    base=load(SEAM/'qualify_contract_adapter.py','fusion_parent_adapter')
    guarded=load(HERE/'qualify_guarded_adapter.py','fusion_guarded_adapter')
    guarded.WORK=WORK;guarded.GRAPH=GRAPH;guarded.RUNTIME=RUNTIME
    graph=guarded.build_real_candidate_graph(base)

    base.run(sys.executable,SCRIPTS/'validate-math-retrieval-candidate-set.py',graph['candidate_path'])
    report_path=WORK/'graph-report.json'
    base.run(sys.executable,SCRIPTS/'validate-math-retrieval-graph-v1.1.py',graph['bundle_path'],'--repo-root',ROOT,'--report',report_path)
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if not report.get('all_checks_passed'): raise RuntimeError('graph did not qualify')
    if report['shared_candidate_set_sha256']!=graph['candidate_digest']: raise RuntimeError('candidate digest drift')

    f_binding_path=GRAPH/'bindings'/'F.json';s_path=GRAPH/'indices'/'S.json';n_path=GRAPH/'indices'/'N.json';fusion_path=GRAPH/'fusion.json'
    fb=json.loads(f_binding_path.read_text());sx=json.loads(s_path.read_text());nx=json.loads(n_path.read_text());fusion=json.loads(fusion_path.read_text())
    if fb['mode']!='Fusion': raise RuntimeError('F binding is not Fusion')
    if fusion['fusion']['method']!='ReciprocalRankFusion': raise RuntimeError('001E fixture freezes RRF')
    if fb['syntax_index_manifest_sha256']!=base.digest_bytes(s_path.read_bytes()): raise RuntimeError('F/S binding drift')
    if fb['normal_form_index_manifest_sha256']!=base.digest_bytes(n_path.read_bytes()): raise RuntimeError('F/N binding drift')
    if fb['fusion_policy_sha256']!=base.digest_bytes(fusion_path.read_bytes()): raise RuntimeError('F fusion binding drift')

    cfg0=base.write_runtime_config(graph,report,report_path)
    vals={}
    for line in cfg0.read_text().splitlines():
        if line.strip():
            k,v=line.split('=',1);vals[k]=v
    vals.update({
        'retrieval_binding_sha256':base.digest_bytes(f_binding_path.read_bytes()),
        'syntax_index_manifest_sha256':base.digest_bytes(s_path.read_bytes()),
        'syntax_index_artifact_sha256':sx['index']['index_artifact_sha256'],
        'normal_index_manifest_sha256':base.digest_bytes(n_path.read_bytes()),
        'normal_index_artifact_sha256':nx['index']['index_artifact_sha256'],
        'fusion_policy_sha256':base.digest_bytes(fusion_path.read_bytes()),
        'rrf_k':fusion['fusion']['rrf_k'],
        'candidate_source_sha256s':','.join(graph['candidate_sources']),
    })
    cfg=WORK/'fusion-runtime-config.txt';write_config(cfg,vals)

    # Strong raw-channel membership negative: illegal 0xee enters only N.
    base.run('cargo','run','--quiet','--manifest-path',HERE/'Cargo.toml','--locked','--bin','emit_guarded_fusion_fixture','--',cfg,NEG,'--inject-illegal-normal')
    if NEG.exists(): raise RuntimeError('illegal normal-form source created evidence')

    base.run('cargo','run','--quiet','--manifest-path',HERE/'Cargo.toml','--locked','--bin','emit_guarded_fusion_fixture','--',cfg,RUNTIME)
    trace=RUNTIME/'trace.json';trace_report=WORK/'trace-report.json'
    base.run(sys.executable,SCRIPTS/'validate-math-retrieval-trace.py',trace,graph['bundle_path'],'--repo-root',ROOT,'--report',trace_report)
    tr=json.loads(trace_report.read_text());
    if not tr.get('all_checks_passed'): raise RuntimeError('fusion trace did not qualify')

    membership=WORK/'candidate-membership-report.json'
    base.run(sys.executable,SCRIPTS/'validate-math-retrieval-candidate-membership.py',graph['candidate_path'],trace,graph['bundle_path'],'--repo-root',ROOT,'--report',membership)
    mr=json.loads(membership.read_text());
    if not mr.get('all_checks_passed'): raise RuntimeError('fusion candidate membership failed')

    audit=base.build_payload_audit(trace,graph,report_path)
    # Parent helper's audit_id is intentionally non-authoritative but rename it
    # here so the successor evidence is human-unambiguous.
    ad=json.loads(audit.read_text());ad['audit_id']='math-ret-runtime-001e-fixture-audit';base.write_json(audit,ad)
    payload_report=WORK/'payload-report.json'
    base.run(sys.executable,SCRIPTS/'validate-math-retrieval-payload-audit.py',audit,trace,graph['bundle_path'],'--repo-root',ROOT,'--payload-root',RUNTIME,'--report',payload_report)
    pr=json.loads(payload_report.read_text());
    if not pr.get('all_checks_passed'): raise RuntimeError('fusion payload audit failed')

    td=json.loads(trace.read_text())
    expected=[f'sha256:{0xa2:064x}',f'sha256:{0xa1:064x}',f'sha256:{0xa3:064x}']
    actual=td['retrieval']['fusion']['fused_ranked_source_object_digests']
    if actual!=expected: raise RuntimeError(f'unexpected frozen RRF order: {actual}')
    if td['packing']['input_ranked_source_object_digests']!=expected: raise RuntimeError('packer input differs from fused rank')
    if td['resources']['retrieval_queries_used']!=2: raise RuntimeError('fusion did not use exactly two retrieval queries')

    summary={'authority':AUTH,'candidate_set_sha256':graph['candidate_digest'],'candidate_count':len(graph['candidate_sources']),'experiment_sha256':graph['experiment_digest'],'fusion_binding_sha256':vals['retrieval_binding_sha256'],'syntax_index_sha256':vals['syntax_index_manifest_sha256'],'normal_index_sha256':vals['normal_index_manifest_sha256'],'fusion_policy_sha256':vals['fusion_policy_sha256'],'bundle_sha256':base.digest_bytes(graph['bundle_path'].read_bytes()),'trace_sha256':base.digest_bytes(trace.read_bytes()),'candidate_membership_report_sha256':base.digest_bytes(membership.read_bytes()),'payload_audit_sha256':base.digest_bytes(audit.read_bytes()),'fused_order':actual,'illegal_normal_candidate_rejected_before_evidence':True,'graph_all_checks_passed':True,'trace_all_checks_passed':True,'candidate_membership_all_checks_passed':True,'payload_all_checks_passed':True}
    (WORK/'qualification-summary.json').write_bytes(base.canonical_bytes(summary));print(json.dumps(summary,sort_keys=True,indent=2));return 0

if __name__=='__main__': raise SystemExit(main())
