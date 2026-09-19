#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

class ValidationError(ValueError): pass

def load_parent():
    p=Path(__file__).with_name('validate-math-retrieval-graph.py')
    s=importlib.util.spec_from_file_location('sym_graph_v1_parent',p)
    if s is None or s.loader is None: raise ValidationError(f'cannot load parent graph validator: {p}')
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
    m.VALIDATORS['FusionPolicy']=('validate-math-retrieval-fusion-v1.1.py','validate')
    return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('bundle',nargs='?',type=Path); ap.add_argument('--repo-root',type=Path,default=Path.cwd()); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--report',type=Path); a=ap.parse_args()
    try:
        p=load_parent()
        if a.self_test:
            p.self_test()
            f=Path(__file__).with_name('validate-math-retrieval-fusion-v1.1.py')
            s=importlib.util.spec_from_file_location('sym_fusion_v11',f); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.self_test()
            print('math-retrieval graph v1.1 profile self-test: PASS'); return 0
        if a.bundle is None: ap.error('bundle path required unless --self-test')
        r=p.validate_graph(a.bundle,a.repo_root,True)
    except (OSError,json.JSONDecodeError,ValidationError,ValueError) as e:
        print(f'INVALID: {e}',file=sys.stderr); return 1
    out=json.dumps(r,sort_keys=True,separators=(',',':'))
    if a.report: a.report.write_text(out+'\n',encoding='utf-8')
    print(out); return 0
if __name__=='__main__': raise SystemExit(main())
