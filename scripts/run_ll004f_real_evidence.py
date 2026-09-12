#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, shutil, subprocess, sys, tempfile, urllib.request, venv
from typing import Any, Iterable

WHEEL_NAME='spiceypy-8.2.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl'
WHEEL_URL=f'https://github.com/AndrewAnnex/SpiceyPy/releases/download/v8.2.0/{WHEEL_NAME}'
WHEEL_SHA256='b7abbcc53320d74a1d1cf3e86f1c4bcd4bcc2e733a25c3b32fffbac677606ee1'
SPICEYPY_VERSION='8.2.0'
MANIFEST_SCHEMA='ll004g.real-evidence-run.v1'
CONFIGS=(
 'configs/lunar_transport/ll004f_de421_pgda_bridge.json',
 'configs/lunar_transport/ll004f_de440_reference.json',
)
TOOLS=(
 'scripts/generate_ll004f_spice_snapshots.py',
 'scripts/verify_ll004f_kernel_locks.py',
 'scripts/compare_ll004f_lunar_frames.py',
)

class RunnerError(RuntimeError): pass

def canonical_bytes(v:Any)->bytes:
    return (json.dumps(v,sort_keys=True,indent=2,separators=(',', ': '))+'\n').encode()

def sha256_file(p:pathlib.Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def verify_sha256(p:pathlib.Path, expected:str)->None:
    got=sha256_file(p)
    if got.lower()!=expected.lower(): raise RunnerError(f'SHA-256 mismatch for {p}: {got} != {expected}')

def write_immutable(path:pathlib.Path,payload:bytes)->None:
    if path.exists():
        if path.read_bytes()!=payload: raise RunnerError(f'refusing to overwrite differing immutable output: {path}')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_bytes(payload); os.replace(tmp,path)

def load_json(p:pathlib.Path)->dict[str,Any]:
    try:v=json.loads(p.read_text())
    except Exception as e: raise RunnerError(f'cannot read JSON {p}: {e}') from e
    if not isinstance(v,dict): raise RunnerError(f'expected object in {p}')
    return v

def collect_kernel_sources(repo:pathlib.Path)->dict[str,str]:
    out={}
    for rel in CONFIGS:
        cfg=load_json(repo/rel)
        ks=cfg.get('kernels')
        if not isinstance(ks,list): raise RunnerError(f'{rel}: kernels must be list')
        for e in ks:
            if not isinstance(e,dict) or not isinstance(e.get('filename'),str) or not isinstance(e.get('source_url'),str):
                raise RunnerError(f'{rel}: malformed kernel entry')
            fn,url=e['filename'],e['source_url']
            if fn in out and out[fn]!=url: raise RunnerError(f'conflicting source URLs for {fn}: {out[fn]} vs {url}')
            out[fn]=url
    return dict(sorted(out.items()))

def download(url:str,dest:pathlib.Path,offline:bool)->None:
    if dest.exists(): return
    if offline: raise RunnerError(f'offline mode missing required artifact: {dest}')
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.name+'.part')
    req=urllib.request.Request(url,headers={'User-Agent':'symthaea-ll004g-evidence/1'})
    try:
        with urllib.request.urlopen(req,timeout=120) as r,tmp.open('wb') as f:
            while True:
                b=r.read(1024*1024)
                if not b:break
                f.write(b)
        os.replace(tmp,dest)
    except Exception as e:
        tmp.unlink(missing_ok=True); raise RunnerError(f'download failed {url}: {e}') from e

def run(cmd:list[str],*,cwd:pathlib.Path,stdout_path:pathlib.Path|None=None)->None:
    if stdout_path:
        stdout_path.parent.mkdir(parents=True,exist_ok=True)
        with stdout_path.open('wb') as out:
            cp=subprocess.run(cmd,cwd=cwd,stdout=out,stderr=subprocess.PIPE)
    else:
        cp=subprocess.run(cmd,cwd=cwd)
    if cp.returncode!=0:
        err=(cp.stderr or b'').decode(errors='replace') if stdout_path else ''
        raise RunnerError(f'command failed ({cp.returncode}): {cmd}\n{err}')

def venv_python(vdir:pathlib.Path)->pathlib.Path:
    return vdir/('Scripts/python.exe' if os.name=='nt' else 'bin/python')

def ensure_venv(repo:pathlib.Path,work:pathlib.Path,offline:bool)->pathlib.Path:
    bootstrap=work/'bootstrap'; wheel=bootstrap/WHEEL_NAME
    download(WHEEL_URL,wheel,offline); verify_sha256(wheel,WHEEL_SHA256)
    vd=work/'venv'
    if not venv_python(vd).exists(): venv.EnvBuilder(with_pip=True,clear=False).create(vd)
    py=venv_python(vd)
    subprocess.run([str(py),'-m','pip','install','--disable-pip-version-check','--no-deps',str(wheel)],check=True,cwd=repo)
    cp=subprocess.run([str(py),'-c','import spiceypy; print(spiceypy.__version__)'],capture_output=True,text=True,check=True,cwd=repo)
    if cp.stdout.strip()!=SPICEYPY_VERSION: raise RunnerError(f'wrong SpiceyPy version: {cp.stdout.strip()}')
    return py

def tree_files(root:pathlib.Path)->list[dict[str,Any]]:
    arr=[]
    for p in sorted(x for x in root.rglob('*') if x.is_file()):
        if p.name=='artifact-manifest.json': continue
        arr.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p)})
    return arr

def resolve_git_sha(repo:pathlib.Path,declared:str|None)->str:
    if declared:return declared
    cp=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True)
    if cp.returncode!=0: raise RunnerError('git SHA must be supplied when repo is not a Git checkout')
    return cp.stdout.strip()

def build_manifest(repo:pathlib.Path,work:pathlib.Path,evidence:pathlib.Path,git_sha:str,py:pathlib.Path,kernel_sources:dict[str,str])->dict[str,Any]:
    cp=subprocess.run([str(py),'-c','import platform,spiceypy,json; print(json.dumps({"python":platform.python_version(),"spiceypy":spiceypy.__version__}))'],capture_output=True,text=True,check=True,cwd=repo)
    versions=json.loads(cp.stdout)
    tool_hashes={rel:sha256_file(repo/rel) for rel in TOOLS}
    config_hashes={rel:sha256_file(repo/rel) for rel in CONFIGS}
    kernel_hashes={fn:sha256_file(work/'kernels'/fn) for fn in kernel_sources}
    runner_rel='scripts/run_ll004f_real_evidence.py'
    return {'schema_version':MANIFEST_SCHEMA,'git_sha':git_sha,'python_version':versions['python'],'spiceypy_version':versions['spiceypy'],
            'spiceypy_wheel':{'filename':WHEEL_NAME,'source_url':WHEEL_URL,'sha256':WHEEL_SHA256},
            'runner_sha256':sha256_file(repo/runner_rel) if (repo/runner_rel).exists() else None,
            'config_sha256':config_hashes,'tool_sha256':tool_hashes,'kernel_sha256':kernel_hashes,
            'files':tree_files(evidence),'non_claim':'Generated SPICE evidence is Phase-0 research input, not navigation, launch, rendezvous, site, or release qualification.'}

def execute(repo:pathlib.Path,work:pathlib.Path,offline:bool,git_sha_arg:str|None)->dict[str,Any]:
    for rel in (*CONFIGS,*TOOLS):
        if not (repo/rel).is_file(): raise RunnerError(f'missing repository input: {rel}')
    for rel in TOOLS: run([sys.executable,rel,'--self-test'],cwd=repo)
    sources=collect_kernel_sources(repo)
    py=ensure_venv(repo,work,offline)
    kdir=work/'kernels'; [download(url,kdir/fn,offline) for fn,url in sources.items()]
    ev=work/'evidence'; (ev/'verification').mkdir(parents=True,exist_ok=True); (ev/'snapshots').mkdir(parents=True,exist_ok=True); (ev/'frame-sensitivity').mkdir(parents=True,exist_ok=True)
    for cfg,tag in ((CONFIGS[0],'de421'),(CONFIGS[1],'de440')):
        run([sys.executable,'scripts/verify_ll004f_kernel_locks.py','--config',cfg,'--kernel-dir',str(kdir)],cwd=repo,stdout_path=ev/'verification'/f'{tag}-kernel-lock-receipt.json')
        run([str(py),'scripts/generate_ll004f_spice_snapshots.py','--config',cfg,'--kernel-dir',str(kdir),'--output-dir',str(ev/'snapshots')],cwd=repo)
    run([sys.executable,'scripts/compare_ll004f_lunar_frames.py','--source',str(ev/'snapshots/ll004f_de421_pgda_bridge_v1.json'),'--destination',str(ev/'snapshots/ll004f_de440_reference_v1.json'),'--output',str(ev/'frame-sensitivity/de421-to-de440-south-pole-sensitivity.json')],cwd=repo)
    manifest=build_manifest(repo,work,ev,resolve_git_sha(repo,git_sha_arg),py,sources)
    payload=canonical_bytes(manifest); write_immutable(ev/'artifact-manifest.json',payload)
    return manifest

def self_test()->None:
    with tempfile.TemporaryDirectory() as td:
        root=pathlib.Path(td); f=root/'a'; f.write_bytes(b'x'); verify_sha256(f,hashlib.sha256(b'x').hexdigest())
        try:verify_sha256(f,'0'*64); raise AssertionError('digest mismatch not rejected')
        except RunnerError:pass
        imm=root/'imm'; write_immutable(imm,b'a'); write_immutable(imm,b'a')
        try:write_immutable(imm,b'b'); raise AssertionError('immutable overwrite not rejected')
        except RunnerError:pass
        repo=root/'repo'; (repo/'configs/lunar_transport').mkdir(parents=True)
        for rel in CONFIGS:
            (repo/rel).write_text(json.dumps({'kernels':[{'filename':'x','source_url':'u'}]}))
        assert collect_kernel_sources(repo)=={'x':'u'}
        (repo/CONFIGS[1]).write_text(json.dumps({'kernels':[{'filename':'x','source_url':'v'}]}))
        try:collect_kernel_sources(repo); raise AssertionError('conflict not rejected')
        except RunnerError:pass

def parse(argv:Iterable[str]|None=None)->argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument('--repo-root',type=pathlib.Path,default=pathlib.Path('.')); p.add_argument('--work-root',type=pathlib.Path,default=pathlib.Path('target/ll004f-real-run')); p.add_argument('--git-sha'); p.add_argument('--offline',action='store_true'); p.add_argument('--self-test',action='store_true'); return p.parse_args(argv)

def main(argv:Iterable[str]|None=None)->int:
    a=parse(argv)
    try:
        if a.self_test:self_test(); print('LL-004G self-test: PASS'); return 0
        m=execute(a.repo_root.resolve(),a.work_root.resolve(),a.offline,a.git_sha); print(json.dumps(m,sort_keys=True,indent=2)); return 0
    except (RunnerError,subprocess.CalledProcessError) as e:
        print(f'LL-004G ERROR: {e}',file=sys.stderr); return 2
if __name__=='__main__': raise SystemExit(main())
