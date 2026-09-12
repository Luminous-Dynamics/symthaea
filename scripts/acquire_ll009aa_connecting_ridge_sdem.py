#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, shutil, stat, tempfile, urllib.parse, urllib.request, zipfile
from typing import Any

POLICY='ll009aa.sdem-archive-policy.v1'; LOCK='ll009aa.sdem-archive-source-lock.v1'; MANIFEST='ll009aa.sdem-archive-member-manifest.v1'; VERIFY='ll009aa.sdem-offline-verification.v1'
class AAError(RuntimeError): pass

def cb(v:Any)->bytes:return (json.dumps(v,sort_keys=True,indent=2,separators=(',',': '))+'\n').encode()
def hb(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def hf(p:pathlib.Path,alg='sha256')->str:
 h=hashlib.new(alg)
 with p.open('rb') as f:
  for c in iter(lambda:f.read(4<<20),b''):h.update(c)
 return h.hexdigest()
def read(p,n):
 try:v=json.loads(p.read_text())
 except (OSError,json.JSONDecodeError) as e:raise AAError(f'cannot read {n}: {e}') from e
 if not isinstance(v,dict):raise AAError(f'{n} must be object')
 return v
def selfhash(v,n):
 x=v.get('receipt_sha256');b=json.loads(json.dumps(v));b.pop('receipt_sha256',None)
 if not isinstance(x,str) or len(x)!=64 or x!=hb(cb(b)):raise AAError(f'{n} self-hash mismatch')
def write_immutable(p:pathlib.Path,v:dict):
 b=cb(v)
 if p.exists():
  if p.read_bytes()!=b:raise AAError(f'refusing to overwrite differing immutable output {p}')
  return
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def safe_url(url:str,hosts:set[str]):
 u=urllib.parse.urlsplit(url)
 if u.scheme!='https' or not u.hostname or u.hostname.lower() not in hosts or u.username or u.password or u.port not in (None,443):raise AAError(f'unsafe URL {url}')
 return u
def policy(v):
 if v.get('schema_version')!=POLICY:raise AAError('policy schema mismatch')
 for k in ('study_id','source_url','record_url','record_doi','pgda_doi','archive_filename'):
  if not isinstance(v.get(k),str) or not v[k]:raise AAError(f'policy missing {k}')
 hosts=v.get('allowed_hosts');
 if not isinstance(hosts,list) or not hosts or not all(isinstance(x,str) and x for x in hosts):raise AAError('allowed_hosts required')
 safe_url(v['source_url'],{x.lower() for x in hosts})
 c=v.get('publisher_checksum',{})
 if c.get('algorithm')!='md5' or not isinstance(c.get('value'),str) or len(c['value'])!=32:raise AAError('publisher MD5 required')
 if c.get('semantics')!='publisher_record_identity_only_not_cryptographic':raise AAError('MD5 semantics must be explicit')
 lim=v.get('limits',{}); req={'max_entries','max_member_uncompressed_bytes','max_total_uncompressed_bytes','max_compression_ratio'}
 if not req<=set(lim) or not all(isinstance(lim[k],(int,float)) and not isinstance(lim[k],bool) and lim[k]>0 for k in req):raise AAError('invalid extraction limits')
 blocked=v.get('blocked_nested_archive_suffixes')
 if not isinstance(blocked,list) or not blocked:raise AAError('blocked_nested_archive_suffixes required')
 return v
class Redirect(urllib.request.HTTPRedirectHandler):
 def __init__(self,hosts):super().__init__();self.hosts=hosts
 def redirect_request(self,req,fp,code,msg,headers,newurl):safe_url(newurl,self.hosts);return super().redirect_request(req,fp,code,msg,headers,newurl)
def make_lock(pv:dict,archive:pathlib.Path,transport:str):
 if not archive.is_file():raise AAError('archive missing')
 md5=hf(archive,'md5')
 if md5.lower()!=pv['publisher_checksum']['value'].lower():raise AAError('publisher MD5 mismatch')
 out={'schema_version':LOCK,'status':'pass','study_id':pv['study_id'],'source_url':pv['source_url'],'record_url':pv['record_url'],'record_doi':pv['record_doi'],'pgda_doi':pv['pgda_doi'],'archive_filename':pv['archive_filename'],'publisher_checksum':pv['publisher_checksum'],'archive_md5':md5,'archive_sha256':hf(archive),'archive_bytes':archive.stat().st_size,'cryptographic_identity':'archive_sha256','transport':transport,'policy_sha256':hb(cb(pv)),'non_claims':['Publisher MD5 is retained only to match the published record; SHA-256 is the promoted cryptographic identity.','This lock does not assign scientific roles to archive members or make terrain claims.']};out['receipt_sha256']=hb(cb(out));return out
def acquire(pp:pathlib.Path,dest:pathlib.Path,lockp:pathlib.Path):
 pv=policy(read(pp,'policy'));hosts={x.lower() for x in pv['allowed_hosts']};safe_url(pv['source_url'],hosts)
 if dest.exists():raise AAError('archive destination already exists')
 dest.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.ll009aa-',dir=dest.parent);os.close(fd);tp=pathlib.Path(tmp)
 try:
  req=urllib.request.Request(pv['source_url'],headers={'Accept-Encoding':'identity','User-Agent':'Symthaea-LL009AA/1'})
  op=urllib.request.build_opener(Redirect(hosts))
  with op.open(req,timeout=120) as r,tp.open('wb') as out:
   safe_url(r.geturl(),hosts);enc=(r.headers.get('Content-Encoding') or 'identity').lower()
   if enc not in ('identity',''):raise AAError(f'unsupported Content-Encoding {enc}')
   expected=r.headers.get('Content-Length');expected=int(expected) if expected is not None else None;count=0
   while True:
    chunk=r.read(4<<20)
    if not chunk:break
    out.write(chunk);count+=len(chunk)
   out.flush();os.fsync(out.fileno())
  if expected is not None and count!=expected:raise AAError('Content-Length mismatch')
  lk=make_lock(pv,tp,'https_acquisition')
  try:os.link(tp,dest)
  except FileExistsError:raise AAError('archive destination raced into existence')
  tp.unlink();write_immutable(lockp,lk);return lk
 finally:
  if tp.exists():tp.unlink()
def safe_member(name:str)->pathlib.PurePosixPath:
 if '\\' in name or not name:raise AAError(f'unsafe ZIP member {name!r}')
 p=pathlib.PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts or any(part in ('','.','..') for part in p.parts):raise AAError(f'unsafe ZIP member {name!r}')
 return p
def inspect_zip(z:zipfile.ZipFile,pv:dict):
 infos=z.infolist();lim=pv['limits']
 if len(infos)>int(lim['max_entries']):raise AAError('ZIP entry-count limit exceeded')
 seen=set();total=0;blocked=tuple(x.lower() for x in pv['blocked_nested_archive_suffixes'])
 for i in infos:
  p=safe_member(i.filename.rstrip('/') if i.is_dir() else i.filename);key=p.as_posix()
  if key in seen:raise AAError(f'duplicate normalized ZIP path {key}')
  seen.add(key)
  if i.flag_bits&1:raise AAError(f'encrypted ZIP member {key}')
  mode=(i.external_attr>>16)&0xffff;typ=stat.S_IFMT(mode) if mode else 0
  if typ not in (0,stat.S_IFREG,stat.S_IFDIR):raise AAError(f'special ZIP member {key}')
  if i.is_dir():continue
  if key.lower().endswith(blocked):raise AAError(f'nested archive member blocked {key}')
  if i.file_size>int(lim['max_member_uncompressed_bytes']):raise AAError('member size limit exceeded')
  total+=i.file_size
  if total>int(lim['max_total_uncompressed_bytes']):raise AAError('total uncompressed limit exceeded')
  ratio=i.file_size/max(i.compress_size,1)
  if ratio>float(lim['max_compression_ratio']):raise AAError(f'compression ratio limit exceeded for {key}')
 return infos,total
def extract(pp,lockp,archive,dest,manifestp):
 pv=policy(read(pp,'policy'));lk=read(lockp,'source lock');selfhash(lk,'source lock')
 if lk.get('policy_sha256')!=hb(cb(pv)) or lk.get('archive_sha256')!=hf(archive) or lk.get('archive_md5')!=hf(archive,'md5') or lk.get('archive_bytes')!=archive.stat().st_size:raise AAError('archive/source-lock mismatch')
 if lk.get('archive_md5').lower()!=pv['publisher_checksum']['value'].lower():raise AAError('publisher MD5 mismatch')
 if dest.exists():raise AAError('extraction destination already exists')
 dest.parent.mkdir(parents=True,exist_ok=True);tmp=pathlib.Path(tempfile.mkdtemp(prefix='.ll009aa-extract-',dir=dest.parent));members=[]
 try:
  with zipfile.ZipFile(archive,'r') as z:
   infos,total=inspect_zip(z,pv)
   for i in infos:
    nm=i.filename.rstrip('/') if i.is_dir() else i.filename;p=safe_member(nm);target=tmp/pathlib.Path(*p.parts)
    if i.is_dir():target.mkdir(parents=True,exist_ok=True);continue
    target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();count=0
    with z.open(i,'r') as src,target.open('xb') as out:
     while True:
      c=src.read(4<<20)
      if not c:break
      out.write(c);h.update(c);count+=len(c)
     out.flush();os.fsync(out.fileno())
    if count!=i.file_size:raise AAError(f'extracted size mismatch {p.as_posix()}')
    members.append({'path':p.as_posix(),'crc32':f'{i.CRC:08x}','compressed_bytes':i.compress_size,'uncompressed_bytes':i.file_size,'sha256':h.hexdigest()})
  members.sort(key=lambda x:x['path']);os.replace(tmp,dest)
  out={'schema_version':MANIFEST,'status':'pass','study_id':pv['study_id'],'source_lock_sha256':hf(lockp),'policy_sha256':hb(cb(pv)),'archive_filename':pv['archive_filename'],'archive_md5':lk['archive_md5'],'archive_sha256':lk['archive_sha256'],'archive_bytes':lk['archive_bytes'],'member_count':len(members),'total_extracted_file_bytes':sum(x['uncompressed_bytes'] for x in members),'members':members,'role_assignment':'intentionally_unresolved_until_ll009ab','non_claims':['Archive member names are not promoted to scientific roles by LL-009AA.','This manifest is exact-byte/extraction evidence only, not terrain or calibration evidence.']};out['receipt_sha256']=hb(cb(out));write_immutable(manifestp,out);return out
 except Exception:
  if tmp.exists():shutil.rmtree(tmp,ignore_errors=True)
  raise
def verify(manifestp,root):
 m=read(manifestp,'manifest');selfhash(m,'manifest');expected={x['path']:x for x in m.get('members',[]) if isinstance(x,dict)}
 actual=[]
 for p in root.rglob('*'):
  if p.is_file():actual.append(p.relative_to(root).as_posix())
 if set(actual)!=set(expected):raise AAError('extracted file set differs from manifest')
 for rel,e in expected.items():
  p=root/pathlib.Path(*safe_member(rel).parts)
  if p.stat().st_size!=e.get('uncompressed_bytes') or hf(p)!=e.get('sha256'):raise AAError(f'member verification failed {rel}')
 out={'schema_version':VERIFY,'status':'pass','manifest_sha256':hf(manifestp),'member_count':len(expected),'total_verified_bytes':sum(expected[x]['uncompressed_bytes'] for x in expected),'complete_file_set_match':True};out['receipt_sha256']=hb(cb(out));return out
def selftest():
 with tempfile.TemporaryDirectory() as td:
  r=pathlib.Path(td);a=r/'a.zip'
  with zipfile.ZipFile(a,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('terrain/elev.tif',b'elev'*100);z.writestr('terrain/diff.tif',b'diff'*100);z.writestr('README.txt',b'ok')
  pv={'schema_version':POLICY,'study_id':'aa','source_url':'https://zenodo.org/x.zip','record_url':'https://zenodo.org/records/1','record_doi':'10.x/a','pgda_doi':'10.x/b','archive_filename':'a.zip','allowed_hosts':['zenodo.org'],'publisher_checksum':{'algorithm':'md5','value':hf(a,'md5'),'semantics':'publisher_record_identity_only_not_cryptographic'},'limits':{'max_entries':20,'max_member_uncompressed_bytes':1_000_000,'max_total_uncompressed_bytes':2_000_000,'max_compression_ratio':500},'blocked_nested_archive_suffixes':['.zip','.tar','.tgz','.gz','.7z','.rar']};pp=r/'p.json';pp.write_bytes(cb(pv));lk=make_lock(pv,a,'local_test');lp=r/'lock.json';lp.write_bytes(cb(lk));dest=r/'out';mp=r/'m.json';m=extract(pp,lp,a,dest,mp);assert m['member_count']==3 and verify(mp,dest)['status']=='pass';assert cb(m)==cb(read(mp,'m'))
  (dest/'README.txt').write_bytes(b'bad')
  try:verify(mp,dest);raise AssertionError('tamper accepted')
  except AAError:pass
  bad=json.loads(json.dumps(pv));bad['publisher_checksum']['value']='0'*32
  try:make_lock(bad,a,'local');raise AssertionError('bad md5 accepted')
  except AAError:pass
  trav=r/'trav.zip'
  with zipfile.ZipFile(trav,'w') as z:z.writestr('../x',b'x')
  t=json.loads(json.dumps(pv));t['publisher_checksum']['value']=hf(trav,'md5');tp=r/'tp';tp.write_bytes(cb(t));tl=make_lock(t,trav,'local');tlp=r/'tl';tlp.write_bytes(cb(tl))
  try:extract(tp,tlp,trav,r/'travout',r/'travm');raise AssertionError('traversal accepted')
  except AAError:pass
  sl=r/'sym.zip';zi=zipfile.ZipInfo('link');zi.create_system=3;zi.external_attr=(stat.S_IFLNK|0o777)<<16
  with zipfile.ZipFile(sl,'w') as z:z.writestr(zi,'target')
  sv=json.loads(json.dumps(pv));sv['publisher_checksum']['value']=hf(sl,'md5');sp=r/'sp';sp.write_bytes(cb(sv));slk=make_lock(sv,sl,'local');slp=r/'slp';slp.write_bytes(cb(slk))
  try:extract(sp,slp,sl,r/'slout',r/'slm');raise AssertionError('symlink accepted')
  except AAError:pass
  print('LL-009AA SDEM archive capsule self-test: PASS')
def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
 a=sub.add_parser('acquire');a.add_argument('--policy',required=True);a.add_argument('--archive',required=True);a.add_argument('--lock',required=True)
 l=sub.add_parser('lock-local');l.add_argument('--policy',required=True);l.add_argument('--archive',required=True);l.add_argument('--lock',required=True)
 e=sub.add_parser('extract');e.add_argument('--policy',required=True);e.add_argument('--lock',required=True);e.add_argument('--archive',required=True);e.add_argument('--destination',required=True);e.add_argument('--manifest',required=True)
 v=sub.add_parser('verify');v.add_argument('--manifest',required=True);v.add_argument('--root',required=True);v.add_argument('--output')
 sub.add_parser('self-test');z=ap.parse_args()
 try:
  if z.cmd=='self-test':selftest();return 0
  if z.cmd=='acquire':o=acquire(pathlib.Path(z.policy),pathlib.Path(z.archive),pathlib.Path(z.lock))
  elif z.cmd=='lock-local':
   pp=pathlib.Path(z.policy);pv=policy(read(pp,'policy'));o=make_lock(pv,pathlib.Path(z.archive),'local_file_verification');o['policy_sha256']=hb(cb(pv));write_immutable(pathlib.Path(z.lock),o)
  elif z.cmd=='extract':o=extract(pathlib.Path(z.policy),pathlib.Path(z.lock),pathlib.Path(z.archive),pathlib.Path(z.destination),pathlib.Path(z.manifest))
  else:
   o=verify(pathlib.Path(z.manifest),pathlib.Path(z.root))
   if z.output:write_immutable(pathlib.Path(z.output),o)
  print(json.dumps(o,sort_keys=True,indent=2));return 0
 except (OSError,json.JSONDecodeError,zipfile.BadZipFile,AAError) as e:raise SystemExit(f'LL-009AA failure: {e}') from e
if __name__=='__main__':raise SystemExit(main())
