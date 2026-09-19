#!/usr/bin/env python3
"""Validate canonical mathematical retrieval representation wire bytes."""
from __future__ import annotations
import argparse,base64,copy,json,sys
from pathlib import Path

SPARSE_VERSION='math-canonical-sparse-wire-v1'
HDC_VERSION='math-binary-hdc-wire-v1'
SPARSE_FIELDS={'version','feature_order','features'}
FEATURE_FIELDS={'feature_id','count'}
HDC_FIELDS={'version','dimension_bits','byte_count','bit_semantics','payload_encoding','bytes_base64'}
FEATURE_CHARS=set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:/-')
class ValidationError(ValueError):pass

def closed(o,fields,w):
 if not isinstance(o,dict) or set(o)!=fields:raise ValidationError(f'{w}: exact fields required')
 return o
def canonical_bytes(o):return (json.dumps(o,sort_keys=True,separators=(',',':'))+'\n').encode()
def feature_id(x,w):
 if not isinstance(x,str) or not x or len(x)>256 or any(c not in FEATURE_CHARS for c in x):raise ValidationError(f'{w}: canonical ASCII feature id required')
 return x
def posint(x,w):
 if not isinstance(x,int) or isinstance(x,bool) or not 1<=x<=2147483647:raise ValidationError(f'{w}: integer 1..2147483647 required')
 return x

def validate_sparse(doc,raw=None):
 d=closed(doc,SPARSE_FIELDS,'sparse')
 if d['version']!=SPARSE_VERSION or d['feature_order']!='FeatureIdUtf8Ascending':raise ValidationError('sparse: version/order invariant failed')
 fs=d['features']
 if not isinstance(fs,list) or not fs:raise ValidationError('sparse.features: non-empty list required')
 ids=[]
 for i,r in enumerate(fs):
  r=closed(r,FEATURE_FIELDS,f'sparse.features[{i}]');ids.append(feature_id(r['feature_id'],f'sparse.features[{i}].feature_id'));posint(r['count'],f'sparse.features[{i}].count')
 if ids!=sorted(ids):raise ValidationError('sparse.features: ascending feature_id order required')
 if len(ids)!=len(set(ids)):raise ValidationError('sparse.features: duplicate feature_id')
 if raw is not None and raw!=canonical_bytes(d):raise ValidationError('sparse: bytes are not canonical JSON + newline')
 return d

def validate_hdc(doc,raw=None):
 d=closed(doc,HDC_FIELDS,'hdc')
 if d['version']!=HDC_VERSION or d['dimension_bits']!=16384 or d['byte_count']!=2048:raise ValidationError('hdc: fixed 16,384-bit / 2,048-byte shape required')
 if d['bit_semantics']!='BinaryHVRawByteArray' or d['payload_encoding']!='StandardBase64':raise ValidationError('hdc: raw-byte/base64 invariant failed')
 s=d['bytes_base64']
 if not isinstance(s,str):raise ValidationError('hdc.bytes_base64: string required')
 try:b=base64.b64decode(s,validate=True)
 except Exception as e:raise ValidationError('hdc.bytes_base64: invalid standard base64') from e
 if len(b)!=2048:raise ValidationError('hdc.bytes_base64: decoded length must be 2048 bytes')
 if base64.b64encode(b).decode()!=s:raise ValidationError('hdc.bytes_base64: canonical padded base64 required')
 if raw is not None and raw!=canonical_bytes(d):raise ValidationError('hdc: bytes are not canonical JSON + newline')
 return d

def self_test():
 sparse={'version':SPARSE_VERSION,'feature_order':'FeatureIdUtf8Ascending','features':[{'feature_id':'depth:0:FORMULA_EQ','count':1},{'feature_id':'node:FORMULA_EQ','count':2}]};sr=canonical_bytes(sparse);validate_sparse(sparse,sr)
 hdc={'version':HDC_VERSION,'dimension_bits':16384,'byte_count':2048,'bit_semantics':'BinaryHVRawByteArray','payload_encoding':'StandardBase64','bytes_base64':base64.b64encode(bytes(range(256))*8).decode()};hr=canonical_bytes(hdc);validate_hdc(hdc,hr)
 attacks=[]
 attacks.append(('sparse-order',lambda s,h:s['features'].reverse()))
 attacks.append(('sparse-duplicate',lambda s,h:s['features'].__setitem__(1,copy.deepcopy(s['features'][0]))))
 attacks.append(('sparse-zero',lambda s,h:s['features'][0].__setitem__('count',0)))
 attacks.append(('sparse-feature-char',lambda s,h:s['features'][0].__setitem__('feature_id','bad feature')))
 attacks.append(('hdc-size',lambda s,h:h.__setitem__('byte_count',2047)))
 attacks.append(('hdc-bytes',lambda s,h:h.__setitem__('bytes_base64',base64.b64encode(b'x'*2047).decode())))
 attacks.append(('hdc-encoding',lambda s,h:h.__setitem__('payload_encoding','UrlBase64')))
 for name,attack in attacks:
  s,h=copy.deepcopy(sparse),copy.deepcopy(hdc);attack(s,h)
  try:validate_sparse(s,canonical_bytes(s));validate_hdc(h,canonical_bytes(h))
  except ValidationError:continue
  raise AssertionError(f'adversarial wire self-test unexpectedly passed: {name}')
 pretty=(json.dumps(sparse,indent=2)+'\n').encode()
 try:validate_sparse(json.loads(pretty),pretty)
 except ValidationError:pass
 else:raise AssertionError('noncanonical JSON bytes unexpectedly passed')
 print('math retrieval representation wire v1 self-test: PASS')

def main():
 p=argparse.ArgumentParser();p.add_argument('kind',nargs='?',choices=('sparse','hdc'));p.add_argument('path',nargs='?',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if a.self_test:self_test();return 0
 if not a.kind or not a.path:p.error('kind and path required unless --self-test')
 try:
  raw=a.path.read_bytes();doc=json.loads(raw.decode());(validate_sparse if a.kind=='sparse' else validate_hdc)(doc,raw)
 except (OSError,UnicodeDecodeError,json.JSONDecodeError,ValidationError) as e:print(f'INVALID: {e}',file=sys.stderr);return 1
 print('VALID');return 0
if __name__=='__main__':raise SystemExit(main())
