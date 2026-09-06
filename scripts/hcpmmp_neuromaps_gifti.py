"""Strict GIFTI label decoding and HCP-MMP1 semantic normalization."""
from __future__ import annotations
import base64,gzip,struct,xml.etree.ElementTree as ET,zlib
from pathlib import Path
from typing import Any
from hcpmmp_neuromaps_common import ContractError,VERTICES

def local(tag:str)->str:return tag.rsplit('}',1)[-1]
def _array(a:ET.Element)->list[int]:
 if a.attrib.get('Intent') not in ('NIFTI_INTENT_LABEL','1002') or a.attrib.get('DataType') not in ('NIFTI_TYPE_INT32','8'): raise ContractError('GIFTI: expected INT32 label array')
 if a.attrib.get('Dimensionality')!='1': raise ContractError('GIFTI: expected one dimension')
 try:n=int(a.attrib['Dim0'])
 except (KeyError,ValueError) as e: raise ContractError('GIFTI: invalid Dim0') from e
 if n!=VERTICES: raise ContractError(f'GIFTI: expected {VERTICES} vertices, got {n}')
 if a.attrib.get('ExternalFileName','') or a.attrib.get('ExternalFileOffset','') not in ('','0'): raise ContractError('GIFTI: external payload forbidden')
 nodes=[x for x in a if local(x.tag)=='Data']
 if len(nodes)!=1: raise ContractError('GIFTI: exactly one Data node required')
 text=nodes[0].text or ''; enc=a.attrib.get('Encoding'); endian=a.attrib.get('Endian','LittleEndian'); pre='<' if endian=='LittleEndian' else '>' if endian=='BigEndian' else None
 if pre is None: raise ContractError('GIFTI: unsupported endian')
 if enc=='ASCII':
  try:v=[int(x) for x in text.split()]
  except ValueError as e: raise ContractError('GIFTI: invalid ASCII') from e
  if len(v)!=n: raise ContractError('GIFTI: ASCII length mismatch')
  return v
 try:raw=base64.b64decode(''.join(text.split()),validate=True)
 except Exception as e: raise ContractError('GIFTI: invalid base64') from e
 if enc=='GZipBase64Binary':
  try:raw=zlib.decompress(raw)
  except zlib.error:
   try:raw=gzip.decompress(raw)
   except OSError as e: raise ContractError('GIFTI: invalid compression') from e
 elif enc!='Base64Binary': raise ContractError(f'GIFTI: unsupported encoding {enc!r}')
 if len(raw)!=n*4: raise ContractError('GIFTI: binary length mismatch')
 return list(struct.unpack(f'{pre}{n}i',raw))
def parse_label_gifti(p:Path)->tuple[list[int],dict[int,str]]:
 try:root=ET.parse(p).getroot()
 except (ET.ParseError,OSError) as e: raise ContractError(f'GIFTI parse failed: {p}') from e
 if local(root.tag)!='GIFTI': raise ContractError('GIFTI: wrong root')
 tabs=[x for x in root if local(x.tag)=='LabelTable']; arr=[x for x in root if local(x.tag)=='DataArray']
 if len(tabs)!=1 or len(arr)!=1: raise ContractError('GIFTI: exactly one LabelTable/DataArray required')
 table={}
 for x in tabs[0]:
  if local(x.tag)!='Label':continue
  try:k=int(x.attrib['Key'])
  except (KeyError,ValueError) as e: raise ContractError('GIFTI: invalid label key') from e
  name=(x.text or '').strip()
  if not name or k in table: raise ContractError('GIFTI: empty/duplicate label')
  table[k]=name
 vals=_array(arr[0])
 if set(vals)-set(table): raise ContractError('GIFTI: unresolved label keys')
 return vals,table
def normalize(name:str,hemi:str,areas:set[str])->str|None:
 if name=='???':return None
 prefix='L_' if hemi=='left' else 'R_'
 if not name.startswith(prefix) or not name.endswith('_ROI'): raise ContractError(f'{hemi}: unexpected label {name!r}')
 base=name[len(prefix):-4]
 if base not in areas: raise ContractError(f'{hemi}: unknown area {name!r}')
 return prefix+base
def semantic(label_path:Path,mask_path:Path,hemi:str,areas:list[str],source_digest:str,source_id:str,terms:str,generator_id:str,generator_version:str,generator_implementation_digest:str)->dict[str,Any]:
 vals,table=parse_label_gifti(label_path); mask,_=parse_label_gifti(mask_path); aset=set(areas)
 labels=[None if mv<=0 else normalize(table[k],hemi,aset) for k,mv in zip(vals,mask)]
 prefix='L_' if hemi=='left' else 'R_'; seen={x[len(prefix):] for x in labels if x is not None}
 if any(a not in seen for a in areas): raise ContractError(f'{hemi}: one or more areas empty after transform')
 return {'schema':'symthaea-semantic-surface-labels-v1','space':'fsaverage5','hemisphere':hemi,'vertex_count':VERTICES,'labels':labels,'source':{'source_id':source_id,'source_version':'v1','source_digest':source_digest,'generator_id':generator_id,'generator_version':generator_version,'generator_implementation_digest':generator_implementation_digest,'terms_reference':terms}}
