#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,struct
from fractions import Fraction
from pathlib import Path
from typing import Any
JSHA='4092b971b29a64a7eb622e5fdee1f80d0bafbe7b9e6dc6513995f692ac74371b'
P='ee79f86a6d431c462ae8701094a3e29be88c4a5e'; PT='645b1644112a822c79b97bcd9eb226cc604b3af8'
KS='cffe5574d000a44a87a757a788d08cb7ba474810e9c6a3d860d57ac57352eb3c'; KI='65a15473bf128bd604e72c4f32b20f49fcba1e82797714b29b96479af5f38395'
E={"chron_fb":"31e904c6adf8e21937dd978ef710e8e49a0ab1421c96c29ad5c046a72f16a9bd","chron_ol":"994798f9d79ffc2b1724acbf5b90c77f23b38290923e177801cd4af88b26379f","classes":"9f8b34bc346d5691c363de6ff4a94110f492e3fecc48ed1f4d6233de9d98cfe9","core":"07a5bbc839270ddaf2d21c60ec73519288ecf648f0a778c4b200d914cf08c3a6","embed":"2e59b1770da35a1df87000916cf1bedc73de9756731dcd154ce8e8702bff57f1","ip_fb":"17984644548708695607c3dcb0a5afdaa7dc41a7dd7ec2029b01a9210b2b6d34","ip_ol":"3548205626704f0ce21932783dc7721d1fa8e995925da0fbac7382728c379ad0","match":"53a69be414c7688b8b0da0d86d9a7ba617f18ec06eabf4c3dbead76a07503a91","obs":"d8793739213b6de12202ab46504f12677bb799304f9a637f23f64c2272cb174a","policy":"f84ef73cdebf2ed9ef887b3e7be05bcc3200d36dd43dc9027221522283e84fb3","prereg":"fa62084a562a061445c74088914616480efc6008d2a16db56b23e9340612337e","sep":"7bcb9f91a5f46bab43354cbad19228d0bf51b0dcdee873965522899b11992bc2","strat_fb":"28a53f1bea4734cd54c92a5f5de2617b58a6f961842d24ed6f4ebd87ad47fbe7","strat_ol":"4c47a248a5fdf7ff2b490ce88c792cccd74600d78661cbba039915b8b2b761f3","th_fb":"53f3be5178ed07cbe2c55293ba3bc7a27c92e0823b4f6374bfe7e1acee485b4c","th_ol":"65800e25b0b52e6d49e2bbe5334eebaf781deea50fc9feb8b81ce7dc787b44b0"}
def cj(v): return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sid(d,v):
 p=cj(v); h=hashlib.sha3_256(); h.update(d.encode()+b'\0'); h.update(len(p).to_bytes(8,'big')); h.update(p); return h.hexdigest()
def q(x):
 if '/' in x: a,b=x.split('/'); return Fraction(int(a),int(b))
 return Fraction(int(x),1)
def b(x):
 if len(x)!=16: raise AssertionError('bad bits')
 f=struct.unpack('>d',int(x,16).to_bytes(8,'big'))[0]
 if not (f==f and abs(f)!=float('inf')): raise AssertionError('nonfinite')
 return Fraction.from_float(f)
def walk(v:Any):
 if isinstance(v,dict):
  if set(v)=={'n','q','b'}:
   if q(v['q'])!=b(v['b']): raise AssertionError('scalar '+v['n'])
   return
  for x in v.values(): walk(x)
 elif isinstance(v,list):
  for x in v: walk(x)
def recompute(d):
 I={}; I['core']=sid('m006d.core.v3',d['core'])
 for k in ['ol','fb']:
  I['ip_'+k]=sid('m006d.ip.v3',d['ip'][k]); I['strat_'+k]=sid('m006d.strategy.v2',d['strategy'][k]); I['chron_'+k]=sid('m006d.chron.v3',d['chron'][k])
 I['obs']=sid('m006d.obs.v3',d['obs']); I['classes']=sid('m006d.classes.v2',d['classes']); I['embed']=sid('m006d.embed.v2',d['classes']['embed']); I['policy']=sid('m006d.affine_pi.v3',d['policy'])
 I['th_ol']=sid('m006d.theorem.ol.v3',{'core':I['core'],'ip':I['ip_ol'],'strat':I['strat_ol'],'chron':I['chron_ol'],'classes':I['classes'],'proof':d['proof']['ol']})
 I['th_fb']=sid('m006d.theorem.fb.v3',{'core':I['core'],'ip':I['ip_fb'],'strat':I['strat_fb'],'chron':I['chron_fb'],'obs':I['obs'],'classes':I['classes'],'policy':I['policy'],'proof':d['proof']['fb']})
 m={'core':I['core'],'ip':[I['ip_ol'],I['ip_fb']],'strat':[I['strat_ol'],I['strat_fb']],'chron':[I['chron_ol'],I['chron_fb']],'obs':I['obs'],'classes':I['classes'],'embed':I['embed'],'policy':I['policy'],'delta':d['match_contract']['allowed_delta']}; I['match']=sid('m006d.match.v3',m)
 I['sep']=sid('m006d.sep.v3',d['separation']); x=dict(d); x.pop('ids'); I['prereg']=sid('m006d.prereg.v3',x); return I
def validate(d):
 assert d['schema']=='symthaea.manifold-006d.preregistration.v3' and d['authority']=='preregistration-only'
 assert d['theorem_executed'] is False and d['qualification_granted'] is False and d['semantic_implementation_present'] is False
 assert d['core']['parent']==[P,PT] and d['core']['arith']==[KS,KI]; walk(d['core']); walk(d['policy'])
 assert [s['i'] for s in d['core']['stages']]==[0,1]
 for s in d['core']['stages']:
  assert [x['q'] for x in s['a']]==['1','1','1','0']
 assert d['strategy']['ol']['bind']==['exists_fixed_u0_u1','forall_d0','forall_d1'] and d['strategy']['ol']['u1_reselect'] is False
 assert d['strategy']['fb']['bind']==['exists_fixed_u0_pi','forall_d0','forall_d1'] and d['strategy']['fb']['fixed_before_d0'] is True and d['strategy']['fb']['fresh_history_strategy_exists'] is False
 assert d['chron']['fb']['e']==['select_u0_pi','d0','x1','publish_Obs1','eval_fixed_pi','d1','x2']
 assert d['obs']['allow']==['x1_bits','stage','chron_id','ip_id'] and d['obs']['unknown']=='reject' and d['obs']['raw_d0_channel'] is False
 assert {'d0','d0_bits','d1','d1_bits','x2','x2_bits','future','hidden'}.issubset(set(d['obs']['deny'])) and d['obs']['derived_ok']=='deterministic_from_x1_and_public_core'
 assert d['classes']['fb']['admit']==['u0∈U','pi_fixed_before_d0','pi_defined_on_all_x1_reachable_under_u0','pi(x1)∈U_on_that_domain']
 assert d['classes']['embed']['authority']=='exact_structural_embedding' and d['classes']['embed']['sampling'] is False and d['classes']['embed']['traj_equal_for_all_d0_d1'] is True
 assert d['policy']['select']=='before_d0' and d['policy']['raw_d0_channel'] is False and d['policy']['pure'] is True and d['policy']['domain_meaning']=='reachable_x1_for_u0=0'
 assert d['proof']['ol']['auth']=='complete_negative' and {'search_exhaustion','sampling','optimizer_failure'}.issubset(set(d['proof']['ol']['ban']))
 assert d['proof']['fb']['auth']=='constructive' and d['proof']['incl']['auth']=='exact_structural_embedding' and d['proof']['incl']['sampling'] is False
 assert d['separation']['promote']=='require inclusion AND strict witness under same core' and d['separation']['scope']=='reference_problem_only'
 assert d['authority_lattice']['StrictReachabilityEnlargement']=='inclusion+strict_witness' and d['authority_lattice']['CertifiedNotRobust']=='complete_negative'
 assert d['supersession']['reinterpret_prior'] is False and d['supersession']['status']=='superseded_before_semantic_implementation' and d['supersession']['v2'][3]=='13ae9e827de9d6506500d3b4b30b60cd992b7ac55b7f8137f318ddc5bcbbea3e'
 need={'strategy_before_disturbance','no_fresh_history_existential','constant_embedding','embedding_trajectory_equivalence','derived_state_inference_allowed','strict_composite_promotion'}; assert need.issubset(set(d['vectors']['families']))
 needg={'bind_prereg_id','bind_json_sha256','bind_validator_sha256','bind_prereg_artifact_sha256','bind_workflow_sha256','bind_classes_id','bind_embed_id','bind_supersession'}; assert set(d['implementation_gate'])==needg
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('src',type=Path); ap.add_argument('out',type=Path); a=ap.parse_args(); raw=a.src.read_bytes(); assert hashlib.sha256(raw).hexdigest()==JSHA; d=json.loads(raw); validate(d); A=recompute(d); assert A==E and d['ids']==E
 r={'schema':'symthaea.manifold-006d.prereg-validation.v3','authority':'preregistration-only-independent-python','prereg_json_sha256':JSHA,'prereg_id':A['prereg'],'core_id':A['core'],'th_ol':A['th_ol'],'th_fb':A['th_fb'],'classes_id':A['classes'],'embed_id':A['embed'],'match_id':A['match'],'sep_id':A['sep'],'strategy_fixed_before_disturbance':True,'fresh_history_strategy_existential_forbidden':True,'derived_state_inference_boundary_frozen':True,'strict_enlargement_requires_inclusion_and_strictness':True,'supersession_without_reinterpretation':True,'theorem_executed':False,'qualification_granted':False,'semantic_implementation_present':False,'production_runtime_authority':False,'verdict':'PASS_PREREGISTRATION'}; a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(r,sort_keys=True,indent=2)+'\n'); return 0
if __name__=='__main__': raise SystemExit(main())
