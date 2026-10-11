#!/usr/bin/env python3
from __future__ import annotations
import copy, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional
EPS=1e-9

def nn(v,n='value'):
    if not math.isfinite(v) or v < 0: raise ValueError(f'{n} must be finite and nonnegative')

@dataclass(frozen=True)
class MachineSpec:
    machine_id:str
    maintenance:Dict[str,float]=field(default_factory=dict)
    def validate(self):
        if not self.machine_id: raise ValueError('machine_id required')
        for k,v in self.maintenance.items():
            if not k: raise ValueError('maintenance material id required')
            nn(v,'maintenance input')

@dataclass(frozen=True)
class ProcessSpec:
    process_id:str; machine_id:str; inputs:Dict[str,float]; outputs:Dict[str,float]
    batches_per_machine:float; power_per_batch:float
    def validate(self):
        if not self.process_id or not self.machine_id: raise ValueError('process_id and machine_id required')
        nn(self.batches_per_machine,'batch capacity'); nn(self.power_per_batch,'process power')
        for d in (self.inputs,self.outputs):
            for k,v in d.items():
                if not k: raise ValueError('material id required')
                nn(v,'process material quantity')

@dataclass(frozen=True)
class BuildSpec:
    build_id:str; builder_machine_id:str; inputs:Dict[str,float]; power_per_unit:float
    output_machine_id:Optional[str]=None; output_power_capacity:float=0.0
    def validate(self):
        if not self.build_id or not self.builder_machine_id: raise ValueError('build_id and builder_machine_id required')
        if self.output_machine_id is None and self.output_power_capacity <= 0: raise ValueError('build must produce machine or power')
        nn(self.power_per_unit,'build power'); nn(self.output_power_capacity,'power capacity')
        for k,v in self.inputs.items():
            if not k: raise ValueError('build material id required')
            nn(v,'build material quantity')

@dataclass(frozen=True)
class ProductionAction: process_id:str; batches:float
@dataclass(frozen=True)
class BuildAction: build_id:str; units:int
@dataclass(frozen=True)
class StepPlan:
    imports:Dict[str,float]=field(default_factory=dict)
    maintenance_targets:Dict[str,int]=field(default_factory=dict)
    production:List[ProductionAction]=field(default_factory=list)
    builds:List[BuildAction]=field(default_factory=list)
@dataclass
class State:
    step:int; inventory:Dict[str,float]; machines:Dict[str,int]; power_capacity:float
    cumulative_outputs:Dict[str,float]=field(default_factory=dict)
    cumulative_imports:Dict[str,float]=field(default_factory=dict)
    locally_built_machines:Dict[str,int]=field(default_factory=dict)

class Model:
    def __init__(self,machines,processes,builds):
        self.machines={x.machine_id:x for x in machines}; self.processes={x.process_id:x for x in processes}; self.builds={x.build_id:x for x in builds}
        if len(self.machines)!=len(machines) or len(self.processes)!=len(processes) or len(self.builds)!=len(builds): raise ValueError('duplicate id')
        for m in machines:m.validate()
        for p in processes:
            p.validate()
            if p.machine_id not in self.machines: raise ValueError('unknown process machine')
        for b in builds:
            b.validate()
            if b.builder_machine_id not in self.machines: raise ValueError('unknown builder machine')
            if b.output_machine_id and b.output_machine_id not in self.machines: raise ValueError('unknown output machine')
    def consume(self,inv,needs,mult=1.0):
        for k,v in needs.items():
            need=v*mult
            if inv.get(k,0)+EPS<need: raise ValueError(f'insufficient inventory: {k}')
        for k,v in needs.items():
            inv[k]=inv.get(k,0)-v*mult
            if abs(inv[k])<EPS: inv[k]=0.0
    def maintenance(self,inv,counts,targets):
        op={}; used={}
        unknown=set(targets)-set(self.machines)
        if unknown: raise ValueError('unknown maintenance target')
        for mid in sorted(self.machines):
            cnt=int(counts.get(mid,0)); spec=self.machines[mid]
            if cnt<0: raise ValueError('negative machine count')
            target=targets.get(mid,cnt)
            if not isinstance(target,int) or target<0 or target>cnt: raise ValueError('invalid maintenance target')
            for mat,per in spec.maintenance.items():
                need=per*target
                if inv.get(mat,0)+EPS<need: raise ValueError(f'insufficient maintenance inventory: {mat}')
            op[mid]=target
            for mat,per in spec.maintenance.items():
                amt=per*target
                inv[mat]=inv.get(mat,0)-amt; used[mat]=used.get(mat,0)+amt
                if abs(inv[mat])<EPS: inv[mat]=0.0
        return op,used
    def step(self,state,plan):
        working=copy.deepcopy(state.inventory)
        for k,v in plan.imports.items():
            nn(v,'import'); working[k]=working.get(k,0)+v; state.cumulative_imports[k]=state.cumulative_imports.get(k,0)+v
        op,maint=self.maintenance(working,state.machines,plan.maintenance_targets)
        power=state.power_capacity; pending={}; pending_m={}; pending_power=0.0; used_proc={}; used_builders={}
        for a in plan.production:
            if a.process_id not in self.processes: raise ValueError('unknown process')
            nn(a.batches,'batches'); p=self.processes[a.process_id]
            cap=op.get(p.machine_id,0)*p.batches_per_machine
            if used_proc.get(a.process_id,0)+a.batches>cap+EPS: raise ValueError('process capacity exceeded')
            needp=a.batches*p.power_per_batch
            if needp>power+EPS: raise ValueError('insufficient power')
            self.consume(working,p.inputs,a.batches); power-=needp; used_proc[a.process_id]=used_proc.get(a.process_id,0)+a.batches
            for k,v in p.outputs.items(): pending[k]=pending.get(k,0)+v*a.batches
        for a in plan.builds:
            if a.build_id not in self.builds: raise ValueError('unknown build')
            if not isinstance(a.units,int) or a.units<0: raise ValueError('build units')
            b=self.builds[a.build_id]; slots=op.get(b.builder_machine_id,0)
            if used_builders.get(b.builder_machine_id,0)+a.units>slots: raise ValueError('builder capacity exceeded')
            needp=a.units*b.power_per_unit
            if needp>power+EPS: raise ValueError('insufficient power')
            self.consume(working,b.inputs,a.units); power-=needp; used_builders[b.builder_machine_id]=used_builders.get(b.builder_machine_id,0)+a.units
            if b.output_machine_id: pending_m[b.output_machine_id]=pending_m.get(b.output_machine_id,0)+a.units
            pending_power+=b.output_power_capacity*a.units
        for k,v in pending.items(): working[k]=working.get(k,0)+v; state.cumulative_outputs[k]=state.cumulative_outputs.get(k,0)+v
        for k,v in pending_m.items(): state.machines[k]=state.machines.get(k,0)+v; state.locally_built_machines[k]=state.locally_built_machines.get(k,0)+v
        opening=state.power_capacity; state.power_capacity+=pending_power; state.inventory=working; state.step+=1
        return {'operational':op,'maintenance':maint,'power_used':opening-power,'closing_power':state.power_capacity,'pending':pending,'pending_machines':pending_m}

def synthetic_model():
    ms=[MachineSpec('refinery',{'bearing':1}),MachineSpec('machine_shop',{'bearing':1}),MachineSpec('wire_mill',{'bearing':1})]
    ps=[ProcessSpec('refine_ore','refinery',{'ore':10},{'metal':6,'slag':4},2,5),ProcessSpec('draw_wire','wire_mill',{'metal':2},{'wire':1.9,'scrap':0.1},2,2)]
    bs=[BuildSpec('build_refinery','machine_shop',{'metal':8,'wire':2,'controller':1},4,'refinery'),BuildSpec('build_wire_mill','machine_shop',{'metal':5,'wire':1,'controller':1},3,'wire_mill'),BuildSpec('build_power','machine_shop',{'metal':4,'wire':1,'power_module':1},2,None,8)]
    return Model(ms,ps,bs)

def base_state():
    return State(0,{'ore':100,'bearing':20,'controller':2,'power_module':1,'metal':10,'wire':3},{'refinery':1,'machine_shop':1,'wire_mill':1},20)

def raises(fn,text):
    try: fn()
    except ValueError as e:
        assert text in str(e),(text,str(e)); return
    raise AssertionError('expected failure '+text)

def self_test():
    m=synthetic_model()
    s=base_state(); s.inventory['metal']=0; s.inventory['wire']=0
    raises(lambda:m.step(s,StepPlan(production=[ProductionAction('refine_ore',2)],builds=[BuildAction('build_refinery',1)])),'insufficient inventory')
    s=base_state(); r0=m.step(s,StepPlan(builds=[BuildAction('build_refinery',1)])); assert r0['operational']['refinery']==1 and s.machines['refinery']==2
    r1=m.step(s,StepPlan(production=[ProductionAction('refine_ore',4)])); assert r1['operational']['refinery']==2
    s=base_state(); s.inventory['bearing']=1; raises(lambda:m.step(s,StepPlan(production=[ProductionAction('refine_ore',2)])),'insufficient maintenance inventory')
    s=base_state(); s.inventory['bearing']=1; r=m.step(s,StepPlan(maintenance_targets={'refinery':1,'machine_shop':0,'wire_mill':0},production=[ProductionAction('refine_ore',2)])); assert r['operational']['refinery']==1 and r['operational']['machine_shop']==0
    s=base_state(); s.inventory['controller']=0; m.step(s,StepPlan(production=[ProductionAction('refine_ore',1)])); raises(lambda:m.step(s,StepPlan(builds=[BuildAction('build_refinery',1)])),'controller')
    s=base_state(); r=m.step(s,StepPlan(builds=[BuildAction('build_power',1)])); assert abs(s.power_capacity-28)<EPS
    s=base_state(); s.machines['machine_shop']=2; s.inventory['controller']=1; s.inventory['metal']=20; s.inventory['wire']=5
    raises(lambda:m.step(s,StepPlan(builds=[BuildAction('build_refinery',1),BuildAction('build_wire_mill',1)])),'controller')
    s=base_state(); s.inventory['controller']=0; raises(lambda:m.step(s,StepPlan(builds=[BuildAction('build_refinery',1)])),'controller')
    s=base_state(); s.inventory['controller']=0; m.step(s,StepPlan(imports={'controller':1},builds=[BuildAction('build_refinery',1)])); assert s.cumulative_imports['controller']==1
    s=base_state(); raises(lambda:m.step(s,StepPlan(production=[ProductionAction('refine_ore',3)])),'process capacity exceeded')
    s=base_state(); s.power_capacity=4; raises(lambda:m.step(s,StepPlan(production=[ProductionAction('refine_ore',1)])),'insufficient power')
    s=base_state(); m.step(s,StepPlan(builds=[BuildAction('build_refinery',1)])); m.step(s,StepPlan(production=[ProductionAction('refine_ore',4)])); assert s.locally_built_machines['refinery']==1 and s.cumulative_outputs['metal']>=24
    raises(lambda:Model([MachineSpec('m')],[ProcessSpec('p','missing',{}, {},1,0)],[]),'unknown process machine')

if __name__=='__main__':
    self_test(); print('ok')
