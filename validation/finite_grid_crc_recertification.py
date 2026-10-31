#!/usr/bin/env python3
"""Finite-grid CRC recertification for arbitrary monotone loss tables.

The loss table is indexed by (risk, observation type, action index). Actions
are ordered from least to most conservative; losses must be nonincreasing.
No interpolation or convexity is assumed. Equal-share and deterministic
coordinate sweeps are checked against every augmented count-vector multiset.
"""
from __future__ import annotations
import math
from typing import Iterable
import numpy as np

from finite_state_crc_recertification import count_vectors, add_type, remove_type

class FiniteGridCRC:
    def __init__(self, actions: Iterable[float], loss_table: np.ndarray,
                 alpha: float | Iterable[float], n_calibration: int):
        self.actions=np.asarray(actions,dtype=np.float64)
        self.loss=np.asarray(loss_table,dtype=np.float64)
        if self.actions.ndim!=1 or self.actions.size<1 or np.any(~np.isfinite(self.actions)):
            raise ValueError("actions must be a nonempty finite increasing vector")
        if np.any(np.diff(self.actions)<=0):
            raise ValueError("actions must be strictly increasing")
        if self.loss.ndim!=3 or self.loss.shape[2]!=self.actions.size:
            raise ValueError("loss table must have shape (risks,types,actions)")
        if self.loss.shape[0]<1 or self.loss.shape[1]<1 or np.any(~np.isfinite(self.loss)) or np.any(self.loss<0):
            raise ValueError("loss table must be nonempty, finite and nonnegative")
        if np.any(np.diff(self.loss,axis=2)>1e-12):
            raise ValueError("losses must be nonincreasing with conservativeness")
        alphas=np.asarray([alpha] if np.ndim(alpha)==0 else alpha,dtype=np.float64)
        if alphas.shape!=(self.loss.shape[0],) or np.any(~np.isfinite(alphas)) or np.any((alphas<=0)|(alphas>=1)):
            raise ValueError("one alpha in (0,1) is required per risk")
        if n_calibration<1: raise ValueError("n_calibration must be positive")
        self.alphas=alphas
        self.n=int(n_calibration)
        self.R,self.K,self.A=self.loss.shape
        self.budget=(self.n+1)*alphas
        self.states=count_vectors(self.n,self.K)
        self.augmented=count_vectors(self.n+1,self.K)

    def _check_controller(self,c):
        if set(c)!=set(self.states):raise ValueError("controller must cover every calibration multiset")
        if any(not isinstance(v,(int,np.integer)) or v<0 or v>=self.A for v in c.values()):
            raise ValueError("controller actions must be valid action-grid indices")

    def total(self,c,S):
        if sum(S)!=self.n+1:raise ValueError("augmented multiset must have n+1 points")
        out=np.zeros(self.R)
        for z,mult in enumerate(S):
            if mult:
                D=remove_type(S,z)
                out += mult*self.loss[:,z,c[D]]
        return out

    def validity_report(self,c,tol=1e-10):
        self._check_controller(c)
        totals={S:self.total(c,S) for S in self.augmented}
        excess=np.max(np.stack(list(totals.values()))-self.budget[None,:],axis=0)
        return {"valid":bool(np.all(excess<=tol)),"max_excess":excess.tolist(),
                "totals":{",".join(map(str,S)):v.tolist() for S,v in totals.items()}}

    def _least_action(self,old_index,constraints):
        for idx in range(old_index+1):
            if all(self.loss[r,z,idx]-self.loss[r,z,old_index] <= allowance+1e-10
                   for r,z,allowance in constraints):
                return idx
        return old_index

    def envelope_crc(self):
        c={}
        worst=self.loss.max(axis=1)
        for D in self.states:
            cal=np.zeros(self.R)
            for z,count in enumerate(D):
                cal += count*self.loss[:,z,:]
            feasible=np.ones(self.A,dtype=bool)
            for r in range(self.R):
                feasible &= cal[r,:]+worst[r,:] <= self.budget[r]+1e-10
            hits=np.flatnonzero(feasible)
            if hits.size==0:raise ValueError("no envelope CRC action is feasible for this calibration state")
            c[D]=int(hits[0])
        if not self.validity_report(c)["valid"]:raise ArithmeticError("envelope CRC invalid")
        return c

    def standard_crc(self):
        c={}
        worst=self.loss[:,:,0].max(axis=1)
        for D in self.states:
            cal=np.zeros((self.R,self.A))
            for z,count in enumerate(D):
                cal += count*self.loss[:,z,:]
            feasible=np.ones(self.A,dtype=bool)
            for r in range(self.R):
                feasible &= cal[r,:]+worst[r] <= self.budget[r]+1e-10
            hits=np.flatnonzero(feasible)
            if hits.size==0:raise ValueError("no standard CRC action is feasible")
            c[D]=int(hits[0])
        if not self.validity_report(c)["valid"]:raise ArithmeticError("standard CRC invalid")
        return c

    def recertify_equal_share(self,c):
        if not self.validity_report(c)["valid"]:raise ValueError("starting controller invalid")
        out={}
        for D in self.states:
            old=c[D];constraints=[]
            for z in range(self.K):
                S=add_type(D,z)
                slack=self.budget-self.total(c,S)
                if np.any(slack < -1e-8):raise ArithmeticError("negative slack")
                for r in range(self.R):
                    constraints.append((r,z,max(0.,float(slack[r]))/(self.n+1)))
            out[D]=self._least_action(old,constraints)
        if not self.validity_report(out)["valid"]:raise ArithmeticError("equal-share update invalid")
        if any(out[D]>c[D] for D in self.states):raise ArithmeticError("more conservative update")
        return out

    def recertify_until(self,c,max_rounds=500):
        cur=dict(c);history=[dict(cur)]
        for _ in range(max_rounds):
            nxt=self.recertify_equal_share(cur);history.append(nxt)
            cur=nxt
            if nxt==history[-2]:break
        return cur,history

    def coordinate_sweep(self,c,state_order=None):
        if not self.validity_report(c)["valid"]:raise ValueError("starting controller invalid")
        order=list(self.states if state_order is None else state_order)
        if len(order)!=len(self.states) or set(order)!=set(self.states):
            raise ValueError("state_order must be a permutation")
        out=dict(c)
        for D in order:
            old=out[D];constraints=[]
            for z in range(self.K):
                S=add_type(D,z);slack=self.budget-self.total(out,S)
                if np.any(slack < -1e-8):raise ArithmeticError("negative slack")
                multiplicity=D[z]+1
                for r in range(self.R):
                    constraints.append((r,z,max(0.,float(slack[r]))/multiplicity))
            out[D]=self._least_action(old,constraints)
            if not self.validity_report(out)["valid"]:raise ArithmeticError("coordinate update invalid")
        return out

    def is_coordinatewise_admissible(self,c):
        if not self.validity_report(c)["valid"]:return False
        for D,old in c.items():
            constraints=[]
            for z in range(self.K):
                S=add_type(D,z);slack=self.budget-self.total(c,S)
                for r in range(self.R):
                    constraints.append((r,z,max(0.,float(slack[r]))/(D[z]+1)))
            if self._least_action(old,constraints)<old:return False
        return True
