#!/usr/bin/env python3
from __future__ import annotations
import json,os,stat,tempfile,unittest
from pathlib import Path
import test_derive_hcpmmp1_neuromaps_lineage_b as base
m=base.m

class GeneratorProvenanceTests(unittest.TestCase):
    def fixture(self,td:Path): return base.Tests().fixture(td)
    def env(self,lf:Path,rf:Path):
        old=os.environ.copy();os.environ['FAKE_LEFT_GII']=str(lf);os.environ['FAKE_RIGHT_GII']=str(rf);return old
    def restore(self,old): os.environ.clear();os.environ.update(old)

    def test_implementation_digest_is_closed_and_self_consistent(self):
        impl=m.generator_implementation()
        self.assertEqual(set(impl),{'digest','files'})
        self.assertEqual(set(impl['files']),{'common','gifti','derive'})
        self.assertEqual(impl['digest'],m.digest_bytes(m.canonical_json_bytes(impl['files'])))
        m.validate_generator_implementation(impl)

    def test_generator_digest_changes_scientific_commitment(self):
        with tempfile.TemporaryDirectory() as t:
            td=Path(t);mp,rp,ap,_,_=self.fixture(td);method=m.load_method_manifest(mp);run=m.load_run_manifest(rp,method,mp)
            vd=run['workbench']['version_output_sha256'];impl=m.generator_implementation()['digest']
            a=m.commitment(mp,run,ap,vd,impl);b=m.commitment(mp,run,ap,vd,'sha256:'+'0'*64)
            self.assertNotEqual(a,b)

    def test_outputs_bind_generator_implementation(self):
        with tempfile.TemporaryDirectory() as t:
            td=Path(t);mp,rp,ap,lf,rf=self.fixture(td);old=self.env(lf,rf)
            try: out=td/'o';e=m.derive(mp,rp,ap,out)
            finally:self.restore(old)
            impl=e['generator_implementation'];self.assertEqual(impl,m.validate_generator_implementation(impl))
            left=json.loads((out/'left.semantic.json').read_text())
            self.assertEqual(left['source']['generator_implementation_digest'],impl['digest'])
            self.assertEqual(left['source']['source_digest'],e['scientific_input_commitment'])
            m.validate_evidence(out)

    def test_generator_mutation_during_derivation_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            td=Path(t);mp,rp,ap,lf,rf=self.fixture(td);old=self.env(lf,rf);real=m.generator_implementation;first=real();calls={'n':0}
            def changing():
                calls['n']+=1
                if calls['n']==1:return first
                files=dict(first['files']);files['derive']='sha256:'+'f'*64
                return {'files':files,'digest':m.digest_bytes(m.canonical_json_bytes(files))}
            m.generator_implementation=changing
            try:
                with self.assertRaises(m.DerivationError):m.derive(mp,rp,ap,td/'o')
            finally:m.generator_implementation=real;self.restore(old)

    def test_input_mutation_during_transform_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            td=Path(t);mp,rp,ap,lf,rf=self.fixture(td);run=json.loads(rp.read_text());victim=Path(run['inputs']['fsaverage10k_left_sphere']['path']);wb=Path(run['workbench']['path'])
            wb.write_text(r'''#!/bin/sh
if [ "$1" = "-version" ]; then printf 'Connectome Workbench fake-v1\n'; exit 0; fi
if [ "$1" = "-label-resample" ]; then printf x >> "$MUTATE_INPUT"; out="$6"; else for out do :; done; fi
case "$out" in *left*) cp "$FAKE_LEFT_GII" "$out" ;; *) cp "$FAKE_RIGHT_GII" "$out" ;; esac
''');wb.chmod(wb.stat().st_mode|stat.S_IXUSR);run['workbench']['sha256']=base.sha(wb);rp.write_text(json.dumps(run,sort_keys=True))
            old=self.env(lf,rf);os.environ['MUTATE_INPUT']=str(victim)
            try:
                with self.assertRaises(m.DerivationError):m.derive(mp,rp,ap,td/'o')
            finally:self.restore(old)

if __name__=='__main__':unittest.main(verbosity=2)
