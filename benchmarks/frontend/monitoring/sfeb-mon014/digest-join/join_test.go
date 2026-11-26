package crossprofile

import "testing"

func sha(ch byte) string { b:=make([]byte,64); for i:=range b { b[i]=ch }; return string(b) }
func reqCtx() DigestContext { return DigestContext{"urn:request","1","application/json","RFC8785-JCS","UTF-8","request","dest-a/op-1"} }
func digest(value string) DigestRef { return DigestRef{"sha-256",value,reqCtx(),Verified} }
func stateDigest(value string) DigestRef { return DigestRef{"sha-256",value,DigestContext{"urn:state","1","application/json","postgres-jsonb-text-v1","UTF-8","state-after","dest-a/resource-a/revision-7"},Verified} }

func TestDigestContextRequiredForEquality(t *testing.T) {
 a,b:=digest(sha('a')),digest(sha('a'))
 if CompareDigestRefs(a,b)!=DigestEqual { t.Fatal("identical typed digest should compare equal") }
 b.Context.Profile="urn:other"
 if CompareDigestRefs(a,b)!=DigestIncomparable { t.Fatal("same hex across profiles must be incomparable") }
 b=digest(sha('a')); b.Context.Canonicalization="raw-json"
 if CompareDigestRefs(a,b)!=DigestIncomparable { t.Fatal("canonicalization mismatch") }
 b=digest(sha('a')); b.Context.Purpose="response"
 if CompareDigestRefs(a,b)!=DigestIncomparable { t.Fatal("purpose mismatch") }
 b=digest(sha('b')); if CompareDigestRefs(a,b)!=DigestDifferent { t.Fatal("same context, different value") }
 b=digest(sha('a')); b.Verification=Missing; if CompareDigestRefs(a,b)!=DigestIndeterminate { t.Fatal("missing evidence must be indeterminate") }
 b=digest("bad"); if CompareDigestRefs(a,b)!=DigestInvalid { t.Fatal("malformed digest must be invalid") }
}
func TestCapsule06EffectAttestationSemantics(t *testing.T) {
 gate:=AssessCapsuleEffect(Confirmed,string(GateExecuted),true)
 runtime:=AssessCapsuleEffect(Confirmed,string(RuntimeClaimed),true)
 host:=AssessCapsuleEffect(Confirmed,string(HostServedObserved),true)
 if !gate.Valid || gate.Grade!=GateObserved { t.Fatalf("gate grade: %+v",gate) }
 if !runtime.Valid || runtime.Grade!=RuntimeAssertion || host.Grade!=RuntimeAssertion { t.Fatalf("runtime/host grade mismatch: %+v %+v",runtime,host) }
 unknown:=AssessCapsuleEffect(Confirmed,"future-value",true)
 if !unknown.Valid || !unknown.UnknownAttestation || unknown.Grade!=RuntimeAssertion { t.Fatalf("unknown must not grade up: %+v",unknown) }
 if AssessCapsuleEffect(Confirmed,string(GateExecuted),false).Valid { t.Fatal("confirmed requires response digest") }
 if AssessCapsuleEffect(DispatchedUnconfirmed,"",false).Valid { t.Fatal("dispatch requires attestation") }
 if AssessCapsuleEffect(NotApplicable,string(GateExecuted),true).Valid { t.Fatal("not-applicable cannot carry attestation") }
}
func goodInput() DigitalEffectInput {
 return DigitalEffectInput{
  Producer:ProducerEvidence{Verified,"dest-a","op-1","writer-a","tenant-a","resource-a",digest(sha('a')),AssessCapsuleEffect(Confirmed,string(RuntimeClaimed),true),true},
  Sink:SinkEvidence{Verified,SinkCommitted,"dest-a","op-1","writer-a","tenant-a","resource-a",digest(sha('a')),7,stateDigest(sha('b'))},
  Observer:ObserverEvidence{Verified,"dest-a","op-1","observer-a","tenant-a","resource-a",7,stateDigest(sha('b')),IndependenceVerified},
  AuthorityAtCommit:AuthorityValid,
 }
}
func TestQualifiedDigitalEffectNeedsEveryBoundLayer(t *testing.T) {
 r:=QualifyDigitalEffect(goodInput())
 if r.Result!=QualifiedDigitalEffect || r.RequestBinding!=DigestEqual || r.StateBinding!=DigestEqual { t.Fatalf("got %+v",r) }
}
func TestProducerConfirmationAloneCannotEstablishEffect(t *testing.T) {
 in:=goodInput(); in.Sink.Verification=Missing
 if QualifyDigitalEffect(in).Result==QualifiedDigitalEffect { t.Fatal("producer confirmation substituted for sink") }
}
func TestMissingObserverIsNotSuccess(t *testing.T) {
 in:=goodInput(); in.Observer.Verification=Missing
 if QualifyDigitalEffect(in).Result!=EffectObservedReadbackMissing { t.Fatal("missing readback must remain explicit") }
}
func TestRequestMismatchIsContradicted(t *testing.T) {
 in:=goodInput(); in.Sink.RequestDigest=digest(sha('c'))
 if QualifyDigitalEffect(in).Result!=ContradictedEffect { t.Fatal("request mismatch accepted") }
}
func TestEqualHexWithDifferentRequestContextIsNotEqual(t *testing.T) {
 in:=goodInput(); in.Sink.RequestDigest.Context.Profile="urn:other"
 if QualifyDigitalEffect(in).Result!=IndeterminateEffect { t.Fatal("incomparable request digest accepted") }
}
func TestWrongObserverRevisionOrIdentityIsContradicted(t *testing.T) {
 in:=goodInput(); in.Observer.Revision++
 if QualifyDigitalEffect(in).Result!=ContradictedEffect { t.Fatal("wrong revision accepted") }
 in=goodInput(); in.Observer.PrincipalID=in.Sink.PrincipalID
 if QualifyDigitalEffect(in).Result!=ContradictedEffect { t.Fatal("same principal cannot establish separate observer") }
}
func TestObserverIndependenceAndCommitAuthorityStaySeparate(t *testing.T) {
 in:=goodInput(); in.Observer.Independence=SharedControlRoot
 if QualifyDigitalEffect(in).Result!=EffectObservedNotIndependent { t.Fatal("shared root passed") }
 in=goodInput(); in.AuthorityAtCommit=AuthorityUncheckable
 if QualifyDigitalEffect(in).Result!=EffectObservedAuthorityUnknown { t.Fatal("unknown authority promoted") }
 in=goodInput(); in.AuthorityAtCommit=AuthorityInvalid
 if QualifyDigitalEffect(in).Result!=EffectObservedAuthorityInvalid { t.Fatal("invalid authority promoted") }
}
func TestProducerSuccessAgainstSinkRefusalIsContradicted(t *testing.T) {
 in:=goodInput(); in.Sink.Outcome=SinkRefused
 if QualifyDigitalEffect(in).Result!=ContradictedEffect { t.Fatal("producer success/sink refusal conflict lost") }
}
func TestBilateralClaimDoesNotReplaceControlIndependence(t *testing.T) {
 in:=goodInput(); in.Observer.Independence=IndependenceUnknown
 if QualifyDigitalEffect(in).Result==QualifiedDigitalEffect { t.Fatal("bilateral evidence implies independence") }
}
