package capture

import "testing"

func sha(ch byte) string { b:=make([]byte,64); for i:=range b { b[i]=ch }; return string(b) }
func dg(profile,purpose,subject string,v byte) DigestEvidence {
	return DigestEvidence{Present:true,Valid:true,Algorithm:"sha-256",Value:sha(v),Verification:Verified,
		Context:DigestContext{Profile:profile,Version:"1",PayloadType:"application/json",Canonicalization:"RFC8785-JCS",Encoding:"UTF-8",Purpose:purpose,Subject:subject}}
}
func base(start,end,watermark uint64,items ...CapturedItem) Manifest {
	q:=dg("urn:query","source-query","boundary-a",'a')
	b:=Boundary{BoundaryID:"boundary-a",SourceID:"source-a",StreamID:"stream-a",PartitionID:"partition-0",SnapshotToken:"snapshot-17",QueryDigest:q,StartExclusive:start,EndInclusive:end}
	s:=SourceEvidence{Verification:Verified,QueryBindingVerification:Verified,WatermarkVerification:Verified,
		BoundaryID:b.BoundaryID,SourceID:b.SourceID,StreamID:b.StreamID,PartitionID:b.PartitionID,SnapshotToken:b.SnapshotToken,
		BoundQueryDigest:q,Terminal:true,Watermark:watermark,AttestedWatermark:watermark,ControlRoot:"source-root-A",AttestedControlRoot:"source-root-A"}
	return Manifest{Verification:Verified,Boundary:b,Source:s,Items:items,Digest:dg("urn:manifest","manifest","boundary-a",'b')}
}
func it(pos uint64,id string,v byte) CapturedItem { return CapturedItem{Position:pos,EventID:id,Digest:dg("urn:event","event",id,v)} }
func witness(m Manifest)*Witness { return &Witness{Verification:Verified,BoundaryID:m.Boundary.BoundaryID,ManifestDigest:m.Digest,ControlRoot:"witness-root-B"} }

func TestNonEmptyCaptureIsOnlyBounded(t *testing.T) {
	m:=base(10,12,12,it(11,"e11",'c'),it(12,"e12",'d'))
	r:=VerifyBoundedCapture(m,nil)
	if r.Coverage!=CoverageComplete || r.UniqueItems!=2 || r.NegativeEvidence!=NoEventsNotClaimed { t.Fatalf("%+v",r) }
}
func TestEmptyWindowIsOnlyZeroObservedWithinBoundary(t *testing.T) {
	m:=base(20,20,20); r:=VerifyBoundedCapture(m,nil)
	if r.Coverage!=CoverageComplete || r.NegativeEvidence!=NoEventsObserved || r.NegativeEvidence==NoEventsWitnessed { t.Fatalf("%+v",r) }
}
func TestIndependentWitnessOnlyAnchorsExactEmptyManifest(t *testing.T) {
	m:=base(20,20,20); r:=VerifyBoundedCapture(m,witness(m))
	if r.Coverage!=CoverageComplete || r.Witness!=WitnessIndependent || r.NegativeEvidence!=NoEventsWitnessed { t.Fatalf("%+v",r) }
}
func TestMismatchedWitnessCannotUpgradeNegative(t *testing.T) {
	m:=base(4,4,4); w:=witness(m); w.BoundaryID="other"
	r:=VerifyBoundedCapture(m,w)
	if r.Witness!=WitnessMismatch || r.NegativeEvidence!=NoEventsObserved { t.Fatalf("%+v",r) }
}
func TestSharedRootCannotWitness(t *testing.T) {
	m:=base(3,3,3); w:=witness(m); w.ControlRoot=m.Source.ControlRoot
	r:=VerifyBoundedCapture(m,w)
	if r.Witness!=WitnessSharedRoot || r.NegativeEvidence!=NoEventsObserved { t.Fatalf("%+v",r) }
}
func TestMissingWitnessRemainsObservedOnly(t *testing.T) {
	m:=base(3,3,3); r:=VerifyBoundedCapture(m,nil)
	if r.Witness!=WitnessNotProvided || r.NegativeEvidence!=NoEventsObserved { t.Fatalf("%+v",r) }
}
func TestNonterminalAndLowWatermarkAreIncomplete(t *testing.T) {
	m:=base(5,7,7,it(6,"e6",'c'),it(7,"e7",'d')); m.Source.Terminal=false
	if got:=VerifyBoundedCapture(m,nil).Coverage; got!=CoverageIncomplete { t.Fatalf("%s",got) }
	m=base(5,7,6,it(6,"e6",'c'),it(7,"e7",'d'))
	if got:=VerifyBoundedCapture(m,nil).Coverage; got!=CoverageIncomplete { t.Fatalf("%s",got) }
}
func TestTerminalSequenceGapContradictsCompleteness(t *testing.T) {
	m:=base(10,13,13,it(11,"e11",'c'),it(13,"e13",'d')); r:=VerifyBoundedCapture(m,nil)
	if r.Coverage!=CoverageContradicted || r.MissingPositions!=1 || r.FirstMissing!=12 || r.LastMissing!=12 || r.NegativeEvidence!=NoEventsNotClaimed { t.Fatalf("%+v",r) }
}
func TestIdenticalSamePositionRedeliveryIsCounted(t *testing.T) {
	x:=it(4,"same-event",'c'); m:=base(3,4,4,x,x); r:=VerifyBoundedCapture(m,nil)
	if r.Coverage!=CoverageComplete || r.UniqueItems!=1 || r.DuplicateDelivery!=1 { t.Fatalf("%+v",r) }
}
func TestConflictingSamePositionAndReusedEventIDAreContradicted(t *testing.T) {
	m:=base(3,3,3,it(3,"a",'c'),it(3,"b",'d'))
	if got:=VerifyBoundedCapture(m,nil).Coverage; got!=CoverageContradicted { t.Fatalf("%s",got) }
	m=base(1,2,2,it(1,"a",'c'),it(2,"a",'c'))
	if got:=VerifyBoundedCapture(m,nil).Coverage; got!=CoverageContradicted { t.Fatalf("%s",got) }
}
func TestOutOfRangeItemIsContradicted(t *testing.T) {
	m:=base(10,11,11,it(10,"before",'c'))
	if got:=VerifyBoundedCapture(m,nil).Coverage; got!=CoverageContradicted { t.Fatalf("%s",got) }
}
func TestSourceAttestationMustBindQuerySnapshotAndCursor(t *testing.T) {
	m:=base(2,2,2); m.Source.SnapshotToken="other"
	if got:=VerifyBoundedCapture(m,nil).Coverage; got!=CoverageContradicted { t.Fatalf("snapshot: %s",got) }
	m=base(2,2,2); m.Source.BoundQueryDigest.Value=sha('f')
	if got:=VerifyBoundedCapture(m,nil).Coverage; got!=CoverageContradicted { t.Fatalf("query: %s",got) }
}
func TestWatermarkAndControlRootMustMatchVerifiedAttestation(t *testing.T) {
	m:=base(1,2,2,it(2,"e2",'c')); m.Source.Watermark=200
	if got:=VerifyBoundedCapture(m,nil).Coverage; got!=CoverageContradicted { t.Fatalf("watermark: %s",got) }
	m=base(1,2,2,it(2,"e2",'c')); m.Source.ControlRoot="other-root"
	if got:=VerifyBoundedCapture(m,nil).Coverage; got!=CoverageContradicted { t.Fatalf("control root: %s",got) }
}
func TestMissingProofsAreIndeterminate(t *testing.T) {
	m:=base(0,0,0); m.Source.QueryBindingVerification=Missing
	if got:=VerifyBoundedCapture(m,nil).Coverage; got!=CoverageIndeterminate { t.Fatalf("query proof: %s",got) }
	m=base(0,0,0); m.Source.WatermarkVerification=Missing
	if got:=VerifyBoundedCapture(m,nil).Coverage; got!=CoverageIndeterminate { t.Fatalf("watermark proof: %s",got) }
	m=base(0,0,0); m.Digest.Verification=Missing
	if got:=VerifyBoundedCapture(m,nil).Coverage; got!=CoverageIndeterminate { t.Fatalf("manifest proof: %s",got) }
}
func TestInvalidDigestAndSourceAttestationFailClosed(t *testing.T) {
	m:=base(1,1,1,it(1,"e",'c')); m.Items[0].Digest.Valid=false
	if got:=VerifyBoundedCapture(m,nil).Coverage; got!=CoverageContradicted { t.Fatalf("event digest: %s",got) }
	m=base(1,1,1); m.Source.Verification=Invalid
	if got:=VerifyBoundedCapture(m,nil).Coverage; got!=CoverageContradicted { t.Fatalf("source: %s",got) }
}
