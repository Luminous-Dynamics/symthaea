package custody

import (
 "crypto/ed25519"
 "encoding/base64"
 "encoding/hex"
 "fmt"
 "testing"
 "time"
)

var nowFixture=time.Date(2026,11,28,8,0,0,0,time.UTC)
var seedFixture=[]byte("0123456789abcdef0123456789abcdef")
func testDigest(s string)string{b:=[]byte(s);x:=make([]byte,32);copy(x,b);for i:=len(b);i<len(x);i++{x[i]=byte(i*7)};return hex.EncodeToString(x)}
func testNonce(n int)string{return fmt.Sprintf("%064x",n)}
func makeSigned(t testing.TB,p ed25519.PrivateKey,id,class string,seq uint64,n,prev,src string)Receipt{
 t.Helper()
 r:=Receipt{Profile:ReceiptProfile,Version:Version,ReceiptID:id,SessionID:"session-001",Nonce:n,ClaimID:"claim-001",TrialID:"trial-001",RecoveryBoundarySHA256:testDigest("boundary"),SourceID:src,EvidenceClass:class,CapturePrincipal:"capture-agent-01",CapturedAt:nowFixture.Add(-time.Minute),ExpiresAt:nowFixture.Add(10*time.Minute),Sequence:seq,PreviousReceiptSHA256:prev,ArtifactSHA256:testDigest("artifact-"+id),ArtifactBytes:1024,MediaType:"application/json",SignerKeyID:"capture-key-01"}
 r.SignatureBase64=base64.StdEncoding.EncodeToString(ed25519.Sign(p,CanonicalSigningBytes(r)));return r
}
func resignTest(r *Receipt,p ed25519.PrivateKey){r.SignatureBase64=base64.StdEncoding.EncodeToString(ed25519.Sign(p,CanonicalSigningBytes(*r)))}
func validInput(t testing.TB)Input{
 t.Helper();priv:=ed25519.NewKeyFromSeed(seedFixture);pub:=priv.Public().(ed25519.PublicKey)
 a:=makeSigned(t,priv,"receipt-001","postgres_source_truth",1,testNonce(1),"","postgres-primary")
 b:=makeSigned(t,priv,"receipt-002","kafka_partition_capture",2,testNonce(2),Digest(a),"kafka-cluster-01")
 return Input{Profile:Profile,Version:Version,Mode:OfflineFixture,EvaluatedAt:nowFixture,ClaimID:"claim-001",TrialID:"trial-001",RecoveryBoundarySHA256:testDigest("boundary"),
  RequiredEvidenceClasses:[]string{"postgres_source_truth","kafka_partition_capture"},
  ExpectedSources:map[string]string{"postgres_source_truth":"postgres-primary","kafka_partition_capture":"kafka-cluster-01"},
  IssuedNonces:[]string{testNonce(1),testNonce(2)},UsedNonces:[]string{},
  TrustedKeys:[]TrustedKey{{KeyID:"capture-key-01",Principal:"capture-agent-01",PublicKeyBase64:base64.StdEncoding.EncodeToString(pub),NotBefore:nowFixture.Add(-24*time.Hour),NotAfter:nowFixture.Add(24*time.Hour)}},
  Receipts:[]Receipt{a,b},MaxReceiptAgeSeconds:3600,AllowedClockSkewSeconds:30}
}
func TestValidSyntheticReceiptsQualifyCustodyOnly(t testing.TB){
 got:=Evaluate(validInput(t))
 if got.CustodyVerdict!=Qualified||got.AuthenticatedCount!=2{t.Fatalf("expected two authenticated receipts: %+v",got)}
 if got.CampaignVerdict!=Indeterminate||got.ProductionActionPermitted{t.Fatalf("custody prototype must never authorize promotion: %+v",got)}
}
func TestTamperedSignedFieldBlocks(t testing.TB){in:=validInput(t);in.Receipts[0].ArtifactSHA256=testDigest("changed");if got:=Evaluate(in);got.CustodyVerdict!=Blocked{t.Fatalf("tampering must block: %+v",got)}}
func TestWrongClaimBlocksEvenWithValidSignature(t testing.TB){in:=validInput(t);p:=ed25519.NewKeyFromSeed(seedFixture);in.Receipts[0].ClaimID="other-claim";resignTest(&in.Receipts[0],p);if got:=Evaluate(in);got.CustodyVerdict!=Blocked{t.Fatalf("cross-claim receipt must block: %+v",got)}}
func TestWrongSourceBlocks(t testing.TB){in:=validInput(t);p:=ed25519.NewKeyFromSeed(seedFixture);in.Receipts[0].SourceID="other-db";resignTest(&in.Receipts[0],p);if got:=Evaluate(in);got.CustodyVerdict!=Blocked{t.Fatalf("wrong source must block: %+v",got)}}
func TestUsedNonceBlocks(t testing.TB){in:=validInput(t);in.UsedNonces=[]string{testNonce(1)};if got:=Evaluate(in);got.CustodyVerdict!=Blocked{t.Fatalf("used nonce must block: %+v",got)}}
func TestDuplicateNonceBlocks(t testing.TB){in:=validInput(t);p:=ed25519.NewKeyFromSeed(seedFixture);in.Receipts[1].Nonce=in.Receipts[0].Nonce;resignTest(&in.Receipts[1],p);if got:=Evaluate(in);got.CustodyVerdict!=Blocked{t.Fatalf("duplicate nonce must block: %+v",got)}}
func TestExpiredReceiptBlocks(t testing.TB){in:=validInput(t);p:=ed25519.NewKeyFromSeed(seedFixture);in.Receipts[0].ExpiresAt=nowFixture.Add(-time.Second);resignTest(&in.Receipts[0],p);if got:=Evaluate(in);got.CustodyVerdict!=Blocked{t.Fatalf("expired receipt must block: %+v",got)}}
func TestStaleReceiptBlocks(t testing.TB){in:=validInput(t);p:=ed25519.NewKeyFromSeed(seedFixture);in.Receipts[0].CapturedAt=nowFixture.Add(-2*time.Hour);resignTest(&in.Receipts[0],p);if got:=Evaluate(in);got.CustodyVerdict!=Blocked{t.Fatalf("stale receipt must block: %+v",got)}}
func TestRevokedKeyBlocksWhenRevocationPredatesCapture(t testing.TB){
 in:=validInput(t)
 revoked:=in.Receipts[0].CapturedAt.Add(-time.Second)
 in.TrustedKeys[0].RevokedAt=&revoked
 if got:=Evaluate(in);got.CustodyVerdict!=Blocked{t.Fatalf("receipt captured after effective revocation must block: %+v",got)}
}
func TestRevocationAfterCapturePreservesHistoricalReceipt(t testing.TB){
 in:=validInput(t)
 revoked:=in.Receipts[0].CapturedAt.Add(time.Second)
 in.TrustedKeys[0].RevokedAt=&revoked
 if got:=Evaluate(in);got.CustodyVerdict!=Qualified{t.Fatalf("later revocation must not invalidate a receipt captured before its effective time: %+v",got)}
 if got:=Evaluate(in);got.CampaignVerdict!=Indeterminate||got.ProductionActionPermitted{t.Fatalf("historical custody preflight cannot authorize campaign: %+v",got)}
}
func TestUnknownSignerHoldsIndeterminate(t testing.TB){in:=validInput(t);in.TrustedKeys=nil;if got:=Evaluate(in);got.CustodyVerdict!=Indeterminate{t.Fatalf("missing trust root must hold: %+v",got)}}
func TestMissingRequiredEvidenceHolds(t testing.TB){in:=validInput(t);in.Receipts=in.Receipts[:1];if got:=Evaluate(in);got.CustodyVerdict!=Indeterminate{t.Fatalf("incomplete evidence must hold: %+v",got)}}
func TestBrokenChainBlocks(t testing.TB){in:=validInput(t);p:=ed25519.NewKeyFromSeed(seedFixture);in.Receipts[1].PreviousReceiptSHA256=testDigest("wrong-parent");resignTest(&in.Receipts[1],p);if got:=Evaluate(in);got.CustodyVerdict!=Blocked{t.Fatalf("broken receipt chain must block: %+v",got)}}
func TestSequenceGapBlocks(t testing.TB){in:=validInput(t);p:=ed25519.NewKeyFromSeed(seedFixture);in.Receipts[1].Sequence=3;resignTest(&in.Receipts[1],p);if got:=Evaluate(in);got.CustodyVerdict!=Blocked{t.Fatalf("sequence gap must block: %+v",got)}}
func TestFutureReceiptOutsideClockSkewBlocks(t testing.TB){in:=validInput(t);p:=ed25519.NewKeyFromSeed(seedFixture);in.Receipts[0].CapturedAt=nowFixture.Add(time.Minute);resignTest(&in.Receipts[0],p);if got:=Evaluate(in);got.CustodyVerdict!=Blocked{t.Fatalf("future timestamp must block: %+v",got)}}
func TestUnknownModeBlocks(t testing.TB){in:=validInput(t);in.Mode="promotion";if got:=Evaluate(in);got.CustodyVerdict!=Blocked{t.Fatalf("unknown mode must block: %+v",got)}}
func TestMissingNonceSnapshotHolds(t testing.TB){in:=validInput(t);in.UsedNonces=nil;if got:=Evaluate(in);got.CustodyVerdict!=Indeterminate{t.Fatalf("missing consumed nonce snapshot must hold: %+v",got)}}
func BenchmarkEvaluateTwoReceipts(b *testing.B){
 in:=validInput(b);b.ReportAllocs();b.ResetTimer()
 for i:=0;i<b.N;i++{_ = Evaluate(in)}
}
