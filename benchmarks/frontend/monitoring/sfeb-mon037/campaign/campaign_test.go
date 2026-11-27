package campaign

import (
 "crypto/sha256"
 "encoding/hex"
 "encoding/json"
 "fmt"
 "testing"
 "time"
)

const da="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
const db="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
const dc="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
func fixtureDigest(s string)string{sum:=sha256.Sum256([]byte(s));return hex.EncodeToString(sum[:])}
var now=time.Date(2026,11,27,12,0,0,0,time.UTC)

func raw(v any)json.RawMessage{b,_:=json.Marshal(v);return b}
func goodRestore()map[string]string{m:=map[string]string{};for _,k:=range restoreKeys{m[k]="passed"};return m}
func refreshRecordDigests(in *Input){in.SourceEventsSHA256=sourceEventsDigest(in.SourceEvents);in.ObservedEventsSHA256=observedEventsDigest(in.ObservedEvents);in.TransactionProofsSHA256=transactionProofsDigest(in.Transactions);in.TopicPartitionsSHA256=topicPartitionsDigest(in.TopicPartitions);in.RequiredSurfaceIDsSHA256=surfaceIDsDigest(in.RequiredSurfaceIDs);in.SurfaceAssertionsSHA256=surfaceAssertionsDigest(in.Surfaces)}
func fixture()Input{
 events:=[]Event{
  {EventID:"row-1",TransactionID:"tx-1",Topic:"ledger",DataCollection:"public.ledger",Operation:"INSERT",PayloadSHA256:da,TotalOrder:1,CollectionOrder:1,Partition:0,Offset:10},
  {EventID:"message-1",TransactionID:"tx-1",Topic:"message",DataCollection:"LOGICAL_DECODING_MESSAGE",Operation:"MESSAGE",PayloadSHA256:db,TotalOrder:2,CollectionOrder:1,Partition:0,Offset:7},
 }
 in := Input{
  Profile:Profile,Version:Version,TrialID:"trial-001",ScenarioID:"MON037-P001",Mode:OfflineFixture,EvaluatedAt:now,
  Boundary:RecoveryBoundary{
   IncidentID:"mon037-fixture-incident-001",ClaimID:"claim-cdc",SourceClusterID:"fixture-source-cluster",
   Database:"mon037_fixture",SourceEpoch:"primary-epoch-001",TimelineID:"1",SnapshotID:"snapshot-001",
   TerminalPosition:"0/200",ScopeDigest:da,InventoryDigest:db,
  },
  RequiredClaimIDs:[]string{"claim-cdc","claim-backup"},
  SourceTruthComplete:true,SourceTruthManifestSHA256:fixtureDigest("MON037 synthetic SQL source-truth manifest v1\n"),
  ObservedCaptureComplete:true,ObservedCaptureManifestSHA256:fixtureDigest("MON037 synthetic Kafka capture manifest v1\n"),
  TransactionReportComplete:true,TransactionReportSHA256:fixtureDigest("MON037 synthetic transaction report v1\n"),
  RequiredSurfaceIDs:[]string{"heap","index","large-object"},
  SurfaceInventoryComplete:true,SurfaceInventorySHA256:fixtureDigest("MON037 synthetic required database surface inventory v1\n"),
  SourceEvents:events,ObservedEvents:events,
  TopicPartitions:[]TopicPartition{{Topic:"ledger",PartitionCount:1},{Topic:"message",PartitionCount:1}},
  Transactions:[]TransactionProof{{TransactionID:"tx-1",ExpectedEventCount:2,BeginObserved:true,EndObserved:true,EndEventCount:2,DataCollections:map[string]int{"public.ledger":1,"LOGICAL_DECODING_MESSAGE":1}}},
  SurfaceScope:"full_state",
  Surfaces:[]Surface{
   {ID:"heap",Disposition:"covered",SourceSHA256:da,RestoredSHA256:da},
   {ID:"index",Disposition:"covered",SourceSHA256:db,RestoredSHA256:db},
   {ID:"large-object",Disposition:"not_applicable",JustificationSHA256:dc},
  },
  RestoreChecks:goodRestore(),
  Components:Components{
   TimelineResult:raw(map[string]any{"profile":"mon034-postgres-timeline-ancestry-result-v1","version":"1.0.0","result":map[string]any{"verdict":"qualified"}}),
   ObserverResult:raw(map[string]any{"profile":"mon035-observer-independence-result-v1","version":"1.0.0","result":map[string]any{
    "verdict":"qualified","independent":true,"shared_domains":[]any{},
    "observer_assessments":[]any{map[string]any{"observer_id":"a","eligible":true},map[string]any{"observer_id":"b","eligible":true}},
   }}),
   ContinuityResult:raw(map[string]any{"profile":"mon036-recovery-evidence-continuity-result-v1","version":"1.0.0",
    "ledger_integrity_valid":true,"ledger_complete":true,
    "claims":[]any{map[string]any{"claim_id":"claim-cdc","disposition":"retain"},map[string]any{"claim_id":"claim-backup","disposition":"retain"}},
   }),
   ProofGraphResult:raw(map[string]any{"verdict":"indeterminate","may_enable_traffic":false}),
  },
 }
 if !DeriveFixtureRecordSetDigests(&in) { panic("fixture boundary binding failed") }
 return in
}
func expect(t *testing.T,in Input,want Verdict){t.Helper();got:=Evaluate(in);if got.ComponentVerdict!=want{t.Fatalf("want %s, got %+v",want,got)}}

func TestCleanFixturePassesComponentModelButNotProduction(t *testing.T){
 got:=Evaluate(fixture())
 if got.ComponentVerdict!=Qualified||got.CampaignVerdict!=Indeterminate||got.TrafficActivationPermitted{t.Fatalf("offline fixture must remain production-indeterminate: %+v",got)}
}
func TestMissingAcknowledgedEventBlocks(t *testing.T){in:=fixture();in.ObservedEvents=in.ObservedEvents[:1];refreshRecordDigests(&in);expect(t,in,Blocked)}
func TestUnexpectedEventBlocks(t *testing.T){in:=fixture();e:=in.ObservedEvents[0];e.EventID="extra";e.PayloadSHA256=dc;e.Offset=11;e.TotalOrder=3;e.CollectionOrder=2;in.ObservedEvents=append(in.ObservedEvents,e);refreshRecordDigests(&in);expect(t,in,Blocked)}
func TestConflictingDuplicateIdentityBlocks(t *testing.T){in:=fixture();e:=in.ObservedEvents[0];e.PayloadSHA256=dc;e.Offset=11;in.ObservedEvents=append(in.ObservedEvents,e);refreshRecordDigests(&in);expect(t,in,Blocked)}
func TestExactReplayRequiresDeduplicationProof(t *testing.T){
 in:=fixture();e:=in.ObservedEvents[0];e.Offset=11;in.ObservedEvents=append(in.ObservedEvents,e);refreshRecordDigests(&in)
 expect(t,in,Indeterminate)
 in.ExactReplayDeduplicationProven=true;in.DeduplicationProofSHA256=fixtureDigest("MON037 synthetic replay-deduplication proof v1\n");in.DeduplicationCoverageSHA256=canonicalDigest([]string{"row-1"});expect(t,in,Qualified)
 got:=Evaluate(in);if got.Reconciliation.DuplicateReplays!=1{t.Fatalf("duplicate replay count lost: %+v",got.Reconciliation)}
}
func TestPayloadMismatchBlocks(t *testing.T){in:=fixture();in.ObservedEvents[0].PayloadSHA256=dc;refreshRecordDigests(&in);expect(t,in,Blocked)}
func TestTransactionEndCountMismatchBlocks(t *testing.T){in:=fixture();in.Transactions[0].EndEventCount=1;refreshRecordDigests(&in);expect(t,in,Blocked)}
func TestTransactionDataCollectionsMismatchBlocks(t *testing.T){in:=fixture();in.Transactions[0].DataCollections["LOGICAL_DECODING_MESSAGE"]=2;refreshRecordDigests(&in);expect(t,in,Blocked)}
func TestPerTopicPartitionViolationBlocks(t *testing.T){in:=fixture();in.ObservedEvents[0].Partition=1;refreshRecordDigests(&in);expect(t,in,Blocked)}
func TestMultiplePartitionsAreIndeterminate(t *testing.T){in:=fixture();in.TopicPartitions[0].PartitionCount=2;refreshRecordDigests(&in);expect(t,in,Indeterminate)}
func TestTimelineBlockedPropagates(t *testing.T){in:=fixture();in.Components.TimelineResult=raw(map[string]any{"profile":"mon034-postgres-timeline-ancestry-result-v1","version":"1.0.0","result":map[string]any{"verdict":"blocked"}});expect(t,in,Blocked)}
func TestObserverSharedDomainBlocks(t *testing.T){
 in:=fixture()
 in.Components.ObserverResult=raw(map[string]any{"profile":"mon035-observer-independence-result-v1","version":"1.0.0","result":map[string]any{
  "verdict":"blocked","independent":false,"shared_domains":[]any{map[string]any{"class":"identity_provider","domain_id":"shared-iam"}},
  "observer_assessments":[]any{map[string]any{"observer_id":"a","eligible":true},map[string]any{"observer_id":"b","eligible":true}},
 }})
 expect(t,in,Blocked)
}
func TestMissingObserverEvidenceIsIndeterminate(t *testing.T){in:=fixture();in.Components.ObserverResult=nil;expect(t,in,Indeterminate)}
func TestContinuityRevalidationHoldsCampaign(t *testing.T){
 in:=fixture()
 in.Components.ContinuityResult=raw(map[string]any{"profile":"mon036-recovery-evidence-continuity-result-v1","version":"1.0.0","ledger_integrity_valid":true,"ledger_complete":true,
 "claims":[]any{map[string]any{"claim_id":"claim-cdc","disposition":"revalidate"},map[string]any{"claim_id":"claim-backup","disposition":"retain"}}})
 expect(t,in,Indeterminate)
}
func TestContinuityRevocationBlocks(t *testing.T){
 in:=fixture()
 in.Components.ContinuityResult=raw(map[string]any{"profile":"mon036-recovery-evidence-continuity-result-v1","version":"1.0.0","ledger_integrity_valid":true,"ledger_complete":true,
 "claims":[]any{map[string]any{"claim_id":"claim-cdc","disposition":"invalidate"},map[string]any{"claim_id":"claim-backup","disposition":"retain"}}})
 expect(t,in,Blocked)
}
func TestExactOffsetTargetNotObservedIsIndeterminate(t *testing.T){
 in:=fixture();in.OffsetTargetRequired=true;in.OffsetTarget="lsn_events_processed_1"
 in.Components.OffsetProof=raw(OffsetProof{Profile:"mon033-postgres-offset-target-v1",Version:"1.0.0",Target:in.OffsetTarget,TargetReached:false})
 expect(t,in,Indeterminate)
}
func TestOffsetPredicateMismatchBlocks(t *testing.T){
 in:=fixture();in.OffsetTargetRequired=true;in.OffsetTarget="lsn_events_processed_1"
 p:=OffsetProof{Profile:"mon033-postgres-offset-target-v1",Version:"1.0.0",Target:in.OffsetTarget,TargetReached:true,CapturedAt:now.Add(-time.Minute),OffsetsSHA256:da}
 p.Offset.LSN="0/10";p.Offset.LSNProc="0/F";p.Offset.LSNCommit="0/10";p.Offset.MessageType="MESSAGE";p.Offset.EventsProcessed=1
 in.Components.OffsetProof=raw(p);expect(t,in,Blocked)
}
func TestSurfaceDigestMismatchBlocks(t *testing.T){in:=fixture();in.Surfaces[0].RestoredSHA256=dc;refreshRecordDigests(&in);expect(t,in,Blocked)}
func TestFullStateExclusionBlocks(t *testing.T){in:=fixture();in.Surfaces=append(in.Surfaces,Surface{ID:"generated",Disposition:"excluded"});in.RequiredSurfaceIDs=append(in.RequiredSurfaceIDs,"generated");refreshRecordDigests(&in);expect(t,in,Blocked)}
func TestRowStreamAllowsExplicitExclusion(t *testing.T){in:=fixture();in.SurfaceScope="row_stream";in.Surfaces=append(in.Surfaces,Surface{ID:"generated",Disposition:"excluded",ExclusionManifestSHA256:fixtureDigest("MON037 synthetic generated-surface exclusion manifest v1\n")});in.RequiredSurfaceIDs=append(in.RequiredSurfaceIDs,"generated");refreshRecordDigests(&in);expect(t,in,Qualified)}
func TestUnknownSurfaceIsIndeterminate(t *testing.T){in:=fixture();in.Surfaces=append(in.Surfaces,Surface{ID:"unknown",Disposition:"unknown"});in.RequiredSurfaceIDs=append(in.RequiredSurfaceIDs,"unknown");refreshRecordDigests(&in);expect(t,in,Indeterminate)}
func TestOldPrimaryUnfencedBlocks(t *testing.T){in:=fixture();in.RestoreChecks["old_primary_fenced"]="failed";expect(t,in,Blocked)}
func TestMissingRestoreCheckIsIndeterminate(t *testing.T){in:=fixture();delete(in.RestoreChecks,"application_write_readback");expect(t,in,Indeterminate)}
func TestTrafficAlreadyEnabledBlocks(t *testing.T){in:=fixture();in.TrafficCurrentlyEnabled=true;got:=Evaluate(in);if got.CampaignVerdict!=Blocked||got.TrafficActivationPermitted{t.Fatalf("unsafe traffic not blocked: %+v",got)}}
func TestSyntheticProofPromotionBlocks(t *testing.T){in:=fixture();in.Components.ProofGraphResult=raw(map[string]any{"verdict":"qualified","may_enable_traffic":true});got:=Evaluate(in);if got.CampaignVerdict!=Blocked||got.TrafficActivationPermitted{t.Fatalf("synthetic proof promoted: %+v",got)}}
func TestDeterministicTenThousandUnsafeMutationsNeverPass(t *testing.T){
 for n:=0;n<10000;n++{
  in:=fixture()
  switch n%8{
  case 0:in.ObservedEvents=in.ObservedEvents[:1]
  case 1:in.ObservedEvents[0].PayloadSHA256=dc
  case 2:in.Transactions[0].EndEventCount++
  case 3:in.TopicPartitions[0].PartitionCount=2
  case 4:in.Surfaces[0].RestoredSHA256=dc
  case 5:in.RestoreChecks["dependencies_ready"]="failed"
  case 6:in.Components.TimelineResult=nil
  case 7:in.Components.ContinuityResult=nil
  }
  refreshRecordDigests(&in)
  if got:=Evaluate(in);got.ComponentVerdict==Qualified{t.Fatalf("unsafe mutation %d qualified: %+v",n,got)}
 }
}
func BenchmarkEvaluate1000Events100Transactions(b *testing.B){
 in:=fixture();in.SourceEvents=nil;in.ObservedEvents=nil;in.Transactions=nil
 for tx:=0;tx<100;tx++{
  counts:=map[string]int{"public.ledger":0,"LOGICAL_DECODING_MESSAGE":0}
  for n:=0;n<10;n++{topic:="ledger";collection:="public.ledger";if n%2==1{topic="message";collection="LOGICAL_DECODING_MESSAGE"};counts[collection]++
   e:=Event{EventID:fmt.Sprintf("e-%03d-%02d",tx,n),TransactionID:fmt.Sprintf("tx-%03d",tx),Topic:topic,DataCollection:collection,Operation:"INSERT",PayloadSHA256:da,TotalOrder:n+1,CollectionOrder:counts[collection],Partition:0,Offset:int64(tx*10+n)}
   in.SourceEvents=append(in.SourceEvents,e);in.ObservedEvents=append(in.ObservedEvents,e)
  }
  in.Transactions=append(in.Transactions,TransactionProof{TransactionID:fmt.Sprintf("tx-%03d",tx),ExpectedEventCount:10,BeginObserved:true,EndObserved:true,EndEventCount:10,DataCollections:counts})
 }
 b.ReportAllocs();b.ResetTimer()
 for n:=0;n<b.N;n++{if Evaluate(in).ComponentVerdict!=Qualified{b.Fatal("benchmark fixture failed")}}
}


func TestIncompleteSourceTruthAndTopicCaptureAreIndeterminate(t *testing.T) {
	in := fixture()
	in.SourceTruthComplete = false
	expect(t, in, Indeterminate)
	in = fixture()
	in.ObservedCaptureComplete = false
	expect(t, in, Indeterminate)
	in = fixture()
	in.TransactionReportComplete = false
	expect(t, in, Indeterminate)
}

func TestMissingRequiredSurfaceIsIndeterminate(t *testing.T) {
	in := fixture()
	in.Surfaces = in.Surfaces[:2]
	refreshRecordDigests(&in)
	got := Evaluate(in)
	if got.ComponentVerdict != Indeterminate {
		t.Fatalf("omitted required surface must remain indeterminate: %+v", got)
	}
}

func TestRowStreamExclusionRequiresManifest(t *testing.T) {
	in := fixture()
	in.SurfaceScope = "row_stream"
	in.Surfaces = append(in.Surfaces, Surface{ID:"generated", Disposition:"excluded", ExclusionManifestSHA256:fixtureDigest("MON037 synthetic generated-surface exclusion manifest v1\\n")})
	in.RequiredSurfaceIDs = append(in.RequiredSurfaceIDs, "generated")
	refreshRecordDigests(&in)
	expect(t, in, Qualified)
}


func TestCollectionOrderMayResetAcrossTransactions(t *testing.T) {
	in := fixture()
	row2 := in.SourceEvents[0]
	row2.EventID = "row-2"
	row2.TransactionID = "tx-2"
	row2.Offset = 11
	row2.TotalOrder = 1
	row2.CollectionOrder = 1
	message2 := in.SourceEvents[1]
	message2.EventID = "message-2"
	message2.TransactionID = "tx-2"
	message2.Offset = 8
	message2.TotalOrder = 2
	message2.CollectionOrder = 1
	in.SourceEvents = append(in.SourceEvents, row2, message2)
	in.ObservedEvents = append(in.ObservedEvents, row2, message2)
	in.Transactions = append(in.Transactions, TransactionProof{
		TransactionID: "tx-2", ExpectedEventCount: 2, BeginObserved: true,
		EndObserved: true, EndEventCount: 2,
		DataCollections: map[string]int{"public.ledger": 1, "LOGICAL_DECODING_MESSAGE": 1},
	})
	refreshRecordDigests(&in)
	expect(t, in, Qualified)
}

func TestMessagePseudoCollectionLabelCanVaryAcrossTransactions(t *testing.T) {
 in:=fixture()
 row2:=in.SourceEvents[0]
 row2.EventID="row-2";row2.TransactionID="tx-2";row2.Offset=11;row2.TotalOrder=1;row2.CollectionOrder=1
 message2:=in.SourceEvents[1]
 message2.EventID="message-2";message2.TransactionID="tx-2";message2.Offset=8;message2.TotalOrder=2;message2.CollectionOrder=1
 message2.DataCollection="logical-message-pseudo-v2"
 in.SourceEvents=append(in.SourceEvents,row2,message2)
 in.ObservedEvents=append(in.ObservedEvents,row2,message2)
 in.Transactions=append(in.Transactions,TransactionProof{
  TransactionID:"tx-2",ExpectedEventCount:2,BeginObserved:true,EndObserved:true,EndEventCount:2,
  DataCollections:map[string]int{"public.ledger":1,"logical-message-pseudo-v2":1},
 })
 refreshRecordDigests(&in)
 expect(t,in,Qualified)
}

func TestRowCollectionOrderGapWithinTransactionBlocks(t *testing.T) {
	in := fixture()
	row2 := in.SourceEvents[0]
	row2.EventID = "row-2"
	row2.TotalOrder = 3
	row2.CollectionOrder = 3
	row2.Offset = 11
	in.SourceEvents = append(in.SourceEvents, row2)
	in.ObservedEvents = append(in.ObservedEvents, row2)
	in.Transactions[0].ExpectedEventCount = 3
	in.Transactions[0].EndEventCount = 3
	in.Transactions[0].DataCollections["public.ledger"] = 2
	refreshRecordDigests(&in)
	expect(t, in, Blocked)
}


func TestRecordSetDigestMismatchBlocks(t *testing.T) {
	in := fixture()
	in.SourceEventsSHA256 = dc
	expect(t, in, Blocked)
	in = fixture()
	in.ObservedEventsSHA256 = dc
	expect(t, in, Blocked)
	in = fixture()
	in.TransactionProofsSHA256 = dc
	expect(t, in, Blocked)
	in = fixture()
	in.TopicPartitionsSHA256 = dc
	expect(t, in, Blocked)
	in = fixture()
	in.SurfaceAssertionsSHA256 = dc
	expect(t, in, Blocked)
}

func TestFixtureDigestDerivationCannotRunForRuntimeCandidate(t *testing.T) {
	in := fixture()
	in.Mode = RuntimeCandidate
	in.SourceEventsSHA256 = ""
	if DeriveFixtureRecordSetDigests(&in) {
		t.Fatal("runtime candidate must never derive its own evidence digests")
	}
	in.Mode = OfflineFixture
	if !DeriveFixtureRecordSetDigests(&in) || in.SourceEventsSHA256 != sourceEventsDigest(in.SourceEvents) {
		t.Fatal("offline fixture digest derivation failed")
	}
}
