package capture

import (
 "fmt"
 "testing"
 "time"
)

const da="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
const db="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
const dc="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
const dd="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
var now=time.Date(2026,11,27,20,0,0,0,time.UTC)

func fixture()Input{
 in:=Input{
  Profile:Profile,Version:Version,TrialID:"trial-038-001",CaptureID:"capture-038-001",
  Mode:OfflineFixture,EvaluatedAt:now,
  Boundary:RecoveryBoundary{
   IncidentID:"incident-038-001",ClaimID:"claim-cdc",SourceClusterID:"pg-cluster-a",
   Database:"application",SourceEpoch:"primary-epoch-7",TimelineID:"3",
   SnapshotID:"backup-038-001",TerminalPosition:"0/ABCDEF",
   ScopeDigest:da,InventoryDigest:db,
  },
  ProducerFenced:true,ProducerFenceProofSHA256:dc,ProducerFencedAt:now.Add(-2*time.Minute),
  CaptureStartedAt:now.Add(-90*time.Second),CaptureCompletedAt:now.Add(-20*time.Second),
  TopicInventoryComplete:true,
  TopicInventory:[]TopicInventory{
   {ClusterID:"kafka-cluster-1",Topic:"ledger",TopicID:"topic-id-ledger",PartitionCount:2,CleanupPolicy:"delete",ConfigSHA256:da},
   {ClusterID:"kafka-cluster-1",Topic:"message",TopicID:"topic-id-message",PartitionCount:1,CleanupPolicy:"delete",ConfigSHA256:db},
  },
  PartitionCaptures:[]PartitionCapture{
   {CaptureID:"capture-038-001",ClusterID:"kafka-cluster-1",Topic:"ledger",TopicID:"topic-id-ledger",Partition:0,
    StartOffset:100,EndOffsetExclusive:104,LogStartOffsetAtStart:50,LogStartOffsetAtEnd:50,
    LastStableOffsetBefore:104,LastStableOffsetAfter:104,HighWatermarkBefore:104,HighWatermarkAfter:104,LeaderEpochBefore:3,LeaderEpochAfter:3,
    IsolationLevel:"read_committed",CleanupPolicy:"delete",CaptureStartedAt:now.Add(-80*time.Second),CaptureCompletedAt:now.Add(-40*time.Second),CaptureComplete:true,
    Records:[]CapturedRecord{
     {Topic:"ledger",TopicID:"topic-id-ledger",Partition:0,Offset:100,RecordClass:"row_change",EventID:"row-1",TransactionID:"tx-1",DataCollection:"public.ledger",PayloadSHA256:da},
     {Topic:"ledger",TopicID:"topic-id-ledger",Partition:0,Offset:102,RecordClass:"row_change",EventID:"row-2",TransactionID:"tx-1",DataCollection:"public.ledger",PayloadSHA256:db},
    }},
   {CaptureID:"capture-038-001",ClusterID:"kafka-cluster-1",Topic:"ledger",TopicID:"topic-id-ledger",Partition:1,
    StartOffset:200,EndOffsetExclusive:202,LogStartOffsetAtStart:100,LogStartOffsetAtEnd:100,
    LastStableOffsetBefore:202,LastStableOffsetAfter:202,HighWatermarkBefore:202,HighWatermarkAfter:202,LeaderEpochBefore:8,LeaderEpochAfter:8,
    IsolationLevel:"read_committed",CleanupPolicy:"delete",CaptureStartedAt:now.Add(-80*time.Second),CaptureCompletedAt:now.Add(-40*time.Second),CaptureComplete:true,
    Records:[]CapturedRecord{
     {Topic:"ledger",TopicID:"topic-id-ledger",Partition:1,Offset:200,RecordClass:"row_change",EventID:"row-3",TransactionID:"tx-2",DataCollection:"public.ledger",PayloadSHA256:dc},
    }},
   {CaptureID:"capture-038-001",ClusterID:"kafka-cluster-1",Topic:"message",TopicID:"topic-id-message",Partition:0,
    StartOffset:40,EndOffsetExclusive:43,LogStartOffsetAtStart:0,LogStartOffsetAtEnd:0,
    LastStableOffsetBefore:43,LastStableOffsetAfter:43,HighWatermarkBefore:43,HighWatermarkAfter:43,LeaderEpochBefore:5,LeaderEpochAfter:5,
    IsolationLevel:"read_committed",CleanupPolicy:"delete",CaptureStartedAt:now.Add(-80*time.Second),CaptureCompletedAt:now.Add(-40*time.Second),CaptureComplete:true,
    Records:[]CapturedRecord{
     {Topic:"message",TopicID:"topic-id-message",Partition:0,Offset:40,RecordClass:"logical_message",EventID:"message-1",TransactionID:"tx-1",DataCollection:"LOGICAL_DECODING_MESSAGE",PayloadSHA256:dd},
     {Topic:"message",TopicID:"topic-id-message",Partition:0,Offset:42,RecordClass:"transaction_metadata",TransactionID:"tx-1",PayloadSHA256:da},
    }},
  },
  BusinessEvents:[]BusinessEvent{
   {EventID:"row-1",TransactionID:"tx-1",Topic:"ledger",TopicID:"topic-id-ledger",Partition:0,Offset:100,RecordClass:"row_change",DataCollection:"public.ledger",PayloadSHA256:da},
   {EventID:"row-2",TransactionID:"tx-1",Topic:"ledger",TopicID:"topic-id-ledger",Partition:0,Offset:102,RecordClass:"row_change",DataCollection:"public.ledger",PayloadSHA256:db},
   {EventID:"row-3",TransactionID:"tx-2",Topic:"ledger",TopicID:"topic-id-ledger",Partition:1,Offset:200,RecordClass:"row_change",DataCollection:"public.ledger",PayloadSHA256:dc},
   {EventID:"message-1",TransactionID:"tx-1",Topic:"message",TopicID:"topic-id-message",Partition:0,Offset:40,RecordClass:"logical_message",DataCollection:"LOGICAL_DECODING_MESSAGE",PayloadSHA256:dd},
  },
 }
 if !DeriveFixtureProofDigests(&in){panic("fixture digest derivation failed")}
 return in
}
func expect(t *testing.T,in Input,want Verdict){t.Helper();got:=Evaluate(in);if got.CaptureVerdict!=want{t.Fatalf("want %s, got %+v",want,got)}}
func refresh(in *Input){if !DeriveFixtureProofDigests(in){panic("fixture digest refresh failed")}}

func TestCleanMultiPartitionVectorQualifiesWithoutGlobalOrder(t *testing.T){
 in:=fixture();got:=Evaluate(in)
 if got.CaptureVerdict!=Qualified{t.Fatalf("valid partition vector did not qualify: %+v",got)}
 if got.CampaignVerdict!=Indeterminate||got.ProductionUsePermitted{t.Fatalf("synthetic capture must not authorize production: %+v",got)}
 if got.GlobalOrderInferred{t.Fatal("partition vector must never infer global order")}
 if got.PartitionCount!=3||got.BusinessEventCount!=4{t.Fatalf("wrong capture counts: %+v",got)}
 if got.OffsetGapsAccepted!=2{t.Fatalf("aborted/control offset gaps must be retained as gaps: %+v",got)}
}
func TestMissingPartitionRangeIsIndeterminate(t *testing.T){in:=fixture();in.PartitionCaptures=in.PartitionCaptures[:2];refresh(&in);expect(t,in,Indeterminate)}
func TestDuplicatePartitionRangeBlocks(t *testing.T){in:=fixture();in.PartitionCaptures=append(in.PartitionCaptures,in.PartitionCaptures[0]);refresh(&in);expect(t,in,Blocked)}
func TestPartitionOutsideDeclaredCountBlocks(t *testing.T){in:=fixture();in.PartitionCaptures[0].Partition=2;for i:=range in.PartitionCaptures[0].Records{in.PartitionCaptures[0].Records[i].Partition=2};in.BusinessEvents[0].Partition=2;in.BusinessEvents[1].Partition=2;refresh(&in);expect(t,in,Blocked)}
func TestReadUncommittedCaptureBlocks(t *testing.T){in:=fixture();in.PartitionCaptures[0].IsolationLevel="read_uncommitted";refresh(&in);expect(t,in,Blocked)}
func TestLastStableOffsetMustBeStable(t *testing.T){in:=fixture();in.PartitionCaptures[0].LastStableOffsetBefore=105;refresh(&in);expect(t,in,Indeterminate)}
func TestRangeCannotEndBeforeLastStableOffset(t *testing.T){in:=fixture();in.PartitionCaptures[0].EndOffsetExclusive=103;refresh(&in);expect(t,in,Indeterminate)}
func TestRangeCannotExtendPastLastStableOffset(t *testing.T){in:=fixture();in.PartitionCaptures[0].EndOffsetExclusive=105;refresh(&in);expect(t,in,Blocked)}
func TestLogStartAdvancingPastCaptureStartBlocks(t *testing.T){in:=fixture();in.PartitionCaptures[0].LogStartOffsetAtEnd=101;refresh(&in);expect(t,in,Blocked)}
func TestCompactedTopicBlocksExactHistoryClaim(t *testing.T){in:=fixture();in.TopicInventory[0].CleanupPolicy="compact,delete";in.PartitionCaptures[0].CleanupPolicy="compact,delete";refresh(&in);expect(t,in,Blocked)}
func TestLeaderEpochChangeRemainsIndeterminate(t *testing.T){in:=fixture();in.PartitionCaptures[0].LeaderEpochAfter=4;refresh(&in);expect(t,in,Indeterminate)}
func TestTopicIDRecreationMismatchBlocks(t *testing.T){in:=fixture();in.PartitionCaptures[0].TopicID="new-topic-id";refresh(&in);expect(t,in,Blocked)}
func TestUnfencedProducerCannotCloseVector(t *testing.T){in:=fixture();in.ProducerFenced=false;refresh(&in);expect(t,in,Indeterminate)}
func TestMissingFenceProofCannotCloseVector(t *testing.T){in:=fixture();in.ProducerFenceProofSHA256="";refresh(&in);expect(t,in,Indeterminate)}
func TestBusinessEventMustMatchExactCapturedRecord(t *testing.T){in:=fixture();in.BusinessEvents[0].PayloadSHA256=dd;refresh(&in);expect(t,in,Blocked)}
func TestCapturedBusinessRecordMustAppearInMon037EventSet(t *testing.T){in:=fixture();in.BusinessEvents=in.BusinessEvents[:3];refresh(&in);expect(t,in,Blocked)}
func TestDuplicatePartitionOffsetBlocks(t *testing.T){in:=fixture();duplicate:=in.PartitionCaptures[0].Records[0];in.PartitionCaptures[0].Records=append(in.PartitionCaptures[0].Records,duplicate);refresh(&in);expect(t,in,Blocked)}
func TestOutOfOrderCapturedRecordsBlock(t *testing.T){in:=fixture();records:=in.PartitionCaptures[0].Records;records[0],records[1]=records[1],records[0];refresh(&in);expect(t,in,Blocked)}
func TestOffsetGapsAreNotTreatedAsMissingMessages(t *testing.T){in:=fixture();got:=Evaluate(in);if got.CaptureVerdict!=Qualified{t.Fatalf("legal Kafka offset gaps were misclassified: %+v",got)}}
func TestSameTransactionMaySpanDifferentPartitions(t *testing.T){in:=fixture();got:=Evaluate(in);if got.CaptureVerdict!=Qualified{t.Fatalf("cross-partition transaction was rejected: %+v",got)}}
func TestBoundaryMutationAfterCaptureBindingBlocks(t *testing.T){in:=fixture();in.Boundary.TerminalPosition="0/ABCDF0";expect(t,in,Blocked)}
func TestMissingCaptureCompleteIsIndeterminate(t *testing.T){in:=fixture();in.PartitionCaptures[1].CaptureComplete=false;refresh(&in);expect(t,in,Indeterminate)}
func TestUnknownTopicCaptureBlocks(t *testing.T){in:=fixture();in.PartitionCaptures[0].Topic="unknown";refresh(&in);expect(t,in,Blocked)}
func TestBusinessEventOutsidePartitionRangeBlocks(t *testing.T){in:=fixture();in.BusinessEvents[0].Offset=999;refresh(&in);expect(t,in,Blocked)}
func TestRecordSetDigestMismatchBlocks(t *testing.T){in:=fixture();in.PartitionCaptures[0].RecordsSHA256=dd;expect(t,in,Blocked)}
func TestPartitionManifestDigestMismatchBlocks(t *testing.T){in:=fixture();in.PartitionCaptures[0].ManifestSHA256=dd;expect(t,in,Blocked)}
func TestCaptureWindowCannotPrecedeProducerFence(t *testing.T){in:=fixture();in.ProducerFencedAt=now.Add(time.Minute);refresh(&in);expect(t,in,Blocked)}
func TestTenThousandAdversarialMutationsNeverQualifyUnsafeCapture(t *testing.T){
 for n:=0;n<10000;n++{in:=fixture();switch n%10{
 case 0:in.ProducerFenced=false
 case 1:in.PartitionCaptures[0].IsolationLevel="read_uncommitted"
 case 2:in.PartitionCaptures[0].LastStableOffsetAfter++
 case 3:in.PartitionCaptures[0].LogStartOffsetAtEnd=in.PartitionCaptures[0].StartOffset+1
 case 4:in.PartitionCaptures[0].Records[0].PayloadSHA256=dd
 case 5:in.PartitionCaptures[0].TopicID="foreign-topic-id"
 case 6:in.BusinessEvents[0].Offset++
 case 7:in.PartitionCaptures= in.PartitionCaptures[:2]
 case 8:in.PartitionCaptures[0].CleanupPolicy="compact"
 case 9:in.Boundary.ClaimID="other-claim"
 };refresh(&in);if Evaluate(in).CaptureVerdict==Qualified{t.Fatalf("unsafe mutation %d qualified",n)}}
}
func BenchmarkEvaluate10000Records256Partitions(b *testing.B){
 in:=fixture();in.TopicInventory=[]TopicInventory{{ClusterID:"kafka-cluster-1",Topic:"ledger",TopicID:"topic-id-ledger",PartitionCount:256,CleanupPolicy:"delete",ConfigSHA256:da}}
 in.PartitionCaptures=nil;in.BusinessEvents=nil
 eventID:=0
 for partition:=0;partition<256;partition++{
  p:=PartitionCapture{CaptureID:in.CaptureID,ClusterID:"kafka-cluster-1",Topic:"ledger",TopicID:"topic-id-ledger",Partition:partition,
   StartOffset:0,EndOffsetExclusive:40,LogStartOffsetAtStart:0,LogStartOffsetAtEnd:0,LastStableOffsetBefore:40,LastStableOffsetAfter:40,HighWatermarkBefore:40,HighWatermarkAfter:40,
   LeaderEpochBefore:1,LeaderEpochAfter:1,IsolationLevel:"read_committed",CleanupPolicy:"delete",
   CaptureStartedAt:now.Add(-time.Minute),CaptureCompletedAt:now.Add(-time.Second),CaptureComplete:true}
  for i:=0;i<40;i++{eventID++;payload:=da;id:=fmt.Sprintf("e-%05d",eventID);p.Records=append(p.Records,CapturedRecord{Topic:"ledger",TopicID:"topic-id-ledger",Partition:partition,Offset:int64(i),RecordClass:"row_change",EventID:id,TransactionID:fmt.Sprintf("tx-%d",eventID),DataCollection:"public.ledger",PayloadSHA256:payload});in.BusinessEvents=append(in.BusinessEvents,BusinessEvent{EventID:id,TransactionID:fmt.Sprintf("tx-%d",eventID),Topic:"ledger",TopicID:"topic-id-ledger",Partition:partition,Offset:int64(i),RecordClass:"row_change",DataCollection:"public.ledger",PayloadSHA256:payload})}
  in.PartitionCaptures=append(in.PartitionCaptures,p)
 }
 refresh(&in);b.ReportAllocs();b.ResetTimer()
 for i:=0;i<b.N;i++{if Evaluate(in).CaptureVerdict!=Qualified{b.Fatal("benchmark fixture failed")}}
}
