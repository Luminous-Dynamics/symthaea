// Package capture verifies a Kafka partition-vector capture against one
// shared recovery boundary. It never infers a global order across partitions.
package capture

import (
 "crypto/sha256"
 "encoding/hex"
 "encoding/json"
 "sort"
 "strings"
 "time"
)

const (
 Profile="mon038-kafka-partition-capture-input-v1"
 Version="1.0.0"
 MaxTopics=10000
 MaxPartitions=100000
 MaxRecords=1000000
)

type Verdict string
const (
 Qualified Verdict="qualified"
 Blocked Verdict="blocked"
 Indeterminate Verdict="indeterminate"
)
type Mode string
const (
 OfflineFixture Mode="offline_fixture"
 RuntimeCandidate Mode="runtime_candidate"
)
type RecoveryBoundary struct {
 IncidentID string `json:"incident_id"`
 ClaimID string `json:"claim_id"`
 SourceClusterID string `json:"source_cluster_id"`
 Database string `json:"database"`
 SourceEpoch string `json:"source_epoch"`
 TimelineID string `json:"timeline_id"`
 SnapshotID string `json:"snapshot_id"`
 TerminalPosition string `json:"terminal_position"`
 ScopeDigest string `json:"scope_digest"`
 InventoryDigest string `json:"inventory_digest"`
}
type TopicInventory struct {
 ClusterID string `json:"cluster_id"`
 Topic string `json:"topic"`
 TopicID string `json:"topic_id"`
 PartitionCount int `json:"partition_count"`
 CleanupPolicy string `json:"cleanup_policy"`
 ConfigSHA256 string `json:"config_sha256"`
}
type CapturedRecord struct {
 Topic string `json:"topic"`
 TopicID string `json:"topic_id"`
 Partition int `json:"partition"`
 Offset int64 `json:"offset"`
 RecordClass string `json:"record_class"`
 EventID string `json:"event_id,omitempty"`
 TransactionID string `json:"transaction_id,omitempty"`
 DataCollection string `json:"data_collection,omitempty"`
 PayloadSHA256 string `json:"payload_sha256"`
 KeySHA256 string `json:"key_sha256,omitempty"`
}
type PartitionCapture struct {
 CaptureID string `json:"capture_id"`
 ClusterID string `json:"cluster_id"`
 Topic string `json:"topic"`
 TopicID string `json:"topic_id"`
 Partition int `json:"partition"`
 BoundarySHA256 string `json:"boundary_sha256"`
 StartOffset int64 `json:"start_offset_inclusive"`
 EndOffsetExclusive int64 `json:"end_offset_exclusive"`
 LogStartOffsetAtStart int64 `json:"log_start_offset_at_start"`
 LogStartOffsetAtEnd int64 `json:"log_start_offset_at_end"`
 LastStableOffsetBefore int64 `json:"last_stable_offset_before"`
 LastStableOffsetAfter int64 `json:"last_stable_offset_after"`
 HighWatermarkBefore int64 `json:"high_watermark_before"`
 HighWatermarkAfter int64 `json:"high_watermark_after"`
 LeaderEpochBefore int32 `json:"leader_epoch_before"`
 LeaderEpochAfter int32 `json:"leader_epoch_after"`
 IsolationLevel string `json:"isolation_level"`
 CleanupPolicy string `json:"cleanup_policy"`
 CaptureStartedAt time.Time `json:"capture_started_at"`
 CaptureCompletedAt time.Time `json:"capture_completed_at"`
 CaptureComplete bool `json:"capture_complete"`
 RecordCount int `json:"record_count"`
 RecordsSHA256 string `json:"records_sha256"`
 ManifestSHA256 string `json:"manifest_sha256"`
 Records []CapturedRecord `json:"records"`
}
type BusinessEvent struct {
 EventID string `json:"event_id"`
 TransactionID string `json:"transaction_id"`
 Topic string `json:"topic"`
 TopicID string `json:"topic_id"`
 Partition int `json:"partition"`
 Offset int64 `json:"offset"`
 RecordClass string `json:"record_class"`
 DataCollection string `json:"data_collection"`
 PayloadSHA256 string `json:"payload_sha256"`
}
type Input struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 TrialID string `json:"trial_id"`
 CaptureID string `json:"capture_id"`
 Mode Mode `json:"mode"`
 EvaluatedAt time.Time `json:"evaluated_at"`
 Boundary RecoveryBoundary `json:"recovery_boundary"`
 BoundarySHA256 string `json:"boundary_sha256"`
 ProducerFenced bool `json:"producer_fenced"`
 ProducerFenceProofSHA256 string `json:"producer_fence_proof_sha256"`
 ProducerFencedAt time.Time `json:"producer_fenced_at"`
 CaptureStartedAt time.Time `json:"capture_started_at"`
 CaptureCompletedAt time.Time `json:"capture_completed_at"`
 TopicInventoryComplete bool `json:"topic_inventory_complete"`
 TopicInventorySHA256 string `json:"topic_inventory_sha256"`
 TopicInventory []TopicInventory `json:"topic_inventory"`
 PartitionRangesSHA256 string `json:"partition_ranges_sha256"`
 PartitionCaptures []PartitionCapture `json:"partition_captures"`
 BusinessEventsSHA256 string `json:"business_events_sha256"`
 BusinessEvents []BusinessEvent `json:"business_events"`
}
type PartitionSummary struct {
 Topic string `json:"topic"`
 Partition int `json:"partition"`
 StartOffset int64 `json:"start_offset_inclusive"`
 EndOffsetExclusive int64 `json:"end_offset_exclusive"`
 LSO int64 `json:"last_stable_offset"`
 Records int `json:"record_count"`
 OffsetGapSlots int64 `json:"offset_gap_slots"`
 OffsetGapsObserved bool `json:"offset_gaps_observed"`
}
type Issue struct {
 Code string `json:"code"`
 Detail string `json:"detail"`
}
type Result struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 TrialID string `json:"trial_id"`
 CaptureID string `json:"capture_id"`
 CaptureVerdict Verdict `json:"capture_verdict"`
 CampaignVerdict Verdict `json:"campaign_verdict"`
 ProductionUsePermitted bool `json:"production_use_permitted"`
 GlobalOrderInferred bool `json:"global_order_inferred"`
 TopicCount int `json:"topic_count"`
 PartitionCount int `json:"partition_count"`
 VisibleRecordCount int `json:"visible_record_count"`
 BusinessEventCount int `json:"business_event_count"`
 OffsetGapsAccepted int64 `json:"offset_gaps_accepted"`
 Partitions []PartitionSummary `json:"partitions"`
 Issues []Issue `json:"issues"`
 ReasonCodes []string `json:"reason_codes"`
}

type flags struct { blocked bool; unknown bool; issues []Issue }
func nf()*flags{return &flags{issues:[]Issue{}}}
func(f *flags)block(code,detail string){f.blocked=true;f.issues=append(f.issues,Issue{code,detail})}
func(f *flags)hold(code,detail string){f.unknown=true;f.issues=append(f.issues,Issue{code,detail})}
func(f *flags)verdict()Verdict{if f.blocked{return Blocked};if f.unknown{return Indeterminate};return Qualified}
func(f *flags)codes()[]string{set:=map[string]struct{}{};for _,i:=range f.issues{set[i.Code]=struct{}{}};out:=make([]string,0,len(set));for s:=range set{out=append(out,s)};sort.Strings(out);return out}
func(f *flags)sortedIssues()[]Issue{out:=append([]Issue(nil),f.issues...);sort.Slice(out,func(i,j int)bool{if out[i].Code!=out[j].Code{return out[i].Code<out[j].Code};return out[i].Detail<out[j].Detail});return out}

func validID(s string)bool{return s!=""&&len(s)<=512&&!strings.ContainsAny(s," \t\r\n")}
func validDigest(s string)bool{if len(s)!=64{return false};_,e:=hex.DecodeString(s);return e==nil&&s==strings.ToLower(s)}
func canonicalDigest(v any)string{raw,e:=json.Marshal(v);if e!=nil{return ""};sum:=sha256.Sum256(raw);return hex.EncodeToString(sum[:])}
func boundaryComplete(b RecoveryBoundary)bool{return validID(b.IncidentID)&&validID(b.ClaimID)&&validID(b.SourceClusterID)&&validID(b.Database)&&validID(b.SourceEpoch)&&validID(b.TimelineID)&&validID(b.SnapshotID)&&validID(b.TerminalPosition)&&validDigest(b.ScopeDigest)&&validDigest(b.InventoryDigest)}
func boundaryDigest(b RecoveryBoundary)string{return canonicalDigest(b)}
func topicInventoryDigest(v []TopicInventory)string{
 c:=append([]TopicInventory(nil),v...)
 sort.Slice(c,func(i,j int)bool{if c[i].ClusterID!=c[j].ClusterID{return c[i].ClusterID<c[j].ClusterID};if c[i].Topic!=c[j].Topic{return c[i].Topic<c[j].Topic};return canonicalDigest(c[i])<canonicalDigest(c[j])})
 return canonicalDigest(c)
}
func recordsDigest(v []CapturedRecord)string{return canonicalDigest(v)}
func businessEventsDigest(v []BusinessEvent)string{
 c:=append([]BusinessEvent(nil),v...)
 sort.Slice(c,func(i,j int)bool{a,b:=c[i],c[j];if a.Topic!=b.Topic{return a.Topic<b.Topic};if a.Partition!=b.Partition{return a.Partition<b.Partition};if a.Offset!=b.Offset{return a.Offset<b.Offset};return canonicalDigest(a)<canonicalDigest(b)})
 return canonicalDigest(c)
}
func rangeManifestDigest(p PartitionCapture)string{p.ManifestSHA256="";p.Records=nil;return canonicalDigest(p)}
func partitionRangesDigest(v []PartitionCapture)string{
 c:=append([]PartitionCapture(nil),v...)
 sort.Slice(c,func(i,j int)bool{a,b:=c[i],c[j];if a.Topic!=b.Topic{return a.Topic<b.Topic};if a.Partition!=b.Partition{return a.Partition<b.Partition};if a.CaptureID!=b.CaptureID{return a.CaptureID<b.CaptureID};return canonicalDigest(a)<canonicalDigest(b)})
 for i:=range c{c[i].Records=nil}
 return canonicalDigest(c)
}

func DeriveFixtureProofDigests(in *Input)bool{
 if in==nil||in.Mode!=OfflineFixture||!boundaryComplete(in.Boundary){return false}
 in.BoundarySHA256=boundaryDigest(in.Boundary)
 in.TopicInventorySHA256=topicInventoryDigest(in.TopicInventory)
 for i:=range in.PartitionCaptures{
  p:=&in.PartitionCaptures[i]
  p.BoundarySHA256=in.BoundarySHA256
  p.RecordCount=len(p.Records)
  p.RecordsSHA256=recordsDigest(p.Records)
  p.ManifestSHA256=rangeManifestDigest(*p)
 }
 in.PartitionRangesSHA256=partitionRangesDigest(in.PartitionCaptures)
 in.BusinessEventsSHA256=businessEventsDigest(in.BusinessEvents)
 return true
}

func Evaluate(in Input)Result{
 out:=Result{
  Profile:"mon038-kafka-partition-capture-result-v1",Version:Version,
  TrialID:in.TrialID,CaptureID:in.CaptureID,GlobalOrderInferred:false,
  Partitions:[]PartitionSummary{},Issues:[]Issue{},ReasonCodes:[]string{},
 }
 f:=nf()
 if in.Profile!=Profile||in.Version!=Version{f.block("profile_version_invalid","unsupported input profile or version")}
 if !validID(in.TrialID)||!validID(in.CaptureID){f.block("capture_identity_invalid","trial or capture ID is invalid")}
 if in.Mode!=OfflineFixture&&in.Mode!=RuntimeCandidate{f.block("mode_invalid","unsupported evidence mode")}
 if in.EvaluatedAt.IsZero()||in.CaptureStartedAt.IsZero()||in.CaptureCompletedAt.IsZero(){f.hold("capture_time_missing","evaluation and capture timestamps are required")}
 if !in.CaptureStartedAt.IsZero()&&!in.CaptureCompletedAt.IsZero()&&in.CaptureCompletedAt.Before(in.CaptureStartedAt){f.block("capture_window_invalid","capture completion precedes capture start")}
 if !in.EvaluatedAt.IsZero()&&!in.CaptureCompletedAt.IsZero()&&in.CaptureCompletedAt.After(in.EvaluatedAt){f.block("capture_in_future","capture completed after evaluation time")}
 if !boundaryComplete(in.Boundary){f.hold("recovery_boundary_incomplete","common recovery boundary is incomplete")}else if !validDigest(in.BoundarySHA256){f.hold("recovery_boundary_digest_missing","boundary digest is absent or malformed")}else if in.BoundarySHA256!=boundaryDigest(in.Boundary){f.block("recovery_boundary_digest_mismatch","boundary digest does not match boundary fields")}
 if !in.ProducerFenced{f.hold("producer_not_fenced","partition vector is not closed while the source producer remains active")}
 if !validDigest(in.ProducerFenceProofSHA256){f.hold("producer_fence_proof_missing","producer fence receipt is not digest-bound")}
 if in.ProducerFencedAt.IsZero(){f.hold("producer_fence_time_missing","producer fence timestamp is missing")}else if !in.CaptureStartedAt.IsZero()&&in.ProducerFencedAt.After(in.CaptureStartedAt){f.block("producer_fenced_after_capture_started","fence must precede partition range capture")}
 if !in.TopicInventoryComplete||len(in.TopicInventory)==0{f.hold("topic_inventory_incomplete","complete topic/partition inventory is required")}
 if len(in.TopicInventory)>MaxTopics||len(in.PartitionCaptures)>MaxPartitions||len(in.BusinessEvents)>MaxRecords{f.block("input_limit_exceeded","topic, partition receipt or business-event count exceeds v1 limits");out.CaptureVerdict=f.verdict();out.CampaignVerdict=out.CaptureVerdict;out.Issues=f.sortedIssues();out.ReasonCodes=f.codes();return out}
 if !validDigest(in.TopicInventorySHA256){f.hold("topic_inventory_digest_missing","topic inventory digest is missing")}else if in.TopicInventorySHA256!=topicInventoryDigest(in.TopicInventory){f.block("topic_inventory_digest_mismatch","topic inventory content does not match its digest")}
 if !validDigest(in.PartitionRangesSHA256){f.hold("partition_range_digest_missing","partition range vector digest is absent")}else if in.PartitionRangesSHA256!=partitionRangesDigest(in.PartitionCaptures){f.block("partition_range_digest_mismatch","partition range vector differs from its digest")}
 if !validDigest(in.BusinessEventsSHA256){f.hold("business_event_digest_missing","business event set digest is absent")}else if in.BusinessEventsSHA256!=businessEventsDigest(in.BusinessEvents){f.block("business_event_digest_mismatch","business event set differs from its digest")}

 inventory:=map[string]TopicInventory{}
 kafkaClusters:=map[string]struct{}{}
 topicIDs:=map[string]string{}
 expectedPartitions:=0
 for _,t:=range in.TopicInventory{
  if !validID(t.ClusterID)||!validID(t.Topic)||!validID(t.TopicID)||t.PartitionCount<1||t.PartitionCount>MaxPartitions{f.block("topic_inventory_entry_invalid",t.Topic);continue}
  key:=t.ClusterID+"\x00"+t.Topic
  if _,ok:=inventory[key];ok{f.block("topic_inventory_duplicate",t.Topic);continue}
  inventory[key]=t
  kafkaClusters[t.ClusterID]=struct{}{}
  topicKey:=t.ClusterID+"\x00"+t.TopicID
  if previous,ok:=topicIDs[topicKey];ok&&previous!=t.Topic{f.block("duplicate_topic_id",t.Topic)}else{topicIDs[topicKey]=t.Topic}
  if expectedPartitions>MaxPartitions-t.PartitionCount { expectedPartitions=MaxPartitions+1 } else { expectedPartitions+=t.PartitionCount }
  if !validDigest(t.ConfigSHA256){f.hold("topic_config_digest_missing",t.Topic)}
  if t.CleanupPolicy!="delete"{
   if strings.Contains(t.CleanupPolicy,"compact"){f.block("compaction_not_supported_v1",t.Topic)}else{f.hold("cleanup_policy_not_proven_delete",t.Topic)}
  }
 }
 if len(kafkaClusters)>1{f.block("multiple_kafka_clusters_not_supported_v1","v1 requires one Kafka cluster per partition vector")}
 if expectedPartitions>MaxPartitions{f.block("partition_limit_exceeded","declared topic partition count exceeds v1 limit")}

 type position struct{topic string;partition int;offset int64}
 type partitionKey struct{topic string; partition int}
 capturedByPosition:=map[position]CapturedRecord{}
 rangeByPartition:=map[partitionKey]PartitionCapture{}
 recordTotal:=0
 for _,p:=range in.PartitionCaptures{
  t,ok:=inventory[p.ClusterID+"\x00"+p.Topic]
  if !ok{f.block("partition_for_unknown_topic",p.Topic);continue}
  if p.CaptureID!=in.CaptureID{f.block("partition_capture_id_mismatch",p.Topic)}
  if p.TopicID!=t.TopicID{f.block("topic_id_mismatch",p.Topic)}
  if p.BoundarySHA256!=in.BoundarySHA256||!validDigest(p.BoundarySHA256){f.block("partition_boundary_mismatch",p.Topic)}
  if p.Partition<0||p.Partition>=t.PartitionCount{f.block("partition_index_out_of_range",p.Topic)}
  key:=partitionKey{topic:p.Topic,partition:p.Partition}
  if _,exists:=rangeByPartition[key];exists{f.block("duplicate_partition_range",p.Topic)}
  rangeByPartition[key]=p
  if p.CleanupPolicy!="delete"{if strings.Contains(p.CleanupPolicy,"compact"){f.block("partition_compaction_enabled",p.Topic)}else{f.hold("partition_cleanup_policy_unknown",p.Topic)}}
  if p.IsolationLevel!="read_committed"{f.block("isolation_level_not_read_committed",p.Topic)}
  if p.StartOffset<0||p.EndOffsetExclusive<p.StartOffset{f.block("offset_range_invalid",p.Topic)}
  if p.LogStartOffsetAtStart>p.StartOffset||p.LogStartOffsetAtEnd>p.StartOffset{f.block("capture_range_truncated_by_log_start",p.Topic)}
  if p.LastStableOffsetBefore!=p.LastStableOffsetAfter{f.hold("last_stable_offset_not_stable",p.Topic)}
  if p.HighWatermarkBefore!=p.HighWatermarkAfter{f.hold("high_watermark_not_stable",p.Topic)}
  if p.LastStableOffsetAfter>p.HighWatermarkAfter{f.block("last_stable_offset_exceeds_high_watermark",p.Topic)}else if p.LastStableOffsetAfter<p.HighWatermarkAfter{f.hold("unresolved_transaction_at_capture_end",p.Topic)}
  if p.EndOffsetExclusive!=p.LastStableOffsetAfter{if p.EndOffsetExclusive>p.LastStableOffsetAfter{f.block("capture_end_beyond_lso",p.Topic)}else{f.hold("capture_stops_before_lso",p.Topic)}}
  if p.LeaderEpochBefore<0||p.LeaderEpochAfter<0{f.hold("leader_epoch_missing",p.Topic)}else if p.LeaderEpochBefore!=p.LeaderEpochAfter{f.hold("leader_epoch_changed_during_capture",p.Topic)}
  if !p.CaptureComplete{f.hold("partition_capture_incomplete",p.Topic)}
  if p.CaptureStartedAt.IsZero()||p.CaptureCompletedAt.IsZero(){f.hold("partition_capture_time_missing",p.Topic)}else{
   if p.CaptureCompletedAt.Before(p.CaptureStartedAt){f.block("partition_capture_window_invalid",p.Topic)}
   if !in.CaptureStartedAt.IsZero()&&p.CaptureStartedAt.Before(in.CaptureStartedAt){f.block("partition_capture_outside_window",p.Topic)}
   if !in.CaptureCompletedAt.IsZero()&&p.CaptureCompletedAt.After(in.CaptureCompletedAt){f.block("partition_capture_outside_window",p.Topic)}
  }
  if p.RecordCount!=len(p.Records){f.block("partition_record_count_mismatch",p.Topic)}
  if !validDigest(p.RecordsSHA256){f.hold("partition_record_digest_missing",p.Topic)}else if p.RecordsSHA256!=recordsDigest(p.Records){f.block("partition_record_digest_mismatch",p.Topic)}
  if !validDigest(p.ManifestSHA256){f.hold("partition_manifest_digest_missing",p.Topic)}else if p.ManifestSHA256!=rangeManifestDigest(p){f.block("partition_manifest_digest_mismatch",p.Topic)}
  if !validDigest(t.ConfigSHA256){/* already reported at inventory */}
  if len(p.Records)>MaxRecords-recordTotal{f.block("captured_record_limit_exceeded","captured record count exceeds v1 limit");recordTotal=MaxRecords+1;continue}
  lastOffset:=int64(-1)
  for _,r:=range p.Records{
   recordTotal++
   if r.Topic!=p.Topic||r.TopicID!=p.TopicID||r.Partition!=p.Partition{f.block("record_partition_identity_mismatch",p.Topic)}
   if r.Offset<p.StartOffset||r.Offset>=p.EndOffsetExclusive{f.block("record_outside_partition_range",p.Topic)}
   if r.Offset<=lastOffset{f.block("record_offsets_not_strictly_increasing",p.Topic)}
   lastOffset=r.Offset
   if !validDigest(r.PayloadSHA256){f.block("record_payload_digest_invalid",p.Topic)}
   if r.KeySHA256!=""&&!validDigest(r.KeySHA256){f.block("record_key_digest_invalid",p.Topic)}
   switch r.RecordClass{
   case "row_change","logical_message":
    if !validID(r.EventID)||!validID(r.TransactionID)||!validID(r.DataCollection){f.block("business_record_identity_missing",p.Topic)}
   case "transaction_metadata":
    if !validID(r.TransactionID){f.block("transaction_metadata_id_missing",p.Topic)}
   case "tombstone":if !validDigest(r.KeySHA256){f.block("tombstone_key_digest_missing",p.Topic)}
   case "heartbeat","opaque":
   default:f.block("record_class_invalid",p.Topic)
   }
   pos:=position{r.Topic,r.Partition,r.Offset}
   if _,exists:=capturedByPosition[pos];exists{f.block("duplicate_partition_offset",p.Topic)}
   capturedByPosition[pos]=r
  }
 }
 if recordTotal>MaxRecords{f.block("captured_record_limit_exceeded","captured record count exceeds v1 limit")}
 if expectedPartitions<=MaxPartitions { for _,t:=range inventory{
  for part:=0;part<t.PartitionCount;part++{
   pkey:=partitionKey{topic:t.Topic,partition:part}
   if _,ok:=rangeByPartition[pkey];!ok{f.hold("partition_capture_missing",t.Topic+":"+itoa(part))}
  }
 }}
 byBusinessPosition:=map[position]struct{}{}
 for _,e:=range in.BusinessEvents{
  if !validID(e.EventID)||!validID(e.TransactionID)||!validID(e.Topic)||!validID(e.TopicID)||e.Partition<0||e.Offset<0||!validDigest(e.PayloadSHA256){f.block("business_event_malformed",e.EventID);continue}
  pos:=position{e.Topic,e.Partition,e.Offset}
  if _,ok:=byBusinessPosition[pos];ok{f.block("duplicate_business_event_position",e.Topic);continue}
  byBusinessPosition[pos]=struct{}{}
  rec,ok:=capturedByPosition[pos]
  if !ok{f.block("business_event_not_in_partition_capture",e.EventID);continue}
  if rec.TopicID!=e.TopicID||rec.EventID!=e.EventID||rec.TransactionID!=e.TransactionID||rec.RecordClass!=e.RecordClass||rec.DataCollection!=e.DataCollection||rec.PayloadSHA256!=e.PayloadSHA256{f.block("business_event_capture_mismatch",e.EventID)}
 }
 for pos,rec:=range capturedByPosition{
  if rec.RecordClass=="row_change"||rec.RecordClass=="logical_message"{
   if _,ok:=byBusinessPosition[pos];!ok{f.block("captured_business_record_missing_from_event_set",rec.EventID)}
  }
 }
 out.TopicCount=len(inventory)
 out.PartitionCount=expectedPartitions
 out.VisibleRecordCount=recordTotal
 out.BusinessEventCount=len(in.BusinessEvents)
 for _,p:=range in.PartitionCaptures{
  width:=p.EndOffsetExclusive-p.StartOffset
  gapSlots:=int64(0)
  if width>=int64(p.RecordCount){gapSlots=width-int64(p.RecordCount)}
  out.OffsetGapsAccepted+=gapSlots
  out.Partitions=append(out.Partitions,PartitionSummary{Topic:p.Topic,Partition:p.Partition,StartOffset:p.StartOffset,EndOffsetExclusive:p.EndOffsetExclusive,LSO:p.LastStableOffsetAfter,Records:p.RecordCount,OffsetGapSlots:gapSlots,OffsetGapsObserved:gapSlots>0})
 }
 sort.Slice(out.Partitions,func(i,j int)bool{if out.Partitions[i].Topic!=out.Partitions[j].Topic{return out.Partitions[i].Topic<out.Partitions[j].Topic};return out.Partitions[i].Partition<out.Partitions[j].Partition})
 out.CaptureVerdict=f.verdict()
 out.CampaignVerdict=Indeterminate
 if in.Mode==OfflineFixture{f.hold("offline_fixture_cannot_qualify_live_capture","fixture-only evidence is not a live capture")}
 if in.Mode==RuntimeCandidate{f.hold("runtime_custody_adapter_missing","runtime evidence requires authenticated broker/capture/fence receipts")}
 out.CampaignVerdict=f.verdict()
 out.ProductionUsePermitted=false
 out.GlobalOrderInferred=false
 out.Issues=f.sortedIssues()
 out.ReasonCodes=f.codes()
 return out
}
func itoa(n int)string{if n==0{return "0"};buf:=[]byte{};for n>0{buf=append(buf,byte('0'+n%10));n/=10};for i,j:=0,len(buf)-1;i<j;i,j=i+1,j-1{buf[i],buf[j]=buf[j],buf[i]};return string(buf)}
