package cdccheck

import (
	"fmt"
	"strconv"
	"strings"
)

type Decision string
const (
	Qualified Decision = "qualified"
	Denied Decision = "denied"
	Indeterminate Decision = "indeterminate"
	Divergent Decision = "divergent"
)

type LSN uint64
func ParseLSN(s string) (LSN,error) {
	p:=strings.Split(s,"/")
	if len(p)!=2||p[0]==""||p[1]=="" { return 0,fmt.Errorf("invalid PostgreSQL LSN %q",s) }
	hi,e:=strconv.ParseUint(p[0],16,32);if e!=nil{return 0,e}
	lo,e:=strconv.ParseUint(p[1],16,32);if e!=nil{return 0,e}
	return LSN((hi<<32)|lo),nil
}
func (l LSN) String() string { return fmt.Sprintf("%X/%X",uint64(l)>>32,uint64(l)&0xffffffff) }

type Profile struct {
	SystemIdentifier string
	DatabaseOID uint32
	DatabaseName,SlotName,OutputPlugin string
	PluginProfileDigest,SchemaProfileDigest string
	SourceEpoch string
	TimelineID uint32
	ConsistentPoint *LSN
	SnapshotName,SnapshotDigest string
}
type SlotState struct {
	SlotType string
	DatabaseOID uint32
	DatabaseName,SlotName,OutputPlugin string
	Temporary bool
	InvalidationReason,WALStatus string
	Failover,Synced,OnStandby bool
	RestartLSN,ConfirmedFlushLSN *LSN
	SafeWALSize *int64
	CatalogXmin,Xmin string
	StandbyReplayLSN *LSN
}
type DecodedEvent struct { ID,XID string; CommitLSN LSN; Ordinal uint32; PayloadDigest string }
type Manifest struct {
	SourceEpoch string
	TimelineID uint32
	SlotName string
	DatabaseOID uint32
	PluginProfileDigest,SchemaProfileDigest string
	StartExclusiveLSN,TerminalLSN,SnapshotConsistentPoint LSN
	SnapshotDigest string
	Events []DecodedEvent
	Complete bool
	DroppedEvents,QueueOverflowCount uint64
	EvidenceThroughLSN,AcknowledgementRequestedLSN LSN
	Witnessed bool
	ManifestDigest string
}
type Result struct { Decision Decision; Reason string; Checked []string }
func out(d Decision,s string,c ...string)Result{return Result{d,s,append([]string(nil),c...)}}
func present(xs ...string)bool{for _,x:=range xs{if strings.TrimSpace(x)==""{return false}};return true}
func profileOK(p Profile)bool{return present(p.SystemIdentifier,p.DatabaseName,p.SlotName,p.OutputPlugin,p.PluginProfileDigest,p.SchemaProfileDigest,p.SourceEpoch,p.SnapshotName,p.SnapshotDigest)&&p.DatabaseOID>0&&p.TimelineID>0&&p.ConsistentPoint!=nil}
func slotMatches(p Profile,s SlotState)bool{return s.SlotType=="logical"&&s.DatabaseOID==p.DatabaseOID&&s.DatabaseName==p.DatabaseName&&s.SlotName==p.SlotName&&s.OutputPlugin==p.OutputPlugin}
func slotHealth(s SlotState)(Decision,string){
	if s.SlotType!="logical"{return Denied,"slot is not logical"}
	if s.Temporary{return Denied,"temporary logical slot is not durable"}
	if s.InvalidationReason!=""{return Denied,"logical slot invalidated: "+s.InvalidationReason}
	switch s.WALStatus{case "lost":return Denied,"required WAL has been lost";case "unreserved":return Indeterminate,"required WAL is unreserved and at risk";case "reserved","extended":default:return Indeterminate,"unknown WAL retention state"}
	if s.RestartLSN==nil{return Indeterminate,"restart_lsn is uninitialized/unknown"}
	return Qualified,""
}
func ValidateInitialSnapshot(p Profile,s SlotState,point LSN,digest string)Result{
	c:=[]string{"cluster-database-slot-plugin-identity","snapshot-consistent-point"}
	if !profileOK(p){return out(Indeterminate,"source profile incomplete",c...)}
	if !slotMatches(p,s){return out(Denied,"slot identity/database/plugin does not match source profile",c...)}
	if d,r:=slotHealth(s);d!=Qualified{return out(d,r,c...)}
	if point!=*p.ConsistentPoint||digest!=p.SnapshotDigest||!present(digest){return out(Denied,"exported snapshot does not bind exact slot consistent point/profile",c...)}
	if s.ConfirmedFlushLSN!=nil&&*s.ConfirmedFlushLSN>*p.ConsistentPoint{return out(Indeterminate,"slot acknowledgement is beyond initial snapshot boundary; reconcile with a durable stream manifest",c...)}
	return out(Qualified,"initial snapshot and logical slot are bound to the same source profile",append(c,"slot-retention-health")...)
}
func ValidateCapture(p Profile,s SlotState,m Manifest)Result{
	c:=[]string{"source-slot-profile","capture-range","decoded-event-order","durable-evidence-before-ack"}
	if !profileOK(p){return out(Indeterminate,"source profile incomplete",c...)}
	if !slotMatches(p,s)||m.SourceEpoch!=p.SourceEpoch||m.TimelineID!=p.TimelineID||m.SlotName!=p.SlotName||m.DatabaseOID!=p.DatabaseOID||m.PluginProfileDigest!=p.PluginProfileDigest||m.SchemaProfileDigest!=p.SchemaProfileDigest{return out(Denied,"capture source/slot/plugin/timeline profile mismatch",c...)}
	if d,r:=slotHealth(s);d!=Qualified{return out(d,r,c...)}
	if m.SnapshotConsistentPoint!=*p.ConsistentPoint||m.SnapshotDigest!=p.SnapshotDigest{return out(Denied,"capture snapshot anchor mismatch",c...)}
	if !m.Complete||m.DroppedEvents>0||m.QueueOverflowCount>0{return out(Indeterminate,"capture incomplete or reports dropped/overflowed events",c...)}
	if !present(m.ManifestDigest)||!m.Witnessed{return out(Indeterminate,"capture manifest is not durably sealed/witnessed by the upstream custody boundary",c...)}
	if m.TerminalLSN<m.StartExclusiveLSN{return out(Denied,"terminal LSN precedes capture start",c...)}
	if m.EvidenceThroughLSN!=m.TerminalLSN{return out(Indeterminate,"durable evidence watermark does not equal terminal LSN",c...)}
	if m.AcknowledgementRequestedLSN>m.EvidenceThroughLSN{return out(Denied,"requested acknowledgement advances beyond durable evidence",c...)}
	if s.ConfirmedFlushLSN!=nil&&*s.ConfirmedFlushLSN>m.EvidenceThroughLSN{return out(Denied,"observed confirmed_flush_lsn exceeds durable evidence",c...)}
	if *s.RestartLSN>m.StartExclusiveLSN{return out(Indeterminate,"slot restart_lsn is later than required capture start",c...)}
	seen:=map[string]DecodedEvent{}
	var prev LSN;var prevXID string;var prevOrdinal uint32
	for i,e:=range m.Events{
		if !present(e.ID,e.XID,e.PayloadDigest){return out(Indeterminate,"decoded event identity incomplete",c...)}
		if prior,ok:=seen[e.ID];ok{if prior!=e{return out(Divergent,"same event identity has conflicting fields or payload: "+e.ID,c...)};continue}
		if e.CommitLSN<=m.StartExclusiveLSN||e.CommitLSN>m.TerminalLSN{return out(Denied,"decoded event outside declared exclusive-start/terminal interval: "+e.ID,c...)}
		if i>0&&e.CommitLSN<prev{return out(Denied,"decoded event LSN order regressed",c...)}
		if i>0&&e.CommitLSN==prev&&e.XID==prevXID&&e.Ordinal<=prevOrdinal{return out(Denied,"per-transaction message ordinal is not strictly increasing",c...)}
		seen[e.ID]=e;prev=e.CommitLSN;prevXID=e.XID;prevOrdinal=e.Ordinal
	}
	return out(Qualified,"bounded logical-decoding interval is sealed; acknowledgement does not exceed durable evidence",c...)
}
func ValidateReplay(old,next Manifest)Result{
	c:=[]string{"overlap-replay-reconciliation","no-unaccounted-range-gap"}
	if old.SourceEpoch!=next.SourceEpoch||old.TimelineID!=next.TimelineID||old.SlotName!=next.SlotName||old.DatabaseOID!=next.DatabaseOID||old.PluginProfileDigest!=next.PluginProfileDigest||old.SchemaProfileDigest!=next.SchemaProfileDigest{return out(Divergent,"capture source epoch/slot/timeline/profile changed without handoff",c...)}
	if next.StartExclusiveLSN>old.EvidenceThroughLSN{return out(Indeterminate,"new capture starts after prior durable evidence; unaccounted range gap",c...)}
	prior:=map[string]DecodedEvent{};for _,e:=range old.Events{prior[e.ID]=e}
	for _,e:=range next.Events{if p,ok:=prior[e.ID];ok&&p!=e{return out(Divergent,"replayed event conflicts with prior durable evidence: "+e.ID,c...)}}
	if next.TerminalLSN<old.TerminalLSN{return out(Indeterminate,"replay capture terminal regressed; await terminal watermark",c...)}
	return out(Qualified,"replay overlap matches prior evidence; exact duplicates are idempotently reconcilable",c...)
}
type Failover struct{
	Prior,Next Profile
	StandbySlot SlotState
	OldPrimaryFenced bool
	TimelineHistoryDigest,HandoffManifestDigest string
	HandoffWitnessed bool
}
func ValidateFailover(f Failover,durable LSN)Result{
	c:=[]string{"same-cluster-database-slot","timeline-transition","standby-slot-ready","old-primary-fenced","handoff-sealed"}
	if f.Prior.SystemIdentifier!=f.Next.SystemIdentifier||f.Prior.DatabaseOID!=f.Next.DatabaseOID||f.Prior.DatabaseName!=f.Next.DatabaseName||f.Prior.SlotName!=f.Next.SlotName||f.Prior.OutputPlugin!=f.Next.OutputPlugin{return out(Divergent,"failover changes cluster/database/slot/plugin identity",c...)}
	if f.Prior.TimelineID==f.Next.TimelineID||f.Next.TimelineID==0||f.Next.SourceEpoch==f.Prior.SourceEpoch{return out(Denied,"failover must create new timeline and source epoch",c...)}
	if !present(f.TimelineHistoryDigest){return out(Indeterminate,"timeline history digest missing",c...)}
	s:=f.StandbySlot
	if s.SlotType!="logical"||s.DatabaseOID!=f.Prior.DatabaseOID||s.DatabaseName!=f.Prior.DatabaseName||s.SlotName!=f.Prior.SlotName||s.OutputPlugin!=f.Prior.OutputPlugin{return out(Denied,"standby slot database/name/plugin identity mismatch",c...)}
	if !s.OnStandby||!s.Failover||!s.Synced||s.Temporary||s.InvalidationReason!=""||(s.WALStatus!="reserved"&&s.WALStatus!="extended"){return out(Denied,"standby logical slot is not persistent, synced, failover-enabled and valid",c...)}
	if s.RestartLSN==nil||*s.RestartLSN>durable{return out(Indeterminate,"standby slot does not retain required durable evidence range",c...)}
	if s.ConfirmedFlushLSN!=nil&&*s.ConfirmedFlushLSN>durable{return out(Denied,"standby acknowledgement exceeds durable evidence",c...)}
	if s.StandbyReplayLSN==nil||*s.StandbyReplayLSN<durable{return out(Indeterminate,"standby has not received/replayed through durable evidence boundary",c...)}
	if !f.OldPrimaryFenced{return out(Indeterminate,"old primary not independently fenced after promotion",c...)}
	if !f.HandoffWitnessed||!present(f.HandoffManifestDigest){return out(Indeterminate,"timeline/source-epoch handoff not sealed and independently witnessed",c...)}
	return out(Qualified,"failover lineage and slot continuity supported for declared boundary",c...)
}
