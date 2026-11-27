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
	ServerMajor uint16
	ConfiguredWALLevel,EffectiveWALLevel string
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
	SystemIdentifier string
	ServerMajor uint16
	TimelineID uint32
	SlotType string
	DatabaseOID uint32
	DatabaseName,SlotName,OutputPlugin string
	Temporary bool
	TwoPhase bool
	InvalidationReason,WALStatus string
	Failover,Synced,OnStandby bool
	RestartLSN,ConfirmedFlushLSN *LSN
	SafeWALSize *int64
	CatalogXmin,Xmin string
	StandbyReplayLSN *LSN
}
type DecodedEvent struct { ID,XID string; CommitLSN LSN; Ordinal uint32; PayloadDigest string }
type Transaction struct {
	XID string
	State string // committed, aborted, prepared, committed_prepared, rolled_back_prepared
	BeginLSN,CommitLSN,EndLSN LSN
	EventIDs []string
}
type Manifest struct {
	ServerMajor uint16
	SystemIdentifier,DatabaseName,OutputPlugin string
	SourceEpoch string
	TimelineID uint32
	SlotName string
	DatabaseOID uint32
	PluginProfileDigest,SchemaProfileDigest string
	StartExclusiveLSN,TerminalLSN,SnapshotConsistentPoint LSN
	SnapshotDigest string
	Events []DecodedEvent
	Transactions []Transaction
	Complete bool
	DroppedEvents,QueueOverflowCount uint64
	EvidenceThroughLSN,AcknowledgementRequestedLSN,AcknowledgementSentLSN LSN
	Witnessed bool
	ManifestDigest string
}
type Result struct { Decision Decision; Reason string; Checked []string }
func out(d Decision,s string,c ...string)Result{return Result{d,s,append([]string(nil),c...)}}
func present(xs ...string)bool{for _,x:=range xs{if strings.TrimSpace(x)==""{return false}};return true}
func profileOK(p Profile)bool{return (p.ServerMajor==18||p.ServerMajor==19)&&present(p.SystemIdentifier,p.DatabaseName,p.SlotName,p.OutputPlugin,p.PluginProfileDigest,p.SchemaProfileDigest,p.SourceEpoch,p.SnapshotName,p.SnapshotDigest)&&p.DatabaseOID>0&&p.TimelineID>0&&p.ConsistentPoint!=nil}
func validateWALLevel(p Profile)(Decision,string){
	if p.ServerMajor==18 {
		if p.ConfiguredWALLevel!="logical"||p.EffectiveWALLevel!="logical"{return Denied,"PostgreSQL 18 requires configured and effective wal_level=logical for this profile"}
		return Qualified,""
	}
	if p.ServerMajor==19 {
		if p.EffectiveWALLevel!="logical"{return Denied,"PostgreSQL 19 effective_wal_level is not logical"}
		if p.ConfiguredWALLevel!="logical"&&p.ConfiguredWALLevel!="replica"{return Indeterminate,"PostgreSQL 19 configured wal_level is outside the versioned profile"}
		return Qualified,""
	}
	return Indeterminate,"unsupported PostgreSQL major version"
}
func slotMatches(p Profile,s SlotState)bool{return s.SlotType=="logical"&&s.SystemIdentifier==p.SystemIdentifier&&s.ServerMajor==p.ServerMajor&&s.TimelineID==p.TimelineID&&s.DatabaseOID==p.DatabaseOID&&s.DatabaseName==p.DatabaseName&&s.SlotName==p.SlotName&&s.OutputPlugin==p.OutputPlugin}
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
	if !profileOK(p){return out(Indeterminate,"source profile incomplete or PostgreSQL major version unsupported",c...)}
	if d,r:=validateWALLevel(p);d!=Qualified{return out(d,r,c...)}
	if !slotMatches(p,s){return out(Denied,"slot identity/database/plugin does not match source profile",c...)}
	if d,r:=slotHealth(s);d!=Qualified{return out(d,r,c...)}
	if point!=*p.ConsistentPoint||digest!=p.SnapshotDigest||!present(digest){return out(Denied,"exported snapshot does not bind exact slot consistent point/profile",c...)}
	if s.ConfirmedFlushLSN!=nil&&*s.ConfirmedFlushLSN>*p.ConsistentPoint{return out(Indeterminate,"slot acknowledgement is beyond initial snapshot boundary; reconcile with a durable stream manifest",c...)}
	return out(Qualified,"initial snapshot and logical slot are bound to the same source profile",append(c,"slot-retention-health")...)
}
func transactionMembersConsistent(m Manifest)(Decision,string){
	txs:=map[string]Transaction{}
	events:=map[string]DecodedEvent{}
	for _,tx:=range m.Transactions{
		if !present(tx.XID){return Indeterminate,"transaction envelope missing XID"}
		if _,ok:=txs[tx.XID];ok{return Divergent,"duplicate transaction envelope: "+tx.XID}
		switch tx.State{
		case "committed","aborted":
		case "prepared","committed_prepared","rolled_back_prepared":return Indeterminate,"two-phase transaction state is not qualified: "+tx.XID
		default:return Indeterminate,"unknown transaction terminal state: "+tx.XID
		}
		txs[tx.XID]=tx
	}
	for _,e:=range m.Events{
		if prior,ok:=events[e.ID];ok{if prior!=e{return Divergent,"duplicate event ID has conflicting transaction fields: "+e.ID};continue}
		events[e.ID]=e
	}
	for xid,tx:=range txs{
		members:=map[string]bool{}
		for _,id:=range tx.EventIDs{
			if members[id]{return Divergent,"transaction repeats event membership: "+id}
			members[id]=true
			e,ok:=events[id]
			if !ok||e.XID!=xid||e.CommitLSN!=tx.CommitLSN{return Indeterminate,"transaction/event membership or commit boundary mismatch: "+id}
		}
		for id,e:=range events{if e.XID==xid&&!members[id]{return Indeterminate,"event missing from transaction envelope: "+id}}
	}
	for id,e:=range events{tx,ok:=txs[e.XID];if !ok{return Indeterminate,"event has no transaction envelope: "+id};if tx.State!="committed"||tx.CommitLSN!=e.CommitLSN{return Indeterminate,"event is not attached to a committed transaction boundary: "+id}}
	return Qualified,""
}
func ValidateCapture(p Profile,s SlotState,m Manifest)Result{
	c:=[]string{"source-slot-profile","capture-range","decoded-event-order","durable-evidence-before-ack"}
	if !profileOK(p){return out(Indeterminate,"source profile incomplete or PostgreSQL major version unsupported",c...)}
	if d,r:=validateWALLevel(p);d!=Qualified{return out(d,r,c...)}
	if !slotMatches(p,s)||m.ServerMajor!=p.ServerMajor||m.SystemIdentifier!=p.SystemIdentifier||m.DatabaseName!=p.DatabaseName||m.OutputPlugin!=p.OutputPlugin||m.SourceEpoch!=p.SourceEpoch||m.TimelineID!=p.TimelineID||m.SlotName!=p.SlotName||m.DatabaseOID!=p.DatabaseOID||m.PluginProfileDigest!=p.PluginProfileDigest||m.SchemaProfileDigest!=p.SchemaProfileDigest{return out(Denied,"capture server/cluster/database/slot/plugin/timeline profile mismatch",c...)}
	if d,r:=slotHealth(s);d!=Qualified{return out(d,r,c...)}
	if s.TwoPhase { return out(Indeterminate,"two-phase decoding is not qualified by this capture profile",c...) }
	if m.SnapshotConsistentPoint!=*p.ConsistentPoint||m.SnapshotDigest!=p.SnapshotDigest{return out(Denied,"capture snapshot anchor mismatch",c...)}
	if !m.Complete||m.DroppedEvents>0||m.QueueOverflowCount>0{return out(Indeterminate,"capture incomplete or reports dropped/overflowed events",c...)}
	if !present(m.ManifestDigest)||!m.Witnessed{return out(Indeterminate,"capture manifest is not durably sealed/witnessed by the upstream custody boundary",c...)}
	if m.TerminalLSN<m.StartExclusiveLSN{return out(Denied,"terminal LSN precedes capture start",c...)}
	if m.EvidenceThroughLSN!=m.TerminalLSN{return out(Indeterminate,"durable evidence watermark does not equal terminal LSN",c...)}
	if m.AcknowledgementRequestedLSN>m.EvidenceThroughLSN{return out(Denied,"requested acknowledgement advances beyond durable evidence",c...)}
	if m.AcknowledgementSentLSN>m.EvidenceThroughLSN{return out(Denied,"actual acknowledgement sent advances beyond durable evidence",c...)}
	if s.ConfirmedFlushLSN!=nil&&*s.ConfirmedFlushLSN>m.EvidenceThroughLSN{return out(Denied,"observed confirmed_flush_lsn exceeds durable evidence",c...)}
	if *s.RestartLSN>m.StartExclusiveLSN{return out(Indeterminate,"slot restart_lsn is later than required capture start",c...)}
	txByXID:=map[string]Transaction{}
	commitOwner:=map[LSN]string{}
	for _,tx:=range m.Transactions {
		if !present(tx.XID) { return out(Indeterminate,"transaction envelope missing XID",c...) }
		if _,ok:=txByXID[tx.XID];ok { return out(Divergent,"duplicate transaction envelope: "+tx.XID,c...) }
		if tx.BeginLSN>tx.CommitLSN||tx.CommitLSN>tx.EndLSN||tx.EndLSN>m.TerminalLSN {
			return out(Denied,"transaction envelope has impossible begin/commit/end LSN ordering: "+tx.XID,c...)
		}
		switch tx.State {
		case "committed","aborted":
		case "prepared","committed_prepared","rolled_back_prepared":
			return out(Indeterminate,"two-phase transaction state is outside the qualified v0.1 envelope: "+tx.XID,c...)
		default:
			return out(Indeterminate,"unknown or incomplete transaction terminal state: "+tx.XID,c...)
		}
		if prior,ok:=commitOwner[tx.CommitLSN];ok&&prior!=tx.XID{return out(Divergent,"different XIDs claim the same commit record LSN",c...)}
		commitOwner[tx.CommitLSN]=tx.XID
		txByXID[tx.XID]=tx
	}
	seen:=map[string]DecodedEvent{}
	eventOwner:=map[string]string{}
	ordinalByXID:=map[string][]uint32{}
	var prev LSN
	for _,e:=range m.Events{
		if !present(e.ID,e.XID,e.PayloadDigest)||e.Ordinal==0{return out(Indeterminate,"decoded event identity/ordinal incomplete",c...)}
		if prior,ok:=seen[e.ID];ok{if prior!=e{return out(Divergent,"same event identity has conflicting fields or payload: "+e.ID,c...)};continue}
		if e.CommitLSN<=m.StartExclusiveLSN||e.CommitLSN>m.TerminalLSN{return out(Denied,"decoded event outside declared exclusive-start/terminal interval: "+e.ID,c...)}
		if len(seen)>0&&e.CommitLSN<prev{return out(Denied,"decoded event LSN order regressed",c...)}
		tx,ok:=txByXID[e.XID]
		if !ok{return out(Indeterminate,"decoded event has no complete transaction envelope: "+e.ID,c...)}
		if tx.State!="committed"||tx.CommitLSN!=e.CommitLSN{return out(Denied,"row event is not bound to a committed transaction boundary: "+e.ID,c...)}
		if owner,ok:=eventOwner[e.ID];ok&&owner!=e.XID{return out(Divergent,"event ID attributed to multiple transactions: "+e.ID,c...)}
		eventOwner[e.ID]=e.XID
		seen[e.ID]=e
		ordinalByXID[e.XID]=append(ordinalByXID[e.XID],e.Ordinal)
		prev=e.CommitLSN
	}
	for xid,tx:=range txByXID {
		ids:=map[string]bool{}
		for _,id:=range tx.EventIDs {
			if ids[id]{return out(Divergent,"transaction envelope repeats event ID: "+id,c...)}
			ids[id]=true
			e,ok:=seen[id]
			if !ok||e.XID!=xid{return out(Indeterminate,"transaction envelope/event membership mismatch: "+id,c...)}
		}
		for id,e:=range seen {
			if e.XID==xid&&!ids[id]{return out(Indeterminate,"decoded event omitted from transaction envelope: "+id,c...)}
		}
		ords:=ordinalByXID[xid]
		for i:=1;i<len(ords);i++{if ords[i]<=ords[i-1]{return out(Denied,"per-transaction message ordinal is not strictly increasing: "+xid,c...)}}
		if tx.State=="aborted"&&len(ids)>0{return out(Denied,"aborted transaction envelope contains row events: "+xid,c...)}
	}
	for _,e:=range m.Events{if _,ok:=txByXID[e.XID];!ok{return out(Indeterminate,"decoded event has no transaction envelope: "+e.ID,c...)}}
	return out(Qualified,"bounded logical-decoding interval is sealed with complete committed transaction envelopes; acknowledgement does not exceed durable evidence",c...)
}
func sameTransaction(a,b Transaction)bool{
	if a.XID!=b.XID||a.State!=b.State||a.BeginLSN!=b.BeginLSN||a.CommitLSN!=b.CommitLSN||a.EndLSN!=b.EndLSN||len(a.EventIDs)!=len(b.EventIDs){return false}
	for i:=range a.EventIDs{if a.EventIDs[i]!=b.EventIDs[i]{return false}}
	return true
}
func ValidateReplay(old,next Manifest)Result{
	c:=[]string{"overlap-replay-reconciliation","transaction-envelope-reconciliation","durable-watermark-and-acknowledgement"}
	if d,r:=transactionMembersConsistent(old);d!=Qualified{return out(d,"old manifest: "+r,c...)}
	if d,r:=transactionMembersConsistent(next);d!=Qualified{return out(d,"new manifest: "+r,c...)}
	if old.ServerMajor!=next.ServerMajor||old.SystemIdentifier!=next.SystemIdentifier||old.DatabaseName!=next.DatabaseName||old.OutputPlugin!=next.OutputPlugin||old.SourceEpoch!=next.SourceEpoch||old.TimelineID!=next.TimelineID||old.SlotName!=next.SlotName||old.DatabaseOID!=next.DatabaseOID||old.PluginProfileDigest!=next.PluginProfileDigest||old.SchemaProfileDigest!=next.SchemaProfileDigest{return out(Divergent,"capture server/cluster/database/slot/timeline/profile changed without handoff",c...)}
	if !old.Complete||!old.Witnessed||!present(old.ManifestDigest)||!next.Complete||!next.Witnessed||!present(next.ManifestDigest){return out(Indeterminate,"old or new replay manifest lacks completeness/seal evidence",c...)}
	if next.StartExclusiveLSN>old.EvidenceThroughLSN{return out(Indeterminate,"new capture starts after prior durable evidence; unaccounted range gap",c...)}
	if next.EvidenceThroughLSN!=next.TerminalLSN{return out(Indeterminate,"replay evidence watermark does not equal terminal LSN",c...)}
	if next.AcknowledgementRequestedLSN>next.EvidenceThroughLSN||next.AcknowledgementSentLSN>next.EvidenceThroughLSN{return out(Denied,"replay request/sent acknowledgement advances beyond durable evidence",c...)}
	if next.TerminalLSN<old.TerminalLSN{return out(Indeterminate,"replay capture terminal regressed; await terminal watermark",c...)}
	if next.EvidenceThroughLSN!=next.TerminalLSN{return out(Indeterminate,"replay evidence watermark does not equal terminal LSN",c...)}
	if next.AcknowledgementRequestedLSN>next.EvidenceThroughLSN{return out(Denied,"replay acknowledgement advances beyond durable evidence",c...)}
	prior:=map[string]DecodedEvent{};for _,e:=range old.Events{if p,ok:=prior[e.ID];ok&&p!=e{return out(Divergent,"old manifest contains conflicting duplicate event identity: "+e.ID,c...)};prior[e.ID]=e}
	for _,e:=range next.Events{if p,ok:=prior[e.ID];ok&&p!=e{return out(Divergent,"replayed event conflicts with prior durable evidence: "+e.ID,c...)}}
	priorTx:=map[string]Transaction{};for _,tx:=range old.Transactions{if p,ok:=priorTx[tx.XID];ok&&!sameTransaction(p,tx){return out(Divergent,"old manifest contains conflicting transaction envelope: "+tx.XID,c...)};priorTx[tx.XID]=tx}
	for _,tx:=range next.Transactions{if p,ok:=priorTx[tx.XID];ok&&!sameTransaction(p,tx){return out(Divergent,"replayed transaction envelope conflicts with prior durable evidence: "+tx.XID,c...)}}
	return out(Qualified,"replay overlap reconciles exact event and transaction identities; acknowledgement remains behind durable evidence",c...)
}
type Failover struct{
	Prior,Next Profile
	StandbySlot SlotState
	OldPrimaryFenced bool
	TimelineHistoryDigest,HandoffManifestDigest string
	TimelineHistoryWitnessed bool
	TimelineHistoryParent uint32
	ForkLSN LSN
	HandoffWitnessed bool
}
func ValidateFailover(f Failover,durable LSN)Result{
	c:=[]string{"same-cluster-database-slot","timeline-transition","verified-parent-timeline-fork","standby-slot-ready","old-primary-fenced","handoff-sealed"}
	if !profileOK(f.Prior)||!profileOK(f.Next){return out(Indeterminate,"prior or promoted source profile incomplete",c...)}
	if d,r:=validateWALLevel(f.Prior);d!=Qualified{return out(d,"prior WAL level: "+r,c...)}
	if d,r:=validateWALLevel(f.Next);d!=Qualified{return out(d,"next WAL level: "+r,c...)}
	if f.Prior.SystemIdentifier!=f.Next.SystemIdentifier||f.Prior.DatabaseOID!=f.Next.DatabaseOID||f.Prior.DatabaseName!=f.Next.DatabaseName||f.Prior.SlotName!=f.Next.SlotName||f.Prior.OutputPlugin!=f.Next.OutputPlugin{return out(Divergent,"failover changes cluster/database/slot/plugin identity",c...)}
	if f.Prior.ServerMajor!=f.Next.ServerMajor{return out(Denied,"physical failover changes PostgreSQL major version",c...)}
	if f.Prior.TimelineID==f.Next.TimelineID||f.Next.TimelineID==0||f.Next.SourceEpoch==f.Prior.SourceEpoch{return out(Denied,"failover must create new timeline and source epoch",c...)}
	if !present(f.TimelineHistoryDigest)||!f.TimelineHistoryWitnessed||f.TimelineHistoryParent!=f.Prior.TimelineID||f.ForkLSN==0{return out(Indeterminate,"verified timeline history does not bind the prior timeline and fork point",c...)}
	if durable>f.ForkLSN{return out(Denied,"durable evidence lies beyond the promoted standby timeline fork; history is not shared",c...)}
	if f.Next.ConsistentPoint==nil||*f.Next.ConsistentPoint<f.ForkLSN{return out(Indeterminate,"promoted source consistent point precedes the authenticated timeline fork",c...)}
	s:=f.StandbySlot
	if s.SlotType!="logical"||s.SystemIdentifier!=f.Prior.SystemIdentifier||s.ServerMajor!=f.Next.ServerMajor||s.TimelineID!=f.Next.TimelineID||s.DatabaseOID!=f.Prior.DatabaseOID||s.DatabaseName!=f.Prior.DatabaseName||s.SlotName!=f.Prior.SlotName||s.OutputPlugin!=f.Prior.OutputPlugin{return out(Denied,"standby cluster/timeline/database/name/plugin identity mismatch",c...)}
	if !s.OnStandby||!s.Failover||!s.Synced||s.Temporary||s.TwoPhase||s.InvalidationReason!=""||(s.WALStatus!="reserved"&&s.WALStatus!="extended"){return out(Denied,"standby logical slot is not persistent, synced, failover-enabled and valid for the supported transaction profile",c...)}
	if s.RestartLSN==nil||s.StandbyReplayLSN==nil{return out(Indeterminate,"standby slot restart/replay position unavailable",c...)}
	if *s.StandbyReplayLSN<f.ForkLSN{return out(Indeterminate,"standby has not replayed through the verified timeline fork point",c...)}
	if *s.RestartLSN>*s.StandbyReplayLSN{return out(Indeterminate,"standby restart_lsn is ahead of its replay position",c...)}
	if s.ConfirmedFlushLSN!=nil&&*s.ConfirmedFlushLSN>durable{return out(Denied,"standby logical slot acknowledged beyond the shared durable evidence boundary",c...)}
	if !f.OldPrimaryFenced{return out(Indeterminate,"old primary not independently fenced after promotion",c...)}
	if !f.HandoffWitnessed||!present(f.HandoffManifestDigest){return out(Indeterminate,"timeline/source-epoch handoff not sealed and independently witnessed",c...)}
	return out(Qualified,"failover lineage and slot continuity supported through the authenticated shared fork boundary",c...)
}
