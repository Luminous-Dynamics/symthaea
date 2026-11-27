package cdccheck

import (
	"fmt"
	"testing"
)
func fixtureN(n int)(Profile,SlotState,Manifest){
	cp:=LSN(0x1000);restart:=LSN(0x800);flush:=cp
	p:=Profile{ServerMajor:18,ConfiguredWALLevel:"logical",EffectiveWALLevel:"logical",SystemIdentifier:"sys-bench",DatabaseOID:77,DatabaseName:"bench",SlotName:"bench_slot",OutputPlugin:"pgoutput",PluginProfileDigest:"plugin-v1",SchemaProfileDigest:"schema-v1",SourceEpoch:"bench-epoch-1",TimelineID:1,ConsistentPoint:&cp,SnapshotName:"bench-snapshot",SnapshotDigest:"snapshot-digest"}
	s:=SlotState{SystemIdentifier:"sys-bench",ServerMajor:18,TimelineID:1,SlotType:"logical",DatabaseOID:77,DatabaseName:"bench",SlotName:"bench_slot",OutputPlugin:"pgoutput",WALStatus:"reserved",RestartLSN:&restart,ConfirmedFlushLSN:&flush}
	terminal:=cp+LSN(n)+0x10
	m:=Manifest{ServerMajor:18,SystemIdentifier:p.SystemIdentifier,DatabaseName:p.DatabaseName,OutputPlugin:p.OutputPlugin,SourceEpoch:p.SourceEpoch,TimelineID:1,SlotName:p.SlotName,DatabaseOID:p.DatabaseOID,PluginProfileDigest:p.PluginProfileDigest,SchemaProfileDigest:p.SchemaProfileDigest,StartExclusiveLSN:cp,TerminalLSN:terminal,SnapshotConsistentPoint:cp,SnapshotDigest:p.SnapshotDigest,Complete:true,EvidenceThroughLSN:terminal,AcknowledgementRequestedLSN:terminal,AcknowledgementSentLSN:terminal,Witnessed:true,ManifestDigest:"bench-manifest"}
	m.Events=make([]DecodedEvent,n)
	for i:=range m.Events{
		id:=fmt.Sprintf("event-%08d",i);xid:=fmt.Sprintf("xid-%08d",i);commit:=cp+LSN(i)+1
		m.Events[i]=DecodedEvent{ID:id,XID:xid,CommitLSN:commit,Ordinal:1,PayloadDigest:fmt.Sprintf("digest-%08d",i)}
		m.Transactions=append(m.Transactions,Transaction{XID:xid,State:"committed",BeginLSN:commit-1,CommitLSN:commit,EndLSN:commit+1,EventIDs:[]string{id}})
	}
	return p,s,m
}
func BenchmarkValidateCapture(b *testing.B){for _,n:=range []int{10,100,1000,10000,100000}{b.Run(fmt.Sprintf("events_%d",n),func(b *testing.B){p,s,m:=fixtureN(n);b.ReportAllocs();b.ResetTimer();for i:=0;i<b.N;i++{r:=ValidateCapture(p,s,m);if r.Decision!=Qualified{b.Fatalf("%s: %s",r.Decision,r.Reason)}}})}}
func BenchmarkValidateReplay(b *testing.B){for _,n:=range []int{10,100,1000,10000,100000}{b.Run(fmt.Sprintf("events_%d",n),func(b *testing.B){_,_,m:=fixtureN(n);next:=m;next.StartExclusiveLSN=m.TerminalLSN-32;next.TerminalLSN=m.TerminalLSN+10;next.EvidenceThroughLSN=next.TerminalLSN;next.AcknowledgementRequestedLSN=next.TerminalLSN;next.AcknowledgementSentLSN=next.TerminalLSN;next.Events=nil;next.Transactions=nil;for _,tx:=range m.Transactions{if tx.CommitLSN>next.StartExclusiveLSN{next.Transactions=append(next.Transactions,tx);for _,id:=range tx.EventIDs{for _,e:=range m.Events{if e.ID==id{next.Events=append(next.Events,e)}}}}};commit:=m.TerminalLSN+5;id:="replay-new";xid:="replay-new-xid";next.Events=append(next.Events,DecodedEvent{ID:id,XID:xid,CommitLSN:commit,Ordinal:1,PayloadDigest:"replay-new-digest"});next.Transactions=append(next.Transactions,Transaction{XID:xid,State:"committed",BeginLSN:commit-1,CommitLSN:commit,EndLSN:commit+1,EventIDs:[]string{id}});b.ReportAllocs();b.ResetTimer();for i:=0;i<b.N;i++{r:=ValidateReplay(m,next);if r.Decision!=Qualified{b.Fatalf("%s: %s",r.Decision,r.Reason)}}})}}
