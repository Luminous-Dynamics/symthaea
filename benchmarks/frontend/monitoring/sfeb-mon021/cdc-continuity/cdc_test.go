package cdccheck

import "testing"
func ptr(v LSN)*LSN{return &v}
func base()(Profile,SlotState,Manifest){
	cp:=LSN(0x100);restart:=LSN(0x80);flush:=LSN(0x100)
	p:=Profile{ServerMajor:18,ConfiguredWALLevel:"logical",EffectiveWALLevel:"logical",SystemIdentifier:"sys-1",DatabaseOID:42,DatabaseName:"app",SlotName:"mon_slot",OutputPlugin:"pgoutput",PluginProfileDigest:"plugin-v1",SchemaProfileDigest:"schema-v1",SourceEpoch:"epoch-1",TimelineID:1,ConsistentPoint:&cp,SnapshotName:"snap-1",SnapshotDigest:"snapshot-digest"}
	s:=SlotState{SystemIdentifier:"sys-1",ServerMajor:18,TimelineID:1,SlotType:"logical",DatabaseOID:42,DatabaseName:"app",SlotName:"mon_slot",OutputPlugin:"pgoutput",WALStatus:"reserved",RestartLSN:&restart,ConfirmedFlushLSN:&flush,SafeWALSize:ptrI(1<<30),CatalogXmin:"100",Xmin:"100"}
	m:=Manifest{ServerMajor:18,SystemIdentifier:"sys-1",DatabaseName:"app",OutputPlugin:"pgoutput",SourceEpoch:"epoch-1",TimelineID:1,SlotName:"mon_slot",DatabaseOID:42,PluginProfileDigest:"plugin-v1",SchemaProfileDigest:"schema-v1",StartExclusiveLSN:cp,TerminalLSN:0x140,SnapshotConsistentPoint:cp,SnapshotDigest:"snapshot-digest",Events:[]DecodedEvent{{ID:"e1",XID:"10",CommitLSN:0x110,Ordinal:1,PayloadDigest:"d1"},{ID:"e2",XID:"11",CommitLSN:0x130,Ordinal:1,PayloadDigest:"d2"}},Complete:true,EvidenceThroughLSN:0x140,AcknowledgementRequestedLSN:0x140,AcknowledgementSentLSN:0x140,Witnessed:true,ManifestDigest:"manifest-digest"}
	m.Transactions=[]Transaction{
		{XID:"10",State:"committed",BeginLSN:0x108,CommitLSN:0x110,EndLSN:0x118,EventIDs:[]string{"e1"}},
		{XID:"11",State:"committed",BeginLSN:0x128,CommitLSN:0x130,EndLSN:0x138,EventIDs:[]string{"e2"}},
	}
	return p,s,m
}
func ptrI(v int64)*int64{return &v}
func setEventCommitLSN(m *Manifest,id string,lsn LSN){
	for i:=range m.Events{if m.Events[i].ID==id{m.Events[i].CommitLSN=lsn}}
	for i:=range m.Transactions{if (m.Transactions[i].XID=="10"&&id=="e1")||(m.Transactions[i].XID=="11"&&id=="e2"){m.Transactions[i].CommitLSN=lsn}}
}
func mergeEventsIntoFirstTransaction(m *Manifest){
	m.Events[1].XID=m.Events[0].XID
	m.Events[1].CommitLSN=m.Events[0].CommitLSN
	m.Events[1].Ordinal=m.Events[0].Ordinal+1
	m.Transactions[0].EventIDs=append(m.Transactions[0].EventIDs,m.Events[1].ID)
	m.Transactions=m.Transactions[:1]
}
func want(t *testing.T,r Result,d Decision){t.Helper();if r.Decision!=d{t.Fatalf("got %s (%s), want %s",r.Decision,r.Reason,d)}}
func TestLSNParseAndRoundTrip(t *testing.T){for _,x:=range []string{"0/0","0/16B6C50","A/F"}{v,e:=ParseLSN(x);if e!=nil{t.Fatal(e)};if _,e=ParseLSN(v.String());e!=nil{t.Fatal(e)}}}
func TestLSNRejectMalformed(t *testing.T){for _,x:=range []string{"","A","A/B/C","-1/0","0/xyz"}{if _,e:=ParseLSN(x);e==nil{t.Fatalf("accepted %q",x)}}}
func TestInitialSnapshotQualified(t *testing.T){p,s,_:=base();want(t,ValidateInitialSnapshot(p,s,0x100,"snapshot-digest"),Qualified)}
func TestSnapshotMismatchDenied(t *testing.T){p,s,_:=base();want(t,ValidateInitialSnapshot(p,s,0x101,"snapshot-digest"),Denied)}
func TestDatabaseMismatchDenied(t *testing.T){p,s,_:=base();s.DatabaseOID++;want(t,ValidateInitialSnapshot(p,s,0x100,"snapshot-digest"),Denied)}
func TestTemporarySlotDenied(t *testing.T){p,s,_:=base();s.Temporary=true;want(t,ValidateInitialSnapshot(p,s,0x100,"snapshot-digest"),Denied)}
func TestInvalidatedSlotDenied(t *testing.T){p,s,_:=base();s.InvalidationReason="wal_removed";want(t,ValidateInitialSnapshot(p,s,0x100,"snapshot-digest"),Denied)}
func TestUnreservedWALIndeterminate(t *testing.T){p,s,_:=base();s.WALStatus="unreserved";want(t,ValidateInitialSnapshot(p,s,0x100,"snapshot-digest"),Indeterminate)}
func TestCaptureQualified(t *testing.T){p,s,m:=base();want(t,ValidateCapture(p,s,m),Qualified)}
func TestDroppedEventIndeterminate(t *testing.T){p,s,m:=base();m.DroppedEvents=1;want(t,ValidateCapture(p,s,m),Indeterminate)}
func TestOverflowIndeterminate(t *testing.T){p,s,m:=base();m.QueueOverflowCount=1;want(t,ValidateCapture(p,s,m),Indeterminate)}
func TestAckAheadDurableDenied(t *testing.T){p,s,m:=base();m.AcknowledgementRequestedLSN=0x150;want(t,ValidateCapture(p,s,m),Denied)}
func TestConfirmedFlushAheadDurableDenied(t *testing.T){p,s,m:=base();s.ConfirmedFlushLSN=ptr(0x150);want(t,ValidateCapture(p,s,m),Denied)}
func TestEvidenceWatermarkMustEqualTerminal(t *testing.T){p,s,m:=base();m.EvidenceThroughLSN=0x130;want(t,ValidateCapture(p,s,m),Indeterminate)}
func TestRestartLSNUnknownIndeterminate(t *testing.T){p,s,m:=base();s.RestartLSN=nil;want(t,ValidateCapture(p,s,m),Indeterminate)}
func TestRestartLSNAfterRequiredStartIndeterminate(t *testing.T){p,s,m:=base();s.RestartLSN=ptr(0x120);want(t,ValidateCapture(p,s,m),Indeterminate)}
func TestLSNGapsAllowed(t *testing.T){p,s,m:=base();setEventCommitLSN(&m,"e2",0x138);want(t,ValidateCapture(p,s,m),Qualified)}
func TestLSNRegressionDenied(t *testing.T){p,s,m:=base();setEventCommitLSN(&m,"e2",0x108);want(t,ValidateCapture(p,s,m),Denied)}
func TestSameTransactionMessageOrdinalRegressionDenied(t *testing.T){p,s,m:=base();mergeEventsIntoFirstTransaction(&m);m.Events[1].Ordinal=1;want(t,ValidateCapture(p,s,m),Denied)}
func TestSameCommitBoundaryRequiresSameTransaction(t *testing.T){p,s,m:=base();m.Events[1].CommitLSN=m.Events[0].CommitLSN;want(t,ValidateCapture(p,s,m),Denied)}
func TestSameEventIDConflictingPayloadDivergent(t *testing.T){p,s,m:=base();m.Events[1].ID=m.Events[0].ID;m.Events[1].PayloadDigest="conflict";want(t,ValidateCapture(p,s,m),Divergent)}
func TestDuplicateIDChangedLSNDivergent(t *testing.T){p,s,m:=base();e:=m.Events[1];e.CommitLSN++;m.Events=append(m.Events,e);want(t,ValidateCapture(p,s,m),Divergent)}
func TestExactReplayDuplicateAccepted(t *testing.T){p,s,m:=base();m.Events=append(m.Events,m.Events[1]);want(t,ValidateCapture(p,s,m),Qualified)}
func TestUnsealedManifestIndeterminate(t *testing.T){p,s,m:=base();m.Witnessed=false;want(t,ValidateCapture(p,s,m),Indeterminate)}
func TestPluginProfileChangedDenied(t *testing.T){p,s,m:=base();m.PluginProfileDigest="changed";want(t,ValidateCapture(p,s,m),Denied)}
func TestReplayOverlapQualified(t *testing.T){_,_,m:=base();n:=m;n.Events=[]DecodedEvent{m.Events[1]};n.Transactions=[]Transaction{m.Transactions[1]};n.StartExclusiveLSN=0x120;n.TerminalLSN=0x160;n.EvidenceThroughLSN=0x160;n.AcknowledgementRequestedLSN=0x160;n.AcknowledgementSentLSN=0x160;n.Events=append(n.Events,DecodedEvent{ID:"e3",XID:"12",CommitLSN:0x150,Ordinal:1,PayloadDigest:"d3"});n.Transactions=append(n.Transactions,Transaction{XID:"12",State:"committed",BeginLSN:0x148,CommitLSN:0x150,EndLSN:0x158,EventIDs:[]string{"e3"}});want(t,ValidateReplay(m,n),Qualified)}
func TestReplayGapIndeterminate(t *testing.T){_,_,m:=base();n:=m;n.Events=nil;n.Transactions=nil;n.StartExclusiveLSN=0x150;n.TerminalLSN=0x160;n.EvidenceThroughLSN=0x160;n.AcknowledgementRequestedLSN=0x160;n.AcknowledgementSentLSN=0x160;want(t,ValidateReplay(m,n),Indeterminate)}
func TestReplayConflictingEventDivergent(t *testing.T){_,_,m:=base();n:=m;n.Events=[]DecodedEvent{m.Events[1]};n.Transactions=[]Transaction{m.Transactions[1]};n.StartExclusiveLSN=0x120;n.TerminalLSN=0x160;n.Events[0].PayloadDigest="conflicting";want(t,ValidateReplay(m,n),Divergent)}
func TestReplayTerminalRegressionIndeterminate(t *testing.T){_,_,m:=base();n:=m;n.Events=[]DecodedEvent{m.Events[1]};n.Transactions=[]Transaction{m.Transactions[1]};n.StartExclusiveLSN=0x120;n.TerminalLSN=0x138;n.EvidenceThroughLSN=0x138;n.AcknowledgementRequestedLSN=0x138;n.AcknowledgementSentLSN=0x138;want(t,ValidateReplay(m,n),Indeterminate)}
func failoverFixture() (Profile,Manifest,Failover){p,_,m:=base();replay:=LSN(0x150);restart:=LSN(0x80);s:=SlotState{SystemIdentifier:p.SystemIdentifier,ServerMajor:p.ServerMajor,TimelineID:2,SlotType:"logical",DatabaseOID:p.DatabaseOID,DatabaseName:p.DatabaseName,SlotName:p.SlotName,OutputPlugin:p.OutputPlugin,WALStatus:"reserved",Failover:true,Synced:true,OnStandby:true,RestartLSN:&restart,StandbyReplayLSN:&replay};n:=p;n.SourceEpoch="epoch-2";n.TimelineID=2;nextPoint:=LSN(0x160);n.ConsistentPoint=&nextPoint;f:=Failover{Prior:p,Next:n,StandbySlot:s,OldPrimaryFenced:true,TimelineHistoryDigest:"timeline-history",TimelineHistoryWitnessed:true,TimelineHistoryParent:p.TimelineID,ForkLSN:0x150,HandoffManifestDigest:"handoff",HandoffWitnessed:true};return p,m,f}
func TestFailoverQualified(t *testing.T){_,m,f:=failoverFixture();want(t,ValidateFailover(f,m.EvidenceThroughLSN),Qualified)}
func TestFailoverSlotNotSyncedDenied(t *testing.T){_,m,f:=failoverFixture();f.StandbySlot.Synced=false;want(t,ValidateFailover(f,m.EvidenceThroughLSN),Denied)}
func TestFailoverStandbyBehindIndeterminate(t *testing.T){_,m,f:=failoverFixture();f.StandbySlot.StandbyReplayLSN=ptr(0x130);want(t,ValidateFailover(f,m.EvidenceThroughLSN),Indeterminate)}
func TestFailoverOldPrimaryNotFencedIndeterminate(t *testing.T){_,m,f:=failoverFixture();f.OldPrimaryFenced=false;want(t,ValidateFailover(f,m.EvidenceThroughLSN),Indeterminate)}
func TestFailoverWrongStandbySlotDenied(t *testing.T){_,m,f:=failoverFixture();f.StandbySlot.DatabaseOID++;want(t,ValidateFailover(f,m.EvidenceThroughLSN),Denied)}
func FuzzLSNParse(f *testing.F){f.Add("0/0");f.Add("A/FFFF");f.Fuzz(func(t *testing.T,s string){v,e:=ParseLSN(s);if e==nil&&v.String()==""{t.Fatal("empty normalized LSN")}})}

func TestTransactionEnvelopeMissingIndeterminate(t *testing.T){p,s,m:=base();m.Transactions=m.Transactions[:1];want(t,ValidateCapture(p,s,m),Indeterminate)}
func TestTransactionEventMembershipMissingIndeterminate(t *testing.T){p,s,m:=base();m.Transactions[0].EventIDs=nil;want(t,ValidateCapture(p,s,m),Indeterminate)}
func TestTransactionEnvelopeContradictsCommitBoundaryDenied(t *testing.T){p,s,m:=base();m.Transactions[0].CommitLSN=0x111;want(t,ValidateCapture(p,s,m),Denied)}
func TestPreparedTransactionNotYetQualified(t *testing.T){p,s,m:=base();m.Transactions[0].State="prepared";want(t,ValidateCapture(p,s,m),Indeterminate)}
func TestTwoPhaseSlotNotQualified(t *testing.T){p,s,m:=base();s.TwoPhase=true;want(t,ValidateCapture(p,s,m),Indeterminate)}

func TestReplayAckAheadDenied(t *testing.T){_,_,m:=base();n:=m;n.TerminalLSN=0x160;n.EvidenceThroughLSN=0x150;n.AcknowledgementRequestedLSN=0x160;want(t,ValidateReplay(m,n),Indeterminate)}
func TestReplayAcknowledgementExceedsEvidenceDenied(t *testing.T){_,_,m:=base();n:=m;n.TerminalLSN=0x160;n.EvidenceThroughLSN=0x160;n.AcknowledgementRequestedLSN=0x170;want(t,ValidateReplay(m,n),Denied)}
func TestReplayTransactionConflictDivergent(t *testing.T){_,_,m:=base();n:=m;n.Events=append([]DecodedEvent(nil),m.Events...);n.Transactions=append([]Transaction(nil),m.Transactions...);n.Transactions[0].BeginLSN++;n.TerminalLSN=0x160;n.EvidenceThroughLSN=0x160;n.AcknowledgementRequestedLSN=0x160;n.AcknowledgementSentLSN=0x160;want(t,ValidateReplay(m,n),Divergent)}
func TestSlotTwoPhaseCaptureHolds(t *testing.T){p,s,m:=base();s.TwoPhase=true;want(t,ValidateCapture(p,s,m),Indeterminate)}

func TestPostgreSQL19AutoEffectiveLogicalWithReplicaConfig(t *testing.T){
	p,s,m:=base()
	p.ServerMajor=19
	p.ConfiguredWALLevel="replica"
	p.EffectiveWALLevel="logical"
	want(t,ValidateCapture(p,s,m),Qualified)
}
func TestPostgreSQL19RequiresEffectiveLogical(t *testing.T){
	p,s,m:=base()
	p.ServerMajor=19
	p.ConfiguredWALLevel="replica"
	p.EffectiveWALLevel="replica"
	want(t,ValidateCapture(p,s,m),Denied)
}
func TestPostgreSQL18RequiresConfiguredLogical(t *testing.T){
	p,s,m:=base()
	p.ConfiguredWALLevel="replica"
	want(t,ValidateCapture(p,s,m),Denied)
}
func TestUnsupportedPostgreSQLMajorIndeterminate(t *testing.T){
	p,s,m:=base()
	p.ServerMajor=17
	want(t,ValidateCapture(p,s,m),Indeterminate)
}

func TestDifferentXIDsCannotClaimSameCommitRecordDivergent(t *testing.T){
	p,s,m:=base()
	m.Events[1].CommitLSN=m.Events[0].CommitLSN
	m.Transactions[1].BeginLSN=0x108
	m.Transactions[1].CommitLSN=m.Transactions[0].CommitLSN
	m.Transactions[1].EndLSN=0x118
	want(t,ValidateCapture(p,s,m),Divergent)
}
func TestReplayPreparedTransactionRemainsIndeterminate(t *testing.T){
	_,_,m:=base()
	n:=m
	n.Transactions=append([]Transaction(nil),m.Transactions...)
	n.Transactions[0].State="prepared"
	n.TerminalLSN=0x160
	n.EvidenceThroughLSN=0x160
	want(t,ValidateReplay(m,n),Indeterminate)
}

func TestFailoverDurableEvidenceBeyondForkDenied(t *testing.T){
	_,m,f:=failoverFixture()
	f.ForkLSN=0x130
	f.StandbySlot.StandbyReplayLSN=ptr(0x150)
	want(t,ValidateFailover(f,m.EvidenceThroughLSN),Denied)
}
func TestFailoverTimelineHistoryNotWitnessedIndeterminate(t *testing.T){
	_,m,f:=failoverFixture()
	f.TimelineHistoryWitnessed=false
	want(t,ValidateFailover(f,m.EvidenceThroughLSN),Indeterminate)
}
func TestFailoverWrongParentTimelineIndeterminate(t *testing.T){
	_,m,f:=failoverFixture()
	f.TimelineHistoryParent=99
	want(t,ValidateFailover(f,m.EvidenceThroughLSN),Indeterminate)
}

func TestAcknowledgementActuallySentAheadOfDurableDenied(t *testing.T){
	p,s,m:=base()
	m.AcknowledgementRequestedLSN=0x140
	m.AcknowledgementSentLSN=0x150
	want(t,ValidateCapture(p,s,m),Denied)
}
func TestManifestClusterIdentityMismatchDenied(t *testing.T){
	p,s,m:=base()
	m.SystemIdentifier="other-cluster"
	want(t,ValidateCapture(p,s,m),Denied)
}
func TestManifestServerMajorMismatchDenied(t *testing.T){
	p,s,m:=base()
	m.ServerMajor=19
	want(t,ValidateCapture(p,s,m),Denied)
}
func TestSlotTimelineMismatchDenied(t *testing.T){
	p,s,m:=base()
	s.TimelineID=2
	want(t,ValidateCapture(p,s,m),Denied)
}
func TestFailoverStandbyTimelineMismatchDenied(t *testing.T){
	_,m,f:=failoverFixture()
	f.StandbySlot.TimelineID=1
	want(t,ValidateFailover(f,m.EvidenceThroughLSN),Denied)
}
func TestFailoverNextConsistentPointBeforeForkIndeterminate(t *testing.T){
	_,m,f:=failoverFixture()
	p:=LSN(0x140)
	f.Next.ConsistentPoint=&p
	want(t,ValidateFailover(f,m.EvidenceThroughLSN),Indeterminate)
}
