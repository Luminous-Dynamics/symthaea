package restoreassurance

import "testing"

func stamp(domain string,t,u int64) ClockStamp {return ClockStamp{Domain:domain,UnixNanos:t,UncertaintyNanos:u,Trusted:true}}
func base() Request {
	p:=Plan{ID:"plan",Version:"1",Digest:"plan-digest",RequestedClaim:ServiceRecovered,SourceSystem:"source-sys",Database:"app",SourceControlRoot:"source-root",RestoreTargetSystem:"restore-sys",RestoreControlRoot:"restore-root",RepositoryID:"repo",BackupMethodProfile:"pgbackrest-v1",RequestedTarget:RecoveryTarget{Mode:"lsn",Value:"0/500",LineageID:"lineage",TimelineID:1,LSN:0x500},RequiredWALStart:0x100,RequiredWALStop:0x600,RequireFullDatabaseStateFlag:true,SurfaceInventoryDigest:"surface-inventory",SurfaceClaimScope:"full_database_state",RequiredDependencyIDs:[]string{"kms","dns"},ApplicationReleaseDigest:"release",InvariantSuiteDigest:"suite",MaxRPOUpperNanos:60_000_000_000,MaxDatabaseRTOUpperNanos:600_000_000_000,MaxServiceRTOUpperNanos:900_000_000_000,IncidentAt:stamp("utc",10_000_000_000_000,1_000_000)}
	a:=ArtifactEvidence{SourceSystem:"source-sys",Database:"app",BackupSetID:"backup",RepositoryID:"repo",BackupDigest:"backup-digest",WALManifestDigest:"wal-digest",MethodProfileDigest:"pgbackrest-v1",EvidenceDigest:"artifact-evidence",WALStart:0x100,WALStop:0x700,LineageID:"lineage",TimelineID:1,IntegrityVerified:true,WALComplete:true,NoWALGaps:true,VerifierVerified:true,Complete:true,ObservedAt:9_900_000_000_000,FreshUntil:10_400_000_000_000}
	x:=RestoreExecution{EnvironmentID:"restore-sys",ControlRoot:"restore-root",Isolated:true,NetworkEgressBlocked:true,ProductionWritesBlocked:true,ScopedCredentials:true,CommandProfileDigest:"restore-cmd",RestoreManifestDigest:"restore-manifest",SourceSystem:"source-sys",Database:"app",RestoredSystemIdentifier:"source-sys",TimelineID:1,LineageID:"lineage",TimelineHistoryDigest:"timeline-history",TimelineHistoryVerified:true,RecoveryState:"ready",ConsistencyReached:true,TargetReached:true,CatalogChecksPassed:true,RequestedTarget:p.RequestedTarget,AchievedTarget:RecoveryTarget{Mode:"lsn",Value:"0/500",LineageID:"lineage",TimelineID:1,LSN:0x500,RecoveryTimestamp:9_990_000_000_000,RecoveryClockDomain:"utc",RecoveryClockUncertainty:1_000_000,RecoveryClockTrusted:true},StartedAt:stamp("utc",10_001_000_000_000,1_000_000),DatabaseReadyAt:stamp("utc",10_100_000_000_000,1_000_000),EvidenceDigest:"restore-evidence",VerifierVerified:true,Complete:true,FreshUntil:10_400_000_000_000}
	s:=SurfaceCoverage{Verified:true,FullDatabaseState:true,InventoryDigest:"surface-inventory",Scope:"full_database_state",RequiredCovered:14,RequiredTotal:14,EvidenceDigest:"surface-evidence"}
	deps:=[]Dependency{{ID:"kms",VersionDigest:"v1",ControlRoot:"kms-root",EvidenceDigest:"kms-e",Required:true,Ready:true,Verified:true,ObservedAt:9_900_000_000_000,FreshUntil:10_400_000_000_000},{ID:"dns",VersionDigest:"v1",ControlRoot:"dns-root",EvidenceDigest:"dns-e",Required:true,Ready:true,Verified:true,ObservedAt:9_900_000_000_000,FreshUntil:10_400_000_000_000}}
	probes:=[]Probe{{ID:"read",Kind:"read",Passed:true,Required:true,Verified:true,ApplicationReleaseDigest:"release",SuiteDigest:"suite",EvidenceDigest:"read-e",ObservedAt:9_900_000_000_000,FreshUntil:10_400_000_000_000},{ID:"write",Kind:"write",Passed:true,Required:true,Verified:true,ApplicationReleaseDigest:"release",SuiteDigest:"suite",EvidenceDigest:"write-e",RollbackVerified:true,ObservedAt:9_900_000_000_000,FreshUntil:10_400_000_000_000},{ID:"ledger",Kind:"invariant",Passed:true,Required:true,Verified:true,ApplicationReleaseDigest:"release",SuiteDigest:"suite",EvidenceDigest:"ledger-e",ObservedAt:9_900_000_000_000,FreshUntil:10_400_000_000_000}}
	ac:=ServiceAcceptance{TrafficEnabled:true,OldPrimaryFenced:true,StaleEndpointsFenced:true,ObserverID:"observer",ObserverControlRoot:"observer-root",ObserverIdentityDigest:"observer-id",ObserverMembershipDigest:"observer-membership",ObserverVerified:true,FencingCompletedAt:stamp("utc",10_150_000_000_000,1_000_000),TrafficEnabledAt:stamp("utc",10_180_000_000_000,1_000_000),ReadyAt:stamp("utc",10_200_000_000_000,1_000_000),EvidenceDigest:"service-e"}
	return Request{Plan:p,Artifact:a,Restore:x,Surface:s,Dependencies:deps,Probes:probes,Acceptance:ac,EvaluatedAt:10_300_000_000_000}
}
func check(t *testing.T,r Result,want Decision){t.Helper();if r.Decision!=want{t.Fatalf("got %s (%s), want %s",r.Decision,r.Reason,want)}}
func TestServiceRecoveredAllEvidence(t *testing.T){r:=base();out:=Evaluate(r);check(t,out,Qualified);if !out.ArtifactValidated||!out.DatabaseRecovered||!out.ServiceRecovered{t.Fatalf("claims not composed: %#v",out)}}
func TestArtifactClaimDoesNotPromoteDatabase(t *testing.T){r:=base();r.Plan.RequestedClaim=ArtifactValidated;out:=Evaluate(r);check(t,out,Qualified);if out.DatabaseRecovered||out.ServiceRecovered{t.Fatal("artifact claim promoted to higher recovery claim")}}
func TestDatabaseClaimDoesNotPromoteService(t *testing.T){r:=base();r.Plan.RequestedClaim=DatabaseRecovered;out:=Evaluate(r);check(t,out,Qualified);if out.ServiceRecovered{t.Fatal("database claim promoted to service recovery")}}
func TestWrongBackupSourceDenied(t *testing.T){r:=base();r.Artifact.SourceSystem="other";check(t,Evaluate(r),Denied)}
func TestBackupIntegrityMissingIndeterminate(t *testing.T){r:=base();r.Artifact.IntegrityVerified=false;check(t,Evaluate(r),Indeterminate)}
func TestRequiredWALRangeMissingDenied(t *testing.T){r:=base();r.Artifact.WALStop=0x400;check(t,Evaluate(r),Denied)}
func TestWALGapIndeterminate(t *testing.T){r:=base();r.Artifact.NoWALGaps=false;check(t,Evaluate(r),Indeterminate)}
func TestRestoreNotIsolatedDenied(t *testing.T){r:=base();r.Restore.NetworkEgressBlocked=false;check(t,Evaluate(r),Denied)}
func TestRestoredIdentityMismatchDenied(t *testing.T){r:=base();r.Restore.RestoredSystemIdentifier="other";check(t,Evaluate(r),Denied)}
func TestDatabaseNotConsistentIndeterminate(t *testing.T){r:=base();r.Restore.ConsistencyReached=false;check(t,Evaluate(r),Indeterminate)}
func TestRecoveryTargetNotReachedDenied(t *testing.T){r:=base();r.Restore.TargetReached=false;check(t,Evaluate(r),Denied)}
func TestRecoveryTimelineUnprovenIndeterminate(t *testing.T){r:=base();r.Restore.TimelineHistoryVerified=false;check(t,Evaluate(r),Indeterminate)}
func TestAchievedLSNBeforeTargetDenied(t *testing.T){r:=base();r.Restore.AchievedTarget.LSN=0x400;check(t,Evaluate(r),Denied)}
func TestRPOExceedsObjectiveDenied(t *testing.T){r:=base();r.Restore.AchievedTarget.RecoveryTimestamp=9_000_000_000_000;check(t,Evaluate(r),Denied)}
func TestClockDomainMismatchIndeterminate(t *testing.T){r:=base();r.Restore.DatabaseReadyAt.Domain="other-clock";check(t,Evaluate(r),Indeterminate)}
func TestRTOExceedsObjectiveDenied(t *testing.T){r:=base();r.Restore.DatabaseReadyAt.UnixNanos+=2_000_000_000_000;check(t,Evaluate(r),Denied)}
func TestFullStateSurfaceMissingIndeterminate(t *testing.T){r:=base();r.Surface.Verified=false;check(t,Evaluate(r),Indeterminate)}
func TestFullStateExclusionsDenied(t *testing.T){r:=base();r.Surface.Excluded=[]string{"large-objects"};check(t,Evaluate(r),Denied)}
func TestDatabaseRecoveredWithoutDependencies(t *testing.T){r:=base();r.Plan.RequestedClaim=DatabaseRecovered;r.Dependencies=nil;check(t,Evaluate(r),Qualified)}
func TestServiceMissingDependencyIndeterminate(t *testing.T){r:=base();r.Dependencies=r.Dependencies[:1];check(t,Evaluate(r),Indeterminate)}
func TestRequiredDependencyUnavailableDenied(t *testing.T){r:=base();r.Dependencies[0].Ready=false;check(t,Evaluate(r),Denied)}
func TestFailedBusinessInvariantDenied(t *testing.T){r:=base();r.Probes[2].Passed=false;check(t,Evaluate(r),Denied)}
func TestReadOnlyWithoutSafeWriteIndeterminate(t *testing.T){r:=base();r.Probes=r.Probes[:1];check(t,Evaluate(r),Indeterminate)}
func TestWriteProbeWithoutRollbackIndeterminate(t *testing.T){r:=base();r.Probes[1].RollbackVerified=false;check(t,Evaluate(r),Indeterminate)}
func TestStaleApplicationSuiteIndeterminate(t *testing.T){r:=base();r.Probes[0].SuiteDigest="old";check(t,Evaluate(r),Indeterminate)}
func TestTrafficBeforeFencingDenied(t *testing.T){r:=base();r.Acceptance.OldPrimaryFenced=false;check(t,Evaluate(r),Denied)}
func TestServiceObserverNotIndependentIndeterminate(t *testing.T){r:=base();r.Acceptance.ObserverControlRoot=r.Restore.ControlRoot;check(t,Evaluate(r),Indeterminate)}
func TestServiceAcceptanceMissingIndeterminate(t *testing.T){r:=base();r.Acceptance.ObserverVerified=false;check(t,Evaluate(r),Indeterminate)}
func TestServiceRTOExceededDenied(t *testing.T){r:=base();r.Acceptance.ReadyAt.UnixNanos+=2_000_000_000_000;check(t,Evaluate(r),Denied)}
func TestNegativeRPOClockConflictDivergent(t *testing.T){r:=base();r.Restore.AchievedTarget.RecoveryTimestamp=10_010_000_000_000;r.Restore.AchievedTarget.RecoveryClockUncertainty=0;check(t,Evaluate(r),Divergent)}
func TestValidationOnlyRestoreCannotServe(t *testing.T){r:=base();r.Restore.RecoveryState="validation_only";check(t,Evaluate(r),Indeterminate)}
func TestOptionalDependencyNotRequired(t *testing.T){r:=base();r.Dependencies[0].Required=false;r.Plan.RequiredDependencyIDs=[]string{"dns"};check(t,Evaluate(r),Qualified)}
func TestArtifactClaimMissingWALRangeIndeterminate(t *testing.T){r:=base();r.Plan.RequestedClaim=ArtifactValidated;r.Artifact.WALStop=0x400;check(t,Evaluate(r),Denied)}
func FuzzRPOUpperBound(f *testing.F){f.Add(int64(100),int64(10),int64(10),int64(10));f.Fuzz(func(t *testing.T,incident,incidentUncertainty,point,pointUncertainty int64){if incident<=0||point<=0||incidentUncertainty<0||pointUncertainty<0{return};upper:=(incident+incidentUncertainty)-(point-pointUncertainty);if upper<0{return};if upper>(incident-point)+incidentUncertainty+pointUncertainty{t.Fatal("RPO upper bound not conservative")}})}

func TestIncompleteSurfaceCountIndeterminate(t *testing.T) {
 r:=base();r.Surface.RequiredCovered=13;check(t,Evaluate(r),Indeterminate)
}
func TestTrafficEnabledBeforeFencingDenied(t *testing.T) {
 r:=base();r.Acceptance.TrafficEnabledAt=stamp("utc",10_140_000_000_000,1_000_000);check(t,Evaluate(r),Denied)
}
func TestTrafficEnabledBeforeDatabaseReadyDenied(t *testing.T) {
 r:=base();r.Acceptance.TrafficEnabledAt=stamp("utc",10_090_000_000_000,1_000_000);check(t,Evaluate(r),Denied)
}
func TestRestoreEnvironmentMustMatchDeclaredTarget(t *testing.T) {
 r:=base();r.Restore.EnvironmentID="other-restore";check(t,Evaluate(r),Indeterminate)
}
func TestRestoreControlRootMustMatchDeclaredTarget(t *testing.T) {
 r:=base();r.Restore.ControlRoot="other-root";check(t,Evaluate(r),Indeterminate)
}
func TestRestoreTargetTimelineMismatchIndeterminate(t *testing.T) {
 r:=base();r.Restore.AchievedTarget.TimelineID=2;check(t,Evaluate(r),Indeterminate)
}
func TestDuplicateRequiredDependencyIDDivergent(t *testing.T) {
 r:=base();r.Plan.RequiredDependencyIDs=append(r.Plan.RequiredDependencyIDs,"kms");check(t,Evaluate(r),Divergent)
}
func TestDuplicateApplicationProbeDivergent(t *testing.T) {
 r:=base();r.Probes=append(r.Probes,r.Probes[0]);check(t,Evaluate(r),Divergent)
}
func TestServiceObserverCannotAliasRestoreEnvironment(t *testing.T) {
 r:=base();r.Acceptance.ObserverID=r.Restore.EnvironmentID;check(t,Evaluate(r),Indeterminate)
}

func TestSurfaceClaimScopeMismatchIndeterminate(t *testing.T) {
 r:=base();r.Surface.Scope="scoped_database_state";check(t,Evaluate(r),Indeterminate)
}
func TestUnverifiedApplicationProbeIndeterminate(t *testing.T) {
 r:=base();r.Probes[0].Verified=false;check(t,Evaluate(r),Indeterminate)
}
func TestMissingInvariantProbeIndeterminate(t *testing.T) {
 r:=base();r.Probes=r.Probes[:2];check(t,Evaluate(r),Indeterminate)
}
func TestDatabaseReadyBeforeRestoreStartDivergent(t *testing.T) {
 r:=base();r.Restore.DatabaseReadyAt.UnixNanos=r.Restore.StartedAt.UnixNanos-1;check(t,Evaluate(r),Divergent)
}
func TestServiceReadyMustFollowTrafficEnable(t *testing.T) {
 r:=base();r.Acceptance.ReadyAt.UnixNanos=r.Acceptance.TrafficEnabledAt.UnixNanos-1;check(t,Evaluate(r),Denied)
}
func TestVerifiedDependencyUnavailableDenied(t *testing.T) {
 r:=base();r.Dependencies[0].Ready=false;check(t,Evaluate(r),Denied)
}

func TestUnverifiedRequiredDependencyIndeterminate(t *testing.T) {
 r:=base();r.Dependencies[0].Verified=false;check(t,Evaluate(r),Indeterminate)
}
func TestMissingRequiredDependencyIndeterminate(t *testing.T) {
 r:=base();r.Dependencies=r.Dependencies[:1];check(t,Evaluate(r),Indeterminate)
}
func TestRestoreFreshnessStaleIndeterminate(t *testing.T) {
 r:=base();r.Restore.FreshUntil=r.EvaluatedAt;check(t,Evaluate(r),Indeterminate)
}
func TestArtifactFreshnessStaleIndeterminate(t *testing.T) {
 r:=base();r.Artifact.FreshUntil=r.EvaluatedAt;check(t,Evaluate(r),Indeterminate)
}
func TestServiceClaimRequiresTrafficAcceptance(t *testing.T) {
 r:=base();r.Acceptance.TrafficEnabled=false;check(t,Evaluate(r),Denied)
}
