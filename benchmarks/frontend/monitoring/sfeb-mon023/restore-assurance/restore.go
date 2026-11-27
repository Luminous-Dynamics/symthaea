package restoreassurance

import (
	"fmt"
	"strings"
)

type Claim string
const (
	ArtifactValidated Claim = "artifact_validated"
	DatabaseRecovered Claim = "database_recovered"
	ServiceRecovered Claim = "service_recovered"
)
type Decision string
const (
	Qualified Decision = "qualified"
	Denied Decision = "denied"
	Indeterminate Decision = "indeterminate"
	Divergent Decision = "divergent"
)
type ClockStamp struct {
	Domain string
	UnixNanos int64
	UncertaintyNanos int64
	Trusted bool
}
type RecoveryTarget struct {
	Mode string
	Value string
	LineageID string
	TimelineID uint32
	LSN uint64
	RecoveryTimestamp int64
	RecoveryClockDomain string
	RecoveryClockUncertainty int64
	RecoveryClockTrusted bool
}
func (t RecoveryTarget) TimestampNanos() int64 { return t.RecoveryTimestamp }
func (t RecoveryTarget) TimestampDomain() string { return t.RecoveryClockDomain }
func (t RecoveryTarget) TimestampUncertainty() int64 { return t.RecoveryClockUncertainty }
func (t RecoveryTarget) TimestampTrusted() bool { return t.RecoveryClockTrusted }
func validTargetStamp(t RecoveryTarget) bool {
	return present(t.RecoveryClockDomain) && t.RecoveryTimestamp>0 && t.RecoveryClockUncertainty>=0 && t.RecoveryClockTrusted
}
type Plan struct {
	ID,Version,Digest string
	RequestedClaim Claim
	SourceSystem,Database,SourceControlRoot string
	RestoreTargetSystem,RestoreControlRoot string
	RepositoryID,BackupMethodProfile string
	RequestedTarget RecoveryTarget
	RequiredWALStart,RequiredWALStop uint64
	RequireFullDatabaseStateFlag bool
	SurfaceInventoryDigest string
	RequiredDependencyIDs []string
	ApplicationReleaseDigest,InvariantSuiteDigest string
	MaxRPOUpperNanos,MaxDatabaseRTOUpperNanos,MaxServiceRTOUpperNanos int64
	IncidentAt ClockStamp
}
type ArtifactEvidence struct {
	SourceSystem,Database,BackupSetID,RepositoryID string
	BackupDigest,WALManifestDigest,MethodProfileDigest,EvidenceDigest string
	WALStart,WALStop uint64
	LineageID string
	TimelineID uint32
	IntegrityVerified,WALComplete,NoWALGaps,VerifierVerified,Complete bool
	ObservedAt int64
	FreshUntil int64
}
type RestoreExecution struct {
	EnvironmentID,ControlRoot string
	Isolated,NetworkEgressBlocked,ProductionWritesBlocked,ScopedCredentials bool
	CommandProfileDigest,RestoreManifestDigest string
	SourceSystem,Database,RestoredSystemIdentifier string
	TimelineID uint32
	LineageID,TimelineHistoryDigest string
	TimelineHistoryVerified bool
	RecoveryState string // ready, validation_only, recovering, failed
	ConsistencyReached,TargetReached,CatalogChecksPassed bool
	RequestedTarget,AchievedTarget RecoveryTarget
	StartedAt,DatabaseReadyAt ClockStamp
	EvidenceDigest string
	VerifierVerified,Complete bool
	FreshUntil int64
}
type SurfaceCoverage struct {
	Verified bool
	FullDatabaseState bool
	InventoryDigest string
	Scope string
	RequiredCovered int
	RequiredTotal int
	Excluded []string
	NotApplicable int
	EvidenceDigest string
}
type Dependency struct {
	ID,VersionDigest,ControlRoot,EvidenceDigest string
	Required,Ready,Verified bool
	ObservedAt,FreshUntil int64
}
type Probe struct {
	ID,Kind string // read, write, invariant
	Passed,Required,Verified bool
	ApplicationReleaseDigest,SuiteDigest,EvidenceDigest string
	RollbackVerified bool
	ObservedAt,FreshUntil int64
}
type ServiceAcceptance struct {
	TrafficEnabled,OldPrimaryFenced,StaleEndpointsFenced bool
	ObserverID,ObserverControlRoot,ObserverIdentityDigest,ObserverMembershipDigest string
	ObserverVerified bool
	FencingCompletedAt,TrafficEnabledAt,ReadyAt ClockStamp
	EvidenceDigest string
}
type Request struct {
	Plan Plan
	Artifact ArtifactEvidence
	Restore RestoreExecution
	Surface SurfaceCoverage
	Dependencies []Dependency
	Probes []Probe
	Acceptance ServiceAcceptance
	EvaluatedAt int64
}
type Result struct {
	Decision Decision
	Reason string
	RequestedClaim,HighestQualifiedClaim Claim
	ArtifactValidated,DatabaseRecovered,ServiceRecovered bool
	RPOUpperNanos,DatabaseRTOUpperNanos,ServiceRTOUpperNanos int64
	ExcludedSurfaces []string
	UnreadyDependencies,FailedProbes []string
	Checked []string
}
func present(xs ...string)bool{for _,x:=range xs{if strings.TrimSpace(x)==""{return false}};return true}
func done(d Decision,reason string,r Result)Result{r.Decision=d;r.Reason=reason;return r}
func validStamp(s ClockStamp)bool{return present(s.Domain)&&s.UnixNanos>0&&s.UncertaintyNanos>=0&&s.Trusted}
func withinFreshness(observed,freshUntil,evaluated int64)bool{return observed>0&&observed<=evaluated&&freshUntil>evaluated}

func Evaluate(req Request)Result{
	p:=req.Plan
	r:=Result{RequestedClaim:p.RequestedClaim,Checked:[]string{"recovery-plan","artifact-and-wal-lineage"}}
	if p.RequestedClaim!=ArtifactValidated&&p.RequestedClaim!=DatabaseRecovered&&p.RequestedClaim!=ServiceRecovered{return done(Indeterminate,"unknown recovery claim level",r)}
	if !present(p.ID,p.Version,p.Digest,p.SourceSystem,p.Database,p.SourceControlRoot,p.RestoreTargetSystem,p.RestoreControlRoot,p.RepositoryID,p.BackupMethodProfile,p.RequestedTarget.Mode,p.RequestedTarget.Value,p.RequestedTarget.LineageID,p.SurfaceInventoryDigestForPlan())||p.SourceSystem==p.RestoreTargetSystem||p.SourceControlRoot==p.RestoreControlRoot||p.MaxRPOUpperNanos<0||p.MaxDatabaseRTOUpperNanos<0||p.MaxServiceRTOUpperNanos<0||req.EvaluatedAt<=0{
		return done(Indeterminate,"recovery plan identity, scope or objectives incomplete",r)
	}
	a:=req.Artifact
	if a.SourceSystem!=p.SourceSystem||a.Database!=p.Database||a.RepositoryID!=p.RepositoryID{return done(Denied,"backup artifact source/database/repository does not match plan",r)}
	if !present(a.BackupSetID,a.BackupDigest,a.WALManifestDigest,a.MethodProfileDigest,a.EvidenceDigest,a.LineageID)||a.MethodProfileDigest!=p.BackupMethodProfile{return done(Indeterminate,"backup artifact evidence/profile incomplete",r)}
	if !a.VerifierVerified||!a.Complete||!a.IntegrityVerified||!a.WALComplete||!a.NoWALGaps{return done(Indeterminate,"backup integrity or required WAL evidence is not qualified",r)}
	if a.FreshUntil<=req.EvaluatedAt||!withinFreshness(a.ObservedAt,a.FreshUntil,req.EvaluatedAt){return done(Indeterminate,"backup/WAL evidence stale or outside evaluation boundary",r)}
	if a.LineageID!=p.RequestedTarget.LineageID||a.TimelineID!=p.RequestedTarget.TimelineID{return done(Indeterminate,"backup WAL lineage does not bind requested recovery target",r)}
	if a.WALStart>p.RequiredWALStart||a.WALStop<p.RequiredWALStop{return done(Denied,"backup archive does not cover the required WAL range",r)}
	if p.RequestedTarget.LSN>0&&a.WALStop<p.RequestedTarget.LSN{return done(Denied,"WAL archive ends before requested LSN target",r)}
	r.ArtifactValidated=true;r.HighestQualifiedClaim=ArtifactValidated
	if p.RequestedClaim==ArtifactValidated{return done(Qualified,"backup artifact and required WAL range qualify; no restore claim is implied",r)}

	r.Checked=append(r.Checked,"isolated-restore","recovery-point-lineage","database-consistency","surface-coverage","rpo-rto")
	x:=req.Restore
	if !present(x.EnvironmentID,x.ControlRoot,x.CommandProfileDigest,x.RestoreManifestDigest,x.SourceSystem,x.Database,x.RestoredSystemIdentifier,x.LineageID,x.EvidenceDigest)||x.EnvironmentID!=p.RestoreTargetSystem||x.ControlRoot!=p.RestoreControlRoot||x.EnvironmentID==p.SourceSystem||x.ControlRoot==p.SourceControlRoot{return done(Indeterminate,"restore environment/source identity incomplete or not bound to the declared isolated target",r)}
	if !x.Isolated||!x.NetworkEgressBlocked||!x.ProductionWritesBlocked||!x.ScopedCredentials{return done(Denied,"restore rehearsal is not safely isolated from production",r)}
	if x.SourceSystem!=p.SourceSystem||x.Database!=p.Database||x.RestoredSystemIdentifier!=p.SourceSystem{return done(Denied,"restored cluster/database identity differs from source plan",r)}
	if !x.VerifierVerified||!x.Complete||x.FreshUntil<=req.EvaluatedAt||!withinFreshness(x.StartedAt.UnixNanos,x.FreshUntil,req.EvaluatedAt){return done(Indeterminate,"restore execution evidence unverified, incomplete or stale",r)}
	if !x.ConsistencyReached||!x.CatalogChecksPassed{return done(Indeterminate,"database consistency/catalog checks not established",r)}
	if x.RecoveryState!="ready"&&x.RecoveryState!="validation_only"{return done(Indeterminate,"database is not at a terminal recovered/validation-only state",r)}
	if !validStamp(x.StartedAt)||!validStamp(x.DatabaseReadyAt)||!validStamp(p.IncidentAt)||!validTargetStamp(x.AchievedTarget) {return done(Indeterminate,"incident, recovery-point or database-ready clock evidence incomplete",r)}
	if x.StartedAt.Domain!=p.IncidentAt.Domain||x.DatabaseReadyAt.Domain!=p.IncidentAt.Domain||x.AchievedTarget.TimestampDomain()!=p.IncidentAt.Domain{return done(Indeterminate,"incident/recovery/database-ready clocks are incomparable",r)}
	if !x.TargetReached||x.RequestedTarget!=p.RequestedTarget{return done(Denied,"restore did not reach the exact requested recovery target",r)}
	if x.AchievedTarget.LineageID!=p.RequestedTarget.LineageID||x.AchievedTarget.TimelineID!=p.RequestedTarget.TimelineID||x.LineageID!=p.RequestedTarget.LineageID||x.TimelineID!=p.RequestedTarget.TimelineID||!x.TimelineHistoryVerified||!present(x.TimelineHistoryDigest){return done(Indeterminate,"achieved recovery point is not proven on the requested timeline/lineage",r)}
	if p.RequestedTarget.LSN>0&&x.AchievedTarget.LSN<p.RequestedTarget.LSN{return done(Denied,"achieved recovery LSN is before requested target",r)}
	r.RPOUpperNanos=(p.IncidentAt.UnixNanos+p.IncidentAt.UncertaintyNanos)-(x.AchievedTarget.TimestampNanos()-x.AchievedTarget.TimestampUncertainty())
	r.DatabaseRTOUpperNanos=(x.DatabaseReadyAt.UnixNanos+x.DatabaseReadyAt.UncertaintyNanos)-(p.IncidentAt.UnixNanos-p.IncidentAt.UncertaintyNanos)
	if r.RPOUpperNanos<0||r.DatabaseRTOUpperNanos<0{return done(Divergent,"clock evidence produces a negative RPO or database RTO",r)}
	if r.RPOUpperNanos>p.MaxRPOUpperNanos||r.DatabaseRTOUpperNanos>p.MaxDatabaseRTOUpperNanos{return done(Denied,"RPO or database RTO exceeds recovery objective",r)}
	s:=req.Surface
	if !s.Verified||!present(s.InventoryDigest,s.Scope,s.EvidenceDigest)||s.InventoryDigest!=p.SurfaceInventoryDigest||s.RequiredTotal<=0||s.RequiredCovered!=s.RequiredTotal{return done(Indeterminate,"MON-022 surface inventory result missing, incomplete or stale",r)}
	if p.RequireFullDatabaseStateFlag&&(!s.FullDatabaseState||len(s.Excluded)>0){return done(Denied,"full-state recovery plan has excluded/unqualified database surfaces",r)}
	if !p.RequireFullDatabaseStateFlag&&s.Scope!="row_stream_continuity"&&s.Scope!="scoped_database_state"{return done(Indeterminate,"database recovery scope is unsupported or unnamed",r)}
	if p.RequestedClaim==ServiceRecovered&&x.RecoveryState!="ready"{return done(Indeterminate,"validation-only paused restore cannot qualify service recovery",r)}
	r.ExcludedSurfaces=append([]string(nil),s.Excluded...)
	r.DatabaseRecovered=true;r.HighestQualifiedClaim=DatabaseRecovered
	if p.RequestedClaim==DatabaseRecovered{return done(Qualified,"database recovered at the requested lineage/target with declared surface scope and RPO/RTO",r)}

	r.Checked=append(r.Checked,"required-dependencies","application-invariants","read-write-probes","traffic-fencing","independent-service-acceptance")
	deps:=map[string]Dependency{}
	for _,d:=range req.Dependencies{
		if !present(d.ID){return done(Indeterminate,"dependency identity missing",r)}
		if _,ok:=deps[d.ID];ok{return done(Divergent,"duplicate dependency evidence: "+d.ID,r)}
		deps[d.ID]=d
	}
	requiredDependencyIDs:=map[string]bool{}
	for _,id:=range p.RequiredDependencyIDs{
		if requiredDependencyIDs[id]{return done(Divergent,"duplicate required dependency in recovery plan: "+id,r)}
		requiredDependencyIDs[id]=true
		d,ok:=deps[id]
		if !ok{r.UnreadyDependencies=append(r.UnreadyDependencies,id);continue}
		if !d.Required||!d.Verified||!present(d.VersionDigest,d.ControlRoot,d.EvidenceDigest)||!withinFreshness(d.ObservedAt,d.FreshUntil,req.EvaluatedAt){r.UnreadyDependencies=append(r.UnreadyDependencies,id);continue}
		if !d.Ready{r.UnreadyDependencies=append(r.UnreadyDependencies,id)}
	}
	if len(r.UnreadyDependencies)>0{return done(Indeterminate,"one or more required dependencies are unavailable or unqualified",r)}
	readSeen,writeSeen:=false,false
	probeIDs:=map[string]bool{}
	for _,probe:=range req.Probes{
		if probeIDs[probe.ID]{return done(Divergent,"duplicate application probe ID: "+probe.ID,r)}
		probeIDs[probe.ID]=true
		if !present(probe.ID,probe.Kind){return done(Indeterminate,"application probe identity missing",r)}
		if !probe.Required{continue}
		if !probe.Verified||!present(probe.ApplicationReleaseDigest,probe.SuiteDigest,probe.EvidenceDigest)||probe.ApplicationReleaseDigest!=p.ApplicationReleaseDigest||probe.SuiteDigest!=p.InvariantSuiteDigest||!withinFreshness(probe.ObservedAt,probe.FreshUntil,req.EvaluatedAt){r.FailedProbes=append(r.FailedProbes,probe.ID);continue}
		if !probe.Passed{r.FailedProbes=append(r.FailedProbes,probe.ID);continue}
		switch probe.Kind{
		case "read":readSeen=true
		case "write":if !probe.RollbackVerified{return done(Indeterminate,"safe write probe lacks rollback/containment evidence",r)}else{writeSeen=true}
		case "invariant":
		default:r.FailedProbes=append(r.FailedProbes,probe.ID)
		}
	}
	if len(r.FailedProbes)>0{return done(Denied,"required application probe/invariant failed or is unqualified",r)}
	if !readSeen||!writeSeen{return done(Indeterminate,"service recovery requires qualified read and safe write/readback probes",r)}
	ac:=req.Acceptance
	if !ac.TrafficEnabled||!ac.OldPrimaryFenced||!ac.StaleEndpointsFenced{return done(Denied,"traffic is not safely enabled after old-primary and endpoint fencing",r)}
	if !ac.ObserverVerified||!present(ac.ObserverID,ac.ObserverControlRoot,ac.ObserverIdentityDigest,ac.ObserverMembershipDigest,ac.EvidenceDigest)||ac.ObserverID==p.SourceSystem||ac.ObserverID==x.EnvironmentID||ac.ObserverControlRoot==p.SourceControlRoot||ac.ObserverControlRoot==x.ControlRoot{return done(Indeterminate,"independent service acceptance observer is not qualified",r)}
	if !validStamp(ac.FencingCompletedAt)||!validStamp(ac.TrafficEnabledAt)||!validStamp(ac.ReadyAt)||ac.FencingCompletedAt.Domain!=p.IncidentAt.Domain||ac.TrafficEnabledAt.Domain!=p.IncidentAt.Domain||ac.ReadyAt.Domain!=p.IncidentAt.Domain{return done(Indeterminate,"service fencing/traffic/ready timestamps are incomparable or untrusted",r)}
	if ac.FencingCompletedAt.UnixNanos<x.DatabaseReadyAt.UnixNanos||ac.TrafficEnabledAt.UnixNanos<ac.FencingCompletedAt.UnixNanos||ac.TrafficEnabledAt.UnixNanos<x.DatabaseReadyAt.UnixNanos||ac.ReadyAt.UnixNanos<ac.TrafficEnabledAt.UnixNanos{return done(Denied,"traffic or service acceptance preceded database readiness/fencing",r)}
	r.ServiceRTOUpperNanos=(ac.ReadyAt.UnixNanos+ac.ReadyAt.UncertaintyNanos)-(p.IncidentAt.UnixNanos-p.IncidentAt.UncertaintyNanos)
	if r.ServiceRTOUpperNanos<0{return done(Divergent,"service-ready time precedes incident boundary",r)}
	if r.ServiceRTOUpperNanos>p.MaxServiceRTOUpperNanos{return done(Denied,"service RTO exceeds declared objective",r)}
	if !present(ac.EvidenceDigest){return done(Indeterminate,"independent service-acceptance evidence missing",r)}
	r.ServiceRecovered=true;r.HighestQualifiedClaim=ServiceRecovered
	return done(Qualified,"service recovered: database, surfaces, dependencies, application invariants, safe probes, fencing and independent acceptance all qualify",r)
}


