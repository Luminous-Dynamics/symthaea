package eventorder

import (
	"fmt"
	"sort"
	"strings"
)

type Decision string
const (
	Qualified Decision = "qualified"
	Denied Decision = "denied"
	Indeterminate Decision = "indeterminate"
	Divergent Decision = "divergent"
)

type EventKind string
const (
	EntityGenerated EventKind = "entity_generated"
	EntityUsed EventKind = "entity_used"
	EntityInvalidated EventKind = "entity_invalidated"
	ActivityStarted EventKind = "activity_started"
	ActivityEnded EventKind = "activity_ended"
)

type DigestRef struct {
	Algorithm string
	Profile string
	Version string
	PayloadType string
	Canonicalization string
	Purpose string
	Subject string
	Value string
}
func (d DigestRef) Valid() bool {
	return nonempty(d.Algorithm,d.Profile,d.Version,d.PayloadType,d.Canonicalization,d.Purpose,d.Subject,d.Value)
}
func (d DigestRef) Equal(x DigestRef) (bool,bool) {
	if d.Algorithm!=x.Algorithm || d.Profile!=x.Profile || d.Version!=x.Version ||
		d.PayloadType!=x.PayloadType || d.Canonicalization!=x.Canonicalization ||
		d.Purpose!=x.Purpose || d.Subject!=x.Subject { return false,false }
	return d.Value==x.Value,true
}
type Source struct {
	// ID identifies one monotonic sequence domain/incarnation, not merely a service name.
	ID string
	ControlRoot string
	IdentityDigest DigestRef
	// ControlRootDigest is an independently verified registry/credential statement
	// binding this sequence-domain identity to its administrative control root.
	ControlRootDigest DigestRef
}
type Event struct {
	ID string
	Kind EventKind
	SubjectID string
	ActivityID string
	SourceID string
	Sequence uint64
	RecordDigest DigestRef
}
type OrderWitness struct {
	ID string
	BeforeEventID string
	AfterEventID string
	WitnessSourceID string
	EvidenceDigest DigestRef
}
type VerifierReport struct {
	SubjectID string
	InputDigest DigestRef
	Status string
	VerifierID string
	Version string
	BuildDigest string
	TrustStoreDigest string
	VerifiedAt int64
	FreshUntil int64
}
type VerifierPin struct {
	ID string
	Version string
	BuildDigest string
	TrustStoreDigest string
}
type Input struct {
	Sources []Source
	Events []Event
	Witnesses []OrderWitness
	Reports []VerifierReport
	EvaluationAt int64
}
type Result struct {
	Decision Decision
	Reason string
	RequiredPairs int
	ProvenPairs int
	Checked []string
}
type pair struct { before,after string }

func nonempty(xs ...string) bool {
	for _,x:=range xs { if strings.TrimSpace(x)=="" { return false } }
	return true
}
func finish(d Decision,reason string,required,proven int,checked ...string) Result {
	return Result{d,reason,required,proven,append([]string(nil),checked...)}
}
func reportFor(subject string,digest DigestRef,reports map[string]VerifierReport,conflicts map[string]bool,pins map[string]VerifierPin,at int64) (Decision,string) {
	if conflicts[subject] { return Divergent,"conflicting verifier reports: "+subject }
	r,ok:=reports[subject]
	if !ok { return Indeterminate,"verifier report missing: "+subject }
	if r.Status=="invalid" { return Denied,"signature/profile verification failed: "+subject }
	if r.Status!="verified" { return Indeterminate,"cryptographic verification not established: "+subject }
	p,ok:=pins[r.VerifierID]
	if !ok || p.ID!=r.VerifierID || p.Version!=r.Version || p.BuildDigest!=r.BuildDigest ||
		p.TrustStoreDigest!=r.TrustStoreDigest || !nonempty(r.VerifierID,r.Version,r.BuildDigest,r.TrustStoreDigest) {
		return Indeterminate,"verifier identity/version/build/trust-store not pinned: "+subject
	}
	if r.VerifiedAt<=0 || r.VerifiedAt>at || r.FreshUntil<=at {
		return Indeterminate,"verifier report stale or outside evaluation boundary: "+subject
	}
	eq,comparable:=r.InputDigest.Equal(digest)
	if !comparable { return Indeterminate,"verifier input digest profile incomparable: "+subject }
	if !eq { return Denied,"verifier report not bound to exact bytes: "+subject }
	return Qualified,""
}
func addEdge(adj map[string][]string,indegree map[string]int,a,b string) {
	adj[a]=append(adj[a],b)
	indegree[b]++
}
func hasPath(adj map[string][]string,start,target string) bool {
	if start==target { return true }
	seen:=map[string]bool{start:true}
	queue:=[]string{start}
	for len(queue)>0 {
		x:=queue[0];queue=queue[1:]
		for _,n:=range adj[x] {
			if n==target { return true }
			if !seen[n] { seen[n]=true;queue=append(queue,n) }
		}
	}
	return false
}
func sortedKeys[V any](m map[string]V) []string {
	keys:=make([]string,0,len(m))
	for k:=range m { keys=append(keys,k) }
	sort.Strings(keys)
	return keys
}
func requiredPairs(events []Event) ([]pair,error) {
	byEntity:=map[string][]Event{}
	byActivity:=map[string][]Event{}
	starts:=map[string][]Event{}
	ends:=map[string][]Event{}
	for _,e:=range events {
		switch e.Kind {
		case EntityGenerated,EntityUsed,EntityInvalidated:
			if e.SubjectID=="" { return nil,fmt.Errorf("entity event missing subject: %s",e.ID) }
			if (e.Kind==EntityGenerated||e.Kind==EntityUsed) && e.ActivityID=="" { return nil,fmt.Errorf("entity generation/use missing activity: %s",e.ID) }
			byEntity[e.SubjectID]=append(byEntity[e.SubjectID],e)
			if e.ActivityID!="" { byActivity[e.ActivityID]=append(byActivity[e.ActivityID],e) }
		case ActivityStarted:
			if e.SubjectID==""||e.ActivityID!=e.SubjectID { return nil,fmt.Errorf("activity-start subject mismatch: %s",e.ID) }
			starts[e.SubjectID]=append(starts[e.SubjectID],e)
		case ActivityEnded:
			if e.SubjectID==""||e.ActivityID!=e.SubjectID { return nil,fmt.Errorf("activity-end subject mismatch: %s",e.ID) }
			ends[e.SubjectID]=append(ends[e.SubjectID],e)
		default:
			return nil,fmt.Errorf("unknown event kind: %s",e.Kind)
		}
	}
	pairs:=[]pair{}
	for _,entity:=range sortedKeys(byEntity) {
		es:=byEntity[entity]
		var gens,invalids,uses []Event
		for _,e:=range es {
			switch e.Kind { case EntityGenerated: gens=append(gens,e);case EntityUsed:uses=append(uses,e);case EntityInvalidated:invalids=append(invalids,e) }
		}
		if len(gens)>1 { return nil,fmt.Errorf("multiple generation events require an explicit simultaneous-generation profile: %s",entity) }
		if len(invalids)>1 { return nil,fmt.Errorf("multiple invalidation events unsupported in v0.1: %s",entity) }
		if (len(uses)>0 || len(invalids)>0) && len(gens)!=1 {
			return nil,fmt.Errorf("entity use/invalidation has no unique authenticated generation event: %s",entity)
		}
		if len(gens)==1 {
			for _,u:=range uses { pairs=append(pairs,pair{gens[0].ID,u.ID}) }
			for _,inv:=range invalids { pairs=append(pairs,pair{gens[0].ID,inv.ID}) }
		}
		if len(invalids)==1 {
			for _,u:=range uses { pairs=append(pairs,pair{u.ID,invalids[0].ID}) }
		}
	}
	activities:=map[string]bool{}
	for a:=range byActivity { activities[a]=true }
	for a:=range starts { activities[a]=true }
	for a:=range ends { activities[a]=true }
	for _,activity:=range sortedKeys(activities) {
		ss,ee:=starts[activity],ends[activity]
		if len(ss)!=1||len(ee)!=1 { return nil,fmt.Errorf("activity requires exactly one start and end event: %s",activity) }
		pairs=append(pairs,pair{ss[0].ID,ee[0].ID})
		for _,e:=range byActivity[activity] { pairs=append(pairs,pair{ss[0].ID,e.ID},pair{e.ID,ee[0].ID}) }
	}
	sort.Slice(pairs,func(i,j int)bool{if pairs[i].before==pairs[j].before{return pairs[i].after<pairs[j].after};return pairs[i].before<pairs[j].before})
	return pairs,nil
}

// Evaluate reconstructs only the declared event ordering. Same-source order
// comes from authenticated monotonic sequence values. Cross-source order
// requires a separately signed witness whose control root differs from both
// endpoint sources. It deliberately ignores physical timestamps.
func Evaluate(in Input,pins map[string]VerifierPin) Result {
	checked:=[]string{"source-local-sequence-domains","exact-event-records"}
	if in.EvaluationAt<=0 { return finish(Indeterminate,"evaluation time missing",0,0,checked...) }
	reports:=reportsIndex(in.Reports)
	conflicts:=reportConflicts(in.Reports)
	sources:=map[string]Source{}
	sourceInput:=append([]Source(nil),in.Sources...)
	sort.Slice(sourceInput,func(i,j int)bool{return sourceInput[i].ID<sourceInput[j].ID})
	for _,s:=range sourceInput {
		if !nonempty(s.ID,s.ControlRoot)||!s.IdentityDigest.Valid() || s.IdentityDigest.Subject!=s.ID || s.IdentityDigest.Purpose!="source-identity" ||
			!s.ControlRootDigest.Valid() || s.ControlRootDigest.Subject!=s.ID || s.ControlRootDigest.Purpose!="control-root-membership" {
			return finish(Indeterminate,"source identity/control-root binding incomplete",0,0,checked...)
		}
		if _,ok:=sources[s.ID];ok { return finish(Divergent,"duplicate source ID: "+s.ID,0,0,checked...) }
		sources[s.ID]=s
		d,why:=reportFor("source:"+s.ID,s.IdentityDigest,reports,conflicts,pins,in.EvaluationAt)
		if d!=Qualified { return finish(d,why,0,0,checked...) }
	}
	if len(sources)==0||len(in.Events)==0 { return finish(Indeterminate,"sources or events missing",0,0,checked...) }
	events:=map[string]Event{}
	seqOwner:=map[string]string{}
	bySource:=map[string][]Event{}
	eventInput:=append([]Event(nil),in.Events...)
	sort.Slice(eventInput,func(i,j int)bool{return eventInput[i].ID<eventInput[j].ID})
	for _,e:=range eventInput {
		if !nonempty(e.ID,e.SubjectID,e.SourceID)||e.Sequence==0||!e.RecordDigest.Valid()||e.RecordDigest.Subject!=e.ID||e.RecordDigest.Purpose!="event" { return finish(Indeterminate,"event identity/source/sequence/digest incomplete",0,0,checked...) }
		if _,ok:=events[e.ID];ok { return finish(Divergent,"duplicate event ID: "+e.ID,0,0,checked...) }
		if _,ok:=sources[e.SourceID];!ok { return finish(Indeterminate,"event source is not in the verified source set: "+e.ID,0,0,checked...) }
		key:=fmt.Sprintf("%s\x00%d",e.SourceID,e.Sequence)
		if prior,ok:=seqOwner[key];ok && prior!=e.ID { return finish(Divergent,"source sequence reused by different event IDs: "+e.SourceID,0,0,checked...) }
		seqOwner[key]=e.ID
		d,why:=reportFor("event:"+e.ID,e.RecordDigest,reports,conflicts,pins,in.EvaluationAt)
		if d!=Qualified { return finish(d,why,0,0,checked...) }
		events[e.ID]=e
		bySource[e.SourceID]=append(bySource[e.SourceID],e)
	}
	checked=append(checked,"verified-event-records","unique-source-sequence-positions")
	adj:=map[string][]string{}
	indegree:=map[string]int{}
	for id:=range events { indegree[id]=0 }
	for _,sourceID:=range sortedKeys(bySource) {
		es:=bySource[sourceID]
		sort.Slice(es,func(i,j int)bool{return es[i].Sequence<es[j].Sequence})
		for i:=1;i<len(es);i++ { addEdge(adj,indegree,es[i-1].ID,es[i].ID) }
	}
	witnessInput:=append([]OrderWitness(nil),in.Witnesses...)
	sort.Slice(witnessInput,func(i,j int)bool{return witnessInput[i].ID<witnessInput[j].ID})
	witnessIDs:=map[string]bool{}
	for _,w:=range witnessInput {
		if !nonempty(w.ID,w.BeforeEventID,w.AfterEventID,w.WitnessSourceID)||!w.EvidenceDigest.Valid()||w.EvidenceDigest.Subject!=w.ID||w.EvidenceDigest.Purpose!="order-witness" { return finish(Indeterminate,"order witness incomplete",0,0,checked...) }
		if witnessIDs[w.ID] { return finish(Divergent,"duplicate order witness ID: "+w.ID,0,0,checked...) }
		witnessIDs[w.ID]=true
		before,bok:=events[w.BeforeEventID];after,aok:=events[w.AfterEventID];ws,wok:=sources[w.WitnessSourceID]
		if !bok||!aok||!wok { return finish(Indeterminate,"order witness references missing event/source: "+w.ID,0,0,checked...) }
		bs,as:=sources[before.SourceID],sources[after.SourceID]
		d,why:=reportFor("witness:"+w.ID,w.EvidenceDigest,reports,conflicts,pins,in.EvaluationAt)
		if d!=Qualified { return finish(d,why,0,0,checked...) }
		if before.SourceID==after.SourceID {
			if before.Sequence<after.Sequence { continue }
			return finish(Divergent,"order witness contradicts authenticated same-source sequence: "+w.ID,0,0,checked...)
		}
		if ws.ControlRoot==bs.ControlRoot||ws.ControlRoot==as.ControlRoot { return finish(Indeterminate,"cross-source order witness shares an endpoint control root: "+w.ID,0,0,checked...) }
		addEdge(adj,indegree,w.BeforeEventID,w.AfterEventID)
	}
	checked=append(checked,"independent-cross-source-witnesses")
	queue:=make([]string,0,len(events))
	for id,d:=range indegree { if d==0 { queue=append(queue,id) } }
	visited:=0
	for len(queue)>0 {
		x:=queue[0];queue=queue[1:];visited++
		for _,y:=range adj[x] { indegree[y]--;if indegree[y]==0 { queue=append(queue,y) } }
	}
	if visited!=len(events) { return finish(Divergent,"event-order graph contains a cycle",0,0,checked...) }
	checked=append(checked,"acyclic-event-order-graph")
	required,err:=requiredPairs(in.Events)
	if err!=nil { return finish(Indeterminate,err.Error(),0,0,checked...) }
	proven:=0
	for _,p:=range required {
		before,after:=events[p.before],events[p.after]
		if before.SourceID==after.SourceID {
			if before.Sequence<after.Sequence { proven++;continue }
			return finish(Denied,"required event order is reversed in one authenticated source sequence: "+p.before+" must precede "+p.after,len(required),proven,checked...)
		}
		if hasPath(adj,p.before,p.after) { proven++;continue }
		if hasPath(adj,p.after,p.before) { return finish(Denied,"required event order is reversed: "+p.before+" must precede "+p.after,len(required),proven,checked...) }
		return finish(Indeterminate,"required event order is not established: "+p.before+" -> "+p.after,len(required),proven,checked...)
	}
	checked=append(checked,"generation-use-invalidation-order","activity-boundary-order")
	return finish(Qualified,"all declared PROV-style event-order constraints are established within the supplied evidence boundary",len(required),proven,checked...)
}
func reportsIndex(in []VerifierReport) map[string]VerifierReport {
	m:=map[string]VerifierReport{}
	for _,r:=range in { if _,ok:=m[r.SubjectID];!ok { m[r.SubjectID]=r } }
	return m
}
func reportConflicts(in []VerifierReport) map[string]bool {
	m:=map[string]VerifierReport{};c:=map[string]bool{}
	for _,r:=range in { if prior,ok:=m[r.SubjectID];ok&&prior!=r { c[r.SubjectID]=true } else if !ok { m[r.SubjectID]=r } }
	return c
}
