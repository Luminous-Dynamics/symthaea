package causalgraph

import (
	"reflect"
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
type NodeKind string
const (
	Entity NodeKind = "entity"
	Activity NodeKind = "activity"
	Agent NodeKind = "agent"
)
type EdgeKind string
const (
	Used EdgeKind = "used"
	Generated EdgeKind = "generated"
	DerivedFrom EdgeKind = "derived_from"
	AuthorizedBy EdgeKind = "authorized_by"
	InformedBy EdgeKind = "informed_by"
	AssociatedWith EdgeKind = "associated_with"
	AttributedTo EdgeKind = "attributed_to"
	ObservedBy EdgeKind = "observed_by"
	Precedes EdgeKind = "precedes"
	Supersedes EdgeKind = "supersedes"
)

type DigestRef struct {
	Algorithm string `json:"algorithm"`
	Profile string `json:"profile"`
	Version string `json:"version"`
	PayloadType string `json:"payload_type"`
	Canonicalization string `json:"canonicalization"`
	Purpose string `json:"purpose"`
	Subject string `json:"subject"`
	Value string `json:"value"`
}
func (d DigestRef) Valid() bool {
	return nonempty(d.Algorithm,d.Profile,d.Version,d.PayloadType,d.Canonicalization,d.Purpose,d.Subject,d.Value)
}
func (d DigestRef) Equal(other DigestRef) (bool,bool) {
	if d.Algorithm!=other.Algorithm || d.Profile!=other.Profile || d.Version!=other.Version ||
		d.PayloadType!=other.PayloadType || d.Canonicalization!=other.Canonicalization ||
		d.Purpose!=other.Purpose || d.Subject!=other.Subject { return false,false }
	return d.Value==other.Value,true
}
type Node struct {
	ID string `json:"id"`
	Kind NodeKind `json:"kind"`
	Digest *DigestRef `json:"digest,omitempty"`
	ExternalRoot bool `json:"external_root,omitempty"`
	RequiresVerification bool `json:"requires_verification,omitempty"`
	DeclaredInputs []string `json:"declared_inputs,omitempty"`
	SourceID string `json:"source_id,omitempty"`
}
type Edge struct {
	ID string `json:"id"`
	Kind EdgeKind `json:"kind"`
	From string `json:"from"`
	To string `json:"to"`
	EvidenceID string `json:"evidence_id"`
}
type Manifest struct {
	ActivityID string `json:"activity_id"`
	CollectorID string `json:"collector_id"`
	SourceID string `json:"source_id"`
	SnapshotID string `json:"snapshot_id"`
	QueryID string `json:"query_id"`
	StartCursor string `json:"start_cursor"`
	EndCursor string `json:"end_cursor"`
	TerminalWatermark string `json:"terminal_watermark"`
	ObservedInputs []string `json:"observed_inputs"`
	Complete bool `json:"complete"`
	Gaps []string `json:"gaps,omitempty"`
	ObservedAt int64 `json:"observed_at_micros"`
}
type VerifierReport struct {
	SubjectID string `json:"subject_id"`
	InputDigest DigestRef `json:"input_digest"`
	Status string `json:"status"`
	VerifierID string `json:"verifier_id"`
	Version string `json:"version"`
	BuildDigest string `json:"build_digest"`
	TrustStoreDigest string `json:"trust_store_digest"`
	VerifiedAt int64 `json:"verified_at_micros"`
	FreshUntil int64 `json:"fresh_until_micros"`
}
type VerifierPin struct {
	ID string
	Version string
	BuildDigest string
	TrustStoreDigest string
}
type Graph struct {
	Nodes []Node `json:"nodes"`
	Edges []Edge `json:"edges"`
	Manifests []Manifest `json:"manifests"`
	Reports []VerifierReport `json:"verifier_reports"`
}
type Result struct {
	Decision Decision `json:"decision"`
	Reason string `json:"reason"`
	Checked []string `json:"checked"`
}
func nonempty(xs ...string) bool {
	for _,x:=range xs { if strings.TrimSpace(x)=="" { return false } }
	return true
}
func result(d Decision,r string,c ...string) Result {
	return Result{Decision:d,Reason:r,Checked:append([]string(nil),c...)}
}
func equalSet(a,b []string) bool {
	if len(a)!=len(b) { return false }
	x:=append([]string(nil),a...); y:=append([]string(nil),b...)
	sort.Strings(x); sort.Strings(y)
	for i:=range x { if x[i]!=y[i] { return false } }
	return true
}
func edgeTypeValid(e Edge, nodes map[string]Node) bool {
	a,okA:=nodes[e.From]; b,okB:=nodes[e.To]
	if !okA||!okB { return false }
	switch e.Kind {
	case Used,ObservedBy: return a.Kind==Entity && b.Kind==Activity
	case Generated: return a.Kind==Activity && b.Kind==Entity
	case DerivedFrom,Supersedes: return a.Kind==Entity && b.Kind==Entity
	case AuthorizedBy: return a.Kind==Entity && b.Kind==Activity
	case InformedBy,Precedes: return a.Kind==Activity && b.Kind==Activity
	case AssociatedWith: return a.Kind==Activity && b.Kind==Agent
	case AttributedTo: return a.Kind==Entity && b.Kind==Agent
	default: return false
	}
}
func verifyNode(n Node, reports map[string]VerifierReport, conflicts map[string]bool, pins map[string]VerifierPin, observedAt int64) (Decision,string) {
	if !n.RequiresVerification { return Qualified,"" }
	if n.Digest==nil || !n.Digest.Valid() { return Indeterminate,"required evidence digest missing: "+n.ID }
	if conflicts[n.ID] { return Divergent,"conflicting verifier reports: "+n.ID }
	r,found:=reports[n.ID]
	if !found { return Indeterminate,"required verifier report missing: "+n.ID }
	if r.Status=="invalid" { return Denied,"source evidence signature/profile rejected: "+n.ID }
	if r.Status!="verified" { return Indeterminate,"source evidence not cryptographically verified: "+n.ID }
	p,ok:=pins[r.VerifierID]
	if !ok || p.ID!=r.VerifierID || p.Version!=r.Version || p.BuildDigest!=r.BuildDigest ||
		p.TrustStoreDigest!=r.TrustStoreDigest || !nonempty(r.VerifierID,r.Version,r.BuildDigest,r.TrustStoreDigest) {
		return Indeterminate,"verifier identity/build/trust-store not pinned: "+n.ID
	}
	if r.VerifiedAt<=0 || r.VerifiedAt>observedAt || r.FreshUntil<=observedAt {
		return Indeterminate,"verifier result stale or outside observation boundary: "+n.ID
	}
	eq,comparable:=r.InputDigest.Equal(*n.Digest)
	if !comparable { return Indeterminate,"verifier digest profile incomparable: "+n.ID }
	if !eq { return Denied,"verifier report not bound to exact entity bytes: "+n.ID }
	return Qualified,""
}

// Evaluate checks graph structure, evidence references, bounded activity
// manifests, and pinned verifier reports. It does not verify signatures itself,
// establish factual truth, or prove completeness outside the declared capture
// scope. Causal edges are separate from supersession/history edges.
func Evaluate(g Graph,pins map[string]VerifierPin) Result {
	checked:=[]string{"typed-entity-activity-agent-model"}
	nodes:=map[string]Node{}
	for _,n:=range g.Nodes {
		if !nonempty(n.ID) || (n.Kind!=Entity&&n.Kind!=Activity&&n.Kind!=Agent) {
			return result(Indeterminate,"node identity or kind invalid",checked...)
		}
		if old,ok:=nodes[n.ID]; ok {
			if !reflect.DeepEqual(old,n) { return result(Divergent,"node identifier reused with conflicting content: "+n.ID,checked...) }
			return result(Divergent,"duplicate node identity: "+n.ID,checked...)
		}
		if n.Kind==Entity && n.Digest!=nil && !n.Digest.Valid() {
			return result(Indeterminate,"entity digest malformed: "+n.ID,checked...)
		}
		nodes[n.ID]=n
	}
	if len(nodes)==0 { return result(Indeterminate,"empty provenance graph",checked...) }
	checked=append(checked,"unique-node-identities","typed-digests")
	edges:=map[string]Edge{}
	adj:=map[string][]string{}
	incomingUsed:=map[string][]string{}
	generatedBy:=map[string]int{}
	for _,e:=range g.Edges {
		if !nonempty(e.ID,e.From,e.To,e.EvidenceID) { return result(Indeterminate,"edge missing identity, endpoint or evidence reference",checked...) }
		if old,ok:=edges[e.ID];ok {
			if old!=e { return result(Divergent,"edge identifier reused with conflicting content: "+e.ID,checked...) }
			return result(Divergent,"duplicate edge identity: "+e.ID,checked...)
		}
		if !edgeTypeValid(e,nodes) { return result(Denied,"edge endpoint type/reference invalid: "+e.ID,checked...) }
		ev,ok:=nodes[e.EvidenceID]
		if !ok || ev.Kind!=Entity { return result(Indeterminate,"edge evidence entity missing: "+e.ID,checked...) }
		if !ev.RequiresVerification { return result(Indeterminate,"edge evidence lacks a required authenticated source-verifier claim: "+e.ID,checked...) }
		edges[e.ID]=e
		if e.Kind!=Supersedes && e.Kind!=AttributedTo { adj[e.From]=append(adj[e.From],e.To) }
		if e.Kind==Used { incomingUsed[e.To]=append(incomingUsed[e.To],e.From) }
		if e.Kind==Generated { generatedBy[e.To]++ }
	}
	checked=append(checked,"edge-endpoints","edge-evidence-references")
	attributedTo:=map[string]string{}
	for _,e:=range g.Edges {
		if e.Kind==AttributedTo {
			if prior,ok:=attributedTo[e.From];ok && prior!=e.To { return result(Divergent,"entity attributed to conflicting source agents: "+e.From,checked...) }
			attributedTo[e.From]=e.To
		}
	}
	for id,n:=range nodes {
		if n.Kind==Entity {
			if generatedBy[id]>1 { return result(Divergent,"immutable entity has conflicting generation edges: "+id,checked...) }
			if n.ExternalRoot {
				agent,ok:=nodes[n.SourceID]
				if attributedTo[id]!="" && attributedTo[id]!=n.SourceID {
					return result(Divergent,"external entity source attribution conflicts with its declared source: "+id,checked...)
				}
				if !nonempty(n.SourceID) || !ok || agent.Kind!=Agent || !agent.RequiresVerification || attributedTo[id]!=n.SourceID {
					return result(Indeterminate,"external entity lacks a verifier-qualified source attribution: "+id,checked...)
				}
			}
			if generatedBy[id]==0 && !n.ExternalRoot {
				used:=false
				for _,e:=range g.Edges { if e.Kind==Used&&e.From==id { used=true;break } }
				if used { return result(Indeterminate,"used entity has no generation event or declared external origin: "+id,checked...) }
			}
		}
	}
	state:=map[string]uint8{}
	var visit func(string) bool
	visit=func(v string) bool {
		if state[v]==1 { return false }
		if state[v]==2 { return true }
		state[v]=1
		for _,w:=range adj[v] { if !visit(w) { return false } }
		state[v]=2
		return true
	}
	for id:=range nodes { if !visit(id) { return result(Denied,"causal/ordering edge cycle detected",checked...) } }
	checked=append(checked,"acyclic-causal-partial-order")
	manifestByActivity:=map[string]Manifest{}
	for _,m:=range g.Manifests {
		if _,ok:=nodes[m.ActivityID];!ok || nodes[m.ActivityID].Kind!=Activity {
			return result(Indeterminate,"manifest refers to missing/non-activity node: "+m.ActivityID,checked...)
		}
		if _,ok:=manifestByActivity[m.ActivityID];ok { return result(Divergent,"multiple manifests for one activity without merge profile: "+m.ActivityID,checked...) }
		manifestByActivity[m.ActivityID]=m
	}
	for id,n:=range nodes {
		if n.Kind!=Activity { continue }
		m,ok:=manifestByActivity[id]
		if !ok { return result(Indeterminate,"activity has no independent capture manifest: "+id,checked...) }
		if !nonempty(m.CollectorID,m.SourceID,m.SnapshotID,m.QueryID,m.StartCursor,m.EndCursor,m.TerminalWatermark) || !m.Digest.Valid() || m.ObservedAt<=0 {
			return result(Indeterminate,"activity capture boundary is incomplete: "+id,checked...)
		}
		collector,collectorOK:=nodes[m.CollectorID]
		source,sourceOK:=nodes[m.SourceID]
		if !collectorOK || collector.Kind!=Agent || !collector.RequiresVerification || !sourceOK || source.Kind!=Agent || !source.RequiresVerification {
			return result(Indeterminate,"capture collector/source identity is not independently verified: "+id,checked...)
		}
		if !m.Complete || len(m.Gaps)>0 { return result(Indeterminate,"activity input capture is incomplete: "+id,checked...) }
		if !equalSet(n.DeclaredInputs,m.ObservedInputs) || !equalSet(m.ObservedInputs,incomingUsed[id]) {
			return result(Divergent,"declared, independently observed and graph-linked input sets differ: "+id,checked...)
		}
		for _,input:=range m.ObservedInputs {
			in,ok:=nodes[input]
			if !ok||in.Kind!=Entity { return result(Indeterminate,"captured input entity missing: "+input,checked...) }
		}
	}
	checked=append(checked,"bounded-independent-activity-manifests","declared-observed-graph-input-equality")
	observedAt:=int64(0)
	for _,m:=range g.Manifests { if m.ObservedAt>observedAt { observedAt=m.ObservedAt } }
	reportIndex:=map[string]VerifierReport{}
	reportConflicts:=map[string]bool{}
	for _,r:=range g.Reports {
		if prior,ok:=reportIndex[r.SubjectID];ok && prior!=r { reportConflicts[r.SubjectID]=true } else if !ok { reportIndex[r.SubjectID]=r }
	}
	for _,n:=range g.Nodes {
		d,why:=verifyNode(n,reportIndex,reportConflicts,pins,observedAt)
		if d!=Qualified { return result(d,why,checked...) }
	}
	for _,m:=range g.Manifests {
		fake:=Node{ID:"manifest:"+m.ActivityID,Kind:Entity,Digest:&m.Digest,RequiresVerification:true}
		d,why:=verifyNode(fake,reportIndex,reportConflicts,pins,observedAt)
		if d!=Qualified { return result(d,"capture manifest "+m.ActivityID+": "+why,checked...) }
	}
	checked=append(checked,"capture-manifest-verifier-reports")
	checked=append(checked,"pinned-source-verifier-reports","exact-source-byte-binding","verifier-freshness")
	return result(Qualified,"causal provenance structure and supplied source-verifier evidence qualify only within the declared capture boundary; factual truth and downstream effects require separate gates",checked...)
}
