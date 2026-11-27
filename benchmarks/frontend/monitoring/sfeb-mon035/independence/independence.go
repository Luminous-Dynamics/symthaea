// Package independence evaluates whether two recovery evidence observers are
// separated across all declared critical failure domains. It does not authenticate
// inventory attestations or prove that the declared dependency graph is complete.
package independence

import (
	"encoding/hex"
	"sort"
	"strings"
	"time"
	"unicode"
)

const (
	Profile = "mon035-observer-independence-input-v1"
	Version = "1.0.0"
	GradeRuntimeIndependent = "runtime_independently_verified"
	MaxNodes = 50000
	MaxEdges = 250000
	MaxIdentifierBytes = 256
)

var criticalDomainClasses = []string{
	"operator", "identity_provider", "control_plane", "compute_host",
	"storage", "network", "time_source", "signer",
}

type Verdict string
const (
	Qualified Verdict = "qualified"
	Blocked Verdict = "blocked"
	Indeterminate Verdict = "indeterminate"
)

type Node struct {
	ID string `json:"id"`
	FailureClass string `json:"failure_class,omitempty"`
	FailureDomainID string `json:"failure_domain_id,omitempty"`
	DependsOn []string `json:"depends_on,omitempty"`
}

type Observer struct {
	ID string `json:"id"`
	RootNodeID string `json:"root_node_id"`
	InventoryComplete bool `json:"inventory_complete"`
	InventoryAttestationSHA256 string `json:"inventory_attestation_sha256"`
	EvidenceGrade string `json:"evidence_grade"`
	ObservedAt time.Time `json:"observed_at"`
	ExpiresAt time.Time `json:"expires_at"`
}

type Input struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	Now time.Time `json:"now"`
	Nodes []Node `json:"nodes"`
	Observers []Observer `json:"observers"`
}

type DomainRef struct {
	Class string `json:"class"`
	DomainID string `json:"domain_id"`
}

type ObserverAssessment struct {
	ObserverID string `json:"observer_id"`
	RootNodeID string `json:"root_node_id"`
	Eligible bool `json:"eligible"`
	ClosureNodeCount int `json:"closure_node_count"`
	Domains []DomainRef `json:"domains"`
	ReasonCodes []string `json:"reason_codes"`
}

type Result struct {
	Verdict Verdict `json:"verdict"`
	Independent bool `json:"independent"`
	ObserverIDs []string `json:"observer_ids"`
	SharedDomains []DomainRef `json:"shared_domains"`
	ObserverAssessments []ObserverAssessment `json:"observer_assessments"`
	ReasonCodes []string `json:"reason_codes"`
}

type evalFlags struct {
	blocked bool
	unknown bool
	reasons map[string]struct{}
}
func (f *evalFlags) block(code string) { f.blocked=true; f.reasons[code]=struct{}{} }
func (f *evalFlags) markUnknown(code string) { f.unknown=true; f.reasons[code]=struct{}{} }
func (f *evalFlags) output() []string {
	out:=make([]string,0,len(f.reasons))
	for code:=range f.reasons { out=append(out,code) }
	sort.Strings(out)
	return out
}
func supportedClass(value string) bool {
	for _, class:=range criticalDomainClasses { if class==value { return true } }
	return false
}
func validSHA256(value string) bool {
	if len(value)!=64 { return false }
	_,err:=hex.DecodeString(value)
	return err==nil
}
func validIdentifier(value string) bool {
	return value != "" && len(value) <= MaxIdentifierBytes &&
		strings.TrimSpace(value) == value &&
		return value != "" && len(value) <= MaxIdentifierBytes &&
			strings.IndexFunc(value, func(r rune) bool { return unicode.IsSpace(r) || unicode.IsControl(r) }) < 0
}

func Evaluate(in Input) Result {
	result:=Result{
		Verdict:Qualified,
		ObserverIDs:[]string{},
		SharedDomains:[]DomainRef{},
		ObserverAssessments:[]ObserverAssessment{},
		ReasonCodes:[]string{},
	}
	flags:=&evalFlags{reasons:map[string]struct{}{}}
	if in.Profile!=Profile || in.Version!=Version { flags.block("unsupported_profile_or_version") }
	if in.Now.IsZero() { flags.markUnknown("evaluation_time_missing") }
	if len(in.Observers)<2 { flags.markUnknown("two_observers_required") }
	if len(in.Observers)>2 { flags.block("unsupported_observer_count_v1") }
	if len(in.Nodes)==0 { flags.markUnknown("dependency_inventory_missing") }
	if len(in.Nodes)>MaxNodes { flags.block("dependency_node_limit_exceeded") }

	nodeByID:=make(map[string]Node,len(in.Nodes))
	totalEdges:=0
	for _,node:=range in.Nodes {
		if !validIdentifier(node.ID) {
			flags.block("invalid_dependency_node_id")
			continue
		}
		if _,exists:=nodeByID[node.ID]; exists {
			flags.block("duplicate_dependency_node_id")
			continue
		}
		if (node.FailureClass=="") != (node.FailureDomainID=="") {
			flags.block("incomplete_failure_domain_identity:"+node.ID)
		}
		if node.FailureClass!="" && !supportedClass(node.FailureClass) {
			flags.block("unsupported_failure_domain_class:"+node.ID)
		}
		if node.FailureDomainID!="" && !validIdentifier(node.FailureDomainID) {
			flags.block("invalid_failure_domain_id:"+node.ID)
		}
		totalEdges+=len(node.DependsOn)
		nodeByID[node.ID]=node
	}
	if totalEdges>MaxEdges { flags.block("dependency_edge_limit_exceeded") }

	observerIDs:=map[string]struct{}{}
	assessments:=make([]ObserverAssessment,0,len(in.Observers))
	for _,observer:=range in.Observers {
		local:=&evalFlags{reasons:map[string]struct{}{}}
		assessment:=ObserverAssessment{
			ObserverID:observer.ID,RootNodeID:observer.RootNodeID,
			Domains:[]DomainRef{},ReasonCodes:[]string{},
		}
		result.ObserverIDs=append(result.ObserverIDs,observer.ID)
		if !validIdentifier(observer.ID) {
			local.block("invalid_observer_id")
		} else if _,duplicate:=observerIDs[observer.ID]; duplicate {
			local.block("duplicate_observer_id")
		} else { observerIDs[observer.ID]=struct{}{} }
		if observer.RootNodeID=="" { local.markUnknown("observer_root_missing") } else if !validIdentifier(observer.RootNodeID) { local.block("observer_root_invalid") }
		if !observer.InventoryComplete { local.markUnknown("observer_inventory_not_attested_complete") }
		if observer.InventoryAttestationSHA256=="" {
			local.markUnknown("inventory_attestation_digest_missing")
		} else if !validSHA256(observer.InventoryAttestationSHA256) {
			local.block("inventory_attestation_digest_invalid")
		}
		if observer.EvidenceGrade!=GradeRuntimeIndependent { local.markUnknown("observer_evidence_not_runtime_independent") }
		if in.Now.IsZero() || observer.ObservedAt.IsZero() || observer.ExpiresAt.IsZero() {
			local.markUnknown("observer_freshness_unproven")
		} else {
			if observer.ObservedAt.After(in.Now) { local.block("observer_observation_from_future") }
			if observer.ExpiresAt.Before(observer.ObservedAt) { local.block("observer_validity_interval_invalid") }
			if !observer.ExpiresAt.After(in.Now) { local.markUnknown("observer_inventory_expired") }
		}
		closure,dfsFlags:=dependencyClosure(observer.RootNodeID,nodeByID)
		for code:=range dfsFlags.reasons { local.reasons[code+":"+observer.ID]=struct{}{} }
		local.blocked=local.blocked || dfsFlags.blocked
		local.unknown=local.unknown || dfsFlags.unknown
		assessment.ClosureNodeCount=len(closure)
		domainSet:=map[string]map[string]struct{}{}
		for nodeID:=range closure {
			node:=nodeByID[nodeID]
			if node.FailureClass=="" { continue }
			if domainSet[node.FailureClass]==nil { domainSet[node.FailureClass]=map[string]struct{}{} }
			domainSet[node.FailureClass][node.FailureDomainID]=struct{}{}
		}
		for _,class:=range criticalDomainClasses {
			domains:=domainSet[class]
			if len(domains)==0 { local.markUnknown("critical_domain_missing:"+class); continue }
			for domainID:=range domains { assessment.Domains=append(assessment.Domains,DomainRef{Class:class,DomainID:domainID}) }
		}
		sort.Slice(assessment.Domains,func(i,j int)bool{
			if assessment.Domains[i].Class==assessment.Domains[j].Class { return assessment.Domains[i].DomainID<assessment.Domains[j].DomainID }
			return assessment.Domains[i].Class<assessment.Domains[j].Class
		})
		assessment.ReasonCodes=local.output()
		assessment.Eligible=!local.blocked && !local.unknown && len(assessment.Domains)>=len(criticalDomainClasses)
		if !assessment.Eligible {
			for code:=range local.reasons { flags.reasons[code+":"+observer.ID]=struct{}{} }
			flags.blocked=flags.blocked || local.blocked
			flags.unknown=flags.unknown || local.unknown
		}
		assessments=append(assessments,assessment)
	}
	sort.Strings(result.ObserverIDs)
	sort.Slice(assessments,func(i,j int)bool{return assessments[i].ObserverID<assessments[j].ObserverID})
	result.ObserverAssessments=assessments
	if len(assessments)==2 && assessments[0].RootNodeID!="" && assessments[0].RootNodeID==assessments[1].RootNodeID {
		flags.block("observers_share_same_root_node")
	}
	if len(assessments)==2 && assessments[0].Eligible && assessments[1].Eligible {
		for _,class:=range criticalDomainClasses {
			left,right:=map[string]struct{}{},map[string]struct{}{}
			for _,domain:=range assessments[0].Domains { if domain.Class==class { left[domain.DomainID]=struct{}{} } }
			for _,domain:=range assessments[1].Domains { if domain.Class==class { right[domain.DomainID]=struct{}{} } }
			for domainID:=range left {
				if _,shared:=right[domainID]; shared { result.SharedDomains=append(result.SharedDomains,DomainRef{Class:class,DomainID:domainID}) }
			}
		}
		sort.Slice(result.SharedDomains,func(i,j int)bool{
			if result.SharedDomains[i].Class==result.SharedDomains[j].Class { return result.SharedDomains[i].DomainID<result.SharedDomains[j].DomainID }
			return result.SharedDomains[i].Class<result.SharedDomains[j].Class
		})
		if len(result.SharedDomains)>0 { flags.block("shared_critical_failure_domain") } else if assessments[0].RootNodeID!=assessments[1].RootNodeID {
			result.Independent=true
		}
	}
	switch {
	case flags.blocked: result.Verdict=Blocked
	case flags.unknown: result.Verdict=Indeterminate
	case result.Independent: result.Verdict=Qualified
	default:
		result.Verdict=Blocked
		flags.block("observers_not_independent")
	}
	result.Independent=result.Verdict==Qualified
	result.ReasonCodes=flags.output()
	return result
}

func dependencyClosure(root string,nodes map[string]Node)(map[string]struct{},*evalFlags){
	closure:=map[string]struct{}{}
	state:=map[string]uint8{}
	flags:=&evalFlags{reasons:map[string]struct{}{}}
	var visit func(string)
	visit=func(id string){
		if state[id]==1 { flags.block("dependency_cycle_detected"); return }
		if state[id]==2 { return }
		node,ok:=nodes[id]
		if !ok { flags.markUnknown("dependency_reference_missing"); return }
		state[id]=1
		closure[id]=struct{}{}
		seen:=map[string]struct{}{}
		for _,child:=range node.DependsOn {
			if !validIdentifier(child) { flags.block("invalid_dependency_reference:"+id); continue }
			if _,duplicate:=seen[child]; duplicate { flags.block("duplicate_dependency_edge:"+id); continue }
			seen[child]=struct{}{}
			visit(child)
		}
		state[id]=2
	}
	if root=="" { flags.markUnknown("dependency_root_missing") } else { visit(root) }
	return closure,flags
}
