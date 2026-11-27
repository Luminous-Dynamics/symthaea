package independence

import (
	"fmt"
	"testing"
	"time"
)

var fixtureTime=time.Date(2026,11,27,12,0,0,0,time.UTC)

func fixture() Input {
	nodes:=[]Node{}
	classes:=[]string{"operator","identity_provider","control_plane","compute_host","storage","network","time_source","signer"}
	for _,observer:=range []string{"a","b"} {
		children:=make([]string,0,len(classes))
		for _,class:=range classes {
			id:=observer+"-"+class
			children=append(children,id)
			nodes=append(nodes,Node{ID:id,FailureClass:class,FailureDomainID:observer+":"+class})
		}
		nodes=append(nodes,Node{ID:observer+"-root",DependsOn:children})
	}
	observers:=[]Observer{
		{ID:"observer-a",RootNodeID:"a-root",InventoryComplete:true,InventoryAttestationSHA256:fmt.Sprintf("%064x",1),EvidenceGrade:GradeRuntimeIndependent,ObservedAt:fixtureTime.Add(-time.Minute),ExpiresAt:fixtureTime.Add(time.Hour)},
		{ID:"observer-b",RootNodeID:"b-root",InventoryComplete:true,InventoryAttestationSHA256:fmt.Sprintf("%064x",2),EvidenceGrade:GradeRuntimeIndependent,ObservedAt:fixtureTime.Add(-time.Minute),ExpiresAt:fixtureTime.Add(time.Hour)},
	}
	return Input{Profile:Profile,Version:Version,Now:fixtureTime,Nodes:nodes,Observers:observers}
}

func containsDomain(domains []DomainRef,class,id string)bool{
	for _,item:=range domains { if item.Class==class && item.DomainID==id { return true } }
	return false
}

func TestIndependentObserversQualifyAcrossAllCriticalDomains(t *testing.T){
	got:=Evaluate(fixture())
	if got.Verdict!=Qualified || !got.Independent || len(got.SharedDomains)!=0 { t.Fatalf("expected independent pair: %+v",got) }
}

func TestSharedIdentityProviderBlocks(t *testing.T){
	in:=fixture()
	for i:=range in.Nodes { if in.Nodes[i].ID=="b-identity_provider" { in.Nodes[i].FailureDomainID="a:identity_provider" } }
	got:=Evaluate(in)
	if got.Verdict!=Blocked || !containsDomain(got.SharedDomains,"identity_provider","a:identity_provider") { t.Fatalf("shared IdP must block: %+v",got) }
}

func TestSharedControlPlaneBlocks(t *testing.T){
	in:=fixture()
	for i:=range in.Nodes { if in.Nodes[i].ID=="b-control_plane" { in.Nodes[i].FailureDomainID="a:control_plane" } }
	got:=Evaluate(in)
	if got.Verdict!=Blocked || !containsDomain(got.SharedDomains,"control_plane","a:control_plane") { t.Fatalf("shared control plane must block: %+v",got) }
}

func TestSharedAdministratorBlocks(t *testing.T){
	in:=fixture()
	for i:=range in.Nodes { if in.Nodes[i].ID=="b-operator" { in.Nodes[i].FailureDomainID="a:operator" } }
	got:=Evaluate(in)
	if got.Verdict!=Blocked || !containsDomain(got.SharedDomains,"operator","a:operator") { t.Fatalf("shared operator must block: %+v",got) }
}

func TestSharedClockAndSignerEachBlock(t *testing.T){
	for _,class:=range []string{"time_source","signer"} {
		in:=fixture()
		for i:=range in.Nodes { if in.Nodes[i].ID=="b-"+class { in.Nodes[i].FailureDomainID="a:"+class } }
		got:=Evaluate(in)
		if got.Verdict!=Blocked || !containsDomain(got.SharedDomains,class,"a:"+class) { t.Fatalf("shared %s must block: %+v",class,got) }
	}
}

func TestIncompleteInventoryIsIndeterminate(t *testing.T){
	in:=fixture(); in.Observers[1].InventoryComplete=false
	if got:=Evaluate(in); got.Verdict!=Indeterminate { t.Fatalf("incomplete inventory must be indeterminate: %+v",got) }
}

func TestMissingDependencyIsIndeterminate(t *testing.T){
	in:=fixture()
	for i:=range in.Nodes { if in.Nodes[i].ID=="b-root" { in.Nodes[i].DependsOn=append(in.Nodes[i].DependsOn,"missing-network") } }
	if got:=Evaluate(in); got.Verdict!=Indeterminate { t.Fatalf("missing dependency must be indeterminate: %+v",got) }
}

func TestDependencyCycleBlocks(t *testing.T){
	in:=fixture()
	in.Nodes=append(in.Nodes,Node{ID:"cycle-x",DependsOn:[]string{"cycle-y"}},Node{ID:"cycle-y",DependsOn:[]string{"cycle-x"}})
	for i:=range in.Nodes { if in.Nodes[i].ID=="a-root" { in.Nodes[i].DependsOn=append(in.Nodes[i].DependsOn,"cycle-x") } }
	if got:=Evaluate(in); got.Verdict!=Blocked { t.Fatalf("cycle must block: %+v",got) }
}

func TestExpiredInventoryIsIndeterminate(t *testing.T){
	in:=fixture(); in.Observers[1].ExpiresAt=fixtureTime.Add(-time.Second)
	if got:=Evaluate(in); got.Verdict!=Indeterminate { t.Fatalf("expired inventory must be indeterminate: %+v",got) }
}

func TestFutureObservationBlocks(t *testing.T){
	in:=fixture(); in.Observers[0].ObservedAt=fixtureTime.Add(time.Second)
	if got:=Evaluate(in); got.Verdict!=Blocked { t.Fatalf("future observation must block: %+v",got) }
}

func TestMissingCriticalDomainIsIndeterminate(t *testing.T){
	in:=fixture()
	for i:=range in.Nodes { if in.Nodes[i].ID=="b-storage" { in.Nodes[i].FailureClass=""; in.Nodes[i].FailureDomainID="" } }
	if got:=Evaluate(in); got.Verdict!=Indeterminate { t.Fatalf("missing domain must be indeterminate: %+v",got) }
}

func TestInvalidDomainClassBlocks(t *testing.T){
	in:=fixture()
	for i:=range in.Nodes { if in.Nodes[i].ID=="a-storage" { in.Nodes[i].FailureClass="region-ish" } }
	if got:=Evaluate(in); got.Verdict!=Blocked { t.Fatalf("unsupported domain class must block: %+v",got) }
}

func TestDuplicateRootBlocks(t *testing.T){
	in:=fixture(); in.Observers[1].RootNodeID=in.Observers[0].RootNodeID
	if got:=Evaluate(in); got.Verdict!=Blocked { t.Fatalf("same root must block: %+v",got) }
}

func TestResourceCapsFailClosed(t *testing.T){
	in:=fixture(); in.Nodes=make([]Node,MaxNodes+1)
	for i:=range in.Nodes { in.Nodes[i].ID=fmt.Sprintf("node-%d",i) }
	if got:=Evaluate(in); got.Verdict!=Blocked { t.Fatalf("node cap breach must block: %+v",got) }
}
