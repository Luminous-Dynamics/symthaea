package inventory

import "testing"

func independentDomains(prefix string) Domains {
	return Domains{
		ControlPlane: prefix + "-cp", IdentityRoot: prefix + "-id",
		CredentialAuthority: prefix + "-cred", NetworkPath: prefix + "-net",
		TimeSource: prefix + "-time", AdministrativePath: prefix + "-admin",
		StoragePlane: prefix + "-storage",
	}
}

func fixtureInput() Input {
	return Input{
		ScopeID: "prod/database",
		ExpectedWitnessIDs: []string{"cloud-discovery", "network-discovery"},
		Witnesses: []Witness{
			{ID:"cloud-discovery", ScopeID:"prod/database", TrustQualified:true, Complete:true, Members:[]string{"db/orders","db/users"}, Domains:independentDomains("cloud")},
			{ID:"network-discovery", ScopeID:"prod/database", TrustQualified:true, Complete:true, Members:[]string{"db/orders","db/users"}, Domains:independentDomains("network")},
		},
		CapturedMembers: []string{"db/orders","db/users"},
		Capabilities: []Capability{
			{MemberID:"db/orders",Status:CapabilitySupported},
			{MemberID:"db/users",Status:CapabilitySupported},
		},
	}
}

func TestIndependentCompleteInventory(t *testing.T) {
	r,err:=Evaluate(fixtureInput())
	if err!=nil { t.Fatal(err) }
	if r.Verdict!=Qualified || !r.InventoryCorroborated || r.Coverage.Rate!=1 { t.Fatalf("unexpected: %+v",r) }
}

func TestOneWitnessAndDroppedWitnessCannotQualify(t *testing.T) {
	in:=fixtureInput()
	in.Witnesses=in.Witnesses[:1]
	r,err:=Evaluate(in); if err!=nil { t.Fatal(err) }
	if r.Verdict==Qualified || r.InventoryCorroborated { t.Fatalf("single witness qualified: %+v",r) }
	in=fixtureInput()
	in.ExpectedWitnessIDs=append(in.ExpectedWitnessIDs,"preregistered-cmdb")
	r,err=Evaluate(in); if err!=nil { t.Fatal(err) }
	if r.Verdict==Qualified || r.InventoryCorroborated { t.Fatalf("dropped witness accepted: %+v",r) }
}

func TestSharedAndMissingFailureDomainsFailIndependence(t *testing.T) {
	in:=fixtureInput()
	in.Witnesses[1].Domains.IdentityRoot=in.Witnesses[0].Domains.IdentityRoot
	r,err:=Evaluate(in); if err!=nil { t.Fatal(err) }
	if r.Verdict!=Indeterminate || r.WitnessPairs[0].State!=SharedDomain { t.Fatalf("shared root accepted: %+v",r) }
	in=fixtureInput()
	in.Witnesses[1].Domains.TimeSource=""
	r,err=Evaluate(in); if err!=nil { t.Fatal(err) }
	if r.Verdict!=Indeterminate || r.WitnessPairs[0].State!=UnknownDomains { t.Fatalf("missing time domain accepted: %+v",r) }
}

func TestInventoryDisagreementAndUntrustedWitnessFailClosed(t *testing.T) {
	in:=fixtureInput()
	in.Witnesses[1].Members=append(in.Witnesses[1].Members,"db/payments")
	r,err:=Evaluate(in); if err!=nil { t.Fatal(err) }
	if r.Verdict!=Indeterminate || r.InventoryCorroborated { t.Fatalf("divergent inventories qualified: %+v",r) }
	in=fixtureInput()
	in.Witnesses[0].TrustQualified=false
	r,err=Evaluate(in); if err!=nil { t.Fatal(err) }
	if r.Verdict!=Indeterminate || r.InventoryCorroborated { t.Fatalf("untrusted witness qualified: %+v",r) }
}

func TestMissingUnsupportedUnknownAndExtraCaptureStates(t *testing.T) {
	in:=fixtureInput()
	in.CapturedMembers=[]string{"db/orders"}
	r,err:=Evaluate(in); if err!=nil { t.Fatal(err) }
	if r.Verdict!=Blocked || r.MissingCount!=1 { t.Fatalf("missing member not blocked: %+v",r) }
	for _,status:=range []CapabilityStatus{CapabilityUnsupported,CapabilityUnknown} {
		in=fixtureInput()
		in.Capabilities[1].Status=status
		r,err=Evaluate(in); if err!=nil { t.Fatal(err) }
		if r.Verdict!=Indeterminate || r.Coverage.Numerator!=1 { t.Fatalf("%s incorrectly counted as covered: %+v",status,r) }
	}
	in=fixtureInput()
	in.CapturedMembers=append(in.CapturedMembers,"db/undocumented")
	r,err=Evaluate(in); if err!=nil { t.Fatal(err) }
	if r.Verdict!=Indeterminate || len(r.CapturedExtras)!=1 { t.Fatalf("extra capture silently accepted: %+v",r) }
}

func TestExplicitExclusionIsVisibleButNotCovered(t *testing.T) {
	in:=fixtureInput()
	in.CapturedMembers=[]string{"db/orders"}
	in.Exclusions=[]Exclusion{{MemberID:"db/users",Authorized:true,PolicyID:"scope-waiver-7",Reason:"documented outside scope"}}
	r,err:=Evaluate(in); if err!=nil { t.Fatal(err) }
	if r.Verdict!=Qualified || r.ScopeStatus!="qualified_with_explicit_exclusions" || r.ExcludedCount!=1 || r.Coverage.Rate!=0.5 { t.Fatalf("exclusion hidden or counted as covered: %+v",r) }
}

func TestContradictoryAndUnauthorizedExclusionsCannotHideCoverage(t *testing.T) {
	in:=fixtureInput()
	in.Exclusions=[]Exclusion{{MemberID:"db/users",Authorized:true,PolicyID:"scope-1",Reason:"excluded"}}
	r,err:=Evaluate(in); if err!=nil { t.Fatal(err) }
	if r.Verdict!=Indeterminate || r.UnknownCount!=1 { t.Fatalf("excluded plus captured contradiction accepted: %+v",r) }
	in=fixtureInput()
	in.CapturedMembers=[]string{"db/orders"}
	in.Exclusions=[]Exclusion{{MemberID:"db/users",Authorized:false}}
	r,err=Evaluate(in); if err!=nil { t.Fatal(err) }
	if r.Verdict!=Indeterminate || r.UnknownCount!=1 { t.Fatalf("unauthorized exclusion accepted: %+v",r) }
}
