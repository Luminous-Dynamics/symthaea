package matrix

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"reflect"
	"testing"
)

func profileCatalogFixture() ProfileCatalog {
	return ProfileCatalog{SchemaVersion: "mace-profile-catalog-v2", CatalogID: "catalog-2026-11", Obligations: []ProfileObligation{
		{Criterion: ControlDefinition{ID: "task-oracle", Mandatory: true}, Layers: []SUTLayer{LayerFoundationModel}, Scenarios: []AgentScenario{ScenarioToolUsing, ScenarioDevOps}},
		{Criterion: ControlDefinition{ID: "tool-boundary", Mandatory: true}, Layers: []SUTLayer{LayerInteractionAccess}, Scenarios: []AgentScenario{ScenarioToolUsing, ScenarioDevOps}},
		{Criterion: ControlDefinition{ID: "resolver", Mandatory: true}, Layers: []SUTLayer{LayerExecutionScheduling}, Scenarios: []AgentScenario{ScenarioDevOps, ScenarioMultiAgent}},
		{Criterion: ControlDefinition{ID: "effect-receipt", Mandatory: true}, Layers: []SUTLayer{LayerUnderlyingInfrastructure}, Scenarios: []AgentScenario{ScenarioDevOps, ScenarioEmbodied}},
		{Criterion: ControlDefinition{ID: "revocation", Mandatory: false, NotApplicableAllowed: true}, Layers: []SUTLayer{LayerExecutionScheduling}, Scenarios: []AgentScenario{ScenarioMultiAgent}},
		{Criterion: ControlDefinition{ID: "resolver", Mandatory: false, NotApplicableAllowed: true}, Layers: []SUTLayer{LayerInteractionAccess}, Scenarios: []AgentScenario{ScenarioMultiAgent}},
	}}
}
func profileDigest(s string) string { sum := sha256.Sum256([]byte(s)); return hex.EncodeToString(sum[:]) }
func testProfile(layers, traversal []SUTLayer, scenarios []AgentScenario) SUTProfile {
	lc := make(map[SUTLayer]string, len(layers)); for _, v := range layers { lc[v] = profileDigest("layer:"+string(v)) }
	sc := make(map[AgentScenario]string, len(scenarios)); for _, v := range scenarios { sc[v] = profileDigest("scenario:"+string(v)) }
	return SUTProfile{Layers:layers, TraversalOrder:traversal, Scenarios:scenarios, LayerConfigSHA256:lc, ScenarioConfigSHA256:sc}
}
func TestCompileProfileComposesUnion(t *testing.T) {
	p := testProfile([]SUTLayer{LayerUnderlyingInfrastructure,LayerExecutionScheduling,LayerInteractionAccess}, []SUTLayer{LayerInteractionAccess,LayerExecutionScheduling,LayerUnderlyingInfrastructure}, []AgentScenario{ScenarioMultiAgent,ScenarioDevOps})
	got, err := CompileProfileInventory(p, profileCatalogFixture()); if err != nil { t.Fatal(err) }
	ids:=make([]string,0,len(got.Inventory.Criteria)); for _,c:=range got.Inventory.Criteria { ids=append(ids,c.ID) }
	if !reflect.DeepEqual(ids,[]string{"effect-receipt","resolver","revocation","tool-boundary"}) { t.Fatalf("criteria=%v",ids) }
	for _,c:=range got.Inventory.Criteria { if c.ID=="resolver"&&(!c.Mandatory||c.NotApplicableAllowed) { t.Fatalf("composed control relaxed: %#v",c) } }
	if got.InventorySHA256==""||got.CatalogSHA256=="" { t.Fatal("missing digest") }
}
func TestSetDeclarationOrderDoesNotChangeIdentity(t *testing.T) {
	c:=profileCatalogFixture(); traversal:=[]SUTLayer{LayerInteractionAccess,LayerUnderlyingInfrastructure}
	a:=testProfile([]SUTLayer{LayerInteractionAccess,LayerUnderlyingInfrastructure},traversal,[]AgentScenario{ScenarioDevOps,ScenarioToolUsing})
	b:=testProfile([]SUTLayer{LayerUnderlyingInfrastructure,LayerInteractionAccess},traversal,[]AgentScenario{ScenarioToolUsing,ScenarioDevOps})
	x,err:=CompileProfileInventory(a,c);if err!=nil{t.Fatal(err)}
	y,err:=CompileProfileInventory(b,c);if err!=nil{t.Fatal(err)}
	if x.InventorySHA256!=y.InventorySHA256{t.Fatal("set declaration order changed identity")}
}
func TestTraversalOrderChangesIdentity(t *testing.T) {
	c:=profileCatalogFixture(); layers:=[]SUTLayer{LayerInteractionAccess,LayerExecutionScheduling,LayerUnderlyingInfrastructure}; scenarios:=[]AgentScenario{ScenarioDevOps,ScenarioMultiAgent}
	a:=testProfile(layers,[]SUTLayer{LayerInteractionAccess,LayerExecutionScheduling,LayerUnderlyingInfrastructure},scenarios)
	b:=testProfile(layers,[]SUTLayer{LayerExecutionScheduling,LayerInteractionAccess,LayerUnderlyingInfrastructure},scenarios)
	x,err:=CompileProfileInventory(a,c);if err!=nil{t.Fatal(err)}
	y,err:=CompileProfileInventory(b,c);if err!=nil{t.Fatal(err)}
	if x.InventorySHA256==y.InventorySHA256{t.Fatal("traversal order not bound")}
}
func TestConstituentConfigChangesIdentity(t *testing.T) {
	c:=profileCatalogFixture();p:=testProfile([]SUTLayer{LayerInteractionAccess},[]SUTLayer{LayerInteractionAccess},[]AgentScenario{ScenarioToolUsing})
	a,err:=CompileProfileInventory(p,c);if err!=nil{t.Fatal(err)}
	p.LayerConfigSHA256[LayerInteractionAccess]=profileDigest("gateway-config-v2")
	b,err:=CompileProfileInventory(p,c);if err!=nil{t.Fatal(err)}
	if a.InventorySHA256==b.InventorySHA256{t.Fatal("config change not bound")}
}
func TestMissingOrInvalidConfigRejects(t *testing.T) {
	c:=profileCatalogFixture();p:=testProfile([]SUTLayer{LayerFoundationModel},[]SUTLayer{LayerFoundationModel},[]AgentScenario{ScenarioToolUsing})
	delete(p.LayerConfigSHA256,LayerFoundationModel)
	_,err:=CompileProfileInventory(p,c);if !errors.Is(err,ErrInvalidProfileConfig){t.Fatalf("missing config: %v",err)}
	p=testProfile([]SUTLayer{LayerFoundationModel},[]SUTLayer{LayerFoundationModel},[]AgentScenario{ScenarioToolUsing});p.ScenarioConfigSHA256[ScenarioToolUsing]="bad"
	_,err=CompileProfileInventory(p,c);if !errors.Is(err,ErrInvalidProfileConfig){t.Fatalf("bad config: %v",err)}
}
func TestUnknownAndEmptyProfilesReject(t *testing.T) {
	_,err:=CompileProfileInventory(testProfile([]SUTLayer{"unknown"},[]SUTLayer{"unknown"},[]AgentScenario{ScenarioToolUsing}),profileCatalogFixture())
	if !errors.Is(err,ErrUnknownSUTLayer){t.Fatalf("got %v",err)}
	_,err=CompileProfileInventory(testProfile([]SUTLayer{LayerFoundationModel},[]SUTLayer{LayerFoundationModel},nil),profileCatalogFixture())
	if !errors.Is(err,ErrInvalidProfile){t.Fatalf("got %v",err)}
}
func TestInvalidTraversalRejects(t *testing.T) {
	p:=testProfile([]SUTLayer{LayerFoundationModel,LayerInteractionAccess},[]SUTLayer{LayerFoundationModel},[]AgentScenario{ScenarioToolUsing})
	_,err:=CompileProfileInventory(p,profileCatalogFixture());if !errors.Is(err,ErrInvalidProfile){t.Fatalf("got %v",err)}
}
func TestUnscopedCatalogRejects(t *testing.T) {
	c:=profileCatalogFixture();c.Obligations[0].Scenarios=nil
	_,err:=CompileProfileInventory(testProfile([]SUTLayer{LayerFoundationModel},[]SUTLayer{LayerFoundationModel},[]AgentScenario{ScenarioToolUsing}),c)
	if !errors.Is(err,ErrInvalidProfile)&&!errors.Is(err,ErrInvalidCatalog){t.Fatalf("got %v",err)}
}
func TestNoMatchingObligationsRejects(t *testing.T) {
	_,err:=CompileProfileInventory(testProfile([]SUTLayer{LayerFoundationModel},[]SUTLayer{LayerFoundationModel},[]AgentScenario{ScenarioAutonomous}),profileCatalogFixture())
	if !errors.Is(err,ErrNoProfileObligations){t.Fatalf("got %v",err)}
}
func TestCatalogChangesChangeInventoryIdentity(t *testing.T) {
	p:=testProfile([]SUTLayer{LayerFoundationModel},[]SUTLayer{LayerFoundationModel},[]AgentScenario{ScenarioToolUsing});c:=profileCatalogFixture()
	a,err:=CompileProfileInventory(p,c);if err!=nil{t.Fatal(err)}
	c.Obligations=append(c.Obligations,ProfileObligation{Criterion:ControlDefinition{ID:"unselected-catalog-row",Mandatory:true},Layers:[]SUTLayer{LayerUnderlyingInfrastructure},Scenarios:[]AgentScenario{ScenarioEmbodied}})
	b,err:=CompileProfileInventory(p,c);if err!=nil{t.Fatal(err)}
	if a.InventorySHA256==b.InventorySHA256||a.CatalogSHA256==b.CatalogSHA256{t.Fatal("catalog edit not bound")}
}
func TestCatalogDigestCanonicalizesDeclarationOrder(t *testing.T) {
	c:=profileCatalogFixture();a,err:=ProfileCatalogDigest(c);if err!=nil{t.Fatal(err)}
	for i,j:=0,len(c.Obligations)-1;i<j;i,j=i+1,j-1{c.Obligations[i],c.Obligations[j]=c.Obligations[j],c.Obligations[i]}
	b,err:=ProfileCatalogDigest(c);if err!=nil{t.Fatal(err)}
	if a!=b{t.Fatal("catalogue declaration order changed digest")}
}
