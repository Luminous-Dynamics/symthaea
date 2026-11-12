package matrix

import (
	"errors"
	"reflect"
	"testing"
)

func profileCatalogFixture() ProfileCatalog {
	return ProfileCatalog{SchemaVersion:"mace-profile-catalog-v1",CatalogID:"catalog-2026-11",Obligations:[]ProfileObligation{
		{Criterion:ControlDefinition{ID:"task-oracle",Mandatory:true},Layers:[]SUTLayer{LayerFoundationModel},Scenarios:[]AgentScenario{ScenarioToolUsing,ScenarioDevOps}},
		{Criterion:ControlDefinition{ID:"tool-boundary",Mandatory:true},Layers:[]SUTLayer{LayerInteractionAccess},Scenarios:[]AgentScenario{ScenarioToolUsing,ScenarioDevOps}},
		{Criterion:ControlDefinition{ID:"resolver",Mandatory:true},Layers:[]SUTLayer{LayerOrchestrationScheduler},Scenarios:[]AgentScenario{ScenarioDevOps,ScenarioMultiAgent}},
		{Criterion:ControlDefinition{ID:"effect-receipt",Mandatory:true},Layers:[]SUTLayer{LayerExecutionInfrastructure},Scenarios:[]AgentScenario{ScenarioDevOps,ScenarioEmbodied}},
		{Criterion:ControlDefinition{ID:"revocation",Mandatory:false,NotApplicableAllowed:true},Layers:[]SUTLayer{LayerOrchestrationScheduler},Scenarios:[]AgentScenario{ScenarioMultiAgent}},
		{Criterion:ControlDefinition{ID:"resolver",Mandatory:false,NotApplicableAllowed:true},Layers:[]SUTLayer{LayerInteractionAccess},Scenarios:[]AgentScenario{ScenarioMultiAgent}},
	}}
}
func TestCompileProfileComposesUnion(t *testing.T){
	p:=SUTProfile{Layers:[]SUTLayer{LayerExecutionInfrastructure,LayerOrchestrationScheduler,LayerInteractionAccess},Scenarios:[]AgentScenario{ScenarioMultiAgent,ScenarioDevOps}}
	got,err:=CompileProfileInventory(p,profileCatalogFixture());if err!=nil{t.Fatal(err)}
	ids:=make([]string,0,len(got.Inventory.Criteria));for _,c:=range got.Inventory.Criteria{ids=append(ids,c.ID)}
	want:=[]string{"effect-receipt","resolver","revocation","tool-boundary"}
	if !reflect.DeepEqual(ids,want){t.Fatalf("criteria=%v want=%v",ids,want)}
	for _,c:=range got.Inventory.Criteria{if c.ID=="resolver"&&(!c.Mandatory||c.NotApplicableAllowed){t.Fatalf("composition relaxed criterion: %#v",c)}}
	d,err:=InventoryDigest(got.Inventory);if err!=nil||d!=got.InventorySHA256{t.Fatalf("digest=%q err=%v",d,err)}
}
func TestCompileProfileIsOrderDeterministic(t *testing.T){
	c:=profileCatalogFixture()
	a,err:=CompileProfileInventory(SUTProfile{Layers:[]SUTLayer{LayerInteractionAccess,LayerExecutionInfrastructure},Scenarios:[]AgentScenario{ScenarioDevOps,ScenarioToolUsing}},c);if err!=nil{t.Fatal(err)}
	b,err:=CompileProfileInventory(SUTProfile{Layers:[]SUTLayer{LayerExecutionInfrastructure,LayerInteractionAccess},Scenarios:[]AgentScenario{ScenarioToolUsing,ScenarioDevOps}},c);if err!=nil{t.Fatal(err)}
	if a.InventorySHA256!=b.InventorySHA256||a.Inventory.InventoryID!=b.Inventory.InventoryID{t.Fatal("selection order changed deterministic profile identity")}
}
func TestCompileProfileRejectsUnknownAndEmptyProfile(t *testing.T){
	_,err:=CompileProfileInventory(SUTProfile{Layers:[]SUTLayer{"unreviewed"},Scenarios:[]AgentScenario{ScenarioToolUsing}},profileCatalogFixture())
	if !errors.Is(err,ErrUnknownSUTLayer){t.Fatalf("got %v",err)}
	_,err=CompileProfileInventory(SUTProfile{Layers:[]SUTLayer{LayerFoundationModel}},profileCatalogFixture())
	if !errors.Is(err,ErrInvalidProfile){t.Fatalf("got %v",err)}
}
func TestCompileProfileRejectsUnscopedRows(t *testing.T){
	c:=profileCatalogFixture();c.Obligations[0].Scenarios=nil
	_,err:=CompileProfileInventory(SUTProfile{Layers:[]SUTLayer{LayerFoundationModel},Scenarios:[]AgentScenario{ScenarioToolUsing}},c)
	if !errors.Is(err,ErrInvalidCatalog){t.Fatalf("got %v",err)}
}
func TestCompileProfileRejectsEmptySelection(t *testing.T){
	_,err:=CompileProfileInventory(SUTProfile{Layers:[]SUTLayer{LayerFoundationModel},Scenarios:[]AgentScenario{ScenarioAutonomous}},profileCatalogFixture())
	if !errors.Is(err,ErrNoProfileObligations){t.Fatalf("got %v",err)}
}
