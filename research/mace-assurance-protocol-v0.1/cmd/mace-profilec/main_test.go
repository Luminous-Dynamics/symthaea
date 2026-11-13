package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/Luminous-Dynamics/symthaea/research/mace-assurance-protocol-v0.1/matrix"
)

func profileRequestFixture() []byte {
	profile := matrix.SUTProfile{
		Layers: []matrix.SUTLayer{matrix.LayerFoundationModel, matrix.LayerInteractionAccess, matrix.LayerExecutionScheduling},
		TraversalOrder: []matrix.SUTLayer{matrix.LayerFoundationModel, matrix.LayerInteractionAccess, matrix.LayerExecutionScheduling},
		Scenarios: []matrix.AgentScenario{matrix.ScenarioToolUsing, matrix.ScenarioDevOps},
		LayerConfigSHA256: map[matrix.SUTLayer]string{
			matrix.LayerFoundationModel: strings.Repeat("1",64),
			matrix.LayerInteractionAccess: strings.Repeat("2",64),
			matrix.LayerExecutionScheduling: strings.Repeat("3",64),
		},
		ScenarioConfigSHA256: map[matrix.AgentScenario]string{
			matrix.ScenarioToolUsing: strings.Repeat("4",64),
			matrix.ScenarioDevOps: strings.Repeat("5",64),
		},
	}
	catalog := matrix.ProfileCatalog{SchemaVersion:"mace-profile-catalog-v2",CatalogID:"example-catalog-v1",Obligations:[]matrix.ProfileObligation{
		{Criterion:matrix.ControlDefinition{ID:"task-oracle",Mandatory:true},Layers:[]matrix.SUTLayer{matrix.LayerFoundationModel},Scenarios:[]matrix.AgentScenario{matrix.ScenarioToolUsing,matrix.ScenarioDevOps}},
		{Criterion:matrix.ControlDefinition{ID:"tool-boundary",Mandatory:true},Layers:[]matrix.SUTLayer{matrix.LayerInteractionAccess},Scenarios:[]matrix.AgentScenario{matrix.ScenarioToolUsing,matrix.ScenarioDevOps}},
		{Criterion:matrix.ControlDefinition{ID:"runtime-identity",Mandatory:true},Layers:[]matrix.SUTLayer{matrix.LayerExecutionScheduling},Scenarios:[]matrix.AgentScenario{matrix.ScenarioDevOps}},
	}}
	request:=struct{
		SchemaVersion string `json:"schema_version"`
		Profile matrix.SUTProfile `json:"profile"`
		Catalog matrix.ProfileCatalog `json:"catalog"`
	}{"mace-profile-compile-request-v1",profile,catalog}
	data,_:=json.Marshal(request)
	return data
}
func TestRunCompilesProfileAndInventory(t *testing.T){
	var output bytes.Buffer
	if err:=run(bytes.NewReader(profileRequestFixture()),&output);err!=nil{t.Fatal(err)}
	var result matrix.CompiledProfile
	if err:=json.Unmarshal(output.Bytes(),&result);err!=nil{t.Fatal(err)}
	if result.InventorySHA256==""||result.CatalogSHA256==""{t.Fatalf("missing digest: %#v",result)}
	if len(result.Inventory.Criteria)!=3{t.Fatalf("criteria=%d",len(result.Inventory.Criteria))}
}
func TestRunRejectsUnknownSchemaAndFields(t *testing.T){
	for _,input:=range []string{
		`{"schema_version":"future","profile":{},"catalog":{}}`,
		`{"schema_version":"mace-profile-compile-request-v1","profile":{},"catalog":{},"surprise":true}`,
	}{if err:=run(strings.NewReader(input),&bytes.Buffer{});err==nil{t.Fatalf("accepted %s",input)}}
}
func TestRunRejectsDuplicateAndTrailingJSON(t *testing.T){
	for _,input:=range []string{
		`{"schema_version":"mace-profile-compile-request-v1","schema_version":"mace-profile-compile-request-v1"}`,
		`{"schema_version":"mace-profile-compile-request-v1","profile":{},"catalog":{}} {}`,
	}{if err:=run(strings.NewReader(input),&bytes.Buffer{});err==nil{t.Fatalf("accepted %s",input)}}
}
