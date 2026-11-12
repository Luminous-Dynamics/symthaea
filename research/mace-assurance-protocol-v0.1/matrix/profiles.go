package matrix

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

type SUTLayer string

const (
	LayerFoundationModel SUTLayer = "foundation-model"
	LayerInteractionAccess SUTLayer = "interaction-access"
	LayerOrchestrationScheduler SUTLayer = "orchestration-scheduler"
	LayerExecutionInfrastructure SUTLayer = "execution-infrastructure"
)

type AgentScenario string

const (
	ScenarioToolUsing AgentScenario = "tool-using"
	ScenarioLongTermMemory AgentScenario = "long-term-memory"
	ScenarioDevOps AgentScenario = "devops-development"
	ScenarioEmbodied AgentScenario = "embodied"
	ScenarioMultiAgent AgentScenario = "multi-agent"
	ScenarioAutonomous AgentScenario = "autonomous-iterative"
)

type SUTProfile struct {
	Layers []SUTLayer `json:"layers"`
	Scenarios []AgentScenario `json:"scenarios"`
}

type ProfileObligation struct {
	Criterion ControlDefinition `json:"criterion"`
	Layers []SUTLayer `json:"layers"`
	Scenarios []AgentScenario `json:"scenarios"`
}

type ProfileCatalog struct {
	SchemaVersion string `json:"schema_version"`
	CatalogID string `json:"catalog_id"`
	Obligations []ProfileObligation `json:"obligations"`
}

type CompiledProfile struct {
	CatalogID string `json:"catalog_id"`
	Layers []SUTLayer `json:"layers"`
	Scenarios []AgentScenario `json:"scenarios"`
	Inventory ControlInventory `json:"inventory"`
	InventorySHA256 string `json:"inventory_sha256"`
}

var (
	ErrUnknownSUTLayer = errors.New("unknown SUT layer")
	ErrUnknownScenario = errors.New("unknown agent scenario")
	ErrInvalidProfile = errors.New("invalid SUT profile")
	ErrInvalidCatalog = errors.New("invalid profile catalog")
	ErrNoProfileObligations = errors.New("profile selects no control obligations")
)

var knownLayers = map[SUTLayer]bool{
	LayerFoundationModel:true,
	LayerInteractionAccess:true,
	LayerOrchestrationScheduler:true,
	LayerExecutionInfrastructure:true,
}
var knownScenarios = map[AgentScenario]bool{
	ScenarioToolUsing:true,
	ScenarioLongTermMemory:true,
	ScenarioDevOps:true,
	ScenarioEmbodied:true,
	ScenarioMultiAgent:true,
	ScenarioAutonomous:true,
}

// CompileProfileInventory selects obligations whose layer scope and scenario
// scope both intersect the declared SUT profile. Duplicate criterion IDs are
// merged conservatively: Mandatory=OR and N/A permission=AND.
func CompileProfileInventory(profile SUTProfile, catalog ProfileCatalog) (CompiledProfile, error) {
	if catalog.SchemaVersion != "mace-profile-catalog-v1" || !profileID(catalog.CatalogID) || len(catalog.Obligations)==0 {
		return CompiledProfile{}, fmt.Errorf("%w: schema, catalog ID and obligations are required", ErrInvalidCatalog)
	}
	layers, err := normalizeLayers(profile.Layers)
	if err != nil { return CompiledProfile{}, err }
	scenarios, err := normalizeScenarios(profile.Scenarios)
	if err != nil { return CompiledProfile{}, err }
	layerSet := make(map[SUTLayer]bool,len(layers))
	for _, v := range layers { layerSet[v]=true }
	scenarioSet := make(map[AgentScenario]bool,len(scenarios))
	for _, v := range scenarios { scenarioSet[v]=true }

	selected := make(map[string]ControlDefinition)
	for i, obligation := range catalog.Obligations {
		criterion := obligation.Criterion
		if !profileID(criterion.ID) { return CompiledProfile{},fmt.Errorf("%w at obligation %d: invalid criterion ID",ErrInvalidCatalog,i+1) }
		if len(obligation.Layers)==0 || len(obligation.Scenarios)==0 { return CompiledProfile{},fmt.Errorf("%w at obligation %d: layer and scenario scopes are required",ErrInvalidCatalog,i+1) }
		osLayers, e := normalizeLayers(obligation.Layers)
		if e != nil { return CompiledProfile{},fmt.Errorf("%w at obligation %d: %v",ErrInvalidCatalog,i+1,e) }
		osScenarios, e := normalizeScenarios(obligation.Scenarios)
		if e != nil { return CompiledProfile{},fmt.Errorf("%w at obligation %d: %v",ErrInvalidCatalog,i+1,e) }
		layerMatch, scenarioMatch := false,false
		for _, v := range osLayers { if layerSet[v] { layerMatch=true;break } }
		for _, v := range osScenarios { if scenarioSet[v] { scenarioMatch=true;break } }
		if !layerMatch || !scenarioMatch { continue }
		if prior,ok:=selected[criterion.ID];ok {
			prior.Mandatory = prior.Mandatory || criterion.Mandatory
			prior.NotApplicableAllowed = prior.NotApplicableAllowed && criterion.NotApplicableAllowed
			selected[criterion.ID]=prior
		} else { selected[criterion.ID]=criterion }
	}
	if len(selected)==0 { return CompiledProfile{},ErrNoProfileObligations }
	criteria:=make([]ControlDefinition,0,len(selected))
	for _,v:=range selected { criteria=append(criteria,v) }
	sort.Slice(criteria,func(i,j int)bool{return criteria[i].ID<criteria[j].ID})
	identity:=struct{
		CatalogID string `json:"catalog_id"`
		Layers []SUTLayer `json:"layers"`
		Scenarios []AgentScenario `json:"scenarios"`
	}{catalog.CatalogID,layers,scenarios}
	encoded,err:=json.Marshal(identity)
	if err!=nil{return CompiledProfile{},fmt.Errorf("marshal profile identity: %w",err)}
	sum:=sha256.Sum256(encoded)
	inventory:=ControlInventory{
		SchemaVersion:"mace-control-inventory-v1",
		InventoryID:"profile-"+hex.EncodeToString(sum[:8]),
		Criteria:criteria,
	}
	digest,err:=InventoryDigest(inventory)
	if err!=nil{return CompiledProfile{},err}
	return CompiledProfile{CatalogID:catalog.CatalogID,Layers:layers,Scenarios:scenarios,Inventory:inventory,InventorySHA256:digest},nil
}

func normalizeLayers(values []SUTLayer)([]SUTLayer,error){
	if len(values)==0{return nil,fmt.Errorf("%w: at least one layer is required",ErrInvalidProfile)}
	seen:=make(map[SUTLayer]bool,len(values));out:=make([]SUTLayer,0,len(values))
	for _,v:=range values{
		if !knownLayers[v]{return nil,fmt.Errorf("%w: %q",ErrUnknownSUTLayer,v)}
		if seen[v]{return nil,fmt.Errorf("%w: duplicate layer %q",ErrInvalidProfile,v)}
		seen[v]=true;out=append(out,v)
	}
	sort.Slice(out,func(i,j int)bool{return out[i]<out[j]})
	return out,nil
}
func normalizeScenarios(values []AgentScenario)([]AgentScenario,error){
	if len(values)==0{return nil,fmt.Errorf("%w: at least one scenario is required",ErrInvalidProfile)}
	seen:=make(map[AgentScenario]bool,len(values));out:=make([]AgentScenario,0,len(values))
	for _,v:=range values{
		if !knownScenarios[v]{return nil,fmt.Errorf("%w: %q",ErrUnknownScenario,v)}
		if seen[v]{return nil,fmt.Errorf("%w: duplicate scenario %q",ErrInvalidProfile,v)}
		seen[v]=true;out=append(out,v)
	}
	sort.Slice(out,func(i,j int)bool{return out[i]<out[j]})
	return out,nil
}
func profileID(value string)bool{
	if len(value)==0||len(value)>64||strings.TrimSpace(value)!=value{return false}
	for i:=0;i<len(value);i++{
		c:=value[i]
		if !((c>='a'&&c<='z')||(c>='0'&&c<='9')||(i>0&&(c=='-'||c=='_'))){return false}
	}
	return true
}
