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
	CatalogSHA256 string `json:"catalog_sha256"`
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

// ProfileCatalogDigest hashes the complete versioned catalogue after normalizing
// obligation, layer and scenario declaration order. Any catalogue edit changes it.
func ProfileCatalogDigest(catalog ProfileCatalog) (string, error) {
	if catalog.SchemaVersion != "mace-profile-catalog-v1" || !profileID(catalog.CatalogID) || len(catalog.Obligations)==0 {
		return "",fmt.Errorf("%w: schema, catalog ID and obligations are required",ErrInvalidCatalog)
	}
	canonical:=ProfileCatalog{SchemaVersion:catalog.SchemaVersion,CatalogID:catalog.CatalogID,Obligations:make([]ProfileObligation,0,len(catalog.Obligations))}
	for i,row:=range catalog.Obligations{
		if !profileID(row.Criterion.ID){return "",fmt.Errorf("%w at obligation %d: invalid criterion ID",ErrInvalidCatalog,i+1)}
		layers,err:=normalizeLayers(row.Layers);if err!=nil{return "",fmt.Errorf("%w at obligation %d: %v",ErrInvalidCatalog,i+1,err)}
		scenarios,err:=normalizeScenarios(row.Scenarios);if err!=nil{return "",fmt.Errorf("%w at obligation %d: %v",ErrInvalidCatalog,i+1,err)}
		canonical.Obligations=append(canonical.Obligations,ProfileObligation{Criterion:row.Criterion,Layers:layers,Scenarios:scenarios})
	}
	sort.Slice(canonical.Obligations,func(i,j int)bool{
		a,b:=canonical.Obligations[i],canonical.Obligations[j]
		if a.Criterion.ID!=b.Criterion.ID{return a.Criterion.ID<b.Criterion.ID}
		la,lb:=strings.Join(layerStrings(a.Layers),","),strings.Join(layerStrings(b.Layers),",")
		if la!=lb{return la<lb}
		sa,sb:=strings.Join(scenarioStrings(a.Scenarios),","),strings.Join(scenarioStrings(b.Scenarios),",")
		if sa!=sb{return sa<sb}
		if a.Criterion.Mandatory!=b.Criterion.Mandatory{return !a.Criterion.Mandatory}
		return !a.Criterion.NotApplicableAllowed&&b.Criterion.NotApplicableAllowed
	})
	encoded,err:=json.Marshal(canonical);if err!=nil{return "",fmt.Errorf("marshal profile catalog: %w",err)}
	sum:=sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]),nil
}

// CompileProfileInventory selects obligations whose layer AND scenario scopes
// intersect the declared profile. Duplicate criterion IDs merge conservatively:
// Mandatory=OR; N/A permission=AND.
func CompileProfileInventory(profile SUTProfile,catalog ProfileCatalog)(CompiledProfile,error){
	catalogDigest,err:=ProfileCatalogDigest(catalog);if err!=nil{return CompiledProfile{},err}
	layers,err:=normalizeLayers(profile.Layers);if err!=nil{return CompiledProfile{},err}
	scenarios,err:=normalizeScenarios(profile.Scenarios);if err!=nil{return CompiledProfile{},err}
	layerSet:=make(map[SUTLayer]bool,len(layers));for _,v:=range layers{layerSet[v]=true}
	scenarioSet:=make(map[AgentScenario]bool,len(scenarios));for _,v:=range scenarios{scenarioSet[v]=true}
	selected:=make(map[string]ControlDefinition)
	for i,row:=range catalog.Obligations{
		if len(row.Layers)==0||len(row.Scenarios)==0{return CompiledProfile{},fmt.Errorf("%w at obligation %d: layer and scenario scopes are required",ErrInvalidCatalog,i+1)}
		osLayers,e:=normalizeLayers(row.Layers);if e!=nil{return CompiledProfile{},fmt.Errorf("%w at obligation %d: %v",ErrInvalidCatalog,i+1,e)}
		osScenarios,e:=normalizeScenarios(row.Scenarios);if e!=nil{return CompiledProfile{},fmt.Errorf("%w at obligation %d: %v",ErrInvalidCatalog,i+1,e)}
		layerMatch,scenarioMatch:=false,false
		for _,v:=range osLayers{if layerSet[v]{layerMatch=true;break}}
		for _,v:=range osScenarios{if scenarioSet[v]{scenarioMatch=true;break}}
		if !layerMatch||!scenarioMatch{continue}
		c:=row.Criterion
		if !profileID(c.ID){return CompiledProfile{},fmt.Errorf("%w at obligation %d: invalid criterion ID",ErrInvalidCatalog,i+1)}
		if prior,ok:=selected[c.ID];ok{
			prior.Mandatory=prior.Mandatory||c.Mandatory
			prior.NotApplicableAllowed=prior.NotApplicableAllowed&&c.NotApplicableAllowed
			selected[c.ID]=prior
		}else{selected[c.ID]=c}
	}
	if len(selected)==0{return CompiledProfile{},ErrNoProfileObligations}
	criteria:=make([]ControlDefinition,0,len(selected));for _,v:=range selected{criteria=append(criteria,v)}
	sort.Slice(criteria,func(i,j int)bool{return criteria[i].ID<criteria[j].ID})
	identity:=struct{
		CatalogID string `json:"catalog_id"`
		CatalogSHA256 string `json:"catalog_sha256"`
		Layers []SUTLayer `json:"layers"`
		Scenarios []AgentScenario `json:"scenarios"`
	}{catalog.CatalogID,catalogDigest,layers,scenarios}
	encoded,err:=json.Marshal(identity);if err!=nil{return CompiledProfile{},fmt.Errorf("marshal profile identity: %w",err)}
	sum:=sha256.Sum256(encoded)
	inventory:=ControlInventory{SchemaVersion:"mace-control-inventory-v1",InventoryID:"profile-"+hex.EncodeToString(sum[:8]),Criteria:criteria}
	digest,err:=InventoryDigest(inventory);if err!=nil{return CompiledProfile{},err}
	return CompiledProfile{CatalogID:catalog.CatalogID,CatalogSHA256:catalogDigest,Layers:layers,Scenarios:scenarios,Inventory:inventory,InventorySHA256:digest},nil
}

func layerStrings(values []SUTLayer)[]string{out:=make([]string,len(values));for i,v:=range values{out[i]=string(v)};return out}
func scenarioStrings(values []AgentScenario)[]string{out:=make([]string,len(values));for i,v:=range values{out[i]=string(v)};return out}

func normalizeLayers(values []SUTLayer)([]SUTLayer,error){
	if len(values)==0{return nil,fmt.Errorf("%w: at least one layer is required",ErrInvalidProfile)}
	seen:=make(map[SUTLayer]bool,len(values));out:=make([]SUTLayer,0,len(values))
	for _,v:=range values{if !knownLayers[v]{return nil,fmt.Errorf("%w: %q",ErrUnknownSUTLayer,v)};if seen[v]{return nil,fmt.Errorf("%w: duplicate layer %q",ErrInvalidProfile,v)};seen[v]=true;out=append(out,v)}
	sort.Slice(out,func(i,j int)bool{return out[i]<out[j]});return out,nil
}
func normalizeScenarios(values []AgentScenario)([]AgentScenario,error){
	if len(values)==0{return nil,fmt.Errorf("%w: at least one scenario is required",ErrInvalidProfile)}
	seen:=make(map[AgentScenario]bool,len(values));out:=make([]AgentScenario,0,len(values))
	for _,v:=range values{if !knownScenarios[v]{return nil,fmt.Errorf("%w: %q",ErrUnknownScenario,v)};if seen[v]{return nil,fmt.Errorf("%w: duplicate scenario %q",ErrInvalidProfile,v)};seen[v]=true;out=append(out,v)}
	sort.Slice(out,func(i,j int)bool{return out[i]<out[j]});return out,nil
}
func profileID(value string)bool{
	if len(value)==0||len(value)>64||strings.TrimSpace(value)!=value{return false}
	for i:=0;i<len(value);i++{c:=value[i];if !((c>='a'&&c<='z')||(c>='0'&&c<='9')||(i>0&&(c=='-'||c=='_'))){return false}}
	return true
}
