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
	LayerFoundationModel          SUTLayer = "foundation-model"
	LayerInteractionAccess        SUTLayer = "interaction-access"
	LayerExecutionScheduling      SUTLayer = "execution-scheduling"
	LayerUnderlyingInfrastructure SUTLayer = "underlying-infrastructure"
)

type AgentScenario string

const (
	ScenarioToolUsing      AgentScenario = "tool-using"
	ScenarioLongTermMemory AgentScenario = "long-term-memory"
	ScenarioDevOps         AgentScenario = "devops-development"
	ScenarioEmbodied       AgentScenario = "embodied"
	ScenarioMultiAgent     AgentScenario = "multi-agent"
	ScenarioAutonomous     AgentScenario = "autonomous-iterative"
)

// SUTProfile is a composite boundary. Layers is a set; TraversalOrder is the
// request path through those layers and is semantically significant.
type SUTProfile struct {
	Layers               []SUTLayer              `json:"layers"`
	TraversalOrder       []SUTLayer              `json:"traversal_order"`
	Scenarios            []AgentScenario         `json:"scenarios"`
	LayerConfigSHA256    map[SUTLayer]string     `json:"layer_config_sha256"`
	ScenarioConfigSHA256 map[AgentScenario]string `json:"scenario_config_sha256"`
}

type ProfileObligation struct {
	Criterion ControlDefinition `json:"criterion"`
	Layers    []SUTLayer        `json:"layers"`
	Scenarios []AgentScenario   `json:"scenarios"`
}

type ProfileCatalog struct {
	SchemaVersion string              `json:"schema_version"`
	CatalogID     string              `json:"catalog_id"`
	Obligations   []ProfileObligation `json:"obligations"`
}

type CompiledProfile struct {
	CatalogID             string                  `json:"catalog_id"`
	CatalogSHA256         string                  `json:"catalog_sha256"`
	Layers                []SUTLayer              `json:"layers"`
	TraversalOrder        []SUTLayer              `json:"traversal_order"`
	Scenarios             []AgentScenario         `json:"scenarios"`
	LayerConfigSHA256     map[SUTLayer]string     `json:"layer_config_sha256"`
	ScenarioConfigSHA256  map[AgentScenario]string `json:"scenario_config_sha256"`
	Inventory             ControlInventory        `json:"inventory"`
	InventorySHA256       string                  `json:"inventory_sha256"`
}

var (
	ErrUnknownSUTLayer      = errors.New("unknown SUT layer")
	ErrUnknownScenario      = errors.New("unknown agent scenario")
	ErrInvalidProfile       = errors.New("invalid SUT profile")
	ErrInvalidProfileConfig = errors.New("invalid SUT profile configuration digest set")
	ErrInvalidCatalog       = errors.New("invalid profile catalog")
	ErrNoProfileObligations = errors.New("profile selects no control obligations")
)

var knownLayers = map[SUTLayer]bool{
	LayerFoundationModel: true, LayerInteractionAccess: true,
	LayerExecutionScheduling: true, LayerUnderlyingInfrastructure: true,
}
var knownScenarios = map[AgentScenario]bool{
	ScenarioToolUsing: true, ScenarioLongTermMemory: true, ScenarioDevOps: true,
	ScenarioEmbodied: true, ScenarioMultiAgent: true, ScenarioAutonomous: true,
}

func ProfileCatalogDigest(catalog ProfileCatalog) (string, error) {
	if catalog.SchemaVersion != "mace-profile-catalog-v2" || !profileID(catalog.CatalogID) || len(catalog.Obligations) == 0 {
		return "", fmt.Errorf("%w: schema, catalog ID and obligations are required", ErrInvalidCatalog)
	}
	canonical := ProfileCatalog{SchemaVersion: catalog.SchemaVersion, CatalogID: catalog.CatalogID, Obligations: make([]ProfileObligation, 0, len(catalog.Obligations))}
	for i, row := range catalog.Obligations {
		if !profileID(row.Criterion.ID) {
			return "", fmt.Errorf("%w at obligation %d: invalid criterion ID", ErrInvalidCatalog, i+1)
		}
		layers, err := normalizeLayers(row.Layers)
		if err != nil {
			return "", fmt.Errorf("%w at obligation %d: %v", ErrInvalidCatalog, i+1, err)
		}
		scenarios, err := normalizeScenarios(row.Scenarios)
		if err != nil {
			return "", fmt.Errorf("%w at obligation %d: %v", ErrInvalidCatalog, i+1, err)
		}
		canonical.Obligations = append(canonical.Obligations, ProfileObligation{Criterion: row.Criterion, Layers: layers, Scenarios: scenarios})
	}
	sort.Slice(canonical.Obligations, func(i, j int) bool {
		a, b := canonical.Obligations[i], canonical.Obligations[j]
		if a.Criterion.ID != b.Criterion.ID { return a.Criterion.ID < b.Criterion.ID }
		la, lb := strings.Join(layerStrings(a.Layers), ","), strings.Join(layerStrings(b.Layers), ",")
		if la != lb { return la < lb }
		sa, sb := strings.Join(scenarioStrings(a.Scenarios), ","), strings.Join(scenarioStrings(b.Scenarios), ",")
		if sa != sb { return sa < sb }
		if a.Criterion.Mandatory != b.Criterion.Mandatory { return !a.Criterion.Mandatory }
		return !a.Criterion.NotApplicableAllowed && b.Criterion.NotApplicableAllowed
	})
	encoded, err := json.Marshal(canonical)
	if err != nil { return "", fmt.Errorf("marshal profile catalog: %w", err) }
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}

// CompileProfileInventory selects obligations whose layer AND scenario scopes
// intersect. Duplicate criterion IDs merge conservatively: mandatory=OR and
// N/A permission=AND. The inventory identity binds the complete catalogue,
// selected profiles, traversal order and constituent configuration digests.
func CompileProfileInventory(profile SUTProfile, catalog ProfileCatalog) (CompiledProfile, error) {
	catalogDigest, err := ProfileCatalogDigest(catalog)
	if err != nil { return CompiledProfile{}, err }
	layers, err := normalizeLayers(profile.Layers)
	if err != nil { return CompiledProfile{}, err }
	traversal, err := validateTraversalOrder(profile.TraversalOrder, layers)
	if err != nil { return CompiledProfile{}, err }
	scenarios, err := normalizeScenarios(profile.Scenarios)
	if err != nil { return CompiledProfile{}, err }
	layerConfigs, err := validateLayerConfigs(profile.LayerConfigSHA256, layers)
	if err != nil { return CompiledProfile{}, err }
	scenarioConfigs, err := validateScenarioConfigs(profile.ScenarioConfigSHA256, scenarios)
	if err != nil { return CompiledProfile{}, err }

	layerSet := make(map[SUTLayer]bool, len(layers))
	for _, value := range layers { layerSet[value] = true }
	scenarioSet := make(map[AgentScenario]bool, len(scenarios))
	for _, value := range scenarios { scenarioSet[value] = true }

	selected := make(map[string]ControlDefinition)
	for i, row := range catalog.Obligations {
		osLayers, err := normalizeLayers(row.Layers)
		if err != nil { return CompiledProfile{}, fmt.Errorf("%w at obligation %d: %v", ErrInvalidCatalog, i+1, err) }
		osScenarios, err := normalizeScenarios(row.Scenarios)
		if err != nil { return CompiledProfile{}, fmt.Errorf("%w at obligation %d: %v", ErrInvalidCatalog, i+1, err) }
		layerMatch, scenarioMatch := false, false
		for _, value := range osLayers { if layerSet[value] { layerMatch = true; break } }
		for _, value := range osScenarios { if scenarioSet[value] { scenarioMatch = true; break } }
		if !layerMatch || !scenarioMatch { continue }
		criterion := row.Criterion
		if !profileID(criterion.ID) { return CompiledProfile{}, fmt.Errorf("%w at obligation %d: invalid criterion ID", ErrInvalidCatalog, i+1) }
		if prior, ok := selected[criterion.ID]; ok {
			prior.Mandatory = prior.Mandatory || criterion.Mandatory
			prior.NotApplicableAllowed = prior.NotApplicableAllowed && criterion.NotApplicableAllowed
			selected[criterion.ID] = prior
		} else { selected[criterion.ID] = criterion }
	}
	if len(selected) == 0 { return CompiledProfile{}, ErrNoProfileObligations }

	criteria := make([]ControlDefinition, 0, len(selected))
	for _, value := range selected { criteria = append(criteria, value) }
	sort.Slice(criteria, func(i, j int) bool { return criteria[i].ID < criteria[j].ID })

	identity := struct {
		CatalogID             string                  `json:"catalog_id"`
		CatalogSHA256         string                  `json:"catalog_sha256"`
		Layers                []SUTLayer              `json:"layers"`
		TraversalOrder        []SUTLayer              `json:"traversal_order"`
		Scenarios             []AgentScenario         `json:"scenarios"`
		LayerConfigSHA256     map[SUTLayer]string     `json:"layer_config_sha256"`
		ScenarioConfigSHA256  map[AgentScenario]string `json:"scenario_config_sha256"`
	}{catalog.CatalogID, catalogDigest, layers, traversal, scenarios, layerConfigs, scenarioConfigs}
	encoded, err := json.Marshal(identity)
	if err != nil { return CompiledProfile{}, fmt.Errorf("marshal profile identity: %w", err) }
	sum := sha256.Sum256(encoded)
	inventory := ControlInventory{SchemaVersion: "mace-control-inventory-v1", InventoryID: "profile-" + hex.EncodeToString(sum[:8]), Criteria: criteria}
	digest, err := InventoryDigest(inventory)
	if err != nil { return CompiledProfile{}, err }
	return CompiledProfile{
		CatalogID: catalog.CatalogID, CatalogSHA256: catalogDigest,
		Layers: layers, TraversalOrder: traversal, Scenarios: scenarios,
		LayerConfigSHA256: layerConfigs, ScenarioConfigSHA256: scenarioConfigs,
		Inventory: inventory, InventorySHA256: digest,
	}, nil
}

func validateTraversalOrder(order, layers []SUTLayer) ([]SUTLayer, error) {
	if len(order) != len(layers) { return nil, fmt.Errorf("%w: traversal order must list every selected layer exactly once", ErrInvalidProfile) }
	selected, seen := make(map[SUTLayer]bool, len(layers)), make(map[SUTLayer]bool, len(order))
	for _, value := range layers { selected[value] = true }
	out := make([]SUTLayer, 0, len(order))
	for _, value := range order {
		if !knownLayers[value] || !selected[value] || seen[value] { return nil, fmt.Errorf("%w: traversal order has unknown, unselected or duplicate layer %q", ErrInvalidProfile, value) }
		seen[value] = true
		out = append(out, value)
	}
	return out, nil
}

func validateLayerConfigs(configs map[SUTLayer]string, layers []SUTLayer) (map[SUTLayer]string, error) {
	if len(configs) != len(layers) { return nil, fmt.Errorf("%w: one digest is required for every selected layer", ErrInvalidProfileConfig) }
	out := make(map[SUTLayer]string, len(configs))
	for key, value := range configs {
		if !knownLayers[key] || !validSHA256(value) { return nil, fmt.Errorf("%w: invalid layer config digest for %q", ErrInvalidProfileConfig, key) }
		out[key] = value
	}
	for _, key := range layers { if _, ok := out[key]; !ok { return nil, fmt.Errorf("%w: missing layer config %q", ErrInvalidProfileConfig, key) } }
	return out, nil
}

func validateScenarioConfigs(configs map[AgentScenario]string, scenarios []AgentScenario) (map[AgentScenario]string, error) {
	if len(configs) != len(scenarios) { return nil, fmt.Errorf("%w: one digest is required for every selected scenario", ErrInvalidProfileConfig) }
	out := make(map[AgentScenario]string, len(configs))
	for key, value := range configs {
		if !knownScenarios[key] || !validSHA256(value) { return nil, fmt.Errorf("%w: invalid scenario config digest for %q", ErrInvalidProfileConfig, key) }
		out[key] = value
	}
	for _, key := range scenarios { if _, ok := out[key]; !ok { return nil, fmt.Errorf("%w: missing scenario config %q", ErrInvalidProfileConfig, key) } }
	return out, nil
}

func validSHA256(value string) bool {
	if len(value) != 64 || strings.ToLower(value) != value { return false }
	_, err := hex.DecodeString(value)
	return err == nil
}
func layerStrings(values []SUTLayer) []string { out := make([]string, len(values)); for i, value := range values { out[i] = string(value) }; return out }
func scenarioStrings(values []AgentScenario) []string { out := make([]string, len(values)); for i, value := range values { out[i] = string(value) }; return out }

func normalizeLayers(values []SUTLayer) ([]SUTLayer, error) {
	if len(values) == 0 { return nil, fmt.Errorf("%w: at least one layer is required", ErrInvalidProfile) }
	seen := make(map[SUTLayer]bool, len(values)); out := make([]SUTLayer, 0, len(values))
	for _, value := range values {
		if !knownLayers[value] { return nil, fmt.Errorf("%w: %q", ErrUnknownSUTLayer, value) }
		if seen[value] { return nil, fmt.Errorf("%w: duplicate layer %q", ErrInvalidProfile, value) }
		seen[value] = true; out = append(out, value)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out, nil
}
func normalizeScenarios(values []AgentScenario) ([]AgentScenario, error) {
	if len(values) == 0 { return nil, fmt.Errorf("%w: at least one scenario is required", ErrInvalidProfile) }
	seen := make(map[AgentScenario]bool, len(values)); out := make([]AgentScenario, 0, len(values))
	for _, value := range values {
		if !knownScenarios[value] { return nil, fmt.Errorf("%w: %q", ErrUnknownScenario, value) }
		if seen[value] { return nil, fmt.Errorf("%w: duplicate scenario %q", ErrInvalidProfile, value) }
		seen[value] = true; out = append(out, value)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out, nil
}
func profileID(value string) bool {
	if len(value) == 0 || len(value) > 64 || strings.TrimSpace(value) != value { return false }
	for i := 0; i < len(value); i++ {
		c := value[i]
		if !((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || (i > 0 && (c == '-' || c == '_'))) { return false }
	}
	return true
}
