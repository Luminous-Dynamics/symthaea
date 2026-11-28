// MON-050 is a deterministic dependency-impact benchmark, not a production authorizer.
package impact

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

const graphProfile = "mon-dependency-graph-v1"
const eventProfile = "mon-dependency-event-v1"
const fixtureProfile = "mon-dependency-impact-fixture-v1"

type NodeKind string
type EdgeMode string
type EventKind string

const (
	ClaimNode      NodeKind = "claim"
	EvidenceNode   NodeKind = "evidence"
	PolicyNode     NodeKind = "policy"
	CapabilityNode NodeKind = "capability"
	ScopeNode      NodeKind = "scope"
	RequiredEdge   EdgeMode = "required"
	ContextEdge    EdgeMode = "context"
)

const (
	EvidenceRevoked  EventKind = "evidence_revoked"
	EvidenceExpired  EventKind = "evidence_expired"
	SourceReplaced   EventKind = "source_replaced"
	CapabilityGone   EventKind = "capability_withdrawn"
	PolicySuperseded EventKind = "policy_superseded"
	ScopeChanged     EventKind = "scope_changed"
)

type Node struct {
	ID   string   `json:"id"`
	Kind NodeKind `json:"kind"`
}
type Edge struct {
	From string   `json:"from"`
	To   string   `json:"to"`
	Mode EdgeMode `json:"mode"`
}
type Graph struct {
	Profile string `json:"profile"`
	Nodes   []Node `json:"nodes"`
	Edges   []Edge `json:"edges"`
}
type Event struct {
	Profile string    `json:"profile"`
	ID      string    `json:"id"`
	Kind    EventKind `json:"kind"`
	Target  string    `json:"target"`
}
type Fixture struct {
	Profile            string   `json:"profile"`
	ID                 string   `json:"id"`
	Graph              Graph    `json:"graph"`
	Event              Event    `json:"event"`
	ExpectedAffected   []string `json:"expected_affected_claims"`
	ExpectedUnaffected []string `json:"expected_unaffected_claims"`
}
type Metric struct {
	Numerator   int     `json:"numerator"`
	Denominator int     `json:"denominator"`
	Rate        float64 `json:"rate"`
}
type Gate struct {
	Name     string `json:"name"`
	Passed   bool   `json:"passed"`
	Observed string `json:"observed"`
}
type Result struct {
	Profile                string   `json:"profile"`
	FixtureID              string   `json:"fixture_id"`
	PredictedAffected      []string `json:"predicted_affected_claims"`
	ExpectedAffected       []string `json:"expected_affected_claims"`
	ExpectedUnaffected     []string `json:"expected_unaffected_claims"`
	Precision              Metric   `json:"invalidation_precision"`
	Recall                 Metric   `json:"invalidation_recall"`
	FalseInvalidation      Metric   `json:"false_invalidation_rate"`
	MissedInvalidation     Metric   `json:"missed_invalidation_rate"`
	ExactSet               bool     `json:"exact_affected_set"`
	Gates                  []Gate   `json:"gates"`
	Verdict                string   `json:"verdict"`
	ProductionUsePermitted bool     `json:"production_use_permitted"`
}

func validKind(k NodeKind) bool {
	return k == ClaimNode || k == EvidenceNode || k == PolicyNode || k == CapabilityNode || k == ScopeNode
}
func validEventKind(k EventKind) bool {
	return k == EvidenceRevoked || k == EvidenceExpired || k == SourceReplaced || k == CapabilityGone || k == PolicySuperseded || k == ScopeChanged
}
func eventTargetKind(k EventKind) NodeKind {
	switch k {
	case EvidenceRevoked, EvidenceExpired, SourceReplaced:
		return EvidenceNode
	case CapabilityGone:
		return CapabilityNode
	case PolicySuperseded:
		return PolicyNode
	case ScopeChanged:
		return ScopeNode
	}
	return ""
}

// Validate enforces a typed DAG. Edges point from a dependent node to its dependency.
func (g Graph) Validate() error {
	if g.Profile != graphProfile || len(g.Nodes) == 0 {
		return errors.New("graph profile or nodes invalid")
	}
	nodes := map[string]Node{}
	for _, n := range g.Nodes {
		if strings.TrimSpace(n.ID) == "" || !validKind(n.Kind) {
			return fmt.Errorf("invalid node: %q", n.ID)
		}
		if _, ok := nodes[n.ID]; ok {
			return fmt.Errorf("duplicate node id: %s", n.ID)
		}
		nodes[n.ID] = n
	}
	adj := map[string][]string{}
	edges := map[string]bool{}
	for _, e := range g.Edges {
		if _, ok := nodes[e.From]; !ok {
			return fmt.Errorf("unknown edge source: %s", e.From)
		}
		if _, ok := nodes[e.To]; !ok {
			return fmt.Errorf("unknown edge target: %s", e.To)
		}
		if e.From == e.To {
			return fmt.Errorf("self dependency: %s", e.From)
		}
		if e.Mode != RequiredEdge && e.Mode != ContextEdge {
			return fmt.Errorf("invalid edge mode: %s", e.Mode)
		}
		fromKind, toKind := nodes[e.From].Kind, nodes[e.To].Kind
		validTypedEdge := false
		switch fromKind {
		case ClaimNode:
			validTypedEdge = toKind == EvidenceNode || toKind == PolicyNode || toKind == CapabilityNode || toKind == ScopeNode
		case EvidenceNode:
			validTypedEdge = toKind == CapabilityNode || toKind == PolicyNode
		case PolicyNode:
			validTypedEdge = toKind == CapabilityNode
		case CapabilityNode:
			validTypedEdge = toKind == CapabilityNode
		case ScopeNode:
			validTypedEdge = false
		}
		if !validTypedEdge {
			return fmt.Errorf("type-confused dependency edge: %s (%s) -> %s (%s)", e.From, fromKind, e.To, toKind)
		}
		key := e.From + "\x00" + e.To
		if edges[key] {
			return fmt.Errorf("duplicate dependency edge: %s -> %s", e.From, e.To)
		}
		edges[key] = true
		adj[e.From] = append(adj[e.From], e.To)
	}
	state := map[string]uint8{}
	var visit func(string) error
	visit = func(id string) error {
		if state[id] == 1 {
			return fmt.Errorf("dependency cycle at %s", id)
		}
		if state[id] == 2 {
			return nil
		}
		state[id] = 1
		for _, to := range adj[id] {
			if err := visit(to); err != nil {
				return err
			}
		}
		state[id] = 2
		return nil
	}
	for id := range nodes {
		if err := visit(id); err != nil {
			return err
		}
	}
	return nil
}

func (e Event) Validate(g Graph) error {
	if e.Profile != eventProfile || strings.TrimSpace(e.ID) == "" || !validEventKind(e.Kind) {
		return errors.New("event profile/id/kind invalid")
	}
	for _, n := range g.Nodes {
		if n.ID == e.Target {
			if n.Kind != eventTargetKind(e.Kind) {
				return fmt.Errorf("event target %s has kind %s, expected %s", n.ID, n.Kind, eventTargetKind(e.Kind))
			}
			return nil
		}
	}
	return fmt.Errorf("event target not found: %s", e.Target)
}

// ImpactedClaims returns claim nodes whose transitive path to target contains only required edges.
func ImpactedClaims(g Graph, e Event) ([]string, error) {
	if err := g.Validate(); err != nil {
		return nil, err
	}
	if err := e.Validate(g); err != nil {
		return nil, err
	}
	reverse := map[string][]string{}
	for _, edge := range g.Edges {
		if edge.Mode == RequiredEdge {
			reverse[edge.To] = append(reverse[edge.To], edge.From)
		}
	}
	seen := map[string]bool{e.Target: true}
	queue := []string{e.Target}
	claims := []string{}
	kinds := map[string]NodeKind{}
	for _, n := range g.Nodes {
		kinds[n.ID] = n.Kind
	}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, parent := range reverse[cur] {
			if seen[parent] {
				continue
			}
			seen[parent] = true
			queue = append(queue, parent)
			if kinds[parent] == ClaimNode {
				claims = append(claims, parent)
			}
		}
	}
	sort.Strings(claims)
	return claims, nil
}

func uniqueSorted(in []string) ([]string, error) {
	out := append([]string(nil), in...)
	sort.Strings(out)
	for i := 1; i < len(out); i++ {
		if out[i] == out[i-1] {
			return nil, fmt.Errorf("duplicate claim id: %s", out[i])
		}
	}
	return out, nil
}
func rate(n, d int) Metric {
	v := 0.0
	if d > 0 {
		v = float64(n) / float64(d)
	} else if n == 0 {
		v = 1
	}
	return Metric{n, d, v}
}

func Evaluate(f Fixture) (Result, error) {
	if f.Profile != fixtureProfile || strings.TrimSpace(f.ID) == "" {
		return Result{}, errors.New("fixture profile/id invalid")
	}
	pred, err := ImpactedClaims(f.Graph, f.Event)
	if err != nil {
		return Result{}, err
	}
	affected, err := uniqueSorted(f.ExpectedAffected)
	if err != nil {
		return Result{}, err
	}
	unaffected, err := uniqueSorted(f.ExpectedUnaffected)
	if err != nil {
		return Result{}, err
	}
	claims := map[string]bool{}
	for _, n := range f.Graph.Nodes {
		if n.Kind == ClaimNode {
			claims[n.ID] = true
		}
	}
	if len(claims) != len(affected)+len(unaffected) {
		return Result{}, errors.New("expected affected/unaffected sets do not partition claim nodes")
	}
	wantA, wantU := map[string]bool{}, map[string]bool{}
	for _, id := range affected {
		if !claims[id] || wantA[id] {
			return Result{}, fmt.Errorf("invalid expected affected claim: %s", id)
		}
		wantA[id] = true
	}
	for _, id := range unaffected {
		if !claims[id] || wantU[id] || wantA[id] {
			return Result{}, fmt.Errorf("invalid expected unaffected claim: %s", id)
		}
		wantU[id] = true
	}
	predSet := map[string]bool{}
	for _, id := range pred {
		predSet[id] = true
	}
	correct, extra, missed := 0, 0, 0
	for _, id := range pred {
		if wantA[id] {
			correct++
		} else {
			extra++
		}
	}
	for _, id := range affected {
		if !predSet[id] {
			missed++
		}
	}
	precision := rate(correct, len(pred))
	recall := rate(correct, len(affected))
	falseInv := rate(extra, len(unaffected))
	missRate := rate(missed, len(affected))
	exact := extra == 0 && missed == 0
	gates := []Gate{{"exact_affected_set", exact, fmt.Sprintf("%d false invalidations; %d missed", extra, missed)}, {"no_unrelated_claim_invalidations", extra == 0, fmt.Sprintf("%d/%d", extra, len(unaffected))}, {"no_missed_mandatory_invalidations", missed == 0, fmt.Sprintf("%d/%d", missed, len(affected))}, {"perfect_precision_and_recall", precision.Rate == 1 && recall.Rate == 1, fmt.Sprintf("precision %.4f; recall %.4f", precision.Rate, recall.Rate)}}
	verdict := "blocked"
	passed := true
	for _, g := range gates {
		if !g.Passed {
			passed = false
		}
	}
	if passed {
		verdict = "qualified"
	}
	return Result{Profile: "mon-dependency-impact-result-v1", FixtureID: f.ID, PredictedAffected: pred, ExpectedAffected: affected, ExpectedUnaffected: unaffected, Precision: precision, Recall: recall, FalseInvalidation: falseInv, MissedInvalidation: missRate, ExactSet: exact, Gates: gates, Verdict: verdict, ProductionUsePermitted: false}, nil
}
