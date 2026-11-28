// Package inventory evaluates independently corroborated source inventory and
// capture coverage. It is a deterministic reference oracle, not a discovery agent.
package inventory

import (
	"fmt"
	"sort"
	"strings"
)

type Verdict string
type MemberState string
type PairState string
type CapabilityStatus string

const (
	Qualified     Verdict = "qualified"
	Blocked       Verdict = "blocked"
	Indeterminate Verdict = "indeterminate"

	Covered     MemberState = "covered"
	Missing     MemberState = "missing"
	Excluded    MemberState = "excluded"
	Unsupported MemberState = "unsupported"
	Unknown     MemberState = "unknown"

	DisjointDomains PairState = "disjoint_declared_domains"
	SharedDomain    PairState = "shared_failure_domain"
	UnknownDomains  PairState = "independence_unknown"

	CapabilitySupported   CapabilityStatus = "supported"
	CapabilityUnsupported CapabilityStatus = "unsupported"
	CapabilityUnknown     CapabilityStatus = "unknown"
)

// Domains are externally attested failure-domain identifiers. A value must
// identify the actual authority/path domain, not merely a witness label.
type Domains struct {
	ControlPlane        string `json:"control_plane"`
	IdentityRoot        string `json:"identity_root"`
	CredentialAuthority string `json:"credential_authority"`
	NetworkPath         string `json:"network_path"`
	TimeSource          string `json:"time_source"`
	AdministrativePath  string `json:"administrative_path"`
	StoragePlane        string `json:"storage_plane"`
}

type Witness struct {
	ID             string   `json:"id"`
	ScopeID        string   `json:"scope_id"`
	TrustQualified bool     `json:"trust_qualified"`
	Complete       bool     `json:"complete"`
	Members        []string `json:"members"`
	Domains        Domains  `json:"domains"`
}

type Capability struct {
	MemberID string           `json:"member_id"`
	Status   CapabilityStatus `json:"status"`
}

type Exclusion struct {
	MemberID   string `json:"member_id"`
	Authorized bool   `json:"authorized"`
	PolicyID   string `json:"policy_id"`
	Reason     string `json:"reason"`
}

type Input struct {
	ScopeID            string       `json:"scope_id"`
	ExpectedWitnessIDs []string     `json:"expected_witness_ids"`
	Witnesses          []Witness    `json:"witnesses"`
	CapturedMembers    []string     `json:"captured_members"`
	Capabilities       []Capability `json:"capabilities"`
	Exclusions         []Exclusion  `json:"exclusions"`
}

type PairResult struct {
	WitnessA string    `json:"witness_a"`
	WitnessB string    `json:"witness_b"`
	State    PairState `json:"state"`
	Shared   []string  `json:"shared_failure_domains"`
	Missing  []string  `json:"missing_failure_domains"`
}

type MemberResult struct {
	MemberID string      `json:"member_id"`
	State    MemberState `json:"state"`
	Reason   string      `json:"reason"`
}

type Metric struct {
	Numerator   int     `json:"numerator"`
	Denominator int     `json:"denominator"`
	Rate        float64 `json:"rate"`
}

type Result struct {
	Profile               string         `json:"profile"`
	ScopeID               string         `json:"scope_id"`
	Verdict               Verdict        `json:"verdict"`
	ScopeStatus           string         `json:"scope_status"`
	InventoryCorroborated bool           `json:"inventory_corroborated"`
	ExpectedMembers       []string       `json:"expected_members"`
	Members               []MemberResult `json:"members"`
	CapturedExtras        []string       `json:"captured_extras"`
	WitnessPairs          []PairResult   `json:"witness_pairs"`
	Coverage              Metric         `json:"coverage"`
	CoveredCount          int            `json:"covered_count"`
	MissingCount          int            `json:"missing_count"`
	ExcludedCount         int            `json:"excluded_count"`
	UnsupportedCount      int            `json:"unsupported_count"`
	UnknownCount          int            `json:"unknown_count"`
	Reasons               []string       `json:"reasons"`
}

var domainFields = []struct {
	name string
	get  func(Domains) string
}{
	{"control_plane", func(d Domains) string { return d.ControlPlane }},
	{"identity_root", func(d Domains) string { return d.IdentityRoot }},
	{"credential_authority", func(d Domains) string { return d.CredentialAuthority }},
	{"network_path", func(d Domains) string { return d.NetworkPath }},
	{"time_source", func(d Domains) string { return d.TimeSource }},
	{"administrative_path", func(d Domains) string { return d.AdministrativePath }},
	{"storage_plane", func(d Domains) string { return d.StoragePlane }},
}

func compareDomains(a, b Domains) PairResult {
	result := PairResult{State: DisjointDomains, Shared: []string{}, Missing: []string{}}
	for _, field := range domainFields {
		av, bv := strings.TrimSpace(field.get(a)), strings.TrimSpace(field.get(b))
		if av == "" || bv == "" {
			result.Missing = append(result.Missing, field.name)
			continue
		}
		if av == bv {
			result.Shared = append(result.Shared, field.name)
		}
	}
	if len(result.Shared) > 0 {
		result.State = SharedDomain
	} else if len(result.Missing) > 0 {
		result.State = UnknownDomains
	}
	return result
}

func sortedUnique(values []string, label string) ([]string, error) {
	out := append([]string(nil), values...)
	for _, value := range out {
		if strings.TrimSpace(value) == "" {
			return nil, fmt.Errorf("%s contains an empty member", label)
		}
	}
	sort.Strings(out)
	for i := 1; i < len(out); i++ {
		if out[i] == out[i-1] {
			return nil, fmt.Errorf("%s contains duplicate %q", label, out[i])
		}
	}
	return out, nil
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func metric(n, d int) Metric {
	rate := 0.0
	if d > 0 {
		rate = float64(n) / float64(d)
	}
	return Metric{Numerator: n, Denominator: d, Rate: rate}
}

// Evaluate requires the supplied witness IDs to match the preregistered set
// exactly, all witnesses to be trust-qualified and complete, exact inventory
// agreement, and pairwise disjoint declared failure-domain vectors.
func Evaluate(in Input) (Result, error) {
	if strings.TrimSpace(in.ScopeID) == "" {
		return Result{}, fmt.Errorf("scope ID is required")
	}
	expectedIDs, err := sortedUnique(in.ExpectedWitnessIDs, "expected witness IDs")
	if err != nil {
		return Result{}, err
	}
	if len(expectedIDs) < 2 {
		return Result{}, fmt.Errorf("at least two preregistered witnesses are required")
	}
	if len(in.Witnesses) == 0 {
		return Result{}, fmt.Errorf("witness list is empty")
	}

	result := Result{
		Profile: "mon-independent-inventory-result-v1", ScopeID: in.ScopeID,
		Verdict: Indeterminate, ScopeStatus: "indeterminate",
		ExpectedMembers: []string{}, Members: []MemberResult{}, CapturedExtras: []string{},
		WitnessPairs: []PairResult{}, Reasons: []string{},
	}
	actualIDs := make([]string, 0, len(in.Witnesses))
	witnessByID := make(map[string]Witness, len(in.Witnesses))
	var canonicalMembers []string
	inventoryConflict := false
	for _, witness := range in.Witnesses {
		if strings.TrimSpace(witness.ID) == "" || witness.ScopeID != in.ScopeID {
			return Result{}, fmt.Errorf("witness identity or scope mismatch")
		}
		if _, exists := witnessByID[witness.ID]; exists {
			return Result{}, fmt.Errorf("duplicate witness ID: %s", witness.ID)
		}
		witnessByID[witness.ID] = witness
		actualIDs = append(actualIDs, witness.ID)
		members, err := sortedUnique(witness.Members, "witness members")
		if err != nil {
			return Result{}, fmt.Errorf("witness %s: %w", witness.ID, err)
		}
		if len(members) == 0 {
			result.Reasons = append(result.Reasons, "empty_witness_inventory:"+witness.ID)
		}
		if canonicalMembers == nil {
			canonicalMembers = members
		} else if !sameStrings(canonicalMembers, members) {
			inventoryConflict = true
		}
		if !witness.TrustQualified {
			result.Reasons = append(result.Reasons, "witness_not_trust_qualified:"+witness.ID)
		}
		if !witness.Complete {
			result.Reasons = append(result.Reasons, "witness_completeness_unproven:"+witness.ID)
		}
	}
	actualIDs, err = sortedUnique(actualIDs, "actual witness IDs")
	if err != nil {
		return Result{}, err
	}
	if !sameStrings(expectedIDs, actualIDs) {
		result.Reasons = append(result.Reasons, "witness_set_differs_from_preregistered_set")
	}
	result.ExpectedMembers = append([]string(nil), canonicalMembers...)
	if len(canonicalMembers) == 0 {
		result.Reasons = append(result.Reasons, "source_inventory_empty")
	}

	pairwiseDisjoint := true
	for i := 0; i < len(actualIDs); i++ {
		for j := i + 1; j < len(actualIDs); j++ {
			a, b := witnessByID[actualIDs[i]], witnessByID[actualIDs[j]]
			pair := compareDomains(a.Domains, b.Domains)
			pair.WitnessA, pair.WitnessB = actualIDs[i], actualIDs[j]
			result.WitnessPairs = append(result.WitnessPairs, pair)
			if pair.State != DisjointDomains {
				pairwiseDisjoint = false
			}
		}
	}
	trustAndCompleteness := true
	for _, witness := range in.Witnesses {
		if !witness.TrustQualified || !witness.Complete {
			trustAndCompleteness = false
		}
	}
	if inventoryConflict {
		result.Reasons = append(result.Reasons, "witness_inventory_disagreement")
	}
	if !pairwiseDisjoint {
		result.Reasons = append(result.Reasons, "witness_failure_domains_not_disjoint_or_unknown")
	}
	if !trustAndCompleteness {
		result.Reasons = append(result.Reasons, "witness_trust_or_completeness_unproven")
	}
	result.InventoryCorroborated = sameStrings(expectedIDs, actualIDs) && trustAndCompleteness && pairwiseDisjoint && !inventoryConflict && len(canonicalMembers) > 0

	captured, err := sortedUnique(in.CapturedMembers, "captured members")
	if err != nil {
		return Result{}, err
	}
	want := make(map[string]struct{}, len(canonicalMembers))
	for _, id := range canonicalMembers {
		want[id] = struct{}{}
	}
	capturedSet := make(map[string]struct{}, len(captured))
	for _, id := range captured {
		capturedSet[id] = struct{}{}
		if _, ok := want[id]; !ok {
			result.CapturedExtras = append(result.CapturedExtras, id)
		}
	}

	capabilities := make(map[string]CapabilityStatus, len(in.Capabilities))
	for _, c := range in.Capabilities {
		if strings.TrimSpace(c.MemberID) == "" {
			return Result{}, fmt.Errorf("capability member ID is empty")
		}
		if _, exists := capabilities[c.MemberID]; exists {
			return Result{}, fmt.Errorf("duplicate capability status: %s", c.MemberID)
		}
		if c.Status != CapabilitySupported && c.Status != CapabilityUnsupported && c.Status != CapabilityUnknown {
			return Result{}, fmt.Errorf("invalid capability status for %s", c.MemberID)
		}
		if _, expected := want[c.MemberID]; !expected {
			result.Reasons = append(result.Reasons, "capability_for_uninventoried_member:"+c.MemberID)
		}
		capabilities[c.MemberID] = c.Status
	}
	exclusions := make(map[string]Exclusion, len(in.Exclusions))
	for _, x := range in.Exclusions {
		if strings.TrimSpace(x.MemberID) == "" {
			return Result{}, fmt.Errorf("exclusion member ID is empty")
		}
		if _, exists := exclusions[x.MemberID]; exists {
			return Result{}, fmt.Errorf("duplicate exclusion: %s", x.MemberID)
		}
		if _, expected := want[x.MemberID]; !expected {
			result.Reasons = append(result.Reasons, "exclusion_for_uninventoried_member:"+x.MemberID)
		}
		exclusions[x.MemberID] = x
	}

	for _, id := range canonicalMembers {
		member := MemberResult{MemberID: id}
		if exclusion, ok := exclusions[id]; ok {
			if _, alsoCaptured := capturedSet[id]; alsoCaptured {
				member.State, member.Reason = Unknown, "excluded_member_also_present_in_capture"
			} else if exclusion.Authorized && strings.TrimSpace(exclusion.PolicyID) != "" && strings.TrimSpace(exclusion.Reason) != "" {
				member.State, member.Reason = Excluded, "explicit_authorized_scope_exclusion"
			} else {
				member.State, member.Reason = Unknown, "exclusion_lacks_authority_policy_or_reason"
			}
		} else {
			status, ok := capabilities[id]
			if !ok || status == CapabilityUnknown {
				member.State, member.Reason = Unknown, "capture_capability_unknown"
			} else if status == CapabilityUnsupported {
				member.State, member.Reason = Unsupported, "required_capture_capability_unsupported"
			} else if _, ok := capturedSet[id]; !ok {
				member.State, member.Reason = Missing, "required_member_absent_from_capture"
			} else {
				member.State, member.Reason = Covered, "captured_with_supported_capability"
			}
		}
		result.Members = append(result.Members, member)
		switch member.State {
		case Covered:
			result.CoveredCount++
		case Missing:
			result.MissingCount++
		case Excluded:
			result.ExcludedCount++
		case Unsupported:
			result.UnsupportedCount++
		case Unknown:
			result.UnknownCount++
		}
	}
	result.CapturedExtras, _ = sortedUnique(result.CapturedExtras, "captured extras")
	result.Coverage = metric(result.CoveredCount, len(canonicalMembers))

	if !result.InventoryCorroborated || len(result.CapturedExtras) > 0 || result.UnsupportedCount > 0 || result.UnknownCount > 0 || hasUninventoriedMetadata(result.Reasons) {
		result.Verdict, result.ScopeStatus = Indeterminate, "indeterminate"
	} else if result.MissingCount > 0 {
		result.Verdict, result.ScopeStatus = Blocked, "incomplete_capture"
	} else if result.ExcludedCount > 0 {
		result.Verdict, result.ScopeStatus = Qualified, "qualified_with_explicit_exclusions"
	} else {
		result.Verdict, result.ScopeStatus = Qualified, "fully_covered"
	}
	return result, nil
}

func hasUninventoriedMetadata(reasons []string) bool {
	for _, reason := range reasons {
		if strings.HasPrefix(reason, "capability_for_uninventoried_member:") || strings.HasPrefix(reason, "exclusion_for_uninventoried_member:") {
			return true
		}
	}
	return false
}
