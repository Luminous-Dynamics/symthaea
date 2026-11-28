// Package composition evaluates a bounded, end-to-end database recovery proof graph.
// It validates typed claims and their declared evidence links; it does not authenticate
// signatures, inspect PostgreSQL, or establish the truth of externally supplied facts.
package composition

import (
	"sort"
	"time"
)

type Stage string

const (
	StageCausal   Stage = "causal_provenance"
	StageTemporal Stage = "temporal_order"
	StageCDC      Stage = "cdc_continuity"
	StageSurface  Stage = "surface_coverage"
	StageRestore  Stage = "restore_to_service"
)

var requiredStages = []Stage{StageCausal, StageTemporal, StageCDC, StageSurface, StageRestore}

type Verdict string

const (
	VerdictQualified     Verdict = "qualified"
	VerdictBlocked       Verdict = "blocked"
	VerdictIndeterminate Verdict = "indeterminate"
)

type EvidenceGrade string

const (
	GradeRuntimeIndependent EvidenceGrade = "runtime_independently_verified"
	GradeStructural         EvidenceGrade = "structurally_checked"
	GradeSynthetic          EvidenceGrade = "synthetic"
)

type CheckState string

const (
	CheckPassed  CheckState = "passed"
	CheckFailed  CheckState = "failed"
	CheckUnknown CheckState = "unknown"
)

type SurfaceScope string

const (
	ScopeRowStream SurfaceScope = "row_stream_continuity"
	ScopeFullState SurfaceScope = "full_recoverable_database_state"
)

// Boundary is the identity tuple that every stage must carry unchanged.
// LSN is treated as an opaque, already-qualified source-position identifier here;
// MON-021 owns PostgreSQL-specific LSN and timeline semantics.
type Boundary struct {
	IncidentID       string `json:"incident_id"`
	ClaimID          string `json:"claim_id"`
	SourceClusterID  string `json:"source_cluster_id"`
	Database         string `json:"database"`
	SourceEpoch      string `json:"source_epoch"`
	TimelineID       string `json:"timeline_id"`
	SnapshotID       string `json:"snapshot_id"`
	TerminalPosition string `json:"terminal_position"`
	ScopeDigest      string `json:"scope_digest"`
	InventoryDigest  string `json:"inventory_digest"`
}

// ParentRef binds a consumer stage to a specific output from a specific parent receipt.
type ParentRef struct {
	Stage        Stage  `json:"stage"`
	EvidenceID   string `json:"evidence_id"`
	OutputDigest string `json:"output_digest"`
}

type SurfaceCounts struct {
	Required      int `json:"required"`
	Covered       int `json:"covered"`
	NotApplicable int `json:"not_applicable"`
	Excluded      int `json:"excluded"`
	Unknown       int `json:"unknown"`
}

// Receipt is a normalized stage claim. Checks are tri-state: an absent or unknown
// check can never be interpreted as a pass.
type Receipt struct {
	Stage                  Stage                 `json:"stage"`
	EvidenceID             string                `json:"evidence_id"`
	SubjectDigest          string                `json:"subject_digest"`
	OutputDigest           string                `json:"output_digest"`
	Boundary               Boundary              `json:"boundary"`
	Parents                []ParentRef           `json:"parents"`
	Status                 Verdict               `json:"status"`
	Grade                  EvidenceGrade         `json:"grade"`
	ProducerControlRoot    string                `json:"producer_control_root"`
	ObserverID             string                `json:"observer_id"`
	ObserverControlRoot    string                `json:"observer_control_root"`
	ClockDomain            string                `json:"clock_domain"`
	ClockUncertainty       time.Duration         `json:"clock_uncertainty_ns"`
	ObservedAt             time.Time             `json:"observed_at"`
	ExpiresAt              time.Time             `json:"expires_at"`
	Checks                 map[string]CheckState `json:"checks"`
	SurfaceScope           SurfaceScope          `json:"surface_scope,omitempty"`
	SurfaceCounts          SurfaceCounts         `json:"surface_counts,omitempty"`
	NotApplicableDigest    string                `json:"not_applicable_digest,omitempty"`
	ExcludedManifestDigest string                `json:"excluded_manifest_digest,omitempty"`
}

type Input struct {
	Now                   time.Time     `json:"now"`
	ClockDomain           string        `json:"clock_domain"`
	MaxClockUncertainty   time.Duration `json:"max_clock_uncertainty_ns"`
	Receipts              []Receipt     `json:"receipts"`
	UnresolvedMutationIDs []string      `json:"unresolved_mutation_ids,omitempty"`
}

type Result struct {
	Verdict                  Verdict       `json:"verdict"`
	MayEnableTraffic         bool          `json:"may_enable_traffic"`
	ReasonCodes              []string      `json:"reason_codes"`
	EvidenceIDs              []string      `json:"evidence_ids"`
	IndependentObserverRoots int           `json:"independent_observer_roots"`
	SurfaceCounts            SurfaceCounts `json:"surface_counts"`
}

var expectedParents = map[Stage][]Stage{
	StageCausal:   {},
	StageTemporal: {StageCausal},
	StageCDC:      {StageTemporal},
	StageSurface:  {StageCDC},
	StageRestore:  {StageCDC, StageSurface},
}

var requiredChecks = map[Stage][]string{
	StageCausal: {
		"inputs_attributed",
		"effects_attributed",
		"authority_bound",
	},
	StageTemporal: {
		"generation_before_use",
		"cross_source_handoffs_witnessed",
	},
	StageCDC: {
		"snapshot_bound",
		"transaction_set_closed",
		"ack_within_durable_watermark",
		"timeline_continuous",
	},
	StageSurface: {
		"inventory_complete",
	},
	StageRestore: {
		"artifact_validated",
		"database_recovered",
		"dependencies_ready",
		"application_read_probe",
		"application_write_readback",
		"write_probe_rollback",
		"application_invariants",
		"old_primary_fenced",
		"stale_endpoints_fenced",
		"independent_service_acceptance",
		"activation_order_verified",
		"traffic_disabled_until_ready",
	},
}

// Evaluate returns a fail-closed result. Contradictions and explicit failures block;
// missing, stale, unknown, or insufficiently qualified evidence is indeterminate.
func Evaluate(in Input) Result {
	r := Result{Verdict: VerdictQualified}
	reasons := map[string]struct{}{}
	blocked := false
	indeterminate := false
	addBlocked := func(code string) {
		blocked = true
		reasons[code] = struct{}{}
	}
	addIndeterminate := func(code string) {
		indeterminate = true
		reasons[code] = struct{}{}
	}

	if in.Now.IsZero() {
		addIndeterminate("evaluation_time_missing")
	}
	if in.ClockDomain == "" || in.MaxClockUncertainty <= 0 {
		addIndeterminate("evaluation_clock_bound_missing")
	}
	if len(in.UnresolvedMutationIDs) > 0 {
		addIndeterminate("unresolved_mutation_outcome")
	}

	byStage := make(map[Stage]Receipt, len(requiredStages))
	byID := make(map[string]Receipt, len(in.Receipts))
	for _, receipt := range in.Receipts {
		if _, ok := expectedParents[receipt.Stage]; !ok {
			addBlocked("unsupported_stage")
			continue
		}
		if _, exists := byStage[receipt.Stage]; exists {
			addBlocked("duplicate_stage_receipt")
			continue
		}
		if receipt.EvidenceID == "" {
			addIndeterminate("evidence_id_missing")
		} else if _, exists := byID[receipt.EvidenceID]; exists {
			addBlocked("duplicate_evidence_id")
		} else {
			byID[receipt.EvidenceID] = receipt
		}
		byStage[receipt.Stage] = receipt
	}

	for _, stage := range requiredStages {
		if _, ok := byStage[stage]; !ok {
			addIndeterminate("required_stage_missing:" + string(stage))
		}
	}
	if len(byStage) != len(requiredStages) {
		addIndeterminate("proof_graph_incomplete")
	}

	var reference Boundary
	haveReference := false
	observerRoots := map[string]struct{}{}
	for _, stage := range requiredStages {
		receipt, ok := byStage[stage]
		if !ok {
			continue
		}
		r.EvidenceIDs = append(r.EvidenceIDs, receipt.EvidenceID)
		if !haveReference {
			reference = receipt.Boundary
			haveReference = true
		} else if receipt.Boundary != reference {
			addBlocked("recovery_boundary_mismatch:" + string(stage))
		}
		if !completeBoundary(receipt.Boundary) {
			addIndeterminate("recovery_boundary_incomplete:" + string(stage))
		}
		if receipt.SubjectDigest == "" || receipt.OutputDigest == "" {
			addIndeterminate("receipt_digest_missing:" + string(stage))
		}
		if receipt.Status == VerdictBlocked {
			addBlocked("stage_claim_blocked:" + string(stage))
		} else if receipt.Status != VerdictQualified {
			addIndeterminate("stage_claim_not_qualified:" + string(stage))
		}
		if receipt.Grade != GradeRuntimeIndependent {
			addIndeterminate("evidence_not_runtime_independently_verified:" + string(stage))
		}
		if receipt.ObserverID == "" || receipt.ObserverControlRoot == "" || receipt.ProducerControlRoot == "" {
			addIndeterminate("observer_identity_incomplete:" + string(stage))
		} else {
			if receipt.ObserverControlRoot == receipt.ProducerControlRoot {
				addBlocked("observer_not_independent:" + string(stage))
			}
			observerRoots[receipt.ObserverControlRoot] = struct{}{}
		}
		if receipt.ClockDomain == "" || receipt.ClockDomain != in.ClockDomain {
			addIndeterminate("evidence_clock_domain_unmatched:" + string(stage))
		}
		if receipt.ClockUncertainty < 0 {
			addBlocked("invalid_clock_uncertainty:" + string(stage))
		} else if in.MaxClockUncertainty > 0 && receipt.ClockUncertainty > in.MaxClockUncertainty {
			addIndeterminate("clock_uncertainty_exceeds_bound:" + string(stage))
		}
		if receipt.ObservedAt.IsZero() || receipt.ExpiresAt.IsZero() || in.Now.IsZero() {
			addIndeterminate("evidence_freshness_unproven:" + string(stage))
		} else {
			if receipt.ObservedAt.After(in.Now) {
				addBlocked("evidence_from_future:" + string(stage))
			}
			if receipt.ExpiresAt.Before(in.Now) {
				addIndeterminate("evidence_expired:" + string(stage))
			}
			if receipt.ExpiresAt.Before(receipt.ObservedAt) {
				addBlocked("invalid_evidence_validity_window:" + string(stage))
			}
		}
		validateParents(receipt, byID, addBlocked, addIndeterminate)
		for _, check := range requiredChecks[stage] {
			switch receipt.Checks[check] {
			case CheckPassed:
			case CheckFailed:
				addBlocked("check_failed:" + string(stage) + ":" + check)
			default:
				addIndeterminate("check_unproven:" + string(stage) + ":" + check)
			}
		}
		if stage == StageSurface {
			r.SurfaceCounts = receipt.SurfaceCounts
			validateSurface(receipt, addBlocked, addIndeterminate)
		}
	}

	r.IndependentObserverRoots = len(observerRoots)
	if len(observerRoots) < 2 {
		addIndeterminate("insufficient_independent_observer_domains")
	}

	sort.Strings(r.EvidenceIDs)
	for code := range reasons {
		r.ReasonCodes = append(r.ReasonCodes, code)
	}
	sort.Strings(r.ReasonCodes)
	switch {
	case blocked:
		r.Verdict = VerdictBlocked
	case indeterminate:
		r.Verdict = VerdictIndeterminate
	default:
		r.Verdict = VerdictQualified
	}
	r.MayEnableTraffic = r.Verdict == VerdictQualified
	return r
}

func completeBoundary(b Boundary) bool {
	return b.IncidentID != "" && b.ClaimID != "" && b.SourceClusterID != "" &&
		b.Database != "" && b.SourceEpoch != "" && b.TimelineID != "" &&
		b.SnapshotID != "" && b.TerminalPosition != "" &&
		b.ScopeDigest != "" && b.InventoryDigest != ""
}

func validateParents(receipt Receipt, byID map[string]Receipt, block, indeterminate func(string)) {
	expected := expectedParents[receipt.Stage]
	if len(receipt.Parents) != len(expected) {
		block("parent_edge_count_mismatch:" + string(receipt.Stage))
		return
	}
	seen := map[Stage]struct{}{}
	for _, parent := range receipt.Parents {
		if _, duplicate := seen[parent.Stage]; duplicate {
			block("duplicate_parent_edge:" + string(receipt.Stage))
			continue
		}
		seen[parent.Stage] = struct{}{}
		allowed := false
		for _, candidate := range expected {
			if candidate == parent.Stage {
				allowed = true
				break
			}
		}
		if !allowed {
			block("unexpected_parent_edge:" + string(receipt.Stage))
			continue
		}
		parentReceipt, ok := byID[parent.EvidenceID]
		if !ok {
			indeterminate("parent_evidence_missing:" + string(receipt.Stage))
			continue
		}
		if parentReceipt.Stage != parent.Stage || parentReceipt.OutputDigest != parent.OutputDigest || parent.OutputDigest == "" {
			block("parent_binding_mismatch:" + string(receipt.Stage))
		}
	}
	for _, stage := range expected {
		if _, ok := seen[stage]; !ok {
			block("required_parent_missing:" + string(receipt.Stage) + ":" + string(stage))
		}
	}
}

func validateSurface(receipt Receipt, block, indeterminate func(string)) {
	c := receipt.SurfaceCounts
	if receipt.SurfaceScope != ScopeRowStream && receipt.SurfaceScope != ScopeFullState {
		block("unsupported_surface_claim_scope")
		return
	}
	if c.Required <= 0 || c.Covered < 0 || c.NotApplicable < 0 || c.Excluded < 0 || c.Unknown < 0 {
		indeterminate("surface_inventory_counts_invalid_or_missing")
		return
	}
	if c.Covered+c.NotApplicable+c.Excluded+c.Unknown > c.Required {
		block("surface_counts_exceed_inventory")
		return
	}
	if c.Covered+c.NotApplicable+c.Excluded+c.Unknown < c.Required {
		indeterminate("surface_inventory_not_fully_accounted")
	}
	if c.Unknown > 0 {
		indeterminate("required_surface_coverage_unknown")
	}
	if c.NotApplicable > 0 && receipt.NotApplicableDigest == "" {
		indeterminate("not_applicable_surfaces_unsubstantiated")
	}
	if receipt.SurfaceScope == ScopeFullState {
		if c.Excluded > 0 {
			block("full_state_claim_contains_exclusions")
		}
		if c.Covered+c.NotApplicable != c.Required {
			indeterminate("full_state_surface_coverage_incomplete")
		}
	}
	if receipt.SurfaceScope == ScopeRowStream {
		if c.Covered == 0 {
			indeterminate("row_stream_claim_covers_no_surfaces")
		}
		if c.Excluded > 0 && receipt.ExcludedManifestDigest == "" {
			indeterminate("row_stream_exclusions_not_manifested")
		}
	}
}
