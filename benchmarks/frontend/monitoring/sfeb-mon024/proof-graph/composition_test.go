package composition

import (
	"strings"
	"testing"
	"time"
)

func fixture() Input {
	now := time.Unix(1_800_000_000, 0).UTC()
	boundary := Boundary{
		IncidentID: "incident-01", ClaimID: "claim-01", SourceClusterID: "pg-cluster-01",
		Database: "ledger", SourceEpoch: "epoch-7", TimelineID: "timeline-4",
		SnapshotID: "snapshot-abc", TerminalPosition: "0/ABCDEF0",
		ScopeDigest: "sha256:scope", InventoryDigest: "sha256:inventory",
	}
	mk := func(stage Stage, id, output, observerRoot string, parents []ParentRef, checks []string) Receipt {
		m := make(map[string]CheckState, len(checks))
		for _, key := range checks {
			m[key] = CheckPassed
		}
		return Receipt{
			Stage: stage, EvidenceID: id, SubjectDigest: "sha256:subject-" + string(stage),
			OutputDigest: output, Boundary: boundary, Parents: parents,
			Status: VerdictQualified, Grade: GradeRuntimeIndependent,
			ProducerControlRoot: "producer-root", ObserverID: "observer-" + string(stage),
			ObserverControlRoot: observerRoot, ClockDomain: "qualification-clock",
			ClockUncertainty: 5 * time.Millisecond, ObservedAt: now.Add(-time.Minute),
			ExpiresAt: now.Add(time.Hour), Checks: m,
		}
	}
	causal := mk(StageCausal, "ev-causal", "digest-causal", "observer-root-a", nil,
		[]string{"inputs_attributed", "effects_attributed", "authority_bound"})
	temporal := mk(StageTemporal, "ev-temporal", "digest-temporal", "observer-root-b",
		[]ParentRef{{Stage: StageCausal, EvidenceID: causal.EvidenceID, OutputDigest: causal.OutputDigest}},
		[]string{"generation_before_use", "cross_source_handoffs_witnessed"})
	cdc := mk(StageCDC, "ev-cdc", "digest-cdc", "observer-root-a",
		[]ParentRef{{Stage: StageTemporal, EvidenceID: temporal.EvidenceID, OutputDigest: temporal.OutputDigest}},
		[]string{"snapshot_bound", "transaction_set_closed", "ack_within_durable_watermark", "timeline_continuous"})
	surface := mk(StageSurface, "ev-surface", "digest-surface", "observer-root-b",
		[]ParentRef{{Stage: StageCDC, EvidenceID: cdc.EvidenceID, OutputDigest: cdc.OutputDigest}},
		[]string{"inventory_complete"})
	surface.SurfaceScope = ScopeFullState
	surface.SurfaceCounts = SurfaceCounts{Required: 14, Covered: 12, NotApplicable: 2}
	surface.NotApplicableDigest = "sha256:na-manifest"
	restore := mk(StageRestore, "ev-restore", "digest-restore", "observer-root-a",
		[]ParentRef{
			{Stage: StageCDC, EvidenceID: cdc.EvidenceID, OutputDigest: cdc.OutputDigest},
			{Stage: StageSurface, EvidenceID: surface.EvidenceID, OutputDigest: surface.OutputDigest},
		},
		[]string{
			"artifact_validated", "database_recovered", "dependencies_ready", "application_read_probe",
			"application_write_readback", "write_probe_rollback", "application_invariants",
			"old_primary_fenced", "stale_endpoints_fenced", "independent_service_acceptance",
			"activation_order_verified", "traffic_disabled_until_ready",
		})
	return Input{Now: now, ClockDomain: "qualification-clock", MaxClockUncertainty: 100 * time.Millisecond,
		Receipts: []Receipt{causal, temporal, cdc, surface, restore}}
}

func hasReason(r Result, needle string) bool {
	for _, reason := range r.ReasonCodes {
		if strings.Contains(reason, needle) {
			return true
		}
	}
	return false
}

func TestPositiveChainQualifiesAndTrafficMayBeEnabled(t *testing.T) {
	r := Evaluate(fixture())
	if r.Verdict != VerdictQualified || !r.MayEnableTraffic {
		t.Fatalf("expected qualified chain, got %+v", r)
	}
	if r.SurfaceCounts.Covered != 12 || r.SurfaceCounts.NotApplicable != 2 {
		t.Fatalf("NotApplicable must remain distinct from Covered: %+v", r.SurfaceCounts)
	}
}

func TestMissingStageIsIndeterminate(t *testing.T) {
	in := fixture()
	in.Receipts = in.Receipts[:4]
	r := Evaluate(in)
	if r.Verdict != VerdictIndeterminate || r.MayEnableTraffic || !hasReason(r, "required_stage_missing") {
		t.Fatalf("expected missing-stage indeterminate result, got %+v", r)
	}
}

func TestBoundaryMismatchBlocks(t *testing.T) {
	in := fixture()
	in.Receipts[2].Boundary.SourceEpoch = "epoch-other"
	r := Evaluate(in)
	if r.Verdict != VerdictBlocked || r.MayEnableTraffic || !hasReason(r, "recovery_boundary_mismatch") {
		t.Fatalf("expected boundary mismatch to block, got %+v", r)
	}
}

func TestParentDigestMismatchBlocks(t *testing.T) {
	in := fixture()
	in.Receipts[1].Parents[0].OutputDigest = "digest-substituted"
	r := Evaluate(in)
	if r.Verdict != VerdictBlocked || !hasReason(r, "parent_binding_mismatch") {
		t.Fatalf("expected parent substitution to block, got %+v", r)
	}
}

func TestExpiredEvidenceIsIndeterminate(t *testing.T) {
	in := fixture()
	in.Receipts[0].ExpiresAt = in.Now.Add(-time.Second)
	r := Evaluate(in)
	if r.Verdict != VerdictIndeterminate || r.MayEnableTraffic || !hasReason(r, "evidence_expired") {
		t.Fatalf("expected expired evidence to be indeterminate, got %+v", r)
	}
}

func TestObserverSharingProducerControlRootBlocks(t *testing.T) {
	in := fixture()
	in.Receipts[3].ObserverControlRoot = in.Receipts[3].ProducerControlRoot
	r := Evaluate(in)
	if r.Verdict != VerdictBlocked || !hasReason(r, "observer_not_independent") {
		t.Fatalf("expected observer alias to block, got %+v", r)
	}
}

func TestSingleObserverDomainIsIndeterminate(t *testing.T) {
	in := fixture()
	for i := range in.Receipts {
		in.Receipts[i].ObserverControlRoot = "one-root"
	}
	r := Evaluate(in)
	if r.Verdict != VerdictIndeterminate || !hasReason(r, "insufficient_independent_observer_domains") {
		t.Fatalf("expected insufficient observer diversity, got %+v", r)
	}
}

func TestUnknownMutationOutcomeBlocksTraffic(t *testing.T) {
	in := fixture()
	in.UnresolvedMutationIDs = []string{"restore-promote-command-17"}
	r := Evaluate(in)
	if r.Verdict != VerdictIndeterminate || r.MayEnableTraffic || !hasReason(r, "unresolved_mutation_outcome") {
		t.Fatalf("expected unresolved mutation to hold traffic, got %+v", r)
	}
}

func TestFailedCDCAcknowledgementCheckBlocks(t *testing.T) {
	in := fixture()
	in.Receipts[2].Checks["ack_within_durable_watermark"] = CheckFailed
	r := Evaluate(in)
	if r.Verdict != VerdictBlocked || !hasReason(r, "ack_within_durable_watermark") {
		t.Fatalf("expected unsafe acknowledgement to block, got %+v", r)
	}
}

func TestUnknownCheckIsIndeterminate(t *testing.T) {
	in := fixture()
	in.Receipts[1].Checks["cross_source_handoffs_witnessed"] = CheckUnknown
	r := Evaluate(in)
	if r.Verdict != VerdictIndeterminate || r.MayEnableTraffic {
		t.Fatalf("expected unknown check to be indeterminate, got %+v", r)
	}
}

func TestFullStateClaimCannotExcludeRequiredSurface(t *testing.T) {
	in := fixture()
	in.Receipts[3].SurfaceCounts = SurfaceCounts{Required: 14, Covered: 12, NotApplicable: 1, Excluded: 1}
	in.Receipts[3].ExcludedManifestDigest = "sha256:excluded"
	r := Evaluate(in)
	if r.Verdict != VerdictBlocked || !hasReason(r, "full_state_claim_contains_exclusions") {
		t.Fatalf("expected full-state exclusion to block, got %+v", r)
	}
}

func TestRowStreamAllowsManifestedExclusions(t *testing.T) {
	in := fixture()
	in.Receipts[3].SurfaceScope = ScopeRowStream
	in.Receipts[3].SurfaceCounts = SurfaceCounts{Required: 14, Covered: 8, NotApplicable: 1, Excluded: 5}
	in.Receipts[3].ExcludedManifestDigest = "sha256:excluded"
	r := Evaluate(in)
	if r.Verdict != VerdictQualified || !r.MayEnableTraffic {
		t.Fatalf("scoped row-stream claim with explicit exclusions should qualify: %+v", r)
	}
}

func TestUnknownSurfaceCannotQualify(t *testing.T) {
	in := fixture()
	in.Receipts[3].SurfaceCounts = SurfaceCounts{Required: 14, Covered: 12, NotApplicable: 1, Unknown: 1}
	in.Receipts[3].NotApplicableDigest = "sha256:na-manifest"
	r := Evaluate(in)
	if r.Verdict != VerdictIndeterminate || r.MayEnableTraffic {
		t.Fatalf("unknown surface must prevent qualification: %+v", r)
	}
}

func TestWrongParentEdgeBlocks(t *testing.T) {
	in := fixture()
	in.Receipts[2].Parents[0].Stage = StageCausal
	r := Evaluate(in)
	if r.Verdict != VerdictBlocked || !hasReason(r, "unexpected_parent_edge") {
		t.Fatalf("expected wrong edge to block, got %+v", r)
	}
}

func TestDuplicateStageBlocks(t *testing.T) {
	in := fixture()
	in.Receipts = append(in.Receipts, in.Receipts[0])
	r := Evaluate(in)
	if r.Verdict != VerdictBlocked || !hasReason(r, "duplicate_stage_receipt") {
		t.Fatalf("expected duplicate stage to block, got %+v", r)
	}
}

func TestMissingRollbackBlocksServiceClaim(t *testing.T) {
	in := fixture()
	in.Receipts[4].Checks["write_probe_rollback"] = CheckFailed
	r := Evaluate(in)
	if r.Verdict != VerdictBlocked || r.MayEnableTraffic || !hasReason(r, "write_probe_rollback") {
		t.Fatalf("write probe without rollback must block service claim: %+v", r)
	}
}

func TestAlreadyEnabledTrafficBlocksOrderedActivationClaim(t *testing.T) {
	in := fixture()
	in.Receipts[4].Checks["traffic_disabled_until_ready"] = CheckFailed
	r := Evaluate(in)
	if r.Verdict != VerdictBlocked || !hasReason(r, "traffic_disabled_until_ready") {
		t.Fatalf("traffic enabled before proof must block: %+v", r)
	}
}

func TestStructuralEvidenceCannotQualifyProductionClaim(t *testing.T) {
	in := fixture()
	in.Receipts[2].Grade = GradeStructural
	r := Evaluate(in)
	if r.Verdict != VerdictIndeterminate || r.MayEnableTraffic || !hasReason(r, "evidence_not_runtime_independently_verified") {
		t.Fatalf("structural evidence must not qualify production claim: %+v", r)
	}
}

func FuzzEvaluateNeverPanics(f *testing.F) {
	f.Add("ev-causal")
	f.Add("")
	f.Add("duplicate")
	f.Fuzz(func(t *testing.T, evidenceID string) {
		in := fixture()
		in.Receipts[0].EvidenceID = evidenceID
		_ = Evaluate(in)
	})
}

func TestAdditionalAdversarialControls(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(*Input)
		verdict Verdict
		reason  string
	}{
		{
			name:    "duplicate evidence identity",
			mutate:  func(in *Input) { in.Receipts[1].EvidenceID = in.Receipts[0].EvidenceID },
			verdict: VerdictBlocked, reason: "duplicate_evidence_id",
		},
		{
			name:    "future-dated evidence",
			mutate:  func(in *Input) { in.Receipts[0].ObservedAt = in.Now.Add(time.Second) },
			verdict: VerdictBlocked, reason: "evidence_from_future",
		},
		{
			name:    "unsubstantiated not-applicable surface",
			mutate:  func(in *Input) { in.Receipts[3].NotApplicableDigest = "" },
			verdict: VerdictIndeterminate, reason: "not_applicable_surfaces_unsubstantiated",
		},
		{
			name: "row-stream exclusions lack manifest",
			mutate: func(in *Input) {
				in.Receipts[3].SurfaceScope = ScopeRowStream
				in.Receipts[3].SurfaceCounts = SurfaceCounts{Required: 14, Covered: 8, Excluded: 6}
				in.Receipts[3].ExcludedManifestDigest = ""
			},
			verdict: VerdictIndeterminate, reason: "row_stream_exclusions_not_manifested",
		},
		{
			name:    "dependency readiness failed",
			mutate:  func(in *Input) { in.Receipts[4].Checks["dependencies_ready"] = CheckFailed },
			verdict: VerdictBlocked, reason: "dependencies_ready",
		},
		{
			name:    "old primary not fenced",
			mutate:  func(in *Input) { in.Receipts[4].Checks["old_primary_fenced"] = CheckFailed },
			verdict: VerdictBlocked, reason: "old_primary_fenced",
		},
		{
			name:    "activation order unknown",
			mutate:  func(in *Input) { in.Receipts[4].Checks["activation_order_verified"] = CheckUnknown },
			verdict: VerdictIndeterminate, reason: "activation_order_verified",
		},
		{
			name:    "independent service acceptance missing",
			mutate:  func(in *Input) { delete(in.Receipts[4].Checks, "independent_service_acceptance") },
			verdict: VerdictIndeterminate, reason: "independent_service_acceptance",
		},
		{
			name:    "evidence clock domain mismatch",
			mutate:  func(in *Input) { in.Receipts[2].ClockDomain = "untrusted-clock" },
			verdict: VerdictIndeterminate, reason: "evidence_clock_domain_unmatched",
		},
		{
			name:    "clock uncertainty exceeds bound",
			mutate:  func(in *Input) { in.Receipts[0].ClockUncertainty = time.Second },
			verdict: VerdictIndeterminate, reason: "clock_uncertainty_exceeds_bound",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			in := fixture()
			tc.mutate(&in)
			r := Evaluate(in)
			if r.Verdict != tc.verdict || !hasReason(r, tc.reason) {
				t.Fatalf("expected %s with reason containing %q, got %+v", tc.verdict, tc.reason, r)
			}
		})
	}
}
