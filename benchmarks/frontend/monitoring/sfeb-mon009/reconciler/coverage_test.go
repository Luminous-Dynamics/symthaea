package coverage

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
)

func digestFixture(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func domains(prefix string) Domains {
	return Domains{
		Identity: prefix + "-identity", Privilege: prefix + "-privilege",
		Process: prefix + "-process", Administrator: prefix + "-admin",
		KeyCustody: prefix + "-keys", Storage: prefix + "-storage",
		Network: prefix + "-network", Operator: prefix + "-operator",
	}
}

func baseline() Input {
	m := Manifest{
		BoundaryID: "boundary-01", TenantID: "tenant-a", ResourceID: "resource-7",
		FromSequence: 40, ThroughSequence: 41, InitialRevision: 5, InitialDigest: digestFixture("state-5"), IssuedAtUnix: 100, ExpiresAtUnix: 500,
		AllowedMutationPaths: []string{"sink-procedure-v1"}, ExpectedReadPath: "observer-read-v1",
		RequiredObserverIndependence:  []string{"identity", "privilege", "process", "administrator", "key_custody"},
		RequiredWatermarkIndependence: []string{"identity", "administrator", "key_custody", "operator"},
		WriterDomains:                 domains("writer"),
	}
	digest, err := CanonicalManifestDigest(m)
	if err != nil {
		panic(err)
	}
	return Input{
		Manifest: m, NowUnix: 200, MaxWatermarkAgeSeconds: 300, MaxObservationAgeSeconds: 300,
		Intents:      []Intent{{OperationID: "op-001", TenantID: "tenant-a", ResourceID: "resource-7", RequestDigest: digestFixture("request-1"), AuthorityRevision: "auth-12", FenceToken: 9, MutationPath: "sink-procedure-v1"}},
		Events:       []SinkEvent{{Sequence: 41, OperationID: "op-001", TenantID: "tenant-a", ResourceID: "resource-7", RequestDigest: digestFixture("request-1"), AuthorityRevision: "auth-12", FenceToken: 9, Outcome: "committed", CommittedAtUnix: 180, PriorRevision: 5, PriorDigest: digestFixture("state-5"), ResultRevision: 6, ResultDigest: digestFixture("state-6"), MutationPath: "sink-procedure-v1"}},
		Observations: []Observation{{SinkSequence: 41, OperationID: "op-001", TenantID: "tenant-a", ResourceID: "resource-7", ObservedRevision: 6, ObservedDigest: digestFixture("state-6"), ReadPath: "observer-read-v1", ObservedAtUnix: 190, ReaderDomains: domains("reader")}},
		Watermark:    &Watermark{BoundaryID: "boundary-01", TenantID: "tenant-a", ResourceID: "resource-7", FromSequence: 40, ThroughSequence: 41, ManifestDigest: digest, Authenticated: true, CreatedAtUnix: 195, SignerDomains: domains("witness")},
	}
}

func TestCompleteThreeWayReconciliation(t *testing.T) {
	r := Verify(baseline())
	if !r.EffectQualified {
		t.Fatalf("expected qualified baseline, got %+v", r)
	}
	for name, got := range map[string]Status{
		"manifest": r.Manifest, "watermark": r.Watermark, "sequence": r.SequenceCoverage,
		"intent_sink": r.IntentSinkLinkage, "readback": r.SinkReadback, "independence": r.ObserverIndependence,
	} {
		if got != Verified {
			t.Errorf("%s = %s, want verified", name, got)
		}
	}
}

func TestMissingSinkOutcomeNeverMeansNoEffect(t *testing.T) {
	in := baseline()
	in.Events = nil
	in.Observations = nil
	in.Manifest.ThroughSequence = 40
	digest, _ := CanonicalManifestDigest(in.Manifest)
	in.Watermark.ThroughSequence = 40
	in.Watermark.ManifestDigest = digest
	r := Verify(in)
	if r.IntentSinkLinkage != Missing {
		t.Fatalf("intent/sink = %s, want missing", r.IntentSinkLinkage)
	}
	if r.EffectQualified {
		t.Fatal("missing sink outcome must not qualify an effect")
	}
}

func TestRequestSubstitutionIsContradicted(t *testing.T) {
	in := baseline()
	in.Events[0].RequestDigest = digestFixture("attacker")
	r := Verify(in)
	if r.IntentSinkLinkage != Contradicted {
		t.Fatalf("got %s, want contradicted", r.IntentSinkLinkage)
	}
	if r.EffectQualified {
		t.Fatal("substituted request must not qualify")
	}
}

func TestReadbackMustMatchExactRevisionAndDigest(t *testing.T) {
	in := baseline()
	in.Observations[0].ObservedRevision = 5
	in.Observations[0].ObservedDigest = digestFixture("old-state")
	r := Verify(in)
	if r.SinkReadback != Contradicted {
		t.Fatalf("got %s, want contradicted", r.SinkReadback)
	}
}

func TestSequenceGapMakesCoverageMissing(t *testing.T) {
	in := baseline()
	in.Manifest.FromSequence = 39
	in.Manifest.ThroughSequence = 41
	in.Watermark.FromSequence = 39
	digest, _ := CanonicalManifestDigest(in.Manifest)
	in.Watermark.ManifestDigest = digest
	r := Verify(in)
	if r.SequenceCoverage != Missing {
		t.Fatalf("got %s, want missing", r.SequenceCoverage)
	}
	if r.EffectQualified {
		t.Fatal("sequence gap must not qualify coverage")
	}
}

func TestTailTruncationDetectedByExternalWatermark(t *testing.T) {
	in := baseline()
	in.Manifest.FromSequence = 39
	in.Manifest.ThroughSequence = 42
	in.Watermark.FromSequence = 39
	in.Watermark.ThroughSequence = 42
	digest, _ := CanonicalManifestDigest(in.Manifest)
	in.Watermark.ManifestDigest = digest
	r := Verify(in)
	if r.SequenceCoverage != Missing {
		t.Fatalf("got %s, want missing", r.SequenceCoverage)
	}
}

func TestSharedAdministratorBreaksObserverIndependence(t *testing.T) {
	in := baseline()
	in.Observations[0].ReaderDomains.Administrator = in.Manifest.WriterDomains.Administrator
	r := Verify(in)
	if r.ObserverIndependence != Contradicted {
		t.Fatalf("got %s, want contradicted", r.ObserverIndependence)
	}
	if r.EffectQualified {
		t.Fatal("shared administrator must prevent independent effect qualification")
	}
}

func TestMissingObserverDomainIsIndeterminate(t *testing.T) {
	in := baseline()
	in.Observations[0].ReaderDomains.Identity = ""
	r := Verify(in)
	if r.ObserverIndependence != Indeterminate {
		t.Fatalf("got %s, want indeterminate", r.ObserverIndependence)
	}
}

func TestStaleWatermarkDoesNotEraseHistoricalProofButBlocksCurrentQualification(t *testing.T) {
	in := baseline()
	in.NowUnix = 10_000
	r := Verify(in)
	if r.Watermark != Verified {
		t.Fatalf("historical watermark = %s, want verified", r.Watermark)
	}
	if r.WatermarkFreshness != Stale {
		t.Fatalf("freshness = %s, want stale", r.WatermarkFreshness)
	}
	if r.EffectQualified {
		t.Fatal("stale watermark must not qualify current effect")
	}
}

func TestUnauthenticatedWatermarkIsMissing(t *testing.T) {
	in := baseline()
	in.Watermark.Authenticated = false
	r := Verify(in)
	if r.Watermark != Missing {
		t.Fatalf("got %s, want missing", r.Watermark)
	}
	if r.EffectQualified {
		t.Fatal("unauthenticated watermark must not qualify")
	}
}

func TestObservationWithoutSinkCommitIsContradicted(t *testing.T) {
	in := baseline()
	in.Events[0].Outcome = "rejected"
	r := Verify(in)
	if r.SinkReadback != Contradicted {
		t.Fatalf("got %s, want contradicted", r.SinkReadback)
	}
}

func TestOutOfBoundaryMutationPathIsContradicted(t *testing.T) {
	in := baseline()
	in.Events[0].MutationPath = "unmediated-sql"
	in.Intents[0].MutationPath = "unmediated-sql"
	r := Verify(in)
	if r.IntentSinkLinkage != Contradicted {
		t.Fatalf("got %s, want contradicted", r.IntentSinkLinkage)
	}
}

func TestTamperedManifestBindingRejectsWatermark(t *testing.T) {
	in := baseline()
	in.Manifest.ExpectedReadPath = "attacker-read-path"
	r := Verify(in)
	if r.Watermark != Contradicted {
		t.Fatalf("got %s, want contradicted", r.Watermark)
	}
}

func TestDuplicateOperationIDIsContradicted(t *testing.T) {
	in := baseline()
	in.Events = append(in.Events, in.Events[0])
	in.Manifest.ThroughSequence = 42
	in.Events[1].Sequence = 42
	in.Watermark.ThroughSequence = 42
	digest, _ := CanonicalManifestDigest(in.Manifest)
	in.Watermark.ManifestDigest = digest
	r := Verify(in)
	if r.IntentSinkLinkage != Contradicted {
		t.Fatalf("got %s, want contradicted", r.IntentSinkLinkage)
	}
}

func TestNonRequiredSharedStorageIsDisclosedWithoutFailingDeclaredPolicy(t *testing.T) {
	in := baseline()
	in.Observations[0].ReaderDomains.Storage = in.Manifest.WriterDomains.Storage
	r := Verify(in)
	if r.ObserverIndependence != Verified {
		t.Fatalf("got %s, want verified under declared dimensions", r.ObserverIndependence)
	}
	found := false
	for _, f := range r.Findings {
		if f.Dimension == "storage" && f.Status == Contradicted {
			found = true
		}
	}
	if !found {
		t.Fatal("shared storage domain must remain visible in findings")
	}
}

func TestSharedWatermarkOperatorDomainBreaksWatermarkIndependence(t *testing.T) {
	in := baseline()
	in.Watermark.SignerDomains.Operator = in.Manifest.WriterDomains.Operator
	r := Verify(in)
	if r.WatermarkIndependence != Contradicted {
		t.Fatalf("got %s, want contradicted", r.WatermarkIndependence)
	}
	if r.EffectQualified {
		t.Fatal("watermark signed in the writer operator domain must not qualify")
	}
}

func TestManifestDigestBindsIndependencePolicy(t *testing.T) {
	in := baseline()
	d1, err := CanonicalManifestDigest(in.Manifest)
	if err != nil {
		t.Fatal(err)
	}
	in.Manifest.RequiredObserverIndependence = []string{"identity", "privilege"}
	d2, err := CanonicalManifestDigest(in.Manifest)
	if err != nil {
		t.Fatal(err)
	}
	if d1 == d2 {
		t.Fatal("manifest digest must bind the independence policy")
	}
}

func TestStaleReadbackRetainsHistoricalMatchButBlocksCurrentQualification(t *testing.T) {
	in := baseline()
	in.NowUnix = 10_000
	r := Verify(in)
	if r.SinkReadback != Verified {
		t.Fatalf("historical readback = %s, want verified", r.SinkReadback)
	}
	if r.SinkReadbackFreshness != Stale {
		t.Fatalf("readback freshness = %s, want stale", r.SinkReadbackFreshness)
	}
	if r.EffectQualified {
		t.Fatal("stale readback must not qualify a current effect")
	}
}

func TestReadbackBeforeCommitIsIndeterminate(t *testing.T) {
	in := baseline()
	in.Observations[0].ObservedAtUnix = 170
	r := Verify(in)
	if r.SinkReadbackFreshness != Indeterminate {
		t.Fatalf("readback freshness = %s, want indeterminate", r.SinkReadbackFreshness)
	}
	if r.EffectQualified {
		t.Fatal("pre-commit readback must not qualify")
	}
}

func TestFreshnessBoundRequiresAuditClock(t *testing.T) {
	in := baseline()
	in.NowUnix = 0
	r := Verify(in)
	if r.SinkReadbackFreshness != Indeterminate {
		t.Fatalf("readback freshness = %s, want indeterminate", r.SinkReadbackFreshness)
	}
	if r.EffectQualified {
		t.Fatal("unknown audit time must not qualify current effect")
	}
}

func TestMaximumUint64SequenceRejected(t *testing.T) {
	in := baseline()
	in.Manifest.ThroughSequence = ^uint64(0)
	r := Verify(in)
	if r.Manifest != Contradicted {
		t.Fatalf("manifest = %s, want contradicted", r.Manifest)
	}
	if r.EffectQualified {
		t.Fatal("overflowing sequence range must not qualify")
	}
}

func TestCommittedEventRequiresMonotonicRevisionAndDigest(t *testing.T) {
	in := baseline()
	in.Events[0].ResultRevision = in.Events[0].PriorRevision
	in.Events[0].ResultDigest = ""
	r := Verify(in)
	if r.IntentSinkLinkage != Contradicted {
		t.Fatalf("intent/sink = %s, want contradicted", r.IntentSinkLinkage)
	}
	if r.EffectQualified {
		t.Fatal("empty/non-advancing state transition must not qualify")
	}
}

func TestCommitOutsideBoundaryValidityIntervalIsContradicted(t *testing.T) {
	in := baseline()
	in.Events[0].CommittedAtUnix = 501
	r := Verify(in)
	if r.IntentSinkLinkage != Contradicted {
		t.Fatalf("intent/sink = %s, want contradicted", r.IntentSinkLinkage)
	}
	if r.EffectQualified {
		t.Fatal("commit outside boundary validity interval must not qualify")
	}
}

func TestWatermarkOutsideBoundaryValidityIntervalIsContradicted(t *testing.T) {
	in := baseline()
	in.Watermark.CreatedAtUnix = 501
	r := Verify(in)
	if r.Watermark != Contradicted {
		t.Fatalf("watermark = %s, want contradicted", r.Watermark)
	}
}

func TestRequestDigestMustBeCanonicalSHA256Hex(t *testing.T) {
	for _, bad := range []string{"sha256:request", "", " " + digestFixture("request-1"), strings.ToUpper(digestFixture("request-1"))} {
		in := baseline()
		in.Intents[0].RequestDigest = bad
		r := Verify(in)
		if r.IntentSinkLinkage != Contradicted || r.EffectQualified {
			t.Fatalf("digest %q accepted: %+v", bad, r)
		}
	}
}

func TestCommittedHistoryRequiresRevisionContinuity(t *testing.T) {
	in := baseline()
	in.Manifest.ThroughSequence = 42
	in.Intents = append(in.Intents, Intent{OperationID: "op-002", TenantID: "tenant-a", ResourceID: "resource-7", RequestDigest: digestFixture("request-2"), AuthorityRevision: "auth-12", FenceToken: 9, MutationPath: "sink-procedure-v1"})
	in.Events = append(in.Events, SinkEvent{Sequence: 42, OperationID: "op-002", TenantID: "tenant-a", ResourceID: "resource-7", RequestDigest: digestFixture("request-2"), AuthorityRevision: "auth-12", FenceToken: 9, Outcome: "committed", CommittedAtUnix: 185, PriorRevision: 99, PriorDigest: digestFixture("state-6"), ResultRevision: 100, ResultDigest: digestFixture("state-100"), MutationPath: "sink-procedure-v1"})
	in.Observations = append(in.Observations, Observation{SinkSequence: 42, OperationID: "op-002", TenantID: "tenant-a", ResourceID: "resource-7", ObservedRevision: 100, ObservedDigest: digestFixture("state-100"), ReadPath: "observer-read-v1", ObservedAtUnix: 195, ReaderDomains: domains("reader")})
	in.Watermark.ThroughSequence = 42
	digest, err := CanonicalManifestDigest(in.Manifest)
	if err != nil {
		t.Fatal(err)
	}
	in.Watermark.ManifestDigest = digest
	r := Verify(in)
	if r.IntentSinkLinkage != Contradicted || r.EffectQualified {
		t.Fatalf("revision discontinuity not rejected: %+v", r)
	}
}

func TestCommittedHistoryRejectsFencingRegression(t *testing.T) {
	in := baseline()
	in.Manifest.ThroughSequence = 42
	in.Intents = append(in.Intents, Intent{OperationID: "op-002", TenantID: "tenant-a", ResourceID: "resource-7", RequestDigest: digestFixture("request-2"), AuthorityRevision: "auth-12", FenceToken: 8, MutationPath: "sink-procedure-v1"})
	in.Events = append(in.Events, SinkEvent{Sequence: 42, OperationID: "op-002", TenantID: "tenant-a", ResourceID: "resource-7", RequestDigest: digestFixture("request-2"), AuthorityRevision: "auth-12", FenceToken: 8, Outcome: "committed", CommittedAtUnix: 185, PriorRevision: 6, PriorDigest: digestFixture("state-6"), ResultRevision: 7, ResultDigest: digestFixture("state-7"), MutationPath: "sink-procedure-v1"})
	in.Observations = append(in.Observations, Observation{SinkSequence: 42, OperationID: "op-002", TenantID: "tenant-a", ResourceID: "resource-7", ObservedRevision: 7, ObservedDigest: digestFixture("state-7"), ReadPath: "observer-read-v1", ObservedAtUnix: 195, ReaderDomains: domains("reader")})
	in.Watermark.ThroughSequence = 42
	digest, err := CanonicalManifestDigest(in.Manifest)
	if err != nil {
		t.Fatal(err)
	}
	in.Watermark.ManifestDigest = digest
	r := Verify(in)
	if r.IntentSinkLinkage != Contradicted || r.EffectQualified {
		t.Fatalf("fencing regression not rejected: %+v", r)
	}
}

func TestFirstCommittedEventMustMatchAnchoredBaseline(t *testing.T) {
	in := baseline()
	in.Manifest.InitialRevision = 4
	digest, err := CanonicalManifestDigest(in.Manifest)
	if err != nil {
		t.Fatal(err)
	}
	in.Watermark.ManifestDigest = digest
	r := Verify(in)
	if r.IntentSinkLinkage != Contradicted || r.EffectQualified {
		t.Fatalf("first event was not bound to anchored initial revision: %+v", r)
	}
}

func TestFirstCommittedEventMustMatchAnchoredBaselineDigest(t *testing.T) {
	in := baseline()
	in.Manifest.InitialDigest = digestFixture("unrelated-state")
	digest, err := CanonicalManifestDigest(in.Manifest)
	if err != nil {
		t.Fatal(err)
	}
	in.Watermark.ManifestDigest = digest
	r := Verify(in)
	if r.IntentSinkLinkage != Contradicted || r.EffectQualified {
		t.Fatalf("first event was not bound to anchored initial digest: %+v", r)
	}
}
