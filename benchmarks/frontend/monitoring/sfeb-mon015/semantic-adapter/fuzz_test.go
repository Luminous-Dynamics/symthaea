package assurance

import "testing"

func FuzzDigestMutationNeverEqualsAcrossContextChanges(f *testing.F) {
	f.Add(hexDigest('a'), "urn:profile-b")
	f.Add(hexDigest('b'), "urn:profile-a")
	f.Fuzz(func(t *testing.T, value, profile string) {
		if len(value) != 64 { t.Skip() }
		a := requestEvidence(value)
		b := requestEvidence(value)
		if profile == a.Context.Profile { profile = "urn:different-profile" }
		b.Context.Profile = profile
		if CompareDigest(a, b) == DigestEqual { t.Fatal("same raw digest with changed profile was treated equal") }
	})
}

func FuzzOARConfirmationCannotAppearWithoutExecution(f *testing.F) {
	f.Add("confirmed", "execution-1")
	f.Add("not_confirmed", "missing")
	f.Fuzz(func(t *testing.T, result, ref string) {
		if result != "confirmed" && result != "not_confirmed" { t.Skip() }
		root := OARRecord{ReceiptID: "d", AgentID: "a", RunID: "r", ActionType: "refund", ActionTarget: "stripe", Phase: OARDecision, Result: "allowed", Verification: Verified}
		conf := OARRecord{ReceiptID: "c", DecisionRef: ref, AgentID: "a", RunID: "r", ActionType: "refund", ActionTarget: "stripe", Phase: OARConfirmation, Result: result, Verification: Verified}
		got := ProjectOARLifecycle([]OARRecord{root, conf})
		if got.Projection == OARConfirmed { t.Fatal("confirmation without an execution was promoted") }
	})
}

func FuzzAACOverclaimNeverPasses(f *testing.F) {
	f.Add("dispatched", "confirmed")
	f.Add("confirmed", "confirmed")
	f.Fuzz(func(t *testing.T, statusText, claimedText string) {
		status := EffectStatus(statusText)
		claimed := EffectMode(claimedText)
		if status != Planned && status != Dispatched && status != Confirmed && status != Failed && status != Reverted { t.Skip() }
		if claimed != NotApplicable && claimed != DispatchedUnconfirmed && claimed != EffectConfirmed { t.Skip() }
		rec := EffectRecord{Status: status, EffectAttestation: "runtime_claimed", ClaimedMode: claimed}
		if status == Confirmed {
			rec.ResponseDigest = DigestEvidence{Present: true, Valid: true, Algorithm: "sha-256", Value: hexDigest('c')}
		}
		got := AssessEffect(rec)
		rank, _ := effectRank(got.DerivedMode)
		claim, ok := effectRank(claimed)
		if ok && claim > rank && got.Valid { t.Fatalf("overclaim promoted: %+v", got) }
	})
}

func FuzzCrossTenantScopeNeverQualifies(f *testing.F) {
	f.Add("tenant-b")
	f.Add("tenant-c")
	f.Fuzz(func(t *testing.T, tenant string) {
		if tenant == "" || tenant == "tenant-a" { t.Skip() }
		in := validDigitalInput()
		in.ObserverScope.Tenant = tenant
		if got := QualifyDigitalEffect(in).Result; got == QualifiedDigital { t.Fatalf("cross-tenant observer promoted: %q", tenant) }
	})
}

func FuzzOARCrossRunReferenceNeverConfirms(f *testing.F) {
	f.Add("run-b")
	f.Add("attacker-run")
	f.Fuzz(func(t *testing.T, runID string) {
		if runID == "" || runID == "run-a" { t.Skip() }
		root := OARRecord{ReceiptID: "d", AgentID: "a", RunID: "run-a", ActionType: "refund", ActionTarget: "stripe", Phase: OARDecision, Result: "allowed", Verification: Verified}
		exec := OARRecord{ReceiptID: "e", DecisionRef: "d", AgentID: "a", RunID: runID, ActionType: "refund", ActionTarget: "stripe", Phase: OARExecution, Result: "executed", Verification: Verified}
		conf := OARRecord{ReceiptID: "c", DecisionRef: "e", AgentID: "a", RunID: runID, ActionType: "refund", ActionTarget: "stripe", Phase: OARConfirmation, Result: "confirmed", Verification: Verified}
		if got := ProjectOARLifecycle([]OARRecord{root, exec, conf}).Projection; got == OARConfirmed { t.Fatal("cross-run lineage was confirmed") }
	})
}
