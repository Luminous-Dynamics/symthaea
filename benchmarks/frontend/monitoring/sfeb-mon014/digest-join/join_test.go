package crossprofile

import "testing"

func sha(ch byte) string {
	b := make([]byte, 64)
	for i := range b {
		b[i] = ch
	}
	return string(b)
}

func ctx() DigestContext {
	return DigestContext{
		Profile: "urn:example:request-profile",
		Version: "1",
		PayloadType: "application/json",
		Canonicalization: "RFC8785-JCS",
		Encoding: "UTF-8",
		Purpose: "request",
		Subject: "destination-a/operation-1",
	}
}

func ref(value string) DigestRef {
	return DigestRef{
		Algorithm: "sha-256",
		Value: value,
		Context: ctx(),
		Verification: Verified,
	}
}

func TestDigestEqualityRequiresSameDeclaredContext(t *testing.T) {
	a, b := ref(sha('a')), ref(sha('a'))
	if got := CompareDigestRefs(a, b); got != DigestEqual {
		t.Fatalf("same context/digest => %s", got)
	}
	b.Context.Profile = "urn:other:profile"
	if got := CompareDigestRefs(a, b); got != DigestIncomparable {
		t.Fatalf("equal-looking hex across profiles => %s, want incomparable", got)
	}
}

func TestDigestEqualityRequiresSameCanonicalizationAndPurpose(t *testing.T) {
	a, b := ref(sha('a')), ref(sha('a'))
	b.Context.Canonicalization = "raw-json-bytes"
	if got := CompareDigestRefs(a, b); got != DigestIncomparable {
		t.Fatalf("different canonicalization => %s", got)
	}
	b = ref(sha('a'))
	b.Context.Purpose = "response"
	if got := CompareDigestRefs(a, b); got != DigestIncomparable {
		t.Fatalf("different purpose => %s", got)
	}
}

func TestDigestDifferentWithSameContextIsDifferent(t *testing.T) {
	if got := CompareDigestRefs(ref(sha('a')), ref(sha('b'))); got != DigestDifferent {
		t.Fatalf("got %s, want different", got)
	}
}

func TestDigestUnknownAlgorithmAndUnverifiedEvidenceNeverCompareEqual(t *testing.T) {
	a, b := ref(sha('a')), ref(sha('a'))
	b.Algorithm = "sha-512"
	if got := CompareDigestRefs(a, b); got != DigestIncomparable {
		t.Fatalf("different algorithm => %s", got)
	}
	b = ref(sha('a'))
	b.Verification = Missing
	if got := CompareDigestRefs(a, b); got != DigestIndeterminate {
		t.Fatalf("unverified evidence => %s", got)
	}
}

func TestDigestMalformedAndInvalidEvidenceAreNotAccepted(t *testing.T) {
	a, b := ref(sha('a')), ref("not-a-digest")
	if got := CompareDigestRefs(a, b); got != DigestInvalid {
		t.Fatalf("malformed digest => %s", got)
	}
	b = ref(sha('a'))
	b.Verification = Invalid
	if got := CompareDigestRefs(a, b); got != DigestInvalid {
		t.Fatalf("invalid verifier result => %s", got)
	}
}

func TestCapsuleEffectAttestationGradesStayOrthogonalAndFailClosed(t *testing.T) {
	gate := AssessCapsuleEffect(Confirmed, string(GateExecuted), true)
	if !gate.Valid || gate.Grade != GateObserved {
		t.Fatalf("gate_executed assessment: %+v", gate)
	}
	runtime := AssessCapsuleEffect(Confirmed, string(RuntimeClaimed), true)
	host := AssessCapsuleEffect(Confirmed, string(HostServedObserved), true)
	if !runtime.Valid || runtime.Grade != RuntimeAssertion || host.Grade != RuntimeAssertion {
		t.Fatalf("runtime and host must be equal grade: runtime=%+v host=%+v", runtime, host)
	}
}

func TestUnknownCapsuleAttestationIsInformationalAndNeverGradesUp(t *testing.T) {
	got := AssessCapsuleEffect(Confirmed, "future_registry_value", true)
	if !got.Valid || !got.UnknownAttestation || got.Grade != RuntimeAssertion {
		t.Fatalf("unknown values must remain informational and grade no higher than runtime: %+v", got)
	}
}

func TestCapsuleEffectModeRelationship(t *testing.T) {
	cases := []struct {
		name string
		mode EffectMode
		attestation string
		response bool
		valid bool
	}{
		{"planned-no-grade", NotApplicable, "", false, true},
		{"phantom-grade", NotApplicable, string(GateExecuted), true, false},
		{"dispatch-needs-grade", DispatchedUnconfirmed, "", false, false},
		{"dispatch-with-grade", DispatchedUnconfirmed, string(RuntimeClaimed), false, true},
		{"confirmed-needs-response", Confirmed, string(GateExecuted), false, false},
		{"confirmed-with-response", Confirmed, string(GateExecuted), true, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := AssessCapsuleEffect(tc.mode, tc.attestation, tc.response)
			if got.Valid != tc.valid {
				t.Fatalf("got %+v, valid=%v want %v", got, got.Valid, tc.valid)
			}
		})
	}
}

func stateRef(value string) DigestRef {
	x := ref(value)
	x.Context = DigestContext{
		Profile: "urn:example:state-profile",
		Version: "1",
		PayloadType: "application/json",
		Canonicalization: "postgres-jsonb-text-v1",
		Encoding: "UTF-8",
		Purpose: "state-after",
		Subject: "destination-a/resource-a/revision-7",
	}
	return x
}

func validInput() DigitalEffectInput {
	return DigitalEffectInput{
		Producer: ProducerEvidence{
			Verification: Verified,
			DestinationID: "destination-a",
			OperationID: "operation-1",
			PrincipalID: "principal-a",
			TenantID: "tenant-a",
			ResourceID: "resource-a",
			RequestDigest: ref(sha('a')),
			Assessment: AssessCapsuleEffect(Confirmed, string(RuntimeClaimed), true),
			ClaimsSuccess: true,
		},
		Sink: SinkEvidence{
			Verification: Verified,
			Outcome: SinkCommitted,
			DestinationID: "destination-a",
			OperationID: "operation-1",
			PrincipalID: "principal-a",
			TenantID: "tenant-a",
			ResourceID: "resource-a",
			RequestDigest: ref(sha('a')),
			ResultRevision: 7,
			StateDigest: stateRef(sha('b')),
		},
		Observer: ObserverEvidence{
			Verification: Verified,
			DestinationID: "destination-a",
			OperationID: "operation-1",
			PrincipalID: "observer-principal",
			TenantID: "tenant-a",
			ResourceID: "resource-a",
			Revision: 7,
			StateDigest: stateRef(sha('b')),
			Independence: IndependenceVerified,
		},
		AuthorityAtCommit: AuthorityValid,
	}
}

func TestQualifiedDigitalEffectNeedsExactIndependentReadbackAndCommitAuthority(t *testing.T) {
	got := QualifyDigitalEffect(validInput())
	if got.Result != QualifiedDigitalEffect ||
		got.RequestBinding != DigestEqual || got.StateBinding != DigestEqual {
		t.Fatalf("got %+v", got)
	}
}

func TestProducerConfirmedAloneDoesNotEstablishSinkEffect(t *testing.T) {
	in := validInput()
	in.Sink.Verification = Missing
	got := QualifyDigitalEffect(in)
	if got.Result == QualifiedDigitalEffect {
		t.Fatalf("producer confirmation must not substitute for a sink: %+v", got)
	}
}

func TestSinkCommitWithoutObserverRemainsUnqualified(t *testing.T) {
	in := validInput()
	in.Observer.Verification = Missing
	got := QualifyDigitalEffect(in)
	if got.Result != EffectObservedReadbackMissing {
		t.Fatalf("got %+v", got)
	}
}

func TestMismatchedProducerSinkRequestIsContradicted(t *testing.T) {
	in := validInput()
	in.Sink.RequestDigest = ref(sha('c'))
	got := QualifyDigitalEffect(in)
	if got.Result != ContradictedEffect || got.RequestBinding != DigestDifferent {
		t.Fatalf("got %+v", got)
	}
}

func TestMismatchedObserverRevisionIsContradicted(t *testing.T) {
	in := validInput()
	in.Observer.Revision++
	got := QualifyDigitalEffect(in)
	if got.Result != ContradictedEffect {
		t.Fatalf("got %+v", got)
	}
}

func TestSharedOrUnknownObserverControlRootCannotQualify(t *testing.T) {
	in := validInput()
	in.Observer.Independence = SharedControlRoot
	if got := QualifyDigitalEffect(in); got.Result != EffectObservedNotIndependent {
		t.Fatalf("shared root: %+v", got)
	}
	in.Observer.Independence = IndependenceUnknown
	if got := QualifyDigitalEffect(in); got.Result != EffectObservedNotIndependent {
		t.Fatalf("unknown root: %+v", got)
	}
}

func TestCommitTimeAuthorityIsIndependentOfObservedEffect(t *testing.T) {
	in := validInput()
	in.AuthorityAtCommit = AuthorityUncheckable
	if got := QualifyDigitalEffect(in); got.Result != EffectObservedAuthorityUnknown {
		t.Fatalf("uncheckable authority: %+v", got)
	}
	in.AuthorityAtCommit = AuthorityInvalid
	if got := QualifyDigitalEffect(in); got.Result != EffectObservedAuthorityInvalid {
		t.Fatalf("invalid authority: %+v", got)
	}
}

func TestProducerSuccessConflictsWithSinkRefusal(t *testing.T) {
	in := validInput()
	in.Sink.Outcome = SinkRefused
	got := QualifyDigitalEffect(in)
	if got.Result != ContradictedEffect {
		t.Fatalf("got %+v", got)
	}
}

func TestCrossPartyRungDoesNotSubstituteForObserverIndependence(t *testing.T) {
	// cross_party_rung is intentionally not an input to this function: a
	// bilateral reference may still be controlled by one shared administrator.
	in := validInput()
	in.Observer.Independence = IndependenceUnknown
	if got := QualifyDigitalEffect(in); got.Result == QualifiedDigitalEffect {
		t.Fatalf("bilateral evidence must not imply independent control domains")
	}
}
