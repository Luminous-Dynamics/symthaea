package scopeguard

import (
	"testing"
	"time"
)

var testNow = time.Date(2026, 11, 26, 15, 0, 0, 0, time.UTC)

func digest(ch byte) string {
	b := make([]byte, 64)
	for i := range b {
		b[i] = ch
	}
	return string(b)
}

func validSession() PrincipalAssertion {
	return PrincipalAssertion{
		PrincipalID: "principal-a", DatabaseSessionPrincipal: "principal-a",
		IdentityIssuer: "idp-a", IdentityTrustDomain: "identity-root-a",
		AuthContextDigest: digest('a'), AuthenticatedAt: testNow.Add(-time.Minute),
		ExpiresAt: testNow.Add(time.Hour), Verified: true,
	}
}

func validGrant() ScopeGrant {
	return ScopeGrant{
		GrantID: "grant-1", PrincipalID: "principal-a", TenantID: "tenant-a",
		ResourceID: "resource-a", Action: "mutate", PolicyDigest: digest('b'),
		AuthorityDigest: digest('c'), Issuer: "authority-a", IssuerTrustDomain: "authority-root-a",
		EvidenceDigest: digest('d'), VerifiedAt: testNow.Add(-time.Minute),
		NotBefore: testNow.Add(-time.Hour), ExpiresAt: testNow.Add(time.Hour),
		MaxAge: 10 * time.Minute, Verified: true,
	}
}

func validRequest() ScopeRequest {
	return ScopeRequest{
		PrincipalID: "principal-a", TenantID: "tenant-a", ResourceID: "resource-a",
		Action: "mutate", PolicyDigest: digest('b'), AuthorityDigest: digest('c'),
		RequestDigest: digest('e'), DestinationID: "payments-db-a", OperationID: "op-1",
	}
}

func TestAuthorizeMutationExactScope(t *testing.T) {
	got := AuthorizeMutation(validSession(), validGrant(), validRequest(), testNow)
	if got.Verdict != Allowed {
		t.Fatalf("got %+v, want allowed", got)
	}
}

func TestAuthorizeMutationRejectsIdentityAndScopeSubstitution(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*PrincipalAssertion, *ScopeGrant, *ScopeRequest)
	}{
		{"database-session-principal", func(s *PrincipalAssertion, _ *ScopeGrant, _ *ScopeRequest) { s.DatabaseSessionPrincipal = "principal-b" }},
		{"request-principal", func(_ *PrincipalAssertion, _ *ScopeGrant, r *ScopeRequest) { r.PrincipalID = "principal-b" }},
		{"tenant", func(_ *PrincipalAssertion, g *ScopeGrant, _ *ScopeRequest) { g.TenantID = "tenant-b" }},
		{"resource", func(_ *PrincipalAssertion, g *ScopeGrant, _ *ScopeRequest) { g.ResourceID = "resource-b" }},
		{"action", func(_ *PrincipalAssertion, g *ScopeGrant, _ *ScopeRequest) { g.Action = "delete" }},
		{"policy-digest", func(_ *PrincipalAssertion, _ *ScopeGrant, r *ScopeRequest) { r.PolicyDigest = digest('f') }},
		{"authority-digest", func(_ *PrincipalAssertion, _ *ScopeGrant, r *ScopeRequest) { r.AuthorityDigest = digest('f') }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, g, r := validSession(), validGrant(), validRequest()
			tc.mutate(&s, &g, &r)
			if got := AuthorizeMutation(s, g, r, testNow); got.Verdict != Unauthorized {
				t.Fatalf("got %+v, want unauthorized", got)
			}
		})
	}
}

func TestAuthorizeMutationTemporalAndRevocationBoundaries(t *testing.T) {
	cases := []struct {
		name   string
		change func(*PrincipalAssertion, *ScopeGrant)
		want   Verdict
	}{
		{"session-expired", func(s *PrincipalAssertion, _ *ScopeGrant) { s.ExpiresAt = testNow }, Unauthorized},
		{"grant-expired", func(_ *PrincipalAssertion, g *ScopeGrant) { g.ExpiresAt = testNow }, Unauthorized},
		{"grant-not-yet-valid", func(_ *PrincipalAssertion, g *ScopeGrant) { g.NotBefore = testNow.Add(time.Second) }, Unauthorized},
		{"grant-stale", func(_ *PrincipalAssertion, g *ScopeGrant) { g.VerifiedAt = testNow.Add(-time.Hour) }, Unauthorized},
		{"revoked-before-commit", func(_ *PrincipalAssertion, g *ScopeGrant) { x := testNow.Add(-time.Second); g.RevokedAt = &x }, Unauthorized},
		{"revoked-after-commit", func(_ *PrincipalAssertion, g *ScopeGrant) { x := testNow.Add(time.Second); g.RevokedAt = &x }, Allowed},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, g := validSession(), validGrant()
			tc.change(&s, &g)
			if got := AuthorizeMutation(s, g, validRequest(), testNow); got.Verdict != tc.want {
				t.Fatalf("got %+v, want %s", got, tc.want)
			}
		})
	}
}

func TestAuthorizeMutationMissingOrUnverifiedEvidenceIsIndeterminate(t *testing.T) {
	cases := []struct {
		name   string
		change func(*PrincipalAssertion, *ScopeGrant, *ScopeRequest)
	}{
		{"unverified-session", func(s *PrincipalAssertion, _ *ScopeGrant, _ *ScopeRequest) { s.Verified = false }},
		{"unverified-grant", func(_ *PrincipalAssertion, g *ScopeGrant, _ *ScopeRequest) { g.Verified = false }},
		{"missing-auth-context", func(s *PrincipalAssertion, _ *ScopeGrant, _ *ScopeRequest) { s.AuthContextDigest = "" }},
		{"malformed-request-digest", func(_ *PrincipalAssertion, _ *ScopeGrant, r *ScopeRequest) { r.RequestDigest = "xyz" }},
		{"missing-operation-id", func(_ *PrincipalAssertion, _ *ScopeGrant, r *ScopeRequest) { r.OperationID = "" }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, g, r := validSession(), validGrant(), validRequest()
			tc.change(&s, &g, &r)
			if got := AuthorizeMutation(s, g, r, testNow); got.Verdict != Indeterminate {
				t.Fatalf("got %+v, want indeterminate", got)
			}
		})
	}
}

func validOperation() OperationIdentity {
	return OperationIdentity{
		DestinationID: "destination-a", OperationID: "op-1", PrincipalID: "principal-a",
		TenantID: "tenant-a", ResourceID: "resource-a", Action: "mutate",
		RequestDigest: digest('a'), PolicyDigest: digest('b'), AuthorityDigest: digest('c'),
		AuthorityRevision: 4, FenceToken: 9, ExpectedPriorRevision: 12,
	}
}

func TestCompareOperationIdentity(t *testing.T) {
	base := validOperation()
	if got := CompareOperationIdentity(nil, base); got != NewOperation {
		t.Fatalf("new => %s", got)
	}
	if got := CompareOperationIdentity(&base, base); got != ReplayOriginal {
		t.Fatalf("exact retry => %s", got)
	}
	cases := []struct {
		name   string
		change func(*OperationIdentity)
	}{
		{"cross-tenant-replay", func(x *OperationIdentity) { x.TenantID = "tenant-b" }},
		{"cross-principal-replay", func(x *OperationIdentity) { x.PrincipalID = "principal-b" }},
		{"request-bytes", func(x *OperationIdentity) { x.RequestDigest = digest('d') }},
		{"resource", func(x *OperationIdentity) { x.ResourceID = "resource-b" }},
		{"policy", func(x *OperationIdentity) { x.PolicyDigest = digest('d') }},
		{"authority", func(x *OperationIdentity) { x.AuthorityDigest = digest('d') }},
		{"fence", func(x *OperationIdentity) { x.FenceToken++ }},
		{"expected-revision", func(x *OperationIdentity) { x.ExpectedPriorRevision++ }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x := base
			tc.change(&x)
			if got := CompareOperationIdentity(&base, x); got != IdentityConflict {
				t.Fatalf("got %s, want conflict", got)
			}
		})
	}
	otherDestination := base
	otherDestination.DestinationID = "destination-b"
	if got := CompareOperationIdentity(&base, otherDestination); got != NewOperation {
		t.Fatalf("different destination => %s", got)
	}
}

func TestCompareOperationIdentityIncompleteInputIsIndeterminate(t *testing.T) {
	base := validOperation()
	base.RequestDigest = "bad"
	if got := CompareOperationIdentity(&base, validOperation()); got != ReplayUnknown {
		t.Fatalf("got %s", got)
	}
}

func party(id, label string, suffix byte) PartyControlEvidence {
	roots := map[string]string{}
	for i, domain := range RequiredControlDomains() {
		roots[domain] = string([]byte{suffix}) + domain + string(rune('a'+i))
	}
	return PartyControlEvidence{
		PartyID: id, AdvertisedTrustDomain: label, ControlRoots: roots,
		EvidenceDigest: digest(suffix), Verified: true,
	}
}

func TestAssessIndependenceUsesControlRootsNotLabels(t *testing.T) {
	a, b := party("sink", "different-label-a", 'a'), party("observer", "different-label-b", 'b')
	if got := AssessIndependence(a, b, RequiredControlDomains()); got != Independent {
		t.Fatalf("disjoint roots => %s", got)
	}
	b.ControlRoots["operator_administration"] = a.ControlRoots["operator_administration"]
	if got := AssessIndependence(a, b, RequiredControlDomains()); got != SharedControl {
		t.Fatalf("shared operator => %s", got)
	}
}

func TestAssessIndependenceUnknownWhenControlEvidenceIsIncomplete(t *testing.T) {
	a, b := party("sink", "same-label", 'a'), party("observer", "same-label", 'b')
	delete(b.ControlRoots, "key_custody")
	if got := AssessIndependence(a, b, RequiredControlDomains()); got != UnknownControls {
		t.Fatalf("missing key custody => %s", got)
	}
	b = party("observer", "different-label", 'b')
	b.Verified = false
	if got := AssessIndependence(a, b, RequiredControlDomains()); got != UnknownControls {
		t.Fatalf("unverified evidence => %s", got)
	}
	if got := AssessIndependence(a, b, nil); got != UnknownControls {
		t.Fatalf("no dimensions => %s", got)
	}
}

func TestAssessIndependenceRejectsUnknownControlDomain(t *testing.T) {
	a, b := party("sink", "a", 'a'), party("observer", "b", 'b')
	if got := AssessIndependence(a, b, []string{"invented_control_plane"}); got != UnknownControls {
		t.Fatalf("unknown dimension => %s", got)
	}
}
