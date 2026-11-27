package authorityjoin

import "testing"

func ip(v int64) *int64 { return &v }
func dg(s, v, p string) DigestRef {
	return DigestRef{"sha-256", "test-profile", "1", "application/json", "jcs", "hex", p, s, v}
}
func pins() VerifierPolicy {
	return VerifierPolicy{"v1": {ID: "v1", Version: "1.0", BuildDigest: "build1", TrustStoreDigest: "roots1"}}
}
func fixture() (Bundle, Operation) {
	t := int64(500)
	req := dg("op-1", "aa", "request")
	pi := dg("alice", "pi", "identity")
	ai := dg("agent", "ai", "identity")
	ii := dg("gov", "ii", "identity")
	gd := dg("g1", "gg", "grant")
	g := Grant{ID: "g1", Principal: "alice", Issuer: "gov", Grantee: "agent", Destination: "db", Tenant: "t1", Action: "write", Resource: "r1", Mode: "execute", Epoch: 4, ValidFrom: 100, ExpiresAt: ip(1000), RequestDigest: req, RecordDigest: gd, IssuerDigest: ii, SourceID: "mycelix-dna", SourceProfile: "timeline-v1"}
	cov := Coverage{true, "mycelix-dna", "timeline-v1", "snap1", "query1", "0", "4", "4", 510}
	mk := func(k, id string, d DigestRef) VerifierReport {
		return VerifierReport{k, id, Verified, d, "v1", "1.0", "build1", "roots1", 400, 900}
	}
	b := Bundle{[]Grant{g}, []VerifierReport{mk("identity", "alice", pi), mk("identity", "agent", ai), mk("issuer", "gov", ii), mk("grant", "g1", gd)}, cov}
	op := Operation{ID: "op-1", Principal: "alice", Delegate: "agent", Destination: "db", Tenant: "t1", Action: "write", Resource: "r1", Mode: "execute", Epoch: 4, RequestDigest: req, PrincipalDigest: pi, DelegateDigest: ai, CommitAt: &t}
	return b, op
}
func want(t *testing.T, b Bundle, o Operation, d Decision) {
	t.Helper()
	r := Evaluate(b, o, pins())
	if r.Decision != d {
		t.Fatalf("got %s (%s), want %s", r.Decision, r.Reason, d)
	}
}
func TestDirectGrant(t *testing.T) { b, o := fixture(); want(t, b, o, Qualified) }
func TestProposalIsNotExecution(t *testing.T) {
	b, o := fixture(); b.Grants[0].Mode = "propose"; want(t, b, o, Denied)
}
func TestPrincipalSubstitution(t *testing.T) {
	b, o := fixture(); o.Principal = "mallory"; want(t, b, o, Denied)
}
func TestDelegateSubstitution(t *testing.T) {
	b, o := fixture(); o.Delegate = "other"; want(t, b, o, Denied)
}
func TestRequestSubstitution(t *testing.T) {
	b, o := fixture(); o.RequestDigest.Value = "bb"; want(t, b, o, Denied)
}
func TestDigestProfileMismatch(t *testing.T) {
	b, o := fixture(); o.RequestDigest.Profile = "other"; want(t, b, o, Indeterminate)
}
func TestNotYetEffective(t *testing.T) { b, o := fixture(); *o.CommitAt = 99; want(t, b, o, Denied) }
func TestExclusiveExpiry(t *testing.T) {
	b, o := fixture(); *o.CommitAt = 1000; b.Coverage.ObservedAt = 1010; want(t, b, o, Denied)
}
func attachRevocationEvidence(b *Bundle) {
	id := "revocation-1"
	d := dg(id, "rev", "revocation-event")
	b.Grants[0].RevocationID = id
	b.Grants[0].RevocationDigest = &d
	b.Reports = append(b.Reports, VerifierReport{"revocation", id, Verified, d, "v1", "1.0", "build1", "roots1", 400, 900})
}
func TestRevokedAtCommit(t *testing.T) {
	b, o := fixture(); b.Grants[0].RevokedAt = ip(500); attachRevocationEvidence(&b); want(t, b, o, Denied)
}
func TestRevocationWithoutEvidenceIndeterminate(t *testing.T) {
	b, o := fixture(); b.Grants[0].RevokedAt = ip(700); want(t, b, o, Indeterminate)
}
func TestLaterRevocationKeepsEarlierHistory(t *testing.T) {
	b, o := fixture(); b.Grants[0].RevokedAt = ip(700); attachRevocationEvidence(&b); want(t, b, o, Qualified)
}
func TestUnknownCommitTime(t *testing.T) { b, o := fixture(); o.CommitAt = nil; want(t, b, o, Indeterminate) }
func TestIncompleteCapture(t *testing.T) { b, o := fixture(); b.Coverage.Complete = false; want(t, b, o, Indeterminate) }
func TestMissingTerminalWatermark(t *testing.T) {
	b, o := fixture(); b.Coverage.TerminalWatermark = ""; want(t, b, o, Indeterminate)
}
func TestCaptureBeforeCommit(t *testing.T) {
	b, o := fixture(); b.Coverage.ObservedAt = 499; want(t, b, o, Indeterminate)
}
func TestMissingVerifier(t *testing.T) { b, o := fixture(); b.Reports = b.Reports[:3]; want(t, b, o, Indeterminate) }
func TestInvalidSignature(t *testing.T) { b, o := fixture(); b.Reports[3].Status = Invalid; want(t, b, o, Denied) }
func TestUnverifiedSignature(t *testing.T) { b, o := fixture(); b.Reports[3].Status = Unverified; want(t, b, o, Indeterminate) }
func TestUnpinnedVerifier(t *testing.T) {
	b, o := fixture(); r := Evaluate(b, o, VerifierPolicy{}); if r.Decision != Indeterminate { t.Fatalf("got %s", r.Decision) }
}
func TestWrongVerifierBuildIsIndeterminate(t *testing.T) {
	b, o := fixture(); b.Reports[3].BuildDigest = "changed"; want(t, b, o, Indeterminate)
}
func TestVerifierDigestBinding(t *testing.T) {
	b, o := fixture(); b.Reports[3].InputDigest.Value = "other"; want(t, b, o, Denied)
}
func TestStaleVerifier(t *testing.T) { b, o := fixture(); b.Reports[3].FreshUntil = 510; want(t, b, o, Indeterminate) }
func TestDuplicateConflictingReports(t *testing.T) {
	b, o := fixture(); r := b.Reports[3]; r.InputDigest.Value = "fork"; b.Reports = append(b.Reports, r); want(t, b, o, Divergent)
}
func TestDelegatedLineage(t *testing.T) {
	b, o := fixture()
	p := b.Grants[0]; p.ID = "parent"; p.Grantee = "agent2"; p.RecordDigest = dg("parent", "p", "grant")
	cd := dg("child", "c", "grant"); id := dg("agent2", "i", "identity")
	c := p; c.ID = "child"; c.ParentID = "parent"; c.Principal = "agent2"; c.Issuer = "agent2"; c.Grantee = "agent"; c.ValidFrom = 200; c.ExpiresAt = ip(900); c.RecordDigest = cd; c.IssuerDigest = id
	b.Grants = []Grant{p, c}
	mk := func(k, s string, d DigestRef) VerifierReport { return VerifierReport{k, s, Verified, d, "v1", "1.0", "build1", "roots1", 400, 900} }
	b.Reports = append(b.Reports, mk("grant", "parent", p.RecordDigest), mk("grant", "child", cd), mk("identity", "agent2", id), mk("issuer", "agent2", id))
	want(t, b, o, Qualified)
}
func TestDelegationAmplification(t *testing.T) {
	b, o := fixture(); p := b.Grants[0]; p.ID = "parent"; p.Grantee = "agent2"
	c := p; c.ID = "child"; c.ParentID = "parent"; c.Principal = "agent2"; c.Issuer = "agent2"; c.Grantee = "agent"; c.Action = "delete"
	b.Grants = []Grant{p, c}; want(t, b, o, Denied)
}
func TestDigestContextsAreIncomparable(t *testing.T) {
	a := dg("s", "same", "request"); b := a; b.Purpose = "credential"
	eq, cmp := a.Equal(b); if eq || cmp { t.Fatalf("equal=%v comparable=%v", eq, cmp) }
}
func FuzzDigestPurposeIsolation(f *testing.F) {
	f.Add("request", "credential", "aa")
	f.Fuzz(func(t *testing.T, a, b, v string) {
		if a == b { return }
		x := dg("s", v, a); y := dg("s", v, b); eq, cmp := x.Equal(y)
		if eq || cmp { t.Fatal("different purpose compared") }
	})
}
