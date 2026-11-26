package authorityjoin

import (
	"fmt"
	"strings"
)

type Decision string

const (
	Qualified     Decision = "qualified"
	Denied        Decision = "denied"
	Indeterminate Decision = "indeterminate"
	Divergent     Decision = "divergent"
)

type VerificationStatus string

const (
	Verified   VerificationStatus = "verified"
	Invalid    VerificationStatus = "invalid"
	Unverified VerificationStatus = "unverified"
)

type DigestRef struct {
	Algorithm        string `json:"algorithm"`
	Profile          string `json:"profile"`
	Version          string `json:"version"`
	PayloadType      string `json:"payload_type"`
	Canonicalization string `json:"canonicalization"`
	Encoding         string `json:"encoding"`
	Purpose          string `json:"purpose"`
	Subject          string `json:"subject"`
	Value            string `json:"value"`
}

func (d DigestRef) Valid() bool {
	return d.Algorithm != "" && d.Profile != "" && d.Version != "" && d.PayloadType != "" && d.Canonicalization != "" && d.Encoding != "" && d.Purpose != "" && d.Subject != "" && d.Value != ""
}

func (d DigestRef) Equal(other DigestRef) (bool, bool) {
	if d.Algorithm != other.Algorithm || d.Profile != other.Profile || d.Version != other.Version || d.PayloadType != other.PayloadType || d.Canonicalization != other.Canonicalization || d.Encoding != other.Encoding || d.Purpose != other.Purpose || d.Subject != other.Subject {
		return false, false
	}
	return d.Value == other.Value, true
}

type VerifierPin struct{ ID, Version, BuildDigest, TrustStoreDigest string }
type VerifierPolicy map[string]VerifierPin

type VerifierReport struct {
	Kind             string             `json:"kind"`
	SubjectID        string             `json:"subject_id"`
	Status           VerificationStatus `json:"status"`
	InputDigest      DigestRef          `json:"input_digest"`
	VerifierID       string             `json:"verifier_id"`
	Version          string             `json:"version"`
	BuildDigest      string             `json:"build_digest"`
	TrustStoreDigest string             `json:"trust_store_digest"`
	VerifiedAt       int64              `json:"verified_at_micros"`
	FreshUntil       int64              `json:"fresh_until_micros"`
}

type Coverage struct {
	Complete          bool   `json:"complete"`
	SourceID          string `json:"source_id"`
	Profile           string `json:"profile"`
	SnapshotID        string `json:"snapshot_id"`
	QueryID           string `json:"query_id"`
	StartCursor       string `json:"start_cursor"`
	EndCursor         string `json:"end_cursor"`
	TerminalWatermark string `json:"terminal_watermark"`
	ObservedAt        int64  `json:"observed_at_micros"`
}

type Grant struct {
	ID            string    `json:"id"`
	ParentID      string    `json:"parent_id,omitempty"`
	Principal     string    `json:"principal"`
	Issuer        string    `json:"issuer"`
	Grantee       string    `json:"grantee"`
	Destination   string    `json:"destination"`
	Tenant        string    `json:"tenant"`
	Action        string    `json:"action"`
	Resource      string    `json:"resource"`
	Mode          string    `json:"mode"`
	Epoch         uint64    `json:"authority_epoch"`
	ValidFrom     int64     `json:"effective_at_micros"`
	ExpiresAt     *int64    `json:"expires_at_micros,omitempty"`
	RevokedAt     *int64    `json:"revoked_at_micros,omitempty"`
	RequestDigest DigestRef `json:"request_digest"`
	RecordDigest  DigestRef `json:"record_digest"`
	IssuerDigest  DigestRef `json:"issuer_identity_digest"`
	SourceID      string    `json:"source_id"`
	SourceProfile string    `json:"source_profile"`
}

type Operation struct {
	ID              string    `json:"operation_id"`
	Principal       string    `json:"principal"`
	Delegate        string    `json:"delegate"`
	Destination     string    `json:"destination"`
	Tenant          string    `json:"tenant"`
	Action          string    `json:"action"`
	Resource        string    `json:"resource"`
	Mode            string    `json:"mode"`
	Epoch           uint64    `json:"authority_epoch"`
	RequestDigest   DigestRef `json:"request_digest"`
	PrincipalDigest DigestRef `json:"principal_identity_digest"`
	DelegateDigest  DigestRef `json:"delegate_identity_digest"`
	CommitAt        *int64    `json:"commit_time_micros"`
}

type Bundle struct {
	Grants   []Grant          `json:"grants"`
	Reports  []VerifierReport `json:"reports"`
	Coverage Coverage         `json:"coverage"`
}

type Result struct {
	Decision Decision `json:"decision"`
	Reason   string   `json:"reason"`
	Checked  []string `json:"checked"`
}

func out(d Decision, why string, checked ...string) Result {
	return Result{d, why, append([]string(nil), checked...)}
}

func nonempty(values ...string) bool {
	for _, v := range values {
		if strings.TrimSpace(v) == "" {
			return false
		}
	}
	return true
}

func reportKey(kind, id string) string { return kind + "\x00" + id }

type required struct {
	kind, id string
	digest   DigestRef
}

func requiredReports(b Bundle, op Operation) ([]required, error) {
	if !op.PrincipalDigest.Valid() || !op.DelegateDigest.Valid() {
		return nil, fmt.Errorf("principal/delegate identity digest missing")
	}
	items := []required{}
	identity := map[string]DigestRef{}
	for _, x := range []required{{"identity", op.Principal, op.PrincipalDigest}, {"identity", op.Delegate, op.DelegateDigest}} {
		if prior, ok := identity[x.id]; ok {
			eq, cmp := prior.Equal(x.digest)
			if !eq || !cmp {
				return nil, fmt.Errorf("conflicting identity evidence for %s", x.id)
			}
		} else {
			identity[x.id] = x.digest
			items = append(items, x)
		}
	}
	issuers := map[string]DigestRef{}
	for _, g := range b.Grants {
		if !g.RecordDigest.Valid() || !g.IssuerDigest.Valid() || !nonempty(g.SourceID, g.SourceProfile) {
			return nil, fmt.Errorf("grant evidence incomplete: %s", g.ID)
		}
		items = append(items, required{"grant", g.ID, g.RecordDigest})
		if prior, ok := issuers[g.Issuer]; ok {
			eq, cmp := prior.Equal(g.IssuerDigest)
			if !eq || !cmp {
				return nil, fmt.Errorf("conflicting issuer evidence for %s", g.Issuer)
			}
		} else {
			issuers[g.Issuer] = g.IssuerDigest
			items = append(items, required{"issuer", g.Issuer, g.IssuerDigest})
		}
	}
	return items, nil
}

func verifyReports(b Bundle, op Operation, policy VerifierPolicy) (Decision, string) {
	wants, err := requiredReports(b, op)
	if err != nil {
		return Indeterminate, err.Error()
	}
	bySubject := map[string][]VerifierReport{}
	for _, r := range b.Reports {
		bySubject[reportKey(r.Kind, r.SubjectID)] = append(bySubject[reportKey(r.Kind, r.SubjectID)], r)
	}
	for _, w := range wants {
		rs := bySubject[reportKey(w.kind, w.id)]
		if len(rs) == 0 {
			return Indeterminate, "missing verifier report for " + w.kind + ":" + w.id
		}
		for _, r := range rs[1:] {
			if r != rs[0] {
				return Divergent, "conflicting verifier reports for " + w.kind + ":" + w.id
			}
		}
		r := rs[0]
		if r.Status == Invalid {
			return Denied, "cryptographic verifier rejected " + w.kind + ":" + w.id
		}
		if r.Status != Verified {
			return Indeterminate, "cryptographic verification unestablished for " + w.kind + ":" + w.id
		}
		p, ok := policy[r.VerifierID]
		if !ok || p.ID != r.VerifierID || p.Version != r.Version || p.BuildDigest != r.BuildDigest || p.TrustStoreDigest != r.TrustStoreDigest || !nonempty(r.VerifierID, r.Version, r.BuildDigest, r.TrustStoreDigest) {
			return Indeterminate, "verifier identity/version/build/trust-store is not pinned"
		}
		if r.VerifiedAt <= 0 || r.VerifiedAt > b.Coverage.ObservedAt || r.FreshUntil <= b.Coverage.ObservedAt {
			return Indeterminate, "verifier result stale or outside observation window"
		}
		eq, cmp := r.InputDigest.Equal(w.digest)
		if !cmp {
			return Indeterminate, "verifier input digest profile incomparable"
		}
		if !eq {
			return Denied, "verifier result not bound to exact evidence bytes"
		}
	}
	return Qualified, "all required verifier reports are pinned, fresh and bound to exact evidence"
}

// Evaluate is a semantic join, not a cryptographic verifier. CommitAt and the
// capture boundary must already be authenticated by their MON source adapters.
// v0.1 supports exact-match scopes only; it does not interpret wildcards/ranges.
func Evaluate(b Bundle, op Operation, policy VerifierPolicy) Result {
	checked := []string{"operation-identity", "operation-scope", "typed-request-digest"}
	if !nonempty(op.ID, op.Principal, op.Delegate, op.Destination, op.Tenant, op.Action, op.Resource, op.Mode) || op.Epoch == 0 || !op.RequestDigest.Valid() {
		return out(Indeterminate, "operation identity/scope/digest incomplete")
	}
	if op.CommitAt == nil || *op.CommitAt <= 0 {
		return out(Indeterminate, "destination commit time not established", checked...)
	}
	c := b.Coverage
	if !c.Complete || !nonempty(c.SourceID, c.Profile, c.SnapshotID, c.QueryID, c.StartCursor, c.EndCursor, c.TerminalWatermark) || c.ObservedAt <= 0 {
		return out(Indeterminate, "authority source capture is incomplete or unbounded", checked...)
	}
	if c.ObservedAt < *op.CommitAt {
		return out(Indeterminate, "authority snapshot predates commit; no qualified receiver fence supplied", checked...)
	}
	checked = append(checked, "bounded-authority-source-capture", "capture-after-commit")
	if len(b.Grants) == 0 {
		return out(Indeterminate, "authority chain missing", checked...)
	}
	for i, g := range b.Grants {
		if !nonempty(g.ID, g.Principal, g.Issuer, g.Grantee, g.Destination, g.Tenant, g.Action, g.Resource, g.Mode, g.SourceID, g.SourceProfile) || g.Epoch == 0 || g.ValidFrom <= 0 || !g.RequestDigest.Valid() {
			return out(Indeterminate, "grant evidence incomplete: "+g.ID, checked...)
		}
		if g.SourceID != c.SourceID || g.SourceProfile != c.Profile {
			return out(Indeterminate, "grant source identity/profile does not match capture", checked...)
		}
		if i == 0 {
			if g.ParentID != "" || g.Principal != op.Principal {
				return out(Denied, "invalid root grant binding", checked...)
			}
		} else {
			p := b.Grants[i-1]
			if g.ParentID != p.ID || g.Principal != p.Grantee {
				return out(Denied, "delegation lineage discontinuity", checked...)
			}
			if g.Epoch != p.Epoch || g.Action != p.Action || g.Resource != p.Resource || g.Destination != p.Destination || g.Tenant != p.Tenant || g.Mode != p.Mode {
				return out(Denied, "delegation scope/epoch amplification is not permitted in v0.1", checked...)
			}
			if g.ValidFrom < p.ValidFrom || (p.ExpiresAt != nil && (g.ExpiresAt == nil || *g.ExpiresAt > *p.ExpiresAt)) {
				return out(Denied, "child grant exceeds parent temporal scope", checked...)
			}
		}
		if g.Destination != op.Destination || g.Tenant != op.Tenant || g.Action != op.Action || g.Resource != op.Resource || g.Mode != op.Mode || g.Epoch != op.Epoch {
			return out(Denied, "grant does not bind exact operation scope/epoch", checked...)
		}
		if *op.CommitAt < g.ValidFrom || (g.ExpiresAt != nil && *op.CommitAt >= *g.ExpiresAt) || (g.RevokedAt != nil && *op.CommitAt >= *g.RevokedAt) {
			return out(Denied, "grant inactive at destination commit: "+g.ID, checked...)
		}
		eq, cmp := g.RequestDigest.Equal(op.RequestDigest)
		if !cmp {
			return out(Indeterminate, "request digest profiles incomparable", checked...)
		}
		if !eq {
			return out(Denied, "grant does not bind exact request digest", checked...)
		}
	}
	if b.Grants[len(b.Grants)-1].Grantee != op.Delegate {
		return out(Denied, "leaf grant does not identify executing delegate", checked...)
	}
	checked = append(checked, "delegation-lineage", "exact-scope-attenuation", "commit-time-authority")
	d, why := verifyReports(b, op, policy)
	if d != Qualified {
		return out(d, why, checked...)
	}
	checked = append(checked, "identity-verification", "issuer-verification", "grant-signature-verification", "pinned-verifier-policy", "verifier-input-binding", "verifier-freshness")
	return out(Qualified, "authority evidence binds the exact operation at the supplied commit time; continue only to the separate effect gate", checked...)
}
