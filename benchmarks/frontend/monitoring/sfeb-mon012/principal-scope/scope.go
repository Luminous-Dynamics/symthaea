// Package scopeguard checks principal-, resource-, and trust-domain bindings
// after cryptographic/authentication adapters have verified their evidence.
// It is a semantic reference preflight, not an identity or signature verifier.
package scopeguard

import (
	"encoding/hex"
	"strings"
	"time"
)

type Verdict string

const (
	Allowed       Verdict = "allowed"
	Unauthorized  Verdict = "unauthorized"
	Indeterminate Verdict = "indeterminate"
)

type ScopeResult struct {
	Verdict Verdict
	Reason  string
}

// PrincipalAssertion is a normalized result from an authenticated session
// adapter. DatabaseSessionPrincipal must come from a non-spoofable database
// authentication identity (for PostgreSQL, normally session_user), not from
// a caller-controlled session variable or request field.
type PrincipalAssertion struct {
	PrincipalID              string
	DatabaseSessionPrincipal string
	IdentityIssuer           string
	IdentityTrustDomain      string
	AuthContextDigest        string
	AuthenticatedAt          time.Time
	ExpiresAt                time.Time
	Verified                 bool
}

// ScopeGrant is a normalized authorization assertion already authenticated
// by an adapter. It grants one principal one action on one tenant/resource
// under exact policy and authority digests.
type ScopeGrant struct {
	GrantID           string
	PrincipalID       string
	TenantID          string
	ResourceID        string
	Action            string
	PolicyDigest      string
	AuthorityDigest   string
	Issuer            string
	IssuerTrustDomain string
	EvidenceDigest    string
	VerifiedAt        time.Time
	NotBefore         time.Time
	ExpiresAt         time.Time
	RevokedAt         *time.Time
	MaxAge            time.Duration
	Verified          bool
}

type ScopeRequest struct {
	PrincipalID     string
	TenantID        string
	ResourceID      string
	Action          string
	PolicyDigest    string
	AuthorityDigest string
	RequestDigest   string
	DestinationID   string
	OperationID     string
}

// AuthorizeMutation fails closed on missing/unverifiable evidence, stale
// grants, identity mismatch, scope mismatch, revocation, and digest mismatch.
// A later revocation (RevokedAt > at) does not rewrite an earlier as-of result.
func AuthorizeMutation(session PrincipalAssertion, grant ScopeGrant, req ScopeRequest, at time.Time) ScopeResult {
	if !session.Verified || !grant.Verified {
		return ScopeResult{Indeterminate, "identity_or_grant_evidence_not_verified"}
	}
	if !nonEmpty(session.PrincipalID, session.DatabaseSessionPrincipal, session.IdentityIssuer,
		session.IdentityTrustDomain, grant.GrantID, grant.PrincipalID, grant.TenantID,
		grant.ResourceID, grant.Action, grant.Issuer, grant.IssuerTrustDomain) {
		return ScopeResult{Indeterminate, "required_identity_or_grant_field_missing"}
	}
	if !validDigest(session.AuthContextDigest) || !validDigest(grant.PolicyDigest) ||
		!validDigest(grant.AuthorityDigest) || !validDigest(grant.EvidenceDigest) ||
		!validDigest(req.PolicyDigest) || !validDigest(req.AuthorityDigest) ||
		!validDigest(req.RequestDigest) {
		return ScopeResult{Indeterminate, "digest_missing_or_malformed"}
	}
	if session.PrincipalID != session.DatabaseSessionPrincipal || session.PrincipalID != req.PrincipalID {
		return ScopeResult{Unauthorized, "authenticated_principal_mismatch"}
	}
	if !nonEmpty(req.TenantID, req.ResourceID, req.Action, req.DestinationID, req.OperationID) {
		return ScopeResult{Indeterminate, "request_scope_incomplete"}
	}
	if session.AuthenticatedAt.IsZero() || session.ExpiresAt.IsZero() ||
		at.Before(session.AuthenticatedAt) || !at.Before(session.ExpiresAt) {
		return ScopeResult{Unauthorized, "session_not_valid_at_mutation_time"}
	}
	if grant.VerifiedAt.IsZero() || grant.NotBefore.IsZero() || grant.ExpiresAt.IsZero() ||
		grant.MaxAge <= 0 || at.Before(grant.VerifiedAt) || at.Before(grant.NotBefore) ||
		!at.Before(grant.ExpiresAt) || at.Sub(grant.VerifiedAt) > grant.MaxAge {
		return ScopeResult{Unauthorized, "grant_not_current_at_mutation_time"}
	}
	if grant.RevokedAt != nil && !grant.RevokedAt.After(at) {
		return ScopeResult{Unauthorized, "grant_revoked_at_mutation_time"}
	}
	if grant.PrincipalID != session.PrincipalID || grant.TenantID != req.TenantID ||
		grant.ResourceID != req.ResourceID || grant.Action != req.Action {
		return ScopeResult{Unauthorized, "principal_tenant_resource_or_action_scope_mismatch"}
	}
	if grant.PolicyDigest != req.PolicyDigest || grant.AuthorityDigest != req.AuthorityDigest {
		return ScopeResult{Unauthorized, "policy_or_authority_digest_mismatch"}
	}
	return ScopeResult{Allowed, "exact_verified_scope_and_freshness_match"}
}

type ReplayVerdict string

const (
	NewOperation     ReplayVerdict = "new_operation"
	ReplayOriginal   ReplayVerdict = "replay_original"
	IdentityConflict ReplayVerdict = "identity_conflict"
	ReplayUnknown    ReplayVerdict = "indeterminate"
)

// OperationIdentity defines a destination-global idempotency key:
// (DestinationID, OperationID). Tenant, principal, resource, request, policy,
// authority and concurrency bindings are immutable values under that key.
type OperationIdentity struct {
	DestinationID         string
	OperationID           string
	PrincipalID           string
	TenantID               string
	ResourceID             string
	Action                 string
	RequestDigest          string
	PolicyDigest           string
	AuthorityDigest        string
	AuthorityRevision      int64
	FenceToken             int64
	ExpectedPriorRevision  int64
}

// CompareOperationIdentity compares a proposed retry against the immutable
// identity already recorded for this destination. An identical retry returns
// the original outcome; any changed binding conflicts rather than mutating.
func CompareOperationIdentity(prior *OperationIdentity, current OperationIdentity) ReplayVerdict {
	if !nonEmpty(current.DestinationID, current.OperationID, current.PrincipalID,
		current.TenantID, current.ResourceID, current.Action) ||
		!validDigest(current.RequestDigest) || !validDigest(current.PolicyDigest) ||
		!validDigest(current.AuthorityDigest) || current.AuthorityRevision < 0 ||
		current.FenceToken < 0 || current.ExpectedPriorRevision < 0 {
		return ReplayUnknown
	}
	if prior == nil {
		return NewOperation
	}
	if !nonEmpty(prior.DestinationID, prior.OperationID, prior.PrincipalID,
		prior.TenantID, prior.ResourceID, prior.Action) ||
		!validDigest(prior.RequestDigest) || !validDigest(prior.PolicyDigest) ||
		!validDigest(prior.AuthorityDigest) || prior.AuthorityRevision < 0 ||
		prior.FenceToken < 0 || prior.ExpectedPriorRevision < 0 {
		return ReplayUnknown
	}
	if prior.DestinationID != current.DestinationID || prior.OperationID != current.OperationID {
		return NewOperation
	}
	if *prior == current {
		return ReplayOriginal
	}
	return IdentityConflict
}

type IndependenceVerdict string

const (
	Independent    IndependenceVerdict = "independent_under_declared_controls"
	SharedControl  IndependenceVerdict = "shared_control_domain"
	UnknownControls IndependenceVerdict = "independence_unverifiable"
)

type PartyControlEvidence struct {
	PartyID               string
	AdvertisedTrustDomain string // descriptive only; never used to establish independence
	ControlRoots          map[string]string
	EvidenceDigest        string
	Verified              bool
}

var allowedControlDomains = map[string]struct{}{
	"identity_administration": {},
	"credential_root": {},
	"key_custody": {},
	"cloud_account": {},
	"storage_administration": {},
	"operator_administration": {},
	"network_administration": {},
	"deployment_pipeline": {},
}

// RequiredControlDomains returns the default dimensions whose administrative
// roots should be compared for an independence claim. Deployments may define
// a stricter or narrower policy, but must publish it.
func RequiredControlDomains() []string {
	return []string{
		"identity_administration", "credential_root", "key_custody",
		"cloud_account", "storage_administration", "operator_administration",
		"network_administration", "deployment_pipeline",
	}
}

// AssessIndependence compares evidenced control roots, not trust-domain labels.
// Missing, unverified or malformed control evidence remains UnknownControls.
// A shared root is a decisive failure even when other dimensions are unknown.
func AssessIndependence(left, right PartyControlEvidence, required []string) IndependenceVerdict {
	if !left.Verified || !right.Verified || left.PartyID == "" || right.PartyID == "" ||
		left.PartyID == right.PartyID || !validDigest(left.EvidenceDigest) ||
		!validDigest(right.EvidenceDigest) || len(required) == 0 {
		return UnknownControls
	}
	unknown := false
	for _, dimension := range required {
		if _, ok := allowedControlDomains[dimension]; !ok {
			unknown = true
			continue
		}
		l, lok := left.ControlRoots[dimension]
		r, rok := right.ControlRoots[dimension]
		if !lok || !rok || strings.TrimSpace(l) == "" || strings.TrimSpace(r) == "" {
			unknown = true
			continue
		}
		if l == r {
			return SharedControl
		}
	}
	if unknown {
		return UnknownControls
	}
	return Independent
}

func nonEmpty(values ...string) bool {
	for _, value := range values {
		if strings.TrimSpace(value) == "" {
			return false
		}
	}
	return true
}

func validDigest(value string) bool {
	if len(value) != 64 || strings.ToLower(value) != value {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == 32
}
