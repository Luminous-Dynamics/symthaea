package admission

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

type Verdict string

const (
	Qualified     Verdict = "qualified"
	Blocked       Verdict = "blocked"
	Indeterminate Verdict = "indeterminate"
)

type Decision struct {
	Verdict Verdict `json:"verdict"`
	Reason  string  `json:"reason"`
}

type Interval struct {
	Earliest time.Time `json:"earliest"`
	Latest   time.Time `json:"latest"`
}

// AppraiseCapture returns qualified only when the entire capture interval is
// strictly before the effective revocation time. Equality and overlap are not
// silently interpreted as safe.
func AppraiseCapture(capture Interval, effectiveRevocation *time.Time) (Decision, error) {
	if capture.Earliest.IsZero() || capture.Latest.IsZero() || capture.Latest.Before(capture.Earliest) {
		return Decision{Verdict: Indeterminate, Reason: "capture_interval_invalid"}, errors.New("capture interval is missing or reversed")
	}
	if effectiveRevocation == nil || effectiveRevocation.IsZero() {
		return Decision{Verdict: Indeterminate, Reason: "revocation_time_unknown"}, nil
	}
	r := *effectiveRevocation
	if capture.Latest.Before(r) {
		return Decision{Verdict: Qualified, Reason: "capture_certainly_precedes_revocation"}, nil
	}
	if !r.After(capture.Earliest) {
		return Decision{Verdict: Blocked, Reason: "revocation_effective_before_or_at_capture"}, nil
	}
	return Decision{Verdict: Indeterminate, Reason: "capture_interval_overlaps_revocation"}, nil
}

type DependencyExpiry struct {
	ID      string
	Known   bool
	Expires time.Time
}

// MinimumMandatoryExpiry refuses to omit any unknown mandatory dependency.
func MinimumMandatoryExpiry(dependencies []DependencyExpiry) (time.Time, error) {
	if len(dependencies) == 0 { return time.Time{}, errors.New("mandatory dependency set is empty") }
	var minimum time.Time
	seen := make(map[string]struct{}, len(dependencies))
	for _, dep := range dependencies {
		if strings.TrimSpace(dep.ID) == "" { return time.Time{}, errors.New("mandatory dependency ID is missing") }
		if _, ok := seen[dep.ID]; ok { return time.Time{}, fmt.Errorf("duplicate mandatory dependency: %s", dep.ID) }
		seen[dep.ID] = struct{}{}
		if !dep.Known || dep.Expires.IsZero() { return time.Time{}, fmt.Errorf("mandatory dependency expiry is unknown: %s", dep.ID) }
		if minimum.IsZero() || dep.Expires.Before(minimum) { minimum = dep.Expires }
	}
	return minimum, nil
}

type Claim struct {
	ID                  string
	Digest              string
	LeaseEnds           time.Time
	AppraisedTrustEpoch uint64
	Revoked             bool
}

type Permit struct {
	ClaimID      string
	ClaimDigest  string
	Target       string
	Operation    string
	TrustEpoch   uint64
	FencingToken uint64
	Nonce        string
	ExpiresAt    time.Time
}

type Store struct {
	mu           sync.Mutex
	now          func() time.Time
	trustEpoch   uint64
	targetFences map[string]uint64
	claims       map[string]Claim
	issued       map[string]Permit
	consumed     map[string]struct{}
}

func NewStore(initialTrustEpoch uint64, clock func() time.Time) *Store {
	if clock == nil { clock = time.Now }
	return &Store{
		now: clock, trustEpoch: initialTrustEpoch,
		targetFences: make(map[string]uint64), claims: make(map[string]Claim),
		issued: make(map[string]Permit), consumed: make(map[string]struct{}),
	}
}

func (s *Store) RegisterClaim(claim Claim) error {
	if strings.TrimSpace(claim.ID) == "" || strings.TrimSpace(claim.Digest) == "" || claim.LeaseEnds.IsZero() {
		return errors.New("claim identity, digest and lease end are required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.claims[claim.ID]; exists { return fmt.Errorf("claim already registered: %s", claim.ID) }
	if claim.AppraisedTrustEpoch != s.trustEpoch {
		return fmt.Errorf("claim appraisal epoch %d does not match current epoch %d", claim.AppraisedTrustEpoch, s.trustEpoch)
	}
	s.claims[claim.ID] = claim
	return nil
}

// IssuePermit binds a short-lived permit to claim, target, operation, trust
// epoch, fencing generation and nonce. A newer permit for the same target
// advances that target's fence and invalidates older permits.
func (s *Store) IssuePermit(claimID, claimDigest, target, operation, nonce string, expectedTrustEpoch uint64, ttl time.Duration) (Permit, error) {
	if strings.TrimSpace(claimID) == "" || strings.TrimSpace(claimDigest) == "" ||
		strings.TrimSpace(target) == "" || strings.TrimSpace(operation) == "" || strings.TrimSpace(nonce) == "" {
		return Permit{}, errors.New("permit binding fields must be non-empty")
	}
	if ttl <= 0 { return Permit{}, errors.New("permit TTL must be positive") }
	s.mu.Lock()
	defer s.mu.Unlock()
	if expectedTrustEpoch != s.trustEpoch { return Permit{}, errors.New("trust/revocation epoch is stale") }
	claim, ok := s.claims[claimID]
	if !ok || claim.Digest != claimDigest || claim.Revoked { return Permit{}, errors.New("claim is missing, changed or revoked") }
	if claim.AppraisedTrustEpoch != s.trustEpoch {
		return Permit{}, errors.New("claim has not been re-appraised under the current trust epoch")
	}
	now := s.now()
	if !now.Before(claim.LeaseEnds) { return Permit{}, errors.New("claim lease has expired") }
	if _, ok := s.issued[nonce]; ok { return Permit{}, errors.New("nonce is already reserved") }
	if _, ok := s.consumed[nonce]; ok { return Permit{}, errors.New("nonce has already been consumed") }
	deadline := now.Add(ttl)
	if claim.LeaseEnds.Before(deadline) { deadline = claim.LeaseEnds }
	fence := s.targetFences[target] + 1
	s.targetFences[target] = fence
	permit := Permit{
		ClaimID: claimID, ClaimDigest: claimDigest, Target: target, Operation: operation,
		TrustEpoch: s.trustEpoch, FencingToken: fence, Nonce: nonce, ExpiresAt: deadline,
	}
	s.issued[nonce] = permit
	return permit, nil
}

// AdvanceTrustEpoch invalidates all old permits and makes previous appraisals stale.
func (s *Store) AdvanceTrustEpoch(next uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if next <= s.trustEpoch { return fmt.Errorf("trust epoch must increase: current=%d next=%d", s.trustEpoch, next) }
	s.trustEpoch = next
	return nil
}

// RevokeClaim revokes the claim and increments the global epoch. Outstanding
// permits are stale; unaffected claims must be re-appraised before reuse.
func (s *Store) RevokeClaim(claimID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	claim, ok := s.claims[claimID]
	if !ok { return fmt.Errorf("claim not found: %s", claimID) }
	claim.Revoked = true
	s.claims[claimID] = claim
	s.trustEpoch++
	return nil
}

// ReappraiseClaim refreshes an unaffected claim at the current epoch. Revoked
// claims cannot be resurrected; callers must create a new versioned claim.
func (s *Store) ReappraiseClaim(claimID, newDigest string, leaseEnds time.Time) error {
	if strings.TrimSpace(newDigest) == "" || leaseEnds.IsZero() { return errors.New("reappraisal digest and lease end are required") }
	s.mu.Lock()
	defer s.mu.Unlock()
	claim, ok := s.claims[claimID]
	if !ok || claim.Revoked { return fmt.Errorf("claim is missing or revoked: %s", claimID) }
	if !s.now().Before(leaseEnds) { return errors.New("reappraisal lease is already expired") }
	claim.Digest = newDigest
	claim.LeaseEnds = leaseEnds
	claim.AppraisedTrustEpoch = s.trustEpoch
	s.claims[claimID] = claim
	return nil
}

func (s *Store) AdvanceTargetFence(target string) (uint64, error) {
	if strings.TrimSpace(target) == "" { return 0, errors.New("target is required") }
	s.mu.Lock()
	defer s.mu.Unlock()
	s.targetFences[target]++
	return s.targetFences[target], nil
}

func (s *Store) TrustEpoch() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.trustEpoch
}

// AdmitAndApply is the in-memory linearization boundary: validation, nonce
// consumption and the callback are serialized with revocation and fencing.
// The nonce is consumed before apply; an error/unknown outcome requires
// reconciliation and never makes the nonce reusable. apply MUST be bounded
// and MUST NOT re-enter this Store.
//
// A production adapter must use a durable transactional/linearizable system
// of record and authenticate permit contents under the trusted root.
func (s *Store) AdmitAndApply(permit Permit, apply func() error) error {
	if apply == nil { return errors.New("mutation callback is required") }
	s.mu.Lock()
	defer s.mu.Unlock()
	stored, ok := s.issued[permit.Nonce]
	if !ok || stored != permit { return errors.New("permit is unknown, altered, expired from registry or already used") }
	if _, used := s.consumed[permit.Nonce]; used { return errors.New("permit nonce already consumed") }
	if permit.TrustEpoch != s.trustEpoch { return errors.New("permit trust/revocation epoch is stale") }
	if s.targetFences[permit.Target] != permit.FencingToken { return errors.New("permit fencing token is stale") }
	claim, ok := s.claims[permit.ClaimID]
	if !ok || claim.Revoked || claim.Digest != permit.ClaimDigest { return errors.New("claim is missing, revoked or changed") }
	if claim.AppraisedTrustEpoch != s.trustEpoch { return errors.New("claim appraisal is stale under the current trust epoch") }
	now := s.now()
	if !now.Before(permit.ExpiresAt) || !now.Before(claim.LeaseEnds) { return errors.New("permit or claim lease has expired") }
	s.consumed[permit.Nonce] = struct{}{}
	delete(s.issued, permit.Nonce)
	s.targetFences[permit.Target]++
	if err := apply(); err != nil {
		return fmt.Errorf("mutation callback failed after nonce consumption; reconcile outcome before new mutation: %w", err)
	}
	return nil
}
