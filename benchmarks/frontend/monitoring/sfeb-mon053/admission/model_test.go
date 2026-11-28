package admission

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestCaptureIntervalRevocationBoundaries(t *testing.T) {
	base := time.Date(2026, 11, 28, 12, 0, 0, 0, time.UTC)
	revoke := base.Add(10 * time.Second)
	cases := []struct{name string; interval Interval; want Verdict}{
		{"certainly_before", Interval{base, base.Add(9*time.Second)}, Qualified},
		{"overlap_before_boundary", Interval{base.Add(9*time.Second), base.Add(11*time.Second)}, Indeterminate},
		{"touch_upper_boundary", Interval{base.Add(8*time.Second), revoke}, Indeterminate},
		{"revocation_at_lower_boundary", Interval{revoke, base.Add(12*time.Second)}, Blocked},
		{"certainly_after", Interval{base.Add(11*time.Second), base.Add(12*time.Second)}, Blocked},
		{"point_at_boundary", Interval{revoke, revoke}, Blocked},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := AppraiseCapture(tc.interval, &revoke)
			if err != nil { t.Fatal(err) }
			if got.Verdict != tc.want { t.Fatalf("got %s want %s (%s)", got.Verdict, tc.want, got.Reason) }
		})
	}
	unknown, err := AppraiseCapture(Interval{base, base}, nil)
	if err != nil || unknown.Verdict != Indeterminate { t.Fatalf("unknown revocation must remain indeterminate: %+v %v", unknown, err) }
	bad, err := AppraiseCapture(Interval{base.Add(time.Second), base}, &revoke)
	if err == nil || bad.Verdict != Indeterminate { t.Fatalf("reversed interval must fail closed: %+v %v", bad, err) }
}

func TestMinimumMandatoryDependencyExpiry(t *testing.T) {
	base := time.Now().UTC()
	got, err := MinimumMandatoryExpiry([]DependencyExpiry{{"db", true, base.Add(time.Hour)}, {"kafka", true, base.Add(5*time.Minute)}, {"iam", true, base.Add(30*time.Minute)}})
	if err != nil { t.Fatal(err) }
	if !got.Equal(base.Add(5*time.Minute)) { t.Fatalf("got %v", got) }
	for _, tc := range []struct{name string; deps []DependencyExpiry}{
		{"empty", nil}, {"unknown", []DependencyExpiry{{"db", false, base}}}, {"zero", []DependencyExpiry{{"db", true, time.Time{}}}}, {"duplicate", []DependencyExpiry{{"db", true, base}, {"db", true, base.Add(time.Hour)}}},
	} {
		t.Run(tc.name, func(t *testing.T) { if _, err := MinimumMandatoryExpiry(tc.deps); err == nil { t.Fatal("expected fail-closed error") } })
	}
}

func testStore(t *testing.T) (*Store, *time.Time, uint64) {
	t.Helper()
	now := time.Date(2026, 11, 28, 12, 0, 0, 0, time.UTC)
	clock := &now
	s := NewStore(7, func() time.Time { return *clock })
	if err := s.RegisterClaim(Claim{ID: "claim-1", Digest: "sha256:claim-1", LeaseEnds: now.Add(time.Hour), AppraisedTrustEpoch: 7}); err != nil { t.Fatal(err) }
	return s, clock, 7
}
func issue(t *testing.T, s *Store, epoch uint64, nonce string) Permit {
	t.Helper()
	p, err := s.IssuePermit("claim-1", "sha256:claim-1", "db-primary", "restore-and-promote", nonce, epoch, time.Minute)
	if err != nil { t.Fatal(err) }
	return p
}

func TestOneUsePermitAndReplayRejection(t *testing.T) {
	s, _, epoch := testStore(t); p := issue(t, s, epoch, "nonce-1"); var calls atomic.Int32
	if err := s.AdmitAndApply(p, func() error { calls.Add(1); return nil }); err != nil { t.Fatal(err) }
	if err := s.AdmitAndApply(p, func() error { calls.Add(1); return nil }); err == nil { t.Fatal("replayed permit accepted") }
	if calls.Load() != 1 { t.Fatalf("callback called %d times", calls.Load()) }
}

func TestConcurrentNonceConsumptionAtMostOnce(t *testing.T) {
	s, _, epoch := testStore(t); p := issue(t, s, epoch, "nonce-race")
	const contenders = 256
	var calls atomic.Int32; var accepted atomic.Int32
	start := make(chan struct{}); var wg sync.WaitGroup
	for i:=0; i<contenders; i++ {
		wg.Add(1); go func() {
			defer wg.Done(); <-start
			if err:=s.AdmitAndApply(p,func()error{calls.Add(1);return nil}); err==nil { accepted.Add(1) }
		}()
	}
	close(start); wg.Wait()
	if accepted.Load()!=1 || calls.Load()!=1 { t.Fatalf("accepted=%d callbacks=%d",accepted.Load(),calls.Load()) }
}

func TestRevocationBeforeAdmissionRejectsAndBumpsEpoch(t *testing.T) {
	s, _, epoch := testStore(t); p := issue(t, s, epoch, "nonce-revoked")
	if err:=s.RevokeClaim("claim-1"); err!=nil { t.Fatal(err) }
	var calls atomic.Int32
	if err:=s.AdmitAndApply(p,func()error{calls.Add(1);return nil}); err==nil { t.Fatal("revoked permit accepted") }
	if calls.Load()!=0 { t.Fatal("mutation callback ran after revocation") }
	if s.TrustEpoch()!=epoch+1 { t.Fatalf("epoch=%d want %d",s.TrustEpoch(),epoch+1) }
}

func TestTrustEpochFenceScopeAndExpiryAreEnforced(t *testing.T) {
	t.Run("trust_epoch",func(t *testing.T) {
		s,clock,epoch:=testStore(t);p:=issue(t,s,epoch,"nonce-epoch")
		if err:=s.AdvanceTrustEpoch(epoch+1);err!=nil{t.Fatal(err)}
		if err:=s.AdmitAndApply(p,func()error{return nil});err==nil{t.Fatal("stale epoch accepted")}
		if _,err:=s.IssuePermit("claim-1","sha256:claim-1","db-primary","restore","nonce-not-reappraised",epoch+1,time.Minute);err==nil{t.Fatal("claim reused without fresh appraisal")}
		if err:=s.ReappraiseClaim("claim-1","sha256:claim-1-v2",clock.Add(time.Hour));err!=nil{t.Fatal(err)}
		p2,err:=s.IssuePermit("claim-1","sha256:claim-1-v2","db-primary","restore","nonce-reappraised",epoch+1,time.Minute);if err!=nil{t.Fatal(err)}
		if err:=s.AdmitAndApply(p2,func()error{return nil});err!=nil{t.Fatal(err)}
	})
	t.Run("fencing",func(t *testing.T) {
		s,_,epoch:=testStore(t);p1:=issue(t,s,epoch,"nonce-fence-1");p2:=issue(t,s,epoch,"nonce-fence-2")
		if err:=s.AdmitAndApply(p1,func()error{return nil});err==nil{t.Fatal("old fence accepted")}
		if err:=s.AdmitAndApply(p2,func()error{return nil});err!=nil{t.Fatal(err)}
	})
	t.Run("scope_substitution",func(t *testing.T) {
		s,_,epoch:=testStore(t);p:=issue(t,s,epoch,"nonce-scope");p.Target="db-replica"
		if err:=s.AdmitAndApply(p,func()error{return nil});err==nil{t.Fatal("altered target accepted")}
	})
	t.Run("expired_permit",func(t *testing.T) {
		s,clock,epoch:=testStore(t);p:=issue(t,s,epoch,"nonce-expired");*clock=p.ExpiresAt
		if err:=s.AdmitAndApply(p,func()error{return nil});err==nil{t.Fatal("expired permit accepted")}
	})
}

func TestCallbackFailureConsumesNonce(t *testing.T) {
	s,_,epoch:=testStore(t);p:=issue(t,s,epoch,"nonce-uncertain")
	want:=errors.New("simulated unknown external outcome")
	if err:=s.AdmitAndApply(p,func()error{return want});err==nil{t.Fatal("expected callback error")}
	if err:=s.AdmitAndApply(p,func()error{return nil});err==nil{t.Fatal("failed callback made nonce reusable")}
}

func TestConcurrentRevocationIsLinearizedWithAdmission(t *testing.T) {
	// The mutex defines a single order. If mutation runs, admission held the serialization lock first; if revoke wins, permit rejects.
	for run:=0;run<100;run++ {
		t.Run(fmt.Sprintf("schedule_%03d",run),func(t *testing.T) {
			s,_,_:=testStore(t);p:=issue(t,s,7,fmt.Sprintf("nonce-%d",run));start:=make(chan struct{})
			var calls atomic.Int32;var admitErr error;var revokeErr error;var wg sync.WaitGroup;wg.Add(2)
			go func(){defer wg.Done();<-start;admitErr=s.AdmitAndApply(p,func()error{calls.Add(1);return nil})}()
			go func(){defer wg.Done();<-start;revokeErr=s.RevokeClaim("claim-1")}()
			close(start);wg.Wait()
			if revokeErr!=nil{t.Fatal(revokeErr)}
			if calls.Load()>1{t.Fatal("more than one mutation")}
			if calls.Load()==1&&admitErr!=nil{t.Fatalf("callback ran but admission errored: %v",admitErr)}
			if calls.Load()==0&&admitErr==nil{t.Fatal("successful admission without callback")}
		})
	}
}

func TestTargetOperationAndDigestSubstitutionAreRejected(t *testing.T) {
	for _, field := range []string{"target", "operation", "digest", "claim"} {
		t.Run(field, func(t *testing.T) {
			s,_,epoch:=testStore(t);p:=issue(t,s,epoch,"nonce-substitution-"+field)
			switch field {
			case "target":p.Target="other-target"
			case "operation":p.Operation="delete"
			case "digest":p.ClaimDigest="sha256:other"
			case "claim":p.ClaimID="other-claim"
			}
			if err:=s.AdmitAndApply(p,func()error{return nil});err==nil{t.Fatalf("%s substitution accepted",field)}
		})
	}
}

func TestClaimLeaseExpiryIsEnforcedAfterPermitIssue(t *testing.T) {
	s,clock,epoch:=testStore(t);p:=issue(t,s,epoch,"nonce-lease-expiry")
	*clock=p.ExpiresAt.Add(-time.Nanosecond)
	s.mu.Lock();claim:=s.claims["claim-1"];claim.LeaseEnds=clock.Add(-time.Nanosecond);s.claims["claim-1"]=claim;s.mu.Unlock()
	if err:=s.AdmitAndApply(p,func()error{return nil});err==nil{t.Fatal("expired claim lease accepted")}
}

func TestPermitDeadlineCannotOutliveWeakestDependency(t *testing.T) {
	s,clock,epoch:=testStore(t);short:=clock.Add(20*time.Second)
	if err:=s.RegisterClaim(Claim{ID:"claim-short",Digest:"sha256:short",LeaseEnds:short,AppraisedTrustEpoch:epoch});err!=nil{t.Fatal(err)}
	p,err:=s.IssuePermit("claim-short","sha256:short","db-secondary","restore","nonce-short",epoch,time.Hour);if err!=nil{t.Fatal(err)}
	if !p.ExpiresAt.Equal(short){t.Fatalf("permit outlived claim lease: %v vs %v",p.ExpiresAt,short)}
}
