package main

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	custody "luminousdynamics/mon026-custody"
)

func testHash(value string) string {
	sum:=sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func makeCustodyBundle(t *testing.T) (string,string,string,ed25519.PublicKey,custody.Policy) {
	t.Helper()
	root:=t.TempDir()
	artifactBytes:=[]byte("{\"suite\":\"independent\"}\n")
	if err:=os.WriteFile(filepath.Join(root,"SUITE.json"),artifactBytes,0o600); err!=nil { t.Fatal(err) }
	publicKey,privateKey,err:=ed25519.GenerateKey(rand.Reader)
	if err!=nil { t.Fatal(err) }
	now:=time.Date(2026,11,27,10,0,0,0,time.UTC)
	manifest:=custody.Manifest{
		Profile:custody.ProfileV1,Version:custody.VersionV1,
		RunID:"benchmark-suite-release-001",Sequence:1,
		CapturedAt:now,ClockDomain:"benchmark-authority-clock",ClockUncertaintyNS:int64(5*time.Millisecond),
		Source:custody.SourceBoundary{
			IncidentID:"benchmark-release",SourceClusterID:"benchmark-authority",
			Database:"benchmark",SystemIdentifier:"123456789",TimelineID:"1",
			SnapshotID:"suite-release-001",CaptureLSN:"0/ABCDEF0",
			ScopeDigest:testHash("scope"),InventoryDigest:testHash("inventory"),
		},
		Artifacts:[]custody.Artifact{{
			Path:"SUITE.json",MediaType:"application/json",Role:"independent-benchmark-suite",
			Producer:"benchmark-authority",CommandID:"publish-suite",
			SizeBytes:int64(len(artifactBytes)),SHA256:testHash(string(artifactBytes)),
			CollectedAt:now.Add(-time.Second),ClockDomain:"benchmark-authority-clock",
			ClockUncertaintyNS:int64(5*time.Millisecond),
		}},
	}
	if err:=custody.SignManifest(&manifest,"suite-authority-key",privateKey); err!=nil { t.Fatal(err) }
	manifestPath:=filepath.Join(t.TempDir(),"manifest.signed.json")
	raw,err:=jsonMarshal(manifest)
	if err!=nil { t.Fatal(err) }
	if err:=os.WriteFile(manifestPath,raw,0o600); err!=nil { t.Fatal(err) }
	policy:=custody.Policy{
		Now:now.Add(time.Minute),MaxAge:time.Hour,ClockDomain:"benchmark-authority-clock",
		MaxClockUncertaintyNS:int64(100*time.Millisecond),
		ExpectedRunID:manifest.RunID,ExpectedSequence:1,
		TrustedKeys:map[string]ed25519.PublicKey{"suite-authority-key":publicKey},
	}
	return root,manifestPath,manifest.RunID,publicKey,policy
}

func jsonMarshal(value any) ([]byte,error) {
	return json.MarshalIndent(value,"","  ")
}

func TestVerifyCustodyBundleAcceptsTrustedExactSuite(t *testing.T) {
	root,manifestPath,runID,publicKey,_:=makeCustodyBundle(t)
	manifest,result,err:=verifyCustodyBundle(root,manifestPath,"SUITE.json","suite-authority-key",publicKey,
		runID,1,"","benchmark-authority-clock",24*time.Hour,100*time.Millisecond)
	if err!=nil { t.Fatal(err) }
	if result.Verdict!=custody.VerdictQualified || manifest.RunID!=runID {
		t.Fatalf("expected trusted suite custody, got result=%+v manifest=%+v",result,manifest)
	}
}

func TestVerifyCustodyBundleRejectsUndeclaredSuiteArtifact(t *testing.T) {
	root,manifestPath,runID,publicKey,_:=makeCustodyBundle(t)
	_,_,err:=verifyCustodyBundle(root,manifestPath,"other.json","suite-authority-key",publicKey,
		runID,1,"","benchmark-authority-clock",24*time.Hour,100*time.Millisecond)
	if err==nil { t.Fatal("artifact absent from signed inventory must be rejected") }
}

func TestVerifyCustodyBundleBlocksWrongExpectedRun(t *testing.T) {
	root,manifestPath,_,publicKey,_:=makeCustodyBundle(t)
	_,result,err:=verifyCustodyBundle(root,manifestPath,"SUITE.json","suite-authority-key",publicKey,
		"another-run",1,"","benchmark-authority-clock",24*time.Hour,100*time.Millisecond)
	if err==nil || result.Verdict!=custody.VerdictBlocked {
		t.Fatalf("expected wrong run identity to block, result=%+v err=%v",result,err)
	}
}

func TestTrustSeparationRequiresDistinctIDsAndKeys(t *testing.T) {
	firstPublic,_,err:=ed25519.GenerateKey(rand.Reader)
	if err!=nil { t.Fatal(err) }
	secondPublic,_,err:=ed25519.GenerateKey(rand.Reader)
	if err!=nil { t.Fatal(err) }
	if err:=validateTrustSeparation("authority",firstPublic,"observer",secondPublic); err!=nil {
		t.Fatalf("distinct identities should pass: %v",err)
	}
	if err:=validateTrustSeparation("same",firstPublic,"same",secondPublic); err==nil {
		t.Fatal("same signer identity must not be accepted")
	}
	if err:=validateTrustSeparation("authority",firstPublic,"observer",firstPublic); err==nil {
		t.Fatal("same public key under different labels must not be accepted")
	}
}

func TestVerifyCustodyBundleUnknownSignerIsIndeterminate(t *testing.T) {
	root,manifestPath,runID,publicKey,_:=makeCustodyBundle(t)
	_,result,err:=verifyCustodyBundle(root,manifestPath,"SUITE.json","untrusted-key",publicKey,
		runID,1,"","benchmark-authority-clock",24*time.Hour,100*time.Millisecond)
	if err==nil || result.Verdict!=custody.VerdictIndeterminate {
		t.Fatalf("unknown trust identity must remain indeterminate, result=%+v err=%v",result,err)
	}
}

func TestVerifyCustodyBundleRejectsWrongSequence(t *testing.T) {
	root,manifestPath,runID,publicKey,_:=makeCustodyBundle(t)
	_,result,err:=verifyCustodyBundle(root,manifestPath,"SUITE.json","suite-authority-key",publicKey,
		runID,2,"","benchmark-authority-clock",24*time.Hour,100*time.Millisecond)
	if err==nil || result.Verdict!=custody.VerdictBlocked {
		t.Fatalf("wrong custody sequence must block, result=%+v err=%v",result,err)
	}
}

func TestVerifyCustodyBundleDetectsArtifactMutation(t *testing.T) {
	root,manifestPath,runID,publicKey,_:=makeCustodyBundle(t)
	if err:=os.WriteFile(filepath.Join(root,"SUITE.json"),[]byte("substituted suite"),0o600); err!=nil {
		t.Fatal(err)
	}
	_,result,err:=verifyCustodyBundle(root,manifestPath,"SUITE.json","suite-authority-key",publicKey,
		runID,1,"","benchmark-authority-clock",24*time.Hour,100*time.Millisecond)
	if err==nil || result.Verdict!=custody.VerdictBlocked {
		t.Fatalf("suite artifact substitution must block, result=%+v err=%v",result,err)
	}
}
