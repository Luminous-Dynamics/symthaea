package main

import (
	"bytes"
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	custody "luminousdynamics/mon026-custody"
	"luminousdynamics/mon028-benchmark"
)

func main() { os.Exit(run()) }

func decodeStrictFile[T any](path string) (T,error) {
	var zero T
	file,err:=os.Open(path)
	if err!=nil { return zero,err }
	defer file.Close()
	decoder:=json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	var value T
	if err:=decoder.Decode(&value); err!=nil { return zero,err }
	var trailing any
	if err:=decoder.Decode(&trailing); err!=io.EOF { return zero,errors.New("input must contain exactly one JSON value") }
	return value,nil
}

func readTrustedKey(hexValue string) (ed25519.PublicKey,error) {
	decoded,err:=hex.DecodeString(hexValue)
	if err!=nil || len(decoded)!=ed25519.PublicKeySize {
		return nil,errors.New("trusted public key must be 32-byte Ed25519 key encoded as hex")
	}
	return ed25519.PublicKey(decoded),nil
}

func verifyCustodyBundle(root,manifestPath,artifactPath,keyID string,pub ed25519.PublicKey,
	expectedRun string,expectedSequence uint64,expectedPrevious,clockDomain string,
	maxAge,maxUncertainty time.Duration) (custody.Manifest,custody.Result,error) {
	manifest,err:=decodeStrictFile[custody.Manifest](manifestPath)
	if err!=nil { return custody.Manifest{},custody.Result{},err }
	declared:=false
	for _,artifact:=range manifest.Artifacts {
		if artifact.Path==artifactPath { declared=true; break }
	}
	if !declared {
		return manifest,custody.Result{},fmt.Errorf("selected artifact %q is not in signed inventory",artifactPath)
	}
	policy:=custody.Policy{
		Now:time.Now().UTC(),MaxAge:maxAge,ClockDomain:clockDomain,
		MaxClockUncertaintyNS:int64(maxUncertainty),
		ExpectedPreviousRoot:expectedPrevious,
		ExpectedRunID:expectedRun,ExpectedSequence:expectedSequence,
		TrustedKeys:map[string]ed25519.PublicKey{keyID:pub},
	}
	result:=custody.VerifyBundle(root,manifest,policy)
	if result.Verdict!=custody.VerdictQualified {
		return manifest,result,fmt.Errorf("custody verdict is %s: %v",result.Verdict,result.ReasonCodes)
	}
	return manifest,result,nil
}

func readBundleArtifact[T any](root,relative string) (T,error) {
	var zero T
	full:=filepath.Join(root,filepath.FromSlash(relative))
	return decodeStrictFile[T](full)
}

func run() int {
	suiteRoot:=flag.String("suite-bundle-root","","artifact root containing the benchmark suite")
	suiteManifestPath:=flag.String("suite-manifest","","MON-026 suite custody manifest outside artifact root")
	suiteArtifact:=flag.String("suite-artifact","SUITE.json","suite artifact path within suite bundle")
	suiteKeyID:=flag.String("suite-key-id","","out-of-band suite-authority key ID")
	suitePublicKeyHex:=flag.String("suite-public-key-hex","","suite-authority Ed25519 public key hex")
	suiteRunID:=flag.String("suite-run-id","","out-of-band expected suite custody run ID")
	suiteSequence:=flag.Uint64("suite-sequence",0,"out-of-band expected suite custody sequence")
	suitePrevious:=flag.String("suite-previous-root","","expected prior suite custody root for sequence greater than one")

	resultRoot:=flag.String("result-bundle-root","","artifact root containing independently observed result files")
	resultManifestPath:=flag.String("result-manifest","","MON-026 result custody manifest outside artifact root")
	resultArtifact:=flag.String("result-artifact","","candidate result artifact path within result bundle")
	resultKeyID:=flag.String("result-key-id","","out-of-band independent benchmark-observer key ID")
	resultPublicKeyHex:=flag.String("result-public-key-hex","","benchmark-observer Ed25519 public key hex")
	resultRunID:=flag.String("result-run-id","","out-of-band expected result custody run ID")
	resultSequence:=flag.Uint64("result-sequence",0,"out-of-band expected result custody sequence")
	resultPrevious:=flag.String("result-previous-root","","expected prior result custody root for sequence greater than one")
	baselineArtifact:=flag.String("baseline-artifact","","optional baseline result path in the same result bundle")

	suiteClock:=flag.String("suite-clock-domain","","trusted suite-authority clock domain")
	suiteMaxAge:=flag.Duration("suite-max-age",0,"maximum suite bundle age, e.g. 8760h")
	suiteMaxUncertainty:=flag.Duration("suite-max-clock-uncertainty",0,"maximum suite clock uncertainty, e.g. 100ms")
	resultClock:=flag.String("result-clock-domain","","trusted result-observer clock domain")
	resultMaxAge:=flag.Duration("result-max-age",0,"maximum result bundle age, e.g. 24h")
	resultMaxUncertainty:=flag.Duration("result-max-clock-uncertainty",0,"maximum result clock uncertainty, e.g. 100ms")
	flag.Parse()

	if *suiteRoot=="" || *suiteManifestPath=="" || *suiteArtifact=="" ||
		*suiteKeyID=="" || *suitePublicKeyHex=="" || *suiteRunID=="" || *suiteSequence==0 ||
		*resultRoot=="" || *resultManifestPath=="" || *resultArtifact=="" ||
		*resultKeyID=="" || *resultPublicKeyHex=="" || *resultRunID=="" || *resultSequence==0 ||
		*suiteClock=="" || *suiteMaxAge<=0 || *suiteMaxUncertainty<=0 ||
		*resultClock=="" || *resultMaxAge<=0 || *resultMaxUncertainty<=0 {
		fmt.Fprintln(os.Stderr,"suite and result custody trust flags plus --result-artifact are required; see -h")
		return 64
	}
	if *suiteKeyID==*resultKeyID {
		fmt.Fprintln(os.Stderr,"suite-authority and result-observer key IDs must be distinct trust identities")
		return 64
	}
	suitePub,err:=readTrustedKey(*suitePublicKeyHex)
	if err!=nil { fmt.Fprintf(os.Stderr,"suite trust key: %v\n",err); return 64 }
	resultPub,err:=readTrustedKey(*resultPublicKeyHex)
	if err!=nil { fmt.Fprintf(os.Stderr,"result trust key: %v\n",err); return 64 }
	if bytes.Equal(suitePub,resultPub) {
		fmt.Fprintln(os.Stderr,"suite-authority and result-observer public keys must be distinct")
		return 64
	}

	_,suiteCustody,err:=verifyCustodyBundle(*suiteRoot,*suiteManifestPath,*suiteArtifact,*suiteKeyID,suitePub,
		*suiteRunID,*suiteSequence,*suitePrevious,*suiteClock,*suiteMaxAge,*suiteMaxUncertainty)
	if err!=nil {
		_ = json.NewEncoder(os.Stdout).Encode(suiteCustody)
		fmt.Fprintf(os.Stderr,"suite custody verification failed: %v\n",err)
		if suiteCustody.Verdict==custody.VerdictIndeterminate { return 3 }
		return 2
	}
	_,resultCustody,err:=verifyCustodyBundle(*resultRoot,*resultManifestPath,*resultArtifact,*resultKeyID,resultPub,
		*resultRunID,*resultSequence,*resultPrevious,*resultClock,*resultMaxAge,*resultMaxUncertainty)
	if err!=nil {
		_ = json.NewEncoder(os.Stdout).Encode(resultCustody)
		fmt.Fprintf(os.Stderr,"result custody verification failed: %v\n",err)
		if resultCustody.Verdict==custody.VerdictIndeterminate { return 3 }
		return 2
	}

	suite,err:=readBundleArtifact[benchmark.Suite](*suiteRoot,*suiteArtifact)
	if err!=nil { fmt.Fprintf(os.Stderr,"read suite artifact: %v\n",err); return 65 }
	result,err:=readBundleArtifact[benchmark.CampaignResult](*resultRoot,*resultArtifact)
	if err!=nil { fmt.Fprintf(os.Stderr,"read candidate result artifact: %v\n",err); return 65 }
	score,err:=benchmark.ScoreRun(suite,result)
	if err!=nil { fmt.Fprintf(os.Stderr,"score candidate result: %v\n",err); return 2 }

	output:=struct {
		SuiteCustody custody.Result `json:"suite_custody"`
		ResultCustody custody.Result `json:"result_custody"`
		Score benchmark.Score `json:"score"`
		Comparison *benchmark.Comparison `json:"comparison,omitempty"`
	}{SuiteCustody:suiteCustody,ResultCustody:resultCustody,Score:score}
	if *baselineArtifact!="" {
		baseline,readErr:=readBundleArtifact[benchmark.CampaignResult](*resultRoot,*baselineArtifact)
		if readErr!=nil { fmt.Fprintf(os.Stderr,"read baseline result artifact: %v\n",readErr); return 65 }
		comparison,compareErr:=benchmark.CompareMatched(suite,baseline,result)
		if compareErr!=nil { fmt.Fprintf(os.Stderr,"matched comparison: %v\n",compareErr); return 2 }
		output.Comparison=&comparison
	}
	if err:=json.NewEncoder(os.Stdout).Encode(output); err!=nil {
		fmt.Fprintf(os.Stderr,"encode score: %v\n",err); return 70
	}
	if !score.Complete { return 3 }
	if !score.SafetyGate { return 2 }
	return 0
}
