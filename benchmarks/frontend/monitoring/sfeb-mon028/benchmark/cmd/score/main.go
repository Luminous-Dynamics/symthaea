package main

import (
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

func run() int {
	bundleRoot:=flag.String("suite-bundle-root","","directory containing the independently custodied suite artifact")
	manifestPath:=flag.String("suite-manifest","","MON-026 signed custody manifest outside suite bundle root")
	suiteArtifact:=flag.String("suite-artifact","SUITE.json","relative artifact path of the benchmark suite inside the bundle")
	keyID:=flag.String("suite-key-id","","out-of-band trusted Ed25519 key ID")
	publicKeyHex:=flag.String("suite-public-key-hex","","out-of-band trusted Ed25519 public key as 64 hex characters")
	expectedRunID:=flag.String("suite-run-id","","out-of-band expected custody run ID")
	expectedSequence:=flag.Uint64("suite-sequence",0,"out-of-band expected custody sequence")
	expectedPrevious:=flag.String("suite-previous-root","","out-of-band expected previous custody root for sequence greater than one")
	clockDomain:=flag.String("clock-domain","","trusted verifier clock domain")
	maxAge:=flag.Duration("max-age",0,"maximum suite age, e.g. 365d")
	maxUncertainty:=flag.Duration("max-clock-uncertainty",0,"maximum clock uncertainty, e.g. 100ms")
	resultPath:=flag.String("result","","candidate campaign result JSON")
	baselinePath:=flag.String("baseline","","optional matched baseline result JSON")
	flag.Parse()

	if *bundleRoot=="" || *manifestPath=="" || *suiteArtifact=="" || *keyID=="" ||
		*publicKeyHex=="" || *expectedRunID=="" || *expectedSequence==0 ||
		*clockDomain=="" || *maxAge<=0 || *maxUncertainty<=0 || *resultPath=="" {
		fmt.Fprintln(os.Stderr,"custodied suite trust flags and --result are required; see -h")
		return 64
	}
	publicKeyBytes,err:=hex.DecodeString(*publicKeyHex)
	if err!=nil || len(publicKeyBytes)!=ed25519.PublicKeySize {
		fmt.Fprintln(os.Stderr,"suite trusted public key must be 32-byte Ed25519 public key hex")
		return 64
	}
	manifest,err:=decodeStrictFile[custody.Manifest](*manifestPath)
	if err!=nil { fmt.Fprintf(os.Stderr,"read suite custody manifest: %v\n",err); return 65 }
	declared:=false
	for _,artifact:=range manifest.Artifacts {
		if artifact.Path==*suiteArtifact { declared=true; break }
	}
	if !declared {
		fmt.Fprintln(os.Stderr,"selected suite artifact is not in the signed manifest inventory")
		return 2
	}
	policy:=custody.Policy{
		Now:time.Now().UTC(),MaxAge:*maxAge,ClockDomain:*clockDomain,
		MaxClockUncertaintyNS:int64(*maxUncertainty),
		ExpectedPreviousRoot:*expectedPrevious,
		ExpectedRunID:*expectedRunID,ExpectedSequence:*expectedSequence,
		TrustedKeys:map[string]ed25519.PublicKey{*keyID:ed25519.PublicKey(publicKeyBytes)},
	}
	custodyResult:=custody.VerifyBundle(*bundleRoot,manifest,policy)
	if custodyResult.Verdict!=custody.VerdictQualified {
		_ = json.NewEncoder(os.Stdout).Encode(custodyResult)
		if custodyResult.Verdict==custody.VerdictBlocked { return 2 }
		return 3
	}
	suitePath:=filepath.Join(*bundleRoot,filepath.FromSlash(*suiteArtifact))
	suite,err:=decodeStrictFile[benchmark.Suite](suitePath)
	if err!=nil { fmt.Fprintf(os.Stderr,"read suite artifact: %v\n",err); return 65 }
	result,err:=decodeStrictFile[benchmark.CampaignResult](*resultPath)
	if err!=nil { fmt.Fprintf(os.Stderr,"read candidate result: %v\n",err); return 65 }
	score,err:=benchmark.ScoreRun(suite,result)
	if err!=nil { fmt.Fprintf(os.Stderr,"score candidate result: %v\n",err); return 2 }

	output:=struct {
		Custody custody.Result `json:"suite_custody"`
		Score benchmark.Score `json:"score"`
		Comparison *benchmark.Comparison `json:"comparison,omitempty"`
	}{Custody:custodyResult,Score:score}
	if *baselinePath!="" {
		baseline,readErr:=decodeStrictFile[benchmark.CampaignResult](*baselinePath)
		if readErr!=nil { fmt.Fprintf(os.Stderr,"read baseline: %v\n",readErr); return 65 }
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
