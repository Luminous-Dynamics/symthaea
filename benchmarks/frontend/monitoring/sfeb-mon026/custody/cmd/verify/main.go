package main

import (
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"luminousdynamics/mon026-custody"
)

func main() {
	os.Exit(run())
}

func run() int {
	bundleRoot := flag.String("bundle-root", "", "directory containing exactly the declared artifact files")
	manifestPath := flag.String("manifest", "", "path to the signed manifest JSON (keep outside bundle-root)")
	keyID := flag.String("key-id", "", "trusted signer key ID")
	publicKeyHex := flag.String("public-key-hex", "", "out-of-band trusted Ed25519 public key as 64 hex characters")
	expectedRunID := flag.String("expected-run-id", "", "out-of-band expected run ID")
	expectedSequence := flag.Uint64("expected-sequence", 0, "out-of-band exact expected bundle sequence")
	expectedPrevious := flag.String("expected-previous-root", "", "out-of-band expected previous root SHA-256; required after sequence 1")
	clockDomain := flag.String("clock-domain", "", "trusted verifier clock domain")
	maxAge := flag.Duration("max-age", 0, "maximum accepted manifest age, e.g. 24h")
	maxUncertainty := flag.Duration("max-clock-uncertainty", 0, "maximum clock uncertainty, e.g. 100ms")
	flag.Parse()

	if *bundleRoot == "" || *manifestPath == "" || *keyID == "" || *publicKeyHex == "" ||
		*expectedRunID == "" || *expectedSequence == 0 || *clockDomain == "" ||
		*maxAge <= 0 || *maxUncertainty <= 0 {
		fmt.Fprintln(os.Stderr, "all required flags must be supplied; see -h")
		return 64
	}
	pubBytes, err := hex.DecodeString(*publicKeyHex)
	if err != nil || len(pubBytes) != ed25519.PublicKeySize {
		fmt.Fprintln(os.Stderr, "trusted public key must be a valid Ed25519 public key encoded as hex")
		return 64
	}

	file, err := os.Open(*manifestPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open manifest: %v\n", err)
		return 66
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	var manifest custody.Manifest
	if err := decoder.Decode(&manifest); err != nil {
		fmt.Fprintf(os.Stderr, "decode manifest: %v\n", err)
		return 65
	}

	policy := custody.Policy{
		Now:                   time.Now().UTC(),
		MaxAge:                *maxAge,
		ClockDomain:           *clockDomain,
		MaxClockUncertaintyNS: int64(*maxUncertainty),
		ExpectedPreviousRoot:  *expectedPrevious,
		ExpectedRunID:         *expectedRunID,
		ExpectedSequence:      *expectedSequence,
		TrustedKeys: map[string]ed25519.PublicKey{
			*keyID: ed25519.PublicKey(pubBytes),
		},
	}
	result := custody.VerifyBundle(*bundleRoot, manifest, policy)
	if err := json.NewEncoder(os.Stdout).Encode(result); err != nil {
		fmt.Fprintf(os.Stderr, "encode result: %v\n", err)
		return 70
	}
	switch result.Verdict {
	case custody.VerdictQualified:
		return 0
	case custody.VerdictBlocked:
		return 2
	default:
		return 3
	}
}
