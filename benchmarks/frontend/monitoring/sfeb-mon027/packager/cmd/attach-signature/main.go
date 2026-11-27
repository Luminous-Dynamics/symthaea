package main

import (
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	custody "luminousdynamics/mon026-custody"
	packager "luminousdynamics/mon027-packager"
)

func main() {
	os.Exit(run())
}

func run() int {
	artifactRoot := flag.String("artifact-root", "", "directory containing exactly the declared raw artifacts")
	manifestPath := flag.String("manifest", "", "unsigned MON-026 manifest outside artifact root")
	signaturePath := flag.String("signature", "", "detached signature envelope returned by external signer")
	keyID := flag.String("key-id", "", "out-of-band expected signer key ID")
	publicKeyHex := flag.String("public-key-hex", "", "out-of-band trusted Ed25519 public key as 64 hex characters")
	expectedRunID := flag.String("expected-run-id", "", "out-of-band expected run ID")
	expectedSequence := flag.Uint64("expected-sequence", 0, "out-of-band exact sequence")
	expectedPrevious := flag.String("expected-previous-root", "", "out-of-band expected prior root for sequence greater than one")
	clockDomain := flag.String("clock-domain", "", "trusted verifier clock domain")
	maxAge := flag.Duration("max-age", 0, "maximum evidence age, e.g. 24h")
	maxUncertainty := flag.Duration("max-clock-uncertainty", 0, "maximum clock uncertainty, e.g. 100ms")
	output := flag.String("signed-manifest-out", "", "new signed manifest output path")
	flag.Parse()
	if *artifactRoot == "" || *manifestPath == "" || *signaturePath == "" || *keyID == "" ||
		*publicKeyHex == "" || *expectedRunID == "" || *expectedSequence == 0 ||
		*clockDomain == "" || *maxAge <= 0 || *maxUncertainty <= 0 || *output == "" {
		fmt.Fprintln(os.Stderr, "all flags are required; see -h")
		return 64
	}
	for _, p := range []string{*manifestPath, *signaturePath, *output} {
		root, err := filepath.Abs(*artifactRoot)
		absolute, absErr := filepath.Abs(p)
		if err != nil || absErr != nil {
			fmt.Fprintln(os.Stderr, "cannot resolve artifact/output paths")
			return 64
		}
		rel, relErr := filepath.Rel(root, absolute)
		if relErr != nil || rel == "." || rel == ".." || (len(rel) >= 3 && rel[:3] == ".."+string(filepath.Separator)) {
			fmt.Fprintf(os.Stderr, "manifest, signature and output must be outside artifact root: %s\n", p)
			return 64
		}
	}

	pubBytes, err := hex.DecodeString(*publicKeyHex)
	if err != nil || len(pubBytes) != ed25519.PublicKeySize {
		fmt.Fprintln(os.Stderr, "invalid Ed25519 public key hex")
		return 64
	}
	manifestFile, err := os.Open(*manifestPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open manifest: %v\n", err)
		return 66
	}
	manifest, decodeErr := packager.DecodeStrict[custody.Manifest](manifestFile)
	_ = manifestFile.Close()
	if decodeErr != nil {
		fmt.Fprintf(os.Stderr, "decode manifest: %v\n", decodeErr)
		return 65
	}
	signatureFile, err := os.Open(*signaturePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open signature: %v\n", err)
		return 66
	}
	envelope, decodeErr := packager.DecodeStrict[packager.SignatureEnvelope](signatureFile)
	_ = signatureFile.Close()
	if decodeErr != nil {
		fmt.Fprintf(os.Stderr, "decode detached signature: %v\n", decodeErr)
		return 65
	}
	if manifest.Signature.KeyID != *keyID || envelope.KeyID != *keyID {
		fmt.Fprintln(os.Stderr, "signer key ID does not match independent trust policy")
		return 2
	}
	policy := custody.Policy{
		Now: time.Now().UTC(),
		MaxAge: *maxAge,
		ClockDomain: *clockDomain,
		MaxClockUncertaintyNS: int64(*maxUncertainty),
		ExpectedPreviousRoot: *expectedPrevious,
		ExpectedRunID: *expectedRunID,
		ExpectedSequence: *expectedSequence,
		TrustedKeys: map[string]ed25519.PublicKey{
			*keyID: ed25519.PublicKey(pubBytes),
		},
	}
	signed, result := packager.AttachSignature(*artifactRoot, manifest, envelope, policy)
	encoded, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode verification result: %v\n", err)
		return 70
	}
	fmt.Fprintln(os.Stdout, string(encoded))
	if result.Verdict != custody.VerdictQualified {
		if result.Verdict == custody.VerdictBlocked {
			return 2
		}
		return 3
	}
	manifestBytes, err := json.MarshalIndent(signed, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode signed manifest: %v\n", err)
		return 70
	}
	if err := writeExclusive(*output, append(manifestBytes, '\n')); err != nil {
		fmt.Fprintf(os.Stderr, "write signed manifest: %v\n", err)
		return 73
	}
	fmt.Printf("signed_manifest=%s\n", *output)
	return 0
}

func writeExclusive(path string, data []byte) error {
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	_, writeErr := file.Write(data)
	syncErr := file.Sync()
	closeErr := file.Close()
	if writeErr != nil {
		_ = os.Remove(path)
		return writeErr
	}
	if syncErr != nil {
		_ = os.Remove(path)
		return syncErr
	}
	if closeErr != nil {
		_ = os.Remove(path)
		return closeErr
	}
	return nil
}
