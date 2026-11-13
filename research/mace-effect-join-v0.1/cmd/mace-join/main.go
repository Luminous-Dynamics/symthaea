// mace-join verifies an untrusted evidence bundle against a separate,
// operator-pinned trust store and policy/intent file. Never keep these files in
// the same agent-writable location as the evidence bundle.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/Luminous-Dynamics/symthaea/research/mace-effect-join-v0.1/join"
)

type policyFile struct {
	State    join.PolicyState       `json:"policy_state"`
	Expected join.IntentExpectation `json:"expected_intent"`
}

func main() {
	bundlePath := flag.String("bundle", "-", "signed evidence bundle JSON; use - for stdin")
	trustPath := flag.String("trust-store", "", "operator-pinned trust store JSON (required)")
	policyPath := flag.String("policy", "", "operator-pinned policy and task-intent JSON (required)")
	flag.Parse()
	if *trustPath == "" || *policyPath == "" {
		fail(errors.New("--trust-store and --policy are required and must be provisioned out of band"))
	}
	bundleBytes, err := readInput(*bundlePath)
	if err != nil { fail(err) }
	trustBytes, err := readFileBounded(*trustPath)
	if err != nil { fail(fmt.Errorf("read trusted root configuration: %w", err)) }
	policyBytes, err := readFileBounded(*policyPath)
	if err != nil { fail(fmt.Errorf("read pinned policy: %w", err)) }
	var bundle join.Bundle
	var trust join.TrustStore
	var policy policyFile
	if err := join.DecodeStrict(bundleBytes, &bundle); err != nil { fail(fmt.Errorf("decode evidence bundle: %w", err)) }
	if err := join.DecodeStrict(trustBytes, &trust); err != nil { fail(fmt.Errorf("decode trust store: %w", err)) }
	if err := join.DecodeStrict(policyBytes, &policy); err != nil { fail(fmt.Errorf("decode policy/intent: %w", err)) }
	result, err := join.Verify(bundle, trust, policy.State, policy.Expected)
	if err != nil { fail(err) }
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(result); err != nil { fail(err) }
}
func readInput(path string) ([]byte, error) {
	var reader io.Reader = os.Stdin
	if path != "-" {
		f, err := os.Open(path)
		if err != nil { return nil, err }
		defer f.Close()
		reader = f
	}
	data, err := io.ReadAll(io.LimitReader(reader, join.MaximumRequestBytes+1))
	if err != nil { return nil, err }
	if len(data) == 0 || len(data) > join.MaximumRequestBytes {
		return nil, fmt.Errorf("evidence bundle is empty or exceeds %d bytes", join.MaximumRequestBytes)
	}
	return data, nil
}
func fail(err error) {
	fmt.Fprintln(os.Stderr, "mace-join:", err)
	os.Exit(1)
}
func readFileBounded(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil { return nil, err }
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, join.MaximumRequestBytes+1))
	if err != nil { return nil, err }
	if len(data) == 0 || len(data) > join.MaximumRequestBytes {
		return nil, fmt.Errorf("file %q is empty or exceeds %d bytes", path, join.MaximumRequestBytes)
	}
	return data, nil
}
