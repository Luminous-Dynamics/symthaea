package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"luminousdynamics/mon027-packager"
)

func main() {
	os.Exit(run())
}

func run() int {
	indexPath := flag.String("index", "", "MON-027 allowlist index JSON outside artifact root")
	artifactRoot := flag.String("artifact-root", "", "directory containing exactly the allowlisted evidence files")
	manifestOut := flag.String("manifest-out", "", "unsigned MON-026 manifest output path")
	requestOut := flag.String("signing-request-out", "", "detached signing request output path")
	flag.Parse()
	if *indexPath == "" || *artifactRoot == "" || *manifestOut == "" || *requestOut == "" {
		fmt.Fprintln(os.Stderr, "all flags are required; see -h")
		return 64
	}
	root, err := filepath.Abs(*artifactRoot)
	if err != nil {
		fmt.Fprintf(os.Stderr, "artifact root: %v\n", err)
		return 66
	}
	manifestAbs, err := filepath.Abs(*manifestOut)
	if err != nil {
		fmt.Fprintf(os.Stderr, "manifest output: %v\n", err)
		return 66
	}
	requestAbs, err := filepath.Abs(*requestOut)
	if err != nil {
		fmt.Fprintf(os.Stderr, "signing request output: %v\n", err)
		return 66
	}
	if manifestAbs == requestAbs {
		fmt.Fprintln(os.Stderr, "manifest and signing request outputs must be different paths")
		return 64
	}
	for _, output := range []string{*indexPath, *manifestOut, *requestOut} {
		absolute, err := filepath.Abs(output)
		if err != nil {
			fmt.Fprintf(os.Stderr, "resolve path %s: %v\n", output, err)
			return 66
		}
		relative, err := filepath.Rel(root, absolute)
		if err != nil || (relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator))) {
			fmt.Fprintf(os.Stderr, "index and outputs must be outside artifact root: %s\n", output)
			return 64
		}
	}

	indexFile, err := os.Open(*indexPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open index: %v\n", err)
		return 66
	}
	index, decodeErr := packager.DecodeStrict[packager.InputIndex](indexFile)
	_ = indexFile.Close()
	if decodeErr != nil {
		fmt.Fprintf(os.Stderr, "decode index: %v\n", decodeErr)
		return 65
	}

	built, err := packager.BuildManifest(root, index)
	if err != nil {
		fmt.Fprintf(os.Stderr, "assemble bundle: %v\n", err)
		return 2
	}
	manifestBytes, err := json.MarshalIndent(built.Manifest, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode manifest: %v\n", err)
		return 70
	}
	requestBytes, err := json.MarshalIndent(built.SigningRequest, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode signing request: %v\n", err)
		return 70
	}
	if err := writeExclusive(*manifestOut, append(manifestBytes, '\n')); err != nil {
		fmt.Fprintf(os.Stderr, "write manifest: %v\n", err)
		return 73
	}
	if err := writeExclusive(*requestOut, append(requestBytes, '\n')); err != nil {
		_ = os.Remove(*manifestOut)
		fmt.Fprintf(os.Stderr, "write signing request: %v\n", err)
		return 73
	}
	fmt.Printf("root_sha256=%s\n", built.Manifest.RootSHA256)
	fmt.Printf("signer_key_id=%s\n", built.SigningRequest.KeyID)
	fmt.Printf("artifact_count=%d\n", len(built.Manifest.Artifacts))
	fmt.Println("unsigned manifest and detached signing request written; no key was read or signature created")
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
