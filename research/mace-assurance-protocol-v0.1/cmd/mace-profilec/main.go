// mace-profilec compiles a declared SUT boundary and scenario catalogue into
// a deterministic mandatory-control inventory. It does not authenticate the
// catalogue or decide whether the catalogue is complete.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/Luminous-Dynamics/symthaea/research/mace-assurance-protocol-v0.1/matrix"
)

type compileRequest struct {
	SchemaVersion string                `json:"schema_version"`
	Profile       matrix.SUTProfile     `json:"profile"`
	Catalog       matrix.ProfileCatalog `json:"catalog"`
}

func main() {
	inputPath := flag.String("input", "-", "request JSON file; use - for stdin")
	flag.Parse()
	var input io.Reader = os.Stdin
	if *inputPath != "-" {
		file, err := os.Open(*inputPath)
		if err != nil { fail(err) }
		defer file.Close()
		input = file
	}
	if err := run(input, os.Stdout); err != nil { fail(err) }
}
func fail(err error) { fmt.Fprintln(os.Stderr, "mace-profilec:", err); os.Exit(1) }

func run(input io.Reader, output io.Writer) error {
	data, err := io.ReadAll(io.LimitReader(input, 8*1024*1024+1))
	if err != nil { return fmt.Errorf("read request: %w", err) }
	if len(data) == 0 || len(data) > 8*1024*1024 { return fmt.Errorf("request is empty or exceeds 8 MiB") }
	if err := rejectDuplicateJSONKeys(data); err != nil { return fmt.Errorf("duplicate/ambiguous JSON: %w", err) }
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var request compileRequest
	if err := decoder.Decode(&request); err != nil { return fmt.Errorf("decode request: %w", err) }
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil { return fmt.Errorf("multiple top-level JSON values") }
		return fmt.Errorf("trailing JSON: %w", err)
	}
	if request.SchemaVersion != "mace-profile-compile-request-v1" {
		return fmt.Errorf("unsupported schema_version %q", request.SchemaVersion)
	}
	compiled, err := matrix.CompileProfileInventory(request.Profile, request.Catalog)
	if err != nil { return fmt.Errorf("compile profile: %w", err) }
	encoder := json.NewEncoder(output)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(compiled); err != nil { return fmt.Errorf("encode result: %w", err) }
	return nil
}

func rejectDuplicateJSONKeys(raw []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	if err := scanJSONValue(decoder); err != nil { return err }
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		if err == nil { return fmt.Errorf("multiple top-level JSON values") }
		return err
	}
	return nil
}
func scanJSONValue(decoder *json.Decoder) error {
	token, err := decoder.Token()
	if err != nil { return err }
	delim, ok := token.(json.Delim)
	if !ok { return nil }
	switch delim {
	case '{':
		seen := make(map[string]struct{})
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil { return err }
			key, ok := keyToken.(string)
			if !ok { return fmt.Errorf("object key is not a string") }
			if _, exists := seen[key]; exists { return fmt.Errorf("repeated key %q", key) }
			seen[key] = struct{}{}
			if err := scanJSONValue(decoder); err != nil { return err }
		}
		end, err := decoder.Token()
		if err != nil { return err }
		if end != json.Delim('}') { return fmt.Errorf("malformed object terminator") }
	case '[':
		for decoder.More() { if err := scanJSONValue(decoder); err != nil { return err } }
		end, err := decoder.Token()
		if err != nil { return err }
		if end != json.Delim(']') { return fmt.Errorf("malformed array terminator") }
	default:
		return fmt.Errorf("unexpected delimiter %q", delim)
	}
	return nil
}
