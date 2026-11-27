package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"luminousdynamics/mon031-lineage"
)

const profile = "mon031-backup-lineage-input-v1"

type input struct {
	Profile string                 `json:"profile"`
	Version string                 `json:"version"`
	Policy  lineage.Policy         `json:"policy"`
	Backups []lineage.BackupRecord `json:"backups"`
}

type output struct {
	Profile string         `json:"profile"`
	Result  lineage.Result `json:"result"`
}

func decodeStrict(reader io.Reader) (input, error) {
	var value input
	decoder := json.NewDecoder(reader)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return input{}, err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return input{}, errors.New("input must contain exactly one JSON document")
	}
	if value.Profile != profile || value.Version != "1.0.0" || len(value.Backups) == 0 {
		return input{}, errors.New("unsupported profile/version or empty backup inventory")
	}
	return value, nil
}

func run() int {
	path := flag.String("input", "", "backup-lineage inventory JSON")
	flag.Parse()
	if *path == "" {
		fmt.Fprintln(os.Stderr, "--input is required")
		return 64
	}
	file, err := os.Open(*path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open input: %v\n", err)
		return 66
	}
	defer file.Close()
	value, err := decodeStrict(file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid input: %v\n", err)
		return 65
	}
	result := lineage.Evaluate(value.Policy, value.Backups)
	if err := json.NewEncoder(os.Stdout).Encode(output{Profile: "mon031-backup-lineage-result-v1", Result: result}); err != nil {
		fmt.Fprintf(os.Stderr, "encode result: %v\n", err)
		return 70
	}
	switch result.Verdict {
	case lineage.Qualified:
		return 0
	case lineage.Blocked:
		return 2
	default:
		return 3
	}
}

func main() { os.Exit(run()) }
