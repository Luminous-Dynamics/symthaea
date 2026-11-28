package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"luminousdynamics/mon034-timeline-ancestry"
)

type output struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	Result ancestry.Result `json:"result"`
}

func readInput(path string) (ancestry.Input, error) {
	var input ancestry.Input
	file, err := os.Open(path)
	if err != nil { return input, err }
	defer file.Close()
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&input); err != nil { return input, err }
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return input, errors.New("input must contain exactly one JSON document")
	}
	return input, nil
}

func run() int {
	inputPath := flag.String("input", "", "strict MON-034 timeline ancestry JSON")
	flag.Parse()
	if *inputPath == "" {
		fmt.Fprintln(os.Stderr, "--input is required")
		return 64
	}
	input, err := readInput(*inputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid input: %v\n", err)
		return 65
	}
	result := ancestry.Evaluate(input)
	encoded, err := json.MarshalIndent(output{
		Profile:"mon034-postgres-timeline-ancestry-result-v1",
		Version:"1.0.0",
		Result:result,
	}, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode output: %v\n", err)
		return 70
	}
	fmt.Println(string(encoded))
	switch result.Verdict {
	case ancestry.Qualified: return 0
	case ancestry.Blocked: return 2
	default: return 3
	}
}

func main() { os.Exit(run()) }
