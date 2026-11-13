package main

import (
	"flag"
	"fmt"
	"os"

	evalqual "github.com/Luminous-Dynamics/symthaea/research/mace-evalqual-v0.1"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "validate":
		validate(os.Args[2:])
	case "score":
		score(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: mace-evalqual validate -corpus corpus.json | score -corpus corpus.json -manifest manifest.json -observations observations.json [-out report.json]")
}

func validate(args []string) {
	fs := flag.NewFlagSet("validate", flag.ExitOnError)
	corpusPath := fs.String("corpus", "", "path to pinned qualification corpus")
	manifestPath := fs.String("manifest", "", "path to run manifest")
	_ = fs.Parse(args)
	if *corpusPath == "" {
		fatalf("-corpus is required")
	}
	var corpus evalqual.Corpus
	if _, err := evalqual.ReadJSONFile(*corpusPath, &corpus); err != nil {
		fatalf("%v", err)
	}
	if err := evalqual.ValidateCorpus(corpus); err != nil {
		fatalf("corpus invalid: %v", err)
	}
	if *manifestPath != "" {
		var manifest evalqual.RunManifest
		if _, err := evalqual.ReadJSONFile(*manifestPath, &manifest); err != nil {
			fatalf("%v", err)
		}
		if err := evalqual.ValidateManifest(manifest, corpus); err != nil {
			fatalf("run manifest invalid: %v", err)
		}
	}
	fmt.Println("valid")
}

func score(args []string) {
	fs := flag.NewFlagSet("score", flag.ExitOnError)
	corpusPath := fs.String("corpus", "", "path to pinned qualification corpus")
	manifestPath := fs.String("manifest", "", "path to content-addressed run manifest")
	observationsPath := fs.String("observations", "", "path to oracle/grader observations")
	outPath := fs.String("out", "", "optional report output path (stdout when empty)")
	_ = fs.Parse(args)
	if *corpusPath == "" || *manifestPath == "" || *observationsPath == "" {
		fatalf("-corpus, -manifest and -observations are required")
	}
	var corpus evalqual.Corpus
	corpusBytes, err := evalqual.ReadJSONFile(*corpusPath, &corpus)
	if err != nil {
		fatalf("%v", err)
	}
	var manifest evalqual.RunManifest
	manifestBytes, err := evalqual.ReadJSONFile(*manifestPath, &manifest)
	if err != nil {
		fatalf("%v", err)
	}
	var observations evalqual.ObservationSet
	observationBytes, err := evalqual.ReadJSONFile(*observationsPath, &observations)
	if err != nil {
		fatalf("%v", err)
	}
	report, err := evalqual.Score(corpus, manifest, observations, corpusBytes, manifestBytes, observationBytes)
	if err != nil {
		fatalf("qualification input invalid: %v", err)
	}
	data, err := evalqual.ReportBytes(report)
	if err != nil {
		fatalf("%v", err)
	}
	if *outPath == "" {
		if _, err := os.Stdout.Write(append(data, '\n')); err != nil {
			fatalf("write stdout: %v", err)
		}
	} else if err := os.WriteFile(*outPath, append(data, '\n'), 0o644); err != nil {
		fatalf("write report: %v", err)
	}
	if !report.QualificationPassed {
		os.Exit(1)
	}
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(2)
}
