package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	independence "luminousdynamics/mon035-observer-independence"
)

type output struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	Result independence.Result `json:"result"`
}

func readInput(path string)(independence.Input,error){
	var input independence.Input
	file,err:=os.Open(path); if err!=nil{return input,err}
	defer file.Close()
	decoder:=json.NewDecoder(io.LimitReader(file,16<<20))
	decoder.DisallowUnknownFields()
	if err:=decoder.Decode(&input);err!=nil{return input,err}
	var trailing any
	if err:=decoder.Decode(&trailing);err!=io.EOF{return input,errors.New("input must contain exactly one JSON document")}
	return input,nil
}

func run()int{
	inputPath:=flag.String("input","","strict MON-035 observer-independence JSON")
	flag.Parse()
	if *inputPath==""{fmt.Fprintln(os.Stderr,"--input is required");return 64}
	input,err:=readInput(*inputPath)
	if err!=nil{fmt.Fprintf(os.Stderr,"invalid input: %v\n",err);return 65}
	result:=independence.Evaluate(input)
	encoded,err:=json.MarshalIndent(output{Profile:"mon035-observer-independence-result-v1",Version:"1.0.0",Result:result},"","  ")
	if err!=nil{fmt.Fprintf(os.Stderr,"encode output: %v\n",err);return 70}
	fmt.Println(string(encoded))
	switch result.Verdict{
	case independence.Qualified:return 0
	case independence.Blocked:return 2
	default:return 3
	}
}
func main(){os.Exit(run())}
