package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	continuity "luminousdynamics/mon036-recovery-evidence-continuity"
)

func read(path string)(continuity.Input,error){
	var in continuity.Input
	f,err:=os.Open(path);if err!=nil{return in,err};defer f.Close()
	raw,err:=io.ReadAll(io.LimitReader(f,(32<<20)+1));if err!=nil{return in,err}
	if len(raw)>32<<20{return in,errors.New("input exceeds 32 MiB")}
	d:=json.NewDecoder(bytes.NewReader(raw));d.DisallowUnknownFields()
	if err=d.Decode(&in);err!=nil{return in,err}
	var trailing any
	if err=d.Decode(&trailing);err!=io.EOF{return in,errors.New("expected one JSON document")}
	return in,nil
}
func run()int{
	p:=flag.String("input","","MON-036 continuity JSON");flag.Parse()
	if *p==""{fmt.Fprintln(os.Stderr,"--input is required");return 64}
	in,err:=read(*p);if err!=nil{fmt.Fprintln(os.Stderr,err);return 65}
	out:=continuity.Evaluate(in);raw,err:=json.MarshalIndent(out,"","  ")
	if err!=nil{fmt.Fprintln(os.Stderr,err);return 70}
	fmt.Println(string(raw));return 0
}
func main(){os.Exit(run())}
