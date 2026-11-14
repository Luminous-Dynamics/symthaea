package main

import (
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/Luminous-Dynamics/symthaea/research/mace-evalqual-v0.1/effectbridge"
)

func main() {
	listen := flag.String("listen", "127.0.0.1:8087", "HTTP listen address")
	journal := flag.String("journal", "./mace-effect-service.jsonl", "durable service journal path")
	allowFaults := flag.Bool("allow-test-fault-injection", false, "allow deterministic post-commit acknowledgement-drop test header; never enable in production")
	flag.Parse()
	server, err := effectbridge.NewServer(*journal, *allowFaults)
	if err != nil {
		fail(err)
	}
	defer server.Close()
	listener, err := net.Listen("tcp", *listen)
	if err != nil {
		fail(err)
	}
	fmt.Fprintf(os.Stderr, "mace-effectsvc listening on %s; journal=%s; fault-injection=%v\n", listener.Addr(), *journal, *allowFaults)
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	go func() { <-stop; _ = server.Close() }()
	if err := server.Serve(listener); err != nil && err.Error() != "http: Server closed" {
		fail(err)
	}
}

func fail(err error) { fmt.Fprintln(os.Stderr, err); os.Exit(2) }
