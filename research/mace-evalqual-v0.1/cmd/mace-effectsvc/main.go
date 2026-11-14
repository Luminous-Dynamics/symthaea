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
	backend := flag.String("backend", "journal", "destination backend: journal or sqlite")
	journal := flag.String("journal", "./mace-effect-service.jsonl", "append-only journal path when backend=journal")
	database := flag.String("database", "./mace-effect-service.sqlite3", "SQLite database path when backend=sqlite (Linux+cgo)")
	allowFaults := flag.Bool("allow-test-fault-injection", false, "allow deterministic post-commit acknowledgement-drop test header; never enable in production")
	flag.Parse()
	var server *effectbridge.Server
	var err error
	switch *backend {
	case "journal":
		server, err = effectbridge.NewServer(*journal, *allowFaults)
	case "sqlite":
		server, err = effectbridge.NewSQLiteServer(*database, *allowFaults)
	default:
		fail(fmt.Errorf("unsupported backend %q (choose journal or sqlite)", *backend))
	}
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
