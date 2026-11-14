//go:build linux

package oraclelab

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestFilesystemObserverChildHelper(t *testing.T) {
	if os.Getenv("MACE_FS_CHILD") != "1" {
		return
	}
	path := os.Getenv("MACE_FS_PATH")
	if path == "" {
		t.Fatal("MACE_FS_PATH missing")
	}
	switch os.Getenv("MACE_FS_ACTION") {
	case "write":
		if err := os.WriteFile(path, []byte("child-mutated"), 0o600); err != nil {
			t.Fatal(err)
		}
	case "write-restore":
		if err := os.WriteFile(path, []byte("child-transient"), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("before"), 0o600); err != nil {
			t.Fatal(err)
		}
	default:
		t.Fatalf("unknown MACE_FS_ACTION %q", os.Getenv("MACE_FS_ACTION"))
	}
}

func runFilesystemMutationChild(t *testing.T, path, action string) error {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestFilesystemObserverChildHelper$")
	cmd.Env = append(os.Environ(), "MACE_FS_CHILD=1", "MACE_FS_PATH="+path, "MACE_FS_ACTION="+action)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("child process failed: %w: %s", err, output)
	}
	return nil
}

func TestObserveDirectoryDetectsChildProcessMutation(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "state.txt")
	if err := os.WriteFile(path, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	report := ObserveDirectory(root, func() error {
		return runFilesystemMutationChild(t, path, "write")
	})
	if report.Status != Fail || !report.StateChanged || len(report.Events) == 0 {
		t.Fatalf("child-process mutation missed: %+v", report)
	}
}

func TestObserveDirectoryDetectsChildProcessTransientWriteRestore(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "state.txt")
	original := []byte("before")
	if err := os.WriteFile(path, original, 0o600); err != nil {
		t.Fatal(err)
	}
	report := ObserveDirectory(root, func() error {
		return runFilesystemMutationChild(t, path, "write-restore")
	})
	if report.Status != Fail || report.BeforeSHA256 != report.AfterSHA256 || len(report.Events) == 0 {
		t.Fatalf("child-process transient mutation missed: %+v", report)
	}
}
