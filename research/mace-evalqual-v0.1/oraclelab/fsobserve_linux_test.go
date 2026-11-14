//go:build linux

package oraclelab

import (
	"os"
	"path/filepath"
	"testing"
)

func TestObserveDirectoryReadOnlyPasses(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "state.txt"), []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	report := ObserveDirectory(root, func() error {
		_, err := os.ReadFile(filepath.Join(root, "state.txt"))
		return err
	})
	if report.Status != Pass || report.StateChanged || len(report.Events) != 0 {
		t.Fatalf("read-only filesystem evaluation failed: %+v", report)
	}
}

func TestObserveDirectoryPersistentMutationFails(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "state.txt"), []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	report := ObserveDirectory(root, func() error {
		return os.WriteFile(filepath.Join(root, "state.txt"), []byte("after"), 0o600)
	})
	if report.Status != Fail || !report.StateChanged || len(report.Events) == 0 {
		t.Fatalf("persistent filesystem mutation missed: %+v", report)
	}
}

func TestObserveDirectoryTransientWriteRestoreFails(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "state.txt")
	original := []byte("before")
	if err := os.WriteFile(path, original, 0o600); err != nil {
		t.Fatal(err)
	}
	report := ObserveDirectory(root, func() error {
		if err := os.WriteFile(path, []byte("transient"), 0o600); err != nil {
			return err
		}
		return os.WriteFile(path, original, 0o600)
	})
	if report.Status != Fail || report.BeforeSHA256 != report.AfterSHA256 || len(report.Events) == 0 {
		t.Fatalf("transient mutation not detected: %+v", report)
	}
}

func TestObserveDirectoryRejectsSymlinkCoverage(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(t.TempDir(), "outside.txt")
	if err := os.WriteFile(target, []byte("outside"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, filepath.Join(root, "linked.txt")); err != nil {
		t.Fatal(err)
	}
	called := false
	report := ObserveDirectory(root, func() error { called = true; return nil })
	if report.Status != Indeterminate || report.Coverage.Complete || called {
		t.Fatalf("symlink path was treated as covered: %+v", report)
	}
}

func TestObserveDirectoryNewSubdirectoryMakesCoverageIndeterminate(t *testing.T) {
	root := t.TempDir()
	report := ObserveDirectory(root, func() error {
		return os.Mkdir(filepath.Join(root, "new-dir"), 0o700)
	})
	if report.Status != Indeterminate {
		t.Fatalf("new directory should invalidate recursive coverage: %+v", report)
	}
}

func TestObserveDirectoryCallbackPanicIsStillMeasured(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "state.txt")
	if err := os.WriteFile(path, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	report := ObserveDirectory(root, func() error {
		if err := os.WriteFile(path, []byte("changed"), 0o600); err != nil {
			return err
		}
		panic("injected panic")
	})
	if report.RunError == "" || report.Status != Fail || !report.StateChanged {
		t.Fatalf("panic path did not retain state mutation evidence: %+v", report)
	}
}

func TestObserveDirectoryNestedFileMutationFails(t *testing.T) {
	root := t.TempDir()
	nested := filepath.Join(root, "nested")
	if err := os.Mkdir(nested, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(nested, "state.txt")
	if err := os.WriteFile(path, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	report := ObserveDirectory(root, func() error {
		return os.WriteFile(path, []byte("after!"), 0o600)
	})
	if report.Status != Fail || !report.StateChanged || len(report.Events) == 0 {
		t.Fatalf("nested filesystem mutation missed: %+v", report)
	}
}

func TestObserveDirectoryCreateDeleteTransientFileFails(t *testing.T) {
	root := t.TempDir()
	report := ObserveDirectory(root, func() error {
		path := filepath.Join(root, "transient.txt")
		if err := os.WriteFile(path, []byte("temporary"), 0o600); err != nil {
			return err
		}
		return os.Remove(path)
	})
	if report.Status != Fail || report.BeforeSHA256 != report.AfterSHA256 || len(report.Events) == 0 {
		t.Fatalf("transient create/delete not detected: %+v", report)
	}
}

func TestObserveDirectoryReportsHostObserverLimitations(t *testing.T) {
	root := t.TempDir()
	report := ObserveDirectory(root, func() error { return nil })
	if report.Status != Pass {
		t.Fatalf("read-only run should be non-interfering: %+v", report)
	}
	if report.Coverage.SourceID != "linux-inotify-same-mount-regular-tree" || len(report.Coverage.KnownLimitations) < 4 {
		t.Fatalf("observer trust boundary was not disclosed: %+v", report.Coverage)
	}
}

func TestMountIDLookupUsesMostSpecificMountpoint(t *testing.T) {
	mounts := []mountEntry{
		{id: 1, mountPath: "/"},
		{id: 2, mountPath: "/var"},
		{id: 3, mountPath: "/var/lib"},
	}
	cases := []struct {
		path string
		want int
	}{
		{path: "/etc/config", want: 1},
		{path: "/var/log/app.log", want: 2},
		{path: "/var/lib/db/data", want: 3},
	}
	for _, tc := range cases {
		got, ok := mountIDForPath(tc.path, mounts)
		if !ok || got != tc.want {
			t.Fatalf("mountIDForPath(%q) = (%d,%v), want (%d,true)", tc.path, got, ok, tc.want)
		}
	}
	if _, ok := mountIDForPath("relative/path", mounts); ok {
		t.Fatal("relative path unexpectedly resolved against an absolute mount table")
	}
}

func TestReadMountTableFindsCurrentRoot(t *testing.T) {
	mounts, err := readMountTable()
	if err != nil {
		t.Fatalf("read /proc/self/mountinfo: %v", err)
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := mountIDForPath(wd, mounts); !ok {
		t.Fatalf("current working directory %q did not resolve to a mount ID", wd)
	}
}
