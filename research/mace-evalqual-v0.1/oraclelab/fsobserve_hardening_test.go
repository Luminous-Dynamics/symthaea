//go:build linux

package oraclelab

import (
	"os"
	"path/filepath"
	"testing"
)

func TestObserveDirectoryRootReplacementIsIndeterminate(t *testing.T) {
	parent := t.TempDir()
	root := filepath.Join(parent, "world")
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "state.txt"), []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	report := ObserveDirectory(root, func() error {
		if err := os.Rename(root, root+".moved"); err != nil {
			return err
		}
		return os.Mkdir(root, 0o700)
	})
	if report.Status != Indeterminate {
		t.Fatalf("root replacement was not rejected as incomplete observation: %+v", report)
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
