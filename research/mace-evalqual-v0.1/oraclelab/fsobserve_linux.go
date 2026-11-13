//go:build linux

package oraclelab

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
)

const fsWatchMask = syscall.IN_MODIFY | syscall.IN_ATTRIB | syscall.IN_CLOSE_WRITE |
	syscall.IN_CREATE | syscall.IN_DELETE | syscall.IN_MOVED_FROM | syscall.IN_MOVED_TO |
	syscall.IN_DELETE_SELF | syscall.IN_MOVE_SELF | syscall.IN_UNMOUNT | syscall.IN_Q_OVERFLOW

type FSEvent struct {
	Sequence uint64   `json:"sequence"`
	Path     string   `json:"path"`
	Mask     uint32   `json:"mask"`
	Kinds    []string `json:"kinds"`
	Mutation bool     `json:"mutation"`
}

type FSReport struct {
	SchemaVersion string    `json:"schema_version"`
	ScopeRoot     string    `json:"scope_root"`
	BeforeSHA256  string    `json:"before_sha256"`
	AfterSHA256   string    `json:"after_sha256"`
	Events        []FSEvent `json:"events"`
	Coverage      Coverage  `json:"coverage"`
	StateChanged  bool      `json:"state_changed"`
	Status        Status    `json:"status"`
	RunError      string    `json:"run_error,omitempty"`
	Reasons       []string  `json:"reasons,omitempty"`
}

// ObserveDirectory watches one existing, symlink-free directory tree during run.
// The callback must join all child processes and writers before returning.
// New directories, mount changes, unsupported file types, queue overflow, or
// watch loss make the observation indeterminate.
func ObserveDirectory(root string, run func() error) FSReport {
	report := FSReport{SchemaVersion: "mace-fs-observer-report/v1", Status: Indeterminate}
	if run == nil {
		report.Reasons = append(report.Reasons, "nil evaluation callback")
		return report
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		report.Reasons = append(report.Reasons, "resolve root: "+err.Error())
		return report
	}
	abs = filepath.Clean(abs)
	report.ScopeRoot = abs
	info, err := os.Lstat(abs)
	if err != nil || info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		report.Reasons = append(report.Reasons, "root must be an existing real directory")
		return report
	}
	rootStat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		report.Reasons = append(report.Reasons, "root filesystem identity unavailable")
		return report
	}
	mounts, err := readMountTable()
	if err != nil {
		report.Reasons = append(report.Reasons, "mount-table identity unavailable: "+err.Error())
		return report
	}
	rootMountID, ok := mountIDForPath(abs, mounts)
	if !ok {
		report.Reasons = append(report.Reasons, "root mount ID unavailable")
		return report
	}
	fd, err := syscall.InotifyInit1(syscall.IN_CLOEXEC | syscall.IN_NONBLOCK)
	if err != nil {
		report.Reasons = append(report.Reasons, "inotify initialization failed: "+err.Error())
		return report
	}
	defer syscall.Close(fd)
	wdPaths := map[int]string{}
	coverage := Coverage{
		SourceID: "linux-inotify-same-mount-regular-tree",
		ScopeID:  abs,
		Complete: true,
		KnownLimitations: []string{
			"inotify is a host-kernel event source, not an isolated or remotely witnessed observer",
			"mount identity is read from /proc/self/mountinfo and remains subject to the host mount namespace/kernel trust boundary",
			"the callback must join every child process and writer before returning",
			"the declared scope excludes symlinks, special files, and newly created subdirectories",
			"this snapshot does not independently capture ACLs, xattr contents, remote service state, or privileged host activity outside the watched tree",
		},
	}
	if err := addTreeWatches(fd, abs, rootStat.Dev, rootMountID, mounts, wdPaths, &coverage); err != nil {
		coverage.Complete = false
		coverage.Uninstrumented = append(coverage.Uninstrumented, err.Error())
	}
	report.Coverage = coverage
	if !coverage.Complete {
		report.Reasons = append(report.Reasons, "watch coverage is incomplete; evaluation was not run")
		return report
	}
	before, err := directoryDigest(abs)
	if err != nil {
		report.Coverage.Complete = false
		report.Reasons = append(report.Reasons, "pre-evaluation snapshot failed: "+err.Error())
		return report
	}
	report.BeforeSHA256 = before
	// Discard setup noise and begin an invocation-local event sequence.
	if _, _, err := drainInotify(fd, abs, wdPaths, 0); err != nil {
		report.Coverage.Complete = false
		report.Reasons = append(report.Reasons, "pre-evaluation event drain failed: "+err.Error())
		return report
	}

	runErr := callFSCallback(run)
	if runErr != nil {
		report.RunError = runErr.Error()
	}
	events, lastSeq, drainErr := drainInotify(fd, abs, wdPaths, 0)
	if drainErr != nil {
		report.Coverage.Complete = false
		report.Reasons = append(report.Reasons, "post-evaluation event drain failed: "+drainErr.Error())
	}
	report.Events = events
	report.AfterSHA256, err = directoryDigest(abs)
	if err != nil {
		report.Coverage.Complete = false
		report.Reasons = append(report.Reasons, "post-evaluation snapshot failed: "+err.Error())
		return report
	}
	// Catch records queued while the post-state snapshot was read.
	extra, _, extraErr := drainInotify(fd, abs, wdPaths, lastSeq)
	if extraErr != nil {
		report.Coverage.Complete = false
		report.Reasons = append(report.Reasons, "final event drain failed: "+extraErr.Error())
	}
	if len(extra) > 0 {
		report.Events = append(report.Events, extra...)
		report.AfterSHA256, err = directoryDigest(abs)
		if err != nil {
			report.Coverage.Complete = false
			report.Reasons = append(report.Reasons, "final state snapshot failed: "+err.Error())
			return report
		}
	}
	report.StateChanged = report.BeforeSHA256 != report.AfterSHA256
	for _, event := range report.Events {
		if event.Mutation {
			report.StateChanged = true
		}
	}
	if !report.Coverage.Complete {
		report.Status = Indeterminate
		return report
	}
	if report.StateChanged {
		report.Status = Fail
		report.Reasons = append(report.Reasons, "filesystem state changed or mutation events were observed")
		return report
	}
	report.Status = Pass
	return report
}

func callFSCallback(run func() error) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("evaluation callback panic: %v", recovered)
		}
	}()
	return run()
}

type mountEntry struct {
	id        int
	mountPath string
}

func readMountTable() ([]mountEntry, error) {
	data, err := os.ReadFile("/proc/self/mountinfo")
	if err != nil {
		return nil, err
	}
	unescape := strings.NewReplacer(`\040`, " ", `\011`, "	", `\012`, "
", `\134`, `\`)
	entries := []mountEntry{}
	for _, line := range strings.Split(string(data), "
") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 6 {
			return nil, fmt.Errorf("malformed mountinfo record")
		}
		id, err := strconv.Atoi(fields[0])
		if err != nil {
			return nil, fmt.Errorf("invalid mount ID: %w", err)
		}
		entries = append(entries, mountEntry{id: id, mountPath: filepath.Clean(unescape.Replace(fields[4]))})
	}
	if len(entries) == 0 {
		return nil, fmt.Errorf("empty mount table")
	}
	return entries, nil
}

func mountIDForPath(path string, mounts []mountEntry) (int, bool) {
	path = filepath.Clean(path)
	bestID, bestLength := 0, -1
	for _, entry := range mounts {
		mountPath := filepath.Clean(entry.mountPath)
		prefix := strings.TrimSuffix(mountPath, string(os.PathSeparator)) + string(os.PathSeparator)
		if path != mountPath && !strings.HasPrefix(path, prefix) {
			continue
		}
		if len(mountPath) > bestLength {
			bestID, bestLength = entry.id, len(mountPath)
		}
	}
	return bestID, bestLength >= 0
}

func addTreeWatches(fd int, root string, rootDev uint64, rootMountID int, mounts []mountEntry, wdPaths map[int]string, coverage *Coverage) error {
	return filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			coverage.Complete = false
			coverage.Uninstrumented = append(coverage.Uninstrumented, path)
			return nil
		}
		info, err := os.Lstat(path)
		if err != nil {
			coverage.Complete = false
			coverage.Uninstrumented = append(coverage.Uninstrumented, path)
			return nil
		}
		if info.Mode()&os.ModeSymlink != 0 || (!info.IsDir() && !info.Mode().IsRegular()) {
			coverage.Complete = false
			coverage.Uninstrumented = append(coverage.Uninstrumented, path)
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		st, ok := info.Sys().(*syscall.Stat_t)
		mountID, mountOK := mountIDForPath(path, mounts)
		if !ok || st.Dev != rootDev || !mountOK || mountID != rootMountID {
			coverage.Complete = false
			coverage.Uninstrumented = append(coverage.Uninstrumented, path+" (different or unknown filesystem/mount)")
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if info.IsDir() {
			wd, err := syscall.InotifyAddWatch(fd, path, fsWatchMask|syscall.IN_ONLYDIR|syscall.IN_DONT_FOLLOW)
			if err != nil {
				coverage.Complete = false
				coverage.Uninstrumented = append(coverage.Uninstrumented, path+" (watch failed: "+err.Error()+")")
				return filepath.SkipDir
			}
			wdPaths[wd] = path
		}
		return nil
	})
}

func drainInotify(fd int, root string, wdPaths map[int]string, startSeq uint64) ([]FSEvent, uint64, error) {
	buf := make([]byte, 64*1024)
	seq := startSeq
	out := []FSEvent{}
	headerSize := 16 // Linux wire event header: wd, mask, cookie and len.
	for {
		n, err := syscall.Read(fd, buf)
		if err == syscall.EAGAIN || err == syscall.EWOULDBLOCK {
			return out, seq, nil
		}
		if err == syscall.EINTR {
			continue
		}
		if err != nil {
			return out, seq, err
		}
		if n == 0 {
			return out, seq, nil
		}
		for offset := 0; offset < n; {
			if n-offset < headerSize {
				return out, seq, fmt.Errorf("short inotify event header")
			}
			// Do not cast to syscall.InotifyEvent: that Go struct includes
			// a fixed Name array and is larger than the 16-byte kernel header.
			wd := int(int32(binary.LittleEndian.Uint32(buf[offset : offset+4])))
			mask := binary.LittleEndian.Uint32(buf[offset+4 : offset+8])
			nameLen := binary.LittleEndian.Uint32(buf[offset+12 : offset+16])
			nameStart := offset + headerSize
			if uint64(nameLen) > uint64(n-nameStart) {
				return out, seq, fmt.Errorf("truncated inotify event name")
			}
			nameEnd := nameStart + int(nameLen)
			nameBytes := bytes.TrimRight(buf[nameStart:nameEnd], " ")
			base, known := wdPaths[wd]
			if mask&syscall.IN_Q_OVERFLOW != 0 {
				return out, seq, fmt.Errorf("inotify queue overflow")
			}
			if mask&(syscall.IN_UNMOUNT|syscall.IN_IGNORED) != 0 {
				return out, seq, fmt.Errorf("inotify watch invalidated or filesystem unmounted")
			}
			if !known {
				return out, seq, fmt.Errorf("event from an unknown watch descriptor")
			}
			full := base
			if len(nameBytes) > 0 {
				full = filepath.Join(base, string(nameBytes))
			}
			rel, err := filepath.Rel(root, full)
			if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
				return out, seq, fmt.Errorf("event path escaped watched root")
			}
			if mask&syscall.IN_ISDIR != 0 && mask&(syscall.IN_CREATE|syscall.IN_MOVED_TO) != 0 {
				return out, seq, fmt.Errorf("new directory created during evaluation; child coverage is not established")
			}
			seq++
			out = append(out, FSEvent{
				Sequence: seq,
				Path:     filepath.ToSlash(rel),
				Mask:     mask,
				Kinds:    maskKinds(mask),
				Mutation: mask&(syscall.IN_MODIFY|syscall.IN_ATTRIB|syscall.IN_CLOSE_WRITE|syscall.IN_CREATE|syscall.IN_DELETE|syscall.IN_MOVED_FROM|syscall.IN_MOVED_TO|syscall.IN_DELETE_SELF|syscall.IN_MOVE_SELF) != 0,
			})
			offset = nameEnd
		}
	}
}

func maskKinds(mask uint32) []string {
	pairs := []struct {
		mask uint32
		name string
	}{
		{syscall.IN_MODIFY, "modify"}, {syscall.IN_ATTRIB, "attrib"}, {syscall.IN_CLOSE_WRITE, "close_write"},
		{syscall.IN_CREATE, "create"}, {syscall.IN_DELETE, "delete"}, {syscall.IN_MOVED_FROM, "moved_from"},
		{syscall.IN_MOVED_TO, "moved_to"}, {syscall.IN_DELETE_SELF, "delete_self"}, {syscall.IN_MOVE_SELF, "move_self"},
		{syscall.IN_UNMOUNT, "unmount"}, {syscall.IN_Q_OVERFLOW, "queue_overflow"}, {syscall.IN_IGNORED, "ignored"},
	}
	out := []string{}
	for _, pair := range pairs {
		if mask&pair.mask != 0 {
			out = append(out, pair.name)
		}
	}
	sort.Strings(out)
	return out
}

func directoryDigest(root string) (string, error) {
	paths := []string{}
	entries := map[string]os.FileInfo{}
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		info, err := os.Lstat(path)
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 || (!info.IsDir() && !info.Mode().IsRegular()) {
			return fmt.Errorf("unsupported symlink or special file %s", path)
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		paths = append(paths, rel)
		entries[rel] = info
		return nil
	})
	if err != nil {
		return "", err
	}
	sort.Strings(paths)
	var builder strings.Builder
	for _, rel := range paths {
		info := entries[rel]
		fmt.Fprintf(&builder, "%d:%s:%d:%d:", len(rel), rel, uint32(info.Mode()), info.Size())
		if info.IsDir() {
			builder.WriteString("dir")
			continue
		}
		data, err := os.ReadFile(filepath.Join(root, rel))
		if err != nil {
			return "", err
		}
		current, err := os.Lstat(filepath.Join(root, rel))
		if err != nil || current.Size() != info.Size() || current.ModTime() != info.ModTime() || current.Mode() != info.Mode() {
			return "", fmt.Errorf("file changed while snapshotting %s", rel)
		}
		builder.WriteString(Digest(data))
	}
	return Digest([]byte(builder.String())), nil
}
