//go:build linux

package witness

import (
	"fmt"
	"os"
	"syscall"
)

func lockFile(file *os.File) error {
	if err := syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		return fmt.Errorf("another witness process owns the journal: %w", err)
	}
	return nil
}

func unlockFile(file *os.File) error {
	return syscall.Flock(int(file.Fd()), syscall.LOCK_UN)
}
