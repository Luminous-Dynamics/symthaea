//go:build linux

package effectbridge

import (
	"fmt"
	"os"
	"syscall"
)

func lockJournal(file *os.File) error {
	if err := syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		return fmt.Errorf("another process owns the effect journal: %w", err)
	}
	return nil
}

func unlockJournal(file *os.File) error {
	return syscall.Flock(int(file.Fd()), syscall.LOCK_UN)
}
