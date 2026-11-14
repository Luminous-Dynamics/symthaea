//go:build !linux

package effectbridge

import (
	"errors"
	"os"
)

func lockJournal(_ *os.File) error {
	return errors.New("the MACE effectbridge durable journal currently requires Linux advisory file locking")
}

func unlockJournal(_ *os.File) error { return nil }
