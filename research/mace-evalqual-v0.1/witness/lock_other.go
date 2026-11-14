//go:build !linux

package witness

import (
	"errors"
	"os"
)

func lockFile(*os.File) error {
	return errors.New("durable single-writer witness locking is supported only on Linux")
}

func unlockFile(*os.File) error { return nil }
